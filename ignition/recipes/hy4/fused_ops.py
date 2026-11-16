"""consult18 P1 (host-bound prefill): one-kernel replacements for eager op chains.

The overlapped prefill is host-bound: one Python thread issues both lanes'
~350 kernels per layer, so the GPU idles. Each function here replaces an eager
chain of tiny torch ops with ONE Triton kernel that reproduces the eager
arithmetic bit for bit:
- every multiply and add is separately rounded (`mul_rn` / `add_rn`, never a
  contracted FMA);
- transcendentals are libdevice's accurate ones (cosf / sinf / exp2f / log2f,
  what torch's float kernels call), never the approximate PTX forms;
- divisions are `div_rn`;
- small source-axis sums run in torch's order.
Bit-exactness against the eager path is the gate (tests on CUDA, and the
serve's numerics hash must not move when a switch flips).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

__all__ = ["ihc_post_pre_split", "merge_head_major", "rms_norm", "rope_interleaved"]

_INV_FREQ = {}


def _inv_freq(rope_dim, theta, device):
    key = (int(rope_dim), float(theta), str(device))
    value = _INV_FREQ.get(key)
    if value is None:
        # Exactly the eager expression (attention.apply_interleaved_rope).
        value = 1.0 / (theta ** (torch.arange(0, rope_dim, 2, device=device, dtype=torch.float32)
                                 / rope_dim))
        _INV_FREQ[key] = value.contiguous()
    return value


@triton.jit
def _rope_kernel(x_ptr, out_ptr, pos_ptr, inv_ptr, stride_n, stride_d, D, HEADS,
                 ROPE: tl.constexpr, BLOCK_D: tl.constexpr):
    n = tl.program_id(0)
    row = n // HEADS
    offs = tl.arange(0, BLOCK_D)
    live = offs < D
    x = tl.load(x_ptr + n * stride_n + offs * stride_d, mask=live, other=0.0)
    start = D - ROPE
    j = offs - start
    rot = live & (offs >= start)
    jj = tl.where(rot, j, 0)
    pair = jj // 2
    even = (jj % 2) == 0
    pos = tl.load(pos_ptr + row).to(tl.float32)
    inv = tl.load(inv_ptr + pair, mask=rot, other=0.0)
    angle = libdevice.mul_rn(pos, inv)
    c = libdevice.cos(angle)
    s = libdevice.sin(angle)
    partner = tl.where(even, offs + 1, offs - 1)
    xp = tl.load(x_ptr + n * stride_n + partner * stride_d, mask=rot, other=0.0).to(tl.float32)
    xf = x.to(tl.float32)
    other = tl.where(even, -xp, xp)
    r = libdevice.add_rn(libdevice.mul_rn(xf, c), libdevice.mul_rn(other, s))
    y = tl.where(rot, r.to(out_ptr.dtype.element_ty), x)
    tl.store(out_ptr + n * D + offs, y, mask=live)


def rope_interleaved(x, positions, *, rope_dim, theta):
    """`attention.apply_interleaved_rope` in one kernel (same output, bitwise)."""
    D = int(x.shape[-1])
    rows = int(positions.numel())
    lead = x.shape[:-1]
    n = 1
    for v in lead:
        n *= int(v)
    if n % max(rows, 1):
        raise ValueError("rope rows must cover the positions")
    heads = n // rows if rows else 1
    flat = x if x.dim() == 2 else x.reshape(n, D)     # 2-D views keep their strides
    out = torch.empty(lead + (D,), dtype=x.dtype, device=x.device)
    if n == 0:
        return out
    inv = _inv_freq(rope_dim, theta, x.device)
    pos = positions.reshape(-1)
    _rope_kernel[(n,)](flat, out, pos, inv, flat.stride(0), flat.stride(1), D, heads,
                       ROPE=int(rope_dim), BLOCK_D=triton.next_power_of_2(D))
    return out


@triton.jit
def _merge_kernel(p_ptr, l_ptr, out_ptr, lse_ptr, HR, W, S: tl.constexpr, BLOCK_W: tl.constexpr):
    hr = tl.program_id(0)
    wb = tl.program_id(1)
    offs = wb * BLOCK_W + tl.arange(0, BLOCK_W)
    live = offs < W
    peak = tl.load(l_ptr + hr).to(tl.float32)
    for s in tl.static_range(1, S):
        peak = tl.maximum(peak, tl.load(l_ptr + s * HR + hr).to(tl.float32))
    finite = (peak == peak) & (peak != float("inf")) & (peak != float("-inf"))
    peak = tl.where(finite, peak, 0.0)
    total = 0.0
    acc = tl.zeros((BLOCK_W,), dtype=tl.float32)
    for s in tl.static_range(0, S):
        w = libdevice.exp2(libdevice.sub_rn(tl.load(l_ptr + s * HR + hr).to(tl.float32), peak))
        total = libdevice.add_rn(total, w)
        p = tl.load(p_ptr + (s * HR + hr) * W + offs, mask=live, other=0.0).to(tl.float32)
        acc = libdevice.add_rn(acc, libdevice.mul_rn(p, w))
    seen = total > 0
    safe = tl.maximum(total, 1e-30)
    out = tl.where(seen, libdevice.div_rn(acc, safe), 0.0)
    tl.store(out_ptr + hr * W + offs, out, mask=live)
    if wb == 0:
        lse = tl.where(seen, libdevice.add_rn(peak, libdevice.log2(safe)), float("-inf"))
        tl.store(lse_ptr + hr, lse)


def merge_head_major(partials, lses):
    """`collectives._merge_head_major` in one kernel (same outputs, bitwise):
    `(out [heads, rows, width] FP32, lse [heads, rows])` from `partials`
    `[sources, heads, rows, width]` and base-2 `lses` `[sources, heads, rows]`."""
    S, H, R, W = (int(v) for v in partials.shape)
    p = partials.contiguous()
    lv = lses.contiguous()
    out = torch.empty((H, R, W), dtype=torch.float32, device=partials.device)
    lse = torch.empty((H, R), dtype=torch.float32, device=partials.device)
    block = min(1024, triton.next_power_of_2(max(W, 1)))
    _merge_kernel[(H * R, triton.cdiv(W, block))](p, lv, out, lse, H * R, W, S=S, BLOCK_W=block)
    return out, lse


@triton.jit
def _rms_tail_kernel(x_ptr, ms_ptr, w_ptr, out_ptr, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    blk = tl.program_id(1)
    offs = blk * BLOCK + tl.arange(0, BLOCK)
    live = offs < N
    x = tl.load(x_ptr + row * N + offs, mask=live, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + offs, mask=live, other=0.0).to(tl.float32)
    r = libdevice.rsqrt(libdevice.add_rn(tl.load(ms_ptr + row), eps))
    y = libdevice.mul_rn(w, libdevice.mul_rn(x, r))
    tl.store(out_ptr + row * N + offs, y.to(out_ptr.dtype.element_ty), mask=live)


def rms_norm(x, weight, eps):
    """`attention.rms_norm` with the elementwise tail in one kernel (bitwise the
    same): the mean of squares stays torch's own reduction (its order), then
    `(w * (x * rsqrt(ms + eps))).to(x.dtype)` runs fused."""
    N = int(x.shape[-1])
    flat = x.reshape(-1, N)
    if not flat.is_contiguous():
        flat = flat.contiguous()
    ms = flat.float().square().mean(-1)
    out = torch.empty_like(flat)
    rows = int(flat.shape[0])
    if rows:
        block = min(2048, triton.next_power_of_2(N))
        _rms_tail_kernel[(rows, triton.cdiv(N, block))](flat, ms, weight.contiguous(), out, N,
                                                        float(eps), BLOCK=block)
    return out.view(x.shape)


# --------------------------------------------------------------------------
# consult18 P2: the iHC sublayer boundary (`run_post_pre`) for decode rows.
# b12x's `_ihc_post_pre_kernel` runs ONE program per token row: at one decode
# row one CTA streams the 786 KB FP32 `fn` and four 6144-wide passes (117.7 us
# per call, 155 calls = 18.3 ms of a 51.5 ms 16k decode step). The same math in
# three launches over (row, chunk) grids. Bit-identical by construction: every
# chunk's `tl.sum` is the original's (same BLOCK_K / BLOCK_D, same expression,
# same num_warps), and the chunk partials are accumulated in the original's
# sequential order from 0.0; the gates, RMS and stores are its expressions.
# --------------------------------------------------------------------------
@triton.jit
def _ihc_pp_chunks_kernel(x_ptr, residual_ptr, prev_post_ptr, fn_ptr, out_ptr, part_ptr,
                          H, K, NC, BLOCK_K: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    c = tl.program_id(1)
    k = c * BLOCK_K
    offs = k + tl.arange(0, BLOCK_K)
    si = offs // H
    d = offs - si * H
    rv = tl.load(residual_ptr + t * K + offs).to(tl.float32)
    xv = tl.load(x_ptr + t * H + d).to(tl.float32)
    p = tl.load(prev_post_ptr + t * 4 + si)
    ob = (p * xv + rv).to(tl.bfloat16).to(tl.float32)
    tl.store(out_ptr + t * K + offs, ob.to(tl.bfloat16))
    base = part_ptr + (t * NC + c) * 9
    tl.store(base + 0, tl.sum(ob * ob))
    tl.store(base + 1, tl.sum(ob * tl.load(fn_ptr + 0 * K + offs).to(tl.float32)))
    tl.store(base + 2, tl.sum(ob * tl.load(fn_ptr + 1 * K + offs).to(tl.float32)))
    tl.store(base + 3, tl.sum(ob * tl.load(fn_ptr + 2 * K + offs).to(tl.float32)))
    tl.store(base + 4, tl.sum(ob * tl.load(fn_ptr + 3 * K + offs).to(tl.float32)))
    tl.store(base + 5, tl.sum(ob * tl.load(fn_ptr + 4 * K + offs).to(tl.float32)))
    tl.store(base + 6, tl.sum(ob * tl.load(fn_ptr + 5 * K + offs).to(tl.float32)))
    tl.store(base + 7, tl.sum(ob * tl.load(fn_ptr + 6 * K + offs).to(tl.float32)))
    tl.store(base + 8, tl.sum(ob * tl.load(fn_ptr + 7 * K + offs).to(tl.float32)))


@triton.jit
def _ihc_pp_gates(part_ptr, scale_ptr, base_ptr, t, K, NC, magnitude, hc_eps, norm_eps):
    sq = 0.0
    a0 = 0.0
    a1 = 0.0
    a2 = 0.0
    a3 = 0.0
    a4 = 0.0
    a5 = 0.0
    a6 = 0.0
    a7 = 0.0
    for c in range(0, NC):
        base = part_ptr + (t * NC + c) * 9
        sq += tl.load(base + 0)
        a0 += tl.load(base + 1)
        a1 += tl.load(base + 2)
        a2 += tl.load(base + 3)
        a3 += tl.load(base + 4)
        a4 += tl.load(base + 5)
        a5 += tl.load(base + 6)
        a6 += tl.load(base + 7)
        a7 += tl.load(base + 8)
    rms = tl.rsqrt(sq / K + norm_eps)
    s_pre = tl.load(scale_ptr + 0)
    s_post = tl.load(scale_ptr + 1)
    pre0 = tl.sigmoid((a0 * rms) * s_pre + tl.load(base_ptr + 0)) + hc_eps
    pre1 = tl.sigmoid((a1 * rms) * s_pre + tl.load(base_ptr + 1)) + hc_eps
    pre2 = tl.sigmoid((a2 * rms) * s_pre + tl.load(base_ptr + 2)) + hc_eps
    pre3 = tl.sigmoid((a3 * rms) * s_pre + tl.load(base_ptr + 3)) + hc_eps
    post0 = magnitude * tl.sigmoid((a4 * rms) * s_post + tl.load(base_ptr + 4)) + hc_eps
    post1 = magnitude * tl.sigmoid((a5 * rms) * s_post + tl.load(base_ptr + 5)) + hc_eps
    post2 = magnitude * tl.sigmoid((a6 * rms) * s_post + tl.load(base_ptr + 6)) + hc_eps
    post3 = magnitude * tl.sigmoid((a7 * rms) * s_post + tl.load(base_ptr + 7)) + hc_eps
    return pre0, pre1, pre2, pre3, post0, post1, post2, post3


@triton.jit
def _ihc_pp_collapse_kernel(part_ptr, scale_ptr, base_ptr, out_ptr, post_ptr, y_ptr, sqy_ptr,
                            H, K, NC, ND, magnitude, hc_eps, norm_eps,
                            BLOCK_D: tl.constexpr, FUSE_NORM: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    j = tl.program_id(1)
    pre0, pre1, pre2, pre3, post0, post1, post2, post3 = _ihc_pp_gates(
        part_ptr, scale_ptr, base_ptr, t, K, NC, magnitude, hc_eps, norm_eps)
    if j == 0:
        tl.store(post_ptr + t * 4 + 0, post0)
        tl.store(post_ptr + t * 4 + 1, post1)
        tl.store(post_ptr + t * 4 + 2, post2)
        tl.store(post_ptr + t * 4 + 3, post3)
    offs = j * BLOCK_D + tl.arange(0, BLOCK_D)
    o0 = tl.load(out_ptr + t * K + 0 * H + offs).to(tl.float32)
    o1 = tl.load(out_ptr + t * K + 1 * H + offs).to(tl.float32)
    o2 = tl.load(out_ptr + t * K + 2 * H + offs).to(tl.float32)
    o3 = tl.load(out_ptr + t * K + 3 * H + offs).to(tl.float32)
    y = pre0 * o0 + pre1 * o1 + pre2 * o2 + pre3 * o3
    if FUSE_NORM:
        yb = y.to(tl.bfloat16).to(tl.float32)
        tl.store(sqy_ptr + t * ND + j, tl.sum(yb * yb))
    else:
        tl.store(y_ptr + t * H + offs, y.to(tl.bfloat16))


@triton.jit
def _ihc_pp_norm_kernel(part_ptr, scale_ptr, base_ptr, out_ptr, y_ptr, sqy_ptr, norm_weight_ptr,
                        H, K, NC, ND, magnitude, hc_eps, norm_eps, layernorm_eps,
                        BLOCK_D: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    j = tl.program_id(1)
    pre0, pre1, pre2, pre3, post0, post1, post2, post3 = _ihc_pp_gates(
        part_ptr, scale_ptr, base_ptr, t, K, NC, magnitude, hc_eps, norm_eps)
    sqy = 0.0
    for i in range(0, ND):
        sqy += tl.load(sqy_ptr + t * ND + i)
    rmsy = tl.rsqrt(sqy / H + layernorm_eps)
    offs = j * BLOCK_D + tl.arange(0, BLOCK_D)
    o0 = tl.load(out_ptr + t * K + 0 * H + offs).to(tl.float32)
    o1 = tl.load(out_ptr + t * K + 1 * H + offs).to(tl.float32)
    o2 = tl.load(out_ptr + t * K + 2 * H + offs).to(tl.float32)
    o3 = tl.load(out_ptr + t * K + 3 * H + offs).to(tl.float32)
    y = pre0 * o0 + pre1 * o1 + pre2 * o2 + pre3 * o3
    yb = y.to(tl.bfloat16).to(tl.float32)
    tl.store(y_ptr + t * H + offs, (yb * rmsy * tl.load(norm_weight_ptr + offs)).to(tl.bfloat16))


def ihc_post_pre_split(x, residual, prev_post, fn, hc_scale, hc_base, *, magnitude, hc_eps,
                       norm_eps, norm_weight=None, layernorm_eps, block_k, block_d, num_warps):
    """`run_post_pre` (b12x iHC) over (row, chunk) grids: `(out_streams, post, y)`,
    the same bits as the one-program-per-row kernel at the same launch config."""
    hidden = int(residual.shape[-1])
    k_all = 4 * hidden
    rows = residual.numel() // k_all
    outer = tuple(residual.shape[:-2])
    out = torch.empty_like(residual)
    post = torch.empty(outer + (4,), dtype=torch.float32, device=residual.device)
    y = torch.empty(outer + (hidden,), dtype=torch.bfloat16, device=residual.device)
    if prev_post.ndim == residual.ndim and prev_post.shape[-1] == 1:
        prev_post = prev_post.squeeze(-1)
    nc, nd = k_all // block_k, hidden // block_d
    parts = torch.empty((rows, nc, 9), dtype=torch.float32, device=residual.device)
    sqy = torch.empty((rows, nd), dtype=torch.float32, device=residual.device)
    _ihc_pp_chunks_kernel[(rows, nc)](x, residual, prev_post, fn, out, parts, hidden, k_all, nc,
                                      BLOCK_K=block_k, num_warps=num_warps)
    _ihc_pp_collapse_kernel[(rows, nd)](parts, hc_scale, hc_base, out, post, y, sqy, hidden, k_all,
                                        nc, nd, float(magnitude), float(hc_eps), float(norm_eps),
                                        BLOCK_D=block_d, FUSE_NORM=norm_weight is not None,
                                        num_warps=num_warps)
    if norm_weight is not None:
        _ihc_pp_norm_kernel[(rows, nd)](parts, hc_scale, hc_base, out, y, sqy, norm_weight, hidden,
                                        k_all, nc, nd, float(magnitude), float(hc_eps),
                                        float(norm_eps), float(layernorm_eps), BLOCK_D=block_d,
                                        num_warps=num_warps)
    return out, post, y
