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

__all__ = ["merge_head_major", "rope_interleaved"]

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
