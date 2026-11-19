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

import math

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

__all__ = ["dequant_rows_int8", "dequant_slots_int8", "ihc_post_pre_split", "quant_rows_int8", "moe_route", "scores_only_select", "merge_head_major", "rms_norm", "rope_interleaved", "sigmoid_gate", "swiglu_f32", "store_owned_rows", "candidate_pick", "rms_norm_tree", "index_key_prep", "latent_prep", "expand_gate", "index_scores_fused"]

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
def _rope_kernel(x_ptr, out_ptr, pos_ptr, inv_ptr, stride_n, stride_d, D, HEADS, out_stride_n,
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
    # rounded to the INPUT dtype, then widened (bitwise the old store when the dtypes match;
    # round 15: BF16 in, FP32 out = rope then .float())
    y = tl.where(rot, r.to(x_ptr.dtype.element_ty).to(out_ptr.dtype.element_ty),
                 x.to(out_ptr.dtype.element_ty))
    tl.store(out_ptr + n * out_stride_n + offs, y, mask=live)


def rope_interleaved(x, positions, *, rope_dim, theta, out=None):
    """`attention.apply_interleaved_rope` in one kernel (same output, bitwise).
    `out` (round 15): write into a caller's view of the same shape whose rows
    (all leading dims flattened) are evenly strided with unit last stride."""
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
    if out is None:
        out = torch.empty(lead + (D,), dtype=x.dtype, device=x.device)
        out_stride = D
    else:
        out_stride = _even_row_stride(out, lead + (D,), out.dtype)
        if out.dtype != x.dtype and not (x.dtype == torch.bfloat16 and out.dtype == torch.float32):
            raise ValueError("rope out dtype: the input's, or FP32 from BF16")
    if n == 0:
        return out
    inv = _inv_freq(rope_dim, theta, x.device)
    pos = positions.reshape(-1)
    _rope_kernel[(n,)](flat, out, pos, inv, flat.stride(0), flat.stride(1), D, heads, out_stride,
                       ROPE=int(rope_dim), BLOCK_D=triton.next_power_of_2(D))
    return out


def _even_row_stride(out, shape, dtype):
    """The row stride of `out` viewed as [prod(shape[:-1]), shape[-1]] (every
    leading dim flattened), which must be uniform with a unit last stride."""
    if tuple(out.shape) != tuple(shape) or out.dtype != dtype or out.stride(-1) != 1:
        raise ValueError(f"out must be {tuple(shape)} {dtype} with unit last stride")
    if out.dim() == 1:
        return int(shape[-1])
    stride = int(out.stride(-2))
    for d in range(out.dim() - 2, 0, -1):
        if int(out.stride(d - 1)) != int(out.shape[d]) * int(out.stride(d)):
            raise ValueError("out rows must be evenly strided")
    return stride


# --------------------------------------------------------------------------
# Round 15: `rms_norm` (square -> torch mean -> tail) in ONE launch for the
# decode family's few rows. The mean is torch's own reduction tree for a
# contiguous FP32 last-dim mean (ATen Reduce.cuh, vectorized input): block
# width BW from `set_block_dimension`, each thread t keeps 4 accumulators
# acc_i = sum_k x[4 (t + k BW) + i] (sequential in k), combines
# ((acc0 + acc1) + acc2) + acc3, then halves: a shared-memory fold down to 32
# lanes and the shuffle fold with DECREASING offsets 16..1 (this ATen's CUDA
# block_x_reduce), and `* (1 / N)`. The square and the tail are the two
# kernels' own expressions.
# --------------------------------------------------------------------------
@triton.jit
def _halve(v, OFF: tl.constexpr):
    # v[t] + v[t + OFF] for t < OFF  (v has 2 * OFF entries)
    lo, hi = tl.split(tl.permute(tl.reshape(v, [2, OFF]), [1, 0]))
    return lo + hi


@triton.jit
def _tree_mean_sq(base, inv_n, BW: tl.constexpr, K: tl.constexpr):
    """ATen's mean of x^2 over 4 * BW * K contiguous values at `base`."""
    t = tl.arange(0, BW)
    a0 = tl.zeros([BW], dtype=tl.float32)
    a1 = tl.zeros([BW], dtype=tl.float32)
    a2 = tl.zeros([BW], dtype=tl.float32)
    a3 = tl.zeros([BW], dtype=tl.float32)
    for k in tl.static_range(K):
        e = 4 * (t + k * BW)
        v0 = tl.load(base + e + 0).to(tl.float32)
        v1 = tl.load(base + e + 1).to(tl.float32)
        v2 = tl.load(base + e + 2).to(tl.float32)
        v3 = tl.load(base + e + 3).to(tl.float32)
        a0 = a0 + v0 * v0
        a1 = a1 + v1 * v1
        a2 = a2 + v2 * v2
        a3 = a3 + v3 * v3
    v = ((a0 + a1) + a2) + a3
    if BW >= 512:
        v = _halve(v, 256)
    if BW >= 256:
        v = _halve(v, 128)
    if BW >= 128:
        v = _halve(v, 64)
    if BW >= 64:
        v = _halve(v, 32)
    v = _halve(v, 16)
    v = _halve(v, 8)
    v = _halve(v, 4)
    v = _halve(v, 2)
    v = _halve(v, 1)
    return tl.sum(v, axis=0) * inv_n     # one element: the sum is the value itself


@triton.jit
def _rms_tree_kernel(x_ptr, w_ptr, out_ptr, N, eps, inv_n, x_stride, out_stride, ms_ptr,
                     BW: tl.constexpr, K: tl.constexpr, STORE_MS: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    base = x_ptr + row * x_stride
    ms = _tree_mean_sq(base, inv_n, BW, K)
    if STORE_MS:
        tl.store(ms_ptr + row, ms)
    r = libdevice.rsqrt(libdevice.add_rn(ms, eps))
    offs = tl.arange(0, BW * 4 * K)
    x = tl.load(base + offs).to(tl.float32)
    w = tl.load(w_ptr + offs).to(tl.float32)
    y = libdevice.mul_rn(w, libdevice.mul_rn(x, r))
    tl.store(out_ptr + row * out_stride + offs, y.to(out_ptr.dtype.element_ty))


def _torch_mean_block_width(rows, n):
    """ATen's block width for a contiguous FP32 [rows, n] last-dim mean, or None
    when that reduction is not the single-block vectorized form emulated here."""
    if n % 4 or n < 128:
        return None                      # ATen vectorizes the input from dim0 >= 128
    dim0, dim1 = n // 4, rows
    mnt = 512

    def last_pow2(v):
        return 1 << (int(v).bit_length() - 1)

    d0 = last_pow2(dim0) if dim0 < mnt else mnt
    d1 = last_pow2(dim1) if dim1 < mnt else mnt
    bw = min(d0, 32)
    bh = min(d1, mnt // bw)
    bw = min(d0, mnt // bh)
    per_thread = -(-n // bw)
    if per_thread >= bh * 16 or per_thread >= 256:
        return None                      # inter-warp / multi-block split: not emulated
    if bw < 32 or n % (4 * bw):
        return None
    k = n // (4 * bw)
    if k & (k - 1):
        return None                      # the tail's arange needs a power-of-two width
    return bw


@triton.jit
def _latent_prep_kernel(kv_ptr, w_ptr, pos_ptr, inv_ptr, out_ptr, kv_stride, eps, inv_n,
                        BW: tl.constexpr, K: tl.constexpr, ROPE: tl.constexpr):
    """Round 15: the MLA latent row in one launch — `rms_norm(kv[:, :N])` (the
    tree kernel's expressions) and the rope of `kv[:, N:N + ROPE]` (the rope
    kernel's expressions) written side by side."""
    row = tl.program_id(0).to(tl.int64)
    base = kv_ptr + row * kv_stride
    N: tl.constexpr = 4 * BW * K
    ms = _tree_mean_sq(base, inv_n, BW, K)
    r = libdevice.rsqrt(libdevice.add_rn(ms, eps))
    offs = tl.arange(0, N)
    x = tl.load(base + offs).to(tl.float32)
    w = tl.load(w_ptr + offs).to(tl.float32)
    y = libdevice.mul_rn(w, libdevice.mul_rn(x, r))
    tl.store(out_ptr + row * (N + ROPE) + offs, y.to(out_ptr.dtype.element_ty))
    j = tl.arange(0, ROPE)
    xr = tl.load(base + N + j)
    xf = xr.to(tl.float32)
    ev, od = tl.split(tl.reshape(xf, [ROPE // 2, 2]))
    swapped = tl.reshape(tl.join(od, ev), [ROPE])
    even = (j % 2) == 0
    pos = tl.load(pos_ptr + row).to(tl.float32)
    inv = tl.load(inv_ptr + j // 2)
    angle = libdevice.mul_rn(pos, inv)
    other = tl.where(even, -swapped, swapped)
    rr = libdevice.add_rn(libdevice.mul_rn(xf, libdevice.cos(angle)),
                          libdevice.mul_rn(other, libdevice.sin(angle)))
    tl.store(out_ptr + row * (N + ROPE) + N + j, rr.to(xr.dtype).to(out_ptr.dtype.element_ty))


def latent_prep(kv_raw, weight, eps, positions, *, lora, rope_dim, theta):
    """`cat(rms_norm(kv_raw[:, :lora]), rope(kv_raw[:, lora:lora + rope_dim]))` for the
    decode family's few rows in one launch, or None when not that shape."""
    rows = int(kv_raw.shape[0]) if kv_raw.dim() == 2 else -1
    if rows < 1 or rows > 4 or kv_raw.stride(-1) != 1 or int(kv_raw.shape[1]) != lora + rope_dim:
        return None
    bw = _torch_mean_block_width(rows, int(lora))
    if bw is None or rope_dim & (rope_dim - 1) or int(positions.numel()) != rows:
        return None
    out = torch.empty((rows, lora + rope_dim), dtype=kv_raw.dtype, device=kv_raw.device)
    _latent_prep_kernel[(rows,)](kv_raw, weight.contiguous(), positions.reshape(-1),
                                 _inv_freq(int(rope_dim), theta, kv_raw.device), out, kv_raw.stride(0),
                                 float(eps), _mean_factor(rows, int(lora)), BW=bw,
                                 K=int(lora) // (4 * bw), ROPE=int(rope_dim), num_warps=4,
                                 enable_fp_fusion=False)
    return out


# --------------------------------------------------------------------------
# Round 15 (numerics class): the W_UV expansion of the folded attention output
# and the sigmoid gate in ONE launch for the decode family: per (row, head,
# 32 value columns) the BF16 output row and W_UV rows are widened (exact),
# their products (exact in FP32) summed in FP32 (Triton's order, not cuBLAS
# SGEMM's), rounded once to BF16 (= the einsum's `.to(bf16)`), then gated with
# torch's sigmoid / product rounding (exact; libdevice without FTZ). Replaces
# out.float(), the per-step FP32 copy of the W_UV shard, the GEMM, the cast
# and the gate kernel.
# --------------------------------------------------------------------------
@triton.jit
def _expand_gate_kernel(o_ptr, w_ptr, g_ptr, out_ptr, HEADS, V, D, o_sr, o_sh, w_sh, w_sv, g_stride,
                        BLOCK_V: tl.constexpr, BLOCK_D: tl.constexpr):
    rh = tl.program_id(0)
    vb = tl.program_id(1)
    r = (rh // HEADS).to(tl.int64)
    h = rh % HEADS
    d = tl.arange(0, BLOCK_D)
    v = vb * BLOCK_V + tl.arange(0, BLOCK_V)
    o = tl.load(o_ptr + r * o_sr + h * o_sh + d, mask=d < D, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + h.to(tl.int64) * w_sh + v[:, None].to(tl.int64) * w_sv + d[None, :],
                mask=(v[:, None] < V) & (d[None, :] < D), other=0.0).to(tl.float32)
    e = tl.sum(w * o[None, :], axis=1).to(tl.bfloat16).to(tl.float32)
    col = h * V + v
    g = tl.load(g_ptr + r * g_stride + col, mask=v < V, other=0.0).to(tl.float32)
    s = libdevice.div_rn(1.0, 1.0 + libdevice.exp(-g)).to(tl.bfloat16).to(tl.float32)
    tl.store(out_ptr + r * (HEADS * V) + col, libdevice.mul_rn(e, s).to(tl.bfloat16), mask=v < V)


def expand_gate(out, wuv, gate_raw, *, block_v=32):
    """`(einsum("rhd,hvd->rhv", out.float(), wuv.float()).to(bf16) * sigmoid(gate)).reshape(rows, -1)`
    up to the expansion's FP32 summation order, for BF16 operands; None if not that shape."""
    if out.dim() != 3 or wuv.dim() != 3 or out.dtype != torch.bfloat16 or wuv.dtype != torch.bfloat16:
        return None
    rows, heads, D = (int(v) for v in out.shape)
    V = int(wuv.shape[1])
    if (int(wuv.shape[0]) != heads or int(wuv.shape[2]) != D or wuv.stride(2) != 1
            or out.stride(2) != 1 or D & (D - 1) or rows < 1 or rows > 16):
        return None
    g = gate_raw.reshape(rows, -1)
    if int(g.shape[1]) != heads * V or g.stride(-1) != 1 or g.dtype != torch.bfloat16:
        return None
    flat = torch.empty((rows, heads * V), dtype=torch.bfloat16, device=out.device)
    _expand_gate_kernel[(rows * heads, triton.cdiv(V, block_v))](
        out, wuv, g, flat, heads, V, D, out.stride(0), out.stride(1), wuv.stride(0), wuv.stride(1),
        g.stride(0),
        BLOCK_V=block_v, BLOCK_D=D, num_warps=4, enable_reflect_ftz=False)
    return flat


# --------------------------------------------------------------------------
# Round 16 (numerics class): the lightning indexer's scores for the decode
# family in two launches. (1) index weights `((x @ Wp^T) * a) * b` with FP32
# sums; (2) per (row, key block): keys widened in registers, the 32-head query
# against them as a BF16 tensor-core dot (the query is BF16-exact: exact
# products, FP32 accumulation), relu, the head-weighted sum, the causal mask. Replaces the x / key casts, the weights GEMV, two scalings, the score
# SGEMM + its split-K reduce, relu, the head-sum GEMV and the mask. The
# products are FP32 as before; only summation orders differ (cuBLAS -> Triton).
# --------------------------------------------------------------------------
@triton.jit
def _index_weights_kernel(x_ptr, w_ptr, out_ptr, K, x_stride, a, b, H, BLOCK_K: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1)
    acc = tl.zeros([BLOCK_K], dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        kk = k0 + tl.arange(0, BLOCK_K)
        kin = kk < K
        xv = tl.load(x_ptr + r * x_stride + kk, mask=kin, other=0.0).to(tl.float32)
        wv = tl.load(w_ptr + h.to(tl.int64) * K + kk, mask=kin, other=0.0)
        acc += wv * xv
    tl.store(out_ptr + r * H + h, (tl.sum(acc, axis=0) * a) * b)


@triton.jit
def _index_scores_kernel(q_ptr, wt_ptr, key_ptr, owned_ptr, out_ptr, width, out_stride, base_ptr,
                         H: tl.constexpr, D: tl.constexpr, BLOCK_W: tl.constexpr,
                         PER_ROW_BASE: tl.constexpr):
    # The rotated query is BF16-exact (the BF16 rope output widened) and the keys
    # are BF16, so the BF16 tensor-core dot forms exactly the FP32 GEMM's products;
    # its FP32 accumulation is the MMA's (logits within ~8e-7 of FP64 vs cuBLAS's
    # ~3.5e-7; 0 of 2,520 top-2048 selections changed at 16k / 64k). Rejected
    # forms: an IEEE-FP32 tl.dot spilled ~1 KB per thread (a spilling kernel grows
    # the driver's GPU-wide local-memory reservation: 0.93 GB less KV pool), a
    # per-head loop and CUDA-core rank-1 updates were latency-bound (22-37 us).
    r = tl.program_id(0).to(tl.int64)
    wb = tl.program_id(1)
    h = tl.arange(0, H)
    d = tl.arange(0, D)
    w = wb * BLOCK_W + tl.arange(0, BLOCK_W)
    q = tl.load(q_ptr + r * (H * D) + h[:, None] * D + d[None, :]).to(tl.bfloat16)
    if PER_ROW_BASE:
        # batched decode: query row r scores its own request row's key segment
        key_ptr += tl.load(base_ptr + r) * D
    keys = tl.load(key_ptr + w[:, None].to(tl.int64) * D + d[None, :], mask=w[:, None] < width,
                   other=0.0)
    prod = tl.maximum(tl.dot(q, tl.trans(keys)), 0.0)
    wt = tl.load(wt_ptr + r * H + h)
    logit = tl.sum(wt[:, None] * prod, axis=0)
    owned = tl.load(owned_ptr + r)
    logit = tl.where(w < owned, logit, float("-inf"))
    tl.store(out_ptr + r * out_stride + w, logit, mask=w < width)


def index_scores_fused(q, x, weights_proj, key_rows, owned, *, width, scale_a, scale_b,
                       block_w=64, bases=None):
    """The masked indexer logits [rows, width] FP32 from the FP32 rotated query
    [rows, H, D], the layer input x [rows, K] (BF16), the FP32 weights_proj
    [H, K], the row's key segment key_rows [>= width, D] (BF16, contiguous) and
    the owned key counts [rows] (int64); None when not that shape. `bases`
    [rows] (int64): each query row's first key slot inside `key_rows` (query
    rows of different request rows); None: every row reads from slot 0."""
    if q.dim() != 3 or q.dtype != torch.float32 or not q.is_contiguous():
        return None
    rows, H, D = (int(v) for v in q.shape)
    if (rows < 1 or rows > 64 or H & (H - 1) or D & (D - 1) or H < 16 or D < 16
            or weights_proj.dtype != torch.float32 or not weights_proj.is_contiguous()
            or int(weights_proj.shape[0]) != H or key_rows.stride(-1) != 1 or key_rows.dtype != torch.bfloat16
            or int(key_rows.stride(0)) != D or int(owned.numel()) != rows or x.stride(-1) != 1):
        return None
    K = int(weights_proj.shape[1])
    wt = torch.empty((rows, H), dtype=torch.float32, device=q.device)
    _index_weights_kernel[(rows, H)](x, weights_proj, wt, K, x.stride(0), float(scale_a), float(scale_b),
                                     H, BLOCK_K=1024, num_warps=4)
    out = torch.empty((rows, width), dtype=torch.float32, device=q.device)
    if width:
        per_row = bases is not None
        _index_scores_kernel[(rows, triton.cdiv(width, block_w))](
            q, wt, key_rows, owned.reshape(-1), out, width, width,
            bases.reshape(-1) if per_row else owned, H=H, D=D, BLOCK_W=block_w,
            PER_ROW_BASE=per_row, num_warps=4)
    return out


_MEAN_FACTORS = {}


def _mean_factor(rows, n):
    """ATen's mean factor: static_cast<float>(num_outputs) / numel, an FP32 division."""
    key = (rows, n)
    value = _MEAN_FACTORS.get(key)
    if value is None:
        value = _MEAN_FACTORS[key] = float(torch.tensor(float(rows), dtype=torch.float32)
                                           / torch.tensor(float(rows * n), dtype=torch.float32))
    return value


def rms_norm_tree(x, weight, eps, out=None):
    """`rms_norm` in one launch when the decode-family shape's torch mean is the
    emulated tree; None otherwise (the caller falls back)."""
    N = int(x.shape[-1])
    flat = x.reshape(-1, N)
    rows = int(flat.shape[0])
    if not (1 <= rows <= 4) or flat.stride(-1) != 1 or not x.is_cuda:
        return None
    bw = _torch_mean_block_width(rows, N)
    if bw is None or not weight.is_contiguous():
        return None
    if out is None:
        target, out_stride = torch.empty((rows, N), dtype=x.dtype, device=x.device), N
    else:
        target, out_stride = out, _even_row_stride(out, tuple(x.shape), x.dtype)
    factor = _mean_factor(rows, N)
    # no FMA contraction: the square is rounded on its own (the square kernel's
    # store), then added (torch's reduce)
    _rms_tree_kernel[(rows,)](flat, weight, target, N, float(eps), factor, flat.stride(0), out_stride,
                              target, BW=bw, K=N // (4 * bw), STORE_MS=False, num_warps=4,
                              enable_fp_fusion=False)
    return target if out is not None else target.view(x.shape)


# --------------------------------------------------------------------------
# Round 15: the indexer key prep `rope(layer_norm(k.float()))` for the decode
# family's few rows in ONE launch (128-wide keys). Every statistic is ATen's
# own reduction tree: the mean (MeanOps, 4-wide vectors over 32 lanes, the
# shuffle fold 16..1, `* 1/128`) and the biased variance (WelfordOps, 2-wide
# vectors over 64 lanes: each lane combines its pair, then the shared fold 32
# and the shuffle fold 16..1). Every Welford combine joins equal counts, so
# its `nb / n` is exactly 0.5 and nvcc's FMA contraction cannot change a bit.
# The normalization, affine and rope are the torch ops' / rope kernel's own
# expressions (no contraction: launched with enable_fp_fusion=False).
# --------------------------------------------------------------------------
@triton.jit
def _welford_fold(mu, m2, nf, OFF: tl.constexpr):
    mu_lo, mu_hi = tl.split(tl.permute(tl.reshape(mu, [2, OFF]), [1, 0]))
    m2_lo, m2_hi = tl.split(tl.permute(tl.reshape(m2, [2, OFF]), [1, 0]))
    delta = mu_hi - mu_lo
    new_count = nf + nf
    nb = libdevice.div_rn(nf, new_count)
    return mu_lo + delta * nb, (m2_lo + m2_hi) + ((delta * delta) * nf) * nb, new_count


@triton.jit
def _index_key_prep_kernel(k_ptr, w_ptr, b_ptr, pos_ptr, inv_ptr, out_ptr, k_stride, eps, inv_n,
                           ROPE: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    base = k_ptr + row * k_stride
    # mean: MeanOps over 4-wide vectors, 32 lanes
    t = tl.arange(0, 32)
    x0 = tl.load(base + 4 * t + 0).to(tl.float32)
    x1 = tl.load(base + 4 * t + 1).to(tl.float32)
    x2 = tl.load(base + 4 * t + 2).to(tl.float32)
    x3 = tl.load(base + 4 * t + 3).to(tl.float32)
    v = ((x0 + x1) + x2) + x3
    v = _halve(v, 16)
    v = _halve(v, 8)
    v = _halve(v, 4)
    v = _halve(v, 2)
    v = _halve(v, 1)
    mean = tl.sum(v, axis=0) * inv_n
    # biased variance: WelfordOps over 2-wide vectors, 64 lanes
    u = tl.arange(0, 64)
    e0 = tl.load(base + 2 * u + 0).to(tl.float32)
    e1 = tl.load(base + 2 * u + 1).to(tl.float32)
    d = e1 - e0
    mu = e0 + d * 0.5
    m2 = 0.0 + ((d * d) * 1.0) * 0.5          # (+0 + +0) + delta^2 * 1 * 0.5
    mu, m2, nf = _welford_fold(mu, m2, 2.0, 32)
    mu, m2, nf = _welford_fold(mu, m2, nf, 16)
    mu, m2, nf = _welford_fold(mu, m2, nf, 8)
    mu, m2, nf = _welford_fold(mu, m2, nf, 4)
    mu, m2, nf = _welford_fold(mu, m2, nf, 2)
    mu, m2, nf = _welford_fold(mu, m2, nf, 1)
    var = libdevice.div_rn(tl.sum(m2, axis=0), nf)
    # (xf - mean) * rsqrt(var + eps) * weight + bias
    c = tl.arange(0, 128)
    x = tl.load(base + c).to(tl.float32)
    r = libdevice.rsqrt(var + eps)
    y = (x - mean) * r
    y = y * tl.load(w_ptr + c) + tl.load(b_ptr + c)
    # interleaved rope on the last ROPE columns (the rope kernel's expressions)
    ev, od = tl.split(tl.reshape(y, [64, 2]))
    swapped = tl.reshape(tl.join(od, ev), [128])
    start = 128 - ROPE
    rot = c >= start
    jj = tl.where(rot, c - start, 0)
    even = (jj % 2) == 0
    p = tl.load(pos_ptr + row).to(tl.float32)
    inv = tl.load(inv_ptr + jj // 2, mask=rot, other=0.0)
    angle = libdevice.mul_rn(p, inv)
    cs = libdevice.cos(angle)
    sn = libdevice.sin(angle)
    other = tl.where(even, -swapped, swapped)
    rr = libdevice.add_rn(libdevice.mul_rn(y, cs), libdevice.mul_rn(other, sn))
    tl.store(out_ptr + row * 128 + c, tl.where(rot, rr, y))


def index_key_prep(k_raw, weight, bias, eps, positions, *, rope_dim, theta):
    """`rope_interleaved(layer_norm(k_raw.float(), weight, bias, eps), positions)`
    for <= 8 rows of 128-wide keys (FP32 out), or None when not that shape."""
    rows = int(k_raw.shape[0]) if k_raw.dim() == 2 else -1
    if (rows < 1 or rows > 8 or int(k_raw.shape[1]) != 128 or k_raw.stride(-1) != 1
            or int(rope_dim) != 64 or weight.dtype != torch.float32 or bias.dtype != torch.float32
            or int(positions.numel()) != rows):
        return None
    out = torch.empty((rows, 128), dtype=torch.float32, device=k_raw.device)
    inv = _inv_freq(int(rope_dim), theta, k_raw.device)
    _index_key_prep_kernel[(rows,)](k_raw, weight.contiguous(), bias.contiguous(),
                                    positions.reshape(-1), inv, out, k_raw.stride(0), float(eps),
                                    _mean_factor(rows, 128), ROPE=int(rope_dim), num_warps=1,
                                    enable_fp_fusion=False)
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
def _rms_tail_kernel(x_ptr, ms_ptr, w_ptr, out_ptr, N, eps, out_stride, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    blk = tl.program_id(1)
    offs = blk * BLOCK + tl.arange(0, BLOCK)
    live = offs < N
    x = tl.load(x_ptr + row * N + offs, mask=live, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + offs, mask=live, other=0.0).to(tl.float32)
    r = libdevice.rsqrt(libdevice.add_rn(tl.load(ms_ptr + row), eps))
    y = libdevice.mul_rn(w, libdevice.mul_rn(x, r))
    tl.store(out_ptr + row * out_stride + offs, y.to(out_ptr.dtype.element_ty), mask=live)


@triton.jit
def _square_f32_kernel(x_ptr, out_ptr, N, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(x_ptr + i, mask=i < N).to(tl.float32)
    tl.store(out_ptr + i, v * v, mask=i < N)


@triton.jit
def _rms_rows_kernel(x_ptr, w_ptr, out_ptr, N, eps, x_stride, out_stride, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    live = offs < N
    x = tl.load(x_ptr + row * x_stride + offs, mask=live, other=0.0).to(tl.float32)
    ms = tl.sum(x * x, axis=0) / N
    r = libdevice.rsqrt(libdevice.add_rn(ms, eps))
    w = tl.load(w_ptr + offs, mask=live, other=0.0).to(tl.float32)
    y = libdevice.mul_rn(w, libdevice.mul_rn(x, r))
    tl.store(out_ptr + row * out_stride + offs, y.to(out_ptr.dtype.element_ty), mask=live)


def rms_norm_rows(x, weight, eps, out=None):
    """c8 campaign L8: `rms_norm` over any row count in one launch — the mean of
    squares is an in-kernel FP32 tree sum (not torch's reduction order), the
    tail the same `(w * (x * rsqrt(ms + eps))).to(dtype)`. None when the row
    does not fit one block (the caller falls back)."""
    N = int(x.shape[-1])
    if N > 8192 or not x.is_cuda or not weight.is_contiguous():
        return None
    flat = x.reshape(-1, N)
    if flat.stride(-1) != 1:
        flat = flat.contiguous()
    rows = int(flat.shape[0])
    if out is None:
        target, out_stride = torch.empty((rows, N), dtype=x.dtype, device=x.device), N
    else:
        target, out_stride = out, _even_row_stride(out, tuple(x.shape), x.dtype)
    if rows:
        block = triton.next_power_of_2(N)
        _rms_rows_kernel[(rows,)](flat, weight, target, N, float(eps), flat.stride(0), out_stride,
                                  BLOCK=block, num_warps=8 if block >= 4096 else 4)
    return target if out is not None else target.view(x.shape)


def rms_norm(x, weight, eps, out=None):
    """`attention.rms_norm` with the elementwise tail in one kernel (bitwise the
    same): the mean of squares stays torch's own reduction (its order), then
    `(w * (x * rsqrt(ms + eps))).to(x.dtype)` runs fused. `out` (round 15): a
    caller's evenly row-strided view of x's shape to write into."""
    N = int(x.shape[-1])
    if x.is_cuda and 1 <= x.numel() // max(N, 1) <= 4:
        from .runtime import SWITCHES

        if SWITCHES.get("rms_tree") == "on":
            done = rms_norm_tree(x if x.reshape(-1, N).stride(-1) == 1 else x.contiguous(),
                                 weight, eps, out=out)
            if done is not None:
                return done
    if x.is_cuda and x.numel() // max(N, 1) > 4:
        from .runtime import SWITCHES

        if SWITCHES.get("rms_rows") == "on":
            done = rms_norm_rows(x, weight, eps, out=out)
            if done is not None:
                return done
    flat = x.reshape(-1, N)
    if not flat.is_contiguous():
        flat = flat.contiguous()
    if flat.is_cuda:
        sq = torch.empty(flat.shape, dtype=torch.float32, device=flat.device)
        n_el = int(flat.numel())
        if n_el:
            _square_f32_kernel[(triton.cdiv(n_el, 4096),)](flat, sq, n_el, BLOCK=4096)
        ms = sq.mean(-1)
    else:
        ms = flat.float().square().mean(-1)
    rows = int(flat.shape[0])
    if out is None:
        target, out_stride = torch.empty_like(flat), N
    else:
        target, out_stride = out, _even_row_stride(out, tuple(x.shape), x.dtype)
    if rows:
        block = min(2048, triton.next_power_of_2(N))
        _rms_tail_kernel[(rows, triton.cdiv(N, block))](flat, ms, weight.contiguous(), target, N,
                                                        float(eps), out_stride, BLOCK=block)
    return target if out is not None else target.view(x.shape)


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
def _ihc_pp_chunks_rows_kernel(x_ptr, residual_ptr, prev_post_ptr, fn_ptr, out_ptr, part_ptr,
                               H, K, NC, ROWS, BLOCK_K: tl.constexpr, R: tl.constexpr):
    """`_ihc_pp_chunks_kernel` for R rows of one chunk per program: the chunk's
    eight `fn` vectors are loaded once instead of once per row (32 KB per row at
    BLOCK_K 1,024); each row's sums are the same 1-D reductions — the same bits."""
    t0 = tl.program_id(0).to(tl.int64) * R
    c = tl.program_id(1)
    k = c * BLOCK_K
    offs = k + tl.arange(0, BLOCK_K)
    si = offs // H
    d = offs - si * H
    f0 = tl.load(fn_ptr + 0 * K + offs).to(tl.float32)
    f1 = tl.load(fn_ptr + 1 * K + offs).to(tl.float32)
    f2 = tl.load(fn_ptr + 2 * K + offs).to(tl.float32)
    f3 = tl.load(fn_ptr + 3 * K + offs).to(tl.float32)
    f4 = tl.load(fn_ptr + 4 * K + offs).to(tl.float32)
    f5 = tl.load(fn_ptr + 5 * K + offs).to(tl.float32)
    f6 = tl.load(fn_ptr + 6 * K + offs).to(tl.float32)
    f7 = tl.load(fn_ptr + 7 * K + offs).to(tl.float32)
    for r in tl.static_range(R):
        t = t0 + r
        if t < ROWS:
            rv = tl.load(residual_ptr + t * K + offs).to(tl.float32)
            xv = tl.load(x_ptr + t * H + d).to(tl.float32)
            p = tl.load(prev_post_ptr + t * 4 + si)
            ob = (p * xv + rv).to(tl.bfloat16).to(tl.float32)
            tl.store(out_ptr + t * K + offs, ob.to(tl.bfloat16))
            base = part_ptr + (t * NC + c) * 9
            tl.store(base + 0, tl.sum(ob * ob))
            tl.store(base + 1, tl.sum(ob * f0))
            tl.store(base + 2, tl.sum(ob * f1))
            tl.store(base + 3, tl.sum(ob * f2))
            tl.store(base + 4, tl.sum(ob * f3))
            tl.store(base + 5, tl.sum(ob * f4))
            tl.store(base + 6, tl.sum(ob * f5))
            tl.store(base + 7, tl.sum(ob * f6))
            tl.store(base + 8, tl.sum(ob * f7))


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



@triton.jit
def _ihc_pp_y_block(out_ptr, t, K, H, offs, pre0, pre1, pre2, pre3):
    o0 = tl.load(out_ptr + t * K + 0 * H + offs).to(tl.float32)
    o1 = tl.load(out_ptr + t * K + 1 * H + offs).to(tl.float32)
    o2 = tl.load(out_ptr + t * K + 2 * H + offs).to(tl.float32)
    o3 = tl.load(out_ptr + t * K + 3 * H + offs).to(tl.float32)
    y = pre0 * o0 + pre1 * o1 + pre2 * o2 + pre3 * o3
    return y.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _ihc_pp_collapse_norm_kernel(part_ptr, scale_ptr, base_ptr, out_ptr, post_ptr, y_ptr,
                                 norm_weight_ptr, H, K, NC, magnitude, hc_eps, norm_eps,
                                 layernorm_eps, ND: tl.constexpr, BLOCK_D: tl.constexpr):
    """Round 14: `_ihc_pp_collapse_kernel` (FUSE_NORM) + `_ihc_pp_norm_kernel` in
    one program per row. The pair recomputes the gates in each of its 2 x ND
    programs per row and reads the four streams twice; here the gates run once,
    pass 1 reads the streams once, stores the BF16 collapse and sums each
    block's squares (the collapse kernel's expression and `tl.sum`, the norm
    kernel's sequential order from 0.0), pass 2 rereads the BF16 collapse (the
    same value the pair recomputes) and stores the norm kernel's expression.
    Bit-identical by construction at the same BLOCK_D / num_warps."""
    t = tl.program_id(0).to(tl.int64)
    pre0, pre1, pre2, pre3, post0, post1, post2, post3 = _ihc_pp_gates(
        part_ptr, scale_ptr, base_ptr, t, K, NC, magnitude, hc_eps, norm_eps)
    tl.store(post_ptr + t * 4 + 0, post0)
    tl.store(post_ptr + t * 4 + 1, post1)
    tl.store(post_ptr + t * 4 + 2, post2)
    tl.store(post_ptr + t * 4 + 3, post3)
    sqy = 0.0
    for j in tl.static_range(ND):
        offs = j * BLOCK_D + tl.arange(0, BLOCK_D)
        yb = _ihc_pp_y_block(out_ptr, t, K, H, offs, pre0, pre1, pre2, pre3)
        tl.store(y_ptr + t * H + offs, yb.to(tl.bfloat16))
        sqy += tl.sum(yb * yb)
    rmsy = tl.rsqrt(sqy / H + layernorm_eps)
    tl.debug_barrier()
    for j in tl.static_range(ND):
        offs = j * BLOCK_D + tl.arange(0, BLOCK_D)
        yb = tl.load(y_ptr + t * H + offs).to(tl.float32)
        tl.store(y_ptr + t * H + offs, (yb * rmsy * tl.load(norm_weight_ptr + offs)).to(tl.bfloat16))


IHC_FUSE_MIN_ROWS = 512


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
    from .runtime import SWITCHES

    per = 8 if rows >= 512 and SWITCHES.get("ihc_rows") == "8" else 1
    if per == 1 and 2 <= rows < 512 and SWITCHES.get("ihc_small") == "on":
        # c8 campaign: 2 rows per program at decode row counts (fn read once per two
        # rows; bit-identical; 16.77 -> 16.48 ms per 8-row step, 4/8 rows per program lose)
        per = 2
    if per > 1:
        # consult18 P9: 8 rows per program (bit-identical; 63 -> 50 us at 1,024 rows)
        _ihc_pp_chunks_rows_kernel[(triton.cdiv(rows, per), nc)](
            x, residual, prev_post, fn, out, parts, hidden, k_all, nc, rows, BLOCK_K=block_k, R=per,
            num_warps=num_warps)
    else:
        _ihc_pp_chunks_kernel[(rows, nc)](x, residual, prev_post, fn, out, parts, hidden, k_all, nc,
                                          BLOCK_K=block_k, num_warps=num_warps)
    if norm_weight is not None and rows >= IHC_FUSE_MIN_ROWS and SWITCHES.get("ihc_fuse") == "on":
        # round 14: collapse + norm in one program per row (bit-identical;
        # 99 -> 29 us at 2,048 rows; slower below a few hundred rows)
        _ihc_pp_collapse_norm_kernel[(rows,)](parts, hc_scale, hc_base, out, post, y, norm_weight,
                                              hidden, k_all, nc, float(magnitude), float(hc_eps),
                                              float(norm_eps), float(layernorm_eps), ND=nd,
                                              BLOCK_D=block_d, num_warps=num_warps)
        return out, post, y
    sqy = torch.empty((rows, nd), dtype=torch.float32, device=residual.device)
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


# --------------------------------------------------------------------------
# consult18 P2: the scores-only selection (collectives.scores_only_threshold +
# scores_only_flags + select_own_candidates, sync-free form) in ONE launch per
# row. Exact for candidate rows as `local_candidates` produces them — scores
# descending, tied scores in ascending position (torch.topk sorted: stable
# radix sort of the index-ordered gather), -inf / -1 padding last: then the
# selected set is a prefix (every key above the global k-th offered score,
# then the first `mine` keys of the tied plateau) and the final stable sort is
# the identity, so values[:n] / positions[:n] then -inf / -1 IS the reference
# output. The program also checks that ordering on device: a row that breaks
# it (or carries a NaN) raises the deferred flag, and the caller reruns the
# forward on the exact path — as for a saturated offer.
# --------------------------------------------------------------------------
@triton.jit
def _scores_only_select_kernel(offers_ptr, cand_ptr, pos_ptr, out_val_ptr, out_pos_ptr, flag_ptr,
                               sel_ptr, cnt_ptr, base_ptr, LCAP,
                               WIDTH, TOPK, KIDX, RANK, OFFER: tl.constexpr, WORLD: tl.constexpr,
                               NOFF: tl.constexpr, NCAND: tl.constexpr, HANDBACK: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)
    j = tl.arange(0, NOFF)
    stacked = tl.load(offers_ptr + r * (WORLD * OFFER) + j, mask=j < WORLD * OFFER,
                      other=float("-inf"))
    finite_s = (stacked == stacked) & (stacked != float("inf")) & (stacked != float("-inf"))
    total = tl.sum(finite_s.to(tl.int32))
    ordered = tl.sort(stacked, descending=True)
    kth = tl.max(tl.where(j == KIDX, ordered, float("-inf")))
    limit = tl.where(total > TOPK, kth, float("-inf"))
    above = tl.sum((stacked > limit).to(tl.int32))
    live = limit != float("-inf")
    owner = j // OFFER
    earlier = tl.sum(((owner < RANK) & (stacked == limit) & live).to(tl.int32))
    mine = tl.maximum(TOPK - above - earlier, 0)
    # saturation: some rank's last offered score is finite and reaches the cut
    last = (j % OFFER) == OFFER - 1
    # (the reference checks the offers only when they are narrower than the row)
    sat = (tl.sum((last & finite_s & (stacked >= limit)).to(tl.int32)) > 0) & (OFFER < WIDTH)
    nan = tl.sum((stacked != stacked).to(tl.int32)) > 0

    i = tl.arange(0, NCAND)
    inside = i < WIDTH
    v = tl.load(cand_ptr + r * WIDTH + i, mask=inside, other=float("-inf"))
    p = tl.load(pos_ptr + r * WIDTH + i, mask=inside, other=-1)
    finite_v = (v == v) & (v != float("inf")) & (v != float("-inf"))
    n_gt = tl.sum((finite_v & (v > limit)).to(tl.int32))
    n_pl = tl.sum((finite_v & (v == limit)).to(tl.int32))
    n = n_gt + tl.minimum(n_pl, mine)
    # truncation: this rank's first un-offered candidate reaches the cut
    trunc = tl.sum(((i == OFFER) & inside & finite_v & (v >= limit)).to(tl.int32)) > 0
    # the input contract: descending scores, ties in ascending position, no NaN
    v_next = tl.load(cand_ptr + r * WIDTH + i + 1, mask=i + 1 < WIDTH, other=float("-inf"))
    p_next = tl.load(pos_ptr + r * WIDTH + i + 1, mask=i + 1 < WIDTH, other=-1)
    pair = (i + 1) < WIDTH
    broken = pair & ((v < v_next) | ((v == v_next) & finite_v & (p >= p_next)))
    broken = broken | (inside & (v != v))
    bad = tl.sum(broken.to(tl.int32)) > 0
    keep = i < n
    out_mask = i < TOPK
    merged = tl.where(keep, p, -1)
    if HANDBACK:
        # the decoder's position -> slot map + b12x selection_handback_reference:
        # owned, in-row entries become `p // world + row_base`, compacted
        # stably to the front, -1 padding, live count.
        pp = merged.to(tl.int64)
        local = tl.where(pp >= 0, pp, 0) // WORLD
        live = out_mask & (pp >= 0) & (local < LCAP) & ((pp % WORLD) == RANK)
        dest = tl.cumsum(live.to(tl.int32), axis=0) - 1
        count = tl.sum(live.to(tl.int32))
        base = tl.load(base_ptr + r)
        slot = (local + base).to(tl.int32)
        tl.store(sel_ptr + r * TOPK + dest, slot, mask=live)
        tl.store(sel_ptr + r * TOPK + i, tl.full((NCAND,), -1, tl.int32), mask=out_mask & (i >= count))
        tl.store(cnt_ptr + r, count)
    else:
        tl.store(out_val_ptr + r * TOPK + i, tl.where(keep, v, float("-inf")), mask=out_mask)
        tl.store(out_pos_ptr + r * TOPK + i, merged.to(tl.int32), mask=out_mask)
    tl.store(flag_ptr + r, (sat | trunc | bad | nan).to(tl.int8))


def _stacked_offers(offers, world):
    """`[rows, world, offer]` FP32 over the per-rank offers — a view when they
    are the rank slices of one gathered tensor (the DCP score gather), else a copy."""
    first = offers[0]
    rows, offer = (int(v) for v in first.shape)
    if (first.dtype == torch.float32 and all(o.dtype == torch.float32 for o in offers)
            and first.stride() == (world * offer, 1)
            and all(o.stride() == first.stride() and o.data_ptr() == first.data_ptr() + k * offer * 4
                    for k, o in enumerate(offers))):
        return torch.as_strided(first, (rows, world, offer), (world * offer, offer, 1))
    return torch.stack([o.float() for o in offers], dim=1).contiguous()


def scores_only_select(offers, candidate_scores, candidate_positions, *, topk, world, rank,
                       handback=None):
    """`(values [rows, topk] FP32, positions [rows, topk] int32, flag [] bool)` —
    the sync-free scores-only selection in one launch (see the kernel note).
    `offers`: the `world` per-rank `[rows, offer]` FP32 offers in rank order.
    `handback=(selected [rows, topk] int32, counts [rows] int32, row_base [rows]
    int64, local_capacity)`: write this rank's compacted local slots + counts
    (the decoder map + selection hand-back) instead; values/positions are None."""
    rows, width = (int(v) for v in candidate_scores.shape)
    offer = int(offers[0].shape[1])
    stacked = _stacked_offers(offers, world)
    scores = candidate_scores.float().contiguous()
    positions = candidate_positions.contiguous()
    flags = torch.empty((rows,), dtype=torch.int8, device=scores.device)
    if handback is None:
        out_val = torch.empty((rows, topk), dtype=torch.float32, device=scores.device)
        out_pos = torch.empty((rows, topk), dtype=torch.int32, device=scores.device)
        sel, cnt, base, lcap = out_pos, flags, flags, 0
    else:
        sel, cnt, base, lcap = handback
        out_val = out_pos = None
    noff = triton.next_power_of_2(world * offer)
    ncand = triton.next_power_of_2(max(width, topk))
    _scores_only_select_kernel[(rows,)](stacked, scores, positions,
                                        flags if out_val is None else out_val,
                                        sel if out_pos is None else out_pos, flags,
                                        sel, cnt, base, int(lcap),
                                        width, topk, min(topk, world * offer) - 1, rank,
                                        OFFER=offer, WORLD=world, NOFF=noff, NCAND=ncand,
                                        HANDBACK=handback is not None, num_warps=8)
    flag = flags[0] if rows == 1 else flags.any()
    return out_val, out_pos, flag


# --------------------------------------------------------------------------
# consult18 P2: the MoE route (moe._route_from_logits, n_group == 1, top-8 of
# 256, renormalised, scaled) in one launch per row. torch's behaviour pinned
# on the card (harness/route_probe.py): sigmoid = 1 / (1 + expf(-x)) (IEEE
# div); topk(sorted=False) returns the keys above the k-th in ascending index,
# then the k-th's ties in ascending index; the 8-way sum adds
# ((a0+a4)+(a2+a6))+((a1+a5)+(a3+a7)). Bit-identical (route_check.py).
# --------------------------------------------------------------------------
@triton.jit
def _moe_route_kernel(logits_ptr, bias_ptr, w_ptr, ids_ptr, E, scale, stride_r,
                      EB: tl.constexpr, K: tl.constexpr, NORM: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)
    e = tl.arange(0, EB)
    inside = e < E
    x = tl.load(logits_ptr + r * stride_r + e, mask=inside, other=0.0)
    score = libdevice.div_rn(1.0, 1.0 + libdevice.exp(-x))
    choice = score + tl.load(bias_ptr + e, mask=inside, other=0.0)
    choice = tl.where(inside, choice, float("-inf"))
    ordered = tl.sort(choice, descending=True)
    kth = tl.max(tl.where(e == K - 1, ordered, float("-inf")))
    gt = choice > kth
    eq = choice == kth
    n_gt = tl.sum(gt.to(tl.int32))
    eq_rank = tl.cumsum(eq.to(tl.int32), axis=0) - 1
    take = gt | (eq & (eq_rank < K - n_gt))
    # torch's order: the keys above the k-th in ascending index, then its ties
    dest = tl.where(gt, tl.cumsum(gt.to(tl.int32), axis=0) - 1, n_gt + eq_rank)
    a0 = tl.sum(tl.where(take & (dest == 0), score, 0.0))
    a1 = tl.sum(tl.where(take & (dest == 1), score, 0.0))
    a2 = tl.sum(tl.where(take & (dest == 2), score, 0.0))
    a3 = tl.sum(tl.where(take & (dest == 3), score, 0.0))
    a4 = tl.sum(tl.where(take & (dest == 4), score, 0.0))
    a5 = tl.sum(tl.where(take & (dest == 5), score, 0.0))
    a6 = tl.sum(tl.where(take & (dest == 6), score, 0.0))
    a7 = tl.sum(tl.where(take & (dest == 7), score, 0.0))
    if NORM:
        total = ((a0 + a4) + (a2 + a6)) + ((a1 + a5) + (a3 + a7))
        denom = total + 1e-20
        w = libdevice.div_rn(score, denom) * scale
    else:
        w = score * scale
    tl.store(w_ptr + r * K + dest, w, mask=take)
    tl.store(ids_ptr + r * K + dest, e.to(tl.int64), mask=take)


def moe_route(logits, bias, *, top_k, scale, norm, ids_dtype=torch.int64):
    """`(weights [rows, top_k] FP32, ids [rows, top_k] ids_dtype)` — moe._route_from_logits
    for n_group == 1 and top_k == 8 in one launch (bit-identical). Row-strided
    logits (unit column stride) are read in place (round 15)."""
    rows, experts = (int(v) for v in logits.shape)
    if top_k != 8:
        raise ValueError("moe_route covers the top-8 route (the 8-way sum order is pinned)")
    if logits.stride(-1) != 1:
        logits = logits.contiguous()
    w = torch.empty((rows, top_k), dtype=torch.float32, device=logits.device)
    ids = torch.empty((rows, top_k), dtype=ids_dtype, device=logits.device)
    _moe_route_kernel[(rows,)](logits, bias.contiguous(), w, ids, experts, float(scale),
                               int(logits.stride(0)) if rows else experts,
                               EB=triton.next_power_of_2(experts), K=top_k, NORM=bool(norm),
                               num_warps=4)
    return w, ids



# --------------------------------------------------------------------------
# Round 15 (decode kernel diet): elementwise chains in one launch each, every
# op in torch's own formula and rounding (exhaustively checked over all BF16
# inputs in /opt/h4/state/dev/diet_ops.py): sigmoid / silu are
# `1 / (1 + expf(-x))` and `x / (1 + expf(-x))` in FP32 (libdevice expf, IEEE
# division) rounded to BF16 once; products are FP32 products rounded once.
# Launched with `enable_reflect_ftz=False`: torch keeps FP32 subnormals, and
# Triton's default libdevice flush-to-zero differs there (sigmoid at x <= -87.5,
# silu of subnormal inputs).
# --------------------------------------------------------------------------
@triton.jit
def _sigmoid_gate_kernel(a_ptr, g_ptr, out_ptr, N, V, a_sr, a_sh, a_sv, g_stride,
                         BLOCK: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    live = offs < N
    g = tl.load(g_ptr + r * g_stride + offs, mask=live, other=0.0).to(tl.float32)
    s = libdevice.div_rn(1.0, 1.0 + libdevice.exp(-g)).to(tl.bfloat16).to(tl.float32)
    h = offs // V
    a = tl.load(a_ptr + r * a_sr + h * a_sh + (offs - h * V) * a_sv, mask=live,
                other=0.0).to(tl.float32)
    tl.store(out_ptr + r * N + offs, libdevice.mul_rn(a, s).to(tl.bfloat16), mask=live)


def sigmoid_gate(attn_output, gate_raw):
    """`(attn_output * sigmoid(gate_raw).view_as(attn_output)).reshape(rows, -1)`
    for BF16 operands, contiguous [rows, heads * v] out (bitwise the same);
    `attn_output` [rows, heads, v] at any strides (the expansion's permuted view)."""
    rows, heads, V = (int(v) for v in attn_output.shape)
    N = heads * V
    g = gate_raw.reshape(rows, -1)
    if int(g.shape[1]) != N or g.stride(-1) != 1:
        raise ValueError("sigmoid_gate needs a [rows, heads * v] gate with unit column stride")
    out = torch.empty((rows, N), dtype=torch.bfloat16, device=attn_output.device)
    if rows and N:
        block = min(4096, triton.next_power_of_2(N))
        sr, sh, sv = (int(v) for v in attn_output.stride())
        _sigmoid_gate_kernel[(rows, triton.cdiv(N, block))](attn_output, g, out, N, V, sr, sh, sv,
                                                            g.stride(0), BLOCK=block,
                                                            enable_reflect_ftz=False)
    return out


@triton.jit
def _swiglu_f32_kernel(g_ptr, u_ptr, out_ptr, N, g_stride, u_stride, BLOCK: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    live = offs < N
    g = tl.load(g_ptr + r * g_stride + offs, mask=live, other=0.0).to(tl.bfloat16).to(tl.float32)
    u = tl.load(u_ptr + r * u_stride + offs, mask=live, other=0.0).to(tl.bfloat16).to(tl.float32)
    act = libdevice.div_rn(g, 1.0 + libdevice.exp(-g)).to(tl.bfloat16).to(tl.float32)
    tl.store(out_ptr + r * N + offs, libdevice.mul_rn(act, u).to(tl.bfloat16), mask=live)


def swiglu_f32(gate32, up32):
    """`silu(gate32.to(bf16)) * up32.to(bf16)` (BF16 out, contiguous) from FP32
    row-strided views — the cast, the activation and the product in one launch."""
    rows, N = (int(v) for v in gate32.shape)
    if tuple(up32.shape) != (rows, N) or gate32.stride(-1) != 1 or up32.stride(-1) != 1:
        raise ValueError("swiglu_f32 needs two [rows, N] FP32 views with unit column stride")
    out = torch.empty((rows, N), dtype=torch.bfloat16, device=gate32.device)
    if rows and N:
        block = min(4096, triton.next_power_of_2(N))
        _swiglu_f32_kernel[(rows, triton.cdiv(N, block))](gate32, up32, out, N, gate32.stride(0),
                                                          up32.stride(0), BLOCK=block,
                                                          enable_reflect_ftz=False)
    return out


@triton.jit
def _candidate_pick_kernel(v_ptr, i_ptr, out_ptr, n, world, rank, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = offs < n
    v = tl.load(v_ptr + offs, mask=live, other=0.0)
    idx = tl.load(i_ptr + offs, mask=live, other=0)
    finite = (v == v) & (tl.abs(v) != float("inf"))
    tl.store(out_ptr + offs, tl.where(finite, idx * world + rank, -1).to(tl.int32), mask=live)


def candidate_pick(values, indices, *, world, rank):
    """`where(isfinite(values), columns[indices], -1).to(int32)` with
    columns[j] = j * world + rank (j < width) — the padded columns past the
    width only ever carry -inf scores, so the gather is the affine map."""
    v = values.contiguous()
    i = indices.contiguous()
    out = torch.empty(v.shape, dtype=torch.int32, device=v.device)
    n = int(v.numel())
    if n:
        block = 1024
        _candidate_pick_kernel[(triton.cdiv(n, block),)](v, i, out, n, int(world), int(rank),
                                                         BLOCK=block)
    return out


@triton.jit
def _store_owned_rows_kernel(table_ptr, values_ptr, kept_ptr, slots_ptr, safe_ptr, D, v_stride,
                             BLOCK_D: tl.constexpr):
    j = tl.program_id(0)
    if tl.load(slots_ptr + j) >= 0:
        k = tl.load(kept_ptr + j).to(tl.int64)
        dst = tl.load(safe_ptr + j).to(tl.int64)
        offs = tl.arange(0, BLOCK_D)
        live = offs < D
        v = tl.load(values_ptr + k * v_stride + offs, mask=live, other=0.0)
        tl.store(table_ptr + dst * D + offs, v.to(table_ptr.dtype.element_ty), mask=live)


def store_owned_rows(table, values, kept, slots, safe):
    """The decode-graph `_store`: rows with `slots >= 0` write
    `values[kept].to(table.dtype)` at `safe`; the others are left untouched (the
    original rewrites them with their own bytes) — the same table bytes."""
    D = int(table.shape[1])
    n = int(slots.numel())
    if values.stride(-1) != 1 or not table.is_contiguous():
        raise ValueError("store_owned_rows needs unit column strides and a contiguous table")
    if n:
        _store_owned_rows_kernel[(n,)](table, values, kept, slots, safe, D, values.stride(0),
                                       BLOCK_D=triton.next_power_of_2(D))


# --------------------------------------------------------------------------
# consult18 P3 (numerics class): INT8 wire for the CE head gather / LSE
# partials — symmetric INT8 per row (one FP32 scale per `width` values).
# --------------------------------------------------------------------------
@triton.jit
def _quant_rows_int8_kernel(x_ptr, q_ptr, s_ptr, N, W, BLOCK_W: tl.constexpr, ROWS: tl.constexpr):
    r = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    c = tl.arange(0, BLOCK_W)
    mask = (r[:, None] < N) & (c[None, :] < W)
    x = tl.load(x_ptr + r[:, None].to(tl.int64) * W + c[None, :], mask=mask, other=0.0).to(tl.float32)
    amax = tl.max(tl.abs(x), axis=1)
    scale = tl.where(amax > 0, amax / 127.0, 1.0)
    q = libdevice.rint(x / scale[:, None])
    q = tl.minimum(tl.maximum(q, -127.0), 127.0)
    tl.store(q_ptr + r[:, None].to(tl.int64) * W + c[None, :], q.to(tl.int8), mask=mask)
    tl.store(s_ptr + r, scale, mask=r < N)


@triton.jit
def _dequant_rows_int8_kernel(q_ptr, s_ptr, out_ptr, N, W, BLOCK_W: tl.constexpr, ROWS: tl.constexpr):
    r = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    c = tl.arange(0, BLOCK_W)
    mask = (r[:, None] < N) & (c[None, :] < W)
    q = tl.load(q_ptr + r[:, None].to(tl.int64) * W + c[None, :], mask=mask, other=0).to(tl.float32)
    s = tl.load(s_ptr + r, mask=r < N, other=0.0)
    tl.store(out_ptr + r[:, None].to(tl.int64) * W + c[None, :], (q * s[:, None]).to(tl.bfloat16),
             mask=mask)


def quant_rows_int8(x, q_out, s_out):
    """`x` [N, W] (contiguous) -> `q_out` int8 [N, W], `s_out` FP32 [N] (amax / 127)."""
    n, w = (int(v) for v in x.shape)
    rows = max(1, 4096 // triton.next_power_of_2(w))
    _quant_rows_int8_kernel[(triton.cdiv(n, rows),)](x, q_out, s_out, n, w,
                                                     BLOCK_W=triton.next_power_of_2(w), ROWS=rows,
                                                     num_warps=4)


def dequant_rows_int8(q, s, out):
    """int8 [N, W] x FP32 [N] -> BF16 `out` [N, W]."""
    n, w = (int(v) for v in q.shape)
    rows = max(1, 4096 // triton.next_power_of_2(w))
    _dequant_rows_int8_kernel[(triton.cdiv(n, rows),)](q, s, out, n, w,
                                                       BLOCK_W=triton.next_power_of_2(w), ROWS=rows,
                                                       num_warps=4)


@triton.jit
def _dequant_slots_kernel(base_ptr, src_ptr, dst_ptr, out_ptr, N, W, SLOT, SOFF, HPR, QROWS,
                          HEAD_MAJOR: tl.constexpr, BLOCK_W: tl.constexpr, ROWS: tl.constexpr):
    k = tl.program_id(1)
    slot = tl.load(src_ptr + k).to(tl.int64)
    dst = tl.load(dst_ptr + k).to(tl.int64)
    r = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    c = tl.arange(0, BLOCK_W)
    mask = (r[:, None] < N) & (c[None, :] < W)
    q_base = base_ptr + slot * SLOT
    q = tl.load(q_base + r[:, None].to(tl.int64) * W + c[None, :], mask=mask, other=0).to(tl.float32)
    s_base = (base_ptr + slot * SLOT + SOFF).to(tl.pointer_type(tl.float32))
    sc = tl.load(s_base + r, mask=r < N, other=0.0)
    if HEAD_MAJOR:
        # Option 1 (PERF-PLAN.md Step 3): head-major store — flat `r` is
        # `row * HPR + head_local` and lands at `[dst * HPR + head_local, row]`
        # of the contiguous `[world * HPR, QROWS, W]` gather target, so the
        # gather needs no permute/reshape copy after the wire.
        off = ((dst * HPR + r % HPR) * QROWS + r // HPR).to(tl.int64) * W
        tl.store(out_ptr + off[:, None] + c[None, :], (q * sc[:, None]).to(tl.bfloat16),
                 mask=mask)
    else:
        out = out_ptr + dst * N * W
        tl.store(out + r[:, None].to(tl.int64) * W + c[None, :], (q * sc[:, None]).to(tl.bfloat16),
                 mask=mask)


def dequant_slots_int8(slots, sources, soff, n, width, out, dests=None,
                       head_major=False, hpr=None, rows=None):
    """`slots` int8 [S, slot] (one inbound slot per source rank: n*width INT8
    values, then n FP32 scales at byte `soff`); `sources` int32 device index of
    the slots to read -> `out` BF16 [*, n, width] at positions `dests` (int32;
    default 0..k-1) — one launch.

    `head_major=True` (Option 1, PERF-PLAN.md Step 3): `out` is the contiguous
    head-major gather target `[world * hpr, rows, width]` (`hpr * rows == n`)
    and slot row `r` (`row * hpr + head_local`) stores at
    `[origin * hpr + head_local, row]` — the gather then needs no
    permute/reshape copy (and downstream `q_nope` slices keep disjoint batch
    strides)."""
    k = int(sources.numel())
    if dests is None:
        raise ValueError("dequant_slots_int8 needs its destination index (built outside "
                         "the overlapped loop)")
    if head_major and (hpr is None or rows is None or int(hpr) * int(rows) != int(n)):
        raise ValueError("dequant_slots_int8(head_major=True) needs hpr and rows "
                         "with hpr * rows == n")
    rows_per_block = max(1, 4096 // triton.next_power_of_2(width))
    _dequant_slots_kernel[(triton.cdiv(n, rows_per_block), k)](slots.view(torch.int8), sources,
                                                              dests, out, n, width,
                                                              int(slots.stride(0)), int(soff),
                                                              int(hpr or 1), int(rows or 1),
                                                              HEAD_MAJOR=bool(head_major),
                                                              BLOCK_W=triton.next_power_of_2(width),
                                                              ROWS=rows_per_block, num_warps=4)
