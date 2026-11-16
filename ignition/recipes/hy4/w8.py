# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0
"""consult18 P6: INT8 (W8A16) storage for the decode-hot dense projections.

A decode step reads ~8 GB of BF16 dense weights per rank, at the memory
roofline. The per-layer same-input pack (`q_a` + `kv_a` + gate + indexer `wk`,
~59 MB) and `o_proj` (~25 MB) are stored as symmetric INT8 with one FP32 scale
per (output row, 128-column group) — reconstruction error ~0.7 % RMS (FP8 E4M3
measured 2.6 % and failed the quality gate; INT8 group-128 passed it).

`Int8Weight` stands in for the BF16 tensor at its call sites (`x @ W.T`,
`W.rows`-style row slices, `.shape` / `.dtype`): up to `GEMV_ROWS` rows run a
Triton GEMV that dequantizes in registers (FP32 accumulation, half the bytes);
more rows (prefill) dequantize the weight to BF16 (one Triton pass) and use the
cuBLAS GEMM — the same values the quality gate measured.
"""
import contextlib
import contextvars

import torch

GROUP = 128
GEMV_ROWS = 16
# INT8 reads are scoped to the decode-family bodies (decode step, verify rows,
# draft forwards — DecodeGraphs) when the BF16 original is kept: every prefill
# call, short tails included, stays the original BF16 op.
_DECODE_SCOPE = contextvars.ContextVar("hy4_w8_decode_scope", default=False)


@contextlib.contextmanager
def decode_scope():
    token = _DECODE_SCOPE.set(True)
    try:
        yield
    finally:
        _DECODE_SCOPE.reset(token)


def in_decode_scope(x):
    """A decode-family call that may read the INT8 copies: CUDA, <= GEMV_ROWS rows,
    inside `decode_scope()`. `x` is the matmul's input as it multiplies the
    weight (rows = every dimension but the last)."""
    return (_DECODE_SCOPE.get() and x.is_cuda
            and int(x.numel() // max(int(x.shape[-1]), 1)) <= GEMV_ROWS)


def int8_rows_ok(weight, x2):
    """Whether this call reads the INT8 copy."""
    return (x2.is_cuda and x2.shape[0] <= GEMV_ROWS and weight.dtype == torch.bfloat16
            and (weight.bf16 is None or _DECODE_SCOPE.get()))


class Int8Weight:
    """`[N, K]` INT8 weight with FP32 scales `[N, K / GROUP]`; logical dtype BF16."""

    def __init__(self, q, scale, dtype=torch.bfloat16, bf16=None):
        self.q = q
        self.scale = scale
        self.dtype = dtype
        # consult18 P6: optional original weight for the many-row (prefill)
        # GEMMs — prefill then stays bit-exact; only <= GEMV_ROWS rows read INT8.
        self.bf16 = bf16

    @classmethod
    def quantize(cls, w, *, chunk=512, keep=False):
        n, k = (int(v) for v in w.shape)
        if k % GROUP:
            raise ValueError(f"INT8 weight: K={k} is not a multiple of {GROUP}")
        q = torch.empty((n, k), dtype=torch.int8, device=w.device)
        scale = torch.empty((n, k // GROUP), dtype=torch.float32, device=w.device)
        for lo in range(0, n, chunk):
            hi = min(lo + chunk, n)
            g = w[lo:hi].float().view(hi - lo, k // GROUP, GROUP)
            s = (g.abs().amax(dim=2, keepdim=True) / 127.0).clamp_min(1e-12)
            q[lo:hi] = torch.round(g / s).clamp(-127, 127).to(torch.int8).view(hi - lo, k)
            scale[lo:hi] = s.squeeze(2)
        return cls(q, scale, w.dtype, bf16=w if keep else None)

    @property
    def shape(self):
        return self.q.shape

    @property
    def device(self):
        return self.q.device

    @property
    def ndim(self):
        return 2

    def __getitem__(self, index):
        if not isinstance(index, slice):
            raise TypeError("Int8Weight supports row slices only")
        return Int8Weight(self.q[index], self.scale[index], self.dtype,
                          bf16=None if self.bf16 is None else self.bf16[index])

    @property
    def T(self):
        return _Int8Transposed(self)

    def dequant(self):
        """BF16 `[N, K]` (rounded once — the values the quality gate measured)."""
        if self.q.is_cuda and self.dtype == torch.bfloat16:
            return _dequant_cuda(self.q, self.scale)
        n, k = self.q.shape
        return (self.q.float().view(n, k // GROUP, GROUP) * self.scale.unsqueeze(2)).view(n, k).to(self.dtype)

    def linear(self, x):
        """`x @ W.T` for `x [..., K]` BF16 -> `[..., N]` BF16."""
        k = int(self.q.shape[1])
        lead = x.shape[:-1]
        x2 = x.reshape(-1, k)
        if int8_rows_ok(self, x2):
            out = _gemv_cuda(x2.to(torch.bfloat16).contiguous(), self.q, self.scale)
        elif self.bf16 is not None:
            return x @ self.bf16.T          # the original op, bit-exact
        else:
            out = torch.nn.functional.linear(x2.to(self.dtype), self.dequant())
        return out.reshape(*lead, int(self.q.shape[0]))


class _Int8Transposed:
    """`W.T` of an `Int8Weight`: `x @ W.T` (via `__rmatmul__`), `.to()` a no-op."""

    def __init__(self, weight):
        self.weight = weight

    def __rmatmul__(self, x):
        return self.weight.linear(x)

    def to(self, *args, **kwargs):
        return self


_KERNELS = {}


def _kernels():
    if _KERNELS:
        return _KERNELS
    import triton
    import triton.language as tl

    @triton.jit
    def gemv(x_ptr, w_ptr, s_ptr, o_ptr, M, N, K, stride_xm, stride_wn, stride_sn, stride_om,
             BLOCK_N: tl.constexpr, SUB: tl.constexpr, G: tl.constexpr, M_PAD: tl.constexpr):
        pid_n = tl.program_id(0)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rm = tl.arange(0, M_PAD)
        acc = tl.zeros((M_PAD, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, G * SUB):
            for j in tl.static_range(SUB):
                rk = k0 + j * G + tl.arange(0, G)
                w = tl.load(w_ptr + rn[:, None] * stride_wn + rk[None, :],
                            mask=(rn[:, None] < N) & (rk[None, :] < K), other=0).to(tl.bfloat16)
                x = tl.load(x_ptr + rm[:, None] * stride_xm + rk[None, :],
                            mask=(rm[:, None] < M) & (rk[None, :] < K), other=0.0)
                s = tl.load(s_ptr + rn * stride_sn + (k0 // G + j),
                            mask=(rn < N) & (k0 + j * G < K), other=0.0)
                acc += tl.dot(x, tl.trans(w)) * s[None, :]
        tl.store(o_ptr + rm[:, None] * stride_om + rn[None, :], acc.to(tl.bfloat16),
                 mask=(rm[:, None] < M) & (rn[None, :] < N))

    @triton.jit
    def dequant(w_ptr, s_ptr, o_ptr, N, K, stride_wn, stride_sn, stride_on,
                BLOCK_N: tl.constexpr, G: tl.constexpr):
        pid_n = tl.program_id(0)
        g = tl.program_id(1)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rk = g * G + tl.arange(0, G)
        mask = (rn[:, None] < N) & (rk[None, :] < K)
        w = tl.load(w_ptr + rn[:, None] * stride_wn + rk[None, :], mask=mask, other=0).to(tl.float32)
        s = tl.load(s_ptr + rn * stride_sn + g, mask=rn < N, other=0.0)
        tl.store(o_ptr + rn[:, None] * stride_on + rk[None, :], (w * s[:, None]).to(tl.bfloat16),
                 mask=mask)

    _KERNELS.update(gemv=gemv, dequant=dequant, cdiv=triton.cdiv)
    return _KERNELS


def _gemv_cuda(x, q, scale):
    k = _kernels()
    m, kk = x.shape
    n = int(q.shape[0])
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    # dg31c sweep: the wide-N o_proj shape prefers 64-row tiles with 8 warps
    block_n, warps = (64, 8) if n >= 6144 and kk <= 4096 else (32, 4)
    k["gemv"][(k["cdiv"](n, block_n),)](
        x, q, scale, out, m, n, kk, x.stride(0), q.stride(0), scale.stride(0), out.stride(0),
        BLOCK_N=block_n, SUB=4 if kk % (4 * GROUP) == 0 else 1, G=GROUP, M_PAD=16,
        num_warps=warps, num_stages=3)
    return out


def _dequant_cuda(q, scale):
    k = _kernels()
    n, kk = (int(v) for v in q.shape)
    out = torch.empty((n, kk), dtype=torch.bfloat16, device=q.device)
    k["dequant"][(k["cdiv"](n, 64), kk // GROUP)](
        q, scale, out, n, kk, q.stride(0), scale.stride(0), out.stride(0), BLOCK_N=64, G=GROUP,
        num_warps=4)
    return out


def quantize_attention(attention, *, keep_bf16=False):
    """With `keep_bf16`, the BF16 fields stay exactly as loaded and the INT8 copies
    go to `w8_pack` / `w8_o_proj` (read only inside `decode_scope()`): every other
    call is the original code path. Without it, INT8 replaces the BF16 storage."""
    if keep_bf16:
        from dataclasses import replace as _replace

        w = attention.weights
        packed = w.packed_input
        pack8 = (Int8Weight.quantize(packed.packed)
                 if packed is not None and packed.packed.shape[1] % GROUP == 0 else None)
        o8 = Int8Weight.quantize(w.o_proj) if w.o_proj.shape[1] % GROUP == 0 else None
        attention.weights = _replace(w, w8_pack=pack8, w8_o_proj=o8)
        return 0
    return _quantize_replacing(attention)


def _quantize_replacing(attention):
    """Replace one attention's same-input pack and `o_proj` with INT8 storage;
    the loose same-input weights become row views of the INT8 pack (no BF16
    reference remains). Returns the BF16 bytes released."""
    from dataclasses import replace

    w = attention.weights
    packed = w.packed_input
    released = 0
    if (packed is not None and not isinstance(packed.packed, Int8Weight)
            and packed.packed.shape[1] % GROUP == 0):
        released += packed.packed.numel() * packed.packed.element_size()
        q_pack = replace(packed, packed=Int8Weight.quantize(packed.packed))
        w = replace(w, packed_input=q_pack, q_a_proj=q_pack.rows("q_a_proj"),
                    kv_a_proj=q_pack.rows("kv_a_proj"), gate_proj=q_pack.rows("linear_gate"),
                    indexer=(None if w.indexer is None
                             else replace(w.indexer, wk=q_pack.rows("indexer_wk"))))
    if not isinstance(w.o_proj, Int8Weight) and w.o_proj.shape[1] % GROUP == 0:
        released += w.o_proj.numel() * w.o_proj.element_size()
        w = replace(w, o_proj=Int8Weight.quantize(w.o_proj))
    attention.weights = w
    return released


class Bf16HeadView:
    """consult18 P9: the FP32 `lm_head` stand-in over its BF16 checkpoint values
    (the FP32 head is their exact upcast): `x @ view.T` is cuBLAS BF16 x BF16
    with FP32 accumulate and output — the FP32 head's products, summation order
    aside (bit-identical at one row, round-7 head_check). Frees the FP32 copy
    (354 MB per rank); `.shape` / `.dtype` / `.device` read like the FP32 head."""

    def __init__(self, head16):
        self.head16 = head16
        self.shape = head16.shape
        self.dtype = torch.float32
        self.device = head16.device

    @property
    def T(self):
        return _Bf16HeadTransposed(self.head16)


class _Bf16HeadTransposed:
    def __init__(self, head16):
        self.head16 = head16

    def __rmatmul__(self, x):
        lead = x.shape[:-1]
        x2 = x.reshape(-1, x.shape[-1])
        if x2.dtype != torch.bfloat16:
            x2 = x2.to(torch.bfloat16)      # the BF16 hidden's FP32 upcast: exact
        out = torch.mm(x2, self.head16.T, out_dtype=torch.float32)
        return out.reshape(*lead, int(self.head16.shape[0]))
