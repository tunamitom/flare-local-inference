"""Small-row GEMV for BF16 activations against 128x128 block-FP8 weights.

Activations stay BF16 (no activation quantization).  Each program owns
``block_n`` output features and a contiguous range of 128-wide K blocks.  An
E4M3 weight tile converts to BF16 exactly, so each block's tensor-core dot is
BF16 activations times the unscaled FP8 weights accumulated in FP32; the
block's FP32 scale then applies to that FP32 sum.  The only rounding beyond an
FP32 reference over dequantized weights is the final BF16 store.

With ``split`` > 1 the K blocks divide among ``split`` programs per feature
tile.  Each writes an FP32 partial, and the last program to arrive (counted
per tile) sums the partials in fixed split order, stores BF16 and re-arms the
counter, so the result is deterministic and the launch stays single-kernel and
graph-capturable.
"""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

MAX_ROWS = 32
BLOCK = 128


@triton.jit
def _w8a16_gemv(x, w, scales, out, partials, counters, rows,
                N: tl.constexpr, K: tl.constexpr, BLOCK_M: tl.constexpr,
                BLOCK_N: tl.constexpr, SPLIT: tl.constexpr,
                BLOCKS_PER_SPLIT: tl.constexpr, GUARDED: tl.constexpr):
    tile = tl.program_id(0)
    part = tl.program_id(1)
    KB: tl.constexpr = K // 128
    offs_n = tile * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, 128)
    n_mask = offs_n < N
    m_mask = offs_m < rows
    # Rows are clamped rather than masked so loads stay unpredicated.
    w_rows = tl.minimum(offs_n, N - 1)
    x_rows = tl.minimum(offs_m, rows - 1)
    w_ptr = w + w_rows[:, None].to(tl.int64) * K + offs_k[None, :]
    x_ptr = x + x_rows[:, None] * K + offs_k[None, :]
    # A tile of at most 128 features never straddles a scale row.
    scale_ptr = scales + ((tile * BLOCK_N) // 128) * KB
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    first = part * BLOCKS_PER_SPLIT
    for step in tl.range(0, BLOCKS_PER_SPLIT):
        kb = first + step
        if GUARDED:
            # Skips blocks past K; the branch changes scheduling, which helps some shapes.
            if kb < KB:
                wt = tl.load(w_ptr + kb * 128, cache_modifier=".cg").to(tl.bfloat16)
                xt = tl.load(x_ptr + kb * 128)
                acc += tl.dot(wt, tl.trans(xt)) * tl.load(scale_ptr + kb)
        else:
            if KB % BLOCKS_PER_SPLIT != 0:
                # The last split is short: clamp its tail onto a real block with zero scale.
                scale = tl.load(scale_ptr + kb, mask=kb < KB, other=0.0)
                kb = tl.minimum(kb, KB - 1)
            else:
                scale = tl.load(scale_ptr + kb)
            wt = tl.load(w_ptr + kb * 128, cache_modifier=".cg").to(tl.bfloat16)
            xt = tl.load(x_ptr + kb * 128)
            acc += tl.dot(wt, tl.trans(xt)) * scale
    out_ptr = out + offs_m[None, :] * N + offs_n[:, None]
    mask = n_mask[:, None] & m_mask[None, :]
    if SPLIT == 1:
        tl.store(out_ptr, acc.to(tl.bfloat16), mask=mask)
    else:
        stride = BLOCK_M * N
        local = offs_m[None, :] * N + offs_n[:, None]
        tl.store(partials + part * stride + local, acc, mask=mask)
        arrived = tl.atomic_add(counters + tile, 1, sem="acq_rel")
        if arrived == SPLIT - 1:
            total = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
            for s in tl.static_range(SPLIT):
                total += tl.load(partials + s * stride + local, mask=mask, other=0.0,
                                 cache_modifier=".cg")
            tl.store(out_ptr, total.to(tl.bfloat16), mask=mask)
            tl.store(counters + tile, 0)


@dataclass(frozen=True)
class W8A16Config:
    block_n: int
    split: int
    num_warps: int = 4
    num_stages: int = 3
    guarded: bool = False


# (N, K) -> ((last row count, config), ...) in increasing row order, swept on SM120
# (RTX PRO 6000, 188 SMs) with cold L2 over the MiMo drafter's TP8 projections.
_TUNED = {
    (2304, 6144): ((8, W8A16Config(32, 16, 4, 3, guarded=True)),
                   (16, W8A16Config(32, 12, 2, 2)),
                   (32, W8A16Config(32, 12, 2, 2, guarded=True))),
    (6144, 2048): ((16, W8A16Config(64, 4, 4, 2)), (32, W8A16Config(128, 2, 4, 3))),
    (4096, 6144): ((16, W8A16Config(128, 3, 8, 3)), (32, W8A16Config(64, 4, 2, 3))),
    (768, 30720): ((16, W8A16Config(32, 30, 4, 2)), (32, W8A16Config(64, 12, 8, 4))),
    (1280, 6144): ((16, W8A16Config(32, 12, 4, 4)), (32, W8A16Config(64, 6, 8, 4))),
}


def default_config(rows: int, n: int, k: int) -> W8A16Config:
    tuned = _TUNED.get((n, k))
    if tuned is not None:
        return next(config for last, config in tuned if rows <= last)
    # Untuned shapes: about two waves of programs over the SMs, whole K blocks each.
    block_n = 64 if n >= 4096 else 32
    tiles = triton.cdiv(n, block_n)
    return W8A16Config(block_n, max(1, min(k // BLOCK, round(376 / tiles))))


def _block_m(rows: int) -> int:
    return max(16, triton.next_power_of_2(rows))


class W8A16Gemv:
    """``out[m, n] = sum_kb (x[m, kb] . w[n, kb]) * scales[n // 128, kb]`` for a fixed shape.

    ``x`` is (rows, K) BF16, ``w`` is (N, K) E4M3, ``scales`` is (ceil(N/128), K/128) FP32,
    output is (rows, N) BF16.  Owns its split-K workspace, so one instance must not be
    launched concurrently on two streams.
    """

    def __init__(self, rows: int, n: int, k: int, device, config: W8A16Config | None = None):
        if not 1 <= rows <= MAX_ROWS or k % BLOCK or n <= 0:
            raise ValueError("W8A16 GEMV needs 1..32 rows and K a multiple of 128")
        self.rows, self.n, self.k = rows, n, k
        self.config = config or default_config(rows, n, k)
        if 128 % self.config.block_n:
            raise ValueError("block_n must divide 128")
        self.block_m = _block_m(rows)
        blocks = k // BLOCK
        split = min(self.config.split, blocks)
        self.blocks_per_split = triton.cdiv(blocks, split)
        self.split = triton.cdiv(blocks, self.blocks_per_split)
        tiles = triton.cdiv(n, self.config.block_n)
        self.grid = (tiles, self.split, 1)
        self.partials = torch.empty((self.split, self.block_m, n) if self.split > 1 else (1,),
                                    dtype=torch.float32, device=device)
        self.counters = torch.zeros(tiles, dtype=torch.int32, device=device)

    def args(self, x, w, scales, out):
        return (x, w, scales, out, self.partials, self.counters, x.shape[0])

    def options(self):
        return dict(N=self.n, K=self.k, BLOCK_M=self.block_m, BLOCK_N=self.config.block_n,
                    SPLIT=self.split, BLOCKS_PER_SPLIT=self.blocks_per_split,
                    GUARDED=self.config.guarded,
                    num_warps=self.config.num_warps, num_stages=self.config.num_stages)

    def prepare(self, x, w, scales, out):
        """A prepared launch (see ignition.preparation.triton) bound to these tensors."""
        from ignition.preparation.triton import prepare_triton
        self._check(x, w, scales, out)
        return prepare_triton(_w8a16_gemv, self.args(x, w, scales, out), self.grid,
                              **self.options())

    def __call__(self, x, w, scales, out=None):
        if out is None:
            out = x.new_empty((x.shape[0], self.n))
        self._check(x, w, scales, out)
        _w8a16_gemv[self.grid](*self.args(x, w, scales, out), **self.options())
        return out

    def _check(self, x, w, scales, out):
        if (x.dtype != torch.bfloat16 or x.shape[1] != self.k or not 1 <= x.shape[0] <= self.rows
                or w.dtype != torch.float8_e4m3fn or w.shape != (self.n, self.k)
                or scales.dtype != torch.float32
                or scales.shape != (triton.cdiv(self.n, BLOCK), self.k // BLOCK)
                or out.dtype != torch.bfloat16 or out.shape != (x.shape[0], self.n)
                or not all(t.is_contiguous() for t in (x, w, scales, out))):
            raise ValueError("W8A16 GEMV operand mismatch")
