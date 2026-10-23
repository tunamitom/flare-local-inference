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
    # The draft head's vocabulary shard; rows 17-32 reuse the 16-row config.
    (19072, 6144): ((8, W8A16Config(128, 3, 4, 2)), (32, W8A16Config(64, 1, 4, 4))),
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
    output is (rows, N) BF16.  Owns its split-K workspace, or binds a shared W8A16Workspace,
    so one instance must not be launched concurrently on two streams.
    """

    def __init__(self, rows: int, n: int, k: int, device, config: W8A16Config | None = None, *,
                 workspace=None):
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
        shape = (self.split, self.block_m, n) if self.split > 1 else (1,)
        if workspace is None:
            self.partials = torch.empty(shape, dtype=torch.float32, device=device)
            self.counters = torch.zeros(tiles, dtype=torch.int32, device=device)
        else:
            self.partials, self.counters = workspace.bind(shape, tiles)

    @staticmethod
    def extent(rows: int, n: int, k: int, config: W8A16Config | None = None):
        """FP32 partial elements and counters a launch of this shape uses."""
        config = config or default_config(rows, n, k)
        blocks = k // BLOCK
        split = triton.cdiv(blocks, triton.cdiv(blocks, min(config.split, blocks)))
        return (split * _block_m(rows) * n if split > 1 else 1), triton.cdiv(n, config.block_n)

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


def quantize_block_fp8(weight: torch.Tensor):
    """E4M3 values with one FP32 scale, amax / 448, per 128x128 block of a BF16 (N, K) matrix.

    The layout is BlockFP8Weight's: values (N, K), scales (ceil(N / 128), K / 128).  A ragged
    last row block takes its scale over the rows present.
    """
    from ignition.components.block_fp8 import BlockFP8Weight

    if weight.ndim != 2 or weight.dtype != torch.bfloat16 or weight.shape[1] % BLOCK:
        raise ValueError("block-FP8 quantization takes BF16 matrices with K a multiple of 128")
    n, k = weight.shape
    row_blocks = triton.cdiv(n, BLOCK)
    padded = torch.nn.functional.pad(weight.float(), (0, 0, 0, row_blocks * BLOCK - n))
    blocks = padded.view(row_blocks, BLOCK, k // BLOCK, BLOCK)
    amax = blocks.abs().amax(dim=(1, 3))
    scales = torch.where(amax > 0, amax / 448, torch.ones_like(amax))
    values = (blocks / scales[:, None, :, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return BlockFP8Weight(values.view(row_blocks * BLOCK, k)[:n].contiguous(), scales.contiguous())


class W8A16Workspace:
    """Split-K partials and arrival counters shared by W8A16 launches that run in stream order.

    Each launch consumes its partials and re-arms its counters before the next launch on the
    stream begins, so every projection and graph on one stream can share one workspace.  It is
    sized for every row count up to MAX_ROWS of the given (N, K) shapes.  Counters must be zero
    when no launch is in flight; zeroing the whole workspace, as package export does for
    disposable storage, leaves it ready.
    """

    def __init__(self, shapes, device):
        extents = [W8A16Gemv.extent(rows, n, k) for n, k in shapes for rows in range(1, MAX_ROWS + 1)]
        if not extents:
            raise ValueError("a W8A16 workspace needs at least one projection shape")
        self.shapes = frozenset(shapes)
        self.partials = torch.zeros(max(p for p, _ in extents), dtype=torch.float32, device=device)
        self.counters = torch.zeros(max(t for _, t in extents), dtype=torch.int32, device=device)

    def bind(self, shape, tiles):
        count = 1
        for size in shape:
            count *= size
        if count > self.partials.numel() or tiles > self.counters.numel():
            raise ValueError("W8A16 launch exceeds its shared workspace")
        return self.partials[:count].view(shape), self.counters[:tiles]

    def scratch_tensors(self):
        return (self.partials, self.counters)


class W8A16Projection:
    """``output = source @ dequantized(weight).T`` with BF16 rows and a block-FP8 weight.

    Rows run as prepared W8A16 GEMV launches of at most MAX_ROWS rows each; larger sources
    (prefill context projections) take one launch per row slice, each reading the whole weight.
    Calls may pass any source with the prepared shape, such as a norm's fresh output.
    """

    def __init__(self, source, weight, output, *, session, name, workspace: W8A16Workspace):
        from ignition.preparation.tensor_metadata import record_numeric_tensor
        from ignition.preparation.tuning import register_finalizer

        values, scales = weight.values, weight.scales
        n, k = values.shape
        rows = source.shape[0]
        if (source.ndim != 2 or source.shape[1] != k or output.shape != (rows, n)
                or source.dtype != torch.bfloat16 or output.dtype != torch.bfloat16
                or not source.is_contiguous() or not output.is_contiguous() or rows == 0):
            raise ValueError(f"W8A16 projection {name} requires contiguous BF16 rows matching its weight")
        for tensor in (values, scales):
            record_numeric_tensor(tensor)
        self.weight, self.output, self.name = weight, output, name
        self.shape = tuple(source.shape)
        self.slices = tuple((start, min(rows, start + MAX_ROWS)) for start in range(0, rows, MAX_ROWS))
        self.gemvs = tuple(W8A16Gemv(end - start, n, k, source.device, workspace=workspace)
                           for start, end in self.slices)
        self._source = source
        self.launches = None
        register_finalizer(session, self.prepare)

    def prepare(self):
        w, s = self.weight.values, self.weight.scales
        self.launches = tuple(gemv.prepare(self._source[start:end], w, s, self.output[start:end])
                              for gemv, (start, end) in zip(self.gemvs, self.slices))
        self._source = None

    def __call__(self, source):
        from ignition.preparation.triton import launch_triton

        if self.launches is None:
            raise RuntimeError(f"W8A16 projection {self.name} must prepare before it runs")
        if tuple(source.shape) != self.shape or not source.is_contiguous():
            raise ValueError(f"W8A16 projection {self.name} requires contiguous {self.shape} sources")
        w, s = self.weight.values, self.weight.scales
        for gemv, launch, (start, end) in zip(self.gemvs, self.launches, self.slices):
            launch_triton(launch.kernel, launch.compiled, gemv.args(source[start:end], w, s, self.output[start:end]),
                          launch.grid, **launch.options)
        return self.output
