"""test_hy4_attention_bmm_strides.py — CPU zero-overlap regression for the
cuBLAS 13.1.0..13.6.1 SM120 strided-batched TMA OOB-read fix (`_bmm_rhi`).

`nvidia-cublas 13.4.1.3`'s `nvjet_sm120_*_tmaAB` kernels perform out-of-bounds
hardware TMA reads when `torch.bmm`/`torch.einsum` lower interleaved,
overlapping batch strides (`stride_batch < slice_span`) on small tensors. The
fix materializes contiguous head-major operands (`Hy4LayerAttention._bmm_rhi`)
whenever `rows > 1` on CUDA, so no overlapping batch slice ever reaches
`cublasGemmStridedBatchedEx`.

Regression contract (CPU only, zero GPU risk — the unfixed pattern fires real
`Xid 13`/`Xid 31` hardware faults on SM120 and must never run on a GPU):

1. `TorchDispatchMode` hooks `aten.bmm` / `aten.bmm.out` and asserts that every
   3-D tensor argument (`A`, `B`, `out`) with `batch > 1` satisfies
   `stride[0] >= (shape[1] - 1) * stride[1] + (shape[2] - 1) * stride[2] + 1`
   (no overlapping batch slices). All four call sites — `_absorb_into`,
   `absorb_queries`, `expand_output` (BF16 + FP32) and `expand_partials`
   (BF16 + FP32) — run through the fixed CUDA path on CPU via a tiny tensor
   subclass faking `is_cuda=True`.
2. `_bmm_rhi` and the call sites are verified bit-exact (`torch.equal`) against
   the `torch.einsum` reference (bmm/einsum are bit-exact on CPU for both
   BF16 and FP32, verified), and `_bmm_rhi`'s result is head-major contiguous
   after `.transpose(0, 1)` (the `collectives.py` head-major stride contract).
3. Negative control (CPU only): the raw overlapping `transpose(0, 1)` lowering
   is asserted to TRIP the detector, proving the detector has teeth. The
   unfixed pattern is never executed against a GPU.
"""
from types import SimpleNamespace

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from ignition.recipes.hy4 import fused_ops
from ignition.recipes.hy4 import runtime as hy4_runtime
from ignition.recipes.hy4.attention import Hy4LayerAttention

ROWS = (1, 2, 3, 4, 8, 12, 16, 20, 32)
HEADS = (8, 64)


def _non_overlapping_batch(t):
    """No overlapping batch slices: one slice spans at most `stride[0]`."""
    return int(t.stride()[0]) >= (
        (int(t.shape[1]) - 1) * int(t.stride()[1])
        + (int(t.shape[2]) - 1) * int(t.stride()[2]) + 1)


class _NoOverlapBmm(TorchDispatchMode):
    """Assert every `aten.bmm` 3-D operand with batch > 1 is non-overlapping."""

    def __init__(self):
        super().__init__()
        self.bmm_calls = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = str(func)
        if ".bmm" in name or name.endswith("bmm"):
            self.bmm_calls += 1
            tensors = [a for a in args if isinstance(a, torch.Tensor)]
            tensors += [v for v in kwargs.values() if isinstance(v, torch.Tensor)]
            for t in tensors:
                if t.dim() == 3 and int(t.shape[0]) > 1:
                    assert _non_overlapping_batch(t), (
                        f"{name}: overlapping batch strides — "
                        f"shape={tuple(t.shape)} stride={tuple(t.stride())}")
        return func(*args, **kwargs)


class _FakeCuda(torch.Tensor):
    """CPU tensor whose `is_cuda` reports True — exercises the CUDA fix path
    (the `rows > 1` contiguous materialization) without touching a GPU."""

    @property
    def is_cuda(self):
        return True


def _fc(x):
    # NOTE: wrap outside any TorchDispatchMode block — the subclass constructor
    # cannot run through the dispatcher.
    return _FakeCuda(x.as_strided(x.shape, x.stride()))


def _attention(heads, *, dtype=torch.bfloat16):
    """Minimal `Hy4LayerAttention` with the attributes the four call sites read
    (the real constructor needs a full checkpoint; the geometry matches
    `Hy4-nvfp4-w4a16-gptq-q4`: qk_nope=128, qk_rope=64, v_head=128, lora=512)."""
    a = Hy4LayerAttention.__new__(Hy4LayerAttention)
    a.heads = heads
    a.qk_nope = 128
    a.qk_rope = 64
    a.rope_theta = 10000.0
    a.v_head = 128
    a.kv_lora = 512
    gen = torch.Generator().manual_seed(1234)
    per_head = torch.randn(heads, a.qk_nope + a.v_head, a.kv_lora,
                           generator=gen).to(dtype)
    a.absorb_wuk = per_head[:, :a.qk_nope]
    a.expand_wuv = per_head[:, a.qk_nope:]
    a.sharding = SimpleNamespace(head_range=(0, heads))
    return a


def _rope_identity(x, positions, *, rope_dim, theta, out=None):
    """CPU stand-in for `fused_ops.rope_interleaved` (Triton, GPU-only): copies
    the rope block through unchanged — the bmm routing under test is orthogonal
    to the rotation math."""
    if out is not None:
        out.copy_(x)
        return out
    return x.clone()


@pytest.fixture
def rope_identity(monkeypatch):
    monkeypatch.setattr(fused_ops, "rope_interleaved", _rope_identity)


@pytest.fixture
def absorb_bf16_switch(monkeypatch):
    monkeypatch.setitem(hy4_runtime.SWITCHES, "expand_bf16", "on")


# ---------------------------------------------------------------- reference
def _ref_absorb_into(q_unroped, attn):
    absorbed = torch.einsum("rhi,hio->rho", q_unroped[..., :attn.qk_nope],
                            attn.absorb_wuk)
    return torch.cat((absorbed, q_unroped[..., attn.qk_nope:]), dim=-1)


# ------------------------------------------------------------------- tests
def test_bmm_rhi_matches_einsum_and_head_major_contract():
    """`_bmm_rhi` bit-exact vs the einsum reference and head-major contiguous."""
    for dtype in (torch.bfloat16, torch.float32):
        for heads in HEADS:
            for rows in ROWS:
                x = torch.randn(rows, heads, 128).to(dtype)
                w = torch.randn(heads, 128, 512).to(dtype)
                out = Hy4LayerAttention._bmm_rhi(x, w)
                ref = torch.einsum("rhi,hio->hro", x, w)
                assert torch.equal(out, ref), (dtype, heads, rows)
                # head-major contract: the call-site result
                # `_bmm_rhi(...).transpose(0, 1)` holds head-major storage
                res = out.transpose(0, 1)
                assert res.shape == (rows, heads, 512), (heads, rows)
                assert res.transpose(0, 1).is_contiguous(), (heads, rows)


def test_absorb_into_zero_overlap_and_exact(rope_identity):
    for dtype in (torch.bfloat16,):
        for heads in HEADS:
            for rows in ROWS:
                attn = _attention(heads, dtype=dtype)
                q = torch.randn(rows, heads, attn.qk_nope + attn.qk_rope).to(dtype)
                positions = torch.arange(rows)
                fq = _fc(q)
                mode = _NoOverlapBmm()
                with mode:
                    out = attn._absorb_into(fq, positions)
                assert mode.bmm_calls >= 1, (heads, rows)
                assert torch.equal(out, _ref_absorb_into(q, attn)), (heads, rows)


def test_absorb_queries_zero_overlap_and_exact():
    for heads in HEADS:
        for rows in ROWS:
            attn = _attention(heads)
            q = torch.randn(rows, heads, attn.qk_nope + attn.qk_rope).to(torch.bfloat16)
            fq = _fc(q)
            mode = _NoOverlapBmm()
            with mode:
                out = attn.absorb_queries(fq)
            assert mode.bmm_calls >= 1, (heads, rows)
            ref = torch.cat(
                (torch.einsum("rhi,hio->rho", q[..., :attn.qk_nope], attn.absorb_wuk),
                 q[..., attn.qk_nope:]), dim=-1)
            assert torch.equal(out, ref), (heads, rows)


def test_expand_output_zero_overlap_and_exact(absorb_bf16_switch):
    for dtype in (torch.bfloat16, torch.float32):
        for heads in HEADS:
            for rows in ROWS:
                attn = _attention(heads, dtype=torch.bfloat16)
                out_in = torch.randn(rows, heads, attn.kv_lora).to(dtype)
                fout = _fc(out_in)
                mode = _NoOverlapBmm()
                with mode:
                    out = attn.expand_output(fout, local=True)
                assert mode.bmm_calls >= 1, (dtype, heads, rows)
                wuv = attn.expand_wuv
                if dtype == torch.bfloat16:
                    ref = torch.einsum("rhd,hvd->rhv", out_in, wuv)
                else:
                    ref = torch.einsum("rhd,hvd->rhv", out_in,
                                       wuv.float()).to(dtype)
                assert torch.equal(out, ref), (dtype, heads, rows)


def test_expand_partials_zero_overlap_and_exact():
    for dtype in (torch.bfloat16, torch.float32):
        for heads in HEADS:
            for rows in ROWS:
                attn = _attention(heads, dtype=torch.bfloat16)
                out_in = torch.randn(rows, heads, attn.kv_lora).to(dtype)
                fout = _fc(out_in)
                mode = _NoOverlapBmm()
                with mode:
                    out = attn.expand_partials(fout)
                assert mode.bmm_calls >= 1, (dtype, heads, rows)
                wuv = attn.expand_wuv
                if dtype == torch.bfloat16:
                    ref = torch.einsum("rhd,hvd->rhv", out_in, wuv)
                else:
                    ref = torch.einsum("rhd,hvd->rhv", out_in,
                                       wuv.float()).to(dtype)
                assert torch.equal(out, ref), (dtype, heads, rows)


def test_negative_control_detector_flags_raw_overlapping_bmm():
    """CPU-only negative control: the raw interleaved `transpose(0, 1)` operands
    (rows > 1) DO overlap and must TRIP the detector. Never run on a GPU — the
    unfixed pattern triggers real SM120 TMA hardware faults."""
    rows, heads = 12, 8
    q = torch.randn(rows, heads, 128).to(torch.bfloat16)
    w = torch.randn(heads, 128, 512).to(torch.bfloat16)
    a = q[..., :128].transpose(0, 1)
    assert not _non_overlapping_batch(a)
    with pytest.raises(AssertionError):
        with _NoOverlapBmm():
            torch.bmm(a, w)


# --------------------------------------------- Option 1 (head-major zero-overlap)
def _attention_absorb_geom(heads, *, qk_nope=192, qk_rope=64, lora=512,
                           dtype=torch.bfloat16):
    """Option 1 geometry (PERF-PLAN.md Step 3): head-major `q_hrd` width
    `256 = 192 + 64`, absorbed query `576 = 512 + 64` (the `Hy4-nvfp4` wire
    contract `unabsorbed = qk_nope + qk_rope`, `absorbed = kv_lora + qk_rope`)."""
    a = Hy4LayerAttention.__new__(Hy4LayerAttention)
    a.heads = heads
    a.qk_nope = qk_nope
    a.qk_rope = qk_rope
    a.rope_theta = 10000.0
    a.v_head = 128
    a.kv_lora = lora
    gen = torch.Generator().manual_seed(4321)
    a.absorb_wuk = torch.randn(heads, qk_nope, lora, generator=gen).to(dtype)
    a.expand_wuv = torch.randn(heads, a.v_head, lora, generator=gen).to(dtype)
    a.sharding = SimpleNamespace(head_range=(0, heads))
    return a


class _BmmOperandWatch(TorchDispatchMode):
    """Records every 3-D `aten.bmm` operand (the no-copy proof needs the exact
    tensor object the GEMM received)."""

    def __init__(self):
        super().__init__()
        self.operands = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if ".bmm" in str(func):
            for t in tuple(args) + tuple(kwargs.values()):
                if isinstance(t, torch.Tensor) and t.dim() == 3:
                    self.operands.append(t)
        return func(*args, **kwargs)


def test_absorb_queries_head_major_zero_overlap_no_copy_and_exact():
    """Option 1 fast path (PERF-PLAN.md Step 3): head-major `q_hrd` keeps
    `q_nope`'s batch strides DISJOINT — `stride[0] == rows * 256` strictly
    exceeds `(rows - 1) * 256 + 192` — so the GEMM runs on the SLICE ITSELF
    (same `data_ptr`: no `contiguous()` copy, the b86a9926 HBM copy tax) and
    the fused transpose-concat returns the exact `[rows, heads, 576]` query."""
    for rows in ROWS:
        attn = _attention_absorb_geom(64)
        q = torch.randn(rows, 64, attn.qk_nope + attn.qk_rope).to(torch.bfloat16)
        q_hrd = q.transpose(0, 1).contiguous()      # CeExchange head-major output
        watch = _BmmOperandWatch()
        mode = _NoOverlapBmm()
        with mode, watch:
            out = attn.absorb_queries(q_hrd, head_major=True)
        assert mode.bmm_calls >= 1, rows
        # the GEMM's A operand IS the q_nope slice of the input — no copy:
        a_ops = [t for t in watch.operands if tuple(t.shape) == (64, rows, attn.qk_nope)]
        assert a_ops, rows
        assert a_ops[0].data_ptr() == q_hrd.data_ptr(), rows
        assert _non_overlapping_batch(a_ops[0]), rows
        # the plan's disjoint-stride math: batch stride is exactly rows * 256
        assert a_ops[0].stride()[0] == rows * (attn.qk_nope + attn.qk_rope), rows
        assert a_ops[0].stride()[2] == 1, rows
        ref = torch.cat(
            (torch.einsum("rhi,hio->rho", q[..., :attn.qk_nope], attn.absorb_wuk),
             q[..., attn.qk_nope:]), dim=-1)
        assert out.shape == (rows, 64, attn.kv_lora + attn.qk_rope), rows
        assert torch.equal(out, ref), rows


def test_absorb_pack_fused_matches_cat_reference():
    """The fused transpose-concat is pure movement: bit-identical to the
    `torch.cat((absorbed_h.transpose(0, 1), q_hrd[..., qk_nope:].transpose(0, 1)))`
    it replaces (the CPU reference path of `absorb_pack_triton.py`)."""
    from ignition.recipes.hy4 import absorb_pack_triton

    for rows in (1, 2, 7, 16, 33):
        attn = _attention_absorb_geom(64)
        q_hrd = torch.randn(64, rows, attn.qk_nope + attn.qk_rope).to(torch.bfloat16)
        absorbed_h = torch.randn(64, rows, attn.kv_lora).to(torch.bfloat16)
        got = absorb_pack_triton.absorb_pack_fused(
            q_hrd, absorbed_h, qk_nope=attn.qk_nope, lora=attn.kv_lora)
        ref = torch.cat((absorbed_h.transpose(0, 1),
                         q_hrd[..., attn.qk_nope:].transpose(0, 1)), dim=-1)
        assert got.shape == (rows, 64, attn.kv_lora + attn.qk_rope), rows
        assert torch.equal(got, ref), rows


def test_absorb_pack_fused_rejects_non_contiguous():
    """The pack indexes flat [H*R, W]/[H*R, L] rows (and a flat out): a
    strided view must fail loudly instead of mis-indexing (contiguity guard)."""
    from ignition.recipes.hy4 import absorb_pack_triton

    attn = _attention_absorb_geom(64)
    rows = 8
    w = attn.qk_nope + attn.qk_rope
    q_hrd = torch.randn(64, rows, w).to(torch.bfloat16)
    absorbed_h = torch.randn(64, rows, attn.kv_lora).to(torch.bfloat16)
    q_strided = torch.randn(64, rows * 2, w).to(torch.bfloat16)[:, :rows, :]
    a_strided = torch.randn(64, rows * 2, attn.kv_lora).to(torch.bfloat16)[:, :rows, :]
    out_strided = torch.randn(rows, 64, (attn.kv_lora + attn.qk_rope) * 2).to(torch.bfloat16)[
        ..., :attn.kv_lora + attn.qk_rope]
    assert not q_strided.is_contiguous() and not a_strided.is_contiguous()
    assert not out_strided.is_contiguous()
    for bad_q, bad_a, bad_out in ((q_strided, absorbed_h, None),
                                 (q_hrd, a_strided, None),
                                 (q_hrd, absorbed_h, out_strided)):
        try:
            absorb_pack_triton.absorb_pack_fused(bad_q, bad_a, qk_nope=attn.qk_nope,
                                                 lora=attn.kv_lora, out=bad_out)
        except ValueError as e:
            assert "contiguous" in str(e)
        else:
            raise AssertionError("non-contiguous absorb_pack_fused input accepted")


def test_row_major_path_unchanged_by_option1():
    """The `head_major=False` contract is byte-identical to `b86a9926`: the
    default call keeps the row-major `[rows, heads, width]` input contract."""
    attn = _attention_absorb_geom(64)
    q = torch.randn(12, 64, attn.qk_nope + attn.qk_rope).to(torch.bfloat16)
    fq = _fc(q)
    out = attn.absorb_queries(fq)
    ref = torch.cat(
        (torch.einsum("rhi,hio->rho", q[..., :attn.qk_nope], attn.absorb_wuk),
         q[..., attn.qk_nope:]), dim=-1)
    assert torch.equal(out, ref)
