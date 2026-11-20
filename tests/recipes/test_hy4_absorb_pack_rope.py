# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0
"""N3 — `_absorb_into` 1-launch fused pack + RoPE (CPU gates).

The fused helper's CPU path, its dispatch and its guards are pure Python /
torch, so they run here; the Triton kernel's 0-ULP contract against
`fused_ops.rope_interleaved` is the GPU harness (`n3_bitexact_gpu.py`, the
microbench lane). What is gated on CPU:

1. `_rope_reference` (the helper's CPU rope) is BIT-IDENTICAL to the
   production eager `attention.apply_interleaved_rope`.
2. `absorb_pack_rope_fused`'s assembly is bit-identical to the two-launch
   chain it replaces (`out[..., :lora].copy_(absorbed.transpose(0, 1))` +
   rope into `out[..., lora:]`), at the real geometry (qk_nope=192,
   qk_rope=64, lora=512, H=8) across rows 2..2048 and hostile positions.
3. `_absorb_into` routes `rows > 1` contiguous-CUDA queries through the fused
   helper (the old chain would call `rope_interleaved`, which is booby-trapped
   in those tests) and keeps the old chain for `rows == 1` and for
   non-contiguous queries — every case still bit-exact.
4. The helper's layout/shape/dtype guards fail loudly (b725daa7's contract).
"""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from ignition.recipes.hy4 import absorb_pack_triton
from ignition.recipes.hy4 import fused_ops
from ignition.recipes.hy4.attention import Hy4LayerAttention, apply_interleaved_rope

# the real `Hy4-nvfp4-w4a16-gptq-q4` geometry (config.json)
HEADS = 8
QK_NOPE, QK_ROPE, LORA = 192, 64, 512
THETA = 10000.0
ROWS_SWEEP = (2, 3, 4, 8, 12, 16, 32, 256, 2048)


def _attention(heads=HEADS, *, qk_nope=QK_NOPE, qk_rope=QK_ROPE, lora=LORA,
               dtype=torch.bfloat16):
    a = Hy4LayerAttention.__new__(Hy4LayerAttention)
    a.heads = heads
    a.qk_nope = qk_nope
    a.qk_rope = qk_rope
    a.rope_theta = THETA
    a.v_head = 128
    a.kv_lora = lora
    gen = torch.Generator().manual_seed(4321)
    a.absorb_wuk = torch.randn(heads, qk_nope, lora, generator=gen).to(dtype)
    a.expand_wuv = torch.randn(heads, a.v_head, lora, generator=gen).to(dtype)
    a.sharding = SimpleNamespace(head_range=(0, heads))
    return a


class _FakeCuda(torch.Tensor):
    """CPU tensor whose `is_cuda` reports True — drives the `rows > 1` CUDA
    dispatch (and the fused helper) without a GPU."""

    @property
    def is_cuda(self):
        return True


def _fc(x):
    # wrap outside any TorchDispatchMode block — the subclass constructor
    # cannot run through the dispatcher
    return _FakeCuda(x.as_strided(x.shape, x.stride()))


def _rope_ref_apply(x, positions):
    """Independent reference: the production eager rope (plain CPU tensors)."""
    return apply_interleaved_rope(x, positions, rope_dim=int(x.shape[-1]), theta=THETA)


def _chain_ref(q, attn, positions):
    """The current `_absorb_into` output: the exact GEMM + pack copy + eager
    rope of the two-launch chain (`_bmm_rhi` is bit-exact vs einsum — the
    other file's first gate)."""
    absorbed = torch.einsum("rhi,hio->rho", q[..., :attn.qk_nope], attn.absorb_wuk)
    out = torch.empty((int(q.shape[0]), attn.heads, attn.kv_lora + attn.qk_rope),
                      dtype=q.dtype)
    out[..., :attn.kv_lora].copy_(absorbed)
    out[..., attn.kv_lora:].copy_(_rope_ref_apply(q[..., attn.qk_nope:], positions))
    return out


def _eager_rope_interleaved(x, positions, *, rope_dim, theta, out=None):
    """CPU stand-in for `fused_ops.rope_interleaved` (Triton, GPU-only): the
    eager rope with an `out=` view, for the old-chain dispatch tests."""
    rotated = absorb_pack_triton._rope_reference(x, positions, rope_dim=rope_dim,
                                                 theta=theta)
    if out is not None:
        out.copy_(rotated)
        return out
    return rotated


@pytest.fixture
def eager_rope(monkeypatch):
    monkeypatch.setattr(fused_ops, "rope_interleaved", _eager_rope_interleaved)


def _count_fused(monkeypatch):
    """Wrap `absorb_pack_rope_fused` with a call counter (the fused-path
    dispatch proof)."""
    calls = []
    real = absorb_pack_triton.absorb_pack_rope_fused

    def wrapped(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(absorb_pack_triton, "absorb_pack_rope_fused", wrapped)
    return calls


# ------------------------------------------------- the rope reference itself
@pytest.mark.parametrize("rows", (1, 2, 5, 17))
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float32))
def test_rope_reference_matches_apply_interleaved_rope(rows, dtype):
    gen = torch.Generator().manual_seed(rows * 31 + int(dtype == torch.float32))
    x = torch.randn(rows, HEADS, QK_ROPE, generator=gen).to(dtype)
    for positions in (torch.arange(rows),
                      torch.arange(rows) + 1_000_000,          # large angles
                      torch.randint(0, 1 << 20, (rows,), generator=gen),
                      torch.zeros(rows, dtype=torch.int64),    # degenerate
                      torch.full((1,), 7)):                    # one row for all
        got = absorb_pack_triton._rope_reference(x, positions, rope_dim=QK_ROPE,
                                                 theta=THETA)
        want = _rope_ref_apply(x, positions)
        assert torch.equal(got, want), (rows, dtype, positions[:3])


# --------------------------------------------- the fused pack+rope assembly
@pytest.mark.parametrize("rows", ROWS_SWEEP)
def test_absorb_pack_rope_fused_matches_the_two_launch_chain(rows):
    attn = _attention()
    gen = torch.Generator().manual_seed(900 + rows)
    q = torch.randn(rows, HEADS, attn.qk_nope + attn.qk_rope, generator=gen).to(torch.bfloat16)
    absorbed_h = torch.randn(HEADS, rows, attn.kv_lora, generator=gen).to(torch.bfloat16)
    positions = (torch.arange(rows) * 7919) % (1 << 21)        # hostile angles
    out = torch.empty((rows, HEADS, attn.kv_lora + attn.qk_rope), dtype=torch.bfloat16)
    got = absorb_pack_triton.absorb_pack_rope_fused(
        q, absorbed_h, positions, qk_nope=attn.qk_nope, lora=attn.kv_lora,
        rope_dim=attn.qk_rope, theta=attn.rope_theta, out=out)
    assert got is out
    ref = torch.cat((absorbed_h.transpose(0, 1),
                     _rope_ref_apply(q[..., attn.qk_nope:], positions)), dim=-1)
    assert torch.equal(out, ref), rows
    # FP32 operands too (the CPU reference path, same assembly)
    qf, af = q.float(), absorbed_h.float()
    out_f = absorb_pack_triton.absorb_pack_rope_fused(
        qf, af, positions, qk_nope=attn.qk_nope, lora=attn.kv_lora,
        rope_dim=attn.qk_rope, theta=attn.rope_theta)
    ref_f = torch.cat((af.transpose(0, 1),
                       _rope_ref_apply(qf[..., attn.qk_nope:], positions)), dim=-1)
    assert torch.equal(out_f, ref_f), rows


def test_absorb_pack_rope_fused_generic_geometry():
    """Width-generic (the kernel takes qk_nope/lora/rope as parameters)."""
    for qk_nope, rope, lora in ((128, 64, 256), (512, 64, 512), (64, 2, 33)):
        rows, heads = 5, 3
        gen = torch.Generator().manual_seed(qk_nope + lora)
        q = torch.randn(rows, heads, qk_nope + rope, generator=gen).to(torch.float32)
        absorbed_h = torch.randn(heads, rows, lora, generator=gen).to(torch.float32)
        positions = torch.arange(rows)
        got = absorb_pack_triton.absorb_pack_rope_fused(
            q, absorbed_h, positions, qk_nope=qk_nope, lora=lora,
            rope_dim=rope, theta=THETA)
        ref = torch.cat((absorbed_h.transpose(0, 1),
                         _rope_ref_apply(q[..., qk_nope:], positions)), dim=-1)
        assert torch.equal(got, ref), (qk_nope, rope, lora)


# ------------------------------------------------------- _absorb_into routing
def test_absorb_into_fused_path_matches_current_chain(eager_rope, monkeypatch):
    """rows > 1 + contiguous CUDA query -> the fused helper (the booby-trapped
    `rope_interleaved` must NOT be reached), output bit-identical to the
    current two-launch chain."""
    calls = _count_fused(monkeypatch)
    for rows in (2, 3, 4, 8, 12, 16, 33):
        attn = _attention()
        gen = torch.Generator().manual_seed(500 + rows)
        q = torch.randn(rows, HEADS, attn.qk_nope + attn.qk_rope,
                        generator=gen).to(torch.bfloat16)
        positions = torch.arange(rows) + 12
        got = attn._absorb_into(_fc(q), positions)
        assert calls, rows                       # the fused path ran
        assert torch.equal(got, _chain_ref(q, attn, positions)), rows


def test_absorb_into_rows_one_keeps_the_old_chain(eager_rope, monkeypatch):
    calls = _count_fused(monkeypatch)
    attn = _attention()
    q = torch.randn(1, HEADS, attn.qk_nope + attn.qk_rope).to(torch.bfloat16)
    positions = torch.zeros(1, dtype=torch.int64)
    got = attn._absorb_into(_fc(q), positions)
    assert not calls                             # rows == 1 never fuses
    assert torch.equal(got, _chain_ref(q, attn, positions))


def test_absorb_into_non_contiguous_query_keeps_the_old_chain(eager_rope, monkeypatch):
    """The fused kernel indexes flat [rows, heads] rows; a strided query falls
    back to the two-launch chain (still bit-exact), never mis-indexes."""
    calls = _count_fused(monkeypatch)
    rows = 6
    attn = _attention()
    gen = torch.Generator().manual_seed(606)
    q = torch.randn(rows, HEADS, 2 * (attn.qk_nope + attn.qk_rope),
                    generator=gen).to(torch.bfloat16)[..., :attn.qk_nope + attn.qk_rope]
    assert not q.is_contiguous()
    positions = torch.arange(rows)
    got = attn._absorb_into(_fc(q), positions)
    assert not calls
    assert torch.equal(got, _chain_ref(q, attn, positions))


# ------------------------------------------------------------------- guards
def test_absorb_pack_rope_fused_guards_fail_loudly():
    rows, heads = 4, HEADS
    q = torch.randn(rows, heads, QK_NOPE + QK_ROPE).to(torch.bfloat16)
    absorbed_h = torch.randn(heads, rows, LORA).to(torch.bfloat16)
    positions = torch.arange(rows)
    call = dict(qk_nope=QK_NOPE, lora=LORA, rope_dim=QK_ROPE, theta=THETA)

    def expect(**kwargs):
        args = dict(q_rhi=q, absorbed_h=absorbed_h, positions=positions, **call)
        args.update(kwargs)
        with pytest.raises(ValueError):
            absorb_pack_triton.absorb_pack_rope_fused(**args)

    # the baseline really works first (the `expect` cases below must be the
    # only thing that raises)
    out = absorb_pack_triton.absorb_pack_rope_fused(q, absorbed_h, positions, **call)
    assert tuple(out.shape) == (rows, heads, LORA + QK_ROPE)
    expect(q_rhi=torch.randn(rows, heads, QK_NOPE).to(torch.bfloat16))     # width split
    expect(absorbed_h=torch.randn(heads, rows, LORA + 1).to(torch.bfloat16))
    expect(absorbed_h=absorbed_h.float())                                    # dtype
    expect(rope_dim=QK_ROPE + 1)                                             # odd rope
    expect(qk_nope=0)
    expect(positions=torch.arange(rows * heads + 1))                         # coverage
    expect(q_rhi=torch.randn(rows, heads, 2 * (QK_NOPE + QK_ROPE))
           .to(torch.bfloat16)[..., :QK_NOPE + QK_ROPE])                     # contiguous
    expect(absorbed_h=torch.randn(heads, 2 * rows, LORA)
           .to(torch.bfloat16)[:, :rows])                                   # contiguous
    out_bad_shape = torch.zeros(rows, heads, LORA + QK_ROPE + 8)
    expect(out=out_bad_shape)
    out_bad_dtype = torch.zeros(rows, heads, LORA + QK_ROPE, dtype=torch.float32)
    expect(out=out_bad_dtype)
    out_strided = torch.zeros(rows, heads, 2 * (LORA + QK_ROPE))[
        ..., :LORA + QK_ROPE]
    expect(out=out_strided)
