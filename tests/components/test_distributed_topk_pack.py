# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Retained-buffer compaction of the distributed top-k record compaction.

The compaction kernel writes the caller's retained physical/counts buffers
directly (``out=``/``count_out=``); the guards reject byte-range overlap with
the kernel's inputs while accepting disjoint buffers carved from one shared
pool chunk. CPU only except the kernel equality test, which needs a GPU.
"""
from __future__ import annotations

import pytest
import torch

from ignition.components.distributed_topk import (
    _overlaps,
    _pack_pairs_reference,
    pack_score_id_pairs,
    selected_local_records,
)
from ignition.preparation.scratch import GraphBufferPool


def _shipped_pack(local, indices, scores, rank, world):
    live = indices >= 0
    local[:, :, 0].copy_(torch.where(live, scores, -torch.inf))
    local.view(torch.int32)[:, :, 1].copy_(
        torch.where(live, indices.long() * world + rank, -1))


def _inputs(rows=3, width=16):
    gen = torch.Generator().manual_seed(3)
    selected = (torch.randint(-4, 512, (rows, width), generator=gen)).to(torch.int32)
    positions = torch.full((rows,), 400, dtype=torch.int32)
    pages = torch.randint(0, 97, (rows, 4), generator=gen).to(torch.int32)
    return selected, positions, pages


def test_compaction_aliasing_guards_reject_true_overlap():
    selected, positions, pages = _inputs()
    rank = torch.tensor([1], dtype=torch.int64)
    with pytest.raises(ValueError, match="must not overlap"):
        selected_local_records(selected, positions, pages, world=8, rank=rank, page_size=32,
                               out=selected)
    with pytest.raises(ValueError, match="must not overlap"):
        selected_local_records(selected, positions, pages, world=8, rank=rank, page_size=32,
                               out=selected.clone(), count_out=positions)
    with pytest.raises(ValueError, match="must not overlap"):
        base = torch.zeros_like(selected)
        selected_local_records(selected, positions, pages, world=8, rank=rank, page_size=32,
                               out=base, count_out=base.reshape(-1)[: selected.shape[0]])
    with pytest.raises(ValueError, match="contiguous"):
        selected_local_records(selected, positions, pages, world=8, rank=rank, page_size=32,
                               out=selected[:, :8])


def test_compaction_guards_accept_disjoint_shared_pool_buffers():
    # Pool cursor buffers are carved out of one shared chunk: they share an
    # untyped storage without sharing any byte. A storage-level comparison
    # reports these as aliased and raises on legitimate callers.
    pool = GraphBufferPool("cpu", chunk_bytes=2 << 20)
    cursor = pool.cursor()
    selected, positions, pages = _inputs()
    pooled_selected = cursor.zeros(selected.shape, torch.int32)
    pooled_positions = cursor.zeros(positions.shape, torch.int32)
    pooled_pages = cursor.zeros(pages.shape, torch.int32)
    out = cursor.zeros(selected.shape, torch.int32)
    count_out = cursor.zeros((selected.shape[0],), torch.int32)
    buffers = (pooled_selected, pooled_positions, pooled_pages, out, count_out)
    assert len({t.untyped_storage().data_ptr() for t in buffers}) == 1
    assert not _overlaps(out, (pooled_selected, pooled_positions, pooled_pages))
    assert not _overlaps(count_out, (pooled_selected, pooled_positions, pooled_pages, out))
    # A true alias inside the shared chunk is still reported.
    assert _overlaps(out, (out,))
    assert _overlaps(out, (out[:, :8],))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the compaction kernel is a Triton GPU kernel")
def test_compaction_retained_pool_buffers_match_the_allocating_path():
    gen = torch.Generator().manual_seed(3)
    rows, width = 5, 64
    world = 8
    selected = (torch.randint(-4, 512, (rows, width), generator=gen)).to(torch.int32).cuda()
    positions = torch.full((rows,), 400, dtype=torch.int32, device="cuda")
    pages = torch.randint(0, 97, (rows, 16), generator=gen).to(torch.int32).cuda()
    rank = torch.tensor([2], dtype=torch.int64, device="cuda")
    pool = GraphBufferPool("cuda", chunk_bytes=2 << 20)
    cursor = pool.cursor()
    out_buf = cursor.zeros((rows, width), torch.int32)
    count_buf = cursor.zeros((rows,), torch.int32)
    out_buf.fill_(-7)
    count_buf.fill_(-7)
    ref_out, ref_count = selected_local_records(
        selected, positions, pages, world=world, rank=rank, page_size=32)
    got_out, got_count = selected_local_records(
        selected, positions, pages, world=world, rank=rank, page_size=32,
        out=out_buf, count_out=count_buf)
    assert got_out is out_buf and got_count is count_buf
    assert torch.equal(ref_out, out_buf)
    assert torch.equal(ref_count, count_buf)


def _packed_pair_inputs(rows, width, seed, *, ties=True, specials=True):
    gen = torch.Generator().manual_seed(seed)
    indices = torch.randint(-1, 896, (rows, width), generator=gen).to(torch.int32)
    scores = torch.randn(rows, width, generator=gen)
    if ties and width > 4:
        scores[0, 1] = scores[0, 2] = 0.25
    if specials and width > 8:
        scores[0, 3] = float("nan")
        scores[0, 4] = float("-inf")
    return indices, scores


def test_fused_pack_matches_the_shipped_pack_bitwise():
    for rows, width in ((4, 32), (1, 16), (3, 48)):
        indices, scores = _packed_pair_inputs(rows, width, 1234 + rows)
        rank = torch.tensor([3], dtype=torch.int64)
        world = 8
        shipped = torch.zeros((rows, width, 2), dtype=torch.float32)
        fused = torch.zeros((rows, width, 2), dtype=torch.float32)
        _shipped_pack(shipped, indices, scores, rank, world)
        _pack_pairs_reference(fused, indices, scores, rank, world)
        # Compare bits: dead entries carry id -1 whose FP32 view is NaN, and
        # torch.equal would report NaN != NaN even for identical bits.
        assert torch.equal(shipped.view(torch.int32), fused.view(torch.int32))
        # dead entries pack as score -inf and id -1
        dead = indices < 0
        if dead.any():
            assert torch.isinf(fused[:, :, 0][dead]).all() and (fused[:, :, 0][dead] < 0).all()
            assert (fused.view(torch.int32)[:, :, 1][dead] == -1).all()


def test_pack_score_id_pairs_cpu_body_matches_reference():
    indices, scores = _packed_pair_inputs(2, 32, 7)
    rank = torch.tensor([5], dtype=torch.int64)
    direct = torch.zeros((2, 32, 2), dtype=torch.float32)
    via_op = torch.zeros((2, 32, 2), dtype=torch.float32)
    _pack_pairs_reference(direct, indices, scores, rank, 8)
    pack_score_id_pairs(via_op, indices, scores, rank, 8)
    assert torch.equal(direct.view(torch.int32), via_op.view(torch.int32))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the fused pack is a Triton GPU kernel")
@pytest.mark.parametrize(("rows", "width", "world"),
                         ((1, 48, 8), (3, 256, 8), (8, 2048, 8), (5, 1024, 16), (2, 48, 16)))
def test_fused_pack_matches_the_shipped_pack_on_cuda(rows, width, world):
    # Bit-exact GPU cover of the fused pack against the shipped pack: tied and
    # non-finite scores, dead entries, and ids whose int32 narrowing overflows.
    indices, scores = _packed_pair_inputs(rows, width, 99 + rows + width)
    if width > 16:
        indices[0, 8] = 1 << 28  # (id * world + rank) wraps past int32
    rank = torch.tensor([world - 1], dtype=torch.int64, device="cuda")
    shipped = torch.zeros((rows, width, 2), dtype=torch.float32, device="cuda")
    fused = torch.zeros((rows, width, 2), dtype=torch.float32, device="cuda")
    _shipped_pack(shipped, indices.cuda(), scores.cuda(), rank, world)
    pack_score_id_pairs(fused, indices.cuda(), scores.cuda(), rank, world)
    assert torch.equal(shipped.view(torch.int32), fused.view(torch.int32))


def test_the_component_never_reads_the_environment_to_choose_behaviour():
    """Source-text check, not a behavioural proof: it asserts only that the module's
    source text contains no ``os.environ`` reference and no ``getenv`` call. It does not
    observe runtime behaviour and does not prove the component could never read the
    environment by another spelling; it pins the two idioms the component must not use.
    The gate for captured behaviour is the ``decode_fusion`` constructor argument."""
    import inspect

    import ignition.components.distributed_topk as module

    source = inspect.getsource(module)
    assert "os.environ" not in source, "source-text check: module source must not contain os.environ"
    assert "getenv" not in source, "source-text check: module source must not contain getenv"
