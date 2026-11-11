"""W22 riders: fused decode score/ID pack (decode_fusion) + out=/count_out= compaction.

Ported invariants from the W11 threat rows (levers/w11-results.md §2): C1 pack
layout (rank-major, FP32 score + int32 id), C3 tie/dead-entry symmetry, the
default-off bit-equality of the shipped pack, and the T17 aliasing guards of
the compaction's retained buffers. CPU only; the GPU node gate (T14) and the
world-8 C3 run are the W11 evidence this rider already carries.
"""
from __future__ import annotations

import pytest
import torch

from ignition.components.distributed_topk import (
    _pack_pairs_reference,
    pack_score_id_pairs,
    selected_local_records,
)


def _shipped_pack(local, indices, scores, rank, world):
    live = indices >= 0
    local[:, :, 0].copy_(torch.where(live, scores, -torch.inf))
    local.view(torch.int32)[:, :, 1].copy_(
        torch.where(live, indices.long() * world + rank, -1))


@pytest.mark.parametrize("rows,width", ((4, 32), (1, 16), (3, 48)))
def test_fused_pack_matches_the_shipped_pack_bitwise(rows, width):
    gen = torch.Generator().manual_seed(1234 + rows)
    indices = torch.randint(-1, 896, (rows, width), generator=gen).to(torch.int32)
    # ties on purpose: duplicate one row's scores across ranks
    scores = torch.randn(rows, width, generator=gen)
    scores[0, 1] = scores[0, 2] = 0.25
    rank = torch.tensor([3], dtype=torch.int64)
    world = 8
    shipped = torch.zeros((rows, width, 2), dtype=torch.float32)
    fused = torch.zeros((rows, width, 2), dtype=torch.float32)
    _shipped_pack(shipped, indices, scores, rank, world)
    _pack_pairs_reference(fused, indices, scores, rank, world)
    # Compare bits: dead entries carry id -1 whose FP32 view is NaN, and
    # torch.equal would report NaN != NaN even for identical bits.
    assert torch.equal(shipped.view(torch.int32), fused.view(torch.int32))
    # dead entries pack as score -inf and id -1 (C1 contract)
    dead = indices < 0
    if dead.any():
        assert torch.isinf(fused[:, :, 0][dead]).all() and (fused[:, :, 0][dead] < 0).all()
        assert (fused.view(torch.int32)[:, :, 1][dead] == -1).all()


def test_pack_score_id_pairs_cpu_body_matches_reference():
    gen = torch.Generator().manual_seed(7)
    indices = torch.randint(-1, 896, (2, 32), generator=gen).to(torch.int32)
    scores = torch.randn(2, 32, generator=gen)
    rank = torch.tensor([5], dtype=torch.int64)
    direct = torch.zeros((2, 32, 2), dtype=torch.float32)
    via_op = torch.zeros((2, 32, 2), dtype=torch.float32)
    _pack_pairs_reference(direct, indices, scores, rank, 8)
    pack_score_id_pairs(via_op, indices, scores, rank, 8)
    assert torch.equal(direct.view(torch.int32), via_op.view(torch.int32))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the compaction kernel is a Triton GPU kernel")
def test_compaction_retained_buffers_match_the_allocating_path():
    gen = torch.Generator().manual_seed(3)
    rows, width = 5, 64
    world = 8
    selected = (torch.randint(-4, 512, (rows, width), generator=gen)).to(torch.int32).cuda()
    positions = torch.full((rows,), 400, dtype=torch.int32, device="cuda")
    pages = torch.randint(0, 97, (rows, 16), generator=gen).to(torch.int32).cuda()
    rank = torch.tensor([2], dtype=torch.int64, device="cuda")
    out_buf = torch.full((rows, width), -7, dtype=torch.int32, device="cuda")
    count_buf = torch.full((rows,), -7, dtype=torch.int32, device="cuda")
    ref_out, ref_count = selected_local_records(
        selected, positions, pages, world=world, rank=rank, page_size=32)
    got_out, got_count = selected_local_records(
        selected, positions, pages, world=world, rank=rank, page_size=32,
        out=out_buf, count_out=count_buf)
    assert got_out is out_buf and got_count is count_buf
    assert torch.equal(ref_out, out_buf)
    assert torch.equal(ref_count, count_buf)


def test_compaction_overlap_guards():
    rows, width = 3, 16
    selected = torch.zeros((rows, width), dtype=torch.int32)
    positions = torch.zeros((rows,), dtype=torch.int32)
    pages = torch.zeros((rows, 4), dtype=torch.int32)
    rank = torch.tensor([1], dtype=torch.int64)
    with pytest.raises(ValueError, match="must not overlap"):
        selected_local_records(selected, positions, pages, world=8, rank=rank, page_size=32,
                               out=selected)
    with pytest.raises(ValueError, match="must not overlap"):
        selected_local_records(selected, positions, pages, world=8, rank=rank, page_size=32,
                               out=selected.clone(), count_out=positions)
    with pytest.raises(ValueError, match="contiguous"):
        selected_local_records(selected, positions, pages, world=8, rank=rank, page_size=32,
                               out=selected[:, :width // 2])


def test_compaction_overlap_guards_accept_disjoint_shared_storage():
    rows, width = 3, 16
    span = rows * width
    total = span + rows + rows * 4 + span + rows
    base = torch.zeros((total,), dtype=torch.int32)
    selected = base[:span].view(rows, width)
    positions = base[span:span + rows]
    pages = base[span + rows:span + rows + rows * 4].view(rows, 4)
    out = base[span + rows + rows * 4:span + rows + rows * 4 + span].view(rows, width)
    count_out = base[span + rows + rows * 4 + span:]
    # All five buffers are disjoint slices of one storage: the storage-equality
    # guard rejected this legitimate pooled layout; the byte-range guard must
    # accept it (the guard is what this exercises — kernel-stage errors on CPU
    # are outside its scope).
    assert selected.untyped_storage().data_ptr() == out.untyped_storage().data_ptr()
    try:
        selected_local_records(selected, positions, pages, world=8,
                               rank=torch.tensor([1], dtype=torch.int64), page_size=32,
                               out=out, count_out=count_out)
    except ValueError as exc:
        # Triton's own "cpu tensor?" ValueError is kernel-stage and out of scope;
        # only the overlap guard's message is this test's business.
        if "must not overlap" in str(exc):
            pytest.fail(f"disjoint shared-storage buffers must not trip the overlap guard: {exc}")
    except Exception:
        pass
