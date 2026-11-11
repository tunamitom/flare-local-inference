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

from ignition.components.distributed_topk import _overlaps, selected_local_records
from ignition.preparation.scratch import GraphBufferPool


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
