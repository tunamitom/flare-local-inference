# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for the shared DSA selection metadata.

The shared selection metadata is plain torch and is compared bitwise against the
per-layer sequence it replaces. The candidate packing fusion is covered by our
fused pack rider's tests (tests/recipes/test_kimi31_w22_riders.py).
"""

from types import SimpleNamespace

import pytest
import torch

from ignition.recipes.kimi3 import dsa
from ignition.recipes.kimi3.dcp import ContextShards, RECORD_BYTES
from ignition.recipes.kimi3.selection import PreparedSelectionMetadata


def _per_layer_metadata(metadata, shards, rows):
    expanded = metadata.token_page_table[:rows, :, None] * 2 + torch.arange(2, dtype=torch.int32)
    pages = torch.where(metadata.token_page_table[:rows, :, None] >= 0, expanded, -1).flatten(1)
    lengths = shards.local_tokens((metadata.positions[:rows] + 1).clamp_min(0)).to(torch.int32)
    capacity = min(rows, dsa.SELECTION_ROWS)
    active = torch.stack([lengths[stage:stage + capacity].max().reshape(1)[0]
                          for stage in range(0, rows, capacity)])
    return pages, lengths, active


@pytest.mark.parametrize("rows,width,world,rank", [
    (1, 7, 9, 0), (7, 7, 9, 8), (24, 7, 9, 4), (32, 911, 9, 3),
    (33, 7, 9, 5), (40, 7, 2, 1), (96, 911, 9, 8), (32, 1, 1, 0),
])
def test_shared_selection_metadata_matches_per_layer_sequence(rows, width, world, rank):
    generator = torch.Generator().manual_seed(rows * 977 + width * 13 + world * 7 + rank)
    metadata = SimpleNamespace(
        positions=torch.randint(-1, 1 << 20, (rows,), generator=generator, dtype=torch.int64),
        token_page_table=torch.randint(-1, 512, (rows, width), generator=generator, dtype=torch.int32))
    shards = ContextShards.on(rank, world, "cpu")
    buffers = {}
    component = PreparedSelectionMetadata(metadata, shards, rows=rows, device="cpu", buffers=buffers)
    view = component()
    expected = _per_layer_metadata(metadata, shards, rows)
    assert torch.equal(view.pages, expected[0])
    assert torch.equal(view.lengths, expected[1])
    assert torch.equal(view.active, expected[2])
    # A second call rewrites the same buffers from new metadata values.
    metadata.positions.add_(3)
    view = component()
    expected = _per_layer_metadata(metadata, shards, rows)
    assert torch.equal(view.pages, expected[0])
    assert torch.equal(view.lengths, expected[1])
    assert torch.equal(view.active, expected[2])


def test_shared_buffers_keep_a_stale_tail_beyond_the_graph_rows():
    buffers = {}
    def metadata(rows, page):
        return SimpleNamespace(positions=torch.arange(rows, dtype=torch.int64),
            token_page_table=torch.full((rows, 3), page, dtype=torch.int32))
    large = PreparedSelectionMetadata(metadata(96, 7), ContextShards.on(0, 9, "cpu"),
                                      rows=96, device="cpu", buffers=buffers)
    large()
    tail = {key: value.clone() for key, value in buffers.items()}
    small = PreparedSelectionMetadata(metadata(8, 5), ContextShards.on(0, 9, "cpu"),
                                      rows=8, device="cpu", buffers=buffers)
    view = small()
    assert view.pages.shape[0] == 8 and view.lengths.shape[0] == 8 and view.active.shape[0] == 1
    for key, value in tail.items():
        assert torch.equal(buffers[key][view.pages.shape[0] if key == "selection_pages" else 8:],
                           value[view.pages.shape[0] if key == "selection_pages" else 8:]), key
    assert torch.equal(buffers["selection_pages"][:8], view.pages)


def _construct_shared(monkeypatch, rows, world):
    def selector(query, weights, cache, pages, lengths, active, output, **kwargs):
        return SimpleNamespace(query=query, weights=weights, cache=cache, pages=pages,
            lengths=lengths, active=active, output=output, scores=kwargs["scores"])

    def merge(indices, scores, output, **kwargs):
        return SimpleNamespace(indices=indices, scores=scores, output=output)

    def attention(query, cache, output, pages, lengths, starts, **kwargs):
        return SimpleNamespace(query=query, cache=cache, output=output, pages=pages,
            lengths=lengths, starts=starts, lse=kwargs["lse"])

    def collective(source, output, **kwargs):
        return SimpleNamespace(source=source, output=output)

    def combine(partial, lse, output, **kwargs):
        return SimpleNamespace(partial=partial, lse=lse, output=output)

    monkeypatch.setattr(dsa, "prepare_projection", lambda *args, **kwargs: None)
    monkeypatch.setattr(dsa, "PreparedFP8Indexer", selector)
    monkeypatch.setattr(dsa, "PreparedDistributedTopK", merge)
    monkeypatch.setattr(dsa, "PreparedDenseMLA", attention)
    monkeypatch.setattr(dsa, "PreparedAllGather", collective)
    monkeypatch.setattr(dsa, "PreparedLseReduceScatter", combine)
    monkeypatch.setattr(dsa.dense_mla, "Caps", lambda **kwargs: SimpleNamespace(**kwargs))
    capacity = dsa.SELECTION_ROWS
    metadata = SimpleNamespace(positions=torch.arange(rows, dtype=torch.int64),
        token_page_table=torch.zeros((rows, 2), dtype=torch.int32))
    selection_rows = rows if rows <= capacity else capacity
    shared = SimpleNamespace(
        pages=torch.zeros((rows, 4), dtype=torch.int32),
        lengths=torch.zeros((rows,), dtype=torch.int32),
        active=torch.zeros((-(-rows // selection_rows),), dtype=torch.int32))
    context = None if world == 1 else SimpleNamespace(
        shards=ContextShards.on(0, world, "cpu"), candidates=object(), heads=object())
    op = dsa.PreparedDSA(
        {"index_query": None, "index_norm": torch.ones(128), "index_bias": torch.zeros(128)},
        torch.zeros((rows, 8)), torch.zeros((rows, 128)), torch.zeros((rows, 32)),
        torch.zeros((rows, 1, RECORD_BYTES), dtype=torch.bfloat16),
        torch.zeros((rows, 1, 512), dtype=torch.bfloat16), torch.zeros(rows, dtype=torch.int64), metadata,
        torch.zeros((2, 128, RECORD_BYTES), dtype=torch.float8_e4m3fn),
        torch.zeros((4, 64 * 132), dtype=torch.uint8), context=context,
        session=None, coordinator=None, name="attention", workspace=None, buffers={}, selection=shared)
    return op, shared, capacity


@pytest.mark.parametrize("rows,world", [(8, 9), (33, 9), (96, 9), (8, 1)])
def test_layer_binds_shared_selection_slices(monkeypatch, rows, world):
    op, shared, capacity = _construct_shared(monkeypatch, rows, world)
    assert len(op.stages) == -(-rows // capacity)
    for begin, end, selector, merge, quantized, gather, attention, combine, lse, paired in op.stages:
        stage = begin // capacity
        assert torch.equal(selector.pages, shared.pages[begin:end])
        assert torch.equal(selector.lengths, shared.lengths[begin:end])
        assert torch.equal(selector.active, shared.active[stage:stage + 1])


