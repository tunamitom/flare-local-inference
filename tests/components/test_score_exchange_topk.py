# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""The score-exchange top-k declares one FP32 score per candidate slot and runs pack, gather, share in order.

Ported from TP9 8b1be5b (tests/components/test_score_exchange_topk.py); extended for the
ada7d02 collect()/merge_result() split that lets a caller overlap the merge with other work.
"""
from types import SimpleNamespace

import pytest
import torch

from ignition.components import candidate_share, distributed_topk


class _Buffers:
    def zeros(self, shape, dtype):
        return torch.zeros(shape, dtype=dtype)


def test_score_exchange_gathers_scores_and_selects_this_rank_share(monkeypatch):
    rows, topk, world = 3, 2048, 9
    events = []

    class Gather:
        def __init__(self, source, output, *, runtime, session, coordinator, name):
            self.arguments = (source, output, runtime, name)
            assert source.dtype == output.dtype == torch.float32
            assert source.shape == (rows, topk) and output.shape == (rows, world * topk)

        def __call__(self, source):
            assert source is self.arguments[0]
            events.append("gather")

    class Share:
        def __init__(self, gathered, indices, scores, rank, output, *, world, session, workspace, name):
            self.arguments = (gathered, indices, scores, rank, output, world, name)

        def __call__(self):
            events.append("share")

    monkeypatch.setattr(distributed_topk, "PreparedAllGather", Gather)
    monkeypatch.setattr(candidate_share, "PreparedCandidateShare", Share)
    monkeypatch.setattr(distributed_topk, "_pack_local_keys",
                        lambda indices, scores, keys: events.append(("pack", indices, scores, keys)))
    indices = torch.zeros((rows, topk), dtype=torch.int32)
    scores = torch.zeros((rows, topk))
    output = torch.zeros((rows, topk), dtype=torch.int32)
    rank = torch.tensor([5])
    runtime = SimpleNamespace()
    exchange = distributed_topk.PreparedScoreExchangeTopK(indices, scores, output, world=world, rank=rank,
        runtime=runtime, session=None, coordinator=None, name="layer3.rows0.candidate_scores",
        buffers=_Buffers(), workspace=None)
    assert exchange.collective.arguments[2:] == (runtime, "layer3.rows0.candidate_scores")
    gathered = exchange.collective.arguments[1]
    assert exchange.share.arguments[0] is gathered
    assert exchange.share.arguments[1:] == (indices, scores, rank, output, world, "layer3.rows0.candidate_scores.share")
    assert exchange() is output
    assert [event if isinstance(event, str) else event[0] for event in events] == ["pack", "gather", "share"]
    assert events[0][1:] == (indices, scores, exchange.keys)
    with pytest.raises(ValueError):
        distributed_topk.PreparedScoreExchangeTopK(indices, scores, output, world=1, rank=rank, runtime=runtime,
            session=None, coordinator=None, name="n", buffers=_Buffers(), workspace=None)


def test_collect_and_merge_result_split_the_stage(monkeypatch):
    # ada7d02's split: collect() packs and gathers, merge_result() selects the share — so a
    # caller can start other work between the exchange and the selection (the DSA query
    # quantize + all-gather branch overlaps the row top-k and the select on the main stream).
    rows, topk, world = 2, 2048, 4
    events = []

    class Gather:
        def __init__(self, source, output, **kwargs):
            pass

        def __call__(self, source):
            events.append("gather")

    class Share:
        def __init__(self, gathered, indices, scores, rank, output, **kwargs):
            pass

        def __call__(self):
            events.append("share")

    monkeypatch.setattr(distributed_topk, "PreparedAllGather", Gather)
    monkeypatch.setattr(candidate_share, "PreparedCandidateShare", Share)
    monkeypatch.setattr(distributed_topk, "_pack_local_keys",
                        lambda indices, scores, keys: events.append("pack"))
    indices = torch.zeros((rows, topk), dtype=torch.int32)
    scores = torch.zeros((rows, topk))
    output = torch.zeros((rows, topk), dtype=torch.int32)
    exchange = distributed_topk.PreparedScoreExchangeTopK(indices, scores, output, world=world,
        rank=torch.tensor([0]), runtime=SimpleNamespace(), session=None, coordinator=None,
        name="layer0.rows0.candidate_scores", buffers=_Buffers(), workspace=None)
    assert exchange.collect() is None
    assert events == ["pack", "gather"]
    assert exchange.merge_result() is output
    assert events == ["pack", "gather", "share"]


def test_share_scratch_holds_the_order_and_the_values_of_the_k_greatest_scores():
    query = candidate_share.Query(rows=24, world=9, topk=2048)
    assert query.width == 18432 and query.scratch_bytes == 2 * 24 * 2048 * 4
    order, values = candidate_share._buffers(torch.zeros(query.scratch_bytes, dtype=torch.uint8), query)
    assert order.dtype == torch.int32 and values.dtype == torch.float32
    assert order.shape == values.shape == (24, 2048)
    assert values.data_ptr() - order.data_ptr() == 24 * 2048 * 4


def test_pack_local_keys_writes_one_score_per_slot():
    # The CPU body of ignition::pack_candidate_keys: a slot holds its candidate's score and
    # -inf for a slot without a candidate or with a -inf or NaN score (the kernel's
    # `score > -inf` test; NaN does not compare greater than -inf).
    rows, topk = 2, 512
    indices = torch.full((rows, topk), -1, dtype=torch.int32)
    scores = torch.randn((rows, topk))
    indices[0, :4] = torch.tensor([7, 8, 9, 10], dtype=torch.int32)
    indices[1, :3] = torch.tensor([1, 2, 3], dtype=torch.int32)
    scores[0, :4] = torch.tensor([1.5, -torch.inf, torch.nan, 0.0])
    scores[1, :3] = torch.tensor([2.0, 3.0, 4.0])
    keys = torch.zeros((rows, topk))
    distributed_topk._pack_local_keys(indices, scores, keys)
    assert keys[0, 0] == 1.5 and keys[0, 3] == 0.0
    assert keys[0, 1] == -torch.inf and keys[0, 2] == -torch.inf
    assert torch.equal(keys[1, :3], torch.tensor([2.0, 3.0, 4.0]))
    assert (keys[0, 4:] == -torch.inf).all() and (keys[1, 3:] == -torch.inf).all()
    with pytest.raises(ValueError):
        distributed_topk._pack_local_keys(indices, scores, torch.zeros((rows, topk, 2)))
    with pytest.raises(ValueError):
        distributed_topk._pack_local_keys(indices, scores.double(), keys)
