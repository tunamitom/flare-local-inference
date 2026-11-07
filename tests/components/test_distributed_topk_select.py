# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Scores-only selection picks a valid global top-k with every rank deciding alone."""

import pytest
import torch

from ignition.components.distributed_topk import select_local_topk


def selected(scores, indices, topk):
    world, rows = scores.shape[:2]
    gathered = scores.permute(1, 0, 2).reshape(rows, -1)
    columns = torch.arange(gathered.shape[1])[None]
    picks = []
    for rank in range(world):
        rank_tensor = torch.tensor([rank])
        out = select_local_topk(gathered, scores[rank], indices[rank], topk=topk, world=world, rank=rank_tensor,
                                columns=columns, values=torch.empty(rows, topk),
                                positions=torch.empty((rows, topk), dtype=torch.int64))
        assert (out[out >= 0] % world == rank).all()
        picks.append(out)
    return torch.stack(picks)


@pytest.mark.parametrize("levels", [None, 4, 2])
def test_every_rank_agrees_on_exactly_k_top_scores(levels):
    torch.manual_seed(1 + (levels or 0))
    world, rows, topk = 16, 8, 64
    scores = torch.randn(world, rows, topk)
    if levels:
        # Quantized scores make threshold ties the common case.
        scores = (scores * levels).round() / levels
    indices = torch.arange(topk)[None, None].expand(world, rows, -1).clone()
    picks = selected(scores, indices, topk)
    for row in range(rows):
        row_scores, row_picks = scores[:, row].flatten(), picks[:, row].flatten()
        kept = row_picks >= 0
        assert kept.sum() == topk
        threshold = row_scores.topk(topk).values
        assert torch.equal(row_scores[kept].sort(descending=True).values, threshold)


def test_no_ties_match_the_position_merge_exactly():
    torch.manual_seed(7)
    world, rows, topk = 16, 8, 64
    scores = torch.randn(world, rows, topk)
    indices = torch.randperm(topk * 1000)[:topk][None, None].expand(world, rows, -1).clone()
    picks = selected(scores, indices, topk)
    gathered = scores.permute(1, 0, 2).reshape(rows, -1)
    global_ids = (indices.long() * world + torch.arange(world)[:, None, None]).permute(1, 0, 2).reshape(rows, -1)
    expected = global_ids.gather(1, gathered.topk(topk, dim=1).indices).sort(1).values
    for row in range(rows):
        row_picks = picks[:, row].flatten()
        assert torch.equal(row_picks[row_picks >= 0].sort().values.long(), expected[row])


def test_missing_candidates_are_never_selected():
    world, rows, topk = 16, 2, 8
    scores = torch.full((world, rows, topk), -torch.inf)
    indices = torch.full((world, rows, topk), -1, dtype=torch.int32)
    scores[3, 0, :3] = torch.tensor([3., 2., 1.])
    indices[3, 0, :3] = torch.tensor([5, 6, 7])
    picks = selected(scores, indices, topk)
    assert sorted(picks[picks >= 0].tolist()) == [5 * world + 3, 6 * world + 3, 7 * world + 3]
