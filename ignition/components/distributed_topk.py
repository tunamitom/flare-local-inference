# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Exact selection from context shards, exchanging candidate scores and never positions."""

import torch

from .collectives import PreparedAllGather


def select_local_topk(gathered, local, indices, *, topk, world, rank, columns, values, positions):
    """Mark this rank's share of the row-wise top-k from every rank's candidate scores.

    `gathered` holds rank-major scores [rows, world * topk] and `local` this rank's [rows, topk]. Every rank
    derives the same threshold and tie count from the same scores, so a rank never needs a peer's positions.
    Ties at the threshold go to lower ranks first, then to earlier local candidates, which selects exactly
    min(topk, live candidates) in total. Returns global positions, -1 where this rank keeps nothing.
    """
    torch.topk(gathered, k=topk, dim=1, largest=True, sorted=False, out=(values, positions))
    threshold = values.amin(1, keepdim=True)
    remaining = topk - (gathered > threshold).sum(1, keepdim=True)
    before = ((gathered == threshold) & (columns < rank * topk)).sum(1, keepdim=True)
    tied = local == threshold
    chosen = (local > threshold) | (tied & (before + tied.cumsum(1) <= remaining))
    chosen &= indices >= 0
    return torch.where(chosen, indices.long() * world + rank, -1).to(torch.int32)


class PreparedDistributedTopK:
    """Select each rank's share of the global top-k by exchanging candidate scores only.

    Context position ``p`` belongs to rank ``p % world`` at local offset ``p // world``.
    Every rank contributes k scores: keeping only k/world could discard global winners.
    Scores travel as bits; the collective transports bytes. Missing local positions have score -inf.
    The output holds global logical positions this rank owns and -1 elsewhere.
    """

    def __init__(self, indices, scores, output, *, world, rank, runtime, session, coordinator,
                 name, buffers):
        if (indices.ndim != 2 or indices.dtype != torch.int32 or scores.shape != indices.shape
                or scores.dtype != torch.float32 or output.shape != indices.shape or output.dtype != indices.dtype
                or world < 2 or rank.shape != (1,) or rank.dtype != torch.int64):
            raise ValueError("distributed top-k requires matching int32 positions, FP32 scores and a rank tensor")
        rows, self.topk = indices.shape
        self.indices, self.scores, self.output = indices, scores, output
        self.world, self.rank = world, rank
        self.local = buffers.zeros((rows, self.topk), torch.float32)
        self.gathered = buffers.zeros((rows, world * self.topk), torch.float32)
        self.values = buffers.zeros(indices.shape, torch.float32)
        self.merge_positions = buffers.zeros(indices.shape, torch.int64)
        self.columns = torch.arange(world * self.topk, device=indices.device)[None]
        self.collective = PreparedAllGather(self.local.view(torch.bfloat16).view(rows, -1),
            self.gathered.view(torch.bfloat16).view(rows, -1), runtime=runtime, session=session,
            coordinator=coordinator, name=name)

    def __call__(self):
        self.local.copy_(torch.where(self.indices >= 0, self.scores, -torch.inf))
        self.collective(self.local.view(torch.bfloat16).view(self.indices.shape[0], -1))
        self.output.copy_(select_local_topk(self.gathered, self.local, self.indices, topk=self.topk,
            world=self.world, rank=self.rank, columns=self.columns, values=self.values,
            positions=self.merge_positions))
        return self.output


def selected_local_records(selected, positions, pages, *, world, rank, page_size):
    """Compact selected global positions into this rank's physical records.

    ``pages`` is a model-page table per query row. The selected positions are independently
    bounded by that row's causal position; negative positions denote inactive rows.
    """
    live = (selected >= 0) & (selected <= positions[:, None]) & (selected % world == rank)
    logical = torch.sort(torch.where(live, selected, torch.iinfo(torch.int32).max), dim=1).values
    count = live.sum(1).to(torch.int32)
    local = logical.long() // world
    page = pages.gather(1, (local // page_size).clamp_max(pages.shape[1] - 1))
    present = torch.arange(selected.shape[1], device=selected.device)[None] < count[:, None]
    physical = torch.where(present & (page >= 0), page.long() * page_size + local % page_size, -1)
    return physical.to(torch.int32), count
