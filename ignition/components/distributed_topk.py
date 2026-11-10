# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Exact selection from context shards, with bounded candidate communication."""

import torch
import triton
import triton.language as tl
from .candidate_merge import PreparedCandidateMerge

from .collectives import PreparedAllGather


class PreparedDistributedTopK:
    """Merge each rank's local top-k into global logical positions.

    Context position ``p`` belongs to rank ``p % world`` at local offset ``p // world``.
    Every rank contributes k candidates: keeping only k/world could discard global winners.
    Candidate IDs travel as integer bits alongside FP32 scores; the collective transports bytes.
    Missing local positions have score -inf and ID -1. All ranks receive identical candidates.

    With ``piggyback=(local_query, gathered_query)`` the candidate exchange also carries
    the caller's query bytes in the same launch: one exchange and one rank barrier replace
    the separate query gather and the candidate gather (W4 R6). The piggybacked fields are
    packed rank-major beside each rank's candidates and split back out after the exchange,
    so the merged candidate layout is unchanged.
    """

    def __init__(self, indices, scores, output, *, world, rank, runtime, session, coordinator,
                 name, buffers, workspace, piggyback=None):
        if (indices.ndim != 2 or indices.dtype != torch.int32 or scores.shape != indices.shape
                or scores.dtype != torch.float32 or output.shape != indices.shape or output.dtype != indices.dtype
                or world < 2 or rank.shape != (1,) or rank.dtype != torch.int64):
            raise ValueError("distributed top-k requires matching int32 positions, FP32 scores and a rank tensor")
        rows, self.topk = indices.shape
        self.indices, self.scores, self.output = indices, scores, output
        self.world, self.rank = world, rank
        self.local = buffers.zeros((rows, self.topk, 2), torch.float32)
        self.gathered = buffers.zeros((rows, world * self.topk, 2), torch.float32)
        self.merge = PreparedCandidateMerge(self.gathered, output, session=session,
            workspace=workspace, name=name + ".merge")
        self.paired = self.collective = None
        if piggyback is None:
            self.collective = PreparedAllGather(self.local.view(torch.bfloat16).view(rows, -1),
                self.gathered.view(torch.bfloat16).view(rows, -1), runtime=runtime, session=session,
                coordinator=coordinator, name=name)
        else:
            from .peer_rows import PreparedPairedGather

            local_query, gathered_query = piggyback
            candidate_bytes = self.topk * 2 * 4
            query_bytes = local_query.numel() * local_query.element_size() // rows
            if (local_query.shape[0] != rows or gathered_query.shape[0] != rows
                    or gathered_query.numel() * gathered_query.element_size() != rows * query_bytes * world
                    or query_bytes % 16 or candidate_bytes % 16):
                raise ValueError("piggybacked queries need 16-byte row fields matching the candidate rows")
            self.paired = PreparedPairedGather(self.local.view(rows, -1), local_query,
                self.gathered.view(rows, -1), gathered_query, runtime=runtime, session=session,
                coordinator=coordinator, name=name, buffers=buffers)
            self.piggyback_first = self.local.view(rows, -1)
            self.piggyback_second = local_query

    def __call__(self):
        live = self.indices >= 0
        self.local[:, :, 0].copy_(torch.where(live, self.scores, -torch.inf))
        self.local.view(torch.int32)[:, :, 1].copy_(
            torch.where(live, self.indices.long() * self.world + self.rank, -1))
        if self.paired is not None:
            self.paired(self.piggyback_first, self.piggyback_second)
        else:
            self.collective(self.local.view(torch.bfloat16).view(self.indices.shape[0], -1))
        self.merge()
        return self.output



@triton.jit
def _records(Selected, Positions, Pages, Rank, Output, Counts,
             selected_row_stride, selected_col_stride, position_stride,
             page_row_stride, page_col_stride,
             WIDTH: tl.constexpr, COLUMNS: tl.constexpr, WORLD: tl.constexpr,
             PAGE: tl.constexpr, BLOCK: tl.constexpr, TENSOR_RANK: tl.constexpr,
             STATIC_RANK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    selected = tl.load(Selected + row*selected_row_stride + col*selected_col_stride,
                       mask=col < WIDTH, other=-1)
    position = tl.load(Positions + row*position_stride)
    rank = STATIC_RANK
    if TENSOR_RANK:
        rank = tl.load(Rank)
    live = (col < WIDTH) & (selected >= 0) & (selected <= position) & (selected % WORLD == rank)
    count = tl.sum(live.to(tl.int32), 0)
    logical = tl.sort(tl.where(live, selected, 2147483647), descending=False)
    local = logical.to(tl.int64)//WORLD
    page_col = tl.minimum(local//PAGE, COLUMNS-1)
    page = tl.load(Pages + row*page_row_stride + page_col*page_col_stride,
                   mask=(col < WIDTH) & (col < count), other=-1).to(tl.int64)
    physical = tl.where((col < count) & (page >= 0), page*PAGE + local % PAGE, -1)
    tl.store(Output + row*WIDTH + col, physical.to(tl.int32), mask=col < WIDTH)
    tl.store(Counts + row, count)


def selected_local_records(selected, positions, pages, *, world, rank, page_size):
    """Compact selected global positions into this rank's physical records.

    ``pages`` is a model-page table per query row. The selected positions are independently
    bounded by that row's causal position; negative positions denote inactive rows.
    """
    output = torch.empty_like(selected, memory_format=torch.contiguous_format)
    count = torch.empty((selected.shape[0],), dtype=torch.int32, device=selected.device)
    tensor_rank = isinstance(rank, torch.Tensor)
    torch.library.wrap_triton(_records)[(selected.shape[0],)](
        selected, positions, pages, rank if tensor_rank else positions, output, count,
        *selected.stride(), positions.stride(0), *pages.stride(),
        selected.shape[1], pages.shape[1], world, page_size,
        triton.next_power_of_2(selected.shape[1]), tensor_rank,
        0 if tensor_rank else rank, num_warps=8)
    return output, count
