# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Exact selection from context shards, with bounded candidate communication."""

import torch
import triton
import triton.language as tl

from .candidate_merge import PreparedCandidateMerge
from .collectives import PreparedAllGather

# Fused decode score/ID pack (the ``decode_fusion`` plan setting) -----------
#
# The shipped pack writes the score/ID pairs with 8 dispatched kernels per call
# (ge, where, copy, cast, mul, add, where, copy). The fused pack replaces them
# with one small compiled kernel that reads ``indices``, ``scores`` and ``rank``
# and writes the pairs straight into the operator's existing ``self.local``
# cursor buffer (``buffers.zeros``, the ``GraphBufferPool`` pattern used by
# ``_records``). It is select, copy and integer arithmetic only, so it is exact
# by construction, and it allocates nothing: no process-local temporary and no
# registry. The gate is a constructor argument threaded from the preparation
# config (``--decode-fusion`` / ``Plan.decode_fusion``); default off keeps the
# shipped 8-kernel pack bit for bit. The component never reads the environment.


def _pack_pairs_reference(local, indices, scores, rank, world):
    """CPU semantics of ``_pack_pairs``: the same select/copy/integer arithmetic.

    Dead entries (``indices < 0``) pack as score ``-inf`` and ID ``-1``; live
    entries pack as their FP32 score and ``int64(index) * world + rank`` narrowed
    to int32 bits in the second slot (rank-major, FP32 score and int32 id).
    Ties are plain byte copies, so equal scores on different ranks pack
    symmetrically.
    """
    live = indices >= 0
    packed = indices.to(torch.int64).mul_(world).add_(rank)
    packed.masked_fill_(~live, -1)
    local.view(torch.int32)[:, :, 1].copy_(packed)
    local[:, :, 0].copy_(torch.where(live, scores, -torch.inf))


@triton.jit
def _pack_pairs(Indices, Scores, Rank, Local, Ids,
                WIDTH: tl.constexpr, WORLD: tl.constexpr, BLOCK: tl.constexpr,
                TENSOR_RANK: tl.constexpr, STATIC_RANK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    mask = col < WIDTH
    index = tl.load(Indices + row*WIDTH + col, mask=mask, other=-1)
    score = tl.load(Scores + row*WIDTH + col, mask=mask, other=0.0)
    rank = STATIC_RANK
    if TENSOR_RANK:
        rank = tl.load(Rank)
    live = mask & (index >= 0)
    slot = (row*WIDTH + col)*2
    tl.store(Local + slot, tl.where(live, score, -float('inf')), mask=mask)
    tl.store(Ids + slot + 1,
             tl.where(live, (index.to(tl.int64)*WORLD + rank).to(tl.int32), -1), mask=mask)


@torch.library.custom_op("ignition::pack_score_id_pairs", mutates_args=("local",))
def pack_score_id_pairs(local: torch.Tensor, indices: torch.Tensor, scores: torch.Tensor,
                        rank: torch.Tensor, world: int) -> None:
    """One compiled kernel replacing the 8-kernel pack (``decode_fusion``).

    Writes score/ID pairs straight into ``local``, an existing cursor buffer:
    it allocates nothing and re-views no storage across element sizes. The CUDA
    body launches ``_pack_pairs`` (one kernel = one round-graph node per call);
    the CPU body runs the same arithmetic for the CPU tests.
    """
    if (local.stride() != (local.shape[1] * 2, 2, 1) or indices.stride() != (indices.shape[1], 1)
            or scores.stride() != indices.stride()):
        raise ValueError("fused pack requires contiguous row-major pair, index and score tensors")
    if local.is_cuda:
        rows, width = indices.shape
        torch.library.wrap_triton(_pack_pairs)[(rows,)](
            indices, scores, rank, local, local.view(torch.int32),
            width, world, triton.next_power_of_2(width), True, 0, num_warps=8)
    else:
        _pack_pairs_reference(local, indices, scores, rank, world)


@pack_score_id_pairs.register_fake
def _pack_score_id_pairs_fake(local, indices, scores, rank, world):
    pass


class PreparedDistributedTopK:
    """Merge each rank's local top-k into global logical positions.

    Context position ``p`` belongs to rank ``p % world`` at local offset ``p // world``.
    Every rank contributes k candidates: keeping only k/world could discard global winners.
    Candidate IDs travel as integer bits alongside FP32 scores; the collective transports bytes.
    Missing local positions have score -inf and ID -1. All ranks receive identical candidates.
    """

    def __init__(self, indices, scores, output, *, world, rank, runtime, session, coordinator,
                 name, buffers, workspace, decode_fusion):
        if (indices.ndim != 2 or indices.dtype != torch.int32 or scores.shape != indices.shape
                or scores.dtype != torch.float32 or output.shape != indices.shape or output.dtype != indices.dtype
                or world < 2 or rank.shape != (1,) or rank.dtype != torch.int64):
            raise ValueError("distributed top-k requires matching int32 positions, FP32 scores and a rank tensor")
        rows, self.topk = indices.shape
        self.indices, self.scores, self.output = indices, scores, output
        self.world, self.rank = world, rank
        self.local = buffers.zeros((rows, self.topk, 2), torch.float32)
        self.gathered = buffers.zeros((rows, world * self.topk, 2), torch.float32)
        self.decode_fusion = bool(decode_fusion)
        self.merge = PreparedCandidateMerge(self.gathered, output, session=session,
            workspace=workspace, name=name + ".merge")
        self.collective = PreparedAllGather(self.local.view(torch.bfloat16).view(rows, -1),
            self.gathered.view(torch.bfloat16).view(rows, -1), runtime=runtime, session=session,
            coordinator=coordinator, name=name)

    def __call__(self):
        if self.decode_fusion:
            pack_score_id_pairs(self.local, self.indices, self.scores, self.rank, self.world)
        else:
            # Shipped pack: 8 dispatched kernels per call.
            live = self.indices >= 0
            self.local[:, :, 0].copy_(torch.where(live, self.scores, -torch.inf))
            self.local.view(torch.int32)[:, :, 1].copy_(
                torch.where(live, self.indices.long() * self.world + self.rank, -1))
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


def _extent(tensor):
    """The tensor's byte range in its storage: the strided bounding box [start, end)."""
    start = tensor.data_ptr()
    if tensor.numel() == 0:
        return start, start
    span = 1 + sum((size - 1) * abs(stride) for size, stride in zip(tensor.shape, tensor.stride()))
    return start, start + span * tensor.element_size()


def _overlaps(tensor, others):
    """True when ``tensor`` shares any byte with one of ``others``.

    Buffers carved from one shared pool chunk are disjoint and must pass: the
    check compares byte ranges, never whole storages.
    """
    start, end = _extent(tensor)
    return any(start < hi and lo < end for lo, hi in (_extent(other) for other in others))


def selected_local_records(selected, positions, pages, *, world, rank, page_size, out=None, count_out=None):
    """Compact selected global positions into this rank's physical records.

    ``pages`` is a model-page table per query row. The selected positions are independently
    bounded by that row's causal position; negative positions denote inactive rows.

    ``out``/``count_out`` are the caller's retained buffers: when given, the compaction
    kernel writes them directly and the caller's follow-up copy disappears (one write per
    element instead of two). Default ``None`` allocates exactly as before.

    ``out`` and ``count_out`` must not overlap ``selected``, ``positions`` or ``pages``
    (or each other) by byte range -- ``out=`` aliasing an input of the same kernel is
    undefined. Disjoint buffers carved from one shared pool chunk are legal and pass.
    Bytes beyond each row's ``count`` are ``-1`` and stay don't care for every consumer:
    consumers read exactly ``[0, count)`` records and nothing past them.
    """
    if out is None:
        output = torch.empty_like(selected, memory_format=torch.contiguous_format)
    else:
        if out.shape != selected.shape or out.dtype != selected.dtype or not out.is_contiguous():
            raise ValueError("out must be a contiguous buffer matching selected")
        # The overlap contract is a caller invariant, checked eagerly only: Dynamo cannot
        # trace data-pointer comparisons (and at capture time the caller's retained
        # buffers are the prepared, statically distinct ones).
        if not torch.compiler.is_compiling() and _overlaps(out, (selected, positions, pages)):
            raise ValueError("out must not overlap selected, positions or pages")
        output = out
    if count_out is None:
        count = torch.empty((selected.shape[0],), dtype=torch.int32, device=selected.device)
    else:
        if count_out.shape != (selected.shape[0],) or count_out.dtype != torch.int32 or not count_out.is_contiguous():
            raise ValueError("count_out must be a contiguous int32 buffer with one entry per row")
        if not torch.compiler.is_compiling() and _overlaps(count_out, (selected, positions, pages, output)):
            raise ValueError("count_out must not overlap out, selected, positions or pages")
        count = count_out
    tensor_rank = isinstance(rank, torch.Tensor)
    torch.library.wrap_triton(_records)[(selected.shape[0],)](
        selected, positions, pages, rank if tensor_rank else positions, output, count,
        *selected.stride(), positions.stride(0), *pages.stride(),
        selected.shape[1], pages.shape[1], world, page_size,
        triton.next_power_of_2(selected.shape[1]), tensor_rank,
        0 if tensor_rank else rank, num_warps=8)
    return output, count
