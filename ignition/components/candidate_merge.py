# SPDX-License-Identifier: Apache-2.0

"""Prepared top-k over valid, interleaved FP32 scores and integer token IDs."""
from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from b12x._lib.compile_plan import attach_programs, compile_only_launches, launch_triton, load_programs
from b12x._lib.compile_pool import CompileJob
from b12x._lib.program_cache import program_cache
from b12x.attention.dsa_indexer.tiled_topk import run_row_topk
from b12x.preparation import PreparedCall
from b12x.preparation.types import plan_from_handle, require_prepared

from ignition.preparation.scratch import planned_scratch, prime_owners
from ignition.preparation.tuning import benchmark_resource, declare, register_finalizer
from .native_operator import native_contract, native_plan


@triton.jit
def _count(Packed, Counts, WIDTH: tl.constexpr, TILES: tl.constexpr, BLOCK: tl.constexpr):
    row, tile = tl.program_id(0), tl.program_id(1)
    col = tile*BLOCK+tl.arange(0, BLOCK)
    index = tl.load(Packed.to(tl.pointer_type(tl.int32))+row*WIDTH*2+col*2+1,
                   mask=col < WIDTH, other=-1)
    tl.store(Counts+row*TILES+tile, tl.sum((index >= 0).to(tl.int32), 0))


@triton.jit
def _compact(Packed, Counts, Values, Ids, Lengths, WIDTH: tl.constexpr,
             TILES: tl.constexpr, BLOCK: tl.constexpr, COUNT_BLOCK: tl.constexpr):
    row, tile = tl.program_id(0), tl.program_id(1)
    prefix_col = tl.arange(0, COUNT_BLOCK)
    counts = tl.load(Counts+row*TILES+prefix_col,
                     mask=(prefix_col < tile) & (prefix_col < TILES), other=0)
    prefix = tl.sum(counts, 0)
    col = tile*BLOCK+tl.arange(0, BLOCK)
    base = row*WIDTH*2+col*2
    score = tl.load(Packed+base, mask=col < WIDTH, other=-float('inf'))
    index = tl.load(Packed.to(tl.pointer_type(tl.int32))+base+1,
                    mask=col < WIDTH, other=-1)
    live = (col < WIDTH) & (index >= 0)
    destination = prefix+tl.cumsum(live.to(tl.int32), 0)-1
    tl.store(Values+row*WIDTH+destination, score, mask=live)
    tl.store(Ids+row*WIDTH+destination, index, mask=live)
    if tile == TILES-1:
        tl.store(Lengths+row, prefix+tl.sum(live.to(tl.int32), 0))


@triton.jit
def _threshold(TopValues, Lengths, Cutoffs, Output, K: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    if tl.load(Lengths+row) > K:
        col = tl.arange(0, BLOCK)
        value = tl.load(TopValues+row*K+col, mask=col < K, other=float('inf'))
        tl.store(Cutoffs+row, tl.min(value,0))
        tl.store(Output+row*K+col, -1, mask=col < K)


@triton.jit
def _counts(Values, Lengths, Cutoffs, Greater, Ties, WIDTH: tl.constexpr,
            K: tl.constexpr, TILES: tl.constexpr, BLOCK: tl.constexpr):
    row, tile = tl.program_id(0), tl.program_id(1)
    length = tl.load(Lengths+row)
    if length > K:
        col = tile*BLOCK+tl.arange(0,BLOCK)
        live = (col < length) & (col < WIDTH)
        value = tl.load(Values+row*WIDTH+col, mask=live, other=-float('inf'))
        cutoff = tl.load(Cutoffs+row)
        tl.store(Greater+row*TILES+tile, tl.sum((live & (value > cutoff)).to(tl.int32),0))
        tl.store(Ties+row*TILES+tile, tl.sum((live & (value == cutoff)).to(tl.int32),0))


@triton.jit
def _select(Values, Ids, Lengths, Cutoffs, Greater, Ties, Output,
            WIDTH: tl.constexpr, K: tl.constexpr, TILES: tl.constexpr,
            BLOCK: tl.constexpr, COUNT_BLOCK: tl.constexpr):
    row, tile = tl.program_id(0), tl.program_id(1)
    length = tl.load(Lengths+row)
    if length > K:
        bins = tl.arange(0,COUNT_BLOCK)
        greater = tl.load(Greater+row*TILES+bins, mask=bins<TILES, other=0)
        ties = tl.load(Ties+row*TILES+bins, mask=bins<tile, other=0)
        greater_before = tl.sum(tl.where(bins<tile,greater,0),0)
        ties_before = tl.sum(ties,0)
        ties_needed = K-tl.sum(greater,0)
        col = tile*BLOCK+tl.arange(0,BLOCK)
        live = (col<length) & (col<WIDTH)
        value = tl.load(Values+row*WIDTH+col, mask=live, other=-float('inf'))
        cutoff = tl.load(Cutoffs+row)
        gt, eq = live & (value>cutoff), live & (value==cutoff)
        gp = tl.cumsum(gt.to(tl.int32),0)
        tp = tl.cumsum(eq.to(tl.int32),0)
        selected = gt | (eq & (ties_before+tp<=ties_needed))
        destination = greater_before+gp+tl.minimum(ties_before+tp,ties_needed)-1
        ids = tl.load(Ids+row*WIDTH+col, mask=selected, other=-1)
        tl.store(Output+row*K+destination,ids,mask=selected)


@triton.jit
def _radix_counts(Values, Ids, Lengths, Cutoffs, Greater, Ties, Output, WIDTH: tl.constexpr, K: tl.constexpr,
                  TILES: tl.constexpr, BLOCK: tl.constexpr, ROW_BLOCK: tl.constexpr, BITS: tl.constexpr):
    """Row top-k + ``_threshold`` + ``_counts`` in one program per row (the ``candidate_merge_radix`` plan setting).

    The K-th largest order key of the compacted row (FP32 bits mapped to the unsigned total order the
    radix top-k uses) is found exactly by a radix select, BITS per pass, most significant digit first.
    It stops as soon as the digit holding the K-th key holds exactly the keys still needed; then (as
    after the last pass) the keys at or above the resolved prefix are precisely a top-K set. The cutoff
    is ``tl.min`` over that set's values, the reduction ``_threshold`` applies to the top-k values
    (minnum: the NaN padding is neutral). Per-tile greater/tie counts and the -1 fill are written for the
    unchanged ``_select``. Rows with at most K candidates copy their compacted IDs, as the top-k does.
    """
    row = tl.program_id(0)
    col = tl.arange(0, ROW_BLOCK)
    value = tl.load(Values+row*WIDTH+col, mask=col < WIDTH, other=0.0)
    length = tl.load(Lengths+row)
    kcol = tl.arange(0, K)
    if length > K:
        live = (col < length) & (col < WIDTH)
        bits = value.to(tl.uint32, bitcast=True)
        # The unsigned total order of FP32 keys (invert negatives, flip the sign bit); a bitwise
        # complement of the widened bits keeps the Python interpreter's uint32 in-range.
        key = tl.where((bits & 0x80000000) != 0, (bits ^ 0xFFFFFFFF).to(tl.uint32), bits | 0x80000000)
        bins = tl.arange(0, 1 << BITS).to(tl.int64)
        remaining = length*0+K
        prefix = (length*0).to(tl.uint32)
        resolved = length*0
        for p in tl.static_range(32 // BITS):
            if resolved == 0:
                if p == 0:
                    match = live
                else:
                    match = live & ((key >> (32-p*BITS)) == prefix)
                digit = ((key >> (32-(p+1)*BITS)) & ((1 << BITS)-1)).to(tl.int32)
                hist = tl.histogram(digit, 1 << BITS, mask=match)
                at_least = tl.cumsum(hist, 0, reverse=True)
                # The digit holding the K-th key is the largest bin with at_least >= remaining; one max
                # reduction returns it with the count above it (non-increasing in the bin) and its own count.
                packed = tl.max(tl.where(at_least >= remaining,
                                         (bins << 32) | ((at_least-hist).to(tl.int64) << 16) | hist.to(tl.int64),
                                         -1), 0)
                above = ((packed >> 16) & 0xFFFF).to(tl.int32)
                remaining = remaining-above
                prefix = (prefix << BITS) | (packed >> 32).to(tl.uint32)
                if (packed & 0xFFFF).to(tl.int32) == remaining:
                    resolved = length*0+(p+1)*BITS
        if resolved == 0:
            resolved = length*0+32
        lower = prefix << (32-resolved).to(tl.uint32)
        cutoff = tl.min(tl.where(live & (key >= lower), value, float("nan")), 0)
        tl.store(Cutoffs+row, cutoff)
        tl.store(Output+row*K+kcol, -1)
        greater = tl.sum(tl.reshape((live & (value > cutoff)).to(tl.int32), (ROW_BLOCK//BLOCK, BLOCK)), 1)
        ties = tl.sum(tl.reshape((live & (value == cutoff)).to(tl.int32), (ROW_BLOCK//BLOCK, BLOCK)), 1)
        tiles = tl.arange(0, ROW_BLOCK//BLOCK)
        tl.store(Greater+row*TILES+tiles, greater, mask=tiles < TILES)
        tl.store(Ties+row*TILES+tiles, ties, mask=tiles < TILES)
    else:
        ids = tl.load(Ids+row*WIDTH+kcol, mask=(kcol < length) & (kcol < WIDTH), other=-1)
        tl.store(Output+row*K+kcol, ids)



@dataclass(frozen=True)
class Query:
    rows: int
    width: int
    topk: int

    @property
    def tiles(self):
        return triton.cdiv(self.width, 512)

    @property
    def scratch_bytes(self):
        # Metadata-only compilation assumes 16-byte-aligned tensor bases.
        offset = 0
        for columns in (self.width, self.width, 1, self.tiles, self.topk, 1, self.tiles):
            offset = (offset+3)//4*4 + self.rows*columns
        return offset*4


CONTRACT = native_contract(component_id="flare.candidate_merge", query_type=Query,
                           tactics=lambda q: (0,), semantic_version=2)


@dataclass(frozen=True)
class RadixQuery(Query):
    """The radix-select variant's query; its own contract keeps the default query and cache keys unchanged."""
    radix: bool = True


CONTRACT_RADIX = native_contract(component_id="flare.candidate_merge_radix", query_type=RadixQuery,
                                 tactics=lambda q: (0,), semantic_version=1)
# 8-bit digits (4 passes at most) at 16 warps measured fastest on SM120 (4/8/10/11/12 bits, 8/16/32 warps).
RADIX_BITS, RADIX_WARPS, RADIX_MAX_WIDTH = 8, 16, 16384


def radix_variant(width, enabled):
    """Whether a merge of ``width`` candidates takes the radix path: the plan setting, and only at most
    RADIX_MAX_WIDTH candidates (the counts travel in 16-bit fields and the row stays in registers);
    wider merges keep the default contract."""
    return bool(enabled) and width <= RADIX_MAX_WIDTH


def merge_query(rows, width, topk, radix):
    """The query and contract for one merge: the radix select replaces the row top-k, threshold and
    cutoff-count kernels (6 -> 4 launches) when the plan asks for it."""
    return (RadixQuery if radix else Query)(rows, width, topk)


def _buffers(scratch, query):
    storage, offset = scratch.view(torch.float32), 0

    def take(columns, dtype):
        nonlocal offset
        offset = (offset+3)//4*4
        count = query.rows*columns
        result = storage[offset:offset+count].view(dtype).view(query.rows, columns)
        offset += count
        return result

    return (take(query.width, torch.float32), take(query.width, torch.int32),
            take(1, torch.int32).view(-1), take(query.tiles, torch.int32),
            take(query.topk, torch.float32), take(1, torch.float32).view(-1),
            take(query.tiles, torch.int32))


@program_cache(scope="preparation")
def compile_merge(payload, ordinal):
    payload = dict(payload)
    q = merge_query(payload["rows"], payload["width"], payload["topk"], payload.get("radix", False))
    packed = torch.empty((q.rows, q.width, 2), dtype=torch.float32, device="meta")
    output = torch.empty((q.rows, q.topk), dtype=torch.int32, device="meta")
    scratch = torch.empty(q.scratch_bytes, dtype=torch.uint8, device="meta")
    values, ids, lengths, counts, top_values, cutoffs, ties = _buffers(scratch, q)
    grid = (q.rows, q.tiles, 1)
    with torch.cuda.device(ordinal), compile_only_launches():
        count = launch_triton(_count, grid, packed, counts, q.width, q.tiles, 512, num_warps=4)
        compact = launch_triton(_compact, grid, packed, counts, values, ids, lengths,
                               q.width, q.tiles, 512, triton.next_power_of_2(q.tiles), num_warps=4)
        if getattr(q, "radix", False):
            radix = launch_triton(_radix_counts, (q.rows, 1, 1), values, ids, lengths, cutoffs, counts, ties, output,
                                  q.width, q.topk, q.tiles, 512, triton.next_power_of_2(q.tiles*512), RADIX_BITS,
                                  num_warps=RADIX_WARPS)
            select = launch_triton(_select, grid, values, ids, lengths, cutoffs, counts, ties, output,
                                   q.width, q.topk, q.tiles, 512, triton.next_power_of_2(q.tiles), num_warps=4)
            return dict(count=count, compact=compact, radix=radix, select=select)
        topk = {}
        run_row_topk(row_logits=values, lengths=lengths, topk=q.topk,
                     output_values=top_values, output_indices=output,
                     output_gather_table=ids, write_values=True, launcher_sink=topk)
        threshold = launch_triton(_threshold, (q.rows, 1, 1), top_values, lengths, cutoffs, output,
                                   q.topk, triton.next_power_of_2(q.topk), num_warps=4)
        count_cutoff = launch_triton(_counts, grid, values, lengths, cutoffs, counts, ties,
                                      q.width, q.topk, q.tiles, 512, num_warps=4)
        select = launch_triton(_select, grid, values, ids, lengths, cutoffs, counts, ties, output,
                               q.width, q.topk, q.tiles, 512, triton.next_power_of_2(q.tiles), num_warps=4)
    return dict(count=count, compact=compact, topk=topk,
                threshold=threshold, count_cutoff=count_cutoff, select=select)


@dataclass(frozen=True)
class State:
    query: Query
    programs: dict

    def run(self, packed, output, scratch):
        q, p = self.query, self.programs
        values, ids, lengths, counts, top_values, cutoffs, ties = _buffers(scratch, q)
        grid = (q.rows, q.tiles, 1)
        p["count"][grid](packed, counts, q.width, q.tiles, 512)
        p["compact"][grid](packed, counts, values, ids, lengths, q.width, q.tiles,
                           512, triton.next_power_of_2(q.tiles))
        if getattr(q, "radix", False):
            p["radix"][(q.rows, 1, 1)](values, ids, lengths, cutoffs, counts, ties, output, q.width, q.topk,
                                       q.tiles, 512, triton.next_power_of_2(q.tiles*512), RADIX_BITS)
            p["select"][grid](values, ids, lengths, cutoffs, counts, ties, output,
                              q.width, q.topk, q.tiles, 512, triton.next_power_of_2(q.tiles))
            return
        run_row_topk(row_logits=values, lengths=lengths, topk=q.topk,
                     output_values=top_values, output_indices=output,
                     output_gather_table=ids, write_values=True, launcher=p["topk"])
        p["threshold"][(q.rows, 1, 1)](top_values, lengths, cutoffs, output,
                                      q.topk, triton.next_power_of_2(q.topk))
        p["count_cutoff"][grid](values, lengths, cutoffs, counts, ties,
                                 q.width, q.topk, q.tiles, 512)
        p["select"][grid](values, ids, lengths, cutoffs, counts, ties, output,
                           q.width, q.topk, q.tiles, 512, triton.next_power_of_2(q.tiles))


@torch.library.custom_op("flare::merge_candidates", mutates_args=("output", "scratch"))
def _execute(packed: torch.Tensor, output: torch.Tensor, scratch: torch.Tensor, handle: int) -> None:
    plan = plan_from_handle(handle)
    require_prepared(plan, plan.component_id, packed.device).run(packed, output, scratch)


@_execute.register_fake
def _fake(packed, output, scratch, handle):
    pass


class PreparedCandidateMerge:
    """Select valid score/ID pairs with deterministic input-order ties.

    IDs occupy the second FP32 slot as int32 bits; negative IDs denote padding.
    Scores for valid IDs must be finite. The selected ID set has the greatest
    available scores, with -1 padding when fewer than top-k IDs are valid.
    Equal cutoff scores prefer earlier gathered positions, so replicas reading
    identical candidate bytes select the same token set regardless of CTA order.
    Sequential calls borrow their compaction storage from the shared arena.
    """

    def __init__(self, packed, output, *, session, workspace, name, radix=False):
        if (packed.ndim != 3 or packed.shape[-1] != 2 or packed.dtype != torch.float32
                or not packed.is_contiguous() or output.ndim != 2 or output.dtype != torch.int32
                or not output.is_contiguous() or output.shape[0] != packed.shape[0]
                or output.device != packed.device or output.shape[1] not in (512, 1024, 2048)):
            raise ValueError("candidate merge requires contiguous FP32 score/ID pairs and int32 top-k output")
        self.packed, self.output = packed, output
        radix = radix_variant(packed.shape[1], radix)
        contract = CONTRACT_RADIX if radix else CONTRACT
        q = merge_query(packed.shape[0], packed.shape[1], output.shape[1], radix)
        payload = contract.encode_query(q)

        def materialize(tactic):
            programs = compile_merge(payload, packed.device.index)
            load_programs(programs)
            if radix:
                return attach_programs(State(q, programs), programs["count"], programs["compact"],
                                       programs["radix"], programs["select"])
            return attach_programs(State(q, programs), programs["count"], programs["compact"],
                                   *programs["topk"].values(), programs["threshold"],
                                   programs["count_cutoff"], programs["select"])

        self.plan = native_plan(contract=CONTRACT, query=q, device=packed.device,
            scratch_bytes=lambda tactic: q.scratch_bytes, materialize=materialize,
            compile_jobs=lambda tactic, device: (CompileJob.create(
                "ignition.components.candidate_merge:compile_merge", payload, device.ordinal),))

        def call(state):
            def inputs():
                source, target = torch.empty_like(packed), torch.empty_like(output)
                source[:, :, 0].normal_()
                source.view(torch.int32)[:, :, 1].copy_(torch.arange(q.width, device=packed.device))
                return source, target
            source, target = benchmark_resource(session, ("candidate_merge", name), inputs)
            scratch = planned_scratch(self.plan.scratch_specs(), workspace)
            return PreparedCall(run=lambda: state.run(source, target, scratch), output=target,
                                owners=prime_owners(workspace, scratch, source, target))

        def prepare(state):
            # Binding uses the existing graph operands. Private benchmark pools
            # are needed only for a measured trial, not for every alias of the
            # same plan in the model's graph catalog.
            scratch = planned_scratch(self.plan.scratch_specs(), workspace)
            return PreparedCall(run=lambda: state.run(packed, output, scratch),
                                owners=prime_owners(workspace, scratch, packed, output))

        declare(session, (self.plan.request(name=name, prepare_call=prepare, benchmark_call=call),))
        register_finalizer(session, lambda: self._bind(workspace), workspace=workspace)

    def _bind(self, workspace):
        self.scratch = planned_scratch(self.plan.scratch_specs(), workspace)

    def __call__(self):
        _execute(self.packed, self.output, self.scratch, self.plan.handle)
        return self.output
