"""One-program speculative commit for request states without token history.

The compiled commit in ``PreparedSpeculativeCommit`` lowers to about forty
small kernels separated by gathers, scatters and the record copy. Batches are
at most 64 rows and the request capacity at most 64 slots, so one CTA holds
every row and slot: it evaluates all rows from the state before the round,
writes the per-row outputs, then lets each slot's last selected row write that
slot, exactly as the unfused commit orders its reads and writes.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ignition.components.sampling import (
    DUPLICATE_SLOT,
    EOS_CAPACITY,
    ERROR,
    FINISHED,
    INVALID_DISTRIBUTION,
    RUNNING,
)
from ignition.preparation.triton import prepare_triton

INVALID_SPECULATION = 6
RECORD_FIELDS = 24


@triton.jit
def _commit(Generation, Consumed, consumed_stride, Generated, Pending, Status, Error, Counter, Histogram,
            MaxNew, EosIds, EosCount,
            TokenIds, Proposed, AcceptedIn, Offered, ValidIn, Increment,
            Inputs, Slots, Generations, Live, Limits, CopyMask, TargetOffset, RngGeneration,
            Sequence, Frame, Records, Idle,
            EmittedIds, CommittedInputs, EmittedCount, AcceptedCount, AcceptedDrafts, CommitColumn,
            ConsumedAfter, Owners, Staged,
            B: tl.constexpr, W: tl.constexpr, C: tl.constexpr, V: tl.constexpr, EOS: tl.constexpr,
            LAG: tl.constexpr, HAS_LAG: tl.constexpr, HAS_LIMITS: tl.constexpr, HAS_RNG: tl.constexpr,
            RUNNING_: tl.constexpr, FINISHED_: tl.constexpr, ERROR_: tl.constexpr,
            DUPLICATE_: tl.constexpr, INVALID_DIST_: tl.constexpr, INVALID_SPEC_: tl.constexpr,
            FIELDS: tl.constexpr):
    rows = tl.arange(0, B)
    cols = tl.arange(0, 8)
    in_width = cols < W
    slots = tl.load(Slots + rows)
    safe = tl.minimum(tl.maximum(slots, 0), C - 1).to(tl.int64)
    live = tl.load(Live)
    selected = ((rows < live) & (live >= 0) & (live <= B) & (slots >= 0) & (slots < C)
                & (tl.load(Generation + safe) == tl.load(Generations + rows)))

    # Each row commits from its slot's last selected row.
    matches = selected[:, None] & selected[None, :] & (slots[:, None] == slots[None, :])
    count = tl.sum(matches.to(tl.int32), 1)
    row = tl.max(tl.where(matches, rows[None, :], 0), 1)
    leader = selected & (row == rows)

    grid2 = row[:, None] * W + cols[None, :]
    ids = tl.load(TokenIds + grid2, mask=in_width[None, :], other=0)
    inputs = tl.load(Inputs + grid2, mask=in_width[None, :], other=0)
    shifted = tl.load(Inputs + grid2 + 1, mask=(cols < W - 1)[None, :], other=0)
    proposed = tl.load(Proposed + row).to(tl.int64)
    accepted = tl.load(AcceptedIn + row).to(tl.int64)
    offered = tl.load(Offered + row).to(tl.int64)
    valid = tl.load(ValidIn + row) != 0
    increment = tl.load(Increment + row)
    status_before = tl.load(Status + safe)
    error_before = tl.load(Error + safe)
    generated_before = tl.load(Generated + safe)
    pending_before = tl.load(Pending + safe)
    max_new = tl.load(MaxNew + safe)
    remaining = tl.maximum(max_new - generated_before, 0)
    enabled = (count == 1) & (status_before == RUNNING_) & (remaining > 0)
    proposal = (cols[None, :] < proposed[:, None]) & in_width[None, :]
    shape_valid = ((proposed >= 1) & (proposed <= W) & (accepted == proposed - 1)
                   & (offered >= accepted) & (offered <= W - 1))
    bad_ids = proposal & ~((ids >= 0) & (ids < V))
    bad_inputs = proposal & ~((inputs >= 0) & (inputs < V))
    chain = (cols[None, :] < W - 1) & (cols[None, :] < accepted[:, None]) & (ids != shifted)
    shape_valid &= tl.sum((bad_ids | bad_inputs | chain).to(tl.int32), 1) == 0
    first_input = tl.sum(tl.where(cols[None, :] == 0, inputs, 0), 1)
    shape_valid &= (pending_before >= 0) & (first_input == pending_before)
    if HAS_RNG:
        shape_valid &= increment >= 0
    else:
        shape_valid &= increment == 0
    valid = valid & shape_valid

    eos_count = tl.load(EosCount + safe)
    eos = tl.zeros((B, 8), dtype=tl.int1)
    for e in tl.static_range(EOS):
        token = tl.load(EosIds + safe * EOS + e)
        eos |= (ids == token[:, None]) & (e < eos_count)[:, None]
    eos_position = tl.min(tl.where(eos & proposal, cols[None, :] + 1, W), 1).to(tl.int64)
    emitted = tl.minimum(tl.minimum(proposed, remaining), eos_position)
    if HAS_LIMITS:
        emitted = tl.minimum(emitted, tl.load(Limits + row).to(tl.int64))
    emitted = tl.where(enabled & valid, emitted, 0)
    emitted_drafts = tl.minimum(tl.maximum(accepted, 0), emitted)
    take = (cols[None, :] < emitted[:, None]) & in_width[None, :]
    last = tl.minimum(tl.maximum(emitted - 1, 0), W - 1)
    pending = tl.sum(tl.where(cols[None, :] == last[:, None], ids, 0), 1)
    consumed = tl.load(Consumed + safe * consumed_stride) + emitted
    generated = generated_before + emitted
    terminal = (generated >= max_new) | (tl.sum((eos & take).to(tl.int32), 1) > 0)
    failed = enabled & ~valid
    duplicate = count > 1
    status = tl.where(duplicate | failed, ERROR_,
                      tl.where((count == 1) & (status_before == RUNNING_) & (terminal | (remaining == 0)),
                               FINISHED_, status_before))
    error = tl.where(duplicate, DUPLICATE_,
                     tl.where(failed, tl.where(shape_valid, INVALID_DIST_, INVALID_SPEC_), error_before))
    generation = tl.load(Generation + safe)
    counters = tl.load(Counter + safe) + emitted
    if HAS_RNG:
        offset = tl.load(TargetOffset + safe)
        rng_generation = tl.load(RngGeneration + safe)
        base = tl.where(rng_generation == generation, offset, 0)
        advance = (emitted > 0) & (increment > 0)
        rng_offset = tl.where(advance, base + increment, offset)
        rng_generation = tl.where(advance, generation, rng_generation)
    live_row = enabled & valid
    sequence = tl.load(Sequence)
    if HAS_LAG:
        lagged = tl.maximum(consumed - LAG, 0)
    else:
        lagged = tl.full((B,), -1, tl.int64)
    copy_mask = tl.load(CopyMask + row)

    # The record: sixteen scalar fields, then the emitted ids padded with -1.
    k = tl.arange(0, 32)[None, :]
    one = tl.zeros((B, 32), dtype=tl.int64)
    record = tl.where(k == 0, generation[:, None] + one, -1)
    record = tl.where(k == 1, sequence + one, record)
    record = tl.where(k == 2, consumed[:, None] + one, record)
    record = tl.where(k == 3, generated[:, None] + one, record)
    record = tl.where(k == 4, tl.where(emitted > 0, pending, -1)[:, None] + one, record)
    record = tl.where(k == 5, status.to(tl.int64)[:, None] + one, record)
    record = tl.where(k == 6, error.to(tl.int64)[:, None] + one, record)
    record = tl.where(k == 7, one, record)
    record = tl.where(k == 8, emitted[:, None] + one, record)
    record = tl.where(k == 9, tl.where(live_row, offered, 0)[:, None] + one, record)
    record = tl.where(k == 10, emitted_drafts[:, None] + one, record)
    record = tl.where(k == 11, tl.where(live_row, copy_mask, 0)[:, None] + one, record)
    record = tl.where(k == 13, lagged[:, None] + one, record)
    record = tl.where(k == 14, one, record)
    record = tl.where(k == 15, W + one, record)
    for c in tl.static_range(W):
        value = tl.sum(tl.where(cols[None, :] == c, tl.where(take, ids, -1), 0), 1)
        record = tl.where(k == 16 + c, value[:, None] + one, record)

    # Per-row outputs precede request-state mutation.
    out2 = rows[:, None] * W + cols[None, :]
    tl.store(EmittedIds + out2, tl.where(selected[:, None] & take, ids, -1), mask=in_width[None, :])
    tl.store(CommittedInputs + out2, tl.where(selected[:, None] & take, inputs, -1), mask=in_width[None, :])
    tl.store(EmittedCount + rows, tl.where(selected, emitted, 0).to(tl.int32))
    tl.store(AcceptedCount + rows, tl.where(selected, emitted, 0).to(tl.int32))
    tl.store(AcceptedDrafts + rows, tl.where(selected, emitted_drafts, 0).to(tl.int32))
    tl.store(CommitColumn + rows, tl.where(selected, emitted - 1, -1).to(tl.int32))

    # Slot defaults: unowned slots keep their state and receive the idle record.
    slot_ids = tl.arange(0, C)
    fields = tl.arange(0, 32)
    in_record = fields < FIELDS
    tl.store(Owners + slot_ids, tl.full((C,), -1, tl.int64))
    tl.store(Owners + C, -1)
    tl.store(ConsumedAfter + slot_ids, tl.load(Consumed + slot_ids.to(tl.int64) * consumed_stride))
    idle = tl.load(Idle + fields, mask=in_record, other=-1)
    tl.store(Staged + slot_ids[:, None] * FIELDS + fields[None, :], idle[None, :] + tl.zeros((C, 32), tl.int64),
             mask=in_record[None, :])
    tl.debug_barrier()

    # Each slot's leader writes that slot.
    tl.store(Owners + safe, rows.to(tl.int64), mask=leader)
    tl.store(ConsumedAfter + safe, consumed, mask=leader)
    tl.store(Staged + safe[:, None] * FIELDS + k, record, mask=leader[:, None] & (k < FIELDS))
    tl.atomic_add(Histogram + safe[:, None] * V + tl.minimum(tl.maximum(ids, 0), V - 1),
                  tl.full((B, 8), 1, tl.int32), mask=take & leader[:, None], sem="relaxed")
    tl.store(Generated + safe, generated, mask=leader)
    tl.store(Pending + safe, tl.where(emitted > 0, pending, pending_before), mask=leader)
    tl.store(Counter + safe, counters, mask=leader)
    tl.store(Status + safe, status, mask=leader)
    tl.store(Error + safe, error, mask=leader)
    if HAS_RNG:
        tl.store(TargetOffset + safe, rng_offset, mask=leader)
        tl.store(RngGeneration + safe, rng_generation, mask=leader)
    tl.store(Consumed + safe * consumed_stride, consumed, mask=leader)
    tl.debug_barrier()

    frame = tl.load(Frame).to(tl.int64)
    staged = tl.load(Staged + slot_ids[:, None] * FIELDS + fields[None, :], mask=in_record[None, :], other=0)
    tl.store(Records + frame * C * FIELDS + slot_ids[:, None] * FIELDS + fields[None, :], staged,
             mask=in_record[None, :])


def fusable(commit) -> bool:
    """Whether this commit's operands fit the one-program commit."""
    state = commit.state
    return (commit.history is None and state.generation.is_cuda and state.capacity <= 64
            and commit.inputs.shape[0] <= 64 and state.eos_token_ids.shape[1] == EOS_CAPACITY
            and state.rng_counter.dim() == 1 and commit.records.is_contiguous())


def prepare_fused_commit(commit):
    """Bind the one-program commit to a prepared ``PreparedSpeculativeCommit``."""
    state, acceptance = commit.state, commit.acceptance
    batch, width = commit.inputs.shape
    rng, limits = commit.rng, commit.token_limits
    lag = commit.draft_context_lag
    args = (state.generation, state.consumed_tokens, state.consumed_tokens.stride(0), state.generated_tokens,
            state.pending_token, state.status, state.error_code, state.rng_counter.view(torch.int64),
            state.histogram, state.max_new_tokens, state.eos_token_ids, state.eos_count,
            acceptance.token_ids, acceptance.proposed_count, acceptance.accepted_drafts, acceptance.offered_drafts,
            acceptance.valid, acceptance.rng_increment,
            commit.inputs, commit.slots, commit.generations, commit.live,
            commit.slots if limits is None else limits, commit.copy_offered_mask,
            state.generation if rng is None else rng.target_offset,
            state.generation if rng is None else rng.generation,
            commit.sequence, commit.frame, commit.records, commit.idle_record,
            commit.emitted_ids, commit.committed_input_ids, commit.emitted_count, commit.accepted_count,
            commit.accepted_drafts, commit.commit_column, commit.consumed_after, commit.owners, commit._records,
            batch, width, state.capacity, state.vocab_size, EOS_CAPACITY,
            0 if lag is None else lag, lag is not None, limits is not None, rng is not None,
            RUNNING, FINISHED, ERROR, DUPLICATE_SLOT, INVALID_DISTRIBUTION, INVALID_SPECULATION, RECORD_FIELDS)
    return prepare_triton(_commit, args, (1, 1, 1), num_warps=4)
