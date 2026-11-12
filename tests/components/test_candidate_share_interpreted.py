# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Score exchange against pair exchange for a distributed top-k, in Triton's CPU interpreter.

Ported from TP9 8b1be5b (tests/components/test_candidate_share_interpreted.py), adapted to
this tree's programs: the pair pack is ``distributed_topk._pack_pairs`` and the pair merge runs
``_threshold`` + ``_counts`` where the source line folded them into ``_threshold_counts``.

The Triton programs of both forms run on CPU tensors; the b12x row top-k, a CuTe kernel, is
replaced by torch.topk, which returns the same K greatest values. For every row, every rank's
share must be the pair form's selection restricted to that rank. The cases cover seven kinds of
candidate sets (random, ties, fewer than K, exactly K, one owner, all equal, -inf scores) at two
seeds. Three mutations of the share rule (own ties counted as earlier, exactly-K rows through
the cut-off path, -inf kept) must each fail the comparison on their discriminating kinds: they
prove the cases discriminate. Two explicit tests pin the tie rule (earlier gathered positions
win: earlier ranks, then earlier slots) and the one recorded divergence (-inf, and NaN outside
both contracts). The interpreter is chosen when a Triton program is declared, so the cases run
in a child process that sets TRITON_INTERPRET before importing them (tests/triton_interpreter.py).
"""
import pytest
import torch
import triton
import triton.language as tl

from tests.triton_interpreter import interpreted, run_interpreted

WORLD, TOPK, ROWS = 9, 512, 2
KINDS = ("random", "ties", "short", "exact-k", "one-owner", "all-equal", "minus-inf")
MUTATIONS = ("own-ties", "exact-k-cutoff", "keep-minus-inf")


def test_every_rank_share_equals_the_pair_selection_restricted_to_the_rank():
    run_interpreted(__file__, 20, "test_case", "test_mutation", "test_tie_rule",
                    "test_minus_inf", "test_nan")


def _pair_form(indices, scores):
    from ignition.components import candidate_merge as merge, distributed_topk

    width = WORLD * TOPK
    packed = torch.zeros((ROWS, width, 2), dtype=torch.float32)
    for rank in range(WORLD):
        local = torch.zeros((ROWS, TOPK, 2), dtype=torch.float32)
        distributed_topk._pack_pairs[(ROWS,)](
            indices[rank], scores[rank], torch.tensor([rank], dtype=torch.int64), local,
            local.view(torch.int32), TOPK, WORLD, triton.next_power_of_2(TOPK), True, 0, num_warps=8)
        packed[:, rank * TOPK:(rank + 1) * TOPK] = local
    tiles = triton.cdiv(width, 512)
    values, ids = torch.zeros((ROWS, width)), torch.full((ROWS, width), -7, dtype=torch.int32)
    lengths, counts = torch.zeros(ROWS, dtype=torch.int32), torch.zeros((ROWS, tiles), dtype=torch.int32)
    top, cutoffs = torch.zeros((ROWS, TOPK)), torch.zeros(ROWS)
    ties, output = torch.zeros((ROWS, tiles), dtype=torch.int32), torch.full((ROWS, TOPK), -777, dtype=torch.int32)
    grid = (ROWS, tiles, 1)
    merge._count[grid](packed, counts, width, tiles, 512)
    merge._compact[grid](packed, counts, values, ids, lengths, width, tiles, 512, triton.next_power_of_2(tiles))
    for row in range(ROWS):
        length = int(lengths[row])
        if length > TOPK:
            top[row] = torch.topk(values[row, :length], TOPK).values
        else:
            # With at most K candidates the row top-k returns every candidate and -1 padding.
            output[row] = -1
            output[row, :length] = ids[row, :length]
    merge._threshold[(ROWS, 1, 1)](top, lengths, cutoffs, output, TOPK, triton.next_power_of_2(TOPK))
    merge._counts[grid](values, lengths, cutoffs, counts, ties, width, TOPK, tiles, 512)
    merge._select[grid](values, ids, lengths, cutoffs, counts, ties, output, width, TOPK, tiles, 512,
                        triton.next_power_of_2(tiles))
    return output


def _score_form(indices, scores, program=None):
    from ignition.components import candidate_share as share, distributed_topk

    gathered = torch.zeros((ROWS, WORLD * TOPK), dtype=torch.float32)
    for rank in range(WORLD):
        keys = torch.full((ROWS, TOPK), 3.0)
        distributed_topk._pack_keys[(ROWS, triton.cdiv(TOPK, 1024))](
            indices[rank], scores[rank], keys, TOPK, TOPK=TOPK, BLOCK=1024)
        reference = torch.full((ROWS, TOPK), 3.0)
        distributed_topk.pack_candidate_keys(reference, indices[rank], scores[rank])
        # The custom op's CPU body is the kernel's arithmetic.
        assert torch.equal(keys, reference), rank
        gathered[:, rank * TOPK:(rank + 1) * TOPK] = keys
    assert not torch.isnan(gathered).any()
    top = torch.topk(gathered, TOPK, dim=1).values.clone()
    for row in range(ROWS):
        # A row with at most K candidates must not depend on what the row top-k returns for it.
        if int((gathered[row] > -torch.inf).sum()) <= TOPK:
            top[row] = torch.nan
    shares = torch.full((WORLD, ROWS, TOPK), -777, dtype=torch.int32)
    select = share._select_share if program is None else program
    for rank in range(WORLD):
        select[(ROWS, 1, 1)](top, gathered, indices[rank], scores[rank], torch.tensor([rank]),
                             shares[rank], TOPK, TOPK, WORLD, triton.next_power_of_2(TOPK))
    return shares


def _candidates(kind, seed, generator):
    indices = torch.full((WORLD, ROWS, TOPK), -1, dtype=torch.int32)
    scores = torch.zeros((WORLD, ROWS, TOPK))
    for rank in range(WORLD):
        for row in range(ROWS):
            if kind == "short":
                count = int(torch.randint(0, TOPK // WORLD, (1,), generator=generator))
            elif kind == "exact-k":
                count = TOPK // WORLD + (TOPK % WORLD if rank == 0 else 0)
            elif kind == "one-owner":
                count = TOPK if rank == 4 else int(torch.randint(0, TOPK // 4, (1,), generator=generator))
            else:
                count = int(torch.randint(TOPK // 2, TOPK + 1, (1,), generator=generator))
            slots = torch.randperm(TOPK, generator=generator)[:count]
            indices[rank, row, slots] = torch.randperm(4 * TOPK, generator=generator)[:count].int()
            if kind == "ties":
                values = torch.randint(-2, 3, (count,), generator=generator).float()
            elif kind == "all-equal":
                values = torch.full((count,), 1.5)
            else:
                values = torch.randn(count, generator=generator) + (100.0 if kind == "one-owner" and rank == 4 else 0.0)
            if kind == "minus-inf" and count:
                # Half of the candidates, or all but a few, carry a score of -inf.
                lost = count // 2 if seed == 0 else count - count // 20
                values[torch.randperm(count, generator=generator)[:lost]] = -torch.inf
            scores[rank, row, slots] = values
            # A slot without a candidate carries an arbitrary score: its index alone marks it.
            empty = indices[rank, row] < 0
            scores[rank, row, empty] = torch.randn(int(empty.sum()), generator=generator) * 50
    return indices, scores


def _finite_ids(indices, scores, row):
    return {int(indices[rank, row, slot]) * WORLD + rank
            for rank in range(WORLD) for slot in range(TOPK)
            if indices[rank, row, slot] >= 0 and scores[rank, row, slot] > -torch.inf}


def _expected_share(reference, finite, rank):
    return [value for value in reference if value % WORLD == rank and value in finite]


@interpreted
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("seed", (0, 1))
def test_case(kind, seed):
    generator = torch.Generator().manual_seed(1000 + 17 * seed + KINDS.index(kind))
    indices, scores = _candidates(kind, seed, generator)
    selected, shares = _pair_form(indices, scores), _score_form(indices, scores)
    for row in range(ROWS):
        reference = sorted(int(value) for value in selected[row] if value >= 0)
        total = int((indices[:, row] >= 0).sum())
        assert len(reference) == min(total, TOPK) and len(set(reference)) == len(reference)
        if kind == "exact-k":
            assert total == TOPK
        # A candidate with a score of -inf is no candidate of the score exchange; the pair form
        # keeps such candidates only while fewer than K finite ones exist (the recorded
        # divergence: test_minus_inf_candidates_diverge_as_documented).
        finite = _finite_ids(indices, scores, row)
        assert kind == "minus-inf" or len(finite) == total
        for rank in range(WORLD):
            own = sorted(int(value) for value in shares[rank, row] if value >= 0)
            assert own == sorted(_expected_share(reference, finite, rank)), (row, rank)
            assert int((shares[rank, row] < -1).sum()) == 0, (row, rank)


# -- mutations of the share rule ---------------------------------------------------------------
#
# Each mutation changes one rule of ``_select_share`` (the program under
# ignition/components/candidate_share.py). A mutation must fail the comparison of
# test_case on its discriminating kinds — if one did not, the equivalence cases
# would not discriminate that rule and the port would be unproven there.


@triton.jit
def _select_share_mutation_own_ties(TopValues, Keys, Indices, Scores, Rank, Output, row_stride,
                                    K: tl.constexpr, WORLD: tl.constexpr, BLOCK: tl.constexpr):
    # "own ties counted as earlier": `earlier` counts the equal scores of earlier ranks AND
    # this rank's own (rank >= peer), so this rank's ties use its places twice.
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    mask = col < K
    rank = tl.load(Rank).to(tl.int32)
    top = tl.load(TopValues + row*K + col, mask=mask, other=float('inf'))
    cutoff = tl.min(top, 0)
    total = 0
    greater = 0
    earlier = 0
    for peer in tl.static_range(WORLD):
        value = tl.load(Keys + (row*WORLD + peer)*K + col, mask=mask, other=-float('inf'))
        total += tl.sum((value > -float('inf')).to(tl.int32), 0)
        greater += tl.sum((value > cutoff).to(tl.int32), 0)
        earlier += tl.sum((value == cutoff).to(tl.int32), 0) * (rank >= peer).to(tl.int32)
    index = tl.load(Indices + row*row_stride + col, mask=mask, other=-1)
    score = tl.load(Scores + row*row_stride + col, mask=mask, other=-float('inf'))
    valid = mask & (index >= 0) & (score > -float('inf'))
    equal = valid & (score == cutoff)
    places = K - greater - earlier
    chosen = valid & ((total <= K) | (score > cutoff) | (equal & (tl.cumsum(equal.to(tl.int32), 0) <= places)))
    tl.store(Output + row*K + col, tl.where(chosen, index*WORLD + rank, -1), mask=mask)


@triton.jit
def _select_share_mutation_cutoff_path(TopValues, Keys, Indices, Scores, Rank, Output, row_stride,
                                       K: tl.constexpr, WORLD: tl.constexpr, BLOCK: tl.constexpr):
    # "exactly K candidates through the cut-off path": the select-all shortcut is dropped, so
    # rows with at most K candidates are cut against whatever the row top-k returned for them.
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    mask = col < K
    rank = tl.load(Rank).to(tl.int32)
    top = tl.load(TopValues + row*K + col, mask=mask, other=float('inf'))
    cutoff = tl.min(top, 0)
    greater = 0
    earlier = 0
    for peer in tl.static_range(WORLD):
        value = tl.load(Keys + (row*WORLD + peer)*K + col, mask=mask, other=-float('inf'))
        greater += tl.sum((value > cutoff).to(tl.int32), 0)
        earlier += tl.sum((value == cutoff).to(tl.int32), 0) * (rank > peer).to(tl.int32)
    index = tl.load(Indices + row*row_stride + col, mask=mask, other=-1)
    score = tl.load(Scores + row*row_stride + col, mask=mask, other=-float('inf'))
    valid = mask & (index >= 0) & (score > -float('inf'))
    equal = valid & (score == cutoff)
    places = K - greater - earlier
    chosen = valid & ((score > cutoff) | (equal & (tl.cumsum(equal.to(tl.int32), 0) <= places)))
    tl.store(Output + row*K + col, tl.where(chosen, index*WORLD + rank, -1), mask=mask)


@triton.jit
def _select_share_mutation_keep_minus_inf(TopValues, Keys, Indices, Scores, Rank, Output, row_stride,
                                          K: tl.constexpr, WORLD: tl.constexpr, BLOCK: tl.constexpr):
    # "-inf kept": a slot with a candidate is valid even at a score of -inf, so the exchange
    # keeps -inf candidates where the pair form's finite-candidate reference does not.
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    mask = col < K
    rank = tl.load(Rank).to(tl.int32)
    top = tl.load(TopValues + row*K + col, mask=mask, other=float('inf'))
    cutoff = tl.min(top, 0)
    total = 0
    greater = 0
    earlier = 0
    for peer in tl.static_range(WORLD):
        value = tl.load(Keys + (row*WORLD + peer)*K + col, mask=mask, other=-float('inf'))
        total += tl.sum((value > -float('inf')).to(tl.int32), 0)
        greater += tl.sum((value > cutoff).to(tl.int32), 0)
        earlier += tl.sum((value == cutoff).to(tl.int32), 0) * (rank > peer).to(tl.int32)
    index = tl.load(Indices + row*row_stride + col, mask=mask, other=-1)
    valid = mask & (index >= 0)
    equal = valid & (index >= 0)
    places = K - greater - earlier
    chosen = valid & ((total <= K) | (equal & (tl.cumsum(equal.to(tl.int32), 0) <= places)))
    tl.store(Output + row*K + col, tl.where(chosen, index*WORLD + rank, -1), mask=mask)


# Kinds that expose each mutation: own ties must be taken (ties / all-equal), a row with at
# most K candidates must take the select-all path (short / exact-k), and -inf candidates must
# be dropped (minus-inf).
MUTATION_KINDS = {"own-ties": ("ties", "all-equal"), "exact-k-cutoff": ("short", "exact-k"),
                  "keep-minus-inf": ("minus-inf",)}


@interpreted
@pytest.mark.parametrize("name", MUTATIONS)
def test_mutation(name):
    program = {"own-ties": _select_share_mutation_own_ties,
               "exact-k-cutoff": _select_share_mutation_cutoff_path,
               "keep-minus-inf": _select_share_mutation_keep_minus_inf}[name]
    diverged = False
    for kind in MUTATION_KINDS[name]:
        for seed in (0, 1):
            generator = torch.Generator().manual_seed(1000 + 17 * seed + KINDS.index(kind))
            indices, scores = _candidates(kind, seed, generator)
            selected, shares = _pair_form(indices, scores), _score_form(indices, scores, program=program)
            for row in range(ROWS):
                reference = sorted(int(value) for value in selected[row] if value >= 0)
                finite = _finite_ids(indices, scores, row)
                for rank in range(WORLD):
                    own = sorted(int(value) for value in shares[rank, row] if value >= 0)
                    if own != sorted(_expected_share(reference, finite, rank)):
                        diverged = True
    assert diverged, f"the equivalence cases do not discriminate the {name} mutation"


# -- the tie rule and the recorded divergence, pinned explicitly --------------------------------


def _fixture(indices_values, scores_values):
    indices = torch.full((WORLD, ROWS, TOPK), -1, dtype=torch.int32)
    scores = torch.zeros((WORLD, ROWS, TOPK))
    for (rank, row, slot), value in indices_values.items():
        indices[rank, row, slot] = value
    for (rank, row, slot), value in scores_values.items():
        scores[rank, row, slot] = value
    return indices, scores


def _id(index, rank):
    """The global position a rank's candidate carries in both forms' outputs (index * world + rank)."""
    return index * WORLD + rank


@interpreted
def test_tie_rule_prefers_earlier_ranks_then_earlier_slots():
    # The pair merge takes equal cutoff scores in gathered order (rank-major, slot order
    # inside a rank); the share rule must give each rank exactly its candidates among them.
    # Row 0: 600 equal-score candidates (300 per rank on ranks 0 and 1). Row 1: 69 greater
    # scores (ranks 0-2), then a 500-candidate tie block at the cut-off on ranks 4-8.
    index_values, score_values = {}, {}
    for slot in range(300):
        for rank in (0, 1):
            index_values[rank, 0, slot] = rank * 1000 + slot
            score_values[rank, 0, slot] = 2.0
    for slot in range(100):
        for rank in (4, 5, 6, 7, 8):
            index_values[rank, 1, slot] = rank * 1000 + slot
            score_values[rank, 1, slot] = 5.0
    for slot in range(23):
        for rank in (0, 1, 2):
            index_values[rank, 1, slot + 100] = rank * 1000 + 500 + slot
            score_values[rank, 1, slot + 100] = 9.0
    indices, scores = _fixture(index_values, score_values)
    selected, shares = _pair_form(indices, scores), _score_form(indices, scores)
    for row in range(ROWS):
        reference = sorted(int(value) for value in selected[row] if value >= 0)
        assert len(reference) == TOPK
        for rank in range(WORLD):
            own = sorted(int(value) for value in shares[rank, row] if value >= 0)
            assert own == [value for value in reference if value % WORLD == rank], (row, rank)
    # Row 0 places fill in gathered order: rank 0's 300, then rank 1's first 212 slots.
    assert sorted(int(v) for v in shares[0, 0] if v >= 0) == sorted(_id(0 * 1000 + s, 0) for s in range(300))
    assert sorted(int(v) for v in shares[1, 0] if v >= 0) == sorted(_id(1 * 1000 + s, 1) for s in range(212))
    # Row 1: 69 greater scores leave 443 places to the tie block, filled in rank order with
    # earlier slots first inside each rank: ranks 4-7 take all 100, rank 8 its first 43.
    for rank in (4, 5, 6, 7):
        assert sorted(int(v) for v in shares[rank, 1] if v >= 0) == sorted(_id(rank * 1000 + s, rank)
                                                                         for s in range(100))
    assert sorted(int(v) for v in shares[8, 1] if v >= 0) == sorted(_id(8 * 1000 + s, 8) for s in range(43))


@interpreted
def test_minus_inf_candidates_diverge_as_documented():
    # RECORDED DIVERGENCE (not papered over): a candidate whose score is -inf is no candidate
    # of the score exchange, where the pair merge kept -inf candidates while fewer than K
    # finite ones existed (8b1be5b's commit message). Row 0 holds 400 finite candidates and
    # 212 live slots scored -inf (612 candidates, more than K): the pair selection fills its
    # remaining 112 places with -inf candidates in gathered order; each rank's share holds
    # only its finite candidates.
    index_values, score_values = {}, {}
    for slot in range(300):
        index_values[0, 0, slot] = slot
        score_values[0, 0, slot] = 3.0
    for slot in range(100):
        index_values[1, 0, slot] = 1000 + slot
        score_values[1, 0, slot] = 1.0
    for slot in range(212):
        index_values[0, 0, 300 + slot] = 2000 + slot
        score_values[0, 0, 300 + slot] = -torch.inf
    indices, scores = _fixture(index_values, score_values)
    selected, shares = _pair_form(indices, scores), _score_form(indices, scores)
    reference = sorted(int(value) for value in selected[0] if value >= 0)
    finite = _finite_ids(indices, scores, 0)
    assert len(reference) == TOPK and len(finite) == 400
    # The pair form keeps 112 -inf candidates: this is the divergence.
    assert len([value for value in reference if value not in finite]) == 112
    for rank in range(WORLD):
        own = sorted(int(value) for value in shares[rank, 0] if value >= 0)
        assert own == sorted(_expected_share(reference, finite, rank)), rank
        assert all(value in finite for value in own), rank


@interpreted
def test_nan_scores_are_no_candidates_of_the_exchange():
    # RECORDED CORNER (outside both contracts: PreparedCandidateShare documents "Scores must
    # not be NaN" and PreparedCandidateMerge "Scores for valid IDs must be finite"): the
    # exchange deterministically treats a NaN score as no candidate — _pack_keys writes -inf
    # for it and _select_share never selects it — even in a row with fewer than K finite
    # candidates where every finite candidate is selected. The pair merge's behavior on NaN
    # inputs is not compared here (undefined by its contract; the GPU wave's bit-exact arm
    # comparison is the gate for what the indexer can return).
    index_values, score_values = {}, {}
    for slot in range(100):
        index_values[0, 0, slot] = slot
        score_values[0, 0, slot] = 3.0
    for slot in range(50):
        index_values[1, 0, slot] = 1000 + slot
        score_values[1, 0, slot] = torch.nan
    indices, scores = _fixture(index_values, score_values)
    shares = _score_form(indices, scores)
    selected = sorted(int(value) for value in shares[0, 0].tolist() if value >= 0)
    assert selected == sorted(_id(slot, 0) for slot in range(100))
    assert not any(value >= 0 for value in shares[1, 0].tolist())
