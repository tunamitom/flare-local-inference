# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0
"""Injected cross-rank collectives for the Hy4 decoder body (W6 §4).

Every collective the body needs goes through this protocol, so one body serves
the eager world-1 path, the W6 CPU simulation and the W7 GPU path:

- `all_gather_heads` — the head all-gather of projected queries
  (`pcie_dcp_a2a.all_gather_heads`, rank-major head blocks: this rank's
  `heads // world` heads out, all `heads` in), so every attention call runs
  ALL heads against this rank's own record shard (amendment 1);
- `lse_reduce` — the base-2 LSE reduce-scatter of the ranks' partials over
  ALL heads (`pcie_dcp_a2a.lse_reduce_scatter`), returning this rank's
  contiguous head shard (`lse_reduce_scatter_reference` is the exact math);
- `all_reduce` — the TP sum after `o_proj` and after `layer.mlp.forward`
  (the layer-0 dense FFN included) — D4;
- `embed_lookup` / `gather_logits` — the vocabulary-sharded embedding
  (masked lookup + all-reduce) and FP32 head (all-gather to the sampler) — D5;
- `merge_topk` — the selection exchange: the W ranks' local top-k candidates
  into ONE global top-k per query row
  (`dsa_indexer.tiled_topk.merge_tiled_topk_candidates` over the
  `pcie_dcp_topk` planes), consumed by `SelectionCell.handback` — D3.

The CPU simulation (`SimulatedGroup`) runs W ranks of the same body in
lockstep — one thread per rank, barrier-synchronized exchanges combined over
the simulated ranks in rank order — which is the W6 gates' instrument. The
GPU implementation (`PcieDcpCollectives` over the real `PCIeDCPA2A` /
`PCIeDCPTopKOwnerExchange` b12x transport, prepared plans only) is W7's; the
body is identical either way. `SimulatedDcpTransport` is the CPU twin of the
transport's call surfaces, so the wired `PcieDcpCollectives` path is what the
2-rank CPU gates run.

LSE convention: Hy4 partials are base 2 (`sparse_mla` `lse_scale="base2"`,
the sink folded exactly once after the merge), so `lse_reduce` always states
`lse_base="base2"` — the natural-log hard-code this replaced re-weighted
every rank (W6 gate c's wrong-LSE-base mutant).
"""

import threading

import torch

__all__ = [
    "Collectives",
    "SoloCollectives",
    "SimulatedGroup",
    "SimulatedCollectives",
    "merge_candidate_topk",
    "scores_only_threshold",
    "scores_only_offer_saturated",
    "scores_only_merge",
    "parse_all_reduce_arms",
    "all_reduce_arm_for",
    "audit_scores_only_selection",
    "check_scores_only_exactness",
    "select_own_candidates",
    "SimulatedDcpTransport",
    "PcieDcpTransport",
    "PcieDcpCollectives",
]

LSE_BASE = "base2"


def _trace(message):
    """HY4_GATE_TRACE=1: one flushed marker per transport collective (hang
    diagnosis at the call sites; permanent code stays silent)."""
    import os
    import time

    if os.environ.get("HY4_GATE_TRACE"):
        print(f"[transport-trace r{os.environ.get('RANK', 0)} t+{time.monotonic():.1f}s] {message}",
              flush=True)


class Collectives:
    """The injected interface (structural — any object with these members).

    `world_size`/`rank` are this rank's CP identity (the ownership rule is
    `p % world` / `p // world`, `decoder.Hy4Cache.owner`/`.local`). All
    methods are called by every rank of the group, once per body step each.
    """

    world_size: int
    rank: int

    def all_gather_heads(self, q_local):
        """`[rows, heads // world, width]` -> `[rows, heads, width]` (rank-major)."""
        raise NotImplementedError

    def lse_reduce(self, out_partial, lse_partial, *, partials_sink_folded):
        """`[rows, heads, dim]`/`[rows, heads]` partials -> this rank's `[rows, heads // world, dim]` shard.

        `partials_sink_folded` is the producing attention call's provenance
        (`SparseMLAProvenance.sink_folded`) — declared into the merge so the
        double-sink-fold guard can reject a per-rank fold loudly (D6).
        """
        raise NotImplementedError

    def all_reduce(self, value):
        """TP sum over the group (in place semantics; may return a new tensor)."""
        raise NotImplementedError

    def embed_lookup(self, token_ids, embed, *, vocabulary_range):
        """Global token ids against this rank's vocabulary shard, summed over the group."""
        raise NotImplementedError

    def gather_logits(self, logits, *, vocabulary_range):
        """Per-shard `[rows, tokens, vocab // world]` logits -> the sampler's full-vocab logits."""
        raise NotImplementedError

    def merge_topk(self, candidate_scores, candidate_positions, *, topk):
        """Every rank's `[rows, topk]` local candidates -> one global `(values, positions)` top-k per row."""
        raise NotImplementedError


class SoloCollectives:
    """World 1: every collective is identity — the eager path, unchanged.

    Only correct at `world_size == 1` (asserted): at world > 1 the body's CP
    branch needs a real transport (`SimulatedGroup` for the W6 gates,
    `PCIeDCPA2A` wiring for W7) and construction refuses to guess.
    """

    def __init__(self, *, world_size=1, rank=0):
        if world_size != 1 or rank != 0:
            raise ValueError(
                "SoloCollectives is the world-1 identity; inject a Collectives "
                "for a context-parallel rank (W6 SimulatedGroup / W7 PCIeDCPA2A)"
            )
        self.world_size = 1
        self.rank = 0

    def all_gather_heads(self, q_local):
        return q_local  # one rank owns every head

    def lse_reduce(self, out_partial, lse_partial, *, partials_sink_folded):
        return out_partial, lse_partial

    def all_reduce(self, value):
        return value

    def embed_lookup(self, token_ids, embed, *, vocabulary_range):
        return embed[token_ids]

    def gather_logits(self, logits, *, vocabulary_range):
        return logits

    def merge_topk(self, candidate_scores, candidate_positions, *, topk):
        return merge_candidate_topk(((candidate_scores, candidate_positions),), topk)


def merge_candidate_topk(candidates, topk):
    """One global top-k per row from every rank's local candidates.

    `candidates` is the per-rank `(scores [rows, topk], positions [rows,
    topk])` in rank order — the `pcie_dcp_topk` planes — merged through
    `dsa_indexer.tiled_topk.merge_tiled_topk_candidates` (its `chunks` axis is
    the ranks). Returns `(values, positions)` `[rows, topk]` in GLOBAL
    positions (`-inf` / `-1` padding survives the merge and drops in the
    hand-back). `torch.topk`'s tie order is unspecified and layout-dependent,
    so it does NOT reproduce the single rank's `select` pick when the top-k cut
    lands inside a score plateau (the tiny indexer's relu makes exact 0.0
    plateaus): the W6 gate-b tie cases are constructed so the selected set is
    tie-order independent there, and content gates verify any plateau swap as
    an exact score tie (`test_hy4_w7_wiring`'s tie-aware content gate).
    """
    from b12x.attention.dsa_indexer.tiled_topk import merge_tiled_topk_candidates

    scores = torch.stack([score for score, _ in candidates], dim=0).float()
    positions = torch.stack([pos for _, pos in candidates], dim=0)
    return merge_tiled_topk_candidates(
        candidate_values=scores, candidate_indices=positions, topk=topk
    )


class SimulatedGroup:
    """A CPU world of `world_size` ranks running the same body in lockstep.

    Each rank runs in its own thread; the collectives rendezvous at barriers
    and combine over the simulated ranks in rank order (deterministic). The
    body is shape-deterministic per rank — every rank calls each collective
    the same number of times — so lockstep is total. A rank that raises aborts
    the barrier so peers fail fast instead of deadlocking, and the lowest-rank
    error is re-raised.
    """

    def __init__(self, world_size, *, timeout=300.0):
        if type(world_size) is not int or world_size < 1:
            raise ValueError(f"simulated world size must be a positive int, got {world_size!r}")
        self.world_size = world_size
        self.timeout = timeout
        self._barrier = threading.Barrier(world_size)
        self._slots = [None] * world_size

    def collectives(self, rank):
        return SimulatedCollectives(self, rank)

    def _exchange(self, rank, value):
        self._slots[rank] = value
        self._barrier.wait(self.timeout)
        taken = tuple(self._slots)
        self._barrier.wait(self.timeout)  # every rank read before reuse
        return taken

    def run(self, bodies):
        """`bodies[rank](collectives)` in one thread per rank; results in rank order."""
        if len(bodies) != self.world_size:
            raise ValueError(f"expected {self.world_size} rank bodies, got {len(bodies)}")
        results = [None] * self.world_size
        errors = []

        def entry(rank):
            try:
                results[rank] = bodies[rank](self.collectives(rank))
            except BaseException as exc:  # noqa: BLE001 - reported below in rank order
                errors.append((rank, exc))
                self._barrier.abort()

        threads = [threading.Thread(target=entry, args=(rank,), daemon=True)
                   for rank in range(self.world_size)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if errors:
            errors.sort(key=lambda item: item[0])
            raise errors[0][1]
        return results


class SimulatedCollectives:
    """One simulated rank's collectives: exchange with the peers, combine in rank order.

    The combination is the shipped contract per collective: `all_gather_heads`
    concatenates rank-major head blocks, `lse_reduce` runs
    `lse_reduce_scatter_reference` (the `pcie_dcp_a2a` merge math, base 2,
    `partials_sink_folded=False` — the sink folds once after the merge),
    `merge_topk` runs `merge_tiled_topk_candidates`, `all_reduce` sums and the
    vocabulary ops shard `sharding.vocabulary_range`.
    """

    def __init__(self, group, rank):
        if not 0 <= rank < group.world_size:
            raise ValueError(f"rank must be in [0, {group.world_size}), got {rank!r}")
        self.group = group
        self.world_size = group.world_size
        self.rank = rank
        self.lse_base = LSE_BASE
        # Selection-exchange policy (the W7 bake-ins): `"candidates"` is the
        # exact full-candidate merge (the W6 gate instrument); `"scores-only"`
        # is the baked wire contract — only score words cross and each rank
        # selects its own keys (see `scores_only_threshold`). The W6 gate-b
        # exactness test overrides `local_offer` to the full selection width.
        self.exchange_mode = "candidates"
        self.local_offer = None

    def all_gather_heads(self, q_local):
        # GPU: `PcieDcpCollectives.all_gather_heads` -> `PCIeDCPA2A.all_gather_heads`
        # (pcie_dcp_a2a:1331-1340) with its prepared plan.
        taken = self.group._exchange(self.rank, q_local.contiguous())
        return torch.cat(taken, dim=1)

    def lse_reduce(self, out_partial, lse_partial, *, partials_sink_folded):
        # GPU: `PcieDcpCollectives.lse_reduce` -> `PCIeDCPA2A.lse_reduce_scatter`
        # (pcie_dcp_a2a:308-324) — partials over ALL heads in, this rank's
        # contiguous head shard out. This is the exact math either way.
        from b12x.comm.pcie import lse_reduce_scatter_reference

        taken = self.group._exchange(
            self.rank, (out_partial.contiguous(), lse_partial.contiguous())
        )
        stacked_out = torch.stack([out for out, _ in taken], dim=0)
        stacked_lse = torch.stack([lse for _, lse in taken], dim=0)
        merged, merged_lse = lse_reduce_scatter_reference(
            stacked_out,
            stacked_lse,
            self.rank,
            lse_base=self.lse_base,
            return_lse=True,
            partials_sink_folded=partials_sink_folded,
        )
        return merged, merged_lse

    def all_reduce(self, value):
        # GPU: `PcieDcpCollectives.all_reduce` (the TP all-reduce — NCCL or the
        # one-shot PCIe reduce over the TP group).
        taken = self.group._exchange(self.rank, value.contiguous())
        return torch.stack(taken, dim=0).sum(dim=0)

    def embed_lookup(self, token_ids, embed, *, vocabulary_range):
        # Masked lookup over this rank's vocabulary shard + all-reduce (D5).
        first, last = vocabulary_range
        inside = (token_ids >= first) & (token_ids < last)
        out = embed.new_zeros((int(token_ids.numel()), int(embed.shape[1])))
        if bool(inside.any()):
            out[inside] = embed[token_ids[inside] - first]
        return self.all_reduce(out)

    def gather_logits(self, logits, *, vocabulary_range):
        # All-gather the vocabulary shards to the sampler (D5). The shards are
        # contiguous rank-major partitions, so rank order is vocab order.
        first, last = vocabulary_range
        if int(logits.shape[-1]) != last - first:
            raise ValueError(
                f"logits width {int(logits.shape[-1])} != the shard's "
                f"vocabulary range {vocabulary_range}"
            )
        taken = self.group._exchange(self.rank, logits.contiguous())
        return torch.cat(taken, dim=-1)

    def merge_topk(self, candidate_scores, candidate_positions, *, topk):
        """`"candidates"`: the exact full-candidate merge. `"scores-only"`: the
        baked wire contract — only score words cross (`exchange_scores`) and
        each rank selects its own keys above the global k-th score.
        """
        if self.exchange_mode == "scores-only":
            return scores_only_merge(
                lambda scores: self.group._exchange(self.rank, scores),
                candidate_scores, candidate_positions,
                topk=topk, world=self.world_size, rank=self.rank, local_offer=self.local_offer)
        if self.exchange_mode != "candidates":
            raise ValueError(f"unknown selection exchange mode {self.exchange_mode!r}")
        taken = self.group._exchange(
            self.rank, (candidate_scores.contiguous(), candidate_positions.contiguous())
        )
        return merge_candidate_topk(taken, topk)


# --------------------------------------------------------------------------
# the scores-only selection exchange (consult7 §5, consult8 §5)
# --------------------------------------------------------------------------
def exchange_scores(group, rank, candidate_scores, local_offer):
    """The wire: each rank's top-`local_offer` SCORES per row, nothing else.

    "Scores-only selection exchange (each rank needs only the global k-th
    score and its own keys)" — key material and foreign positions never move;
    each rank keeps its own candidates (`candidate_scores` /
    `candidate_positions` stay local). `local_offer=None` offers the full
    candidate width (the W6 gate-b override: 2,048-selection equality).
    Returns the per-rank offers in rank order, each `[rows, offer]` sorted
    descending with `-inf` padding.
    """
    rows, width = candidate_scores.shape
    offer = width if local_offer is None else min(int(local_offer), width)
    return group._exchange(rank, candidate_scores[:, :offer].contiguous())


def scores_only_threshold(offers, topk):
    """Per row: `(threshold, above)` — the global k-th score and the count above it.

    `offers` is the per-rank `[rows, offer]` score offers in rank order. The
    threshold is the topk-th largest offered score; `above` is how many
    offered scores are strictly greater (so `topk - above` slots remain for
    the threshold plateau). Padding `-inf` never counts. Returns two `[rows]`
    tensors (FP32, int64) — one batched top-k, no per-row host reads.
    """
    stacked = torch.cat([offer.float() for offer in offers], dim=1)   # [rows, world * offer]
    total = torch.isfinite(stacked).sum(dim=1)
    k = min(int(topk), int(stacked.shape[1]))
    if k == 0:
        thresholds = stacked.new_full((stacked.shape[0],), float("-inf"))
    else:
        kth = stacked.topk(k, dim=1).values[:, k - 1]
        # At most `topk` finite offers: every offered key is in (threshold -inf).
        thresholds = torch.where(total > topk, kth, torch.full_like(kth, float("-inf")))
    aboves = (stacked > thresholds.unsqueeze(1)).sum(dim=1)
    return thresholds, aboves


def _threshold_tensor(thresholds, like):
    return torch.as_tensor(thresholds, dtype=torch.float32, device=like.device).reshape(-1)


def scores_only_offer_saturated(offers, thresholds):
    """True when some rank's offer may hide keys at or above the cut.

    A rank's offer is sorted descending, so its LAST offered score bounds every
    score it did not offer. If that last score is finite and reaches the row's
    threshold, un-offered keys could clear the cut. The test reads the shared
    offers only, so every rank computes the same answer and widens together
    (one host read per exchange).
    """
    limit = _threshold_tensor(thresholds, offers[0])
    for offer in offers:
        if not offer.shape[1]:
            continue
        last = offer[:, -1].float()
        if bool((torch.isfinite(last) & (last >= limit)).any()):
            return True
    return False


def check_scores_only_exactness(thresholds, local_offer, candidate_scores):
    """The exactness guard: the offer must not truncate at the cut.

    Each rank guards its OWN truncation from its own keys (the wire carries
    only scores): if its first UN-offered candidate reaches that row's global
    k-th score, un-offered keys might clear the threshold and the selection
    would not equal the global top-k (the W6 gate-b 2,048-selection equality).
    Raise instead of returning an inexact selection — the fix is a wider offer
    (`local_offer` up to the full width). `-inf` padding past a rank's live
    keys is not a candidate and never trips the guard.
    """
    if local_offer is None:
        return  # full-width offer: nothing is ever truncated
    rows, width = candidate_scores.shape
    if width <= int(local_offer):
        return  # everything this rank has was offered
    first_unoffered = candidate_scores[:, int(local_offer)].float()
    limit = _threshold_tensor(thresholds, candidate_scores)
    truncated = torch.isfinite(first_unoffered) & (first_unoffered >= limit)
    if bool(truncated.any()):
        row = int(truncated.nonzero()[0])
        raise ValueError(
            f"scores-only offer truncated the selection at row {row}: the first "
            f"un-offered score {float(first_unoffered[row])} reaches the global k-th score "
            f"{float(limit[row])}; widen local_offer (up to the full width) — "
            "the selection must equal the global top-k exactly (W6 gate b)")


def select_own_candidates(offers, candidate_scores, candidate_positions, *, topk, world, rank,
                          local_offer, thresholds=None, aboves=None):
    """This rank's own keys in the global top-k, from the k-th score alone.

    Keys with score strictly above the threshold are all in; the threshold
    plateau fills in deterministic (rank, position) order — rank-major, each
    rank's tied keys in ascending global position — so every rank derives the
    same cut from the exchanged scores and its own keys. Returns
    `(values, positions)` `[rows, topk]` with `-inf` / `-1` padding.

    The plateau order is world-DEPENDENT by construction: the scores-only wire
    carries no positions (consult8 §5's baked exchange), so global-position
    order is not derivable across ranks. When the cut lands inside a plateau
    the W-rank set may therefore differ from the world-1 `torch.topk` pick —
    a tie-order artifact of an exactly tied cut, not a selection error; the
    content gates verify such swaps as exact score ties.

    Batched over the rows (consult16 lever 2): the per-row Python loop cost a
    dozen host reads per query row per full layer.
    """
    if thresholds is None:
        thresholds, aboves = scores_only_threshold(offers, topk)
    rows, width = candidate_scores.shape
    device = candidate_scores.device
    limit = _threshold_tensor(thresholds, candidate_scores).unsqueeze(1)
    values = candidate_scores.float()
    positions = candidate_positions.long()
    finite = torch.isfinite(values)
    selected = finite & (values > limit)
    plateau = finite & (values == limit)
    if bool(plateau.any()):
        above = (torch.zeros(rows, dtype=torch.int64, device=device) if aboves is None
                 else torch.as_tensor(aboves, dtype=torch.int64, device=device).reshape(-1))
        # Earlier ranks' plateau keys fill first (a `-inf` threshold has no plateau).
        earlier = torch.zeros(rows, dtype=torch.int64, device=device)
        live = torch.isfinite(limit)
        for other in range(rank):
            earlier += ((offers[other].float() == limit) & live).sum(dim=1)
        mine = (topk - above - earlier).clamp_min(0)
        # This rank's plateau keys in ascending global position: keep the first `mine`.
        key = torch.where(plateau, positions, torch.full_like(positions, torch.iinfo(torch.int64).max))
        order = key.argsort(dim=1, stable=True).argsort(dim=1, stable=True)
        selected = selected | (plateau & (order < mine.unsqueeze(1)))
    # Selected keys first, by score descending (stable: ties keep candidate order).
    masked = torch.where(selected, values, torch.full_like(values, float("-inf")))
    ranked, index = masked.sort(dim=1, descending=True, stable=True)
    picked = torch.where(torch.isfinite(ranked), positions.gather(1, index),
                         torch.full_like(positions, -1))
    if width < topk:
        ranked = torch.cat((ranked, ranked.new_full((rows, topk - width), float("-inf"))), dim=1)
        picked = torch.cat((picked, picked.new_full((rows, topk - width), -1)), dim=1)
    return ranked[:, :topk].to(torch.float32), picked[:, :topk].to(torch.int32)


def audit_scores_only_selection(own_values, own_positions, exact_values, exact_positions, *,
                                world, rank):
    """The tie-aware exactness gate, in the run's own scores (consult16 lever 2).

    `own_*` is this rank's scores-only result, `exact_*` the full-candidate
    global merge of the same call. This rank's keys must be exactly the exact
    merge's keys it owns (`position % world == rank`), except for keys whose
    score EQUALS the row's k-th score: a tied cut may keep either key. Any
    other difference raises. Returns the number of rows with such a tie swap.
    """
    big = torch.iinfo(torch.int64).max
    own = own_positions.long()
    exact = exact_positions.long()
    exact_mine = torch.where((exact >= 0) & (exact % world == rank), exact, torch.full_like(exact, -1))
    left = torch.where(own >= 0, own, torch.full_like(own, big)).sort(dim=1).values
    right = torch.where(exact_mine >= 0, exact_mine, torch.full_like(exact_mine, big)).sort(dim=1).values
    differing = (left != right).any(dim=1).nonzero(as_tuple=True)[0].tolist()
    swaps = 0
    for row in differing:
        finite = torch.isfinite(exact_values[row])
        full = int(finite.sum()) == int(exact_values.shape[1])
        cut = float(exact_values[row][finite].min()) if full else None
        mine = {int(p): float(v) for p, v in zip(own[row].tolist(), own_values[row].tolist()) if p >= 0}
        theirs = {int(p): float(v) for p, v in zip(exact[row].tolist(), exact_values[row].tolist())
                  if p >= 0 and p % world == rank}
        for position in set(mine) ^ set(theirs):
            score = mine.get(position, theirs.get(position))
            if cut is None or score != cut:
                raise ValueError(
                    f"scores-only selection differs from the exact merge at row {row}, position "
                    f"{position} (score {score}, k-th score {cut}): not a tie at the cut")
        swaps += 1
    return swaps


def scores_only_selection(offers, candidate_scores, candidate_positions, *, topk, world, rank,
                          local_offer):
    """The full scores-only merge for one rank: threshold, guard, own keys."""
    thresholds, aboves = scores_only_threshold(offers, topk)
    check_scores_only_exactness(thresholds, local_offer, candidate_scores)
    return select_own_candidates(
        offers, candidate_scores, candidate_positions, topk=topk, world=world, rank=rank,
        local_offer=local_offer, thresholds=thresholds, aboves=aboves)


def scores_only_merge(exchange, candidate_scores, candidate_positions, *, topk, world, rank,
                      local_offer):
    """The scores-only exchange with a narrow first offer and an exact fallback.

    `exchange(scores)` gathers every rank's `[rows, offer]` slice in rank
    order. Each rank first offers its top `local_offer` scores; when any
    rank's offer is saturated at the cut (`scores_only_offer_saturated`) every
    rank re-offers its FULL candidate width — the decision reads only the
    shared offers, so the ranks stay in lockstep — and the selection is the
    exact global top-k either way. `local_offer=None` offers the full width
    at once (the W6 gate-b form).
    """
    width = int(candidate_scores.shape[1])
    offer = width if local_offer is None else min(int(local_offer), width)
    offers = exchange(candidate_scores[:, :offer].contiguous())
    thresholds, aboves = scores_only_threshold(offers, topk)
    if offer < width and scores_only_offer_saturated(offers, thresholds):
        offers = exchange(candidate_scores.contiguous())
        thresholds, aboves = scores_only_threshold(offers, topk)
        offer = width
    check_scores_only_exactness(thresholds, None if offer == width else offer, candidate_scores)
    return select_own_candidates(
        offers, candidate_scores, candidate_positions, topk=topk, world=world, rank=rank,
        local_offer=local_offer, thresholds=thresholds, aboves=aboves)


# --------------------------------------------------------------------------
# the W7 transport wiring: the real b12x DCP primitives
# --------------------------------------------------------------------------
class SimulatedDcpTransport:
    """CPU twin of the `PCIeDCPA2A` / `PCIeDCPTopKOwnerExchange` call surfaces.

    Implements exactly the surfaces `PcieDcpCollectives` consumes —
    `all_gather_heads(local_input, out, *, plan)`,
    `lse_reduce_scatter(partial_output, partial_lse, out, *, plan, lse_base,
    partials_sink_folded, return_lse)` and `exchange_scores(offers, plan)` —
    over a `SimulatedGroup`, with the shipped math
    (`lse_reduce_scatter_reference`, rank-major head concat). This is what
    makes the wired `PcieDcpCollectives` path runnable in the 2-rank CPU
    gates against world 1.
    """

    def __init__(self, group):
        self.group = group
        self.rank = None  # bound per SimulatedCollectives-style rank handle

    def bind(self, rank):
        bound = SimulatedDcpTransport(self.group)
        bound.rank = rank
        return bound

    def all_gather_heads(self, local_input, out=None, *, plan):
        taken = self.group._exchange(self.rank, local_input.contiguous())
        gathered = torch.cat(taken, dim=1)
        return gathered if out is None else out.copy_(gathered)

    def lse_reduce_scatter(self, partial_output, partial_lse, out=None, *, plan, lse_base,
                           partials_sink_folded=None, return_lse=False):
        from b12x.comm.pcie import lse_reduce_scatter_reference

        taken = self.group._exchange(
            self.rank, (partial_output.contiguous(), partial_lse.contiguous()))
        stacked_out = torch.stack([value for value, _ in taken], dim=0)
        stacked_lse = torch.stack([value for _, value in taken], dim=0)
        merged, merged_lse = lse_reduce_scatter_reference(
            stacked_out, stacked_lse, self.rank, lse_base=lse_base, return_lse=True,
            partials_sink_folded=partials_sink_folded)
        if out is not None:
            out.copy_(merged)
            merged = out
        return (merged, merged_lse) if return_lse else merged

    def exchange_scores(self, offers, *, plan):
        return self.group._exchange(self.rank, offers.contiguous())

    def stage_candidates(self, local_indices, local_scores, local_counts=None, *, plan):
        """The `pcie_dcp_topk` owner-plane surface: candidate packs in rank order."""
        taken = self.group._exchange(
            self.rank, (local_scores.contiguous(), local_indices.contiguous()))
        return taken

    def all_reduce(self, value):
        taken = self.group._exchange(self.rank, value.contiguous())
        return torch.stack(taken, dim=0).sum(dim=0)

    def all_gather(self, value):
        taken = self.group._exchange(self.rank, value.contiguous())
        return torch.cat(taken, dim=-1)


# ------------------------------------------------------------------------
# the W7 GPU transport: the real b12x PCIe DCP surfaces
# ------------------------------------------------------------------------
# The b12x PCIe all-reduce arms (the c1 cells' names): the one-shot pool's
# transports, and the BF16 two-shot runtime the manager builds beside it.
ALL_REDUCE_ARMS = {
    "oneshot": "remote_push_full",
    "rsag": "remote_push_rsag",
    "pull": "pull",
    "twoshot": None,
}
DEFAULT_ALL_REDUCE_ARMS = "1:oneshot,2-:rsag"


def parse_all_reduce_arms(spec):
    """`"1:oneshot,2-1024:twoshot,1025-:rsag@256x32"` -> row-range rules.

    Each rule is `ROWS:ARM` or `ROWS:ARM@THREADSxBLOCKS`; `ROWS` is `N`,
    `LO-HI` or `LO-` (open). Rules are tried in order; every row count the
    serve prepares must match one. Returns `((lo, hi, arm, threads, blocks), ...)`
    with `hi=None` open and `threads=blocks=None` meaning the arm's default
    launch (256 x 16 for the one-shot arms — the c1 cells' — and the prepared
    default for two-shot and pull).
    """
    rules = []
    for item in str(spec).split(","):
        item = item.strip()
        rows, sep, arm = item.partition(":")
        if not sep:
            raise ValueError(f"all-reduce arm rule {item!r} is not ROWS:ARM")
        threads = blocks = None
        arm, at, launch = arm.partition("@")
        if at:
            left, x, right = launch.partition("x")
            if not x or not left.isdigit() or not right.isdigit():
                raise ValueError(f"all-reduce launch {launch!r} is not THREADSxBLOCKS")
            threads, blocks = int(left), int(right)
            if threads < 32 or threads % 32 or blocks < 1:
                raise ValueError(f"all-reduce launch {launch!r}: threads in 32s, blocks >= 1")
        if arm not in ALL_REDUCE_ARMS:
            raise ValueError(f"unknown all-reduce arm {arm!r}; one of {sorted(ALL_REDUCE_ARMS)}")
        lo, dash, hi = rows.partition("-")
        if not lo.isdigit() or (dash and hi and not hi.isdigit()):
            raise ValueError(f"all-reduce row range {rows!r} is not N, LO-HI or LO-")
        lo = int(lo)
        hi = lo if not dash else (int(hi) if hi else None)
        if lo < 1 or (hi is not None and hi < lo):
            raise ValueError(f"all-reduce row range {rows!r} is empty")
        rules.append((lo, hi, arm, threads, blocks))
    if not rules:
        raise ValueError("the all-reduce arm spec is empty")
    return tuple(rules)


def all_reduce_arm_for(rows, rules):
    """The first rule covering `rows`: `(arm, threads, blocks)`."""
    for lo, hi, arm, threads, blocks in rules:
        if rows >= lo and (hi is None or rows <= hi):
            return arm, threads, blocks
    raise ValueError(f"no all-reduce arm rule covers rows={rows}")


class PcieDcpTransport:
    """GPU transport over the real b12x PCIe DCP kernels (W7 gate c / P5).

    `PCIeDCPA2APool` carries the two CP-attention exchanges — the head
    all-gather of projected queries and the base-2 fused `lse_reduce_scatter`
    — through PREPARED PLANS ONLY (`_dcp_preparation` queries and plans; no
    JIT at run time).  The sink never enters the merge (`partials_sink_folded`
    is declared into the run; the one fold is `PreparedSinkFold` after it).
    The call surfaces are exactly `SimulatedDcpTransport`'s, so
    `PcieDcpCollectives` rides either transport and the body is unchanged.

    The TP reductions and vocabulary gathers (`all_reduce`, `all_gather`) and
    the selection wires (`exchange_scores`, `stage_candidates`) ride the
    caller's torch.distributed control group here: the `pcie_dcp_topk` owner
    planes partition ROWS across owners while the Hy4 body needs every row's
    merged top-k on every rank (`SelectionCell.handback` runs per rank), and
    the scores-only wire's real-kernel unit cost is measured separately
    (c1-unit-costs.md §5: 7.88 us at world 8).  What rides which path is
    recorded in `surfaces()`.
    """

    def __init__(self, *, group, device, pools, rank, world_size, total_heads,
                 all_reduce=None, reduce_width=None, all_reduce_arms=None, score_offer=None):
        self.group = group
        self.device = torch.device(device)
        self.rank = rank
        self.world_size = world_size
        self.total_heads = int(total_heads)
        self._pools = pools            # {(width, kind): pool}
        self._plans = {}               # (surface, rows, width, lse_base) -> plan
        self._prepared = 0
        self._ar = all_reduce          # consult15 §2 lever 1: b12x PCIe all-reduce
        self.reduce_width = None if reduce_width is None else int(reduce_width)
        self._ar_plans = {}            # rows -> (plan, runtime, arm)
        # consult17: the scores-only offer width carried on the DCP all-gather
        # (`None` = the offers ride the control group, as before).
        self.score_offer = None if score_offer is None else int(score_offer)
        # HY4_SCORE_WIRE_AUDIT=1 (a gate boot): every GPU offer gather is also
        # run over the control group and the two must agree bit for bit.
        import os
        self.score_wire_audit = bool(os.environ.get("HY4_SCORE_WIRE_AUDIT"))
        self.score_wire_checked = 0
        # consult17: which arm each row count runs (the A/B flag's rules).
        self.all_reduce_arms = parse_all_reduce_arms(all_reduce_arms or DEFAULT_ALL_REDUCE_ARMS)

    # -- construction ------------------------------------------------------

    @classmethod
    def from_process_group(cls, process_group, device, *, total_heads, query_widths,
                           value_width, max_rows, reduce_width=None, all_reduce_arms=None,
                           score_offer=None, extra_value_widths=(), value_max_rows=None):
        """One channel pool per wire geometry (the query gather's absorbed and
        unabsorbed widths, and the 512-wide latent partials).

        `reduce_width` (consult15 §2 lever 1) additionally builds the b12x
        PCIe all-reduce pool over the biggest `[rows, reduce_width]` BF16
        reduction the body runs (`max_rows` rows) — the c1 cells' runtime and
        arms (c1-unit-costs.md §1: oneshot at 1 row, rsag above).
        """
        from b12x.comm.pcie.pcie_dcp_a2a import PCIeDCPA2APool

        import torch.distributed as dist

        device = torch.device(device)
        rank = dist.get_rank(group=process_group)
        world_size = dist.get_world_size(group=process_group)
        pools = {}
        for width in sorted(set(query_widths)):
            _trace(f"PCIeDCPA2APool gather width={width}")
            pools[(width, "gather")] = PCIeDCPA2APool.from_process_group(
                process_group=process_group, device=device,
                max_batch_size=max_rows, total_heads=total_heads,
                head_dim=width, query_head_dim=width, single_channel=True)
            _trace(f"PCIeDCPA2APool gather width={width} done")
        # consult17 (lever 9): `extra_value_widths` adds partial pools at other
        # widths (the 256-wide expanded prefill partials); `value_max_rows`
        # caps the rows of the `value_width` pool when only decode still uses
        # it (pools are sized by their row capacity).
        for width, capacity in ((value_width, value_max_rows or max_rows),
                                *((extra, max_rows) for extra in extra_value_widths)):
            _trace(f"PCIeDCPA2APool partials width={width} rows={capacity}")
            pools[(width, "partials")] = PCIeDCPA2APool.from_process_group(
                process_group=process_group, device=device,
                max_batch_size=capacity, total_heads=total_heads,
                head_dim=width, query_head_dim=width, single_channel=True)
            _trace(f"PCIeDCPA2APool partials width={width} done")
        if score_offer is not None:
            # consult17 (lever 2 step 2): one more gather channel for the
            # scores-only offer. The gather moves 16-bit words, so each rank's
            # `[rows, offer]` FP32 scores travel as `[rows, 1, 2 * offer]`
            # BF16 words — one "head" per rank — and are viewed back unchanged.
            words = 2 * int(score_offer)
            if words % 8:
                raise ValueError("the score offer must be a multiple of 4 scores")
            _trace(f"PCIeDCPA2APool scores words={words}")
            pools[(words, "scores")] = PCIeDCPA2APool.from_process_group(
                process_group=process_group, device=device,
                max_batch_size=max_rows, total_heads=world_size,
                head_dim=words, query_head_dim=words, single_channel=True)
            _trace(f"PCIeDCPA2APool scores words={words} done")
        all_reduce = None
        if reduce_width is not None:
            from ignition.preparation.collectives import create_all_reduce_runtime

            _trace(f"PCIeAllReduce width={reduce_width} rows={max_rows}")
            all_reduce = create_all_reduce_runtime(
                process_group, device, max_tokens=max_rows, hidden_size=int(reduce_width))
            all_reduce.prepare_channels(("h4:allreduce",))
            _trace(f"PCIeAllReduce width={reduce_width} rows={max_rows} done")
        return cls(group=process_group, device=device, pools=pools,
                   rank=rank, world_size=world_size, total_heads=total_heads,
                   all_reduce=all_reduce, reduce_width=reduce_width,
                   all_reduce_arms=all_reduce_arms, score_offer=score_offer)

    def bind(self, rank):
        if rank != self.rank:
            raise ValueError(f"transport is rank {self.rank}, cannot bind rank {rank}")
        return self

    def surfaces(self):
        """What rides the real DCP kernels versus the control group."""
        return {
            "all_gather_heads": "PCIeDCPA2APool.all_gather_heads (prepared plans)",
            "lse_reduce_scatter": "PCIeDCPA2APool.lse_reduce_scatter (prepared plans, base-2)",
            "all_reduce": ("b12x PCIeAllReduce (oneshot at 1 row, rsag above; "
                           "prepared plans per row shape)" if self._ar is not None
                           else "torch.distributed control group (unit costs: c1-unit-costs.md §5)"),
            "all_gather": "torch.distributed control group",
            "exchange_scores": "torch.distributed control group",
            "stage_candidates": "torch.distributed control group",
        }

    def close(self):
        for pool in self._pools.values():
            pool.close()
        self._pools = {}
        if self._ar is not None:
            self._ar.close()
            self._ar = None
        self._ar_plans = {}

    # -- prepared plans (no JIT at run time) -------------------------------

    def prepare_gather(self, session, rows, width, *, name=None):
        """Prepare the head all-gather plan for `rows` at query width `width`."""
        from b12x.comm.pcie import _dcp_preparation as preparation
        from b12x.preparation import CollectiveRequirement

        key = ("all_gather_heads", int(rows), int(width), "")
        if key in self._plans:
            return self._plans[key]
        pool = self._pools[(int(width), "gather")]
        runtime = pool.for_stream()
        inp = torch.zeros((rows, self._total_heads_local(), width),
                          dtype=torch.bfloat16, device=self.device)
        out = torch.empty((rows, self._total_heads(), width),
                          dtype=torch.bfloat16, device=self.device)
        call = dict(local_input=inp, out=out)
        query = preparation.query_from_runtime(
            runtime, surface="DcpAllToAll.all_gather_heads", call=call)
        plan = preparation.plan(query, runtime=runtime)
        label = name or f"gather-{rows}-{width}"
        session.prepare(
            (plan.request(
                name=label,
                collective=CollectiveRequirement(key=label, ranks=tuple(range(self.world_size))),
                prepare_call=lambda state: preparation.prepare_call(state, **call),
            ),),
            coordinator=lambda progress: progress.ready_collectives[0].key
            if progress.ready_collectives else None,
        )
        self._plans[key] = plan
        self._prepared += 1
        return plan

    def prepare_score_gather(self, session, rows):
        """Prepare the offer gather for `rows` (consult17: scores off the host)."""
        from b12x.comm.pcie import _dcp_preparation as preparation
        from b12x.preparation import CollectiveRequirement

        if self.score_offer is None:
            raise RuntimeError("no score-gather pool (construct with score_offer)")
        words = 2 * self.score_offer
        key = ("score_gather", int(rows), words, "")
        if key in self._plans:
            return self._plans[key]
        runtime = self._pools[(words, "scores")].for_stream()
        inp = torch.zeros((rows, 1, words), dtype=torch.bfloat16, device=self.device)
        out = torch.empty((rows, self.world_size, words), dtype=torch.bfloat16,
                          device=self.device)
        call = dict(local_input=inp, out=out)
        query = preparation.query_from_runtime(
            runtime, surface="DcpAllToAll.all_gather_heads", call=call)
        plan = preparation.plan(query, runtime=runtime)
        label = f"scores-{rows}-{words}"
        session.prepare(
            (plan.request(
                name=label,
                collective=CollectiveRequirement(key=label, ranks=tuple(range(self.world_size))),
                prepare_call=lambda state: preparation.prepare_call(state, **call),
            ),),
            coordinator=lambda progress: progress.ready_collectives[0].key
            if progress.ready_collectives else None,
        )
        self._plans[key] = plan
        self._prepared += 1
        return plan

    def prepare_lse(self, session, rows, width, *, lse_base=LSE_BASE):
        """Prepare the fused LSE reduce-scatter plan (`width`-wide latent values)."""
        from b12x.comm.pcie import _dcp_preparation as preparation
        from b12x.preparation import CollectiveRequirement

        key = ("lse_reduce_scatter", int(rows), int(width), lse_base)
        if key in self._plans:
            return self._plans[key]
        pool = self._pools[(int(width), "partials")]
        runtime = pool.for_stream()
        partial = torch.zeros((rows, self._total_heads(), width),
                              dtype=torch.bfloat16, device=self.device)
        lse = torch.zeros((rows, self._total_heads()), device=self.device)
        out = torch.empty((rows, self._total_heads() // self.world_size, width),
                          dtype=torch.bfloat16, device=self.device)
        call = dict(partial_output=partial, partial_lse=lse, out=out,
                    lse_base=lse_base, partials_sink_folded=False)
        query = preparation.query_from_runtime(
            runtime, surface="DcpAllToAll.lse_reduce_scatter", call=call)
        plan = preparation.plan(query, runtime=runtime)
        label = f"lse-{rows}-{width}-{lse_base}"
        session.prepare(
            (plan.request(
                name=label,
                collective=CollectiveRequirement(key=label, ranks=tuple(range(self.world_size))),
                prepare_call=lambda state: preparation.prepare_call(state, **call),
            ),),
            coordinator=lambda progress: progress.ready_collectives[0].key
            if progress.ready_collectives else None,
        )
        self._plans[key] = plan
        self._prepared += 1
        return plan

    def _total_heads(self):
        return self.total_heads

    def prepare_all_reduce(self, session, rows, *, name=None):
        """Prepare the b12x PCIe all-reduce plan for `rows` at the reduce width.

        The arm comes from `self.all_reduce_arms` (consult17; the default is
        consult15's lever 1: `oneshot` at 1 row, `rsag` above, 256 x 16).
        `oneshot` / `rsag` / `pull` are transports of the one-shot pool;
        `twoshot` is the BF16 two-shot runtime the manager builds beside it.
        Each is prepared exactly as the c1 cells' bench prepares it
        (`benchmark_pcie_allreduce_transports.prepare`, minus graph replay).
        """
        from dataclasses import replace

        from b12x.preparation import CollectiveRequirement, FrozenMapping

        if self._ar is None:
            raise RuntimeError("no all-reduce runtime (construct with reduce_width)")
        rows = int(rows)
        if rows in self._ar_plans:
            return self._ar_plans[rows][0]
        arm, threads, blocks = all_reduce_arm_for(rows, self.all_reduce_arms)
        inp = torch.zeros((rows, self.reduce_width), dtype=torch.bfloat16, device=self.device)
        out = torch.empty_like(inp)
        if arm == "twoshot":
            from b12x.comm.pcie import _twoshot_preparation as twoshot

            target = getattr(self._ar, "_bf16_twoshot", None)
            if target is None or not target.accepts(inp):
                raise RuntimeError(
                    f"the two-shot all-reduce is unavailable for rows={rows} "
                    "(no BF16 two-shot runtime, or the shape is not whole 16-byte packs)")
            call = {"inp": inp, "out": out}
            if threads is not None:
                call.update(threads=threads, block_limit=blocks)
            query = twoshot.query_from_runtime(
                target, surface="PCIeTwoShotBF16.all_reduce", call=call)
            plan = twoshot.plan(query, runtime=target)

            def prime(state):
                return twoshot.prepared_call(state, payload=inp, out=out)
        else:
            from b12x.comm.pcie import _oneshot_preparation as oneshot

            target = self._ar._prepared_channel_for_stream(None, None)
            query = oneshot.query_from_runtime(
                target, surface="OneshotAllReduce.all_reduce", call={"inp": inp, "out": out})
            if arm != "pull" and not query.call["payload_ready_region_packs"]:
                raise RuntimeError(
                    "the b12x all-reduce pool has no payload-ready region for "
                    f"rows={rows} (remote_push arms unavailable at this setup)")
            call = {**query.call, "transport": ALL_REDUCE_ARMS[arm]}
            if threads is not None:
                call.update(threads=threads, blocks=blocks)
            elif arm == "pull":
                call.update(threads=query.call["eager_threads"],
                            blocks=query.call["eager_blocks"])
            else:
                call.update(threads=256, blocks=16)
            query = replace(query, call=FrozenMapping(call))
            plan = oneshot.plan(query, runtime=target)

            def prime(state):
                return oneshot._prepare_plain_call(state, inp=inp, out=out)

        label = name or f"ar-{arm}-{rows}-{self.reduce_width}"
        session.prepare(
            (plan.request(
                name=label,
                collective=CollectiveRequirement(key=label, ranks=tuple(range(self.world_size))),
                prepare_call=prime,
            ),),
            coordinator=lambda progress: progress.ready_collectives[0].key
            if progress.ready_collectives else None,
        )
        self._ar_plans[rows] = (plan, target, arm)
        self._prepared += 1
        return plan

    def _total_heads_local(self):
        return self.total_heads // self.world_size

    # -- the SimulatedDcpTransport call surfaces ---------------------------

    @staticmethod
    def _wire(tensor):
        """The DCP kernels' wire dtype (BF16 partials / FP16; the FP32 gate body
        stages through the production wire and is restored on return)."""
        return tensor.to(torch.bfloat16) if tensor.dtype == torch.float32 else tensor

    def all_gather_heads(self, local_input, out=None, *, plan=None):
        rows, _, width = (int(v) for v in local_input.shape)
        plan = self._lookup("all_gather_heads", rows, width, "")
        pool = self._pools[(width, "gather")]
        _trace(f"all_gather_heads rows={rows} width={width}")
        result = pool.for_stream().all_gather_heads(
            self._wire(local_input.contiguous()), None if out is None else self._wire(out), plan=plan)
        _trace(f"all_gather_heads rows={rows} width={width} done")
        return result.to(local_input.dtype) if out is None else out

    def lse_reduce_scatter(self, partial_output, partial_lse, out=None, *, plan=None,
                           lse_base=LSE_BASE, partials_sink_folded=None, return_lse=False):
        rows, heads, width = (int(v) for v in partial_output.shape)
        if partials_sink_folded is None:
            raise ValueError(
                "lse_reduce_scatter requires declared partials_sink_folded "
                "(the sink never enters the merge; D6)")
        plan = self._lookup("lse_reduce_scatter", rows, width, lse_base)
        pool = self._pools[(width, "partials")]
        _trace(f"lse_reduce_scatter rows={rows} width={width} base={lse_base}")
        wire_dtype = partial_output.dtype
        result = pool.for_stream().lse_reduce_scatter(
            self._wire(partial_output.contiguous()), partial_lse.float().contiguous(),
            None if out is None else self._wire(out),
            plan=plan, lse_base=lse_base,
            partials_sink_folded=partials_sink_folded, return_lse=return_lse)
        _trace(f"lse_reduce_scatter rows={rows} width={width} base={lse_base} done")
        if return_lse:
            merged, merged_lse = result
            return merged.to(wire_dtype), merged_lse
        return result.to(wire_dtype)

    def exchange_scores(self, offers, *, plan=None):
        """Every rank's `[rows, offer]` FP32 scores, in rank order.

        With a prepared score gather for this exact `(rows, offer)` the offer
        rides the DCP all-gather as raw 16-bit words (no host wire); any other
        shape — the full-width re-offer after a saturated cut, an unprepared
        row count — takes the control group, as before. Every rank holds the
        same plans, so every rank takes the same path.
        """
        rows, width = (int(v) for v in offers.shape)
        key = ("score_gather", rows, 2 * width, "")
        if (self.score_offer != width or offers.dtype != torch.float32
                or key not in self._plans):
            return self._gather_all(offers)
        words = offers.contiguous().view(torch.bfloat16).reshape(rows, 1, 2 * width)
        _trace(f"score_gather rows={rows} offer={width}")
        gathered = self._pools[(2 * width, "scores")].for_stream().all_gather_heads(
            words, None, plan=self._plans[key])
        _trace(f"score_gather rows={rows} offer={width} done")
        scores = gathered.contiguous().view(torch.float32)      # [rows, world, offer]
        result = tuple(scores[:, rank, :] for rank in range(self.world_size))
        if self.score_wire_audit:
            for rank, (got, want) in enumerate(zip(result, self._gather_all(offers))):
                if not torch.equal(got.contiguous().view(torch.int32),
                                   want.contiguous().view(torch.int32)):
                    raise RuntimeError(
                        f"score gather returned different bits for rank {rank}'s offer "
                        f"(rows={rows}, offer={width}): the DCP gather is not a pure copy here")
            self.score_wire_checked += 1
        return result

    def stage_candidates(self, local_indices, local_scores, local_counts=None, *, plan=None):
        scores = self._gather_all(local_scores)
        indices = self._gather_all(local_indices)
        return tuple(zip(scores, indices))

    def all_reduce(self, value):
        from .profile import TIMER

        if self._ar is not None:
            # consult15 §2 lever 1: the b12x PCIe all-reduce over the prepared
            # pool (oneshot at 1 row, rsag above) — GPU to GPU, no host wire.
            if (value.dim() != 2 or value.dtype != torch.bfloat16
                    or int(value.shape[1]) != self.reduce_width):
                raise RuntimeError(
                    f"unprepared all-reduce value {tuple(value.shape)} {value.dtype}; "
                    f"the GPU path reduces [rows, {self.reduce_width}] BF16")
            entry = self._ar_plans.get(int(value.shape[0]))
            if entry is None:
                raise RuntimeError(
                    f"no prepared all-reduce plan for rows={int(value.shape[0])}; "
                    "plans bind before the run — there is no JIT at run time")
            plan, target, arm = entry
            inp = value.contiguous()
            out = torch.empty_like(inp)
            _trace(f"all_reduce(gpu) rows={int(value.shape[0])} arm={arm}")
            if arm == "twoshot":
                if not target.accepts(inp):
                    inp = inp.clone()      # two-shot wants 16-byte-aligned packs
                prepared = plan.prepared.state.query.call
                result = target.all_reduce(inp, plan=plan, out=out,
                                           threads=prepared["threads"],
                                           block_limit=prepared["block_limit"])
            else:
                result = target.all_reduce(inp, plan=plan, out=out)
            _trace(f"all_reduce(gpu) rows={int(value.shape[0])} done")
            return result
        import torch.distributed as dist

        _trace(f"all_reduce shape={tuple(value.shape)}")
        with TIMER.phase("host.d2h"):
            host = value.contiguous().cpu()
        with TIMER.phase("host.gloo"):
            dist.all_reduce(host, op=dist.ReduceOp.SUM, group=self.group)
        TIMER.add_host("all_reduce", host.nbytes)
        _trace(f"all_reduce shape={tuple(value.shape)} done")
        with TIMER.phase("host.h2d"):
            return host.to(value.device)

    def all_gather(self, value):
        import torch.distributed as dist

        from .profile import TIMER

        _trace(f"all_gather shape={tuple(value.shape)}")
        with TIMER.phase("host.d2h"):
            host = value.contiguous().cpu()
        with TIMER.phase("host.gloo"):
            taken = [torch.empty_like(host) for _ in range(self.world_size)]
            dist.all_gather(taken, host, group=self.group)
        TIMER.add_host("all_gather", host.nbytes, copies=self.world_size)
        _trace(f"all_gather shape={tuple(value.shape)} done")
        # Cat on the host and move ONCE: moving each shard first then catting
        # holds 2x the full-vocab result on GPU (world x shard + the cat) —
        # at 1024 prefill rows that is ~1 GiB of eager transient per chunk.
        # Same bytes, one device transfer instead of world+1.
        with TIMER.phase("host.h2d"):
            return torch.cat(taken, dim=-1).to(value.device)

    def _gather_all(self, value):
        """Every rank's copy of `value`, in rank order (the control-group wire)."""
        import torch.distributed as dist

        from .profile import TIMER

        _trace(f"gather_all shape={tuple(value.shape)}")
        with TIMER.phase("host.d2h"):
            host = value.contiguous().cpu()
        with TIMER.phase("host.gloo"):
            taken = [torch.empty_like(host) for _ in range(self.world_size)]
            dist.all_gather(taken, host, group=self.group)
        TIMER.add_host("gather_all", host.nbytes, copies=self.world_size)
        _trace(f"gather_all shape={tuple(value.shape)} done")
        with TIMER.phase("host.h2d"):
            return tuple(t.to(value.device) for t in taken)

    def _lookup(self, surface, rows, width, lse_base):
        key = (surface, int(rows), int(width), lse_base)
        if key not in self._plans:
            raise RuntimeError(
                f"no prepared {surface!r} plan for rows={rows}, width={width}, "
                f"lse_base={lse_base!r}; plans bind before the run — there is "
                "no JIT at run time")
        return self._plans[key]


class PcieDcpCollectives:
    """The W7 collectives: the real b12x transport through the `Collectives` protocol.

    Wires the recipe's TODO hooks to the shipped primitives — `all_gather_heads`
    and `lse_reduce_scatter` (`PCIeDCPA2A`, pcie_dcp_a2a:1331-1340 / 308-324),
    the selection exchange over the `pcie_dcp_topk` owner planes, and the TP
    all-reduce — through PREPARED PLANS ONLY (`KernelPlanBook`; no JIT at run
    time). The LSE contract is base 2 with `partials_sink_folded` declared
    into the merge; the sink never enters the merge (the ONE post-merge fold
    is `PreparedSinkFold` in `attention.fold_sink`) — passing
    `attention_sink` here instead would double-fold and is rejected loudly.

    The selection exchange runs the baked scores-only contract by default
    (each rank needs only the global k-th score and its own keys; the exact
    top-k is guarded by `check_scores_only_exactness`). `exchange_mode=
    "candidates"` runs the exact full-candidate merge instead (the W6 gate-b
    instrument).

    `transport` exposes the call surfaces (`SimulatedDcpTransport` on CPU;
    `PcieDcpTransport` wraps `PCIeDCPA2A` + `PCIeDCPTopKOwnerExchange` on
    GPU), so the same wiring is what the CPU gates and the GPU serve run.
    """

    def __init__(self, *, rank, world_size, transport, plans=None,
                 exchange_mode="scores-only", local_offer=None):
        if not 0 <= rank < world_size:
            raise ValueError(f"rank must be in [0, {world_size}), got {rank!r}")
        self.rank = rank
        self.world_size = world_size
        self.transport = transport.bind(rank) if hasattr(transport, "bind") else transport
        self.plans = plans  # KernelPlanBook (prepared plans per call shape)
        self.exchange_mode = exchange_mode
        self.local_offer = local_offer
        self.lse_base = LSE_BASE
        # HY4_SELECTION_AUDIT=1: also run the exact full-candidate merge and
        # check the scores-only result against it, tie-aware (a gate boot only:
        # it doubles the selection wire). Counters are read by the serve.
        import os
        self.audit = bool(os.environ.get("HY4_SELECTION_AUDIT"))
        self.audit_rows = self.audit_swaps = 0

    def _plan(self, kernel, shape, *tensors):
        """The prepared plan for (kernel, call shape); `shape=None` shares one plan."""
        if self.plans is None:
            return None
        if shape is None:
            return self.plans.shared(kernel)
        return self.plans.plan_for(shape, kernel)

    def all_gather_heads(self, q_local, *, call_shape=None):
        """`[rows, heads // world, width]` -> `[rows, heads, width]` (rank-major)."""
        return self.transport.all_gather_heads(
            q_local.contiguous(), plan=self._plan("all_gather_heads", call_shape))

    def lse_reduce(self, out_partial, lse_partial, *, partials_sink_folded, call_shape=None):
        """Base-2 fused reduce-scatter over ALL heads -> this rank's head shard."""
        merged, merged_lse = self.transport.lse_reduce_scatter(
            out_partial.contiguous(), lse_partial.contiguous(),
            plan=self._plan("lse_reduce_scatter", call_shape),
            lse_base=self.lse_base, partials_sink_folded=partials_sink_folded, return_lse=True)
        return merged, merged_lse

    def all_reduce(self, value):
        return self.transport.all_reduce(value.contiguous())

    def embed_lookup(self, token_ids, embed, *, vocabulary_range):
        first, last = vocabulary_range
        inside = (token_ids >= first) & (token_ids < last)
        out = embed.new_zeros((int(token_ids.numel()), int(embed.shape[1])))
        if bool(inside.any()):
            out[inside] = embed[token_ids[inside] - first]
        return self.all_reduce(out)

    def gather_logits(self, logits, *, vocabulary_range):
        first, last = vocabulary_range
        if int(logits.shape[-1]) != last - first:
            raise ValueError(
                f"logits width {int(logits.shape[-1])} != the shard's "
                f"vocabulary range {vocabulary_range}")
        # Vocabulary shards are contiguous rank-major partitions, so rank
        # order is vocab order (the sampler's all-gather).
        return self.transport.all_gather(logits.contiguous())

    def merge_topk(self, candidate_scores, candidate_positions, *, topk):
        if self.exchange_mode == "scores-only":
            plan = self._plan("selection_exchange", None)
            values, positions = scores_only_merge(
                lambda scores: self.transport.exchange_scores(scores, plan=plan),
                candidate_scores, candidate_positions,
                topk=topk, world=self.world_size, rank=self.rank,
                local_offer=self.local_offer or None)
            if self.audit:
                exact_values, exact_positions = merge_candidate_topk(
                    self.transport.stage_candidates(
                        candidate_positions.contiguous(), candidate_scores.contiguous(), None,
                        plan=plan), topk)
                self.audit_swaps += audit_scores_only_selection(
                    values, positions, exact_values, exact_positions,
                    world=self.world_size, rank=self.rank)
                self.audit_rows += int(values.shape[0])
            return values, positions
        if self.exchange_mode != "candidates":
            raise ValueError(f"unknown selection exchange mode {self.exchange_mode!r}")
        candidates = self.transport.stage_candidates(
            candidate_positions.contiguous(), candidate_scores.contiguous(), None,
            plan=self._plan("selection_exchange", None))
        return merge_candidate_topk(candidates, topk)
