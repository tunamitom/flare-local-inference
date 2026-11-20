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

from contextlib import contextmanager

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
    "AR_AUDIT_REL_L2",
    "CE_AUDIT_REL_L2",
    "CeExchange",
    "merge_lse_partials",
    "relay_partner",
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
                          local_offer, thresholds=None, aboves=None, sync_free=False):
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
    # `sync_free` (the overlapped loop): run the plateau fill unconditionally
    # instead of reading the device first — with no plateau the mask is all
    # False and the OR below adds nothing, so the selection is identical.
    if sync_free or bool(plateau.any()):
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


def scores_only_flags(offers, thresholds, local_offer, candidate_scores):
    """Device-side twin of the widen + exactness host reads: one bool tensor,
    True when some rank's offer is saturated at the cut or this rank's first
    un-offered key reaches it (the narrow selection might then be inexact).
    No host read — the overlapped loop checks it after the loop."""
    limit = _threshold_tensor(thresholds, offers[0])
    flag = torch.zeros((), dtype=torch.bool, device=limit.device)
    for offer in offers:
        if offer.shape[1]:
            last = offer[:, -1].float()
            flag = flag | (torch.isfinite(last) & (last >= limit)).any()
    if local_offer is not None and candidate_scores.shape[1] > int(local_offer):
        first = candidate_scores[:, int(local_offer)].float()
        flag = flag | (torch.isfinite(first) & (first >= limit)).any()
    return flag


def scores_only_merge(exchange, candidate_scores, candidate_positions, *, topk, world, rank,
                      local_offer, deferred=None):
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
    if deferred is not None:
        from .runtime import SWITCHES

        if candidate_scores.is_cuda and SWITCHES.get("sel_fused") == "on":
            # consult18 P2: threshold + flags + own selection in one launch
            # per row (fused_ops.scores_only_select; bit-identical for the
            # sorted candidate rows local_candidates produces, and its flag
            # also fires on any row outside that contract -> exact rerun).
            from .fused_ops import scores_only_select

            values, positions, flag = scores_only_select(
                offers, candidate_scores, candidate_positions, topk=topk, world=world, rank=rank)
            deferred.append(flag)
            return values, positions
        thresholds, aboves = scores_only_threshold(offers, topk)
        # consult18 D: inside the overlapped loop no host read — the widen and
        # exactness tests become a device flag the caller checks after the loop
        # (a set flag reruns the pair on the single-lane path).
        if offer < width:
            deferred.append(scores_only_flags(offers, thresholds, offer, candidate_scores))
        return select_own_candidates(
            offers, candidate_scores, candidate_positions, topk=topk, world=world, rank=rank,
            local_offer=local_offer, thresholds=thresholds, aboves=aboves, sync_free=True)
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

    def gather_head_major(self, rows, width):
        """`PcieDcpCollectives.gather_head_major`'s transport surface (Option 1,
        PERF-PLAN.md Step 3). The CPU twin's gather is the BF16 wire's row-major
        `[rows, heads, width]` (see `all_gather_heads` above), so it never
        reports the INT8 wire's head-major layout and the absorb path stays on
        the row-major contract every consumer assumes."""
        return False

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
    # consult18 stage A1: b12x's copy-engine ring (`DmaAllReduce`,
    # comm/pcie/pcie_dma.py) — reduce-scatter + all-gather ring whose data
    # plane is cudaMemcpyAsync peer copies; SMs only flag and add.
    "dma": None,
}
# Relative L2 error allowed between the ring and the incumbent kernel arm on
# the same input (HY4_AR_AUDIT=1). The ring rounds to BF16 at each of its 7
# reduce hops; the kernel arms round once. Expected ~2-5e-3; refuse above.
AR_AUDIT_REL_L2 = 1.5e-2
# consult18 B1: the CE merge against the kernel reduce-scatter (HY4_CE_AUDIT=1).
CE_AUDIT_REL_L2 = 1.5e-2
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
        if arm == "dma" and at:
            raise ValueError("the dma ring has no thread/block launch to pin")
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



# --------------------------------------------------------------------------
# consult18 stage B1: the prefill DCP exchanges on copy-engine peer copies
# --------------------------------------------------------------------------
def _merge_head_major(partials, lses):
    """FP32 base-2 LSE merge over the source axis: `(out [heads, rows, width] FP32,
    lse [heads, rows])`; rows no source saw get zero / `-inf`."""
    if partials.is_cuda and int(partials.shape[0]) <= 4:
        from .runtime import SWITCHES

        if SWITCHES.get("fused_merge") == "on":
            # consult18 P1: one kernel, bit-identical to the chain below for up to
            # four sources (the relay's switch-local and pair merges).
            from .fused_ops import merge_head_major

            return merge_head_major(partials, lses)
    lses = lses.float()
    peak = lses.max(dim=0).values
    peak = torch.where(torch.isfinite(peak), peak, torch.zeros_like(peak))
    weights = torch.exp2(lses - peak)                         # -inf sources weigh 0
    total = weights.sum(dim=0)
    acc = (weights.unsqueeze(-1) * partials.float()).sum(dim=0)
    seen = total > 0
    out = torch.where(seen.unsqueeze(-1), acc / total.clamp_min(1e-30).unsqueeze(-1),
                      torch.zeros_like(acc))
    lse = torch.where(seen, peak + torch.log2(total.clamp_min(1e-30)),
                      torch.full_like(total, float("-inf")))
    return out, lse


def merge_lse_partials(partials, lses):
    """Merge per-source attention partials by their base-2 LSE (B1's local merge).

    `partials` `[sources, heads, rows, width]`, `lses` `[sources, heads, rows]`
    FP32 base 2. Returns `(out [rows, heads, width] in the partials' dtype,
    lse [rows, heads] FP32)`: FP32 accumulation, one rounding. A row no source
    saw (every LSE `-inf`) gets a zero output and `-inf` LSE, as
    `lse_reduce_scatter_reference` returns. The sink is not involved: it folds
    once after the merge (D6). The merge is associative, so B2's two-level
    form (four sources per switch, then the two switch results) is the same
    map up to rounding.
    """
    out, lse = _merge_head_major(partials, lses)
    return out.to(partials.dtype).permute(1, 0, 2).contiguous(), lse.t().contiguous()


class _DeviceBytes:
    """A zero-copy uint8 view of raw device memory (`__cuda_array_interface__`)."""

    def __init__(self, pointer, nbytes):
        self.__cuda_array_interface__ = {
            "shape": (int(nbytes),), "typestr": "|u1", "data": (int(pointer), False),
            "version": 3, "strides": None}


def relay_partner(rank, world_size, island_size):
    """consult18 B2: the rank's cross-switch partner (same position, other island)."""
    if world_size != 2 * island_size:
        raise ValueError("the pairwise relay needs exactly two islands")
    return (rank + island_size) % world_size


class CeExchange:
    """consult18 stage B1/B2: the prefill head gather and LSE reduce-scatter on
    copy-engine peer copies instead of SM copy kernels.

    One CUDA-IPC slab per rank holds the flags and, per operation, two banks
    of inbound slots. Every rank pushes blocks straight into the destinations'
    slots with `cudaMemcpyAsync` (cross-switch and in-switch copies on separate
    streams), publishes one monotonic flag per (source, destination) after its
    copies, and the destination waits its flags on the main stream before it
    reads. Flag and copy primitives are the copy-engine ring's (`DmaAllReduce`,
    whose launchers its prepared plan installed), so this needs the ring.

    Direct (B1): every block crosses straight to its destination — a quarter
    of each rank's blocks for every cross-switch peer cross the B<->D link.
    Relay (B2, `relay=True`): only the cross-switch partner pair exchanges
    over the link. The gather sends each block across once and the partner
    forwards it inside its switch; the LSE reduce-scatter merges the four
    partials of a switch first and sends one merged block per destination
    across (BF16 on the wire, FP32 merges). About a quarter of the link bytes.

    Bank reuse needs no end handshake: a rank writes a peer's bank b again two
    calls later, only after it waited for data the peer sent after its main
    stream had finished reading bank b (the argument of
    `pcie_dma._all_reduce_fp8`, which holds per phase here).
    """

    FLAG_SLOTS = 256
    FLAG_STRIDE = 128
    _OPS = ("lse", "gather", "scores")

    def _streams(self):
        from .runtime import SWITCHES

        return self._stream_sets["high" if SWITCHES.get("ce_priority") == "high" else "normal"]

    @property
    def _cross_stream(self):
        return self._streams()[0]

    @property
    def _local_stream(self):
        return self._streams()[1]

    @property
    def _flag_stream(self):
        return self._streams()[2]

    def __init__(self, *, group, ring, rank, world_size, device, max_rows, heads_per_rank,
                 lse_width, gather_width, island_size=4, score_offer=512):
        from b12x.comm.pcie.pcie_oneshot import PCIeOneshotAllReduce

        if world_size > 8:
            raise ValueError("the CE exchange's counter layout holds at most 8 ranks")
        def align(n):
            return (int(n) + 255) // 256 * 256
        self.ring = ring
        self.rank, self.world = int(rank), int(world_size)
        self.device = torch.device(device)
        self.max_rows, self.hpr = int(max_rows), int(heads_per_rank)
        self.lse_width, self.gather_width = int(lse_width), int(gather_width)
        self.lse_part = align(self.hpr * self.max_rows * self.lse_width * 2)
        self.lse_slot = self.lse_part + align(self.hpr * self.max_rows * 4)
        self.gather_slot = align(self.max_rows * self.hpr * self.gather_width * 2)
        self.score_offer = int(score_offer)
        self.score_slot = align(self.max_rows * self.score_offer * 4)
        self.lse_base = self.FLAG_SLOTS * self.FLAG_STRIDE
        # Two halves per LSE bank: [0, world) blocks for this rank's heads by
        # source (B2: index partner = the partner's merged block), [world, 2w)
        # B2's blocks for the partner's heads by local source.
        self.gather_base = self.lse_base + 2 * 2 * self.world * self.lse_slot
        self.scores_base = self.gather_base + 2 * self.world * self.gather_slot
        self.nbytes = self.scores_base + 2 * self.world * self.score_slot
        self._slab = PCIeOneshotAllReduce._allocate_shared_buffer(
            group, self.nbytes, zero_fill=True, ipc=ring._ipc)
        self.ptrs = [int(ptr) for ptr in self._slab.peer_ptrs]
        self._memory = torch.as_tensor(_DeviceBytes(self.ptrs[self.rank], self.nbytes),
                                       device=self.device)
        self._send = torch.zeros(self.FLAG_SLOTS, dtype=torch.int32, device=self.device)
        self._wait = torch.zeros(self.FLAG_SLOTS, dtype=torch.int32, device=self.device)
        island = [peer // island_size for peer in range(self.world)]
        peers = [peer for peer in range(self.world) if peer != self.rank]
        self.cross = [peer for peer in peers if island[peer] != island[self.rank]]
        self.local = [peer for peer in peers if island[peer] == island[self.rank]]
        self.partner = (relay_partner(self.rank, self.world, island_size)
                        if self.world == 2 * island_size else None)
        # consult18 P11: two stream sets — default priority, and the device's
        # highest priority (runtime switch `ce_priority`). The exchanges' copy,
        # flag and relay launches are small and latency-bound; at high priority
        # their pending blocks dispatch ahead of the lanes' compute blocks as SMs
        # free up (the chain of collectives is what paces the overlapped pair).
        _, high = torch.cuda.Stream.priority_range()
        self._stream_sets = {
            "normal": tuple(torch.cuda.Stream(device=self.device) for _ in range(3)),
            "high": tuple(torch.cuda.Stream(device=self.device, priority=high) for _ in range(3)),
        }
        self._ready = torch.cuda.Event()
        self._events = {}
        self.calls = {op: 0 for op in self._OPS}
        # The LSE relay's in-switch source slots (device index, built once and
        # outside any overlapped loop — see lse_reduce_scatter).
        self._switch_index = torch.tensor(sorted((self.rank, *self.local)), dtype=torch.long,
                                          device=self.device)
        # consult18 P3 INT8 wire: every source's slot (gather) / the switch's (LSE)
        # (built here: the overlapped loop must not make host transfers)
        self._all_sources = torch.arange(self.world, dtype=torch.int32, device=self.device)
        order = sorted((self.rank, *self.local))
        self._local_i32 = (
            torch.tensor(sorted(self.local), dtype=torch.int32, device=self.device),
            torch.tensor([order.index(src) for src in sorted(self.local)], dtype=torch.int32,
                         device=self.device),
            order.index(self.rank), len(order))

    # -- flags, counters, events ------------------------------------------
    def _flag_ptr(self, rank, op, source):
        return self.ptrs[rank] + (self._OPS.index(op) * 16 + source) * self.FLAG_STRIDE

    def _send_ctr(self, op, dest, source):
        return self._send.data_ptr() + (32 + self._OPS.index(op) * 64 + dest * 8 + source) * 4

    def _wait_ctr(self, op, source, relay=False):
        return self._wait.data_ptr() + ((128 if relay else 0) + self._OPS.index(op) * 16 + source) * 4

    def _event(self, key):
        event = self._events.get(key)
        if event is None:
            event = self._events[key] = torch.cuda.Event()
        return event

    def _bank(self, op):
        bank = self.calls[op] % 2
        self.calls[op] += 1
        return bank

    def _mark(self, label, stream=None):
        """HY4_TRACE breadcrumb on `stream` (default: the current stream)."""
        tracer = getattr(self, "tracer", None)
        if tracer is not None:
            tracer.mark(f"{getattr(self, 'tag', 'ce')} {label}", stream)

    def _send_block(self, stream, dest, copies, sources, op, key):
        """`copies` [(dest_offset, src_ptr, nbytes)] on `stream`, then flag `dest`
        once per source in `sources` (after the copies) on the flag stream."""
        kernels = self.ring._kernels
        event = self._event(key)
        call = self.calls[op]
        from .runtime import SWITCHES

        # consult18 P3 TIMING PROBE ONLY (results are garbage): halve the bytes
        # of the head gather and the LSE partials — the price of FP8 wires.
        half = SWITCHES.get("wire_probe") == "ce_half" and op in ("gather", "lse")
        with torch.cuda.stream(stream):
            for offset, source, nbytes in copies:
                if half:
                    nbytes = max(16, (nbytes // 2) // 16 * 16)
                kernels.dma_copy(self.ptrs[dest] + offset, source, nbytes)
            self._mark(f"copied {key[0]}->{dest} {op}#{call}", stream)
            event.record(stream)
        with torch.cuda.stream(self._flag_stream):
            self._flag_stream.wait_event(event)
            for origin in sources:
                kernels.dma_set_flag(self._flag_ptr(dest, op, origin),
                                     self._send_ctr(op, dest, origin))
            self._mark(f"flagged {key[0]}->{dest} src{tuple(sources)} {op}#{call}",
                       self._flag_stream)

    def _await(self, op, source, *, relay=False):
        """Wait (on the current stream) for `source`'s data of this call."""
        self.ring._kernels.dma_wait_flag(self._flag_ptr(self.rank, op, source),
                                         self._wait_ctr(op, source, relay))
        self._mark(f"got {op} src{source}{' relay' if relay else ''} #{self.calls[op]}")

    def _await_all(self, op, sources):
        """Wait (on the current stream) for every source's data of this call.
        consult18 P1: with the `batched_waits` switch on, ONE multi-flag wait
        launch (the same per-source protocol, in the same order) instead of
        one launch per source."""
        sources = tuple(sources)
        from .runtime import SWITCHES

        kernels = self.ring._kernels
        if (SWITCHES.get("batched_waits") == "on" and len(sources) > 1
                and hasattr(kernels, "dma_wait_flags")):
            kernels.dma_wait_flags([self._flag_ptr(self.rank, op, s) for s in sources],
                                   [self._wait_ctr(op, s, False) for s in sources])
            self._mark(f"got {op} src{sources} #{self.calls[op]}")
            return
        for source in sources:
            self._await(op, source)

    def _drain(self):
        main = torch.cuda.current_stream()
        for stream in (self._cross_stream, self._local_stream, self._flag_stream):
            main.wait_stream(stream)

    def _start(self):
        main = torch.cuda.current_stream()
        self._ready.record(main)
        for stream in (self._cross_stream, self._local_stream):
            stream.wait_event(self._ready)

    def accepts_lse(self, partials):
        rows, heads, width = (int(v) for v in partials.shape)
        return (partials.dtype == torch.bfloat16 and rows <= self.max_rows
                and heads == self.world * self.hpr and width == self.lse_width)

    def accepts_gather(self, local):
        rows, heads, width = (int(v) for v in local.shape)
        return (local.dtype == torch.bfloat16 and rows <= self.max_rows
                and heads == self.hpr and width == self.gather_width)

    # -- LSE reduce-scatter -----------------------------------------------
    def _lse_views(self, bank, half, rows, width):
        hpr = self.hpr
        first = self.lse_base + (bank * 2 + half) * self.world * self.lse_slot
        slots = self._memory[first:first + self.world * self.lse_slot].view(self.world, self.lse_slot)
        part_bytes, lse_bytes = hpr * rows * width * 2, hpr * rows * 4
        received = slots[:, :part_bytes].view(torch.bfloat16).view(self.world, hpr, rows, width)
        received_lse = slots[:, self.lse_part:self.lse_part + lse_bytes] \
            .view(torch.float32).view(self.world, hpr, rows)
        return received, received_lse

    def _lse_offset(self, bank, half, source):
        return self.lse_base + ((bank * 2 + half) * self.world + source) * self.lse_slot

    def _lse_relay_int8(self, partials, lses):
        """consult18 P3 (numerics class): the LSE reduce-scatter relay over an
        INT8 wire (one FP32 scale per (head, row) vector of the partials; the
        LSEs stay FP32; this rank's own partials stay exact)."""
        from .fused_ops import dequant_rows_int8, dequant_slots_int8, quant_rows_int8

        rows, heads, width = (int(v) for v in partials.shape)
        head_major = partials.permute(1, 0, 2)
        if not head_major.is_contiguous():
            head_major = head_major.contiguous()
        lse_major = lses.float().t().contiguous()
        hpr, me = self.hpr, self.rank
        n = hpr * rows
        soff = self._align(n * width)
        lse_bytes = n * 4
        q_all = torch.empty((heads * rows, width), dtype=torch.int8, device=self.device)
        s_all = torch.empty((heads * rows,), dtype=torch.float32, device=self.device)
        quant_rows_int8(head_major.view(heads * rows, width), q_all, s_all)

        def block(owner, offset):
            first = owner * n
            return ((offset, q_all[first].data_ptr(), n * width),
                    (offset + soff, s_all[first:].data_ptr(), n * 4),
                    (offset + self.lse_part, lse_major[owner * hpr].data_ptr(), lse_bytes))

        def views(bank, half):
            first = self.lse_base + (bank * 2 + half) * self.world * self.lse_slot
            slots = self._memory[first:first + self.world * self.lse_slot] \
                .view(self.world, self.lse_slot)
            q = slots[:, :n * width].view(torch.int8).view(self.world, n, width)
            sc = slots[:, soff:soff + n * 4].view(torch.float32).view(self.world, n)
            ls = slots[:, self.lse_part:self.lse_part + lse_bytes].view(torch.float32) \
                .view(self.world, hpr, rows)
            return q, sc, ls

        bank = self._bank("lse")
        self._start()
        partner = self.partner
        if partner is None:
            raise RuntimeError("the LSE relay needs two islands")
        for dest in self.local:
            other = relay_partner(dest, self.world, len(self.local) + 1)
            copies = (*block(dest, self._lse_offset(bank, 0, me)),
                      *block(other, self._lse_offset(bank, 1, me)))
            self._send_block(self._local_stream, dest, copies, (me,), "lse", ("lse1", dest))
        q0, s0, l0 = views(bank, 0)
        q1, s1, l1 = views(bank, 1)
        own = slice(me * hpr, (me + 1) * hpr)
        partner_heads = slice(partner * hpr, (partner + 1) * hpr)
        l0[me].copy_(lse_major[own])
        l1[me].copy_(lse_major[partner_heads])
        self._await_all("lse", self.local)
        local_src, local_dst, me_pos, k = self._local_i32
        first0 = self.lse_base + (bank * 2 + 0) * self.world * self.lse_slot
        first1 = self.lse_base + (bank * 2 + 1) * self.world * self.lse_slot
        slots0 = self._memory[first0:first0 + self.world * self.lse_slot].view(self.world, self.lse_slot)
        slots1 = self._memory[first1:first1 + self.world * self.lse_slot].view(self.world, self.lse_slot)
        mine4_in = torch.empty((k, hpr, rows, width), dtype=partials.dtype, device=self.device)
        theirs4_in = torch.empty_like(mine4_in)
        # own partials exact; the three in-switch sources dequantized in place
        mine4_in[me_pos].copy_(head_major[own])
        theirs4_in[me_pos].copy_(head_major[partner_heads])
        dequant_slots_int8(slots0, local_src, soff, n, width, mine4_in, local_dst)
        dequant_slots_int8(slots1, local_src, soff, n, width, theirs4_in, local_dst)
        switch = self._switch_index
        mine4, mine4_lse = _merge_head_major(mine4_in, l0.index_select(0, switch))
        out4, out4_lse = _merge_head_major(theirs4_in, l1.index_select(0, switch))
        # Phase 2, across the link: the merged block for the partner's heads, INT8.
        q4 = torch.empty((n, width), dtype=torch.int8, device=self.device)
        s4 = torch.empty((n,), dtype=torch.float32, device=self.device)
        quant_rows_int8(out4.to(torch.bfloat16).contiguous().view(n, width), q4, s4)
        out4_lse = out4_lse.contiguous()
        main = torch.cuda.current_stream()
        merged = self._event("lse-merged")
        merged.record(main)
        self._cross_stream.wait_event(merged)
        base = self._lse_offset(bank, 0, me)
        self._send_block(self._cross_stream, partner,
                         ((base, q4.data_ptr(), n * width), (base + soff, s4.data_ptr(), n * 4),
                          (base + self.lse_part, out4_lse.data_ptr(), lse_bytes)),
                         (me,), "lse", ("lse2", partner))
        self._await("lse", partner)
        theirs_partner = torch.empty((n, width), dtype=partials.dtype, device=self.device)
        dequant_rows_int8(q0[partner], s0[partner], theirs_partner)
        both = torch.stack((mine4, theirs_partner.view(hpr, rows, width).float()))
        both_lse = torch.stack((mine4_lse, l0[partner]))
        out, lse = _merge_head_major(both, both_lse)
        result = out.to(partials.dtype).permute(1, 0, 2).contiguous(), lse.t().contiguous()
        self._drain()
        return result

    def lse_reduce_scatter(self, partials, lses, *, relay=False):
        """`[rows, heads, width]` partials + `[rows, heads]` base-2 LSE over
        ALL heads -> this rank's merged head shard `(out, lse)`."""
        if relay and self._wire_int8():
            return self._lse_relay_int8(partials, lses)
        rows, heads, width = (int(v) for v in partials.shape)
        head_major = partials.permute(1, 0, 2)
        if not head_major.is_contiguous():
            head_major = head_major.contiguous()
        lse_major = lses.float().t().contiguous()
        hpr, me = self.hpr, self.rank
        part_bytes, lse_bytes = hpr * rows * width * 2, hpr * rows * 4

        def block(owner, offset):
            return ((offset, head_major[owner * hpr].data_ptr(), part_bytes),
                    (offset + self.lse_part, lse_major[owner * hpr].data_ptr(), lse_bytes))

        bank = self._bank("lse")
        self._start()
        own = slice(me * hpr, (me + 1) * hpr)
        mine, mine_lse = self._lse_views(bank, 0, rows, width)
        if not relay:
            for dest in (*self.cross, *self.local):
                stream = self._cross_stream if dest in self.cross else self._local_stream
                self._send_block(stream, dest, block(dest, self._lse_offset(bank, 0, me)),
                                 (me,), "lse", ("lse", dest))
            mine[me].copy_(head_major[own])
            mine_lse[me].copy_(lse_major[own])
            self._await_all("lse", (*self.cross, *self.local))
            result = merge_lse_partials(mine, mine_lse)
            self._drain()
            return result
        partner = self.partner
        if partner is None:
            raise RuntimeError("the LSE relay needs two islands")
        # Phase 1, inside the switch: each local peer gets its own heads' block
        # and its partner's heads' block.
        for dest in self.local:
            other = relay_partner(dest, self.world, len(self.local) + 1)
            copies = (*block(dest, self._lse_offset(bank, 0, me)),
                      *block(other, self._lse_offset(bank, 1, me)))
            self._send_block(self._local_stream, dest, copies, (me,), "lse", ("lse1", dest))
        theirs, theirs_lse = self._lse_views(bank, 1, rows, width)
        mine[me].copy_(head_major[own])
        mine_lse[me].copy_(lse_major[own])
        partner_heads = slice(partner * hpr, (partner + 1) * hpr)
        theirs[me].copy_(head_major[partner_heads])
        theirs_lse[me].copy_(lse_major[partner_heads])
        self._await_all("lse", self.local)
        # The switch's source slots as a cached DEVICE index: indexing with a
        # Python list uploads it on every call (a host transfer the overlapped
        # loop must not make); index_select gathers the same rows.
        switch = self._switch_index
        if switch is None:
            switch = self._switch_index = torch.tensor(sorted((me, *self.local)),
                                                       dtype=torch.long, device=self.device)
        mine4, mine4_lse = _merge_head_major(mine.index_select(0, switch),
                                             mine_lse.index_select(0, switch))
        out4, out4_lse = _merge_head_major(theirs.index_select(0, switch),
                                           theirs_lse.index_select(0, switch))
        out4 = out4.to(torch.bfloat16)
        # Phase 2, across the link: one merged block per partner pair.
        main = torch.cuda.current_stream()
        merged = self._event("lse-merged")
        merged.record(main)
        self._cross_stream.wait_event(merged)
        self._send_block(self._cross_stream, partner,
                         ((self._lse_offset(bank, 0, me), out4.data_ptr(), part_bytes),
                          (self._lse_offset(bank, 0, me) + self.lse_part, out4_lse.data_ptr(),
                           lse_bytes)),
                         (me,), "lse", ("lse2", partner))
        self._await("lse", partner)
        both = torch.stack((mine4, mine[partner].float()))
        both_lse = torch.stack((mine4_lse, mine_lse[partner]))
        out, lse = _merge_head_major(both, both_lse)
        result = out.to(partials.dtype).permute(1, 0, 2).contiguous(), lse.t().contiguous()
        self._drain()
        return result

    # -- scores-only offer gather (consult18 stage D: one channel per lane) --
    def accepts_scores(self, offers):
        rows, width = (int(v) for v in offers.shape)
        return offers.dtype == torch.float32 and rows <= self.max_rows and width == self.score_offer

    def exchange_scores(self, offers, *, relay=False):
        """Every rank's `[rows, offer]` FP32 scores in rank order (byte copies).
        The returned views live in this lane's bank: consume them before the
        lane's next-but-one scores call.

        N1: `relay=True` runs the two-hop island relay, the exact pattern of
        `all_gather_heads(relay=True)` (consult18 B2): each rank sends its
        offer ONCE across the B<->D link (to its cross-island partner) plus
        to its 3 in-switch peers, awaits the partner on `_local_stream`, then
        forwards the partner's offer inside the switch. Cross-RC score bytes
        drop 4x (16 MiB -> 4 MiB egress per rank per call) with zero extra
        buffer slots (the same `world * score_slot` bank) and byte-identical
        received scores — pure element movement either way."""
        rows, width = (int(v) for v in offers.shape)
        source = offers.contiguous()
        nbytes = rows * width * 4
        me = self.rank
        bank = self._bank("scores")

        def offset(origin):
            return self.scores_base + (bank * self.world + origin) * self.score_slot

        self._start()
        own = ((offset(me), source.data_ptr(), nbytes),)
        if not relay:
            for dest in (*self.cross, *self.local):
                stream = self._cross_stream if dest in self.cross else self._local_stream
                self._send_block(stream, dest, own, (me,), "scores", ("scores", dest))
        else:
            partner = self.partner
            if partner is None:
                raise RuntimeError("the scores relay needs two islands")
            self._send_block(self._cross_stream, partner, own, (me,), "scores", ("s-x", partner))
            for dest in self.local:
                self._send_block(self._local_stream, dest, own, (me,), "scores", ("s-own", dest))
            # Forward the partner's offer inside the switch once it has arrived.
            with torch.cuda.stream(self._local_stream):
                self._await("scores", partner, relay=True)
            forward = ((offset(partner), self.ptrs[me] + offset(partner), nbytes),)
            for dest in self.local:
                self._send_block(self._local_stream, dest, forward, (partner,), "scores",
                                 ("s-fwd", dest))
        slots = self._memory[offset(0):offset(0) + self.world * self.score_slot]
        received = slots.view(self.world, self.score_slot)[:, :nbytes] \
            .view(torch.float32).view(self.world, rows, width)
        received[me].copy_(source)
        self._await_all("scores", (*self.cross, *self.local))
        self._drain()
        return tuple(received[origin] for origin in range(self.world))

    # -- head all-gather --------------------------------------------------
    @staticmethod
    def _wire_int8():
        from .runtime import SWITCHES

        return SWITCHES.get("wire") == "int8"

    @staticmethod
    def _align(n):
        return (int(n) + 255) // 256 * 256

    def _all_gather_heads_int8(self, local, *, relay):
        """consult18 P3 (numerics class): the head gather over an INT8 wire —
        one FP32 scale per (row, head) vector; every rank (this one included)
        attends with the same dequantized queries."""
        from .fused_ops import dequant_slots_int8, quant_rows_int8

        rows, hpr, width = (int(v) for v in local.shape)
        n = rows * hpr
        soff = self._align(n * width)
        nbytes = soff + n * 4
        me = self.rank
        bank = self._bank("gather")

        def offset(origin):
            return self.gather_base + (bank * self.world + origin) * self.gather_slot

        slots = self._memory[offset(0):offset(0) + self.world * self.gather_slot] \
            .view(self.world, self.gather_slot)
        mine = slots[me]
        q_own = mine[:n * width].view(torch.int8).view(n, width)
        s_own = mine[soff:soff + n * 4].view(torch.float32)
        # quantized straight into this rank's own slot (sent from there)
        quant_rows_int8(local.contiguous().view(n, width), q_own, s_own)
        self._start()
        own = ((offset(me), mine.data_ptr(), nbytes),)
        if not relay:
            for dest in (*self.cross, *self.local):
                stream = self._cross_stream if dest in self.cross else self._local_stream
                self._send_block(stream, dest, own, (me,), "gather", ("gather", dest))
        else:
            partner = self.partner
            if partner is None:
                raise RuntimeError("the gather relay needs two islands")
            self._send_block(self._cross_stream, partner, own, (me,), "gather", ("g-x", partner))
            for dest in self.local:
                self._send_block(self._local_stream, dest, own, (me,), "gather", ("g-own", dest))
            with torch.cuda.stream(self._local_stream):
                self._await("gather", partner, relay=True)
            forward = ((offset(partner), self.ptrs[me] + offset(partner), nbytes),)
            for dest in self.local:
                self._send_block(self._local_stream, dest, forward, (partner,), "gather",
                                 ("g-fwd", dest))
        out = torch.empty((self.world * hpr, rows, width), dtype=torch.bfloat16,
                          device=self.device)
        self._await_all("gather", (*self.cross, *self.local))
        dequant_slots_int8(slots, self._all_sources, soff, n, width, out, self._all_sources,
                           head_major=True, hpr=hpr, rows=rows)
        self._drain()
        # Option 1 (PERF-PLAN.md Step 3): head-major `[heads, rows, width]`
        # (rank-major head blocks preserved) — no permute/reshape copy after
        # the wire, and the absorb GEMM's `q_nope` slice is disjoint-stride.
        return out

    @property
    def gather_head_major(self):
        """Option 1 (PERF-PLAN.md Step 3): the INT8 wire's gather returns
        head-major `[heads, rows, width]` (the BF16 wire stays row-major)."""
        return self._wire_int8()

    def all_gather_heads(self, local, *, relay=False):
        """`[rows, heads // world, width]` -> rank-major heads.

        INT8 wire (Option 1): contiguous head-major `[heads, rows, width]`;
        BF16 wire: row-major `[rows, heads, width]`. `gather_head_major` says
        which one a caller is looking at."""
        if self._wire_int8():
            return self._all_gather_heads_int8(local, relay=relay)
        rows, hpr, width = (int(v) for v in local.shape)
        source = local.contiguous()
        nbytes = rows * hpr * width * 2
        me = self.rank
        bank = self._bank("gather")

        def offset(origin):
            return self.gather_base + (bank * self.world + origin) * self.gather_slot

        self._start()
        own = ((offset(me), source.data_ptr(), nbytes),)
        if not relay:
            for dest in (*self.cross, *self.local):
                stream = self._cross_stream if dest in self.cross else self._local_stream
                self._send_block(stream, dest, own, (me,), "gather", ("gather", dest))
        else:
            partner = self.partner
            if partner is None:
                raise RuntimeError("the gather relay needs two islands")
            self._send_block(self._cross_stream, partner, own, (me,), "gather", ("g-x", partner))
            for dest in self.local:
                self._send_block(self._local_stream, dest, own, (me,), "gather", ("g-own", dest))
            # Forward the partner's block inside the switch once it has arrived.
            with torch.cuda.stream(self._local_stream):
                self._await("gather", partner, relay=True)
            forward = ((offset(partner), self.ptrs[me] + offset(partner), nbytes),)
            for dest in self.local:
                self._send_block(self._local_stream, dest, forward, (partner,), "gather",
                                 ("g-fwd", dest))
        slots = self._memory[offset(0):offset(0) + self.world * self.gather_slot]
        received = slots.view(self.world, self.gather_slot)[:, :nbytes] \
            .view(torch.bfloat16).view(self.world, rows, hpr, width)
        received[me].copy_(source)
        self._await_all("gather", (*self.cross, *self.local))
        gathered = received.permute(1, 0, 2, 3).reshape(rows, self.world * hpr, width)
        self._drain()
        return gathered


def a2a_pool_row_caps(*, absorbed_width, unabsorbed_width, expanded_width,
                      decode_rows, prefill_widths, ce_rows, audit,
                      force_caps=False):
    """PERF-PLAN.md Step 5a — dead `PCIeDCPA2APool` row caps (pure arithmetic).

    The absorbed-query gather (`absorbed_width`, 576) is decode/verify/batch
    only, so its A2A pool caps at `max(*decode_rows, 1)` UNCONDITIONALLY. When
    the CE exchange covers `ce_rows` and above, the `unabsorbed_width` (256)
    gather pool and the `expanded_width` (256) partials pool only serve
    `w < ce_rows` — they cap at the largest such prepared prefill width, and
    the caller must skip `prepare_gather`/`prepare_lse` plan binding at
    `w >= ce_rows` (`skip_prepare_above`; those widths ride the CE exchange).

    `audit` (HY4_CE_AUDIT=1) runs the SM-kernel arm at EVERY CE width, so the
    full pools and bound plans must stay: the 256 pools keep `CHUNK` rows
    (absent from `caps`, the caller falls back to `max_rows`). Combining the
    audit with explicitly forced caps raises loudly — the audit's kernel arm
    would have neither pool rows nor plans above the cap — mirroring the
    HY4_AR_AUDIT fail-loud at `h4_serve.py:2810`.

    Returns `{"caps": {(width, kind): rows}, "skip_prepare_above": rows | None}`;
    missing keys mean "keep `max_rows` (CHUNK)".
    """
    caps = {(int(absorbed_width), "gather"): max((int(w) for w in decode_rows), default=1)}
    ce = None if ce_rows is None else int(ce_rows)
    if audit and force_caps and ce is not None:
        raise ValueError(
            "HY4_CE_AUDIT runs the SM-kernel arm at every CE width: capped dead-A2A "
            "pools leave that arm with neither pool rows nor bound plans — leave the "
            "caps off (unset HY4_A2A_CAP)")
    if ce is None or audit:
        # Nothing is dead (no CE coverage) or nothing may be dead (audit boot).
        return {"caps": caps, "skip_prepare_above": None}
    below = [int(w) for w in prefill_widths if int(w) < ce]
    cap = max(below) if below else 1
    caps[(int(unabsorbed_width), "gather")] = cap
    if expanded_width is not None:
        caps[(int(expanded_width), "partials")] = cap
    return {"caps": caps, "skip_prepare_above": ce}


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
                 all_reduce=None, reduce_width=None, all_reduce_arms=None, score_offer=None,
                 dma=None, kernel_rows=None, ce=None, ce_lse_rows=None,
                 ce_gather_rows=None, ce_lse_relay=False, ce_gather_relay=False,
                 ce_scores_relay=False, lane_rings=(), lane_ces=()):
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
        # consult18 stage A1: the copy-engine ring (None = not built) and the
        # largest row count the kernel pool was sized for (None = unbounded).
        self._dma = dma
        # consult18 B1: the copy-engine DCP exchange and the narrowest row
        # count each operation sends through it (None = that op stays on the
        # SM kernels). HY4_CE_AUDIT=1 also runs the kernel path on every CE
        # call: gathers must be byte-identical, merges within CE_AUDIT_REL_L2.
        self._ce = ce
        self.ce_lse_rows = None if ce_lse_rows is None else int(ce_lse_rows)
        self.ce_gather_rows = None if ce_gather_rows is None else int(ce_gather_rows)
        self.ce_lse_relay, self.ce_gather_relay = bool(ce_lse_relay), bool(ce_gather_relay)
        # N1: the scores exchange's two-hop island relay (its `relay=True`).
        self.ce_scores_relay = bool(ce_scores_relay)
        # consult18 stage D: one copy-engine ring and one CE exchange per
        # extra lane (index i -> lane i + 1); every channel is single-stream,
        # so two in-flight halves never share one. Lane plans: (lane, rows).
        self._lane_rings = tuple(lane_rings)
        self._lane_ces = tuple(lane_ces)
        self._lane_ar_plans = {}
        # consult18 D: HY4_TRACE=1 — host-mapped per-stream breadcrumbs (hangdump.Tracer).
        self.tracer = None
        import os as _os_t
        if _os_t.environ.get("HY4_TRACE") and dma is not None:
            from .hangdump import Tracer

            self.tracer = Tracer(dma._kernels)
            self.tracer.name_stream(torch.cuda.current_stream(self.device), "main")
            for index, channel in enumerate((ce, *self._lane_ces)):
                if channel is None:
                    continue
                channel.tracer, channel.tag = self.tracer, f"ce{index}"
                self.tracer.name_stream(channel._cross_stream, f"ce{index}.cross")
                self.tracer.name_stream(channel._local_stream, f"ce{index}.local")
                self.tracer.name_stream(channel._flag_stream, f"ce{index}.flag")
        import os as _os
        self.ce_audit = bool(_os.environ.get("HY4_CE_AUDIT"))
        self.ce_audit_calls = {"lse": 0, "gather": 0}
        self.ce_audit_max = 0.0
        # consult18 D: HY4_LANE_AUDIT=1 (a gate boot) checks the overlapped
        # halves' channels against the host control group. Gathers and score
        # offers every call (byte-equal); merges and ring sums every
        # HY4_LANE_AUDIT_EVERY-th call (their reference moves every rank's
        # full partials over gloo). WARNING: the 626151a9 audit boot deadlocked
        # across ranks inside this audit (switch-B ranks in the gloo gather,
        # switch-D ranks unable to drain their lane-0 stream): a host barrier
        # inside the overlapped loop is NOT safe, so this audit must become
        # post-hoc (snapshot in the loop, compare after it) before it is used.
        # HY4_LANE_AUDIT=1 records in the loop and compares after it
        # (flush_lane_audit, on the main stream once both lanes have joined):
        # nothing host-blocking runs between two lanes' exchanges. "inloop" is
        # the old form that compared inside the loop and deadlocked the ranks
        # (kept only to reproduce that hang under HY4_TRACE).
        mode = (_os.environ.get("HY4_LANE_AUDIT") or "").strip()
        self.lane_audit = bool(mode)
        self.lane_audit_inloop = mode == "inloop"
        self._posthoc = []
        self._digest_weights = None
        self._projection = None
        self.lane_audit_every = max(1, int(_os.environ.get("HY4_LANE_AUDIT_EVERY") or "8"))
        # consult18 D (F1/F2): inside the overlapped loop every score offer rides
        # the lane's own CE exchange, padded to the offer width with -inf (and
        # the rare full-width widen in offer-width chunks): no host wire and no
        # kernel-pool (full-residency, peer-waiting) grid between two lanes.
        self.overlap_ce_scores = (_os.environ.get("HY4_OVERLAP_CE_SCORES", "1") or "1") != "0"
        # consult18 D (fix C): serialize collectives across lanes (see _comm_token).
        self.overlap_comm_token = (_os.environ.get("HY4_OVERLAP_COMM_TOKEN", "1") or "1") != "0"
        self._comm_last = None
        self.lane_audit_calls = {"gather": 0, "scores": 0, "lse": 0, "all_reduce": 0}
        self.lane_audit_seen = {"lse": 0, "all_reduce": 0}
        self.lane_audit_max = {"lse": 0.0, "all_reduce": 0.0}
        self.kernel_rows = None if kernel_rows is None else int(kernel_rows)
        # HY4_AR_AUDIT=1 (a gate boot): every ring call also runs the rsag
        # kernel arm on the same input and the two must agree within
        # AR_AUDIT_REL_L2. The kernel pool must then cover the ring's rows.
        import os
        self.ar_audit = bool(os.environ.get("HY4_AR_AUDIT"))
        self._ar_audit_plans = {}      # rows -> (plan, runtime) of the incumbent arm
        self.ar_audit_calls = 0
        self.ar_audit_max = 0.0
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
                           score_offer=None, extra_value_widths=(), value_max_rows=None,
                           dma=False, kernel_rows=None, ce_lse_rows=None,
                           ce_gather_rows=None, ce_lse_width=None, ce_gather_width=None,
                           ce_lse_relay=False, ce_gather_relay=False, ce_scores_relay=False,
                           lanes=1, lane_rows=None,
                           pool_row_caps=None):
        """One channel pool per wire geometry (the query gather's absorbed and
        unabsorbed widths, and the 512-wide latent partials).

        `pool_row_caps` (PERF-PLAN.md Step 5a, `a2a_pool_row_caps`) shrinks
        individual pools' `max_batch_size` to their dead-A2A row cap — a key
        absent from the mapping keeps `max_rows`. The scores pools and the
        all-reduce sizing are untouched (pool-neutrality: no
        `release_preparation` / `wire_decode_ops` changes).

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
        caps = dict(pool_row_caps or {})
        for width in sorted(set(query_widths)):
            capacity = int(caps.get((int(width), "gather"), max_rows))
            _trace(f"PCIeDCPA2APool gather width={width} rows={capacity}")
            pools[(width, "gather")] = PCIeDCPA2APool.from_process_group(
                process_group=process_group, device=device,
                max_batch_size=capacity, total_heads=total_heads,
                head_dim=width, query_head_dim=width, single_channel=True)
            _trace(f"PCIeDCPA2APool gather width={width} done")
        # consult17 (lever 9): `extra_value_widths` adds partial pools at other
        # widths (the 256-wide expanded prefill partials); `value_max_rows`
        # caps the rows of the `value_width` pool when only decode still uses
        # it (pools are sized by their row capacity).
        for width, capacity in ((value_width, value_max_rows or max_rows),
                                *((extra, caps.get((int(extra), "partials"), max_rows))
                                  for extra in extra_value_widths)):
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
            # consult18 P2: the one-row (decode) offer gets its own channel.
            # A captured decode step switches its pools to device-side slot
            # selection for good; the prefill offers must not share that
            # channel (an 8k cell measured -1.3 % after one capture with a
            # shared score pool). The other decode pools (absorbed gather,
            # latent partials) are decode-only already.
            pools[(words, "scores1")] = PCIeDCPA2APool.from_process_group(
                process_group=process_group, device=device,
                max_batch_size=1, total_heads=world_size,
                head_dim=words, query_head_dim=words, single_channel=True)
            _trace(f"PCIeDCPA2APool scores words={words} done")
        all_reduce = None
        ring = None
        if reduce_width is not None:
            from ignition.preparation.collectives import (
                create_all_reduce_runtime, create_dma_all_reduce_runtime,
            )

            # consult18 A1: with the ring carrying the wide rows, the kernel
            # pool only needs `kernel_rows` rows (its IPC storage scales with them).
            kernel_capacity = max_rows if kernel_rows is None else int(kernel_rows)
            _trace(f"PCIeAllReduce width={reduce_width} rows={kernel_capacity}")
            all_reduce = create_all_reduce_runtime(
                process_group, device, max_tokens=kernel_capacity, hidden_size=int(reduce_width))
            all_reduce.prepare_channels(("h4:allreduce",))
            _trace(f"PCIeAllReduce width={reduce_width} rows={kernel_capacity} done")
            if dma:
                _trace(f"DmaAllReduce width={reduce_width} rows={max_rows}")
                ring = create_dma_all_reduce_runtime(
                    process_group, device, max_tokens=max_rows, hidden_size=int(reduce_width))
                _trace(f"DmaAllReduce width={reduce_width} rows={max_rows} done")
        ce = None
        if ce_lse_rows is not None or ce_gather_rows is not None:
            if ring is None:
                raise ValueError("the CE exchange borrows the copy-engine ring's primitives: "
                                 "enable a dma all-reduce arm")
            _trace("CeExchange")
            ce = CeExchange(group=process_group, ring=ring, rank=rank, world_size=world_size,
                            device=device, max_rows=max_rows,
                            heads_per_rank=int(total_heads) // world_size,
                            lse_width=int(ce_lse_width or value_width),
                            gather_width=int(ce_gather_width or min(query_widths)),
                            score_offer=int(score_offer or 512))
            _trace(f"CeExchange done ({ce.nbytes} B slab)")
        lane_rings, lane_ces = [], []
        for extra in range(1, int(lanes)):
            # consult18 stage D: lane `extra`'s own channels, sized for a half.
            if ring is None or ce is None:
                raise ValueError("overlap lanes need the dma ring and the CE exchange")
            from ignition.preparation.collectives import create_dma_all_reduce_runtime

            rows_l = int(lane_rows or max_rows)
            _trace(f"lane {extra}: DmaAllReduce + CeExchange rows={rows_l}")
            lane_ring = create_dma_all_reduce_runtime(
                process_group, device, max_tokens=rows_l, hidden_size=int(reduce_width))
            lane_rings.append(lane_ring)
            lane_ces.append(CeExchange(group=process_group, ring=lane_ring, rank=rank,
                                       world_size=world_size, device=device, max_rows=rows_l,
                                       heads_per_rank=int(total_heads) // world_size,
                                       lse_width=ce.lse_width, gather_width=ce.gather_width,
                                       score_offer=ce.score_offer))
            _trace(f"lane {extra} done")
        return cls(group=process_group, device=device, pools=pools,
                   rank=rank, world_size=world_size, total_heads=total_heads,
                   all_reduce=all_reduce, reduce_width=reduce_width,
                   all_reduce_arms=all_reduce_arms, score_offer=score_offer,
                   dma=ring, kernel_rows=kernel_rows, ce=ce, ce_lse_rows=ce_lse_rows,
                   ce_gather_rows=ce_gather_rows, ce_lse_relay=ce_lse_relay,
                   ce_gather_relay=ce_gather_relay, ce_scores_relay=ce_scores_relay,
                   lane_rings=lane_rings, lane_ces=lane_ces)

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

    def _score_pool_key(self, words, rows):
        """The one-row offer rides its own channel when the transport has it."""
        if int(rows) == 1 and (int(words), "scores1") in self._pools:
            return (int(words), "scores1")
        return (int(words), "scores")

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
        runtime = self._pools[self._score_pool_key(words, rows)].for_stream()
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
        `twoshot` is the BF16 two-shot runtime the manager builds beside it;
        `dma` (consult18 A1) is the copy-engine ring. Each is prepared exactly
        as the c1 cells' bench or FLARE's `PreparedAllReduce` prepares it.
        """
        if self._ar is None:
            raise RuntimeError("no all-reduce runtime (construct with reduce_width)")
        rows = int(rows)
        if rows in self._ar_plans:
            return self._ar_plans[rows][0]
        arm, threads, blocks = all_reduce_arm_for(rows, self.all_reduce_arms)
        label = name or f"ar-{arm}-{rows}-{self.reduce_width}"
        if arm == "dma":
            plan = self._prepare_ring(session, rows, label)
            self._ar_plans[rows] = (plan, self._dma, "dma")
            if self.ar_audit:
                self._ar_audit_plans[rows] = self._prepare_kernel_arm(
                    session, rows, "rsag", None, None, label + "-audit")
        else:
            plan, target = self._prepare_kernel_arm(session, rows, arm, threads, blocks, label)
            self._ar_plans[rows] = (plan, target, arm)
        self._prepared += 1
        return plan

    def _prepare_ring(self, session, rows, label):
        """consult18 A1: declare and prime the copy-engine ring for `rows`."""
        from b12x.comm.pcie import _dma_preparation as dma
        from b12x.preparation import CollectiveRequirement

        if self._dma is None:
            raise RuntimeError("the dma all-reduce arm needs the ring (construct with dma=True)")
        inp = torch.zeros((rows, self.reduce_width), dtype=torch.bfloat16, device=self.device)
        out = torch.empty_like(inp)
        if not self._dma.should_allreduce(inp):
            raise RuntimeError(
                f"the dma ring refuses rows={rows} at width {self.reduce_width} "
                "(capacity or 8-element shard alignment)")
        plan = dma.plan(dma.query_from_metadata(self._dma, shape=tuple(inp.shape), dtype=inp.dtype),
                        runtime=self._dma)
        session.prepare(
            (plan.request(
                name=label,
                collective=CollectiveRequirement(key=label, ranks=tuple(range(self.world_size))),
                prepare_call=lambda state: dma.prepared_call(state, inp=inp, out=out),
            ),),
            coordinator=lambda progress: progress.ready_collectives[0].key
            if progress.ready_collectives else None,
        )
        return plan

    def prepare_lane_all_reduce(self, session, lane_index, rows):
        """consult18 stage D: the ring plan for `rows` on lane `lane_index`'s ring."""
        from b12x.comm.pcie import _dma_preparation as dma
        from b12x.preparation import CollectiveRequirement

        ring = self._lane_rings[lane_index - 1]
        rows = int(rows)
        inp = torch.zeros((rows, self.reduce_width), dtype=torch.bfloat16, device=self.device)
        out = torch.empty_like(inp)
        label = f"ar-lane{lane_index}-{rows}-{self.reduce_width}"
        plan = dma.plan(dma.query_from_metadata(ring, shape=tuple(inp.shape), dtype=inp.dtype),
                        runtime=ring)
        session.prepare(
            (plan.request(
                name=label,
                collective=CollectiveRequirement(key=label, ranks=tuple(range(self.world_size))),
                prepare_call=lambda state: dma.prepared_call(state, inp=inp, out=out),
            ),),
            coordinator=lambda progress: progress.ready_collectives[0].key
            if progress.ready_collectives else None,
        )
        self._lane_ar_plans[(lane_index, rows)] = (plan, ring, "dma")
        ring.min_bytes = rows * self.reduce_width * 2
        if getattr(ring, "_fp8", "") not in ("a2a", "i8_a2a", "mx_a2a"):
            # Eager replay covers the lossless ring and the ring-topology
            # compressed wires (B12X_PCIE_DMA_FP8 ag / i8 / mx / *_ring).
            ring.prepare_eager_replay(torch.bfloat16, max_elements=rows * self.reduce_width,
                                      state=plan.prepared.state)
        return plan

    def prepare_dma_replay(self):
        """consult18 A1: capture the ring's eager-replay graphs (call collectively,
        after every ring plan is prepared). One graph launch per call instead
        of ~110 host launches; only capacities from the narrowest ring row
        count up are captured."""
        if self._dma is None:
            return 0
        ring_rows = [rows for rows, entry in self._ar_plans.items() if entry[2] == "dma"]
        if not ring_rows:
            return 0
        self._dma.min_bytes = min(ring_rows) * self.reduce_width * 2
        state = self._ar_plans[max(ring_rows)][0].prepared.state
        self._dma.prepare_eager_replay(torch.bfloat16,
                                       max_elements=max(ring_rows) * self.reduce_width,
                                       state=state)
        return len(self._dma._eager_replays.get(torch.bfloat16, ()))

    def _prepare_kernel_arm(self, session, rows, arm, threads, blocks, label):
        """One of the one-shot pool's / two-shot runtime's arms; `(plan, runtime)`."""
        from dataclasses import replace

        from b12x.preparation import CollectiveRequirement, FrozenMapping

        if self.kernel_rows is not None and rows > self.kernel_rows:
            raise RuntimeError(
                f"rows={rows} runs the {arm} kernel arm but the kernel pool holds "
                f"{self.kernel_rows} rows; route these rows to dma or raise the pool")
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

        session.prepare(
            (plan.request(
                name=label,
                collective=CollectiveRequirement(key=label, ranks=tuple(range(self.world_size))),
                prepare_call=prime,
            ),),
            coordinator=lambda progress: progress.ready_collectives[0].key
            if progress.ready_collectives else None,
        )
        return plan, target

    def _audit_ring(self, inp, result):
        """HY4_AR_AUDIT: the ring's sum against the rsag kernel arm's, same input."""
        plan, target = self._ar_audit_plans[int(inp.shape[0])]
        reference = target.all_reduce(inp, plan=plan, out=torch.empty_like(inp))
        want = reference.float()
        error = float((result.float() - want).norm() / want.norm().clamp_min(1e-30))
        self.ar_audit_calls += 1
        self.ar_audit_max = max(self.ar_audit_max, error)
        if not error <= AR_AUDIT_REL_L2:
            raise RuntimeError(
                f"dma ring all-reduce differs from the rsag arm by rel-L2 {error:.3e} "
                f"(rows={int(inp.shape[0])}; band {AR_AUDIT_REL_L2:.1e})")

    def _total_heads_local(self):
        return self.total_heads // self.world_size

    # -- the SimulatedDcpTransport call surfaces ---------------------------

    @staticmethod
    def _wire(tensor):
        """The DCP kernels' wire dtype (BF16 partials / FP16; the FP32 gate body
        stages through the production wire and is restored on return)."""
        return tensor.to(torch.bfloat16) if tensor.dtype == torch.float32 else tensor

    def _lane_ce(self):
        """The current lane's CE exchange (lane 0: the base one)."""
        from .lanes import lane

        current = lane()
        if current == 0:
            return self._ce
        if current > len(self._lane_ces):
            raise RuntimeError(f"no CE exchange for lane {current}")
        return self._lane_ces[current - 1]

    def _lane_check(self, what):
        from .lanes import lane, overlapped

        if lane() or overlapped():
            raise RuntimeError(f"lane {lane()} reached the single-channel {what}: "
                               "an overlapped half must stay on its lane's channels "
                               "(CE exchange or dma ring: single-CTA peer waits only)")

    def _all_gather_heads_impl(self, local_input, out=None, *, plan=None):
        rows, _, width = (int(v) for v in local_input.shape)
        ce = self._lane_ce()
        if (out is None and ce is not None and self.ce_gather_rows is not None
                and rows >= self.ce_gather_rows and ce.accepts_gather(local_input)):
            _trace(f"all_gather_heads(ce) rows={rows} width={width}")
            gathered = ce.all_gather_heads(local_input, relay=self.ce_gather_relay)
            # Option 1 layout: the INT8 CE gather is head-major — the audits
            # compare the row-major map (a view; the same bytes per element).
            audit_view = gathered.transpose(0, 1) if ce.gather_head_major else gathered
            if self.ce_audit and not self._in_lane():
                self._audit_gather(local_input, audit_view)
            if self.lane_audit and self._in_lane():
                self._lane_audit_gather(local_input, audit_view)
            return gathered
        return self._kernel_gather(local_input, out)

    def gather_head_major(self, rows, width):
        """Option 1 (PERF-PLAN.md Step 3): will this gather dispatch to the
        INT8 CE wire (whose output is head-major `[heads, rows, width]`)?"""
        ce = self._lane_ce()
        return bool(ce is not None and ce.gather_head_major
                    and self.ce_gather_rows is not None
                    and int(rows) >= self.ce_gather_rows
                    and int(width) == int(ce.gather_width))

    def _kernel_gather(self, local_input, out=None):
        self._lane_check("DCP gather pool")
        rows, _, width = (int(v) for v in local_input.shape)
        plan = self._lookup("all_gather_heads", rows, width, "")
        pool = self._pools[(width, "gather")]
        _trace(f"all_gather_heads rows={rows} width={width}")
        result = pool.for_stream().all_gather_heads(
            self._wire(local_input.contiguous()), None if out is None else self._wire(out), plan=plan)
        _trace(f"all_gather_heads rows={rows} width={width} done")
        return result.to(local_input.dtype) if out is None else out

    def _lse_reduce_scatter_impl(self, partial_output, partial_lse, out=None, *, plan=None,
                                 lse_base=LSE_BASE, partials_sink_folded=None, return_lse=False):
        rows, heads, width = (int(v) for v in partial_output.shape)
        if partials_sink_folded is None:
            raise ValueError(
                "lse_reduce_scatter requires declared partials_sink_folded "
                "(the sink never enters the merge; D6)")
        ce = self._lane_ce()
        if (out is None and return_lse and partials_sink_folded is False
                and lse_base == LSE_BASE and ce is not None
                and self.ce_lse_rows is not None and rows >= self.ce_lse_rows
                and ce.accepts_lse(partial_output)):
            _trace(f"lse_reduce_scatter(ce) rows={rows} width={width}")
            merged = ce.lse_reduce_scatter(partial_output, partial_lse,
                                           relay=self.ce_lse_relay)
            if self.ce_audit and not self._in_lane():
                self._audit_lse(partial_output, partial_lse, merged)
            if self.lane_audit and self._in_lane():
                self._lane_audit_lse(partial_output, partial_lse, merged)
            return merged
        return self._kernel_lse(partial_output, partial_lse, out, lse_base=lse_base,
                                partials_sink_folded=partials_sink_folded,
                                return_lse=return_lse)

    def _kernel_lse(self, partial_output, partial_lse, out=None, *, lse_base,
                    partials_sink_folded, return_lse):
        self._lane_check("DCP partials pool")
        rows, heads, width = (int(v) for v in partial_output.shape)
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

    @staticmethod
    def _in_lane():
        """Inside an overlapped half (either lane): the lane audit's scope."""
        from .lanes import lane, overlapped

        return overlapped() or lane() != 0

    # -- post-hoc lane audit (records in the loop, compares after it) ------
    def _digest(self, tensor):
        """Two position-weighted int64 sums of `tensor`'s raw 16/32-bit words
        (an exact-equality fingerprint; computed on the current stream)."""
        words = tensor.contiguous().view(-1)
        words = words.view(torch.int16 if words.element_size() == 2 else torch.int32).to(torch.int64)
        n = int(words.numel())
        if self._digest_weights is None or int(self._digest_weights.shape[1]) < n:
            index = torch.arange(n, device=words.device, dtype=torch.int64)
            self._digest_weights = torch.stack((index % 65521 + 1, (index * 40503) % 65519 + 1))
        return (self._digest_weights[:, :n] * words).sum(dim=1)

    def prepare_lane_audit(self):
        """Build the post-hoc audit's constants BEFORE the overlapped loop (an
        upload inside it is exactly the transfer the loop must never make):
        the ring-sum projection (seeded device generator: same on every rank)
        and the digest weights at the widest audited tensor."""
        if not self.lane_audit or self.lane_audit_inloop:
            return
        if self._projection is None:
            generator = torch.Generator(device=self.device).manual_seed(18_04)
            self._projection = torch.randn(int(self.reduce_width), 4, generator=generator,
                                           device=self.device)
        if self._digest_weights is None:
            widest = 0
            for channel in (self._ce, *self._lane_ces):
                if channel is not None:
                    widest = max(widest, channel.max_rows * channel.hpr * channel.gather_width,
                                 channel.max_rows * channel.score_offer * 4)
            index = torch.arange(max(widest, 1), device=self.device, dtype=torch.int64)
            self._digest_weights = torch.stack((index % 65521 + 1, (index * 40503) % 65519 + 1))

    def _project(self, value):
        """Four random projections of a [rows, width] sum (linear: sum of the
        inputs' projections == the output's, up to the output's rounding)."""
        if self._projection is None:
            raise RuntimeError("prepare_lane_audit() must run before the overlapped loop")
        return (value.float() @ self._projection).sum(dim=0)

    def _record(self, kind, *items):
        self._posthoc.append((kind, *items))

    def flush_lane_audit(self):
        """Compare everything the overlapped halves recorded (call on every
        rank, after the loop: gloo + host reads are safe here)."""
        records, self._posthoc = self._posthoc, []
        if not records:
            return
        world = self.world_size

        def stacked(kind, index):
            return [r[index] for r in records if r[0] == kind]

        for kind in ("gather", "scores"):
            mine = stacked(kind, 1)
            if not mine:
                continue
            segments = torch.stack(stacked(kind, 2))               # [n, world, 2]
            everyone = self._gather_all(torch.stack(mine))          # world x [n, 2]
            for origin in range(world):
                bad = (segments[:, origin] != everyone[origin]).any(dim=1)
                if bool(bad.any()):
                    call = int(bad.nonzero()[0])
                    raise RuntimeError(
                        f"lane audit: {kind} call {call} received different bytes from rank "
                        f"{origin} than it sent (digest mismatch)")
            self.lane_audit_calls[kind] += len(mine)
        projected_in = stacked("ar", 1)
        if projected_in:
            outs = torch.stack(stacked("ar", 2))
            want = sum(part.double() for part in self._gather_all(torch.stack(projected_in)))
            rel = ((outs.double() - want).norm(dim=1) / want.norm(dim=1).clamp_min(1e-30))
            worst = float(rel.max())
            self.lane_audit_max["ar_projection"] = max(self.lane_audit_max.get("ar_projection", 0.0), worst)
            self.lane_audit_calls["ar_projection"] = self.lane_audit_calls.get("ar_projection", 0) + len(projected_in)
            if not worst <= AR_AUDIT_REL_L2:
                raise RuntimeError(f"lane audit: a ring sum's projections differ by {worst:.3e}")
        for record in records:
            if record[0] == "arfull":
                self._lane_audit_all_reduce_now(record[1], record[2])
            elif record[0] == "lse":
                self._lane_audit_lse_now(record[1], record[2], (record[3], record[4]))

    def _lane_audit_gather(self, local_input, gathered):
        """HY4_LANE_AUDIT: an overlapped half's CE gather == every rank's block."""
        if not self.lane_audit_inloop:
            hpr = int(local_input.shape[1])
            segments = torch.stack([self._digest(gathered[:, origin * hpr:(origin + 1) * hpr])
                                    for origin in range(self.world_size)])
            self._record("gather", self._digest(local_input), segments)
            return
        parts = self._gather_all(local_input.contiguous())
        want = torch.cat(parts, dim=1)
        self.lane_audit_calls["gather"] += 1
        if not torch.equal(gathered.contiguous().view(torch.int16), want.view(torch.int16)):
            raise RuntimeError(
                f"lane CE head gather differs from the control-group gather "
                f"(rows={int(local_input.shape[0])})")

    def _lane_audit_lse(self, partial_output, partial_lse, merged):
        """HY4_LANE_AUDIT: an overlapped half's CE merge vs an FP32 merge of every rank's
        partials gathered over gloo (sampled: the partials are the big wire)."""
        self.lane_audit_seen["lse"] += 1
        if (self.lane_audit_seen["lse"] - 1) % self.lane_audit_every:
            return
        if not self.lane_audit_inloop:
            got, got_lse = merged
            self._record("lse", partial_output.clone(), partial_lse.float().clone(),
                         got.clone(), got_lse.clone())
            return
        self._lane_audit_lse_now(partial_output, partial_lse, merged)

    def _lane_audit_lse_now(self, partial_output, partial_lse, merged):
        outs = self._gather_all(partial_output.contiguous())
        lses = self._gather_all(partial_lse.float().contiguous())
        hpr = int(partial_output.shape[1]) // self.world_size
        mine = slice(self.rank * hpr, (self.rank + 1) * hpr)
        stacked = torch.stack([o[:, mine].permute(1, 0, 2) for o in outs])      # [src, h, rows, w]
        stacked_lse = torch.stack([l[:, mine].t() for l in lses])                # [src, h, rows]
        want, want_lse = merge_lse_partials(stacked, stacked_lse)
        got, got_lse = merged
        error = float((got.float() - want.float()).norm() / want.float().norm().clamp_min(1e-30))
        finite = torch.isfinite(want_lse)
        lse_gap = (float((got_lse[finite] - want_lse[finite]).abs().max())
                   if bool(finite.any()) else 0.0)
        if not torch.equal(torch.isfinite(got_lse), finite):
            lse_gap = float("inf")
        self.lane_audit_calls["lse"] += 1
        self.lane_audit_max["lse"] = max(self.lane_audit_max["lse"], error)
        if not (error <= CE_AUDIT_REL_L2 and lse_gap <= 1e-2):
            raise RuntimeError(
                f"lane CE LSE merge differs from the gloo reference: rel-L2 {error:.3e}, "
                f"LSE gap {lse_gap:.3e} (rows={int(partial_output.shape[0])})")

    def _lane_audit_scores(self, offers, got):
        """HY4_LANE_AUDIT: an overlapped half's CE score offers == the gloo offers, bit for bit."""
        if not self.lane_audit_inloop:
            self._record("scores", self._digest(offers),
                         torch.stack([self._digest(part) for part in got]))
            return
        want = self._gather_all(offers.contiguous())
        self.lane_audit_calls["scores"] += 1
        for rank, (a, b) in enumerate(zip(got, want)):
            if not torch.equal(a.contiguous().view(torch.int32), b.contiguous().view(torch.int32)):
                raise RuntimeError(
                    f"lane CE score offers differ from the gloo offers for rank {rank} "
                    f"(rows={int(offers.shape[0])})")

    def _lane_audit_all_reduce(self, inp, result):
        """HY4_LANE_AUDIT: an overlapped half's ring sum vs an FP32 sum over gloo (sampled)."""
        self.lane_audit_seen["all_reduce"] += 1
        if not self.lane_audit_inloop:
            self._record("ar", self._project(inp), self._project(result))
            if not (self.lane_audit_seen["all_reduce"] - 1) % (4 * self.lane_audit_every):
                self._record("arfull", inp.clone(), result.clone())
            return
        if (self.lane_audit_seen["all_reduce"] - 1) % self.lane_audit_every:
            return
        self._lane_audit_all_reduce_now(inp, result)

    def _lane_audit_all_reduce_now(self, inp, result):
        want = sum(part.float() for part in self._gather_all(inp.contiguous()))
        error = float((result.float() - want).norm() / want.norm().clamp_min(1e-30))
        self.lane_audit_calls["all_reduce"] += 1
        self.lane_audit_max["all_reduce"] = max(self.lane_audit_max["all_reduce"], error)
        if not error <= AR_AUDIT_REL_L2:
            raise RuntimeError(
                f"lane ring all-reduce differs from the gloo FP32 sum by rel-L2 {error:.3e} "
                f"(rows={int(inp.shape[0])}; band {AR_AUDIT_REL_L2:.1e})")

    def _audit_gather(self, local_input, gathered):
        """HY4_CE_AUDIT: the CE gather must be byte-identical to the kernel's."""
        from .lanes import lane

        if lane():
            # The kernel path is single-channel (lane 0 only); lane 1 runs the
            # same CE code on its own instance and is not cross-checked here.
            self.ce_audit_skipped = getattr(self, "ce_audit_skipped", 0) + 1
            return
        want = self._kernel_gather(local_input)
        self.ce_audit_calls["gather"] += 1
        if not torch.equal(gathered.view(torch.int16), want.view(torch.int16)):
            raise RuntimeError(
                f"CE head gather differs from the kernel gather (rows={int(local_input.shape[0])})")

    def _audit_lse(self, partial_output, partial_lse, merged):
        """HY4_CE_AUDIT: the CE merge against the kernel reduce-scatter."""
        from .lanes import lane

        if lane():
            # The kernel path is single-channel (lane 0 only); lane 1 runs the
            # same CE code on its own instance and is not cross-checked here.
            self.ce_audit_skipped = getattr(self, "ce_audit_skipped", 0) + 1
            return
        want, want_lse = self._kernel_lse(partial_output, partial_lse, lse_base=LSE_BASE,
                                          partials_sink_folded=False, return_lse=True)
        got, got_lse = merged
        error = float((got.float() - want.float()).norm()
                      / want.float().norm().clamp_min(1e-30))
        finite = torch.isfinite(want_lse)
        lse_gap = float((got_lse[finite] - want_lse[finite]).abs().max()) if bool(finite.any()) else 0.0
        if not torch.equal(torch.isfinite(got_lse), finite):
            lse_gap = float("inf")
        self.ce_audit_calls["lse"] += 1
        self.ce_audit_max = max(self.ce_audit_max, error)
        if not (error <= CE_AUDIT_REL_L2 and lse_gap <= 1e-2):
            raise RuntimeError(
                f"CE LSE merge differs from the kernel: rel-L2 {error:.3e}, LSE gap "
                f"{lse_gap:.3e} (rows={int(partial_output.shape[0])})")

    def _exchange_scores_impl(self, offers, *, plan=None):
        """Every rank's `[rows, offer]` FP32 scores, in rank order.

        With a prepared score gather for this exact `(rows, offer)` the offer
        rides the DCP all-gather as raw 16-bit words (no host wire); any other
        shape — the full-width re-offer after a saturated cut, an unprepared
        row count — takes the control group, as before. Every rank holds the
        same plans, so every rank takes the same path.
        """
        rows, width = (int(v) for v in offers.shape)
        from .lanes import lane, overlapped

        if self.overlap_ce_scores and (lane() or overlapped()):
            got = self._ce_scores(self._lane_ce(), offers)
            if self.lane_audit:
                self._lane_audit_scores(offers, got)
            return got
        if lane():
            ce = self._lane_ce()
            if ce.accepts_scores(offers):
                got = ce.exchange_scores(offers, relay=self.ce_scores_relay)
                if self.lane_audit:
                    self._lane_audit_scores(offers, got)
                return got
            # Any other shape (an offer narrower than score_offer, the
            # full-width re-offer) takes the host wire. KNOWN HAZARD (consult18
            # D): a host barrier inside the overlapped loop has deadlocked the
            # ranks (the audit boot, 2026-11-15); --overlap stays off until the
            # loop is free of host barriers and device-wide synchronizes.
            return self._gather_all(offers)
        if overlapped():
            self._lane_check("score pool / control group")
        key = ("score_gather", rows, 2 * width, "")
        if (self.score_offer != width or offers.dtype != torch.float32
                or key not in self._plans):
            return self._gather_all(offers)
        words = offers.contiguous().view(torch.bfloat16).reshape(rows, 1, 2 * width)
        _trace(f"score_gather rows={rows} offer={width}")
        if self.tracer is not None:
            self.tracer.mark(f"pool scores rows={rows}")
        gathered = self._pools[self._score_pool_key(2 * width, rows)].for_stream().all_gather_heads(
            words, None, plan=self._plans[key])
        if self.tracer is not None:
            self.tracer.mark("pool scores done")
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

    # -- consult18 S1: cooperative admission contract ---------------------
    # A pending cooperative grid (the W4A16 routed experts) is admitted only
    # whole, and until then the device dispatches no later kernel from any
    # stream. Beside a resident kernel that waits on a PEER, that is a
    # cross-rank cycle (harness/coop_block.py). Inside the overlap the peer
    # waits are bounded instead: the comm token keeps one collective in flight
    # per rank; its peer waits are single-CTA b12x flag waits on at most
    # PEER_WAIT_STREAMS streams (CE: the lane stream + the gather relay's
    # local stream; the dma ring: the lane stream, its reduce-phase waits
    # split out of the multi-CTA wait_add by B12X_PCIE_DMA_SINGLE_CTA_WAITS);
    # every other collective path raises inside the overlap. The fused grid
    # leaves B12X_W4A16_COOP_RESERVE_SMS >= PEER_WAIT_STREAMS SMs free, so it
    # always fits beside them: once ordinary kernels drain it is admitted.
    PEER_WAIT_STREAMS = 2

    def admission_contract(self):
        """The overlap's cooperative-admission regime, or raise if none holds."""
        from b12x._lib.env import env_flag
        from b12x.moe._shared.kernels.w4a16.kernel.fused_launch import (
            cooperative_reserve_sms)

        sms = torch.cuda.get_device_properties(self.device).multi_processor_count
        reserve = cooperative_reserve_sms(sms)
        single = env_flag("PCIE_DMA_SINGLE_CTA_WAITS")
        # consult18 P1: with the comm token off each lane keeps one collective in
        # flight on its own channels, so the resident peer waits are bounded by
        # PEER_WAIT_STREAMS PER LANE.
        in_flight = 1 if self._token_active() else max(int(getattr(self, "lanes", 2) or 2), 2)
        need = self.PEER_WAIT_STREAMS * in_flight
        if reserve >= need and single:
            return (f"reserve {reserve} of {sms} SMs >= {need} single-CTA peer waits "
                    f"({in_flight} collective(s) in flight)")
        return None

    def _token_active(self):
        """The comm token's state for this call: HY4_OVERLAP_COMM_TOKEN at
        construction AND the `comm_token` runtime switch (A/B per request)."""
        from .runtime import SWITCHES

        return self.overlap_comm_token and SWITCHES.get("comm_token", "on") != "off"

    # -- consult18 D (fix C): one collective in flight per rank -----------
    # b12x's PCIe runtimes are single ordered channels ("callers must not
    # overlap collective streams"). Inside the overlap each collective waits
    # on a device event the previous collective of EITHER lane recorded when
    # it finished (after its side streams drained into the lane stream), so
    # collectives execute one at a time in host program order — identical on
    # every rank, the single-lane (B2) regime — while the other lane's
    # COMPUTE, which never waits on a peer, runs beside them.
    @contextmanager
    def _comm_token(self):
        from .lanes import overlapped

        if not (self._token_active() and overlapped() and self.device.type == "cuda"):
            yield
            return
        stream = torch.cuda.current_stream()
        if self._comm_last is not None:
            stream.wait_event(self._comm_last)
        try:
            yield
        finally:
            done = torch.cuda.Event()
            done.record(torch.cuda.current_stream())
            self._comm_last = done

    def all_gather_heads(self, local_input, out=None, *, plan=None):
        with self._comm_token():
            return self._all_gather_heads_impl(local_input, out, plan=plan)

    def lse_reduce_scatter(self, partial_output, partial_lse, out=None, *, plan=None,
                           lse_base=LSE_BASE, partials_sink_folded=None, return_lse=False):
        with self._comm_token():
            return self._lse_reduce_scatter_impl(
                partial_output, partial_lse, out, plan=plan, lse_base=lse_base,
                partials_sink_folded=partials_sink_folded, return_lse=return_lse)

    def exchange_scores(self, offers, *, plan=None):
        with self._comm_token():
            return self._exchange_scores_impl(offers, plan=plan)

    def all_reduce(self, value):
        with self._comm_token():
            return self._all_reduce_impl(value)

    def _ce_scores(self, ce, offers):
        """consult18 D (F1/F2): every rank's `[rows, width]` FP32 offer over this
        lane's CE exchange, whatever the width: padded with -inf to whole
        offer-width units (one CE call each) and trimmed back, so the result is
        bit-identical to the host wire's. -inf padding is not a candidate
        (`scores_only_threshold` counts finite scores only)."""
        rows, width = (int(v) for v in offers.shape)
        unit = int(ce.score_offer)
        if offers.dtype != torch.float32 or rows > ce.max_rows:
            raise RuntimeError(f"the lane CE cannot carry a {tuple(offers.shape)} {offers.dtype} offer")
        units = max(1, -(-width // unit))
        if units * unit == width:
            padded = offers.contiguous()
        else:
            padded = offers.new_full((rows, units * unit), float("-inf"))
            padded[:, :width] = offers
        if units == 1:
            got = ce.exchange_scores(padded, relay=self.ce_scores_relay)
            return tuple(part[:, :width] for part in got)
        # A bank is rewritten two calls later: copy each unit out first.
        pieces = [[] for _ in range(self.world_size)]
        for index in range(units):
            got = ce.exchange_scores(padded[:, index * unit:(index + 1) * unit].contiguous(),
                                     relay=self.ce_scores_relay)
            for origin, part in enumerate(got):
                pieces[origin].append(part.clone())
        return tuple(torch.cat(parts, dim=1)[:, :width] for parts in pieces)

    def stage_candidates(self, local_indices, local_scores, local_counts=None, *, plan=None):
        scores = self._gather_all(local_scores)
        indices = self._gather_all(local_indices)
        return tuple(zip(scores, indices))

    def _all_reduce_impl(self, value):
        from .profile import TIMER

        if self._ar is not None:
            # consult15 §2 lever 1: the b12x PCIe all-reduce over the prepared
            # pool (oneshot at 1 row, rsag above) — GPU to GPU, no host wire.
            if (value.dim() != 2 or value.dtype != torch.bfloat16
                    or int(value.shape[1]) != self.reduce_width):
                raise RuntimeError(
                    f"unprepared all-reduce value {tuple(value.shape)} {value.dtype}; "
                    f"the GPU path reduces [rows, {self.reduce_width}] BF16")
            from .lanes import lane

            entry = (self._ar_plans.get(int(value.shape[0])) if lane() == 0
                     else self._lane_ar_plans.get((lane(), int(value.shape[0]))))
            if entry is None:
                raise RuntimeError(
                    f"no prepared all-reduce plan for rows={int(value.shape[0])}; "
                    "plans bind before the run — there is no JIT at run time")
            plan, target, arm = entry
            if arm != "dma":
                from .lanes import overlapped

                if overlapped():
                    raise RuntimeError(
                        f"all-reduce arm {arm!r} (rows={int(value.shape[0])}) inside the "
                        "overlapped loop: only the dma ring has single-CTA peer waits")
            inp = value.contiguous()
            out = torch.empty_like(inp)
            _trace(f"all_reduce(gpu) rows={int(value.shape[0])} arm={arm}")
            if self.tracer is not None:
                self.tracer.mark(f"ar {arm} rows={int(value.shape[0])} lane{lane()}")
            if arm == "twoshot":
                if not target.accepts(inp):
                    inp = inp.clone()      # two-shot wants 16-byte-aligned packs
                prepared = plan.prepared.state.query.call
                result = target.all_reduce(inp, plan=plan, out=out,
                                           threads=prepared["threads"],
                                           block_limit=prepared["block_limit"])
            else:
                result = target.all_reduce(inp, plan=plan, out=out)
            if self.tracer is not None:
                self.tracer.mark(f"ar {arm} done lane{lane()}")
            if arm == "dma" and self.ar_audit and not self._in_lane():
                self._audit_ring(inp, result)
            if arm == "dma" and self.lane_audit and self._in_lane():
                self._lane_audit_all_reduce(inp, result)
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

    def gather_to_root(self, value):
        """consult18 P2: rank 0 gets every rank's `value` concatenated on its last
        dim as a HOST tensor (the sampler's input; it samples on the host);
        other ranks get None. One D2H per rank and a gloo gather — no all-gather
        and no H2D (the decode step's logits were all-gathered and moved back
        to the device only for the sampler to copy them out again)."""
        import torch.distributed as dist

        host = value.contiguous().cpu()
        if self.rank == 0:
            taken = [torch.empty_like(host) for _ in range(self.world_size)]
            dist.gather(host, taken, dst=0, group=self.group)
            return torch.cat(taken, dim=-1)
        dist.gather(host, None, dst=0, group=self.group)
        return None

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
        # consult18 D: device flags of the overlapped loop's narrow selections
        # (scores_only_flags); Decoder.forward_overlapped reads them after the loop.
        self.deferred_checks = []

    def _plan(self, kernel, shape, *tensors):
        """The prepared plan for (kernel, call shape); `shape=None` shares one plan."""
        if self.plans is None:
            return None
        if shape is None:
            return self.plans.shared(kernel)
        return self.plans.plan_for(shape, kernel)

    def all_gather_heads(self, q_local, *, call_shape=None):
        """`[rows, heads // world, width]` -> rank-major heads (INT8 CE wire:
        head-major `[heads, rows, width]` — see `gather_head_major`)."""
        return self.transport.all_gather_heads(
            q_local.contiguous(), plan=self._plan("all_gather_heads", call_shape))

    def gather_head_major(self, rows, width):
        """Option 1 (PERF-PLAN.md Step 3): this gather returns head-major
        `[heads, rows, width]` (the INT8 CE wire dispatch above `ce_gather_rows`).
        Mock-safe: callers `getattr`-fall back to row-major."""
        return self.transport.gather_head_major(rows, width)

    def lse_reduce(self, out_partial, lse_partial, *, partials_sink_folded, call_shape=None):
        """Base-2 fused reduce-scatter over ALL heads -> this rank's head shard."""
        # No `.contiguous()` on the partial: the CE exchange (consult18 B1)
        # sends head-major blocks and the expand einsum already lays them out
        # head-major; the kernel path makes its own contiguous copy.
        merged, merged_lse = self.transport.lse_reduce_scatter(
            out_partial, lse_partial.contiguous(),
            plan=self._plan("lse_reduce_scatter", call_shape),
            lse_base=self.lse_base, partials_sink_folded=partials_sink_folded, return_lse=True)
        return merged, merged_lse

    def all_reduce(self, value):
        return self.transport.all_reduce(value.contiguous())

    def embed_lookup(self, token_ids, embed, *, vocabulary_range):
        first, last = vocabulary_range
        inside = (token_ids >= first) & (token_ids < last)
        from .lanes import sync_free

        if sync_free():
            # consult18 D: no host read inside the overlapped loop (the masked
            # assignment below syncs through nonzero); same values. Also the
            # sync-free single-lane forward (consult18 P1/P2).
            rows = (token_ids - first).clamp(0, int(embed.shape[0]) - 1)
            out = torch.where(inside.reshape(-1, 1), embed[rows.reshape(-1)],
                              embed.new_zeros(()))
            return self.all_reduce(out)
        out = embed.new_zeros((int(token_ids.numel()), int(embed.shape[1])))
        if bool(inside.any()):
            out[inside] = embed[token_ids[inside] - first]
        return self.all_reduce(out)

    def gather_logits(self, logits, *, vocabulary_range, root_host=False):
        first, last = vocabulary_range
        if int(logits.shape[-1]) != last - first:
            raise ValueError(
                f"logits width {int(logits.shape[-1])} != the shard's "
                f"vocabulary range {vocabulary_range}")
        # Vocabulary shards are contiguous rank-major partitions, so rank
        # order is vocab order (the sampler's all-gather).
        if root_host and hasattr(self.transport, "gather_to_root"):
            # consult18 P2: the decode step's sampler input — rank 0 only, on
            # the host (other ranks return their local shard, never read).
            gathered = self.transport.gather_to_root(logits)
            return logits if gathered is None else gathered
        return self.transport.all_gather(logits.contiguous())

    def argmax_tokens(self, logits, *, vocabulary_range):
        """consult18 P4: greedy tokens on device, the same on every rank: `[rows]` int64.

        Each rank offers its shard's (max, first argmax as a global vocabulary
        id) per row over the score gather (the selection exchange's offer
        channel: a pure copy of FP32 bits). The winner is the largest value,
        ties to the lowest rank, i.e. the lowest vocabulary id: exactly
        `torch.argmax` over the gathered full row (ids < 2**24 are exact in
        FP32). No host read, so it is capture-safe at every prepared row count."""
        rows = int(logits.shape[0])
        local = logits.reshape(rows, -1)
        idx = local.argmax(dim=-1)
        val = local.gather(1, idx.unsqueeze(1)).reshape(rows).float()
        width = int(getattr(self.transport, "score_offer", None) or 2)
        offer = torch.full((rows, width), float("-inf"), dtype=torch.float32, device=local.device)
        offer[:, 0] = val
        offer[:, 1] = (idx + int(vocabulary_range[0])).float()
        got = self.transport.exchange_scores(offer, plan=self._plan("selection_exchange", None))
        stacked = torch.stack([g[:, :2] for g in got], dim=1)          # [rows, world, 2]
        best = stacked[:, :, 0].argmax(dim=1)                          # first max: lowest rank
        return stacked[torch.arange(rows, device=local.device), best, 1].long()

    def fused_handback_ok(self):
        """The fused selection + hand-back serves the scores-only exchange
        without the selection audit (the audit compares merge_topk outputs)."""
        return self.exchange_mode == "scores-only" and not self.audit

    def merge_topk_handback(self, candidate_scores, candidate_positions, *, topk, selected, counts,
                            base, local_capacity):
        """consult18 P2 (sync-free): the scores-only offer exchange, then ONE
        launch doing the selection, the decoder's position->slot map and the
        selection hand-back into `selected` / `counts` (this rank's compacted
        local slots); its deferred flag joins `deferred_checks`."""
        from .fused_ops import scores_only_select

        width = int(candidate_scores.shape[1])
        offer = width if not self.local_offer else min(int(self.local_offer), width)
        plan = self._plan("selection_exchange", None)
        offers = self.transport.exchange_scores(candidate_scores[:, :offer].contiguous(), plan=plan)
        _, _, flag = scores_only_select(
            offers, candidate_scores, candidate_positions, topk=topk, world=self.world_size,
            rank=self.rank, handback=(selected, counts, base.reshape(-1).to(torch.int64).contiguous(),
                                      int(local_capacity)))
        self.deferred_checks.append(flag)

    def merge_topk(self, candidate_scores, candidate_positions, *, topk):
        if self.exchange_mode == "scores-only":
            plan = self._plan("selection_exchange", None)
            from .lanes import overlapped, sync_free

            values, positions = scores_only_merge(
                lambda scores: self.transport.exchange_scores(scores, plan=plan),
                candidate_scores, candidate_positions,
                topk=topk, world=self.world_size, rank=self.rank,
                local_offer=self.local_offer or None,
                deferred=self.deferred_checks if sync_free() else None)
            if self.audit and not overlapped():
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
