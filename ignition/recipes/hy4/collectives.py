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
            offers = exchange_scores(self.group, self.rank, candidate_scores, self.local_offer)
            return scores_only_selection(
                offers, candidate_scores, candidate_positions,
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
    the threshold plateau). Padding `-inf` never counts.
    """
    stacked = torch.stack([offer.float() for offer in offers], dim=0)  # [world, rows, offer]
    thresholds, aboves = [], []
    for row in range(stacked.shape[1]):
        values = stacked[:, row].flatten()
        values = values[torch.isfinite(values)].sort(descending=True).values
        total = int(values.numel())
        if total <= topk:
            thresholds.append(float("-inf"))
            aboves.append(total)
            continue
        thresholds.append(float(values[topk - 1]))
        above = int((values > values[topk - 1]).sum())
        aboves.append(above)
    return thresholds, aboves


def check_scores_only_exactness(thresholds, local_offer, candidate_scores):
    """The exactness guard: the offer must not truncate at the cut.

    Each rank guards its OWN truncation from its own keys (the wire carries
    only scores): if its first UN-offered candidate reaches that row's global
    k-th score, un-offered keys might clear the threshold and the selection
    would not equal the global top-k (the W6 gate-b 2,048-selection equality).
    Raise instead of returning an inexact selection — the fix is a wider offer
    (`local_offer` up to the full width).
    """
    if local_offer is None:
        return  # full-width offer: nothing is ever truncated
    rows, width = candidate_scores.shape
    if width <= int(local_offer):
        return  # everything this rank has was offered
    for row in range(rows):
        first_unoffered = float(candidate_scores[row, int(local_offer)])
        if first_unoffered >= thresholds[row]:
            raise ValueError(
                f"scores-only offer truncated the selection at row {row}: the first "
                f"un-offered score {first_unoffered} reaches the global k-th score "
                f"{thresholds[row]}; widen local_offer (up to the full width) — "
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
    """
    if thresholds is None:
        thresholds, aboves = scores_only_threshold(offers, topk)
    rows, width = candidate_scores.shape
    out_values = candidate_scores.new_full((rows, topk), float("-inf"), dtype=torch.float32)
    out_positions = torch.full((rows, topk), -1, dtype=torch.int64, device=candidate_scores.device)
    for row in range(rows):
        threshold = thresholds[row]
        values = candidate_scores[row].float()
        positions = candidate_positions[row].long()
        plateau = torch.isfinite(values) & (values == threshold)
        # Global plateau fill order: (rank, ascending global position). This
        # rank's plateau keys in ascending position; earlier ranks fill first.
        plateau_positions = positions[plateau]
        plateau_values = values[plateau]
        order = plateau_positions.argsort()
        plateau_positions, plateau_values = plateau_positions[order], plateau_values[order]
        earlier = 0
        for other in range(rank):
            other_plateau = int((offers[other][row] == threshold).sum()) \
                if float("-inf") != threshold else 0
            earlier += other_plateau
        budget = topk - (aboves[row] if aboves is not None else 0)
        mine = max(0, min(int(plateau_positions.numel()), budget - earlier))
        keep = torch.isfinite(values) & (values > threshold)
        kept_values = values[keep]
        kept_positions = positions[keep]
        fill_values = torch.cat((kept_values, plateau_values[:mine]))
        fill_positions = torch.cat((kept_positions, plateau_positions[:mine]))
        count = min(topk, int(fill_values.numel()))
        if count:
            sort = fill_values[:count].sort(descending=True).indices
            out_values[row, :count] = fill_values[:count][sort].to(out_values.dtype)
            out_positions[row, :count] = fill_positions[:count][sort]
    return out_values, out_positions.to(torch.int32)


def scores_only_selection(offers, candidate_scores, candidate_positions, *, topk, world, rank,
                          local_offer):
    """The full scores-only merge for one rank: threshold, guard, own keys."""
    thresholds, aboves = scores_only_threshold(offers, topk)
    check_scores_only_exactness(thresholds, local_offer, candidate_scores)
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
                 all_reduce=None, reduce_width=None):
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
        self._ar_plans = {}            # rows -> (plan, channel)

    # -- construction ------------------------------------------------------

    @classmethod
    def from_process_group(cls, process_group, device, *, total_heads, query_widths,
                           value_width, max_rows, reduce_width=None):
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
        _trace(f"PCIeDCPA2APool partials width={value_width}")
        pools[(value_width, "partials")] = PCIeDCPA2APool.from_process_group(
            process_group=process_group, device=device,
            max_batch_size=max_rows, total_heads=total_heads,
            head_dim=value_width, query_head_dim=value_width, single_channel=True)
        _trace(f"PCIeDCPA2APool partials width={value_width} done")
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
                   all_reduce=all_reduce, reduce_width=reduce_width)

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

        consult15 §2 lever 1 — the c1 cells' arms (c1-unit-costs.md §1, the
        w8/w4prod allreduce cells): `oneshot` (remote_push_full) at 1 row,
        `rsag` (remote_push_rsag) above, threads 256 / blocks 16. Prepared
        through the same session + CollectiveRequirement form as the DCP
        gather/lse plans (the bench's `benchmark_pcie_allreduce_transports`
        prepare path, minus graph replay).
        """
        from dataclasses import replace

        from b12x.comm.pcie import _oneshot_preparation as oneshot
        from b12x.preparation import CollectiveRequirement, FrozenMapping

        if self._ar is None:
            raise RuntimeError("no all-reduce runtime (construct with reduce_width)")
        rows = int(rows)
        if rows in self._ar_plans:
            return self._ar_plans[rows][0]
        inp = torch.zeros((rows, self.reduce_width), dtype=torch.bfloat16, device=self.device)
        out = torch.empty_like(inp)
        target = self._ar._prepared_channel_for_stream(None, None)
        query = oneshot.query_from_runtime(
            target, surface="OneshotAllReduce.all_reduce", call={"inp": inp, "out": out})
        transport = "remote_push_full" if rows <= 1 else "remote_push_rsag"
        if not query.call["payload_ready_region_packs"]:
            raise RuntimeError(
                "the b12x all-reduce pool has no payload-ready region for "
                f"rows={rows} (remote_push arms unavailable at this setup)")
        call = {**query.call, "transport": transport, "threads": 256, "blocks": 16}
        query = replace(query, call=FrozenMapping(call))
        plan = oneshot.plan(query, runtime=target)

        def prime(state):
            return oneshot._prepare_plain_call(state, inp=inp, out=out)

        label = name or f"ar-{rows}-{self.reduce_width}"
        session.prepare(
            (plan.request(
                name=label,
                collective=CollectiveRequirement(key=label, ranks=tuple(range(self.world_size))),
                prepare_call=prime,
            ),),
            coordinator=lambda progress: progress.ready_collectives[0].key
            if progress.ready_collectives else None,
        )
        self._ar_plans[rows] = (plan, target)
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
        return self._gather_all(offers)

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
            plan, target = entry
            inp = value.contiguous()
            out = torch.empty_like(inp)
            _trace(f"all_reduce(gpu) rows={int(value.shape[0])}")
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
            offers = self.transport.exchange_scores(
                candidate_scores[:, :self.local_offer].contiguous() if self.local_offer
                else candidate_scores.contiguous(),
                plan=self._plan("selection_exchange", None))
            return scores_only_selection(
                offers, candidate_scores, candidate_positions,
                topk=topk, world=self.world_size, rank=self.rank, local_offer=self.local_offer)
        if self.exchange_mode != "candidates":
            raise ValueError(f"unknown selection exchange mode {self.exchange_mode!r}")
        candidates = self.transport.stage_candidates(
            candidate_positions.contiguous(), candidate_scores.contiguous(), None,
            plan=self._plan("selection_exchange", None))
        return merge_candidate_topk(candidates, topk)
