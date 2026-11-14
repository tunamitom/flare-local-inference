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
GPU implementation over `PCIeDCPA2A` is W7's (TODO hooks below name the
primitives per collective); the body is identical either way.

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
]

LSE_BASE = "base2"


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
    hand-back). Ties resolve as `torch.topk` resolves them on the single
    rank's `select` — the W6 gate-b tie cases are constructed so the selected
    set is tie-order independent.
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

    def all_gather_heads(self, q_local):
        # TODO(W7, GPU): `PCIeDCPA2A.all_gather_heads` (pcie_dcp_a2a:1331-1340).
        taken = self.group._exchange(self.rank, q_local.contiguous())
        return torch.cat(taken, dim=1)

    def lse_reduce(self, out_partial, lse_partial, *, partials_sink_folded):
        # TODO(W7, GPU): `PCIeDCPA2A.lse_reduce_scatter` fused reduce-scatter
        # (pcie_dcp_a2a:308-324) — partials over ALL heads in, this rank's
        # contiguous head shard out.
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
        # TODO(W7, GPU): the TP all-reduce (NCCL/one-shot over the TP group).
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
        # TODO(W7, GPU): `pcie_dcp_topk` candidate exchange into each row
        # owner, then `merge_tiled_topk_candidates`.
        taken = self.group._exchange(
            self.rank, (candidate_scores.contiguous(), candidate_positions.contiguous())
        )
        return merge_candidate_topk(taken, topk)
