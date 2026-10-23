"""Exact vocabulary-parallel speculative verification over sharded target scores.

Each tensor-parallel rank owns a contiguous vocabulary shard of the filtered
target scores (penalties, temperature and grammar already applied). Instead of
all-gathering full-vocabulary logits, every rank publishes a small summary per
row: its shard maximum, the exponential sum relative to that maximum, and its
K best scores with their global token IDs. Every rank merges the gathered
summaries with identical arithmetic into the global normalizer and a compact
candidate vocabulary, filters it with the ordinary top-k and top-p policy, and
verifies the chain over that compact vocabulary.

The compact result equals the dense one whenever the top-p nucleus provably
lies inside the gathered candidates: each rank's unpublished tokens score at or
below its K-th published score, so when the smallest kept probability exceeds
every such bound, no unpublished token can enter the nucleus. A top-k of at
most K is exact for the same reason. Rows that fail the check are flagged, and
the caller runs the dense path for the round; the flag is computed from
gathered data, so every rank takes the same branch.

Sampling keeps the dense path's per-row seed and offset. The compact vocabulary
is sorted by global token ID, so inverse-CDF draws visit the nucleus in dense
order and select the same token; only floating-point summation order differs.
"""

from dataclasses import dataclass

import torch

SUMMARY_CANDIDATES = 64


@dataclass
class ShardSummary:
    """One rank's per-row summary: [R] maximum and exp-sum, [R, K] scores and global IDs."""
    maximum: torch.Tensor
    exp_sum: torch.Tensor
    scores: torch.Tensor
    token_ids: torch.Tensor


def shard_summary(local_scores, rank_offset, candidates=SUMMARY_CANDIDATES):
    """Summarize a rank's [R, Vl] filtered FP32 scores for the candidate all-gather."""
    maximum = local_scores.amax(-1)
    safe = torch.where(torch.isfinite(maximum), maximum, 0.)
    exp_sum = torch.exp(local_scores - safe[:, None]).sum(-1)
    scores, local_ids = local_scores.topk(candidates, dim=-1)
    return ShardSummary(maximum, exp_sum, scores, local_ids + rank_offset)


@dataclass
class MergedTarget:
    """Compact filtered target probabilities: [R, C] global IDs (sorted, -1 padded), probabilities, [R] exactness."""
    token_ids: torch.Tensor
    probabilities: torch.Tensor
    complete: torch.Tensor


def merge_summaries(maximum, exp_sum, scores, token_ids, top_k, top_p, vocab):
    """Merge gathered [world, R(, K)] summaries into compact filtered probabilities.

    top_k is [R] int32 (values <= 0 or >= vocab disable it); top_p is [R] float32.
    """
    world, rows, candidates = scores.shape
    global_max = maximum.amax(0)
    safe_max = torch.where(torch.isfinite(global_max), global_max, 0.)
    shard_scale = torch.where(torch.isfinite(maximum), torch.exp(maximum - safe_max[None]), 0.)
    total = (exp_sum * shard_scale).sum(0)
    pooled_scores = scores.permute(1, 0, 2).reshape(rows, world * candidates)
    pooled_ids = token_ids.permute(1, 0, 2).reshape(rows, world * candidates)
    ordered, order = pooled_scores.sort(-1, descending=True)
    ordered_ids = pooled_ids.gather(-1, order)
    k_enabled = (top_k > 0) & (top_k < vocab)
    k_exact = ~k_enabled | (top_k <= candidates)
    rank = torch.arange(world * candidates, device=scores.device)[None, :]
    kth = ordered.gather(-1, (top_k.clamp(1, world * candidates) - 1).long()[:, None])
    in_top_k = ~k_enabled[:, None] | (ordered >= kth)
    exp_scores = torch.where(torch.isfinite(ordered), torch.exp(ordered - safe_max[:, None]), 0.)
    exp_scores = torch.where(in_top_k, exp_scores, 0.)
    # Top-k renormalizes over the kept tokens, all of which are pooled when top_k <= K.
    normalizer = torch.where(k_enabled, exp_scores.sum(-1), total)
    probabilities = exp_scores / normalizer[:, None]
    # Nucleus: the smallest prefix of descending probabilities whose mass reaches top_p.
    before = probabilities.cumsum(-1) - probabilities
    kept = (before < top_p[:, None]) & (probabilities > 0)
    smallest = torch.where(kept, probabilities, torch.inf).amin(-1)
    # An unpublished token on rank r scores at most that rank's K-th published score.
    bound_scores = torch.where(torch.isfinite(scores[:, :, -1]), scores[:, :, -1], -torch.inf)
    bound = torch.exp(bound_scores - safe_max[None]).amax(0) / normalizer
    nucleus_exact = smallest > bound
    complete = torch.isfinite(global_max) & k_exact & (k_enabled & (top_k <= candidates) | nucleus_exact)
    nucleus = torch.where(kept, probabilities, 0.)
    nucleus = nucleus / nucleus.sum(-1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)
    sorted_ids, by_id = torch.where(kept, ordered_ids, vocab).sort(-1)
    return MergedTarget(torch.where(sorted_ids < vocab, sorted_ids, -1), nucleus.gather(-1, by_id), complete)


def compact_chain_verify(merged, batch, width, draft_ids, draft_token_probs, seed, offset):
    """Verify [B, W-1] drafts against [B*W] merged rows over a per-row compact vocabulary.

    draft_token_probs gives the draft distribution at every compact candidate:
    a callable (ids [B, W-1, C'] -> probabilities) evaluated on the compact
    vocabulary. Returns (tokens [B, W], accepted [B], emitted [B]) like
    FlashInfer's chain_speculative_sampling on the dense vocabulary.
    """
    from flashinfer import sampling

    ids = merged.token_ids.view(batch, width, -1)
    probs = merged.probabilities.view(batch, width, -1)
    # One compact vocabulary per request: every position's nucleus plus its drafts, in global ID order.
    union = torch.cat([ids.reshape(batch, -1), draft_ids], -1)
    union = torch.where(union >= 0, union, torch.iinfo(torch.int64).max)
    union, _ = union.sort(-1)
    duplicate = torch.zeros_like(union, dtype=torch.bool)
    duplicate[:, 1:] = union[:, 1:] == union[:, :-1]
    union = torch.where(duplicate, torch.iinfo(torch.int64).max, union)
    union, _ = union.sort(-1)
    columns = union.shape[-1]
    valid = union != torch.iinfo(torch.int64).max
    # Scatter each position's nucleus into the shared compact columns.
    location = torch.searchsorted(union, torch.where(ids >= 0, ids, 0).reshape(batch, -1)).view_as(ids)
    target = torch.zeros(batch, width, columns, device=probs.device)
    target.scatter_add_(-1, location, torch.where(ids >= 0, probs, 0.))
    draft_columns = torch.searchsorted(union, draft_ids)
    draft = draft_token_probs(torch.where(valid, union, 0)[:, None, :].expand(batch, width - 1, columns))
    draft = torch.where(valid[:, None, :], draft, 0.)
    accepted = torch.zeros(batch, dtype=torch.int32, device=probs.device)
    emitted = torch.zeros(batch, dtype=torch.int32, device=probs.device)
    tokens, _, _ = sampling.chain_speculative_sampling(draft.contiguous(), draft_columns.int().contiguous(),
        target.contiguous(), accepted, emitted, deterministic=True, seed=seed, offset=offset,
        row_independent_rng=True)
    tokens = torch.where(tokens >= 0, union.gather(-1, tokens.clamp_min(0).long()), -1)
    return tokens, accepted, emitted
