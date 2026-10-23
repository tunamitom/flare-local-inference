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
import triton
import triton.language as tl

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


# Fused kernels. Scores travel as int64 sort keys: an order-preserving int32 image of the FP32 score in the high
# word and the complement of the token ID in the low word, so a descending sort orders by score, then by ID.
_LOW = tl.constexpr(0xFFFFFFFF)


@triton.jit
def _score_key(x):
    bits = x.to(tl.int32, bitcast=True)
    return tl.where(bits >= 0, bits, bits ^ 0x7FFFFFFF)


@triton.jit
def _key_score(key):
    bits = tl.where(key >= 0, key, key ^ 0x7FFFFFFF)
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def _pack(x, ids):
    return (_score_key(x).to(tl.int64) << 32) | (_LOW - ids.to(tl.int64))


@triton.jit
def _summary_chunk_kernel(Scores, ChunkMax, ChunkSum, ChunkTop, stride, local, chunks,
                          K: tl.constexpr, BLOCK: tl.constexpr):
    # One program per (row, chunk): chunk maximum, exp-sum relative to it, and its K best keys.
    row, chunk = tl.program_id(0), tl.program_id(1)
    ids = chunk * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(Scores + row * stride + ids, mask=ids < local, other=-float("inf")).to(tl.float32)
    maximum = tl.max(x, 0)
    safe = tl.where(maximum == -float("inf"), 0., maximum)
    tl.store(ChunkMax + row * chunks + chunk, maximum)
    tl.store(ChunkSum + row * chunks + chunk, tl.sum(tl.exp(x - safe), 0))
    tl.store(ChunkTop + (row * chunks + chunk) * K + tl.arange(0, K), tl.topk(_pack(x, ids), K))


@triton.jit
def _summary_merge_kernel(ChunkMax, ChunkSum, ChunkTop, Maximum, ExpSum, TopScores, TopIds, chunks, offset,
                          K: tl.constexpr, CHUNKS: tl.constexpr):
    row = tl.program_id(0)
    lane = tl.arange(0, CHUNKS)
    maxima = tl.load(ChunkMax + row * chunks + lane, mask=lane < chunks, other=-float("inf"))
    sums = tl.load(ChunkSum + row * chunks + lane, mask=lane < chunks, other=0.)
    maximum = tl.max(maxima, 0)
    safe = tl.where(maximum == -float("inf"), 0., maximum)
    scale = tl.where(maxima == -float("inf"), 0., tl.exp(maxima - safe))
    slot = tl.arange(0, CHUNKS * K)
    keys = tl.load(ChunkTop + row * chunks * K + slot, mask=slot < chunks * K, other=-9223372036854775807 - 1)
    best = tl.topk(keys, K)
    out = tl.arange(0, K)
    tl.store(Maximum + row, maximum)
    tl.store(ExpSum + row, tl.sum(sums * scale, 0))
    tl.store(TopScores + row * K + out, _key_score((best >> 32).to(tl.int32)))
    tl.store(TopIds + row * K + out, _LOW - (best & _LOW) + offset)


def fused_shard_summary(local_scores, rank_offset, candidates=SUMMARY_CANDIDATES, block=2048):
    """shard_summary in two kernels: per-chunk summaries in parallel, then one merge program per row."""
    rows, local = local_scores.shape
    device = local_scores.device
    chunks = triton.cdiv(local, block)
    chunk_max = torch.empty(rows, chunks, device=device)
    chunk_sum = torch.empty(rows, chunks, device=device)
    chunk_top = torch.empty(rows, chunks, candidates, dtype=torch.int64, device=device)
    _summary_chunk_kernel[(rows, chunks)](local_scores, chunk_max, chunk_sum, chunk_top, local_scores.stride(0),
                                          local, chunks, K=candidates, BLOCK=block, num_warps=4)
    maximum = torch.empty(rows, device=device)
    exp_sum = torch.empty(rows, device=device)
    scores = torch.empty(rows, candidates, device=device)
    token_ids = torch.empty(rows, candidates, dtype=torch.int64, device=device)
    _summary_merge_kernel[(rows,)](chunk_max, chunk_sum, chunk_top, maximum, exp_sum, scores, token_ids, chunks,
                                   int(rank_offset), K=candidates, CHUNKS=triton.next_power_of_2(chunks),
                                   num_warps=4)
    return ShardSummary(maximum, exp_sum, scores, token_ids)


@triton.jit
def _merge_kernel(Maximum, ExpSum, Scores, Ids, TopK, TopP, OutIds, OutProbs, Complete, rows, vocab,
                  WORLD: tl.constexpr, K: tl.constexpr, NUCLEUS: tl.constexpr):
    row = tl.program_id(0)
    C: tl.constexpr = WORLD * K
    ranks = tl.arange(0, WORLD)
    maxima = tl.load(Maximum + ranks * rows + row)
    sums = tl.load(ExpSum + ranks * rows + row)
    global_max = tl.max(maxima, 0)
    safe_max = tl.where(global_max == -float("inf"), 0., global_max)
    scale = tl.where(maxima == -float("inf"), 0., tl.exp(maxima - safe_max))
    total = tl.sum(sums * scale, 0)
    slot = tl.arange(0, C)
    rank_of = slot // K
    flat = rank_of * rows * K + row * K + slot % K
    pooled = _pack(tl.load(Scores + flat), tl.load(Ids + flat))
    ordered = tl.sort(pooled, descending=True)
    scores = _key_score((ordered >> 32).to(tl.int32))
    ids = _LOW - (ordered & _LOW)
    top_k = tl.load(TopK + row)
    top_p = tl.load(TopP + row)
    k_enabled = (top_k > 0) & (top_k < vocab)
    k_index = tl.minimum(tl.maximum(top_k, 1), C) - 1
    kth = tl.sum(tl.where(slot == k_index, scores, 0.), 0)
    in_top_k = (k_enabled == 0) | (scores >= kth)
    exp_scores = tl.where(scores == -float("inf"), 0., tl.exp(scores - safe_max))
    exp_scores = tl.where(in_top_k, exp_scores, 0.)
    normalizer = tl.where(k_enabled, tl.sum(exp_scores, 0), total)
    probabilities = exp_scores / normalizer
    before = tl.cumsum(probabilities, 0) - probabilities
    kept = (before < top_p) & (probabilities > 0)
    smallest = tl.min(tl.where(kept, probabilities, float("inf")), 0)
    last = tl.load(Scores + ranks * rows * K + row * K + (K - 1))
    bound = tl.max(tl.where(last == -float("inf"), 0., tl.exp(last - safe_max)), 0) / normalizer
    complete = (global_max != -float("inf")) & (((k_enabled == 0) | (top_k <= K)) &
                                                 ((k_enabled & (top_k <= K)) | (smallest > bound)))
    # Only the first NUCLEUS IDs are published; a larger nucleus takes the dense path.
    complete = complete & (tl.sum(kept.to(tl.int32), 0) <= NUCLEUS)
    nucleus = tl.where(kept, probabilities, 0.)
    nucleus = nucleus / tl.maximum(tl.sum(nucleus, 0), 1.1754943508222875e-38)
    by_id = tl.sort((tl.where(kept, ids, vocab) << 32) | nucleus.to(tl.int32, bitcast=True).to(tl.int64))
    out_ids = by_id >> 32
    published = slot < NUCLEUS
    tl.store(OutIds + row * NUCLEUS + slot, tl.where(out_ids < vocab, out_ids, -1), mask=published)
    tl.store(OutProbs + row * NUCLEUS + slot, (by_id & _LOW).to(tl.int32).to(tl.float32, bitcast=True),
             mask=published)
    tl.store(Complete + row, complete)


NUCLEUS_SLOTS = 128


def fused_merge_summaries(maximum, exp_sum, scores, token_ids, top_k, top_p, vocab, nucleus=NUCLEUS_SLOTS):
    """merge_summaries in one single-CTA program per row, publishing at most `nucleus` IDs per row.

    Rows whose nucleus holds more IDs are reported incomplete, so the caller verifies them densely.
    """
    world, rows, candidates = scores.shape
    device = scores.device
    out_ids = torch.empty(rows, nucleus, dtype=torch.int64, device=device)
    out_probs = torch.empty(rows, nucleus, device=device)
    complete = torch.empty(rows, dtype=torch.bool, device=device)
    _merge_kernel[(rows,)](maximum.contiguous(), exp_sum.contiguous(), scores.contiguous(), token_ids.contiguous(),
                           top_k, top_p, out_ids, out_probs, complete, rows, vocab, WORLD=world, K=candidates,
                           NUCLEUS=nucleus, num_warps=4)
    return MergedTarget(out_ids, out_probs, complete)


@triton.jit
def _union_kernel(Ids, Probs, Drafts, Target, UnionIds, DraftColumns, Scratch, width, entries,
                  C: tl.constexpr, N: tl.constexpr):
    # One program per request: sort every position's nucleus IDs and the drafts together, number the distinct IDs
    # in ascending order, and write each position's probabilities and each draft's column at that numbering.
    request = tl.program_id(0)
    entry = tl.arange(0, N)
    nucleus = entry < width * C
    draft = (entry >= width * C) & (entry < entries)
    ids = tl.load(Ids + request * width * C + entry, mask=nucleus, other=-1)
    ids = tl.where(draft, tl.load(Drafts + request * (width - 1) + entry - width * C, mask=draft, other=-1), ids)
    ids = tl.where(ids >= 0, ids, 1 << 40)
    ordered = tl.sort((ids << 16) | entry.to(tl.int64))
    tl.store(Scratch + request * N + entry, ordered >> 16)
    # Unused columns: zero target mass, and a valid ID (0) the draft lookup may read; zero target makes them inert.
    tl.store(UnionIds + request * N + entry, tl.zeros([N], tl.int64))
    for position in range(width):
        tl.store(Target + (request * width + position) * N + entry, tl.zeros([N], tl.float32))
    tl.debug_barrier()
    previous = tl.load(Scratch + request * N + entry - 1, mask=entry > 0, other=-1)
    sorted_ids = ordered >> 16
    source = (ordered & 0xFFFF).to(tl.int32)
    valid = sorted_ids < (1 << 40)
    fresh = valid & (sorted_ids != previous)
    column = tl.cumsum(fresh.to(tl.int32), 0) - 1
    tl.store(UnionIds + request * N + column, sorted_ids, mask=fresh)
    from_nucleus = valid & (source < width * C)
    probability = tl.load(Probs + request * width * C + source, mask=from_nucleus, other=0.)
    position = source // C
    tl.store(Target + (request * width + position) * N + column, probability, mask=from_nucleus)
    tl.store(DraftColumns + request * (width - 1) + source - width * C, column,
             mask=valid & (source >= width * C))


def fused_compact_chain_verify(merged, batch, width, draft_ids, draft_token_probs, seed, offset):
    """compact_chain_verify with the compact vocabulary built by one sort-and-number kernel per request."""
    from flashinfer import sampling

    C = merged.token_ids.shape[-1]
    entries = width * C + width - 1
    N = triton.next_power_of_2(entries)
    device = merged.token_ids.device
    target = torch.empty(batch, width, N, device=device)
    union = torch.empty(batch, N, dtype=torch.int64, device=device)
    draft_columns = torch.empty(batch, width - 1, dtype=torch.int32, device=device)
    scratch = torch.empty(batch, N, dtype=torch.int64, device=device)
    _union_kernel[(batch,)](merged.token_ids, merged.probabilities, draft_ids.contiguous(), target, union,
                            draft_columns, scratch, width, entries, C=C, N=N, num_warps=8)
    # Unused columns read token 0's draft probability; with zero target mass they affect neither the acceptance
    # ratio of a draft (whose column is always used) nor the residual max(target - draft, 0).
    draft = draft_token_probs(union[:, None, :].expand(batch, width - 1, N))
    accepted = torch.zeros(batch, dtype=torch.int32, device=device)
    emitted = torch.zeros(batch, dtype=torch.int32, device=device)
    tokens, _, _ = sampling.chain_speculative_sampling(draft.contiguous(), draft_columns, target, accepted, emitted,
        deterministic=True, seed=seed, offset=offset, row_independent_rng=True)
    tokens = torch.where(tokens >= 0, union.gather(-1, tokens.clamp_min(0).long()), -1)
    return tokens, accepted, emitted


def compact_eligible(parameters, top_k, grammar_enabled, logprob_requested, slots, live,
                     candidates=SUMMARY_CANDIDATES):
    """[R] rows whose sampling policy the compact path reproduces exactly.

    ``parameters`` is the request state's [S, 6] temperature, top-p, min-p,
    repetition, presence and frequency columns; ``top_k``, ``grammar_enabled``
    and ``logprob_requested`` are per-slot; ``slots`` maps rows to slots and
    ``live`` marks live rows. Penalties count prior tokens across the full
    vocabulary and grammar masks do too, so either forces the dense path;
    min-p filters relative to the global maximum, which the summaries carry,
    but stays dense until verified; logprob records read full-vocabulary
    logits. Top-k above the published candidates may need unpublished tokens.
    Dead rows are eligible so they never force a fallback.
    """
    safe = slots.long().clamp(0, parameters.shape[0] - 1)
    p = parameters[safe]
    bits = p.view(torch.int32)
    neutral = (p[:, 3] == 1.) & (bits[:, 4] == 0) & (bits[:, 5] == 0) & (bits[:, 2] == 0)
    k = top_k[safe]
    bounded = (k <= 0) | (k <= candidates)
    plain = (grammar_enabled[safe] == 0) & (logprob_requested[safe] == 0)
    return (neutral & bounded & plain) | ~live


def compact_round(eligible, complete, fallbacks=None):
    """Rank-uniform [1] bool: every row is eligible and its gathered nucleus complete.

    ``complete`` comes from gathered summaries, so every rank computes the same
    flag without communication. ``fallbacks``, an int64 [1] debug tensor, counts
    rounds that fall back to the dense path; it accumulates on device only.
    """
    take = (eligible & complete).all().reshape(1)
    if fallbacks is not None:
        fallbacks.add_((~take).long())
    return take
