# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0
"""consult18 P5: speculative SAMPLING math over vocabulary-sharded logits.

Standard speculative sampling (Leviathan et al. 2023; Chen et al. 2023) at a
temperature: draft token x_i ~ q_i, accepted with probability
min(1, p_i(x_i) / q_i(x_i)); the first rejection at i emits a sample of
norm(max(0, p_i - q_i)); with every draft accepted the bonus is a sample of
p_k. The emitted tokens are distributed exactly as sampling from p.

Under tensor parallelism every rank holds one contiguous vocabulary shard of
the target and draft logits. Nothing is gathered:
- a sample of softmax(s) over the shards is the Gumbel-max over the shards'
  perturbed scores, `argmax_v (s_v + G_v)` with independent standard Gumbel
  noise per entry, reduced across ranks by the same offer gather as the
  greedy tokens (`argmax_tokens`: ties to the lowest vocabulary id);
- the acceptance needs the global log-sum-exp of p_i and q_i and both
  logits at x_i: one more offer gather of per-rank partials;
- the uniform draw of the acceptance test is rank 0's, read from the gather,
  so every rank takes the same decision.
Noise and uniforms come from caller buffers (filled outside CUDA graphs, per
rank independent generators); the functions are capture-safe.
"""
import torch

NEG = float("-inf")


def gumbel(uniform):
    """Standard Gumbel noise from U(0, 1) draws (0 -> -inf: that entry is never chosen)."""
    u = uniform.clamp(max=1.0 - 2.0 ** -24)
    return -torch.log(-torch.log(u))


def sample_tokens(collectives, scores, noise, *, vocabulary_range):
    """One exact sample per row of softmax(scores) over the vocabulary shards:
    `scores` [rows, V/w] this rank's shard (already temperature-scaled),
    `noise` [rows, V/w] U(0, 1). Returns [rows] int64 global ids, the same on
    every rank."""
    perturbed = scores + gumbel(noise)
    if collectives is None or int(getattr(collectives, "world_size", 1)) == 1:
        return perturbed.argmax(dim=-1)
    return collectives.argmax_tokens(perturbed, vocabulary_range=vocabulary_range)


def _exchange_rows(collectives, offer):
    """Every rank's `[rows, c]` FP32 offer, stacked `[rows, world, c]` (the score
    gather; padded to its offer width with -inf)."""
    rows, cols = (int(v) for v in offer.shape)
    if collectives is None or int(getattr(collectives, "world_size", 1)) == 1:
        return offer.reshape(rows, 1, cols)
    width = int(getattr(collectives.transport, "score_offer", None) or cols)
    padded = torch.full((rows, max(width, cols)), NEG, dtype=torch.float32, device=offer.device)
    padded[:, :cols] = offer
    got = collectives.transport.exchange_scores(padded, plan=collectives._plan("selection_exchange", None))
    return torch.stack([g[:, :cols] for g in got], dim=1)


def verify_sample(collectives, target_scores, draft_scores, drafts, uniforms, noise, *,
                  vocabulary_range):
    """The speculative-sampling decision of one round on this rank's shards.

    `target_scores` [k+1, V/w]: temperature-scaled target logits at the k+1
    verified positions; `draft_scores` [k, V/w]: the scaled draft logits each
    draft x_i was sampled from; `drafts` [k] global ids; `uniforms` [k]
    U(0, 1) (only rank 0's are used); `noise` [k+1, V/w] U(0, 1).
    Returns `(accept [k] bool, tokens [k+1] int64)`: row i < k is the residual
    sample emitted if draft i is the first rejection, row k the bonus sample of
    p_k. Identical on every rank."""
    k = int(draft_scores.shape[0])
    lo = int(vocabulary_range[0])
    width = int(target_scores.shape[1])
    sp, sq = target_scores.float(), draft_scores.float()
    lse_p = torch.logsumexp(sp, dim=-1)                                  # [k+1]
    lse_q = torch.logsumexp(sq, dim=-1)                                  # [k]
    local = drafts.long() - lo
    own = (local >= 0) & (local < width)
    index = local.clamp(0, width - 1).unsqueeze(1)
    px = torch.where(own, sp[:k].gather(1, index).squeeze(1), torch.full_like(lse_q, NEG))
    qx = torch.where(own, sq.gather(1, index).squeeze(1), torch.full_like(lse_q, NEG))
    rank = int(getattr(collectives, "rank", 0)) if collectives is not None else 0
    offer = torch.full((k + 1, 5), NEG, dtype=torch.float32, device=sp.device)
    offer[:, 0] = lse_p
    offer[:k, 1] = lse_q
    offer[:k, 2] = px
    offer[:k, 3] = qx
    offer[:k, 4] = uniforms.float() if rank == 0 else 0.0
    got = _exchange_rows(collectives, offer)                             # [k+1, world, 5]
    g_lse_p = torch.logsumexp(got[:, :, 0], dim=1)                       # [k+1]
    g_lse_q = torch.logsumexp(got[:k, :, 1], dim=1)                      # [k]
    g_px = got[:k, :, 2].max(dim=1).values
    g_qx = got[:k, :, 3].max(dim=1).values
    u = got[:k, 0, 4]
    log_ratio = (g_px - g_lse_p[:k]) - (g_qx - g_lse_q)
    accept = torch.log(u) <= log_ratio
    residual = (torch.exp(sp[:k] - g_lse_p[:k].unsqueeze(1))
                - torch.exp(sq - g_lse_q.unsqueeze(1))).clamp_min(0.0)
    scores = torch.cat([torch.log(residual), sp[k:]], dim=0)
    tokens = sample_tokens(collectives, scores, noise, vocabulary_range=vocabulary_range)
    return accept, tokens
