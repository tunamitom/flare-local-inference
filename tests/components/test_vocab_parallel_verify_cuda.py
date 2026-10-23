"""Vocabulary-parallel verification matches dense FlashInfer verification wherever it reports exactness."""
import os

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires FLARE_RUN_CUDA_TESTS=1 and an assigned GPU")

WORLD, LOCAL = 8, 4096
VOCAB = WORLD * LOCAL


def _scores(rows, sharpness, generator):
    base = torch.randn(rows, VOCAB, device="cuda", generator=generator)
    strong = torch.randint(0, VOCAB, (rows, 8), device="cuda", generator=generator)
    boost = torch.linspace(sharpness, sharpness / 4, 8, device="cuda").expand(rows, 8).contiguous()
    return base.scatter_add_(-1, strong, boost)


def _dense(scores, top_k, top_p):
    from flashinfer import sampling

    masked = sampling.top_k_mask_logits(scores.contiguous(), top_k)
    return sampling.top_p_renorm_probs(sampling.softmax(masked), top_p, is_deterministic=True)


def _verify(batch, width, sharpness, seed, top_k_value=0, fused=False):
    from flashinfer import sampling

    from ignition.components import vocab_parallel_verify as vp

    shard_summary, merge_summaries, compact_chain_verify = (
        (vp.fused_shard_summary, vp.fused_merge_summaries, vp.fused_compact_chain_verify) if fused
        else (vp.shard_summary, vp.merge_summaries, vp.compact_chain_verify))

    g = torch.Generator(device="cuda").manual_seed(seed)
    rows = batch * width
    scores = _scores(rows, sharpness, g)
    top_k = torch.full((rows,), top_k_value, dtype=torch.int32, device="cuda")
    top_p = torch.full((rows,), 0.95, device="cuda")
    target = _dense(scores, torch.where(top_k > 0, top_k, VOCAB), top_p).view(batch, width, VOCAB)
    draft_scores = scores.view(batch, width, VOCAB)[:, :-1].reshape(-1, VOCAB)
    draft_scores = draft_scores + 0.7 * torch.randn(draft_scores.shape, device="cuda", generator=g)
    draft = _dense(draft_scores, torch.full((draft_scores.shape[0],), VOCAB, dtype=torch.int32, device="cuda"),
                   top_p[: draft_scores.shape[0]]).view(batch, width - 1, VOCAB)
    drafts = torch.multinomial(draft.view(-1, VOCAB), 1, generator=g).view(batch, width - 1)
    seeds = torch.randint(0, 2**62, (batch,), device="cuda", generator=g)
    offsets = torch.randint(0, 2**20, (batch,), device="cuda", generator=g) * 4
    accepted = torch.zeros(batch, dtype=torch.int32, device="cuda")
    emitted = torch.zeros(batch, dtype=torch.int32, device="cuda")
    dense, _, _ = sampling.chain_speculative_sampling(draft, drafts.int(), target, accepted, emitted,
        deterministic=True, seed=seeds, offset=offsets, row_independent_rng=True)
    parts = [shard_summary(scores[:, r * LOCAL:(r + 1) * LOCAL], r * LOCAL) for r in range(WORLD)]
    gathered = [torch.stack([getattr(p, name) for p in parts]) for name in ("maximum", "exp_sum", "scores", "token_ids")]
    merged = merge_summaries(*gathered, top_k, top_p, VOCAB)
    tokens, _, compact_emitted = compact_chain_verify(merged, batch, width, drafts,
                                                      lambda ids: draft.gather(-1, ids), seeds, offsets)
    complete = merged.complete.view(batch, width).all(-1)
    return complete, (tokens == dense).all(-1) & (compact_emitted == emitted)


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("batch,width", [(1, 5), (4, 2)])
@pytest.mark.parametrize("top_k", [0, 20])
def test_complete_rows_match_dense_verification(batch, width, top_k, fused):
    complete_rows = 0
    for seed in range(20):
        complete, same = _verify(batch, width, 18., seed, top_k, fused)
        assert bool(same[complete].all())
        complete_rows += int(complete.sum())
    assert complete_rows >= 10 * batch


@pytest.mark.parametrize("fused", [False, True])
def test_flat_rows_request_the_dense_fallback(fused):
    complete, _ = _verify(4, 2, 1., 3, fused=fused)
    assert not bool(complete.any())


def test_compact_eligibility_and_round_flag_on_cpu():
    from ignition.components.vocab_parallel_verify import compact_eligible, compact_round

    params = torch.tensor([[1., .95, 0., 1., 0., 0.],      # default sampling: eligible
                           [1., .95, 0., 1.1, 0., 0.],     # repetition penalty
                           [1., .95, 0., 1., -0., 0.],     # negative-zero presence is not neutral
                           [1., .95, .05, 1., 0., 0.],     # min-p
                           [0., 1., 0., 1., 0., 0.],       # greedy: eligible
                           [.7, .9, 0., 1., 0., 0.]])
    top_k = torch.tensor([20, 20, 20, 20, -1, 100])
    grammar = torch.zeros(6, dtype=torch.int32)
    logprobs = torch.zeros(6, dtype=torch.int32)
    slots = torch.arange(6, dtype=torch.int32)
    live = torch.ones(6, dtype=torch.bool)
    eligible = compact_eligible(params, top_k, grammar, logprobs, slots, live)
    assert eligible.tolist() == [True, False, False, False, True, False]
    grammar[0] = 1
    logprobs[4] = 1
    assert compact_eligible(params, top_k, grammar, logprobs, slots, live)[[0, 4]].tolist() == [False, False]
    live[1] = False
    assert compact_eligible(params, top_k, grammar, logprobs, slots, live)[1].item()
    fallbacks = torch.zeros(1, dtype=torch.int64)
    ok = torch.tensor([True, True])
    assert compact_round(ok, ok, fallbacks).item() and fallbacks.item() == 0
    assert not compact_round(ok, torch.tensor([True, False]), fallbacks).item() and fallbacks.item() == 1
