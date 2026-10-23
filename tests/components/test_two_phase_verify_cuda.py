"""Two-phase verification's compact continuation reproduces the dense verifier wherever it takes the round."""

import os

import pytest
import torch

from ignition.components.sampling import DeviceRequestState, RequestAdmission
from ignition.components.speculation import (
    PreparedSpeculativeDistributions, PreparedStochasticVerification, SpeculativeRngState,
)
from ignition.components.two_phase_verify import PreparedTwoPhaseVerification

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires FLARE_RUN_CUDA_TESTS=1 and an assigned GPU")

VOCAB = 19072 * 2


def _round(batch, width, *, sharpness, seed, top_k=0, presence=0.):
    g = torch.Generator(device="cuda").manual_seed(seed)
    state = DeviceRequestState.allocate("cuda", vocab_size=VOCAB, capacity=batch)
    for slot in range(batch):
        state.admit(slot, RequestAdmission(generation=3, seed=11 + seed * 97 + slot, max_new_tokens=100,
            temperature=.8, top_k=top_k, top_p=.95, presence_penalty=presence, eos_token_ids=()), [1])
    slots = torch.arange(batch, dtype=torch.int32, device="cuda")
    generations = torch.full((batch,), 3, dtype=torch.int64, device="cuda")
    live = torch.tensor([batch], dtype=torch.int32, device="cuda")
    logits = torch.randn(batch, width, VOCAB, device="cuda", generator=g)
    strong = torch.randint(0, VOCAB, (batch, width, 6), device="cuda", generator=g)
    logits.scatter_add_(-1, strong, torch.linspace(sharpness, sharpness / 3, 6, device="cuda").expand_as(strong).contiguous())
    logits = logits.bfloat16()
    draft_probs = torch.softmax(logits[:, :-1].float() + .5 * torch.randn(batch, width - 1, VOCAB, device="cuda",
                                generator=g), -1).contiguous()
    drafts = torch.multinomial(draft_probs.view(-1, VOCAB), 1, generator=g).view(batch, width - 1)
    valid = torch.ones(batch, width - 1, dtype=torch.bool, device="cuda")
    limits = torch.full((batch,), width, dtype=torch.int32, device="cuda")
    target = PreparedSpeculativeDistributions(logits, state, slots, generations, live, drafts)
    target.prepare()

    def verifier():
        rng = SpeculativeRngState.allocate("cuda", capacity=batch)
        verify = PreparedStochasticVerification(draft_probs, drafts, target.probabilities, state, slots, generations,
            live, rng=rng, draft_valid=valid, target_valid=target.valid, token_limits=limits)
        verify.prepare()
        return verify, rng

    dense, _ = verifier()
    target()
    dense()
    shadow, rng = verifier()
    flag = torch.zeros(64, dtype=torch.uint8, device="cuda")
    phases = PreparedTwoPhaseVerification(logits.view(batch * width, VOCAB), state, slots, generations, live, drafts,
        draft_probs, valid, limits, verifier=shadow, rng=rng, logprob_requested=torch.zeros(batch, dtype=torch.int32,
        device="cuda"), flag=flag, rank=0, world=1, runtime=None, session=None, coordinator=None, name="test")
    phases.prepare()
    take = bool(phases.decide().item())
    assert int(flag[0]) == int(take)
    phases.compact()
    return take, dense.output, shadow.output


@pytest.mark.parametrize("batch,width", [(1, 5), (4, 2), (2, 8)])
@pytest.mark.parametrize("top_k", [0, 20])
def test_taken_rounds_match_dense_verification(batch, width, top_k):
    taken = 0
    for seed in range(12):
        take, dense, compact = _round(batch, width, sharpness=22., seed=seed, top_k=top_k)
        if take:
            taken += 1
            for name in ("token_ids", "accepted_drafts", "proposed_count", "offered_drafts", "valid", "rng_increment"):
                assert torch.equal(getattr(dense, name), getattr(compact, name)), name
    assert taken >= 6


def test_flat_rounds_and_penalized_requests_fall_back():
    assert not _round(4, 2, sharpness=1., seed=1)[0]
    assert not _round(1, 5, sharpness=22., seed=2, presence=.3)[0]


def test_captured_phases_replay_on_new_logits():
    batch, width = 2, 4
    take, _, _ = _round(batch, width, sharpness=22., seed=5)
    assert take
    g = torch.Generator(device="cuda").manual_seed(41)
    state = DeviceRequestState.allocate("cuda", vocab_size=VOCAB, capacity=batch)
    for slot in range(batch):
        state.admit(slot, RequestAdmission(generation=3, seed=5 + slot, max_new_tokens=100, temperature=.8,
                                           top_k=0, top_p=.95, eos_token_ids=()), [1])
    slots = torch.arange(batch, dtype=torch.int32, device="cuda")
    generations = torch.full((batch,), 3, dtype=torch.int64, device="cuda")
    live = torch.tensor([batch], dtype=torch.int32, device="cuda")
    logits = torch.zeros(batch * width, VOCAB, dtype=torch.bfloat16, device="cuda")
    drafts = torch.zeros(batch, width - 1, dtype=torch.int64, device="cuda")
    draft_probs = torch.zeros(batch, width - 1, VOCAB, device="cuda")
    valid = torch.ones(batch, width - 1, dtype=torch.bool, device="cuda")
    limits = torch.full((batch,), width, dtype=torch.int32, device="cuda")

    def fill(seed):
        g.manual_seed(seed)
        base = torch.randn(batch * width, VOCAB, device="cuda", generator=g)
        strong = torch.randint(0, VOCAB, (batch * width, 6), device="cuda", generator=g)
        base.scatter_add_(-1, strong, torch.linspace(22., 7., 6, device="cuda").expand_as(strong).contiguous())
        logits.copy_(base.bfloat16())
        draft_probs.copy_(torch.softmax(logits.view(batch, width, VOCAB)[:, :-1].float(), -1))
        drafts.copy_(torch.multinomial(draft_probs.view(-1, VOCAB), 1, generator=g).view(batch, width - 1))

    fill(1)
    target = PreparedSpeculativeDistributions(logits.view(batch, width, VOCAB), state, slots, generations, live, drafts)
    target.prepare()
    rng = SpeculativeRngState.allocate("cuda", capacity=batch)
    verify = PreparedStochasticVerification(draft_probs, drafts, target.probabilities, state, slots, generations, live,
        rng=rng, draft_valid=valid, target_valid=target.valid, token_limits=limits)
    verify.prepare()
    flag = torch.zeros(64, dtype=torch.uint8, device="cuda")
    phases = PreparedTwoPhaseVerification(logits, state, slots, generations, live, drafts, draft_probs, valid, limits,
        verifier=verify, rng=rng, logprob_requested=torch.zeros(batch, dtype=torch.int32, device="cuda"), flag=flag,
        rank=0, world=1, runtime=None, session=None, coordinator=None, name="captured")
    phases.prepare()
    stream = torch.cuda.Stream()
    graphs = []
    for phase in (phases.decide, phases.compact):
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            phase()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            phase()
        graphs.append(graph)
    for seed in (2, 3, 4):
        fill(seed)
        graphs[0].replay()
        torch.cuda.synchronize()
        replayed_flag = int(flag[0])
        graphs[1].replay()
        torch.cuda.synchronize()
        replayed = verify.output.token_ids.clone(), verify.output.accepted_drafts.clone()
        eager_flag = int(phases.decide().item())
        phases.compact()
        assert replayed_flag == eager_flag
        assert torch.equal(replayed[0], verify.output.token_ids) and torch.equal(replayed[1], verify.output.accepted_drafts)
