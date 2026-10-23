"""The one-program speculative commit reproduces the unfused commit exactly."""

import copy
import os

import pytest
import torch

from tests.components.completion_support import completion_outputs

from ignition.components.sampling import DeviceRequestState, EOS_CAPACITY, FINISHED, RUNNING
from ignition.components.speculation import PreparedSpeculativeCommit, SpeculativeAcceptance, SpeculativeRngState

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires explicitly assigned CUDA device and FLARE_RUN_CUDA_TESTS=1")
VOCAB = 1000


def _random_round(generator, batch, width, capacity, *, rng, lag, limits):
    def randint(low, high, shape):
        return torch.randint(low, high, shape, generator=generator, device="cuda")
    state = DeviceRequestState.allocate("cuda", vocab_size=VOCAB, capacity=capacity)
    state.generation.copy_(randint(6, 9, (capacity,)))
    state.status.copy_(torch.where(randint(0, 10, (capacity,)) < 8, RUNNING, FINISHED).int())
    state.error_code.copy_(randint(0, 3, (capacity,)).int())
    state.max_new_tokens.copy_(randint(0, 40, (capacity,)))
    state.generated_tokens.copy_(randint(0, 40, (capacity,)))
    state.consumed_tokens.copy_(randint(0, 5000, (capacity,)))
    state.rng_counter.copy_(randint(0, 2**40, (capacity,)).to(torch.uint64))
    state.histogram.copy_(randint(0, 3, (capacity, VOCAB)).int())
    state.eos_token_ids.copy_(randint(0, 20, (capacity, EOS_CAPACITY)))
    state.eos_count.copy_(randint(0, EOS_CAPACITY + 1, (capacity,)).int())
    slots = randint(-1, capacity + 1, (batch,)).int()
    if batch > 1 and bool(randint(0, 2, ()).item()):
        slots[1] = slots[0]
    generations = torch.where(randint(0, 10, (batch,)) < 8,
                              state.generation[slots.long().clamp(0, capacity - 1)], 99)
    live = torch.tensor([int(randint(0, batch + 2, ()).item()) - (1 if batch == 1 else 0)], device="cuda",
                        dtype=torch.int32)
    inputs = randint(0, 20 if bool(randint(0, 2, ()).item()) else VOCAB, (batch, width))
    if bool(randint(0, 5, ()).item() == 0):
        inputs[0, -1] = VOCAB + 3
    state.pending_token[slots.long().clamp(0, capacity - 1)] = torch.where(
        randint(0, 10, (batch,)) < 9, inputs[:, 0], -1)
    acceptance = SpeculativeAcceptance.allocate(batch, width, "cuda")
    acceptance.token_ids.copy_(randint(0, 20, (batch, width)))
    acceptance.token_ids[:, :-1] = torch.where(randint(0, 10, (batch, width - 1)) < 9, inputs[:, 1:],
                                               acceptance.token_ids[:, :-1])
    proposed = randint(0, width + 2, (batch,))
    acceptance.proposed_count.copy_(proposed.int())
    acceptance.accepted_drafts.copy_((proposed - torch.where(randint(0, 10, (batch,)) < 9, 1, 0)).int())
    acceptance.offered_drafts.copy_(randint(0, width + 1, (batch,)).int())
    acceptance.valid.copy_(randint(0, 10, (batch,)) < 9)
    acceptance.rng_increment.copy_(randint(-1, 5, (batch,)) * (1 if rng else 0))
    rng_state = None
    if rng:
        rng_state = SpeculativeRngState.allocate("cuda", capacity=capacity)
        rng_state.generation.copy_(randint(6, 9, (capacity,)))
        rng_state.target_offset.copy_(randint(0, 10**6, (capacity,)))
    if bool(randint(0, 2, ()).item()):
        # A clean round: distinct live slots whose drafts form a valid chain, so tokens commit.
        slots = torch.randperm(capacity, generator=generator, device="cuda")[:batch].int()
        generations = state.generation[slots.long()].clone()
        live.fill_(batch)
        state.status[slots.long()] = RUNNING
        state.max_new_tokens[slots.long()] = state.generated_tokens[slots.long()] + randint(1, 12, (batch,))
        inputs = randint(0, VOCAB, (batch, width))
        state.pending_token[slots.long()] = inputs[:, 0]
        acceptance.token_ids[:, :-1] = inputs[:, 1:]
        proposed = randint(1, width + 1, (batch,))
        acceptance.proposed_count.copy_(proposed.int())
        acceptance.accepted_drafts.copy_((proposed - 1).int())
        acceptance.offered_drafts.copy_((proposed - 1).int())
        acceptance.valid.fill_(True)
        acceptance.rng_increment.copy_(randint(0, 5, (batch,)) * (1 if rng else 0))
    records = torch.full((2, capacity, 24), -99, device="cuda", dtype=torch.int64)
    frame = torch.tensor([int(randint(0, 2, ()).item())], device="cuda", dtype=torch.int32)
    sequence = torch.tensor([int(randint(0, 100, ()).item())], device="cuda")
    logits = torch.zeros((batch, width, VOCAB), device="cuda")
    token_limits = randint(0, width + 1, (batch,)).int() if limits else None
    copy_mask = randint(0, 2**width, (batch,))
    return dict(state=state, acceptance=acceptance, inputs=inputs, slots=slots, generations=generations,
                live=live, records=records, frame=frame, sequence=sequence, logits=logits,
                rng=rng_state, lag=lag, limits=token_limits, copy_mask=copy_mask)


def _commit(round_, fused):
    r = copy.deepcopy(round_)
    commit = PreparedSpeculativeCommit(r["state"], r["acceptance"], r["inputs"], r["slots"], r["generations"],
        r["live"], completion_outputs(r["records"]), r["frame"], r["sequence"], logits=r["logits"], rng=r["rng"],
        copy_offered_mask=r["copy_mask"], draft_context_lag=r["lag"], token_limits=r["limits"])
    if not fused:
        commit.fused = -1  # keep prepare() from binding the fused program
    commit.prepare()
    if not fused:
        commit.fused = None
    assert (commit.fused is not None) == fused
    commit()
    torch.cuda.synchronize()
    return commit, r


def _compare(reference, fused):
    (a, ra), (b, rb) = reference, fused
    for name in ra["state"].__dataclass_fields__:
        assert torch.equal(getattr(ra["state"], name), getattr(rb["state"], name)), name
    if ra["rng"] is not None:
        assert torch.equal(ra["rng"].target_offset, rb["rng"].target_offset)
        assert torch.equal(ra["rng"].generation, rb["rng"].generation)
    assert torch.equal(ra["records"], rb["records"])
    for name in ("emitted_ids", "committed_input_ids", "emitted_count", "accepted_count", "accepted_drafts",
                 "commit_column", "consumed_after", "_records"):
        assert torch.equal(getattr(a, name), getattr(b, name)), name
    # owners[capacity] collects unowned rows in an unspecified scatter order; only owned entries are read.
    assert torch.equal(a.owners[:-1], b.owners[:-1])


@pytest.mark.parametrize("batch,width", [(1, 1), (1, 4), (1, 8), (2, 3), (4, 2), (8, 1), (8, 5), (64, 2)])
@pytest.mark.parametrize("rng,lag,limits", [(True, 0, True), (False, None, False), (True, 1, False)])
def test_fused_commit_matches_unfused(batch, width, rng, lag, limits):
    generator = torch.Generator(device="cuda").manual_seed(batch * 131 + width * 17 + int(rng) + 3 * int(limits))
    emitted = errors = 0
    for capacity in sorted({max(4, batch), 64}):
        for _ in range(40):
            round_ = _random_round(generator, batch, width, capacity, rng=rng, lag=lag, limits=limits)
            reference = _commit(round_, False)
            _compare(reference, _commit(round_, True))
            emitted += int(reference[0].emitted_count.sum())
            errors += int((reference[1]["state"].error_code != round_["state"].error_code).sum())
    # The random rounds exercise both committed tokens and rejected rows.
    assert emitted > 0 and errors > 0
