"""One request's draft positions sampled as one sharded row block equal position-by-position proposals."""

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                              reason="requires an explicitly assigned CUDA device")

VOCAB = 152576
WORLD = 8
LOCAL = VOCAB // WORLD
BLOCK = 8  # drafter block: pending token + 7 draft positions


def _state(slots, *, top_k, top_p, temperature, generator):
    from ignition.components.sampling import RUNNING, DeviceRequestState

    state = DeviceRequestState.allocate("cuda", vocab_size=VOCAB, capacity=slots)
    state.generation.copy_(torch.arange(slots) + 7)
    state.status.fill_(RUNNING)
    counts = (torch.rand((slots, VOCAB), generator=generator) < 1e-3).to(torch.int32)
    state.histogram.copy_(counts)  # committed-history penalties still apply inside a block
    state.parameters.copy_(torch.tensor([temperature, top_p, .02, 1.2, .1, .05]).repeat(slots, 1))
    state.top_k.fill_(top_k)
    state.grammar_enabled.zero_()
    return state


@pytest.mark.parametrize("positions", [2, 4, 7])
@pytest.mark.parametrize("top_k", [-1, 40])
@pytest.mark.parametrize("top_p", [.95, 1.])
@pytest.mark.parametrize("temperature", [1., .6])
@torch.no_grad()
def test_position_block_equals_independent_positions(positions, top_k, top_p, temperature):
    from ignition.components.draft_sampling import PreparedShardedDraftBlock, PreparedShardedDraftDistribution

    generator = torch.Generator().manual_seed(positions * 101 + (top_k > 0) * 7 + int(top_p * 100) + int(temperature * 10))
    capacity = 4
    state = _state(capacity, top_k=top_k, top_p=top_p, temperature=temperature, generator=generator)
    slot = torch.tensor([2], dtype=torch.int32, device="cuda")
    generation = state.generation[slot.long()].clone()
    live = torch.tensor([1], dtype=torch.int32, device="cuda")
    seed = torch.randint(0, 2**62, (1,), generator=generator).cuda()
    base = torch.randint(0, 2**40, (1,), generator=generator).cuda()
    # The drafter's local logits per rank: [request, block, vocabulary/world].
    logits = (torch.randn((1, BLOCK, VOCAB), generator=generator) * 3).to(torch.bfloat16).cuda()
    local = [logits[:, :, rank * LOCAL:(rank + 1) * LOCAL].contiguous() for rank in range(WORLD)]

    blocks = [PreparedShardedDraftBlock(local[rank][0, 1:1 + positions], state, slot, generation, live,
        rank=rank, world=WORLD, runtime=None, session=None, coordinator=None, name=f"block{rank}",
        seed=seed, base_offset=base) for rank in range(WORLD)]
    for block in blocks:
        block.prepare()
        block.expand()
    for block in blocks:
        block.distribution.shard_candidates()
    gathered = torch.cat([block.distribution.shard_payload for block in blocks], 1)
    for block in blocks:
        block.distribution.gathered_payload.copy_(gathered)
        block.distribution.pooled_sample()
    reference = blocks[0].distribution

    prefix = torch.zeros((1, 1), dtype=torch.int64, device="cuda")
    for position in range(positions):
        offset = base + 4 * position
        sampled = torch.zeros(1, dtype=torch.int64, device="cuda")
        shards = [PreparedShardedDraftDistribution(local[rank][:, 1 + position], state, slot, generation, live,
            prefix, rank=rank, world=WORLD, runtime=None, session=None, coordinator=None, name=f"s{rank}",
            seed=seed, offset=offset, sampled=sampled) for rank in range(WORLD)]
        for shard in shards:
            shard.prepare()
            shard.shard_candidates()
        payload = torch.cat([shard.shard_payload for shard in shards], 1)
        for shard in shards:
            shard.gathered_payload.copy_(payload)
            shard.pooled_sample()
        single = shards[0]
        assert bool(single.valid[0, 0]) and bool(reference.valid[position, 0])
        assert int(reference.sampled[position]) == int(single.sampled[0]), position
        assert torch.equal(reference.probabilities[position].view(torch.int32), single.probabilities[0].view(torch.int32))
        assert torch.equal(reference.proposal_selected[position], single.proposal_selected[0])
    for block in blocks:
        assert torch.equal(block.distribution.sampled, reference.sampled)


@torch.no_grad()
def test_position_block_rows_follow_request_liveness():
    from ignition.components.draft_sampling import PreparedShardedDraftBlock

    generator = torch.Generator().manual_seed(5)
    state = _state(4, top_k=-1, top_p=1., temperature=1., generator=generator)
    slot = torch.tensor([1], dtype=torch.int32, device="cuda")
    logits = (torch.randn((1, BLOCK, LOCAL), generator=generator)).to(torch.bfloat16).cuda()
    for live_rows, generation_shift, expected in ((1, 0, True), (0, 0, False), (1, 3, False)):
        generation = state.generation[slot.long()].clone() + generation_shift
        live = torch.tensor([live_rows], dtype=torch.int32, device="cuda")
        block = PreparedShardedDraftBlock(logits[0, 1:6], state, slot, generation, live, rank=0, world=WORLD,
            runtime=None, session=None, coordinator=None, name="b", seed=torch.ones(1, dtype=torch.int64, device="cuda"),
            base_offset=torch.zeros(1, dtype=torch.int64, device="cuda"))
        block.prepare()
        block.expand()
        assert block.offset.tolist() == [0, 4, 8, 12, 16] and int(block.live_rows[0]) == 5 * live_rows
        block.distribution.shard_candidates()
        payload = block.distribution.shard_payload
        block.distribution.gathered_payload.copy_(payload.repeat(1, WORLD))
        block.distribution.pooled_sample()
        assert block.distribution.valid[:, 0].tolist() == [expected] * 5
