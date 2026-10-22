"""Vocabulary-sharded draft proposals equal proposals drawn from gathered full-vocabulary logits."""

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                              reason="requires an explicitly assigned CUDA device")

VOCAB = 152576
WORLD = 8
LOCAL = VOCAB // WORLD
DRAFTS = 7


def _state(rows, *, top_k, top_p, temperature, generator):
    from ignition.components.sampling import RUNNING, DeviceRequestState

    state = DeviceRequestState.allocate("cuda", vocab_size=VOCAB, capacity=rows)
    state.generation.copy_(torch.arange(rows) + 7)
    state.status.fill_(RUNNING)
    # Sparse penalty counts on the first half of the slots, neutral penalties on the rest.
    counts = (torch.rand((rows, VOCAB), generator=generator) < 1e-3).to(torch.int32)
    counts[rows // 2:] = 0
    state.histogram.copy_(counts)
    parameters = torch.tensor([temperature, top_p, 0., 1., 0., 0.]).repeat(rows, 1)
    parameters[:rows // 2, 3:] = torch.tensor([1.3, .2, .05])
    parameters[1::4, 2] = .05
    state.parameters.copy_(parameters)
    if top_k:
        choices = torch.tensor([1, 5, 40, 64, 200, 0])
        state.top_k.copy_(choices[torch.arange(rows) % len(choices)].to(torch.int32))
    else:
        state.top_k.fill_(-1)
    masks = torch.randint(-2**31, 2**31 - 1, state.grammar_mask.shape, generator=generator, dtype=torch.int64)
    state.grammar_mask.copy_(masks.to(torch.int32))
    state.grammar_enabled.copy_((torch.arange(rows) % 5 == 2).to(torch.int32))
    return state


def _logits(rows, generator):
    logits = torch.randn((rows, VOCAB), generator=generator) * 3
    top = logits.max() + 4
    edges = [shard * LOCAL + offset for shard in range(1, WORLD) for offset in (-2, -1, 0, 1)]
    # Exact maxima duplicated across shard boundaries, from a few to more than the candidate bound.
    for row in range(0, rows, 3):
        logits[row, edges[:4 + row % 29]] = top
    # More cutoff ties than candidate slots, behind a few strictly higher scores in later shards.
    for row in range(1, rows, 3):
        logits[row, torch.arange(70) * 1999 + row] = top
        logits[row, [VOCAB - 1, 5 * LOCAL + 3]] = top + 1
    logits[2::3, :] = logits[2::3, :].round()  # many equal scores everywhere
    return logits.to(torch.bfloat16).cuda()


def _invalidate(rows, slots, generations, drafts, logits):
    # A stale generation, a duplicate slot, an invalid draft prefix, a non-finite maximum, a fully
    # masked row and a row past live_rows.
    generations[1] = 99
    slots[3], generations[3] = slots[4], generations[4]
    drafts[5, 0] = -1
    logits[6, 4 * LOCAL + 7] = float("inf")
    if rows > 8:
        logits[9, 11] = float("nan")
        logits[10] = -float("inf")
        logits[12, 3 * LOCAL] = float("nan")
        logits[12, 9] = float("inf")
    return rows - 1


def _bits(tensor):
    return tensor.contiguous().view(torch.int32 if tensor.element_size() == 4 else torch.uint8)


def _sharded(logits, state, slots, generations, live, drafts, prefix, seed, offset, sampled):
    from ignition.components.draft_sampling import PreparedShardedDraftDistribution

    shards = [PreparedShardedDraftDistribution(logits[:, rank * LOCAL:(rank + 1) * LOCAL], state, slots,
        generations, live, drafts, rank=rank, world=WORLD, runtime=None, session=None, coordinator=None,
        name=f"shard{rank}", seed=seed, offset=offset, sampled=sampled, prefix_offset=prefix)
        for rank in range(WORLD)]
    for shard in shards:
        shard.prepare()

    def run():
        for shard in shards:
            shard.shard_candidates()
        # The rank-ordered all-gather: each output row holds rank zero's payload, then rank one's, ...
        gathered = torch.cat([shard.shard_payload for shard in shards], 1)
        for shard in shards:
            shard.gathered_payload.copy_(gathered)
            shard.pooled_sample()
        return shards
    return run


@pytest.mark.parametrize("rows", [8, 32, 64])
@pytest.mark.parametrize("top_k", [False, True])
@pytest.mark.parametrize("top_p", [.95, 1.])
@pytest.mark.parametrize("temperature", [1., .7])
@pytest.mark.parametrize("prefix", [0, 3])
@torch.no_grad()
def test_sharded_proposals_match_gathered_logits(rows, top_k, top_p, temperature, prefix):
    from ignition.components.draft_sampling import PreparedDraftDistribution

    generator = torch.Generator().manual_seed(rows * 131 + top_k * 17 + int(top_p * 100) + int(temperature * 10)
                                              + prefix)
    state = _state(rows, top_k=top_k, top_p=top_p, temperature=temperature, generator=generator)
    slots = torch.randperm(rows, generator=generator).to(torch.int32).cuda()
    generations = state.generation[slots.long()].clone()
    drafts = torch.randint(0, VOCAB, (rows, DRAFTS), generator=generator).cuda()
    logits = _logits(rows, generator)
    # Earlier drafts hit tied maxima, so prefix penalties break some of the injected ties.
    drafts[0, :2] = torch.tensor([LOCAL - 1, 2 * LOCAL])
    live_rows = _invalidate(rows, slots, generations, drafts, logits)
    live = torch.tensor([live_rows], dtype=torch.int32, device="cuda")
    seed = torch.randint(0, 2**62, (rows,), generator=generator).cuda()
    offset = torch.randint(0, 2**40, (rows,), generator=generator).cuda()

    reference_sampled = torch.zeros(rows, dtype=torch.int64, device="cuda")
    reference = PreparedDraftDistribution(logits.float().view(rows, 1, VOCAB), state, slots, generations, live,
        drafts, seed=seed, offset=offset, sampled=reference_sampled, prefix_offset=prefix)
    reference.prepare()
    sampled = torch.full((rows,), -5, dtype=torch.int64, device="cuda")
    run = _sharded(logits, state, slots, generations, live, drafts, prefix, seed, offset, sampled)
    for _ in range(2):
        reference()
        shards = run()
        torch.cuda.synchronize()
        for shard in shards:
            assert torch.equal(shard.valid, reference.valid)
            assert torch.equal(_bits(shard.probabilities), _bits(reference.probabilities))
            assert torch.equal(_bits(shard.maximum), _bits(reference.maximum))
            # Support keys agree as sets; pool order differs only by the block decomposition.
            assert torch.equal(shard.proposal_selected.sort(1).values, reference.proposal_selected.sort(1).values)
        assert torch.equal(sampled, reference_sampled)
        offset.add_(4)
    valid = reference.valid[:, 0]
    assert not valid[[1, 3, 4, rows - 1] + ([5] if prefix else [])].any()
    assert valid.sum() >= rows - 10


@torch.no_grad()
def test_single_shard_graph_replay_matches_gathered_logits():
    from ignition.components.draft_sampling import PreparedDraftDistribution, PreparedShardedDraftDistribution
    from ignition.preparation.compilation import compile_forward

    rows = 8
    generator = torch.Generator().manual_seed(5)
    state = _state(rows, top_k=True, top_p=.95, temperature=.7, generator=generator)
    slots = torch.arange(rows, dtype=torch.int32, device="cuda")
    generations = state.generation[slots.long()].clone()
    drafts = torch.randint(0, VOCAB, (rows, DRAFTS), generator=generator).cuda()
    live = torch.tensor([rows], dtype=torch.int32, device="cuda")
    # Logits of every draft position; each proposal reads one position through a strided view.
    block = 4
    positions = torch.randn((rows, block, VOCAB), generator=generator).mul(3).to(torch.bfloat16).cuda()
    seed = torch.arange(rows, dtype=torch.int64, device="cuda") + 3
    offset = torch.zeros(rows, dtype=torch.int64, device="cuda")
    sampled = torch.zeros(rows, dtype=torch.int64, device="cuda")
    operation = PreparedShardedDraftDistribution(positions[:, 2], state, slots, generations, live, drafts,
        rank=0, world=1, runtime=None, session=None, coordinator=None, name="proposal", seed=seed, offset=offset,
        sampled=sampled, prefix_offset=2)
    operation.prepare()
    compiled = compile_forward(operation)
    compiled()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled()
    reference_sampled = torch.zeros_like(sampled)
    reference = PreparedDraftDistribution(positions[:, 2:3].float().contiguous(), state, slots, generations, live,
        drafts, seed=seed, offset=offset, sampled=reference_sampled, prefix_offset=2)
    reference.prepare()
    for step in range(3):
        positions.copy_(torch.randn(positions.shape, generator=generator).mul(3).to(torch.bfloat16))
        offset.fill_(4 * step)
        operation.probabilities.fill_(float("nan"))
        graph.replay()
        reference.logits.copy_(positions[:, 2:3].float())
        reference()
        torch.cuda.synchronize()
        assert torch.equal(operation.valid, reference.valid) and operation.valid.all()
        assert torch.equal(_bits(operation.probabilities), _bits(reference.probabilities))
        assert torch.equal(sampled, reference_sampled)
    graph.reset()
