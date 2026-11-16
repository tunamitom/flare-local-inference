# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0
"""consult18 P4: the MTP draft block under context + tensor parallelism.

The CP draft path (`Hy4Decoder._draft_cp`: vocabulary-sharded embedding, the
CP attention of layer `config.layers` with owned record / index-key writes and
the selection exchange, the sharded MoE, all-reduces) against the world-1
`MtpBlock.forward` reference on the tiny checkpoint: prompt absorption plus
three chained draft steps, worlds 2 and 8, wired to the real collectives over
the simulated DCP transport. Same tolerance as the CP gate a (1e-3 relative),
and the greedy chain's drafted tokens must agree.
"""
import pytest

torch = pytest.importorskip("torch")

from tests.recipes.test_hy4_w6_cp import (  # noqa: E402,F401
    TINY, golden_tokens, rel_error, tiny_available,
)

CAPACITY = 40
N_CHAIN = 3


@pytest.fixture(scope="module")
def mtp_worlds(tiny_available):
    from ignition.recipes.hy4.bootstrap import PreparationConfig, construct_rank
    from ignition.recipes.hy4.collectives import SimulatedGroup
    from tests.recipes.test_hy4_reference_gates import to_fp32_weights

    settings = PreparationConfig(devices=(0,), mtp=True, prefill_chunk_size=64,
                                 speculative_widths=(1, 2, 3))
    built = {}
    for world in (1, 2, 8):
        placeholder = SimulatedGroup(world)
        ranks = []
        for rank in range(world):
            constructed = construct_rank(TINY, settings, rank=rank, pinned=False, mtp=True,
                                         cache_rows=2, cache_capacity=CAPACITY, world_size=world,
                                         collectives=placeholder.collectives(rank))
            constructed.weights = to_fp32_weights(constructed.weights)
            ranks.append(constructed)
        built[world] = ranks
    return built


def _decoders(ranks, group):
    from ignition.recipes.hy4.collectives import PcieDcpCollectives, SimulatedDcpTransport
    from ignition.recipes.hy4.decoder import Hy4Cache, Hy4Decoder

    world = len(ranks)
    transport = SimulatedDcpTransport(group) if group is not None else None
    out = []
    for index, rank in enumerate(ranks):
        cache = Hy4Cache.allocate(rank.model_config, rows=2, capacity=CAPACITY,
                                  device=torch.device("cpu"), draft=True, dtype=torch.float32,
                                  world=world, rank=index)
        collectives = (None if group is None else
                       PcieDcpCollectives(rank=index, world_size=world, transport=transport))
        out.append(Hy4Decoder(rank.weights, cache, compute_dtype=torch.float32,
                              collectives=collectives))
    return out


def _scenario(decoder):
    _, tokens = golden_tokens()
    ids = torch.tensor([tokens], dtype=torch.long)
    logits, hidden, _ = decoder.forward(ids, return_hidden=True)
    decoder.complete([0], ids)
    n_abs = ids.shape[1] - 1
    absorb_logits, absorb_hidden, _ = decoder.draft_absorb(ids[:, 1:], hidden[:, :n_abs])
    chain_logits, chain_tokens = [], []
    d_in = int(logits[0, -1].argmax())
    prev = hidden[:, n_abs:n_abs + 1]
    for _ in range(N_CHAIN):
        step_logits, step_hidden = decoder.draft_step(torch.tensor([[d_in]], dtype=torch.long), prev)
        chain_logits.append(step_logits)
        d_in = int(step_logits[0].argmax())
        chain_tokens.append(d_in)
        prev = step_hidden.unsqueeze(1)
    return absorb_logits, absorb_hidden, torch.stack(chain_logits), chain_tokens


@pytest.mark.parametrize("world", (2, 8))
def test_cp_draft_matches_world_one(mtp_worlds, world):
    from ignition.recipes.hy4.collectives import SimulatedGroup

    ref = _scenario(_decoders(mtp_worlds[1], None)[0])
    group = SimulatedGroup(world)
    results = group.run([lambda col, d=d: _scenario(d) for d in _decoders(mtp_worlds[world], group)])
    for rank, got in enumerate(results):
        for name, want, have in (("absorb logits", ref[0], got[0]),
                                 ("absorb hidden", ref[1], got[1]),
                                 ("chain logits", ref[2], got[2])):
            assert rel_error(have, want) < 1e-3, f"world {world} rank {rank} {name}: {rel_error(have, want)}"
        assert got[3] == ref[3], f"world {world} rank {rank}: chain {got[3]} != {ref[3]}"
