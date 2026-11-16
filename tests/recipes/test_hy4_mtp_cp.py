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


def _spec_body(graphs, kind, start, count, shape):
    entry = graphs._spec_state(kind, count)
    graphs._stage_spec(entry, 0, start, count)
    width = graphs.index_width(start + count)
    body = graphs._body_verify if kind == "verify" else graphs._body_draft
    return body(0, shape, width, count)


@pytest.mark.parametrize("world", (2, 8))
def test_spec_graph_bodies_equal_eager(mtp_worlds, world):
    """consult18 P4: the captured speculative bodies (verify T=4 and MTP draft
    T=3 / T=1 with predicated writes, device-side greedy tokens) against the
    eager forward / draft_absorb / draft_step — bit-equal hidden, the same
    greedy tokens, and a following decode step (reads the verify's KV) and
    draft step (reads the draft KV) bit-equal."""
    from ignition.recipes.hy4.collectives import SimulatedGroup
    from ignition.recipes.hy4.decode_graph import DecodeGraphs

    prompt = 20

    def scenario(graph):
        def body(decoder):
            _, tokens = golden_tokens()
            ids = torch.tensor([tokens], dtype=torch.long)
            _, hidden, _ = decoder.forward(ids[:, :prompt], return_hidden=True)
            decoder.append([0], prompt)
            decoder.draft_absorb(ids[:, 1:prompt], hidden[:, :prompt - 1], logits="none")
            verify = ids[:, prompt:prompt + 4]
            g = DecodeGraphs(decoder, granule=2)
            if graph:
                toks, _ = g.spec_inputs("verify", 4)
                toks.copy_(verify)
                v_greedy, v_hidden, flag = _spec_body(
                    g, "verify", prompt, 4, decoder.default_call_shape((1, 4)))
                assert not bool(flag)
            else:
                logits, vh, _ = decoder.forward(verify, return_hidden=True)
                v_greedy, v_hidden = logits[0].argmax(dim=-1), vh[0]
            decoder.append([0], 4)
            # re-absorb rows prompt-1 .. prompt+1 (target hidden), then one step
            nxt = ids[:, prompt:prompt + 3]
            prev = torch.cat([hidden[:, prompt - 1:prompt], v_hidden[None, :2]], dim=1)
            if graph:
                toks, hid = g.spec_inputs("draft", 3)
                toks.copy_(nxt)
                hid.copy_(prev)
                a_greedy, a_hidden, _ = _spec_body(
                    g, "draft", prompt - 1, 3, decoder.default_call_shape((1, 3)))
                toks1, hid1 = g.spec_inputs("draft", 1)
                toks1.copy_(a_greedy.reshape(1, 1))
                hid1.copy_(a_hidden[-1:].reshape(1, 1, -1))
                s_greedy, s_hidden, _ = _spec_body(
                    g, "draft", prompt + 2, 1, decoder.default_call_shape((1, 1)))
                decoder._draft_positions[0] = prompt + 3
            else:
                decoder._draft_positions[0] = prompt - 1
                dl, dh, _ = decoder.draft_absorb(nxt, prev, logits="last")
                a_greedy, a_hidden = dl.reshape(1, -1).argmax(dim=-1), dh[0]
                sl, sh = decoder.draft_step(a_greedy.reshape(1, 1), dh[:, -1])
                s_greedy, s_hidden = sl.argmax(dim=-1), sh
            step = decoder.decode_step(torch.tensor([[5]], dtype=torch.long))
            dl2, _ = decoder.draft_step(torch.tensor([[7]], dtype=torch.long), s_hidden.reshape(1, -1))
            return [v_greedy.reshape(-1), v_hidden, a_greedy.reshape(-1), a_hidden.reshape(3, -1),
                    s_greedy.reshape(-1), s_hidden.reshape(1, -1), step, dl2]
        return body

    def run(graph):
        from tests.recipes.test_hy4_mtp_cp import _decoders  # noqa: F401

        group = SimulatedGroup(world)
        return group.run([lambda col, d=d: scenario(graph)(d)
                          for d in _decoders(mtp_worlds[world], group)])

    names = ("verify greedy", "verify hidden", "absorb greedy", "absorb hidden",
             "step greedy", "step hidden", "decode after", "draft after")
    for rank, (exact, graphed) in enumerate(zip(run(False), run(True))):
        for name, want, got in zip(names, exact, graphed):
            assert torch.equal(got, want), f"world {world} rank {rank} {name}: {got} vs {want}"
