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


@pytest.mark.parametrize("world", (2, 8))
def test_spec_decode_rows_match_prefill_family(mtp_worlds, world):
    """consult18 P4: the verify (T=4) and the draft re-absorption (T=3) as the
    decode-family verification shape (T query rows of one request row, the
    owner-side absorbed query wiring) against the same forwards as one
    prefill-family row — the same greedy tokens and FP32 hidden within the
    query-wiring rounding, and the following decode step reads equal KV."""
    from ignition.recipes.hy4.collectives import SimulatedGroup
    from ignition.recipes.hy4.decode_graph import DecodeGraphs
    from ignition.recipes.hy4.wiring import verify_shape

    prompt = 20

    def scenario(decode_rows):
        def body(decoder):
            _, tokens = golden_tokens()
            ids = torch.tensor([tokens], dtype=torch.long)
            _, hidden, _ = decoder.forward(ids[:, :prompt], return_hidden=True)
            decoder.append([0], prompt)
            decoder.draft_absorb(ids[:, 1:prompt], hidden[:, :prompt - 1], logits="none")
            g = DecodeGraphs(decoder, granule=2)

            def shape(count):
                return verify_shape(count) if decode_rows else decoder.default_call_shape((1, count))

            toks, _ = g.spec_inputs("verify", 4)
            toks.copy_(ids[:, prompt:prompt + 4])
            v_greedy, v_hidden, _ = _spec_body(g, "verify", prompt, 4, shape(4))
            v_greedy, v_hidden = v_greedy.clone(), v_hidden.clone()
            decoder.append([0], 4)
            toks, hid = g.spec_inputs("draft", 3)
            toks.copy_(ids[:, prompt:prompt + 3])
            hid.copy_(torch.cat([hidden[:, prompt - 1:prompt], v_hidden[None, :2]], dim=1))
            a_greedy, a_hidden, _ = _spec_body(g, "draft", prompt - 1, 3, shape(3))
            step = decoder.decode_step(torch.tensor([[5]], dtype=torch.long))
            return [v_greedy.reshape(-1), v_hidden, a_greedy.reshape(-1), a_hidden.clone(), step]
        return body

    def run(decode_rows):
        group = SimulatedGroup(world)
        return group.run([lambda col, d=d: scenario(decode_rows)(d)
                          for d in _decoders(mtp_worlds[world], group)])

    for rank, (want, got) in enumerate(zip(run(False), run(True))):
        assert torch.equal(got[0], want[0]), f"world {world} rank {rank}: verify greedy"
        assert torch.equal(got[2], want[2]), f"world {world} rank {rank}: absorb greedy"
        for name, index in (("verify hidden", 1), ("absorb hidden", 3), ("decode after", 4)):
            assert rel_error(got[index], want[index]) < 1e-4, \
                f"world {world} rank {rank} {name}: {rel_error(got[index], want[index])}"


@pytest.mark.parametrize("world", (2, 8))
def test_spec_sample_bodies_match_reference_math(mtp_worlds, world):
    """consult18 P5: the sampling verify / draft bodies (temperature 0.7, fixed
    noise and uniforms) emit what `verify_sample` / `sample_tokens` give on the
    eager forward's own logits shard and the draft's scores."""
    from ignition.recipes.hy4.collectives import SimulatedGroup
    from ignition.recipes.hy4.decode_graph import DecodeGraphs
    from ignition.recipes.hy4.spec_sample import sample_tokens, verify_sample

    prompt, temp = 20, 0.7

    def body(decoder):
        _, tokens = golden_tokens()
        ids = torch.tensor([tokens], dtype=torch.long)
        _, hidden, _ = decoder.forward(ids[:, :prompt], return_hidden=True)
        decoder.append([0], prompt)
        decoder.draft_absorb(ids[:, 1:prompt], hidden[:, :prompt - 1], logits="none")
        g = DecodeGraphs(decoder, granule=2)
        g.spec_inv_temp().fill_(1.0 / temp)
        rank = decoder.collectives.rank
        gen = torch.Generator().manual_seed(11 + rank)
        vr = decoder.weights.sharding.vocabulary_range
        # draft T=1 in sampling mode vs the reference on the same scores
        entry = g.spec_sample_state("draft", 1)
        entry["noise"].uniform_(generator=gen)
        toks, hid = g.spec_inputs("draft", 1)
        toks.copy_(ids[:, prompt:prompt + 1])
        hid.copy_(hidden[:, prompt - 1:prompt])
        en = g._spec_state("draft", 1)
        g._stage_spec(en, 0, prompt - 1, 1)
        d_tok, d_hid, _ = g._body_draft(0, decoder.default_call_shape((1, 1)),
                                        g.index_width(prompt), 1, True)
        want_scores = (d_hid[-1:].float() @ decoder.weights.head.T) / temp
        want_tok = sample_tokens(decoder.collectives, want_scores, entry["noise"], vocabulary_range=vr)
        draft = (d_tok.clone(), want_tok, entry["out_q"].clone(), want_scores)
        # verify T=3 in sampling mode vs verify_sample on the eager forward's shard
        ventry = g.spec_sample_state("verify", 3)
        ventry["noise"].uniform_(generator=gen)
        ventry["uniform"].uniform_(generator=gen)
        ventry["q_in"].copy_(torch.randn(ventry["q_in"].shape, generator=gen))
        vt, _ = g.spec_inputs("verify", 3)
        vt.copy_(ids[:, prompt:prompt + 3])
        g._stage_spec(ventry, 0, prompt, 3)
        v_tok, _, _ = g._body_verify(0, decoder.default_call_shape((1, 3)), g.index_width(prompt + 3),
                                     3, True)
        got = (v_tok.clone(), ventry["out_accept"].clone())
        full = decoder.forward(ids[:, prompt:prompt + 3], logits_rows="all")
        lo, hi = vr
        want = verify_sample(decoder.collectives, full[0, :, lo:hi].float() / temp, ventry["q_in"],
                             vt[0, 1:], ventry["uniform"], ventry["noise"], vocabulary_range=vr)
        return draft, got, want

    group = SimulatedGroup(world)
    results = group.run([lambda col, d=d: body(d) for d in _decoders(mtp_worlds[world], group)])
    for rank, (draft, got, want) in enumerate(results):
        assert torch.equal(draft[0], draft[1]), f"rank {rank}: draft sample"
        assert torch.allclose(draft[2], draft[3]), f"rank {rank}: draft scores"
        assert torch.equal(got[0], want[1]), f"rank {rank}: verify tokens {got[0]} vs {want[1]}"
        assert torch.equal(got[1], want[0]), f"rank {rank}: accept {got[1]} vs {want[0]}"
        assert torch.equal(results[0][1][0], got[0]), "ranks disagree"


@pytest.mark.parametrize("world", (2, 8))
def test_absorb_with_target_embeddings_is_bit_equal(mtp_worlds, world):
    """consult18 P5: the prompt absorption may reuse the target forward's
    all-reduced embeddings (row t's next token is the target's row t+1): the
    published embeddings shifted by one give bit-equal draft outputs."""
    from ignition.recipes.hy4.collectives import SimulatedGroup

    def body(decoder):
        _, tokens = golden_tokens()
        ids = torch.tensor([tokens], dtype=torch.long)
        n = 20
        decoder.embed_sink = {}
        t = ids[:, :n]
        _, hidden, _ = decoder.forward(t, return_hidden=True)
        emb = decoder.embed_sink.pop(id(t))
        decoder.embed_sink = None
        decoder.append([0], n)
        nxt_ids = torch.cat([ids[:, 1:n], ids[:, n - 1:n]], dim=1)
        nxt_emb = torch.cat([emb[1:], emb[-1:]], dim=0)
        decoder._draft_positions[0] = 0
        a = decoder.draft_absorb(nxt_ids, hidden, logits="all")
        decoder._draft_positions[0] = 0
        b = decoder.draft_absorb(nxt_ids, hidden, logits="all", next_embeds=nxt_emb)
        return a[0], a[1], b[0], b[1]

    group = SimulatedGroup(world)
    for rank, (la, ha, lb, hb) in enumerate(group.run([lambda col, d=d: body(d)
                                                       for d in _decoders(mtp_worlds[world], group)])):
        assert torch.equal(la, lb) and torch.equal(ha, hb), f"world {world} rank {rank}"


@pytest.mark.parametrize("world", (2, 8))
def test_int8_attention_weights_match_dequantized_reference(mtp_worlds, world):
    """consult18 P6: INT8 storage of the same-input pack + o_proj (w8.py) gives
    exactly the forward of a model whose weights ARE the dequantized values
    (one packed GEMM split vs per-slice GEMMs: equal on CPU fp32), and a
    decode step after it."""
    from dataclasses import replace

    from ignition.recipes.hy4.collectives import SimulatedGroup
    from ignition.recipes.hy4.w8 import Int8Weight, quantize_attention

    def run(mode):
        def body(decoder):
            for layer in decoder.weights.layers:
                attn = layer.attention
                w = attn.weights
                if mode == "int8":
                    quantize_attention(attn)
                elif mode == "dequant":
                    pk = w.packed_input
                    ok_pack = pk is not None and pk.packed.shape[1] % 128 == 0
                    deq = Int8Weight.quantize(pk.packed).dequant() if ok_pack else pk.packed
                    pk2 = replace(pk, packed=deq)
                    o = (Int8Weight.quantize(w.o_proj).dequant() if w.o_proj.shape[1] % 128 == 0
                         else w.o_proj)
                    attn.weights = replace(w, packed_input=pk2, o_proj=o)
            _, tokens = golden_tokens()
            ids = torch.tensor([tokens], dtype=torch.long)
            out = decoder.forward(ids[:, :20], logits_rows="all")
            decoder.append([0], 20)
            step = decoder.decode_step(torch.tensor([[5]], dtype=torch.long))
            return out, step
        return body

    results = {}
    for mode in ("dequant", "int8"):
        group = SimulatedGroup(world)
        results[mode] = group.run([lambda col, d=d, m=mode: run(m)(d)
                                   for d in _decoders(mtp_worlds[world], group)])
    for rank, (a, b) in enumerate(zip(results["dequant"], results["int8"])):
        for name, x, y in (("forward", a[0], b[0]), ("decode", a[1], b[1])):
            assert rel_error(y, x) < 1e-5, f"world {world} rank {rank} {name}: {rel_error(y, x)}"
