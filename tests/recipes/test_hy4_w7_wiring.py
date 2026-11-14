# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0
"""W7 bake-in wiring gates (P1/P2/P3) and their mutants — CPU.

P1 (transport + kernel wiring): the `PcieDcpCollectives` wiring of the real
b12x DCP surfaces (`all_gather_heads`, `lse_reduce_scatter` with base-2 LSE and
declared `partials_sink_folded`; the sink NEVER enters the merge — the one fold
is `PreparedSinkFold`), prepared plans per call shape (no JIT at run time; the
one-row decode shape is distinct from the verification-rows shape), the fused
iHC post+pre+norm, and a 2-rank CPU run of the WIRED paths equal to world 1.

P2 (decode bake-ins, consult7 §5): packed same-input projections at load,
shared expert overlapped with routed experts on decode graphs only, the
selection path on a second stream overlapping projections + head gather FOR
DECODE GRAPHS ONLY, the scores-only selection exchange, and the short-context
bypass. Mutant: wrong-stream overlap must FAIL the placement gate.

P3 (prefill bake-ins, consult8 §5): prefill graphs gather UNABSORBED queries
and absorb + expand locally (the `kv_b` two-wirings — decode keeps the
owner-side absorb), no two wire-heavy exchanges overlap in prefill graphs,
deterministic length-only chunking, the prefill graph catalogue (128..4,096
rows) with per-size content gates, and the scores-only exchange with the local
top-512 offer + exactness guard (the 2,048-selection equality). Mutants:
absorbed-query prefill gather and chunk-position nondeterminism must each FAIL.
"""

import pytest

torch = pytest.importorskip("torch")

from tests.recipes.test_hy4_w6_cp import (  # noqa: E402
    CAPACITY, TINY, golden_tokens, fresh_decoders, rel_error, tiny_available, worlds,
)

from ignition.recipes.hy4 import wiring
from ignition.recipes.hy4.wiring import (
    DECODE_ONE_ROW, DEFAULT_LOCAL_OFFER, PREFILL_GRAPH_SIZES, CallShape,
    StreamSchedule, chunk_plan, prefill_shape, verify_shape,
)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def wired_decoders(ranks, *, group, exchange_mode="scores-only", local_offer=None,
                   rows=2, capacity=CAPACITY, plans=None):
    """Fresh decoders per rank wired to `PcieDcpCollectives` over the transport."""
    from ignition.recipes.hy4.collectives import PcieDcpCollectives, SimulatedDcpTransport
    from ignition.recipes.hy4.decoder import Hy4Cache, Hy4Decoder

    world = len(ranks)
    transport = SimulatedDcpTransport(group)
    decoders = []
    for rank_index, rank in enumerate(ranks):
        cache = Hy4Cache.allocate(rank.model_config, rows=rows, capacity=capacity,
                                  device=torch.device("cpu"), draft=False,
                                  dtype=torch.float32, world=world, rank=rank_index)
        collectives = PcieDcpCollectives(
            rank=rank_index, world_size=world, transport=transport, plans=plans,
            exchange_mode=exchange_mode, local_offer=local_offer)
        decoders.append(Hy4Decoder(rank.weights, cache, compute_dtype=torch.float32,
                                   collectives=collectives))
    return decoders


def run_wired(ranks, body, *, group, **kwargs):
    decoders = wired_decoders(ranks, group=group, **kwargs)
    return group.run([lambda col, decoder=decoder: body(decoder) for decoder in decoders])


def trace_of(decoder, name):
    return [entry for entry in decoder.trace if entry[0] == name]


def assert_gate_fails(gate, *args, **kwargs):
    """The mutant gate pattern (gate-c style): the gate MUST fail here."""
    with pytest.raises(AssertionError):
        gate(*args, **kwargs)


def scenario_prompt_and_step(decoder):
    """One prefill over the short golden prompt, then one decode step."""
    _, tokens = golden_tokens()
    batch = torch.tensor([tokens], dtype=torch.long)
    prefill = decoder.prefill(batch)
    decoder.complete([0], batch)
    step = decoder.decode_step(torch.tensor([[7]], dtype=torch.long), request_rows=[0])
    return [prefill, step]


# --------------------------------------------------------------------------
# P1: call shapes, prepared plans, the transport wiring
# --------------------------------------------------------------------------
def test_one_row_decode_shape_is_distinct_from_the_verification_shape():
    """consult7 §5: "a one-row decode graph distinct from the verification graph"."""
    verify = verify_shape(4)
    assert DECODE_ONE_ROW.name == "decode" and verify.name == "verify"
    assert DECODE_ONE_ROW.is_decode_graph and verify.is_decode_graph
    assert DECODE_ONE_ROW.mla_mode == "decode" and verify.mla_mode == "verify"
    assert DECODE_ONE_ROW != verify and DECODE_ONE_ROW.rows == 1 and verify.rows == 4
    prefill = prefill_shape(1, 128)
    assert prefill.is_prefill_graph and prefill.mla_mode == "extend"
    with pytest.raises(ValueError):
        CallShape("decode", rows=2, tokens=1)   # decode is THE one-row shape
    with pytest.raises(ValueError):
        CallShape("verify", rows=1, tokens=1)   # verification is the multi-row shape


def test_prepared_plans_are_per_call_shape_and_refuse_jit(worlds):
    """One plan per call shape; a plan book bound means no JIT at run time."""
    from ignition.recipes.hy4.wiring import KernelPlanBook

    rank = worlds[1][0]
    attn = rank.weights.layers[0].attention
    saved = (attn._plan_book, dict(attn._mla_plans), dict(attn._sink_folds))
    try:
        book = KernelPlanBook()
        shapes = (DECODE_ONE_ROW, verify_shape(2))
        attn.bind_plan_book(book)
        attn.prepare_plans(shapes)
        decode_plan = book.plan_for(DECODE_ONE_ROW, "sparse_mla")
        verify_plan = book.plan_for(verify_shape(2), "sparse_mla")
        assert decode_plan is not verify_plan, "the one-row decode plan is not the verification plan"
        assert book.has(DECODE_ONE_ROW, "sink_fold") and book.has(verify_shape(2), "sink_fold")
        # No JIT at run time: an unprepared shape (a prefill chunk) refuses.
        with pytest.raises(RuntimeError, match="no JIT at run time"):
            attn.attend(torch.zeros(8, attn.heads, attn.kv_lora), None, None, None,
                        call_shape=prefill_shape(1, 8))
        with pytest.raises(RuntimeError, match="no JIT at run time"):
            attn.attend(torch.zeros(8, attn.heads, attn.kv_lora), None, None, None)
    finally:
        attn._plan_book, attn._mla_plans, attn._sink_folds = saved
        attn._mla_plans.clear()
        attn._mla_plans.update(saved[1])
        attn._sink_folds.clear()
        attn._sink_folds.update(saved[2])


def test_pcie_dcp_wiring_calls_the_transport_surfaces():
    """The TODO hooks wired: `all_gather_heads` / `lse_reduce_scatter` on the
    b12x surfaces with prepared plans, base-2 LSE, declared partial provenance,
    and NO sink inside the merge (the fold is `PreparedSinkFold`, after)."""
    from ignition.recipes.hy4.collectives import PcieDcpCollectives

    class RecordingTransport:
        def __init__(self):
            self.calls = []

        def bind(self, rank):
            return self

        def all_gather_heads(self, local_input, out=None, *, plan):
            self.calls.append(("all_gather_heads", tuple(local_input.shape), plan))
            return torch.cat([local_input, local_input], dim=1)

        def lse_reduce_scatter(self, partial_output, partial_lse, out=None, *, plan, lse_base,
                               partials_sink_folded=None, return_lse=False, **extra):
            self.calls.append(("lse_reduce_scatter", lse_base, partials_sink_folded,
                               "attention_sink" in extra, plan))
            heads = partial_output.shape[1] // 2
            merged = partial_output[:, :heads]
            return merged, partial_lse[:, :heads]

        def exchange_scores(self, offers, *, plan):
            self.calls.append(("exchange_scores", tuple(offers.shape), plan))
            return [offers, offers]

        def all_reduce(self, value):
            return value

        def all_gather(self, value):
            return value

    transport = RecordingTransport()
    col = PcieDcpCollectives(rank=0, world_size=2, transport=transport)
    q = torch.zeros(3, 4, 8)
    gathered = col.all_gather_heads(q)
    assert tuple(gathered.shape) == (3, 8, 8)
    out, lse = col.lse_reduce(torch.zeros(3, 8, 6), torch.zeros(3, 8),
                              partials_sink_folded=False)
    assert tuple(out.shape) == (3, 4, 6)
    assert transport.calls[0][0] == "all_gather_heads"
    surface = transport.calls[1]
    assert surface[0] == "lse_reduce_scatter"
    assert surface[1] == "base2", "the merge must state the base-2 LSE explicitly"
    assert surface[2] is False, "partials_sink_folded must be declared into the merge"
    assert surface[3] is False, "the sink must NOT enter the merge (one post-merge fold)"
    # The scores-only exchange moves SCORES only (offer width), never keys.
    col.merge_topk(torch.tensor([[3.0, 2.0, 1.0]]), torch.tensor([[5, 3, 1]]), topk=2)
    scores_call = transport.calls[2]
    assert scores_call[0] == "exchange_scores" and scores_call[1] == (1, 3)


def test_wired_two_rank_paths_equal_world_1(worlds):
    """Gate-a style over the WIRED `PcieDcpCollectives` path (2 and 8 ranks)."""
    from ignition.recipes.hy4.collectives import SimulatedGroup

    for world in (2, 8):
        reference = [scenario_prompt_and_step(decoder)
                     for decoder in fresh_decoders(worlds[1])]
        group = SimulatedGroup(world)
        simulated = run_wired(worlds[world], scenario_prompt_and_step, group=group)
        for rank_out, want in zip(simulated, reference):
            for got, expected in zip(rank_out, want):
                assert rel_error(got, expected) <= 1e-3, (
                    f"wired path (world {world}) drifted from world 1: "
                    f"{rel_error(got, expected)}")


def test_fused_ihc_post_pre_norm_runs_the_boundary(worlds):
    """The layer loop runs the FUSED iHC post+pre+norm boundary (one crossing
    one fused op), not a post + pre + norm sequence."""
    from ignition.recipes.hy4.state import IhcBoundary

    seen = {"pre": 0, "post": 0, "post_pre": 0}
    originals = (IhcBoundary.pre, IhcBoundary.post, IhcBoundary.post_pre)

    def counted(name, original):
        def wrapper(*args, **kwargs):
            seen[name] += 1
            return original(*args, **kwargs)
        return wrapper

    IhcBoundary.pre = counted("pre", originals[0])
    IhcBoundary.post = counted("post", originals[1])
    IhcBoundary.post_pre = counted("post_pre", originals[2])
    try:
        scenario_prompt_and_step(fresh_decoders(worlds[1])[0])
    finally:
        IhcBoundary.pre, IhcBoundary.post, IhcBoundary.post_pre = originals
    layers = len(worlds[1][0].weights.layers)
    # Per forward: layer-0 `pre`, the tail `post`, and every interior boundary
    # as ONE fused post+pre+norm call — (layers - 1) ffn->attn crossings plus
    # `layers` attn->ffn crossings = 2 * layers - 1.
    assert seen["post_pre"] == 2 * (2 * layers - 1), (
        "every interior boundary must run as ONE fused post+pre+norm call")
    assert (seen["pre"], seen["post"]) == (2, 2), (
        "only layer 0's pre and the tail post run unfused")


# --------------------------------------------------------------------------
# P2: packed same-input projections, overlap, scores-only, short bypass
# --------------------------------------------------------------------------
def test_packed_same_input_projections_are_built_at_load(worlds):
    """q_a + kv_a + linear_gate + index wk pack at load; router + shared gate/up
    pack at load; the packed rows are bitwise the loose weights."""
    rank = worlds[1][0]
    for layer in rank.weights.layers:
        attn = layer.attention
        packed = attn.weights.packed_input
        assert packed is not None
        assert set(packed.slices) >= {"q_a_proj", "kv_a_proj", "linear_gate"}
        assert torch.equal(packed.rows("q_a_proj"), attn.weights.q_a_proj)
        assert torch.equal(packed.rows("kv_a_proj"), attn.weights.kv_a_proj)
        assert torch.equal(packed.rows("linear_gate"), attn.weights.gate_proj)
        if attn.weights.indexer is not None:
            assert torch.equal(packed.rows("indexer_wk"), attn.weights.indexer.wk)
        moe = layer.mlp
        if hasattr(moe, "packed_input") and moe.packed_input is not None:
            assert torch.equal(moe.packed_input.rows("router"), moe.router.gate)
            assert torch.equal(moe.packed_input.rows("shared_gate"), moe.shared.gate_proj)
            assert torch.equal(moe.packed_input.rows("shared_up"), moe.shared.up_proj)


def test_packed_projection_math_is_the_separate_math(worlds):
    rank = worlds[1][0]
    layer = rank.weights.layers[1]
    attn = layer.attention
    rows = 3
    x = torch.randn(rows, rank.model_config.hidden, dtype=torch.float32)
    positions = torch.arange(rows, dtype=torch.long)
    bundle = attn.project_packed(x, positions, with_index_keys=attn.weights.indexer is not None)
    q_resid, q, latent = attn.project(x, positions)
    assert torch.equal(bundle.q_resid, q_resid) and torch.equal(bundle.q, q)
    assert torch.equal(bundle.latent, latent)
    if attn.weights.indexer is not None:
        assert torch.equal(bundle.index_keys, attn.prepare_index_keys(x, positions))
    assert torch.equal(bundle.gate_raw, x @ attn.weights.gate_proj.T.to(x.dtype))


def test_shared_expert_overlap_is_per_graph_family(worlds):
    """consult7 §5 overlaps shared with routed experts on decode graphs;
    consult8 §6 makes overlap a property of the graph family — prefill does
    not overlap (and never two wire-heavy exchanges either)."""
    assert DECODE_ONE_ROW.overlap_shared_expert is True
    assert verify_shape(2).overlap_shared_expert is True
    assert prefill_shape(1, 8).overlap_shared_expert is False
    for family, want in ((wiring.GraphFamily.DECODE, True), (wiring.GraphFamily.PREFILL, False)):
        schedule = StreamSchedule.for_family(family)
        assert (schedule.placement("shared_expert", family) != schedule.placement("routed_experts", family)) is want


def gate_stream_placement(for_family):
    """The wiring gate: decode overlaps selection with projections + gather;
    prefill keeps BOTH wire-heavy exchanges on the wire lane (serialized)."""
    decode = wiring.GraphFamily.DECODE
    prefill = wiring.GraphFamily.PREFILL
    decode_schedule, prefill_schedule = for_family(decode), for_family(prefill)
    assert decode_schedule.overlaps("selection_exchange", "projections", decode), (
        "decode must run the selection path on a second stream")
    assert decode_schedule.placement("selection_exchange", decode) == "selection"
    assert decode_schedule.placement("query_exchange", decode) == "compute"
    assert not prefill_schedule.wire_heavy_overlap(prefill), (
        "prefill must not overlap two wire-heavy exchanges")
    assert prefill_schedule.placement("selection_exchange", prefill) == "wire"
    assert prefill_schedule.placement("query_exchange", prefill) == "wire"


def test_selection_overlap_gate_passes_on_the_shipped_schedule():
    gate_stream_placement(StreamSchedule.for_family)


def test_mutant_wrong_stream_overlap_fails_the_gate():
    """Mutant: the prefill selection exchange placed on the second stream
    (the decode-optimal overlap leaking into prefill — consult8 §6)."""
    from ignition.recipes.hy4.wiring import _Placement

    def mutant_for_family(family):
        schedule = StreamSchedule()
        schedule._placements = {}
        for item in StreamSchedule.for_family(family)._placements:
            stream = item.stream
            if family == wiring.GraphFamily.PREFILL and item.op == "selection_exchange":
                stream = "selection"   # wrong-stream overlap
            schedule._placements[_Placement(item.op, family, stream)] = True
        return schedule

    assert_gate_fails(gate_stream_placement, mutant_for_family)


def test_short_context_bypass_skips_indexer_and_exchange(worlds):
    """decode family, context <= top-k: no indexer, no exchange — and the
    result still equals world 1 (which scores everything)."""
    from ignition.recipes.hy4.collectives import SimulatedGroup

    topk = worlds[1][0].model_config.text["index_topk"]

    def short_step(decoder):
        warm = torch.tensor([[7, 3, 11, 5]], dtype=torch.long)
        decoder.prefill(warm, request_rows=[0])
        decoder.complete([0], warm)
        return decoder.decode_step(torch.tensor([[13]], dtype=torch.long), request_rows=[0])

    reference = short_step(fresh_decoders(worlds[1])[0])
    group = SimulatedGroup(2)
    decoders = wired_decoders(worlds[2], group=group)

    def warm_up(decoder):
        warm = torch.tensor([[7, 3, 11, 5]], dtype=torch.long)
        decoder.prefill(warm, request_rows=[0])
        decoder.complete([0], warm)
        decoder.trace.clear()          # only the DECODE step's wiring is gated
        return None

    group.run([lambda col, decoder=decoder: warm_up(decoder) for decoder in decoders])
    for decoder in decoders:
        decoder.collectives.merge_topk = lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("the short-context bypass must not run the exchange"))
    out = group.run([lambda col, decoder=decoder: decoder.decode_step(
        torch.tensor([[13]], dtype=torch.long), request_rows=[0])
        for decoder in decoders])[0]
    assert rel_error(out, reference) <= 1e-3
    for decoder in decoders:
        assert trace_of(decoder, "selection_bypass"), "the bypass must be recorded"
        assert not trace_of(decoder, "selection_exchange"), "no exchange at short context"
    assert topk >= 5, "the scenario must sit under top-k for the bypass to bind"


def test_scores_only_exchange_is_the_2048_selection_equality(worlds):
    """The scores-only exchange (each rank needs only the global k-th score and
    its own keys) selects exactly the global top-2,048 — the W6 gate-b
    equality; and the local top-512 offer carries its exactness guard."""
    from ignition.recipes.hy4.collectives import (
        check_scores_only_exactness, scores_only_selection, scores_only_threshold,
    )

    topk = 2048
    world = 2
    for visible in (3000, 500):
        # Tie plateaus of four; the cut falls between plateaus (tie-order free).
        scores = torch.tensor([[1000.0 - (p // 4) for p in range(visible)]], dtype=torch.float32)
        want = sorted(torch.topk(scores[0], min(topk, visible)).indices.tolist())
        per_rank = []
        for rank in range(world):
            width = -(-visible // world)
            owned = [p for p in range(visible) if p % world == rank][:topk]
            values = torch.full((1, topk), float("-inf"))
            positions = torch.full((1, topk), -1, dtype=torch.int64)
            picked = sorted(owned, key=lambda p: -scores[0, p])[:topk]
            for j, p in enumerate(picked):
                values[0, j] = scores[0, p]
                positions[0, j] = p
            per_rank.append((values, positions))
        # Full-width offer: exact by construction (the gate-b override).
        offers = [values[:, :topk] for values, _ in per_rank]
        union = []
        for rank in range(world):
            values, positions = per_rank[rank]
            got_values, got_positions = scores_only_selection(
                offers, values, positions, topk=topk, world=world, rank=rank, local_offer=None)
            union += [int(p) for p in got_positions[0].tolist() if p >= 0]
        assert sorted(union) == want, (
            f"scores-only selection != the 2,048 selection (visible {visible})")
        assert len(sorted(set(union))) == min(topk, visible)
        # The 512-offer guard: a full-width offer never truncates.
        thresholds, _ = scores_only_threshold(offers, topk)
        check_scores_only_exactness(thresholds, None, per_rank[0][0])


def test_local_512_offer_exactness_guard_raises_on_truncation():
    """The guard: an offer whose cut sits at/above the global k-th score cannot
    be exact — it must raise rather than return a wrong selection."""
    from ignition.recipes.hy4.collectives import (
        check_scores_only_exactness, scores_only_threshold,
    )

    topk = 3
    # Two ranks offer two of their candidates; rank 0's un-offered keys clear the cut.
    candidates = [torch.tensor([[9.0, 8.0, 7.0, 6.0, 5.0]]),
                  torch.tensor([[4.0, 3.0]])]
    offers = [values[:, :2] for values in candidates]
    thresholds, _ = scores_only_threshold(offers, topk)
    assert thresholds[0] == 4.0
    check_scores_only_exactness(thresholds, None, candidates[0])  # full offer: fine
    with pytest.raises(ValueError, match="truncated"):
        check_scores_only_exactness(thresholds, 2, candidates[0])  # 2-wide offer truncates rank 0


# --------------------------------------------------------------------------
# P3: the prefill wiring, chunking, the graph catalogue
# --------------------------------------------------------------------------
def gate_query_wiring(trace, family, wiring_name, width=None):
    """The wiring gate: prefill gathers UNABSORBED queries and absorbs locally;
    decode keeps the owner-side absorb."""
    exchanges = [entry for entry in trace if entry[0] == "query_exchange"]
    assert exchanges, "the query exchange must be recorded"
    for entry in exchanges:
        assert entry[2] == wiring_name, (
            f"{family} must run the {wiring_name} query wiring, got {entry[2]}")
        if width is not None:
            assert entry[3] == width, (
                f"{family} gather width {entry[3]} != the wiring's {width}")


def test_prefill_gathers_unabsorbed_and_decode_absorbs_owner_side(worlds):
    """The `kv_b` two-wirings (consult8 §5), observed at the exchange."""
    from ignition.recipes.hy4.collectives import SimulatedGroup

    _, tokens = golden_tokens()
    qk_width = (worlds[1][0].model_config.text["qk_nope_head_dim"]
                + worlds[1][0].model_config.text["qk_rope_head_dim"])

    def prompt(decoder):
        batch = torch.tensor([tokens], dtype=torch.long)
        return decoder.prefill(batch)

    group = SimulatedGroup(2)
    decoders = wired_decoders(worlds[2], group=group)
    group.run([lambda col, decoder=decoder: prompt(decoder) for decoder in decoders])
    for decoder in decoders:
        gate_query_wiring(decoder.trace, "prefill", "unabsorbed-local-absorb", width=qk_width)

    def step(decoder):
        warm = torch.tensor([[7, 3, 11, 5, 13, 2, 17, 19, 23, 29, 31, 37,
                              41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89]],
                            dtype=torch.long)
        decoder.prefill(warm, request_rows=[0])
        decoder.complete([0], warm)
        decoder.trace.clear()   # only the DECODE step's query wiring is gated
        return decoder.decode_step(torch.tensor([[7]], dtype=torch.long), request_rows=[0])

    group = SimulatedGroup(2)
    decoders = wired_decoders(worlds[2], group=group)
    group.run([lambda col, decoder=decoder: step(decoder) for decoder in decoders])
    for decoder in decoders:
        gate_query_wiring(decoder.trace, "decode", "owner-absorbed")


def test_mutant_absorbed_query_prefill_gather_fails_the_gate(worlds):
    """Mutant: prefill gathering ABSORBED queries (the decode wiring leaking
    into prefill — prefill must gather unabsorbed and absorb locally)."""
    from ignition.recipes.hy4.collectives import SimulatedGroup

    _, tokens = golden_tokens()

    def prompt(decoder):
        batch = torch.tensor([tokens], dtype=torch.long)
        return decoder.prefill(batch)

    original = CallShape.query_wiring
    CallShape.query_wiring = property(lambda self: "owner-absorbed")
    try:
        group = SimulatedGroup(2)
        decoders = wired_decoders(worlds[2], group=group)
        group.run([lambda col, decoder=decoder: prompt(decoder) for decoder in decoders])
    finally:
        CallShape.query_wiring = original
    for decoder in decoders:
        assert_gate_fails(gate_query_wiring, decoder.trace, "prefill",
                          "unabsorbed-local-absorb")


def test_no_two_wire_heavy_exchanges_overlap_in_prefill(worlds):
    from ignition.recipes.hy4.collectives import SimulatedGroup

    _, tokens = golden_tokens()

    def prompt(decoder):
        batch = torch.tensor([tokens], dtype=torch.long)
        return decoder.prefill(batch)

    group = SimulatedGroup(2)
    decoders = wired_decoders(worlds[2], group=group)
    group.run([lambda col, decoder=decoder: prompt(decoder) for decoder in decoders])
    schedule = StreamSchedule.for_family(wiring.GraphFamily.PREFILL)
    for decoder in decoders:
        for name in ("selection_exchange", "query_exchange"):
            placements = [entry[1] for entry in trace_of(decoder, name)]
            assert placements and all(p == "wire" for p in placements)
    assert not schedule.wire_heavy_overlap(wiring.GraphFamily.PREFILL)


def test_deterministic_length_only_chunking(worlds):
    """Chunk boundaries come from LENGTHS only (consult8 §5) and the chunked
    forward equals the one-shot forward."""
    plan = chunk_plan([24, 17], 8)
    assert plan == chunk_plan([24, 17], 8)
    assert plan == ((0, 0, 8), (0, 8, 16), (0, 16, 24), (1, 0, 8), (1, 8, 16), (1, 16, 17))

    from ignition.recipes.hy4.decoder import Hy4Cache, Hy4Decoder

    rank = worlds[1][0]
    outcomes = []
    for content in ([7, 3, 11, 5, 13, 2, 17, 19, 23, 29, 31, 37,
                     41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]):
        cache = Hy4Cache.allocate(rank.model_config, rows=1, capacity=CAPACITY,
                                  device=torch.device("cpu"), draft=False,
                                  dtype=torch.float32)
        decoder = Hy4Decoder(rank.weights, cache, compute_dtype=torch.float32,
                             prefill_chunk_size=8)
        ids = torch.tensor([content], dtype=torch.long)
        decoder.prefill(ids)
        outcomes.append([entry for entry in decoder.trace if entry[0] == "chunk"])
    assert outcomes[0] == outcomes[1], (
        "chunk boundaries must not depend on the token content")
    assert outcomes[0] == [("chunk", 0, 8), ("chunk", 8, 16), ("chunk", 16, 24)]

    # Chunked == one-shot (the chunking is batching only).
    chunked_cache = Hy4Cache.allocate(rank.model_config, rows=1, capacity=CAPACITY,
                                      device=torch.device("cpu"), draft=False,
                                      dtype=torch.float32)
    chunked = Hy4Decoder(rank.weights, chunked_cache, compute_dtype=torch.float32,
                         prefill_chunk_size=8)
    one_cache = Hy4Cache.allocate(rank.model_config, rows=1, capacity=CAPACITY,
                                  device=torch.device("cpu"), draft=False,
                                  dtype=torch.float32)
    one = Hy4Decoder(rank.weights, one_cache, compute_dtype=torch.float32)
    ids = torch.tensor([[7, 3, 11, 5, 13, 2, 17, 19, 23, 29, 31, 37,
                         41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89]], dtype=torch.long)
    got, want = chunked.prefill(ids), one.prefill(ids)
    assert rel_error(got, want) <= 1e-5, "chunking must not change the forward"


def test_mutant_chunk_position_nondeterminism_fails_the_gate(worlds):
    """Mutant: a content-dependent chunker (the K3.1 TTFT-bump defect shape) —
    the determinism gate must catch it."""
    from ignition.recipes.hy4 import decoder as decoder_module
    from ignition.recipes.hy4.decoder import Hy4Cache, Hy4Decoder

    rank = worlds[1][0]
    poison = {"content": None}

    def bad_chunk_plan(lengths, chunk_size):
        # The defect: boundaries move with the content, not just the lengths.
        shift = (sum(poison["content"]) % 5) if poison["content"] else 0
        return tuple((row, start + shift, min(start + shift + chunk_size, length))
                     for row, length in enumerate(lengths)
                     for start in range(0, length, chunk_size))

    outcomes = []
    original = decoder_module.chunk_plan
    decoder_module.chunk_plan = bad_chunk_plan
    try:
        for content in ([7, 3, 11, 5, 13, 2, 17, 19, 23, 29, 31, 37,
                         41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]):
            poison["content"] = content
            cache = Hy4Cache.allocate(rank.model_config, rows=1, capacity=CAPACITY,
                                      device=torch.device("cpu"), draft=False,
                                      dtype=torch.float32)
            decoder = Hy4Decoder(rank.weights, cache, compute_dtype=torch.float32,
                                 prefill_chunk_size=8)
            decoder.prefill(torch.tensor([content], dtype=torch.long))
            outcomes.append([entry for entry in decoder.trace if entry[0] == "chunk"])
    finally:
        decoder_module.chunk_plan = original
    assert outcomes[0] != outcomes[1], "sanity: the mutant must actually diverge"

    def gate(outcome_pair):
        assert outcome_pair[0] == outcome_pair[1], (
            "chunk boundaries must not depend on the token content")

    assert_gate_fails(gate, outcomes)


# --------------------------------------------------------------------------
# the tie-aware content comparison (the selection-tie contract)
# --------------------------------------------------------------------------
SELECTION_TIE_BAND = 5e-3


class SelectionTape:
    """Every selection a run hands its attention, and the score field behind it.

    The content gates compare exact forwards; exact SELECTION equality needs
    tie-free score fields at the top-k cut (the W6 gates' rule). The indexer's
    relu makes exact 0.0 plateaus, so a cut inside a plateau is legitimately
    world-dependent — the scores-only wire carries no positions (its plateau
    fill is the documented (rank, position) order) while world 1 picks in
    `torch.topk`'s unspecified order (gate b's top-2048 rule: "the gate must
    not depend on it"). This tape records what each path selected and the
    scores it selected from, so a gate can PROVE that a forward difference is
    exactly such a plateau swap — an equal-score swap at the cut in each run's
    own field — and not a CP machinery bug.
    """

    def __init__(self):
        self.cells = {}        # id(attention cell) -> (tag, layer)
        self.selections = {}   # (tag, layer) -> (selected, counts)
        self.scores = {}       # (tag, layer) -> (score rows, call kwargs)

    def __enter__(self):
        from ignition.recipes.hy4 import attention as attention_module

        tape, module = self, attention_module
        self._attend = module.Hy4LayerAttention.attend
        self._index_scores = module.Hy4LayerAttention.index_scores
        attend, index_scores = self._attend, self._index_scores

        def captured_attend(cell, q, latent_table, selected, counts, **kwargs):
            out = attend(cell, q, latent_table, selected, counts, **kwargs)
            tag, layer = tape.cells.get(id(cell), ("?", -1))
            tape.selections[(tag, layer)] = (selected.detach().clone(),
                                             counts.detach().clone())
            return out

        def captured_scores(cell, x, q_resid, positions, visible, key_table,
                            slot_base, **kwargs):
            out = index_scores(cell, x, q_resid, positions, visible, key_table,
                               slot_base, **kwargs)
            tag, layer = tape.cells.get(id(cell), ("?", -1))
            tape.scores[(tag, layer)] = (out.detach().clone(), dict(kwargs))
            return out

        module.Hy4LayerAttention.attend = captured_attend
        module.Hy4LayerAttention.index_scores = captured_scores
        return self

    def __exit__(self, *exc):
        from ignition.recipes.hy4 import attention as attention_module

        attention_module.Hy4LayerAttention.attend = self._attend
        attention_module.Hy4LayerAttention.index_scores = self._index_scores

    def register(self, tag, weights):
        self.cells.update({id(layer.attention): (tag, index)
                           for index, layer in enumerate(weights.layers)})

    def selected_set(self, tag, layer, row):
        selected, _ = self.selections[(tag, layer)]
        return {int(v) for v in selected[row].tolist() if int(v) >= 0}

    def score(self, tag, layer, row, key):
        """The key's score in the run's OWN field (CP: the owning rank's)."""
        if tag == "solo":
            scores, kwargs = self.scores[(tag, layer)]
            world = 1
        else:
            scores, kwargs = self.scores[(f"cp{key % 2}", layer)]
            world = 2
        column = key // world   # world 1: the position; CP: local index
        return float(scores[row][column])

    def tied_swaps(self, layer, row, want, got):
        """True when the set difference is exactly an equal-score cut swap."""
        if not (want - got) or len(want - got) != len(got - want):
            return False
        for dropped in sorted(want - got):
            for added in sorted(got - want):
                for tag in ("solo", "cp"):
                    if self.score(tag, layer, row, dropped) != self.score(tag, layer, row, added):
                        return False
        return True


def test_prefill_graph_catalogue_declared_with_content_gates(worlds):
    """consult8 §5: "the prefill graph catalogue (128 … 4,096 rows) declared,
    so the gates can cover it". Every size resolves to a prefill call shape on
    the prefill wiring with a deterministic chunk cover; the affordable sizes
    additionally run CONTENT gates (the wired CP forward equals world 1,
    tie-aware: a cut inside an exact score plateau is a documented
    world-dependent tie and must show up as a proven equal-score swap —
    anything else fails the gate)."""
    from ignition.recipes.hy4.collectives import SimulatedGroup

    assert PREFILL_GRAPH_SIZES == (128, 256, 512, 1024, 2048, 4096)
    for size in PREFILL_GRAPH_SIZES:
        shape = prefill_shape(1, size)
        assert shape.is_prefill_graph and shape.mla_mode == "extend"
        assert shape.query_wiring == "unabsorbed-local-absorb"
        assert not shape.overlap_selection and not shape.overlap_shared_expert
        assert shape.local_offer == DEFAULT_LOCAL_OFFER
        cover = chunk_plan([size], 128)
        assert sum(stop - start for _, start, stop in cover) == size
        assert cover == chunk_plan([size], 128)

    # Per-size content gates at the sizes the CPU simulation can afford; the
    # larger catalogue entries are covered by the structural gates above (the
    # full-shape content gates run in the P5 GPU window against the capture).
    from ignition.recipes.hy4.bootstrap import PreparationConfig, construct_rank
    from ignition.recipes.hy4.collectives import PcieDcpCollectives, SimulatedDcpTransport, SimulatedGroup
    from ignition.recipes.hy4.decoder import Hy4Cache, Hy4Decoder
    from tests.recipes.test_hy4_reference_gates import to_fp32_weights

    for size in (128, 256):
        settings = PreparationConfig(devices=(0,), mtp=False, prefill_chunk_size=size)
        # Distinct token ids per position keep the score fields as tie-free as
        # the data allows (the W6 gates' rule) — the indexer's relu still makes
        # exact 0.0 plateaus, so the gate below proves any selection
        # difference is an exact cut tie (see `SelectionTape`).
        tokens = list(range(size))
        batch = torch.tensor([tokens], dtype=torch.long)

        with SelectionTape() as tape:
            # World 1 reference from a fresh rank sized for this catalogue entry
            # (the fresh FP32 decoder — the constructed entry point runs BF16).
            solo = construct_rank(TINY, settings, rank=0, pinned=False, mtp=False,
                                  cache_rows=1, cache_capacity=size + 8, world_size=1)
            solo.weights = to_fp32_weights(solo.weights)
            tape.register("solo", solo.weights)
            solo_cache = Hy4Cache.allocate(solo.model_config, rows=1, capacity=size + 8,
                                           device=torch.device("cpu"), draft=False,
                                           dtype=torch.float32)
            reference = Hy4Decoder(solo.weights, solo_cache,
                                   compute_dtype=torch.float32).prefill(batch)
            # The wired 2-rank CP run at the same size.
            group = SimulatedGroup(2)
            decoders = []
            for rank_index in range(2):
                built = construct_rank(TINY, settings, rank=rank_index, pinned=False, mtp=False,
                                       cache_rows=1, cache_capacity=size + 8, world_size=2,
                                       collectives=group.collectives(rank_index))
                built.weights = to_fp32_weights(built.weights)
                tape.register(f"cp{rank_index}", built.weights)
                cache = Hy4Cache.allocate(built.model_config, rows=1, capacity=size + 8,
                                          device=torch.device("cpu"), draft=False,
                                          dtype=torch.float32, world=2, rank=rank_index)
                decoders.append(Hy4Decoder(
                    built.weights, cache, compute_dtype=torch.float32,
                    collectives=PcieDcpCollectives(
                        rank=rank_index, world_size=2,
                        transport=SimulatedDcpTransport(group))))
            out = group.run([lambda col, decoder=decoder: decoder.prefill(batch)
                             for decoder in decoders])[0]

        # Tie-aware content gate. Every query whose selection matches world 1
        # must meet the fp band (1e-3; measured 2.5e-6 at both sizes — the
        # residual is reduction-order noise only). A selection difference is
        # allowed ONLY where the top-k cut lands inside an exact score
        # plateau: the swapped keys must tie at the cut in each run's own
        # field (proven via `SelectionTape` — a real selection bug cannot
        # hide), and those rows get the selection-tie band instead of
        # loosening the fp gate. Measured at 256 rows: one query (row 214)
        # selects 15 strictly-positive keys plus one 0.0-plateau key at the
        # layer-1 indexer — world 1 picks 121 (`torch.topk`'s unspecified
        # order), the wired run picks 168 (the scores-only (rank, position)
        # plateau fill) — moving that row's logits by 1.3e-3 while every
        # other row stays at fp noise. `SELECTION_TIE_BAND` (5e-3) bounds a
        # swap landing on a higher-weight key (~4x the measured); the 128-row
        # gate is tie-free and passes the strict band alone.
        tie_rows = set()
        for layer in sorted({layer for (tag, layer) in tape.scores if tag == "solo"}):
            for row in range(size):
                want = tape.selected_set("solo", layer, row)
                got = set()
                for rank_index in range(2):
                    got |= {2 * slot + rank_index
                            for slot in tape.selected_set(f"cp{rank_index}", layer, row)}
                if got == want:
                    continue
                assert tape.tied_swaps(layer, row, want, got), (
                    f"catalogue {size} rows, layer {layer}, query {row}: selection "
                    f"{sorted(want)} != wired {sorted(got)} without an exact score "
                    f"tie at the cut — a selection bug, not tie order")
                tie_rows.add(row)
        scale = float(reference.abs().max())
        errors = (out - reference).abs().amax(dim=(0, 2)) / scale
        strict = [row for row in range(size) if row not in tie_rows]
        assert max(float(errors[row]) for row in strict) <= 1e-3, (
            f"prefill catalogue content gate failed at {size} rows")
        if tie_rows:
            assert max(float(errors[row]) for row in tie_rows) <= SELECTION_TIE_BAND, (
                f"catalogue {size} rows: boundary-tie rows moved more than "
                f"{SELECTION_TIE_BAND}: {sorted(tie_rows)}")
