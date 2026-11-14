# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0
"""W6 gates (D8): the CP body against world 1 on the tiny goldens — CPU only.

gate a: the 2-rank and 8-rank CPU simulation equals world 1 on the tiny goldens
        within 1e-3 relative — including a context longer than top-k and a
        batch of two rows (plus a decode step).
gate b: the W-rank selected set equals the single-rank top-k — including ties
        and contexts under top-k — through the shipped merge
        (`merge_tiled_topk_candidates`) and the per-row hand-back; and at the
        literal production scale: top-2,048 with tie plateaus and contexts
        under 2,048.
gate c: the sink-off, sink-per-rank and wrong-LSE-base mutants each FAIL the
        gate — each diverges from world 1 (and the double sink fold is
        rejected loudly by `partials_sink_folded`).
gate d: the position-ownership rule (amendment 2) asserted in code:
        `Hy4Cache.owner`/`.local` equal
        `dsa_indexer.selection_handback.rank_local_slots` (`p % world`, local
        `p // world`) over full 1,024-position pages and a batch of two rows,
        and a write->read round trip at world 2 and 8 finds every owned
        position.

The CP world is `hy4/collectives.SimulatedGroup`: W ranks of the SAME
`Hy4Decoder` body in lockstep, every collective combining over the simulated
ranks through the shipped contract (`lse_reduce_scatter_reference`,
`merge_tiled_topk_candidates`, `rank_local_slots`). The world-1 leg is the
eager path (`SoloCollectives`), the simulation's ground truth, unchanged; it
is additionally anchored to `goldens/golden.json`'s short-run logits rows.
"""

import copy
import dataclasses
import json
import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from tests.recipes.test_hy4_reference_gates import to_fp32_weights

REF = Path(os.environ.get("HY4_REFERENCE_DIR", "/home/tom/hy4/reference"))
TINY = REF / "tiny-hy4"
GOLDEN_PATH = REF / "goldens" / "golden.json"

CAPACITY = 32      # global positions per cache row (24-token golden + headroom)
WORLD_SIZES = (2, 8)


def golden_tokens():
    assert GOLDEN_PATH.is_file(), f"goldens missing: {GOLDEN_PATH}"
    golden = json.loads(GOLDEN_PATH.read_text())
    tokens = list(golden["runs"]["short"]["tokens"])
    assert tokens, "the short golden run must carry its tokens"
    return golden, tokens


def rel_error(got, want):
    """max |got - want| / max |want| — the tolerance file's own metric shape."""
    got, want = got.float(), want.float()
    return float((got - want).abs().max() / want.abs().max().clamp_min(1e-12))


# --------------------------------------------------------------------------
# fixtures: one constructed rank per (world, rank), FP32 weights
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def tiny_available():
    if not (TINY / "config.json").is_file() or not (TINY / "model.safetensors").is_file():
        pytest.skip(f"tiny-hy4 reference checkpoint not present under {TINY}")


@pytest.fixture(scope="module")
def worlds(tiny_available):
    """{world: [PreparedHy4Rank per rank]} for worlds 1, 2 and 8 (FP32 weights)."""
    from ignition.recipes.hy4.bootstrap import PreparationConfig, construct_rank
    from ignition.recipes.hy4.collectives import SimulatedGroup

    settings = PreparationConfig(devices=(0,), mtp=False, prefill_chunk_size=64)
    built = {}
    for world in (1,) + WORLD_SIZES:
        ranks = []
        # Construction wires a decoder per rank; the gates run FRESH decoders
        # bound to their run's `SimulatedGroup` (`fresh_decoders`), so the
        # construct-time collectives are a wiring placeholder.
        placeholder = SimulatedGroup(world)
        for rank in range(world):
            constructed = construct_rank(TINY, settings, rank=rank, pinned=False, mtp=False,
                                         cache_rows=2, cache_capacity=CAPACITY,
                                         world_size=world,
                                         collectives=placeholder.collectives(rank))
            constructed.weights = to_fp32_weights(constructed.weights)
            ranks.append(constructed)
        built[world] = ranks
    return built


def fresh_decoders(ranks, *, group=None, rows=2, capacity=CAPACITY):
    """One fresh decoder+cache per rank (FP32), wired to the group's collectives."""
    from ignition.recipes.hy4.decoder import Hy4Cache, Hy4Decoder

    world = len(ranks)
    decoders = []
    for rank_index, rank in enumerate(ranks):
        cache = Hy4Cache.allocate(rank.model_config, rows=rows, capacity=capacity,
                                  device=torch.device("cpu"), draft=False,
                                  dtype=torch.float32, world=world, rank=rank_index)
        collectives = None if group is None else group.collectives(rank_index)
        decoders.append(Hy4Decoder(rank.weights, cache, compute_dtype=torch.float32,
                                   collectives=collectives))
    return decoders


def run_world(ranks, body, *, group=None):
    """Run `body(decoder)` on every rank (lockstep through the group); results in rank order."""
    decoders = fresh_decoders(ranks, group=group)
    if group is None:
        return [body(decoder) for decoder in decoders]
    return group.run([lambda col, decoder=decoder: body(decoder) for decoder in decoders])


# --------------------------------------------------------------------------
# gate a: 2-rank and 8-rank simulation equals world 1 within 1e-3
# --------------------------------------------------------------------------
def scenario_short(decoder):
    """The short golden prompt, one row: 24 tokens > top-k 16."""
    _, tokens = golden_tokens()
    batch = torch.tensor([tokens], dtype=torch.long)
    return [decoder.prefill(batch)]


def scenario_two_rows(decoder):
    """A batch of two rows: row 0 continues at position 5, row 1 starts at 0."""
    _, tokens = golden_tokens()
    warm = torch.tensor([tokens[:5]], dtype=torch.long)
    decoder.prefill(warm, request_rows=[0])
    decoder.complete([0], warm)
    batch = torch.tensor([tokens, tokens], dtype=torch.long)
    prefill_logits = decoder.prefill(batch, request_rows=[0, 1])
    decoder.complete([0, 1], batch)
    step = decoder.decode_step(torch.tensor([[7], [3]], dtype=torch.long), request_rows=[0, 1])
    return [prefill_logits, step]


def _gate_a(worlds, scenario, world):
    from ignition.recipes.hy4.collectives import SimulatedGroup

    reference = run_world(worlds[1], scenario)[0]
    group = SimulatedGroup(world)
    simulated = run_world(worlds[world], scenario, group=group)[0]
    assert len(reference) == len(simulated)
    errors = [rel_error(got, want) for got, want in zip(simulated, reference)]
    assert max(errors) <= 1e-3, (
        f"gate a (world {world}, {scenario.__name__}): sim != world 1, "
        f"relative errors {errors} exceed 1e-3")


@pytest.mark.parametrize("scenario", [scenario_short, scenario_two_rows],
                         ids=["golden-short-1row", "two-rows-mixed-offsets"])
def test_gate_a_simulation_equals_world_1(worlds, scenario):
    for world in WORLD_SIZES:
        _gate_a(worlds, scenario, world)


def test_gate_a_world_1_is_the_golden_anchor(worlds):
    """The world-1 leg is the golden's own forward (finding C: goldens must be read)."""
    golden, tokens = golden_tokens()
    logits = scenario_short(fresh_decoders(worlds[1])[0])[0]
    for key, want in golden["runs"]["short"]["logits_rows"].items():
        row = logits[0, int(key)]
        want = torch.tensor(want, dtype=torch.float32)
        assert rel_error(row, want) <= 1e-3, f"world 1 drifted from golden.json row {key}"


# --------------------------------------------------------------------------
# gate b: the W-rank selected set equals the single-rank top-k
# --------------------------------------------------------------------------
FULL_LAYERS = (0, 1, 5)  # tiny-hy4's full-indexer layers


def capture_merged(ranks, world, body):
    """Run `body` in the simulated world; return (results, merged selections per rank)."""
    from ignition.recipes.hy4.collectives import SimulatedGroup

    group = SimulatedGroup(world)
    decoders = fresh_decoders(ranks, group=group)
    recorded = [[] for _ in decoders]
    for rank_index, decoder in enumerate(decoders):
        original = decoder.collectives.merge_topk

        def wrapped(scores, positions, *, topk, _original=original, _sink=recorded[rank_index]):
            values, indices = _original(scores, positions, topk=topk)
            _sink.append(indices.clone())
            return values, indices

        decoder.collectives.merge_topk = wrapped
    results = group.run([lambda col, decoder=decoder: body(decoder) for decoder in decoders])
    return results, recorded


def world1_positions(rank, layer_id, rows, token_rows):
    """The world-1 selection's global positions per query token (sorted)."""
    selected, counts = rank.selections[layer_id].buffers(rows=rows)
    capacity = rank.cache.capacity
    base = token_rows.long() * capacity
    out = []
    for row in range(rows):
        slots = selected[row][: int(counts[row])].long()
        out.append(sorted((slots - base[row]).tolist()))
    return out


def handed_back_positions(rank, world, rank_index, layer_id, rows, token_rows):
    """A simulated rank's handed-back selection, mapped back to global positions."""
    selected, counts = rank.selections[layer_id].buffers(rows=rows)
    local_capacity = rank.cache.local_capacity
    base = token_rows.long() * local_capacity
    out = []
    for row in range(rows):
        slots = selected[row][: int(counts[row])].long()
        out.append(sorted((int(v) * world + rank_index) for v in (slots - base[row]).tolist()))
    return out


def scenario_batch(decoder, rows, width):
    tokens = torch.tensor([[7, 3, 11, 5, 13, 2, 17, 19, 23, 29, 31, 37,
                            41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89][:width]]
                          * rows, dtype=torch.long)
    decoder.prefill(tokens)
    return tokens


@pytest.mark.parametrize("rows,width", [(1, 24), (2, 24), (1, 8)],
                         ids=["context-over-topk", "batch-of-two", "context-under-topk"])
def test_gate_b_w_rank_selected_set_equals_single_rank(worlds, rows, width):
    """Merged sets, handed-back partitions and per-row counts all match world 1."""
    topk = worlds[1][0].model_config.text["index_topk"]
    assert min(width, topk) >= 1
    body = lambda decoder: scenario_batch(decoder, rows, width)  # noqa: E731

    reference = run_world(worlds[1], body)[0]
    reference_positions = [world1_positions(worlds[1][0], layer, rows * width,
                                            torch.arange(rows).reshape(-1, 1).expand(rows, width).reshape(-1))
                           for layer in FULL_LAYERS]
    for world in WORLD_SIZES:
        results, recorded = capture_merged(worlds[world], world, body)
        token_rows = torch.arange(rows).reshape(-1, 1).expand(rows, width).reshape(-1)
        assert len(recorded[0]) == len(FULL_LAYERS)
        for index, layer in enumerate(FULL_LAYERS):
            merged = recorded[0][index]
            assert merged.shape == (rows * width, topk)
            for row in range(rows * width):
                want = reference_positions[index][row]
                got = sorted(int(p) for p in merged[row].tolist() if p >= 0)
                assert got == want, (
                    f"gate b (world {world}, layer {layer}, query {row}): "
                    f"merged set {got[:8]}... != single-rank {want[:8]}...")
            # The hand-back partitions the merged set per rank (per-row slot map).
            for rank_index in range(world):
                handed = handed_back_positions(worlds[world][rank_index], world, rank_index,
                                               layer, rows * width, token_rows)
                for row in range(rows * width):
                    want = [p for p in reference_positions[index][row] if p % world == rank_index]
                    assert handed[row] == want, (
                        f"gate b (world {world}, layer {layer}, rank {rank_index}, query "
                        f"{row}): hand-back {handed[row][:8]} != owned {want[:8]}")


def synthetic_index_scores(x, q_resid, positions, visible, key_table, slot_base, *, world, rank):
    """The `index_scores` contract over a synthetic global score field (gate b's tie legs)."""
    scores = SYNTHETIC["scores"]
    rows = int(visible.shape[0])
    owned = torch.div((visible.long() - rank).clamp_min(0) + world - 1, world,
                      rounding_mode="floor")
    width = int(owned.max())
    out = torch.full((rows, width), float("-inf"))
    for row in range(rows):
        live = int(owned[row])
        for j in range(live):
            out[row, j] = scores[j * world + rank]
    return out


SYNTHETIC = {}


def test_gate_b_top2048_ties_and_short_contexts(worlds):
    """Literal production scale: the W-rank merged top-2,048 equals the single-rank top-2,048.

    The score field carries tie plateaus of four (ties inside the selected set
    and below the cut; the 2,048 boundary falls between plateaus, so the
    selected SET is tie-order independent — the shipped merge's `torch.topk`
    tie order is unspecified and the gate must not depend on it). Contexts
    over top-k (3,000 positions) and under top-k (500) are both covered.
    """
    from ignition.recipes.hy4.collectives import merge_candidate_topk

    topk = 2048
    attn = worlds[1][0].weights.layers[0].attention
    saved_scores, saved_topk = attn.index_scores, attn.config.text["index_topk"]
    attn.index_scores = synthetic_index_scores
    attn.config.text["index_topk"] = topk
    try:
        for visible_count in (3000, 500):
            SYNTHETIC["scores"] = torch.tensor(
                [1000.0 - (p // 4) for p in range(visible_count)], dtype=torch.float32)
            visible = torch.tensor([visible_count], dtype=torch.long)
            want = sorted(torch.topk(SYNTHETIC["scores"], min(topk, visible_count)).indices.tolist())
            for world in WORLD_SIZES:
                candidates = [
                    attn.local_candidates(None, None, None, visible, None, None,
                                          world=world, rank=rank)
                    for rank in range(world)
                ]
                # The candidate contract is exactly `topk` wide per rank; the
                # short context's padding is `-inf`/`-1` and drops in the merge.
                _, merged = merge_candidate_topk(candidates, topk)
                got = sorted(int(p) for p in merged[0].tolist() if p >= 0)
                assert got == want, (
                    f"gate b top-2048 (world {world}, visible {visible_count}): "
                    f"{len(got)} merged entries != {len(want)} single-rank")
                assert len(got) == min(topk, visible_count)
    finally:
        attn.index_scores, attn.config.text["index_topk"] = saved_scores, saved_topk


# --------------------------------------------------------------------------
# gate c: sink-off, sink-per-rank and wrong-LSE-base mutants each FAIL
# --------------------------------------------------------------------------
def _gate_c_reference(worlds):
    return run_world(worlds[1], scenario_short)[0]


def mutant_run(worlds, world, mutant):
    from ignition.recipes.hy4.collectives import SimulatedGroup

    group = SimulatedGroup(world)
    decoders = fresh_decoders(worlds[world], group=group)
    for decoder in decoders:
        # Mutants shadow methods on the shared weight objects — private copies.
        decoder.weights = copy.deepcopy(decoder.weights)
    mutant(decoders)
    return group.run([lambda col, decoder=decoder: scenario_short(decoder)
                      for decoder in decoders])[0]


def assert_killed(worlds, world, mutant, label):
    reference = _gate_c_reference(worlds)
    poisoned = mutant_run(worlds, world, mutant)
    error = max(rel_error(got, want) for got, want in zip(poisoned, reference))
    assert error > 1e-3, (
        f"gate c mutant '{label}' (world {world}) SURVIVED: relative error "
        f"{error:.3e} is inside 1e-3 of world 1")


def mutant_sink_off(decoders):
    """No sink at all: the post-merge fold never runs."""
    for decoder in decoders:
        for layer in decoder.weights.layers:
            layer.attention.fold_sink = lambda out, lse: (out, lse)


def mutant_sink_per_rank_factory(full_sinks):
    """Each rank folds the sink into its own partial (the phantom sink row per rank).

    The mutant also misreports its provenance (`sink_folded=False`) — the
    silent variant of the defect: with truthful provenance the merge's
    `partials_sink_folded` guard rejects the double fold loudly, which
    `test_gate_c_double_sink_fold_is_rejected_loudly` pins; gate c kills the
    SILENT variant numerically.
    """
    def mutant(decoders):
        from ignition.recipes.hy4.attention import SparseMLAProvenance

        for decoder in decoders:
            for index, layer in enumerate(decoder.weights.layers):
                unit = layer.attention
                full = full_sinks[index]

                def attend(q, table, selected, counts, _unit=unit, _full=full,
                           _original=unit.attend, **_ignored):
                    saved = _unit.weights
                    _unit.weights = dataclasses.replace(saved, sink=_full)
                    try:
                        out = _original(q, table, selected, counts, fold_sink=True, expand=False)
                    finally:
                        _unit.weights = saved
                    return dataclasses.replace(
                        out, provenance=SparseMLAProvenance(sink_folded=False))

                unit.attend = attend
                unit.fold_sink = lambda out, lse: (out, lse)   # and never after the merge
    return mutant


def mutant_wrong_lse_base(decoders):
    """The natural-log merge over base-2 partials (the retired hard-code)."""
    for decoder in decoders:
        decoder.collectives.lse_base = "natural"


@pytest.mark.parametrize("world", WORLD_SIZES)
def test_gate_c_mutants_each_fail(worlds, world):
    assert_killed(worlds, world, mutant_sink_off, "sink-off")

    full_sinks = [torch.cat([rank.weights.layers[index].attention.weights.sink
                             for rank in worlds[world]]) for index in range(len(worlds[world][0].weights.layers))]
    assert_killed(worlds, world, mutant_sink_per_rank_factory(full_sinks), "sink-per-rank")
    assert_killed(worlds, world, mutant_wrong_lse_base, "wrong-LSE-base")


def test_gate_c_double_sink_fold_is_rejected_loudly():
    """A merge folding a sink over already-folded partials must raise (defect 11)."""
    from b12x.comm.pcie import lse_reduce_scatter_reference

    world, rows, heads, dim = 2, 1, 4, 8
    out = torch.zeros(world, rows, heads, dim)
    lse = torch.zeros(world, rows, heads)
    sink = torch.zeros(heads)
    with pytest.raises(ValueError, match="double sink fold"):
        lse_reduce_scatter_reference(out, lse, 0, lse_base="base2",
                                     attention_sink=sink, partials_sink_folded=True)
    with pytest.raises(ValueError, match="partials_sink_folded"):
        lse_reduce_scatter_reference(out, lse, 0, lse_base="base2",
                                     attention_sink=sink)


# --------------------------------------------------------------------------
# gate d: the ownership rule asserted in code (rank_local_slots is the contract)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("world", WORLD_SIZES)
def test_gate_d_ownership_matches_rank_local_slots(worlds, world):
    from b12x.attention.dsa_indexer.selection_handback import rank_local_slots

    from ignition.recipes.hy4.decoder import Hy4Cache

    config = worlds[1][0].model_config
    cache = Hy4Cache.allocate(config, rows=2, capacity=1024, device=torch.device("cpu"),
                              dtype=torch.float32, world=world, rank=0)
    positions = torch.arange(2048, dtype=torch.int64)   # two full 1,024-position pages
    owner, local = cache.owner(positions), cache.local(positions)
    assert torch.equal(owner, positions % world) and torch.equal(local, positions // world)
    for rank in range(world):
        slots = rank_local_slots(positions.reshape(1, -1), world_size=world, rank=rank).reshape(-1)
        expected = torch.where(positions % world == rank, positions // world,
                               torch.full_like(positions, -1)).to(torch.int32)
        assert torch.equal(slots, expected.to(slots.dtype))
    # The shared-page framing (amendment 2): one shared page is
    # `PAGE_TOKENS * world` global positions (1,024 at CP8 — the "full
    # 1,024-position page" of the design at world 8) and contributes exactly
    # 128 positions to every rank, at local slots [128k, 128(k+1)).
    page_global = 128 * world
    for page in range(2048 // page_global):
        page_positions = positions[page * page_global:(page + 1) * page_global]
        owned = page_positions[page_positions % world == 0]
        local_slots = owned // world
        assert int(owned.numel()) == 128
        assert int(local_slots.min()) == page * 128 and int(local_slots.max()) == page * 128 + 127

    # A batch of two rows: the per-row slot map equals rank_local_slots' map.
    token_rows = torch.tensor([0, 1], dtype=torch.long)
    slot_map = cache.local_slot_map(token_rows)
    merged = torch.tensor([[0, 3, 5, 7], [1, 2, 4, 6]], dtype=torch.int64)
    for rank in range(world):
        mapped = rank_local_slots(merged, world_size=world, rank=rank, slot_map=slot_map)
        for row in range(2):
            expected = [int(cache.slots_for(token_rows[row])) + int(p) // world
                        for p in merged[row].tolist() if p % world == rank]
            got = [int(v) for v in mapped[row].tolist() if v >= 0]
            assert got == expected, f"per-row slot map diverged at rank {rank} row {row}"


@pytest.mark.parametrize("world", WORLD_SIZES)
def test_gate_d_write_read_round_trip(worlds, world):
    """Write->read at world 2 and 8 finds every owned position (and nothing else)."""
    from ignition.recipes.hy4.decoder import Hy4Cache

    config = worlds[1][0].model_config
    width = config.text["kv_lora_rank"] + config.text["qk_rope_head_dim"]
    positions = torch.arange(24, dtype=torch.long)
    found = {}
    for rank in range(world):
        cache = Hy4Cache.allocate(config, rows=2, capacity=CAPACITY, device=torch.device("cpu"),
                                  dtype=torch.float32, world=world, rank=rank)
        for row in range(2):
            tokens = torch.full((24,), row, dtype=torch.long)
            values = torch.zeros(24, width)
            values[:, 0] = row * 1000 + positions.float()
            cache.write_latent(0, tokens, positions, values)   # stores OWNED only (D1)
            keep = (positions % world) == rank
            for p in positions[keep].tolist():
                slot = row * cache.local_capacity + p // world
                record = cache.latents(0)[slot]
                assert float(record[0]) == row * 1000 + p, f"lost ({row}, {p}) at rank {rank}"
                found[(row, p)] = found.get((row, p), 0) + 1
            for p in positions[~keep].tolist():
                slot = row * cache.local_capacity + p // world
                # Non-owned writes were dropped: the slot holds another owned
                # position's record or a zero, never this one.
                assert float(cache.latents(0)[slot][0]) != row * 1000 + p
    assert set(found) == {(row, p) for row in range(2) for p in positions.tolist()}, (
        "every position must be found by exactly its owning rank")
    assert all(count == 1 for count in found.values()), "a position must be stored exactly once"
