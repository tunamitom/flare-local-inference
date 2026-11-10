# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the W10 E2 prototypes: R9 temporary, union pattern, R10b fold.

Every test carries the R9 threat id from
``ignition/components/process_local_temporary.py`` (checklist carried from the
lever documents). Nothing here runs a GPU kernel, a collective or the allocation
microbenchmark (``scripts/benchmark_e2_allocation.py`` is ready but deliberately
not run: the window is CPU-only).
"""

from pathlib import Path
import importlib.util

import pytest
import torch

from ignition.components.distributed_topk import PreparedDistributedTopK, selected_local_records
from ignition.components.glue_union import GlueStage, GlueUnionPlan, PreparedGlueUnion
from ignition.components.process_local_temporary import (
    GlueScratch,
    ProcessLocalTemporary,
    R9_THREATS,
    TemporaryVariant,
    release_all_temporaries,
)
from ignition.recipes.kimi3 import e2_union
from ignition.recipes.kimi3.e2_decode_step_sketch import DecodeStepCaptureSketch
from ignition.recipes.kimi3.e2_head_folding import GateReport, head_folding_hook


@pytest.fixture(autouse=True)
def _clean_temporaries():
    release_all_temporaries()
    yield
    release_all_temporaries()


# -- R9: process-local temporary -----------------------------------------


def test_T9_variant_order_is_r9d_then_r9a_then_r9b():
    assert TemporaryVariant.ORDER == ("r9d", "r9a", "r9b")
    assert set(R9_THREATS) == {f"T{i}" for i in range(1, 11)}


def test_T3_T9_variants_are_bit_identical():
    catalog = (("exchange.pack", (4, 32), torch.float32), ("exchange.query", (4, 16), torch.bfloat16),
               ("aggregation.scores", (4, 8), torch.float32))
    results = {}
    for variant in TemporaryVariant.ORDER:
        temporary = ProcessLocalTemporary("test", "cpu", variant=variant, max_bytes=1 << 20)
        views = {}
        for name, shape, dtype in catalog:
            view = temporary.reserve(name, shape, dtype)
            values = torch.arange(view.numel(), dtype=torch.float32).reshape(shape)
            view.copy_(values.to(dtype))
            views[name] = view.clone()
        results[variant] = views
    for variant in TemporaryVariant.ORDER[1:]:
        for name, _, _ in catalog:
            assert torch.equal(results[variant][name], results[TemporaryVariant.POOL][name]), name


def test_T4_no_steady_state_allocation_in_the_round_loop():
    for variant in TemporaryVariant.ORDER:
        temporary = ProcessLocalTemporary("test", "cpu", variant=variant, max_bytes=1 << 20)
        temporary.reserve("exchange.pack", (8, 64), torch.float32)
        for _ in range(5):
            temporary.carve("exchange.pack", (8, 64), torch.float32)
        assert temporary.allocations == 1, variant
        first = temporary.reserve("exchange.pack", (8, 64), torch.float32)
        second = temporary.carve("exchange.pack", (8, 64), torch.float32)
        assert first.data_ptr() == second.data_ptr()
        assert temporary.allocations == 1, variant


def test_T4_re_reserving_a_different_extent_is_rejected():
    temporary = ProcessLocalTemporary("test", "cpu", variant=TemporaryVariant.POOL, max_bytes=1 << 20)
    temporary.reserve("exchange.pack", (8, 64), torch.float32)
    with pytest.raises(ValueError):
        temporary.reserve("exchange.pack", (8, 32), torch.float32)
    with pytest.raises(KeyError):
        temporary.carve("never.reserved", (2, 2), torch.float32)


def test_T10_upper_bound_admission_is_explicit():
    temporary = ProcessLocalTemporary("test", "cpu", variant=TemporaryVariant.POOL, max_bytes=512)
    temporary.reserve("exchange.pack", (8, 8), torch.float32)  # 256 B aligned
    with pytest.raises(ValueError, match="upper bound"):
        temporary.reserve("aggregation.scores", (8, 64), torch.float32)


def test_T4_single_reservation_and_two_stage_fallbacks_stay_within_their_contract():
    reservation = ProcessLocalTemporary("test", "cpu", variant=TemporaryVariant.RESERVATION, max_bytes=1 << 20)
    reservation.reserve("a", (8,), torch.float32)
    reservation.reserve("b", (8,), torch.float32)
    assert reservation.allocations == 1  # one multi-round reservation
    two_stage = ProcessLocalTemporary("test", "cpu", variant=TemporaryVariant.TWO_STAGE, max_bytes=4 << 20)
    two_stage.reserve("exchange.first", (8,), torch.float32)
    two_stage.reserve("exchange.second", (8,), torch.float32)
    two_stage.reserve("aggregation.first", (8,), torch.float32)
    # one coarse block per group, then silent stage-two carving inside it
    assert two_stage.allocations == 1


# -- union pattern: hoisted strides/offsets, copy-count elimination -------


def _union_plan():
    return GlueUnionPlan((
        GlueStage("scores", 4, 8, torch.float32, 0),
        GlueStage("query", 4, 4, torch.bfloat16, 4 * 8 * 4),
        GlueStage("counts", 4, 1, torch.int32, 4 * 8 * 4 + 4 * 4 * 2),
    ))


def test_copy_count_elimination_one_copy_per_element_and_hoisted_strides():
    plan = _union_plan()
    temporary = ProcessLocalTemporary("test", "cpu", variant=TemporaryVariant.POOL, max_bytes=1 << 20)
    union = PreparedGlueUnion(plan, temporary, name="decode")
    sources = {"scores": torch.arange(32, dtype=torch.float32).reshape(4, 8),
               "query": torch.arange(16, dtype=torch.float32).reshape(4, 4).to(torch.bfloat16),
               "counts": torch.arange(4, dtype=torch.int32)}
    union.pack(sources)
    assert union.copies == len(plan.stages)          # one copy per field, not per row
    assert union.copied_elements == plan.elements
    destinations = {name: torch.zeros_like(value) for name, value in sources.items()}
    union.unpack(destinations)
    assert union.copies == 2 * len(plan.stages)
    assert plan.recomputed_strides == 0              # strides/offsets hoisted out of the loop
    for name, value in sources.items():
        assert torch.equal(destinations[name], value), name  # T3: bit-identical bytes
    plan.unhoisted_copy(sources, union.staging)
    assert plan.recomputed_strides > 0               # the anti-pattern the union removed


def test_union_plan_rejects_overlaps_duplicates_and_bad_offsets():
    with pytest.raises(ValueError, match="overlap"):
        GlueUnionPlan((GlueStage("a", 4, 4, torch.float32, 0), GlueStage("b", 8, 4, torch.float32, 8)))
    with pytest.raises(ValueError, match="duplicate"):
        GlueUnionPlan((GlueStage("a", 4, 4, torch.float32, 0), GlueStage("a", 4, 4, torch.float32, 64)))
    with pytest.raises(ValueError, match="offset"):
        GlueUnionPlan((GlueStage("a", 4, 4, torch.float32, 1),))
    with pytest.raises(ValueError, match="at least one stage"):
        GlueUnionPlan(())


def test_union_pack_rejects_mismatched_fields():
    plan = _union_plan()
    temporary = ProcessLocalTemporary("test", "cpu", variant=TemporaryVariant.POOL, max_bytes=1 << 20)
    union = PreparedGlueUnion(plan, temporary, name="decode")
    sources = {"scores": torch.zeros(4, 4), "query": torch.zeros((4, 4), dtype=torch.bfloat16),
               "counts": torch.zeros(4, dtype=torch.int32)}
    with pytest.raises(ValueError):
        union.pack(sources)


# -- R10b fold into the union pattern ------------------------------------


def test_r10b_fold_splitk_funnel_rides_the_same_union(monkeypatch):
    outputs = {"kda_input": torch.arange(24, dtype=torch.float32).reshape(8, 3),
               "latent_down": torch.arange(8, dtype=torch.float32)}
    monkeypatch.setenv("IGNITION_KIMI_E2_UNION", "1")
    for variant in TemporaryVariant.ORDER:
        monkeypatch.setenv("IGNITION_KIMI_E2_R9_VARIANT", variant)
        union = e2_union.fold_splitk_funnel("layer0.projection", dict(outputs), device=torch.device("cpu"))
        union.pack(outputs)
        assert union.copies == len(outputs)
        back = {name: torch.zeros_like(value) for name, value in outputs.items()}
        union.unpack(back)
        for name, value in outputs.items():
            assert torch.equal(back[name], value), (variant, name)  # T3 across R9 variants


def test_r10b_fold_requires_the_union_flag():
    outputs = {"out": torch.zeros(4, 4)}
    with pytest.raises(RuntimeError, match="IGNITION_KIMI_E2_UNION"):
        e2_union.fold_splitk_funnel("layer0", outputs, device=torch.device("cpu"))


def test_killed_levers_are_never_engaged_by_the_union_path():
    # J-3: the R6 exchange merge and FP8 partials are KILLED; the union path must
    # refuse to run with them set (never re-add).
    assert e2_union.killed_levers_engaged() == []
    import os
    os.environ["IGNITION_KIMI_E2_UNION"] = "1"
    os.environ["IGNITION_KIMI_DSA_EXCHANGE_MERGE"] = "1"
    try:
        with pytest.raises(RuntimeError, match="killed lever"):
            e2_union.decode_temporary(torch.device("cpu"))
    finally:
        os.environ.pop("IGNITION_KIMI_E2_UNION", None)
        os.environ.pop("IGNITION_KIMI_DSA_EXCHANGE_MERGE", None)


def test_r9_variant_flag_order_and_validation():
    assert e2_union.e2_r9_variant() == "r9d"  # J-3: r9d first
    import os
    os.environ["IGNITION_KIMI_E2_R9_VARIANT"] = "r9b"
    try:
        assert e2_union.e2_r9_variant() == "r9b"
        os.environ["IGNITION_KIMI_E2_R9_VARIANT"] = "r9z"
        with pytest.raises(ValueError):
            e2_union.e2_r9_variant()
    finally:
        os.environ.pop("IGNITION_KIMI_E2_R9_VARIANT", None)


# -- exchange + aggregation scratch (dsa wiring shape) --------------------


class _CapturingMerge:
    def __init__(self, indices, scores, output, **kwargs):
        self.indices, self.scores, self.output = indices, scores, output
        self.scratch = kwargs.get("scratch")
        self.piggyback = kwargs.get("piggyback")


def _build_dsa(monkeypatch):
    from types import SimpleNamespace
    from ignition.recipes.kimi3 import dsa
    from ignition.recipes.kimi3.dcp import ContextShards, RECORD_BYTES

    monkeypatch.setattr(dsa, "prepare_projection", lambda *args, **kwargs: None)
    monkeypatch.setattr(dsa, "PreparedFP8Indexer", lambda *args, **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(dsa, "PreparedDistributedTopK", _CapturingMerge)
    monkeypatch.setattr(dsa, "PreparedDenseMLA", lambda *args, **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(dsa, "PreparedAllGather", lambda *args, **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(dsa, "PreparedLseReduceScatter", lambda *args, **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(dsa.dense_mla, "Caps", lambda **kwargs: SimpleNamespace(**kwargs))
    context = SimpleNamespace(shards=ContextShards.on(0, 2, "cpu"), candidates=object(), heads=object())
    cache = torch.zeros((2, 128, RECORD_BYTES), dtype=torch.float8_e4m3fn)
    return dsa.PreparedDSA(
        {"index_query": None, "index_norm": torch.ones(128), "index_bias": torch.zeros(128)},
        torch.zeros((2, 8)), torch.zeros((2, 128)), torch.zeros((2, 32)),
        torch.zeros((2, 1, RECORD_BYTES), dtype=torch.bfloat16),
        torch.zeros((2, 1, 512), dtype=torch.bfloat16), torch.zeros(2, dtype=torch.int64),
        SimpleNamespace(token_page_table=torch.zeros((2, 2), dtype=torch.int32)),
        cache, torch.zeros((4, 64 * 132), dtype=torch.uint8), context=context,
        session=None, coordinator=None, name="attention", workspace=None, buffers={})


def test_dsa_scratch_wiring_is_off_by_default_and_on_with_the_flag(monkeypatch):
    op = _build_dsa(monkeypatch)
    assert op.union_scratch is None
    assert all(stage[3].scratch is None for stage in op.stages)  # default wiring unchanged
    monkeypatch.setenv("IGNITION_KIMI_E2_UNION", "1")
    op = _build_dsa(monkeypatch)
    assert isinstance(op.union_scratch, GlueScratch)
    assert all(isinstance(stage[3].scratch, GlueScratch) for stage in op.stages)


def test_selected_local_records_out_contract_is_copy_count_elimination(monkeypatch):
    # The compaction kernel writes the caller's retained buffers directly (one
    # write per element) instead of allocating and copying again.
    import ignition.components.distributed_topk as distributed_topk

    class _FakeLaunch:
        def __getitem__(self, grid):
            return self

        def __call__(self, *args, **kwargs):
            args[4].fill_(7)
            args[5].fill_(3)

    monkeypatch.setattr(distributed_topk.torch.library, "wrap_triton", lambda kernel: _FakeLaunch())
    selected = torch.zeros(2, 4, dtype=torch.int32)
    positions = torch.zeros(2, dtype=torch.int64)
    pages = torch.zeros((2, 2), dtype=torch.int32)
    out = torch.zeros_like(selected)
    count_out = torch.zeros(2, dtype=torch.int32)
    returned_out, returned_count = selected_local_records(
        selected, positions, pages, world=2, rank=0, page_size=64, out=out, count_out=count_out)
    assert returned_out is out and returned_count is count_out  # no intermediate allocation
    assert torch.equal(out, torch.full_like(out, 7))
    with pytest.raises(ValueError):
        selected_local_records(selected, positions, pages, world=2, rank=0, page_size=64,
                               out=torch.zeros(2, 2, dtype=torch.int32))


def test_distributed_topk_scratch_path_is_bit_identical():
    # T3: the zero-allocation packing path writes exactly the values the original
    # per-call allocation path writes.
    def build(scratch):
        op = object.__new__(PreparedDistributedTopK)
        op.indices = torch.tensor([[5, -1, 9, 2], [-1, -1, 3, -1]], dtype=torch.int32)
        op.scores = torch.tensor([[1.5, -2.0, 3.25, 4.0], [0.5, 1.0, 2.0, 3.0]])
        op.world, op.rank = 4, torch.tensor([2], dtype=torch.int64)
        op.local = torch.zeros(2, 4, 2, dtype=torch.float32)
        op.paired = None
        op.collective = lambda packed: None
        op.merge = lambda: None
        op.output = torch.zeros(2, 4, dtype=torch.int32)
        op._live = op._packed_ids = None
        if scratch is not None:
            op._live = scratch.zeros("topk.live", (2, 4), torch.bool)
            op._packed_ids = scratch.zeros("topk.packed_ids", (2, 4), torch.int64)
        return op

    stock = build(None)
    stock()
    temporary = ProcessLocalTemporary("test", "cpu", variant=TemporaryVariant.POOL, max_bytes=1 << 20)
    united = build(GlueScratch(temporary))
    allocations_before = temporary.allocations
    united()
    assert temporary.allocations == allocations_before  # T4: round loop allocates nothing
    # T3: bit-identical packing (compare id slots as bits; -1 reads as NaN in float)
    assert torch.equal(united.local.view(torch.int32), stock.local.view(torch.int32))


# -- R7 hook + gate harness stub -----------------------------------------


def test_R7_hook_never_claims_without_both_gates():
    assert head_folding_hook()["folded"] is False and "gated" in head_folding_hook()["reason"]
    partial = GateReport(bit_equal={"1": True, "2": True, "3": True, "4": True, "6": True, "8": True},
                         latency={"1": 100.0})
    assert head_folding_hook(gate_report=partial)["folded"] is False
    assert head_folding_hook(gate_report="not a report")["folded"] is False
    passing = GateReport(bit_equal={str(w): True for w in (1, 2, 3, 4, 6, 8)},
                         latency={str(w): 100.0 for w in (1, 2, 3, 4, 6, 8)})
    engaged = head_folding_hook(gate_report=passing)
    assert engaged["folded"] is True and engaged["schedule"] == "per_head_small_m_folded"


def test_R7_gate_harness_stub_refuses_to_produce_a_report():
    spec = importlib.util.spec_from_file_location(
        "qualify_e2_head_folding", Path(__file__).resolve().parents[2] / "scripts" / "qualify_e2_head_folding.py")
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)
    with pytest.raises(NotImplementedError, match="spare-card"):
        harness.produce_report([])


# -- R4 sketch (engineering only, low priority per J-3) -------------------


def test_R4_decode_step_capture_is_a_sketch_only():
    sketch = DecodeStepCaptureSketch()
    boundaries = sketch.boundaries()
    assert "low priority" in boundaries["priority"]
    assert any("allocation-free" in requirement for requirement in boundaries["requires"])
    with pytest.raises(NotImplementedError):
        sketch.capture()
    with pytest.raises(NotImplementedError):
        sketch.replay()


# -- allocation microbench harness: threshold documented, not run ---------


def test_microbench_documents_the_radiance_gate_and_is_not_run():
    spec = importlib.util.spec_from_file_location(
        "benchmark_e2_allocation", Path(__file__).resolve().parents[2] / "scripts" / "benchmark_e2_allocation.py")
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)
    assert harness.RADIANCE_THRESHOLD == 0.10 and harness.ALLOCATION_THRESHOLD == 0.10
    # The radiance (all-reduce) bench refuses without the mechanics approval; the
    # allocation bench itself is deliberately not executed here (no GPU).
    assert harness.run_radiance_bench(torch.device("cpu"), approval=None) is None
    with pytest.raises(NotImplementedError, match="radiance-mechanics"):
        harness.run_radiance_bench(torch.device("cpu"), approval="j-mechanics-1")
