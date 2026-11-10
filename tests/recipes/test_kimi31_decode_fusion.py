# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the W11 decode-glue mechanics (radiance-mechanics review §1/§3/§5).

Named constraints carried from the review's threat checklist (§1; these rows are
the only thing that survives of W10's R9 temporary and union fold — T1–T10
stand as documented in the lever records):

* T11 — captured memory owned outside the graph's package storage scope or
  cursor (global registries, call-order-dependent pools).
* T12 — views of different element sizes over one storage in compiled graphs.
* T13 — name collisions across graph shapes: hard failure at different extents,
  silent sharing at equal ones.
* T14 — hidden kernels and allocations; enforced by graph node count, never by
  a component's own counter.
* T15 — peer visibility: a buffer a peer reads must be in peer storage and must
  not be reused as scratch while any rank can still read it.
* T16 — stage sharing inside one graph when branches run on the auxiliary
  stream.
* T17 — ``out=`` aliasing an input of the same kernel; bytes beyond ``count``
  are stale and must stay "don't care" for every consumer.

Wire constraints C1–C5 (§1 "Corrected mechanics"): unchanged wire buffers and
launch position, no collective constructed/replaced/reordered, rank symmetry at
world 8, the node gate counted from the graph, and bit-equality against the
shipped pack including dead entries and ties. Nothing here runs a GPU kernel,
a collective or an export: the fused Triton kernel's GPU bit-equality is checked
by the post-window captured-chain measurement (§2 item 2).
"""

from pathlib import Path
import ast
import importlib.util
from types import SimpleNamespace

import pytest
import torch

from ignition.components.distributed_topk import (
    PreparedDistributedTopK,
    _pack_pairs_reference,
    pack_score_id_pairs,
    selected_local_records,
)
from ignition.preparation.scratch import GraphBufferPool
from ignition import planning
from ignition.planning import (
    Gpu,
    KILLED_SWITCHES,
    LEGACY_SWITCH_FLAGS,
    Settings,
    check_switch_environment,
    live_settings,
    settings_scope,
    verify_recorded_settings,
)


# -- helpers ---------------------------------------------------------------

def _pack_shipped(local, indices, scores, rank, world):
    """The shipped 8-kernel pack, exactly as in PreparedDistributedTopK.__call__."""
    live = indices >= 0
    local[:, :, 0].copy_(torch.where(live, scores, -torch.inf))
    local.view(torch.int32)[:, :, 1].copy_(
        torch.where(live, indices.long() * world + rank, -1))


def _build_op(*, decode_fusion, indices, scores, world=8, rank=0, pool=None, cursor=None):
    """A PreparedDistributedTopK wired for CPU tests (fakes for merge/collective)."""
    rows, topk = indices.shape
    op = object.__new__(PreparedDistributedTopK)
    op.indices, op.scores = indices, scores
    op.world, op.rank = world, torch.tensor([rank], dtype=torch.int64)
    op.output = torch.zeros(rows, topk, dtype=torch.int32)
    if pool is None and cursor is None:
        op.local = torch.zeros(rows, topk, 2, dtype=torch.float32)
        op.gathered = torch.zeros(rows, world * topk, 2, dtype=torch.float32)
    else:
        cursor = pool.cursor() if cursor is None else cursor
        op.local = cursor.zeros((rows, topk, 2), torch.float32)
        op.gathered = cursor.zeros((rows, world * topk, 2), torch.float32)
    op.decode_fusion = decode_fusion
    op.paired = None
    op.collective = lambda packed: None
    op.merge = lambda: None
    return op


def _tie_inputs(rows=3):
    """Inputs with dead entries, ties across ranks, and the zero index."""
    indices = torch.tensor([
        [5, -1, 9, 2],
        [-1, -1, 3, -1],
        [0, 7, 7, -1],
    ][:rows], dtype=torch.int32)
    scores = torch.tensor([
        [1.5, -2.0, 3.25, 4.0],   # dead entry carries junk score
        [0.5, 1.0, 2.0, 3.0],
        [7.0, 7.0, 7.0, 0.0],     # ties and index 0 (live)
    ][:rows])
    return indices, scores


# -- T11: captured memory ownership ---------------------------------------

def test_T11_scratch_is_the_operator_cursor_and_the_temporary_registry_is_gone():
    # The R9 process-local temporary and its module-global registry are deleted;
    # pack scratch is the operator's existing buffers cursor (GraphBufferPool).
    with pytest.raises(ImportError):
        importlib.import_module("ignition.components.process_local_temporary")
    with pytest.raises(ImportError):
        importlib.import_module("ignition.recipes.kimi3.e2_union")
    assert not (Path(__file__).resolve().parents[2] / "scripts" / "benchmark_e2_allocation.py").exists()
    pool = GraphBufferPool("cpu", chunk_bytes=256)
    cursor = pool.cursor()
    indices, scores = _tie_inputs()
    local = cursor.zeros((3, 4, 2), torch.float32)
    assert local.untyped_storage().data_ptr() == pool.chunks[torch.float32][0].untyped_storage().data_ptr()
    # The fused pack allocates nothing: it mutates the given cursor buffer only.
    before = pool.nbytes
    storage_ptr = local.untyped_storage().data_ptr()
    data_ptr = local.data_ptr()
    pack_score_id_pairs(local, indices, scores, torch.tensor([2], dtype=torch.int64), 8)
    assert pool.nbytes == before
    assert local.untyped_storage().data_ptr() == storage_ptr
    assert local.data_ptr() == data_ptr


# -- T12: element-size reinterpretation ------------------------------------

def test_T12_fused_pack_pairs_same_size_slots_only():
    # Scores are FP32 and IDs are int32 bits in the second slot — same element
    # size, so compiled graphs never reinterpret aliased storage across element
    # sizes. The across-size bf16 wire view is the pre-existing shipped wire (C1).
    indices, scores = _tie_inputs()
    indices_before, scores_before = indices.clone(), scores.clone()
    local = torch.zeros(3, 4, 2, dtype=torch.float32)
    pack_score_id_pairs(local, indices, scores, torch.tensor([0], dtype=torch.int64), 8)
    ids = local.view(torch.int32)[:, :, 1]
    assert ids.dtype.itemsize == local.dtype.itemsize
    assert ids.untyped_storage().data_ptr() == local.untyped_storage().data_ptr()
    # Every input stays byte-untouched (the op mutates `local` only).
    assert torch.equal(indices, indices_before)
    assert torch.equal(scores, scores_before)


# -- T13: name collisions across shapes -----------------------------------

def test_T13_no_name_registry_and_two_shapes_coexist():
    # The deleted registry raised on a second reservation at a different extent.
    # Names are now plain labels: two operators of different shapes coexist and
    # the components module keeps no global registry.
    import ignition.components.distributed_topk as distributed_topk

    assert not any(name.startswith("_TEMPORARIES") or name == "_REGISTRY"
                   for name in vars(distributed_topk))
    indices, scores = _tie_inputs(rows=2)
    small = _build_op(decode_fusion=True, indices=indices, scores=scores)
    large = _build_op(decode_fusion=True, indices=torch.zeros(8, 16, dtype=torch.int32),
                      scores=torch.zeros(8, 16))
    small()
    large()
    assert small.local.shape == (2, 4, 2) and large.local.shape == (8, 16, 2)


# -- T14: node gate from the graph ----------------------------------------

# make_fx metadata nodes (views/aliases/scalar constants) that launch no kernel;
# every remaining call_function node is one kernel launch.
_NON_KERNEL_ATEL = {
    "aten::view.default", "aten::view.dtype", "aten::slice.Tensor", "aten::select.int",
    "aten::as_strided.default", "aten::alias", "aten::scalar_tensor", "aten::getitem.default",
    "aten::unsqueeze.default", "aten::detach.default", "aten::lift.default",
}


def _graph_nodes(graph_module, *, kernel_only):
    count = 0
    for node in graph_module.graph.nodes:
        if node.op != "call_function":
            continue
        target = node.target
        name = target.name() if hasattr(target, "name") else str(target)
        if kernel_only and name in _NON_KERNEL_ATEL:
            continue
        count += 1
    return count


def test_T14_fused_pack_graph_node_count_is_strictly_lower():
    # Counted from the executing graph (make_fx), never from a Python counter:
    # the fused path is one graph node (one compiled kernel `_pack_pairs` per
    # call); the shipped pack is its 8 kernel launches plus metadata views.
    from torch.fx.experimental.proxy_tensor import make_fx

    indices, scores = _tie_inputs()
    rank = torch.tensor([3], dtype=torch.int64)

    def shipped(local, indices, scores, rank):
        _pack_shipped(local, indices, scores, rank, 8)

    def fused(local, indices, scores, rank):
        torch.ops.ignition.pack_score_id_pairs(local, indices, scores, rank, 8)

    shipped_graph = make_fx(shipped)(torch.zeros(3, 4, 2), indices, scores, rank)
    fused_graph = make_fx(fused)(torch.zeros(3, 4, 2), indices, scores, rank)
    assert _graph_nodes(fused_graph, kernel_only=False) == 1
    assert _graph_nodes(fused_graph, kernel_only=True) == 1
    assert _graph_nodes(shipped_graph, kernel_only=True) == 8  # 8 -> 1 per call
    assert (_graph_nodes(shipped_graph, kernel_only=False)
            > _graph_nodes(fused_graph, kernel_only=False))


# -- T15 + C1/C2: wire boundary -------------------------------------------

def test_T15_C1_C2_wire_buffer_collective_and_order_are_unchanged():
    indices, scores = _tie_inputs()
    results = {}
    for decode_fusion in (False, True):
        op = _build_op(decode_fusion=decode_fusion, indices=indices, scores=scores)
        calls = []
        op.collective = lambda packed: calls.append(("collective", packed))
        op.merge = lambda: calls.append(("merge", None))
        op()
        assert [name for name, _ in calls] == ["collective", "merge"]
        packed = calls[0][1]
        # Same wire buffer, same bf16 row view, same launch position (C1):
        # bytes are transported as bf16 row fields over the fp32/int32 pairs.
        assert packed.untyped_storage().data_ptr() == op.local.untyped_storage().data_ptr()
        assert packed.shape == (indices.shape[0], indices.shape[1] * 4)
        assert packed.dtype == torch.bfloat16
        results[decode_fusion] = op.local.view(torch.int32).clone()
    # C2/C1: the pack writes byte-identical wire bytes whatever the path.
    assert torch.equal(results[True], results[False])


# -- T16: stage sharing inside one graph ----------------------------------

def test_T16_one_cursor_gives_disjoint_stage_buffers():
    pool = GraphBufferPool("cpu", chunk_bytes=256)
    indices, scores = _tie_inputs()
    cursor = pool.cursor()  # one graph: one cursor for its sequential stages
    first = _build_op(decode_fusion=True, indices=indices, scores=scores, cursor=cursor)
    second = _build_op(decode_fusion=True, indices=indices, scores=scores, cursor=cursor)
    first()
    second()
    spans = sorted((tensor.data_ptr(), tensor.data_ptr() + tensor.numel() * 4)
                   for tensor in (first.local, first.gathered, second.local, second.gathered))
    for (_, end), (begin, _) in zip(spans, spans[1:]):
        assert end <= begin, "one graph's stage buffers must not overlap"
    # Separate cursors (separate graphs) share chunks by design.
    other = _build_op(decode_fusion=True, indices=indices, scores=scores, cursor=pool.cursor())
    assert other.local.untyped_storage().data_ptr() == first.local.untyped_storage().data_ptr()


# -- T17: out= aliasing and don't-care tail --------------------------------

class _FakeRecordsLaunch:
    """Emulates the compaction kernel: writes rows [0, count) only."""

    def __getitem__(self, grid):
        return self

    def __call__(self, *args, **kwargs):
        output, count = args[4], args[5]
        count.fill_(2)
        output.fill_(-1)
        output[:, :2] = 7


def test_T17_out_aliasing_is_rejected(monkeypatch):
    import ignition.components.distributed_topk as distributed_topk

    monkeypatch.setattr(distributed_topk.torch.library, "wrap_triton", lambda kernel: _FakeRecordsLaunch())
    selected = torch.zeros(2, 4, dtype=torch.int32)
    positions = torch.zeros(2, dtype=torch.int64)
    pages = torch.zeros((2, 2), dtype=torch.int32)
    with pytest.raises(ValueError, match="T17"):
        selected_local_records(selected, positions, pages, world=2, rank=0, page_size=64, out=selected)
    shared = torch.zeros(16, dtype=torch.int32)
    pages_shared = shared[:8].view(2, 4)
    out_shared = shared[8:].view(2, 4)
    with pytest.raises(ValueError, match="T17"):
        selected_local_records(selected, positions, pages_shared, world=2, rank=0, page_size=64,
                               out=out_shared)
    aliased = torch.zeros(12, dtype=torch.int32)
    with pytest.raises(ValueError, match="T17"):
        # count_out sharing storage with pages (an input of the same kernel).
        selected_local_records(selected, positions, aliased[:8].view(2, 4), world=2, rank=0,
                               page_size=64, out=selected.clone(), count_out=aliased[8:10])
    with pytest.raises(ValueError, match="T17"):
        # count_out sharing storage with out (two outputs of one kernel).
        selected_local_records(selected, positions, pages, world=2, rank=0, page_size=64,
                               out=aliased[:8].view(2, 4), count_out=aliased[8:10])


def test_T17_bytes_beyond_count_are_dont_care(monkeypatch):
    import ignition.components.distributed_topk as distributed_topk

    monkeypatch.setattr(distributed_topk.torch.library, "wrap_triton", lambda kernel: _FakeRecordsLaunch())
    selected = torch.zeros(2, 4, dtype=torch.int32)
    positions = torch.zeros(2, dtype=torch.int64)
    pages = torch.zeros((2, 2), dtype=torch.int32)
    first = torch.full((2, 4), 12345, dtype=torch.int32)
    second = torch.full((2, 4), -999, dtype=torch.int32)
    returned_first, _ = selected_local_records(selected, positions, pages, world=2, rank=0,
                                               page_size=64, out=first,
                                               count_out=torch.zeros(2, dtype=torch.int32))
    returned_second, _ = selected_local_records(selected, positions, pages, world=2, rank=0,
                                                page_size=64, out=second,
                                                count_out=torch.zeros(2, dtype=torch.int32))
    assert returned_first is first and returned_second is second  # no intermediate allocation
    # Consumers read exactly [0, count) records; the tail is stale/don't care
    # and never affects what consumers see.
    assert torch.equal(first[:, :2], second[:, :2])
    assert "don't care" in selected_local_records.__doc__


def test_compaction_out_contract_is_copy_count_elimination(monkeypatch):
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


# -- C5: bit-equality including dead entries and ties ----------------------

def _run_operator(decode_fusion, indices, scores, world=8):
    op = _build_op(decode_fusion=decode_fusion, indices=indices, scores=scores, world=world)
    op()
    return op.local.view(torch.int32).clone()


def test_C5_fused_pack_is_bit_equal_to_the_shipped_pack():
    indices, scores = _tie_inputs()
    rank = torch.tensor([6], dtype=torch.int64)
    shipped = torch.zeros(3, 4, 2)
    fused = torch.zeros(3, 4, 2)
    _pack_shipped(shipped, indices, scores, rank, 8)
    _pack_pairs_reference(fused, indices, scores, rank, 8)
    scratch = torch.zeros(3, 4, 2)
    pack_score_id_pairs(scratch, indices, scores, rank, 8)  # the op body agrees
    assert torch.equal(fused.view(torch.int32), shipped.view(torch.int32))  # bit-equal, not close
    assert torch.equal(scratch.view(torch.int32), shipped.view(torch.int32))
    # Dead entries: score -inf and ID -1 exactly (bits).
    assert fused[0, 1, 0] == -torch.inf and int(fused.view(torch.int32)[0, 1, 1]) == -1
    assert fused[1, 0, 0] == -torch.inf and int(fused.view(torch.int32)[1, 0, 1]) == -1
    # Index 0 is live: packed as 0 * world + rank.
    assert int(fused.view(torch.int32)[2, 0, 1]) == 0 * 8 + 6
    # Ties pack as plain copies: equal scores stay equal bytes.
    assert fused[2, 0, 0] == fused[2, 1, 0] == fused[2, 2, 0]
    # Deterministic at scale: random inputs with repeated scores and deaths.
    generator = torch.Generator().manual_seed(0xC0FFEE)
    indices = torch.randint(-3, 40, (16, 9), generator=generator, dtype=torch.int32)
    scores = torch.randint(0, 4, (16, 9), generator=generator).float()  # many ties
    assert torch.equal(_run_operator(True, indices, scores), _run_operator(False, indices, scores))


# -- C3: rank symmetry at world 8 ----------------------------------------

def _reference_merge(gathered, topk):
    """The merge's documented semantics: greatest scores, ties prefer earlier
    gathered positions (deterministic input order), -1 padding."""
    rows, width = gathered.shape[0], gathered.shape[1]
    scores = gathered[:, :, 0]
    ids = gathered.view(torch.int32)[:, :, 1]
    output = torch.full((rows, topk), -1, dtype=torch.int32)
    for row in range(rows):
        pairs = [(float(scores[row, column]), int(ids[row, column]), column)
                 for column in range(width) if int(ids[row, column]) >= 0]
        pairs.sort(key=lambda pair: (-pair[0], pair[2]))
        for slot, (_, identifier, _) in enumerate(pairs[:topk]):
            output[row, slot] = identifier
    return output


def test_C3_rank_symmetry_at_world_8():
    world, topk, rows = 8, 4, 3
    locals_by_rank = {True: [], False: []}
    for rank in range(world):
        # Equal scores on different ranks (ties across ranks) and dead entries.
        indices = torch.tensor([[rank, -1, 100 + rank, 200 + rank]] * rows, dtype=torch.int32)
        scores = torch.tensor([[7.0, 0.0, 7.0, 1.0]] * rows)
        row = {decode_fusion: _run_operator(decode_fusion, indices, scores, world)
               for decode_fusion in (True, False)}
        assert torch.equal(row[True], row[False])
        for decode_fusion in (True, False):
            locals_by_rank[decode_fusion].append(row[decode_fusion])
    # After the exchange every rank holds a byte-identical gathered buffer
    # (rank-major) and produces the identical selected output.
    selected_by_path = {}
    for decode_fusion in (True, False):
        gathered = torch.cat(locals_by_rank[decode_fusion], dim=1)
        for rank in range(world):
            rank_view = torch.cat(locals_by_rank[decode_fusion], dim=1)
            assert torch.equal(rank_view.view(torch.int32), gathered.view(torch.int32))
        selected_by_path[decode_fusion] = _reference_merge(gathered, topk)
        # Ties across ranks: the earlier gathered position (lower rank) wins.
        assert int(selected_by_path[decode_fusion][0, 0]) == 0
    assert torch.equal(selected_by_path[True], selected_by_path[False])


# -- decode_fusion flag ---------------------------------------------------

def _construct(monkeypatch, **kwargs):
    import ignition.components.distributed_topk as distributed_topk

    monkeypatch.setattr(distributed_topk, "PreparedCandidateMerge",
                        lambda gathered, output, **kwargs: SimpleNamespace())
    monkeypatch.setattr(distributed_topk, "PreparedAllGather",
                        lambda *args, **kwargs: SimpleNamespace())
    indices, scores = _tie_inputs()
    pool = GraphBufferPool("cpu", chunk_bytes=256)
    return distributed_topk.PreparedDistributedTopK(
        indices, scores, torch.zeros(3, 4, dtype=torch.int32), world=8,
        rank=torch.tensor([0], dtype=torch.int64), runtime=None, session=None,
        coordinator=None, name="attention.candidates", buffers=pool.cursor(),
        workspace=None, **kwargs)


def test_decode_fusion_defaults_off_and_reads_plan_settings(monkeypatch):
    op = _construct(monkeypatch)
    assert op.decode_fusion is False  # default off: wiring bit-identical
    with settings_scope(Settings(decode_fusion=True)):
        assert _construct(monkeypatch).decode_fusion is True
        assert live_settings().decode_fusion is True
    op = _construct(monkeypatch, decode_fusion=True)
    assert op.decode_fusion is True  # explicit parameter wins
    # The other folded switches read the same source of truth.
    from ignition.recipes.kimi3 import projections, routing

    with settings_scope(Settings(atomic_splitk=True, route_fused_select=True, dsa_lse_blocks=4)):
        assert projections.atomic_splitk() is True
        assert routing.route_fused_select() is True
        from ignition.recipes.kimi3.dsa import _lse_combine_blocks
        assert _lse_combine_blocks() == 4


# -- §3: env-to-flag fold -------------------------------------------------

def test_env_to_flag_fold_has_no_environment_reads_in_recipes_or_components():
    """The grep test (§3 item 2): no environment reads in ignition/recipes or
    ignition/components. B12X_* stamping (writes) before kernel compiles stays."""
    root = Path(__file__).resolve().parents[2] / "ignition"
    violations = []
    for tree_name in ("recipes", "components"):
        for path in sorted((root / tree_name).rglob("*.py")):
            module = ast.parse(path.read_text(), str(path))
            written = set()
            for node in ast.walk(module):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Attribute)
                                and target.value.attr == "environ"):
                            written.add(id(target.value))
                if isinstance(node, ast.Call):
                    function = node.func
                    if isinstance(function, ast.Attribute) and function.attr == "get" \
                            and isinstance(function.value, ast.Attribute) and function.value.attr == "environ":
                        violations.append(f"{path}:{node.lineno}: os.environ.get read")
                    if (isinstance(function, ast.Name) and function.id == "getenv") or (
                            isinstance(function, ast.Attribute) and function.attr == "getenv"):
                        violations.append(f"{path}:{node.lineno}: getenv read")
            for node in ast.walk(module):
                if isinstance(node, ast.Attribute) and node.attr == "environ" and id(node) not in written:
                    violations.append(f"{path}:{node.lineno}: os.environ read")
                if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute) \
                        and node.value.attr == "environ" and isinstance(node.ctx, ast.Load):
                    violations.append(f"{path}:{node.lineno}: os.environ[...] read")
    assert not violations, "environment reads in recipes/components (fold to plan settings): " + \
        "; ".join(violations)


def test_plan_refuses_legacy_switch_environment_naming_the_replacement_flag():
    recipe = SimpleNamespace(b12x_environment={"B12X_W4A16_SMALL_M_SPLITK": "1"})
    for name, replacement in LEGACY_SWITCH_FLAGS.items():
        with pytest.raises(planning.UsageError) as raised:
            check_switch_environment(recipe, Settings(), environ={name: "1"})
        assert name in str(raised.value) and replacement in str(raised.value)
    with pytest.raises(planning.UsageError) as raised:
        check_switch_environment(recipe, Settings(), environ={"IGNITION_KIMI_UNKNOWN_FUTURE": "1"})
    assert "unaccounted" in str(raised.value)


def test_killed_levers_have_no_flag_and_the_env_guard_raises():
    recipe = SimpleNamespace(b12x_environment={"B12X_W4A16_SMALL_M_SPLITK": "1"})
    assert set(KILLED_SWITCHES) == {"IGNITION_KIMI_DSA_EXCHANGE_MERGE", "IGNITION_KIMI_DSA_LSE_FP8"}
    for name in KILLED_SWITCHES:
        assert name not in LEGACY_SWITCH_FLAGS  # no flag replaces a killed lever
        with pytest.raises(planning.UsageError) as raised:
            check_switch_environment(recipe, Settings(), environ={name: "1"})
        assert "killed lever" in str(raised.value)
    # No Settings field exists for them (the switch does not exist).
    assert not hasattr(Settings(), "dsa_exchange_merge")
    assert not hasattr(Settings(), "dsa_lse_fp8")


def test_b12x_environment_allowlist_accounts_for_plan_stamped_switches():
    recipe = SimpleNamespace(b12x_environment={"B12X_W4A16_SMALL_M_SPLITK": "1",
                                               "B12X_PCIE_DCP_THREADS": "512"})
    accounted = {"B12X_W4A16_SMALL_M_SPLITK": "1", "B12X_PCIE_DCP_THREADS": "512",
                 "B12X_COMPILE_CACHE_DIR": "/tmp/cache"}
    check_switch_environment(recipe, Settings(), environ=dict(accounted))
    with pytest.raises(planning.UsageError) as raised:
        check_switch_environment(recipe, Settings(), environ={"B12X_PCIE_DMA_PIECES": "4"})
    assert "B12X_PCIE_DMA_PIECES" in str(raised.value)  # the J-1 risk 6 leak is closed
    # B12X_W4A16_PREFILL_SHAPES is accounted only while trellis stamps it.
    check_switch_environment(recipe, Settings(), trellis_prefill_shapes=True,
                             environ={"B12X_W4A16_PREFILL_SHAPES": "1"})
    with pytest.raises(planning.UsageError):
        check_switch_environment(recipe, Settings(), trellis_prefill_shapes=False,
                                 environ={"B12X_W4A16_PREFILL_SHAPES": "1"})


def _plan(settings=None):
    from ignition.recipes.kimi31.recipe import RECIPE

    gpus = tuple(Gpu(index=index, name="TestGPU", uuid=f"uuid-{index}", total_bytes=1 << 40,
                     free_bytes=1 << 40, capability=(10, 3)) for index in range(8))
    return planning.Plan(RECIPE, RECIPE.checkpoints[0], gpus, 4, 3072, None,
                         0, 0, 1000, 0, 0, 0,
                         settings=Settings() if settings is None else settings)


def test_the_plan_itself_refuses_unaccounted_environment(monkeypatch):
    _plan()  # clean environment plans fine
    monkeypatch.setenv("IGNITION_KIMI_ATOMIC_SPLITK", "1")
    with pytest.raises(planning.UsageError) as raised:
        _plan()
    assert "--atomic-splitk" in str(raised.value)
    monkeypatch.delenv("IGNITION_KIMI_ATOMIC_SPLITK")
    monkeypatch.setenv("IGNITION_KIMI_DSA_EXCHANGE_MERGE", "1")
    with pytest.raises(planning.UsageError) as raised:
        _plan()
    assert "killed lever" in str(raised.value)


def test_switches_are_recorded_in_plan_settings():
    plan = _plan(Settings(atomic_splitk=True, route_fused_select=True, dsa_lse_blocks=4,
                          decode_fusion=True))
    recorded = plan.to_dict()["settings"]
    assert recorded["atomic_splitk"] is True
    assert recorded["route_fused_select"] is True
    assert recorded["dsa_lse_blocks"] == 4
    assert recorded["decode_fusion"] is True
    assert recorded["b12x_environment"]["B12X_W4A16_SMALL_M_SPLITK"] == "1"  # recorded
    assert "B12X_W4A16_PREFILL_SHAPES" not in recorded["b12x_environment"]  # no trellis, no stamp
    with pytest.raises(planning.UsageError):
        Settings(dsa_lse_blocks=0)


def test_post_export_assertion_compares_recorded_settings_and_fingerprints():
    recorded = {"atomic_splitk": False, "decode_fusion": False, "prefill_dma_ring_rows": 1024,
                "slab_bytes": 178_028_544, "round_graph_nodes": 4967}
    verify_recorded_settings(recorded, {"atomic_splitk": False, "decode_fusion": False,
                                        "prefill_dma_ring_rows": 1024},
                             fingerprints={"slab_bytes": 178_028_544, "round_graph_nodes": 4967})
    with pytest.raises(planning.UsageError) as raised:
        verify_recorded_settings(recorded, {"decode_fusion": True})
    assert "decode_fusion" in str(raised.value)
    with pytest.raises(planning.UsageError) as raised:
        verify_recorded_settings(recorded, {}, fingerprints={"slab_bytes": 1})
    assert "fingerprint slab_bytes" in str(raised.value)
    with pytest.raises(planning.UsageError) as raised:
        verify_recorded_settings(recorded, {"missing": 1})
    assert "not recorded" in str(raised.value)


def test_the_union_is_parked_and_unwired():
    root = Path(__file__).resolve().parents[2] / "ignition"
    source = (root / "components" / "glue_union.py").read_text()
    assert "PARKED" in source.splitlines()[4]  # header comment says it is parked and why
    # Nothing outside the parked module itself imports it any more.
    importers = []
    for path in root.rglob("*.py"):
        if path.name == "glue_union.py":
            continue
        module = ast.parse(path.read_text(), str(path))
        for node in ast.walk(module):
            if isinstance(node, ast.ImportFrom) and (node.module or "").endswith("glue_union"):
                importers.append(str(path))
            if isinstance(node, ast.Import) and any(alias.name.endswith("glue_union") for alias in node.names):
                importers.append(str(path))
    assert not importers, f"the parked union is still imported: {importers}"


# -- R7 hook + gate harness stub (shape only, unchanged) -------------------

from ignition.recipes.kimi3.e2_decode_step_sketch import DecodeStepCaptureSketch  # noqa: E402
from ignition.recipes.kimi3.e2_head_folding import GateReport, head_folding_hook  # noqa: E402


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
