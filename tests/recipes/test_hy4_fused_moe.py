# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""The M2 fused routed-experts wiring: the `MoEBlock._prepared` seam, the
call-shaped output contract, and the resident-slice loader aliases.

CPU only: the b12x CUDA boundary (`PreparedRoutedExperts`,
`prepared_moe_out`, `load_fp4_experts`) is stubbed to capture exactly what the
runtime would receive — the same discipline as
`test_hy4_expert_loading.py`. The fused-vs-reference NUMERIC gate runs on the
spare GPU card against the real package (`scratch/gate_m2_fused_numerics.py`);
nothing here touches a GPU.
"""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from ignition.recipes.hy4 import contract
from ignition.recipes.hy4.moe import FusedRoutedExperts, FusedWireScratch, MoEBlock
from ignition.recipes.hy4.weights import Hy4WeightLoader

from tests.recipes.test_hy4_expert_loading import (COUNT, HIDDEN, INTER, PREFIX,
                                                   SWIGLU_LIMIT, make_loader,
                                                   packed_checkpoint)

LOCAL = INTER // 2          # one rank's intermediate slice (world 2)


class FakePreparationSession:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeOp:
    """Stand-in for `ignition.components.moe.PreparedRoutedExperts`."""

    def __init__(self, source, experts, ids, weights, output, *, session, name,
                 workspace=None):
        self.source, self.experts = source, experts
        self.ids, self.weights, self.output = ids, weights, output
        self.name = name
        self.workspace_nbytes = 4096
        self.scale_layouts = (((3, 2), (2, 1), torch.float32),)
        self.scale_inputs = ("scale-inputs",)
        self.plan = SimpleNamespace(handle="plan-handle")
        self.precision = {"a4_prefill": False}
        if workspace is not None:
            # The real op's eager binding finalizer sizes a `ScratchArena`
            # through `planned_scratch` -> `views` -> `reserve` before the
            # wrapper adopts it; mirror that one effect.
            workspace.reserve(self.workspace_nbytes)

    def functional(self, *, session):
        self.functional_session = session
        return self


class FakeCall:
    def __init__(self):
        self.calls = []

    def __call__(self, a, weights, ids, output, scratch, scale_inputs, storage,
                 handle, input_scales_static, fast_math, activation_amax,
                 layer_idx, route_expert_map, output_expert_map, scales_expanded,
                 a4_prefill):
        self.calls.append({"a": a, "weights": weights, "ids": ids, "output": output,
                           "scratch": scratch, "scale_inputs": scale_inputs,
                           "storage": storage, "handle": handle,
                           "a4_prefill": a4_prefill})
        assert a.shape[0] == ids.shape[0] == weights.shape[0] == output.shape[0]
        output.fill_(1.0)


@pytest.fixture()
def fake_boundary(monkeypatch):
    """Stub the two CUDA-bound b12x entry points the wrapper touches."""
    import b12x.moe.fused_moe._binding_ops as binding_ops
    import b12x.preparation
    import ignition.components.moe as components_moe

    call = FakeCall()
    monkeypatch.setattr(components_moe, "PreparedRoutedExperts", FakeOp)
    monkeypatch.setattr(b12x.preparation, "PreparationSession", FakePreparationSession)
    monkeypatch.setattr(binding_ops, "prepared_moe_out", call)
    return call


def fused(experts="experts", *, hidden=HIDDEN, top_k=2):
    return FusedRoutedExperts(experts, hidden=hidden, top_k=top_k,
                              device=torch.device("cpu"), name="layer1.experts.fused")


# --- the no-JIT-at-run-time contract ---------------------------------------

def test_unwired_fused_experts_refuse_run_time_planning():
    op = fused()
    x = torch.zeros((2, HIDDEN), dtype=torch.bfloat16)
    ids = torch.zeros((2, 2), dtype=torch.int64)
    weights = torch.full((2, 2), 0.5)
    with pytest.raises(RuntimeError, match="no JIT at run time"):
        op.forward(x, ids, weights)
    with pytest.raises(RuntimeError, match="no JIT at run time"):
        op(x, ids=ids, weights=weights)
    assert not op.wired


# --- wire: the prepared plan binds at wire_operators time -------------------

def test_wire_builds_one_prepared_op_and_warms_the_extremes(fake_boundary):
    call = fake_boundary
    op = fused()
    op.wire(session="prepared", max_tokens=16)   # the recipe's sentinel form
    assert op.wired and op.max_tokens == 16
    assert isinstance(op.op, FakeOp)
    assert op.op.name == "layer1.experts.fused"
    assert op._workspace.shape == (4096,) and op._workspace.dtype == torch.uint8
    assert [tuple(t.shape) for t in op._storage] == [(3, 2)]
    # `warm` (run by wire_operators after wire) exercises the extremes
    op.warm()
    assert [c["a"].shape[0] for c in call.calls] == [1, 16]
    assert op.calls == 0                      # warm is not a model call
    # wire is idempotent: the prepared plan is not rebuilt
    before = op.op
    op.wire(session="prepared", max_tokens=64)
    assert op.op is before and op.max_tokens == 16


# --- the shared fused wire scratch (consult 13 patch A) ----------------------

def test_shared_wire_scratch_gives_every_op_one_workspace(fake_boundary):
    scratch = FusedWireScratch(device=torch.device("cpu"), max_tokens=16,
                               hidden=HIDDEN, top_k=2)
    first, second = fused(), fused()
    first.name = "layer1.experts.fused"
    second.name = "layer2.experts.fused"
    first.wire(session="prepared", max_tokens=16, scratch=scratch)
    second.wire(session="prepared", max_tokens=16, scratch=scratch)
    # both ops ride ONE arena buffer and ONE scale-storage list
    assert first._workspace.data_ptr() == second._workspace.data_ptr()
    assert first._workspace.data_ptr() == scratch.arena.tensor.data_ptr()
    assert first._storage[0] is second._storage[0]
    assert scratch.nbytes == first.op.workspace_nbytes
    scratch.seal()                       # every layer wired: growth forbidden
    assert scratch.arena.frozen


def test_shared_wire_scratch_refuses_a_different_plan(fake_boundary, monkeypatch):
    import ignition.components.moe as components_moe

    scratch = FusedWireScratch(device=torch.device("cpu"), max_tokens=16,
                               hidden=HIDDEN, top_k=2)
    first = fused()
    first.wire(session="prepared", max_tokens=16, scratch=scratch)

    class BiggerOp(FakeOp):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.workspace_nbytes = 8192

    monkeypatch.setattr(components_moe, "PreparedRoutedExperts", BiggerOp)
    second = fused()
    with pytest.raises(ValueError, match="sized for a different plan"):
        second.wire(session="prepared", max_tokens=16, scratch=scratch)


def test_forward_shapes_and_route_contract(fake_boundary):
    call = fake_boundary
    op = fused()
    op.wire(session="prepared", max_tokens=8)
    call.calls.clear()
    x = torch.zeros((5, HIDDEN), dtype=torch.bfloat16)
    ids = torch.arange(10).reshape(5, 2)      # the router's int64 indices
    weights = torch.full((5, 2), 0.5)
    out = op.forward(x, ids, weights)
    assert out.shape == (5, HIDDEN) and out.dtype == torch.bfloat16
    seen = call.calls[-1]
    assert seen["a"] is x and seen["output"].shape == (5, HIDDEN)
    assert seen["ids"].dtype == torch.int32 and seen["ids"].shape == (5, 2)
    assert torch.equal(seen["ids"], ids.to(torch.int32))
    assert seen["weights"].dtype == torch.float32
    assert seen["handle"] == "plan-handle" and seen["a4_prefill"] is False
    # the persistent workspace and scale storage ride every call
    assert seen["scratch"] is op._workspace
    assert seen["storage"][0] is op._storage[0]
    assert op.calls == 1
    # capacity is the wire-time call-shape catalogue, enforced, not padded
    with pytest.raises(ValueError, match="above the prepared capacity"):
        op.forward(torch.zeros((9, HIDDEN), dtype=torch.bfloat16),
                   torch.zeros((9, 2), dtype=torch.int64), weights)
    with pytest.raises(ValueError, match="BF16"):
        op.forward(torch.zeros((3, HIDDEN)), ids, weights)


# --- the MoEBlock seam ------------------------------------------------------

def test_moe_block_dispatches_to_the_prepared_path():
    seen = {}

    def prepared(x, *, ids, weights):
        seen["x"], seen["ids"], seen["weights"] = x, ids, weights
        return torch.full_like(x, 2.0)

    def boom(*a, **k):
        raise AssertionError("the reference loop must not run when _prepared is set")

    reference = SimpleNamespace(forward=boom)
    # the seam is the CUDA path: `device.type == "cuda"` and `_prepared` set
    block = MoEBlock(router=None, shared=None, routed=reference,
                     config=None, device=SimpleNamespace(type="cuda"))
    block._prepared = prepared
    x = torch.zeros((4, HIDDEN), dtype=torch.bfloat16)
    ids = torch.zeros((4, 2), dtype=torch.int64)
    weights = torch.full((4, 2), 0.5)
    out = block._routed_forward(x, ids, weights)
    assert torch.equal(out, torch.full_like(x, 2.0))
    assert seen["ids"] is ids and seen["weights"] is weights


# --- the loader's resident-slice aliases ------------------------------------

def resident_sources(*, first=0, last=LOCAL):
    """The state.py GPU slicing over the packed contract's tensors."""
    generator = torch.Generator().manual_seed(5)
    full = packed_checkpoint(seed=5)
    local = last - first
    half = (slice(None), slice(first, last), slice(None))
    half_up = (slice(None), slice(INTER + first, INTER + last), slice(None))
    down_extent = (slice(None), slice(None), slice(first // 2, last // 2))
    down_scale_extent = (slice(None), slice(None), slice(first // 16, last // 16))
    resident = {}
    for projection in ("gate_up_proj", "down_proj"):
        for suffix in ("weight", "weight_scale", "weight_scale_2"):
            name = f"{PREFIX}.{projection}.{suffix}"
            source = full[name]
            if projection == "gate_up_proj" and suffix != "weight_scale_2":
                resident[name] = torch.cat((source[half], source[half_up]), dim=1)
            elif projection == "down_proj" and suffix == "weight":
                resident[name] = source[down_extent]
            elif projection == "down_proj" and suffix == "weight_scale":
                resident[name] = source[down_scale_extent]
            else:
                resident[name] = source[(slice(None),)]
    assert resident[f"{PREFIX}.gate_up_proj.weight"].shape == (COUNT, 2 * local, HIDDEN // 2)
    return resident


@pytest.fixture()
def captured_fp4(monkeypatch):
    record = {}

    def load_fp4_experts(self, prefixes, **kwargs):
        record["prefixes"] = tuple(prefixes)
        record["kwargs"] = kwargs
        record["sources"] = {name: self._sources[name] for name in list(self._sources)
                            if name.startswith(f"{PREFIX}.experts.")}
        return "prepared-experts"

    monkeypatch.setattr(Hy4WeightLoader, "load_fp4_experts", load_fp4_experts)
    return record


def test_load_resident_experts_serves_local_alias_views(captured_fp4):
    record = captured_fp4
    loader = make_loader({})
    resident = resident_sources()
    out = loader.load_resident_experts(PREFIX, resident,
                                       intermediate_range=(0, LOCAL),
                                       swiglu_limit=SWIGLU_LIMIT)
    assert out == "prepared-experts"
    assert record["prefixes"] == tuple(f"{PREFIX}.experts.{i}" for i in range(COUNT))
    kwargs = record["kwargs"]
    # the LOCAL intermediate with no further range: byte-identical planes to
    # load_quantized_experts' full-tensor path at the same world size
    assert kwargs["hidden_size"] == HIDDEN
    assert kwargs["intermediate_size"] == LOCAL
    assert kwargs["intermediate_range"] is None
    assert kwargs["format"] == "nvfp4" and kwargs["activation_mode"] == "a16"
    assert kwargs["swiglu_limit"] == SWIGLU_LIMIT
    assert kwargs["swiglu_clamp_after_silu"] is False
    sources = record["sources"]
    assert len(sources) == COUNT * 9
    for index in range(COUNT):
        target = f"{PREFIX}.experts.{index}"
        gate = sources[f"{target}.gate_proj.weight"]
        up = sources[f"{target}.up_proj.weight"]
        fused_rows = resident[f"{PREFIX}.gate_up_proj.weight"][index]
        assert torch.equal(gate, fused_rows[:LOCAL])
        assert torch.equal(up, fused_rows[LOCAL:])
        assert sources[f"{target}.gate_proj.weight"].shape == (LOCAL, HIDDEN // 2)
        assert sources[f"{target}.down_proj.weight"].shape == (HIDDEN, LOCAL // 2)
        gain = sources[f"{target}.gate_proj.weight_scale_2"]
        assert gain.shape == (1,)
        assert float(gain[0]) == float(resident[f"{PREFIX}.gate_up_proj.weight_scale_2"][index])
    # the alias views leave no synthetic names behind in the loader session
    assert not [name for name in loader._sources if name.startswith(f"{PREFIX}.experts.")]


def test_load_resident_experts_rejects_wrong_slice_geometry():
    loader = make_loader({})
    resident = resident_sources(first=0, last=LOCAL)
    with pytest.raises(ValueError, match="resident rank slice has shape"):
        loader.load_resident_experts(PREFIX, resident,
                                     intermediate_range=(0, INTER),   # lies about the range
                                     swiglu_limit=SWIGLU_LIMIT)


# --- the session's hand-back: fatal, with the offender named -----------------

class RefusingSession:
    """`DirectWeightSession`'s worst case: every transfer handed back False
    (the pre-flight and the fatal error must name the offender)."""

    def __init__(self):
        self.transfers = []
        self.torch_copy_bytes = 0

    def __call__(self, destination, source):
        self.torch_copy_bytes += destination.nbytes
        return False

    def flush(self):
        pass

    def stats(self):
        return {}


class B12xShapedRefusingSession(RefusingSession):
    """Carries the `DirectWeightSession` surface the pre-flight keys on
    (`sources` registry + `device`) while refusing every transfer."""

    device = 0
    sources = {}


def test_copy_is_fail_closed_and_names_the_refused_transfer():
    loader = make_loader({})
    loader._session = RefusingSession()
    staged = torch.zeros((4, 6), dtype=torch.uint8)
    destination = torch.empty_like(staged)
    with pytest.raises(RuntimeError, match="b12x rejected the direct weight transfer") as raised:
        loader._copy(destination, staged, name="model.layers.1.mlp.experts.0.gate_proj.weight")
    # the diagnostic names the tensor and both sides
    message = str(raised.value)
    assert "model.layers.1.mlp.experts.0.gate_proj.weight" in message
    assert "source device=cpu" in message and "contiguous=True" in message


def test_copy_pre_flight_names_the_offender_before_the_session():
    loader = make_loader({})
    session = B12xShapedRefusingSession()
    loader._session = session
    # a non-contiguous staged source violates the host-copy conditions
    with pytest.raises(RuntimeError, match="b12x would refuse this transfer") as raised:
        loader._copy(torch.empty((4, 6), dtype=torch.uint8),
                     torch.zeros((4, 12), dtype=torch.uint8)[:, ::2],
                     name="sliced", extent=(slice(None), slice(0, 12, 2)))
    assert "not contiguous" in str(raised.value) and "sliced" in str(raised.value)
    # a meta checkpoint view with a shape mismatch is refused with name + extent
    with pytest.raises(RuntimeError, match="b12x would refuse this transfer") as raised:
        loader._copy(torch.empty((4, 6), dtype=torch.uint8),
                     torch.empty((2, 3), dtype=torch.uint8, device="meta"),
                     name="model.layers.1.mlp.experts.0.down_proj.weight",
                     extent=(slice(None), slice(0, 128)))
    assert "A5" in str(raised.value) and "down_proj.weight" in str(raised.value)
    assert session.torch_copy_bytes == 0          # nothing reached the session


def test_load_resident_experts_aliases_meta_checkpoint_views(captured_fp4):
    """Consult 12's direct form: the fused sources are the checkpoint's own
    meta views; each expert is aliased as a FULL-WIDTH view and
    `load_fp4_experts` gets the FULL intermediate plus the real rank range, so
    its own `source[first:last]` slicing fills each plane by one direct
    transfer (`torch_copy_bytes` stays 0)."""
    record = captured_fp4
    loader = make_loader({})
    packed = packed_checkpoint()
    meta = {name: torch.empty(tuple(tensor.shape), dtype=tensor.dtype, device="meta")
            for name, tensor in packed.items() if name != f"{PREFIX}.gate_up_proj.weight_scale_2"
            and name != f"{PREFIX}.down_proj.weight_scale_2"}
    # gains stay value-loaded CPU tensors
    meta[f"{PREFIX}.gate_up_proj.weight_scale_2"] = packed[f"{PREFIX}.gate_up_proj.weight_scale_2"]
    meta[f"{PREFIX}.down_proj.weight_scale_2"] = packed[f"{PREFIX}.down_proj.weight_scale_2"]
    out = loader.load_resident_experts(PREFIX, meta,
                                       intermediate_range=(0, LOCAL),
                                       swiglu_limit=SWIGLU_LIMIT)
    assert out == "prepared-experts"
    kwargs = record["kwargs"]
    assert kwargs["intermediate_size"] == INTER          # FULL intermediate
    assert kwargs["intermediate_range"] == (0, LOCAL)    # the rank's real range
    sources = record["sources"]
    assert len(sources) == COUNT * 9
    for index in range(COUNT):
        target = f"{PREFIX}.experts.{index}"
        gate = sources[f"{target}.gate_proj.weight"]
        up = sources[f"{target}.up_proj.weight"]
        assert gate.is_meta and up.is_meta
        # full-width aliases of the checkpoint's fused [gate; up] rows
        assert gate.shape == (INTER, HIDDEN // 2) and up.shape == (INTER, HIDDEN // 2)
        fused_storage = meta[f"{PREFIX}.gate_up_proj.weight"].untyped_storage()._cdata
        assert gate.untyped_storage()._cdata == fused_storage
        assert up.untyped_storage()._cdata == fused_storage
        assert gate.storage_offset() == index * 2 * INTER * (HIDDEN // 2)
        assert up.storage_offset() == (index * 2 + 1) * INTER * (HIDDEN // 2)
        down = sources[f"{target}.down_proj.weight"]
        assert down.shape == (HIDDEN, INTER // 2)
        assert down.untyped_storage()._cdata == \
            meta[f"{PREFIX}.down_proj.weight"].untyped_storage()._cdata
        assert down.storage_offset() == index * HIDDEN * (INTER // 2)
        gain = sources[f"{target}.gate_proj.weight_scale_2"]
        assert gain.shape == (1,) and not gain.is_meta
        assert float(gain[0]) == float(meta[f"{PREFIX}.gate_up_proj.weight_scale_2"][index])
    # the alias views leave no synthetic names behind in the loader session
    assert not [name for name in loader._sources if name.startswith(f"{PREFIX}.experts.")]


# --- allocated_bytes must survive the fused graph's object cycles -----------

def test_allocated_bytes_is_cycle_safe_over_fused_plan_graphs():
    """The prepared fused plan holds Enum members (`PackedSourceFormat`,
    `AutotuneEffort`); every Enum member reaches its class's member maps and
    back (`vars(member)['__objclass__']`), so a `vars()`-walking graph visitor
    without identity dedupe recurses forever — the serve's `allocated_bytes`
    RecursionError during construct_rank."""
    from enum import Enum

    from ignition.recipes.hy4.state import Hy4ModelWeights

    class Packing(Enum):
        MODEL = "modelopt_nvfp4"

    plan = SimpleNamespace(format=Packing.MODEL, packing=Packing.MODEL)
    routed = SimpleNamespace(plan=plan, source=SimpleNamespace(format=Packing.MODEL))
    layer = SimpleNamespace(mlp=SimpleNamespace(routed=routed))
    embed = torch.zeros(2, 2)
    weights = Hy4ModelWeights(config=None, sharding=None, device=torch.device("cpu"),
                              embed=embed, norm=torch.zeros(2), head=torch.zeros(2, 2),
                              hc_head=SimpleNamespace(weight=embed[:1]),   # same storage
                              layers=(layer,), draft=None)
    # terminates through the Enum cycle; one storage counts once
    assert weights.allocated_bytes == embed.untyped_storage().nbytes() \
        + weights.norm.untyped_storage().nbytes() + weights.head.untyped_storage().nbytes()
