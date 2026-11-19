# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""P3 R0 gate: the h4 MoE wiring with CSF experts constructs, on CPU.

Two serve boots (2026-11-19 09:20 and 09:29, cards 5-12) died in construction
because nothing below a real boot reached the decode-op planning path:
- boot 1 (6ac4c8d0): `functional compressed experts require an arena
  preparation-release boundary` (decode ops planned without a ScratchArena);
- boot 2 (9598a030): `scratch binding declarations are frozen` (decode ops
  handed the wire arena after `FusedWireScratch.seal()` froze it).

This test drives the recipe's own wiring (`wire_routed_layers`, the block
`bootstrap.wire_operators` runs) through ignition's REAL `FusedRoutedExperts`,
`wire_decode_ops`, `PreparedRoutedExperts`, `register_finalizer` and
`ScratchArena`. Only the b12x boundary under them is stubbed: plan builders,
`declare`, `bind`, the scale-template helpers and the kernel call. That is
where the arena lifecycle lives, so both historical crashes reproduce here,
and the fix is pinned:
- CSF decode ops register on the live arena before the seal;
- after the seal the arena is finalized and its preparation releases run,
  so b12x `release_scale_templates` drops every layer's expanded CSF scale
  destinations before the serve sizes its KV pool;
- raw-scale wiring keeps its order and its workspace-free decode plans.
"""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

CPU = torch.device("cpu")
HIDDEN, TOP_K, EXPERTS, CAPACITY = 8, 2, 4, 64


class FakePlan:
    """b12x execution plan: one uint8 workspace sized by the call capacity.

    Like b12x (`_tuning`: inline CSF decoding at >= 16 tokens and low expert
    occupancy), a CSF plan at 16..CAPACITY-1 rows decodes its scales INLINE
    and declares no scale storage; the others expand into three planes.
    """

    def __init__(self, max_tokens, events, csf):
        self.nbytes = 256 * max_tokens
        self.handle = f"plan-{max_tokens}"
        self.csf = csf
        self.inline = csf and 16 <= max_tokens < CAPACITY
        events.append(("plan", max_tokens))

    def scratch_specs(self):
        return [SimpleNamespace(shape=(self.nbytes,), dtype=torch.uint8, device=CPU,
                                nbytes=self.nbytes)]

    def request(self, *, name, prepare_call, benchmark_call, dependencies=()):
        return SimpleNamespace(name=name)


def scale_buffers(binding):
    """b12x `scale_buffers`: ([], planes) for an expanded CSF plan, else ([], [])."""
    plan = binding.plan
    if not plan.csf or plan.inline:
        return [], []
    return [], [torch.zeros((EXPERTS, 6)), torch.zeros((EXPERTS, 3)), torch.zeros((2,))]


class FakeScalePlan:
    def request(self, *, name, prepare_call, benchmark_call):
        return SimpleNamespace(name=name)


class Boundary:
    """Records what the stubbed b12x boundary saw."""

    def __init__(self):
        self.events = []                  # ("plan", rows) / ("seal",) / ("release", rows)
        self.released = []                # (experts, plan) per release_scale_templates call
        self.calls = 0
        self.storage = []                 # (plan handle, scale storage planes) per kernel call

    def prepared_moe_out(self, a, weights, ids, output, scratch, scale_inputs, storage,
                         handle, input_scales_static, fast_math, activation_amax,
                         layer_idx, route_expert_map, output_expert_map, scales_expanded,
                         a4_prefill):
        self.calls += 1
        self.storage.append((handle, len(storage)))
        output.zero_()

    def release_scale_templates(self, experts, plan):
        self.events.append(("release", plan.nbytes // 256))
        self.released.append((experts, plan))


class FakePreparationSession:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


@pytest.fixture()
def boundary(monkeypatch):
    import b12x.moe.fused_moe._binding_ops as binding_ops
    import b12x.preparation
    import ignition.components.moe as components_moe
    from ignition.recipes.hy4.moe import FusedWireScratch

    record = Boundary()
    fused_moe = SimpleNamespace(
        plan_scale_expansion=lambda experts, **kwargs: FakeScalePlan(),
        plan_execution=lambda *, capacity, experts, **kwargs: FakePlan(
            capacity.max_tokens, record.events, experts.has_csf_blocks),
        ExecutionCapacity=lambda **kwargs: SimpleNamespace(**kwargs),
        RoutingSpec=lambda **kwargs: SimpleNamespace(**kwargs),
        bind=lambda plan, **kwargs: SimpleNamespace(plan=plan),
    )
    monkeypatch.setattr(components_moe, "fused_moe", fused_moe)
    monkeypatch.setattr(components_moe, "declare", lambda session, requests, **kwargs: None)
    monkeypatch.setattr(binding_ops, "materialize_scale_templates", lambda experts: None)
    monkeypatch.setattr(binding_ops, "scale_buffers", scale_buffers)
    monkeypatch.setattr(binding_ops, "release_weight_views", lambda impl: None)
    monkeypatch.setattr(binding_ops, "release_scale_templates", record.release_scale_templates)
    monkeypatch.setattr(binding_ops, "prepared_moe_out", record.prepared_moe_out)
    monkeypatch.setattr(b12x.preparation, "PreparationSession", FakePreparationSession)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args, **kwargs: None)
    seal = FusedWireScratch.seal

    def recorded_seal(scratch):
        record.events.append(("seal",))
        seal(scratch)

    monkeypatch.setattr(FusedWireScratch, "seal", recorded_seal)
    monkeypatch.setenv("HY4_MOE_DECODE_OP", "1,4,16")        # 16: an inline-CSF plan
    for name in ("HY4_MOE_DECODE_ENV_ALT", "HY4_MOE_A4", "HY4_MOE_CALIBRATE",
                 "HY4_MOE_AUTOTUNE", "HY4_MOE_DETERMINISTIC"):
        monkeypatch.delenv(name, raising=False)
    return record


def _experts(csf):
    impl = SimpleNamespace(**{field: torch.zeros(1) for field in (
        "w1_fp4", "w2_fp4", "w1_blockscale", "w2_blockscale",
        "w1_alphas", "w2_alphas", "a1_gscale", "a2_gscale")},
        representation=None, nvfp4_csf=None)
    return SimpleNamespace(_impl=impl, has_csf_blocks=csf, num_experts=EXPERTS,
                           plan=SimpleNamespace(activation=SimpleNamespace(outer_scale="group")))


def _layers(csf, count=3):
    from ignition.recipes.hy4.moe import FusedRoutedExperts

    layers = []
    for index in range(count):
        routed = FusedRoutedExperts(_experts(csf), hidden=HIDDEN, top_k=TOP_K, device=CPU,
                                    name=f"model.layers.{index + 1}.mlp.experts.fused")
        layers.append(SimpleNamespace(mlp=SimpleNamespace(routed=routed)))
    return layers


def _wire(layers):
    from ignition.recipes.hy4.moe import wire_routed_layers

    return wire_routed_layers(layers, device=CPU, capacity=CAPACITY, session="prepared")


def test_csf_wiring_constructs_with_the_decode_ops(boundary):
    layers = _layers(csf=True)
    session, scratch, small = _wire(layers)
    assert small is None
    for layer in layers:
        routed = layer.mlp.routed
        assert layer.mlp._prepared is routed
        assert sorted(routed.decode_ops) == [1, 4, 16]
        assert routed._decode_workspace is not None and len(routed._decode_storage) == 3
    # the decode ops planned on the LIVE arena: every plan precedes the seal
    events = boundary.events
    seal = events.index(("seal",))
    plans = [index for index, event in enumerate(events) if event[0] == "plan"]
    assert len(plans) == 4 * len(layers) and max(plans) < seal


def test_csf_inline_decode_plans_take_no_scale_storage(boundary):
    """b12x decodes CSF inline for some plans (no scale storage) and expands
    for others (GPU R2, 10:03: `the decode MoE ops disagree on their scale
    layouts`). The shared decode storage serves the expanded plans only."""
    from ignition.recipes.hy4.runtime import SWITCHES

    layers = _layers(csf=True)
    _wire(layers)
    routed = layers[0].mlp.routed
    assert routed.decode_ops[16].scale_layouts == ()
    assert routed.decode_ops[4].scale_layouts == routed.decode_ops[1].scale_layouts != ()
    before = SWITCHES.get("moe_decode_op")
    SWITCHES["moe_decode_op"] = "on"
    try:
        boundary.storage.clear()
        for rows in (1, 3, 10, 30):
            routed.forward(torch.zeros((rows, HIDDEN), dtype=torch.bfloat16),
                           torch.zeros((rows, TOP_K), dtype=torch.int64), torch.ones((rows, TOP_K)))
    finally:
        SWITCHES["moe_decode_op"] = before
    assert boundary.storage == [("plan-1", 3), ("plan-4", 3), ("plan-16", 0), (f"plan-{CAPACITY}", 3)]


def test_csf_wiring_releases_the_expanded_scale_templates(boundary):
    layers = _layers(csf=True)
    _, scratch, _ = _wire(layers)
    # every CSF op (prefill and both decode capacities, every layer)
    # registered its release and the arena ran them all, once, after the seal
    # — the serve sizes its KV pool after wiring returns, so the freed
    # expanded scale planes count toward it
    released = {(id(experts), plan.handle) for experts, plan in boundary.released}
    assert len(boundary.released) == len(released) == 4 * len(layers)
    assert released == {(id(layer.mlp.routed.experts), f"plan-{rows}")
                        for layer in layers for rows in (CAPACITY, 16, 4, 1)}
    events = boundary.events
    seal = events.index(("seal",))
    assert all(index > seal for index, event in enumerate(events) if event[0] == "release")
    assert scratch.arena.frozen and scratch.arena.nbytes == 0
    with pytest.raises(RuntimeError, match="released"):
        scratch.arena.register_preparation_release(lambda: None)
    # the ops keep serving: the main workspace view owns the arena's bytes
    routed = layers[0].mlp.routed
    assert routed._workspace.numel() == 256 * CAPACITY
    calls = boundary.calls
    routed.forward(torch.zeros((2, HIDDEN), dtype=torch.bfloat16),
                   torch.zeros((2, TOP_K), dtype=torch.int64), torch.ones((2, TOP_K)))
    assert boundary.calls == calls + 1


def test_raw_scale_wiring_keeps_its_order(boundary):
    layers = _layers(csf=False)
    _, scratch, _ = _wire(layers)
    for layer in layers:
        assert sorted(layer.mlp.routed.decode_ops) == [1, 4, 16]
        assert layer.mlp.routed._decode_storage == []
    # unchanged pre-CSF order: main plans, seal, then the decode plans; no
    # preparation release (nothing to release, the arena keeps its bytes)
    events = boundary.events
    seal = events.index(("seal",))
    assert [event for event in events[:seal] if event[0] == "plan"] == [("plan", CAPACITY)] * 3
    assert sorted(event[1] for event in events[seal + 1:]) == [1] * 3 + [4] * 3 + [16] * 3
    assert boundary.released == []
    assert scratch.arena.frozen and scratch.arena.nbytes > 0
