# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepared reductions preserve the selected surface's capture and ownership contract."""

from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("surface", ["PCIeTwoShotBF16.all_reduce", "OneshotAllReduce.all_reduce"])
def test_preparation_uses_selected_surface_and_retains_storage(monkeypatch, surface):
    from b12x.comm import pcie
    from b12x.comm.pcie import _twoshot_preparation
    from b12x.preparation import PreparedCall
    from ignition.components import collectives

    monkeypatch.setattr(pcie, "joint_finalist_policy", lambda: object(), raising=False)
    source = torch.full((32, 5120), 3, dtype=torch.bfloat16)
    output = torch.empty_like(source)
    callbacks, calls, selected = [], [], []
    plan = SimpleNamespace(query=SimpleNamespace(surface=surface))
    plan.request = lambda **kwargs: callbacks.append(kwargs) or object()
    runtime = SimpleNamespace(algorithm="oneshot", world_size=4, single_channel=True,
                              exchange_group=object(), for_stream=lambda: SimpleNamespace(),
                              plan=lambda *_args, **_kwargs: plan)
    monkeypatch.setattr(collectives, "_roce", lambda _: False)
    monkeypatch.setattr(collectives, "declare", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(torch.distributed, "get_process_group_ranks", lambda _: [0, 1, 2, 3])

    def two_call(state, *, payload, out):
        calls.append("two")
        assert payload is source and out is output
        call = PreparedCall(run=lambda: out.copy_(payload), output=out, capture_safe=False,
                            owners=(state, payload, out))
        selected.append(call)
        return call

    monkeypatch.setattr(_twoshot_preparation, "prepared_call", two_call)
    operation = collectives.PreparedAllReduce(source, output, runtime=runtime,
        session=object(), coordinator=object(), name="hidden.reduce")
    state = SimpleNamespace(query=plan.query,
        bind_input=lambda tensor: calls.append("bind") or None,
        run_plain=lambda inp, out: out.copy_(inp))
    prepared = callbacks[0]["prepare_call"](state)
    if selected:
        assert prepared is selected[0]
    else:
        assert all(actual is expected for actual, expected in zip(
            prepared.owners, (runtime, source, output), strict=True))
    assert prepared.capture_safe is (surface != "PCIeTwoShotBF16.all_reduce")
    prepared.run()
    assert torch.equal(source, output)
    assert calls == (["two"] if surface == "PCIeTwoShotBF16.all_reduce" else ["bind"])
    assert operation.runtime is runtime and operation.plan is plan


@pytest.mark.parametrize("mode", ["static", "runtime"])
@pytest.mark.parametrize("effort", ["off", "default", "audit"])
def test_runtime_geometry_is_pinned_for_every_tuning_effort(monkeypatch, effort, mode):
    from b12x.comm import pcie
    from b12x.preparation import AutotuneEffort
    from ignition.components import collectives
    from ignition.preparation import startup_tuning

    calls, requests, registered = [], [], []
    plan = SimpleNamespace(request=lambda **kwargs: requests.append(kwargs) or object())
    runtime = SimpleNamespace(algorithm="oneshot", world_size=4, single_channel=True,
        exchange_group=object(), for_stream=lambda: SimpleNamespace(_joint_workspace=object()),
        plan=lambda *args, **kwargs: calls.append(kwargs) or plan)
    from ignition.preparation.collective_ladder import load, select
    ladder = load("ignition/preparation/ladders/gb202_pcie_tp4.json")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (12, 0))
    monkeypatch.setattr(collectives, "_roce", lambda _: False)
    monkeypatch.setattr(collectives, "declare", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.distributed, "get_process_group_ranks", lambda _: [0, 1, 2, 3])
    monkeypatch.setattr(startup_tuning, "register_collective", lambda owner, session: registered.append(owner))
    source = torch.ones(2048, dtype=torch.bfloat16)
    output = torch.empty_like(source)
    operation = collectives.PreparedAllReduce(source, output, runtime=runtime,
        session=SimpleNamespace(_autotune_effort=AutotuneEffort(effort), collective_tuning=mode, collective_ladder=ladder), coordinator=object(), name="reduce")
    assert "finalist_policy" not in requests[0]
    if mode == "static":
        assert calls == [dict(out=output)]
        assert registered == []
    else:
        row = select(ladder, 4096, world_size=4, capability=(12, 0), inplace=False)
        assert calls == [dict(out=output, joint_only=True, override=pcie.CollectiveConfig(
            implementation=row["implementation"], threads=row["threads"], blocks=row["blocks"],
            submode=next(iter(row["profiles"])), active_blocks=1))]
        assert operation.runtime_profiles == tuple((mode, count) for mode, counts in row["profiles"].items() for count in counts)
        assert operation.runtime_cache_policy == "cold"
        assert registered == [operation]


def _fp8_channel(rows=16, columns=64, world=2, calls=None):
    runtime = SimpleNamespace(world_size=world, rank=0, device=torch.device("cpu"), exchange_group=object(),
        max_rows=rows, row_elems=columns,
        _pack_stride=256, _scale_offset=4096, _scale_stride=32, _slot_bytes=8192,
        _signal_ptrs=tuple(range(8)), _staging_ptrs=(tuple(range(8)), tuple(range(8))))
    calls = [] if calls is None else calls
    runtime.reduce_scatter_fp8 = lambda payload, scale, out, plan=None: calls.append(
        ("reduce_scatter_fp8", tuple(payload.shape), tuple(scale.shape), tuple(out.shape))) or out.zero_()
    runtime.all_gather_fp8 = lambda payload, scale, out, plan=None: calls.append(
        ("all_gather_fp8", tuple(payload.shape), tuple(scale.shape), tuple(out.shape))) or out.zero_()
    from ignition.components.collectives import Fp8TwoShotAllReduce

    return Fp8TwoShotAllReduce(runtime, max_rows=rows, row_elems=columns), calls


def test_fp8_two_shot_composes_quantized_reduce_scatter_and_all_gather(monkeypatch):
    from b12x.comm.pcie import _twoshot_preparation
    from b12x.comm.pcie.pcie_twoshot import quantize_per_row
    from b12x.preparation import PreparedCall
    from ignition.components import collectives

    rows, columns, world = 16, 64, 2
    channel, calls = _fp8_channel(rows, columns, world)
    requests, primed = [], []
    monkeypatch.setattr(collectives, "declare", lambda session, values, **kwargs: requests.extend(values))
    monkeypatch.setattr(torch.distributed, "get_process_group_ranks", lambda group: list(range(world)))

    def two_call(state, *, payload, scale=None, out=None):
        primed.append((payload, scale, out))
        return PreparedCall(run=lambda: None, output=out, capture_safe=False)

    monkeypatch.setattr(_twoshot_preparation, "prepared_call", two_call)
    source = torch.randn(rows, columns).to(torch.bfloat16)
    output = torch.empty_like(source)
    operation = collectives.PreparedAllReduce(source, output, runtime=channel, session=object(),
        coordinator=object(), name="layer.reduce")
    # Both halves declare their own prepared surface against shared transport buffers.
    assert (operation.plan.query.surface, operation.ag_plan.query.surface) == (
        "TwoShotReduceScatter.reduce_scatter_fp8", "TwoShotReduceScatter.all_gather_fp8")
    assert [request.name for request in requests] == ["layer.reduce.fp8_reduce_scatter", "layer.reduce.fp8_all_gather"]
    assert channel._plans == [operation.plan]
    for request in requests:
        request.prepare_call(SimpleNamespace())
    assert [(tuple(p.shape), tuple(s.shape), tuple(o.shape)) for p, s, o in primed] == [
        ((rows, columns), (rows, 1), (rows // world, columns)),
        ((rows // world, columns), (rows // world, 1), (rows, columns))]
    # The captured launch quantizes exactly as b12x's reference, then runs
    # quantize -> reduce-scatter -> requantize -> all-gather into the output.
    operation.launch(source, output)
    assert calls == [("reduce_scatter_fp8", (rows, columns), (rows, 1), (rows // world, columns)),
                     ("all_gather_fp8", (rows // world, columns), (rows // world, 1), (rows, columns))]
    expected_payload, expected_scale = quantize_per_row(source)
    assert torch.equal(channel.scale[:rows], expected_scale.view(-1, 1))
    assert torch.equal(channel.payload[:rows].view(torch.uint8), expected_payload.view(torch.uint8))
    # A zero reduced shard quantizes to zero payloads on the all-gather wire.
    assert not channel.shard_payload[:rows // world].view(torch.uint8).any()
    assert operation.runtime is channel and operation.output is output


def test_fp8_two_shot_capacity_matches_the_exact_twin_envelope(monkeypatch):
    from ignition.components import collectives

    channel, _ = _fp8_channel()
    accept = torch.zeros(16, 64, dtype=torch.bfloat16)
    assert channel.accepts(accept) and channel.accepts(torch.zeros(8, 128, dtype=torch.bfloat16))
    assert not channel.accepts(torch.zeros(15, 64, dtype=torch.bfloat16))  # transport rows must divide the world
    assert not channel.accepts(torch.zeros(32, 64, dtype=torch.bfloat16))  # beyond the twin's envelope
    assert not channel.accepts(torch.zeros(16, 64)) and not channel.accepts(torch.zeros(16, 65)[:, :64])
    with pytest.raises(ValueError, match="fp8 two-shot reductions"):
        collectives.PreparedAllReduce(torch.zeros(16, 64), torch.empty(4, 4, dtype=torch.bfloat16),
            runtime=channel, session=object(), coordinator=object(), name="bad")
    monkeypatch.setattr(torch.distributed, "get_process_group_ranks", lambda group: [0, 1])
    monkeypatch.setattr(collectives, "declare", lambda *args, **kwargs: None)
    # The reduction contract is unchanged: matching contiguous BF16 source and output.
    source, output = torch.zeros(16, 64, dtype=torch.bfloat16), torch.empty(16, 64, dtype=torch.bfloat16)
    collectives.PreparedAllReduce(source, output, runtime=channel, session=object(),
        coordinator=object(), name="good")
    assert output.shape == source.shape and output.dtype == source.dtype


def _island_runtime(rows=32, columns=64):
    """A real PCIeIslandARAllReduce shell with mocked launch side effects."""
    from b12x.comm.pcie.pcie_island_ar import PCIeIslandARAllReduce

    runtime = PCIeIslandARAllReduce.__new__(PCIeIslandARAllReduce)
    runtime.group = runtime.exchange_group = object()
    runtime.rank, runtime.world_size = 0, 8
    runtime.device = torch.device("cpu")
    runtime.max_rows, runtime.row_elems = rows, columns
    runtime.max_elements = rows * columns
    runtime.blocks = None
    runtime.threads, runtime.wait_nanosleep_cycles = 512, 24
    runtime.quarter_capacity = 256
    runtime.stage_offset, runtime.part_offset, runtime.final_offset = 49664, 60000, 62000
    runtime.slab_bytes = 128 * 1024
    runtime._mapped_peers = (1, 2, 3, 4)
    runtime._closed = False
    calls = []
    runtime.all_reduce = lambda source, out=None, plan=None: calls.append(
        ("all_reduce", tuple(source.shape), tuple(out.shape), plan)) or out
    return runtime, calls


def test_island_channel_prepares_the_owner_surface_and_launches_all_reduce(monkeypatch):
    """L1: the island channel prepares one owner-surface plan and launches unchanged shapes."""
    import contextlib

    from b12x.preparation import PreparedCall
    from ignition.components import collectives

    runtime, calls = _island_runtime()
    requests, primed = [], []
    monkeypatch.setattr(collectives, "_roce", lambda _: False)
    monkeypatch.setattr(collectives, "declare", lambda session, values, **kwargs: requests.extend(values))
    monkeypatch.setattr(torch.distributed, "get_process_group_ranks", lambda group: list(range(8)))
    monkeypatch.setattr(torch.cuda, "device", lambda device: contextlib.nullcontext())

    import b12x.comm.pcie._owner_preparation as owner

    def owner_call(state, **actual):
        primed.append(actual)
        return PreparedCall(run=lambda: state.launcher(), output=actual.get("out"),
                            capture_safe=True)

    monkeypatch.setattr(owner, "prepared_call", owner_call)
    source = torch.zeros(8, 64, dtype=torch.bfloat16)
    output = torch.empty_like(source)
    operation = collectives.PreparedAllReduce(source, output, runtime=runtime,
        session=object(), coordinator=object(), name="layer.reduce")
    assert operation.plan.query.surface == "PCIeIslandARAllReduce.all_reduce"
    assert operation.plan.query.call["inp"]["shape"] == (8, 64)
    assert [request.name for request in requests] == ["layer.reduce"]
    assert requests[0].collective.key == "layer.reduce"
    for request in requests:
        request.prepare_call(SimpleNamespace())
    assert [(call["inp"].shape, call["out"].shape) for call in primed] == [((8, 64), (8, 64))]
    # The captured launch keeps the reduction contract: whole-tensor all_reduce.
    operation.launch(source, output)
    assert calls == [("all_reduce", (8, 64), (8, 64), operation.plan)]
    assert operation.runtime is runtime and operation.output is output


def test_island_capacity_matches_the_exact_twin_envelope(monkeypatch):
    from ignition.components import collectives

    runtime, _ = _island_runtime(rows=32, columns=64)
    assert runtime.accepts(torch.zeros(32, 64, dtype=torch.bfloat16))
    assert not runtime.accepts(torch.zeros(33, 64, dtype=torch.bfloat16))  # beyond the envelope
    monkeypatch.setattr(collectives, "_roce", lambda _: False)
    monkeypatch.setattr(collectives, "declare", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.distributed, "get_process_group_ranks", lambda group: list(range(8)))
    with pytest.raises(ValueError, match="island reductions require"):
        collectives.PreparedAllReduce(torch.zeros(33, 64, dtype=torch.bfloat16),
            torch.empty(4, 4, dtype=torch.bfloat16), runtime=runtime,
            session=object(), coordinator=object(), name="bad")
    with pytest.raises(ValueError, match="island reduction capacity must cover the full input"):
        collectives.PreparedChunkedAllReduce(torch.zeros(64, 64, dtype=torch.bfloat16),
            torch.empty(64, 64, dtype=torch.bfloat16), runtime=runtime,
            session=object(), coordinator=object(), name="chunked.bad")


def test_direct_collective_requires_joint_api_before_capability_lookup(monkeypatch):
    from importlib import import_module
    from ignition.components import collectives

    pcie = import_module("b12x.comm.pcie")
    api = import_module("b12x.comm.pcie.api")
    monkeypatch.delattr(pcie, "CollectiveConfig", raising=False)
    monkeypatch.delattr(api, "CollectiveConfig", raising=False)
    monkeypatch.setattr(collectives, "_roce", lambda _: False)
    def channel():
        raise AssertionError("capability lookup must follow the required API import")
    runtime = SimpleNamespace(algorithm="oneshot", world_size=2, single_channel=True, for_stream=channel)
    source = torch.ones(2048, dtype=torch.bfloat16)
    with pytest.raises(ImportError, match="CollectiveConfig"):
        collectives.PreparedAllReduce(source, torch.empty_like(source), runtime=runtime,
            session=object(), coordinator=object(), name="reduce")


@pytest.mark.parametrize("first_profile", [("payload_rsag_phased", 8), ("payload_rsag", 64)])
def test_clustered_ladder_prepares_a_valid_initial_profile(monkeypatch, first_profile):
    from copy import deepcopy
    from b12x.comm.pcie import CollectiveConfig
    from ignition.components import collectives
    from ignition.preparation import startup_tuning
    from ignition.preparation.collective_ladder import load, validate

    data = deepcopy(load("ignition/preparation/ladders/gb202_pcie_tp4.json"))
    mode, active = first_profile
    data["buckets"][0].update(threads=512, blocks=64, cluster_size=8, profiles={mode: [active]})
    validate(data)
    calls = []
    plan = SimpleNamespace(request=lambda **kwargs: object())
    runtime = SimpleNamespace(algorithm="oneshot", world_size=4, single_channel=True,
        exchange_group=object(), for_stream=lambda: SimpleNamespace(_joint_workspace=object()),
        plan=lambda *args, **kwargs: calls.append(kwargs) or plan)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (12, 0))
    monkeypatch.setattr(collectives, "_roce", lambda _: False)
    monkeypatch.setattr(collectives, "declare", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.distributed, "get_process_group_ranks", lambda _: [0, 1, 2, 3])
    monkeypatch.setattr(startup_tuning, "register_collective", lambda *args: None)
    source = torch.ones(8, dtype=torch.bfloat16)
    output = torch.empty_like(source)
    collectives.PreparedAllReduce(source, output, runtime=runtime,
        session=SimpleNamespace(collective_tuning="runtime", collective_ladder=data),
        coordinator=object(), name="hidden.reduce")
    assert calls == [dict(out=output, joint_only=True, joint_cluster_size=8,
                         override=CollectiveConfig(implementation="joint_latency", threads=512,
                             blocks=64, submode=mode, active_blocks=active))]


def test_collective_choice_keys_distinguish_alternatives_and_finalist_policy():
    from b12x.comm.pcie import joint_finalist_policy
    from b12x.comm.pcie._collective_tuning import CollectiveConfig, CollectiveQuery, TUNING
    from b12x.preparation import FrozenMapping
    from b12x.preparation.session import PreparationJob

    def key(implementation, policy):
        config = CollectiveConfig(implementation=implementation, threads=256, blocks=64)
        query = CollectiveQuery(operation="all_reduce", world_size=2, rank=0,
            alternatives=FrozenMapping({implementation: {"call": {"shape": (32, 2560)}}}), default=config)
        plan = SimpleNamespace(contract=TUNING, invocation=FrozenMapping())
        request = SimpleNamespace(plan=plan, dependencies=(), finalist_policy=policy)
        obligation = SimpleNamespace(request=request, candidate_program_identity="same-program",
            configuration=SimpleNamespace(encoded_query=FrozenMapping(TUNING.encode_query(query)), pinned=None))
        return PreparationJob._choice_key(None, obligation, {})

    static = key("oneshot", None)
    runtime_without_finalists = key("joint_latency", None)
    runtime = key("joint_latency", joint_finalist_policy())
    assert len({static, runtime_without_finalists, runtime}) == 3
    assert static == key("oneshot", None)
    assert runtime == key("joint_latency", joint_finalist_policy())


@pytest.mark.parametrize("geometry,expected", [({}, (256, 16)),
    ({"threads": 512, "block_limit": 8}, (512, 8))])
def test_lse_geometry_is_declaration_metadata_without_replay_overrides(monkeypatch, geometry, expected):
    def no_cuda(*args, **kwargs):
        pytest.fail("collective metadata tests must not initialize CUDA")

    monkeypatch.setattr(torch.cuda, "_lazy_init", no_cuda)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    from types import MethodType
    from b12x.comm.pcie.pcie_dcp_a2a import PCIeDCPA2A
    from ignition.components import collectives

    runtime = SimpleNamespace(world_size=16, rank=0, device=torch.device("cpu"), max_batch_size=4,
        total_heads=16, heads_per_rank=1, head_dim=128, query_head_dim=128,
        _slot_bytes=1 << 20, _signal_ptrs=tuple(range(16)), _threads_override=0,
        _block_limit_override=0, exchange_group=object(), _owned_buffers=())
    runtime._resolve_launch_config = MethodType(PCIeDCPA2A._resolve_launch_config, runtime)
    requests, primed, replayed = [], [], []
    runtime.lse_reduce_scatter = lambda *args, **kwargs: replayed.append(kwargs)
    monkeypatch.setattr(torch.distributed, "get_process_group_ranks", lambda _: list(range(16)))
    monkeypatch.setattr(collectives, "declare", lambda session, values, **kwargs: requests.extend(values))
    partial = torch.zeros((4, 16, 128), dtype=torch.bfloat16)
    lse = torch.zeros((4, 16), dtype=torch.float32)
    output = torch.empty((4, 1, 128), dtype=torch.bfloat16)
    operation = collectives.PreparedLseReduceScatter(partial, lse, output, runtime=runtime,
        session=object(), coordinator=object(), name="attention.combine", **geometry)
    call = operation.plan.query.call
    assert (call["threads"], call["block_limit"]) == expected
    prepared = requests[0].prepare_call(SimpleNamespace(run=lambda **kwargs: primed.append(kwargs)))
    prepared.run()
    operation(partial, lse)
    for invocation in (*primed, *replayed):
        assert "threads" not in invocation and "block_limit" not in invocation
    assert replayed[0]["plan"] is operation.plan
