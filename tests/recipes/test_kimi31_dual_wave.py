# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""L9 dual-wave prefill overlap: plan-setting wiring, the schedule gate, the pool cap, the launch knob.

The arithmetic contract (substreams == serial mixes bitwise) rides
test_kimi31_dual_wave_residual.py; the ring transports ride flashinfer's CPU
emulation. These cases pin the wiring: the setting is recorded and gated, the
gate selects the two-wave schedule exactly where intended, the pool cap bounds
the second prepared-module set, and arming stamps the capped expert launch.
"""

from dataclasses import replace
from types import MappingProxyType, SimpleNamespace

import pytest

from ignition import planning
from ignition.errors import UsageError
from ignition.recipes.declaration import Checkpoint, RankCosts, Recipe, SamplingDefaults
from ignition.recipes.kimi3.decoder import (check_wire_scope, dual_wave_armed_shapes, dual_wave_extra_bytes,
                                            dual_wave_prefill_enabled, dual_wave_shape_enabled)

GIB, MIB = 1 << 30, 1 << 20


def recipe(name, device_gib, ranks):
    checkpoint = Checkpoint(name, f"lab/{name}", "ab12cd34" + "0" * 32, "DemoForCausalLM",
                            stored_bytes=device_gib * GIB, device_bytes=device_gib * GIB,
                            mapped_host_bytes=0, draft_bytes=0)
    return Recipe(name=name, served_model_name=name, checkpoints=(checkpoint,),
                  context=262144, page_tokens=64, concurrency=4,
                  sampling_defaults=SamplingDefaults(temperature=1., top_p=1., top_k=-1),
                  max_concurrency=64, max_draft_tokens=0, prefill_chunk=4096, prepare=None,
                  usage=None, mapped_host_name="lookup table", speculation=None,
                  ranks=MappingProxyType({size: RankCosts(100 * MIB // size, 2 * MIB // size, 4 * GIB)
                                          for size in ranks}))


def gpu(index, free_gib):
    return planning.Gpu(index, "NVIDIA RTX PRO 6000 Blackwell", f"GPU-{index:04d}", 96 * GIB,
                        int(free_gib * GIB), (12, 0))


KIMI = recipe("kimi31", 60, (2, 4, 8))
SOLO = recipe("solo", 60, (1, 2, 4))
OPTIONS = dict(concurrency=4, reserve_bytes=0, speculation=None)


def test_prefill_dual_wave_is_a_gated_plan_setting():
    exact = planning.plan(KIMI, KIMI.checkpoints[0], tuple(gpu(i, 95) for i in range(8)),
                          selected=tuple(range(8)), **OPTIONS)
    assert exact.prefill_dual_wave is False
    assert exact.to_dict()["prefill_dual_wave"] is False
    assert exact.prefill_dual_wave_pool == 0.09
    assert exact.to_dict()["prefill_dual_wave_pool"] == 0.09
    armed = planning.plan(KIMI, KIMI.checkpoints[0], tuple(gpu(i, 95) for i in range(8)),
                          selected=tuple(range(8)), prefill_dual_wave=True,
                          prefill_dma_ring_rows=1024, **OPTIONS)
    assert armed.prefill_dual_wave is True and armed.to_dict()["prefill_dual_wave"] is True
    assert replace(armed, prefill_dual_wave=False, prefill_dma_ring_rows=0) == exact
    # The setting is Kimi-only and needs the copy-engine ring at TP8.
    with pytest.raises(UsageError, match="--prefill-dual-wave supports"):
        planning.plan(SOLO, SOLO.checkpoints[0], (gpu(0, 95),), selected=(0,),
                      prefill_dual_wave=True,
                      concurrency=4, reserve_bytes=0, speculation=None)
    with pytest.raises(UsageError, match="copy-engine ring"):
        planning.plan(KIMI, KIMI.checkpoints[0], tuple(gpu(i, 95) for i in range(8)),
                      selected=tuple(range(8)), prefill_dual_wave=True, **OPTIONS)
    with pytest.raises(UsageError, match="TP8 copy-engine ring"):
        planning.plan(KIMI, KIMI.checkpoints[0], tuple(gpu(i, 95) for i in range(4)),
                      selected=(0, 1, 2, 3), prefill_dual_wave=True, prefill_dma_ring_rows=1024,
                      concurrency=4, reserve_bytes=0, speculation=None)
    with pytest.raises(UsageError, match="share in"):
        planning.plan(KIMI, KIMI.checkpoints[0], tuple(gpu(i, 95) for i in range(8)),
                      selected=tuple(range(8)), prefill_dual_wave=True, prefill_dma_ring_rows=1024,
                      prefill_dual_wave_pool=0.0, **OPTIONS)


class _Arena:
    overlap = True


class _FlatArena:
    overlap = False


def test_dual_wave_gate_selects_the_schedule_exactly_where_intended():
    full = dict(setting=True, mode="prefill", direct_ring=True, workspace=_Arena(), tokens=2048,
                request_count=1, auxiliary_layers=(), auxiliary_source=None)
    assert dual_wave_prefill_enabled(**full) is True
    # 4096-row chunks arm; 1024-row chunks stay on the serial path (eb72cab's floor).
    assert dual_wave_prefill_enabled(**{**full, "tokens": 4096}) is True
    assert dual_wave_prefill_enabled(**{**full, "tokens": 1024}) is False
    for broken in ({"setting": False}, {"mode": "decode"}, {"mode": "verify"}, {"direct_ring": False},
                   {"workspace": None}, {"workspace": _FlatArena()}, {"tokens": 2304},
                   {"tokens": 2047}, {"auxiliary_layers": (3,), "auxiliary_source": "residual_output"}):
        assert dual_wave_prefill_enabled(**{**full, **broken}) is False, broken
    # F6 (l9-transplant-fable-review.md §6): the single-request term is an
    # assertion, not a gate — the dual wave engages at any stream count, and a
    # prefill graph that somehow carried another request count must fail loudly
    # instead of silently keeping the serial path.
    for wrong in (0, 2, -1):
        try:
            dual_wave_prefill_enabled(**{**full, "request_count": wrong})
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted request_count {wrong}")
    # Auxiliary features tapping the attention input keep the schedule.
    assert dual_wave_prefill_enabled(**{**full, "auxiliary_layers": (3,),
                                        "auxiliary_source": "attention_input"}) is True


def test_dual_wave_shape_side_mirrors_the_gate_for_wire_binding():
    # F2 (l9-transplant-fable-review.md finding 3): bootstrap binds the direct
    # ring with exactly this predicate, so the decoder's wire-scope check and the
    # gate cannot disagree about which graphs split.
    assert dual_wave_shape_enabled(True, 3072, workspace=_Arena()) is True
    assert dual_wave_shape_enabled(True, 2048, workspace=_Arena()) is True
    for broken in (dict(setting=False, tokens=3072, workspace=_Arena()),
                   dict(setting=True, tokens=1536, workspace=_Arena()),
                   dict(setting=True, tokens=2304, workspace=_Arena()),
                   dict(setting=True, tokens=3072, workspace=None),
                   dict(setting=True, tokens=3072, workspace=_FlatArena()),
                   dict(setting=True, tokens=3072, workspace=_Arena(), auxiliary_layers=(3,),
                        auxiliary_source="residual_output")):
        assert dual_wave_shape_enabled(**broken) is False, broken
    assert dual_wave_shape_enabled(True, 3072, workspace=_Arena(), auxiliary_layers=(3,),
                                  auxiliary_source="attention_input") is True


def test_wire_scope_fires_when_the_direct_ring_binds_a_serial_graph():
    # F2: the direct-ring runtime binds only dual-wave decoders.
    check_wire_scope(prefill_dual_wave=True, direct_ring=True, dual_wave=True)
    check_wire_scope(prefill_dual_wave=False, direct_ring=False, dual_wave=False)
    check_wire_scope(prefill_dual_wave=True, direct_ring=False, dual_wave=False)
    for wrong in (dict(prefill_dual_wave=True, direct_ring=True, dual_wave=False),
                  dict(prefill_dual_wave=False, direct_ring=True, dual_wave=False),
                  dict(prefill_dual_wave=True, direct_ring=False, dual_wave=True)):
        try:
            check_wire_scope(**wrong)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted {wrong}")


def test_wave_b_active_is_guaranteed_at_plan_time():
    # F5 (l9-transplant-fable-review.md §6): a dual-wave shape arms only with a
    # prepared prefill shape of at least half its rows and strictly below them, so
    # smallest-fit scheduling never replays the shape's graph with wave B empty
    # (the untested lb = 0 KDA pass). Refused shapes stay serial.
    ladder = (128, 256, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096)
    assert dual_wave_armed_shapes(ladder) == (2048, 2560, 3072, 3584, 4096)
    # A shape whose only smaller shapes are under half its rows is refused.
    assert dual_wave_armed_shapes((1024, 2048, 3072)) == (2048, 3072)
    assert dual_wave_armed_shapes((2048, 3072)) == (3072,)
    assert dual_wave_armed_shapes((2048,)) == ()
    # 512-multiples below 2,048 rows never split; odd ladders arm nothing.
    assert dual_wave_armed_shapes((1024, 1536, 2048)) == (2048,)
    assert dual_wave_armed_shapes((100, 2500)) == ()
    assert dual_wave_armed_shapes(()) == ()


def test_pool_cap_bounds_the_second_prepared_module_set():
    import torch

    buffers = {
        "dual_mlp_in": torch.zeros((4096, 7168), dtype=torch.bfloat16),
        "dual_total_w0": torch.zeros((2048, 7168), dtype=torch.bfloat16),
        "dual_total_w1": torch.zeros((2048, 7168), dtype=torch.bfloat16),
        "kda_dw": {"w0": {"output": torch.zeros((2048, 7168), dtype=torch.bfloat16)},
                   "w1": {"output": torch.zeros((2048, 7168), dtype=torch.bfloat16)}},
        "moe_dw": {"w0": {"output": torch.zeros((2048, 7168), dtype=torch.bfloat16)},
                   "w1": {"output": torch.zeros((2048, 7168), dtype=torch.bfloat16)}},
        "kda": {"output_w0": torch.zeros((2048, 7168), dtype=torch.bfloat16),
                "output_w1": torch.zeros((2048, 7168), dtype=torch.bfloat16)},
        # Serial buffers are not counted.
        "total": torch.zeros((4096, 7168), dtype=torch.bfloat16),
    }
    dual_layers = [("w", "kda", object(), object(), object(), object(), object(), object())]
    counted = dual_wave_extra_bytes(buffers, dual_layers)
    row = 7168 * 2
    expected = (4096 + 8 * 2048) * row
    assert counted == expected
    # The default 9% share of a preparation budget that fits prices the set inside the cap.
    assert counted <= 0.09 * (expected / 0.09)
    # The enforcement raises in the decoder when the budget is smaller (its error
    # path is construction-time; the accounting it compares is this function).
    assert counted > 1_000_000


def test_preparation_config_validates_the_pool_share():
    from ignition.recipes.kimi3 import bootstrap

    config = bootstrap.PreparationConfig(devices=tuple(range(8)), prefill_chunk_size=4096,
        max_active=1, vision=False, speculative_widths=(), prefill_dma_ring_rows=1024)
    assert config.prefill_dual_wave is False
    on = replace(config, prefill_dual_wave=True)
    assert on.prefill_dual_wave_pool == 0.09
    with pytest.raises(ValueError, match="share in"):
        replace(config, prefill_dual_wave=True, prefill_dual_wave_pool=0.3)
    with pytest.raises(ValueError, match="boolean"):
        replace(config, prefill_dual_wave=1)


def test_armed_preparation_selects_the_direct_ring_wire(monkeypatch):
    from ignition.preparation import collectives

    created = []
    import b12x.comm.pcie as pcie

    class FakeDma:
        def __init__(self, **kwargs):
            created.append(kwargs)

    monkeypatch.setattr(pcie, "DmaAllReduce", FakeDma)
    collectives.create_dma_all_reduce_runtime(object(), "cuda:0", max_tokens=4096, hidden_size=7168,
                                              ipc=None, bf16_direct=True)
    assert created[-1]["bf16_direct"] is True and "fp24_ring" not in created[-1]
    collectives.create_dma_all_reduce_runtime(object(), "cuda:0", max_tokens=4096, hidden_size=7168,
                                              ipc=None, fp24_ring=True)
    assert created[-1]["fp24_ring"] is True
    collectives.create_dma_all_reduce_runtime(object(), "cuda:0", max_tokens=4096, hidden_size=7168, ipc=None)
    assert "bf16_direct" not in created[-1] and "fp24_ring" not in created[-1]
    with pytest.raises(ValueError, match="mutually exclusive"):
        collectives.create_dma_all_reduce_runtime(object(), "cuda:0", max_tokens=4096, hidden_size=7168,
                                                  ipc=None, bf16_direct=True, fp24_ring=True)
    with pytest.raises(ValueError, match="two-byte activations"):
        collectives.create_dma_all_reduce_runtime(object(), "cuda:0", max_tokens=4096, hidden_size=7168,
                                                  ipc=None, bf16_direct=True, wire_mode="mx")


def test_sm_cap_is_not_stamped_armed_or_not(monkeypatch):
    # F4 (l9-transplant-fable-review.md §5): the B12X_W4A16_NONCOOP_PREFILL_SMS
    # stamp is gone from the recipe. It is process-wide, so it capped every
    # prefill expert launch — serial graphs included — for no measured gain, on a
    # guessed consumer. The knob's reader (b12x w4a16_prefill_sms, flashinfer
    # test_w4a16_prefill_sms.py) stays landed; the stamp returns only on armed
    # A/B evidence.
    import os

    from ignition.recipes.kimi3 import recipe
    from ignition.recipes.kimi3.recipe import DUAL_WAVE_NONCOOP_PREFILL_SMS

    monkeypatch.delenv("B12X_W4A16_NONCOOP_PREFILL_SMS", raising=False)
    assert not hasattr(recipe, "stamp_dual_wave_launch_env"), \
        "the armed recipe must not stamp the SM cap (F4)"
    source = open(recipe.__file__).read()
    assert 'os.environ["B12X_W4A16_NONCOOP_PREFILL_SMS"]' not in source, \
        "recipe.py must not stamp B12X_W4A16_NONCOOP_PREFILL_SMS (F4)"
    assert DUAL_WAVE_NONCOOP_PREFILL_SMS == "170"  # the documented value, unstamped
