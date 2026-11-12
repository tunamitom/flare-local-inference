# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""F3 (l9-transplant-fable-review.md § mandate 2): the dual-wave aliasing assertions.

The two-wave schedule rests on an aliasing argument (sibling stage cursors reuse
storage; wave outputs of the different kinds alias by lifetime; the parent-cursor
buffers are safe by construction order; one staging-backed ring op per phase).
This module asserts it at dual-set construction. Each assertion must FIRE on a
violating layout — these tests trigger every one — and pass on the schedule's
geometry (the phase table of dual_wave_alias_phases over disjoint wave sets).
"""

from types import SimpleNamespace

import pytest
import torch

from ignition.recipes.kimi3.decoder import (DualWaveAliasCheck, assert_parent_buffers_isolated,
                                            assert_phase_disjoint, assert_single_staging_op,
                                            dual_wave_alias_phases)

HIDDEN, HALF, ROWS = 256, 64, 128


def bf16(*shape):
    return torch.zeros(shape, dtype=torch.bfloat16)


def fake_kda():
    return SimpleNamespace(buffers={"output": bf16(HALF, HIDDEN), "input": bf16(HALF, 32),
                                    "mixed": bf16(HALF, 96), "forget_input": bf16(HALF, 8)})


def fake_mlp():
    return SimpleNamespace(input=bf16(HALF, 32), gate_up=SimpleNamespace(output=bf16(HALF, 64)),
                           down=SimpleNamespace(output=bf16(HALF, HIDDEN)))


def fake_moe():
    module = SimpleNamespace(
        router_input=torch.zeros((HALF, 32), dtype=torch.float32),
        gathered_logits=torch.zeros((HALF, 128), dtype=torch.float32),
        router_logits=torch.zeros((HALF, 32), dtype=torch.float32),
        source=bf16(HALF, 32), latent=bf16(HALF, 64), expert_output=bf16(HALF, 64),
        up_input=bf16(HALF, 32), output=bf16(HALF, HIDDEN),
        routes=SimpleNamespace(ids=torch.zeros((HALF, 4), dtype=torch.int32),
                               weights=torch.zeros((HALF, 4), dtype=torch.float32)),
        down=SimpleNamespace(output=bf16(HALF, 64)), shared=fake_mlp(),
        down_reduce=SimpleNamespace(padded=None), expert_reduce=SimpleNamespace(padded=None))
    module.latent_sum = module.latent
    return module


def ring(*, staging=False):
    return SimpleNamespace(padded=bf16(1, 1) if staging else None)


def state():
    residual = SimpleNamespace(prefix=bf16(ROWS, HIDDEN), blocks=bf16(ROWS, 2, HIDDEN),
                               output=bf16(ROWS, HIDDEN))
    return dict(residual=residual, delta=bf16(ROWS, HIDDEN), attn_in=residual.output,
                mlp_in=bf16(ROWS, HIDDEN), tot_a=bf16(HALF, HIDDEN), tot_b=bf16(HALF, HIDDEN),
                auxiliary=None, half=HALF)


def kda_layer(alias_wave_output=False):
    kda_a, kda_b = fake_kda(), fake_kda()
    out_p0, out_p1 = kda_a.buffers["output"], kda_b.buffers["output"]
    if alias_wave_output:
        kda_b.buffers = dict(kda_b.buffers, output=out_p0)
    m_a, m_b = fake_mlp(), fake_mlp()
    mlp_p0, mlp_p1 = m_a.down.output, m_b.down.output
    return [["w", "kda", (kda_a, kda_b, out_p0, out_p1), ("dense", m_a, m_b, mlp_p0, mlp_p1),
             ring(), ring(), ring(), ring()]]


def moe_layer():
    m_a, m_b = fake_moe(), fake_moe()
    out_p0, out_p1 = bf16(HALF, HIDDEN), bf16(HALF, HIDDEN)
    attention = SimpleNamespace(out_pads=(out_p0, out_p1))
    return [("w", "mla", attention,
             ("moe", m_a, m_b, m_a.output[:HALF], m_b.output[:HALF]), ring(), ring(), ring(), ring())]


def test_phase_disjointness_assertion_fires_on_shared_storage():
    shared = bf16(ROWS, HIDDEN)
    # Overlapping views of one storage on opposite sides: the J-fault class.
    with pytest.raises(RuntimeError, match="share storage"):
        assert_phase_disjoint("phase", [("comm", shared[:HALF])], [("comp", shared[HALF - 4:])])
    with pytest.raises(RuntimeError, match="share storage"):
        assert_phase_disjoint("phase", [("comm", shared)], [("comp", shared)])
    # Disjoint tensors pass.
    assert assert_phase_disjoint("phase", [("comm", bf16(HALF, HIDDEN))], [("comp", bf16(HALF, HIDDEN))]) == []
    assert assert_phase_disjoint("phase", [("comm", shared[:HALF])], [("comp", shared[HALF:])]) == []


def test_parent_buffer_isolation_assertion_fires():
    live = bf16(4, HIDDEN)
    parent = bf16(ROWS, HIDDEN)
    # A parent-cursor buffer sharing storage with a live graph tensor fires.
    with pytest.raises(RuntimeError, match="parent-cursor"):
        assert_parent_buffers_isolated([("dual_mlp_in", parent)],
                                       [("residual.prefix", parent), ("live", live)])
    # The buffer's own entry (same label) is not a violation; disjoint passes.
    assert assert_parent_buffers_isolated([("dual_mlp_in", parent)],
                                          [("dual_mlp_in", parent), ("live", live)]) == []


def test_single_staging_op_assertion_fires():
    with pytest.raises(RuntimeError, match="staging"):
        assert_single_staging_op("phase", 2)
    assert_single_staging_op("phase", 0)
    assert_single_staging_op("phase", 1)
    with pytest.raises(ValueError):
        assert_single_staging_op("phase", -1)


def test_alias_check_fires_on_a_violating_phase():
    check = DualWaveAliasCheck("prefill3072")
    shared = bf16(ROWS, HIDDEN)
    with pytest.raises(RuntimeError, match="share storage"):
        check.phase("layer0.attn_reduce.w0", [("out_p0", shared)], [("waveB.output", shared[HALF:])], 0)


def test_phase_table_passes_on_the_schedule_geometry():
    # One KDA + dense layer and one MLA + MoE layer: 4 + 8 overlap phases.
    layers = (kda_layer()[0], moe_layer()[0])
    phases = dual_wave_alias_phases(layers, **state())
    assert len(phases) == 4 + 8, [label for label, *_ in phases]
    check = DualWaveAliasCheck("prefill3072")
    for label, comm, comp, staging_ops in phases:
        check.phase(label, comm, comp, staging_ops)
    parents = [("dual_mlp_in", bf16(ROWS, HIDDEN)), ("dual_total_w0", bf16(HALF, HIDDEN))]
    assert check.finish(parents, [("live", bf16(8, HIDDEN))]) == 12


def test_phase_table_fires_when_wave_storage_aliases():
    # The wave-B KDA output aliases wave-A's reduced output: phase 1 must fire.
    layers = [kda_layer(alias_wave_output=True)[0]]
    with pytest.raises(RuntimeError, match="share storage"):
        for label, comm, comp, staging_ops in dual_wave_alias_phases(layers, **state()):
            assert_phase_disjoint(label, comm, comp)
    # The wave-B dense MLP writes where wave-A's output reduction reads: the
    # mlp_reduce.w0 phase must fire.
    layer = kda_layer()[0]
    layer[3] = ("dense", layer[3][1], layer[3][1], layer[3][3], layer[3][3])
    with pytest.raises(RuntimeError, match="share storage"):
        for label, comm, comp, staging_ops in dual_wave_alias_phases([layer], **state()):
            assert_phase_disjoint(label, comm, comp)


def test_alias_check_fires_on_two_staging_backed_ops_in_one_phase():
    # The schedule puts one ring operation in each overlap phase; a second
    # staging-backed one would share prefill_direct_staging concurrently.
    check = DualWaveAliasCheck("prefill3072")
    with pytest.raises(RuntimeError, match="staging"):
        check.phase("layer0.attn_reduce.w0", [("out_p0", bf16(HALF, HIDDEN))],
                    [("waveB.output", bf16(HALF, HIDDEN))], 2)


def test_phase_table_fires_when_parent_buffers_alias_graph_tensors():
    layers = (kda_layer(),)
    s = state()
    # dual_mlp_in is the wave-A MLP target: a live tensor sharing it fires F3.2.
    with pytest.raises(RuntimeError, match="parent-cursor"):
        assert_parent_buffers_isolated([("dual_mlp_in", s["mlp_in"])],
                                       [("dual_mlp_in", s["mlp_in"]), ("residual.prefix", s["mlp_in"][:4])])
