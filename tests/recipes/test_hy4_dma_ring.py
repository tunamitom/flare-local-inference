# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0
"""consult18 stage A1: the copy-engine ring all-reduce arm (CPU, fake runtimes)."""

import pytest

torch = pytest.importorskip("torch")

from tests.recipes.test_hy4_w6_cp import tiny_available, worlds  # noqa: E402,F401


# --------------------------------------------------------------------------
# consult18 stage A1: the copy-engine ring arm
# --------------------------------------------------------------------------
def test_dma_arm_parses_and_refuses_a_launch_pin():
    from ignition.recipes.hy4.collectives import all_reduce_arm_for, parse_all_reduce_arms

    rules = parse_all_reduce_arms("1:oneshot,2-511:rsag,512-:dma")
    assert all_reduce_arm_for(1, rules)[0] == "oneshot"
    assert all_reduce_arm_for(256, rules)[0] == "rsag"
    assert all_reduce_arm_for(512, rules) == ("dma", None, None)
    assert all_reduce_arm_for(1024, rules) == ("dma", None, None)
    with pytest.raises(ValueError, match="no thread/block launch"):
        parse_all_reduce_arms("1:oneshot,2-:dma@256x16")


def _ring_transport(**kwargs):
    from ignition.recipes.hy4.collectives import PcieDcpTransport

    return PcieDcpTransport(group=None, device="cpu", pools={}, rank=0, world_size=8,
                            total_heads=64, all_reduce=object(), reduce_width=8,
                            all_reduce_arms="1:oneshot,2-:dma", **kwargs)


class _FakeSum:
    """Stands in for a b12x runtime: `all_reduce` returns 8 x the input."""

    def __init__(self, bias=0.0):
        self.calls, self.bias = [], bias

    def all_reduce(self, inp, *, plan, out):
        self.calls.append(plan)
        out.copy_(inp.float().mul(8).add(self.bias).to(out.dtype))
        return out


def test_dma_arm_runs_the_ring_and_the_audit_compares_it_with_rsag():
    from ignition.recipes.hy4.collectives import AR_AUDIT_REL_L2

    value = torch.randn(4, 8).to(torch.bfloat16)
    transport = _ring_transport()
    ring, kernel = _FakeSum(), _FakeSum()
    transport._ar_plans[4] = ("ring-plan", ring, "dma")
    transport._ar_audit_plans[4] = ("rsag-plan", kernel)
    out = transport.all_reduce(value)
    assert ring.calls == ["ring-plan"] and kernel.calls == []      # audit off: no extra work
    assert torch.equal(out, value.float().mul(8).to(torch.bfloat16))
    transport.ar_audit = True
    transport.all_reduce(value)
    assert kernel.calls == ["rsag-plan"] and transport.ar_audit_calls == 1
    assert transport.ar_audit_max <= AR_AUDIT_REL_L2
    transport._ar_plans[4] = ("ring-plan", _FakeSum(bias=5.0), "dma")   # a wrong sum
    with pytest.raises(RuntimeError, match="differs from the rsag arm"):
        transport.all_reduce(value)


def test_kernel_pool_size_guard_and_ring_replay_capacities():
    transport = _ring_transport(kernel_rows=256)
    with pytest.raises(RuntimeError, match="kernel pool holds 256 rows"):
        transport._prepare_kernel_arm(None, 512, "rsag", None, None, "x")

    class FakeRing:
        min_bytes = 0
        _eager_replays = {}

        def prepare_eager_replay(self, dtype, *, max_elements):
            self.seen = (dtype, max_elements, self.min_bytes)
            self._eager_replays = {dtype: ("g1", "g2")}

    ring = FakeRing()
    transport = _ring_transport(dma=ring)
    assert transport.prepare_dma_replay() == 0                     # no ring plans yet
    transport._ar_plans[1] = ("p", None, "oneshot")
    transport._ar_plans[512] = ("p", ring, "dma")
    transport._ar_plans[1024] = ("p", ring, "dma")
    assert transport.prepare_dma_replay() == 2
    assert ring.seen == (torch.bfloat16, 1024 * 8, 512 * 8 * 2)
