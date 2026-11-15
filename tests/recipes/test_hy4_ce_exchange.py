# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0
"""consult18 stage B1: the copy-engine DCP exchange's merge and routing (CPU)."""

import pytest

torch = pytest.importorskip("torch")


def test_merge_matches_the_b12x_reference_merge():
    """The CE merge (sources gathered head-major) equals
    `lse_reduce_scatter_reference` for every destination rank, empty rows included."""
    from b12x.comm.pcie import lse_reduce_scatter_reference
    from ignition.recipes.hy4.collectives import merge_lse_partials

    torch.manual_seed(3)
    world, rows, hpr, width = 8, 5, 2, 16
    heads = world * hpr
    partials = torch.randn(world, rows, heads, width)
    lses = torch.randn(world, rows, heads) * 4
    lses[1, 0, :] = float("-inf")             # a rank with no keys for row 0
    lses[:, 3, 5] = float("-inf")             # a (row, head) nobody saw
    for dest in range(world):
        want, want_lse = lse_reduce_scatter_reference(
            partials, lses, dest, lse_base="base2", return_lse=True, partials_sink_folded=False)
        heads_of = slice(dest * hpr, (dest + 1) * hpr)
        got, got_lse = merge_lse_partials(partials[:, :, heads_of].permute(0, 2, 1, 3),
                                          lses[:, :, heads_of].permute(0, 2, 1))
        assert tuple(got.shape) == tuple(want.shape) and tuple(got_lse.shape) == tuple(want_lse.shape)
        torch.testing.assert_close(got, want.to(got.dtype), rtol=1e-5, atol=1e-6)
        assert torch.equal(torch.isfinite(got_lse), torch.isfinite(want_lse))
        finite = torch.isfinite(want_lse)
        torch.testing.assert_close(got_lse[finite], want_lse[finite].float(), rtol=1e-5, atol=1e-5)


class _FakeCe:
    def __init__(self, gather=None, merge=None):
        self.gather, self.merge, self.calls = gather, merge, []

    def accepts_lse(self, partials):
        return partials.shape[2] == 4

    def accepts_gather(self, local):
        return local.shape[2] == 4

    def lse_reduce_scatter(self, partials, lses, relay=False):
        self.calls.append("lse")
        return self.merge(partials, lses)

    def all_gather_heads(self, local, relay=False):
        self.calls.append("gather")
        return self.gather(local)


def _transport(ce, **kwargs):
    from ignition.recipes.hy4.collectives import PcieDcpTransport

    transport = PcieDcpTransport(group=None, device="cpu", pools={}, rank=0, world_size=2,
                                 total_heads=4, ce=ce, **kwargs)
    kernel_calls = []

    def kernel_gather(local, out=None):
        kernel_calls.append("gather")
        return torch.cat([local, local], dim=1)

    def kernel_lse(partials, lses, out=None, *, lse_base, partials_sink_folded, return_lse):
        kernel_calls.append("lse")
        return partials[:, :2].clone(), lses[:, :2].clone()

    transport._kernel_gather = kernel_gather
    transport._kernel_lse = kernel_lse
    return transport, kernel_calls


def test_ce_routing_thresholds_widths_and_audits():
    good_gather = lambda local: torch.cat([local, local], dim=1)          # noqa: E731
    good_merge = lambda partials, lses: (partials[:, :2].clone(), lses[:, :2].clone())  # noqa: E731
    ce = _FakeCe(good_gather, good_merge)
    transport, kernel = _transport(ce, ce_lse_rows=4, ce_gather_rows=4)
    wide, narrow = torch.randn(4, 2, 4).to(torch.bfloat16), torch.randn(2, 2, 4).to(torch.bfloat16)
    transport.all_gather_heads(wide)
    transport.all_gather_heads(narrow)                        # under the row threshold
    transport.all_gather_heads(torch.randn(4, 2, 8).to(torch.bfloat16))  # other width
    assert ce.calls == ["gather"] and kernel == ["gather", "gather"]
    partials, lses = torch.randn(4, 4, 4).to(torch.bfloat16), torch.randn(4, 4)
    transport.lse_reduce_scatter(partials, lses, partials_sink_folded=False, return_lse=True)
    transport.lse_reduce_scatter(partials, lses, partials_sink_folded=True, return_lse=True)
    assert ce.calls == ["gather", "lse"] and kernel[-1] == "lse"
    off, kernel_off = _transport(_FakeCe(good_gather, good_merge))          # flags off
    off.all_gather_heads(wide)
    off.lse_reduce_scatter(partials, lses, partials_sink_folded=False, return_lse=True)
    assert kernel_off == ["gather", "lse"]
    transport.ce_audit = True
    transport.all_gather_heads(wide)
    transport.lse_reduce_scatter(partials, lses, partials_sink_folded=False, return_lse=True)
    assert transport.ce_audit_calls == {"lse": 1, "gather": 1} and transport.ce_audit_max == 0.0
    bad = _FakeCe(lambda local: torch.cat([local, local + 1], dim=1),
                  lambda partials, lses: (partials[:, :2] * 1.1, lses[:, :2].clone()))
    broken, _ = _transport(bad, ce_lse_rows=4, ce_gather_rows=4)
    broken.ce_audit = True
    with pytest.raises(RuntimeError, match="differs from the kernel gather"):
        broken.all_gather_heads(wide)
    with pytest.raises(RuntimeError, match="CE LSE merge differs"):
        broken.lse_reduce_scatter(partials, lses, partials_sink_folded=False, return_lse=True)


def test_two_level_merge_equals_the_flat_merge():
    """B2's LSE relay merges each switch's four partials, then the two switch
    results; that equals the flat eight-source merge up to rounding, empty
    sources and rows included."""
    from ignition.recipes.hy4.collectives import _merge_head_major

    torch.manual_seed(9)
    sources, heads, rows, width = 8, 2, 6, 16
    partials = torch.randn(sources, heads, rows, width)
    lses = torch.randn(sources, heads, rows) * 6
    lses[0:4, 1, 2] = float("-inf")            # one switch saw nothing for (head 1, row 2)
    lses[:, 0, 4] = float("-inf")              # nobody saw (head 0, row 4)
    flat, flat_lse = _merge_head_major(partials, lses)
    first, first_lse = _merge_head_major(partials[:4], lses[:4])
    second, second_lse = _merge_head_major(partials[4:], lses[4:])
    out, lse = _merge_head_major(torch.stack((first, second)), torch.stack((first_lse, second_lse)))
    torch.testing.assert_close(out, flat, rtol=1e-5, atol=1e-6)
    assert torch.equal(torch.isfinite(lse), torch.isfinite(flat_lse))
    finite = torch.isfinite(flat_lse)
    torch.testing.assert_close(lse[finite], flat_lse[finite], rtol=1e-5, atol=1e-5)
    assert float(out[0, 4].abs().max()) == 0.0


def test_relay_partner_pairs_the_two_switches():
    from ignition.recipes.hy4.collectives import relay_partner

    assert [relay_partner(rank, 8, 4) for rank in range(8)] == [4, 5, 6, 7, 0, 1, 2, 3]
    with pytest.raises(ValueError, match="two islands"):
        relay_partner(0, 8, 2)


def test_relay_flags_route_to_the_relay_paths():
    good = _FakeCe(lambda local: torch.cat([local, local], dim=1),
                   lambda partials, lses: (partials[:, :2].clone(), lses[:, :2].clone()))
    seen = {}
    good.all_gather_heads = lambda local, relay=False: seen.setdefault("gather", relay) or torch.cat([local, local], dim=1)
    good.lse_reduce_scatter = lambda partials, lses, relay=False: (seen.setdefault("lse", relay), (partials[:, :2].clone(), lses[:, :2].clone()))[1]
    transport, _ = _transport(good, ce_lse_rows=4, ce_gather_rows=4, ce_lse_relay=True,
                              ce_gather_relay=False)
    transport.all_gather_heads(torch.randn(4, 2, 4).to(torch.bfloat16))
    transport.lse_reduce_scatter(torch.randn(4, 4, 4).to(torch.bfloat16), torch.randn(4, 4),
                                 partials_sink_folded=False, return_lse=True)
    assert seen == {"gather": False, "lse": True}
