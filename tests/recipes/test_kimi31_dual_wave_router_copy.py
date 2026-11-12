# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The dual-wave router-logit copy (l9-final-whole-set-fable-review.md § 1).

``phase_down_pack``'s ``down_gather`` branch feeds the forked router gather, and
the gather's source is ``router_input``. The serial ``__call__`` copies the local
router logits into that buffer before its gather on both prefill branches; the
dual-wave prologue must do the same or the router gathers zeros and every row
routes to the same experts.

Fable's required test: on a module with ``down_gather`` set, ``phase_down_pack``,
the gathers and ``phase_router`` must select the routes ``__call__`` selects.
"""

from types import SimpleNamespace

import torch

from ignition.recipes.kimi3.decoder import PreparedLatentMoE

TOKENS, EXPERTS, PICK, HIDDEN = 8, 32, 4, 64


class FakeProjection:
    """Writes per-row logits into the module's ``router_logits`` buffer."""

    def __init__(self, logits):
        self.logits = logits

    def __call__(self, source):
        self.logits.copy_(source[:, :EXPERTS].float())
        return self.logits


class FakeGather:
    """A gather whose transport execution copies its source into its output."""

    def __init__(self, output):
        self.output = output
        self.seen = []

    def __call__(self, source):
        self.seen.append(source)
        self.output.copy_(source.view(torch.float32)[:, : self.output.shape[1]])
        return self.output


class FakeDown:
    def __init__(self):
        self.output = torch.zeros((TOKENS, HIDDEN))

    def __call__(self, source):
        self.output.copy_(source[:, :HIDDEN])
        return self.output


class FakeRouter:
    """Top-k selection: the route set is a pure function of the logits it reads."""

    def __init__(self, routes):
        self.routes = routes
        self.reads = []

    def __call__(self, logits):
        self.reads.append(logits.clone())
        top = torch.topk(logits.float(), PICK, dim=1)
        self.routes.ids.copy_(top.indices.to(torch.int32))
        self.routes.weights.copy_(top.values.to(torch.float32))


class FakeShared:
    concurrent = False

    def __init__(self):
        self.output = torch.zeros((TOKENS, HIDDEN))

    def __call__(self, source):
        self.output.copy_(source[:, :HIDDEN])
        return self.output


def build():
    """A real-class instance carrying only the attributes the real methods read.

    Buffer wiring mirrors ``PreparedLatentMoE.__init__``'s no-route-gather
    branch at world size 1: ``router_output`` is the projection's ``logits``
    buffer viewed per rank and ``router_columns`` the gathered buffer's local
    block, so ``phase_router``'s copy lands where the router reads. Returns
    ``(module, fakes)``; the fakes are read directly (the instance attributes
    are assigned dynamically, as the real construction does).
    """
    logits = torch.zeros((TOKENS, EXPERTS))
    router_input = torch.zeros((TOKENS, EXPERTS), dtype=torch.float32)
    gathered = torch.zeros((TOKENS, EXPERTS), dtype=torch.float32)
    routes = SimpleNamespace(ids=torch.zeros((TOKENS, PICK), dtype=torch.int32),
                             weights=torch.zeros((TOKENS, PICK)))
    fakes = {"router": FakeRouter(routes), "projection": FakeProjection(logits),
             "router_gather": FakeGather(gathered), "down": FakeDown(),
             "down_gather": FakeGather(torch.zeros((TOKENS, HIDDEN)))}
    attrs = {"router_logits": logits, "router_input": router_input,
             "gathered_logits": gathered,
             "router_columns": gathered.view(TOKENS, 1, EXPERTS)[:, :, :EXPERTS],
             "router_output": logits.view(TOKENS, 1, EXPERTS),
             "routes": routes, "route_gather": None,
             "down_gather": fakes["down_gather"],
             "down_reduce": FakeGather(torch.zeros((TOKENS, HIDDEN))),
             "router_gather": fakes["router_gather"],
             "down": fakes["down"], "router_projection": fakes["projection"],
             "router": fakes["router"], "expert_output": torch.zeros((TOKENS, HIDDEN)),
             "latent_sum": torch.randn((TOKENS, HIDDEN)),
             "latent_norm": torch.ones((HIDDEN // 2,)),
             "up_input": torch.zeros((TOKENS, HIDDEN // 2)),
             "eps": 1e-5, "latent_begin": None,
             "output": torch.zeros((TOKENS, HIDDEN)), "shared": FakeShared()}
    module = PreparedLatentMoE.__new__(PreparedLatentMoE)
    for key, value in attrs.items():
        setattr(module, key, value)
    module.experts = lambda: module.expert_output
    module.expert_reduce = lambda out: out
    module.up = lambda up_input: up_input
    return module, fakes


def test_phase_down_pack_routes_match_call_routes():
    """The dual-wave phase chain must select exactly the routes ``__call__`` selects."""
    torch.manual_seed(20261113)
    source = torch.randn((TOKENS, HIDDEN))
    # Per-row distinct logits (the arange fixture gave every row the same top-k).
    source[:, :EXPERTS] = torch.rand((TOKENS, EXPERTS)) * 10.0

    # Serial reference: the real forward.
    serial, sf = build()
    serial(source)
    serial_ids = serial.routes.ids.clone()
    serial_reads = sf["router"].reads[-1].clone()

    # Dual-wave chain: the real prologue, the forked gathers as the transport
    # executes them, then the real phase_router.
    dual, df = build()
    dual.router_input.zero_()  # the buffer must be written by phase_down_pack itself
    dual.phase_down_pack(source)
    for operation, launch_source in dual.route_ring_launches():
        operation(launch_source)
    dual.phase_router()

    # The forked router gather takes the router_input buffer the prologue writes.
    assert df["router_gather"].seen, "the route ring never ran the router gather"
    assert df["router_gather"].seen[0].data_ptr() == dual.router_input.data_ptr()
    # Dual-wave routing equals the serial forward's routing, per row.
    assert torch.equal(dual.routes.ids, serial_ids), "dual-wave routes differ from __call__ routes"
    assert torch.equal(df["router"].reads[-1], serial_reads), "the router read different logits"
    # ...and it is per-row routing, not the zero-logit single-set selection.
    sets = {tuple(sorted(row.tolist())) for row in dual.routes.ids}
    assert len(sets) == TOKENS, f"rows collapsed onto {len(sets)} distinct expert sets"
