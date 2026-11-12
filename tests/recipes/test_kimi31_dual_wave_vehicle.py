# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""L9 fork/join vehicle (fable): the two ops' schemas and fakes, and the traced
surface through the real worker entry — handles and the transport stream must
enter the artifact as symbolic graph inputs (aot.py's unbacked-sources pattern
matches only ``.handle`` / ``.stream_handle`` attribute paths; trap 1). A baked
constant handle would mean no sharing across ranks, a guard failure on every
later boot, and a recompile each time."""
import torch

from ignition.preparation.aot import aot_forward
from ignition.preparation.forward_metadata import dumps, loads
from ignition.recipes.kimi3.decoder import TransportStream, _fork_transport, _join_transport


class _Ring:
    """The launch surface the real PreparedAllReduce.launches exposes."""

    def __init__(self, output, handle):
        self.single, self.output, self.handle = False, output, handle

    def launches(self, source):
        return [(source, self.output, self.handle)]


class _VehicleStep:
    """One overlap pair as the traced forward sees it: launches through the real
    method, fork and join through the real ops, compute between them."""

    def __init__(self, ring, source, output, transport):
        self.ring, self.source, self.output = ring, source, output
        self.dual_transport = TransportStream(stream_handle=transport)

    def __call__(self):
        sources, outputs, handles = [], [], []
        for operation, source in ((self.ring, self.source),):
            for item_source, item_output, handle in operation.launches(source):
                sources.append(item_source)
                outputs.append(item_output)
                handles.append(handle)
        transport = self.dual_transport.stream_handle
        _fork_transport(sources, outputs, handles, transport)
        result = self.source.add(1)
        _join_transport(sources, outputs, [result], transport)
        return result


class _MisnamedStep(_VehicleStep):
    """Trap 1's defect class: the handle read through a ``comm_stream_handle``-style
    attribute — the character before ``stream_handle`` is not a dot, so aot.py's
    unbacked-sources pattern misses it and the int is baked into the artifact."""

    def __init__(self, ring, source, output, transport):
        super().__init__(ring, source, output, transport)
        self.comm_stream_handle = transport

    def __call__(self):
        sources, outputs, handles = [], [], []
        for operation, source in ((self.ring, self.source),):
            for item_source, item_output, handle in operation.launches(source):
                sources.append(item_source)
                outputs.append(item_output)
                handles.append(handle)
        _fork_transport(sources, outputs, handles, self.comm_stream_handle)
        return self.source


def _compile(step):
    """The worker's entry: pickle the forward, then the real aot compile."""
    return aot_forward(loads(dumps(step)))


def _integer_inputs(backend):
    return [entry for entry in backend.signature if entry is int]


def test_fork_join_schemas_mutate_outputs_and_take_symbolic_ints():
    fork = str(torch.ops.flare.kimi3_fork_transport.default._schema)
    join = str(torch.ops.flare.kimi3_join_transport.default._schema)
    assert fork == "flare::kimi3_fork_transport(Tensor[] sources, Tensor(a1!)[] outputs, " \
                   "SymInt[] handles, SymInt transport_stream) -> ()"
    assert join == "flare::kimi3_join_transport(Tensor[] sources, Tensor(a1!)[] outputs, " \
                   "Tensor[] dependency, SymInt transport_stream) -> ()"


def test_fork_join_fakes_do_nothing_on_meta_tensors():
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        sources = [torch.zeros(2, 2), torch.zeros(2, 2)]
        outputs = [torch.zeros(2, 2), torch.zeros(2, 2)]
        torch.ops.flare.kimi3_fork_transport.default(sources, outputs, [3, 4], 99)
        torch.ops.flare.kimi3_join_transport.default(sources, outputs, [torch.zeros(2, 2)], 99)


def test_traced_vehicle_takes_handles_and_stream_as_graph_inputs():
    source, output = torch.zeros(4, 8), torch.zeros(4, 8)
    step = _VehicleStep(_Ring(output, 7), source, output, transport=1234)
    backend = _compile(step)._artifacts.compiled_fn
    # Two unbacked integers entered the artifact: the plan handle and the stream.
    assert len(_integer_inputs(backend)) == 2
    fork = [call for call in backend.invocations
            if call.target is torch.ops.flare.kimi3_fork_transport.default]
    join = [call for call in backend.invocations
            if call.target is torch.ops.flare.kimi3_join_transport.default]
    # Program order: fork, join; non-empty integer operands on both.
    assert [call.target for call in backend.invocations][-2:] == \
        [torch.ops.flare.kimi3_fork_transport.default, torch.ops.flare.kimi3_join_transport.default]
    assert len(fork) == 1 and len(join) == 1
    assert fork[0].integers and join[0].integers
    # Both the plan handle (leaf 2) and the transport stream (leaf 3) are symbolic.
    assert fork[0].integers == [2, 3] and join[0].integers == [3]
    # The fork reads the source and mutates the output; the join reads the
    # compute result as its dependency.
    assert join[0].operands


def test_misnamed_stream_handle_is_baked_a_constant():
    source, output = torch.zeros(4, 8), torch.zeros(4, 8)
    step = _MisnamedStep(_Ring(output, 7), source, output, transport=1234)
    backend = _compile(step)._artifacts.compiled_fn
    # Only the plan handle survives as a graph input; the comm_stream_handle value
    # was baked as a constant — the defect trap 1 exists to prevent.
    assert len(_integer_inputs(backend)) == 1
    fork = [call for call in backend.invocations
            if call.target is torch.ops.flare.kimi3_fork_transport.default][0]
    assert fork.integers == [2] and fork.leaves[3] == 1234
