# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""L9 fork/join vehicle (fable): ``launches()`` on the three transport classes, the
padded refusal, and ``launch_transport``'s dispatch — the transport fork's launch
table, unit-tested without a device or a b12x runtime (real methods over real
tensors on bare instances; the CUDA/boot path is gate G1)."""
import pytest
import torch

from ignition.components import collectives
from ignition.components.collectives import (PreparedAllGather, PreparedAllReduce, PreparedChunkedAllGather,
                                            launch_transport)
from ignition.components.owner_packet_reduce import PreparedDirectRingAllReduce


def bare(cls, **attrs):
    """An instance of the real class with only the attributes its real methods read."""
    instance = cls.__new__(cls)
    for key, value in attrs.items():
        setattr(instance, key, value)
    return instance


def test_all_reduce_launches_returns_source_output_handle():
    source, output = torch.zeros(4, 8), torch.zeros(4, 8)
    operation = bare(PreparedAllReduce, single=False, output=output, handle=17)
    assert operation.launches(source) == [(source, output, 17)]


def test_all_reduce_launches_refuses_single_rank():
    operation = bare(PreparedAllReduce, single=True, output=torch.zeros(4, 8), handle=1)
    with pytest.raises(ValueError, match="single-rank"):
        operation.launches(torch.zeros(4, 8))


def test_all_gather_launches_returns_source_output_handle():
    source, output = torch.zeros(4, 8), torch.zeros(4, 32)
    operation = bare(PreparedAllGather, single=False, output=output, handle=23)
    assert operation.launches(source) == [(source, output, 23)]


def test_all_gather_launches_refuses_single_rank():
    operation = bare(PreparedAllGather, single=True, output=torch.zeros(4, 8), handle=1)
    with pytest.raises(ValueError, match="single-rank"):
        operation.launches(torch.zeros(4, 8))


def test_chunked_all_gather_launches_one_triple_per_part():
    source = torch.arange(4 * 8, dtype=torch.bfloat16).view(4, 8)
    output = torch.zeros(4, 32)
    parts = [bare(PreparedAllGather, single=False, output=output[:2], handle=3),
             bare(PreparedAllGather, single=False, output=output[2:], handle=4)]
    operation = bare(PreparedChunkedAllGather, bounds=[(0, 2), (2, 4)], parts=parts, output=output)
    launches = operation.launches(source)
    assert [(tuple(s.shape), h) for s, _, h in launches] == [((2, 8), 3), ((2, 8), 4)]
    assert launches[0][1].data_ptr() == output[:2].data_ptr() and launches[1][1].data_ptr() == output[2:].data_ptr()
    assert launches[0][0].data_ptr() == source[:2].data_ptr()
    assert launches[1][0].data_ptr() == source[2:].data_ptr()


def test_direct_ring_launches_refuses_padded_staging():
    source = torch.zeros(4, 8, dtype=torch.bfloat16)
    output = torch.zeros(4, 8, dtype=torch.bfloat16)
    operation = bare(PreparedDirectRingAllReduce, padded=torch.zeros(1, dtype=torch.bfloat16),
                     output=output, elements=32)
    with pytest.raises(ValueError, match="padded direct ring"):
        operation.launches(source)


def test_direct_ring_launches_delegates_to_its_inner_reduction():
    source = torch.zeros(4, 8, dtype=torch.bfloat16)
    output = torch.zeros(4, 8, dtype=torch.bfloat16)
    inner = bare(PreparedAllReduce, single=False, output=output, handle=9)
    operation = bare(PreparedDirectRingAllReduce, padded=None, reduce=inner, output=output, elements=32)
    assert operation.launches(source) == [(source, output, 9)]


class _GatherRuntime:
    heads_per_rank, query_head_dim, total_heads = 2, 4, 8

    def __init__(self):
        self.calls = []

    def all_gather_heads(self, source, *, out, plan):
        self.calls.append((source, out, plan))


def _register(owner):
    handle = 1000 + len(collectives._OWNERS)
    collectives._OWNERS[handle] = owner
    return handle


def test_launch_transport_dispatches_gathers_through_gather_columns():
    source = torch.zeros(2, 8, dtype=torch.bfloat16)
    output = torch.zeros(2, 32, dtype=torch.bfloat16)
    runtime = _GatherRuntime()
    owner = bare(PreparedAllGather, single=False, runtime=runtime, roce=False, plan="plan",
                 output=output, handle=0)
    handle = _register(owner)
    launch_transport(source, output, handle)
    assert len(runtime.calls) == 1
    gathered, out, plan = runtime.calls[0]
    assert gathered.shape == (2, 2, 4) and out.shape == (2, 8, 4) and plan == "plan"
    assert gathered.data_ptr() == source.data_ptr() and out.data_ptr() == output.data_ptr()


def test_launch_transport_dispatches_reductions_through_launch():
    source = torch.zeros(2, 8, dtype=torch.bfloat16)
    output = torch.zeros(2, 8, dtype=torch.bfloat16)
    calls = []
    owner = bare(PreparedAllReduce, single=False, output=output, handle=0,
                 launch=lambda s, o: calls.append((s, o)))
    handle = _register(owner)
    launch_transport(source, output, handle)
    assert calls == [(source, output)]


def test_launch_transport_rejects_other_owners():
    class _Other:
        pass

    owner = _Other()
    handle = _register(owner)
    with pytest.raises(TypeError, match="prepared reduction or gather"):
        launch_transport(torch.zeros(2, 8), torch.zeros(2, 8), handle)
