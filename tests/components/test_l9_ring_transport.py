# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""L9 transport pieces: direct-ring aligned extents, the direct ring reduction, and reduce-scatter.

The ring schedules themselves are proven in flashinfer's CPU random-interleaving
emulation (`tests/experimental/b12x/comm/test_pcie_dma_direct_ring.py`, world 8
and 9); these cases pin the flare-side construction contracts and the
`flare::pcie_reduce_scatter` dispatch on CPU.
"""

import torch

from b12x.comm import pcie

from ignition.components.collectives import PreparedReduceScatter, _OWNERS, _reduce_scatter
from ignition.components.owner_packet_reduce import (
    PreparedDirectRingAllReduce,
    copy_engine_reduction,
    direct_aligned_extent,
)


class FakeDmaRuntime:
    """Minimal direct-channel stand-in: enough identity for plan construction."""

    def __init__(self, world=8, rank=0, max_bytes=3072 * 7168 * 2):
        from b12x.comm.pcie import pcie_dma

        self._runtime = pcie_dma.PCIeDmaAllReduce.__new__(pcie_dma.PCIeDmaAllReduce)
        rt = self._runtime
        rt.world_size, rt.rank, rt.device = world, rank, "fake"
        rt._closed = False
        rt._direct_mode = "fp32"
        rt.max_bytes = max_bytes
        rt._fp8 = ""
        rt._pieces_override = 0
        rt._a2a_chunks_override = 0
        rt.shard_capacity = 1 << 20
        rt._fp8_stage = None
        rt._fp8_stage_stride = 0
        rt._direct_stage = torch.zeros(2 * rt.shard_capacity, dtype=torch.uint8)
        rt._direct_max_shard_elems = 1 << 19
        rt.group = object()
        self.reduce_scatter_calls = []

    def __getattr__(self, item):
        return getattr(self._runtime, item)

    def reduce_scatter(self, source, *, out, plan):
        self.reduce_scatter_calls.append((source.shape, out.shape, plan))


class RecordingRuntime(FakeDmaRuntime, pcie.DmaAllReduce):
    """Passes isinstance(runtime, DmaAllReduce) while recording reduce_scatter calls."""


def _patch_declaration(monkeypatch):
    import ignition.components.collectives as collectives
    import torch.distributed as dist

    captured = {}

    def fake_declare(session, requests, *, coordinator=None):
        captured["requests"] = requests

    monkeypatch.setattr(collectives, "declare", fake_declare)
    monkeypatch.setattr(dist, "get_process_group_ranks", lambda group: tuple(range(8)))
    return captured


def test_direct_aligned_extent_rounds_rank_shards_to_256_element_groups():
    assert direct_aligned_extent(128 * 8 * 256, 8) == 128 * 256
    # Unaligned payloads round the per-rank extent up, never down.
    assert direct_aligned_extent(1000, 8) == 256
    assert direct_aligned_extent(256 * 8, 8) == 256
    assert direct_aligned_extent(257 * 8, 8) == 512
    for elements in (1, 127, 2048, 3072 * 7168):
        for world in (2, 8, 9):
            extent = direct_aligned_extent(elements, world)
            assert extent % 256 == 0
            assert extent * world >= elements
            # Two ring pieces of the padded extent stay 128-element aligned.
            assert (extent // 2) % 128 == 0 or extent % 256 == 0
    for bad in (0, -1, 1.5):
        try:
            direct_aligned_extent(bad, 8)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted bad extent {bad}")


def test_reduce_scatter_requires_a_direct_ring_channel():
    from b12x.comm.pcie import pcie_dma

    plain = pcie_dma.PCIeDmaAllReduce.__new__(pcie_dma.PCIeDmaAllReduce)
    plain._direct_mode = ""
    src = torch.zeros((8, 128, 128), dtype=torch.bfloat16)
    out = torch.zeros((128, 128), dtype=torch.bfloat16)
    try:
        PreparedReduceScatter(src, out, runtime=plain, session=object(), coordinator=object(), name="x")
    except ValueError as error:
        assert "direct" in str(error)
    else:
        raise AssertionError("accepted a non-direct channel")


def test_reduce_scatter_geometry_contract():
    runtime = RecordingRuntime()
    src = torch.zeros((8, 128, 128), dtype=torch.bfloat16)
    out = torch.zeros((128, 128), dtype=torch.bfloat16)
    cases = [
        (src.float(), out, "dtype"),
        (src, out.float(), "dtype"),
        (src, torch.zeros((64, 128), dtype=torch.bfloat16), "output shape"),
        (src.reshape(16, 64, 128)[::2], out, "contiguous"),
    ]
    for source, output, label in cases:
        try:
            PreparedReduceScatter(source, output, runtime=runtime, session=object(),
                                  coordinator=object(), name="x")
        except ValueError as error:
            assert "PreparedReduceScatter requires contiguous 128-aligned BF16" in str(error), (label, error)
        else:
            raise AssertionError(f"accepted {label} violation")
    # 128-element shard alignment: 100 elements per rank is rejected.
    bad = torch.zeros((8, 100), dtype=torch.bfloat16)
    try:
        PreparedReduceScatter(bad, torch.zeros((100,), dtype=torch.bfloat16), runtime=runtime,
                              session=object(), coordinator=object(), name="x")
    except ValueError as error:
        assert "128-aligned" in str(error)
    else:
        raise AssertionError("accepted unaligned shards")


def test_reduce_scatter_constructs_and_dispatches_through_the_custom_op(monkeypatch):
    captured = _patch_declaration(monkeypatch)
    runtime = RecordingRuntime()
    source = torch.zeros((8, 128, 128), dtype=torch.bfloat16)
    output = torch.zeros((128, 128), dtype=torch.bfloat16)
    prepared = PreparedReduceScatter(source, output, runtime=runtime, session=object(),
                                     coordinator=object(), name="layer.expert_rs")
    assert prepared.output is output
    assert _OWNERS[prepared.handle] is prepared
    request = captured["requests"][0]
    assert request.name == "layer.expert_rs"
    assert request.collective.key == "layer.expert_rs"
    # The replayed call drives the channel's reduce_scatter on the exact buffers.
    _reduce_scatter(source, output, prepared.handle)
    assert runtime.reduce_scatter_calls == [(tuple(source.shape), tuple(output.shape), prepared.plan)]
    assert prepared(source) is output
    assert len(runtime.reduce_scatter_calls) == 2


def test_direct_ring_reduction_keeps_aligned_shapes_and_pads_unaligned(monkeypatch):
    _patch_declaration(monkeypatch)
    runtime = RecordingRuntime(world=8)
    session, coordinator = object(), object()
    # Aligned: no staging, one direct reduction over the caller's buffers.
    source = torch.zeros((256, 2048), dtype=torch.bfloat16)
    aligned = PreparedDirectRingAllReduce(source, source.clone(), runtime=runtime,
                                          session=session, coordinator=coordinator, name="a")
    assert aligned.padded is None
    # Unaligned: padded staging, the reduction runs over the padded packet view.
    source = torch.zeros((3, 1000), dtype=torch.bfloat16)
    staging = torch.zeros((direct_aligned_extent(source.numel(), 8) * 8,), dtype=torch.bfloat16)
    padded = PreparedDirectRingAllReduce(source, source.clone(), runtime=runtime, staging=staging,
                                         session=session, coordinator=coordinator, name="b")
    assert padded.padded.data_ptr() == staging.data_ptr()
    assert padded.padded.numel() == direct_aligned_extent(3000, 8) * 8
    assert padded.packet.shape == (8, direct_aligned_extent(3000, 8))
    try:
        PreparedDirectRingAllReduce(source, source.clone(), runtime=runtime,
                                    session=session, coordinator=coordinator, name="c")
    except ValueError as error:
        assert "staging" in str(error)
    else:
        raise AssertionError("accepted unaligned shapes without staging")


def test_copy_engine_reduction_selects_the_direct_ring(monkeypatch):
    _patch_declaration(monkeypatch)
    runtime = RecordingRuntime()
    source = torch.zeros((256, 2048), dtype=torch.bfloat16)
    reduction = copy_engine_reduction(source, source.clone(), runtime=runtime,
                                      session=object(), coordinator=object(), name="r")
    assert isinstance(reduction, PreparedDirectRingAllReduce)
