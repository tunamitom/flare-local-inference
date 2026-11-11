# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""MoonViT checkpoint partitions and caller-owned storage, without CUDA execution."""

from dataclasses import dataclass
import json
from types import SimpleNamespace

import pytest
import torch

from ignition.components.parallel import TensorParallelShard
from ignition.components.weights import DirectWeightLoader
from ignition.preparation.scratch import GraphBufferPool
from ignition.recipes.kimi3.vision import (
    HEAD_SIZE, HIDDEN, INTERMEDIATE, PATCH_CAPACITIES, VISION_HEADS, PreparedVision, ProjectionGatherScratch,
    VisionInputs, VisionScratch, VisionWeights, head_range, projection_partitions,
)


@pytest.mark.parametrize('world', (1, 2, 4, 8, 16))
def test_head_partitions_cover_without_padding_or_replication(world):
    partitions = [head_range(TensorParallelShard(rank, world)) for rank in range(world)]
    assert [head for first, last in partitions for head in range(first, last)] == list(range(VISION_HEADS))
    counts = [last - first for first, last in partitions]
    assert max(counts) - min(counts) <= 1
    if world == 8:
        assert counts == [2] * 4 + [1] * 4
    if world == 16:
        assert counts == [1] * 12 + [0] * 4


@pytest.mark.parametrize('world', (1, 8, 16))
def test_fused_qkv_rows_preserve_head_order(world):
    weight = torch.arange(3 * VISION_HEADS * HEAD_SIZE)[:, None]
    pieces = []
    for rank in range(world):
        shard = TensorParallelShard(rank, world)
        ranges, columns = projection_partitions('vision_tower.encoder.blocks.0.wqkv.weight', shard)
        assert columns is None
        piece = torch.cat([weight[first:last] for first, last in ranges])
        first, last = head_range(shard)
        assert piece.shape == (3 * (last - first) * HEAD_SIZE, 1)
        pieces.append(piece.view(3, last - first, HEAD_SIZE, 1))
    torch.testing.assert_close(torch.cat(pieces, dim=1).reshape_as(weight), weight)


@pytest.mark.parametrize('name,rows,columns', (
    ('vision_tower.encoder.blocks.0.wo.weight', 1024, 1536),
    ('vision_tower.encoder.blocks.0.mlp.fc0.weight', 4096, 1024),
    ('vision_tower.encoder.blocks.0.mlp.fc1.weight', 1024, 4096),
    ('mm_projector.proj.0.weight', 4096, 4096),
    ('mm_projector.proj.2.weight', 7168, 4096),
))
@pytest.mark.parametrize('world', (8, 16))
def test_partitioned_linear_matches_unpartitioned_arithmetic(name, rows, columns, world):
    # Dyadic entries make these FP32 sums exact, isolating checkpoint slicing from GEMM rounding.
    source = ((torch.arange(2 * columns).view(2, columns) % 5) - 2).float() / 8
    weight = (((torch.arange(rows) % 11) - 5).float()[:, None]
              * ((torch.arange(columns) % 7) - 3).float()[None, :]) / 8
    expected = torch.nn.functional.linear(source, weight)
    partials = []
    row_partitioned = None
    for rank in range(world):
        ranges, selected_columns = projection_partitions(name, TensorParallelShard(rank, world))
        assert len(ranges) == 1
        selected_rows = ranges[0]
        row_partitioned = selected_rows is not None
        local_weight = weight[slice(None) if selected_rows is None else slice(*selected_rows),
                              slice(None) if selected_columns is None else slice(*selected_columns)]
        local_source = source if selected_columns is None else source[:, slice(*selected_columns)]
        partials.append(torch.nn.functional.linear(local_source, local_weight))
    actual = torch.cat(partials, dim=1) if row_partitioned else torch.stack(partials).sum(0)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize('world', (1, 8, 16))
def test_scratch_layout_and_pool_offsets_agree_on_all_ranks(world):
    pool = GraphBufferPool('cpu', chunk_bytes=1 << 20)
    allocated = None
    after_pointer = None
    for rank in range(world):
        shard = TensorParallelShard(rank, world)
        cursor = pool.cursor()
        scratch = VisionScratch.allocate(256, 'cpu', shard=shard, buffers=cursor)
        trailing = cursor.zeros((256, HIDDEN // world), torch.bfloat16)
        if allocated is None:
            allocated, after_pointer = pool.nbytes, trailing.data_ptr()
        assert pool.nbytes == allocated
        assert trailing.data_ptr() == after_pointer
        first, last = head_range(shard)
        heads = last - first
        assert scratch.query.shape == (256, heads, HEAD_SIZE)
        assert scratch.qkv.shape == (256, 3 * heads * HEAD_SIZE)
        assert scratch.attended.shape == (256, heads * HEAD_SIZE)
        assert scratch.mlp_up.shape == (256, INTERMEDIATE // world)
        assert scratch.activated.shape == (64, INTERMEDIATE // world)
        assert scratch.output.shape == (64, HIDDEN)
        assert all(value.is_contiguous() for value in scratch.__dict__.values())
        for stage in ((scratch.query, scratch.key, scratch.value, scratch.qkv, scratch.projected),
                      (scratch.query, scratch.key, scratch.value, scratch.attended, scratch.projected),
                      (scratch.mlp_up, scratch.mlp_down), (scratch.merged, scratch.activated),
                      (scratch.activated, scratch.output)):
            intervals = sorted((value.data_ptr(), value.data_ptr() + value.nbytes) for value in stage if value.numel())
            assert all(left[1] <= right[0] for left, right in zip(intervals, intervals[1:]))
        smaller = scratch.rows(64)
        assert smaller.query.shape == (64, heads, HEAD_SIZE)
        assert smaller.activated.shape == (16, INTERMEDIATE // world)


def test_sharded_workspace_is_bounded_by_merger_output():
    assert VisionScratch.workspace_elements(65536) == 65536 * 4608
    assert VisionScratch.workspace_elements(65536, TensorParallelShard(0, 8)) == 65536 * 1792
    assert VisionScratch.workspace_elements(65536, TensorParallelShard(7, 8)) == 65536 * 1792


def test_patch_capacities_bound_vision_input_and_scratch_storage():
    # Cap 65,536 -> 16,384 patches: the dropped tier freed 320,274,432 B per rank:
    # pixels 115,605,504 + projected 100,663,296 + attention_storage 100,663,296 +
    # interpolation/merge indices 3,342,336. Images above 16,384 patches (about
    # 1,790 x 1,790 px) exceed every admitted tier. The pixel buffer stays float32:
    # the frontend stages f32 bytes (helios/src/frontend/images/packing.rs
    # tensor_layout and spatial_merge).
    assert PATCH_CAPACITIES == (256, 1024, 4096, 16384)
    patches = PATCH_CAPACITIES[-1]
    inputs = VisionInputs.allocate('cpu')
    assert inputs.vision_pixel_values.shape == (patches, 588)
    assert inputs.vision_pixel_values.dtype == torch.float32
    scratch = VisionScratch.allocate(patches, 'cpu', shard=TensorParallelShard(0, 8))
    assert scratch.projected.shape == (patches, 1024) and scratch.projected.dtype == torch.bfloat16
    # TP8 attention storage slab is max(3 * patches * 256, patches * 1024) bf16 elements.
    assert scratch.normalized.shape == (patches, 1024) and scratch.normalized.dtype == torch.bfloat16
    dropped = 65536 - patches
    assert dropped * (588 * 4 + 1024 * 2 + 1024 * 2) + dropped * 68 == 320_274_432


def test_sharded_encoder_rejects_missing_gather_before_preparation():
    shard = TensorParallelShard(0, 8)
    scratch = VisionScratch.allocate(4, 'cpu', shard=shard)
    with pytest.raises(ValueError, match='caller-owned prepared all-gather'):
        PreparedVision(VisionWeights({}, shard), scratch, None, session=None, workspace=None, name='vision')
    with pytest.raises(ValueError, match='scratch does not match'):
        PreparedVision(VisionWeights({}), scratch, None, session=None, workspace=None, name='vision')


def _cpu_projection(source, weight, output, **kwargs):
    def run(value):
        output.copy_(torch.nn.functional.linear(value.float(), weight.float()).bfloat16())
        return output
    return run


def _gather_fixture(full_source, full_weight, shard, chunk_rows, *, heads, calls):
    def factory(source, output, *, name):
        index = int(name.rsplit('.', 1)[1])
        begin, end = index * chunk_rows, index * chunk_rows + source.shape[0]

        def gather(value):
            calls.append(name)
            complete = full_source[begin:end]
            if '.input.' in name:
                if heads:
                    capacity = -(-VISION_HEADS // shard.world_size) * HEAD_SIZE
                    first, last = head_range(shard)
                    width = (last - first) * HEAD_SIZE
                    torch.testing.assert_close(value[:, :width], complete[:, first * HEAD_SIZE:last * HEAD_SIZE],
                                               atol=0, rtol=0)
                    assert torch.count_nonzero(value[:, width:]) == 0
                    for rank in range(shard.world_size):
                        first, last = head_range(TensorParallelShard(rank, shard.world_size))
                        offset = rank * capacity
                        output[:, offset:offset + capacity].zero_()
                        output[:, offset:offset + (last - first) * HEAD_SIZE].copy_(
                            complete[:, first * HEAD_SIZE:last * HEAD_SIZE])
                else:
                    first, last = shard.partition(full_source.shape[1])
                    torch.testing.assert_close(value, complete[:, first:last], atol=0, rtol=0)
                    output.copy_(complete)
            else:
                expected = torch.nn.functional.linear(complete.float(), full_weight.float()).bfloat16()
                first, last = shard.partition(expected.shape[1])
                torch.testing.assert_close(value, expected[:, first:last], atol=0, rtol=0)
                output.copy_(expected)
            return output
        return gather
    return factory


@pytest.mark.parametrize('rank,world,heads', ((0, 8, True), (7, 8, True), (15, 16, True), (3, 8, False)))
def test_full_k_projection_gathers_rows_without_partial_sum_rounding(monkeypatch, rank, world, heads):
    from ignition.recipes.kimi3 import vision

    shard = TensorParallelShard(rank, world)
    rows, chunk = 9, 4
    width = VISION_HEADS * HEAD_SIZE if heads else 64
    source = ((torch.arange(rows * width).view(rows, width) % 5 - 2).float() / 8).bfloat16()
    weight = (((torch.arange(64) % 11 - 5).float()[:, None]
              * (torch.arange(width) % 7 - 3).float()[None, :]) / 8).bfloat16()
    first, last = head_range(shard) if heads else shard.partition(width)
    first, last = (first * HEAD_SIZE, last * HEAD_SIZE) if heads else (first, last)
    local = source[:, first:last].contiguous()
    out_first, out_last = shard.partition(64)
    output = torch.empty((rows, 64), dtype=torch.bfloat16)
    scratch = ProjectionGatherScratch.allocate(chunk, shard, 'cpu')
    calls = []
    monkeypatch.setattr(vision, 'prepare_projection', _cpu_projection)
    operation = vision._GatherProjection(weight[out_first:out_last], local, output, shard=shard,
        scratch=scratch, gather_factory=_gather_fixture(source, weight, shard, chunk, heads=heads, calls=calls),
        session=None, workspace=None, name='projection', heads=heads)
    assert [(part[0], part[1]) for part in operation.parts] == [(0, 4), (4, 8), (8, 9)]
    for scale in (1, -2, .5):
        source.mul_(scale)
        local.copy_(source[:, first:last])
        for field in ('input', 'head_send', 'head_received', 'output'):
            getattr(scratch, field).fill_(float('nan'))
        output.fill_(float('nan'))
        expected = torch.nn.functional.linear(source.float(), weight.float()).bfloat16()
        torch.testing.assert_close(operation(local), expected, atol=0, rtol=0)
    assert len(calls) == 18


def test_empty_head_rank_computes_output_channels_from_peer_heads(monkeypatch):
    from ignition.recipes.kimi3 import vision

    shard = TensorParallelShard(15, 16)
    scratch = VisionScratch.allocate(4, 'cpu', shard=shard)
    gathered = ProjectionGatherScratch.allocate(4, shard, 'cpu')
    source = torch.ones((4, 1536), dtype=torch.bfloat16)
    weight = torch.full((1024, 1536), 1 / 64, dtype=torch.bfloat16)
    first, last = shard.partition(1024)
    calls = []
    monkeypatch.setattr(vision, 'prepare_projection', _cpu_projection)
    attention = vision._Attention({'wo.weight': weight[first:last]}, scratch, None, None,
        session=None, workspace=None, name='vision.0', shard=shard, gather_scratch=gathered,
        gather_factory=_gather_fixture(source, weight, shard, 4, heads=True, calls=calls))
    output = attention(scratch.hidden, cu_seqlens=None, position_embeddings=None)
    assert calls == ['vision.0.output.input.0', 'vision.0.output.output.0']
    torch.testing.assert_close(output, torch.full_like(output, 24), atol=0, rtol=0)


def test_empty_head_rank_keeps_attention_scratch_binding_order(monkeypatch):
    from ignition.preparation.scratch import ScratchArena, _shared_views
    from ignition.recipes.kimi3.vision import _Attention

    arena = ScratchArena('cpu', overlap=False)
    shard = TensorParallelShard(15, 16)
    scratch = VisionScratch.allocate(4, 'cpu', shard=shard)
    gathered = ProjectionGatherScratch.allocate(4, shard, 'cpu')
    monkeypatch.setattr('ignition.recipes.kimi3.vision.prepare_projection', _cpu_projection)
    arena.register_finalizer(lambda: arena.views(((32,), torch.uint8)))
    for layer in range(3):
        _Attention({'wo.weight': torch.zeros((64, 1536), dtype=torch.bfloat16)}, scratch, None, None,
            session=None, workspace=arena, shard=shard, gather_scratch=gathered,
            name=f'vision.{layer}', gather_factory=lambda source, output, **kwargs: lambda value: value)
    arena.register_finalizer(lambda: arena.views(((64,), torch.uint8)))
    arena._recorded_views = []
    for callback in arena._finalizers:
        callback()
    empty_layout = arena._recorded_views
    arena._recorded_views = None
    assert empty_layout == [(((size,), torch.uint8),) for size in (32, 0, 0, 0, 64)]
    populated_layout = [(((size,), torch.uint8),) for size in (32, 256, 512, 256, 64)]
    shared = _shared_views([populated_layout, empty_layout])
    assert shared == populated_layout
    arena._binding_views = iter(shared)
    for callback in arena._finalizers:
        callback()
    assert next(arena._binding_views, None) is None
    arena._binding_views = None


@dataclass
class _Packed:
    shape: tuple


class _CPUTransfer:
    def __call__(self, destination, source):
        destination.copy_(source)
        return True

    def flush(self):
        pass


class _CPUWeightLoader(DirectWeightLoader):
    """Replace only checkpoint transport; retain real tensor slicing and quantization APIs."""

    def __enter__(self):
        self._sources = self.manifest.tensors
        self._session = _CPUTransfer()
        return self

    def __exit__(self, *args):
        self._session = None


def _manifest(tmp_path, tensors):
    (tmp_path / 'config.json').write_text(json.dumps({'quantization_config': {'quant_method': 'exl3'}}))
    return SimpleNamespace(tensors=tensors, metadata_path=tmp_path, snapshot=tmp_path)


@pytest.mark.parametrize('rank,world', ((0, 1), (0, 8), (7, 8), (15, 16)))
@pytest.mark.parametrize('online_mxfp8', (False, True))
def test_loader_shards_before_quantizing_and_retains_unquantized_parameters(monkeypatch, tmp_path, rank, world, online_mxfp8):
    from ignition.recipes.kimi3 import vision

    prefix = 'vision_tower.encoder.blocks.0.'
    shapes = {
        prefix + 'wqkv.weight': (4608, 1024),
        prefix + 'wo.weight': (1024, 1536),
        prefix + 'mlp.fc0.weight': (4096, 1024),
        prefix + 'mlp.fc1.weight': (1024, 4096),
        prefix + 'norm0.weight': (1024,),
        prefix + 'norm1.weight': (1024,),
        'mm_projector.proj.0.weight': (4096, 4096),
        'mm_projector.proj.2.weight': (HIDDEN, 4096),
        'mm_projector.post_norm.weight': (HIDDEN,),
        'vision_tower.patch_embed.proj.weight': (1024, 3, 14, 14),
        'vision_tower.patch_embed.pos_emb.weight': (64, 64, 1024),
    }
    source = {name: (torch.arange(shape[0]) % 251).bfloat16().view(shape[0], *((1,) * (len(shape) - 1))).expand(shape)
              for name, shape in shapes.items()}
    converted = []

    def quantize(weight, *, retain_scale_rows=True):
        assert weight.ndim == 2 and weight.shape[1] % 128 == 0 and weight.numel() > 0
        assert retain_scale_rows is False
        converted.append(tuple(weight.shape))
        return _Packed(tuple(weight.shape))

    monkeypatch.setattr(vision, 'DirectWeightLoader', _CPUWeightLoader)
    monkeypatch.setattr(vision, 'quantize_bf16_linear_as_mxfp8', quantize)
    shard = TensorParallelShard(rank, world)
    manifest = _manifest(tmp_path, source)
    weights = VisionWeights.load(manifest, 'cpu', shard=shard, online_mxfp8=online_mxfp8)
    assert weights.shard == shard
    heads = head_range(shard)[1] - head_range(shard)[0]
    assert len(converted) == ((6 if heads else 5) if online_mxfp8 else 0)
    for name in shapes:
        weight = weights.tensors[name]
        if projection_partitions(name, shard) is None:
            torch.testing.assert_close(weight, source[name], atol=0, rtol=0)
        elif not heads and name.endswith('.wqkv.weight'):
            assert weight is None
        else:
            assert isinstance(weight, _Packed if online_mxfp8 else torch.Tensor)
            if not online_mxfp8:
                ranges, columns = projection_partitions(name, shard)
                expected = torch.cat([source[name][slice(None) if extent is None else slice(*extent),
                    slice(None) if columns is None else slice(*columns)] for extent in ranges])
                torch.testing.assert_close(weight, expected, atol=0, rtol=0)
    if heads:
        assert weights.tensors[prefix + 'wqkv.weight'].shape == (3 * heads * HEAD_SIZE, 1024)
    assert weights.tensors[prefix + 'wo.weight'].shape == (1024 // world, 1536)
    assert weights.tensors['mm_projector.proj.2.weight'].shape == (HIDDEN // world, INTERMEDIATE)


def test_exl3_metadata_reproduces_generic_linear_loader_rejection(tmp_path):
    prefix = 'vision_tower.encoder.blocks.0.wqkv'
    manifest = _manifest(tmp_path, {prefix + '.weight': torch.zeros((4608, 1024), dtype=torch.bfloat16)})
    with _CPUWeightLoader(manifest, 'cpu') as loader:
        with pytest.raises(ValueError, match='expected modelopt MIXED_PRECISION'):
            loader.load_linear_group((prefix,))


@pytest.mark.parametrize('rank,world', ((0, 8), (15, 16)))
@pytest.mark.parametrize('shape,dtype', (((4608, 1024), torch.float32), ((4608, 1024), torch.uint8),
                                       ((4608,), torch.bfloat16)))
def test_loader_rejects_non_bf16_projection_even_on_empty_head_rank(monkeypatch, tmp_path, rank, world, shape, dtype):
    from ignition.recipes.kimi3 import vision

    name = 'vision_tower.encoder.blocks.0.wqkv.weight'
    manifest = _manifest(tmp_path, {name: torch.zeros(1, dtype=dtype).expand(shape)})
    monkeypatch.setattr(vision, 'DirectWeightLoader', _CPUWeightLoader)
    with pytest.raises(ValueError, match='expected unquantized BF16 vision projection'):
        VisionWeights.load(manifest, 'cpu', shard=TensorParallelShard(rank, world))
