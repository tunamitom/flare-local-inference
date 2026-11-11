# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MoonViT image execution with learned positions, axial RoPE, and spatial merging."""

from dataclasses import dataclass, replace
from functools import partial
from types import SimpleNamespace

import torch
from torch import nn
from transformers.models.kimi_k25.modeling_kimi_k25 import Kimi_K25VisionEncoderLayer

from ignition.components.linear import QuantizedProjection, UnquantizedProjection, prepare_projection
from ignition.components.parallel import TensorParallelShard
from ignition.components.rotary import apply_cached_rotary, prepare_cached_rotary
from ignition.components.varlen_attention import PreparedVarlenAttention
from ignition.components.weights import DirectWeightLoader, quantize_bf16_linear_as_mxfp8
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession, declare, register_finalizer

# Cap the admitted tiers at 16,384 patches. VisionInputs keeps PATCH_CAPACITIES[-1] x 588
# f32 pixels plus VisionScratch's projected and attention slabs, so the dropped 65,536-slot
# tier frees 320,274,432 B per rank (pixels 115,605,504 + projected 100,663,296 + attention
# 100,663,296 + indices 3,342,336) at zero decode/prefill cost; the slabs are permanently
# reserved per rank on every deployment. Images above 16,384 patches (about 1,790 x 1,790 px
# before the processor's own resizing) exceed every admitted tier.
PATCH_CAPACITIES = (256, 1024, 4096, 16384)
FEATURES = 16384
HIDDEN = 7168
VISION_HEADS = 12
HEAD_SIZE = 128
INTERMEDIATE = 4096


def head_range(shard):
    """Contiguous, unpadded MoonViT heads; empty ranks still join feature exchanges."""
    width, extra = divmod(VISION_HEADS, shard.world_size)
    first = shard.rank * width + min(shard.rank, extra)
    return first, first + width + int(shard.rank < extra)


def projection_partitions(name, shard):
    """Checkpoint row groups and columns for one MoonViT or merger projection."""
    first, last = head_range(shard)
    channels = first * HEAD_SIZE, last * HEAD_SIZE
    intermediate = shard.partition(INTERMEDIATE, alignment=128)
    if name.endswith('.wqkv.weight'):
        return tuple((offset + channels[0], offset + channels[1])
                     for offset in (0, VISION_HEADS * HEAD_SIZE, 2 * VISION_HEADS * HEAD_SIZE)), None
    if name.endswith('.wo.weight'):
        return (shard.partition(1024, alignment=64),), None
    if name.endswith('.mlp.fc0.weight') or name == 'mm_projector.proj.0.weight':
        return (intermediate,), None
    if name.endswith('.mlp.fc1.weight'):
        return (shard.partition(1024, alignment=64),), None
    if name == 'mm_projector.proj.2.weight':
        return (shard.partition(HIDDEN, alignment=64),), None
    return None


@dataclass(frozen=True)
class VisionWeights:
    tensors: dict[str, object]
    shard: TensorParallelShard = TensorParallelShard(0, 1)

    @classmethod
    def load(cls, manifest, device, progress=None, *, shard=None, online_mxfp8=False):
        """Load head and MLP partitions before optional MXFP8 conversion.

        Conversion covers attention QKV/output, encoder MLPs and merger linears.
        Patch convolution, learned positions and normalization retain their stored precision.
        Omitting ``shard`` retains the replicated encoder layout.
        """
        shard = TensorParallelShard(0, 1) if shard is None else shard
        shard.partition(INTERMEDIATE, alignment=128)
        tensors = {}
        with DirectWeightLoader(manifest, device, prefixes=('vision_tower.', 'mm_projector.'), progress=progress) as loader:
            for name in sorted(loader._selected_names()):
                partitions = projection_partitions(name, shard)
                if partitions is None:
                    tensors[name] = loader.load_tensor(name)
                    continue
                rows, columns = partitions
                source = loader._sources[name]
                if source.ndim != 2 or source.dtype != torch.bfloat16:
                    raise ValueError(f"{name}: expected unquantized BF16 vision projection")
                if (rows[0] is not None and rows[0][0] == rows[0][1]) or (
                        columns is not None and columns[0] == columns[1]):
                    # Empty-head ranks still join feature exchanges but execute no local attention.
                    tensors[name] = None
                    continue
                # Routed-expert checkpoint metadata does not describe these BF16 vision matrices.
                pieces = tuple(loader.load_tensor(name, extent=(
                    slice(*(extent or (0, source.shape[0]))),
                    slice(*(columns or (0, source.shape[1]))))) for extent in rows)
                weight = pieces[0] if len(pieces) == 1 else torch.cat(pieces)
                del pieces
                if online_mxfp8:
                    weight = quantize_bf16_linear_as_mxfp8(weight, retain_scale_rows=False)
                tensors[name] = weight
        return cls(tensors, shard)


@dataclass(frozen=True)
class VisionInputs:
    vision_pixel_values: torch.Tensor
    vision_position_ids: torch.Tensor
    vision_interpolation_indices: torch.Tensor
    vision_interpolation_weights: torch.Tensor
    vision_merge_indices: torch.Tensor
    vision_cu_seqlens: torch.Tensor
    vision_num_patches: torch.Tensor
    vision_num_features: torch.Tensor
    vision_slot: torch.Tensor
    vision_generation: torch.Tensor
    vision_feature_offset: torch.Tensor

    @classmethod
    def allocate(cls, device):
        def zeros(shape, dtype=torch.int32):
            return torch.zeros(shape, dtype=dtype, device=device)
        n = PATCH_CAPACITIES[-1]
        return cls(zeros((n, 588), torch.float32), zeros((n, 2), torch.int64),
            zeros((n, 4), torch.int64), zeros((n, 4), torch.float32), zeros((n // 4, 4)),
            zeros(2), zeros(1), zeros(1), zeros(1) - 1, zeros(1, torch.int64), zeros(1))

    def buffers(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class VisionScratch:
    patches: torch.Tensor
    hidden: torch.Tensor
    qkv: torch.Tensor
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    attended: torch.Tensor
    projected: torch.Tensor
    mlp_up: torch.Tensor
    mlp_down: torch.Tensor
    merged: torch.Tensor
    activated: torch.Tensor
    output: torch.Tensor
    normalized: torch.Tensor

    @staticmethod
    def workspace_elements(patches, shard=None):
        shard = TensorParallelShard(0, 1) if shard is None else shard
        shard.partition(INTERMEDIATE, alignment=128)
        if type(patches) is not int or patches <= 0 or patches % 4:
            raise ValueError("vision patch capacity must be a positive multiple of four")
        heads = -(-VISION_HEADS // shard.world_size)
        return max(patches * width for width in (588, 3 * heads * HEAD_SIZE,
                   INTERMEDIATE // shard.world_size, INTERMEDIATE // 4, HIDDEN // 4))

    @classmethod
    def allocate(cls, patches, device, *, shard=None, work=None, buffers=None):
        shard = TensorParallelShard(0, 1) if shard is None else shard
        first, last = head_range(shard)
        heads = last - first
        intermediate = INTERMEDIATE // shard.world_size
        elements = cls.workspace_elements(patches, shard)
        def zeros(*shape):
            return (torch.zeros(shape, dtype=torch.bfloat16, device=device) if buffers is None
                    else buffers.zeros(shape, torch.bfloat16))
        work = zeros(elements) if work is None else work
        if (work.ndim != 1 or work.numel() < elements or not work.is_contiguous()
                or work.dtype != torch.bfloat16 or work.device != torch.device(device)):
            raise ValueError("vision workspace requires contiguous BF16 storage for its projection and merge capacities")
        projected = zeros(patches, 1024)
        # Reserve identical pool offsets on all ranks, including the smaller and empty head shards.
        attention_width = patches * -(-VISION_HEADS // shard.world_size) * HEAD_SIZE
        attention_storage = zeros(max(3 * attention_width, patches * 1024))
        attention = attention_storage[:3 * attention_width].view(3, attention_width)
        q, k, v = (plane[:patches * heads * HEAD_SIZE].view(patches, heads, HEAD_SIZE) for plane in attention)
        def view(rows, width):
            return work[:rows * width].view(rows, width)
        # Patch projection, attention output, MLP activation, and spatial merging execute serially.
        return cls(view(patches, 588), projected, view(patches, 3 * heads * HEAD_SIZE), q, k, v,
            view(patches, heads * HEAD_SIZE), projected, view(patches, intermediate), projected,
            view(patches // 4, INTERMEDIATE), projected.view(-1)[:patches // 4 * intermediate].view(patches // 4, intermediate),
            view(patches // 4, HIDDEN), attention_storage[:patches * 1024].view(patches, 1024))

    def rows(self, patches):
        return type(self)(*(getattr(self, name)[:patches // 4 if name in ('merged', 'activated', 'output') else patches]
                           for name in self.__dataclass_fields__))


def _projection(source, weight, output, *, session, workspace, name, retain_packed_weights=False,
                output_partition=None):
    if retain_packed_weights or output_partition is not None:
        return QuantizedProjection(source, weight, output, session=session, workspace=workspace,
            activation_mode='a16', retain_packed_weights=retain_packed_weights,
            output_partition=output_partition, name=name)
    return prepare_projection(source, weight, output, session=session, workspace=workspace,
                              activation_mode='a16', name=name)


class _Projection(nn.Module):
    def __init__(self, weight, source, output, *, session, workspace, name, retain_packed_weights=False):
        super().__init__()
        self.operation = _projection(source, weight, output, session=session, workspace=workspace,
            name=name, retain_packed_weights=retain_packed_weights)

    def forward(self, source):
        return self.operation(source)


@dataclass(frozen=True)
class ProjectionGatherScratch:
    """Bounded, serially reused storage, disjoint from live encoder inputs and outputs."""

    input: torch.Tensor
    head_send: torch.Tensor
    head_received: torch.Tensor
    output: torch.Tensor
    head_columns: torch.Tensor
    rows: int

    @classmethod
    def allocate(cls, rows, shard, device, *, buffers=None):
        if type(rows) is not int or not 0 < rows <= 1024:
            raise ValueError("vision gather scratch requires one through 1024 rows")
        heads = -(-VISION_HEADS // shard.world_size)
        def zeros(width):
            shape = (rows * width,)
            return (torch.zeros(shape, dtype=torch.bfloat16, device=device) if buffers is None
                    else buffers.zeros(shape, torch.bfloat16))
        columns = [rank * heads * HEAD_SIZE + column for rank in range(shard.world_size)
                   for column in range((head_range(TensorParallelShard(rank, shard.world_size))[1]
                                        - head_range(TensorParallelShard(rank, shard.world_size))[0]) * HEAD_SIZE)]
        return cls(zeros(INTERMEDIATE), zeros(heads * HEAD_SIZE),
                   zeros(heads * shard.world_size * HEAD_SIZE), zeros(HIDDEN // shard.world_size),
                   torch.tensor(columns, dtype=torch.int64, device=device), rows)


class _GatherProjection(nn.Module):
    """Gather full-K inputs, compute output-row shards, and gather their BF16 results.

    Projection partial sums never cross ranks. Transport padding applies only to
    uneven head partitions; checkpoint weights and the GEMM input remain unpadded.
    """

    def __init__(self, weight, source, output, *, shard, scratch, gather_factory,
                 session, workspace, name, heads=False):
        super().__init__()
        columns = weight.shape[1] if isinstance(weight, torch.Tensor) else weight.in_features
        local_output = weight.shape[0] if isinstance(weight, torch.Tensor) else weight.out_features
        if (source.ndim != 2 or output.shape != (source.shape[0], local_output * shard.world_size)
                or source.dtype != torch.bfloat16 or output.dtype != source.dtype
                or source.device != output.device or not source.is_contiguous() or not output.is_contiguous()
                or columns > INTERMEDIATE or local_output > HIDDEN // shard.world_size):
            raise ValueError("vision output-row projection requires matching contiguous BF16 rows")
        if source.numel() and (source.data_ptr() < output.data_ptr() + output.nbytes
                              and output.data_ptr() < source.data_ptr() + source.nbytes):
            raise ValueError("vision output-row projection requires disjoint input and output storage")
        self.heads, self.output, self.source_columns = heads, output, source.shape[1]
        self.head_columns = scratch.head_columns
        padding = -(-VISION_HEADS // shard.world_size) * HEAD_SIZE
        if (heads and (columns != VISION_HEADS * HEAD_SIZE or source.shape[1] > padding)
                or not heads and source.shape[1] * shard.world_size != columns):
            raise ValueError("vision input partitions do not cover the projection's full reduction dimension")
        self.parts = []
        projections = {}
        for index, begin in enumerate(range(0, source.shape[0], scratch.rows)):
            end = min(begin + scratch.rows, source.shape[0])
            rows = end - begin
            x = scratch.input[:rows * columns].view(rows, columns)
            y = scratch.output[:rows * local_output].view(rows, local_output)
            send = scratch.head_send[:rows * padding].view(rows, padding) if heads else None
            gathered = (scratch.head_received[:rows * padding * shard.world_size]
                        .view(rows, padding * shard.world_size)) if heads else x
            input_gather = gather_factory(send if heads else source[begin:end], gathered,
                                          name=f'{name}.input.{index}')
            if rows not in projections:
                packed = not isinstance(weight, torch.Tensor)
                # Small expanded GEMMs change accumulation when N is partitioned.
                # Preserve the full reference geometry only when M is unchunked;
                # stored weights and the input/output exchanges stay partitioned.
                full_extent = packed and source.shape[0] <= scratch.rows and rows >= 128
                partition = ((output.shape[1], shard.rank * local_output,
                              (shard.rank + 1) * local_output) if full_extent else None)
                projections[rows] = _projection(x, weight, y, session=session, workspace=workspace,
                    name=f'{name}.rows{rows}', retain_packed_weights=packed and not full_extent,
                    output_partition=partition)
            output_gather = gather_factory(y, output[begin:end], name=f'{name}.output.{index}')
            self.parts.append((begin, end, send, gathered, x, y, input_gather, projections[rows], output_gather))

    def forward(self, source):
        for begin, end, send, gathered, x, y, input_gather, projection, output_gather in self.parts:
            if self.heads:
                if self.source_columns < send.shape[1]:
                    send.zero_()
                if self.source_columns:
                    send[:, :self.source_columns].copy_(source[begin:end])
                input_gather(send)
                torch.index_select(gathered, 1, self.head_columns, out=x)
            else:
                input_gather(source[begin:end])
            projection(x)
            output_gather(y)
        return self.output


class _InplaceGELU(nn.Module):
    def forward(self, source):
        return torch.ops.aten.gelu.out(source, approximate="tanh", out=source)


class _RMSNorm(nn.Module):
    """Prepared normalization with caller storage and native Torch rounding."""

    def __init__(self, weight, source, output, eps=None, *, session, name):
        super().__init__()
        from ignition.components.rms_norm import PreparedTorchRMSNorm

        self.weight = nn.Parameter(weight, requires_grad=False)
        self.output, self.eps = output, eps
        self.operation = PreparedTorchRMSNorm(source, self.weight, output, eps=eps, session=session, name=name)

    def forward(self, source):
        return self.operation(source)


class _Attention(nn.Module):
    def __init__(self, weights, scratch, cu, positions, *, session, workspace, name,
                 shard, gather_scratch, gather_factory):
        super().__init__()
        s = scratch
        self.scratch = scratch
        self.positions = positions
        self.heads = s.query.shape[1]
        self.attention_name = name + '.attention'
        if self.heads:
            self.qkv = _Projection(weights['wqkv.weight'], s.hidden, s.qkv,
                session=session, workspace=workspace, name=name + '.qkv',
                retain_packed_weights=shard.world_size > 1 and not isinstance(weights['wqkv.weight'], torch.Tensor))
        if shard.world_size == 1:
            self.output = _Projection(weights['wo.weight'], s.attended, s.projected,
                session=session, workspace=workspace, name=name + '.output')
        else:
            self.output = _GatherProjection(weights['wo.weight'], s.attended, s.projected,
                shard=shard, scratch=gather_scratch, gather_factory=gather_factory,
                session=session, workspace=workspace, name=name + '.output', heads=True)
        if self.heads:
            self.attend = PreparedVarlenAttention(s.query, s.key, s.value, cu, s.attended.view_as(s.query),
                session=session, workspace=workspace, name=name + '.attention')
        elif isinstance(workspace, ScratchArena):
            # Varlen attention binds one byte-buffer view per layer on head-owning ranks.
            # Empty ranks reserve the corresponding view without launching attention.
            register_finalizer(session, lambda: workspace.views(((0,), torch.uint8)), workspace=workspace)

    def forward(self, source, *, cu_seqlens, position_embeddings):
        s = self.scratch
        if self.heads:
            q, k, v = self.qkv(source).view(-1, 3, self.heads, HEAD_SIZE).unbind(1)
            apply_cached_rotary(q, k, position_embeddings, self.positions, s.query, s.key, True)
            s.value.copy_(v)
            self.attend()
        return self.output(s.attended)


def _declare_reference_attention(rows, device, *, session, workspace, name):
    """Tune the unpartitioned head geometry without retaining its Q/K/V storage."""
    from torch._subclasses.fake_tensor import FakeTensorMode
    from b12x.attention import varlen
    from b12x.preparation import PreparedCall
    from ignition.preparation.fixtures import varlen_attention as fixtures

    # Declarations need only metadata. Private benchmark operands belong to
    # one live tuning race and are released before serving capture.
    with FakeTensorMode():
        q, k, v, out = (torch.empty((rows, VISION_HEADS, HEAD_SIZE),
            dtype=torch.bfloat16, device=device) for _ in range(4))
        cu = torch.empty((2,), dtype=torch.int32, device=device)
        plan = varlen.plan(q, k, v, cu, max_seqlen_q=rows, max_seqlen_k=rows, causal=False)
    arguments = dict(q=q, k=k, v=v, out=out, cu_seqlens_q=cu,
        max_seqlen_q=rows, max_seqlen_k=rows, softmax_scale=HEAD_SIZE**-.5, causal=False)
    inputs = fixtures.input_factory(arguments, storage=session.benchmark_storage, session=session)
    call = partial(fixtures.benchmark_call, arguments=arguments, inputs=inputs, workspace=workspace)

    def retain_selection(state):
        # This declaration supplies a numerical schedule, not a serving call.
        # Only the tuning races need Q/K/V operands; retain no reference binding.
        return PreparedCall(run=lambda: None)

    declare(session, (plan.request(name=name, prepare_call=retain_selection, benchmark_call=call),))


def resolve_reference_attention(session, requests, finalists, *, encoders, coordinator=None):
    """Make local head partitions inherit the unpartitioned numerical schedule.

    BF16 tile-probability rounding depends on the attention reduction tile.
    Independently tuning one, two, and twelve heads can change image features.
    The unpartitioned shape is tuned normally; its legal tile selection is
    transferred to local heads through the existing contextual preparation phase.
    """
    from b12x.attention import varlen
    from b12x.preparation import FrozenMapping

    del finalists
    choices, obligations, replacements, schedules = {}, [], {}, []
    references = tuple(encoder.reference_attention_name for encoder in encoders
                       if encoder.reference_attention_name is not None)
    if coordinator is not None and not coordinator.same_on_every_rank(references):
        raise ValueError('vision ranks declared different unpartitioned attention schedules')
    for encoder in encoders:
        name = encoder.reference_attention_name
        if name is None:
            continue
        reference = requests[name].plan
        config = reference.selection.config
        payload = reference.contract.config_payload(config).to_dict()
        source_rank = None
        if coordinator is not None:
            import torch.distributed as dist

            # Head partitions need one unpartitioned schedule even when the
            # enclosing preparation session uses rank-local tuning.
            source_rank = coordinator.ranks[0]
            selected = [payload if dist.get_rank() == source_rank else None]
            dist.broadcast_object_list(selected, src=source_rank, group=coordinator.group)
            payload = selected[0]
            config = reference.contract.decode_config(FrozenMapping(payload))
        schedules.append(dict(reference=name, reference_rank=source_rank, config=payload))
        for block in encoder.blocks:
            attention = getattr(block.attn, 'attend', None)
            if attention is None:
                continue
            request_name = block.attn.attention_name
            request = requests[request_name]
            original = request.plan
            key = id(original)
            if key not in replacements:
                alternative = varlen.plan(attention.q, attention.k, attention.v, attention.cu,
                    max_seqlen_q=attention.max_seqlen, max_seqlen_k=attention.max_seqlen,
                    causal=False, override=config)
                if not original.same_declaration_except_override(alternative):
                    raise ValueError('head partition scheduling changed the attention operation contract')
                replacements[key] = alternative
                obligations.append(replace(request, name=request_name + '.unpartitioned_schedule',
                    plan=alternative, benchmark_call=None, finalist_policy=None))
            elif replacements[key].override != config:
                raise ValueError('one shared attention plan cannot inherit different reduction schedules')
            choices[request_name] = replacements[key]
    if obligations:
        result = session.prepare_context_plans(obligations)
        result.close()
    for encoder in encoders:
        encoder.attention_schedule_ready = True
        encoder.reference_attention_name = None
    return choices, dict(objective='Preserve unpartitioned attention arithmetic across head shards',
                         schedules=schedules, partitioned_operators=len(choices))


class PreparedVision:
    """MoonViT with output-row projections and bounded, exact BF16 exchanges.

    ``gather_factory(source, output, name=...)`` prepares a column all-gather
    into caller-owned output. Every runtime must hold at least ``chunk_rows``
    rows; serial projections reuse bounded input and local-output staging.
    """

    def __init__(self, weights, scratch, inputs, *, session, workspace, name, buffers=None,
                 gather_factory=None, chunk_rows=1024):
        self.scratch, self.inputs = scratch, inputs
        s, tensors = scratch, weights.tensors
        # Output projections remain present even on ranks with no local heads.
        packed_partition = (weights.shard.world_size > 1
                            and not isinstance(tensors.get('vision_tower.encoder.blocks.0.wo.weight'), torch.Tensor))
        first, last = head_range(weights.shard)
        if (s.query.shape[1] != last - first or s.mlp_up.shape[1] != INTERMEDIATE // weights.shard.world_size):
            raise ValueError("vision scratch does not match the checkpoint's head and intermediate partitions")
        if weights.shard.world_size > 1 and gather_factory is None:
            raise ValueError("sharded vision requires a caller-owned prepared all-gather factory")
        if type(chunk_rows) is not int or not 0 < chunk_rows <= 1024:
            raise ValueError("vision projection chunks require one through 1024 rows")
        self.compilation_variant = f'heads{last - first}-intermediate{s.mlp_up.shape[1]}'
        n, device = s.patches.shape[0], s.patches.device
        def zeros(shape, dtype=torch.int32):
            return (torch.zeros(shape, dtype=dtype, device=device) if buffers is None
                    else buffers.zeros(shape, dtype))
        self.positions = zeros((n, 2), torch.int64)
        self.interpolation_indices = zeros((n, 4), torch.int64)
        self.interpolation_weights = zeros((n, 4), torch.float32)
        self.merge_indices = zeros((n // 4, 4), torch.int64)
        self.cu = zeros((2,))
        self.num_patches, self.num_features = zeros((1,)), zeros((1,))
        gather_scratch = (None if weights.shard.world_size == 1 else ProjectionGatherScratch.allocate(
            min(n, chunk_rows), weights.shard, device, buffers=buffers))

        def output_projection(weight, source, output, *, name):
            if gather_scratch is None:
                return _Projection(weight, source, output, session=session, workspace=workspace, name=name)
            return _GatherProjection(weight, source, output, shard=weights.shard, scratch=gather_scratch,
                gather_factory=gather_factory, session=session, workspace=workspace, name=name)

        self.frequencies = 1. / (10000 ** (torch.arange(0, 128, 4, dtype=torch.float32, device=device) / 128))
        angles = torch.arange(512, dtype=torch.float32, device=device)[:, None] * self.frequencies
        self.rotary = torch.view_as_real(torch.polar(torch.ones_like(angles), angles))
        self.rotary_positions = torch.arange(n, dtype=torch.int32, device=device)
        prepare_cached_rotary(session)
        self.position_weight = tensors['vision_tower.patch_embed.pos_emb.weight'].reshape(-1, 1024)
        self.patch = UnquantizedProjection(s.patches, tensors['vision_tower.patch_embed.proj.weight'].flatten(1),
            s.hidden, session=session, name=name + '.patch')
        self.blocks = []
        config = SimpleNamespace(hidden_size=1024, num_attention_heads=12, intermediate_size=4096, hidden_act='gelu_pytorch_tanh')
        for layer in range(27):
            prefix = f'vision_tower.encoder.blocks.{layer}.'
            w = {key.removeprefix(prefix): value for key, value in tensors.items() if key.startswith(prefix)}
            with torch.device('meta'):
                block = Kimi_K25VisionEncoderLayer(config)
            block.norm1 = _RMSNorm(w['norm0.weight'], s.hidden, s.normalized,
                                  session=session, name=f'{name}.{layer}.norm1')
            block.norm2 = _RMSNorm(w['norm1.weight'], s.hidden, s.normalized,
                                  session=session, name=f'{name}.{layer}.norm2')
            block.attn = _Attention(w, s, self.cu, self.rotary_positions,
                session=session, workspace=workspace, name=f'{name}.{layer}', shard=weights.shard,
                gather_scratch=gather_scratch, gather_factory=gather_factory)
            block.mlp.fc1 = _Projection(w['mlp.fc0.weight'], s.hidden, s.mlp_up,
                session=session, workspace=workspace, name=f'{name}.{layer}.up',
                retain_packed_weights=packed_partition)
            # The activation is the only consumer of the up projection's output.
            block.mlp.act = _InplaceGELU()
            block.mlp.fc2 = output_projection(w['mlp.fc1.weight'], s.mlp_up, s.mlp_down,
                                              name=f'{name}.{layer}.down')
            block.eval()
            self.blocks.append(block)
        self.norm = _RMSNorm(tensors['vision_tower.encoder.final_layernorm.weight'], s.hidden, s.normalized,
                             session=session, name=name + '.norm')
        self.up = _Projection(tensors['mm_projector.proj.0.weight'], s.merged, s.activated,
                             session=session, workspace=workspace, name=name + '.merge_up',
                             retain_packed_weights=packed_partition)
        self.down = output_projection(tensors['mm_projector.proj.2.weight'], s.activated, s.output,
                                      name=name + '.merge_down')
        self.post_norm = _RMSNorm(tensors['mm_projector.post_norm.weight'], s.output, s.output, 1e-5,
                                  session=session, name=name + '.post_norm')
        self.reference_attention_name = None
        self.attention_schedule_ready = True
        if packed_partition:
            if not isinstance(session, RecordedPreparationSession):
                raise ValueError("sharded MXFP8 MoonViT requires recorded reference-attention preparation")
            self.reference_attention_name = name + '.unpartitioned_attention'
            _declare_reference_attention(n, device, session=session, workspace=workspace,
                                         name=self.reference_attention_name)
            self.attention_schedule_ready = False

    def rotary_embeddings(self):
        rotary = self.rotary[self.positions.flip((1,))].transpose(1, 2).flatten(1, 2)
        return torch.cat((rotary[..., 0], rotary[..., 1]), dim=-1)

    def __call__(self):
        if not self.attention_schedule_ready:
            raise RuntimeError('sharded MXFP8 vision requires resolve_reference_attention during session.tune')
        s = self.scratch
        positions = (self.position_weight[self.interpolation_indices].float()
                     * self.interpolation_weights[:, :, None]).sum(1).to(s.hidden.dtype)
        hidden = self.patch(s.patches) + positions
        embeddings = self.rotary_embeddings()
        for block in self.blocks:
            hidden = block(hidden, self.cu, position_embeddings=embeddings)
        hidden = self.norm(hidden)
        s.merged.copy_(hidden[self.merge_indices].flatten(1))
        return self.post_norm(self.down(torch.nn.functional.gelu(self.up(s.merged))))
