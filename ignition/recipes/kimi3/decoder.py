# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Kimi K3 attention residuals and latent MoE composed into complete forwards."""

from dataclasses import dataclass, field
import json
import os

import torch
import triton

from ignition.components.collectives import PreparedChunkedAllGather, PreparedChunkedAllReduce
from ignition.components.owner_packet_reduce import (
    PreparedOwnerPacketAllReduce,
    copy_engine_reduction,
    direct_aligned_extent,
)
from ignition.components.embedding import PreparedShardedEmbedding
from ignition.components.linear import UnquantizedProjection
from ignition.components.moe import PreparedRouter, PreparedRoutedExperts, RouterScratch
from ignition.components.recurrent import DeltaMetadata, _commit_convolution
from ignition.components.vocabulary import PreparedVocabularyProjection
from ignition.components.kda_speculation import PreparedKDAReplay
from ignition.preparation.tuning import register_finalizer

from .attention import PreparedMLA
from .kda import PreparedKDA
from .pointwise import rms_norm_columns, situ
from .projections import prepare_projection
from .residual import ADD, REPLACE, ResidualStream
from .routing import MAX_ROWS as ROUTE_ROWS, PreparedRouteGather

# L4 (upstream flare #16): the prefill latent gather's measured launch geometry,
# (threads, block_limit) of the DcpAllToAll.all_gather_heads channel: 512:32 was
# the fastest measured at TP8 for the 3072x3584 gather. A caller-chosen geometry
# rides the collective's plan; None would keep the channel default (256:16), and
# the export's recipe environment clamps the block dimension (see the probe
# record). The two-island transport is armed by the export's W22 switches.
LATENT_GATHER_LAUNCH = (512, 32)


def latent_gather_supported(world_size, latent):
    """The latent gather needs 64-column rank partitions (3584: world 2, 4, 7, 8).

    Prefill rows project their own latent columns from the full hidden row with
    the decode row-sharded ``latent_down_rows`` weights and all-gather them,
    replacing the latent down-projection's all-reduce. Each latent value is then
    one FP32-accumulated dot product over all 7168 inputs, rounded to BF16 once
    (the reduction rounds eight BF16 partials, sums in FP32 and rounds again).
    Decode, verification and draft graphs are unchanged."""
    return world_size > 1 and latent % (world_size * 64) == 0


# A concurrent shared-expert MLP forks onto the auxiliary stream after its input is written and joins before its
# output is added; the explicit mutated outputs order both against the routed branch through compilation.
@torch.library.custom_op("flare::kimi3_start_shared_expert", mutates_args=("outputs", "workspace"))
def _start_shared_expert(source: torch.Tensor, weights: list[torch.Tensor], scales: list[torch.Tensor | None],
                         outputs: list[torch.Tensor], workspace: torch.Tensor, plans: list[int], auxiliary_stream: int) -> None:
    from ignition.components.sm103_projection import bf16_out, packed_out_workspace

    def project(value, index, output):
        if scales[index] is None:
            bf16_out(value, weights[index], output, plans[index])
        else:
            packed_out_workspace(value, weights[index], scales[index], None, None, output,
                                 workspace, "none", None, plans[index], "any")

    main = torch.cuda.current_stream(source.device)
    auxiliary = torch.cuda.ExternalStream(auxiliary_stream, device=source.device)
    auxiliary.wait_stream(main)
    gate_up, intermediate, projected = outputs
    with torch.cuda.stream(auxiliary):
        project(source, 0, gate_up)
        situ(gate_up, intermediate)
        project(intermediate, 1, projected)


@_start_shared_expert.register_fake
def _fake_start_shared_expert(source, weights, scales, outputs, workspace, plans, auxiliary_stream) -> None:
    pass


@torch.library.custom_op("flare::kimi3_join_shared_expert", mutates_args=("outputs", "workspace"))
def _join_shared_expert(source: torch.Tensor, routed: torch.Tensor, outputs: list[torch.Tensor],
                        workspace: torch.Tensor, auxiliary_stream: int) -> None:
    main = torch.cuda.current_stream(source.device)
    main.wait_stream(torch.cuda.ExternalStream(auxiliary_stream, device=source.device))


@_join_shared_expert.register_fake
def _fake_join_shared_expert(source, routed, outputs, workspace, auxiliary_stream) -> None:
    pass


@dataclass(frozen=True)
class DecoderStorage:
    scratch: dict
    hidden_columns: torch.Tensor
    latent_begin: torch.Tensor
    allocator: object = None
    stage_allocators: dict = field(default_factory=dict)

    @classmethod
    def allocate(cls, config, shard, device, *, scratch=None, allocator=None):
        # Rank-specific offsets stay in tensors so every rank replays rank zero's compiled forward.
        return cls({} if scratch is None else scratch, torch.arange(*shard.partition(config.hidden), device=device),
            torch.tensor(shard.partition(config.text["routed_expert_hidden_size"])[:1], dtype=torch.int32, device=device),
            allocator)

    def scratch_tensors(self):
        return tuple(self.scratch.values())

    def stage_allocator(self, name):
        """Serial attention and MLP stages share storage beyond persistent decoder values."""
        if self.allocator is None:
            return None
        if name not in self.stage_allocators:
            self.stage_allocators[name] = self.allocator.fork()
        return self.stage_allocators[name]


@dataclass(frozen=True)
class Scratch:
    activations: torch.Tensor
    hidden: torch.Tensor
    local_logits: torch.Tensor
    logits: torch.Tensor

    @classmethod
    def allocate(cls, tokens, rows, config, world_size, device, *, activations=None, buffers=None):
        def zeros(shape):
            return (torch.zeros(shape, dtype=torch.bfloat16, device=device) if buffers is None
                    else buffers.zeros(shape, torch.bfloat16))
        return cls(zeros((tokens, config.hidden)) if activations is None else activations, zeros((rows, config.hidden)),
                   zeros((rows, config.vocabulary // world_size)), zeros((rows, config.vocabulary)))

    def rows(self, tokens, rows):
        return type(self)(self.activations[:tokens], self.hidden[:rows], self.local_logits[:rows], self.logits[:rows])


def retained_auxiliary_rows(weights, tokens, *, mode):
    """Bound accepted-prefill features where the projection preserves enclosing-row arithmetic."""
    from b12x.gemm.blockscaled._linear import MXFP8LinearWeight
    from ignition.components.sm103_projection import is_sm103
    from ignition.recipes.kimi31.dflash_config import DFlash2Config

    if (mode == "prefill" and isinstance(weights.config.draft, DFlash2Config)
            and weights.draft is not None and isinstance(weights.draft.projection, MXFP8LinearWeight)
            and not is_sm103(weights.embedding.device)):
        from ignition.recipes.kimi31.dflash_config import CONTEXT_ROW_ALIGNMENT
        return min(tokens, weights.config.draft.window + CONTEXT_ROW_ALIGNMENT)
    return tokens


class PreparedMLP:
    """A SiTU MLP; with an overlapping workspace, ``start`` and ``finish`` run it on the auxiliary stream."""

    def __init__(self, weights, source, *, session, name, buffers, workspace=None, allocator=None,
                 parallel_plans=()):
        rows, device = source.shape[0], source.device
        def buffer(field, width):
            if field not in buffers or buffers[field].shape[0] < rows:
                buffers[field] = (torch.zeros((rows, width), dtype=source.dtype, device=device) if allocator is None
                                  else allocator.zeros((rows, width), source.dtype))
            return buffers[field][:rows]
        width = (weights["gate_up"].shape[0] if isinstance(weights["gate_up"], torch.Tensor)
                 else weights["gate_up"].out_features)
        self.input = buffer("input", width // 2)
        self.gate_up = prepare_projection(source, weights["gate_up"], buffer("gate_up", width),
                                          session=session, name=name + ".gate_up", workspace=workspace)
        self.down = prepare_projection(self.input, weights["down"], buffer("output", source.shape[1]),
                                       session=session, name=name + ".down", workspace=workspace)
        self.concurrent = bool(parallel_plans) and workspace is not None and workspace.overlap
        if self.concurrent:
            self.arena, self.parallel_plans = workspace, tuple(parallel_plans)
            self.stream_handle = workspace.auxiliary_stream.cuda_stream
            self.weights, self.scales = [], []
            for projection in (self.gate_up, self.down):
                if isinstance(projection, UnquantizedProjection):
                    self.weights.append(projection.weight)
                    self.scales.append(None)
                else:
                    values, scales, _ = projection.parts
                    self.weights.append(values)
                    self.scales.append(scales.view(torch.uint8))
                for main in self.parallel_plans:
                    workspace.register_parallel(main, projection.plan)
            self.outputs = [self.gate_up.output, self.input, self.down.output]
            register_finalizer(session, self.refresh, workspace=workspace)

    def refresh(self):
        main_bytes = max(plan.memory_requirements().scratch_nbytes for plan in self.parallel_plans)
        side_bytes = max(p.plan.memory_requirements().scratch_nbytes for p in (self.gate_up, self.down))
        _, self.workspace = self.arena.views(((main_bytes,), torch.uint8), ((side_bytes,), torch.uint8))

    def __call__(self, source):
        situ(self.gate_up(source), self.input)
        return self.down(self.input)

    def start(self, source):
        _start_shared_expert(source, self.weights, self.scales, self.outputs, self.workspace,
                             [self.gate_up.plan.handle, self.down.plan.handle], self.stream_handle)

    def finish(self, source, routed):
        _join_shared_expert(source, routed, self.outputs, self.workspace, self.stream_handle)
        return self.down.output


class PreparedLatentMoE:
    def __init__(self, weights, source, config, shard, *, runtime, router_runtime, coordinator, session, name, workspace, buffers, storage,
                 route_runtime=None, allocator=None, accumulation_buffers=None, latent_gather_runtime=None,
                 copy_engine=False, reduction_staging=None):
        tokens, hidden = source.shape
        c, device = config.text, source.device
        latent = c["routed_expert_hidden_size"]

        def buffer(field, width, dtype=torch.bfloat16):
            # Decode-only fields first appear in a decoder of any row count, so each holds every route row.
            if field not in buffers or buffers[field].shape[0] < tokens:
                shape = (max(tokens, ROUTE_ROWS), width)
                buffers[field] = (torch.zeros(shape, dtype=dtype, device=device) if allocator is None
                                  else allocator.zeros(shape, dtype))
            return buffers[field][:tokens]

        self.source = buffer("source", hidden // shard.world_size)
        self.hidden_columns, self.latent_begin = storage.hidden_columns, storage.latent_begin
        self.latent = buffer("latent", latent)
        # The routed experts write where prefill reduces the latent down-projection from, and their sum
        # replaces the latent rows they consumed.
        self.expert_output = buffer("down", latent)
        self.latent_sum = self.latent
        self.up_input = buffer("up_input", latent // shard.world_size)
        self.output = buffer("output", hidden)
        self.up = prepare_projection(self.up_input, weights["latent_up"], self.output,
                                      session=session, name=name + ".latent_up", workspace=workspace)
        # Each rank normalizes the full latent row and keeps the columns its up-projection shard reads.
        self.latent_norm, self.eps = weights["latent_norm"], c["rms_norm_eps"]
        if copy_engine:
            # L9: the direct ring's zero-copy reduction (plus padded staging when unaligned).
            reduction = copy_engine_reduction
            reduction_args = dict(workspace=accumulation_buffers, staging=reduction_staging)
        else:
            reduction = PreparedChunkedAllReduce if accumulation_buffers is None else PreparedOwnerPacketAllReduce
            reduction_args = {} if accumulation_buffers is None else dict(workspace=accumulation_buffers)
        self.expert_reduce = reduction(self.expert_output, self.latent_sum, runtime=runtime,
            coordinator=coordinator, session=session, name=name + ".expert_reduce", **reduction_args)
        local_logits = buffer("local_router", weights["router"].shape[0], torch.float32)
        self.router_projection = UnquantizedProjection(source, weights["router"], local_logits, session=session, name=name + ".router")
        if "routes" not in buffers:
            buffers["routes"] = RouterScratch.allocate(tokens, device, experts=c["num_experts"], top_k=c["num_experts_per_token"])
        routes = buffers["routes"].rows(tokens)
        self.route_gather = self.down_gather = None
        if latent_gather_runtime is not None:
            # Prefill rows project their own latent columns from the full hidden row (the decode row-sharded
            # weights) and gather them: no reduction, one BF16 rounding per latent value.
            self.down = prepare_projection(source, weights["latent_down_rows"],
                buffer("prefill_local_down", latent // shard.world_size), session=session,
                name=name + ".latent_down_rows", workspace=workspace)
            self.down_gather = PreparedChunkedAllGather(self.down.output, self.latent, runtime=latent_gather_runtime,
                coordinator=coordinator, session=session, name=name + ".down_gather", launch=LATENT_GATHER_LAUNCH)
        elif route_runtime is not None and tokens <= ROUTE_ROWS:
            # Decode rows gather row-sharded latent and router shards and select experts in one exchange.
            self.down = prepare_projection(source, weights["latent_down_rows"],
                buffer("local_down", latent // shard.world_size), session=session, name=name + ".latent_down_rows",
                workspace=workspace)
            self.route_gather = PreparedRouteGather(self.down.output, local_logits, weights["correction"],
                self.latent, routes.weights, routes.ids, runtime=route_runtime, session=session,
                coordinator=coordinator, name=name + ".route_gather")
        else:
            self.down = prepare_projection(self.source, weights["latent_down"], buffer("down", latent),
                                            session=session, name=name + ".latent_down", workspace=workspace)
            self.down_reduce = reduction(self.down.output, self.latent, runtime=runtime,
                coordinator=coordinator, session=session, name=name + ".down_reduce", **reduction_args)
        if self.route_gather is None:
            logits = buffer("router", c["num_experts"], torch.float32)
            self.router_logits = logits
            local_experts = local_logits.shape[1]
            router_width = (local_experts + 31) // 32 * 32
            self.router_input = buffer("router_input", router_width, torch.float32)
            self.gathered_logits = buffer("gathered_router", router_width * shard.world_size, torch.float32)
            self.router_columns = self.gathered_logits.view(tokens, shard.world_size, router_width)[:, :, :local_experts]
            self.router_output = logits.view(tokens, shard.world_size, local_experts)
            # Gathering BF16 views transports the FP32 logits without conversion or arithmetic.
            self.router_gather = PreparedChunkedAllGather(self.router_input.view(torch.bfloat16), self.gathered_logits.view(torch.bfloat16),
                runtime=router_runtime, coordinator=coordinator, session=session, name=name + ".router_gather")
            self.router = PreparedRouter(logits, routes, score_func="sigmoid", correction_bias=weights["correction"],
                renormalize=True, routed_scaling_factor=c["routed_scaling_factor"], session=session, name=name + ".route")
        self.experts = PreparedRoutedExperts(self.latent, weights["experts"], routes.ids, routes.weights,
            self.expert_output, workspace=workspace, session=session, name=name + ".experts")
        # Decode and verification rows run the shared experts beside routing and the routed experts.
        self.shared = PreparedMLP(weights["shared"], source, session=session, name=name + ".shared",
                                  buffers=buffers.setdefault("shared", {}), allocator=allocator,
                                  workspace=workspace, parallel_plans=(self.router_projection.plan, self.down.plan,
                                      self.experts.plan, self.up.plan) if self.route_gather is not None else ())

    def phase_down_pack(self, source):
        """L9 dual-wave: the MoE's compute prologue (router projection, latent down projection)."""
        logits = self.router_projection(source)
        if self.route_gather is not None:
            self.down(source)
        elif self.down_gather is not None:
            self.down(source)
        else:
            self.router_input[:, :logits.shape[1]].copy_(logits)
            self.source.copy_(source.index_select(1, self.hidden_columns))
            self.down(self.source)

    def phase_route_ring(self):
        """L9 dual-wave: the MoE's transport step (route/latent gathers or the latent reduce)."""
        if self.route_gather is not None:
            self.route_gather()
        else:
            self.router_gather(self.router_input.view(torch.bfloat16))
            self.router_output.copy_(self.router_columns)
            if self.down_gather is not None:
                self.down_gather(self.down.output)
            else:
                self.down_reduce(self.down.output)

    def phase_router(self):
        """L9 dual-wave: expert selection on the gathered logits (after phase_route_ring)."""
        if getattr(self, "router", None) is not None:
            self.router(self.router_logits)

    def phase_experts(self):
        return self.experts()

    def phase_expert_reduce(self):
        """L9 dual-wave: the expert reduce over the ring and the latent norm (RS-norm stays unported)."""
        self.expert_reduce(self.expert_output)
        rms_norm_columns(self.latent_sum, self.latent_norm, self.up_input, self.eps, begin=self.latent_begin)

    def phase_up_combine(self, shared_out):
        self.up(self.up_input)
        self.output.add_(shared_out)
        return self.output

    def __call__(self, source):
        if self.shared.concurrent:
            self.shared.start(source)
        logits = self.router_projection(source)
        if self.route_gather is not None:
            self.down(source)
            self.route_gather()
        else:
            self.router_input[:, :logits.shape[1]].copy_(logits)
            self.router_gather(self.router_input.view(torch.bfloat16))
            self.router_output.copy_(self.router_columns)
            self.router(self.router_logits)
            if self.down_gather is not None:
                self.down_gather(self.down(source))
            else:
                self.source.copy_(source.index_select(1, self.hidden_columns))
                self.down_reduce(self.down(self.source))
        self.expert_reduce(self.experts())
        rms_norm_columns(self.latent_sum, self.latent_norm, self.up_input, self.eps, begin=self.latent_begin)
        self.up(self.up_input)
        shared = self.shared.finish(source, self.output) if self.shared.concurrent else self.shared(source)
        self.output.add_(shared)
        return self.output


def _scratch_spans(value, path, found):
    """Record (path, start, end) byte spans for every tensor reachable from decoder scratch."""
    if isinstance(value, torch.Tensor):
        if value.numel():
            found.append((path or "scratch", value.data_ptr(), value.data_ptr() + value.numel() * value.element_size()))
    elif isinstance(value, dict):
        for key, sub in value.items():
            _scratch_spans(sub, f"{path}.{key}" if path else str(key), found)
    elif isinstance(value, (list, tuple)):
        for index, sub in enumerate(value):
            _scratch_spans(sub, f"{path}[{index}]", found)


def dual_wave_prefill_enabled(*, setting, mode, direct_ring, workspace, tokens, request_count,
                              auxiliary_layers, auxiliary_source):
    """The L9 dual-wave gate (29aac9e/eb72cab/2d3635f): True when this prefill decoder
    splits its rows into two waves. Large even chunks, the direct ring and an
    overlapping workspace (the auxiliary compute stream) are required; drafts'
    auxiliary features must tap the attention input (the schedule preserves that
    tap order). Every other decoder keeps the serial path.

    F6 (l9-transplant-fable-review.md §6): the single-request term is an
    assertion, not a gate — prefill graphs always carry one live request per
    replay, and the dual wave engages at any stream count (concurrent prefill
    streams interleave whole graph replays; no two requests ever share one).
    """
    if (setting and mode == "prefill" and tokens >= 2048 and tokens % 512 == 0
            and request_count != 1):
        raise ValueError("a prefill graph carries one live request per replay; "
                         f"got {request_count} (the dual wave is not a per-request gate)")
    return bool(
        setting and mode == "prefill" and direct_ring
        and workspace is not None and workspace.overlap
        and tokens >= 2048 and tokens % 512 == 0
        and (not auxiliary_layers or auxiliary_source == "attention_input")
    )


def dual_wave_extra_bytes(buffers, dual_layers):
    """Bytes the dual-wave second prepared-module set adds beyond the serial decoders.

    Conservative accounting for the pool-cap knob: both waves' per-wave module
    buffers (kda/moe/dense dw dicts), the shared dual buffers and the attention
    wave outputs. Counted against the preparation memory budget.
    """
    def tensor_bytes(value, seen=None):
        seen = set() if seen is None else seen
        if id(value) in seen:
            return 0
        if isinstance(value, torch.Tensor):
            return value.numel() * value.element_size()
        if isinstance(value, dict):
            seen.add(id(value))
            return sum(tensor_bytes(item, seen) for item in value.values())
        if isinstance(value, (list, tuple)):
            seen.add(id(value))
            return sum(tensor_bytes(item, seen) for item in value)
        state = getattr(value, "__dict__", None)
        if state is not None:
            seen.add(id(value))
            return tensor_bytes(state, seen)
        return 0

    extra = sum(tensor_bytes(buffers[key]) for key in ("dual_mlp_in", "dual_total_w0", "dual_total_w1"))
    for dual_waves in ("kda_dw", "moe_dw", "dense_dw"):
        extra += tensor_bytes(buffers.get(dual_waves, {}))
    for kind in {entry[1] for entry in dual_layers}:
        stage = buffers.get(kind, {})
        extra += sum(tensor_bytes(stage.get(key)) for key in ("output_w0", "output_w1"))
    return extra


class SelectionOverlapCheck:
    """J-fault in-export gate (levers/j-fault-fable-review.md §3.1), at construction only.

    Before the layer loop it takes the four selection shapes from a fork of the
    shared cursor without writing them — the ranges the hoist's tables would have
    occupied before the fix — and afterwards it asserts the actually bound tables
    share no byte with any decoder-storage tensor and reports the would-be ranges
    against the buffers this decoder newly bound. The 20-row round graph must show
    shared-cursor overlap (the CPU-reproduced cause); its absence falsifies the
    diagnosis and stops the export before compile (restart with
    IGNITION_KIMI_SELECTION_HOIST=0).
    """

    def __init__(self, selection, buffers, allocator, metadata, source, name, mode):
        from .dsa import SELECTION_ROWS

        self.name, self.mode = name, mode
        self.rows = source.shape[0]
        self.buffers = buffers
        self.tables = ({label: getattr(selection, label)
                        for label in ("pages", "lengths", "active", "slots")}
                       if selection is not None else None)
        self.before = []
        _scratch_spans(buffers.scratch, "", self.before)
        self.would_be = {}
        if allocator is not None:
            width = metadata.token_page_table.shape[1]
            capacity = min(self.rows, SELECTION_ROWS)
            fork = allocator.fork()
            for label, shape, dtype in (("pages", (self.rows, 2 * width), torch.int32),
                                       ("lengths", (self.rows,), torch.int32),
                                       ("active", (-(-self.rows // capacity),), torch.int32),
                                       ("slots", (self.rows,), torch.int64)):
                taken = fork._take(shape, dtype)        # never written
                self.would_be[label] = [taken.data_ptr(), taken.data_ptr() + taken.numel() * taken.element_size()]

    def finish(self):
        after = []
        _scratch_spans(self.buffers.scratch, "", after)
        anchored = {start for _, start, _ in self.before}
        fresh = {path: [start, end] for path, start, end in after if start not in anchored}
        violations = []
        if self.tables:
            for label, tensor in self.tables.items():
                start, count = tensor.data_ptr(), tensor.numel() * tensor.element_size()
                for path, other_start, other_end in after:
                    if max(start, other_start) < min(start + count, other_end):
                        violations.append(f"{label} overlaps {path}")
        hits = []
        for label, (start, end) in self.would_be.items():
            for path, (other_start, other_end) in fresh.items():
                shared = max(0, min(end, other_end) - max(start, other_start))
                if shared:
                    hits.append({"table": label, "buffer": path, "bytes": shared})
        print("[overlap-check] " + json.dumps({"graph": self.name.split(".")[0], "decoder": self.name,
              "mode": self.mode, "rows": self.rows, "hoist": self.tables is not None,
              "assert": "clear" if not violations else violations,
              "would_be_ranges": self.would_be, "newly_bound": fresh, "would_be_overlaps": hits}),
              flush=True)
        if violations:
            raise RuntimeError(f"J-fault overlap check FAILED on {self.name}: the selection tables share memory "
                               f"with decoder storage: {violations}")
        if self.rows == 20 and self.tables is not None and not hits:
            raise RuntimeError(f"J-fault overlap check: the 20-row graph ({self.name}) reports NO overlap on the "
                               "shared-cursor side — the diagnosis in levers/j-fault-fable-review.md §1 is wrong. "
                               "Stop before compile and restart the export with IGNITION_KIMI_SELECTION_HOIST=0.")
        return hits


class PreparedDecoder:
    """Kimi K3's decoder over packed rows.

    ``verify`` rows are speculative candidates: KDA layers record them without advancing committed states
    until ``commit`` replays each request's accepted prefix. With a draft model, every forward also keeps
    DSpark's AttnRes mixtures entering its auxiliary layers in ``auxiliary``.
    """

    def __init__(self, weights, scratch, metadata, pool, *, mode, runtime, vocabulary_runtime, embedding_runtime, router_runtime, vision_state,
                 coordinator, session, name, workspace, buffers, kda_records=None, route_runtime=None, context=None,
                 latent_runtimes=(), scoring=None, mix_split_rows=0, score_exchange=True,
                 latent_gather_runtime=None, prefill_dual_wave=False, dual_wave_budget_bytes=None):
        # Verification records its candidate rows; other modes ignore records.
        if mode not in ("prefill", "decode", "verify") or mode == "verify" and kda_records is None:
            raise ValueError("Kimi K3 verification decodes recorded candidate rows")
        self.weights, self.scratch, self.metadata, self.pool = weights, scratch, metadata, pool
        self.vision_state = vision_state if mode == "prefill" else None
        config, source = weights.config, scratch.activations
        allocator = buffers.allocator
        accumulation_buffers = None
        reduction_staging = None
        from b12x.comm.pcie import DmaAllReduce

        direct_ring = isinstance(runtime, DmaAllReduce) and bool(getattr(runtime, "direct_mode", ""))
        if isinstance(runtime, DmaAllReduce) and getattr(runtime, "kimi_bf16_ring", False):
            # Fable dig 2026-11-10: plain copy-engine ring with BF16 hop sums (one launch per reduction,
            # no FP32 owner packets): 1.44 ms per 3072x7168 reduction and about half the graph nodes.
            if mode != "prefill":
                raise ValueError("Kimi copy-engine reductions are prepared only for prefill")
        elif direct_ring:
            # L9 (bcc93b3): the direct ring reduces BF16 activations without FP32 owner
            # packets; unaligned shapes pad through caller-owned BF16 staging.
            if mode != "prefill":
                raise ValueError("Kimi copy-engine reductions are prepared only for prefill")
            ring_elements = direct_aligned_extent(source.numel(), runtime.world_size) * runtime.world_size
            key = "prefill_direct_staging"
            if key not in buffers.scratch or buffers.scratch[key].numel() < ring_elements:
                buffers.scratch[key] = (torch.zeros(ring_elements, dtype=source.dtype, device=source.device)
                    if allocator is None else allocator.zeros((ring_elements,), source.dtype))
            reduction_staging = buffers.scratch[key]
        elif isinstance(runtime, DmaAllReduce):
            if mode != "prefill":
                raise ValueError("Kimi copy-engine reductions are prepared only for prefill")
            count = min(source.numel(), runtime.max_bytes // 4)
            key = "prefill_reduction_workspace"
            if key not in buffers.scratch or buffers.scratch[key].numel() < count:
                buffers.scratch[key] = (torch.zeros(count, dtype=torch.float32, device=source.device)
                    if allocator is None else allocator.zeros((count,), torch.float32))
            accumulation_buffers = buffers.scratch[key]
        if ("local_embedding" not in buffers.scratch
                or buffers.scratch["local_embedding"].shape[0] < source.shape[0]):
            shape = (source.shape[0], weights.embedding.shape[1])
            buffers.scratch["local_embedding"] = (torch.zeros(shape, dtype=source.dtype, device=source.device)
                if allocator is None else allocator.zeros(shape, source.dtype))
        self.local_embedding = buffers.scratch["local_embedding"][:source.shape[0]]
        self.embedding = PreparedShardedEmbedding(weights.embedding, metadata.token_ids, self.local_embedding, source,
            runtime=embedding_runtime, coordinator=coordinator,
            num_rows=metadata.live_counts[:1], session=session, name=name + ".embedding")
        self.delta = DeltaMetadata.allocate(metadata.starts.numel(), source.device,
            verification_columns=8 if mode == "verify" else 1)
        self.buffers, self.layers = buffers.scratch, []
        def zeros(shape):
            return (torch.zeros(shape, dtype=source.dtype, device=source.device) if allocator is None
                    else allocator.zeros(shape, source.dtype))

        # Residual blocks stay live across layers; reserve them before smaller layer intermediates.
        tokens, hidden = source.shape
        if "residual" not in self.buffers:
            committed = -(-len(weights.layers) // config.text["attn_res_block_size"])
            self.buffers["residual"] = dict(blocks=zeros((tokens, committed, hidden)), prefix=zeros(source.shape))
        self.auxiliary = None
        if "total" not in self.buffers:
            self.buffers["total"] = zeros(source.shape)
        self.auxiliary_layers = config.draft.auxiliary_layers if weights.draft is not None else ()
        self.auxiliary_count = None
        self.auxiliary_window, self.auxiliary_alignment = None, 1
        if self.auxiliary_layers:
            auxiliary_rows = retained_auxiliary_rows(weights, tokens, mode=mode)
            if auxiliary_rows < tokens:
                if metadata.lengths.numel() != 1 or config.draft.auxiliary_source != "attention_input":
                    raise ValueError("bounded Kimi draft features require a single accepted prefill and attention-input taps")
                self.auxiliary_count = metadata.lengths
                from ignition.recipes.kimi31.dflash_config import CONTEXT_ROW_ALIGNMENT
                self.auxiliary_window = config.draft.window
                self.auxiliary_alignment = CONTEXT_ROW_ALIGNMENT
            if "auxiliary" not in self.buffers:
                self.buffers["auxiliary"] = zeros((auxiliary_rows, len(self.auxiliary_layers) * hidden))
            self.auxiliary = self.buffers["auxiliary"][:auxiliary_rows]
        # One request-metadata derivation per replay, shared by every DSA layer.
        # J-fault fix (levers/j-fault-fable-review.md §2): the hoist's four tables
        # live on private torch.zeros storage — allocator=None (was
        # allocator=allocator), so no cursor of the shared decoder storage can
        # ever hand a layer buffer the same bytes. IGNITION_KIMI_SELECTION_HOIST=0
        # selects the per-layer path kept in dsa.py/attention.py (selection=None).
        # Read here, in the decoder constructor: never in a serialized settings
        # dataclass (that moves compile keys and re-races the dense set) and never
        # inside a @program_cache factory. Unset keeps the hoist armed.
        hoist = os.environ.get("IGNITION_KIMI_SELECTION_HOIST", "1") != "0"
        self.selection = None
        if mode != "prefill" and "dsa" in config.layers and hoist:
            from .dcp import ContextShards
            from .selection import PreparedSelectionMetadata

            shards = context.shards if context is not None else ContextShards()
            self.selection = PreparedSelectionMetadata(metadata, shards, rows=source.shape[0],
                device=source.device, allocator=None)
        # J-fault §3.1 stop-the-line gate, inside "Construct graph catalog": the
        # four table shapes are taken from a fork of the shared cursor before the
        # layer loop (the pre-fix placement, never written); finish() asserts and
        # reports after the loop.
        overlap = (SelectionOverlapCheck(self.selection, buffers, allocator, metadata, source, name, mode)
                   if mode != "prefill" else None)
        # L9 dual-wave gate (29aac9e/eb72cab/2d3635f): split a prefill chunk's rows into
        # two waves whose compute and ring transfers interleave. Requires the direct
        # ring, an overlapping workspace (the auxiliary compute stream), large
        # even chunks and a single live request. Every other graph keeps the serial
        # path. The gate drops g0san's stats_runtime and gather-ring requirements:
        # those belong to the unported RS-norm and ring column gathers.
        self.dual_wave = dual_wave_prefill_enabled(
            setting=prefill_dual_wave, mode=mode, direct_ring=direct_ring, workspace=workspace,
            tokens=tokens, request_count=metadata.lengths.numel(),
            auxiliary_layers=self.auxiliary_layers,
            auxiliary_source=(config.draft.auxiliary_source if config.draft is not None else None))
        if self.dual_wave:
            self.half_tokens = tokens // 2
            # Pad half-reductions to whole rank shards ((pad_r * hidden) % (world * 256) == 0)
            # so the direct ring reduces them with zero staging copies.
            pad_cand = -(-self.half_tokens // runtime.world_size) * runtime.world_size
            self.half_pad_rows = pad_cand if pad_cand * hidden * 2 <= runtime.max_bytes else self.half_tokens
            for key_dw, r_cap in (("dual_mlp_in", tokens), ("dual_total_w0", self.half_pad_rows),
                                  ("dual_total_w1", self.half_pad_rows)):
                if key_dw not in self.buffers or self.buffers[key_dw].shape[0] < r_cap:
                    self.buffers[key_dw] = zeros((r_cap, hidden))
            self.dual_mlp_in = self.buffers["dual_mlp_in"][:tokens]
            self.dual_total_w0 = self.buffers["dual_total_w0"][:self.half_pad_rows]
            self.dual_total_w1 = self.buffers["dual_total_w1"][:self.half_pad_rows]
            self.delta_a = DeltaMetadata.allocate(1, source.device)
            self.delta_b = DeltaMetadata.allocate(1, source.device)
            self.comm_priority_stream = torch.cuda.Stream(device=source.device, priority=-1)
            self.comm_stream_handle = self.comm_priority_stream.cuda_stream
            self.comp_stream_handle = workspace.auxiliary_stream.cuda_stream
            self.dual_layers = []
        for index, w in enumerate(weights.layers):
            prefix = f"{name}.layer{index}"
            kind = config.layers[index]
            kwargs = dict(mode=mode, session=session, name=prefix + ".attention", workspace=workspace,
                          buffers=self.buffers.setdefault(kind, {}), allocator=buffers.stage_allocator(kind))
            if self.dual_wave:
                half, pad_r = self.half_tokens, self.half_pad_rows
                attn_in = self.buffers["moe"]["output"][:tokens]
                mlp_in = self.dual_mlp_in
                tot_w0, tot_w1 = self.dual_total_w0, self.dual_total_w1
                reduce_args_dw = dict(runtime=runtime, staging=reduction_staging,
                                      session=session, coordinator=coordinator)
                if kind == "kda":
                    # 2d3635f: KDA prefill runs two complete half-chunk passes bound to
                    # half-chunk DeltaMetadata, preserving inactive-wave state slots.
                    kda_dw = self.buffers.setdefault("kda_dw", {})
                    for wk in ("w0", "w1"):
                        kb = kda_dw.setdefault(wk, {})
                        if "output" not in kb or kb["output"].shape[0] < pad_r:
                            kb["output"] = buffers.stage_allocator("kda").zeros((pad_r, hidden), torch.bfloat16)
                    kda_a = PreparedKDA(w["attention"], attn_in[:half], pool.recurrent(index), self.delta_a,
                        eps=config.text["rms_norm_eps"], mode=mode, session=session,
                        name=prefix + ".attention.w0", workspace=workspace, buffers=kda_dw["w0"],
                        allocator=buffers.stage_allocator("kda"))
                    kda_b = PreparedKDA(w["attention"], attn_in[half:], pool.recurrent(index), self.delta_b,
                        eps=config.text["rms_norm_eps"], mode=mode, session=session,
                        name=prefix + ".attention.w1", workspace=workspace, buffers=kda_dw["w1"],
                        allocator=buffers.stage_allocator("kda"))
                    attention = (kda_a, kda_b, kda_dw["w0"]["output"][:pad_r], kda_dw["w1"]["output"][:pad_r])
                else:
                    attention = PreparedMLA(w["attention"], attn_in, metadata, pool.attention(index),
                            context=context, index_cache=pool.index(index) if kind == "dsa" else None,
                            eps=config.text["rms_norm_eps"],
                            peers=context.peers.get(index) if context is not None else None,
                            index_peers=(context.index_peers.get(index)
                                         if context is not None and context.index_peers is not None else None),
                            selection=self.selection, score_exchange=score_exchange, coordinator=coordinator,
                            latents=next((r for r in latent_runtimes if source.shape[0] <= r.max_batch_size),
                                         latent_runtimes[-1] if latent_runtimes else None),
                            dual_wave=True, out_pad_rows=pad_r, **kwargs)
                out_p0, out_p1 = attention[2:] if kind == "kda" else attention.out_pads
                attn_red_a = copy_engine_reduction(out_p0, tot_w0, name=prefix + ".attn_reduce.w0", **reduce_args_dw)
                attn_red_b = copy_engine_reduction(out_p1, tot_w1, name=prefix + ".attn_reduce.w1", **reduce_args_dw)
                if "mlp" in w:
                    dense_dw = self.buffers.setdefault("dense_dw", {})
                    for wk in ("w0", "w1"):
                        wb = dense_dw.setdefault(wk, {})
                        if "output" not in wb or wb["output"].shape[0] < pad_r:
                            wb["output"] = buffers.stage_allocator("dense").zeros((pad_r, hidden), torch.bfloat16)
                    mlp_a = PreparedMLP(w["mlp"], mlp_in[:half], session=session, name=prefix + ".mlp.w0",
                                        buffers=dense_dw["w0"], allocator=buffers.stage_allocator("dense"),
                                        workspace=workspace)
                    mlp_b = PreparedMLP(w["mlp"], mlp_in[half:], session=session, name=prefix + ".mlp.w1",
                                        buffers=dense_dw["w1"], allocator=buffers.stage_allocator("dense"),
                                        workspace=workspace)
                    mlp_p0, mlp_p1 = dense_dw["w0"]["output"][:pad_r], dense_dw["w1"]["output"][:pad_r]
                    mlp_red_a = copy_engine_reduction(mlp_p0, tot_w0, name=prefix + ".mlp_reduce.w0", **reduce_args_dw)
                    mlp_red_b = copy_engine_reduction(mlp_p1, tot_w1, name=prefix + ".mlp_reduce.w1", **reduce_args_dw)
                    self.dual_layers.append((w, kind, attention, ("dense", mlp_a, mlp_b, mlp_p0, mlp_p1),
                                             attn_red_a, attn_red_b, mlp_red_a, mlp_red_b))
                else:
                    moe_dw = self.buffers.setdefault("moe_dw", {})
                    for wk in ("w0", "w1"):
                        wb = moe_dw.setdefault(wk, {})
                        if "output" not in wb or wb["output"].shape[0] < pad_r:
                            wb["output"] = buffers.stage_allocator("moe").zeros((pad_r, hidden), torch.bfloat16)
                    moe_a = PreparedLatentMoE(w, mlp_in[:half], config, weights.shard, runtime=runtime,
                        router_runtime=router_runtime, coordinator=coordinator, session=session,
                        name=prefix + ".moe.w0", workspace=workspace, buffers=moe_dw["w0"], storage=buffers,
                        allocator=buffers.stage_allocator("moe"), accumulation_buffers=accumulation_buffers,
                        reduction_staging=reduction_staging, copy_engine=direct_ring,
                        latent_gather_runtime=latent_gather_runtime if mode == "prefill" else None)
                    moe_b = PreparedLatentMoE(w, mlp_in[half:], config, weights.shard, runtime=runtime,
                        router_runtime=router_runtime, coordinator=coordinator, session=session,
                        name=prefix + ".moe.w1", workspace=workspace, buffers=moe_dw["w1"], storage=buffers,
                        allocator=buffers.stage_allocator("moe"), accumulation_buffers=accumulation_buffers,
                        reduction_staging=reduction_staging, copy_engine=direct_ring,
                        latent_gather_runtime=latent_gather_runtime if mode == "prefill" else None)
                    mlp_p0, mlp_p1 = moe_dw["w0"]["output"][:pad_r], moe_dw["w1"]["output"][:pad_r]
                    mlp_red_a = copy_engine_reduction(mlp_p0, tot_w0, name=prefix + ".mlp_reduce.w0", **reduce_args_dw)
                    mlp_red_b = copy_engine_reduction(mlp_p1, tot_w1, name=prefix + ".mlp_reduce.w1", **reduce_args_dw)
                    self.dual_layers.append((w, kind, attention, ("moe", moe_a, moe_b, mlp_p0, mlp_p1),
                                             attn_red_a, attn_red_b, mlp_red_a, mlp_red_b))
                continue
            attention = (PreparedKDA(w["attention"], source, pool.recurrent(index), self.delta,
                eps=config.text["rms_norm_eps"], records=kda_records.layer(index) if mode == "verify" else None,
                **kwargs) if kind == "kda" else
                PreparedMLA(w["attention"], source, metadata, pool.attention(index), context=context,
                            index_cache=pool.index(index) if kind == "dsa" else None,
                            eps=config.text["rms_norm_eps"],
                            peers=context.peers.get(index) if context is not None else None,
                            index_peers=(context.index_peers.get(index)
                                         if context is not None and context.index_peers is not None else None),
                            selection=self.selection,
                            score_exchange=score_exchange,
                            coordinator=coordinator,
                            # The smallest latent exchange that carries the step, else the largest in chunks.
                            latents=next((r for r in latent_runtimes if source.shape[0] <= r.max_batch_size),
                                         latent_runtimes[-1] if latent_runtimes else None),
                            **kwargs))
            if "mlp" in w:
                # The dense MLP writes the MoE layers' output, which its reduction consumes before the next mixture.
                dense = self.buffers.setdefault("dense", {})
                if "output" in self.buffers.get("moe", {}):
                    dense.setdefault("output", self.buffers["moe"]["output"])
                mlp = PreparedMLP(w["mlp"], source, session=session, name=prefix + ".mlp", buffers=dense,
                                  allocator=buffers.stage_allocator("dense"))
            else:
                # Shared experts write the sum buffer, which the preceding mixture consumed and the MoE
                # reduction overwrites only after their output is added.
                if "total" in self.buffers:
                    self.buffers.setdefault("moe", {}).setdefault("shared", {}).setdefault("output", self.buffers["total"])
                mlp = PreparedLatentMoE(w, source, config, weights.shard, runtime=runtime,
                    router_runtime=router_runtime, route_runtime=route_runtime if mode != "prefill" else None,
                    coordinator=coordinator, session=session, name=prefix + ".moe", workspace=workspace,
                    buffers=self.buffers.setdefault("moe", {}), storage=buffers, allocator=buffers.stage_allocator("moe"),
                    accumulation_buffers=accumulation_buffers,
                    # Only prefill graphs gather the latent; decode, verification and drafts keep their paths.
                    latent_gather_runtime=latent_gather_runtime if mode == "prefill" else None)
            # The sum buffer is consumed before the next attention or MLP reduction.
            total = self.buffers["total"][:source.shape[0]]
            if direct_ring:
                reduce = copy_engine_reduction(source, total, runtime=runtime, staging=reduction_staging,
                                               session=session, coordinator=coordinator, name=prefix + ".reduce")
            else:
                reduction = PreparedChunkedAllReduce if accumulation_buffers is None else PreparedOwnerPacketAllReduce
                reduction_args = {} if accumulation_buffers is None else dict(workspace=accumulation_buffers)
                reduce = reduction(source, total, runtime=runtime, coordinator=coordinator,
                                   session=session, name=prefix + ".reduce", **reduction_args)
            self.layers.append((w, attention, mlp, reduce))
        if allocator is not None:
            allocator.join(*buffers.stage_allocators.values())
        if overlap is not None:
            overlap.finish()
        if self.dual_wave:
            # The pool-cap knob (prefill_dual_wave_pool): bound the second prepared-module
            # set's added buffers so the KV pool keeps its token count inside the owner's band.
            self.dual_extra_bytes = extra = dual_wave_extra_bytes(self.buffers, self.dual_layers)
            if dual_wave_budget_bytes is not None and extra > dual_wave_budget_bytes:
                raise ValueError(
                    f"dual-wave second prepared-module set needs {extra} bytes on {self.name},"
                    f" above the {dual_wave_budget_bytes}-byte pool cap; lower --prefill-dual-wave-pool"
                    f" consumers or keep --prefill-dual-wave off")
        # The running block sum and committed blocks persist through one forward; every forward
        # rewrites them from its embeddings before reading them.
        r = self.buffers["residual"]
        # The normalized mixture entering attention takes the MoE output, which the previous reduction
        # consumed; the one entering the MLP takes the attention output, which its reduction consumed.
        self.mlp_input = self.buffers["kda"]["output"][:tokens]
        self.residual = ResidualStream(r["prefix"][:tokens], r["blocks"][:tokens], self.buffers["moe"]["output"][:tokens],
                                      mix_split_rows=mix_split_rows)
        self.scoring = None if scoring is None else scoring.prepare(self.residual.output, weights.head,
            vocabulary=weights.config.text["vocab_size"], session=session, coordinator=coordinator, name=name + ".prompt")
        self.head = PreparedVocabularyProjection(scratch.hidden, weights.head, scratch.local_logits, scratch.logits,
            runtime=vocabulary_runtime, coordinator=coordinator, session=session, name=name + ".head")
        self.replay = None
        if mode == "verify":
            indices = kda_records.layers
            states = [pool.recurrent(i).recurrent[:kda_records.corrections.shape[1]] for i in indices]
            self.replay = PreparedKDAReplay(pool.backing, states, kda_records, self.layers[indices[0]][1].plan,
                self.delta.state_indices[:, 0], session=session, workspace=workspace)
            regions = [pool.layout.region(f"layer{i}.convolution") for i in indices]
            self.convolution_offsets = torch.tensor([r.offset // 2 for r in regions], dtype=torch.int64, device=source.device)
            self.convolution_shape = regions[0].shape

    def commit(self, counts):
        self.replay(counts)
        channels, columns = self.convolution_shape
        torch.library.wrap_triton(_commit_convolution)[
            (self.convolution_offsets.numel(), counts.numel(), triton.cdiv(channels, 128))](
            self.pool.backing.view(torch.bfloat16), self.convolution_offsets, self.metadata.state_rows, counts,
            self.pool.layout.row_bytes // 2, channels, columns, 128)

    def __call__(self):
        m, d, c = self.metadata, self.delta, self.weights.config.text
        active = m.lengths > 0
        d.query_start_loc.copy_(m.query_start_loc)
        d.convolution_slots.copy_(torch.where(active, m.state_rows, -1))
        torch.eq(m.starts, 0, out=d.fresh)
        d.state_indices[:, 0].copy_(m.state_rows)
        d.initial_state_indices.copy_(torch.where(m.starts > 0, m.state_rows, 0))
        d.final_state_indices.copy_(m.state_rows)
        d.num_tokens.copy_(m.live_counts[:1])
        d.num_seqs.copy_(m.live_counts[1:])
        local = self.embedding.lookup()
        if self.vision_state is not None:
            self.vision_state.inject(m, local)
        # Each layer's residual source is the running sum of its block plus every committed block;
        # a block commits its running sum when the next block starts.
        size, eps = c["attn_res_block_size"], c["rms_norm_eps"]
        delta, mode = self.embedding.gather(local), REPLACE
        if self.dual_wave:
            return self._call_dual_wave(delta, mode, size, eps)
        if self.selection is not None:
            self.selection()
        for index, (w, attention, mlp, reduce) in enumerate(self.layers):
            committed, boundary = -(-index // size), index % size == 0
            mixture = None
            if (index in self.auxiliary_layers
                    and self.weights.config.draft.auxiliary_source == "attention_input"):
                column = self.auxiliary_layers.index(index) * self.residual.prefix.shape[1]
                mixture = self.auxiliary[:, column:column + self.residual.prefix.shape[1]]
            hidden = self.residual.mix(delta, mode, sources=committed, write=index // size if boundary else -1,
                score=w["attention_res_score"], norm=w["input_layernorm"], eps=eps, output_eps=eps, mixture=mixture,
                mixture_count=self.auxiliary_count if mixture is not None else None,
                mixture_window=self.auxiliary_window, mixture_alignment=self.auxiliary_alignment)
            attended = reduce(attention(hidden))
            hidden = self.residual.mix(attended, REPLACE if boundary else ADD, sources=committed + boundary, write=-1,
                score=w["mlp_res_score"], norm=w["post_attention_layernorm"], eps=eps, output_eps=eps,
                output=self.mlp_input)
            delta, mode = reduce(mlp(hidden)), ADD
            if (index in self.auxiliary_layers
                    and self.weights.config.draft.auxiliary_source == "residual_output"):
                column = self.auxiliary_layers.index(index) * self.residual.prefix.shape[1]
                self.auxiliary[:, column:column + self.residual.prefix.shape[1]].copy_(self.residual.prefix + delta)
        hidden = self.residual.mix(delta, mode, sources=-(-len(self.layers) // size), write=-1,
            score=self.weights.output_res_score, norm=self.weights.norm, eps=eps, output_eps=eps)
        if self.scoring is not None:
            self.scoring()
        torch.index_select(hidden, 0, m.sample_indices, out=self.scratch.hidden)
        self.head(self.scratch.hidden)

    def _call_dual_wave(self, delta, mode, size, eps):
        """L9 (29aac9e/eb72cab/2d3635f): interleave half-chunk (Wave A [0:H], Wave B [H:2H])
        SM compute on the auxiliary compute stream with the other wave's ring reductions
        and gathers on the priority -1 transport stream.

        The rows of one chunk split into two waves; every wave runs the same per-row
        arithmetic as the serial path (see the numerics contract in
        levers/evidence/l9-dualwave/port-map.md) through its own prepared module set.
        """
        half = self.half_tokens
        m = self.metadata
        la = m.lengths.clamp(min=0, max=half).to(torch.int32)
        lb = (m.lengths - half).clamp(min=0, max=half).to(torch.int32)
        act_a = la > 0
        act_b = lb > 0
        da, db = self.delta_a, self.delta_b
        da.query_start_loc[0].zero_()
        da.query_start_loc[1:].copy_(la)
        da.convolution_slots.copy_(torch.where(act_a, m.state_rows, -1))
        torch.eq(m.starts, 0, out=da.fresh)
        da.state_indices[:, 0].copy_(m.state_rows)
        da.initial_state_indices.copy_(torch.where(m.starts > 0, m.state_rows, 0))
        da.final_state_indices.copy_(torch.where(act_a, m.state_rows, 0))
        da.num_tokens.copy_(la[:1])
        da.num_seqs.copy_(act_a.to(torch.int32)[:1])
        db.query_start_loc[0].zero_()
        db.query_start_loc[1:].copy_(lb)
        db.convolution_slots.copy_(torch.where(act_b, m.state_rows, -1))
        db.fresh.zero_()
        db.state_indices[:, 0].copy_(m.state_rows)
        db.initial_state_indices.copy_(torch.where(act_b, m.state_rows, 0))
        db.final_state_indices.copy_(torch.where(act_b, m.state_rows, 0))
        db.num_tokens.copy_(lb[:1])
        db.num_seqs.copy_(act_b.to(torch.int32)[:1])

        main = torch.cuda.current_stream(delta.device)
        comm = torch.cuda.ExternalStream(self.comm_stream_handle, device=delta.device)
        comp = torch.cuda.ExternalStream(self.comp_stream_handle, device=delta.device)
        res_a = self.residual.substream(0, half)
        res_b = self.residual.substream(half, 2 * half)
        attn_in = self.residual.output
        mlp_in = self.dual_mlp_in
        tot_a = self.dual_total_w0[:half]
        tot_b = self.dual_total_w1[:half]
        n_layers = len(self.dual_layers)

        def on_comp(fn):
            comp.wait_stream(main)
            with torch.cuda.stream(comp):
                fn()
            main.wait_stream(comp)

        def overlap(comm_fn, comp_fn):
            comm.wait_stream(main)
            comp.wait_stream(main)
            with torch.cuda.stream(comm):
                comm_fn()
            with torch.cuda.stream(comp):
                comp_fn()
            main.wait_stream(comm)
            main.wait_stream(comp)

        def attn_mix_kwargs(idx, w, wave):
            committed, boundary = -(-idx // size), idx % size == 0
            mixture = None
            if idx in self.auxiliary_layers and self.weights.config.draft.auxiliary_source == "attention_input":
                col = self.auxiliary_layers.index(idx) * self.residual.prefix.shape[1]
                mix_full = self.auxiliary[:, col:col + self.residual.prefix.shape[1]]
                mixture = mix_full if self.auxiliary_count is not None else (
                    mix_full[:half] if wave == 0 else mix_full[half:])
            return dict(
                sources=committed,
                write=idx // size if boundary else -1,
                score=w["attention_res_score"],
                norm=w["input_layernorm"],
                eps=eps,
                output_eps=eps,
                mixture=mixture,
                mixture_count=self.auxiliary_count if mixture is not None else None,
                mixture_window=self.auxiliary_window,
                mixture_alignment=self.auxiliary_alignment,
            )

        delta_a, delta_b = delta[:half], delta[half:]
        mix_a_done = False
        kda_a_done = False
        mix_b_done = False
        for index, (w, kind, attention, mlp_spec, attn_red_a, attn_red_b, mlp_red_a, mlp_red_b) in enumerate(self.dual_layers):
            committed, boundary = -(-index // size), index % size == 0
            out_p0, out_p1 = attention[2:] if kind == "kda" else attention.out_pads
            if not mix_a_done:
                on_comp(lambda w=w, index=index, da=delta_a, md=mode: res_a.mix(da, md, **attn_mix_kwargs(index, w, 0)))
            if kind == "kda":
                kda_a, kda_b = attention[:2]
                if not kda_a_done:
                    on_comp(lambda kda_a=kda_a: kda_a(attn_in[:half]))

                def run_kda_wave_b(w=w, index=index, kda_b=kda_b, db=delta_b, md=mode, mix_b_done=mix_b_done):
                    if not mix_b_done:
                        res_b.mix(db, md, **attn_mix_kwargs(index, w, 1))
                    kda_b(attn_in[half:])
                overlap(
                    lambda attn_red_a=attn_red_a, out_p0=out_p0: attn_red_a(out_p0),
                    run_kda_wave_b,
                )
            else:
                if not mix_b_done:
                    on_comp(lambda w=w, index=index, db=delta_b, md=mode: res_b.mix(db, md, **attn_mix_kwargs(index, w, 1)))
                attention.pre_output(attn_in)
                on_comp(lambda attention=attention: attention.run_out_half(0))
                overlap(
                    lambda attn_red_a=attn_red_a, out_p0=out_p0: attn_red_a(out_p0),
                    lambda attention=attention: attention.run_out_half(1),
                )

            mlp_mode = REPLACE if boundary else ADD
            mlp_sources = committed + boundary
            mlp_kind, m_a, m_b, mlp_p0, mlp_p1 = mlp_spec
            if mlp_kind == "dense":
                def run_dense_a(w=w, m_a=m_a):
                    res_a.mix(tot_a, mlp_mode, sources=mlp_sources, write=-1,
                              score=w["mlp_res_score"], norm=w["post_attention_layernorm"],
                              eps=eps, output_eps=eps, output=mlp_in[:half])
                    m_a(mlp_in[:half])
                overlap(
                    lambda attn_red_b=attn_red_b, out_p1=out_p1: attn_red_b(out_p1),
                    run_dense_a,
                )

                def run_dense_b(w=w, m_b=m_b):
                    res_b.mix(tot_b, mlp_mode, sources=mlp_sources, write=-1,
                              score=w["mlp_res_score"], norm=w["post_attention_layernorm"],
                              eps=eps, output_eps=eps, output=mlp_in[half:])
                    m_b(mlp_in[half:])
                overlap(
                    lambda mlp_red_a=mlp_red_a, mlp_p0=mlp_p0: mlp_red_a(mlp_p0),
                    run_dense_b,
                )
                next_idx = index + 1
                nw, nkind, nattn = self.dual_layers[next_idx][:3]

                def pre_next_a(nw=nw, nkind=nkind, nattn=nattn, next_idx=next_idx):
                    res_a.mix(tot_a, ADD, **attn_mix_kwargs(next_idx, nw, 0))
                    if nkind == "kda":
                        nattn[0](attn_in[:half])
                overlap(
                    lambda mlp_red_b=mlp_red_b, mlp_p1=mlp_p1: mlp_red_b(mlp_p1),
                    pre_next_a,
                )
                delta_a, delta_b, mode = tot_a, tot_b, ADD
                mix_a_done = True
                kda_a_done = nkind == "kda"
                mix_b_done = False
                continue

            # MoE layer: the fully overlapped dual-wave pipeline.

            def prep_moe_a(w=w, m_a=m_a):
                res_a.mix(tot_a, mlp_mode, sources=mlp_sources, write=-1,
                          score=w["mlp_res_score"], norm=w["post_attention_layernorm"],
                          eps=eps, output_eps=eps, output=mlp_in[:half])
                m_a.phase_down_pack(mlp_in[:half])
            on_comp(prep_moe_a)

            overlap(m_a.phase_route_ring, lambda m_a=m_a: m_a.shared(mlp_in[:half]))

            def run_exp_a(m_a=m_a):
                m_a.phase_router()
                m_a.phase_experts()
            overlap(
                lambda attn_red_b=attn_red_b, out_p1=out_p1: attn_red_b(out_p1),
                run_exp_a,
            )

            def prep_moe_b(w=w, m_b=m_b):
                res_b.mix(tot_b, mlp_mode, sources=mlp_sources, write=-1,
                          score=w["mlp_res_score"], norm=w["post_attention_layernorm"],
                          eps=eps, output_eps=eps, output=mlp_in[half:])
                m_b.phase_down_pack(mlp_in[half:])
            overlap(m_a.phase_expert_reduce, prep_moe_b)

            def run_up_a_sh_b(m_a=m_a, m_b=m_b):
                m_a.phase_up_combine(m_a.shared.down.output)
                m_b.shared(mlp_in[half:])
            overlap(m_b.phase_route_ring, run_up_a_sh_b)

            def run_exp_b(m_b=m_b):
                m_b.phase_router()
                m_b.phase_experts()
            overlap(
                lambda mlp_red_a=mlp_red_a, mlp_p0=mlp_p0: mlp_red_a(mlp_p0),
                run_exp_b,
            )

            has_next = index + 1 < n_layers
            if has_next:
                nw, nkind, nattn = self.dual_layers[index + 1][:3]
                overlap(
                    m_b.phase_expert_reduce,
                    lambda nw=nw, next_idx=index + 1: res_a.mix(tot_a, ADD, **attn_mix_kwargs(next_idx, nw, 0)),
                )
            else:
                overlap(m_b.phase_expert_reduce, lambda: None)

            on_comp(lambda m_b=m_b: m_b.phase_up_combine(m_b.shared.down.output))
            if has_next and nkind == "kda":
                overlap(
                    lambda mlp_red_b=mlp_red_b, mlp_p1=mlp_p1: mlp_red_b(mlp_p1),
                    lambda nkda_a=nattn[0]: nkda_a(attn_in[:half]),
                )
                kda_a_done = True
            else:
                overlap(lambda mlp_red_b=mlp_red_b, mlp_p1=mlp_p1: mlp_red_b(mlp_p1), lambda: None)
                kda_a_done = False
            delta_a, delta_b, mode = tot_a, tot_b, ADD
            mix_a_done = has_next
            mix_b_done = False

        final_sources = -(-n_layers // size)
        res_a.mix(delta_a, mode, sources=final_sources, write=-1,
                  score=self.weights.output_res_score, norm=self.weights.norm, eps=eps, output_eps=eps)
        res_b.mix(delta_b, mode, sources=final_sources, write=-1,
                  score=self.weights.output_res_score, norm=self.weights.norm, eps=eps, output_eps=eps)
        if self.scoring is not None:
            self.scoring()
        torch.index_select(attn_in, 0, m.sample_indices, out=self.scratch.hidden)
        self.head(self.scratch.hidden)
