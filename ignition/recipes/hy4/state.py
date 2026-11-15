# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0
"""Per-rank Hy4 model state: shard-assignment-driven construction.

`build_model_state` consumes a `Hy4WeightLoader` end to end — every selected
checkpoint tensor lands in exactly one rank slot, at the contract's dtype (the
do-not-quantize floor: iHC `fn`/`base`/`scale`, sinks, the indexer
`weights_proj`/`k_norm`, the router correction bias and `lm_head` stay FP32,
the attention/indexer projections and router `gate` stay BF16) — and refuses
to finish with an unconsumed tensor. Construction runs on CPU for the
reference package's `tiny-hy4` (the rank-construction dry run) and on GPU for
the real checkpoint.

The iHC boundary (`norm.mhc.ihc`): `IhcBoundary.pre` / `.post` /
`.post_pre` wire the fused b12x operator on CUDA
(`B12XIHCCaps` -> `ihc.plan` -> PreparationSession -> `ihc.bind` ->
`run_post_pre`) and the operator's documented pure-torch reference on CPU
(`_ihc_torch_ref`, "usable directly in a FLARE recipe before the kernel
lands"). The two are bit-comparable by construction; the CPU path is what the
tiny dry run executes.
"""

from dataclasses import dataclass, field, replace

import torch

from .attention import (AttentionWeights, Hy4LayerAttention, IndexerWeights,
                        PackedInputProjections, SelectionCell)
from .config import ModelConfig
from .moe import (DenseBlock, DenseMlpWeights, MoEBlock, PackedMoEInputProjections,
                  ReferenceRoutedExperts, RouterWeights)
from .shard import Hy4Sharding
from .weights import kept_fp32


@dataclass(frozen=True)
class IhcWeights:
    """One iHC boundary's parameters, kept FP32 (`weights.kept_fp32`)."""

    fn: torch.Tensor    # [2 * hc_mult, hc_mult * hidden]
    base: torch.Tensor  # [2 * hc_mult]
    scale: torch.Tensor  # [2]


@dataclass(frozen=True)
class IhcHeadWeights:
    fn: torch.Tensor    # [hc_mult, hc_mult * hidden]
    base: torch.Tensor  # [hc_mult]
    scale: torch.Tensor  # [1]


class IhcBoundary:
    """One sublayer boundary of the 4-stream iHC state.

    `pre` collapses the streams into the sublayer input and emits the post
    gates; `post` redistributes the sublayer output (`post_i * branch +
    stream_i`); `post_pre` fuses the update with the next boundary's collapse
    and RMSNorm fold (`run_post_pre`, one launch). CUDA runs the prepared
    operator; CPU runs `_ihc_torch_ref` — the exact numerics the operator is
    tuned against.
    """

    def __init__(self, weights: IhcWeights, config: ModelConfig, device,
                 *, magnitude=None, plan=None):
        self.weights = weights
        self.device = device
        self.magnitude = config.text["hc_magnitude"] if magnitude is None else magnitude
        self.hc_eps = config.text["hc_eps"]
        self.norm_eps = config.text["rms_norm_eps"]
        self.plan = plan  # b12x Plan bound at prepare time (CUDA): the fused `post_pre`
        # A plan pins ONE invocation (`execution prepares post_pre, not pre`),
        # so the two single-op entries carry their own plans; unbound they take
        # the operator's lazy default (the CPU path never binds any).
        self.pre_plan = None
        self.post_plan = None

    def _reference(self):
        from b12x.norm.mhc import ihc

        return ihc

    def pre(self, streams):
        """`(post [rows, 4] FP32, y [rows, hidden])` from streams `[rows, 4, hidden]`."""
        w = self.weights
        if self.device.type == "cuda":
            from b12x.norm.mhc import ihc

            return ihc.run_pre(streams, w.fn, w.scale, w.base,
                               magnitude=self.magnitude, hc_eps=self.hc_eps, norm_eps=self.norm_eps,
                               plan=self.pre_plan)  # prepared plan: no JIT at run time
        ihc = self._reference()
        return ihc.ihc_pre_reference(streams, w.fn, w.scale, w.base,
                                     magnitude=self.magnitude, hc_eps=self.hc_eps, norm_eps=self.norm_eps)

    def post(self, branch, streams, post):
        """Redistribute `branch` over the streams (`post_i * branch + stream_i`)."""
        w = self.weights
        if self.device.type == "cuda":
            from b12x.norm.mhc import ihc

            return ihc.run_post(branch, streams, post, plan=self.post_plan)
        ihc = self._reference()
        return ihc.ihc_post_reference(branch, streams, post)

    def post_pre(self, branch, streams, prev_post, nxt, norm_weight):
        """Fused boundary: update the streams, then collapse for `nxt` + RMSNorm.

        `(streams, post, y) = run_post_pre(branch, streams, prev_post, nxt.fn,
        nxt.scale, nxt.base, norm_weight=...)` — the post update, the next
        pre, and the next sublayer's input RMSNorm in one launch.
        """
        if self.device.type == "cuda":
            from b12x.norm.mhc import ihc

            w = nxt.weights
            return ihc.run_post_pre(branch, streams, prev_post, w.fn, w.scale, w.base,
                                    magnitude=nxt.magnitude, hc_eps=nxt.hc_eps,
                                    norm_eps=nxt.norm_eps, norm_weight=norm_weight,
                                    layernorm_eps=self.norm_eps, plan=self.plan)
        ihc = self._reference()
        w = nxt.weights
        return ihc.ihc_post_pre_reference(branch, streams, prev_post, w.fn, w.scale, w.base,
                                          magnitude=nxt.magnitude, hc_eps=nxt.hc_eps,
                                          norm_eps=nxt.norm_eps, norm_weight=norm_weight,
                                          layernorm_eps=self.norm_eps)


class IhcHead:
    """The final stream collapse (`model.hc_head`, `run_head` before `model.norm`)."""

    def __init__(self, weights: IhcHeadWeights, config: ModelConfig, device):
        self.weights, self.device = weights, device
        self.hc_eps = config.text["hc_eps"]
        self.norm_eps = config.text["rms_norm_eps"]

    def merge(self, streams):
        w = self.weights
        if self.device.type == "cuda":
            from b12x.norm.mhc import ihc

            return ihc.run_head(streams, w.fn, w.scale, w.base,
                                hc_eps=self.hc_eps, norm_eps=self.norm_eps)
        from b12x.norm.mhc import ihc

        return ihc.ihc_head_reference(streams, w.fn, w.scale, w.base,
                                      hc_eps=self.hc_eps, norm_eps=self.norm_eps)


@dataclass(frozen=True)
class LayerState:
    """One decoder layer on this rank."""

    index: int
    dense: bool  # layer 0 carries the dense FFN; the rest are sparse MoE
    attention: Hy4LayerAttention
    attn_ihc: IhcBoundary
    ffn_ihc: IhcBoundary
    input_norm: torch.Tensor
    post_attention_norm: torch.Tensor
    mlp: object


@dataclass
class Hy4ModelWeights:
    """One rank's complete Hy4 body plus its MTP draft."""

    config: ModelConfig
    sharding: Hy4Sharding
    device: torch.device
    embed: torch.Tensor          # [vocab_local, hidden] BF16
    norm: torch.Tensor           # [hidden]
    head: torch.Tensor           # [vocab_local, hidden] FP32 (`enable_lm_head_fp32`)
    hc_head: IhcHead
    layers: tuple
    draft: object | None = None  # `mtp.MtpBlock` when speculation is prepared
    used_names: frozenset = frozenset()
    _prepared: dict = field(default_factory=dict)  # GPU session wiring

    @property
    def allocated_bytes(self):
        total = 0
        seen = set()
        visited = {}

        def visit(item):
            nonlocal total
            if isinstance(item, torch.Tensor):
                if item.device.type != "cpu" or True:
                    storage = item.untyped_storage()
                    if id(storage) not in seen:
                        seen.add(id(storage))
                        total += storage.nbytes()
                return
            # Object graphs legitimately cycle (Enum members reach their class's
            # member maps; prepared ops hold parent references), so dedupe by
            # identity before descending. The reference is kept so ids cannot be
            # reused while the walk runs.
            marker = id(item)
            if marker in visited:
                return
            visited[marker] = item
            if isinstance(item, (list, tuple)):
                for sub in item:
                    visit(sub)
            elif isinstance(item, dict):
                for sub in item.values():
                    visit(sub)
            elif hasattr(item, "__dataclass_fields__"):
                for sub in vars(item).values():
                    visit(sub)
            elif hasattr(item, "__dict__"):
                for sub in vars(item).values():
                    visit(sub)

        visit((self.embed, self.norm, self.head, self.hc_head, self.layers, self.draft))
        return total


class _Builder:
    """Name-checked tensor consumption from one open `Hy4WeightLoader`."""

    def __init__(self, loader, config, sharding):
        self.loader, self.config, self.sharding = loader, config, sharding
        self.used = set()

    def take(self, name):
        if name not in self.loader._sources:
            raise ValueError(f"Hy4 construction consumed a tensor the loader did not select: {name}")
        self.used.add(name)
        return name

    def norm(self, name, width):
        self.take(name)
        return self.loader.load_norm(name, width)

    def fp32(self, name, shape, *, dtype=torch.float32):
        """One do-not-quantize-floor tensor, loaded FP32 with its exact shape."""
        self.take(name)
        source = self.loader._sources[name]
        if tuple(source.shape) != tuple(shape):
            raise ValueError(f"{name}: expected shape {shape}, checkpoint has {tuple(source.shape)}")
        return self.loader.load_tensor(name, dtype=dtype)

    def linear(self, name, shape, *, row_range=None, column_range=None):
        """`name.weight` of full checkpoint `shape`, sliced to this rank's ranges.

        `row_range`/`column_range` are `(first, last)` global partitions (from
        `Hy4Sharding`); omitted axes load whole. The checkpoint dtype is
        preserved (BF16 projections stay BF16 — the do-not-quantize floor).
        `kept_fp32` tensors upcast AT LOAD whatever the checkpoint stores
        (`weights.kept_fp32`: the real shards hold `lm_head` as BF16
        [120832, 6144] and the runtime floor is FP32), so a BF16 source is
        consumed, not refused.
        """
        self.take(name + ".weight")
        source = self.loader._sources[name + ".weight"]
        if source.ndim != 2 or tuple(source.shape) != tuple(shape):
            raise ValueError(f"{name}.weight: expected {shape}, checkpoint has {tuple(source.shape)}")
        if kept_fp32(name + ".weight"):
            weight = self._load_upcast(name, shape, row_range=row_range, column_range=column_range)
        else:
            weight = self.loader.load_linear(name, rows=row_range, columns=column_range)
        expected = ((row_range[1] - row_range[0] if row_range else shape[0]),
                    (column_range[1] - column_range[0] if column_range else shape[1]))
        if tuple(weight.shape) != expected:
            raise ValueError(f"{name}.weight: rank slice {tuple(weight.shape)} != {expected}")
        return weight

    def _load_upcast(self, name, shape, *, row_range=None, column_range=None):
        """One `kept_fp32` matrix loaded into FP32 destinations straight from storage."""
        rows = slice(*(row_range if row_range else (0, shape[0])))
        columns = slice(*(column_range if column_range else (0, shape[1])))
        weight = self.loader.load_tensor(name + ".weight", dtype=torch.float32, extent=(rows, columns))
        if weight.dtype != torch.float32:
            raise ValueError(f"{name}.weight must load FP32 (`kept_fp32` floor)")
        return weight

    def matrix(self, name, shape, *, row_range=None, column_range=None):
        """Like `linear`, for prefixes with sibling buffers (`mlp.gate.*`).

        `load_linear` refuses prefixes whose module owns extra tensors (the
        router's `e_score_correction_bias` buffer); this reads the weight
        directly with the same rank slicing and shape checks. `kept_fp32`
        tensors upcast at load, as in `linear`.
        """
        self.take(name + ".weight")
        source = self.loader._sources[name + ".weight"]
        if source.ndim != 2 or tuple(source.shape) != tuple(shape):
            raise ValueError(f"{name}.weight: expected {shape}, checkpoint has {tuple(source.shape)}")
        if kept_fp32(name + ".weight"):
            return self._load_upcast(name, shape, row_range=row_range, column_range=column_range)
        rows = slice(*(row_range if row_range else (0, shape[0])))
        columns = slice(*(column_range if column_range else (0, shape[1])))
        weight = self.loader.load_tensor(name + ".weight", extent=(rows, columns))
        expected = (rows.stop - rows.start, columns.stop - columns.start)
        if tuple(weight.shape) != expected:
            raise ValueError(f"{name}.weight: rank slice {tuple(weight.shape)} != {expected}")
        return weight

    def finish(self, expected):
        leftover = set(expected) - self.used
        if leftover:
            raise ValueError(f"Hy4 construction left unconsumed checkpoint tensors: {sorted(leftover)[:8]}")
        return frozenset(self.used)


def _load_ihc(builder, prefix, *, head=False):
    config = builder.config
    streams = config.text["hc_mult"]
    rows = streams if head else 2 * streams
    shape = (rows, streams * config.hidden)
    suffix = ("hc_head_fn", "hc_head_base", "hc_head_scale") if head else ("hc_fn", "hc_base", "hc_scale")
    builder.take(f"{prefix}.{suffix[0]}")
    builder.take(f"{prefix}.{suffix[1]}")
    builder.take(f"{prefix}.{suffix[2]}")
    loaded = builder.loader.load_ihc(prefix, head=head)
    if tuple(loaded[suffix[0]].shape) != shape:
        raise ValueError(f"{prefix}.{suffix[0]}: expected {shape}, got {tuple(loaded[suffix[0]].shape)}")
    for part in suffix:
        if loaded[part].dtype != torch.float32:
            raise ValueError(f"{prefix}.{part}: the iHC floor is FP32")
    return loaded


def _attention_state(builder, layer, prefix, selection):
    config, sharding, loader = builder.config, builder.sharding, builder.loader
    c = config.text
    hidden = config.hidden
    qk = c["qk_nope_head_dim"] + c["qk_rope_head_dim"]
    attn = f"{prefix}.self_attn"

    q_a = builder.linear(f"{attn}.q_a_proj", (c["q_lora_rank"], hidden))
    q_a_norm = builder.norm(f"{attn}.q_a_layernorm.weight", c["q_lora_rank"])
    q_b = builder.linear(f"{attn}.q_b_proj", (config.heads * qk, c["q_lora_rank"]),
                         row_range=sharding.q_b_rows)
    kv_a = builder.linear(f"{attn}.kv_a_proj_with_mqa", (c["kv_lora_rank"] + c["qk_rope_head_dim"], hidden))
    kv_a_norm = builder.norm(f"{attn}.kv_a_layernorm.weight", c["kv_lora_rank"])
    # `kv_b_proj` loads REPLICATED — whole on every rank (consult8 §5's `kv_b`
    # two-wirings: the prefill graphs gather UNABSORBED queries and absorb +
    # expand LOCALLY over ALL heads; decode keeps the owner-side absorb and
    # reads only `sharding.head_range`'s rows).
    kv_b = builder.linear(f"{attn}.kv_b_proj",
                          (config.heads * (c["qk_nope_head_dim"] + c["v_head_dim"]), c["kv_lora_rank"]))
    o_proj = builder.linear(f"{attn}.o_proj", (hidden, config.heads * c["v_head_dim"]),
                            column_range=sharding.o_proj_columns)
    gate = builder.linear(f"{attn}.linear_gate", (config.heads * c["v_head_dim"], hidden),
                          row_range=sharding.gate_rows)
    builder.take(f"{attn}.learnable_sink_param")
    sink = loader.load_sink(attn)
    if sink.dtype != torch.float32 or sink.shape != (config.heads,):
        raise ValueError(f"{attn}.learnable_sink_param: expected {config.heads} FP32 sinks")
    sink = sink[sharding.head_range[0]:sharding.head_range[1]].contiguous()

    indexer = None
    if config.indexer_type(layer) == "full":
        ip = f"{attn}.indexer"
        indexer = IndexerWeights(
            wq_b=builder.linear(f"{ip}.wq_b", (c["index_n_heads"] * c["index_head_dim"], c["q_lora_rank"])),
            wk=builder.linear(f"{ip}.wk", (c["index_head_dim"], hidden)),
            k_norm_weight=builder.fp32(f"{ip}.k_norm.weight", (c["index_head_dim"],)),
            k_norm_bias=builder.fp32(f"{ip}.k_norm.bias", (c["index_head_dim"],)),
            weights_proj=builder.fp32(f"{ip}.weights_proj.weight", (c["index_n_heads"], hidden)))
        for name, tensor in (("wq_b", indexer.wq_b), ("wk", indexer.wk)):
            if tensor.dtype != torch.bfloat16:
                raise ValueError(f"{ip}.{name}: indexer projections stay BF16")
    attention_weights = AttentionWeights(
        q_a_proj=q_a, q_a_norm=q_a_norm, q_b_proj=q_b, kv_a_proj=kv_a,
        kv_a_norm=kv_a_norm, kv_b_proj=kv_b, o_proj=o_proj, gate_proj=gate,
        sink=sink, indexer=indexer)
    # The consult7 §5 bake-in "packed same-input projections at load": one
    # GEMM for `q_a_proj` + `kv_a_proj` + `linear_gate` + the indexer `wk`.
    attention_weights = replace(
        attention_weights, packed_input=PackedInputProjections.build(attention_weights))
    return Hy4LayerAttention(
        weights=attention_weights,
        config=config, sharding=sharding, selection=selection, device=builder.loader.device)


def _moe_state(builder, prefix, *, dense=False):
    config, sharding, loader = builder.config, builder.sharding, builder.loader
    c = config.text
    hidden = config.hidden
    if dense:
        weights = DenseMlpWeights(
            gate_proj=builder.linear(f"{prefix}.gate_proj", (c["intermediate_size"], hidden),
                                    row_range=sharding.dense_range),
            up_proj=builder.linear(f"{prefix}.up_proj", (c["intermediate_size"], hidden),
                                   row_range=sharding.dense_range),
            down_proj=builder.linear(f"{prefix}.down_proj", (hidden, c["intermediate_size"]),
                                    column_range=sharding.dense_range))
        for name in ("gate_proj", "up_proj", "down_proj"):
            if getattr(weights, name).dtype != torch.bfloat16:
                raise ValueError(f"{prefix}.{name}: dense FFN stays BF16")
        return DenseBlock(weights, loader.device)
    router = RouterWeights(
        gate=builder.matrix(f"{prefix}.gate", (config.experts, hidden)),
        correction_bias=builder.fp32(f"{prefix}.gate.e_score_correction_bias", (config.experts,)))
    if router.gate.dtype != torch.bfloat16:
        raise ValueError(f"{prefix}.gate.weight: the router floor is BF16")
    shared = DenseMlpWeights(
        gate_proj=builder.linear(f"{prefix}.shared_experts.gate_proj", (c["moe_intermediate_size"], hidden),
                                 row_range=sharding.expert_range),
        up_proj=builder.linear(f"{prefix}.shared_experts.up_proj", (c["moe_intermediate_size"], hidden),
                               row_range=sharding.expert_range),
        down_proj=builder.linear(f"{prefix}.shared_experts.down_proj", (hidden, c["moe_intermediate_size"]),
                                column_range=sharding.expert_range))
    expert_prefix = f"{prefix}.experts"
    if loader._packed:
        # Materialize through the loader session when the raw `_sources` entry
        # is a declared (meta) placeholder on GPU (see `weights.expert_stacks`).
        # The fused tensors hold EVERY routed expert's intermediate width; only
        # this rank's `sharding.expert_range` channels are resident (the same
        # slicing the BF16 branch applies below) — materializing whole fused
        # tensors overruns the card at full shape (w13: the 8-rank construction
        # OOMed at 91.7 GiB mid-experts loading 3 GiB all-expert tensors).
        # `load_tensor(extent=)` keeps the transfer over the checkpoint's byte
        # ranges; the fused [gate; up] row axis splits into the rank's two
        # half-ranges and re-fuses to [E, 2 * local, hidden // 2].
        first, last = sharding.expert_range
        inter = c["moe_intermediate_size"]
        half = (slice(None), slice(first, last), slice(None))
        half_up = (slice(None), slice(inter + first, inter + last), slice(None))
        down_extent = (slice(None), slice(None), slice(first // 2, last // 2))
        down_scale_extent = (slice(None), slice(None), slice(first // 16, last // 16))

        def load_sliced(name, extent):
            source = loader._sources[name]
            return loader.load_tensor(name, extent=extent) \
                if getattr(source, "is_meta", False) else source[extent]

        import torch as _torch
        packed_sources = {}
        for projection in ("gate_up_proj", "down_proj"):
            for suffix in ("weight", "weight_scale", "weight_scale_2"):
                name = f"{expert_prefix}.{projection}.{suffix}"
                if projection == "gate_up_proj" and suffix != "weight_scale_2":
                    packed_sources[name] = _torch.cat(
                        (load_sliced(name, half), load_sliced(name, half_up)), dim=1)
                elif projection == "down_proj" and suffix == "weight":
                    packed_sources[name] = load_sliced(name, down_extent)
                elif projection == "down_proj" and suffix == "weight_scale":
                    packed_sources[name] = load_sliced(name, down_scale_extent)
                else:  # per-expert gains: whole first axis
                    packed_sources[name] = load_sliced(name, (slice(None),))
        builder.used.update(packed_sources)
        if loader.device.type == "cuda":
            # M2 lever: the fused GPU routed-experts path. The rank keeps ONLY
            # the fused b12x planes (holding the reference's packed bytes as
            # well would double the weights); the torch reference loop stays
            # the CPU dry-run / gate contract (`ReferenceRoutedExperts`).
            from .moe import FusedRoutedExperts

            experts = loader.load_resident_experts(
                expert_prefix, packed_sources,
                intermediate_range=sharding.expert_range,
                swiglu_limit=c["swiglu_limit"])
            routed = FusedRoutedExperts(
                experts, hidden=hidden, top_k=c["num_experts_per_tok"],
                device=loader.device, name=f"{expert_prefix}.fused")
        else:
            routed = ReferenceRoutedExperts.from_packed(
                packed_sources, expert_prefix, swiglu_limit=c["swiglu_limit"])
    else:
        for projection in ("gate_up_proj", "down_proj"):
            builder.take(f"{expert_prefix}.{projection}")
        stacks = loader.expert_stacks(expert_prefix)
        first, last = sharding.expert_range
        sliced = {f"{expert_prefix}.gate_proj.weight": stacks[f"{expert_prefix}.gate_proj.weight"][:, first:last],
                  f"{expert_prefix}.up_proj.weight": stacks[f"{expert_prefix}.up_proj.weight"][:, first:last],
                  f"{expert_prefix}.down_proj.weight": stacks[f"{expert_prefix}.down_proj.weight"][:, :, first:last]}
        routed = ReferenceRoutedExperts.from_bf16(sliced, expert_prefix, swiglu_limit=c["swiglu_limit"])
    return MoEBlock(router=router, shared=shared, routed=routed, config=config,
                    device=loader.device,
                    # consult7 §5: router + shared gate/up pack at load.
                    packed_input=PackedMoEInputProjections.build(router, shared))


def build_model_state(loader, config, sharding, *, selections, mtp=True):
    """Construct one rank's Hy4 body from the loader's selected tensors.

    `selections` maps layer id -> `SelectionCell` (the IndexCache shared-selection
    wiring); the MTP block's cell lives at `config.layers`. Every selected
    checkpoint tensor must be consumed exactly once — construction raises on
    anything left over (the streaming loader's silent-drop failure mode is
    deliberately impossible here).
    """
    from .mtp import MtpBlock

    builder = _Builder(loader, config, sharding)
    hidden = config.hidden
    embed = builder.linear("model.embed_tokens", (config.vocabulary, hidden),
                           row_range=sharding.vocabulary_range)
    norm = builder.norm("model.norm.weight", hidden)
    # `kept_fp32` floor: the real shards store `lm_head` as BF16 [120832, 6144]
    # and it upcasts at load (`weights.kept_fp32`, `stored_parameter_counts`);
    # the runtime head is always FP32 (`enable_lm_head_fp32`).
    head = builder.linear("lm_head", (config.vocabulary, hidden), row_range=sharding.vocabulary_range)
    if head.dtype != torch.float32:
        raise ValueError("lm_head.weight must be loaded FP32 (`enable_lm_head_fp32`)")
    head_parts = _load_ihc(builder, "model.hc_head", head=True)
    hc_head = IhcHead(IhcHeadWeights(fn=head_parts["hc_head_fn"], base=head_parts["hc_head_base"],
                                     scale=head_parts["hc_head_scale"]), config, loader.device)

    layers = []
    for layer in range(config.layers):
        prefix = f"model.layers.{layer}"
        attn_parts = _load_ihc(builder, f"{prefix}.hc_attn_layer.hc_pre")
        ffn_parts = _load_ihc(builder, f"{prefix}.hc_mlp_layer.hc_pre")
        attention = _attention_state(builder, layer, prefix, selections[layer])
        input_norm = builder.norm(f"{prefix}.input_layernorm.weight", hidden)
        post_norm = builder.norm(f"{prefix}.post_attention_layernorm.weight", hidden)
        mlp = _moe_state(builder, f"{prefix}.mlp", dense=config.text["mlp_layer_types"][layer] == "dense")
        layers.append(LayerState(
            index=layer, dense=layer == 0, attention=attention,
            attn_ihc=IhcBoundary(IhcWeights(fn=attn_parts["hc_fn"], base=attn_parts["hc_base"],
                                            scale=attn_parts["hc_scale"]), config, loader.device),
            ffn_ihc=IhcBoundary(IhcWeights(fn=ffn_parts["hc_fn"], base=ffn_parts["hc_base"],
                                           scale=ffn_parts["hc_scale"]), config, loader.device),
            input_norm=input_norm, post_attention_norm=post_norm, mlp=mlp))

    draft = None
    if mtp:
        draft = MtpBlock.build(builder, selections[config.layers])
    builder.finish(set(loader._selected_names()))
    return Hy4ModelWeights(
        config=config, sharding=sharding, device=loader.device, embed=embed, norm=norm, head=head,
        hc_head=hc_head, layers=tuple(layers), draft=draft, used_names=frozenset(builder.used))
