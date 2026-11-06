# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Block-causal sliding-window MLA DFlash2 proposals over target-fed latent rings.

Accepted target positions write one E4M3 latent record per draft layer into a window-sized ring in the
request's continuation row, so restored continuations resume drafting. Every rank holds every record and
attends with its own heads. A proposal block is the pending token followed by mask tokens; each layer stages
the window beside the block's own records and convolves both attention branches across the block, and a
low-rank selector samples the block as a path.
"""

from dataclasses import dataclass, field

import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

from ignition.components.candidate_selector import PreparedCandidateSelector
from ignition.components.collectives import PreparedAllReduce
from ignition.components.embedding import PreparedShardedEmbedding
from ignition.components.grouped_convolution import PreparedGroupedConvolution
from ignition.components.mlp import MLPScratch, PreparedMLP
from ignition.components.vocabulary import PreparedVocabularyProjection
from ignition.components.weights import DirectWeightLoader, MLPWeights
from ignition.preparation.checkpoint import inspect_indexed_checkpoint

from .attention import PreparedMLA
from .dcp import RECORD_BYTES, write_records
from .latent_dspark import LayerWeights as LatentLayerWeights, PreparedContext as LatentContext, YarnRotary

PAGE_TOKENS = 128


def _norm(weight, eps):
    with torch.device("meta"):
        module = Qwen3RMSNorm(weight.numel(), eps=eps)
    module.weight = torch.nn.Parameter(weight, requires_grad=False)
    return module.eval()


@dataclass(frozen=True)
class LayerWeights(LatentLayerWeights):
    attention_conv: dict
    mlp_conv: dict


@dataclass(frozen=True)
class DraftWeights:
    projection: torch.Tensor
    hidden_norm: torch.Tensor
    context_kv: torch.Tensor
    norm: torch.Tensor
    embedding: torch.Tensor
    head: None
    selector: dict
    layers: tuple[LayerWeights, ...]

    @classmethod
    def load(cls, snapshot, config, device, shard, *, target_embedding, target_head, progress=None,
             online_mxfp8=False):
        if online_mxfp8:
            raise ValueError("the MLA DFlash2 package keeps its projections in BF16")
        d, hidden = config.draft, config.hidden
        d.validate_shard(shard)
        h0, h1 = shard.partition(d.heads)
        q_range, kv_range = shard.partition(d.text["q_lora_rank"]), shard.partition(RECORD_BYTES)
        intermediate = shard.partition(d.text["intermediate_size"], alignment=32)
        with DirectWeightLoader(inspect_indexed_checkpoint(snapshot), device, progress=progress,
                                progress_label="Load draft weights") as loader:
            def matrix(name, rows=None, columns=None):
                value = loader.load_tensor(name + ".weight", extent=(slice(*(rows or (None, None))),
                    slice(*(columns or (None, None)))))
                if value.ndim != 2 or value.dtype != torch.bfloat16:
                    raise ValueError(f"MLA DFlash2 requires BF16 matrices: {name}")
                return value

            def convolution(name):
                base, projection = loader.load_tensors((name + ".base_kernel", name + ".kernel_projection.weight"))
                return dict(base=base, projection=projection)

            project = matrix("fc", rows=shard.partition(hidden, alignment=64))
            if project.shape[1] != hidden * len(d.auxiliary_layers):
                raise ValueError("draft context projection does not match the selected target features")
            hidden_norm, norm = loader.load_tensors(("hidden_norm.weight", "norm.weight"))
            layers, context = [], []
            for i in range(d.layers):
                prefix = f"layers.{i}"
                a = prefix + ".self_attn"
                latent = (matrix(a + ".q_a_proj", rows=q_range), matrix(a + ".kv_a_proj_with_mqa", rows=kv_range))
                width = sum(x.shape[0] for x in latent)
                fused = torch.cat((*latent, latent[0].new_zeros((-width % 8, hidden))))
                kv = matrix(a + ".kv_b_proj", rows=(h0 * 256, h1 * 256)).view(h1 - h0, 256, 512)
                attention = dict(latent_gate=fused, q_b=matrix(a + ".q_b_proj", rows=(h0 * 192, h1 * 192)),
                    key=kv[:, :128].contiguous(), value=kv[:, 128:].transpose(1, 2).contiguous(),
                    query_norm=loader.load_tensor(a + ".q_a_layernorm.weight"),
                    kv_norm=loader.load_tensor(a + ".kv_a_layernorm.weight"),
                    output=matrix(a + ".o_proj", columns=(h0 * 128, h1 * 128)))
                context.append(matrix(a + ".kv_a_proj_with_mqa"))
                mlp = MLPWeights(torch.cat([matrix(prefix + ".mlp." + n, rows=intermediate)
                    for n in ("gate_proj", "up_proj")]), matrix(prefix + ".mlp.down_proj", columns=intermediate))
                layers.append(LayerWeights(attention, loader.load_tensor(prefix + ".input_layernorm.weight"),
                    loader.load_tensor(prefix + ".post_attention_layernorm.weight"), mlp,
                    convolution(prefix + ".attention_conv"), convolution(prefix + ".mlp_conv")))
            selector = loader.load_tensors(("candidate_selector.hidden_projection.weight",
                "candidate_selector.predecessor_codebook", "candidate_selector.successor_codebook"))
        selector = dict(zip(("projection", "predecessor", "successor"), selector, strict=True))
        return cls(project, hidden_norm, torch.cat(context), norm, target_embedding, None, selector, tuple(layers))


@dataclass(frozen=True)
class DraftScratch:
    local_embedding: torch.Tensor
    embedding: torch.Tensor
    normalized: torch.Tensor
    output: torch.Tensor
    mlp: MLPScratch
    hidden: torch.Tensor
    local_logits: torch.Tensor
    gathered_logits: torch.Tensor
    logits: torch.Tensor
    # Serial layers share one staged window, its block's records and the attention intermediates.
    window: torch.Tensor
    record_slots: torch.Tensor
    convolution: dict
    attention_buffers: dict
    selectors: dict = field(default_factory=dict)
    block_size: int = 8

    @classmethod
    def allocate(cls, config, shard, tokens, block, device):
        rows = tokens // block
        d, hidden = config.draft, config.hidden

        def zeros(*shape, dtype=torch.bfloat16):
            return torch.zeros(shape, dtype=dtype, device=device)
        pages = (d.window + block + PAGE_TOKENS - 1) // PAGE_TOKENS
        return cls(zeros(tokens, hidden // shard.world_size), zeros(tokens, hidden), zeros(tokens, hidden),
            zeros(tokens, hidden), MLPScratch.allocate(tokens, hidden, d.text["intermediate_size"] // shard.world_size, device),
            zeros(tokens, hidden), zeros(tokens, config.vocabulary // shard.world_size),
            zeros(tokens, config.vocabulary), zeros(tokens, config.vocabulary, dtype=torch.float32),
            zeros(rows, pages * PAGE_TOKENS, RECORD_BYTES, dtype=torch.uint8), zeros(tokens, dtype=torch.int64),
            dict(coefficients=zeros(tokens, 2 * 2 * hidden // 16), before=zeros(tokens, hidden), after=zeros(tokens, hidden)),
            {}, block_size=block)

    def rows(self, tokens, samples):
        if samples != tokens or not 0 < tokens <= self.embedding.shape[0] or tokens % self.block_size:
            raise ValueError("MLA DFlash2 scratch requires complete proposal blocks")
        shared = ("window", "convolution", "attention_buffers", "selectors", "block_size")
        return type(self)(*(getattr(self, name) if name in shared else self.mlp.rows(tokens) if name == "mlp"
                           else getattr(self, name)[:tokens] for name in self.__dataclass_fields__))

    def scratch_tensors(self):
        tensors = tuple(getattr(self, name) for name in self.__dataclass_fields__ if name != "selectors")
        return (*tensors, *((s.projected, s.unary, s.ids) for s in self.selectors.values()))


class WindowStage:
    """Linearize each request's committed ring and write the block's own records after it."""

    def __init__(self, ring, staged, slots, metadata, window):
        rows = metadata.lengths.numel()
        self.ring, self.metadata, self.window, self.slots = ring, metadata, window, slots
        self.staged = staged[:rows]
        self.capacity = staged.shape[1]
        self.table = self.staged.view(torch.float8_e4m3fn).view(rows * self.capacity, RECORD_BYTES)
        self.offsets = torch.arange(self.capacity, device=staged.device)

    def __call__(self, latent):
        m = self.metadata
        prefix = m.starts.clamp(max=self.window)
        history = (m.starts - prefix)[:, None] + self.offsets[None]
        self.staged.copy_(self.ring[m.state_rows.clamp_min(0)[:, None], history.remainder(self.window)])
        requests = m.requests.clamp_min(0)
        self.slots.copy_(torch.where(m.positions >= 0,
            requests * self.capacity + prefix[requests] + m.positions - m.starts[requests], -1))
        write_records(latent, self.slots, self.table)


class WindowView:
    """Query metadata whose lengths count the staged window, with the block's absolute rotary positions."""

    def __init__(self, metadata, pages):
        self.metadata, self.page_table = metadata, pages
        self.token_page_table = pages
        self.positions, self.query_start_loc = metadata.positions, metadata.query_start_loc
        self.lengths, self.requests = metadata.lengths, metadata.requests
        self.starts = torch.zeros_like(metadata.starts)


class DraftAttention:
    def __init__(self, index, weights, source, config, scratch, metadata, pool, *, runtime, coordinator, session,
                 name, workspace, latent_runtimes):
        rows, tokens = metadata.lengths.numel(), source.shape[0]
        d, count = config.draft, scratch.window.shape[1] // PAGE_TOKENS
        latents = next((r for r in latent_runtimes if tokens <= r.max_batch_size),
                       latent_runtimes[-1] if latent_runtimes else None)
        self.window = d.window
        pages = torch.arange(rows * count, dtype=torch.int32, device=source.device).view(rows, count)
        self.view = WindowView(metadata, pages)
        self.metadata = metadata
        cache = scratch.window[:rows].view(torch.float8_e4m3fn).view(rows * count, PAGE_TOKENS, RECORD_BYTES)
        stage = WindowStage(pool.draft_ring(index), scratch.window, scratch.record_slots, metadata, d.window)
        self.attention = PreparedMLA(weights.attention, source, self.view, cache, mode="verify", session=session,
            name=name, workspace=workspace, buffers=scratch.attention_buffers, latents=latents,
            coordinator=coordinator, eps=d.text["rms_norm_eps"], rotary=YarnRotary(config, source.device),
            gated=False, window=d.window, records=stage)
        self.reduce = PreparedAllReduce(self.attention.projections["output"].output, scratch.output, runtime=runtime,
            coordinator=coordinator, session=session, name=name + ".reduce")

    def __call__(self, source):
        self.view.starts.copy_(self.metadata.starts.clamp(max=self.window))
        return self.reduce(self.attention(source))


class PreparedContext(LatentContext):
    """Write accepted target feedback into every draft layer's committed ring."""

    def __init__(self, weights, auxiliary, metadata, pool, *, session, name, workspace, runtime, coordinator, buffers):
        super().__init__(weights, auxiliary, metadata, pool, session=session, name=name, workspace=workspace,
                         runtime=runtime, coordinator=coordinator, buffers=buffers)
        self.window = weights.config.draft.window

    def record_slots(self, counts):
        m = self.metadata
        offset, count = m.positions - m.starts[m.requests], counts[m.requests]
        live = (m.positions >= 0) & (offset < count) & (offset >= (count - self.window).clamp_min(0))
        return torch.where(live, m.state_rows[m.requests] * self.window + m.positions % self.window, -1)

    def records(self, index):
        return self.pool.draft_ring(index).view(torch.float8_e4m3fn)


class PreparedDraft:
    """The pending token and mask tokens through convolved MLA layers, the shared head and the selector."""

    def __init__(self, weights, scratch, metadata, pool, *, block, runtime, vocabulary_runtime, embedding_runtime,
                 coordinator, session, name, workspace, latent_runtimes=()):
        self.weights, self.scratch, self.metadata = weights, scratch, metadata
        config, draft = weights.config, weights.draft
        c, s = config.draft, scratch
        eps = c.text["rms_norm_eps"]
        self.embedding = PreparedShardedEmbedding(draft.embedding, metadata.token_ids, s.local_embedding, s.embedding,
            runtime=embedding_runtime, coordinator=coordinator, session=session, name=name + ".embedding",
            num_rows=metadata.live_counts[:1])
        self.layers = []
        for index, layer in enumerate(draft.layers):
            prefix = f"{name}.layer{index}"
            conv = {kind: PreparedGroupedConvolution(s.normalized, getattr(layer, kind), block_size=block,
                group_size=c.text["dflash_config"]["conv_group_size"], session=session, name=prefix + "." + kind,
                buffers=s.convolution, token_capacity=s.convolution["before"].shape[0])
                for kind in ("attention_conv", "mlp_conv")}
            attention = DraftAttention(index, layer, conv["attention_conv"].before, config, s, metadata, pool,
                runtime=runtime, coordinator=coordinator, session=session, name=prefix + ".attention",
                workspace=workspace, latent_runtimes=latent_runtimes)
            mlp = PreparedMLP(conv["mlp_conv"].before, layer.mlp, s.mlp.gate_up, s.mlp.intermediate, s.mlp.output,
                session=session, name=prefix + ".mlp", workspace=workspace, activation_mode="a16")
            reduction = PreparedAllReduce(s.mlp.output, s.output, runtime=runtime, coordinator=coordinator,
                session=session, name=prefix + ".mlp.reduce")
            self.layers.append((_norm(layer.input_norm, eps), conv["attention_conv"], attention,
                _norm(layer.post_attention_norm, eps), conv["mlp_conv"], mlp, reduction))
        self.norm = _norm(draft.norm, eps)
        head = weights.head if draft.head is None else draft.head
        self.head = PreparedVocabularyProjection(s.hidden, head, s.local_logits, s.gathered_logits,
            runtime=vocabulary_runtime, coordinator=coordinator, session=session, name=name + ".head")
        key = s.hidden.shape[0]
        if key not in s.selectors:
            s.selectors[key] = PreparedCandidateSelector(s.hidden.view(-1, block, s.hidden.shape[-1]), draft.selector,
                top_k=c.text["dflash_config"]["selector_top_k"], session=session, name=name + ".selector")
        self.selector, self.block = s.selectors[key], block

    def __call__(self):
        s = self.scratch
        self.embedding()
        for norm, conv, attention, post_norm, mlp_conv, mlp, reduce in self.layers:
            s.normalized.copy_(norm(s.embedding))
            s.embedding.add_(conv.finish(attention(conv.prepare(s.normalized))))
            s.normalized.copy_(post_norm(s.embedding))
            s.embedding.add_(mlp_conv.finish(reduce(mlp(mlp_conv.prepare(s.normalized)))))
        s.hidden.copy_(self.norm(s.embedding[self.metadata.sample_indices]))
        self.head(s.hidden)
        # The unary transform consumes the head result in FP32.
        s.logits.copy_(s.gathered_logits.float())
        self.selector.prepare(s.hidden.view(-1, self.block, s.hidden.shape[-1]))
