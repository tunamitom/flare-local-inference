"""Captured DFlash context projections and noncausal parallel proposals."""

from dataclasses import dataclass, fields
from functools import partial

import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RMSNorm

from ignition.components.collectives import PreparedAllGather, PreparedAllReduce, PreparedAllReduceRMSNorm
from ignition.components.embedding import PreparedEmbedding
from ignition.components.linear import UnquantizedProjection
from ignition.components.mlp import MLPScratch, PreparedMLP
from ignition.components.paged_attention import PreparedPagedAttention, PreparedPagedKVWriter
from ignition.components.speculation import SpeculativeRngState
from ignition.components.vocabulary import PreparedVocabularyProjection, vocabulary_shard_projection

from .attention import partial_rotary, stage_windows
from .decoder import PreparedDecoder, canonical_norm


# Draft blocks of up to eight rows (96 KiB) reduce through the one-shot channel with the residual add
# and the following RMSNorm in the same launch, as the target decoder does; above that, two-shot.
DRAFT_FUSED_REDUCE_NORM_MAX_BYTES = 96 << 10


def inverse_frequencies(geometry, device):
    return 1.0 / (geometry.theta ** (torch.arange(0, geometry.rotary_dim, 2,
        dtype=torch.float32, device=device) / geometry.rotary_dim))


@dataclass(frozen=True)
class ContextBuffers:
    """Context projection planes for the largest graph; graphs replay serially and slice rows."""

    local: torch.Tensor
    projected: torch.Tensor
    kv: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    slots: torch.Tensor

    @classmethod
    def allocate(cls, weights, tokens, device, dtype=torch.bfloat16):
        c, a = weights.config.draft, weights.config.draft.attention
        _, kv_heads = a.local(weights.shard)
        hidden = c.text["hidden_size"]
        return cls(torch.zeros((tokens, hidden // weights.shard.world_size), dtype=dtype, device=device),
                   torch.zeros((tokens, hidden), dtype=dtype, device=device),
                   torch.zeros((tokens, weights.draft.context_kv.shape[0]), dtype=dtype, device=device),
                   torch.zeros((tokens, kv_heads, a.key_dim), dtype=dtype, device=device),
                   torch.zeros((tokens, kv_heads, a.value_dim), dtype=dtype, device=device),
                   torch.full((tokens,), -1, dtype=torch.int64, device=device))

    def rows(self, tokens):
        if tokens > self.slots.shape[0]:
            raise ValueError("context graph exceeds the shared context capacity")
        return type(self)(*(getattr(self, f.name)[:tokens] for f in fields(self)))


class PreparedContext:
    """Project accepted target features once into all draft layers' context K/V."""

    def __init__(self, weights, auxiliary, metadata, pool, *, session, name, workspace, runtime, coordinator,
                 buffers=None):
        self.metadata, self.pool = metadata, pool
        c, a = weights.config.draft, weights.config.draft.attention
        self.window = a.window
        self.value_scale = c.text["dflash_config"].get("attention_value_scale", 1.)
        tokens, device = auxiliary.shape[0], auxiliary.device
        planes = (ContextBuffers.allocate(weights, tokens, device, auxiliary.dtype) if buffers is None
                  else buffers.rows(tokens))
        self.local, self.projected = planes.local, planes.projected
        self.project = UnquantizedProjection(auxiliary, weights.draft.projection, self.local,
                                             session=session, name=name + ".project")
        self.gather = PreparedAllGather(self.local, self.projected, runtime=runtime, coordinator=coordinator,
                                       session=session, name=name + ".gather")
        self.norm = canonical_norm(Qwen3RMSNorm, weights.draft.hidden_norm, c.text["rms_norm_eps"])
        self.kv = planes.kv
        self.project_kv = UnquantizedProjection(self.projected, weights.draft.context_kv, self.kv,
                                               session=session, name=name + ".kv")
        self.key, self.value, self.slots = planes.key, planes.value, planes.slots
        self.inverse = inverse_frequencies(a, device)
        self.stages = []
        for index, layer in enumerate(weights.draft.layers):
            norm = canonical_norm(Qwen3RMSNorm, layer.key_norm, c.text["rms_norm_eps"])
            writer = PreparedPagedKVWriter(self.key, self.value, pool.reserved_region(f"draft{index}.key"),
                pool.reserved_region(f"draft{index}.value"), self.slots, session=session, name=f"{name}.{index}.write")
            self.stages.append((norm, writer))

    def __call__(self, auxiliary, counts):
        m = self.metadata
        offset = m.positions - m.starts[m.requests]
        count = counts[m.requests]
        live = (m.positions >= 0) & (offset < count) & (offset >= (count - self.window).clamp_min(0))
        self.slots.copy_(torch.where(live, m.state_rows[m.requests] * self.window + m.positions % self.window, -1))
        self.projected.copy_(self.norm(self.gather(self.project(auxiliary))))
        self.project_kv(self.projected)
        width = self.key.shape[1] * self.key.shape[2]
        chunks = self.kv.view(self.kv.shape[0], len(self.stages), -1)
        for index, (norm, writer) in enumerate(self.stages):
            key = norm(chunks[:, index, :width].view_as(self.key))
            self.key.copy_(partial_rotary(key, m.positions, self.inverse))
            self.value.copy_(chunks[:, index, width:].view_as(self.value) * self.value_scale)
            writer()

    def scratch_tensors(self):
        return (self.local, self.projected, self.kv, self.key, self.value, self.slots)


@dataclass(frozen=True)
class DraftScratch:
    embedding: torch.Tensor
    qkv: torch.Tensor
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    attended: torch.Tensor
    partial: torch.Tensor
    output: torch.Tensor
    window_key: torch.Tensor
    window_value: torch.Tensor
    mlp: MLPScratch
    hidden: torch.Tensor
    local_logits: torch.Tensor
    logits: torch.Tensor

    @classmethod
    def allocate(cls, config, shard, tokens, device, *, full_logits=True):
        a, c = config.attention, config.text
        heads, kv_heads = a.local(shard)
        rows = tokens // config.block_size
        pages = rows * ((a.window + config.block_size + 127) // 128)
        def zeros(*shape, dtype=torch.bfloat16):
            return torch.zeros(shape, dtype=dtype, device=device)
        hidden = c["hidden_size"]
        return cls(zeros(tokens, hidden), zeros(tokens, heads * a.key_dim + kv_heads * (a.key_dim + a.value_dim)),
            zeros(tokens, heads, a.key_dim), zeros(tokens, kv_heads, a.key_dim), zeros(tokens, kv_heads, a.value_dim),
            zeros(tokens, heads, a.value_dim), zeros(tokens, hidden), zeros(tokens, hidden),
            zeros(pages, 128, kv_heads, a.key_dim, dtype=torch.float8_e4m3fn),
            zeros(pages, 128, kv_heads, a.value_dim, dtype=torch.float8_e4m3fn),
            MLPScratch.allocate(tokens, hidden, c["intermediate_size"] // shard.world_size, device),
            zeros(tokens, hidden), zeros(tokens, c["vocab_size"] // shard.world_size),
            zeros(tokens, c["vocab_size"] if full_logits else 0))

    def rows(self, tokens, samples):
        if samples != tokens:
            raise ValueError("DFlash projects every block position")
        return type(self)(*(getattr(self, field) if field.startswith("window_")
                           else self.mlp.rows(tokens) if field == "mlp"
                           else getattr(self, field)[:tokens] for field in self.__dataclass_fields__))


class DraftAttention(torch.nn.Module):
    def __init__(self, index, weights, config, scratch, metadata, pool, *, runtime, coordinator, session, name, workspace,
                 reduce_output=True):
        super().__init__()
        self.scratch, self.metadata = scratch, metadata
        a = config.attention
        self.value_scale = config.text["dflash_config"].get("attention_value_scale", 1.)
        self.qkv = UnquantizedProjection(scratch.embedding, weights.qkv, scratch.qkv, session=session, name=name + ".qkv")
        self.q_norm = canonical_norm(Qwen3RMSNorm, weights.query_norm, config.text["rms_norm_eps"])
        self.k_norm = canonical_norm(Qwen3RMSNorm, weights.key_norm, config.text["rms_norm_eps"])
        self.inverse = inverse_frequencies(a, scratch.embedding.device)
        self.key_cache, self.value_cache = (pool.reserved_region(f"draft{index}.{kind}") for kind in ("key", "value"))
        rows = metadata.lengths.numel()
        count = (a.window + config.block_size + 127) // 128
        self.key, self.value = scratch.window_key[:rows * count], scratch.window_value[:rows * count]
        self.pages = torch.arange(rows * count, device=scratch.embedding.device, dtype=torch.int32).view(rows, count)
        self.lengths = torch.zeros_like(metadata.lengths)
        self.window = a.window
        self.attention = PreparedPagedAttention(scratch.query, self.key, self.value, self.pages, self.lengths,
            metadata.query_start_loc, scratch.attended, sink=weights.sink, mode="verify", width=config.block_size,
            window=a.window - 1, causal=False, session=session, workspace=workspace, name=name + ".attention")
        self.output = UnquantizedProjection(scratch.attended.flatten(1), weights.output, scratch.partial,
                                            session=session, name=name + ".output")
        self.reduce = PreparedAllReduce(scratch.partial, scratch.output, runtime=runtime, coordinator=coordinator,
                                       session=session, name=name + ".reduce") if reduce_output else None

    def forward(self, hidden_states, **kwargs):
        s, m = self.scratch, self.metadata
        q, k, v = self.qkv(hidden_states).split((s.query[0].numel(), s.key[0].numel(), s.value[0].numel()), -1)
        s.query.copy_(partial_rotary(self.q_norm(q.view_as(s.query)), m.positions, self.inverse))
        s.key.copy_(partial_rotary(self.k_norm(k.view_as(s.key)), m.positions, self.inverse))
        s.value.copy_(v.view_as(s.value) * self.value_scale)
        stage_windows(self.key_cache, s.key, self.key, m)
        stage_windows(self.value_cache, s.value, self.value, m)
        self.lengths.copy_(m.starts.clamp(max=self.window).int() + m.lengths)
        self.attention()
        partial = self.output(s.attended.flatten(1))
        return (partial if self.reduce is None else self.reduce(partial)), None


class DraftMLP(torch.nn.Module):
    def __init__(self, weights, scratch, *, runtime, coordinator, session, name, workspace, reduce_output=True):
        super().__init__()
        s = scratch.mlp
        self.mlp = PreparedMLP(scratch.embedding, weights, s.gate_up, s.intermediate, s.output,
                              session=session, workspace=workspace, name=name)
        self.reduce = PreparedAllReduce(s.output, scratch.output, runtime=runtime, coordinator=coordinator,
                                       session=session, name=name + ".reduce") if reduce_output else None

    def forward(self, hidden_states):
        partial = self.mlp(hidden_states)
        return partial if self.reduce is None else self.reduce(partial)


class PreparedDraft:
    """The draft decoder; its head gathers full-vocabulary logits, or with ``gather_logits=False``
    leaves each rank's vocabulary shard in the local logits for sharded proposals."""

    def __init__(self, weights, scratch, metadata, pool, *, runtime, vocabulary_runtime, coordinator,
                 session, name, workspace, gather_logits=True):
        self.weights, self.scratch, self.metadata = weights, scratch, metadata
        c = weights.config.draft
        self.embedding = PreparedEmbedding(weights.embedding, metadata.token_ids, scratch.embedding,
            num_rows=metadata.live_counts[:1], session=session, name=name + ".embedding")
        oneshot = getattr(runtime, "oneshot", runtime)
        self.fused = (scratch.partial.numel() * scratch.partial.element_size() <= DRAFT_FUSED_REDUCE_NORM_MAX_BYTES
                      and scratch.mlp.output.dtype == scratch.partial.dtype
                      and PreparedAllReduceRMSNorm.fits(oneshot, scratch.partial))
        self.layers, self.reductions = [], []
        for index, layer_weights in enumerate(weights.draft.layers):
            layer = Qwen3DecoderLayer.__new__(Qwen3DecoderLayer)
            torch.nn.Module.__init__(layer)
            layer.self_attn = DraftAttention(index, layer_weights, c, scratch, metadata, pool,
                runtime=runtime, coordinator=coordinator, session=session, name=f"{name}.layer{index}.attention",
                workspace=workspace, reduce_output=not self.fused)
            layer.mlp = DraftMLP(layer_weights.mlp, scratch, runtime=runtime, coordinator=coordinator,
                                session=session, name=f"{name}.layer{index}.mlp", workspace=workspace,
                                reduce_output=not self.fused)
            layer.input_layernorm = canonical_norm(Qwen3RMSNorm, layer_weights.input_norm, c.text["rms_norm_eps"])
            layer.post_attention_layernorm = canonical_norm(Qwen3RMSNorm, layer_weights.post_attention_norm,
                                                            c.text["rms_norm_eps"])
            self.layers.append(layer.eval())
        self.norm = canonical_norm(Qwen3RMSNorm, weights.draft.norm, c.text["rms_norm_eps"])
        if self.fused:
            # The residual stream lives in the reduction output rows; each reduction adds into it and
            # writes the following norm (post-attention, the next input norm, or the final norm) into
            # the embedding rows every projection reads.
            layers = weights.draft.layers
            following = [w.input_norm for w in layers[1:]] + [weights.draft.norm]
            common = {"eps": c.text["rms_norm_eps"], "runtime": oneshot, "coordinator": coordinator,
                      "session": session}
            for index, (layer_weights, after) in enumerate(zip(layers, following)):
                self.reductions.append((
                    PreparedAllReduceRMSNorm(scratch.partial, scratch.output, layer_weights.post_attention_norm,
                        scratch.embedding, name=f"{name}.layer{index}.attention_reduce_norm", **common),
                    PreparedAllReduceRMSNorm(scratch.mlp.output, scratch.output, after, scratch.embedding,
                        name=f"{name}.layer{index}.mlp_reduce_norm", **common)))
        self.head = (PreparedVocabularyProjection(scratch.hidden, weights.head, scratch.local_logits, scratch.logits,
            runtime=vocabulary_runtime, coordinator=coordinator, session=session, name=name + ".head")
            if gather_logits else vocabulary_shard_projection(scratch.hidden, weights.head, scratch.local_logits,
                session=session, name=name + ".head.projection"))

    def __call__(self):
        hidden = torch.where((self.metadata.token_ids == self.weights.config.draft.noise_token)[:, None],
                             self.weights.draft.mask_embedding, self.embedding())
        if self.fused:
            s = self.scratch
            s.output.copy_(hidden)
            s.embedding.copy_(self.layers[0].input_layernorm(s.output))
            for layer, (attention, mlp) in zip(self.layers, self.reductions):
                attention(layer.self_attn(s.embedding)[0], s.output)
                mlp(layer.mlp(s.embedding), s.output)
            s.hidden.copy_(s.embedding[self.metadata.sample_indices])
            self.head(s.hidden)
            return
        for layer in self.layers:
            hidden = layer(hidden)
        self.scratch.hidden.copy_(self.norm(hidden[self.metadata.sample_indices]))
        self.head(self.scratch.hidden)


@dataclass(frozen=True)
class DraftForwards:
    decoder: object
    context: object
    draft: object = PreparedDraft
    markov: object = None


class SpeculativePreparation:
    def __init__(self, weights, canonical, pool, requests, inputs, scratch, *, completion, frame, sequence,
                 session, workspace, runtime, vocabulary_runtime, context_runtime, coordinator, context_tokens=None,
                 draft_candidates=None):
        from .pool import PAGE_TOKENS

        self.page_tokens = PAGE_TOKENS
        self.weights, self.tables, self.pool = weights, canonical, pool
        self.requests, self.inputs, self.scratch = requests, inputs, scratch
        self.completion, self.frame, self.sequence = completion, frame, sequence
        self.session, self.workspace = session, workspace
        self.collectives = dict(runtime=runtime, vocabulary_runtime=vocabulary_runtime, coordinator=coordinator)
        self.markov_runtime = None
        # Every prefill and verification graph projects its context through one shared set of planes.
        self.context_buffers = (None if context_tokens is None
                                else ContextBuffers.allocate(weights, context_tokens, pool.backing.device))
        # A candidate channel proposes from each rank's draft vocabulary shard instead of gathered logits.
        self.draft_candidates = draft_candidates
        self.forwards = DraftForwards(PreparedDecoder,
            partial(PreparedContext, runtime=context_runtime, coordinator=coordinator, buffers=self.context_buffers),
            partial(PreparedDraft, gather_logits=draft_candidates is None))
        self.rng = SpeculativeRngState.allocate(pool.backing.device, capacity=requests.capacity)
        self.generation, self.offset = torch.zeros_like(requests.generation), torch.zeros_like(requests.generation)
        self.block_size, self.logit_offset = weights.config.draft.block_size, 1
        self.draft_scratch = DraftScratch.allocate(weights.config.draft, weights.shard,
            requests.capacity * self.block_size, pool.backing.device, full_logits=draft_candidates is None)
