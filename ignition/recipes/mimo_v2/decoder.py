"""Canonical MiMo residual layers with prepared attention, experts and vocabulary shards."""

import os
from dataclasses import dataclass

import torch

from ignition.components.block_fp8 import BlockFP8Weight
from ignition.components.collectives import PreparedAllReduce, PreparedAllReduceRMSNorm
from ignition.components.embedding import PreparedEmbedding
from ignition.components.l2_prefetch import L2Prefetcher
from ignition.components.linear import UnquantizedProjection
from ignition.components.mlp import MLPScratch, PreparedMLP
from ignition.components.moe import PreparedRouter, PreparedRoutedExperts, RouterScratch
from ignition.components.vocabulary import PreparedVocabularyProjection

from .attention import AttentionScratch, PreparedAttention

# Decode and verification graphs prefetch upcoming projection weights into L2 beside the
# all-reduces, as the vLLM reference serves MiMo; FLARE_MIMO_L2_PREFETCH=0 at export disables it.
L2_PREFETCH = os.environ.get("FLARE_MIMO_L2_PREFETCH", "1") != "0"
L2_PREFETCH_MAX_TOKENS = 256
# Decode and verification reductions up to this payload run b12x's one-shot all-reduce with the
# residual add and RMSNorm in the same launch, the vLLM reference's 84 KiB default
# (VLLM_PCIE_ONESHOT_FUSED_ADD_RMS_NORM_MAX_SIZE): seven 6144-wide BF16 rows, every c1 width
# below eight. Larger payloads keep the two-shot route and the separate add and norm;
# FLARE_MIMO_FUSED_REDUCE_NORM_MAX_BYTES=0 at export disables the fusion.
FUSED_REDUCE_NORM_MAX_BYTES = int(os.environ.get("FLARE_MIMO_FUSED_REDUCE_NORM_MAX_BYTES", 84 << 10))


@dataclass(frozen=True)
class Scratch:
    embedding: torch.Tensor
    normalized: torch.Tensor
    attention: AttentionScratch
    dense: MLPScratch
    router_logits: torch.Tensor
    router: RouterScratch
    reduced: torch.Tensor
    hidden: torch.Tensor
    local_logits: torch.Tensor
    logits: torch.Tensor
    # Target-layer features the DFlash context projects within the same graph; graphs replay
    # serially, so one row-sliced plane serves every capacity (zero columns without drafting).
    auxiliary: torch.Tensor

    @classmethod
    def allocate(cls, config, shard, tokens, samples, device, *, buffers=None, auxiliary_columns=0):
        from ignition.preparation.scratch import GraphBufferPool

        c = config.text
        if buffers is None:
            buffers = GraphBufferPool(device, chunk_bytes=2 << 20).cursor()
        def zeros(rows, columns, dtype=torch.bfloat16, *, cursor=buffers):
            return cursor.zeros((rows, columns), dtype)
        embedding, normalized = zeros(tokens, c["hidden_size"]), zeros(tokens, c["hidden_size"])
        reduced = zeros(tokens, c["hidden_size"])
        attn, ffn = buffers.fork(), buffers.fork()
        attention = AttentionScratch.allocate(config, shard, tokens, device, requests=samples, buffers=attn)
        dense = MLPScratch.allocate(tokens, c["hidden_size"], c["intermediate_size"] // shard.world_size, device, buffers=ffn)
        router_logits = zeros(tokens, c["n_routed_experts"], torch.float32, cursor=ffn)
        router = RouterScratch.allocate(tokens, device, experts=c["n_routed_experts"], top_k=c["num_experts_per_tok"], buffers=ffn)
        buffers.join(attn, ffn)
        return cls(embedding, normalized, attention, dense, router_logits, router, reduced,
            zeros(samples, c["hidden_size"]), zeros(samples, c["vocab_size"] // shard.world_size),
            zeros(samples, c["vocab_size"]), zeros(tokens, auxiliary_columns))

    def rows(self, tokens, samples):
        return type(self)(self.embedding[:tokens], self.normalized[:tokens], self.attention.rows(tokens),
            self.dense.rows(tokens), self.router_logits[:tokens], self.router.rows(tokens), self.reduced[:tokens],
            self.hidden[:samples], self.local_logits[:samples], self.logits[:samples], self.auxiliary[:tokens])


class PreparedFeedForward(torch.nn.Module):
    def __init__(self, weights, config, scratch, *, runtime, coordinator, session, name, workspace,
                 reduce_output=True, a4_prefill=None):
        super().__init__()
        self.scratch = scratch
        self.dense = self.router_projection = self.router = self.experts = None
        source, output = scratch.normalized, scratch.dense.output
        if weights.mlp is not None:
            self.dense = PreparedMLP(source, weights.mlp, scratch.dense.gate_up, scratch.dense.intermediate,
                output, session=session, name=name, workspace=workspace)
        else:
            self.router_projection = UnquantizedProjection(source, weights.router, scratch.router_logits,
                                                           session=session, name=name + ".gate")
            c = config.text
            self.router = PreparedRouter(scratch.router_logits, scratch.router, session=session,
                name=name + ".route", score_func=c["scoring_func"], renormalize=c["norm_topk_prob"],
                routed_scaling_factor=c.get("routed_scaling_factor") or 1., correction_bias=weights.correction)
            self.experts = PreparedRoutedExperts(source, weights.experts, scratch.router.ids, scratch.router.weights,
                output, session=session, name=name + ".experts", workspace=workspace,
                router_plan=self.router.plan, a4_prefill=a4_prefill)
        self.reduction = PreparedAllReduce(output, scratch.reduced, runtime=runtime, coordinator=coordinator,
            session=session, name=name + ".reduce") if reduce_output else None
        # (prefetcher, plan) issued after the experts, before their reduction.
        self.prefetch = None

    def forward(self, hidden_states):
        self.scratch.normalized.copy_(hidden_states.reshape_as(self.scratch.normalized))
        return self.reduction(self.partial()).view_as(hidden_states)

    def partial(self):
        """This rank's unreduced output from the normalized scratch rows."""
        if self.dense is not None:
            return self.dense(self.scratch.normalized)
        self.router(self.router_projection(self.scratch.normalized))
        output = self.experts()
        if self.prefetch is not None:
            self.prefetch[0].issue(self.prefetch[1])
        return output


def _projection_tensors(weight):
    # Block-FP8 projections carry values and scales; BF16 ones are a single tensor.
    return (weight.values, weight.scales) if isinstance(weight, BlockFP8Weight) else (weight,)


_PREFETCH = {}


def _shared_prefetch(weights, device):
    """One side stream and one plan pair per layer for every decode and verification graph.

    Window A (attention, after the QKV projection): this layer's output projection, router
    and post-attention norm. Window C (after the experts, before their all-reduce): the next
    layer's input norm and QKV projection.
    """
    key = (id(weights), device)
    if key not in _PREFETCH:
        prefetcher, layers = L2Prefetcher(device), weights.layers
        plans = []
        for index, layer in enumerate(layers):
            attention = prefetcher.plan((*_projection_tensors(layer.output), layer.router, layer.post_attention_norm))
            experts = None
            if layer.experts is not None and index + 1 < len(layers):
                experts = prefetcher.plan((layers[index + 1].input_norm, *_projection_tensors(layers[index + 1].qkv)))
            plans.append((attention, experts))
        _PREFETCH[key] = prefetcher, tuple(plans)
    return _PREFETCH[key]


def install_l2_windows(layers, plans, prefetcher):
    for layer, (attention, experts) in zip(layers, plans):
        layer.self_attn.prefetch = (prefetcher, attention)
        if layer.mlp.experts is not None and experts is not None:
            layer.mlp.prefetch = (prefetcher, experts)


def canonical_norm(norm_type, weight, eps):
    with torch.device("meta"):
        module = norm_type(weight.numel(), eps=eps)
    module.weight = torch.nn.Parameter(weight, requires_grad=False)
    return module.eval()


class PreparedDecoder:
    def __init__(self, weights, environment, scratch, metadata, pool, *, mode, runtime,
                 vocabulary_runtime, coordinator, session, name, workspace, scoring=None, a4_prefill=None):
        layer_type, norm_type = environment.canonical
        self.weights, self.scratch, self.metadata, self.pool = weights, scratch, metadata, pool
        self.vision = environment.vision if mode == "prefill" else None
        c = weights.config.text
        self.embedding = PreparedEmbedding(weights.embedding, metadata.token_ids, scratch.embedding,
            num_rows=metadata.live_counts[:1], session=session, name=name + ".embedding")
        oneshot = getattr(runtime, "oneshot", runtime)
        partial = scratch.attention.partial
        self.fused = (mode != "prefill" and partial.numel() * partial.element_size() <= FUSED_REDUCE_NORM_MAX_BYTES
                      and scratch.dense.output.dtype == partial.dtype
                      and PreparedAllReduceRMSNorm.fits(oneshot, partial))
        self.layers, self.reductions = [], []
        for index, layer_weights in enumerate(weights.layers):
            # Populate the canonical layer's execution modules without constructing unloaded expert tensors.
            layer = layer_type.__new__(layer_type)
            torch.nn.Module.__init__(layer)
            layer.self_attn = PreparedAttention(index, layer_weights, weights.config, weights.shard,
                scratch.normalized, scratch.attention, metadata, pool, mode=mode, runtime=runtime,
                coordinator=coordinator, session=session, name=f"{name}.layer{index}.attention", workspace=workspace,
                reduce_output=not self.fused)
            layer.mlp = PreparedFeedForward(layer_weights, weights.config, scratch, runtime=runtime,
                coordinator=coordinator, session=session, name=f"{name}.layer{index}.mlp", workspace=workspace,
                reduce_output=not self.fused, a4_prefill=a4_prefill)
            layer.input_layernorm = canonical_norm(norm_type, layer_weights.input_norm, c["layernorm_epsilon"])
            layer.post_attention_layernorm = canonical_norm(norm_type, layer_weights.post_attention_norm,
                                                            c["layernorm_epsilon"])
            self.layers.append(layer.eval())
        self.norm = canonical_norm(norm_type, weights.norm, c["layernorm_epsilon"])
        if self.fused:
            # Each reduction adds into the residual stream (the embedding rows) and writes the
            # following norm into the normalized rows both projections read: the attention
            # partial with this layer's post-attention norm, the MLP partial with the next
            # layer's input norm or, after the last layer, the final norm.
            norms = [w.input_norm for w in weights.layers[1:]] + [weights.norm]
            for index, (layer_weights, following) in enumerate(zip(weights.layers, norms)):
                common = {"eps": c["layernorm_epsilon"], "runtime": oneshot, "coordinator": coordinator,
                          "session": session}
                self.reductions.append((
                    PreparedAllReduceRMSNorm(partial, scratch.embedding, layer_weights.post_attention_norm,
                        scratch.normalized, name=f"{name}.layer{index}.attention_reduce_norm", **common),
                    PreparedAllReduceRMSNorm(scratch.dense.output, scratch.embedding, following,
                        scratch.normalized, name=f"{name}.layer{index}.mlp_reduce_norm", **common)))
        self.target_layers = () if weights.draft is None else weights.config.draft.target_layers
        # Graphs prepared without drafting (the -no-mtp variants) take no auxiliary columns.
        columns = len(self.target_layers) * c["hidden_size"]
        if scratch.auxiliary.shape[0] != scratch.embedding.shape[0] or scratch.auxiliary.shape[1] < columns:
            raise ValueError("decoder scratch must carry one auxiliary column block per draft target layer")
        self.auxiliary = scratch.auxiliary[:, :columns]
        self.scoring = None if scoring is None else scoring.prepare(scratch.embedding, weights.head,
            vocabulary=weights.config.text["vocab_size"], session=session, coordinator=coordinator,
            name=name + ".prompt", copy_source=True)
        self.head = PreparedVocabularyProjection(scratch.hidden, weights.head, scratch.local_logits, scratch.logits,
            runtime=vocabulary_runtime, coordinator=coordinator, session=session, name=name + ".head")
        self.prefetcher = None
        if L2_PREFETCH and mode != "prefill" and scratch.embedding.shape[0] <= L2_PREFETCH_MAX_TOKENS:
            self.prefetcher, plans = _shared_prefetch(weights, scratch.embedding.device)
            install_l2_windows(self.layers, plans, self.prefetcher)

    def __call__(self):
        hidden = self.embedding()
        if self.vision is not None:
            self.vision.inject(self.metadata, hidden)
        if self.fused:
            return self._fused(hidden)
        for index, layer in enumerate(self.layers):
            hidden = layer(hidden)
            if index in self.target_layers:
                column = self.target_layers.index(index) * hidden.shape[-1]
                self.auxiliary[:, column:column + hidden.shape[-1]].copy_(hidden)
        if self.prefetcher is not None:
            self.prefetcher.join()
        if self.scoring is not None:
            self.scoring.source.copy_(self.norm(hidden))
            self.scoring()
            self.scratch.hidden.copy_(self.scoring.source[self.metadata.sample_indices])
        else:
            self.scratch.hidden.copy_(self.norm(hidden[self.metadata.sample_indices]))
        self.head(self.scratch.hidden)

    def _fused(self, residual):
        s = self.scratch
        s.normalized.copy_(self.layers[0].input_layernorm(residual))
        for index, (layer, (attention, mlp)) in enumerate(zip(self.layers, self.reductions)):
            attention(layer.self_attn(s.normalized)[0], residual)
            mlp(layer.mlp.partial(), residual)
            if index in self.target_layers:
                column = self.target_layers.index(index) * residual.shape[-1]
                self.auxiliary[:, column:column + residual.shape[-1]].copy_(residual)
        if self.prefetcher is not None:
            self.prefetcher.join()
        s.hidden.copy_(s.normalized[self.metadata.sample_indices])
        self.head(s.hidden)

    def commit(self, counts):
        for layer in self.layers:
            layer.self_attn.commit(counts)

    def scratch_tensors(self):
        return (self.auxiliary, *(layer.self_attn.scratch_tensors() for layer in self.layers))
