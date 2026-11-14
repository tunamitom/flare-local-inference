# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0
"""Hy4 gated sparse-MLA attention with the DSA indexer and shared selection.

One decoder layer's attention is:

1. `q_resid = q_a_layernorm(q_a_proj(x))` and the MLA latents
   `k_pass = kv_a_layernorm(kv_a_proj(x))`, `k_rot` — both rotary blocks get
   the interleaved (adjacent-pair, PR #50589) rotation, applied to the indexer
   keys and queries at write/prepare time (`dsa_indexer.rope`).
2. Selection: `full` layers run the lightning indexer and write their top-k into
   the `IndexCache` owner buffer (`dsa_indexer` physical slot space);
   `shared` layers bind the previous full layer's buffer verbatim — same
   `data_ptr`, no copy, no recompute (the stale-selection mutant must die).
3. Attention: queries are absorbed into latent space (`q_nope @ W_UK`), the
   sparse MLA attends to the caller-selected latent records with the per-head
   learnable sink folded exactly once, the output is expanded back
   (`@ W_UV`), elementwise-gated (`sigmoid(linear_gate(x))`) and projected.

Operator wiring (the GPU path): `sparse_mla.plan -> bind -> run` with the sink
provenance the context-parallel merge needs (`Binding.sink_folded`,
`Binding.lse_base`), `dsa_indexer.plan -> bind -> run` against the shared
`IndexCache` cells, and `IndexCache.handback` /
`dsa_indexer.PreparedSelectionHandback` for the cross-rank selection. The CPU
dry-run path executes the same contract in torch (the plan's P3 torch-op
composition): identical operands, identical provenance fields, device-agnostic
math. Both paths return the same LSE contract: base-2 LSE with the sink already
folded (`sink_folded=True` when a sink is bound), because the kernels fold the
sink into both the normalizer and the LSE (`_shared/mla/decode_math.py`).
"""

from dataclasses import dataclass, field

import torch

from .config import ModelConfig
from .shard import Hy4Sharding

# Epsilon convention (reference package, defect 19): the indexer `k_norm` is a
# LayerNorm fed by `rms_norm_eps` (the GGUF port's tie-breaker). b12x's
# `dsa_indexer.rope.INDEX_K_NORM_EPS` default is 1e-6 (vLLM's deviation) — the
# recipe always passes the epsilon explicitly so the decided 1e-5 wins.
K_NORM_EPS_SOURCE = "config.rms_norm_eps"


def rotate_half_interleaved(x):
    """Adjacent-pair rotation half (PR #50589), over the last dimension."""
    if x.shape[-1] % 2:
        raise ValueError("Hy4 rotary embeddings require an even final dimension")
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_interleaved_rope(x, positions, *, rope_dim, theta):
    """Interleaved rotary embedding on the LAST `rope_dim` dims of `x`.

    The checkpoint stores the rotary block last (`torch.split(x,
    [dim - rope_dim, rope_dim])` in the witness) with consecutive rotary
    pairs, so the rotation is the GPT-J adjacent-pair form. `positions` is
    int/long per row. Matches `dsa_indexer.rope.apply_indexer_rope` at
    width 128 (checked in `tests/recipes/test_hy4_decoder_body.py`).
    """
    if rope_dim <= 0 or rope_dim % 2:
        raise ValueError("rope_dim must be a positive even number")
    if x.shape[-1] < rope_dim:
        raise ValueError("rotary block exceeds the input width")
    pe = x[..., x.shape[-1] - rope_dim:]
    inv_freq = 1.0 / (theta ** (torch.arange(0, rope_dim, 2, device=x.device, dtype=torch.float32) / rope_dim))
    freqs = positions.reshape(-1, 1).float() * inv_freq.reshape(1, -1)
    angles = freqs.repeat_interleave(2, dim=-1)  # interleaved pairs, NOT split-half
    cos, sin = angles.cos(), angles.sin()
    while cos.ndim < pe.ndim:
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    rotated = pe.float() * cos + rotate_half_interleaved(pe.float()) * sin
    out = x.clone()
    out[..., x.shape[-1] - rope_dim:] = rotated.to(x.dtype)
    return out


@dataclass(frozen=True)
class SparseMLAProvenance:
    """What a context-parallel merge needs to know about one attention call.

    Mirrors `sparse_mla.Binding.sink_folded` / `.lse_base`: under CP every rank
    folds its shard WITHOUT the sink and the post-merge `PreparedSinkFold` folds
    it exactly once (`partials_sink_folded=provenance.sink_folded`,
    `lse_base=provenance.lse_base`). The eager path folds per call (no CP), so
    it reports `sink_folded=True` whenever a sink is bound — the same contract
    the kernels implement.
    """

    sink_folded: bool
    lse_base: str = "base2"

    @classmethod
    def from_binding(cls, binding):
        return cls(sink_folded=binding.sink_folded, lse_base=binding.lse_base)


@dataclass(frozen=True)
class SparseMLAOutput:
    """Attention output [rows, heads, v_head_dim] plus its base-2 folded LSE."""

    output: torch.Tensor
    lse: torch.Tensor | None
    provenance: SparseMLAProvenance


@dataclass(frozen=True)
class IndexerWeights:
    """The lightning indexer of one `full` layer (or the MTP block).

    `wq_b`/`wk` decode BF16; `weights_proj`/`k_norm` stay FP32
    (`weights.kept_fp32`), and `k_norm` is a LayerNorm over the full index head.
    """

    wq_b: torch.Tensor        # [n_index_heads * index_head_dim, q_lora]
    wk: torch.Tensor          # [index_head_dim, hidden]
    k_norm_weight: torch.Tensor  # [index_head_dim] FP32
    k_norm_bias: torch.Tensor     # [index_head_dim] FP32
    weights_proj: torch.Tensor    # [n_index_heads, hidden] FP32

    @property
    def heads(self):
        return self.weights_proj.shape[0]

    @property
    def head_dim(self):
        return int(self.wk.shape[0])


@dataclass(frozen=True)
class AttentionWeights:
    """One layer's gated-MLA projections on this rank (see `shard.Hy4Sharding`)."""

    q_a_proj: torch.Tensor        # [q_lora, hidden] replicated
    q_a_norm: torch.Tensor        # [q_lora]
    q_b_proj: torch.Tensor        # [heads_local * qk_head_dim, q_lora]
    kv_a_proj: torch.Tensor       # [kv_lora + qk_rope, hidden] replicated
    kv_a_norm: torch.Tensor       # [kv_lora]
    kv_b_proj: torch.Tensor       # [heads_local * (qk_nope + v), kv_lora]
    o_proj: torch.Tensor          # [hidden, heads_local * v]
    gate_proj: torch.Tensor       # [heads_local * v, hidden] (`linear_gate`)
    sink: torch.Tensor            # [heads_local] FP32 (`learnable_sink_param`)
    indexer: IndexerWeights | None  # None on `shared` layers

    @property
    def heads(self):
        return self.sink.shape[0]


@dataclass
class Hy4LayerAttention:
    """Per-layer attention execution: projection, selection, sparse attention.

    `selection` is the layer's IndexCache cell (shared layers alias their owner
    full layer's buffers). `operator` wires the b12x sparse-MLA plan on CUDA and
    the torch-op reference on CPU; both surface the same provenance.
    """

    weights: AttentionWeights
    config: ModelConfig
    sharding: Hy4Sharding
    selection: "SelectionCell | None"
    device: torch.device
    _mla: object = field(default=None, repr=False)
    _indexer_plan: object = field(default=None, repr=False)

    def __post_init__(self):
        c = self.config.text
        self.hidden = self.config.hidden
        self.heads = self.weights.heads
        self.qk_nope = c["qk_nope_head_dim"]
        self.qk_rope = c["qk_rope_head_dim"]
        self.kv_lora = c["kv_lora_rank"]
        self.v_head = c["v_head_dim"]
        self.rope_theta = c["rope_parameters"]["rope_theta"]
        self.scaling = c["qk_head_dim"] ** -0.5  # rope_type default: no mscale
        # `kv_b_proj` rows are PER HEAD, `[key qk_nope; value v]` (reference
        # `expand_kv`, modeling_hy_v4_patched.py:431-434: `kv_b_proj(x).view(B,
        # S, heads, qk_nope + v)` then a dim=-1 split). The split is therefore
        # within each head's row block — slicing "all heads' keys then all
        # heads' values" only ever got head 0 right (consult3 finding B), and
        # contradicts `shard.kv_b_rows`, which slices whole-head blocks.
        per_head = self.weights.kv_b_proj.view(self.heads, self.qk_nope + self.v_head, self.kv_lora)
        self.absorb_wuk = per_head[:, :self.qk_nope]
        self.expand_wuv = per_head[:, self.qk_nope:]

    # --- projection --------------------------------------------------------

    def project(self, x, positions):
        """(q_resid, absorbed query [rows,heads,kv_lora+rope], latent [rows,kv_lora+rope]).

        The absorbed query is `q_nope @ W_UK` per head (512-wide at production
        geometry) concatenated with the rotated `q_rot` (64) — the 576-wide
        `sparse_mla` query of the GLM_NSA contract.
        """
        w = self.weights
        q_resid = rms_norm(x @ w.q_a_proj.T, w.q_a_norm, self.config.text["rms_norm_eps"])
        q = (q_resid @ w.q_b_proj.T).view(-1, self.heads, self.qk_nope + self.qk_rope)
        q = apply_interleaved_rope(q, positions, rope_dim=self.qk_rope, theta=self.rope_theta)
        compressed = x @ w.kv_a_proj.T
        kv_pass, k_rot = compressed[:, :self.kv_lora], compressed[:, self.kv_lora:]
        k_pass = rms_norm(kv_pass, w.kv_a_norm, self.config.text["rms_norm_eps"])
        k_rot = apply_interleaved_rope(k_rot, positions, rope_dim=self.qk_rope, theta=self.rope_theta)
        latent = torch.cat((k_pass, k_rot), dim=-1)  # [rows, kv_lora + rope] — the cache record
        # Absorb W_UK into the query: q_nope @ W_UK (per head) puts the nope
        # block in latent space, so scoring runs against the raw latent record.
        absorbed_nope = torch.einsum("rhi,hio->rho", q[..., :self.qk_nope], self.absorb_wuk)
        q_absorbed = torch.cat((absorbed_nope, q[..., self.qk_nope:]), dim=-1)
        return q_resid, q_absorbed, latent

    # --- selection ---------------------------------------------------------

    def prepare_index_keys(self, x, positions):
        """Project and prepare this layer's index keys.

        The `k_norm` runs before the nope/pe split and the interleaved RoPE
        applies at write time (`dsa_indexer.rope.prepare_indexer_keys`): the
        key cache stores post-rope keys and the scorer never rotates. The
        epsilon is `rms_norm_eps` (the decided convention), never b12x's 1e-6
        default.
        """
        iw = self.weights.indexer
        if iw is None:
            raise ValueError("shared layers prepare no index keys")
        k = (x @ iw.wk.T).float()
        k = layer_norm(k, iw.k_norm_weight, iw.k_norm_bias, self.config.text["rms_norm_eps"])
        return apply_interleaved_rope(k, positions, rope_dim=self.config.text["qk_rope_head_dim"],
                                     theta=self.rope_theta)

    def select(self, x, q_resid, positions, visible, key_table, slot_base):
        """Write this layer's top-k selection into its IndexCache cell.

        `full` layers score with the indexer (dsa_indexer on CUDA, the same
        logits contract in torch on CPU) and own the result; `shared` layers
        never write — their `selection.selected` resolves to the owner's live
        buffer. `visible` is each token's causal cache length and `slot_base`
        its row's first physical slot (top-k picks are within-row positions,
        stored as absolute slots). Returns the cell's `(selected, counts)`.
        """
        cell = self.selection
        if cell is None:
            raise ValueError("layer has no selection cell")
        if cell.kind == "shared":
            # No indexer run, no copy: the owner's buffers are bound verbatim.
            return cell.buffers(rows=x.shape[0])
        if self.weights.indexer is None:
            raise ValueError("full layer lacks indexer weights")
        scores = self.index_scores(x, q_resid, positions, visible, key_table, slot_base)
        topk = min(self.config.text["index_topk"], scores.shape[-1])
        indices = scores.topk(topk, dim=-1).indices.to(torch.int32)
        cell.write_selection(indices, visible, slot_base)
        return cell.buffers(rows=x.shape[0])

    def index_scores(self, x, q_resid, positions, visible, key_table, slot_base):
        """Dense causal indexer logits [rows, width] over the key cache.

        The reference contract (`HYV4Indexer.forward`, `dsa_indexer`'s
        `paged_decode_logits_reference`): per-head `relu(q_rot · k_rot)` scores
        combined with `weights_proj(x) * n_heads^-0.5 * head_dim^-0.5` (the
        weight projection consumes the layer input `x`, the query projection
        the `q_resid`), keys rotated at write time, queries at prepare time.
        `visible` is each token's causal cache length (within-row positions
        0 .. visible-1), `slot_base` the row's first physical slot.
        """
        iw = self.weights.indexer
        c = self.config.text
        n_heads, head_dim = iw.heads, iw.head_dim
        q = (q_resid @ iw.wq_b.T).view(-1, n_heads, head_dim)
        q = apply_interleaved_rope(q, positions, rope_dim=c["qk_rope_head_dim"], theta=self.rope_theta)
        width = int(visible.max())
        rows = int(q.shape[0])
        keys = key_table.new_zeros((rows, width, key_table.shape[-1]))
        for row in range(rows):
            live = int(visible[row])
            keys[row, :live] = key_table[int(slot_base[row]):int(slot_base[row]) + live]
        scores = torch.relu(torch.einsum("rhd,rwd->rhw", q.float(), keys.float()))
        weights = (x.float() @ iw.weights_proj.float().T) * (n_heads ** -0.5) * (head_dim ** -0.5)
        logits = torch.einsum("rh,rhw->rw", weights, scores)
        # Causal masking over the token's visible cache length (`visible` counts
        # the entries each query may attend to, within-row positions 0..visible-1).
        causal = torch.arange(width, device=self.device).reshape(1, width) < visible.reshape(-1, 1)
        return logits.masked_fill(~causal, float("-inf"))

    # --- attention ---------------------------------------------------------

    def attend(self, q_absorbed, latent_table, selected, counts, *, return_lse=True, cache_lengths=None):
        """Sparse gated MLA against the selected latent records.

        Wires `sparse_mla.plan -> bind -> run` (with `attention_sink`) on CUDA;
        the torch-op reference on CPU. Both fold the per-head sink exactly once
        and return a base-2 LSE with `sink_folded=True`. `cache_lengths` is each
        row's KV length (the `sparse_mla` `cache_lengths` operand); it defaults
        to the selection width.
        """
        sink = self.weights.sink
        if self.device.type == "cuda":
            output, lse, binding = self._attend_cuda(q_absorbed, latent_table, selected, counts,
                                                     cache_lengths=cache_lengths)
            provenance = SparseMLAProvenance.from_binding(binding)
        else:
            output, lse = self._attend_reference(q_absorbed, latent_table, selected, counts)
            provenance = SparseMLAProvenance(sink_folded=sink is not None)
        return SparseMLAOutput(output=output, lse=lse if return_lse else None, provenance=provenance)

    def _attend_reference(self, q_absorbed, latent_table, selected, counts):
        """The eager sparse-MLA contract in torch (sink column + folded LSE)."""
        rows, heads, width = selected.shape[0], self.heads, selected.shape[1]
        slots = selected.long().clamp_min(0)
        keys = latent_table[slots]                       # [rows, width, kv_lora + rope]
        valid = (selected >= 0) & (torch.arange(width, device=self.device).reshape(1, width)
                                   < counts.reshape(-1, 1))
        logits = torch.einsum("rhd,rwd->rhw", q_absorbed.float(), keys.float()) * self.scaling
        logits = logits.masked_fill(~valid.reshape(rows, 1, width), float("-inf"))
        sink = self.weights.sink.float().reshape(1, heads, 1)
        combined = torch.cat((logits, sink.expand(rows, heads, 1)), dim=-1)
        probs = torch.softmax(combined, dim=-1)
        values = keys[..., :self.kv_lora]                # the absorbed value is the latent itself
        out_latent = torch.einsum("rhw,rwd->rhd", probs[..., :-1].float(), values.float())
        # LSE in base 2, sink folded exactly once (logaddexp of token LSE and sink).
        lse = torch.logsumexp(combined, dim=-1) * (1.0 / torch.log(torch.tensor(2.0)))
        expanded = torch.einsum("rhd,hvd->rhv", out_latent, self.expand_wuv.float()).to(q_absorbed.dtype)
        return expanded, lse.float()

    def _attend_cuda(self, q_absorbed, latent_table, selected, counts, *, cache_lengths=None):
        """`sparse_mla.plan -> bind -> run` on CUDA with sink provenance."""
        from b12x.attention import sparse_mla

        rows = q_absorbed.shape[0]
        if self._mla is None:
            caps = sparse_mla.Caps(
                device=self.device, num_q_heads=self.heads, max_q_rows=rows,
                max_width=selected.shape[1], softmax_scale=self.scaling,
                head_dim=self.kv_lora + self.qk_rope, v_head_dim=self.kv_lora,
                kv_dtype=latent_table.dtype, model_type=sparse_mla.ModelType.GLM_NSA,
                cache_record_bytes=self.config.text["kv_lora_rank"] + self.config.text["qk_rope_head_dim"] + 80,
                mode="decode", return_lse=True, lse_scale="base2", has_attention_sink=True)
            self._mla = sparse_mla.plan(caps)
        if cache_lengths is None:
            cache_lengths = counts.new_full((rows,), selected.shape[1])
        binding = sparse_mla.bind(self._mla, scratch=self._scratch(), q=q_absorbed.to(torch.bfloat16),
                                  kv_cache=latent_table, selected_indices=selected,
                                  cache_lengths=cache_lengths,
                                  selected_lengths=counts, attention_sink=self.weights.sink)
        result = sparse_mla.run(binding)
        output, lse = result
        return output, lse, binding

    def _scratch(self):
        raise NotImplementedError("sparse-MLA scratch is sized by the b12x PreparationSession at prepare time")

    def gate_and_project(self, attn_output, x):
        """Elementwise per-head gate (`gated_mla`, `gating_type=elementwise`) then `o_proj`."""
        gate = torch.sigmoid(x @ self.weights.gate_proj.T.to(x.dtype))
        gated = attn_output * gate.view(attn_output.shape[0], self.heads, self.v_head)
        return gated.reshape(gated.shape[0], -1).contiguous() @ self.weights.o_proj.T


def rms_norm(x, weight, eps):
    """`HYV4RMSNorm`: FP32 statistics, one output rounding."""
    xf = x.float()
    out = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
    return (weight.float() * out).to(x.dtype)


def layer_norm(x, weight, bias, eps):
    """The indexer `k_norm` LayerNorm (FP32, over the full index head)."""
    xf = x.float()
    mean, var = xf.mean(-1, keepdim=True), xf.var(-1, keepdim=True, unbiased=False)
    out = (xf - mean) * torch.rsqrt(var + eps)
    return (out * weight.float().reshape(1, -1) + bias.float().reshape(1, -1))


@dataclass
class SelectionCell:
    """One layer's view of the shared-selection `IndexCache`.

    `kind`/`owner` follow `dsa_indexer.index_cache.hy4_indexer_layers`:
    `full` layers own `[max_q_rows, topk]` selected buffers and index-key
    pages; `shared` layers resolve to their owner's buffers by `data_ptr`
    (consult2 defect 15b: the hand-back runs once per group and stays live).
    """

    layer_id: int
    kind: str
    owner_id: int
    cache: object  # b12x.attention.dsa_indexer.index_cache.IndexCache

    def buffers(self, rows=None):
        """The owner's `(selected, counts)`; with `rows`, the live row views."""
        selected, counts = self.cache.selected(self.layer_id), self.cache.counts(self.layer_id)
        if rows is None:
            return selected, counts
        return selected[:rows], counts[:rows]

    def write_selection(self, indices, visible, slot_base):
        """Scatter the row-wise top-k into the owner's physical-slot buffer.

        `indices` are within-row causal positions; the stored slots are
        absolute (`+ slot_base`) and the buffer pads with -1 past each token's
        live count (the buffer is `topk` wide even when the sparse width is
        shorter). Masked tie-fill picks past the query's causal bound are
        dropped — the reference package's defect 16 (causal-only selections).
        """
        selected, counts = self.cache.selected(self.layer_id), self.cache.counts(self.layer_id)
        rows, width = indices.shape
        ranks = torch.arange(width, device=indices.device).reshape(1, width)
        live = torch.minimum(visible, torch.full_like(visible, width))
        in_row = indices < visible.reshape(-1, 1)
        within = torch.where((ranks < live.reshape(-1, 1)) & in_row, indices,
                             torch.full_like(indices, -1))
        slots = within + slot_base.reshape(-1, 1).to(within.dtype)
        slots = torch.where(within >= 0, slots, torch.full_like(slots, -1))
        target = selected[:rows]
        target.fill_(-1)
        target[:, :width].copy_(slots.to(target.dtype))
        counts[:rows].copy_(live.to(counts.dtype))

    def handback(self, merged, *, world_size, rank, prepared=None):
        """Cross-rank selection hand-back into the owner's buffers.

        Under context parallelism the merged global top-k lands in GLOBAL
        positions; `IndexCache.handback` (backed by
        `dsa_indexer.selection_handback.PreparedSelectionHandback`) recovers
        this rank's local selection once per group.
        """
        if prepared is not None:
            selected, counts = prepared.run(merged, output=self.cache.selected(self.owner_id),
                                            counts=self.cache.counts(self.owner_id))
            return selected, counts
        return self.cache.handback(self.layer_id, merged, world_size=world_size, rank=rank)
