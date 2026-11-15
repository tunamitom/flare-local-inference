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
class PackedInputProjections:
    """One GEMM for every projection reading the same layer input (packed at load).

    The consult7 §5 bake-in "packed same-input projections at load time":
    `q_a_proj`, `kv_a_proj`, `linear_gate` and the indexer `wk` all consume the
    layer input `x`, so their weight rows concatenate into ONE matrix and the
    forward runs one `x @ packed.T` instead of four reads of `x`. Pure layout:
    each output row is the same dot product as the separate GEMM. Built by
    `pack_same_input_projections` at weight-load time.
    """

    packed: torch.Tensor    # [q_lora + kv_lora+rope + heads*v + index_head_dim, hidden]
    slices: dict            # name -> (first, last) rows of `packed`

    @classmethod
    def build(cls, weights):
        parts, slices, first = [], {}, 0
        items = [("q_a_proj", weights.q_a_proj), ("kv_a_proj", weights.kv_a_proj),
                 ("linear_gate", weights.gate_proj)]
        if weights.indexer is not None:
            items.append(("indexer_wk", weights.indexer.wk))
        for name, tensor in items:
            parts.append(tensor)
            slices[name] = (first, first + tensor.shape[0])
            first += tensor.shape[0]
        return cls(packed=torch.cat(parts, dim=0).contiguous(), slices=slices)

    def rows(self, name):
        first, last = self.slices[name]
        return self.packed[first:last]


@dataclass(frozen=True)
class LayerProjections:
    """One layer call's projections out of the packed same-input GEMM."""

    q_resid: torch.Tensor
    q: torch.Tensor            # absorbed or unabsorbed query, per the wiring
    latent: torch.Tensor
    index_keys: torch.Tensor | None
    gate_raw: torch.Tensor     # `x @ linear_gate.T`, pre-sigmoid


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
    packed_input: PackedInputProjections | None = None  # same-input pack (load time)

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
    _mla_plans: dict = field(default_factory=dict, repr=False)
    _mla_scratch: dict = field(default_factory=dict, repr=False)
    _mla_scratch_lanes: dict = field(default_factory=dict, repr=False)   # consult18 D
    _plan_book: object = field(default=None, repr=False)
    _sink_folds: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        c = self.config.text
        self._active_call_shape = None
        self._packed_gate = None
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
        # `kv_b_proj` is REPLICATED (consult8 §5's `kv_b` two-wirings): the
        # tensors are all heads here, and the DECODE wiring reads only this
        # rank's head rows (`absorb_queries` / `expand_output` slice
        # `sharding.head_range` — decode's weight reads do not change).
        per_head = self.weights.kv_b_proj.view(-1, self.qk_nope + self.v_head, self.kv_lora)
        self.absorb_wuk = per_head[:, :self.qk_nope]
        self.expand_wuv = per_head[:, self.qk_nope:]

    # --- projection --------------------------------------------------------

    def project(self, x, positions, *, absorb=True):
        """(q_resid, absorbed query [rows,heads,kv_lora+rope], latent [rows,kv_lora+rope]).

        The absorbed query is `q_nope @ W_UK` per head (512-wide at production
        geometry) concatenated with the rotated `q_rot` (64) — the 576-wide
        `sparse_mla` query of the GLM_NSA contract. `absorb=False` returns
        the UNABSORBED query `[rows, heads, qk_nope + qk_rope]` instead — the
        prefill graph wiring gathers unabsorbed queries and absorbs + expands
        LOCALLY (consult8 §5's `kv_b` two-wirings); decode graphs keep the
        owner-side absorb here and read no extra weights.
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
        if not absorb:
            return q_resid, q, latent
        # Absorb W_UK into the query: q_nope @ W_UK (per head) puts the nope
        # block in latent space, so scoring runs against the raw latent record.
        # Owner-side: this rank's head shard only (the decode wiring — the
        # weight reads stay this rank's `kv_b_proj` rows).
        return q_resid, self.absorb_queries(q), latent

    def project_packed(self, x, positions, *, absorb=True, with_index_keys=False):
        """All same-input projections of one layer call from ONE packed weight.

        Packed at load (`AttentionWeights.packed_input`): `q_a_proj` +
        `kv_a_proj` + `linear_gate` + the indexer `wk` live in ONE contiguous
        matrix the forward reads. The per-projection GEMMs run over the packed
        ROWS (`x @ packed.rows(name).T`) — same shapes, same values, bitwise
        the same results as the separate weights: the golden content gates
        hash selections and must not drift, and fusing the launches is not the
        lever at one row anyway (consult7 §6: launch-count hunting). Without
        packing this falls back to the loose weights (identical math). The
        `gate_raw` output is stashed for `gate_and_project` (which keeps its
        two-argument contract) keyed on the input's storage.
        """
        w = self.weights
        packed = w.packed_input
        if packed is None:
            q_resid, q, latent = self.project(x, positions, absorb=absorb)
            index_keys = self.prepare_index_keys(x, positions) if with_index_keys else None
            gate_raw = x @ w.gate_proj.T.to(x.dtype)
            self._packed_gate = (x.data_ptr(), gate_raw)
            return LayerProjections(q_resid, q, latent, index_keys, gate_raw)
        q_a_raw = x @ packed.rows("q_a_proj").T
        kv_raw = x @ packed.rows("kv_a_proj").T
        gate_raw = x @ packed.rows("linear_gate").T.to(x.dtype)
        q_resid = rms_norm(q_a_raw, w.q_a_norm, self.config.text["rms_norm_eps"])
        q = (q_resid @ w.q_b_proj.T).view(-1, self.heads, self.qk_nope + self.qk_rope)
        q = apply_interleaved_rope(q, positions, rope_dim=self.qk_rope, theta=self.rope_theta)
        kv_pass, k_rot = kv_raw[:, :self.kv_lora], kv_raw[:, self.kv_lora:]
        k_pass = rms_norm(kv_pass, w.kv_a_norm, self.config.text["rms_norm_eps"])
        k_rot = apply_interleaved_rope(k_rot, positions, rope_dim=self.qk_rope, theta=self.rope_theta)
        latent = torch.cat((k_pass, k_rot), dim=-1)
        q_out = q if not absorb else self.absorb_queries(q)
        index_keys = None
        if with_index_keys:
            # Through `prepare_index_keys` — the single key-prep entry point
            # (a wrong-rotation mutant wraps this method and must stay live).
            index_keys = self.prepare_index_keys(x, positions)
        self._packed_gate = (x.data_ptr(), gate_raw)
        return LayerProjections(q_resid, q_out, latent, index_keys, gate_raw)

    def absorb_queries(self, q_unabsorbed):
        """`q_nope @ W_UK` per head over the query's head axis (the local absorb).

        The prefill wiring gathers UNABSORBED queries and runs this locally
        for ALL gathered heads (with `kv_b_proj` replicated — consult8 §5);
        `project(absorb=True)` runs it owner-side for this rank's head shard.
        """
        heads = int(q_unabsorbed.shape[1])
        first, last = self.sharding.head_range
        if heads == self.heads:
            wuk = self.absorb_wuk[first:last]
        elif heads == self.absorb_wuk.shape[0]:
            wuk = self.absorb_wuk
        else:
            raise ValueError(
                f"absorb_queries: {heads} heads is neither this rank's shard "
                f"({self.heads}) nor the full head count ({self.absorb_wuk.shape[0]})")
        absorbed_nope = torch.einsum("rhi,hio->rho", q_unabsorbed[..., :self.qk_nope], wuk)
        return torch.cat((absorbed_nope, q_unabsorbed[..., self.qk_nope:]), dim=-1)

    def expand_output(self, out, *, local=True):
        """`W_UV` expansion of a latent-space attention output (per head).

        `local=True` expands this rank's contiguous head shard (the
        post-merge output); `local=False` expands every head (the prefill
        wiring's local expand over gathered heads).
        """
        first, last = self.sharding.head_range
        wuv = self.expand_wuv[first:last] if local else self.expand_wuv
        # fp32 accumulation whatever the partial's dtype: the real sparse-MLA
        # wire carries BF16 partials (the DCP merge returns a BF16 head shard),
        # while the CPU reference partials are fp32.
        return torch.einsum("rhd,hvd->rhv", out.float(), wuv.float()).to(out.dtype)

    def expand_partials(self, out):
        """`W_UV` over EVERY head, before the exchange (consult17, lever 9).

        The merge is a per (row, head) convex combination and the sink fold a
        per (row, head) scale, so expanding the all-head partial here and
        merging the expanded values is the same map as merging latents and
        expanding the shard afterwards — but the wire then carries `v_head_dim`
        values per head instead of `kv_lora_rank` (256 instead of 512). Needs
        the whole `kv_b_proj` on every rank (the prefill wiring has it). BF16
        partials expand in BF16 (FP32 accumulate inside the GEMM, rounded once
        to the wire dtype); the FP32 twin expands in FP32.
        """
        wuv = self.expand_wuv
        if out.dtype == torch.bfloat16 and wuv.dtype == torch.bfloat16:
            return torch.einsum("rhd,hvd->rhv", out, wuv)
        return torch.einsum("rhd,hvd->rhv", out.float(), wuv.float()).to(out.dtype)

    # --- prepared plans (no JIT at run time) -------------------------------
    def bind_plan_book(self, book):
        """Attach the rank's `KernelPlanBook` (from `wire_operators`)."""
        self._plan_book = book

    def prepare_plans(self, shapes, *, device=None, sink_variants=(True, False),
                      scratch_pool=None, expanded_partials=False, lane_pools=()):
        """Bind one `sparse_mla` plan per call shape, plus the sink fold.

        The one-row decode plan is NOT the verification-rows plan (consult7
        §5: "a one-row decode graph distinct from the verification graph") —
        each `CallShape` gets its own `sparse_mla.plan` (`Caps.mode` =
        shape.mla_mode: decode / verify / extend). On CUDA these are the real
        b12x plans; on CPU the book records the bound shape (the torch-op
        reference needs no lower). Once a plan book is bound, `attend` refuses
        a call shape with no prepared plan — there is no JIT at run time.

        Each shape gets a plan per BIND variant: the world-1 variant (the sink
        bound, folded in kernel) and the CP-partial variant (sink UNBOUND —
        D2/D6: the merge has no sink input and the one `PreparedSinkFold`
        folds after it). `sparse_mla.bind` requires `attention_sink` presence
        to match `Caps.has_attention_sink`, so the variants are distinct
        plans with distinct scratch. `sink_variants` selects which to prepare
        (default: both); a CP serve that never runs the world-1 call prepares
        `(False,)` and adds the bound variant only at its graph-capture shapes
        — the plan set is a measured memory term and the full two-variant book
        at 14 call shapes exhausted the card on the first h4 boot.
        """
        from .wiring import KernelPlanBook

        self._plan_book = self._plan_book or KernelPlanBook()
        # One shared plan per (bind variant, call-shape name): `sparse_mla`
        # plans are MAX-capacity (`_validate_q` admits q rows <= max_total_q),
        # so a plan built at the group's largest geometry serves every smaller
        # call — the per-shape plan/scratch duplication is what exhausted the
        # card at 14 shapes x 64-head partials. Heads follow the variant: the
        # world-1 call is this rank's shard, the CP partial is total-head.
        groups = {}
        for shape in shapes:
            for bound in sink_variants:
                groups.setdefault((bool(bound), shape.name), []).append(shape)
        for (bound, name), group_shapes in groups.items():
            eff = self.weights.sink is not None and bool(bound)
            heads = self.heads if bound else int(self.config.text["num_attention_heads"])
            proto = group_shapes[0]
            plan = self._make_mla_plan(
                proto, has_sink=eff, num_q_heads=heads,
                max_q_rows=max(s.rows * s.tokens for s in group_shapes))
            if self.device.type != "cuda":
                scratch = None
            elif scratch_pool is None:
                scratch = self._allocate_scratch(plan)
            else:
                spec, = plan.scratch_specs()
                pool_key = (bool(bound), name, heads, tuple(spec.shape), spec.dtype)
                scratch = scratch_pool.get(pool_key)
                if scratch is None:
                    scratch = scratch_pool[pool_key] = self._allocate_scratch(plan)
            # consult18 stage D: every extra lane gets its own scratch (from
            # its own shared pool): sparse-MLA outputs alias their scratch, so
            # two in-flight halves must never share one.
            lane_scratch = []
            for index, pool in enumerate(lane_pools, start=1):
                if self.device.type != "cuda":
                    lane_scratch.append((index, None))
                    continue
                spec, = plan.scratch_specs()
                pool_key = (bool(bound), name, heads, tuple(spec.shape), spec.dtype)
                if pool_key not in pool:
                    pool[pool_key] = self._allocate_scratch(plan)
                lane_scratch.append((index, pool[pool_key]))
            for shape in group_shapes:
                self._mla_plans[(shape.name, shape.rows, shape.tokens, eff)] = plan
                if scratch is not None:
                    self._mla_scratch[(shape.name, shape.rows, shape.tokens, eff)] = scratch
                for index, extra in lane_scratch:
                    if extra is not None:
                        self._mla_scratch_lanes.setdefault(index, {})[
                            (shape.name, shape.rows, shape.tokens, eff)] = extra
                if bound == sink_variants[0]:
                    self._plan_book.bind(shape, "sparse_mla", plan)
        for shape in shapes:
            if self.weights.sink is not None:
                # The fold geometry is the call shape's flattened query rows:
                # [rows*tokens, heads, kv_lora] out + [rows*tokens, heads] LSE
                # (placeholder operands pin the plan).
                # With expanded partials (consult17) a prefill graph folds
                # the merged EXPANDED values: `v_head_dim` wide, not latent.
                batch = shape.rows * shape.tokens
                width = (int(self.expand_wuv.shape[1])
                         if expanded_partials and shape.is_prefill_graph else self.kv_lora)
                out = self.weights.sink.new_zeros(
                    (batch, self.heads, width), dtype=self.weights.q_b_proj.dtype)
                lse = self.weights.sink.new_zeros((batch, self.heads))
                fold = self._make_sink_fold(out, lse)
                self._sink_folds[(shape.name, shape.rows, shape.tokens)] = fold
                self._plan_book.bind(shape, "sink_fold", fold)
        return self

    def _make_mla_plan(self, shape, *, has_sink=True, num_q_heads=None, max_q_rows=None):
        if self.device.type != "cuda":
            return ("reference-sparse-mla", shape.name, shape.rows, shape.tokens, shape.mla_mode,
                    has_sink, num_q_heads, max_q_rows)
        from b12x.attention import sparse_mla

        caps = sparse_mla.Caps(
            device=self.device,
            # The world-1 call runs this rank's head shard; the CP partial (D2)
            # runs the head all-gather's ALL-HEAD output, so its plan and
            # scratch are total-head (`num_q_heads`).
            num_q_heads=self.heads if num_q_heads is None else int(num_q_heads),
            # Queries flatten to rows * tokens (the sink-fold geometry's
            # `batch`); `max_q_rows` may cover a whole call-shape family —
            # `sparse_mla.bind` admits any call at or under the planned rows.
            max_q_rows=shape.rows * shape.tokens if max_q_rows is None else int(max_q_rows),
            max_width=self.config.text["index_topk"],
            softmax_scale=self.scaling,
            head_dim=self.kv_lora + self.qk_rope, v_head_dim=self.kv_lora,
            # The engine contract is the fused 656-byte FP8 record (uint8 byte
            # record: 512 E4M3 nope + 16 B inline fp32 scales + 128 B raw BF16
            # RoPE) — the record `concat_and_cache_glm_next_mla` writes.
            kv_dtype=torch.uint8, model_type=sparse_mla.ModelType.GLM_NSA,
            cache_record_bytes=self.kv_lora + self.qk_rope + 80,
            mode=shape.mla_mode, return_lse=True, lse_scale="base2",
            # `sparse_mla.bind` requires the sink's presence at bind time to
            # match the caps: the world-1 variant folds it in kernel, the CP
            # partial (D2/D6) binds without it and folds once post-merge.
            has_attention_sink=bool(has_sink))
        return sparse_mla.plan(caps)

    def _mla_plan_for(self, call_shape, *, sink_bound=True):
        """The prepared plan for `call_shape` at this sink-binding mode."""
        if call_shape is not None:
            key = (call_shape.name, call_shape.rows, call_shape.tokens, bool(sink_bound))
            if key in self._mla_plans:
                return self._mla_plans[key]
        if self._plan_book is not None:
            if call_shape is None:
                raise RuntimeError(
                    "a prepared plan book is bound: attend must state its call "
                    "shape (decode one-row, verify rows or prefill) — there is "
                    "no JIT at run time")
            raise RuntimeError(
                f"no prepared sparse_mla plan for call shape {call_shape.name} "
                f"(rows={call_shape.rows}, tokens={call_shape.tokens}); wire_operators "
                "binds one plan per call shape — there is no JIT at run time")
        return self._mla  # legacy: built lazily in `_attend_cuda`

    def _make_sink_fold(self, out, lse):
        """The ONE post-merge sink fold's prepared plan (`prepare_sink_fold`)."""
        from b12x.comm.pcie.api import prepare_sink_fold

        return prepare_sink_fold(out=out, lse=lse, sink=self.weights.sink, lse_base="base2")

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
        # The packed same-input weight supplies `wk` when present (same rows,
        # bitwise the same GEMM); the loose tensor otherwise.
        packed = self.weights.packed_input
        wk = iw.wk if packed is None else packed.rows("indexer_wk")
        k = (x @ wk.T).float()
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
        This is the world-1 path (queries and records on one rank); under
        context parallelism the decoder runs `local_candidates` -> the
        selection exchange -> `SelectionCell.handback` instead (D3).
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

    def local_candidates(self, x, q_resid, positions, visible, key_table, slot_base, *, world, rank):
        """This rank's local top-k candidates over its OWN key table (D3).

        Returns `(candidate_scores [rows, topk] FP32, candidate_positions
        [rows, topk] int32 GLOBAL)` — exactly `topk` wide, the
        `merge_tiled_topk_candidates` candidate contract — padded `-inf` / `-1`
        past each token's owned live keys. Score column `j` is the global
        position `j * world + rank` (the interleaved ownership rule, amendment
        2), so the candidate merge compares global positions across ranks; a
        candidate whose score was masked to `-inf` is offered as `-1` and
        drops in the hand-back (never a not-yet-visible record).
        """
        if self.weights.indexer is None:
            raise ValueError("full layer lacks indexer weights")
        topk = self.config.text["index_topk"]
        # Bound the [rows, width] logits and the top-k's sort copies (about
        # 20 B per query row per owned key): above the budget, score the rows
        # in blocks. Each block is this same method on a row slice.
        count = int(visible.shape[0])
        owned = torch.div((visible.long() - rank).clamp_min(0) + world - 1, world,
                          rounding_mode="floor")
        widest = max(int(owned.max()) if count else 0, topk)
        block = max(1, self.INDEX_CANDIDATE_BLOCK_BYTES // (widest * 20))
        if count > block:
            parts = [self.local_candidates(
                x[start:start + block], q_resid[start:start + block],
                positions[start:start + block], visible[start:start + block], key_table,
                slot_base[start:start + block], world=world, rank=rank)
                for start in range(0, count, block)]
            return (torch.cat([values for values, _ in parts]),
                    torch.cat([picked for _, picked in parts]))
        scores = self.index_scores(x, q_resid, positions, visible, key_table, slot_base,
                                  world=world, rank=rank)
        rows, width = scores.shape
        if width < topk:
            scores = torch.cat((scores, scores.new_full((rows, topk - width), float("-inf"))), dim=-1)
        columns = torch.cat((torch.arange(width, device=self.device, dtype=torch.int64) * world + rank,
                             torch.full((max(topk - width, 0),), -1, device=self.device,
                                        dtype=torch.int64)))
        values, indices = scores.topk(topk, dim=-1)
        picked = torch.where(torch.isfinite(values), columns[indices],
                             torch.full_like(indices, -1, dtype=torch.int64))
        return values.to(torch.float32), picked.to(torch.int32)

    # Byte budgets of the torch indexer scoring (consult14): one score block
    # `[block_rows, heads, live]` FP32, and one candidate block's logits plus
    # top-k copies. They bound the request-time transient at any context.
    INDEX_SCORE_BLOCK_BYTES = 256 << 20
    INDEX_CANDIDATE_BLOCK_BYTES = 256 << 20

    def index_scores(self, x, q_resid, positions, visible, key_table, slot_base, *, world=1, rank=0):
        """Dense causal indexer logits [rows, width] over the key cache.

        The reference contract (`HYV4Indexer.forward`, `dsa_indexer`'s
        `paged_decode_logits_reference`): per-head `relu(q_rot · k_rot)` scores
        combined with `weights_proj(x) * n_heads^-0.5 * head_dim^-0.5` (the
        weight projection consumes the layer input `x`, the query projection
        the `q_resid`), keys rotated at write time, queries at prepare time.
        `visible` is each token's causal cache length (within-row GLOBAL
        positions 0 .. visible-1), `slot_base` the row's first physical slot.

        Context parallelism (world > 1): the key table holds only the rank's
        OWNED keys — global position `p` at local index `p // world`
        (amendment 2) — so the score row covers the token's owned live keys
        (`p < visible`, `p % world == rank`) and score column `j` is global
        position `j * world + rank`. The indexer is replicated and rank
        invariant (never head-sharded — §2), so the local scores equal the
        single-rank scores of the same positions.
        """
        iw = self.weights.indexer
        c = self.config.text
        n_heads, head_dim = iw.heads, iw.head_dim
        q = (q_resid @ iw.wq_b.T).view(-1, n_heads, head_dim)
        q = apply_interleaved_rope(q, positions, rope_dim=c["qk_rope_head_dim"], theta=self.rope_theta)
        # Owned live keys per query token (world 1: every position is owned).
        owned = torch.div((visible.long() - rank).clamp_min(0) + world - 1, world,
                          rounding_mode="floor")
        width = int(owned.max()) if owned.numel() else 0
        rows = int(q.shape[0])
        q = q.float()
        weights = (x.float() @ iw.weights_proj.float().T) * (n_heads ** -0.5) * (head_dim ** -0.5)
        # No `[rows, width, key_dim]` key copy: the tokens of one request row
        # share one `[live, key_dim]` key segment, scored in row blocks of at
        # most `INDEX_SCORE_BLOCK_BYTES` (the copy was 1 KiB per query row per
        # owned key — 4.0 GiB at a 32k context, the consult14 request OOM).
        # Columns past a token's own live keys are masked below, as before.
        logits = q.new_zeros((rows, width))
        for base in torch.unique(slot_base).tolist() if width else ():
            members = (slot_base == base).nonzero(as_tuple=True)[0]
            live = int(owned[members].max())
            if not live:
                continue
            keys = key_table[base:base + live].float()
            block = max(1, self.INDEX_SCORE_BLOCK_BYTES // (n_heads * live * 4))
            for start in range(0, int(members.numel()), block):
                part = members[start:start + block]
                scores = torch.relu_(q[part].reshape(-1, head_dim) @ keys.T).view(-1, n_heads, live)
                logits[part, :live] = torch.einsum("rh,rhw->rw", weights[part], scores)
        # Causal masking over the token's owned live keys (world 1: `owned` is
        # exactly the visible cache length).
        causal = torch.arange(width, device=self.device).reshape(1, width) < owned.reshape(-1, 1)
        return logits.masked_fill(~causal, float("-inf"))

    # --- attention ---------------------------------------------------------

    def attend(self, q_absorbed, latent_table, selected, counts, *, return_lse=True,
               cache_lengths=None, fold_sink=True, expand=True, call_shape=None):
        """Sparse gated MLA against the selected latent records.

        Wires `sparse_mla.plan -> bind -> run` (with `attention_sink`) on CUDA;
        the torch-op reference on CPU. Both surface the same provenance.
        `cache_lengths` is each row's KV length (the `sparse_mla` `cache_lengths`
        operand); it defaults to the selection width. `call_shape` selects the
        PREPARED plan (one per call shape — the one-row decode plan is not the
        verification-rows plan); with a plan book bound, an unprepared shape
        raises (no JIT at run time).

        World 1 (`fold_sink=True`, `expand=True`): the per-head sink folds per
        call and the output expands through `W_UV` — the eager path, the
        simulation's ground truth, unchanged. The CP partial (D2) calls with
        `fold_sink=False, expand=False`: the sink stays UNBOUND (provenance
        `sink_folded=False`) and the output stays in the wide latent, because
        `kv_b_proj` is head-sharded and only this rank's head shard can expand
        — the partials exchange in latent space and the single post-merge
        `fold_sink` + `W_UV` finish the shard (`q_b_proj`/`kv_b_proj`
        replication, the 256-wide partial variant, is the later measured
        lever). The merge is per (row, head) convex and `W_UV` per head, so
        expanding before or after the merge is the same map.
        """
        sink = self.weights.sink
        sink_bound = sink is not None and fold_sink
        if self.device.type == "cuda":
            output, lse, binding = self._attend_cuda(q_absorbed, latent_table, selected, counts,
                                                     fold_sink=fold_sink, cache_lengths=cache_lengths,
                                                     call_shape=call_shape)
            provenance = SparseMLAProvenance.from_binding(binding)
        else:
            plan = self._mla_plan_for(call_shape, sink_bound=sink_bound)  # the no-JIT check runs on CPU too
            output, lse = self._attend_reference(q_absorbed, latent_table, selected, counts,
                                                 fold_sink=fold_sink)
            provenance = SparseMLAProvenance(sink_folded=bool(fold_sink and sink is not None))
        if expand:
            output = torch.einsum("rhd,hvd->rhv", output.float(), self.expand_wuv.float()).to(q_absorbed.dtype)
        return SparseMLAOutput(output=output, lse=lse if return_lse else None, provenance=provenance)

    def fold_sink(self, out, lse, *, call_shape=None):
        """The ONE post-merge sink fold (D2): `out *= sigmoid(lse - sink); lse = logaddexp(lse, sink)`.

        `PreparedSinkFold` (`b12x.comm.pcie.prepare_sink_fold`) on this rank's
        head-shard sink and the base-2 merged LSE, run exactly once after
        `Collectives.lse_reduce`. The prepared plan is pinned per call shape
        at `prepare_plans` time (CUDA-graph safe, allocation-free `run`);
        folding per shard instead is the double-fold defect the merge's
        `partials_sink_folded` guard rejects (gate c's sink-per-rank mutant).
        """
        prepared = None
        call_shape = call_shape if call_shape is not None else self._active_call_shape
        if call_shape is not None:
            prepared = self._sink_folds.get((call_shape.name, call_shape.rows, call_shape.tokens))
        if prepared is not None and (int(prepared.batch), int(prepared.heads), int(prepared.head_dim)) != \
                (int(out.shape[0]), int(out.shape[1]), int(out.shape[2])):
            prepared = None  # stale geometry (an unpinned call) — rebind below
        if prepared is None:
            if self._plan_book is not None:
                raise RuntimeError(
                    "no prepared sink-fold plan for this call shape; prepare_plans "
                    "binds one per call shape — there is no JIT at run time")
            prepared = self._sink_folds.get("lazy")
            if prepared is None or (int(prepared.batch), int(prepared.heads), int(prepared.head_dim)) != \
                    (int(out.shape[0]), int(out.shape[1]), int(out.shape[2])):
                prepared = self._make_sink_fold(out, lse)
                self._sink_folds["lazy"] = prepared
        return prepared.run(out, lse, self.weights.sink)

    def _attend_reference(self, q_absorbed, latent_table, selected, counts, *, fold_sink=True):
        """The eager sparse-MLA contract in torch (latent-space output)."""
        rows, heads, width = selected.shape[0], int(q_absorbed.shape[1]), selected.shape[1]
        slots = selected.long().clamp_min(0)
        keys = latent_table[slots]                       # [rows, width, kv_lora + rope]
        valid = (selected >= 0) & (torch.arange(width, device=self.device).reshape(1, width)
                                   < counts.reshape(-1, 1))
        logits = torch.einsum("rhd,rwd->rhw", q_absorbed.float(), keys.float()) * self.scaling
        logits = logits.masked_fill(~valid.reshape(rows, 1, width), float("-inf"))
        values = keys[..., :self.kv_lora]                # the absorbed value is the latent itself
        ln2 = 1.0 / torch.log(torch.tensor(2.0))
        if fold_sink:
            sink = self.weights.sink.float().reshape(1, heads, 1)
            combined = torch.cat((logits, sink.expand(rows, heads, 1)), dim=-1)
            probs = torch.softmax(combined, dim=-1)
            out_latent = torch.einsum("rhw,rwd->rhd", probs[..., :-1].float(), values.float())
            # LSE in base 2, sink folded exactly once (logaddexp of token LSE and sink).
            lse = torch.logsumexp(combined, dim=-1) * ln2
        else:
            # The CP partial: sink UNBOUND over this rank's own record shard.
            probs = torch.softmax(logits, dim=-1)
            out_latent = torch.einsum("rhw,rwd->rhd", probs.float(), values.float())
            lse = torch.logsumexp(logits, dim=-1) * ln2
        return out_latent, lse.float()

    def _attend_cuda(self, q_absorbed, latent_table, selected, counts, *, fold_sink=True,
                     cache_lengths=None, call_shape=None):
        """`sparse_mla.plan -> bind -> run` on CUDA with sink provenance.

        `fold_sink=False` (the CP partial) binds WITHOUT `attention_sink`, so
        `Binding.sink_folded` is False and the post-merge `PreparedSinkFold`
        owns the one fold. The kernel emits the latent-space value output
        (`Caps.v_head_dim == kv_lora_rank`), which `attend` expands. The plan
        is the call shape's PREPARED plan (bound by `prepare_plans` — one per
        shape, no JIT at run time); without a plan book this keeps the legacy
        lazy single plan.
        """
        from b12x.attention import sparse_mla

        sink_bound = self.weights.sink is not None and fold_sink
        rows = q_absorbed.shape[0]
        record_bytes = self.config.text["kv_lora_rank"] + self.config.text["qk_rope_head_dim"] + 80
        # The kernel reads the FUSED 656-byte FP8 record (`Hy4Cache` with
        # `fused_records=True` writes it through `concat_and_cache_glm_next_mla`);
        # a decoded BF16 table is the device-agnostic twin and cannot bind.
        if (latent_table is None or latent_table.dtype != torch.uint8
                or int(latent_table.shape[-1]) != record_bytes):
            got = "None" if latent_table is None else f"{latent_table.dtype}, width {int(latent_table.shape[-1])}"
            raise ValueError(
                f"the CUDA sparse-MLA bind needs the fused {record_bytes}-byte FP8 record "
                f"cache (uint8) written by concat_and_cache_glm_next_mla — got {got}; "
                "allocate the cache with fused_records=True")
        plan = self._mla_plan_for(call_shape, sink_bound=sink_bound)
        key = ((call_shape.name, call_shape.rows, call_shape.tokens, bool(sink_bound))
               if call_shape is not None else "lazy")
        if plan is None:
            caps = sparse_mla.Caps(
                device=self.device, num_q_heads=int(q_absorbed.shape[1]), max_q_rows=rows,
                max_width=selected.shape[1], softmax_scale=self.scaling,
                head_dim=self.kv_lora + self.qk_rope, v_head_dim=self.kv_lora,
                kv_dtype=torch.uint8, model_type=sparse_mla.ModelType.GLM_NSA,
                cache_record_bytes=record_bytes,
                mode="decode", return_lse=True, lse_scale="base2",
                has_attention_sink=bool(sink_bound))
            plan = sparse_mla.plan(caps)
            self._mla = plan
            self._mla_scratch["lazy"] = self._allocate_scratch(plan)
        from .lanes import lane

        current = lane()
        if current:
            if current not in self._mla_scratch_lanes or key not in self._mla_scratch_lanes[current]:
                raise RuntimeError(f"no sparse-MLA scratch for lane {current} at {key}")
            scratch = self._mla_scratch_lanes[current][key]
        else:
            scratch = self._mla_scratch.get(key)
        if scratch is None and plan is self._mla:
            scratch = self._mla_scratch.get("lazy")
        if scratch is None:
            raise RuntimeError(
                "no prepared sparse-MLA scratch for this call shape; prepare_plans "
                "allocates one per call shape — there is no JIT at run time")
        if cache_lengths is None:
            cache_lengths = counts.new_full((rows,), selected.shape[1])
        binding = sparse_mla.bind(plan, scratch=scratch, q=q_absorbed.to(torch.bfloat16),
                                  # The Hy4 cache is the flat physical-slot table
                                  # (`output_index_space="physical"`); the kernel
                                  # contract takes it as one batch of rows.
                                  kv_cache=latent_table.unsqueeze(0) if latent_table.ndim == 2
                                  else latent_table,
                                  selected_indices=selected,
                                  cache_lengths=cache_lengths,
                                  selected_lengths=counts,
                                  attention_sink=self.weights.sink if sink_bound else None)
        result = sparse_mla.run(binding)
        output, lse = result
        return output, lse, binding

    @staticmethod
    def _allocate_scratch(plan):
        """One caller-owned scratch buffer per prepared plan (`plan.scratch_specs`).

        The b12x sparse-MLA contract is eager plan -> bind -> kernel over a
        caller-owned scratch tensor sized at prepare time (`b12x.attention.
        sparse_mla._scratch`); nothing allocates inside a step or a capture.
        """
        specs = plan.scratch_specs()
        if len(specs) != 1:
            raise ValueError(f"sparse-MLA scratch expects one buffer, got {len(specs)}")
        spec = specs[0]
        return torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)

    def gate_and_project(self, attn_output, x):
        """Elementwise per-head gate (`gated_mla`, `gating_type=elementwise`) then `o_proj`.

        The gate projection comes from the packed same-input GEMM when this
        call's `project_packed` stashed it (same `x` storage); otherwise it is
        computed here as always. The two-argument contract is load-bearing (a
        gate-after-o_proj mutant replaces this method), so the packed result
        travels out of band.
        """
        gate_raw = None
        stashed = getattr(self, "_packed_gate", None)
        if stashed is not None and stashed[0] == x.data_ptr():
            gate_raw = stashed[1]
            self._packed_gate = None
        if gate_raw is None:
            gate_raw = x @ self.weights.gate_proj.T.to(x.dtype)
        gate = torch.sigmoid(gate_raw)
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
    # consult18 stage D: one more IndexCache per extra execution lane, so a
    # half's selection (kept live through its shared layers) is never
    # overwritten by the other half's.
    lane_caches: tuple = ()

    def _cache(self):
        from .lanes import lane

        current = lane()
        if current == 0:
            return self.cache
        if current > len(self.lane_caches):
            raise RuntimeError(f"selection cell {self.layer_id} has no buffers for lane {current}")
        return self.lane_caches[current - 1]

    def buffers(self, rows=None):
        """The owner's `(selected, counts)`; with `rows`, the live row views."""
        selected, counts = self._cache().selected(self.layer_id), self._cache().counts(self.layer_id)
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
        selected, counts = self._cache().selected(self.layer_id), self._cache().counts(self.layer_id)
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

    def fill_visible(self, visible, slot_base, *, world=1, rank=0):
        """The short-context bypass selection: every VISIBLE OWNED key, no indexer.

        At context <= top-k the global top-k selection IS the whole visible
        prefix (consult7 §5: "the short-context bypass of indexer and
        exchange") — the same SET `write_selection` + the hand-back produce
        over the scored path. Under context parallelism each rank fills its
        OWNED local slots (`j * world + rank < visible`, the interleaved
        ownership rule) — a slot past the row's visible prefix would alias a
        not-yet-written record.
        """
        selected, counts = self._cache().selected(self.layer_id), self._cache().counts(self.layer_id)
        rows = int(visible.shape[0])
        width = int(selected.shape[1])
        target = selected[:rows]
        target.fill_(-1)
        for row in range(rows):
            vis = int(visible[row])
            live = 0 if vis <= rank else min((vis - rank + world - 1) // world, width)
            if live:
                target[row, :live] = (int(slot_base[row]) + torch.arange(
                    live, device=target.device)).to(target.dtype)
        lives = torch.where(visible > rank,
                            torch.div(visible - rank + world - 1, world, rounding_mode="floor"),
                            torch.zeros_like(visible)).clamp_max(width)
        counts[:rows].copy_(lives.to(counts.dtype))
        return target, counts[:rows]

    def handback(self, merged, *, world_size, rank, slot_map=None, prepared=None):
        """Cross-rank selection hand-back into the owner's buffers (D3).

        Under context parallelism the merged global top-k lands in GLOBAL
        positions; `IndexCache.handback` (backed by
        `dsa_indexer.selection_handback.PreparedSelectionHandback`) recovers
        this rank's local selection once per group. `slot_map` is the PER-ROW
        local-index -> physical-slot table (`[rows, local_capacity]`,
        `Hy4Cache.local_slot_map`) — one shared map would send every query row
        into row zero's cache (`selection_handback.py` fix-on-the-way).
        """
        cache = self._cache()
        if prepared is not None:
            selected, counts = prepared.run(merged, slot_map=slot_map,
                                            output=cache.selected(self.owner_id),
                                            counts=cache.counts(self.owner_id))
            return selected, counts
        return cache.handback(self.layer_id, merged, world_size=world_size, rank=rank,
                              slot_map=slot_map)
