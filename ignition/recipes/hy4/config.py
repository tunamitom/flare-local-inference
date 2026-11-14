# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Tencent Hy4 checkpoint geometry, validated value by value against the model config.

Every number the decoder, the loader, and the export contract rely on is pinned
here: an upstream config drift raises instead of silently reshaping weights.
The reference values are the `tencent/Hy4-preview` config at revision
`705d81ee51566a186d645b74c974d642ef2828fe` (2026-08-27).

IndexCache: only `indexer_types == "full"` layers run the DSA indexer (21 of the
78 target layers: 0, 1, then every fourth from 5); the 57 `shared` layers carry
no indexer weights and reuse the previous full layer's top-2048 selection verbatim.
"""

import json
from dataclasses import dataclass
from pathlib import Path

PAGE_TOKENS = 128

# The 21 indexer-owning layers: layers 0 and 1, then 5, 9, ..., 77.
FULL_INDEXER_LAYERS = (0, 1, *range(5, 78, 4))
assert len(FULL_INDEXER_LAYERS) == 21 and FULL_INDEXER_LAYERS[-1] == 77

# CSF (`csf_32x2_v1`) resident size as a fraction of the raw block scales,
# measured on real Hy4 scales (`/home/tom/hy4/consult2-plan.md` section 1c).
# The MSE-searched scales are the worse case and therefore the planning
# number; max scales compress slightly better (0.539 / 0.595).
CSF_RESIDENT_SCALE_RATIOS = {"gate_up_proj": 0.557, "down_proj": 0.600}

# Bytes of one rank-local cache record per token: the FP8 latent record (512
# no-RoPE + 64 RoPE dims plus the record header) and, on full indexer layers,
# the FP8 index-key record. 656 x 78 + 132 x 21 = 53,940 B per token.
LATENT_RECORD_BYTES = 656
INDEX_RECORD_BYTES = 132


def _require(config, name, expected, *, pin=True):
    value = config.get(name)
    if pin and value != expected:
        raise ValueError(f"Hy4 preparation requires {name}={expected!r}; checkpoint has {value!r}")


def _require_in(config, name, choices, *, pin=True):
    value = config.get(name)
    if pin and value not in choices:
        raise ValueError(f"Hy4 preparation requires {name} in {choices!r}; checkpoint has {value!r}")


def _smoke_relations(config):
    """Structural Hy4 invariants at any geometry (the tiny-parity path).

    `from_dict(config, pinned=False)` accepts reduced geometries (the reference
    package's `tiny-hy4`) but never a structural drift: the iHC 4-stream form,
    the gated-MLA geometry relations, the IndexCache full/shared pattern and the
    dense-then-sparse MLP pattern are checked exactly as the pinned path checks
    their production values.
    """
    for name in ("hidden_size", "num_hidden_layers", "vocab_size", "intermediate_size",
                 "q_lora_rank", "kv_lora_rank", "moe_intermediate_size", "n_routed_experts",
                 "num_experts_per_tok", "index_n_heads", "index_head_dim", "index_topk",
                 "num_attention_heads", "num_key_value_heads", "head_dim",
                 "qk_head_dim", "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim"):
        if type(config.get(name)) is not int or config[name] <= 0:
            raise ValueError(f"Hy4 smoke geometry requires positive integer {name}")
    layers = config["num_hidden_layers"]
    c = config
    if (c["qk_nope_head_dim"] + c["qk_rope_head_dim"] != c["qk_head_dim"]
            or c["head_dim"] != c["qk_rope_head_dim"]
            or c["num_attention_heads"] % c["num_key_value_heads"]
            or c["num_experts_per_tok"] > c["n_routed_experts"]):
        raise ValueError("Hy4 attention/MoE geometry must preserve rotary pairs, KV groups and top-k bounds")
    if not (c.get("use_mla") and c.get("gated_mla") and c.get("use_dsa") and c.get("enable_ihc")
            and c.get("learnable_sink") and not c.get("tie_word_embeddings")):
        raise ValueError("Hy4 requires gated MLA, DSA, iHC and learnable sinks, untied")
    if c.get("hc_mult") != 4 or c.get("hc_eps", 0) <= 0 or c.get("hc_magnitude", 0) <= 0:
        raise ValueError("Hy4 iHC requires hc_mult=4 with positive eps and magnitude")
    if c.get("n_shared_experts") != 1 or c.get("swiglu_limit", 0) <= 0 or c.get("routed_scaling_factor", 0) <= 0:
        raise ValueError("Hy4 MoE requires one shared expert and positive clamp/scaling")
    if type(c.get("num_nextn_predict_layers")) is not int or c["num_nextn_predict_layers"] < 1:
        raise ValueError("Hy4 requires at least one MTP predictor layer")
    rope = c.get("rope_parameters")
    if not isinstance(rope, dict) or "rope_theta" not in rope or "rope_type" not in rope:
        raise ValueError("Hy4 requires rope_parameters with rope_theta and rope_type")
    indexer = c.get("indexer_types")
    expected_indexer = ["full" if layer in FULL_INDEXER_LAYERS else "shared" for layer in range(layers)]
    if indexer != expected_indexer:
        raise ValueError(
            "Hy4 requires the exact indexer_types list: full layers 0, 1, then every fourth from 5; "
            f"checkpoint differs at {[i for i, (a, b) in enumerate(zip(expected_indexer, indexer or ())) if a != b][:8]}")
    if c.get("mlp_layer_types") != ["dense"] + ["sparse"] * (layers - 1):
        raise ValueError("Hy4 requires mlp_layer_types dense at layer 0 and sparse elsewhere")
    if not (isinstance(c.get("layer_types"), list) and len(c["layer_types"]) == layers
            and set(c["layer_types"]) <= {"deepseek_sparse_attention", "indexed_attention"}):
        raise ValueError("Hy4 requires one uniform sparse-attention layer_types list")


# Structural flags the checkpoint's `config.json` carries at full size but the
# HF `HYV4Config` class supplies as defaults; the smoke path fills them so the
# decoder body can rely on one effective config either way.
SMOKE_DEFAULTS = {
    "use_mla": True, "gated_mla": True, "gating_type": "elementwise", "use_dsa": True,
    "enable_ihc": True, "learnable_sink": True, "learnable_sink_init": 0.0,
    "enable_lm_head_fp32": True, "mtp_loss_factor": 0.1, "bitwise_backward_align": False,
    "num_nextn_predict_layers": 1,
}


@dataclass(frozen=True)
class ModelConfig:
    """The validated `config.json` of a Hy4 checkpoint."""

    text: dict

    @classmethod
    def load(cls, snapshot, *, pinned=True):
        return cls.from_dict(json.loads((Path(snapshot) / "config.json").read_text()), pinned=pinned)

    @classmethod
    def from_dict(cls, config, *, pinned=True):
        """Validate a Hy4 `config.json`.

        `pinned=True` (the serving path) demands the exact production geometry
        of `tencent/Hy4-preview` at revision `705d81ee`. `pinned=False` is the
        smoke/tiny path: it fills HF's structural defaults and validates the
        same Hy4 structural invariants at reduced geometry (`_smoke_relations`)
        — for the reference package's `tiny-hy4` rank-construction dry-run.
        """
        if not isinstance(config, dict):
            raise ValueError("Hy4 preparation requires a JSON object config")
        if pinned:
            cls._validate_pinned(config)
            return cls(config)
        effective = {name: value for name, value in SMOKE_DEFAULTS.items() if name not in config}
        effective.update(config)
        _smoke_relations(effective)
        return cls(effective)

    @classmethod
    def _validate_pinned(cls, config):
        # Identity.
        _require(config, "architectures", ["HYV4ForCausalLM"])
        _require(config, "model_type", "hy_v4")
        _require(config, "torch_dtype", "bfloat16")
        _require(config, "dtype", "bfloat16")
        _require(config, "hidden_act", "silu")
        _require(config, "attention_bias", False)
        _require(config, "attention_dropout", 0.0)
        _require(config, "use_cache", True)
        _require(config, "tie_word_embeddings", False)
        _require(config, "bitwise_backward_align", False)
        # Geometry.
        _require(config, "hidden_size", 6144)
        _require(config, "num_hidden_layers", 78)
        _require(config, "vocab_size", 120832)
        _require(config, "intermediate_size", 18432)
        _require(config, "max_position_embeddings", 1048576)
        _require(config, "initializer_range", 0.006)
        _require(config, "rms_norm_eps", 1e-5)
        # Gated MLA attention.
        _require(config, "use_mla", True)
        _require(config, "gated_mla", True)
        _require(config, "gating_type", "elementwise")
        _require(config, "num_attention_heads", 64)
        _require(config, "num_key_value_heads", 8)
        _require(config, "head_dim", 64)
        _require(config, "qk_head_dim", 256)
        _require(config, "qk_nope_head_dim", 192)
        _require(config, "qk_rope_head_dim", 64)
        _require(config, "v_head_dim", 256)
        _require(config, "q_lora_rank", 2048)
        _require(config, "kv_lora_rank", 512)
        if (config["qk_nope_head_dim"] + config["qk_rope_head_dim"] != config["qk_head_dim"]
                or config["num_attention_heads"] % config["num_key_value_heads"]):
            raise ValueError("Hy4 attention geometry must preserve complete rotary pairs and KV groups")
        # DeepSeek sparse indexer and IndexCache.
        _require(config, "use_dsa", True)
        _require(config, "index_n_heads", 32)
        _require(config, "index_head_dim", 128)
        _require(config, "index_topk", 2048)
        _require(config, "layer_types", ["deepseek_sparse_attention"] * 78)
        _require(config, "mlp_layer_types", ["dense"] + ["sparse"] * 77)
        indexer_types = config.get("indexer_types")
        expected_indexer = ["full" if layer in FULL_INDEXER_LAYERS else "shared" for layer in range(78)]
        if indexer_types != expected_indexer:
            raise ValueError(
                "Hy4 requires the exact indexer_types list: full layers 0, 1, 5..77 step 4; "
                f"checkpoint differs at {[i for i, (a, b) in enumerate(zip(expected_indexer, indexer_types or ())) if a != b][:8]}")
        # iHC: four independent residual streams, input-dependent collapse.
        _require(config, "enable_ihc", True)
        _require(config, "hc_mult", 4)
        _require(config, "hc_eps", 1e-6)
        _require(config, "hc_magnitude", 2.0)
        # GPT-OSS-style learnable attention sinks.
        _require(config, "learnable_sink", True)
        _require(config, "learnable_sink_init", 0.0)
        # MoE: 256 routed + 1 shared expert, top-8 renormalised then x2.827,
        # routed SwiGLU clamped to +/-10 (shared expert and layer 0 unclamped).
        _require(config, "n_routed_experts", 256)
        _require(config, "n_shared_experts", 1)
        _require(config, "num_experts_per_tok", 8)
        _require(config, "moe_intermediate_size", 2048)
        _require(config, "n_group", 1)
        _require(config, "topk_group", 1)
        _require(config, "norm_topk_prob", True)
        _require(config, "routed_scaling_factor", 2.827)
        _require(config, "swiglu_limit", 10.0)
        # MTP: one single-stream GLM-style predictor with its own indexer, sink,
        # gate and 256-expert MoE; the head is shared.
        _require(config, "num_nextn_predict_layers", 1)
        _require(config, "mtp_loss_factor", 0.1)
        # Rotary and head.
        _require(config, "rope_parameters", {"rope_theta": 10000000.0, "rope_type": "default"})
        _require(config, "enable_lm_head_fp32", True)
        # Special tokens.
        _require(config, "bos_token_id", 120000)
        _require(config, "eos_token_id", 120025)
        _require(config, "pad_token_id", 120002)
        _require_in(config, "transformers_version", ("5.16.2",))

    @property
    def layers(self):
        return self.text["num_hidden_layers"]

    @property
    def hidden(self):
        return self.text["hidden_size"]

    @property
    def vocabulary(self):
        return self.text["vocab_size"]

    @property
    def heads(self):
        return self.text["num_attention_heads"]

    @property
    def experts(self):
        return self.text["n_routed_experts"]

    @property
    def draft_layers(self):
        return self.text["num_nextn_predict_layers"]

    @property
    def full_indexer_layers(self):
        """Layers that run the indexer and own the only index-key pages."""
        layers = tuple(i for i, kind in enumerate(self.text["indexer_types"]) if kind == "full")
        assert len(self.text["indexer_types"]) != 78 or layers == FULL_INDEXER_LAYERS
        return layers

    @property
    def shared_indexer_layers(self):
        """Layers that consume the previous full layer's selection verbatim."""
        return tuple(layer for layer in range(self.layers) if layer not in FULL_INDEXER_LAYERS)

    def indexer_type(self, layer):
        if not 0 <= layer < self.layers + self.draft_layers:
            raise ValueError(f"no Hy4 layer {layer}")
        return self.text["indexer_types"][layer] if layer < self.layers else "full"

    def token_record_bytes(self, *, draft=False):
        """One rank's cache bytes per token across the layer stack."""
        layers = self.layers + int(draft)
        indexers = len(self.full_indexer_layers) + int(draft)
        return layers * LATENT_RECORD_BYTES + indexers * INDEX_RECORD_BYTES

    def parameter_census(self):
        """Exact parameter counts per tensor family, at the preview geometry.

        The totals reproduce the checkpoint's published parameter counts
        (779,930,197,312 BF16 + 30,795,421 FP32 = 779,960,992,733).
        """
        c = self.text
        h, v = self.hidden, self.vocabulary
        heads, qk, nope, rope, vd = self.heads, c["qk_head_dim"], c["qk_nope_head_dim"], c["qk_rope_head_dim"], c["v_head_dim"]
        ql, kv = c["q_lora_rank"], c["kv_lora_rank"]
        indexers = len(self.full_indexer_layers) + self.draft_layers
        blocks = self.layers + self.draft_layers
        moe_blocks = self.layers - 1 + self.draft_layers
        e, mi, dense = self.experts, c["moe_intermediate_size"], c["intermediate_size"]
        return {
            "embed_tokens": v * h,
            "lm_head": v * h,
            "model_norm": h,
            "q_a_proj": blocks * ql * h,
            "q_a_layernorm": blocks * ql,
            "q_b_proj": blocks * heads * qk * ql,
            "kv_a_proj_with_mqa": blocks * (kv + rope) * h,
            "kv_a_layernorm": blocks * kv,
            "kv_b_proj": blocks * heads * (nope + vd) * kv,
            "o_proj": blocks * h * heads * vd,
            # gated DSA: per-head scalar gate over the attention output.
            "linear_gate": blocks * heads * vd * h,
            "sinks": blocks * heads,
            "input_layernorm": blocks * h,
            "post_attention_layernorm": blocks * h,
            "indexer.wq_b": indexers * (32 * 128) * ql,
            "indexer.wk": indexers * 128 * h,
            "indexer.weights_proj": indexers * 32 * h,
            "indexer.k_norm": indexers * 2 * 128,
            "hc_attn_layer.hc_pre": self.layers * (8 * (4 * h) + 8 + 2),
            "hc_mlp_layer.hc_pre": self.layers * (8 * (4 * h) + 8 + 2),
            "hc_head": 4 * (4 * h) + 4 + 1,
            "mlp.dense": 3 * dense * h,
            "mlp.experts": moe_blocks * e * (2 * mi * h + h * mi),
            "mlp.gate": moe_blocks * e * h,
            "mlp.e_score_correction_bias": moe_blocks * e,
            "mlp.shared_experts": moe_blocks * (2 * mi * h + h * mi),
            "mtp.eh_proj": self.draft_layers * 2 * h * h,
            "mtp.norms": self.draft_layers * 3 * h,
        }

    def stored_parameter_counts(self):
        """(BF16, FP32) storage counts reproducing the checkpoint's published census.

        Storage FP32 is the iHC `fn`/`base`/`scale` parameters, the routed
        correction biases and the learnable sinks of the 78 target layers. The
        MTP block sits outside transformers' strict-FP32 list (it drops
        `model.mtp_layers.*` on load) and writes its correction bias and sink in
        BF16; `weights_proj`, `k_norm` and `lm_head` are stored BF16 and kept
        FP32 at load (see `weights.py`).
        """
        census = self.parameter_census()
        fp32 = sum(census[name] for name in
                   ("hc_attn_layer.hc_pre", "hc_mlp_layer.hc_pre", "hc_head"))
        fp32 += (self.layers - 1) * self.experts  # target-layer correction biases
        fp32 += self.layers * self.heads          # target-layer sinks
        return sum(census.values()) - fp32, fp32

    def packed_byte_arithmetic(self):
        """Exact resident byte arithmetic of the W4A16-packed checkpoint.

        Every figure is documented derivation from `parameter_census`, not a
        measurement; `tests/recipes/test_hy4_csf.py` reproduces it independently.

        - `packed_resident_bytes`: every tensor once at its **resident** dtype:
          routed experts packed to W4A16 NVFP4 (4 bits per weight, one FP8
          block scale per 16 weights, one FP32 gain pair per expert), `lm_head`,
          the indexer `weights_proj`/`k_norm`, and the MTP block's correction
          bias and sink upgraded to FP32 by `weights.kept_fp32`.
        - `replicated_bytes`: resident bytes every tensor-parallel rank carries
          whole. The partition policy follows the reference vLLM port: the
          `q_a_proj`/`kv_a_proj_with_mqa` down-projections, the whole indexer,
          the router `gate` and correction bias, every norm, the learnable
          sinks, and the iHC streams (`fn`/`base`/`scale`, `hc_head`) replicate;
          `kv_b_proj` replicates too with the `kv_b` two-wirings (consult8 §5).
          `q_b_proj`, `o_proj`, `linear_gate`, the experts, the
          shared experts, the dense layer-0 FFN, `eh_proj`, the embedding and
          the FP32 head shard. (If the embedding and FP32 head do not shard,
          every rank pays 4,454,141,952 bytes more.)
        - `draft_bytes`: the packed MTP block, loaded only for learned drafts.
        - `expert_scale_bytes`/`csf_scale_bytes`: raw block scales and their
          CSF-resident form at the measured ratios (worse MSE case).
        - `scale_scratch_bytes`: one rank's shared CSF expansion scratch at TP8
          (two scale layouts: an expansion writes one while the next layer's
          compressed bytes stage in the other), about 0.15 GiB per rank.
        """
        import math

        c = self.text
        h, v = self.hidden, self.vocabulary
        heads, qk, nope, rope, vd = self.heads, c["qk_head_dim"], c["qk_nope_head_dim"], c["qk_rope_head_dim"], c["v_head_dim"]
        ql, kv = c["q_lora_rank"], c["kv_lora_rank"]
        indexers = len(self.full_indexer_layers) + self.draft_layers
        blocks = self.layers + self.draft_layers
        moe_blocks = self.layers - 1 + self.draft_layers
        e, mi = self.experts, c["moe_intermediate_size"]
        census = self.parameter_census()
        stored_bf16, stored_fp32 = self.stored_parameter_counts()

        # Resident dtype upgrades over storage: `lm_head`, indexer
        # `weights_proj`/`k_norm`, and the MTP block's correction bias and sink
        # load FP32 from BF16 storage.
        upgrades = (v * h + indexers * 32 * h + indexers * 2 * 128 + e + heads) * 2
        expert_params = census["mlp.experts"]
        packed_experts = expert_params * 4 // 8 + expert_params // 16 + moe_blocks * e * 2 * 4
        # `kv_b_proj` of the target layers is replicated (the `kv_b`
        # two-wirings) — placement only: `packed_resident` counts every tensor
        # once at resident dtype regardless of placement, and `replicated`
        # carries the whole matrix on every rank.
        packed_resident = (packed_experts + (stored_bf16 - expert_params) * 2
                           + stored_fp32 * 4 + upgrades)

        replicated = (
            blocks * ql * h * 2                    # q_a_proj
            + blocks * (kv + rope) * h * 2         # kv_a_proj_with_mqa
            + self.layers * heads * (nope + vd) * kv * 2  # kv_b_proj (replicated — the `kv_b` two-wirings)
            + blocks * ql * 2 + blocks * kv * 2    # q_a/kv_a layernorms
            + blocks * h * 2 * 2                   # input/post_attention layernorms
            + h * 2                                # model.norm
            + blocks * heads * 4                   # learnable sinks (FP32)
            + moe_blocks * e * h * 2               # router gate.weight
            + moe_blocks * e * 4                   # correction biases (FP32 resident)
            + indexers * (32 * 128) * ql * 2       # indexer.wq_b
            + indexers * 128 * h * 2               # indexer.wk
            + indexers * 32 * h * 4                # indexer.weights_proj (FP32 resident)
            + indexers * 2 * 128 * 4               # indexer.k_norm (FP32 resident)
            + self.layers * 2 * (8 * (4 * h) + 8 + 2) * 4  # iHC hc_pre, attn and mlp
            + (4 * (4 * h) + 4 + 1) * 4            # hc_head
            + self.draft_layers * 3 * h * 2)       # MTP enorm/hnorm/final_layernorm

        draft_experts = e * (2 * mi * h + h * mi)
        draft_bytes = (
            draft_experts * 4 // 8 + draft_experts // 16 + e * 2 * 4  # packed experts
            + (2 * mi * h + h * mi) * 2            # shared experts
            + 2 * h * h * 2                        # eh_proj
            + ql * h * 2 + heads * qk * ql * 2     # q_a_proj, q_b_proj
            + (kv + rope) * h * 2 + heads * (nope + vd) * kv * 2
            + h * heads * vd * 2 + heads * vd * h * 2  # o_proj, linear_gate
            + ((32 * 128) * ql + 128 * h + 32 * h) * 2 + 2 * 128 * 2  # indexer
            + e * h * 2                            # router gate
            + (5 * h + ql + kv) * 2                # enorm/hnorm/final/input/post_attention norms
            + (e + heads) * 2 + (32 * h + 2 * 128 + e + heads) * 2)  # bias/sink and FP32 upgrades

        gate_up_scales = moe_blocks * e * (2 * mi) * (h // 16)
        down_scales = moe_blocks * e * h * (mi // 16)
        ratios = CSF_RESIDENT_SCALE_RATIOS
        csf_scales = (math.ceil(ratios["gate_up_proj"] * gate_up_scales)
                      + math.ceil(ratios["down_proj"] * down_scales))
        # Two expansion layouts of one rank's block scales at TP8 (256 experts,
        # 256 local intermediate channels).
        scale_scratch = 2 * (e * (2 * mi // 8) * (h // 16) + e * h * ((mi // 8) // 16))
        return {
            "packed_resident_bytes": packed_resident,
            "replicated_bytes": replicated,
            "draft_bytes": draft_bytes,
            "expert_scale_bytes": gate_up_scales + down_scales,
            "csf_scale_bytes": csf_scales,
            "scale_scratch_bytes": scale_scratch,
        }
