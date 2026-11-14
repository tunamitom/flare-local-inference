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

# Bytes of one rank-local cache record per token: the FP8 latent record (512
# no-RoPE + 64 RoPE dims plus the record header) and, on full indexer layers,
# the FP8 index-key record. 656 x 78 + 132 x 21 = 53,940 B per token.
LATENT_RECORD_BYTES = 656
INDEX_RECORD_BYTES = 132


def _require(config, name, expected):
    value = config.get(name)
    if value != expected:
        raise ValueError(f"Hy4 preparation requires {name}={expected!r}; checkpoint has {value!r}")


def _require_in(config, name, choices):
    value = config.get(name)
    if value not in choices:
        raise ValueError(f"Hy4 preparation requires {name} in {choices!r}; checkpoint has {value!r}")


@dataclass(frozen=True)
class ModelConfig:
    """The validated `config.json` of a Hy4 checkpoint."""

    text: dict

    @classmethod
    def load(cls, snapshot):
        return cls.from_dict(json.loads((Path(snapshot) / "config.json").read_text()))

    @classmethod
    def from_dict(cls, config):
        if not isinstance(config, dict):
            raise ValueError("Hy4 preparation requires a JSON object config")
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
        return cls(config)

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
        return FULL_INDEXER_LAYERS

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
