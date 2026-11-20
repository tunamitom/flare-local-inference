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

# Official `tencent/Hy4` (`hy_v4_vl`) DFly2 target layers and alignment.
OFFICIAL_DFLY_LAYERS = (61, 65, 69, 73, 77)
DFLY_TOPK_ALIGNMENT = 128


def _normalize_hy4_config(config):
    """Flatten nested `hy_v4_vl` `text_config` while checking duplicate fields.

    Official `tencent/Hy4` wraps text geometry under `text_config` alongside
    top-level text keys, `vision_config`, and `dfly_*` metadata. Flat preview
    `hy_v4` dicts pass through untouched.
    """
    if config.get("model_type") != "hy_v4_vl" or not isinstance(config.get("text_config"), dict):
        return dict(config)
    text_cfg = config["text_config"]
    merged = dict(text_cfg)
    for key, value in config.items():
        if key == "text_config":
            continue
        if key in text_cfg and key not in ("model_type", "architectures"):
            if text_cfg[key] != value:
                raise ValueError(
                    f"Hy4 VL config disagrees between top-level and text_config on {key!r}: "
                    f"{value!r} vs {text_cfg[key]!r}")
        else:
            merged[key] = value
    merged["model_type"] = config["model_type"]
    merged["architectures"] = config.get("architectures", text_cfg.get("architectures"))
    merged["_text_model_type"] = text_cfg.get("model_type")
    # Official `hy_v4_vl` omits `use_mla` / `use_dsa` (implicit from `layer_types`
    # and `gated_mla`); preserve explicit False so validation rejects it.
    merged.setdefault("use_mla", True)
    merged.setdefault("use_dsa", True)
    return merged


def _validate_dfly_relations(config):
    """Validate structural DFly2 config invariants when bundled."""
    dfly_layers = config.get("dfly_num_hidden_layers")
    if dfly_layers is None or dfly_layers == 0:
        return
    if type(dfly_layers) is not int or dfly_layers < 1:
        raise ValueError(f"Hy4 DFly requires a positive integer dfly_num_hidden_layers, got {dfly_layers!r}")
    for key, expected in (("dfly_enable_ihc", False), ("dfly_mlp_type", "dense"),
                          ("dfly_attn_causal", True), ("dfly_mtp_fusion", "slot0"),
                          ("dfly_enable_hidden_correction", True)):
        value = config.get(key, expected)
        if type(value) is not type(expected) or value != expected:
            raise ValueError(f"Hy4 DFly requires {key}={expected!r}; checkpoint has {value!r}")
    block_size = config.get("dfly_block_size")
    if type(block_size) is not int or not 2 <= block_size <= 17:
        raise ValueError(f"Hy4 DFly requires dfly_block_size in [2, 17], got {block_size!r}")
    mask_id = config.get("dfly_mask_token_id")
    if type(mask_id) is not int or not 0 <= mask_id < config["vocab_size"]:
        raise ValueError(
            f"Hy4 DFly requires dfly_mask_token_id in [0, {config['vocab_size']}), got {mask_id!r}")
    target_topk = config.get("dfly_target_index_topk", config["index_topk"])
    if type(target_topk) is not int or target_topk != config["index_topk"]:
        raise ValueError(
            f"Hy4 DFly requires dfly_target_index_topk == index_topk ({config['index_topk']}), "
            f"got {target_topk!r}")
    window = config.get("dfly_context_local_window")
    if window is not None and (type(window) is not int or window < 0):
        raise ValueError(f"Hy4 DFly requires dfly_context_local_window None or >= 0, got {window!r}")
    layers = config["num_hidden_layers"]
    indexer_types = config["indexer_types"]
    for key in ("dfly_target_index_layer", "dfly_target_kv_layers"):
        targets = config.get(key)
        if key == "dfly_target_index_layer" and type(targets) is int:
            targets = [targets] * dfly_layers
        if (not isinstance(targets, (list, tuple)) or len(targets) != dfly_layers
                or any(type(i) is not int or not 0 <= i < layers for i in targets)):
            raise ValueError(
                f"Hy4 DFly {key} must list {dfly_layers} valid target layer indices in [0, {layers}), "
                f"got {targets!r}")
        if key == "dfly_target_kv_layers" and len(set(targets)) != dfly_layers:
            raise ValueError(f"Hy4 DFly dfly_target_kv_layers must be distinct, got {targets!r}")
        if key == "dfly_target_index_layer":
            shared = [i for i in targets if indexer_types[i] != "full"]
            if shared:
                raise ValueError(
                    f"Hy4 DFly dfly_target_index_layer must name 'full' indexer layers; got shared {shared}")
    for key, default in (("dfly_intermediate_size", config["intermediate_size"]),
                         ("dfly_hidden_correction_intermediate_size", config["hidden_size"])):
        size = config.get(key)
        size = default if size is None else size
        if type(size) is not int or size < 1:
            raise ValueError(f"Hy4 DFly {key} must be a positive integer, got {size!r}")


def _require(config, name, expected, *, pin=True):
    value = config.get(name)
    if pin and value != expected:
        raise ValueError(f"Hy4 preparation requires {name}={expected!r}; checkpoint has {value!r}")


def _require_in(config, name, choices, *, pin=True):
    value = config.get(name)
    if pin and value not in choices:
        raise ValueError(f"Hy4 preparation requires {name} in {choices!r}; checkpoint has {value!r}")


def _validate_vision_config(config):
    """Pin the official `hy_v4_vl` `vision_config` (VISION-DESIGN section 4.1 item 1).

    Only `hy_v4_vl` checkpoints carry `vision_config`; the pinned values are
    the exact `tencent/Hy4` (`bd0c8665bf93`) vision geometry. `use_2d_rope` is
    ABSENT from the checkpoint config (defaults False) and `rope_2d_theta` /
    `rope_2d_percent` are inert — an explicit `use_2d_rope: true` is a
    structural drift and fails closed.
    """
    vision = config.get("vision_config")
    if not isinstance(vision, dict):
        raise ValueError("Hy4 VL preparation requires a vision_config object")
    for name, expected in (("num_hidden_layers", 40), ("hidden_size", 1536),
                           ("intermediate_size", 6144), ("num_attention_heads", 16),
                           ("patch_size", 16), ("spatial_merge_size", 2),
                           ("max_image_size", 2048),
                           ("projector_type", "rmsnorm_mlp_adapter")):
        value = vision.get(name)
        if value != expected:
            raise ValueError(
                f"Hy4 VL preparation requires vision_config.{name}={expected!r}; "
                f"checkpoint has {value!r}")
    if vision.get("use_2d_rope", False):
        raise ValueError("Hy4 VL vision carries no 2D ViT RoPE (use_2d_rope must be unset)")
    # Special tokens live at top level (merged config) — pinned by numeric id.
    for name, expected in (("image_token_id", 120120), ("image_start_token_id", 120118),
                           ("image_end_token_id", 120119)):
        _require(config, name, expected)


def _smoke_relations(config):
    """Structural Hy4 invariants at any geometry (the tiny-parity path).

    `from_dict(config, pinned=False)` accepts reduced geometries (the reference
    package's `tiny-hy4` and official `hy_v4_vl` DFly2 tiny fixtures) but never
    a structural drift: the iHC 4-stream form, the gated-MLA geometry relations,
    the IndexCache full/shared pattern, the dense-then-sparse MLP pattern, and
    any bundled DFly2 draft geometry are checked strictly.
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
    if type(c.get("num_nextn_predict_layers")) is not int or c["num_nextn_predict_layers"] < 0:
        raise ValueError("Hy4 requires a nonnegative integer num_nextn_predict_layers")
    # `draft_layers == 0` is legal (plan 4.1): official `hy_v4_vl` carries no
    # MTP predictor layer at all, so the smoke path must accept zero draft
    # layers whether or not a DFly2 drafter is bundled. Zero draft layers
    # means zero draft KV bytes (`token_record_bytes`) and 77 packed MoE
    # prefixes (59,136 matrices) in the export.
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
    _validate_dfly_relations(c)


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

        `pinned=True` (the serving path) demands either the exact production
        geometry of `tencent/Hy4-preview` (`hy_v4`, revision `705d81ee`) or
        official `tencent/Hy4` (`hy_v4_vl`, revision `bd0c8665bf93` with
        5-layer DFly2). `pinned=False` is the smoke/tiny path: it fills HF's
        structural defaults and validates the same Hy4 structural invariants
        at reduced geometry (`_smoke_relations`).
        """
        if not isinstance(config, dict):
            raise ValueError("Hy4 preparation requires a JSON object config")
        normalized = _normalize_hy4_config(config)
        if pinned:
            cls._validate_pinned(normalized)
            return cls(normalized)
        effective = {name: value for name, value in SMOKE_DEFAULTS.items() if name not in normalized}
        effective.update(normalized)
        if effective.get("dfly_num_hidden_layers") and "num_nextn_predict_layers" not in normalized:
            effective["num_nextn_predict_layers"] = 0
        _smoke_relations(effective)
        return cls(effective)

    @classmethod
    def _validate_pinned(cls, config):
        is_vl = config.get("model_type") == "hy_v4_vl"
        # Identity.
        if is_vl:
            _require(config, "architectures", ["HYV4VLForConditionalGeneration"])
            _require(config, "model_type", "hy_v4_vl")
            _require(config, "_text_model_type", "hy_v4")
            _require(config, "dtype", "bfloat16")
            if "torch_dtype" in config:
                _require(config, "torch_dtype", "bfloat16")
            _require(config, "qk_norm", True)
        else:
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
        if is_vl:
            # Official Hy4: 0 legacy MTP layers; 5-layer dense DFly2 drafter.
            _require(config, "num_nextn_predict_layers", 0)
            _require(config, "dfly_num_hidden_layers", 5)
            _require(config, "dfly_mlp_type", "dense")
            _require(config, "dfly_enable_ihc", False)
            _require(config, "dfly_mtp_fusion", "slot0")
            _require(config, "dfly_enable_hidden_correction", True)
            _require(config, "dfly_target_index_layer", list(OFFICIAL_DFLY_LAYERS))
            _require(config, "dfly_target_kv_layers", list(OFFICIAL_DFLY_LAYERS))
            _require(config, "dfly_target_index_topk", 2048)
            _require(config, "dfly_attn_causal", True)
            _require(config, "dfly_block_size", 7)
            _require(config, "dfly_mask_token_id", 120009)
            _require(config, "dfly_context_local_window", None)
            _validate_dfly_relations(config)
            rope = config.get("rope_parameters")
            expected_rope = {
                "mrope_interleaved": True,
                "mrope_section": [12, 10, 10],
                "rope_theta": 10000000,
                "rope_type": "default",
            }
            if (not isinstance(rope, dict)
                    or rope.get("mrope_interleaved") is not expected_rope["mrope_interleaved"]
                    or rope.get("mrope_section") != expected_rope["mrope_section"]
                    or rope.get("rope_type") != expected_rope["rope_type"]
                    or float(rope.get("rope_theta", -1)) != float(expected_rope["rope_theta"])
                    or set(rope) != set(expected_rope)):
                raise ValueError(
                    f"Hy4 VL preparation requires rope_parameters={expected_rope!r}; checkpoint has {rope!r}")
            _validate_vision_config(config)
            _require_in(config, "transformers_version", ("5.24.0",))
        else:
            # Preview Hy4: one single-stream GLM-style MTP predictor.
            _require(config, "num_nextn_predict_layers", 1)
            _require(config, "mtp_loss_factor", 0.1)
            _require(config, "rope_parameters", {"rope_theta": 10000000.0, "rope_type": "default"})
            _require_in(config, "transformers_version", ("5.16.2",))
        _require(config, "enable_lm_head_fp32", True)
        # Special tokens.
        _require(config, "bos_token_id", 120000)
        _require(config, "eos_token_id", 120025)
        _require(config, "pad_token_id", 120002)

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
    def dfly_layers(self):
        return int(self.text.get("dfly_num_hidden_layers") or 0)

    @property
    def has_dfly(self):
        return self.dfly_layers > 0

    @property
    def has_vision(self):
        return isinstance(self.text.get("vision_config"), dict)

    @property
    def dfly_target_index_layers(self):
        if not self.has_dfly:
            return ()
        raw = self.text["dfly_target_index_layer"]
        if type(raw) is int:
            return (raw,) * self.dfly_layers
        return tuple(int(i) for i in raw)

    @property
    def dfly_target_kv_layers(self):
        if not self.has_dfly:
            return ()
        return tuple(int(i) for i in self.text["dfly_target_kv_layers"])

    @property
    def dfly_block_size(self):
        return int(self.text["dfly_block_size"]) if self.has_dfly else 0

    @property
    def dfly_mask_token_id(self):
        return int(self.text["dfly_mask_token_id"]) if self.has_dfly else -1

    @property
    def dfly_context_local_window(self):
        if not self.has_dfly:
            return 0
        window = self.text.get("dfly_context_local_window")
        return 0 if window is None else int(window)

    @property
    def dfly_intermediate_size(self):
        if not self.has_dfly:
            return 0
        size = self.text.get("dfly_intermediate_size")
        return int(self.text["intermediate_size"] if size is None else size)

    @property
    def dfly_hidden_correction_intermediate_size(self):
        if not self.has_dfly:
            return 0
        size = self.text.get("dfly_hidden_correction_intermediate_size")
        return int(self.hidden if size is None else size)

    @property
    def dfly_index_topk(self):
        """Sparse attention column width for DFly2 layers (history + window + block).

        Rounded up to 128-index blocks when `target_index_topk >= 128` (2,176
        at production geometry) to satisfy sparse-MLA block alignment; kept
        exact on small smoke configs whose `index_topk < 128`.
        """
        if not self.has_dfly:
            return 0
        target_topk = int(self.text.get("dfly_target_index_topk", self.text["index_topk"]))
        width = target_topk + self.dfly_context_local_window + self.dfly_block_size
        if target_topk >= DFLY_TOPK_ALIGNMENT:
            return -(-width // DFLY_TOPK_ALIGNMENT) * DFLY_TOPK_ALIGNMENT
        return width

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
        """One rank's cache bytes per token across the layer stack.

        Legacy MTP (`draft_layers > 0`) adds one persistent latent and index
        layer; DFly2 writes ephemeral block KV into target layers' verify slots
        and adds zero persistent KV bytes.
        """
        persistent_draft = int(bool(draft) and self.draft_layers > 0)
        layers = self.layers + persistent_draft
        indexers = len(self.full_indexer_layers) + persistent_draft
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
        census = {
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
        if self.has_dfly:
            dn = self.dfly_layers
            di = self.dfly_intermediate_size
            ci = self.dfly_hidden_correction_intermediate_size
            census.update({
                "dfly.q_a_proj": dn * ql * h,
                "dfly.q_a_layernorm": dn * ql,
                "dfly.q_b_proj": dn * heads * qk * ql,
                "dfly.kv_a_proj_with_mqa": dn * (kv + rope) * h,
                "dfly.kv_a_layernorm": dn * kv,
                "dfly.kv_b_proj": dn * heads * (nope + vd) * kv,
                "dfly.o_proj": dn * h * heads * vd,
                "dfly.linear_gate": dn * heads * vd * h,
                "dfly.sinks": dn * heads,
                "dfly.norms": dn * 2 * h + h,  # input/post_attention + final_norm
                "dfly.mlp.dense": dn * 3 * di * h,
                "dfly.mtp_fusion": 2 * h * h + 3 * h,  # eh_proj + enorm/hnorm/gate
                "dfly.hidden_correction": (2 * ci * (2 * h) + h * ci) + 2 * h,
            })
        return census

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

    def packed_byte_arithmetic(self, *, vision=False):
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
        - `vision=True` (Phase 4, VISION-DESIGN section 1.1): adds the TP8-A1
          sharded `vit.*` footprint — `vision_bytes` (all 649 tensors once, BF16)
          and `vision_sharded_bytes` (the exact per-rank 355,560,960 B = 339.09
          MiB: replicated embeddings 52,697,088 + col/row-sharded layers
          40 x 7,099,008 + TP8 `perceive` 18,903,552).
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
        # `weights_proj`/`k_norm`, and any BF16-stored draft correction bias /
        # sinks load FP32 from BF16 storage.
        upgrades = (v * h + indexers * 32 * h + indexers * 2 * 128
                    + self.draft_layers * (e + heads) + self.dfly_layers * heads) * 2
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

        if self.has_dfly:
            dn = self.dfly_layers
            di = self.dfly_intermediate_size
            ci = self.dfly_hidden_correction_intermediate_size
            dfly_replicated = (
                dn * (ql * h + (kv + rope) * h + heads * (nope + vd) * kv
                      + ql + kv + 2 * h) * 2
                + (2 * h * h + 3 * h + h + 2 * h) * 2  # mtp_fusion + final_norm + hc norms
            )
            dfly_sharded = (
                dn * (heads * qk * ql + h * heads * vd + heads * vd * h + 3 * di * h) * 2
                + (2 * ci * (2 * h) + h * ci) * 2
                + dn * heads * 4  # sinks head-sharded FP32
            )
            replicated += dfly_replicated
            draft_bytes = dfly_replicated + dfly_sharded
        else:
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
        result = {
            "packed_resident_bytes": packed_resident,
            "replicated_bytes": replicated,
            "draft_bytes": draft_bytes,
            "expert_scale_bytes": gate_up_scales + down_scales,
            "csf_scale_bytes": csf_scales,
            "scale_scratch_bytes": scale_scratch,
        }
        if vision and self.has_vision:
            # Phase 4 (VISION-DESIGN section 1.1): `vit.*` at TP8-A1 — every
            # tensor BF16 (2 B/element), `q/k/v` + `dense_h_to_4h` +
            # `perceive.proj.0` column-shard (dim 0 / 8) and `o_proj` +
            # `dense_4h_to_h` + `perceive.proj.2` row-shard (dim 1 / 8); the
            # embeddings, every norm, `image_newline` and the post-all-reduce
            # biases replicate. (If the world is not 8, `Hy4VisionWeights`
            # shards the same way and `allocated_bytes` measures the result.)
            v = self.text["vision_config"]
            vh, vl = int(v["hidden_size"]), int(v["num_hidden_layers"])
            vi = int(v["intermediate_size"])
            vo = int(v.get("out_hidden_size", self.hidden))
            vp = int(v.get("patch_size", 16))
            vedge = int(v.get("max_image_size", 2048)) // vp
            vpos = vedge * vedge + 1
            embed_elements = vh * 3 * vp * vp + vh + vpos * vh
            layer_elements = (3 * (vh * vh + vh) + (vh * vh + vh)      # q/k/v + o
                              + vi * vh + vi + vh * vi + vh             # fc1 + fc2
                              + 4 * vh)                                 # 2 x LayerNorm(w+b)
            # Per-rank: q/k/v (+biases), `o_proj` weight, `dense_h_to_4h`
            # (+bias) and `dense_4h_to_h` weight shard / 8; `o_proj` /
            # `dense_4h_to_h` biases and the two LayerNorms replicate.
            layer_rank = ((3 * (vh * vh + vh) + vh * vh + vi * vh + vi + vh * vi) // 8
                          + 2 * vh + 4 * vh)
            perceive_elements = vh + vi * vh * 4 + vi + vo * vi + vo + vo
            perceive_rank = ((vi * vh * 4 + vi + vo * vi) // 8 + vh + vo + vo)
            result["vision_bytes"] = (embed_elements + vl * layer_elements
                                      + perceive_elements) * 2
            result["vision_sharded_bytes"] = (embed_elements * 2
                                              + vl * layer_rank * 2
                                              + perceive_rank * 2)
        return result
