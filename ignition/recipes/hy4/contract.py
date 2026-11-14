# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Export contract: ModelOpt `MIXED_PRECISION` declarations with checkpoint source names.

FLARE's generic ModelOpt parser accepts only `quant_method: "modelopt"` with
`quant_algo: "MIXED_PRECISION"`, unquantized KV, and a per-layer
`quantized_layers` map keyed by checkpoint source names
(`ignition/preparation/quantization.py:10-19`). This module encodes that form
for Hy4 and the passthrough rule for everything else in `config.json`: the
export copies the source config unchanged and adds `quantization_config` alone.
Regenerating the config (the quant-toolkit's export rewrites
`num_key_value_heads` and other geometry) is forbidden.

What packs: only the 77 target layers' routed experts and the MTP block's
experts, as `W4A16_NVFP4` with group size 16. Everything else stays plain
storage and is therefore absent from the map — the Tencent FP8 build's
do-not-quantize list is the floor: embeddings, `lm_head`, iHC `fn`, the
`linear_gate`, the router `gate`, and the indexer `weights_proj`; the shared
experts and layer 0 also stay BF16 (they see every token).

The packed W4A16 experts need no external quantization step: the loader packs
the BF16 checkpoint's experts at load (`weights.py`), and the export writes the
same packing out under this contract.
"""

from ignition.preparation.quantization import expert_format, quantized_layers

# Expert packing the export writes and the loader accepts: maximum-scale
# ModelOpt NVFP4, activations in 16 bits (`W4A16_NVFP4` -> ("nvfp4", "a16")).
EXPERT_ALGORITHM = {"quant_algo": "W4A16_NVFP4", "group_size": 16}

# Families that must never appear in `quantized_layers` (do-not-quantize floor).
UNQUANTIZED_FAMILIES = (
    "embed_tokens", "lm_head", "hc_pre", "hc_head", "linear_gate", "learnable_sink_param",
    "mlp.gate", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "mlp.shared_experts",
    "self_attn", "indexer", "layernorm", "norm",
)


def quantized_source_names(config, *, mtp=True):
    """Source names of the packed stacks: one fused expert prefix per MoE block."""
    names = tuple(f"model.layers.{layer}.mlp.experts" for layer in range(1, config.layers))
    return names + (tuple(f"model.mtp_layers.{index}.mlp.experts" for index in range(config.draft_layers))
                    if mtp else ())


def quantization_config(config, *, mtp=True):
    """The export's `quantization_config`, in FLARE's generic ModelOpt form."""
    return {
        "quant_method": "modelopt",
        "quant_algo": "MIXED_PRECISION",
        "kv_cache_quant_algo": None,
        "quantized_layers": {name: dict(EXPERT_ALGORITHM) for name in quantized_source_names(config, mtp=mtp)},
    }


def export_config(source_config, *, mtp=True):
    """The exported `config.json`: the source config plus `quantization_config`.

    Passthrough rule: every source key survives unchanged — including
    `num_key_value_heads`, `architectures`, and the geometry lists — and no
    source key is removed or rewritten.
    """
    from .config import ModelConfig

    model = ModelConfig.from_dict(source_config)
    prior = source_config.get("quantization_config")
    declaration = quantization_config(model, mtp=mtp)
    if prior is not None and prior != declaration:
        raise ValueError("source config already declares a different quantization_config")
    result = dict(source_config)
    result["quantization_config"] = declaration
    if any(result[key] != source_config[key] for key in source_config if key != "quantization_config"):
        raise ValueError("export config must pass every source key through unchanged")
    return result


def validate_export(config, *, mtp=True):
    """Prove FLARE's generic parser accepts the export and nothing more is packed."""
    from .config import ModelConfig

    model = ModelConfig.from_dict(config)
    layers = quantized_layers(config)
    expected = set(quantized_source_names(model, mtp=mtp))
    if set(layers) != expected:
        raise ValueError(f"export packs the wrong source names: {sorted(set(layers) ^ expected)[:8]}")
    for name in layers:
        if expert_format(layers, name) != ("nvfp4", "a16"):
            raise ValueError(f"{name}: expected W4A16 NVFP4 experts")
        if any(fragment in name for fragment in UNQUANTIZED_FAMILIES):
            raise ValueError(f"{name}: do-not-quantize family reached the export map")
    return layers
