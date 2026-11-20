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

Packed expert storage (the export writes exactly these names and shapes; the
loader consumes exactly them through `load_fp4_experts`, slicing each rank's
bytes bit-exactly from the checkpoint). For every fused prefix `P` of
`quantized_source_names` with `e` experts, intermediate width `mi` and hidden
width `h`:

| tensor | dtype | shape | note |
|---|---|---|---|
| `P.gate_up_proj.weight` | uint8 | (e, 2*mi, h//2) | packed NVFP4, rows [gate; up] |
| `P.gate_up_proj.weight_scale` | float8_e4m3fn | (e, 2*mi, h//16) | block scales, group 16 |
| `P.gate_up_proj.weight_scale_2` | float32 | (e,) | max-scaling gain, shared gate/up |
| `P.down_proj.weight` | uint8 | (e, h, mi//2) | packed NVFP4 |
| `P.down_proj.weight_scale` | float8_e4m3fn | (e, h, mi//16) | block scales, group 16 |
| `P.down_proj.weight_scale_2` | float32 | (e,) | max-scaling gain |

The row order keeps the BF16 checkpoint's fused semantics (`gate_up.chunk(2)`:
first block gate, second up), and `weight_scale_2` is a multiplier gain in
ModelOpt's sense (`amax / (448 * 6)`). There is no `input_scale`: the package
is A16 everywhere (A4 is a separate speed lever with its own loss measurement).

The packed W4A16 experts need no external quantization step on the BF16
route: the loader packs the BF16 checkpoint's experts at load (`weights.py`),
and the export writes the same packing out under this contract.
"""

from ignition.preparation.quantization import expert_format, quantized_layers

# Expert packing the export writes and the loader accepts: maximum-scale
# ModelOpt NVFP4, activations in 16 bits (`W4A16_NVFP4` -> ("nvfp4", "a16")).
EXPERT_ALGORITHM = {"quant_algo": "W4A16_NVFP4", "group_size": 16}

# The fused expert projections of one packed prefix, in the export's names.
PACKED_PROJECTIONS = ("gate_up_proj", "down_proj")

# Suffixes each packed projection carries. `weight_scale_2` is the multiplier
# gain; there is deliberately no `input_scale` (A16 package).
PACKED_SUFFIXES = ("weight", "weight_scale", "weight_scale_2")

# Families that must never appear in `quantized_layers` (do-not-quantize floor).
UNQUANTIZED_FAMILIES = (
    "embed_tokens", "lm_head", "hc_pre", "hc_head", "linear_gate", "learnable_sink_param",
    "mlp.gate", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "mlp.shared_experts",
    "self_attn", "indexer", "layernorm", "norm", "dfly_layers",
    # Phase 4 (VISION-DESIGN section 4.1 item 2): the `vit.*` vision tower is
    # strictly BF16 unquantized (649 tensors) — contract validation refuses a
    # quantized vision weight fail-closed.
    "vit",
)


def quantized_source_names(config, *, mtp=True):
    """Source names of the packed stacks: one fused expert prefix per MoE block."""
    names = tuple(f"model.layers.{layer}.mlp.experts" for layer in range(1, config.layers))
    return names + (tuple(f"model.mtp_layers.{index}.mlp.experts" for index in range(config.draft_layers))
                    if mtp else ())


def packed_expert_tensor_names(config, *, mtp=True):
    """Every packed expert tensor name the export writes for the fused prefixes."""
    names = set()
    for prefix in quantized_source_names(config, mtp=mtp):
        names |= {f"{prefix}.{projection}.{suffix}"
                  for projection in PACKED_PROJECTIONS for suffix in PACKED_SUFFIXES}
    return frozenset(names)


def packed_expert_geometry(config, prefix):
    """(dtype, shape) per export tensor name of one fused expert prefix.

    `prefix` must belong to `quantized_source_names`; the geometry is the
    packed checkpoint storage the loader slices (see the module table).
    """
    import torch

    if prefix not in quantized_source_names(config):
        raise ValueError(f"{prefix}: not a packed fused expert prefix of the export")
    e, mi, h = config.experts, config.text["moe_intermediate_size"], config.hidden
    return {
        f"{prefix}.gate_up_proj.weight": (torch.uint8, (e, 2 * mi, h // 2)),
        f"{prefix}.gate_up_proj.weight_scale": (torch.float8_e4m3fn, (e, 2 * mi, h // 16)),
        f"{prefix}.gate_up_proj.weight_scale_2": (torch.float32, (e,)),
        f"{prefix}.down_proj.weight": (torch.uint8, (e, h, mi // 2)),
        f"{prefix}.down_proj.weight_scale": (torch.float8_e4m3fn, (e, h, mi // 16)),
        f"{prefix}.down_proj.weight_scale_2": (torch.float32, (e,)),
    }


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
