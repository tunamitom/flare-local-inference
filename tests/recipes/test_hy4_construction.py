# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""The Hy4 recipe skeleton constructs on CPU from the real checkpoint metadata.

No checkpoint tensors are read and no GPU is touched: the fixtures are the
pinned `tencent/Hy4-preview` `config.json` and `model.safetensors.index.json`
saved under `HY4_REF_DIR` (default `/home/tom/hy4/ref`).
"""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

REF = Path(os.environ.get("HY4_REF_DIR", "/home/tom/hy4/ref"))
PUBLISHED_BF16_PARAMS = 779_930_197_312
PUBLISHED_FP32_PARAMS = 30_795_421


@pytest.fixture(scope="module")
def source():
    config = REF / "bf16_config.json"
    index = REF / "idx.json"
    if not config.is_file() or not index.is_file():
        pytest.skip(f"Hy4 reference metadata not present under {REF}")
    return (json.loads(config.read_text()), json.loads(index.read_text())["weight_map"])


@pytest.fixture(scope="module")
def model(source):
    from ignition.recipes.hy4.config import ModelConfig

    return ModelConfig.from_dict(source[0])


def test_recipe_constructs_from_the_real_index_metadata(source, model):
    from ignition.recipes import resolve
    from ignition.recipes.hy4.recipe import RECIPE

    config, weight_map = source
    # Every geometry value in the pinned config validated during construction.
    assert (model.layers, model.hidden, model.vocabulary) == (78, 6144, 120832)
    assert model.full_indexer_layers == (0, 1, *range(5, 78, 4)) and len(model.shared_indexer_layers) == 57
    assert model.indexer_type(0) == model.indexer_type(77) == "full"
    assert model.indexer_type(2) == "shared" and model.indexer_type(78) == "full"  # 78 is the MTP block

    # The generated tensor inventory is exactly the real index's 2006 names.
    from ignition.recipes.hy4.weights import checkpoint_names

    names = checkpoint_names(model)
    assert names == frozenset(weight_map) and len(names) == 2006
    assert checkpoint_names(model, mtp=False) == frozenset(
        name for name in weight_map if not name.startswith("model.mtp_layers."))
    # IndexCache: index-key tensors exist for the 21 full layers and the MTP block only.
    indexers = {name for name in names if ".self_attn.indexer.wq_b.weight" in name}
    assert len(indexers) == 22 and all(model.indexer_type(int(name.split(".")[2])) == "full"
                                       for name in indexers if not name.startswith("model.mtp_layers"))

    # Parameter census reproduces the checkpoint's published counts.
    assert sum(model.parameter_census().values()) == PUBLISHED_BF16_PARAMS + PUBLISHED_FP32_PARAMS
    assert model.stored_parameter_counts() == (PUBLISHED_BF16_PARAMS, PUBLISHED_FP32_PARAMS)

    # The recipe registers, resolves, and declares both serving worlds.
    resolved = resolve("hy4-preview")
    assert resolved.recipe is RECIPE and resolved.checkpoint.alias == "hy4-preview"
    assert resolved.checkpoint.architecture == "HYV4ForCausalLM"
    assert resolved.checkpoint.revision == "705d81ee51566a186d645b74c974d642ef2828fe"
    assert RECIPE.tensor_parallel == (8, 16) and set(RECIPE.plain_ranks) == {8, 16}
    assert RECIPE.sampling_defaults.as_dict()["temperature"] == 0.9
    assert RECIPE.sampling_defaults.as_dict()["top_p"] == 1.0
    assert RECIPE.sampling_defaults.top_k == -1
    assert RECIPE.speculation.draft_tokens == 3 and RECIPE.speculation.token_capacity == 128
    assert RECIPE.context == 1048576 and RECIPE.context % RECIPE.page_tokens == 0
    # Cache cost arithmetic: 656 B latent records on 78 layers plus 132 B
    # index-key records on the 21 indexer layers (53,940 B per token).
    assert model.token_record_bytes() == 53940
    plain, spec = RECIPE.plain_ranks[8], RECIPE.ranks[8]
    assert plain.page_bytes == 128 * 53940 and spec.page_bytes == 128 * 54728
    assert plain.page_tokens == 128 * 8 and RECIPE.ranks[16].page_tokens == 128 * 16
    # The BF16 checkpoint packs to W4A16 NVFP4 at load: resident weights fit TP8.
    checkpoint = RECIPE.checkpoints[0]
    assert checkpoint.device_bytes < checkpoint.stored_bytes
    assert RECIPE.weights_bytes(checkpoint, 8, RECIPE.speculation) < 64 << 30


def test_bootstrap_configuration_constructs_on_cpu(source, model):
    from ignition.recipes.hy4.bootstrap import PreparationConfig

    config = PreparationConfig(devices=tuple(range(8)), context=1048576, prefill_chunk_size=4096,
                               max_active=32, mtp=True, speculative_widths=(1, 2, 3, 4),
                               speculative_token_capacity=128)
    assert config.mtp and not config.mtp_only and config.speculative_widths == (1, 2, 3, 4)
    assert config.page_tokens == 128 * 8 and config.context % 128 == 0
    assert config.collective_tuning == "static" and config.collective_ladder is None

    only = PreparationConfig(devices=(0,), mtp=True, mtp_only=True, speculative_widths=(1, 2, 3, 4))
    assert only.mtp_only
    from dataclasses import replace

    for replacement in (dict(mtp=False), dict(speculative_widths=(1,), speculative_fixed_width=None),
                        dict(speculative_fixed_width=1)):
        with pytest.raises(ValueError, match="--mtp-only requires enabled learned speculation"):
            replace(config, mtp_only=True, **replacement)
    with pytest.raises(ValueError, match="three draft tokens"):
        PreparationConfig(devices=(0,), mtp=True, speculative_widths=(1, 2, 3, 4, 5))
    with pytest.raises(ValueError, match="distinct CUDA devices"):
        PreparationConfig(devices=(0, 1, 2, 3))


def test_geometry_validation_rejects_config_drift(source):
    from ignition.recipes.hy4.config import ModelConfig

    def drifted(**changes):
        config = dict(source[0], **changes)
        with pytest.raises(ValueError):
            ModelConfig.from_dict(config)

    indexer = list(source[0]["indexer_types"])
    indexer[3] = "full"
    mlp = list(source[0]["mlp_layer_types"])
    mlp[0] = "sparse"
    drifted(indexer_types=indexer)
    drifted(mlp_layer_types=mlp)
    for name, wrong in (("hidden_size", 4096), ("num_hidden_layers", 61), ("num_experts_per_tok", 4),
                        ("n_routed_experts", 128), ("moe_intermediate_size", 1408), ("index_topk", 1024),
                        ("index_n_heads", 16), ("hc_mult", 2), ("hc_magnitude", 1.0), ("hc_eps", 1e-5),
                        ("swiglu_limit", 8.0), ("learnable_sink", False), ("gated_mla", False),
                        ("num_nextn_predict_layers", 3), ("routed_scaling_factor", 2.8),
                        ("q_lora_rank", 1536), ("kv_lora_rank", 513), ("qk_rope_head_dim", 32),
                        ("vocab_size", 128000), ("intermediate_size", 14336),
                        ("rope_parameters", {"rope_theta": 500000.0, "rope_type": "default"}),
                        ("architectures", ["HYV3ForCausalLM"]), ("tie_word_embeddings", True)):
        drifted(**{name: wrong})


def test_export_contract_encodes_the_generic_modelopt_parser(source):
    from ignition.preparation.quantization import expert_format, linear_format, quantized_layers
    from ignition.recipes.hy4 import contract
    from ignition.recipes.hy4.config import ModelConfig

    model = ModelConfig.from_dict(source[0])
    export = contract.export_config(source[0])
    # Passthrough rule: the source config is unchanged except for the declaration.
    assert all(export[key] == source[0][key] for key in source[0] if key != "quantization_config")
    assert export["num_key_value_heads"] == source[0]["num_key_value_heads"] == 8
    assert export["indexer_types"] == source[0]["indexer_types"]
    assert set(export) == set(source[0]) | {"quantization_config"}

    # The generic ModelOpt parser accepts exactly this form.
    declaration = export["quantization_config"]
    assert (declaration["quant_method"], declaration["quant_algo"]) == ("modelopt", "MIXED_PRECISION")
    assert declaration["kv_cache_quant_algo"] is None
    layers = quantized_layers(export)
    assert set(layers) == set(contract.quantized_source_names(model)) and len(layers) == 78
    for name in layers:
        assert expert_format(layers, name) == ("nvfp4", "a16")
    assert contract.validate_export(export) == layers

    # The do-not-quantize floor never reaches the map.
    assert not any(fragment in name for name in layers for fragment in contract.UNQUANTIZED_FAMILIES)
    assert "model.layers.0.mlp" not in {name.rsplit(".", 2)[0] for name in layers}
    # A regenerated or plain-NVFP4 config is rejected by the parser, as encoded.
    bad = dict(export, quantization_config=dict(declaration, quant_algo="NVFP4"))
    with pytest.raises(ValueError, match="MIXED_PRECISION"):
        quantized_layers(bad)
    tampered = dict(source[0], num_key_value_heads=4)
    with pytest.raises(ValueError):
        contract.validate_export(contract.export_config(tampered))
    assert linear_format(layers, "model.layers.1.mlp.experts") == "nvfp4"


def test_fused_experts_split_into_loader_stacks():
    torch = pytest.importorskip("torch")

    from ignition.recipes.hy4.weights import Hy4WeightLoader

    count, inter, hidden = 3, 4, 2
    config = SimpleNamespace(text={"moe_intermediate_size": inter, "n_routed_experts": count, "swiglu_limit": 10.0},
                             hidden=hidden, experts=count)
    fused = torch.arange(count * 2 * inter * hidden, dtype=torch.bfloat16).reshape(count, 2 * inter, hidden)
    down = torch.zeros(count, hidden, inter, dtype=torch.bfloat16)
    loader = Hy4WeightLoader.__new__(Hy4WeightLoader)
    loader.config = config
    loader._sources = {"model.layers.1.mlp.experts.gate_up_proj": fused,
                       "model.layers.1.mlp.experts.down_proj": down}
    stacks = loader.expert_stacks("model.layers.1.mlp.experts")
    gate = stacks["model.layers.1.mlp.experts.gate_proj.weight"]
    up = stacks["model.layers.1.mlp.experts.up_proj.weight"]
    # The reference forward splits `gate, up = gate_up.chunk(2, dim=-1)` on the
    # GEMM output: the fused weight's first intermediate block is the gate.
    assert gate.shape == up.shape == (count, inter, hidden)
    assert torch.equal(gate, fused[:, :inter, :]) and torch.equal(up, fused[:, inter:, :])
    assert torch.equal(fused[1, :inter], gate[1]) and torch.equal(fused[1, inter:], up[1])
    assert stacks["model.layers.1.mlp.experts.down_proj.weight"] is down
    # The loader's canonical names are exactly what packing consumes.
    assert set(stacks) == {"model.layers.1.mlp.experts.gate_proj.weight",
                           "model.layers.1.mlp.experts.up_proj.weight",
                           "model.layers.1.mlp.experts.down_proj.weight"}


def test_fp32_policy_matches_the_declared_families():
    from ignition.recipes.hy4.weights import kept_fp32

    kept = ["model.layers.3.hc_attn_layer.hc_pre.hc_fn", "model.layers.3.hc_attn_layer.hc_pre.hc_base",
            "model.layers.3.hc_attn_layer.hc_pre.hc_scale", "model.hc_head.hc_head_fn",
            "model.hc_head.hc_head_base", "model.hc_head.hc_head_scale",
            "model.layers.5.self_attn.learnable_sink_param",
            "model.layers.5.self_attn.indexer.weights_proj.weight",
            "model.layers.5.self_attn.indexer.k_norm.weight",
            "model.layers.5.self_attn.indexer.k_norm.bias",
            "model.layers.5.mlp.gate.e_score_correction_bias", "lm_head.weight",
            "model.mtp_layers.0.self_attn.learnable_sink_param"]
    plain = ["model.embed_tokens.weight", "model.norm.weight", "model.layers.5.self_attn.o_proj.weight",
             "model.layers.5.self_attn.linear_gate.weight", "model.layers.5.mlp.gate.weight",
             "model.layers.5.mlp.shared_experts.up_proj.weight", "model.layers.5.mlp.experts.gate_up_proj",
             "model.layers.0.mlp.gate_proj.weight", "model.layers.5.input_layernorm.weight",
             "model.layers.5.self_attn.indexer.wq_b.weight", "model.mtp_layers.0.eh_proj.weight"]
    assert all(kept_fp32(name) for name in kept)
    assert not any(kept_fp32(name) for name in plain)
