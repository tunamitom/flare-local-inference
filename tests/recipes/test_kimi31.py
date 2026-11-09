# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Kimi K3.1 checkpoint identity, attention partition and complete history storage."""

from dataclasses import replace
import json
from pathlib import Path

import pytest

from ignition.components.parallel import TensorParallelShard
from ignition.recipes.kimi3.config import ModelConfig, DSparkConfig, LatentDSparkConfig
from ignition.recipes.kimi3.pool import continuation_layout, page_specs, page_bytes
from ignition.recipes.kimi31.recipe import RECIPE

FIXTURES = Path(__file__).parents[1] / "fixtures" / "kimi31"


@pytest.fixture
def raw():
    # Published configuration at moonshotai/Kimi-K3.1-Pro revision 6d0a0202956796543364d666ab19c6ad1a494ee1.
    return json.loads((FIXTURES / "config.json").read_text())


def test_layer_mapping_and_cache_accounting(raw):
    config = ModelConfig.from_dict(raw)
    config.validate_shard(TensorParallelShard(0, 16))
    assert config.layers.count("kda") == 69
    assert config.layers.count("dsa") == 24
    assert {i + 1 for i, kind in enumerate(config.layers) if kind == "dsa"} == set(raw["text_config"]["dsa_config"]["dsa_layers"])
    plain = RECIPE.plain_ranks[16]
    layout = continuation_layout(config, 16)
    assert page_bytes(config) == plain.page_bytes
    assert layout.row_bytes == plain.state_row_bytes
    for name, shape in page_specs(config):
        assert layout.region(name).shape == shape
    draft = DSparkConfig.from_dict(json.loads((FIXTURES / "draft-config.json").read_text()), config.text)
    assert isinstance(draft, LatentDSparkConfig)
    config = replace(config, draft=draft)
    draft.validate_shard(TensorParallelShard(0, 16))
    assert draft.auxiliary_layers == (75, 79, 83, 87, 91)
    assert draft.auxiliary_source == "residual_output"
    layout = continuation_layout(config, 16, speculative_tokens=7)
    assert layout.row_bytes == RECIPE.ranks[16].state_row_bytes
    assert page_bytes(config) == RECIPE.ranks[16].page_bytes
    assert len([name for name, _ in page_specs(config) if name.startswith("draft")]) == 5
    for name, shape in page_specs(config):
        assert layout.region(name).shape == shape


@pytest.mark.parametrize("field,value", [("index_topk", 512), ("index_rope_dim", 64),
    ("index_k_norm_type", "rmsnorm"), ("index_topk_freq", 2), ("dsa_layers", [4])])
def test_rejects_incompatible_sparse_attention(raw, field, value):
    raw["text_config"]["dsa_config"][field] = value
    with pytest.raises(ValueError, match="DSA|K3.1"):
        ModelConfig.from_dict(raw)


def test_dense_k3_cannot_silently_consume_sparse_checkpoint(raw):
    raw.update(model_type="kimi_k3", architectures=["KimiK3ForConditionalGeneration"])
    with pytest.raises(ValueError, match="DSA configuration"):
        ModelConfig.from_dict(raw)
    del raw["text_config"]["dsa_config"]
    config = ModelConfig.from_dict(raw)
    assert config.layers.count("mla") == 24
    assert not any(name.endswith(".index") for name, _ in page_specs(config))


@pytest.mark.parametrize("field,value", [("full_attention_causal", False), ("target_layer_ids", [93]),
    ("layer_types", ["sliding_attention"] * 5), ("vocab_size", 1)])
def test_draft_geometry_and_target_identity(raw, field, value):
    draft = json.loads((FIXTURES / "draft-config.json").read_text())
    draft[field] = value
    with pytest.raises(ValueError, match="DSpark"):
        DSparkConfig.from_dict(draft, raw["text_config"])


def test_recipe_preserves_image_capacity_and_sampling():
    assert RECIPE.max_images == 4
    assert RECIPE.sampling_defaults.top_p == .95
    assert RECIPE.sampling_defaults.temperature == 1.
    assert RECIPE.sampling_defaults.top_k == -1
    assert RECIPE.max_draft_tokens == 7


def test_gather_thread_tuning_keeps_other_kimi_collective_settings():
    from ignition.recipes.kimi3.recipe import RECIPE as KIMI3

    assert KIMI3.b12x_environment["B12X_PCIE_DCP_THREADS"] == "512"
    assert RECIPE.b12x_environment == {**KIMI3.b12x_environment, "B12X_PCIE_DCP_THREADS": "0"}
    assert RECIPE.b12x_environment["B12X_PCIE_DCP_BLOCK_LIMIT"] == "8"


def test_exl3_keeps_sparse_attention_and_bf16_projections(raw):
    raw["quantization_config"] = raw["text_config"]["quantization_config"] = {
        "quant_method": "exl3", "dense_format": "bf16",
        "exl3": {"manifest": "exl3-manifest.json"},
    }
    config = ModelConfig.from_dict(raw)
    config.validate_shard(TensorParallelShard(0, 8))
    assert config.layers.count("dsa") == 24
    prefix = "language_model.model.layers.1.block_sparse_moe"
    for projection in ("w1", "w3", "w2"):
        assert config.projection_format(f"{prefix}.experts.0.{projection}") == "exl3"
    assert config.projection_format(f"{prefix}.shared_experts.up_proj") == "bf16"
    assert config.projection_format("language_model.model.layers.3.self_attn.indexer.wq_b") == "bf16"


@pytest.mark.parametrize("fields", [
    {"dense_format": "mxfp8"}, {"exl3": None}, {"exl3": {"manifest": "other.json"}},
])
def test_exl3_rejects_undeclared_retained_storage(raw, fields):
    raw["text_config"]["quantization_config"] = {
        "quant_method": "exl3", "dense_format": "bf16",
        "exl3": {"manifest": "exl3-manifest.json"}, **fields,
    }
    with pytest.raises(ValueError, match="EXL3"):
        ModelConfig.from_dict(raw)


def test_tp8_costs_match_complete_history_layout(raw):
    config = ModelConfig.from_dict(raw)
    config.validate_shard(TensorParallelShard(0, 8))
    assert continuation_layout(config, 8).row_bytes == RECIPE.plain_ranks[8].state_row_bytes
    assert page_bytes(config) == RECIPE.plain_ranks[8].page_bytes
    draft = DSparkConfig.from_dict(json.loads((FIXTURES / "draft-config.json").read_text()), config.text)
    config = replace(config, draft=draft)
    assert continuation_layout(config, 8, speculative_tokens=7).row_bytes == RECIPE.ranks[8].state_row_bytes
    assert page_bytes(config) == RECIPE.ranks[8].page_bytes


def test_tp8_preparation_keeps_context_parallel_page_geometry():
    from ignition.recipes.kimi3.bootstrap import PreparationConfig

    config = PreparationConfig(devices=tuple(range(8)), prefill_chunk_size=256,
        max_active=1, vision=False, speculative_widths=())
    assert config.page_tokens == 1024
    assert config.preparation_pages == 3
    with pytest.raises(ValueError, match="distinct GPUs"):
        replace(config, devices=(0,) * 8)


@pytest.mark.parametrize("concurrency,expected", [(1, 3), (2, 3), (3, 5), (4, 5), (8, 9)])
def test_tuning_rows_cover_recurrence_without_snapshot_destinations(concurrency, expected):
    from ignition.components.capacity import request_capacity
    from ignition.recipes.kimi3.bootstrap import PreparationConfig

    config = PreparationConfig(devices=tuple(range(8)), max_active=concurrency, vision=False)
    assert config.tuning_state_rows == expected
    assert config.tuning_state_rows >= 1 + request_capacity(concurrency)
    assert config.tuning_state_rows >= 3  # Null, one prefill state and its checkpoint.


def test_exl3_rejects_shards_that_split_intermediate_hadamard_blocks(raw):
    raw["text_config"]["quantization_config"] = {"quant_method": "exl3", "dense_format": "bf16",
        "exl3": {"manifest": "exl3-manifest.json"}}
    config = ModelConfig.from_dict(raw)
    with pytest.raises(ValueError, match="aligned rank partitions"):
        config.validate_shard(TensorParallelShard(0, 16))


def test_prefill_fp8_transport_flag_states_select_the_reduction_channel(monkeypatch):
    """Both P3′ flag states: off keeps the exact BF16 twin's call unchanged, on swaps only the channel."""
    from ignition.recipes.kimi3 import bootstrap

    selected = []
    monkeypatch.setattr(bootstrap, "create_single_round_all_reduce_runtime",
        lambda group, device, *, max_tokens, hidden_size, ipc: selected.append(
            ("exact", max_tokens, hidden_size, ipc)) or "exact-channel")
    monkeypatch.setattr(bootstrap, "create_fp8_two_shot_all_reduce_runtime",
        lambda group, device, *, max_tokens, hidden_size: selected.append(
            ("fp8", max_tokens, hidden_size)) or "fp8-channel")
    config = bootstrap.PreparationConfig(devices=tuple(range(8)), prefill_chunk_size=256,
        max_active=1, vision=False, speculative_widths=())
    assert config.prefill_fp8_transport is False
    assert bootstrap.prefill_all_reduce_runtime(config, object(), object(),
        max_tokens=256, hidden_size=7168, ipc="peer-storage") == "exact-channel"
    assert selected == [("exact", 256, 7168, "peer-storage")]
    on = replace(config, prefill_fp8_transport=True)
    assert bootstrap.prefill_all_reduce_runtime(on, object(), object(),
        max_tokens=256, hidden_size=7168, ipc="peer-storage") == "fp8-channel"
    assert selected == [("exact", 256, 7168, "peer-storage"), ("fp8", 256, 7168)]
    with pytest.raises(ValueError, match="two, four or eight"):
        replace(config, devices=tuple(range(16)), prefill_fp8_transport=True)
