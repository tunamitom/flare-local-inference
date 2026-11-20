# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Phase 3A plan-4.1 official-package loader deltas (CPU-only).

Pins the three `official_format_deltas_deferred_to_phase3a` recorded verbatim in
`perf-campaign/OFFICIAL-P0-BUILD-REPORT.md` (from `official/p0-tiny-roundtrip.json`):

1. `config._smoke_relations` accepts `num_nextn_predict_layers == 0`
   (`draft_layers == 0`), with or without a bundled DFly2 drafter;
2. the production pin accepts official `hy_v4_vl` + nested `text_config`
   (rejecting drift) while preview `hy_v4` pinning stays intact;
3. `weights.checkpoint_names` covers the `model.dfly_layers.*` (82 keys) and
   `vit.*` (649 keys) families cleanly when present, fails closed on any
   unknown key, and `Hy4WeightLoader` carries explicit `dfly=` / `vision=`
   selection flags (plan 4.1).

The real-package tests gate on `/opt/models/Hy4-official-nvfp4-w4a16-mse`
(official P0 base package; the Q4 GPTQ package lands with the same layout) and
run strictly read-only. GPU/Triton compile smoke is deliberately NOT run here —
it is the named gate `GATE-P3A-GPU-LOAD` of PHASE3-PREP-REPORT.md.
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling test helpers

torch = pytest.importorskip("torch")

OFFICIAL_PACKAGE = Path("/opt/models/Hy4-official-nvfp4-w4a16-mse")
OFFICIAL_SNAPSHOT = Path("/storage/Hy4")

CPU = torch.device("cpu")


def _real_package_config():
    if not (OFFICIAL_PACKAGE / "config.json").is_file():
        pytest.skip("official P0 package not present on this host")
    return json.loads((OFFICIAL_PACKAGE / "config.json").read_text())


def _manifest_over(root: Path, names, meta_shard="model-00001.safetensors"):
    """The loader's manifest surface: a config.json dir + a name inventory."""
    return SimpleNamespace(snapshot=root, metadata_path=root,
                           tensors={name: SimpleNamespace(shard=meta_shard) for name in names})


def _full_inventory(config, packed=False):
    """Every key a schema-complete checkpoint of this config may carry."""
    from ignition.recipes.hy4.weights import checkpoint_names, dfly_checkpoint_names

    names = set(checkpoint_names(config, mtp=True, dfly=True, vision=True, packed=packed))
    if config.has_dfly:
        names |= set(dfly_checkpoint_names(config, include_ignored=True))
    return names


# ---------------------------------------------------------------------------
# Delta 1: `draft_layers == 0` is a legal config (plan 4.1)
# ---------------------------------------------------------------------------
def test_smoke_config_accepts_zero_draft_layers():
    from test_hy4_dfly2 import _tiny_official_config

    from ignition.recipes.hy4.config import ModelConfig

    cfg = _tiny_official_config()
    assert cfg["num_nextn_predict_layers"] == 0
    model = ModelConfig.from_dict(cfg, pinned=False)
    assert model.draft_layers == 0
    assert model.has_dfly and model.dfly_layers == 2

    # Zero draft layers without any bundled drafter is legal too: the smoke
    # path must accept `draft_layers == 0` outright (the previous gate raised
    # "Hy4 requires at least one MTP predictor layer" here).
    stripped = {k: v for k, v in cfg.items() if not k.startswith("dfly_")}
    text_stripped = {k: v for k, v in stripped["text_config"].items()
                     if not k.startswith("dfly_")}
    stripped["text_config"] = text_stripped
    plain = ModelConfig.from_dict(stripped, pinned=False)
    assert plain.draft_layers == 0 and not plain.has_dfly

    # The structural nonnegativity check survives (mutate top-level and nested
    # `text_config` together — a lone top-level edit trips the duplicate check).
    for bad_value in (-1, "0"):
        bad = dict(cfg, num_nextn_predict_layers=bad_value,
                   text_config=dict(cfg["text_config"], num_nextn_predict_layers=bad_value))
        with pytest.raises(ValueError, match="nonnegative"):
            ModelConfig.from_dict(bad, pinned=False)


def test_zero_draft_layers_packed_arithmetic():
    """`draft_layers == 0` -> 77 packed MoE prefixes / 59,136 matrices (plan 4.1)."""
    from test_hy4_dfly2 import _tiny_official_config

    from ignition.recipes.hy4 import contract
    from ignition.recipes.hy4.config import ModelConfig

    cfg = _tiny_official_config()
    model = ModelConfig.from_dict(cfg, pinned=False)
    assert model.token_record_bytes(draft=False) == model.token_record_bytes(draft=True)
    assert len(contract.quantized_source_names(model, mtp=True)) == model.layers - 1
    # The real 78-layer geometry yields exactly the P0 export census.
    assert (78 - 1 + 0) * 256 * 3 == 59136
    assert len(contract.packed_expert_tensor_names(
        ModelConfig.from_dict(_real_package_config(), pinned=True), mtp=True)) == 77 * 6


# ---------------------------------------------------------------------------
# Delta 2: production pin accepts `hy_v4_vl` + nested `text_config`
# ---------------------------------------------------------------------------
def test_pinned_accepts_official_vl_and_rejects_drift():
    from ignition.recipes.hy4.config import ModelConfig

    raw = _real_package_config()
    model = ModelConfig.load(OFFICIAL_PACKAGE, pinned=True)
    assert model.layers == 78 and model.draft_layers == 0
    assert model.has_dfly and model.dfly_layers == 5
    assert model.has_vision

    with pytest.raises(ValueError, match="architectures"):
        ModelConfig.from_dict(dict(raw, architectures=["HYV4ForCausalLM"]), pinned=True)
    with pytest.raises(ValueError, match="disagrees"):
        ModelConfig.from_dict(dict(raw, hidden_size=1), pinned=True)
    with pytest.raises(ValueError, match="dfly_num_hidden_layers"):
        ModelConfig.from_dict(dict(raw, dfly_num_hidden_layers=3), pinned=True)


def test_preview_pin_stays_intact():
    """The preview `hy_v4` production pin is untouched by the official deltas."""
    from ignition.recipes.hy4.config import ModelConfig

    raw = _real_package_config()
    # Preview's flat `hy_v4` form: the nested text config plus the structural
    # keys the preview checkpoint carried at top level.
    preview = dict(raw["text_config"])
    preview.update({
        "model_type": "hy_v4", "architectures": ["HYV4ForCausalLM"],
        "torch_dtype": "bfloat16", "dtype": "bfloat16",
        "transformers_version": "5.16.2", "num_nextn_predict_layers": 1,
        "mtp_loss_factor": 0.1, "use_mla": True, "use_dsa": True,
        "rope_parameters": {"rope_theta": 10000000.0, "rope_type": "default"},
    })
    model = ModelConfig.from_dict(preview, pinned=True)
    assert model.draft_layers == 1 and not model.has_dfly and not model.has_vision

    with pytest.raises(ValueError, match="architectures"):
        ModelConfig.from_dict(dict(preview, architectures=["HYV4VLForConditionalGeneration"]),
                              pinned=True)
    with pytest.raises(ValueError, match="num_nextn_predict_layers"):
        ModelConfig.from_dict(dict(preview, num_nextn_predict_layers=0), pinned=True)


# ---------------------------------------------------------------------------
# Delta 3: `checkpoint_names` covers the dfly (82) and vit (649) families
# ---------------------------------------------------------------------------
def test_vision_family_matches_real_package_census():
    from ignition.recipes.hy4.config import ModelConfig
    from ignition.recipes.hy4.weights import (
        checkpoint_names, dfly_checkpoint_names, vision_checkpoint_names,
    )

    _real_package_config()
    config = ModelConfig.load(OFFICIAL_PACKAGE, pinned=True)
    index = json.loads(
        (OFFICIAL_PACKAGE / "model.safetensors.index.json").read_text())["weight_map"]

    vision = vision_checkpoint_names(config)
    real_vit = {k for k in index if k.startswith("vit.")}
    real_dfly = {k for k in index if k.startswith("model.dfly_layers")}
    # The packed package index: 1,825 text non-expert + 462 packed fused +
    # 82 DFly2 + 649 vit = 3,018 keys (the raw /storage/Hy4 census is 2,710).
    assert len(index) == 3018
    assert len(vision) == 649 and vision == real_vit
    assert len(dfly_checkpoint_names(config, include_ignored=True)) == 82
    assert len(real_dfly) == 82

    active = checkpoint_names(config, mtp=True, dfly=True, packed=True)
    ignored = dfly_checkpoint_names(config, include_ignored=True) - dfly_checkpoint_names(config)
    assert len(active) == 1825 + 462 + 80
    assert set(index) == set(active) | ignored | vision
    assert checkpoint_names(config, mtp=True, dfly=True, vision=True,
                            packed=True) == set(active) | vision


def test_loader_flags_select_dfly_and_vision(tmp_path):
    """`Hy4WeightLoader(..., dfly=bool, vision=bool)` selection semantics."""
    from test_hy4_dfly2 import _tiny_official_config

    from ignition.recipes.hy4.config import ModelConfig
    from ignition.recipes.hy4.weights import Hy4WeightLoader, checkpoint_names

    cfg = _tiny_official_config()
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    config = ModelConfig.from_dict(cfg, pinned=False)
    manifest = _manifest_over(tmp_path, _full_inventory(config))

    def selected(**flags):
        loader = Hy4WeightLoader(manifest, CPU, pinned=False, **flags)
        return loader._selected_names()

    text_only = checkpoint_names(config, mtp=False, dfly=False)
    with_dfly = checkpoint_names(config, mtp=False, dfly=True)
    with_vision = checkpoint_names(config, mtp=False, dfly=False, vision=True)

    assert selected(mtp=False) == set(text_only)              # dfly follows mtp
    assert selected(mtp=False, dfly=True) == set(with_dfly)   # explicit drafter load
    assert selected(mtp=False, vision=True) == set(with_vision)
    assert selected(mtp=True) == set(checkpoint_names(config, mtp=True, dfly=True))
    assert not set(with_dfly) & set(with_vision) - set(text_only)


def test_loader_fails_closed_on_unknown_and_missing_keys(tmp_path):
    from test_hy4_dfly2 import _tiny_official_config

    from ignition.recipes.hy4.config import ModelConfig
    from ignition.recipes.hy4.weights import Hy4WeightLoader, checkpoint_names

    cfg = _tiny_official_config()
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    config = ModelConfig.from_dict(cfg, pinned=False)
    inventory = _full_inventory(config)

    for bogus in ("vit.adapter.proj.weight", "vit.layers.9.self_attn.q_proj.weight",
                  "model.dfly_layers.0.bogus.weight", "model.layers.0.unknown.weight"):
        manifest = _manifest_over(tmp_path, inventory | {bogus})
        with pytest.raises(ValueError, match="unexpected"):
            Hy4WeightLoader(manifest, CPU, mtp=False, pinned=False)._selected_names()

    victim = sorted(checkpoint_names(config, mtp=False, dfly=False))[0]
    manifest = _manifest_over(tmp_path, inventory - {victim})
    with pytest.raises(ValueError, match="missing"):
        Hy4WeightLoader(manifest, CPU, mtp=False, pinned=False)._selected_names()


# ---------------------------------------------------------------------------
# Real-package acceptance: the Phase-3 boot loads the official P0 package
# ---------------------------------------------------------------------------
def test_real_package_loader_acceptance():
    from ignition.recipes.hy4 import contract
    from ignition.recipes.hy4.config import ModelConfig
    from ignition.recipes.hy4.weights import (
        Hy4WeightLoader, checkpoint_names, verify_export_package,
    )
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint

    raw = _real_package_config()
    config = ModelConfig.load(OFFICIAL_PACKAGE, pinned=True)

    report = verify_export_package(OFFICIAL_PACKAGE, config)
    assert report["packed_matrices"] == 59136
    layers = contract.validate_export(raw)
    assert len(layers) == 77

    manifest = inspect_indexed_checkpoint(OFFICIAL_PACKAGE)
    text_only = Hy4WeightLoader(manifest, CPU, mtp=False, pinned=True)._selected_names()
    assert text_only == set(checkpoint_names(config, mtp=False, dfly=False, packed=True))
    assert len(text_only) == 1825 + 462                    # 2,287 packed text tensors

    with_dfly = Hy4WeightLoader(manifest, CPU, mtp=False, dfly=True,
                                pinned=True)._selected_names()
    assert len(with_dfly) == len(text_only) + 80            # + 80 active DFly2 tensors


def test_real_package_rank_headers_dry_run():
    """`construct_rank(..., headers_only=True)` over the real packed package."""
    from ignition.recipes.hy4.bootstrap import PreparationConfig, construct_rank
    from ignition.recipes.hy4.collectives import SimulatedGroup

    _real_package_config()
    built = construct_rank(
        str(OFFICIAL_PACKAGE),
        PreparationConfig(devices=tuple(range(8)), mtp=False, prefill_chunk_size=1024),
        rank=0, device=CPU, pinned=True, mtp=False,
        cache_rows=1, cache_capacity=128, headers_only=True, world_size=8,
        collectives=SimulatedGroup(8).collectives(0))
    assert built.weights.dfly is None          # text-only selection (dfly flag off)
    assert len(built.weights.used_names) == 2287
