# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Model resolution and the recipe declarations it reads."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from ignition.errors import FetchError, UsageError
from ignition.recipes import aliases, recipes, resolve

ROOT = Path(__file__).resolve().parents[2]
HUB = Path.home() / ".cache/huggingface/hub"


def test_declarations_import_without_model_dependencies():
    result = subprocess.run(
        [sys.executable, "-c", "import sys; from ignition.recipes import aliases; from ignition import planning;"
         " import ignition.cli.preparation; assert aliases() == ('qwen3.8-flash', 'qwen3.8-27b', 'qwen3.8-27b-qad', 'glm-5.3-flash', 'deepseek-v4.1-flash', 'deepseek-v4-flash', 'mimo-v2.6-pro-mopd', 'kimi-k3-mxfp4-csf', 'kimi-k3', 'kimi-k3-dspark', 'kimi-k3.1-pro', 'step5-preview-nvfp4', 'nemotron-bolt-24gb', 'minimax-m3-1-flash', 'gemma-4-e2b', 'gemma-4-e4b', 'gemma-4-12b', 'gemma-4-26b-a4b', 'gemma-4-31b', 'gemma-4-e4b-nvfp4', 'gemma-4-12b-nvfp4', 'gemma-4-26b-a4b-nvfp4', 'gemma-4-31b-nvfp4', 'muse-spark-open-nvfp4');"
         " assert not {'torch', 'transformers', 'flare_heliopause'} & set(sys.modules)"],
        cwd=ROOT, check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_alias_and_repository_resolve_to_the_declared_checkpoint(tmp_path):
    alias = resolve("qwen3.8-flash")
    assert alias.directory is None and alias.recipe.name == "qwen38" and alias.source == "b9ce18a6"
    assert resolve(alias.checkpoint.repository).checkpoint is alias.checkpoint
    assert resolve(f"{alias.checkpoint.repository}@{alias.checkpoint.revision}").checkpoint is alias.checkpoint
    assert alias.model_id == alias.checkpoint.repository
    assert alias.snapshot(tmp_path) == (
        tmp_path / "models--nvidia--Qwen3.8-Flash-Next-FP4-Local" / "snapshots"
        / "b9ce18a6887e18c089fdea8509a33e63429a3300")


def write_checkpoint(directory, architecture, total_size):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps({"architectures": [architecture]}))
    (directory / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": total_size}}))


def write_exl3_checkpoint(directory, *, declared_bytes=600):
    write_checkpoint(directory, "KimiK31ForConditionalGeneration", 100)
    config = {"architectures": ["KimiK31ForConditionalGeneration"],
              "quantization_config": {"quant_method": "exl3", "dense_format": "bf16",
                  "exl3": {"manifest": "exl3-manifest.json", "weight_file_bytes": declared_bytes}}}
    (directory / "config.json").write_text(json.dumps(config))
    (directory / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": 100}, "weight_map": {"norm": "retained.safetensors"}}))
    (directory / "retained.safetensors").write_bytes(b"\0" * 100)
    (directory / "exl3-manifest.json").write_text(json.dumps({
        "kind": "exl3-manifest", "schema": "exl3-v1",
        "layers": {"1": {"file": "exl3-layer-00001.safetensors"}}}))


def test_exl3_resolution_counts_expert_files_outside_retained_index(tmp_path):
    write_exl3_checkpoint(tmp_path)
    model = resolve(str(tmp_path))
    assert model.recipe.name == "kimi31"
    assert model.checkpoint.stored_bytes == 600
    assert model.checkpoint.device_bytes - model.checkpoint.draft_bytes == 600
    assert not model.cached(HUB)
    (tmp_path / "exl3-layer-00001.safetensors").write_bytes(b"\0" * 500)
    assert model.cached(HUB)


@pytest.mark.parametrize("value", [True, 0, -1, 50, "600"])
def test_exl3_rejects_invalid_total_weight_size(tmp_path, value):
    write_exl3_checkpoint(tmp_path, declared_bytes=value)
    with pytest.raises(UsageError, match="weight_file_bytes"):
        resolve(str(tmp_path))


def test_exl3_local_size_without_hint_uses_every_layer_file(tmp_path):
    write_exl3_checkpoint(tmp_path)
    config = json.loads((tmp_path / "config.json").read_text())
    del config["quantization_config"]["exl3"]["weight_file_bytes"]
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "exl3-layer-00001.safetensors").write_bytes(b"\0" * 500)
    assert resolve(str(tmp_path)).checkpoint.stored_bytes == 600


def test_kimi_exl3_reports_conversion_savings_without_enabling_conversion(tmp_path):
    write_exl3_checkpoint(tmp_path, declared_bytes=797_000_000_000)
    path = tmp_path / "config.json"
    raw = json.loads((ROOT / "tests/fixtures/kimi31/config.json").read_text())
    raw["quantization_config"] = raw["text_config"]["quantization_config"] = {
        "quant_method": "exl3", "dense_format": "bf16",
        "exl3": {"manifest": "exl3-manifest.json", "weight_file_bytes": 797_000_000_000}}
    path.write_text(json.dumps(raw))
    model = resolve(str(tmp_path))
    original = model.recipe.weights_bytes(model.checkpoint, 8, None)
    converted = model.recipe.weights_bytes(model.checkpoint, 8, None, online_mxfp8=True)
    assert original == 99_625_000_000
    # Credit is bounded below the measured worst-rank conversion rate (see _exl3_mxfp8_savings).
    assert 5 << 30 < original - converted < 6 << 30
    assert model.checkpoint.stored_bytes == 797_000_000_000
    assert raw["text_config"]["quantization_config"]["dense_format"] == "bf16"


def test_kimi_conversion_estimate_counts_sharded_mla_and_duplicated_latent_views(tmp_path):
    from ignition.recipes import _exl3_mxfp8_savings

    write_exl3_checkpoint(tmp_path, declared_bytes=797_000_000_000)
    raw = json.loads((ROOT / "tests/fixtures/kimi31/config.json").read_text())
    raw["quantization_config"] = raw["text_config"]["quantization_config"] = {
        "quant_method": "exl3", "dense_format": "bf16",
        "exl3": {"manifest": "exl3-manifest.json", "weight_file_bytes": 797_000_000_000}}
    (tmp_path / "config.json").write_text(json.dumps(raw))
    c = raw["text_config"]
    hidden, layers = c["hidden_size"], c["num_hidden_layers"]
    kda = len(c["linear_attn_config"]["kda_layers"])
    channels = c["linear_attn_config"]["num_heads"] * 128
    mla_channels = c["num_attention_heads"] * 128
    base = hidden * (3 * kda * channels + 2 * (kda * channels + (layers-kda) * mla_channels)
        + 3 * (layers-c["first_k_dense_replace"]) * c["num_shared_experts"] * c["moe_intermediate_size"])
    mla = (layers-kda) * (c["q_lora_rank"] * c["num_attention_heads"]
        * (c["qk_nope_head_dim"] + c["qk_rope_head_dim"])
        + hidden * (c["q_lora_rank"] + c["kv_lora_rank"] + c["qk_rope_head_dim"]))
    latent_pair = (layers-c["first_k_dense_replace"]) * 4 * hidden * c["routed_expert_hidden_size"]
    assert _exl3_mxfp8_savings(tmp_path) == 2 * (base+mla) * 475 // 1000 + latent_pair // 16
    model = resolve(str(tmp_path))
    # The estimate must not silently enable conversion or alter source size.
    assert model.recipe.weights_bytes(model.checkpoint, 8, None) == 99_625_000_000
    assert model.checkpoint.stored_bytes == 797_000_000_000


def test_exl3_cache_does_not_accept_a_layer_path_outside_checkpoint(tmp_path):
    write_exl3_checkpoint(tmp_path)
    model = resolve(str(tmp_path))
    manifest = json.loads((tmp_path / "exl3-manifest.json").read_text())
    manifest["layers"]["1"]["file"] = "../other.safetensors"
    (tmp_path / "exl3-manifest.json").write_text(json.dumps(manifest))
    assert not model.cached(HUB)


def test_a_directory_is_served_by_the_recipe_for_its_architecture(tmp_path):
    qualified = resolve("glm-5.3-flash").checkpoint
    write_checkpoint(tmp_path / "export", "Glm5NextForConditionalGeneration", qualified.stored_bytes // 2)
    local = resolve(str(tmp_path / "export"))
    assert local.recipe.name == "glm53" and local.checkpoint.alias == "glm-5.3-flash"
    assert local.directory == (tmp_path / "export").resolve() and local.snapshot(HUB) == local.directory
    assert local.source == str(local.directory) and local.model_id == "export"
    assert local.checkpoint not in local.recipe.checkpoints
    assert local.checkpoint.stored_bytes == qualified.stored_bytes // 2
    assert local.checkpoint.device_bytes == round(qualified.device_bytes / 2)
    write_checkpoint(tmp_path / "other", "Glm5NextForConditionalGeneration", qualified.stored_bytes)
    assert resolve(str(tmp_path / "other")).checkpoint.revision != local.checkpoint.revision


@pytest.fixture
def hub(monkeypatch, tmp_path):
    """Serve repositories from a dictionary of {repository: (commit, architecture, total size)}."""
    import huggingface_hub
    from huggingface_hub.errors import RepositoryNotFoundError

    repositories, requests = {}, []

    def download(repository, filename, *, revision=None, cache_dir=None):
        requests.append((repository, filename, revision, cache_dir))
        if repository not in repositories:
            missing = RepositoryNotFoundError.__new__(RepositoryNotFoundError)
            Exception.__init__(missing, f"{repository} not found")
            raise missing
        commit, architecture, total_size = repositories[repository]
        if isinstance(commit, Exception):
            raise commit
        snapshot = tmp_path / "hub" / f"models--{repository.replace('/', '--')}" / "snapshots" / (revision or commit)
        write_checkpoint(snapshot, architecture, total_size)
        return str(snapshot / filename)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    return repositories, requests


def test_another_repository_is_served_by_the_recipe_for_its_architecture(hub, tmp_path):
    repositories, requests = hub
    qualified = resolve("qwen3.8-flash").checkpoint
    repositories["lab/qwen-export"] = ("c0ffee" * 6 + "abcd", "Qwen4ExpForConditionalGeneration", 90 << 30)
    model = resolve("lab/qwen-export", tmp_path / "hub")
    assert model.recipe.name == "qwen38" and model.checkpoint.alias == "qwen3.8-flash"
    assert (model.checkpoint.repository, model.checkpoint.revision) == ("lab/qwen-export", "c0ffee" * 6 + "abcd")
    assert model.source == "lab/qwen-export@c0ffeec0" and model.model_id == "lab/qwen-export"
    assert model.checkpoint.stored_bytes == 90 << 30
    assert model.checkpoint.mapped_host_bytes == round(qualified.mapped_host_bytes * (90 << 30) / qualified.stored_bytes)
    assert requests[0] == ("lab/qwen-export", "config.json", None, tmp_path / "hub")
    pinned = resolve("lab/qwen-export@0123abcd", tmp_path / "hub")
    assert pinned.checkpoint.revision == "0123abcd" and requests[-1][2] == "0123abcd"


def test_a_different_revision_of_a_declared_repository_is_not_the_pinned_checkpoint(hub, tmp_path):
    repositories, _ = hub
    qualified = resolve("qwen3.8-flash").checkpoint
    repositories[qualified.repository] = ("0" * 40, "Qwen4ExpForConditionalGeneration", qualified.stored_bytes)
    model = resolve(f"{qualified.repository}@feedbeef", tmp_path / "hub")
    assert model.checkpoint is not qualified and model.checkpoint.revision == "feedbeef"
    assert model.source == f"{qualified.repository}@feedbeef"


@pytest.mark.parametrize("name", ["qwen", "Qwen3.8-Flash", "qwen3.8-flash@main"])
def test_unknown_aliases_list_the_supported_aliases(name):
    with pytest.raises(UsageError, match="supported models: " + ", ".join(aliases())):
        resolve(name)


def test_repositories_that_are_missing_unsupported_or_unreachable_are_rejected(hub, tmp_path):
    repositories, _ = hub
    with pytest.raises(UsageError, match="lab/missing is not a safetensors checkpoint on Hugging Face"):
        resolve("lab/missing", tmp_path / "hub")
    repositories["lab/llama"] = ("1" * 40, "LlamaForCausalLM", 1 << 30)
    with pytest.raises(UsageError, match="no recipe supports architecture"):
        resolve("lab/llama", tmp_path / "hub")
    repositories["lab/offline"] = (OSError("network unreachable"), None, None)
    with pytest.raises(FetchError, match="cannot read lab/offline from Hugging Face: network unreachable"):
        resolve("lab/offline", tmp_path / "hub")


def test_directories_without_a_supported_architecture_are_usage_errors(tmp_path):
    with pytest.raises(UsageError, match="no readable config.json architecture"):
        resolve(str(tmp_path))
    (tmp_path / "config.json").write_text(json.dumps({"architectures": ["LlamaForCausalLM"]}))
    with pytest.raises(UsageError, match="no recipe supports architecture"):
        resolve(str(tmp_path))


def test_cached_requires_the_index_and_every_tensor_file(tmp_path):
    model = resolve("glm-5.3-flash")
    snapshot = model.snapshot(tmp_path)
    assert not model.cached(tmp_path)
    (snapshot / "tensors").mkdir(parents=True)
    (snapshot / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": model.checkpoint.stored_bytes},
        "weight_map": {"a": "tensors/one.safetensors", "b": "tensors/two.safetensors"}}))
    (snapshot / "tensors/one.safetensors").touch()
    assert not model.cached(tmp_path)
    (snapshot / "tensors/two.safetensors").touch()
    assert model.cached(tmp_path)
    model.verify(tmp_path)


def cache_checkpoint(snapshot, checkpoint):
    write_checkpoint(snapshot, checkpoint.architecture, checkpoint.stored_bytes)
    (snapshot / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": checkpoint.stored_bytes},
        "weight_map": {"a": "one.safetensors", "b": "two.safetensors"}}))
    for name in ("one.safetensors", "two.safetensors"):
        (snapshot / name).touch()


@pytest.mark.parametrize("exact_state", ["absent", "partial", "complete"])
def test_cached_snapshot_matches_repository_case_and_prefers_complete_shards(tmp_path, exact_state):
    model = resolve("deepseek-v4.1-flash")
    checkpoint = model.checkpoint
    exact = model.snapshot(tmp_path)
    alternate = (tmp_path / f"models--{checkpoint.repository.swapcase().replace('/', '--')}"
                 / "snapshots" / checkpoint.revision)
    cache_checkpoint(alternate, checkpoint)
    if exact_state != "absent":
        cache_checkpoint(exact, checkpoint)
        if exact_state == "partial":
            (exact / "two.safetensors").unlink()
    assert model.snapshot(tmp_path) == (exact if exact_state == "complete" else alternate)
    assert model.cached(tmp_path)
    model.verify(tmp_path)
    assert model.directory is None and model.checkpoint is checkpoint


@pytest.mark.parametrize("mismatch", ["revision", "repository", "missing_shard", "invalid_index"])
def test_cache_lookup_requires_the_requested_repository_revision_and_all_shards(tmp_path, mismatch):
    model = resolve("deepseek-v4.1-flash")
    checkpoint = model.checkpoint
    exact = model.snapshot(tmp_path)
    repository = checkpoint.repository.swapcase() + ("-other" if mismatch == "repository" else "")
    revision = "0" * 40 if mismatch == "revision" else checkpoint.revision
    alternate = tmp_path / f"models--{repository.replace('/', '--')}" / "snapshots" / revision
    cache_checkpoint(alternate, checkpoint)
    if mismatch == "missing_shard":
        (alternate / "two.safetensors").unlink()
    elif mismatch == "invalid_index":
        (alternate / "model.safetensors.index.json").write_text("{")
    assert model.snapshot(tmp_path) == exact
    assert not model.cached(tmp_path)


@pytest.mark.parametrize("pinned", [False, True])
def test_declared_repository_casing_preserves_the_pinned_checkpoint_without_hub_requests(hub, tmp_path, pinned):
    _, requests = hub
    checkpoint = resolve("deepseek-v4.1-flash").checkpoint
    name = checkpoint.repository.swapcase()
    if pinned:
        name += f"@{checkpoint.revision}"
    model = resolve(name, tmp_path)
    assert model.checkpoint is checkpoint
    assert requests == []


def test_local_export_size_is_validated_by_recipe(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"architectures": ["Qwen4ExpForConditionalGeneration"]}))
    model = resolve(str(tmp_path))
    model.verify(HUB)
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": 7}}))
    model.verify(HUB)


def test_named_checkpoint_allows_different_export_size(tmp_path):
    model = resolve("qwen3.8-flash")
    snapshot = model.snapshot(tmp_path)
    snapshot.mkdir(parents=True)
    (snapshot / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": 7}}))
    model.verify(tmp_path)


@pytest.mark.parametrize("size", [None, 0, -1, True, "7"])
def test_local_export_requires_valid_size(tmp_path, size):
    (tmp_path / "config.json").write_text(json.dumps({"architectures": ["Qwen4ExpForConditionalGeneration"]}))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": size}}))
    with pytest.raises(UsageError, match="invalid safetensors total_size"):
        resolve(str(tmp_path)).verify(HUB)


def test_declared_ranges_are_consistent():
    assert len(set(aliases())) == len(aliases())
    for recipe in recipes():
        assert 1 <= recipe.concurrency <= recipe.max_concurrency <= 64
        assert recipe.context % recipe.page_tokens == 0
        assert recipe.tensor_parallel and all(type(world) is int and world > 0 for world in recipe.tensor_parallel)
        for checkpoint in recipe.checkpoints:
            assert checkpoint.revision == "local" if Path(checkpoint.repository).is_absolute() else len(checkpoint.revision) == 40
            assert 0 <= checkpoint.draft_bytes <= checkpoint.device_bytes
            # A draft model may have its own checkpoint index.
            assert checkpoint.device_bytes - checkpoint.draft_bytes + checkpoint.mapped_host_bytes <= checkpoint.stored_bytes


def test_local_checkpoint_alias_uses_actual_directory(tmp_path, monkeypatch):
    from dataclasses import replace
    import ignition.recipes as catalog
    recipe = next(r for r in recipes() if r.name == "kimi3")
    checkpoint = replace(recipe.checkpoints[0], repository=str(tmp_path))
    monkeypatch.setattr(catalog, 'recipes', lambda: (replace(recipe, checkpoints=(checkpoint,)),))
    (tmp_path / 'config.json').write_text(json.dumps({'architectures': [checkpoint.architecture]}))
    model = resolve(checkpoint.alias)
    assert model.directory == tmp_path and model.snapshot(HUB) == tmp_path
    assert model.checkpoint.repository == str(tmp_path)


def test_alias_preserves_selected_checkpoint_when_directory_is_shared(tmp_path, monkeypatch):
    from dataclasses import replace
    import ignition.recipes as catalog

    recipe = next(r for r in recipes() if r.name == "kimi3")
    checkpoints = tuple(replace(c, repository=str(tmp_path)) for c in recipe.checkpoints[:2])
    recipe = replace(recipe, checkpoints=checkpoints)
    monkeypatch.setattr(catalog, "recipes", lambda: (recipe,))
    for checkpoint in checkpoints:
        model = resolve(checkpoint.alias)
        assert model.recipe is recipe and model.checkpoint is checkpoint
        assert model.directory == tmp_path
    assert resolve(str(tmp_path)).checkpoint is checkpoints[0]
