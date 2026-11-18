# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Hy4 CSF wiring and the corrected per-rank weight arithmetic.

CPU only: no checkpoint tensors are read. The tests reproduce the recipe's
declared byte figures from the parameter census independently, check the plan's
anchors (the replication gap per rank at TP8 — the pre-`kv_b` ~2.9 GB plus the
`kv_b` two-wirings' 1.87 GiB, the 55.6 GiB floor, the ~0.15 GiB-per-rank CSF expansion scratch), prove `planning.fit` accepts
`--expert-scale-compression csf` for this recipe, and trace the loader's
`scale_compression`/`scale_scratch` pass-through into both packing entry
points (the Step-5 call-site pattern).
"""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from ignition.recipes.hy4 import recipe as hy4_recipe
from ignition.recipes.hy4.config import ModelConfig

REF = Path(os.environ.get("HY4_REF_DIR", "/home/tom/hy4/ref"))
GIB = 1 << 30


@pytest.fixture(scope="module")
def model():
    config = REF / "bf16_config.json"
    if not config.is_file():
        pytest.skip(f"Hy4 reference metadata not present under {REF}")
    return ModelConfig.from_dict(json.loads(config.read_text()))


def independent_arithmetic(model):
    """Re-derive the resident byte figures from the census, by family."""
    c = model.text
    h, v = model.hidden, model.vocabulary
    layers, draft = model.layers, model.draft_layers
    blocks, moe = layers + draft, layers - 1 + draft
    heads, e, mi = model.heads, model.experts, c["moe_intermediate_size"]
    qk, nope, rope, vd = c["qk_head_dim"], c["qk_nope_head_dim"], c["qk_rope_head_dim"], c["v_head_dim"]
    ql, kv = c["q_lora_rank"], c["kv_lora_rank"]
    indexers = len(model.full_indexer_layers) + draft
    census = model.parameter_census()
    stored_bf16, stored_fp32 = model.stored_parameter_counts()

    # Packed experts, kept-FP32 upgrades, everything else at storage dtype.
    # `packed` counts every tensor ONCE at resident dtype regardless of
    # placement; `kv_b_proj` of the target layers additionally rides whole on
    # every rank (the `kv_b` two-wirings) — added to `replicated` below.
    expert_params = census["mlp.experts"]
    packed = (expert_params // 2 + expert_params // 16 + moe * e * 8
              + (stored_bf16 - expert_params) * 2 + stored_fp32 * 4
              + (v * h + indexers * 32 * h + indexers * 256 + e + heads) * 2)

    replicated = (
        blocks * (ql * h + (kv + rope) * h + ql + kv + 2 * h) * 2  # q/kv down-projections and layernorms
        + layers * heads * (nope + vd) * kv * 2    # kv_b_proj (replicated — the `kv_b` two-wirings)
        + h * 2 + blocks * heads * 4                               # model.norm and the FP32 sinks
        + moe * (e * h * 2 + e * 4)                                # router gate and FP32 correction biases
        + indexers * ((32 * 128) * ql * 2 + 128 * h * 2 + 32 * h * 4 + 256 * 4)  # indexer, FP32 at load
        + layers * 2 * (8 * 4 * h + 10) * 4                        # iHC hc_pre (attn and mlp)
        + (16 * h + 5) * 4                                         # hc_head
        + draft * 3 * h * 2)                                       # MTP enorm/hnorm/final_layernorm

    draft_experts = e * (2 * mi * h + h * mi)
    draft_bytes = (draft_experts // 2 + draft_experts // 16 + e * 8
                   + (2 * mi * h + h * mi + 2 * h * h) * 2
                   + (ql * h + heads * qk * ql + (kv + rope) * h + heads * (nope + vd) * kv
                      + h * heads * vd + heads * vd * h + (32 * 128) * ql + 128 * h + 32 * h
                      + 256 + e * h + 5 * h + ql + kv + e + heads) * 2
                   + (32 * h + 256 + e + heads) * 2)

    gate_up_scales, down_scales = moe * e * 2 * mi * (h // 16), moe * e * h * (mi // 16)
    import math

    ratios = {name: value for name, value in
              zip(("gate_up_proj", "down_proj"), (0.557, 0.600))}
    csf = math.ceil(ratios["gate_up_proj"] * gate_up_scales) + math.ceil(ratios["down_proj"] * down_scales)
    scratch = 2 * (e * 2 * (mi // 8) * (h // 16) + e * h * ((mi // 8) // 16))
    return {"packed_resident_bytes": packed, "replicated_bytes": replicated,
            "draft_bytes": draft_bytes, "expert_scale_bytes": gate_up_scales + down_scales,
            "csf_scale_bytes": csf, "scale_scratch_bytes": scratch}


def test_byte_arithmetic_reproduces_the_declared_figures(model):
    """The declared constants are the census arithmetic, verified independently."""
    arithmetic = independent_arithmetic(model)
    assert arithmetic == model.packed_byte_arithmetic()
    checkpoint = hy4_recipe.RECIPE.checkpoints[0]
    assert checkpoint.device_bytes == (arithmetic["packed_resident_bytes"]
                                       + 7 * arithmetic["replicated_bytes"])
    assert checkpoint.expansion_bytes == 8 * arithmetic["scale_scratch_bytes"]
    assert checkpoint.draft_bytes == arithmetic["draft_bytes"]
    savings = arithmetic["expert_scale_bytes"] - arithmetic["csf_scale_bytes"]
    assert checkpoint.scale_compressed_device_bytes == {
        "csf": checkpoint.device_bytes - savings}
    assert checkpoint.stored_bytes == hy4_recipe.STORED_BYTES

    # Plan anchors: the replication gap the old TP8 figure missed (now the
    # pre-`kv_b` 2.9 GB plus the `kv_b` two-wirings' 1.87 GiB per rank), the
    # 55.6 GiB floor, and the CSF delta per rank at TP8.
    gap = 7 * arithmetic["replicated_bytes"] // 8
    old_gap = 7 * 3_339_223_924 // 8       # the pre-`kv_b` replicated set
    kv_b_gap = 7 * 2_290_089_984 // 8      # `kv_b_proj`'s target-layer rows
    assert 2_800_000_000 < old_gap < 3_000_000_000  # "roughly 2.9 GB per rank at TP8"
    assert 1.86 * GIB < kv_b_gap < 1.88 * GIB       # "+1.87 GiB per rank" (consult8 §3)
    assert gap == old_gap + kv_b_gap == 4_925_649_669
    native = hy4_recipe.RECIPE.weights_bytes(checkpoint, 8, hy4_recipe.RECIPE.speculation)
    csf = hy4_recipe.RECIPE.weights_bytes(checkpoint, 8, hy4_recipe.RECIPE.speculation, "csf")
    assert native > 55.6 * GIB + gap                    # 55.6 GiB was a floor
    loaded = checkpoint.device_bytes + checkpoint.expansion_bytes
    assert native == -(-loaded // 8)
    assert csf == -(-(loaded - savings) // 8)
    assert (native - csf) / GIB == pytest.approx(2.35, abs=0.02)
    # The scratch reservation is about 0.15 GiB per rank at TP8.
    assert 0.13 < checkpoint.expansion_bytes / 8 / GIB < 0.16


def test_weights_bytes_counts_replicated_tensors_and_expansion(model):
    checkpoint = hy4_recipe.RECIPE.checkpoints[0]
    for world in (8, 16):
        learned = hy4_recipe.RECIPE.weights_bytes(checkpoint, world, hy4_recipe.RECIPE.speculation)
        plain = hy4_recipe.RECIPE.weights_bytes(checkpoint, world, None)
        loaded = checkpoint.device_bytes + checkpoint.expansion_bytes
        assert learned == -(-loaded // world)
        assert plain == -(-(loaded - checkpoint.draft_bytes) // world)
    # Replicated tensors ride every rank whole: the TP8 division is exact for
    # them, so the per-rank figure equals the sharded share plus one full copy.
    replicated = model.packed_byte_arithmetic()["replicated_bytes"]
    sharded = model.packed_byte_arithmetic()["packed_resident_bytes"] - replicated
    expected = -(-((sharded + 8 * replicated) + checkpoint.expansion_bytes) // 8)
    assert hy4_recipe.RECIPE.weights_bytes(checkpoint, 8, hy4_recipe.RECIPE.speculation) == expected


def test_planning_accepts_csf_for_the_hy4_checkpoint():
    """planning's scale-encoding check no longer rejects csf for this recipe."""
    from ignition import planning
    from ignition.errors import UsageError

    checkpoint = hy4_recipe.RECIPE.checkpoints[0]
    gpus = tuple(planning.Gpu(index, "NVIDIA RTX PRO 6000 Blackwell", f"GPU-{index}",
                              96 << 30, 95 << 30, (12, 0)) for index in range(8))
    speculation = hy4_recipe.RECIPE.speculation

    def fit(**settings):
        return planning.fit(hy4_recipe.RECIPE, checkpoint, gpus,
                            concurrency=4, reserve_bytes=0, speculation=speculation, **settings)

    native = fit()
    compressed = fit(expert_scale_compression="csf")
    assert native.expert_scale_compression is None
    assert compressed.expert_scale_compression == "csf"
    assert compressed.weights_bytes == hy4_recipe.RECIPE.weights_bytes(
        checkpoint, 8, speculation, "csf") < native.weights_bytes
    assert compressed.pages > native.pages
    with pytest.raises(UsageError, match="does not prepare this scale encoding"):
        fit(expert_scale_compression="gzip")


def test_preparation_config_and_loader_csf_wiring():
    from ignition.components.parallel import TensorParallelShard
    from ignition.recipes.hy4.bootstrap import PreparationConfig
    from ignition.recipes.hy4.weights import Hy4WeightLoader

    config = PreparationConfig(devices=tuple(range(8)), mtp=True,
                               expert_scale_compression="csf")
    assert config.expert_scale_compression == "csf"
    with pytest.raises(ValueError, match="must be None or 'csf'"):
        PreparationConfig(devices=(0,), expert_scale_compression="gzip")

    # Loader validation and the shared expansion scratch (Step-5's pattern).
    config_json = REF / "bf16_config.json"
    if not config_json.is_file():
        pytest.skip(f"Hy4 reference metadata not present under {REF}")
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        snapshot = Path(directory)
        (snapshot / "config.json").write_text(config_json.read_text())
        manifest = SimpleNamespace(snapshot=snapshot, metadata_path=snapshot, tensors={})
        with pytest.raises(ValueError, match="must be None or 'csf'"):
            Hy4WeightLoader(manifest, torch.device("cpu"), expert_scale_compression="lz4")
        loader = Hy4WeightLoader(manifest, torch.device("cpu"),
                                 shard=TensorParallelShard(0, 8), expert_scale_compression="csf")
        assert loader.scale_scratch[0].shape == (256, 512, 384)
        assert loader.scale_scratch[1].shape == (256, 6144, 16)
        buffers = sum(tensor.numel() for tensor in loader.scale_scratch)
        assert buffers == 75_497_472  # one layer's raw scales, two layouts reserved
        assert hy4_recipe.EXPANSION_BYTES == 2 * 8 * buffers
        assert loader.expert_scale_scratch(256)[0].shape == (256, 512, 384)
        plain = Hy4WeightLoader(manifest, torch.device("cpu"),
                                shard=TensorParallelShard(0, 8))
        assert plain.scale_scratch == () and plain.expert_scale_scratch(256) is None


def test_load_experts_passes_scale_compression_and_scratch(monkeypatch):
    """Both packing entry points receive the CSF settings (Step-5 weights.py)."""
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    import test_hy4_expert_loading as loading

    seen = {}
    loader = loading.make_loader(loading.packed_checkpoint(), expert_scale_compression="csf")

    def fp4(self, prefixes, **kwargs):
        seen["fp4"] = kwargs
        return ("fp4",)

    monkeypatch.setattr(loader, "load_fp4_experts", lambda prefixes, **kwargs: fp4(loader, prefixes, **kwargs))
    assert loader.load_experts(loading.PREFIX) == ("fp4",)
    assert seen["fp4"]["scale_compression"] == "csf"
    scratch = seen["fp4"]["scale_scratch"]
    assert scratch is not None and scratch[0].shape == loader.scale_scratch[0].shape

    # The raw BF16 route passes the same settings to the packer.
    count, inter, hidden = loading.COUNT, loading.INTER, loading.HIDDEN
    sources = {
        f"{loading.PREFIX}.gate_up_proj": torch.zeros(count, 2 * inter, hidden, dtype=torch.bfloat16),
        f"{loading.PREFIX}.down_proj": torch.zeros(count, hidden, inter, dtype=torch.bfloat16),
    }
    loader = loading.make_loader(sources, expert_scale_compression="csf")
    loader.quantization = {}
    monkeypatch.setattr(loader, "load_bf16_experts_as_nvfp4",
                        lambda prefix, **kwargs: seen.update(bf16=kwargs) or ("bf16",))
    assert loader.load_experts(loading.PREFIX) == ("bf16",)
    assert seen["bf16"]["scale_compression"] == "csf"
    assert seen["bf16"]["scale_scratch"] is not None
    assert seen["bf16"]["swiglu_limit"] == loading.SWIGLU_LIMIT
    assert seen["bf16"]["swiglu_clamp_after_silu"] is False


def test_load_resident_experts_forwards_csf_on_the_serve_path(monkeypatch):
    """The h4 CUDA serve call (`state._moe_state` -> `load_resident_experts`).

    The CHECKPOINT form (meta fused views, the rank's real intermediate range)
    must reach `load_fp4_experts` with the loader's CSF setting and the shared
    expansion scratch — this is the wiring the h4 serve path runs per routed
    block, and the one the CSF validation boot then exercises on GPU.
    """
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    import test_hy4_expert_loading as loading

    count, inter, hidden = loading.COUNT, loading.INTER, loading.HIDDEN

    def meta(shape, dtype):
        return torch.empty(shape, dtype=dtype, device="meta")

    resident = {
        f"{loading.PREFIX}.gate_up_proj.weight":
            meta((count, 2 * inter, hidden // 2), torch.uint8),
        f"{loading.PREFIX}.gate_up_proj.weight_scale":
            meta((count, 2 * inter, hidden // 16), torch.float8_e4m3fn),
        f"{loading.PREFIX}.gate_up_proj.weight_scale_2":
            torch.rand(count, generator=torch.Generator().manual_seed(1)) * 0.02 + 0.01,
        f"{loading.PREFIX}.down_proj.weight":
            meta((count, hidden, inter // 2), torch.uint8),
        f"{loading.PREFIX}.down_proj.weight_scale":
            meta((count, hidden, inter // 16), torch.float8_e4m3fn),
        f"{loading.PREFIX}.down_proj.weight_scale_2":
            torch.rand(count, generator=torch.Generator().manual_seed(2)) * 0.02 + 0.01,
    }
    for compressed in ("csf", None):
        loader = loading.make_loader({}, expert_scale_compression=compressed)
        for name, tensor in resident.items():
            loader._sources[name] = tensor
        before = set(loader._sources)
        seen = {}
        monkeypatch.setattr(loader, "load_fp4_experts",
                            lambda prefixes, **kwargs: seen.update(kwargs) or ("fp4",))
        assert loader.load_resident_experts(
            loading.PREFIX, resident, intermediate_range=(0, inter),
            swiglu_limit=loading.SWIGLU_LIMIT) == ("fp4",)
        assert seen["scale_compression"] == compressed
        assert seen["intermediate_size"] == inter
        assert tuple(seen["intermediate_range"]) == (0, inter)
        scratch = seen["scale_scratch"]
        if compressed is None:
            assert scratch is None
        else:
            assert scratch is not None and len(scratch) == 2
            assert scratch[0].shape == (count, 2 * inter, hidden // 16)
            assert scratch[1].shape == (count, hidden, inter // 16)
            # the views share the loader's two expansion buffers (the scratch arena)
            for view, buffer in zip(scratch, loader.scale_scratch, strict=True):
                assert (view.untyped_storage().data_ptr()
                        == buffer.untyped_storage().data_ptr())
        assert set(loader._sources) == before   # the per-expert stacks are cleaned up
