"""MiMo partitions retain each checkpoint row's original quantization scale."""

from types import SimpleNamespace

import pytest
import torch

from ignition.components.parallel import TensorParallelShard
from ignition.recipes.mimo_v2.config import ModelConfig
from ignition.recipes.mimo_v2.weights import MiMoWeightLoader


def config():
    return ModelConfig.from_dict({
        "model_type": "mimo_v2", "architectures": ["MiMoV2ForCausalLM"],
        "attention_projection_layout": "fused_qkv", "attention_bias": False,
        "hidden_size": 128, "hidden_act": "silu", "num_hidden_layers": 2,
        "hybrid_layer_pattern": [0, 1], "moe_layer_freq": [0, 1],
        "num_attention_heads": 128, "num_key_value_heads": 8, "head_dim": 192,
        "v_head_dim": 128, "swa_num_attention_heads": 128, "swa_num_key_value_heads": 8,
        "swa_head_dim": 192, "swa_v_head_dim": 128, "partial_rotary_factor": .334,
        "rope_theta": 10000000, "swa_rope_theta": 10000, "sliding_window_size": 128,
        "add_swa_attention_sink_bias": True, "n_shared_experts": None,
        "scoring_func": "sigmoid", "topk_method": "noaux_tc", "n_group": 1, "topk_group": 1,
        "quantization_config": {"quant_method": "fp8", "store_dtype": "mxfp4",
            "weight_block_size": [128, 128], "mxfp4_block_size": 32,
            "activation_scheme": "dynamic", "fmt": "e4m3",
            "ignored_layers": ["model.layers.0.self_attn.o_proj"]},
    }, checkpoint_ranks=8)


@pytest.mark.parametrize("world", [1, 2, 4, 8])
def test_qkv_keeps_training_chunk_padding_and_scale_ownership(world):
    c = config()
    # Every training rank stores 3392 weight rows and 27 scale blocks (3456 padded rows).
    source = torch.ones((27136, 128), dtype=torch.float8_e4m3fn)
    scales = torch.arange(1, 217, dtype=torch.float32).reshape(216, 1)
    restored = []
    for rank in range(world):
        loader = object.__new__(MiMoWeightLoader)
        loader.config, loader.shard, loader.device = c, TensorParallelShard(rank, world), "cpu"
        loader._session = SimpleNamespace(flush=lambda: None)
        loader._sources = {"model.layers.0.self_attn.qkv_proj.weight": source,
                           "model.layers.0.self_attn.qkv_proj.weight_scale_inv": scales}
        loader._copy = lambda target, value: target.copy_(value)
        result = loader.load_qkv(0)
        chunks = c.qkv_chunks(0, loader.shard)
        offset = 0
        for chunk in chunks:
            n = chunk.weight_rows[1] - chunk.weight_rows[0]
            values = result.values[offset:offset + chunk.padded_rows].float()
            scale = result.scales[offset // 128:(offset + chunk.padded_rows) // 128].repeat_interleave(128, 0)
            restored.append((values * scale)[:n])
            assert torch.count_nonzero(values[n:]) == 0
            assert (chunk.query_rows, chunk.key_rows, chunk.value_rows) == (3072, 192, 128)
            offset += chunk.padded_rows
    reference = torch.cat([scales[i * 27:(i + 1) * 27].repeat_interleave(128, 0)[:3392]
                           for i in range(8)]).expand(-1, 128)
    torch.testing.assert_close(torch.cat(restored), reference, rtol=0, atol=0)


def test_format_follows_metadata():
    c = config()
    assert c.projection_format("model.layers.0.self_attn.o_proj") == "bf16"
    assert c.projection_format("model.layers.1.self_attn.o_proj") == "block_fp8"
    assert c.projection_format("model.layers.1.mlp.experts.0.gate_proj") == "mxfp4"


def test_attention_schedule_has_independent_rotary_and_window_geometry():
    global_attention, window = config().attention(0), config().attention(1)
    assert (global_attention.window, global_attention.theta, global_attention.sinks) == (-1, 10000000, False)
    assert (window.window, window.theta, window.sinks) == (128, 10000, True)
    assert global_attention.rotary_dim == window.rotary_dim == 64
    assert global_attention.local(TensorParallelShard(7, 8)) == (16, 1)


def test_checkpoint_chunk_geometry_rejects_inexact_sharding():
    with pytest.raises(ValueError, match="checkpoint's fused-QKV shard count"):
        config().qkv_chunks(0, TensorParallelShard(0, 3))
    with pytest.raises(ValueError, match="checkpoint's fused-QKV shard count"):
        config().qkv_chunks(0, TensorParallelShard(0, 16))


def test_a4_prefill_variants_follow_the_export_threshold():
    from ignition.recipes.mimo_v2.bootstrap import A4_PREFILL_MIN_TOKENS, PreparationConfig

    assert PreparationConfig(dflash=True).a4_prefill_min_tokens == A4_PREFILL_MIN_TOKENS == 512
    assert PreparationConfig(dflash=True).a4_prefill
    assert not PreparationConfig(dflash=True, a4_prefill_min_tokens=0).a4_prefill
    assert not PreparationConfig(dflash=True, expert_activation="a8").a4_prefill


def test_draft_weight_format_defaults_to_w8a16_and_keeps_bf16_above_32_rows():
    from ignition.recipes.mimo_v2.bootstrap import PreparationConfig
    from ignition.recipes.mimo_v2.draft_weights import DraftWeights

    assert PreparationConfig(dflash=True).draft_weight_format == "w8a16"
    with pytest.raises(ValueError, match="w8a16 or bf16"):
        PreparationConfig(dflash=True, draft_weight_format="fp8")
    draft = SimpleNamespace(block_size=8)
    assert DraftWeights.serving_format("w8a16", draft, 4) == "w8a16"
    assert DraftWeights.serving_format("w8a16", draft, 8) == "bf16"
    assert DraftWeights.serving_format("bf16", draft, 1) == "bf16"


def test_draft_block_fp8_conversion_keeps_shard_blocks_and_shapes():
    from ignition.components.block_fp8 import BlockFP8Weight
    from ignition.components.weights import MLPWeights
    from ignition.recipes.mimo_v2.draft_weights import DraftLayerWeights, DraftWeights

    generator = torch.Generator().manual_seed(3)

    def matrix(n, k):
        return torch.randn((n, k), generator=generator).to(torch.bfloat16)

    layer = DraftLayerWeights(matrix(384, 256), matrix(256, 256), *(torch.ones(128),) * 2, None,
                              *(torch.ones(256),) * 2, MLPWeights(matrix(512, 256), matrix(256, 256)))
    weights = DraftWeights(matrix(128, 640), torch.ones(256), matrix(256, 256), torch.ones(256),
                           torch.zeros(256, dtype=torch.bfloat16), (layer,))
    converted = weights.block_fp8()
    for before, after in ((weights.projection, converted.projection), (weights.context_kv, converted.context_kv),
                          (layer.qkv, converted.layers[0].qkv), (layer.mlp.down, converted.layers[0].mlp.down)):
        assert isinstance(after, BlockFP8Weight) and after.values.shape == before.shape
        assert after.scales.shape == (before.shape[0] // 128, before.shape[1] // 128)
        restored = after.values.float().view(-1, 128, before.shape[1] // 128, 128) * after.scales[:, None, :, None]
        error = (restored.view(before.shape) - before.float()).norm() / before.float().norm()
        assert error < 0.04
    # Quantizing a row shard equals the matching rows of the whole matrix's quantization.
    from ignition.components.w8a16_gemv import quantize_block_fp8

    full, shard = quantize_block_fp8(layer.qkv), quantize_block_fp8(layer.qkv[128:256])
    assert torch.equal(full.values[128:256].view(torch.uint8), shard.values.view(torch.uint8))
    assert torch.equal(full.scales[1:2], shard.scales)
    assert converted.projection_shapes == ((128, 640), (256, 256), (384, 256), (512, 256))
    # A vocabulary head shard joins as a draft-only block-FP8 copy; without one draft logits keep BF16.
    assert converted.head is None
    head = matrix(640, 256)
    with_head = weights.block_fp8(head)
    assert isinstance(with_head.head, BlockFP8Weight) and with_head.head.values.shape == head.shape
    assert torch.equal(with_head.head.values.view(torch.uint8), quantize_block_fp8(head).values.view(torch.uint8))
    assert with_head.projection_shapes == tuple(sorted(converted.projection_shapes + ((640, 256),)))
    assert converted.with_head(head).head.scales.shape == (5, 2)
    assert converted.with_head(None) is converted


def test_csf_scales_keep_a4_prefill_past_the_inline_window():
    from ignition.recipes.mimo_v2.bootstrap import CSF_A4_PREFILL_MIN_TOKENS, PreparationConfig

    native = PreparationConfig(dflash=True)
    compressed = PreparationConfig(dflash=True, expert_scale_compression="csf")
    assert native.expert_scale_compression is None and native.a4_prefill_min_tokens == 512
    assert compressed.a4_prefill_min_tokens == CSF_A4_PREFILL_MIN_TOKENS == 1024 and compressed.a4_prefill
    # MiMo routes eight of 384 experts: inline CSF scales cover calls of at most 768 tokens.
    assert 768 * 8 <= 16 * 384 < CSF_A4_PREFILL_MIN_TOKENS * 8
    assert PreparationConfig(dflash=True, expert_scale_compression="csf", a4_prefill_min_tokens=2048
                             ).a4_prefill_min_tokens == 2048
    with pytest.raises(ValueError, match="scale compression"):
        PreparationConfig(dflash=True, expert_scale_compression="x4t")


def test_planner_prepares_csf_expert_scales_with_the_declared_saving():
    from ignition import planning
    from ignition.recipes.mimo_v2.recipe import RECIPE

    checkpoint = RECIPE.checkpoints[0]
    gpus = tuple(planning.Gpu(i, "NVIDIA RTX PRO 6000 Blackwell", f"GPU-{i:04d}", 96 << 30, 95 << 30, (12, 0))
                 for i in range(8))
    settings = dict(concurrency=4, reserve_bytes=0, speculation=RECIPE.speculation)
    native = planning.fit(RECIPE, checkpoint, gpus, **settings)
    compressed = planning.fit(RECIPE, checkpoint, gpus, expert_scale_compression="csf", **settings)
    assert native.expert_scale_compression is None and compressed.to_dict()["expert_scale_compression"] == "csf"
    saved = native.weights_bytes - compressed.weights_bytes
    assert 2.9e9 < saved < 3.0e9 and compressed.pages > native.pages


def _expert_loader(world, rank, *, compression):
    c = ModelConfig.from_dict({**config().text, "n_routed_experts": 2, "moe_intermediate_size": 64,
                               "num_experts_per_tok": 1}, checkpoint_ranks=8)
    loader = object.__new__(MiMoWeightLoader)
    loader.config, loader.shard, loader.device = c, TensorParallelShard(rank, world), "cpu"
    loader.expert_activation, loader.expert_scale_compression, loader.scale_scratch = "a16", compression, None
    loader._session = SimpleNamespace(flush=lambda: None)
    loader._copy = lambda target, value, numeric=True: target.copy_(value)
    loader._require_open = lambda: None
    loader._sources = {}
    for layer in (0, 1):
        for expert in range(2):
            for index, projection in enumerate(("gate_proj", "up_proj", "down_proj")):
                rows, columns = (128, 64) if projection == "down_proj" else (64, 128)
                prefix = f"model.layers.{layer}.mlp.experts.{expert}.{projection}"
                tag = 16 * layer + 4 * expert + index
                # Each row and scale group carries its projection and row identity.
                loader._sources[prefix + ".weight"] = (
                    torch.arange(rows, dtype=torch.uint8)[:, None] + tag * 8).expand(rows, columns // 2).contiguous()
                loader._sources[prefix + ".weight_scale"] = (
                    torch.arange(columns // 32, dtype=torch.uint8)[None] * 64 + torch.arange(rows, dtype=torch.uint8)[:, None] % 64
                    + tag).to(torch.uint8)
    for name in ("load_qkv", "load_linear", "load_tensor", "load_tensors"):
        setattr(loader, name, lambda *args, **kwargs: (None, None))
    return loader


@pytest.mark.parametrize("world", [1, 2])
def test_csf_experts_load_gate_first_with_one_shared_scratch(monkeypatch, world):
    from b12x.moe import fused_moe as moe

    for rank in range(world):
        prepared = []
        monkeypatch.setattr(moe, "prepare_weights", lambda **kwargs: prepared.append(kwargs) or len(prepared))
        native, compressed = (_expert_loader(world, rank, compression=mode) for mode in (None, "csf"))
        native.load_layer(1)
        compressed.load_layer(1)
        (base,), (csf,) = prepared[:1], prepared[1:]
        first, last = 64 // world * rank, 64 // world * (rank + 1)
        local = last - first
        assert base["scale_scratch"] is None and csf["scale_scratch"] is compressed.scale_scratch
        assert base["plan"].source.w13_layout.value == "w13" and csf["plan"].source.w13_layout.value == "w31"
        assert base["plan"].scale_compression is None and csf["plan"].scale_compression == "csf"
        assert [tuple(t.shape) for t in compressed.scale_scratch] == [(2, 256, 8), (2, 256, 8)]
        for expert in range(2):
            source = lambda projection: compressed._sources[f"model.layers.1.mlp.experts.{expert}.{projection}.weight"]
            scale = lambda projection: compressed._sources[f"model.layers.1.mlp.experts.{expert}.{projection}.weight_scale"]
            w13, s13 = csf["weights"].w13[expert], csf["weights"].w13_block_scales[expert]
            assert torch.equal(w13[:local], source("gate_proj")[first:last])
            assert torch.equal(w13[local:], source("up_proj")[first:last])
            assert torch.equal(s13[:local], scale("gate_proj")[first:last])
            assert torch.equal(s13[local:], scale("up_proj")[first:last])
            assert torch.equal(csf["weights"].w2_block_scales[expert], scale("down_proj")[:, first // 32:last // 32])
            # The native order keeps up first; the halves and down projection hold the same bytes.
            assert torch.equal(base["weights"].w13[expert], torch.cat((w13[local:], w13[:local])))
            assert torch.equal(base["weights"].w2[expert], csf["weights"].w2[expert])
        scratch = compressed.scale_scratch
        compressed.load_layer(1)
        assert prepared[-1]["scale_scratch"] is scratch


def test_switch_continuations_compile_like_speculative_rounds():
    from ignition.recipes.mimo_v2.bootstrap import COMPILED_KINDS

    # An uncompiled continuation captures its target verification eagerly, thousands of extra kernels.
    assert {"speculative", "continuation"} <= set(COMPILED_KINDS)


def test_one_request_switch_catalog_sizes_its_bodies():
    from ignition.recipes.mimo_v2.bootstrap import PreparationConfig

    class OneRequest(PreparationConfig):
        # A one-request catalog: the switched graph owns every round that offers drafts.
        speculative_round_shapes = ((1, 1, 1),)
        speculative_widths = (1, 2, 3, 4, 5, 6, 7, 8)
        request_capacity = 1

    config = OneRequest.__new__(OneRequest)
    config.__dict__.update(dflash=True, switch_speculation=True)
    assert config.switch_widths == 8
    assert config.decode_tokens == 8 and config.decode_samples == 8
