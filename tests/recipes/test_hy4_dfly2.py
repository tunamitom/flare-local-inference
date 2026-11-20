# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for official `tencent/Hy4` (`hy_v4_vl`) and DFly2 integration.

Zero GPU usage: exercises pinned config validation and strict `headers_dry_run`
against real `/storage/Hy4` headers (read-only), numeric parity against the
upstream SGLang `hunyuan_v4_dfly.py` equations at worlds 1/2/4, the per-row live
`row_cap_dev` cache-extent invariant under ragged + idle-zero dynamic extents
and row relocation, `BatchLoop` `bspec` k-gate behavior, text-only multimodal
HTTP-400 rejection, and packed W4A16 official export round-trip.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

OFFICIAL_SNAPSHOT = Path("/storage/Hy4")
CPU = torch.device("cpu")


# ---------------------------------------------------------------------------
# Helpers: synthetic schema-accurate official hy_v4_vl + DFly2 + ViT fixture
# ---------------------------------------------------------------------------
def _tiny_official_config():
    """6-layer official `hy_v4_vl` smoke config with 2 DFly2 layers + vision_config.

    Target layers: 0 (dense, full), 1 (sparse, full), 2..4 (sparse, shared),
    5 (sparse, full) — matching Hy4's exact `FULL_INDEXER_LAYERS` pattern
    (0, 1, 5, ...). DFly2 binds target layers [1, 5].
    """
    text_cfg = {
        "architectures": ["HYV4VLForConditionalGeneration"],
        "model_type": "hy_v4",
        "hidden_size": 64,
        "num_hidden_layers": 6,
        "vocab_size": 96,
        "intermediate_size": 128,
        "moe_intermediate_size": 64,
        "n_routed_experts": 4,
        "n_shared_experts": 1,
        "num_experts_per_tok": 2,
        "n_group": 1,
        "topk_group": 1,
        "norm_topk_prob": True,
        "routed_scaling_factor": 2.5,
        "swiglu_limit": 10.0,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "head_dim": 16,
        "qk_head_dim": 32,
        "qk_nope_head_dim": 16,
        "qk_rope_head_dim": 16,
        "v_head_dim": 16,
        "q_lora_rank": 32,
        "kv_lora_rank": 32,
        "use_mla": True,
        "gated_mla": True,
        "gating_type": "elementwise",
        "use_dsa": True,
        "index_n_heads": 4,
        "index_head_dim": 32,
        "index_topk": 12,
        "indexer_types": ["full", "full", "shared", "shared", "shared", "full"],
        "layer_types": ["deepseek_sparse_attention"] * 6,
        "mlp_layer_types": ["dense"] + ["sparse"] * 5,
        "enable_ihc": True,
        "hc_mult": 4,
        "hc_eps": 1e-6,
        "hc_magnitude": 2.0,
        "learnable_sink": True,
        "learnable_sink_init": 0.0,
        "enable_lm_head_fp32": True,
        "tie_word_embeddings": False,
        "use_cache": True,
        "qk_norm": True,
        "rms_norm_eps": 1e-5,
        "hidden_act": "silu",
        "max_position_embeddings": 4096,
        "num_nextn_predict_layers": 0,
        "rope_parameters": {
            "mrope_interleaved": True,
            "mrope_section": [4, 2, 2],
            "rope_theta": 10000.0,
            "rope_type": "default",
        },
        "bos_token_id": 0,
        "eos_token_id": 1,
        "pad_token_id": 2,
    }
    top = dict(text_cfg)
    top["model_type"] = "hy_v4_vl"
    top["dtype"] = "bfloat16"
    top["text_config"] = text_cfg
    top["vision_config"] = {
        "model_type": "hy_v4_vl",
        "num_hidden_layers": 1,
        "hidden_size": 32,
    }
    top.update({
        "dfly_enable_ihc": False,
        "dfly_mtp_fusion": "slot0",
        "dfly_target_index_topk": 12,
        "dfly_context_local_window": None,
        "dfly_enable_hidden_correction": True,
        "dfly_num_hidden_layers": 2,
        "dfly_mlp_type": "dense",
        "dfly_mask_token_id": 95,
        "dfly_target_index_layer": [1, 5],
        "dfly_target_kv_layers": [1, 5],
        "dfly_attn_causal": True,
        "dfly_block_size": 7,
    })
    return top


def _make_tiny_official_snapshot(root: Path, *, packed: bool = False, seed: int = 1234):
    """Write a complete tiny official Hy4+DFly2+ViT checkpoint on disk."""
    from safetensors.torch import save_file

    from ignition.recipes.hy4 import contract
    from ignition.recipes.hy4.config import ModelConfig

    root.mkdir(parents=True, exist_ok=True)
    cfg = _tiny_official_config()
    model_cfg = ModelConfig.from_dict(cfg, pinned=False)
    gen = torch.Generator().manual_seed(seed)

    def rand(shape, *, dtype=torch.bfloat16, scale=0.08):
        return (torch.randn(shape, generator=gen, dtype=torch.float32) * scale).to(dtype)

    def ones(shape, *, dtype=torch.bfloat16):
        return (1.0 + 0.05 * torch.randn(shape, generator=gen, dtype=torch.float32)).to(dtype)

    c = model_cfg.text
    h, v = model_cfg.hidden, model_cfg.vocabulary
    heads, qk = model_cfg.heads, c["qk_head_dim"]
    nope, rope, vd = c["qk_nope_head_dim"], c["qk_rope_head_dim"], c["v_head_dim"]
    ql, kv = c["q_lora_rank"], c["kv_lora_rank"]
    e, mi, inter = model_cfg.experts, c["moe_intermediate_size"], c["intermediate_size"]
    ih, idim = c["index_n_heads"], c["index_head_dim"]
    hc = c["hc_mult"]

    tensors = {
        "model.embed_tokens.weight": rand((v, h)),
        "model.norm.weight": ones((h,)),
        "lm_head.weight": rand((v, h)),
        "model.hc_head.hc_head_fn": rand((hc, hc * h), dtype=torch.float32, scale=0.05),
        "model.hc_head.hc_head_base": rand((hc,), dtype=torch.float32, scale=0.05),
        "model.hc_head.hc_head_scale": torch.full((1,), 0.1, dtype=torch.float32),
    }

    for layer in range(model_cfg.layers):
        p = f"model.layers.{layer}"
        for blk in ("hc_attn_layer", "hc_mlp_layer"):
            tensors[f"{p}.{blk}.hc_pre.hc_fn"] = rand((2 * hc, hc * h), dtype=torch.float32, scale=0.05)
            tensors[f"{p}.{blk}.hc_pre.hc_base"] = rand((2 * hc,), dtype=torch.float32, scale=0.05)
            tensors[f"{p}.{blk}.hc_pre.hc_scale"] = torch.full((2,), 0.1, dtype=torch.float32)
        tensors[f"{p}.input_layernorm.weight"] = ones((h,))
        tensors[f"{p}.post_attention_layernorm.weight"] = ones((h,))
        ap = f"{p}.self_attn"
        tensors[f"{ap}.q_a_proj.weight"] = rand((ql, h))
        tensors[f"{ap}.q_a_layernorm.weight"] = ones((ql,))
        tensors[f"{ap}.q_b_proj.weight"] = rand((heads * qk, ql))
        tensors[f"{ap}.kv_a_proj_with_mqa.weight"] = rand((kv + rope, h))
        tensors[f"{ap}.kv_a_layernorm.weight"] = ones((kv,))
        tensors[f"{ap}.kv_b_proj.weight"] = rand((heads * (nope + vd), kv))
        tensors[f"{ap}.o_proj.weight"] = rand((h, heads * vd))
        tensors[f"{ap}.linear_gate.weight"] = rand((heads * vd, h))
        tensors[f"{ap}.learnable_sink_param"] = rand((heads,), dtype=torch.float32, scale=0.2)
        if model_cfg.indexer_type(layer) == "full":
            ip = f"{ap}.indexer"
            tensors[f"{ip}.wq_b.weight"] = rand((ih * idim, ql))
            tensors[f"{ip}.wk.weight"] = rand((idim, h))
            tensors[f"{ip}.k_norm.weight"] = ones((idim,))
            tensors[f"{ip}.k_norm.bias"] = rand((idim,), scale=0.02)
            tensors[f"{ip}.weights_proj.weight"] = rand((ih, h))
        if layer == 0:
            tensors[f"{p}.mlp.gate_proj.weight"] = rand((inter, h))
            tensors[f"{p}.mlp.up_proj.weight"] = rand((inter, h))
            tensors[f"{p}.mlp.down_proj.weight"] = rand((h, inter))
        else:
            tensors[f"{p}.mlp.gate.weight"] = rand((e, h))
            tensors[f"{p}.mlp.gate.e_score_correction_bias"] = rand((e,), dtype=torch.float32, scale=0.05)
            tensors[f"{p}.mlp.shared_experts.gate_proj.weight"] = rand((mi, h))
            tensors[f"{p}.mlp.shared_experts.up_proj.weight"] = rand((mi, h))
            tensors[f"{p}.mlp.shared_experts.down_proj.weight"] = rand((h, mi))
            if packed:
                ep = f"{p}.mlp.experts"
                tensors[f"{ep}.gate_up_proj.weight"] = torch.randint(
                    0, 256, (e, 2 * mi, h // 2), generator=gen, dtype=torch.uint8)
                tensors[f"{ep}.gate_up_proj.weight_scale"] = torch.ones(
                    (e, 2 * mi, h // 16), dtype=torch.float8_e4m3fn)
                tensors[f"{ep}.gate_up_proj.weight_scale_2"] = torch.full((e,), 0.02, dtype=torch.float32)
                tensors[f"{ep}.down_proj.weight"] = torch.randint(
                    0, 256, (e, h, mi // 2), generator=gen, dtype=torch.uint8)
                tensors[f"{ep}.down_proj.weight_scale"] = torch.ones(
                    (e, h, mi // 16), dtype=torch.float8_e4m3fn)
                tensors[f"{ep}.down_proj.weight_scale_2"] = torch.full((e,), 0.02, dtype=torch.float32)
            else:
                tensors[f"{p}.mlp.experts.gate_up_proj"] = rand((e, 2 * mi, h))
                tensors[f"{p}.mlp.experts.down_proj"] = rand((e, h, mi))

    # DFly2 layers (pure BF16) + wrapper + ignored confidence_head.
    dp = "model.dfly_layers"
    tensors[f"{dp}.final_norm.weight"] = ones((h,))
    tensors[f"{dp}.mtp_fusion.enorm.weight"] = ones((h,))
    tensors[f"{dp}.mtp_fusion.hnorm.weight"] = ones((h,))
    tensors[f"{dp}.mtp_fusion.eh_proj.weight"] = rand((h, 2 * h))
    tensors[f"{dp}.mtp_fusion.gate"] = rand((h,), scale=0.15)
    tensors[f"{dp}.hidden_correction.hidden_norm.weight"] = ones((h,))
    tensors[f"{dp}.hidden_correction.embed_norm.weight"] = ones((h,))
    tensors[f"{dp}.hidden_correction.gate_proj.weight"] = rand((h, 2 * h))
    tensors[f"{dp}.hidden_correction.up_proj.weight"] = rand((h, 2 * h))
    tensors[f"{dp}.hidden_correction.down_proj.weight"] = rand((h, h))
    tensors[f"{dp}.confidence_head.proj.weight"] = rand((1, h))
    tensors[f"{dp}.confidence_head.proj.bias"] = rand((1,))
    for idx in range(model_cfg.dfly_layers):
        lp = f"{dp}.{idx}"
        tensors[f"{lp}.input_layernorm.weight"] = ones((h,))
        tensors[f"{lp}.post_attention_layernorm.weight"] = ones((h,))
        ap = f"{lp}.self_attn"
        tensors[f"{ap}.q_a_proj.weight"] = rand((ql, h))
        tensors[f"{ap}.q_a_layernorm.weight"] = ones((ql,))
        tensors[f"{ap}.q_b_proj.weight"] = rand((heads * qk, ql))
        tensors[f"{ap}.kv_a_proj_with_mqa.weight"] = rand((kv + rope, h))
        tensors[f"{ap}.kv_a_layernorm.weight"] = ones((kv,))
        tensors[f"{ap}.kv_b_proj.weight"] = rand((heads * (nope + vd), kv))
        tensors[f"{ap}.o_proj.weight"] = rand((h, heads * vd))
        tensors[f"{ap}.linear_gate.weight"] = rand((heads * vd, h))
        tensors[f"{ap}.learnable_sink_param"] = rand((heads,), scale=0.2)
        tensors[f"{lp}.mlp.gate_proj.weight"] = rand((inter, h))
        tensors[f"{lp}.mlp.up_proj.weight"] = rand((inter, h))
        tensors[f"{lp}.mlp.down_proj.weight"] = rand((h, inter))

    # Vision passthrough tensors (skipped by the text-only loader).
    tensors["vit.embeddings.patch_embedding.weight"] = rand((32, 3, 4, 4))
    tensors["vit.adapter.proj.weight"] = rand((h, 32))

    if packed:
        cfg = dict(cfg)
        cfg["quantization_config"] = contract.quantization_config(model_cfg, mtp=True)
        prefixes = contract.quantized_source_names(model_cfg, mtp=True)
        census = {f"{p}.{proj}.{idx}": 0.075
                  for p in prefixes for proj in ("gate_proj", "up_proj", "down_proj")
                  for idx in range(e)}
        report = {
            "byte_verified": True,
            "packed_matrices": len(census),
            "census": census,
            "census_max_relative": 0.075,
            "window_edge_counts": {
                p: {proj: [0] * e for proj in ("gate_proj", "up_proj", "down_proj")}
                for p in prefixes
            },
        }
        (root / "export_report.json").write_text(json.dumps(report))

    shard_name = "model-00001-of-00001.safetensors"
    shard_path = root / shard_name
    save_file(tensors, str(shard_path))
    shard_path.chmod(0o644)
    (root / "config.json").write_text(json.dumps(cfg))
    weight_map = {name: shard_name for name in sorted(tensors)}
    total_size = sum(t.numel() * t.element_size() for t in tensors.values())
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total_size}, "weight_map": weight_map}))
    return root, tensors


def _to_fp32_dfly_weights(weights):
    """Promote BF16 tensors (including DFly2) to FP32 for exact CPU parity checks."""
    from dataclasses import replace

    from ignition.recipes.hy4.attention import AttentionWeights, Hy4LayerAttention, PackedInputProjections
    from ignition.recipes.hy4.dfly import DflyBlock, DflyHiddenCorrection, DflyLayer, DflyMtpFusion
    from ignition.recipes.hy4.moe import DenseBlock, DenseMlpWeights
    from tests.recipes.test_hy4_reference_gates import to_fp32_weights

    base = to_fp32_weights(weights)
    if weights.dfly is None:
        return base

    def f32(t):
        return t.float() if isinstance(t, torch.Tensor) and t.is_floating_point() else t

    dfly = weights.dfly
    fusion = DflyMtpFusion(
        enorm=f32(dfly.mtp_fusion.enorm), hnorm=f32(dfly.mtp_fusion.hnorm),
        eh_proj=f32(dfly.mtp_fusion.eh_proj), gate=f32(dfly.mtp_fusion.gate),
        eps=dfly.mtp_fusion.eps)
    hc = DflyHiddenCorrection(
        hidden_norm=f32(dfly.hidden_correction.hidden_norm),
        embed_norm=f32(dfly.hidden_correction.embed_norm),
        gate_proj=f32(dfly.hidden_correction.gate_proj),
        up_proj=f32(dfly.hidden_correction.up_proj),
        down_proj=f32(dfly.hidden_correction.down_proj),
        eps=dfly.hidden_correction.eps)
    layers = []
    for layer in dfly.layers:
        aw = layer.attention.weights
        aw32 = AttentionWeights(
            q_a_proj=f32(aw.q_a_proj), q_a_norm=f32(aw.q_a_norm), q_b_proj=f32(aw.q_b_proj),
            kv_a_proj=f32(aw.kv_a_proj), kv_a_norm=f32(aw.kv_a_norm), kv_b_proj=f32(aw.kv_b_proj),
            o_proj=f32(aw.o_proj), gate_proj=f32(aw.gate_proj), sink=f32(aw.sink), indexer=None)
        aw32 = replace(aw32, packed_input=PackedInputProjections.build(aw32))
        attn32 = Hy4LayerAttention(
            weights=aw32, config=layer.attention.config, sharding=layer.attention.sharding,
            selection=None, device=layer.attention.device, max_width=layer.attention.max_width)
        mw = layer.mlp.weights
        mlp32 = DenseBlock(DenseMlpWeights(
            gate_proj=f32(mw.gate_proj), up_proj=f32(mw.up_proj), down_proj=f32(mw.down_proj)),
            layer.mlp.device)
        layers.append(DflyLayer(
            index=layer.index, target_kv_layer=layer.target_kv_layer,
            target_index_layer=layer.target_index_layer, producer_plane=layer.producer_plane,
            attention=attn32, input_norm=f32(layer.input_norm),
            post_attention_norm=f32(layer.post_attention_norm), mlp=mlp32))
    return replace(base, dfly=DflyBlock(
        layers=tuple(layers), mtp_fusion=fusion, final_norm=f32(dfly.final_norm),
        hidden_correction=hc, producer_layers=dfly.producer_layers,
        config=dfly.config, device=dfly.device))


def _sglang_dfly_reference(raw_tensors, cfg_dict, *, target_kv_by_layer, target_topk_by_layer,
                           bonus_tokens, target_hidden, anchor, k):
    """Standalone reference implementing SGLang `hunyuan_v4_dfly.py` equations directly.

    Uses raw checkpoint tensors without calling any `ignition.recipes.hy4.dfly`
    code, matching `HYV4DFlyFusion`, `HYV4DFlyDecoderLayer`, `HYV4DFlyModel`,
    and `HYV4DFlyHiddenCorrection` line-for-line in FP32.
    """
    from ignition.recipes.hy4.attention import apply_interleaved_rope, rms_norm

    t = {name: val.float() for name, val in raw_tensors.items()}
    tc = cfg_dict["text_config"]
    eps = tc["rms_norm_eps"]
    heads = tc["num_attention_heads"]
    nope, rope, vd = tc["qk_nope_head_dim"], tc["qk_rope_head_dim"], tc["v_head_dim"]
    kv_lora = tc["kv_lora_rank"]
    theta = tc["rope_parameters"]["rope_theta"]
    scale = (nope + rope) ** -0.5
    W = k + 1
    bs = int(bonus_tokens.shape[0])

    # Input ids: [bonus, mask, ..., mask], positions [anchor .. anchor + k].
    ids = torch.full((bs, W), int(cfg_dict["dfly_mask_token_id"]), dtype=torch.long)
    ids[:, 0] = bonus_tokens.long()
    positions = anchor.long().unsqueeze(1) + torch.arange(W, dtype=torch.long).unsqueeze(0)

    h = t["model.embed_tokens.weight"][ids]  # [B, W, H]
    # Slot-0 HYV4DFlyFusion: embeds + gate * eh_proj(cat(enorm(embeds), hnorm(target_hidden))).
    e0 = h[:, 0]
    fused = torch.cat((
        rms_norm(e0, t["model.dfly_layers.mtp_fusion.enorm.weight"], eps),
        rms_norm(target_hidden.float(), t["model.dfly_layers.mtp_fusion.hnorm.weight"], eps),
    ), dim=-1) @ t["model.dfly_layers.mtp_fusion.eh_proj.weight"].T
    h = torch.cat(((e0 + t["model.dfly_layers.mtp_fusion.gate"] * fused).unsqueeze(1), h[:, 1:]), dim=1)

    kv_layers = cfg_dict["dfly_target_kv_layers"]
    idx_layers = cfg_dict["dfly_target_index_layer"]
    for i in range(cfg_dict["dfly_num_hidden_layers"]):
        lp = f"model.dfly_layers.{i}"
        ap = f"{lp}.self_attn"
        x = rms_norm(h.reshape(bs * W, -1), t[f"{lp}.input_layernorm.weight"], eps)
        pos_flat = positions.reshape(-1)
        q_resid = rms_norm(x @ t[f"{ap}.q_a_proj.weight"].T, t[f"{ap}.q_a_layernorm.weight"], eps)
        q = (q_resid @ t[f"{ap}.q_b_proj.weight"].T).view(bs * W, heads, nope + rope)
        q = apply_interleaved_rope(q, pos_flat, rope_dim=rope, theta=theta)
        kv_raw = x @ t[f"{ap}.kv_a_proj_with_mqa.weight"].T
        k_pass = rms_norm(kv_raw[:, :kv_lora], t[f"{ap}.kv_a_layernorm.weight"], eps)
        k_rot = apply_interleaved_rope(kv_raw[:, kv_lora:], pos_flat, rope_dim=rope, theta=theta)
        block_latent = torch.cat((k_pass, k_rot), dim=-1).view(bs, W, kv_lora + rope)

        per_head = t[f"{ap}.kv_b_proj.weight"].view(heads, nope + vd, kv_lora)
        wuk, wuv = per_head[:, :nope], per_head[:, nope:]
        q_abs = torch.cat((torch.einsum("rhi,hio->rho", q[..., :nope], wuk), q[..., nope:]), dim=-1)
        q_abs = q_abs.view(bs, W, heads, kv_lora + rope)

        sink = t[f"{ap}.learnable_sink_param"].view(1, heads, 1)
        hist_kv = target_kv_by_layer[kv_layers[i]]       # [B, P_max, kv_lora + rope]
        hist_idx = target_topk_by_layer[idx_layers[i]]   # [B, T] (-1 padded)
        out_rows = []
        for b in range(bs):
            valid_hist = hist_idx[b][hist_idx[b] >= 0].long()
            hk = hist_kv[b, valid_hist]                  # [H_valid, D]
            b_outs = []
            for j in range(W):
                keys = torch.cat((hk, block_latent[b, :j + 1]), dim=0)  # [H_valid + j + 1, D]
                logits = torch.einsum("hd,kd->hk", q_abs[b, j], keys) * scale
                comb = torch.cat((logits, sink[0]), dim=-1)
                probs = torch.softmax(comb, dim=-1)[..., :-1]
                lat_out = torch.einsum("hk,kd->hd", probs, keys[:, :kv_lora])
                val_out = torch.einsum("hd,hvd->hv", lat_out, wuv)
                b_outs.append(val_out)
            out_rows.append(torch.stack(b_outs, dim=0))
        attn_val = torch.stack(out_rows, dim=0).reshape(bs * W, heads, vd)
        gate = torch.sigmoid(x @ t[f"{ap}.linear_gate.weight"].T).view(bs * W, heads, vd)
        attn_out = ((attn_val * gate).reshape(bs * W, -1) @ t[f"{ap}.o_proj.weight"].T).view(bs, W, -1)
        h = h + attn_out

        mx = rms_norm(h.reshape(bs * W, -1), t[f"{lp}.post_attention_layernorm.weight"], eps)
        mlp_out = ((F.silu(mx @ t[f"{lp}.mlp.gate_proj.weight"].T)
                    * (mx @ t[f"{lp}.mlp.up_proj.weight"].T))
                   @ t[f"{lp}.mlp.down_proj.weight"].T).view(bs, W, -1)
        h = h + mlp_out

    z = rms_norm(h.reshape(bs * W, -1), t["model.dfly_layers.final_norm.weight"], eps).view(bs, W, -1)
    # Sequential HYV4DFlyHiddenCorrection: hidden_norm FIRST, embed_norm SECOND.
    prev = bonus_tokens.long()
    drafts = []
    for step in range(k):
        prev_emb = t["model.embed_tokens.weight"][prev]
        hc_in = torch.cat((
            rms_norm(z[:, step + 1], t["model.dfly_layers.hidden_correction.hidden_norm.weight"], eps),
            rms_norm(prev_emb, t["model.dfly_layers.hidden_correction.embed_norm.weight"], eps),
        ), dim=-1)
        delta = ((F.silu(hc_in @ t["model.dfly_layers.hidden_correction.gate_proj.weight"].T)
                  * (hc_in @ t["model.dfly_layers.hidden_correction.up_proj.weight"].T))
                 @ t["model.dfly_layers.hidden_correction.down_proj.weight"].T)
        corrected = z[:, step + 1] + delta
        logits = corrected @ t["lm_head.weight"].T
        prev = logits.argmax(dim=-1)
        drafts.append(prev)
    return torch.stack(drafts, dim=1), z


# ---------------------------------------------------------------------------
# 1. Real /storage/Hy4 pinned config + TP8 headers_dry_run (read-only!)
# ---------------------------------------------------------------------------
def test_official_storage_config_and_tp8_headers_dry_run():
    if not (OFFICIAL_SNAPSHOT / "config.json").is_file():
        pytest.skip("/storage/Hy4 not available on this host")

    from ignition.recipes.hy4 import contract
    from ignition.recipes.hy4.bootstrap import PreparationConfig, construct_rank
    from ignition.recipes.hy4.collectives import SimulatedGroup
    from ignition.recipes.hy4.config import ModelConfig
    from ignition.recipes.hy4.weights import checkpoint_names, dfly_checkpoint_names

    raw = json.loads((OFFICIAL_SNAPSHOT / "config.json").read_text())
    cfg = ModelConfig.load(OFFICIAL_SNAPSHOT, pinned=True)

    assert cfg.layers == 78 and cfg.hidden == 6144 and cfg.vocabulary == 120832
    assert cfg.draft_layers == 0
    assert cfg.has_dfly and cfg.dfly_layers == 5
    assert cfg.has_vision
    assert cfg.dfly_target_kv_layers == (61, 65, 69, 73, 77)
    assert cfg.dfly_target_index_layers == (61, 65, 69, 73, 77)
    assert cfg.dfly_block_size == 7 and cfg.dfly_mask_token_id == 120009
    assert cfg.dfly_context_local_window == 0
    assert cfg.dfly_index_topk == 2176
    # DFly2 owns no persistent KV or indexer cache layer.
    assert cfg.token_record_bytes(draft=False) == 53940
    assert cfg.token_record_bytes(draft=True) == 53940

    # Strict inventory check against real 2,710-key safetensors index.
    idx = json.loads((OFFICIAL_SNAPSHOT / "model.safetensors.index.json").read_text())["weight_map"]
    active = checkpoint_names(cfg, mtp=True, dfly=True)
    ignored_dfly = dfly_checkpoint_names(cfg, include_ignored=True) - dfly_checkpoint_names(cfg)
    vit_keys = {k for k in idx if k.startswith("vit.")}
    assert len(idx) == 2710
    assert len(active) == 1979 + 80
    assert len(ignored_dfly) == 2
    assert len(vit_keys) == 649
    assert set(idx) == set(active) | set(ignored_dfly) | vit_keys

    # Export contract keeps DFly2 pure BF16 (77 routed MoE prefixes, 0 dfly_layers).
    export = contract.export_config(raw)
    qlayers = contract.validate_export(export)
    assert len(qlayers) == 77
    assert not any("dfly_layers" in name for name in qlayers)
    assert export["vision_config"] == raw["vision_config"]
    assert export["dfly_target_kv_layers"] == [61, 65, 69, 73, 77]

    # Real TP8 header dry-run: loads 1,979 backbone + 80 DFly2 tensors on meta device.
    settings = PreparationConfig(devices=tuple(range(8)), mtp=True, prefill_chunk_size=1024)
    group = SimulatedGroup(8)
    built = construct_rank(
        OFFICIAL_SNAPSHOT, settings, rank=0, device=CPU, pinned=True, mtp=True,
        cache_rows=1, cache_capacity=128, headers_only=True, world_size=8,
        collectives=group.collectives(0))
    assert built.weights.draft is None
    assert built.weights.dfly is not None
    assert len(built.weights.dfly.layers) == 5
    assert built.memory_report["dfly"] is True
    assert built.weights.used_names == active
    # Pure BF16 DFly2 resident size per rank at TP8 (~1.14 GiB with replicated kv_b_proj).
    dfly_bytes = built.weights.allocated_bytes - construct_rank(
        OFFICIAL_SNAPSHOT, PreparationConfig(devices=tuple(range(8)), mtp=False, prefill_chunk_size=1024),
        rank=0, device=CPU, pinned=True, mtp=False,
        cache_rows=1, cache_capacity=128, headers_only=True, world_size=8,
        collectives=group.collectives(0)).weights.allocated_bytes
    assert 1.05 * (1 << 30) < dfly_bytes < 1.25 * (1 << 30), f"unexpected DFly2 bytes/rank: {dfly_bytes}"
    # Verify DFly2 weights are pure BF16 (except FP32 per-head learnable sinks).
    assert built.weights.dfly.mtp_fusion.eh_proj.dtype == torch.bfloat16
    assert built.weights.dfly.hidden_correction.gate_proj.dtype == torch.bfloat16
    for layer in built.weights.dfly.layers:
        assert layer.attention.weights.q_b_proj.dtype == torch.bfloat16
        assert layer.attention.weights.sink.dtype == torch.float32
        assert layer.mlp.weights.gate_proj.dtype == torch.bfloat16


def test_official_config_validation_rejects_drift():
    from ignition.recipes.hy4.config import ModelConfig

    cfg = _tiny_official_config()
    # Non-full indexer layer in dfly_target_index_layer must fail.
    bad = dict(cfg, dfly_target_index_layer=[1, 2])
    with pytest.raises(ValueError, match="full"):
        ModelConfig.from_dict(bad, pinned=False)
    # Duplicate dfly_target_kv_layers must fail.
    bad_kv = dict(cfg, dfly_target_kv_layers=[1, 1])
    with pytest.raises(ValueError, match="distinct"):
        ModelConfig.from_dict(bad_kv, pinned=False)
    # Mismatch between top-level and nested text_config must fail.
    bad_nested = dict(cfg, hidden_size=128)
    with pytest.raises(ValueError, match="disagrees"):
        ModelConfig.from_dict(bad_nested, pinned=False)


# ---------------------------------------------------------------------------
# 2. Exact numeric parity vs SGLang reference (world=1, 2, 4; short & sparse)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("prompt_len", (8, 22))
@pytest.mark.parametrize("world", (1, 2, 4))
def test_dfly2_matches_sglang_reference_and_cp_parity(tmp_path, prompt_len, world):
    from ignition.recipes.hy4.bootstrap import PreparationConfig, construct_rank
    from ignition.recipes.hy4.collectives import PcieDcpCollectives, SimulatedDcpTransport, SimulatedGroup
    from ignition.recipes.hy4.decoder import Hy4Cache, Hy4Decoder

    snap, raw_tensors = _make_tiny_official_snapshot(tmp_path / f"snap_{prompt_len}_{world}")
    cfg_dict = json.loads((snap / "config.json").read_text())
    cap = 64
    k = 3
    prompt = torch.tensor([[3 + (i * 7) % 80 for i in range(prompt_len)]], dtype=torch.long)

    settings = PreparationConfig(devices=(0,), mtp=True, prefill_chunk_size=64)

    # First run world=1 to obtain ground-truth target KV, target top-k, bonus, hidden.
    built1 = construct_rank(snap, settings, rank=0, pinned=False, mtp=True,
                            cache_rows=2, cache_capacity=cap, world_size=1)
    w1 = _to_fp32_dfly_weights(built1.weights)
    c1 = Hy4Cache.allocate(built1.model_config, rows=2, capacity=cap, device=CPU,
                           draft=False, dtype=torch.float32, world=1, rank=0)
    d1 = Hy4Decoder(w1, c1, compute_dtype=torch.float32)
    logits1, hidden1, _ = d1.forward(prompt, request_rows=[0], return_hidden=True)
    d1.complete([0], prompt)
    bonus = logits1[:, -1, :].argmax(dim=-1)             # [1]
    last_hid = hidden1[:, -1, :]                         # [1, H]

    # Snapshot target KV and tapped top-k BEFORE DFly2 writes into verify slots.
    target_kv_ref = {
        layer_id: c1.latents(layer_id)[:prompt_len].clone().unsqueeze(0)
        for layer_id in cfg_dict["dfly_target_kv_layers"]
    }
    target_topk_ref = {
        layer_id: d1.dfly_state.hist_local[plane, 0].clone().unsqueeze(0)
        for plane, layer_id in enumerate(d1.weights.dfly.producer_layers)
    }
    ref_drafts, ref_z = _sglang_dfly_reference(
        raw_tensors, cfg_dict,
        target_kv_by_layer=target_kv_ref,
        target_topk_by_layer=target_topk_ref,
        bonus_tokens=bonus, target_hidden=last_hid,
        anchor=torch.tensor([prompt_len], dtype=torch.long), k=k)

    if world == 1:
        got_drafts, got_z = d1.dfly_draft(bonus, last_hid, request_rows=[0], k=k)
        assert torch.allclose(got_z, ref_z, atol=1e-4, rtol=1e-4), \
            f"world=1 z max diff {(got_z - ref_z).abs().max().item()}"
        assert torch.equal(got_drafts, ref_drafts), f"{got_drafts} != {ref_drafts}"
        return

    group = SimulatedGroup(world)
    transport = SimulatedDcpTransport(group)
    decoders = []
    for r in range(world):
        built = construct_rank(snap, settings, rank=r, pinned=False, mtp=True,
                               cache_rows=2, cache_capacity=cap, world_size=world,
                               collectives=group.collectives(r))
        wr = _to_fp32_dfly_weights(built.weights)
        cr = Hy4Cache.allocate(built.model_config, rows=2, capacity=cap, device=CPU,
                               draft=False, dtype=torch.float32, world=world, rank=r)
        col = PcieDcpCollectives(rank=r, world_size=world, transport=transport)
        decoders.append(Hy4Decoder(wr, cr, compute_dtype=torch.float32, collectives=col))

    def run_rank(dec):
        lg, hid, _ = dec.forward(prompt, request_rows=[0], return_hidden=True)
        dec.complete([0], prompt)
        b_tok = lg[:, -1, :].argmax(dim=-1)
        h_last = hid[:, -1, :]
        bypass = (prompt_len + k + 1) <= cfg_dict["index_topk"]
        dr, z = dec.dfly_draft(b_tok, h_last, request_rows=[0], k=k, bypass_cp=bypass)
        return dr, z

    results = group.run([lambda col, dec=dec: run_rank(dec) for dec in decoders])
    for r, (got_drafts, got_z) in enumerate(results):
        rel = float((got_z - ref_z).norm() / ref_z.norm().clamp_min(1e-12))
        assert rel < 1e-3, f"world={world} rank={r} prompt={prompt_len} rel_err={rel}"
        assert torch.equal(got_drafts, ref_drafts), f"world={world} rank={r}: {got_drafts} != {ref_drafts}"


def test_dfly2_multi_round_verify_overwrites_target_kv_cleanly(tmp_path):
    """Verify two consecutive DFly2 draft->verify rounds match plain target decode."""
    from ignition.recipes.hy4.bootstrap import PreparationConfig, construct_rank
    from ignition.recipes.hy4.decoder import Hy4Cache, Hy4Decoder

    snap, _ = _make_tiny_official_snapshot(tmp_path / "snap_rounds")
    settings = PreparationConfig(devices=(0,), mtp=True, prefill_chunk_size=64)
    cap = 64
    prompt = torch.tensor([[4 + (i * 5) % 80 for i in range(18)]], dtype=torch.long)

    def make_dec():
        built = construct_rank(snap, settings, rank=0, pinned=False, mtp=True,
                               cache_rows=1, cache_capacity=cap, world_size=1)
        w = _to_fp32_dfly_weights(built.weights)
        c = Hy4Cache.allocate(built.model_config, rows=1, capacity=cap, device=CPU,
                              draft=False, dtype=torch.float32, world=1, rank=0)
        return Hy4Decoder(w, c, compute_dtype=torch.float32)

    # Plain autoregressive reference decode of 8 tokens.
    plain = make_dec()
    lg = plain.forward(prompt, request_rows=[0])
    plain.complete([0], prompt)
    plain_out = [int(lg[0, -1].argmax())]
    for _ in range(7):
        step_lg = plain.decode_step(torch.tensor([[plain_out[-1]]], dtype=torch.long), request_rows=[0])
        plain.complete([0], torch.tensor([[plain_out[-1]]], dtype=torch.long))
        plain_out.append(int(step_lg[0, -1].argmax()))

    # DFly2 speculative greedy rounds (k=3): verify overwrites DFly2's ephemeral KV writes.
    spec_dec = make_dec()
    lg, hid, _ = spec_dec.forward(prompt, request_rows=[0], return_hidden=True)
    spec_dec.complete([0], prompt)
    bonus = int(lg[0, -1].argmax())
    last_hid = hid[:, -1, :]
    spec_out = [bonus]
    k = 3
    while len(spec_out) < len(plain_out):
        drafts, _ = spec_dec.dfly_draft(torch.tensor([bonus]), last_hid, request_rows=[0], k=k)
        verify_ids = torch.cat((torch.tensor([[bonus]], dtype=torch.long), drafts), dim=1)
        spec_dec.arm_dfly_capture(k + 1)
        try:
            v_lg, v_hid, _ = spec_dec.forward(verify_ids, request_rows=[0], return_hidden=True)
        finally:
            spec_dec.disarm_dfly_tap()
        target_pred = v_lg[0].argmax(dim=-1).tolist()
        draft_list = drafts[0].tolist()
        n = 0
        while n < k and draft_list[n] == target_pred[n]:
            n += 1
        spec_dec.append([0], n + 1)
        spec_dec.dfly_commit_capture(0, n)
        spec_out.extend(draft_list[:n] + [target_pred[n]])
        bonus = target_pred[n]
        last_hid = v_hid[:, n, :]

    assert spec_out[:len(plain_out)] == plain_out, f"spec {spec_out} != plain {plain_out}"


# ---------------------------------------------------------------------------
# 3. Per-row live row_cap_dev & relocation invariant on DFly2
# ---------------------------------------------------------------------------
def test_dfly2_preserves_per_row_live_capacity_and_relocation():
    """Ragged + idle-zero extents `[0, 12, 0, 6]` and live `relocate` on DFly2."""
    from ignition.recipes.hy4.decoder import Hy4Cache
    from ignition.recipes.hy4.dfly import DflyState

    # 4 cache rows, world=8, rank=1, 64 global/row => 8 local/row => 32 flat slots.
    cache = Hy4Cache(config=None, rows=4, capacity=64, device=CPU, world=8, rank=1)
    # Idle rows 0 and 2 have cap 0; active rows 1 (cap 12 @ base 0) and 3 (cap 6 @ base 12).
    cache.set_extents([0, 12, 0, 6], bases=[0, 0, 0, 12])

    state = DflyState(num_producers=2, rows=4, target_topk=8, draft_topk=16, device=CPU)

    # Record a target selection covering idle row 0 (with bogus slots) and active row 1.
    token_rows = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    # Physical slots on rank 1: row 1 has valid local offsets [0, 2, 5] and OOB offset 15;
    # row 3 has valid local offsets [1, 4] (base 12 -> physical 13, 16) and OOB offset 9 (21).
    sel = torch.tensor([
        [0, 1, 2, -1, -1, -1, -1, -1],          # row 0 (idle cap 0 -> must drop all!)
        [0, 2, 5, 15, -1, -1, -1, -1],          # row 1 (cap 12 -> keep 0,2,5; drop 15!)
        [0, -1, -1, -1, -1, -1, -1, -1],        # row 2 (idle cap 0 -> drop all!)
        [13, 16, 21, -1, -1, -1, -1, -1],       # row 3 (base 12, cap 6 -> keep loc 1,4; drop 9!)
    ], dtype=torch.int32)
    cnt = torch.tensor([3, 4, 1, 3], dtype=torch.int32)
    state.record_commit(
        0, token_rows, sel, cnt,
        base=cache.slots_for(token_rows), caps=cache.caps_for(token_rows),
        world=8, rank=1, is_global=False)

    assert state.hist_count[0].tolist()[:4] == [0, 3, 0, 2]
    assert state.hist_local[0, 1, :3].tolist() == [0, 2, 5]
    assert state.hist_local[0, 3, :2].tolist() == [1, 4]
    assert (state.hist_local[0, 0] == -1).all()
    assert (state.hist_local[0, 2] == -1).all()

    # Build DFly2 sparse indices for [row 0 (idle), row 1 (active @ P=48), row 3 (active @ P=40)]
    # with K=3 (W=4). On rank 1:
    # - row 1 anchors=48 -> block pos 48,49,50,51; rank 1 owns pos 49 (loc 6 < 12).
    # - row 3 anchors=40 -> block pos 40,41,42,43; rank 1 owns pos 41 (loc 5 < 6).
    rows_q = torch.tensor([0, 1, 3], dtype=torch.long)
    anchors = torch.tensor([8, 48, 40], dtype=torch.long)
    out_sel, out_cnt = state.build_sparse_indices(
        0, rows_q, anchors, cache=cache, query_width=4, world=8, rank=1)
    assert out_sel.shape == (12, 16) and out_cnt.shape == (12,)
    # Idle row 0: all 4 query slots must have count 0 and all -1 (zero OOB!).
    assert out_cnt[:4].tolist() == [0, 0, 0, 0]
    assert (out_sel[:4] == -1).all()
    # Active row 1: slot 0 sees history [0,2,5] (3); slots 1..3 also see block pos 49 -> slot 0+6=6 (4).
    assert out_cnt[4:8].tolist() == [3, 4, 4, 4]
    assert out_sel[4, :3].tolist() == [0, 2, 5] and (out_sel[4, 3:] == -1).all()
    for j in (5, 6, 7):
        assert out_sel[j, :4].tolist() == [0, 2, 5, 6]
        assert (out_sel[j, 4:] == -1).all()
    # Active row 3 (base 12): slot 0 sees [13,16] (2); slots 1..3 also see block pos 41 -> 12+5=17 (3).
    assert out_cnt[8:12].tolist() == [2, 3, 3, 3]
    assert out_sel[9, :3].tolist() == [13, 16, 17] and (out_sel[9, 3:] == -1).all()

    # Dynamic relocation: move row 1 from base 0 -> base 18 in place without mutating DflyState!
    cache.set_extents([0, 12, 0, 6], bases=[0, 18, 0, 0])
    rel_sel, rel_cnt = state.build_sparse_indices(
        0, rows_q, anchors, cache=cache, query_width=4, world=8, rank=1)
    assert rel_cnt[4:8].tolist() == [3, 4, 4, 4]
    # Relocated physical slots = 18 + [0, 2, 5, 6] = [18, 20, 23, 24]!
    assert rel_sel[5, :4].tolist() == [18, 20, 23, 24]
    # And row 3 relocated to base 0 -> [1, 4, 5]!
    assert rel_sel[9, :3].tolist() == [1, 4, 5]

    # Also verify global-to-local tap conversion (local/short_kv regime) respects row_cap_dev.
    S = 64
    g_sel = torch.tensor([
        [0 * S + 1, 0 * S + 9, -1, -1, -1, -1, -1, -1],   # idle row 0 -> 0 kept
        [1 * S + 1, 1 * S + 2, 1 * S + 9, 1 * S + 17, -1, -1, -1, -1],  # rank 1 owns 1, 9, 17 -> loc 0, 1, 2
    ], dtype=torch.int32)
    g_cnt = torch.tensor([2, 4], dtype=torch.int32)
    g_rows = torch.tensor([0, 1], dtype=torch.long)
    state.record_commit(
        1, g_rows, g_sel, g_cnt, base=g_rows * S, caps=cache.caps_for(g_rows),
        world=8, rank=1, is_global=True)
    assert state.hist_count[1, 0].item() == 0
    assert state.hist_count[1, 1].item() == 3
    assert state.hist_local[1, 1, :3].tolist() == [0, 1, 2]


def test_bspec_dfly2_batched_bodies_with_pad_and_ragged_extents(tmp_path):
    """Exercise DecodeGraphs `_body_bdraft_dfly` + `_body_bverify` with size=8, per=3 (rem=2 pad)."""
    from ignition.recipes.hy4.bootstrap import PreparationConfig, construct_rank
    from ignition.recipes.hy4.collectives import PcieDcpCollectives, SimulatedDcpTransport, SimulatedGroup
    from ignition.recipes.hy4.decode_graph import DecodeGraphs
    from ignition.recipes.hy4.decoder import Hy4Cache, Hy4Decoder
    from ignition.recipes.hy4.wiring import batch_shape

    snap, _ = _make_tiny_official_snapshot(tmp_path / "snap_bspec")
    settings = PreparationConfig(devices=(0,), mtp=True, prefill_chunk_size=64)
    world, cap = 2, 64
    group = SimulatedGroup(world)
    transport = SimulatedDcpTransport(group)
    decoders = []
    for r in range(world):
        built = construct_rank(snap, settings, rank=r, pinned=False, mtp=True,
                               cache_rows=4, cache_capacity=cap, world_size=world,
                               collectives=group.collectives(r))
        wr = _to_fp32_dfly_weights(built.weights)
        cr = Hy4Cache.allocate(built.model_config, rows=4, capacity=cap, device=CPU,
                               draft=False, dtype=torch.float32, world=world, rank=r)
        # Ragged + idle-zero extents: rows 0 & 2 idle (cap 0), rows 1 & 3 active (cap 32 each).
        cr.set_extents([0, 32, 0, 32], bases=[0, 0, 0, 32])
        col = PcieDcpCollectives(rank=r, world_size=world, transport=transport)
        decoders.append(Hy4Decoder(wr, cr, compute_dtype=torch.float32, collectives=col))

    # Stage 1: short context (<= index_topk=12) exercises full `_body_bdraft` -> `_body_bverify`
    # -> `commit_capture(target, src_idx)` across `size=8, per=3` (rem=2 trailing pads) AND
    # `size=8, per=4` (full padded slot on active row 1).
    p1_short = torch.tensor([[5, 6, 7, 8, 9]], dtype=torch.long)
    p3_short = torch.tensor([[20, 21, 22, 23, 24, 25]], dtype=torch.long)
    # Stage 2: sparse context (> index_topk=12) exercises `_body_bdraft("cp", 0)` reading
    # tapped sparse target top-12 selections + causal block slots on both ranks.
    p1_sparse = torch.tensor([[5 + i for i in range(16)]], dtype=torch.long)
    p3_sparse = torch.tensor([[20 + i for i in range(18)]], dtype=torch.long)

    def run_rank(dec):
        lg1, h1, _ = dec.forward(p1_short, request_rows=[1], return_hidden=True)
        dec.complete([1], p1_short)
        lg3, h3, _ = dec.forward(p3_short, request_rows=[3], return_hidden=True)
        dec.complete([3], p3_short)
        g = DecodeGraphs(dec, granule=4)
        # Fresh DecodeGraphs must stage both spec and batch rings before any prior run_spec/run_batch.
        assert g._spec_host(4).shape == (4,)
        assert g._batch_host(8).shape == (8,)
        st = g.bspec_state()
        for r, lg, h, L in ((1, lg1, h1, 5), (3, lg3, h3, 6)):
            b = int(lg[0, -1].argmax())
            st["pend_tok"][r].fill_(b)
            st["pend_hid"][r].copy_(h[0, -1].reshape(1, -1).expand(st["pend_hid"].shape[1], -1))
            st["spec_on"][r] = True
            g.bspec_set(r, P=L, c=L - 1, n=0)
        # Greedy rows (inv_temp = 0), size=8, per=3 (K=2, 2 active slots + 2 trailing pad entries).
        st["inv_temp"].zero_()
        rows_8 = [1, 1, 1, 3, 3, 3, 1, 1]
        pads_8 = [False] * 6 + [True, True]
        entry = g._bspec_args(8, 3)
        g._stage_bspec(entry, 8, rows_8, pads_8)
        shape = batch_shape(8)
        drafts, dflag = g._body_bdraft(8, shape, ("bypass", 0), 3)
        assert not bool(dflag)
        n_acc, vflag = g._body_bverify(8, shape, ("bypass", 0), 3)
        assert not bool(vflag)
        p_after_round1 = st["P"][[1, 3]].clone()
        # Verify DFly2 graph regime collapses context lengths past top-k to one key.
        assert g._bspec_regime("draft", 20, shape) == ("cp", 0)
        assert g._bspec_regime("draft", 200_000, shape) == ("cp", 0)
        # Full-slot padding (`B=1` inside `size=8, per=4`, K=3): slot 1 is padded on row 1
        # and writes at `P + 4` without corrupting slot 0's live `P .. P + 3` verify slots.
        rows_pad_slot = [1, 1, 1, 1, 1, 1, 1, 1]
        pads_pad_slot = [False] * 4 + [True] * 4
        entry4 = g._bspec_args(8, 4)
        g._stage_bspec(entry4, 8, rows_pad_slot, pads_pad_slot)
        drafts4, dflag4 = g._body_bdraft(8, shape, ("bypass", 0), 4)
        assert not bool(dflag4)
        n_acc4, vflag4 = g._body_bverify(8, shape, ("bypass", 0), 4)
        assert not bool(vflag4)
        hc_after_verify = dec.dfly_state.hist_count[:, [0, 1, 2, 3]].clone()

        # Stage 2: sparse context (> topk=12) -> `_body_bdraft` in `("cp", 0)` sparse regime.
        dec.release([1, 3])
        s_lg1, s_h1, _ = dec.forward(p1_sparse, request_rows=[1], return_hidden=True)
        dec.complete([1], p1_sparse)
        s_lg3, s_h3, _ = dec.forward(p3_sparse, request_rows=[3], return_hidden=True)
        dec.complete([3], p3_sparse)
        for r, lg, h, L in ((1, s_lg1, s_h1, 16), (3, s_lg3, s_h3, 18)):
            b = int(lg[0, -1].argmax())
            st["pend_tok"][r].fill_(b)
            st["pend_hid"][r].copy_(h[0, -1].reshape(1, -1).expand(st["pend_hid"].shape[1], -1))
            g.bspec_set(r, P=L, c=L - 1, n=0)
        g._stage_bspec(entry, 8, rows_8, pads_8)
        sparse_drafts, s_dflag = g._body_bdraft(8, shape, ("cp", 0), 3)
        assert not bool(s_dflag)
        return (drafts[:2, :2].clone(), n_acc[:2].clone(), p_after_round1,
                hc_after_verify, drafts4[0, :3].clone(), n_acc4[0].clone(),
                sparse_drafts[:2, :2].clone())

    res = group.run([lambda col, d=d: run_rank(d) for d in decoders])
    # All ranks agree identically on bypass + sparse drafts, acceptance counts, and updated P!
    for idx in (0, 1, 2, 4, 5, 6):
        assert torch.equal(res[0][idx], res[1][idx])
    # Idle rows 0 & 2 stayed at 0 count; active rows 1 & 3 updated cleanly!
    for r in range(world):
        hc = res[r][3]
        assert (hc[:, 0] == 0).all() and (hc[:, 2] == 0).all()
        assert (hc[:, 1] > 0).all() and (hc[:, 3] > 0).all()


# ---------------------------------------------------------------------------
# 4. BatchLoop bspec k-gate (unchanged) + text-only multimodal HTTP-400 check
# ---------------------------------------------------------------------------
def test_bspec_k_gate_and_multimodal_rejection():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "h4_serve_mod", Path(__file__).resolve().parents[2] / "deploy" / "hy4" / "h4_serve.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Verify BatchLoop._bs_k policy across B = 1..8 (k=3 at B<=3, k=2 at B=4, k=1 at B>4).
    loop = mod.BatchLoop.__new__(mod.BatchLoop)
    loop.ctx = SimpleNamespace(batch_rows=(2, 4, 8, 12, 16))
    loop.bspec_k3 = True
    loop.bspec_k2 = True
    assert [loop._bs_k(B) for B in range(1, 9)] == [3, 3, 3, 2, 1, 1, 1, 1]

    # Without bucket 12 prepared, k falls back to 1 everywhere.
    loop.ctx = SimpleNamespace(batch_rows=(2, 4, 8))
    assert [loop._bs_k(B) for B in range(1, 9)] == [1] * 8

    # Text-only multimodal rejection (mapped to HTTP 400 in do_POST).
    mod.reject_multimodal_messages([{"role": "user", "content": "hello"}])
    mod.reject_multimodal_messages([{"role": "user", "content": [{"type": "text", "text": "ok"}]}])
    for bad in (
        [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}]}],
        [{"role": "user", "content": [{"type": "input_image", "image_url": "https://x/y.png"}]}],
        [{"role": "user", "content": [{"type": "video_url", "video_url": {"url": "https://x/v.mp4"}}]}],
    ):
        with pytest.raises(ValueError, match="text-only"):
            mod.reject_multimodal_messages(bad)


# ---------------------------------------------------------------------------
# 5. Packed W4A16 official export round-trip with DFly2 BF16 + ViT skipped
# ---------------------------------------------------------------------------
def test_packed_w4a16_official_export_loads_and_drafts_on_cpu(tmp_path):
    from ignition.recipes.hy4.bootstrap import PreparationConfig, construct_rank

    snap, _ = _make_tiny_official_snapshot(tmp_path / "snap_packed", packed=True)
    settings = PreparationConfig(devices=(0,), mtp=True, prefill_chunk_size=64)
    built = construct_rank(snap, settings, rank=0, pinned=False, mtp=True,
                           cache_rows=1, cache_capacity=64, world_size=1)
    assert built.weights.dfly is not None and built.weights.draft is None
    assert built.memory_report["dfly"] is True
    # Verify routed experts loaded from packed W4A16 while DFly2 stayed pure BF16.
    assert built.weights.dfly.mtp_fusion.eh_proj.dtype == torch.bfloat16
    assert built.weights.dfly.layers[0].mlp.weights.gate_proj.dtype == torch.bfloat16

    dec = built.decoder
    prompt = torch.tensor([[3, 7, 11, 15, 19, 23]], dtype=torch.long)
    logits, hidden, _ = dec.forward(prompt, request_rows=[0], return_hidden=True)
    dec.complete([0], prompt)
    bonus = logits[:, -1, :].argmax(dim=-1)
    drafts, z = dec.dfly_draft(bonus, hidden[:, -1, :], request_rows=[0], k=3)
    assert drafts.shape == (1, 3) and z.shape == (1, 4, built.model_config.hidden)
    assert torch.isfinite(z).all()
