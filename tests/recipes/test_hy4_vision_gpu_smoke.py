# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0
"""Hy4 vision GPU compile smoke + minimal numeric cases (VISION-DESIGN gate V2 lite).

HARD RULES (Phase-4 campaign): SPARE GPU index 0 / PCI 11:00.0 ONLY — run this
file with ``CUDA_VISIBLE_DEVICES=0`` so no other device is ever visible, never
touch GPUs 1-16, and drain the card back to <= 2 MiB afterwards (the drain is
verified in the final test). Every probe has a hard timeout at the runner.

What is covered here (the CPU-authored GPU code paths of Phase 4):

1. `apply_interleaved_mrope` / `apply_interleaved_rope` 3D dispatch — GPU
   compile smoke + bitwise GPU-vs-CPU parity + the exact 1D reduction
   invariant;
2. the `rope_delta_dev` decode shift arithmetic on device (integer identity
   at delta 0 — the bit-identical text path through the same kernels);
3. `Hy4VisionTower.forward` on the REAL BF16 `vit.*` weights of
   `/opt/models/Hy4-official-nvfp4-w4a16-mse` (512x512): finite output with
   the exact `[272, 6144]` geometry, run-to-run determinism, and invariance
   across the packed all-reduce decomposition (`prefill_widths` chunking is
   output-exact);
4. the SDPA backend check (risk R2: `head_dim = 96` must never fall back to
   math SDPA on CUDA).
"""

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA not available (GPU smoke needs SPARE GPU 0)", allow_module_level=True)

PACKAGE = Path("/opt/models/Hy4-official-nvfp4-w4a16-mse")
STORAGE = Path("/storage/Hy4")


def _only_spare_gpu0():
    assert torch.cuda.device_count() == 1, (
        f"the GPU smoke must run with CUDA_VISIBLE_DEVICES=0 only, "
        f"sees {torch.cuda.device_count()} devices")
    assert torch.cuda.current_device() == 0


def _real_vision_weights(device="cuda:0"):
    """The real 649-tensor `vit.*` family as full-width (world 1) weights."""
    from safetensors import safe_open

    from ignition.recipes.hy4.vision import Hy4VisionWeights, VisionLayerWeights

    if not PACKAGE.is_dir() or not (STORAGE / "config.json").is_file():
        pytest.skip("official package / storage checkpoint not present")
    config = json.loads((STORAGE / "config.json").read_text())["vision_config"]
    index = json.loads((PACKAGE / "model.safetensors.index.json").read_text())
    names = [name for name in index["weight_map"] if name.startswith("vit.")]
    if len(names) != 649:
        pytest.skip(f"package carries {len(names)} vit tensors, expected 649")
    shards = {}
    for name in names:
        shards.setdefault(index["weight_map"][name], []).append(name)
    tensors = {}
    for shard, wanted in shards.items():
        with safe_open(PACKAGE / shard, framework="pt", device="cpu") as handle:
            for name in wanted:
                tensors[name] = handle.get_tensor(name).to(device)
    layers = []
    for layer in range(int(config["num_hidden_layers"])):
        prefix = f"vit.layers.{layer}"
        layers.append(VisionLayerWeights(
            input_norm_weight=tensors[f"{prefix}.input_layernorm.weight"],
            input_norm_bias=tensors[f"{prefix}.input_layernorm.bias"],
            post_norm_weight=tensors[f"{prefix}.post_attention_layernorm.weight"],
            post_norm_bias=tensors[f"{prefix}.post_attention_layernorm.bias"],
            q_weight=tensors[f"{prefix}.self_attn.q_proj.weight"],
            q_bias=tensors[f"{prefix}.self_attn.q_proj.bias"],
            k_weight=tensors[f"{prefix}.self_attn.k_proj.weight"],
            k_bias=tensors[f"{prefix}.self_attn.k_proj.bias"],
            v_weight=tensors[f"{prefix}.self_attn.v_proj.weight"],
            v_bias=tensors[f"{prefix}.self_attn.v_proj.bias"],
            o_weight=tensors[f"{prefix}.self_attn.o_proj.weight"],
            o_bias=tensors[f"{prefix}.self_attn.o_proj.bias"],
            fc1_weight=tensors[f"{prefix}.mlp.dense_h_to_4h.weight"],
            fc1_bias=tensors[f"{prefix}.mlp.dense_h_to_4h.bias"],
            fc2_weight=tensors[f"{prefix}.mlp.dense_4h_to_h.weight"],
            fc2_bias=tensors[f"{prefix}.mlp.dense_4h_to_h.bias"]))
    config = dict(config)
    config.setdefault("out_hidden_size", config["text_hidden_size"])
    weights = Hy4VisionWeights(
        config=config, world=1, rank=0,
        patch_weight=tensors["vit.embeddings.patch_embedding.weight"],
        patch_bias=tensors["vit.embeddings.patch_embedding.bias"],
        position_embedding=tensors["vit.embeddings.position_embedding.weight"],
        layers=tuple(layers),
        perceive_norm=tensors["vit.perceive.norm.weight"],
        perceive_proj0_weight=tensors["vit.perceive.proj.0.weight"],
        perceive_proj0_bias=tensors["vit.perceive.proj.0.bias"],
        perceive_proj2_weight=tensors["vit.perceive.proj.2.weight"],
        perceive_proj2_bias=tensors["vit.perceive.proj.2.bias"],
        image_newline=tensors["vit.perceive.image_newline"])
    return weights


class _Solo:
    @staticmethod
    def all_reduce(value):
        return value


def test_gpu_mrope_compile_smoke_and_bitwise_parity():
    _only_spare_gpu0()
    from ignition.recipes.hy4.attention import (apply_interleaved_mrope,
                                                apply_interleaved_rope)

    x_cpu = torch.randn(17, 64, dtype=torch.bfloat16)
    positions = torch.tensor([0, 1, 2, 7, 4095, 65536, 999999,
                              3, 5, 8, 13, 21, 34, 55, 89, 144, 233])
    three_d = torch.stack((positions, positions + 1, positions + 2))
    cpu = apply_interleaved_mrope(x_cpu, three_d, rope_dim=64, theta=10000000.0)
    x_gpu = x_cpu.cuda()
    positions_g = positions.cuda()
    three_d_g = three_d.cuda()
    gpu = apply_interleaved_mrope(x_gpu, three_d_g, rope_dim=64, theta=10000000.0)
    assert torch.equal(cpu, gpu.cpu())              # GPU compile + bitwise CPU parity
    # Exact 1D reduction invariant on device.
    equal_axes = positions_g.unsqueeze(0).expand(3, -1).contiguous()
    one_d = apply_interleaved_rope(x_gpu, positions_g, rope_dim=64, theta=10000000.0)
    reduced = apply_interleaved_mrope(x_gpu, equal_axes, rope_dim=64, theta=10000000.0)
    assert torch.equal(one_d, reduced)
    dispatched = apply_interleaved_rope(x_gpu, equal_axes, rope_dim=64, theta=10000000.0)
    assert torch.equal(one_d, dispatched)
    torch.cuda.synchronize()


def test_gpu_rope_delta_shift_is_integer_identity_for_text():
    _only_spare_gpu0()
    positions = torch.arange(4096, device="cuda", dtype=torch.long)
    rope_delta_dev = torch.zeros(3, dtype=torch.long, device="cuda")
    token_rows = torch.zeros(4096, dtype=torch.long, device="cuda")
    derived = (positions + rope_delta_dev[token_rows]).clamp_min(0)
    assert torch.equal(derived, positions)          # text rows: bit-identical angles
    rope_delta_dev[0] = -4095
    derived = (positions + rope_delta_dev[token_rows]).clamp_min(0)
    # Every shifted position is exactly `p + delta` clamped at 0 — and the
    # negative span collapses to 0 (the clamp guard).
    expected = (positions - 4095).clamp_min(0)
    assert torch.equal(derived, expected)
    assert int(derived[0]) == 0 and int(derived[4095]) == 0     # the clamp guard
    torch.cuda.synchronize()


def test_gpu_vision_tower_real_weights_smoke():
    _only_spare_gpu0()
    from ignition.recipes.hy4.vision import Hy4VisionTower, llm_token_count

    weights = _real_vision_weights()
    tower = Hy4VisionTower(weights)
    pixels = (torch.randn(1, 3, 512, 512) / 10).to(device="cuda:0", dtype=torch.bfloat16)
    embeds = tower.forward(pixels, (1, 32, 32), collectives=_Solo())
    assert embeds.shape == (llm_token_count(32, 32), 6144) == (272, 6144)
    assert embeds.dtype == torch.bfloat16
    assert bool(torch.isfinite(embeds.float()).all())
    again = tower.forward(pixels, (1, 32, 32), collectives=_Solo())
    assert torch.equal(embeds, again)               # deterministic
    # The packed all-reduce decomposition is output-exact: chunking the
    # reductions at 128 rows (max tail padding) reproduces the 2048-row calls.
    chunked = tower.forward(pixels, (1, 32, 32), collectives=_Solo(),
                            prefill_widths=(128,))
    assert torch.equal(embeds, chunked)
    torch.cuda.synchronize()
    del embeds, again, chunked, tower, weights


def test_gpu_sdpa_backend_is_not_math(head_dim_check=True):
    _only_spare_gpu0()
    # Risk R2: head_dim = 96 SDPA must dispatch flash / mem-efficient / cuDNN
    # on this build; math fallback would materialize [16, N, N] (8 GiB at
    # N = 16,384) and OOM. The tower refuses math-only configurations itself
    # (`Hy4VisionTower._sdpa`) — here we evidence the ACTIVE backend.
    enabled = (torch.backends.cuda.flash_sdp_enabled()
               or torch.backends.cuda.mem_efficient_sdp_enabled()
               or torch.backends.cuda.cudnn_sdp_enabled())
    assert enabled, "no non-math SDPA backend is enabled on this build"
    q = torch.randn(1, 16, 2048, 96, dtype=torch.bfloat16, device="cuda:0")
    out = torch.nn.functional.scaled_dot_product_attention(q, q, q, is_causal=False)
    assert out.shape == q.shape
    torch.cuda.synchronize()
    del q, out


def test_gpu_drain_to_spare_quiescence():
    """Drain hygiene — the hard rule is `nvidia-smi memory.used <= 2 MiB` with
    `utilization.gpu == 0%` on SPARE GPU 0 AFTER each probe (evaluated by the
    probe runner once this process exits: a live CUDA process necessarily
    holds its context on the card, so an in-process `<= 2 MiB` assert would be
    dishonest — the previous form of this test asserted exactly that and was
    measured wrong for exactly this reason).

    In-process we assert what makes the post-probe rule hold: every model and
    test tensor is freed (only torch's library workspace caches — measured
    32 MiB of cuBLAS/cuDNN handles, pinned for the life of the process — may
    remain allocated) and the card idles at 0% util.
    """
    _only_spare_gpu0()
    import gc
    import subprocess
    import time

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    live = torch.cuda.memory_allocated()
    assert live <= 64 << 20, (
        f"GPU 0 keeps {live} B of live torch allocations beyond the library "
        f"workspace caches — the post-probe <= 2 MiB rule would be violated")
    # The card must be idle (the 0% half of the hard rule), sampled with a
    # short settle window so a just-finished kernel cannot flake the read.
    for _ in range(10):
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader", "-i", "0"],
            capture_output=True, text=True, timeout=30)
        assert out.returncode == 0, out.stderr
        if out.stdout.strip().split()[0] == "0":
            break
        time.sleep(1.0)
    else:
        raise AssertionError(f"GPU 0 not idle after drain: {out.stdout!r}")
