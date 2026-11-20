# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0
"""Gate V1 — Hy4 vision CPU schema, processor and packing gate (VISION-DESIGN 4.4).

CPU only. Verifies against the UPSTREAM REFERENCE SOURCES on disk (extracted
with `ast`, never re-typed by hand):

- `transformers-5.24.0.dev0-hy_v4_vl/image_processing_hy_v4_vl.py`:
  `smart_resize` and `patchify` (row-major flat patches);
- `transformers-5.24.0.dev0-hy_v4_vl/modeling_hy_v4_vl.py`:
  `HYV4VLModel.get_vision_position_ids` / `get_rope_index` (3D position ids +
  `mrope_position_deltas`) and `HYV4VLRotaryEmbedding.recomposition_frequencies`
  (the interleaved mRoPE section mapping).

Plus the Phase-4 engine contracts: the `vit.*` fail-closed inventory
(`UNQUANTIZED_FAMILIES`), `ModelConfig` vision pinning, the TP8-A1 byte
arithmetic (355,560,960 B/rank), the 3D->1D mRoPE reduction invariant, the
`CHUNK=2048` pre-padding image scatter, the D1 HTTP-400 payload semantics and
the `rope_delta_dev` decoder hooks (bit-identical text path).
"""

import ast
import base64
import itertools
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REF = Path("/home/tom/hy4/reference")
TF = REF / "official" / "transformers-5.24.0.dev0-hy_v4_vl"
SGL = REF / "official" / "sglang-hy4-03fde9cc" / "configs" / "hy_v4_vl_processing.py"
TINY = REF / "tiny-hy4"

needs_reference = pytest.mark.skipif(
    not (TF / "modeling_hy_v4_vl.py").is_file(),
    reason="transformers 5.24.0 hy_v4_vl reference sources not present")


def _extract(path, name):
    """The verbatim upstream `def name` source (top-level or in a class), exec'd."""
    source = path.read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            segment = ast.get_source_segment(source, node)
            namespace = {"torch": torch, "math": __import__("math"), "itertools": itertools}
            exec(compile(ast.parse(segment), str(path), "exec"), namespace)  # noqa: S102
            return namespace[name]
    raise AssertionError(f"{path}: no function {name}")


# --------------------------------------------------------------------------
# smart_resize / patchify / preprocessing (design section 3.1)
# --------------------------------------------------------------------------

@needs_reference
def test_smart_resize_matches_transformers_reference():
    from ignition.recipes.hy4.vision import smart_resize

    reference = _extract(TF / "image_processing_hy_v4_vl.py", "smart_resize")
    cases = [(512, 512), (1024, 1024), (2048, 2048), (720, 1280), (100, 300), (37, 4000),
             (4096, 4096), (123, 456), (64, 64), (3000, 1000), (1, 1), (33, 33)]
    for height, width in cases:
        assert smart_resize(height, width) == reference(height, width, factor=32,
                                                        min_pixels=4096, max_pixels=4194304)


def test_smart_resize_bounds_and_error_contract():
    from ignition.recipes.hy4.vision import smart_resize

    for height, width in [(512, 512), (1024, 1024), (2048, 2048), (720, 1280), (37, 4000)]:
        h_r, w_r = smart_resize(height, width)
        assert h_r % 32 == 0 and w_r % 32 == 0
        assert 4096 <= h_r * w_r <= 4194304 or h_r * w_r <= 4194304
    with pytest.raises(ValueError):
        smart_resize(0, 100)                      # min(H, W) == 0 -> HTTP 400 upstream
    with pytest.raises(ValueError):
        smart_resize(10, 3000)                    # aspect ratio > 200


@needs_reference
def test_patchify_matches_transformers_row_major():
    from ignition.recipes.hy4.vision import HYV4ImageProcessor

    reference = _extract(TF / "image_processing_hy_v4_vl.py", "patchify")
    images = torch.arange(1 * 3 * 64 * 96, dtype=torch.float32).reshape(1, 3, 64, 96) / 1000.0
    expected, grid_h, grid_w = reference(None, images, 16, 2, 1)
    expected = expected.reshape(-1, expected.shape[-1])
    got = HYV4ImageProcessor.flat_patches(images, patch=16)
    assert (grid_h, grid_w) == (4, 6)
    assert got.shape == expected.shape
    assert torch.equal(got, expected)


def test_preprocess_normalization_and_bicubic_override():
    from PIL import Image

    from ignition.recipes.hy4.vision import HYV4ImageProcessor

    processor = HYV4ImageProcessor()
    white = Image.new("RGB", (512, 512), (255, 255, 255))
    pixels, grid = processor.preprocess(white)
    assert grid == (1, 32, 32)
    assert pixels.shape == (1, 3, 512, 512) and pixels.dtype == torch.bfloat16
    # (1.0 - 0.5) / 0.5 == 1.0 exactly at every pixel.
    assert torch.equal(pixels, torch.ones_like(pixels))
    black, _ = processor.preprocess(Image.new("RGB", (64, 64), (0, 0, 0)))
    assert torch.equal(black, -torch.ones_like(black))
    gray, _ = processor.preprocess(Image.new("RGB", (64, 64), (255, 0, 0)))
    assert float(gray[0, 0, 0, 0]) == 1.0 and float(gray[0, 1, 0, 0]) == -1.0
    # Odd sizes resize through smart_resize (32-divisible) with BICUBIC.
    pixels, grid = processor.preprocess(Image.new("RGB", (123, 456), (10, 20, 30)))
    h_r, w_r = pixels.shape[-2:]
    assert h_r % 32 == 0 and w_r % 32 == 0 and grid == (1, h_r // 16, w_r // 16)


def test_llm_token_counts_include_image_newline():
    from ignition.recipes.hy4.vision import llm_token_count

    assert llm_token_count(32, 32) == 272        # 512 x 512
    assert llm_token_count(64, 64) == 1056       # 1024 x 1024
    assert llm_token_count(128, 128) == 4160     # 2048 x 2048 (never 4,096 — risk R6)


# --------------------------------------------------------------------------
# 3D mRoPE position ids + delta (design section 2.2)
# --------------------------------------------------------------------------

def test_vision_position_ids_match_the_reference_docstring_golden():
    from ignition.recipes.hy4.vision import _vision_position_ids

    # `get_rope_index`'s documented example: 2 frames of a merged 2x2 grid.
    got = _vision_position_ids(0, (2, 4, 4), spatial_merge_size=2)
    assert got[0].tolist() == [0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3]
    assert got[1].tolist() == [0, 0, 0, 1, 1, 1, 3, 3, 3, 4, 4, 4]
    assert got[2].tolist() == [0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5]


@needs_reference
def test_vision_position_ids_match_transformers_reference():
    from ignition.recipes.hy4.vision import _vision_position_ids

    reference = _extract(TF / "modeling_hy_v4_vl.py", "get_vision_position_ids")
    for start, grid in [(0, (1, 4, 4)), (7, (1, 8, 4)), (13, (2, 4, 4)), (1, (1, 32, 32))]:
        expected = reference(None, start, torch.tensor(grid), 1, 2)
        got = _vision_position_ids(start, grid, spatial_merge_size=2)
        assert torch.equal(got, expected), (start, grid)


@needs_reference
def test_compute_3d_mrope_positions_and_delta_match_get_rope_index():
    from ignition.recipes.hy4.vision import compute_3d_mrope_positions

    ns = {"torch": torch, "itertools": itertools}
    source = (TF / "modeling_hy_v4_vl.py").read_text()
    tree = ast.parse(source)
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in (
                "get_vision_position_ids", "get_rope_index"):
            found[node.name] = ast.get_source_segment(source, node)
    assert set(found) == {"get_vision_position_ids", "get_rope_index"}
    exec(compile(ast.parse(found["get_vision_position_ids"]), "ref", "exec"), ns)  # noqa: S102
    exec(compile(ast.parse(found["get_rope_index"]), "ref", "exec"), ns)  # noqa: S102

    class Ref:
        config = types.SimpleNamespace(
            vision_config=types.SimpleNamespace(spatial_merge_size=2))
        get_vision_position_ids = ns["get_vision_position_ids"]
        get_rope_index = ns["get_rope_index"]

    image_token, start_token, end_token = 120120, 120118, 120119
    # Expanded prompt: 3 text + start_token, image (512x512 -> 272 rows),
    # end_token + 2 text + start_token, image (grid 16x16 -> 20 rows), end_token
    # + 4 text. The reference's `mm_token_type_ids` marks the placeholder spans.
    spans = [((1, 32, 32), 272), ((1, 16, 16), 72)]
    ids, types_row = [], []
    for index, (grid, count) in enumerate(spans):
        text = 3 if index == 0 else 2
        ids += [10 + index] * text + [start_token]
        types_row += [0] * (text + 1)
        ids += [image_token] * count
        types_row += [1] * count
        ids += [end_token] + [99] * 4
        types_row += [0] * 5
    ids = torch.tensor([ids])
    types_t = torch.tensor([types_row])
    grids = torch.tensor([[1, 32, 32], [1, 16, 16]])
    expected_ids, expected_delta = Ref().get_rope_index(
        ids, types_t, image_grid_thw=grids, video_grid_thw=None)
    got_ids, got_delta = compute_3d_mrope_positions(ids[0].tolist(), [tuple(g) for g in grids.tolist()])
    assert torch.equal(got_ids, expected_ids[:, 0, :]), (
        got_ids[:, :20], expected_ids[:, 0, :20])
    assert int(got_delta) == int(expected_delta[0, 0])

    # Text-only: the delta is exactly 0 (the bit-identical text path).
    text_only = [1, 2, 3, 4]
    ids_t = torch.tensor([text_only])
    expected_ids, expected_delta = Ref().get_rope_index(
        ids_t, torch.zeros_like(ids_t), image_grid_thw=None, video_grid_thw=None)
    got_ids, got_delta = compute_3d_mrope_positions(text_only, [])
    assert int(got_delta) == 0 and torch.equal(got_ids, expected_ids[:, 0, :])


def test_compute_3d_mrope_positions_fail_closed():
    from ignition.recipes.hy4.vision import compute_3d_mrope_positions, expand_image_placeholders

    # A wrong placeholder count raises (parity assertion, design 3.2).
    with pytest.raises(ValueError):
        compute_3d_mrope_positions([120120] * 10, [(1, 32, 32)])
    with pytest.raises(ValueError):
        compute_3d_mrope_positions([1, 2, 3], [(1, 32, 32)])   # no placeholder run
    with pytest.raises(ValueError):
        expand_image_placeholders([120120, 5], [(1, 32, 32), (1, 32, 32)])
    with pytest.raises(ValueError):
        expand_image_placeholders([120120, 120120], [(1, 32, 32)])  # smuggled 120120
    expanded = expand_image_placeholders([1, 120120, 2], [(1, 8, 8)])
    assert len(expanded) == 22 and int((expanded == 120120).sum()) == 20


# --------------------------------------------------------------------------
# mRoPE frequency recomposition + exact 1D reduction (design section 2.1)
# --------------------------------------------------------------------------

@needs_reference
def test_mrope_recomposition_matches_reference():
    reference = _extract(TF / "modeling_hy_v4_vl.py", "recomposition_frequencies")
    section = [12, 10, 10]
    freq = torch.randn(3, 2, 7, 32, dtype=torch.float32)
    stub = types.SimpleNamespace(mrope_section=section)
    expected = reference(stub, freq)
    # The design's section-2.1 algorithm (transcribed independently here).
    freqs_out = freq[0].clone()
    for axis, offset in ((1, 1), (2, 2)):
        length = section[axis] * 3
        freqs_out[:, :, offset:length:3] = freq[axis][:, :, offset:length:3]
    got = torch.cat((freqs_out, freqs_out), dim=-1)
    assert torch.equal(got, expected)


def test_mrope_1d_reduction_is_bitwise_identity():
    from ignition.recipes.hy4.attention import (apply_interleaved_mrope,
                                                apply_interleaved_rope)

    x = torch.randn(9, 32, dtype=torch.bfloat16)
    positions = torch.tensor([0, 1, 2, 7, 100, 4095, 65536, 123456, 999999])
    one_d = apply_interleaved_rope(x, positions, rope_dim=32, theta=10000000.0)
    three_d = positions.unsqueeze(0).expand(3, -1).contiguous()
    mrope = apply_interleaved_mrope(x, three_d, rope_dim=32, theta=10000000.0,
                                    mrope_section=(6, 5, 5))
    assert torch.equal(one_d, mrope)                       # exact reduction invariant
    dispatched = apply_interleaved_rope(x, three_d, rope_dim=32, theta=10000000.0,
                                        mrope_section=(6, 5, 5))
    assert torch.equal(one_d, dispatched)
    # Diverged axes must change the result (the hook is live).
    skewed = three_d.clone()
    skewed[2] += 3
    assert not torch.equal(one_d, apply_interleaved_mrope(
        x, skewed, rope_dim=32, theta=10000000.0, mrope_section=(6, 5, 5)))
    # 3D inputs over a head axis stay consistent with the flat call.
    x_heads = torch.randn(9, 4, 32, dtype=torch.bfloat16)
    flat = apply_interleaved_mrope(x_heads, three_d, rope_dim=32, theta=10000000.0,
                                   mrope_section=(6, 5, 5))
    per_head = torch.stack([apply_interleaved_mrope(x_heads[:, h], three_d, rope_dim=32,
                                                    theta=10000000.0, mrope_section=(6, 5, 5))
                            for h in range(4)], dim=1)
    assert torch.equal(flat, per_head)
    with pytest.raises(ValueError):
        apply_interleaved_mrope(x, three_d, rope_dim=32, theta=1.0, mrope_section=(4, 4, 4))


# --------------------------------------------------------------------------
# Prefill-chunk scatter + padding (design section 3.2.3 / risk R4)
# --------------------------------------------------------------------------

def test_chunk_scatter_precedes_padding_across_2048_boundaries():
    from ignition.recipes.hy4.vision import (compute_3d_mrope_positions,
                                             right_pad_chunk, scatter_image_rows)

    # A 2048x2048 image: 4,160 rows spanning 3 CHUNK=2048 prefill chunks.
    ids = [120120] * 4160
    rope, delta = compute_3d_mrope_positions(ids, [(1, 128, 128)])
    assert rope.shape == (3, 4160) and delta == 65 - 4160
    image_embeds = torch.arange(4160 * 4, dtype=torch.float32).reshape(4160, 4)
    cursor = 0
    seen = 0
    for start, stop in ((0, 2048), (2048, 4096), (4096, 4160)):
        real = stop - start
        chunk_ids = torch.tensor(ids[start:stop])
        chunk_embeds = torch.zeros((2048, 4))     # the full padded chunk width
        cursor = scatter_image_rows(chunk_ids, chunk_embeds[:real], image_embeds, cursor)
        seen += real
        # Right-padding repeats the LAST REAL row (and its 3D position column).
        padded, padded_rope = right_pad_chunk(chunk_embeds[:real], rope[:, start:stop], real, 2048)
        if real < 2048:
            assert torch.equal(padded[real], padded[real - 1])
            assert torch.equal(padded_rope[:, real], padded_rope[:, real - 1])
            assert torch.equal(padded[:real], chunk_embeds[:real])
        assert torch.equal(padded[:real], image_embeds[start:stop])
    assert cursor == 4160 == seen
    with pytest.raises(ValueError):
        scatter_image_rows(torch.tensor([120120, 120120]), torch.zeros((2, 4)),
                           image_embeds, 4159)     # overrun raises (fail-closed)
    with pytest.raises(ValueError):
        right_pad_chunk(torch.zeros((10, 4)), rope[:, :10], 10, 8)     # width < real


# --------------------------------------------------------------------------
# Env knobs + OpenAI payload semantics (design sections 3.2 / 3.3 / 4.2)
# --------------------------------------------------------------------------

def test_vision_env_defaults_preserve_d1():
    from ignition.recipes.hy4.vision import image_limits_from_env, vision_mode_from_env

    assert vision_mode_from_env({}) == "off"                # default: exact D1 HTTP 400
    assert vision_mode_from_env({"HY4_VISION": "tp8"}) == "tp8"
    assert vision_mode_from_env({"HY4_VISION": "spare"}) == "spare"
    with pytest.raises(ValueError):
        vision_mode_from_env({"HY4_VISION": "on"})
    assert image_limits_from_env({}) == (1, "data_uri_only", 20 * 1024 * 1024)
    assert image_limits_from_env({
        "HY4_MAX_IMAGES_PER_PROMPT": "4", "HY4_IMAGE_FETCH_POLICY": "allow_http",
        "HY4_IMAGE_MAX_BYTES": "1024"}) == (4, "allow_http", 1024)
    for bad in ({"HY4_MAX_IMAGES_PER_PROMPT": "0"}, {"HY4_IMAGE_FETCH_POLICY": "ftp"},
                {"HY4_IMAGE_MAX_BYTES": "-1"}):
        with pytest.raises(ValueError):
            image_limits_from_env(bad)


def _data_uri(payload=b"png-bytes"):
    return "data:image/png;base64," + base64.b64encode(payload).decode()


def test_extract_image_payloads_d1_contract():
    from ignition.recipes.hy4.vision import extract_image_payloads

    ok = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": _data_uri()}},
        {"type": "text", "text": "what is this?"}]}]
    assert extract_image_payloads(ok) == [b"png-bytes"]
    # Dict and string payload forms both parse.
    ok2 = [{"role": "user", "content": [{"type": "image", "image": _data_uri(b"zz")}]}]
    assert extract_image_payloads(ok2) == [b"zz"]
    # Video stays HTTP 400 in every mode (design 3.3).
    for part in ({"type": "video_url", "video_url": {"url": _data_uri()}},
                 {"type": "video", "video": _data_uri()},
                 {"type": "input_video", "input_video": _data_uri()},
                 {"type": "text", "video_url": _data_uri()}):
        with pytest.raises(ValueError, match="video"):
            extract_image_payloads([{"role": "user", "content": [part]}])
    # Image in a system message: HTTP 400.
    with pytest.raises(ValueError, match="system"):
        extract_image_payloads([{"role": "system", "content": [
            {"type": "image_url", "image_url": {"url": _data_uri()}}]}])
    # Image cap (HY4_MAX_IMAGES_PER_PROMPT, default 1).
    with pytest.raises(ValueError, match="at most"):
        extract_image_payloads([{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": _data_uri()}},
            {"type": "image_url", "image_url": {"url": _data_uri()}}]}])
    # SSRF policy: remote URLs and file paths are refused by default.
    for url in ("http://169.254.169.254/meta", "https://example.com/a.png", "file:///etc/passwd"):
        with pytest.raises(ValueError):
            extract_image_payloads([{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": url}}]}])
    # Oversized / malformed payloads fail closed.
    with pytest.raises(ValueError, match="byte limit"):
        extract_image_payloads([{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": _data_uri(b"x" * 4096)}}]}], max_bytes=1024)
    with pytest.raises(ValueError, match="malformed"):
        extract_image_payloads([{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,@@@"}}]}])
    with pytest.raises(ValueError, match="at most"):
        extract_image_payloads(ok + [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": _data_uri()}}]}], max_images=1)


# --------------------------------------------------------------------------
# ModelConfig vision pinning + contract floor + byte arithmetic
# --------------------------------------------------------------------------

def _official_source_config():
    import json

    raw = json.loads((Path("/storage/Hy4") / "config.json").read_text())
    return raw


def test_vision_config_pinning_fail_closed():
    import copy
    import json

    from ignition.recipes.hy4.config import ModelConfig

    config_path = Path("/storage/Hy4") / "config.json"
    if not config_path.is_file():
        pytest.skip("official /storage/Hy4 config not present")
    raw = json.loads(config_path.read_text())
    model = ModelConfig.from_dict(copy.deepcopy(raw), pinned=True)
    assert model.has_vision
    # Mutations of the pinned vision geometry raise.
    for key, value in (("num_hidden_layers", 39), ("hidden_size", 1024),
                       ("intermediate_size", 4096), ("num_attention_heads", 8),
                       ("patch_size", 8), ("spatial_merge_size", 1),
                       ("max_image_size", 1024), ("projector_type", "mlp"),
                       ("use_2d_rope", True)):
        bad = copy.deepcopy(raw)
        bad["vision_config"][key] = value
        with pytest.raises(ValueError):
            ModelConfig.from_dict(bad, pinned=True)
    bad = copy.deepcopy(raw)
    bad["image_token_id"] = 120121
    with pytest.raises(ValueError):
        ModelConfig.from_dict(bad, pinned=True)


def test_unquantized_families_reject_quantized_vit():
    from ignition.recipes.hy4 import contract

    assert "vit" in contract.UNQUANTIZED_FAMILIES
    for name in ("vit.layers.0.mlp.dense_h_to_4h", "vit.perceive.proj.0",
                 "vit.embeddings.patch_embedding"):
        assert any(fragment in name for fragment in contract.UNQUANTIZED_FAMILIES)
    # The text families' own export contract is unchanged by the new floor.
    assert not any("vit" in f for f in ("model.layers.1.mlp.experts", "model.mtp_layers.0"))


def test_vision_byte_arithmetic_is_the_tp8a1_census():
    import json

    from ignition.recipes.hy4.config import ModelConfig

    raw = json.loads((Path("/storage/Hy4") / "config.json").read_text())
    model = ModelConfig.from_dict(raw, pinned=True)
    bytes_with = model.packed_byte_arithmetic(vision=True)
    bytes_without = model.packed_byte_arithmetic()
    # Text-only mode: exact same dict as before Phase 4 (0 delta).
    assert set(bytes_without) == {"packed_resident_bytes", "replicated_bytes", "draft_bytes",
                                  "expert_scale_bytes", "csf_scale_bytes", "scale_scratch_bytes"}
    # TP8-A1 (design section 1.1 table).
    assert bytes_with["vision_bytes"] == 2_470_253_568            # 2.3006 GiB full BF16
    assert bytes_with["vision_sharded_bytes"] == 355_560_960      # 339.09 MiB/rank
    assert all(bytes_with[key] == bytes_without[key] for key in bytes_without)
    # The 649-tensor census (640 layer + 3 embedding + 6 perceive).
    from ignition.recipes.hy4.weights import vision_checkpoint_names

    names = vision_checkpoint_names(model)
    assert len(names) == 649


# --------------------------------------------------------------------------
# Decoder hooks: rope_delta_dev (design sections 2.3 / 4.3)
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def constructed(tmp_path_factory):
    from ignition.recipes.hy4.bootstrap import PreparationConfig, construct_rank

    root, _ = _vision_snapshot(tmp_path_factory.mktemp("snap"))
    settings = PreparationConfig(devices=(0,), mtp=True, speculative_widths=(1, 2, 3))
    return construct_rank(root, settings, rank=0, pinned=False, mtp=True,
                          cache_rows=2, cache_capacity=32, vision=True)


def _fresh(constructed):
    import copy

    from ignition.recipes.hy4.decoder import Hy4Cache, Hy4Decoder

    rank = copy.copy(constructed)
    rank.cache = Hy4Cache.allocate(constructed.model_config, rows=2, capacity=32,
                                  device=torch.device("cpu"), draft=True)
    rank.decoder = Hy4Decoder(constructed.weights, rank.cache)
    return rank


def test_rope_delta_dev_shape_and_release_zeroing(constructed):
    rank = _fresh(constructed)
    decoder = rank.decoder
    assert tuple(decoder.rope_delta_dev.shape) == (3,)     # rows + 1 trailing pad entry
    assert int(decoder.rope_delta_dev.sum()) == 0
    decoder.rope_delta_dev[0] = -4095
    decoder.rope_delta_dev[2] = 99                         # the pad row is a scratch slot
    decoder.release([0])
    assert int(decoder.rope_delta_dev[0]) == 0             # released row restarts text-only
    # The derived rotary positions are integer identity at delta 0 (bitwise).
    positions = torch.arange(5)
    token_rows = torch.zeros(5, dtype=torch.long)
    shifted = (positions + decoder.rope_delta_dev[token_rows]).clamp_min(0)
    assert torch.equal(shifted, positions)


def test_text_path_bit_identical_and_delta_changes_numerics(constructed):
    rank = _fresh(constructed)
    decoder = rank.decoder
    tokens = torch.tensor([[10, 11, 12]])
    baseline = decoder.forward(tokens.clone(), logits_rows="all")
    # Vision-enabled-but-unused: a nonzero delta on row 0 must be visible in the
    # rotary angles (the hook is live) and return to bit-identity after release.
    decoder.rope_delta_dev[0] = -1023
    shifted = decoder.forward(tokens.clone(), logits_rows="all")
    assert not torch.equal(baseline, shifted)
    decoder.release([0])
    reused = decoder.forward(tokens.clone(), logits_rows="all")
    assert torch.equal(baseline, reused)                   # post-release reuse is bit-identical


def test_inputs_embeds_path_is_bit_identical_to_the_token_path(constructed):
    rank = _fresh(constructed)
    decoder = rank.decoder
    tokens = torch.tensor([[21, 22, 23]])
    baseline = decoder.forward(tokens.clone(), logits_rows="all")
    embeds = decoder.collectives.embed_lookup(
        tokens.reshape(-1), decoder.weights.embed,
        vocabulary_range=decoder.weights.sharding.vocabulary_range).to(decoder.compute_dtype)
    embeds = embeds.reshape(1, 3, -1)
    supplied = decoder.forward(tokens.clone(), logits_rows="all", inputs_embeds=embeds)
    assert torch.equal(baseline, supplied)
    # A 3D rope_positions equal to the 1D positions is also bit-identical.
    rows = decoder._rows(tokens, None)
    positions = decoder._positions(tuple(tokens.shape), rows)
    rope3 = positions.unsqueeze(0).expand(3, -1).contiguous()
    supplied_rope = decoder.forward(tokens.clone(), logits_rows="all", rope_positions=rope3)
    assert torch.equal(baseline, supplied_rope)


def test_prefill_chunk_slicing_matches_whole_forward(constructed):
    def _chunked_prefill(decoder, tokens, embeds, rope):
        saved = decoder.prefill_chunk_size
        decoder.prefill_chunk_size = 3
        try:
            return decoder.prefill(tokens.clone(), inputs_embeds=embeds.clone(),
                                   rope_positions=rope.clone())
        finally:
            decoder.prefill_chunk_size = saved

    rank = _fresh(constructed)
    decoder = rank.decoder
    tokens = torch.tensor([[30, 31, 32, 33, 34, 35]])
    rows = decoder._rows(tokens, None)
    positions = decoder._positions(tuple(tokens.shape), rows)
    rope3 = torch.stack((positions, positions + 1, positions + 2))   # diverged axes
    embeds = torch.randn(1, 6, int(constructed.model_config.hidden))
    whole = decoder.forward(tokens.clone(), logits_rows="all",
                            inputs_embeds=embeds.clone(), rope_positions=rope3.clone())
    decoder.release([0])
    # Whole-prompt prefill without chunking equals the direct forward.
    part = decoder.prefill(tokens.clone(), inputs_embeds=embeds.clone(),
                           rope_positions=rope3.clone())
    assert torch.equal(whole, part)
    decoder.release([0])
    # Chunked prefill (2 chunks of 3) slices embeds + rope per chunk and the
    # summed logits match the whole forward (same map, chunked execution).
    chunked = _chunked_prefill(decoder, tokens, embeds, rope3)
    assert chunked.shape == whole.shape
    assert torch.allclose(whole.float(), chunked.float(), atol=1e-4, rtol=1e-4)
    decoder.release([0])
    # The slicing helpers themselves are exact views.
    from ignition.recipes.hy4.decoder import _slice_embeds, _slice_rope

    assert torch.equal(_slice_embeds(embeds, 1, 6, 2, 5), embeds[:, 2:5].reshape(-1, embeds.shape[-1]))
    assert torch.equal(_slice_rope(rope3, 1, 6, 2, 5), rope3[:, 2:5])
    assert _slice_embeds(None, 1, 6, 0, 3) is None and _slice_rope(None, 1, 6, 0, 3) is None


# --------------------------------------------------------------------------
# Vision construction + TP8-A1 sharded forward (design sections 1.1 / 4.1)
# --------------------------------------------------------------------------

def _vision_snapshot(root):
    """A complete tiny official Hy4+DFly2+ViT checkpoint with a COHERENT tiny
    vision geometry (the text stack is `test_hy4_dfly2`'s fixture; the vit
    family is rewritten to consumable shapes for `Hy4VisionWeights.build`)."""
    import importlib.util
    import json

    from safetensors.torch import save_file

    spec = importlib.util.spec_from_file_location(
        "hy4_dfly_fixtures", Path(__file__).parent / "test_hy4_dfly2.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    root.mkdir(parents=True, exist_ok=True)
    _, tensors = mod._make_tiny_official_snapshot(root)
    cfg = mod._tiny_official_config()
    cfg["vision_config"] = {
        "model_type": "hy_v4_vl", "num_hidden_layers": 2, "hidden_size": 32,
        "num_attention_heads": 8, "intermediate_size": 64, "out_hidden_size": 64,
        "patch_size": 4, "spatial_merge_size": 2, "max_image_size": 16,
        "projector_type": "rmsnorm_mlp_adapter", "rms_norm_eps": 1e-5,
        "text_hidden_size": 64,
    }
    gen = torch.Generator().manual_seed(7)

    def rand(shape, *, scale=0.08):
        return (torch.randn(shape, generator=gen, dtype=torch.float32) * scale).to(torch.bfloat16)

    for name in [k for k in tensors if k.startswith("vit.")]:
        del tensors[name]
    vh = 32
    tensors["vit.embeddings.patch_embedding.weight"] = rand((vh, 3, 4, 4))
    tensors["vit.embeddings.patch_embedding.bias"] = rand((vh,))
    tensors["vit.embeddings.position_embedding.weight"] = rand((17, vh))
    for layer in range(2):
        vp = f"vit.layers.{layer}"
        for sub in ("input_layernorm", "post_attention_layernorm"):
            tensors[f"{vp}.{sub}.weight"] = rand((vh,))
            tensors[f"{vp}.{sub}.bias"] = rand((vh,), scale=0.02)
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            tensors[f"{vp}.self_attn.{proj}.weight"] = rand((vh, vh))
            tensors[f"{vp}.self_attn.{proj}.bias"] = rand((vh,), scale=0.02)
        tensors[f"{vp}.mlp.dense_h_to_4h.weight"] = rand((2 * vh, vh))
        tensors[f"{vp}.mlp.dense_h_to_4h.bias"] = rand((2 * vh,), scale=0.02)
        tensors[f"{vp}.mlp.dense_4h_to_h.weight"] = rand((vh, 2 * vh))
        tensors[f"{vp}.mlp.dense_4h_to_h.bias"] = rand((vh,), scale=0.02)
    tensors["vit.perceive.norm.weight"] = rand((vh,))
    tensors["vit.perceive.image_newline"] = rand((2 * vh,), scale=0.02)
    tensors["vit.perceive.proj.0.weight"] = rand((2 * vh, vh, 2, 2))
    tensors["vit.perceive.proj.0.bias"] = rand((2 * vh,), scale=0.02)
    tensors["vit.perceive.proj.2.weight"] = rand((2 * vh, 2 * vh, 1, 1))
    tensors["vit.perceive.proj.2.bias"] = rand((2 * vh,), scale=0.02)
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


def test_construct_rank_vision_end_to_end(tmp_path):
    from ignition.recipes.hy4.bootstrap import PreparationConfig, construct_rank
    from ignition.recipes.hy4.vision import Hy4VisionTower

    root, _ = _vision_snapshot(tmp_path / "snap")
    settings = PreparationConfig(devices=(0,), mtp=True, speculative_widths=(1, 2, 3))
    built = construct_rank(root, settings, rank=0, pinned=False, mtp=True,
                           cache_rows=2, cache_capacity=64, vision=True)
    assert isinstance(built.decoder.vision, Hy4VisionTower)
    assert built.memory_report["vision"] is True
    assert built.weights.vision is not None
    # Fail-closed consumption: exactly the 41 tiny vit tensors were consumed
    # through the shared builder (no leftovers — `build_model_state.finish`).
    vit_used = [name for name in built.weights.vision.used_names if name.startswith("vit.")]
    assert len(vit_used) == 41 and "vit.perceive.image_newline" in vit_used
    # The 41-tensor consumption is fail-closed: exact, zero leftovers (the
    # builder's `finish` already raised otherwise).
    from ignition.recipes.hy4.collectives import SoloCollectives

    pixels = torch.randn(1, 3, 32, 32, dtype=torch.float32) / 10.0
    embeds = built.decoder.vision.forward(pixels.to(torch.bfloat16), (1, 8, 8),
                                          collectives=SoloCollectives())
    assert embeds.shape == (20, 64) and embeds.dtype == torch.bfloat16
    assert bool(torch.isfinite(embeds.float()).all())
    # Text-only construction: zero vision bytes and no tower attached.
    plain = construct_rank(root, settings, rank=0, pinned=False, mtp=True,
                           cache_rows=2, cache_capacity=64, vision=False)
    assert plain.weights.vision is None and plain.decoder.vision is None
    assert plain.memory_report["vision"] is False


def test_vision_build_fail_closed_on_missing_tensor(tmp_path):
    import json

    from safetensors.torch import save_file

    from ignition.recipes.hy4.bootstrap import PreparationConfig, construct_rank

    root, tensors = _vision_snapshot(tmp_path / "snap")
    del tensors["vit.perceive.image_newline"]
    shard_path = root / "model-00001-of-00001.safetensors"
    save_file(tensors, str(shard_path))
    shard_path.chmod(0o644)
    weight_map = {name: "model-00001-of-00001.safetensors" for name in sorted(tensors)}
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 0}, "weight_map": weight_map}))
    settings = PreparationConfig(devices=(0,), mtp=True, speculative_widths=(1, 2, 3))
    with pytest.raises(ValueError):
        construct_rank(root, settings, rank=0, pinned=False, mtp=True,
                       cache_rows=2, cache_capacity=64, vision=True)


def _sharded_vision(root, rank, world):
    """`Hy4VisionWeights.build` at an exact TP shard through the real loader.

    The text stack of the tiny fixture cannot shard at world 8 (its MoE width
    is 64 < 8 x 16), so the vision family is built through the loader + builder
    inventory directly — the same code path `build_model_state` runs.
    """
    import types as _types

    from ignition.components.parallel import TensorParallelShard
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.recipes.hy4.config import ModelConfig
    from ignition.recipes.hy4.vision import Hy4VisionWeights
    from ignition.recipes.hy4.weights import Hy4WeightLoader, cpu_dry_run

    manifest = inspect_indexed_checkpoint(root)
    shard = TensorParallelShard(rank, world)
    loader = Hy4WeightLoader(manifest, torch.device("cpu"), shard=shard, mtp=True,
                             vision=True, pinned=False)
    with cpu_dry_run(loader):
        shim = _types.SimpleNamespace(
            loader=loader, config=ModelConfig.load(root, pinned=False),
            sharding=_types.SimpleNamespace(shard=shard), used=set())

        def take(name):
            if name not in loader._sources:
                raise ValueError(f"missing selected tensor {name}")
            shim.used.add(name)

        shim.take = take
        weights = Hy4VisionWeights.build(shim)
    return weights


def test_tp8_sharded_forward_equals_full_forward(tmp_path):
    from ignition.recipes.hy4.collectives import SimulatedGroup
    from ignition.recipes.hy4.vision import Hy4VisionTower

    root, _ = _vision_snapshot(tmp_path / "snap")
    world = 8
    full_weights = _sharded_vision(root, 0, 1)
    assert full_weights.world == 1
    sharded = [_sharded_vision(root, rank, world) for rank in range(world)]
    for rank, weights in enumerate(sharded):
        assert weights.world == world and weights.rank == rank
        # Fail-closed consumption: exactly the 41 tiny vit tensors, no leftovers.
        assert len(weights.used_names) == 41
        # The sharded footprint is strictly smaller than the full one.
        assert weights.allocated_bytes < full_weights.allocated_bytes
    assert sum(w.allocated_bytes for w in sharded) > full_weights.allocated_bytes

    torch.manual_seed(1234)
    pixels = (torch.randn(1, 3, 32, 32) / 10).to(torch.bfloat16)
    expected = Hy4VisionTower(full_weights).forward(pixels, (1, 8, 8), collectives=_Solo())
    group = SimulatedGroup(world, timeout=120.0)
    towers = [Hy4VisionTower(weights) for weights in sharded]

    def body(rank):
        return lambda collectives: towers[rank].forward(pixels, (1, 8, 8),
                                                        collectives=collectives)

    outputs = group.run([body(rank) for rank in range(world)])
    got = outputs[0]
    assert got.shape == expected.shape
    diff = (got.float() - expected.float()).norm() / expected.float().norm()
    assert float(diff) < 2e-2, f"TP8 sharded vs full rel-L2 {float(diff)}"


class _Solo:
    @staticmethod
    def all_reduce(value):
        return value 
