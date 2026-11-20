# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0
"""Official `tencent/Hy4` vision tower (`vit.*`) and 3D mRoPE plumbing (Phase 4).

This module implements the VISION-DESIGN.md engineering spec:

- `HYV4ImageProcessor` (pure CPU): `smart_resize` (`factor=32`,
  `min_pixels=4096`, `max_pixels=4194304`, `max_aspect_ratio=200`), the exact
  preprocessing pipeline (`PIL.Image.Resampling.BICUBIC` override of the
  checkpoint's `resample: 1`, `[0.5, 0.5, 0.5]` normalization), row-major
  patchify, and the fail-closed OpenAI image-payload extraction
  (`HY4_IMAGE_FETCH_POLICY=data_uri_only` by default).
- `compute_3d_mrope_positions`: the interleaved 3D mRoPE position ids
  (`get_vision_position_ids` / `get_rope_index` semantics) and the scalar
  `mrope_delta` applied to every post-prompt token through
  `decoder.rope_delta_dev`.
- `Hy4VisionWeights.build(builder)`: exact consumption of all 649 `vit.*`
  tensors in BF16 through the `state._Builder` inventory (TP8-A1 sharding:
  column-shard `q/k/v` + `dense_h_to_4h` + `perceive.proj.0` on dim 0,
  row-shard `o_proj` + `dense_4h_to_h` + `perceive.proj.2` on dim 1; norms,
  `image_newline` and post-all-reduce biases replicate).
- `Hy4VisionTower.forward`: Conv2d patch embed, FP32 bilinear
  `position_embedding` interpolation (`scale_factor = ((h + 0.1) / 128,
  (w + 0.1) / 128)`, `align_corners=False`), 40 pre-norm ViT layers with
  full per-image bidirectional SDPA (`is_causal=False`, never materializing
  the `[16, N, N]` score matrix), and the TP8 `vit.perceive` adapter with
  row-wise `image_newline` append (`(H_m) * (W_m + 1)` LLM tokens per image —
  4,160 at `2048 x 2048`, never 4,096).

All TP8 layer reductions pack zero-copy into `[N // 4, 6144]` BF16 rows
(`6144 / 1536 == 4` and every `smart_resize` patch count is a multiple of 4)
and reduce through the existing `collectives.all_reduce` prepared-plan table
(`PREFILL_WIDTHS`), so the vision tower needs no new transport pool, no NCCL
communicator and zero extra communication VRAM (design section 1.3).

Row-parallel biases (`o_proj`, `dense_4h_to_h`, `perceive.proj.2`) are added
exactly once AFTER the all-reduce from the replicated copy (design section
1.1 bias rule).

Multi-image prompts encode one image at a time (bounded 240 MiB peak
transient activation per design section 1.2); callers concatenate the final
`[N_llm_img, 6144]` embeddings.
"""

import base64
import binascii
import io
import math
import os
import re
from dataclasses import dataclass

import torch
import torch.nn.functional as F

# --------------------------------------------------------------------------
# Preprocessing geometry (design section 3.1; verified against
# `image_processing_hy_v4_vl.py` and SGLang `hy_v4_vl_processing.py`).
# --------------------------------------------------------------------------
PATCH_SIZE = 16
SPATIAL_MERGE_SIZE = 2
SMART_RESIZE_FACTOR = PATCH_SIZE * SPATIAL_MERGE_SIZE      # 32
MIN_PIXELS = 4096
MAX_PIXELS = 4194304                                       # 2048 * 2048
MAX_ASPECT_RATIO = 200
IMAGE_MEAN = (0.5, 0.5, 0.5)                               # processor_config.json
IMAGE_STD = (0.5, 0.5, 0.5)
RESAMPLE_BICUBIC = "bicubic"                               # config `resample: 1` is NEVER used

# Special tokens (numeric ids; the checkpoint pins these at top level).
IMAGE_START_TOKEN_ID = 120118
IMAGE_END_TOKEN_ID = 120119
IMAGE_TOKEN_ID = 120120
IMAGE_NEWLINE_TOKEN_ID = 120121

# Prepared PCIe all-reduce widths (`collectives.PREFILL_WIDTHS`) and the packed
# reduction chunk (design section 1.3): `6144 / 1536 == 4` packing.
PREFILL_WIDTHS = (128, 256, 512, 1024, 2048)
PACK_CHUNK = 2048
REDUCE_WIDTH = 6144
PACK_FACTOR = REDUCE_WIDTH // 1536                         # == 4


def smart_resize(height, width, *, factor=SMART_RESIZE_FACTOR,
                 min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS):
    """Dynamic-resolution resize targets (`image_processing_hy_v4_vl.smart_resize`).

    Both dimensions land divisible by `factor`, the pixel count lands in
    `[min_pixels, max_pixels]` and the aspect ratio is preserved as closely as
    possible. Raises `ValueError` (HTTP 400 upstream) when `min(H, W) == 0` or
    the aspect ratio exceeds 200.
    """
    if min(height, width) <= 0:
        raise ValueError("image dimensions must be positive")
    if max(height, width) / min(height, width) > MAX_ASPECT_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_ASPECT_RATIO}, "
            f"got {max(height, width) / min(height, width)}")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def llm_token_count(grid_h, grid_w):
    """LLM placeholder tokens of one image: `(grid_h // 2) * (grid_w // 2 + 1)`.

    The trailing `+ 1` per merged row is the `image_newline` column, so the
    `2048 x 2048` count is 4,160 — never `img_max_token_num`'s 4,096 (design
    risk R6).
    """
    return (grid_h // SPATIAL_MERGE_SIZE) * (grid_w // SPATIAL_MERGE_SIZE + 1)


class HYV4ImageProcessor:
    """Pure-CPU image preprocessing for `HYV4VLImageProcessor` numerics.

    Pipeline (per image, rank 0 CPU; design section 3.1):
    1. decode to RGB;
    2. `smart_resize(H, W, factor=32, min_pixels=4096, max_pixels=4194304)`;
    3. resize with `PIL.Image.Resampling.BICUBIC` (the `resample: 1` LANCZOS
       recorded in `processor_config.json` is overridden upstream and must
       never be used);
    4. rescale uint8 by `1 / 255` (FP32) and normalize with mean/std
       `[0.5, 0.5, 0.5]`;
    5. return `pixel_bchw [1, 3, H_r, W_r]` (BF16 by default) plus
       `grid_thw (1, grid_h, grid_w)`.
    """

    def __init__(self, *, mean=IMAGE_MEAN, std=IMAGE_STD,
                 factor=SMART_RESIZE_FACTOR, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS):
        self.mean = tuple(float(v) for v in mean)
        self.std = tuple(float(v) for v in std)
        self.factor, self.min_pixels, self.max_pixels = factor, min_pixels, max_pixels

    def resize_target(self, height, width):
        return smart_resize(height, width, factor=self.factor,
                            min_pixels=self.min_pixels, max_pixels=self.max_pixels)

    def preprocess(self, image, *, dtype=torch.bfloat16):
        """`(pixel_bchw [1, 3, H_r, W_r], grid_thw (1, grid_h, grid_w))`."""
        from PIL import Image

        if isinstance(image, (bytes, bytearray, memoryview)):
            image = Image.open(io.BytesIO(bytes(image)))
        if not isinstance(image, Image.Image):
            raise ValueError("HYV4ImageProcessor expects a PIL image or encoded bytes")
        image = image.convert("RGB")
        width, height = image.size
        h_r, w_r = self.resize_target(height, width)
        # BICUBIC regardless of the checkpoint's recorded `resample: 1`.
        image = image.resize((w_r, h_r), resample=Image.Resampling.BICUBIC)
        pixels = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
        pixels = pixels.reshape(h_r, w_r, 3).permute(2, 0, 1).to(torch.float32) / 255.0
        mean = pixels.new_tensor(self.mean).reshape(3, 1, 1)
        std = pixels.new_tensor(self.std).reshape(3, 1, 1)
        pixels = (pixels - mean) / std
        grid = (1, h_r // PATCH_SIZE, w_r // PATCH_SIZE)
        return pixels.unsqueeze(0).to(dtype), grid

    @staticmethod
    def flat_patches(pixel_bchw, *, patch=PATCH_SIZE):
        """Row-major flat patches `[grid_h * grid_w, 3 * patch * patch]` (reference `patchify`).

        The Hy4 patchify permute is row-major across the whole image (unlike
        Qwen2-VL's block-grouped order), so `Conv2d(patch, patch)` over the
        image and this reshape produce the identical patch sequence.
        """
        _, channels, height, width = pixel_bchw.shape
        if height % patch or width % patch:
            raise ValueError("image dimensions must be divisible by patch_size")
        grid_h, grid_w = height // patch, width // patch
        return (pixel_bchw.reshape(1, channels, grid_h, patch, grid_w, patch)
                .permute(0, 2, 4, 1, 3, 5)
                .reshape(grid_h * grid_w, channels * patch * patch))


# --------------------------------------------------------------------------
# 3D interleaved mRoPE position ids (design section 2.2)
# --------------------------------------------------------------------------

def compute_3d_mrope_positions(ids, image_grid_thw, *, spatial_merge_size=SPATIAL_MERGE_SIZE,
                               image_token_id=IMAGE_TOKEN_ID):
    """`(rope_positions_3d [3, L] int64, mrope_delta int)` of one expanded prompt.

    `ids` is the placeholder-expanded prompt (the `i`-th contiguous run of
    `image_token_id` must hold exactly `llm_token_count(grid)` rows for image
    `i`); `image_grid_thw` lists the images' `(1, grid_h, grid_w)` in document
    order. Text tokens (including `image_start_token_id` before and
    `image_end_token_id` after each image span) get 1D positions `T = H = W`;
    the merged image rows receive the row-major 3D block of
    `get_vision_position_ids` (`T = start_pos`, `H = start_pos + r`,
    `W = start_pos + c`, the trailing `image_newline` column at `c = W_m`).

    `mrope_delta = (max_pos + 1) - len(ids)` (always `<= 0`; 0 for text-only
    prompts) — the scalar shift every post-prompt token applies through
    `decoder.rope_delta_dev`.
    """
    if not isinstance(ids, torch.Tensor):
        ids = torch.tensor(list(ids), dtype=torch.long)
    if ids.ndim != 1:
        raise ValueError("ids must be a flat 1D token sequence")
    grids = [(int(t), int(h), int(w)) for t, h, w in image_grid_thw]
    device = ids.device
    L = int(ids.shape[0])
    positions = torch.empty((3, L), dtype=torch.long, device=device)
    cursor = 0            # walking 1D position of the next text token
    written = 0           # rows of `ids` consumed
    span = 0              # index into `grids`
    is_image = ids == image_token_id
    while written < L:
        if bool(is_image[written]):
            # One contiguous image placeholder run.
            start = written
            while written < L and bool(is_image[written]):
                written += 1
            if span >= len(grids):
                raise ValueError("more image placeholder runs than image grids")
            t, grid_h, grid_w = grids[span]
            span += 1
            count = t * llm_token_count(grid_h, grid_w)
            if written - start != count:
                raise ValueError(
                    f"image {span} placeholder run holds {written - start} rows, "
                    f"the grid requires {count}")
            vision = _vision_position_ids(cursor, (t, grid_h, grid_w),
                                          spatial_merge_size=spatial_merge_size, device=device)
            positions[:, start:written] = vision
            cursor = int(vision.max()) + 1
        else:
            positions[:, written] = cursor
            cursor += 1
            written += 1
    if span != len(grids):
        raise ValueError(f"{len(grids)} image grids supplied but only {span} placeholder runs found")
    delta = (int(positions.max()) + 1) - L
    return positions, delta


def _vision_position_ids(start_position, grid_thw, *, spatial_merge_size, device=None):
    """`get_vision_position_ids(start_position, grid_thw, 1, spatial_merge_size)`.

    Every frame is laid out as `H_m` rows of `W_m + 1` tokens (each merged row
    ends with an `image_newline` column) and every frame starts a new block of
    `max(H_m, W_m + 1)` positions — transcribed from
    `modeling_hy_v4_vl.py::HYV4VLModel.get_vision_position_ids`.
    """
    t, grid_h, grid_w = (int(v) for v in grid_thw)
    llm_grid_t = t
    llm_grid_h = grid_h // spatial_merge_size
    llm_grid_w = grid_w // spatial_merge_size
    position_temporal = torch.arange(llm_grid_t, device=device) * max(llm_grid_h, llm_grid_w + 1)
    position_height = torch.arange(llm_grid_h, device=device)
    position_width = torch.arange(llm_grid_w + 1, device=device)
    t_grid, h_grid, w_grid = torch.meshgrid(position_temporal, position_height,
                                            position_width, indexing="ij")
    ids = torch.stack([t_grid, h_grid + t_grid, w_grid + t_grid], dim=0).reshape(3, -1)
    return ids + start_position


def expand_image_placeholders(ids, image_grid_thw, *, image_token_id=IMAGE_TOKEN_ID):
    """Expand each `image_token_id` occurrence into `llm_token_count` copies.

    Fail-closed parity assertions (design section 3.2): exactly one placeholder
    per image BEFORE expansion, exactly `sum(N_llm_img)` AFTER — a mismatch
    (e.g. user text smuggling a raw `image_token_id`) raises `ValueError`
    (HTTP 400 upstream).
    """
    if not isinstance(ids, torch.Tensor):
        ids = torch.tensor(list(ids), dtype=torch.long)
    grids = [(int(t), int(h), int(w)) for t, h, w in image_grid_thw]
    before = int((ids == image_token_id).sum())
    if before != len(grids):
        raise ValueError(
            f"prompt carries {before} image placeholders for {len(grids)} images")
    out = []
    span = 0
    for token in ids.tolist():
        if token == image_token_id:
            t, grid_h, grid_w = grids[span]
            span += 1
            out.extend([image_token_id] * (t * llm_token_count(grid_h, grid_w)))
        else:
            out.append(token)
    expanded = torch.tensor(out, dtype=torch.long, device=ids.device)
    after = int((expanded == image_token_id).sum())
    expected = sum(t * llm_token_count(h, w) for t, h, w in grids)
    if after != expected:
        raise ValueError(f"expansion produced {after} image rows, expected {expected}")
    return expanded


# --------------------------------------------------------------------------
# Prefill-chunk scatter (design section 3.2.3 + risk R4)
# --------------------------------------------------------------------------

def scatter_image_rows(chunk_ids, chunk_embeds, image_embeds, cursor, *,
                       image_token_id=IMAGE_TOKEN_ID):
    """Scatter `image_embeds` rows over one REAL chunk's placeholder rows.

    Runs strictly over `0 .. real - 1` BEFORE any right-padding (padding a
    chunk that ends mid-image must never duplicate `image_token_id` rows and
    overrun `image_embeds`). Returns the advanced cursor.
    """
    real = int(chunk_ids.shape[0])
    if real == 0:
        return cursor
    hit = (chunk_ids == image_token_id).nonzero(as_tuple=True)[0]
    if int(hit.shape[0]) == 0:
        return cursor
    end = cursor + int(hit.shape[0])
    if end > int(image_embeds.shape[0]):
        raise ValueError("image embedding rows overrun the placeholder span")
    chunk_embeds[hit] = image_embeds[cursor:end].to(chunk_embeds.dtype)
    return end


def right_pad_chunk(chunk_embeds, rope_positions, real, width):
    """Right-pad a chunk to `width` rows: `width - real` copies of row `real - 1`.

    In embedding space the pad repeats the last real row; in rotary space it
    repeats the last real 3D position column. Returns
    `(padded_embeds, padded_rope)`.
    """
    if width < real:
        raise ValueError("chunk width cannot shrink below the real token count")
    if width == real:
        return chunk_embeds, rope_positions
    tail = chunk_embeds[real - 1:real].expand(width - real, -1)
    embeds = torch.cat((chunk_embeds[:real], tail), dim=0)
    rope_tail = rope_positions[:, real - 1:real].expand(3, width - real)
    rope = torch.cat((rope_positions[:, :real], rope_tail), dim=1)
    return embeds, rope


# --------------------------------------------------------------------------
# OpenAI image payload extraction (design sections 3.2 / 3.3)
# --------------------------------------------------------------------------

_MULTIMODAL_PART_TYPES = ("image", "image_url", "input_image",
                          "video", "video_url", "input_video")
_VIDEO_PART_TYPES = ("video", "video_url", "input_video")
_IMAGE_PART_TYPES = ("image", "image_url", "input_image")
_DATA_URI = re.compile(r"^data:(?P<mime>[^;,]+)?(?:;[^,]*)*;base64,(?P<body>.*)$",
                       re.DOTALL)

DEFAULT_MAX_IMAGE_BYTES = 20 * 1024 * 1024        # HY4_IMAGE_MAX_BYTES (20 MiB)
DEFAULT_MAX_IMAGES = 1                            # HY4_MAX_IMAGES_PER_PROMPT
DEFAULT_FETCH_POLICY = "data_uri_only"            # HY4_IMAGE_FETCH_POLICY


def _part_kind(part):
    kind = part.get("type")
    if kind in _VIDEO_PART_TYPES or "video_url" in part or "input_video" in part:
        return "video"
    if kind in _IMAGE_PART_TYPES or "image_url" in part or "image" in part:
        return "image"
    return None


def _part_payload(part):
    for key in ("image_url", "image"):
        if key in part:
            value = part[key]
            if isinstance(value, dict):
                return value.get("url")
            return value
    return part.get("url")


def extract_image_payloads(messages, *, fetch_policy=DEFAULT_FETCH_POLICY,
                          max_bytes=DEFAULT_MAX_IMAGE_BYTES,
                          max_images=DEFAULT_MAX_IMAGES):
    """Extract one encoded-byte payload per image content part (document order).

    Raises `ValueError` (mapped to HTTP 400) for video parts anywhere, image
    parts in `system` messages, more than `max_images` images, oversized
    payloads, malformed base64, or non-data URIs under
    `fetch_policy="data_uri_only"` (SSRF / local-file disclosure prevention).
    """
    if fetch_policy not in ("data_uri_only", "allow_http"):
        raise ValueError(f"unknown HY4_IMAGE_FETCH_POLICY {fetch_policy!r}")
    if not isinstance(messages, (list, tuple)):
        raise ValueError("messages must be a list")
    payloads = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        parts = content if isinstance(content, list) else []
        for part in parts:
            if not isinstance(part, dict):
                continue
            kind = _part_kind(part)
            if kind is None:
                continue
            if kind == "video":
                raise ValueError("video inputs are not enabled on this serve")
            if role == "system":
                raise ValueError("image inputs are not allowed in system messages")
            payloads.append(_decode_image_payload(
                _part_payload(part), fetch_policy=fetch_policy, max_bytes=max_bytes))
            if len(payloads) > max_images:
                raise ValueError(
                    f"at most {max_images} image(s) per prompt are enabled on this serve")
    return payloads


def _decode_image_payload(url, *, fetch_policy, max_bytes):
    if not isinstance(url, str) or not url:
        raise ValueError("image content part carries no URL or payload")
    match = _DATA_URI.match(url)
    if match is not None:
        try:
            raw = base64.b64decode(match.group("body"), validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError(f"malformed image data URI: {exc}") from exc
    elif url.startswith(("http://", "https://")):
        if fetch_policy != "allow_http":
            raise ValueError("remote image URLs are disabled on this serve (HY4_IMAGE_FETCH_POLICY)")
        raise ValueError("remote image fetching is not implemented; inline a data URI")
    elif url.startswith("file://"):
        raise ValueError("file image paths are disabled on this serve")
    else:
        raise ValueError("unsupported image URL scheme")
    if len(raw) > max_bytes:
        raise ValueError(f"image payload exceeds the {max_bytes} byte limit")
    if not raw:
        raise ValueError("empty image payload")
    return raw


# --------------------------------------------------------------------------
# Env knobs (design section 4.2)
# --------------------------------------------------------------------------

def vision_mode_from_env(env=None):
    """`HY4_VISION` boot mode: `off` (default, exact D1 HTTP 400), `tp8`, `spare`."""
    env = os.environ if env is None else env
    mode = (env.get("HY4_VISION") or "off").strip().lower()
    if mode not in ("off", "tp8", "spare"):
        raise ValueError(f"HY4_VISION must be off, tp8 or spare; got {mode!r}")
    return mode


def image_limits_from_env(env=None):
    """`(max_images, fetch_policy, max_bytes)` from the boot environment."""
    env = os.environ if env is None else env
    try:
        max_images = int(env.get("HY4_MAX_IMAGES_PER_PROMPT") or DEFAULT_MAX_IMAGES)
    except ValueError as exc:
        raise ValueError("HY4_MAX_IMAGES_PER_PROMPT must be a positive integer") from exc
    if max_images < 1:
        raise ValueError("HY4_MAX_IMAGES_PER_PROMPT must be a positive integer")
    policy = (env.get("HY4_IMAGE_FETCH_POLICY") or DEFAULT_FETCH_POLICY).strip().lower()
    if policy not in ("data_uri_only", "allow_http"):
        raise ValueError(f"HY4_IMAGE_FETCH_POLICY must be data_uri_only or allow_http; got {policy!r}")
    try:
        max_bytes = int(env.get("HY4_IMAGE_MAX_BYTES") or DEFAULT_MAX_IMAGE_BYTES)
    except ValueError as exc:
        raise ValueError("HY4_IMAGE_MAX_BYTES must be a positive integer") from exc
    if max_bytes < 1:
        raise ValueError("HY4_IMAGE_MAX_BYTES must be a positive integer")
    return max_images, policy, max_bytes


# --------------------------------------------------------------------------
# Vision weights (design section 1.1, TP8-A1 sharding)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class VisionLayerWeights:
    """One ViT layer's TP shard. Row-parallel biases are replicated copies
    added once after the all-reduce (design bias rule)."""
    input_norm_weight: torch.Tensor
    input_norm_bias: torch.Tensor
    post_norm_weight: torch.Tensor
    post_norm_bias: torch.Tensor
    q_weight: torch.Tensor          # [hidden // world * heads_ratio ...] col-shard dim 0
    q_bias: torch.Tensor
    k_weight: torch.Tensor
    k_bias: torch.Tensor
    v_weight: torch.Tensor
    v_bias: torch.Tensor
    o_weight: torch.Tensor          # [hidden, local] row-shard dim 1
    o_bias: torch.Tensor            # [hidden] replicated
    fc1_weight: torch.Tensor        # [inter // world, hidden] col-shard dim 0
    fc1_bias: torch.Tensor
    fc2_weight: torch.Tensor        # [hidden, inter // world] row-shard dim 1
    fc2_bias: torch.Tensor          # [hidden] replicated


@dataclass(frozen=True)
class Hy4VisionWeights:
    """The TP-sharded BF16 `vit.*` tower of one rank (649 tensors at TP8:
    339.09 MiB/rank = 355,560,960 B)."""
    config: dict                    # the `vision_config` dict
    world: int
    rank: int
    patch_weight: torch.Tensor      # [hidden, 3, patch, patch] replicated
    patch_bias: torch.Tensor        # [hidden] replicated
    position_embedding: torch.Tensor  # [num_positions, hidden] replicated
    layers: tuple
    perceive_norm: torch.Tensor     # [hidden] replicated
    perceive_proj0_weight: torch.Tensor   # [inter // world, hidden, 2, 2] dim-0 shard
    perceive_proj0_bias: torch.Tensor     # [inter // world]
    perceive_proj2_weight: torch.Tensor   # [out, inter // world, 1, 1] dim-1 shard
    perceive_proj2_bias: torch.Tensor     # [out] replicated (post all-reduce)
    image_newline: torch.Tensor     # [out] replicated
    used_names: frozenset = frozenset()

    @property
    def hidden(self):
        return int(self.config["hidden_size"])

    @property
    def heads(self):
        return int(self.config["num_attention_heads"])

    @property
    def head_dim(self):
        return self.hidden // self.heads

    @property
    def patch(self):
        return int(self.config.get("patch_size", PATCH_SIZE))

    @property
    def merge(self):
        return int(self.config.get("spatial_merge_size", SPATIAL_MERGE_SIZE))

    @property
    def intermediate(self):
        return int(self.config["intermediate_size"])

    @property
    def out_hidden(self):
        return int(self.config.get("out_hidden_size", self.config["text_hidden_size"]))

    @property
    def num_positions(self):
        return int(self.position_embedding.shape[0])

    @property
    def position_edge(self):
        """Learned position grid edge (`max_image_size // patch_size` = 128)."""
        return int(self.config.get("max_image_size", 2048)) // self.patch

    @property
    def rms_eps(self):
        return float(self.config.get("rms_norm_eps", 1e-5))

    @property
    def layer_norm_eps(self):
        return float(self.config.get("layer_norm_eps", 1e-5))

    @property
    def allocated_bytes(self):
        total, seen = 0, set()
        for name in self.__dataclass_fields__:
            item = getattr(self, name)
            for tensor in (item if isinstance(item, tuple) else (item,)):
                if isinstance(tensor, torch.Tensor):
                    storage = tensor.untyped_storage()
                    if id(storage) not in seen:
                        seen.add(id(storage))
                        total += storage.nbytes()
        return total

    @classmethod
    def build(cls, builder):
        """Consume exactly the checkpoint's `vit.*` tensors through `_Builder`.

        TP8-A1 (design section 1.1): column-shard `q/k/v`, `dense_h_to_4h` and
        `perceive.proj.0` on dim 0 (bias rows shard with them); row-shard
        `o_proj`, `dense_4h_to_h` and `perceive.proj.2` on dim 1 (their biases
        replicate and join the output AFTER the all-reduce); norms,
        `position_embedding`, `patch_embedding` and `image_newline` replicate.
        """
        config = dict(builder.config.text["vision_config"])
        world = builder.sharding.shard.world_size
        rank = builder.sharding.shard.rank
        hidden = int(config["hidden_size"])
        layers = int(config["num_hidden_layers"])
        heads = int(config["num_attention_heads"])
        inter = int(config["intermediate_size"])
        out_hidden = int(config.get("out_hidden_size", builder.config.hidden))
        patch = int(config.get("patch_size", PATCH_SIZE))
        merge = int(config.get("spatial_merge_size", SPATIAL_MERGE_SIZE))
        positions = int(config.get("max_image_size", 2048)) // patch
        positions = positions * positions + 1                  # + 1 unused CLS slot
        if heads % world or inter % world:
            raise ValueError(
                f"Hy4 vision sharding needs heads ({heads}) and intermediate ({inter}) "
                f"divisible by the world ({world})")
        if config.get("projector_type", "rmsnorm_mlp_adapter") != "rmsnorm_mlp_adapter":
            raise ValueError("Hy4 vision requires projector_type=rmsnorm_mlp_adapter")
        if config.get("spatial_merge_size", SPATIAL_MERGE_SIZE) != SPATIAL_MERGE_SIZE:
            raise ValueError("Hy4 vision requires spatial_merge_size=2")
        if config.get("use_2d_rope"):
            raise ValueError("Hy4 vision carries no 2D ViT RoPE (use_2d_rope must be unset)")

        def take_tensor(name, shape, *, dtype=torch.bfloat16, extent=None):
            builder.take(name)
            source = builder.loader._sources[name]
            if tuple(source.shape) != tuple(shape):
                raise ValueError(f"{name}: expected {tuple(shape)}, checkpoint has {tuple(source.shape)}")
            if source.dtype != dtype:
                raise ValueError(f"{name}: expected {dtype}, checkpoint stores {source.dtype}")
            return builder.loader.load_tensor(name, extent=extent)

        def shard_range(width, *, alignment=1):
            return builder.sharding.shard.partition(width, alignment=alignment)

        head_width = hidden // heads                     # 96

        patch_weight = take_tensor("vit.embeddings.patch_embedding.weight",
                                   (hidden, 3, patch, patch))
        patch_bias = take_tensor("vit.embeddings.patch_embedding.bias", (hidden,))
        position_embedding = take_tensor("vit.embeddings.position_embedding.weight",
                                         (positions, hidden))

        q_rows = shard_range(hidden)                    # q/k/v shard whole rows (1536/8 = 192)
        inter_rows = shard_range(inter)
        o_cols = shard_range(hidden)                    # o_proj / fc2 shard columns
        layer_states = []
        for layer in range(layers):
            prefix = f"vit.layers.{layer}"
            ln_w1 = take_tensor(f"{prefix}.input_layernorm.weight", (hidden,))
            ln_b1 = take_tensor(f"{prefix}.input_layernorm.bias", (hidden,))
            ln_w2 = take_tensor(f"{prefix}.post_attention_layernorm.weight", (hidden,))
            ln_b2 = take_tensor(f"{prefix}.post_attention_layernorm.bias", (hidden,))
            attn = {}
            for proj in ("q_proj", "k_proj", "v_proj"):
                attn[proj] = (
                    take_tensor(f"{prefix}.self_attn.{proj}.weight", (hidden, hidden),
                                extent=(slice(*q_rows), slice(None))),
                    take_tensor(f"{prefix}.self_attn.{proj}.bias", (hidden,),
                                extent=slice(*q_rows)))
            o_weight = take_tensor(f"{prefix}.self_attn.o_proj.weight", (hidden, hidden),
                                   extent=(slice(None), slice(*o_cols)))
            o_bias = take_tensor(f"{prefix}.self_attn.o_proj.bias", (hidden,))
            fc1_weight = take_tensor(f"{prefix}.mlp.dense_h_to_4h.weight", (inter, hidden),
                                     extent=(slice(*inter_rows), slice(None)))
            fc1_bias = take_tensor(f"{prefix}.mlp.dense_h_to_4h.bias", (inter,),
                                   extent=slice(*inter_rows))
            fc2_weight = take_tensor(f"{prefix}.mlp.dense_4h_to_h.weight", (hidden, inter),
                                     extent=(slice(None), slice(*inter_rows)))
            fc2_bias = take_tensor(f"{prefix}.mlp.dense_4h_to_h.bias", (hidden,))
            layer_states.append(VisionLayerWeights(
                input_norm_weight=ln_w1, input_norm_bias=ln_b1,
                post_norm_weight=ln_w2, post_norm_bias=ln_b2,
                q_weight=attn["q_proj"][0], q_bias=attn["q_proj"][1],
                k_weight=attn["k_proj"][0], k_bias=attn["k_proj"][1],
                v_weight=attn["v_proj"][0], v_bias=attn["v_proj"][1],
                o_weight=o_weight, o_bias=o_bias,
                fc1_weight=fc1_weight, fc1_bias=fc1_bias,
                fc2_weight=fc2_weight, fc2_bias=fc2_bias))

        perceive_norm = take_tensor("vit.perceive.norm.weight", (hidden,))
        proj0_shape = (inter, hidden, merge, merge)
        proj0_weight = take_tensor("vit.perceive.proj.0.weight", proj0_shape)
        # Column-shard the conv's output channels on dim 0.
        proj0_weight = proj0_weight[inter_rows[0]:inter_rows[1]].contiguous()
        proj0_bias = take_tensor("vit.perceive.proj.0.bias", (inter,),
                                 extent=slice(*inter_rows))
        proj2_weight = take_tensor("vit.perceive.proj.2.weight", (out_hidden, inter, 1, 1))
        # Row-shard the conv's input channels on dim 1 (post all-reduce bias).
        proj2_weight = proj2_weight[:, inter_rows[0]:inter_rows[1]].contiguous()
        proj2_bias = take_tensor("vit.perceive.proj.2.bias", (out_hidden,))
        image_newline = take_tensor("vit.perceive.image_newline", (out_hidden,))
        return cls(config=config, world=world, rank=rank,
                   patch_weight=patch_weight, patch_bias=patch_bias,
                   position_embedding=position_embedding, layers=tuple(layer_states),
                   perceive_norm=perceive_norm,
                   perceive_proj0_weight=proj0_weight, perceive_proj0_bias=proj0_bias,
                   perceive_proj2_weight=proj2_weight, perceive_proj2_bias=proj2_bias,
                   image_newline=image_newline,
                   used_names=frozenset(builder.used))


# --------------------------------------------------------------------------
# The forward tower (design sections 1.2 / 1.3)
# --------------------------------------------------------------------------

def _vision_rms_norm(x, weight, eps):
    """`HYV4VLRMSNorm` numerics exactly: FP32 variance, `x * rsqrt` rounded
    ONCE to `x.dtype`, then the BF16 weight multiply (reference
    `modeling_hy_v4_vl.py` rounds the scaled activation before the multiply)."""
    input_dtype = x.dtype
    xf = x.to(torch.float32)
    variance = xf.pow(2).mean(-1, keepdim=True)
    scaled = (xf * torch.rsqrt(variance + eps)).to(input_dtype)
    return weight * scaled


class Hy4VisionTower:
    """One rank's TP-sharded BF16 ViT + `vit.perceive` forward.

    `forward(pixel_bchw, grid_thw, *, collectives, prefill_widths)` encodes ONE
    image (`pixel_bchw [1, 3, H_r, W_r]`, `grid_thw (1, grid_h, grid_w)`) into
    `image_embeds [H_m * (W_m + 1), out_hidden]` BF16. Multi-image prompts loop
    externally and concatenate (design section 1.2: bounded activation peak).
    All layer reductions pack zero-copy into `[N // 4, 6144]` BF16 rows through
    the existing prepared-plan all-reduce table — no new transport, no NCCL.
    """

    def __init__(self, weights: Hy4VisionWeights):
        self.weights = weights
        self.config = weights.config

    # -- packed PCIe all-reduce (design section 1.3) ----------------------
    @staticmethod
    def _reduce_rows(rows_tensor, collectives, prefill_widths):
        """All-reduce `[rows, width]` BF16 partial rows through the prepared table.

        Row counts decompose into `<= 2` full `PACK_CHUNK`-row calls plus at
        most one tail slice zero-padded to the next prepared width
        (`PREFILL_WIDTHS`), sliced back to `rows` afterwards. At production
        geometry `width` is exactly the transport's `reduce_width = 6144`
        (both the packed ViT layer reductions and the `perceive.proj.2`
        reduction land there — design section 1.3).
        """
        if rows_tensor.ndim != 2:
            raise ValueError(f"packed all-reduce expects [rows, width], got {tuple(rows_tensor.shape)}")
        rows = int(rows_tensor.shape[0])
        widths = tuple(sorted(set(int(w) for w in prefill_widths) | set(PREFILL_WIDTHS)))
        out = []
        at = 0
        while at < rows:
            take = min(PACK_CHUNK, rows - at)
            chunk = rows_tensor[at:at + take]
            if take == PACK_CHUNK:
                reduced = collectives.all_reduce(chunk.contiguous())
            else:
                pad_to = next((w for w in widths if w >= take), take)
                padded = torch.zeros((pad_to, int(rows_tensor.shape[1])),
                                     dtype=chunk.dtype, device=chunk.device)
                padded[:take] = chunk
                reduced = collectives.all_reduce(padded.contiguous())[:take]
            out.append(reduced)
            at += take
        return torch.cat(out, dim=0) if len(out) > 1 else out[0]

    def _packed_all_reduce(self, partial, collectives, prefill_widths):
        """Sum `[N, hidden]` partials across the group.

        At production geometry (`hidden == 1536`) the rows pack zero-copy into
        `[N // 4, 6144]` (`6144 / 1536 == 4` exactly; every `smart_resize`
        patch count is a multiple of 4) so the reduction lands on the
        transport's prepared `6144`-wide table; any other width reduces its
        own rows directly (smoke geometries).
        """
        if partial.ndim != 2:
            raise ValueError(f"packed all-reduce expects [N, hidden], got {tuple(partial.shape)}")
        rows = int(partial.shape[0])
        width = int(partial.shape[1])
        if width * PACK_FACTOR == REDUCE_WIDTH:
            if rows % PACK_FACTOR:
                raise ValueError(f"patch count {rows} must be a multiple of {PACK_FACTOR}")
            packed = partial.view(rows // PACK_FACTOR, REDUCE_WIDTH)
            return self._reduce_rows(packed, collectives, prefill_widths).view(rows, width)
        return self._reduce_rows(partial.contiguous(), collectives, prefill_widths)

    # -- position interpolation -------------------------------------------
    def _interpolated_pos_embed(self, grid_h, grid_w, *, device, dtype):
        """FP32 bilinear `position_embedding` interpolation with the reference's
        explicit `scale_factor = ((grid + 0.1) / edge)` shift (even at
        `2048 x 2048` the scale is `128.1 / 128 != 1.0`)."""
        w = self.weights
        edge = w.position_edge
        patch_pos = w.position_embedding[1:].reshape(1, edge, edge, w.hidden)
        patch_pos = patch_pos.permute(0, 3, 1, 2).float().to(device)
        scale_h = (int(grid_h) + 0.1) / edge
        scale_w = (int(grid_w) + 0.1) / edge
        patch_pos = F.interpolate(patch_pos, scale_factor=(scale_h, scale_w),
                                  mode="bilinear", align_corners=False)
        return patch_pos.permute(0, 2, 3, 1).reshape(1, -1, w.hidden).to(dtype)

    # -- attention ---------------------------------------------------------
    @staticmethod
    def _sdpa(q, k, v):
        """Full bidirectional per-image attention WITHOUT materializing scores."""
        if q.is_cuda and not (torch.backends.cuda.flash_sdp_enabled()
                              or torch.backends.cuda.mem_efficient_sdp_enabled()
                              or torch.backends.cuda.cudnn_sdp_enabled()):
            # Risk R2: math SDPA would allocate the [16, N, N] score matrix
            # (8.0 GiB at N = 16,384) — refuse instead of OOM.
            raise RuntimeError(
                "Hy4 ViT needs a non-math SDPA backend (flash/mem-efficient/cuDNN); "
                "math fallback would materialize the attention scores")
        return F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=False)

    # -- the forward -------------------------------------------------------
    def forward(self, pixel_bchw, grid_thw, *, collectives, prefill_widths=PREFILL_WIDTHS):
        w = self.weights
        if pixel_bchw.ndim != 4 or int(pixel_bchw.shape[0]) != 1:
            raise ValueError("Hy4VisionTower.forward encodes one image [1, 3, H_r, W_r]")
        grid = tuple(int(v) for v in grid_thw)
        if len(grid) != 3 or grid[0] != 1:
            raise ValueError(f"expected a single-frame image grid (1, h, w), got {grid}")
        _, _, height, width = pixel_bchw.shape
        grid_h, grid_w = grid[1], grid[2]
        if height != grid_h * w.patch or width != grid_w * w.patch:
            raise ValueError("grid_thw disagrees with the pixel tensor's spatial size")
        patches = grid_h * grid_w
        device = pixel_bchw.device
        dtype = pixel_bchw.dtype

        # Patch embed (row-major patches == Conv2d over the image grid).
        flat = HYV4ImageProcessor.flat_patches(pixel_bchw, patch=w.patch)
        x = F.linear(flat.to(w.patch_weight.dtype), w.patch_weight.flatten(1), w.patch_bias)
        x = x + self._interpolated_pos_embed(grid_h, grid_w, device=device, dtype=x.dtype)
        x = x.unsqueeze(0)      # [1, N, hidden] — reference hidden-state layout

        head_dim = w.head_dim
        local_heads = int(w.layers[0].q_weight.shape[0]) // head_dim
        for layer in w.layers:
            residual = x
            h = F.layer_norm(x, (w.hidden,), layer.input_norm_weight, layer.input_norm_bias,
                             eps=w.layer_norm_eps)
            flat_h = h.reshape(-1, w.hidden)
            q = F.linear(flat_h, layer.q_weight, layer.q_bias).view(-1, local_heads, head_dim)
            k = F.linear(flat_h, layer.k_weight, layer.k_bias).view(-1, local_heads, head_dim)
            v = F.linear(flat_h, layer.v_weight, layer.v_bias).view(-1, local_heads, head_dim)
            q, k, v = (t.transpose(0, 1).unsqueeze(0) for t in (q, k, v))
            attn = self._sdpa(q, k, v).squeeze(0).transpose(0, 1).reshape(-1, q.shape[1] * head_dim)
            partial = F.linear(attn, layer.o_weight)          # bias AFTER the all-reduce
            reduced = self._packed_all_reduce(partial, collectives, prefill_widths)
            x = residual + (reduced + layer.o_bias).unsqueeze(0)

            residual = x
            h = F.layer_norm(x, (w.hidden,), layer.post_norm_weight, layer.post_norm_bias,
                             eps=w.layer_norm_eps)
            flat_h = h.reshape(-1, w.hidden)
            hidden = F.gelu(F.linear(flat_h, layer.fc1_weight, layer.fc1_bias), approximate="none")
            partial = F.linear(hidden, layer.fc2_weight)      # bias AFTER the all-reduce
            reduced = self._packed_all_reduce(partial, collectives, prefill_widths)
            x = residual + (reduced + layer.fc2_bias).unsqueeze(0)

        return self._perceive(x.reshape(patches, w.hidden), (grid_h, grid_w),
                              collectives=collectives, prefill_widths=prefill_widths)

    def _perceive(self, hidden, size, *, collectives, prefill_widths):
        """`vit.perceive`: RMSNorm -> 2x2 merge conv -> exact GELU -> 1x1 conv ->
        per-row `image_newline` column (`H_m * (W_m + 1)` output rows)."""
        w = self.weights
        grid_h, grid_w = (int(v) for v in size)
        h_m, w_m = grid_h // w.merge, grid_w // w.merge
        hidden = _vision_rms_norm(hidden, w.perceive_norm, w.rms_eps)
        # Each frame merges on its own: [1, hidden, grid_h, grid_w].
        frame = hidden.reshape(1, grid_h, grid_w, w.hidden).permute(0, 3, 1, 2)
        merged = F.conv2d(frame, w.perceive_proj0_weight, w.perceive_proj0_bias,
                          stride=w.merge)
        merged = F.gelu(merged, approximate="none")
        partial = F.conv2d(merged, w.perceive_proj2_weight)   # bias AFTER the all-reduce
        # Row-parallel partials over the local input channels: [N // 4, 6144]
        # rows already (`N // 4 == h_m * w_m`), reduced through the same table.
        _, channels, height, width = partial.shape
        flat = partial.reshape(channels, height * width).transpose(0, 1).contiguous()
        reduced = self._reduce_rows(flat, collectives, prefill_widths)
        merged = reduced.transpose(0, 1).reshape(1, channels, height, width)
        merged = merged + w.perceive_proj2_bias.reshape(1, channels, 1, 1)
        newline = w.image_newline.reshape(1, channels, 1, 1).expand(1, channels, height, 1)
        merged = torch.cat([merged, newline.to(merged.dtype)], dim=-1)
        return merged.reshape(1, channels, -1).permute(0, 2, 1).reshape(-1, channels)

    def encode_images(self, pixel_values, grids, *, collectives, prefill_widths=PREFILL_WIDTHS):
        """Encode a list of images one at a time (bounded activation peak) and
        concatenate the `[sum(N_llm_img), out_hidden]` rows in document order."""
        if len(pixel_values) != len(grids):
            raise ValueError("one grid per image")
        outs = [self.forward(pixels, grid, collectives=collectives,
                             prefill_widths=prefill_widths)
                for pixels, grid in zip(pixel_values, grids)]
        return torch.cat(outs, dim=0) if len(outs) > 1 else outs[0]
