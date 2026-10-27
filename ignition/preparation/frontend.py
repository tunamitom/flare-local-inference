"""Extract and validate the structural frontend fields consumed by Helios."""

import json
import hashlib
from enum import IntEnum
from pathlib import Path


class PromptEncoder(IntEnum):
    PROMPT_JINJA = 0
    DEEPSEEK_V4 = 1
    DEEPSEEK_V41 = 2


TOKEN_FIELDS = ("bos_token", "eos_token", "unk_token", "sep_token", "pad_token", "cls_token", "mask_token")


def annotated_template_asset(source: bytes, annotated: Path):
    """Validate output-preserving message annotations against a checkpoint template."""
    annotated = annotated.resolve()
    contents = annotated.read_bytes()
    if contents.replace(b"{%- mark -%}", b"") != source:
        raise ValueError(f"checkpoint chat template does not match {annotated}")
    return {"path": str(annotated), "sha256": hashlib.sha256(contents).hexdigest()}


def _token(value):
    if isinstance(value, dict):
        value = value.get("content")
    if not isinstance(value, str):
        raise ValueError("special token must be a string or an object with string content")
    return value


def tokenizer_settings(config):
    if not isinstance(config, dict):
        raise ValueError("tokenizer configuration must be an object")
    if config.get("clean_up_tokenization_spaces") is True:
        raise ValueError("frontend requires tokenizer clean_up_tokenization_spaces=false")
    result = {name: _token(config[name]) for name in TOKEN_FIELDS if config.get(name) is not None}
    additional = config.get("additional_special_tokens")
    if additional is not None:
        if not isinstance(additional, list):
            raise ValueError("additional_special_tokens must be a list")
        result["additional_special_tokens"] = {"values": [_token(value) for value in additional]}
    return result


def _expected(config, values, label):
    for key, expected in values.items():
        value = config.get(key)
        if value != expected or (type(expected) in (bool, int) and type(value) is not type(expected)):
            raise ValueError(f"unsupported {label} {key}")


def image_settings(config, model, profile):
    if profile == "glm":
        _expected(config, {"processor_class": "Glm5NextProcessor"}, "GLM processor")
        image = config["image_processor"]
        _expected(image, {"image_processor_type": "Glm5NextImageProcessor", "patch_size": 14,
            "temporal_patch_size": 2, "merge_size": 2, "patch_expand_factor": 1,
            "min_image_tokens": 16, "max_image_tokens": 8000,
            "image_mean": [0.48145466, 0.4578275, 0.40821073],
            "image_std": [0.26862954, 0.26130258, 0.27577711]}, "GLM image processor")
        optional = {"resample": 3, "rescale_factor": 1 / 255,
            "do_resize": True, "do_rescale": True, "do_normalize": True, "do_convert_rgb": True}
        _expected(image, {key: value for key, value in optional.items() if key in image}, "GLM image processor")
        _expected(model, {"model_type": "glm5_next", "image_token_id": 154854, "video_token_id": 154855},
                  "GLM model configuration")
        _expected(model["vision_config"], {"patch_size": 14, "temporal_patch_size": 2,
            "spatial_merge_size": 2, "in_channels": 3}, "GLM vision configuration")
        return {"glm": {}}
    if profile == "radio":
        _expected(config, {"image_processor_type": "NemotronH_Omni_Reasoning_V3ImageProcessor",
            "patch_size": 16, "downsample_ratio": .5, "max_model_len": 16384,
            "min_num_patches": 1024, "max_num_patches": 13312,
            "norm_mean": [0.48145466, 0.4578275, 0.40821073],
            "norm_std": [0.26862954, 0.26130258, 0.27577711]}, "RADIO image processor")
        _expected(model, {"patch_size": 16, "downsample_ratio": .5, "ps_version": "v2",
            "img_context_token_id": 18}, "RADIO image configuration")
        _expected(model["vision_config"], {"max_img_size": 2048, "patch_size": 16}, "RADIO vision configuration")
        return {"radio": {}}
    if profile == "moonvit":
        _expected(config["media_proc_cfg"], {"in_patch_limit": 65536, "patch_size": 14,
            "image_mean": [0.5] * 3, "image_std": [0.5] * 3, "merge_kernel_size": 2,
            "fixed_output_tokens": None, "patch_limit_on_one_side": 512,
            "transparent_bg_fill_stage": "after_resize", "transparent_bg_config": {
                "pattern": "chessboard", "chessboard_square_size": 8,
                "chessboard_square_on_top_left": True, "chessboard_white_value": 255,
                "chessboard_gray_value": 180}}, "MoonViT image processor")
        _expected(model["vision_config"], {"patch_size": 14, "init_pos_emb_height": 64,
            "init_pos_emb_width": 64, "pos_emb_interpolation_mode": "bilinear",
            "merge_kernel_size": [2, 2]}, "MoonViT vision configuration")
        return {"moonvit": {}}
    if profile == "deepseek_row_major":
        vision = model["vision_config"]
        _expected(vision, {"patch_size": 14, "downsample_ratio": 3, "max_image_tokens": 1024,
            "min_pixels": 295936, "max_wh_ratio": None}, "DeepSeek row-major vision configuration")
        return {profile: dict(min_pixels=vision["min_pixels"], max_tokens=vision["max_image_tokens"])}
    if profile == "deepseek4":
        _expected(config, {"image_processor_type": "DeepSeekV4ImageProcessor", "patch_size": 14,
            "merge_size": 3, "max_image_tokens": 384, "min_pixels": 147456,
            "max_width_height_ratio": 8}, "DeepSeek image processor")
        _expected(model, {"vision_patch_size": 14, "vision_downsample_ratio": 3,
            "vision_max_n_token": 384, "vision_min_pixels": 147456,
            "vision_max_wh_ratio": 8}, "DeepSeek vision configuration")
        return {"deepseek4": {}}
    if profile not in ("qwen", "mimo"):
        raise ValueError("unsupported image preprocessing profile")
    qwen = profile == "qwen"
    mean = [0.5] * 3 if qwen else [0.485, 0.456, 0.406]
    std = [0.5] * 3 if qwen else [0.229, 0.224, 0.225]
    _expected(config, {"processor_class": "Qwen3VLProcessor" if qwen else "Qwen2_5_VLProcessor",
        "image_processor_type": "Qwen2VLImageProcessorFast" if qwen else "Qwen2VLImageProcessor",
        "patch_size": 16, "temporal_patch_size": 2, "merge_size": 2,
        "image_mean": mean, "image_std": std}, "image processor")
    optional = {"resample": 3, "rescale_factor": 1 / 255,
        "do_resize": True, "do_rescale": True, "do_normalize": True, "do_convert_rgb": True}
    _expected(config, {key: value for key, value in optional.items() if key in config}, "image processor")
    _expected(model, {"model_type": "qwen4_exp" if qwen else "mimo_v2"}, "model configuration")
    vision = model["vision_config"]
    _expected(vision, {"patch_size": 16, "temporal_patch_size": 2, "spatial_merge_size": 2}, "vision configuration")
    if qwen:
        _expected(vision, {"in_channels": 3, "num_position_embeddings": 2304}, "vision configuration")
        _expected(config["size"], {"shortest_edge": 65536, "longest_edge": 16777216}, "resize area")
        result = dict(min_pixels=65536, max_pixels=16777216, position_side=48)
        for name in ("image_token_id", "video_token_id"):
            value = model[name]
            if type(value) is not int or not 0 <= value < 2**32:
                raise ValueError(f"invalid model {name}")
            result[name] = value
    else:
        _expected(config, {"min_pixels": 8192, "max_pixels": 8388608}, "resize area")
        _expected(model, {"image_token_id": 151655}, "model configuration")
        result = dict(min_pixels=8192, max_pixels=8388608, image_mean=mean, image_std=std)
    return {profile: result}


def audio_settings(config, model, profile):
    """Log-mel preprocessing and the tokenizer's frame-to-token arithmetic."""
    if profile != "mimo":
        raise ValueError("unsupported audio preprocessing profile")
    _expected(config, {"sampling_rate": 24000, "hop_length": 240, "window_size": 960, "nfft": 960,
        "n_mels": 128, "fmin": 0, "fmax": None, "kernel_size": 3, "stride_size": 2, "avg_pooler": 2,
        "encoder_layers": 24, "encoder_attention_heads": 16, "encoder_ffn_dim": 4096,
        "encoder_causal": True, "encoder_attn_window_size": [128, 0], "encoder_skip_layer_id": 3,
        "hybrid_attention": True, "swa_per_block": 2, "num_quantizers": 20, "ln_type": "LayerNorm",
        "activation_function": "gelu", "rope_theta": 10000, "rope_type": "default",
        "codebook_size": [1024, 1024, *[256], *[128] * 17]}, "audio tokenizer configuration")
    processor = model["processor_config"]
    _expected(processor, {"audio_sampling_rate": 24000, "audio_nfft": 960, "audio_hop_length": 240,
        "audio_window_size": 960, "audio_n_mels": 128, "audio_kernel_size": 3, "audio_stride_size": 2,
        "audio_avg_pooler": 2, "audio_group_size": 4, "audio_segment_size": 6000}, "audio processor configuration")
    _expected(model["audio_config"], {"audio_channels": 20, "audio_segment_size": 6000, "group_size": 4,
        "input_full_attention": True, "input_local_layers": 6, "input_local_dim": 1024,
        "input_local_attn_heads": 16, "input_local_head_dim": 64, "input_local_intermediate_size": 4096,
        "out_hidden_size": 6144, "projection_layers": 2, "rope_theta": 640000,
        "partial_rotary_factor": 1.0, "add_post_norm": True, "speech_vocab_size": "1280",
        "speech_zeroemb_idx": "1024"}, "audio tower configuration")
    return {"mimo": dict(sampling_rate=24000, n_fft=960, hop_length=240, window_length=960, n_mels=128,
        f_min=0.0, f_max=None, segment_frames=config.get("audio_segment_size", 6000),
        kernel_size=3, stride=2, pooling=2, group_size=4, max_seconds=config["max_audio_seconds"])}


def video_settings(image, profile):
    """Frame sampling and per-frame resize areas for video, which reuses the image resize range.

    The clip budget, frame count and frame ceiling are the reference processor's own constants:
    thirty-two uniform frames per clip, each frame capped so the whole clip stays inside
    total_max_pixels, and the same 0..255 RGB standardization the image path applies.
    """
    if profile != "mimo":
        raise ValueError("unsupported video preprocessing profile")
    # The video frontend divides u8 pixels directly by pixel_std, so it takes the CLIP constants
    # in 0..255 scale; the image path's 0..1 mean/std are the same numbers divided by 255.
    return {"mimo": dict(min_pixels=image["min_pixels"], max_pixels=image["max_pixels"],
        total_max_pixels=268435456, num_frames=32, max_frames=3600,
        pixel_mean=[value * 255.0 for value in image["image_mean"]],
        pixel_std=[value * 255.0 for value in image["image_std"]])}


def _native_image_placeholder(snapshot, model, encoder):
    from tokenizers import Tokenizer

    token = "<｜deepseek_image｜>"
    tokenizer = Tokenizer.from_file(str(snapshot / "tokenizer.json"))
    token_id = tokenizer.token_to_id(token)
    if (token_id != 129264 or token_id >= tokenizer.get_vocab_size()
            or tokenizer.encode(token, add_special_tokens=False).ids != [token_id]):
        raise ValueError("DeepSeek image placeholder must be atomic vocabulary token 129264")
    vocabulary_size = model.get("vocab_size")
    if vocabulary_size is not None and (type(vocabulary_size) is not int or token_id >= vocabulary_size):
        raise ValueError("DeepSeek image placeholder is outside the model vocabulary")
    if encoder == PromptEncoder.DEEPSEEK_V41:
        _expected(model, {"image_token_id": token_id}, "DeepSeek image configuration")
    return token_id


def prepare_frontend(snapshot, *, eos_token_ids, image_profile=None, preprocessor=None,
                     prompt_encoder=PromptEncoder.PROMPT_JINJA, audio_profile=None, video_profile=None):
    """Resolve checkpoint structure during preparation; sampling is recipe-owned."""
    if not isinstance(prompt_encoder, PromptEncoder):
        raise ValueError("prompt_encoder must be a PromptEncoder")
    snapshot = Path(snapshot)
    if type(eos_token_ids) is int:
        eos_token_ids = [eos_token_ids]
    eos_token_ids = list(eos_token_ids)
    if (not eos_token_ids or len(eos_token_ids) > 8 or len(set(eos_token_ids)) != len(eos_token_ids)
            or any(type(value) is not int or not 0 <= value < 2**32 for value in eos_token_ids)):
        raise ValueError("EOS token IDs must be one through eight distinct uint32 values")
    result = {"tokenizer": tokenizer_settings(json.loads((snapshot / "tokenizer_config.json").read_text())),
              "eos_token_ids": eos_token_ids}
    if prompt_encoder != PromptEncoder.PROMPT_JINJA:
        result["prompt_encoder"] = int(prompt_encoder)
    if image_profile is not None:
        expected_profile = {PromptEncoder.DEEPSEEK_V4: "deepseek4",
                            PromptEncoder.DEEPSEEK_V41: "deepseek_row_major"}.get(prompt_encoder)
        if expected_profile is not None:
            if image_profile != expected_profile:
                raise ValueError("image profile differs from the prompt encoder")
            result["image_placeholder_token_id"] = _native_image_placeholder(
                snapshot, json.loads((snapshot / "config.json").read_text()), prompt_encoder)
        path = snapshot / "preprocessor_config.json" if preprocessor is None else Path(preprocessor)
        result["image"] = image_settings({} if image_profile == "deepseek_row_major" else json.loads(path.read_text()),
            json.loads((snapshot / "config.json").read_text()), image_profile)
    if audio_profile is not None:
        result["audio"] = audio_settings(json.loads((snapshot / "audio_tokenizer" / "config.json").read_text()),
            json.loads((snapshot / "config.json").read_text()), audio_profile)
    if video_profile is not None:
        if image_profile is None:
            raise ValueError("video settings reuse the image resize range and need an image profile")
        result["video"] = video_settings(result["image"][image_profile], video_profile)
    return result
