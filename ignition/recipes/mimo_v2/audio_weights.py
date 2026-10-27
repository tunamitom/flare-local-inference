"""BF16 audio tokenizer weights, FP32 residual codebooks, and the local transformer.

The tokenizer's own checkpoint lives beside the language model in `audio_tokenizer/`; the
speech embedding tables, the six-layer local transformer, and the 4096-to-6144 projection are
part of the language model's shards and are read through the shared indexed checkpoint loader.
"""

from dataclasses import dataclass

import torch
from safetensors import safe_open

from ignition.components.weights import DirectWeightLoader

# The tokenizer geometry is fixed by the prepared frontend profile; anything else is a different
# audio tower and must be ported explicitly rather than half-matched.
TOKENIZER = dict(
    max_audio_seconds=300,
    stride_size=2,
    avg_pooler=2,
    d_model=1024,
    scale_embedding=False,
    kernel_size=3,
    activation_function="gelu",
    encoder_layers=24,
    encoder_skip_layer_id=3,
    encoder_attention_heads=16,
    encoder_ffn_dim=4096,
    encoder_causal=True,
    encoder_attn_window_size=[128, 0],
    nfft=960,
    n_mels=128,
    sampling_rate=24000,
    hop_length=240,
    window_size=960,
    fmin=0,
    fmax=None,
    num_quantizers=20,
    codebook_size=[1024, 1024, *[256], *[128] * 17],
    position_embedding_type="rope",
    rope_theta=10000,
    rope_type="default",
    ln_type="LayerNorm",
    hybrid_attention=True,
    swa_per_block=2,
)

LOCAL = dict(
    audio_channels=20,
    audio_segment_size=6000,
    group_size=4,
    input_full_attention=True,
    input_local_attn_heads=16,
    input_local_dim=1024,
    input_local_head_dim=64,
    input_local_intermediate_size=4096,
    input_local_layers=6,
    out_hidden_size=6144,
    partial_rotary_factor=1.0,
    projection_layers=2,
    rope_theta=640000.0,
    speech_vocab_size="1280",
    speech_zeroemb_idx="1024",
    add_post_norm=True,
)

SPEECH_VOCAB = 1280
PADDING_INDEX = 1024
LAYER_NORM_EPS = 1e-5
RMS_NORM_EPS = 1e-6


@dataclass(frozen=True)
class TokenizerLayerWeights:
    attention_norm: tuple[torch.Tensor, torch.Tensor]
    qkv: torch.Tensor
    qkv_bias: torch.Tensor
    output: torch.Tensor
    output_bias: torch.Tensor
    final_norm: tuple[torch.Tensor, torch.Tensor]
    fc1: torch.Tensor
    fc1_bias: torch.Tensor
    fc2: torch.Tensor
    fc2_bias: torch.Tensor
    # Even layers run a 128-frame causal window; odd layers attend the whole prefix.
    window: tuple[int, int] | None


@dataclass(frozen=True)
class LocalLayerWeights:
    input_norm: torch.Tensor
    post_attention_norm: torch.Tensor
    qkv: torch.Tensor
    qkv_bias: torch.Tensor
    output: torch.Tensor
    gate_up: torch.Tensor
    down: torch.Tensor


@dataclass(frozen=True)
class AudioWeights:
    conv1: torch.Tensor
    conv1_bias: torch.Tensor
    conv2: torch.Tensor
    conv2_bias: torch.Tensor
    layers: tuple[TokenizerLayerWeights, ...]
    norm: tuple[torch.Tensor, torch.Tensor]
    downsample: torch.Tensor
    downsample_norm: tuple[torch.Tensor, torch.Tensor]
    codebooks: tuple[torch.Tensor, ...]
    speech: tuple[torch.Tensor, ...]
    local: tuple[LocalLayerWeights, ...]
    local_norm: torch.Tensor
    projection: tuple[torch.Tensor, torch.Tensor]
    # Positions 0..output_length-1 of each segment; the tokenizer's RoPE table is sized for
    # max_audio_seconds of audio before pooling.
    frequencies: torch.Tensor

    @classmethod
    def load(cls, manifest, device, *, progress=None, reader=None):
        """Read the tokenizer sub-checkpoint and the language model's audio rows.

        `reader(name)` returns one language-model tensor on the device; it defaults to the shared
        direct loader and exists so a parity harness can read shards without the pinned-staging
        io_uring path.
        """
        device = torch.device(device)
        tokenizer_path = manifest.metadata_path / "audio_tokenizer" / "model.safetensors"
        if not tokenizer_path.is_file():
            raise ValueError("MiMo audio preparation requires audio_tokenizer/model.safetensors")
        if reader is not None:
            return cls._assemble(reader, tokenizer_path, device)
        with DirectWeightLoader(manifest, device, prefixes=("audio_encoder.", "speech_embeddings."),
                                progress=progress, progress_label="Load audio weights") as loader:
            return cls._assemble(loader.load_tensor, tokenizer_path, device)

    @classmethod
    def _assemble(cls, reader, tokenizer_path, device):
        def local_tensor(name):
            value = reader(name)
            if value.dtype != torch.bfloat16:
                raise ValueError(f"MiMo audio requires BF16 weights: {name}")
            return value

        speech = tuple(local_tensor(f"speech_embeddings.{index}.weight") for index in range(LOCAL["audio_channels"]))
        if any(value.shape != (SPEECH_VOCAB, LOCAL["input_local_dim"]) for value in speech):
            raise ValueError("unsupported MiMo speech embedding geometry")
        local = []
        for index in range(LOCAL["input_local_layers"]):
            prefix = f"audio_encoder.input_local_transformer.layers.{index}"
            qkv = torch.cat([local_tensor(f"{prefix}.self_attn.{role}_proj.weight") for role in "qkv"])
            bias = torch.cat([local_tensor(f"{prefix}.self_attn.{role}_proj.bias") for role in "qkv"])
            local.append(LocalLayerWeights(
                local_tensor(f"{prefix}.input_layernorm.weight"),
                local_tensor(f"{prefix}.post_attention_layernorm.weight"),
                qkv, bias, local_tensor(f"{prefix}.self_attn.o_proj.weight"),
                torch.cat([local_tensor(f"{prefix}.mlp.{kind}_proj.weight") for kind in ("gate", "up")]),
                local_tensor(f"{prefix}.mlp.down_proj.weight")))
        local_norm = local_tensor("audio_encoder.input_local_transformer.norm.weight")
        projection = (local_tensor("audio_encoder.projection.mlp.0.weight"),
                      local_tensor("audio_encoder.projection.mlp.2.weight"))
        tokenizer = safe_open(str(tokenizer_path), framework="pt", device=str(device))

        def token_tensor(name, dtype=torch.bfloat16):
            value = tokenizer.get_tensor("encoder." + name)
            if value.dtype != dtype:
                raise ValueError(f"MiMo audio tokenizer requires {dtype} storage: {name}")
            return value

        conv1, conv2 = token_tensor("conv1.weight"), token_tensor("conv2.weight")
        d_model, mels, kernel = TOKENIZER["d_model"], TOKENIZER["n_mels"], TOKENIZER["kernel_size"]
        if conv1.shape != (d_model, mels, kernel) or conv2.shape != (d_model, d_model, kernel):
            raise ValueError("unsupported MiMo audio tokenizer convolution geometry")
        layers = []
        window = tuple(TOKENIZER["encoder_attn_window_size"])
        for index in range(TOKENIZER["encoder_layers"]):
            prefix = f"layers.{index}"
            # k_proj is bias-free in the reference module; its fused rows carry a zero bias.
            bias = torch.cat([token_tensor(f"{prefix}.self_attn.q_proj.bias"),
                              torch.zeros(d_model, dtype=torch.bfloat16, device=device),
                              token_tensor(f"{prefix}.self_attn.v_proj.bias")])
            local_window = (window if TOKENIZER["hybrid_attention"]
                            and index % TOKENIZER["swa_per_block"] < TOKENIZER["swa_per_block"] - 1 else None)
            layers.append(TokenizerLayerWeights(
                (token_tensor(f"{prefix}.self_attn_layer_norm.weight"),
                 token_tensor(f"{prefix}.self_attn_layer_norm.bias")),
                torch.cat([token_tensor(f"{prefix}.self_attn.{role}_proj.weight") for role in "qkv"]), bias,
                token_tensor(f"{prefix}.self_attn.out_proj.weight"),
                token_tensor(f"{prefix}.self_attn.out_proj.bias"),
                (token_tensor(f"{prefix}.final_layer_norm.weight"),
                 token_tensor(f"{prefix}.final_layer_norm.bias")),
                token_tensor(f"{prefix}.fc1.weight"), token_tensor(f"{prefix}.fc1.bias"),
                token_tensor(f"{prefix}.fc2.weight"), token_tensor(f"{prefix}.fc2.bias"), local_window))
        downsample = token_tensor("down_sample_layer.0.weight")
        if downsample.shape != (d_model, d_model, TOKENIZER["avg_pooler"]):
            raise ValueError("unsupported MiMo audio tokenizer pooling geometry")
        codebooks = []
        for index, size in enumerate(TOKENIZER["codebook_size"]):
            value = tokenizer.get_tensor(f"encoder.quantizer.vq.layers.{index}._codebook.embed")
            if value.shape != (size, d_model):
                raise ValueError(f"unsupported MiMo codebook geometry at layer {index}")
            # The reference upcasts the quantizer and its input to FP32 for nearest-neighbour search.
            codebooks.append(value.float())
        head_dim = d_model // TOKENIZER["encoder_attention_heads"]
        # The reference registers `inv_freq` as a buffer, so loading the tower in BF16 rounds the
        # rotary table; the FP32 rotation then reads those rounded frequencies.
        frequencies = (1.0 / (TOKENIZER["rope_theta"] ** (
            torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))).bfloat16().float()
        return cls(conv1, token_tensor("conv1.bias"), conv2, token_tensor("conv2.bias"), tuple(layers),
                   (token_tensor("layer_norm.weight"), token_tensor("layer_norm.bias")), downsample,
                   (token_tensor("down_sample_norm.weight"), token_tensor("down_sample_norm.bias")),
                   tuple(codebooks), speech, tuple(local), local_norm, projection, frequencies)
