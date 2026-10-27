"""MiMo audio tower: causal log-mel tokenizer, residual VQ codes, and the grouped local transformer.

One captured graph encodes one frontend segment of at most `AUDIO_CAPACITIES[-1]` log-mel frames.
Shorter segments stay inside the same capacity: the mel buffer is zero-padded, and because the
tokenizer encoder is causal (a 128-frame window on even layers, the whole prefix on odd ones) every
row in the real prefix is exactly what a shorter input produces. The pooling and grouping
arithmetic is then derived on device from the segment's own frame count, so the number of produced
features matches the processor's placeholder count without a device-to-host read, which is what
makes the graph capturable.

The reference this reproduces is `vllm/model_executor/models/mimo_audio.py` in the kk-mimo fork:
`AudioEncoder` (conv1/conv2, `encoder_skip_layer_id`, hybrid causal attention, 2x average-pool with
GELU and LayerNorm), `ResidualVectorQuantizer`, and `MimoAudioEncoder` (summed speech embeddings, a
six-layer Qwen2 local transformer over each group of four code rows, and the 4096-16384-6144
projection). `max_seqlen` is a host constant of the captured capacity rather than
`torch.max(output_length).item()`.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ignition.components.linear import UnquantizedProjection
from ignition.components.varlen_attention import PreparedVarlenAttention
from ignition.components.vision_delivery import PreparedFeaturePublication
from ignition.preparation.tensor_metadata import record_numeric_tensor

from .audio_weights import LAYER_NORM_EPS, RMS_NORM_EPS, AudioWeights

# Mel frames one graph encodes: the tokenizer's own `audio_segment_size`, 60 s at 100 frames/s.
AUDIO_CAPACITIES = (6000,)
MELS = 128
D_MODEL = 1024
HEADS = 16
HEAD_DIM = D_MODEL // HEADS
FFN = 4096
GROUP = 4
CHANNELS = 20
LOCAL_DIM = 1024
LOCAL_FFN = 4096
LOCAL_HEAD_DIM = 64
LOCAL_THETA = 640000.0
PROJECTION_HIDDEN = 4 * GROUP * LOCAL_DIM
OUT_HIDDEN = 6144
# `encoder_skip_layer_id` is one-based in the checkpoint; the clone happens after that layer.
SKIP_LAYER = 3


def encoded_frames(mel_frames, kernel=3, stride=2):
    """Code frames the two convolutions produce from `mel_frames` log-mel frames."""
    return (mel_frames + 2 - kernel) // stride + 1


def pooled_frames(frames, pooling=2):
    return -(-frames // pooling)


def grouped_tokens(frames, group=GROUP):
    return -(-frames // group)


def processor_tokens(mel_frames, kernel=3, stride=2, pooling=2, group=GROUP):
    """The placeholder count the processor derives, verbatim from `preprocess_audio`."""
    n = mel_frames + 3 - kernel
    n = (n + 2 - kernel) // stride + 1
    n = n // pooling + (n % pooling != 0)
    return grouped_tokens(n, group)


def encoded_tokens(mel_frames, kernel=3, stride=2, pooling=2, group=GROUP):
    """The count the encoder produces, from its convolution, pooling, and grouping arithmetic."""
    return grouped_tokens(pooled_frames(encoded_frames(mel_frames, kernel, stride), pooling), group)


@dataclass(frozen=True)
class AudioInputs:
    """Named outputs of the audio preparation, uploaded for one segment."""

    audio_features: torch.Tensor
    audio_frames: torch.Tensor
    audio_slot: torch.Tensor
    audio_generation: torch.Tensor
    audio_feature_offset: torch.Tensor

    @classmethod
    def allocate(cls, device, frames=AUDIO_CAPACITIES[-1]):
        def zeros(shape, dtype=torch.int32):
            return torch.zeros(shape, dtype=dtype, device=device)
        return cls(zeros((frames, MELS), torch.float32), zeros((1,)), zeros((1,)) - 1,
                   zeros((1,), torch.int64), zeros((1,)))

    def buffers(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class AudioScratch:
    mel_time: torch.Tensor
    residual: torch.Tensor
    normalized: torch.Tensor
    qkv: torch.Tensor
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    attended: torch.Tensor
    projected: torch.Tensor
    mlp_up: torch.Tensor
    mlp_activated: torch.Tensor
    mlp_down: torch.Tensor
    skip: torch.Tensor
    pooled_input: torch.Tensor
    pooled_time: torch.Tensor
    pooled: torch.Tensor
    quantized: torch.Tensor
    quantization_residual: torch.Tensor
    codes: torch.Tensor
    gather: torch.Tensor
    code_index: torch.Tensor
    group: torch.Tensor
    group_normalized: torch.Tensor
    group_qkv: torch.Tensor
    group_query: torch.Tensor
    group_key: torch.Tensor
    group_value: torch.Tensor
    group_attended: torch.Tensor
    group_projected: torch.Tensor
    group_fused: torch.Tensor
    group_activated: torch.Tensor
    group_output: torch.Tensor
    projection_hidden: torch.Tensor
    projection_activated: torch.Tensor
    output: torch.Tensor

    @classmethod
    def allocate(cls, frames, device, *, buffers=None):
        """Buffers for one capacity: `frames` mel rows, half as many code rows, a quarter pooled."""
        def zeros(shape, dtype=torch.bfloat16):
            return (torch.zeros(shape, dtype=dtype, device=device) if buffers is None
                    else buffers.zeros(shape, dtype))
        code = encoded_frames(frames)
        pooled = pooled_frames(code)
        tokens = grouped_tokens(pooled)
        rows = tokens * GROUP
        return cls(
            zeros((MELS, frames)),
            zeros((code, D_MODEL)),
            zeros((code, D_MODEL)),
            zeros((code, 3 * D_MODEL)),
            zeros((code, HEADS, HEAD_DIM)),
            zeros((code, HEADS, HEAD_DIM)),
            zeros((code, HEADS, HEAD_DIM)),
            zeros((code, D_MODEL)),
            zeros((code, D_MODEL)),
            zeros((code, FFN)),
            zeros((code, FFN)),
            zeros((code, D_MODEL)),
            zeros((code, D_MODEL)),
            zeros((code, D_MODEL)),
            zeros((D_MODEL, code)),
            zeros((pooled, D_MODEL)),
            zeros((pooled, LOCAL_DIM), torch.float32),
            zeros((pooled, LOCAL_DIM), torch.float32),
            zeros((CHANNELS, pooled), torch.int64),
            zeros((rows,), torch.int64),
            zeros((rows,), torch.int64),
            zeros((rows, LOCAL_DIM)),
            zeros((rows, LOCAL_DIM)),
            zeros((rows, 3 * LOCAL_DIM)),
            zeros((rows, HEADS, LOCAL_HEAD_DIM)),
            zeros((rows, HEADS, LOCAL_HEAD_DIM)),
            zeros((rows, HEADS, LOCAL_HEAD_DIM)),
            zeros((rows, LOCAL_DIM)),
            zeros((rows, LOCAL_DIM)),
            zeros((rows, 2 * LOCAL_FFN)),
            zeros((rows, LOCAL_FFN)),
            zeros((rows, LOCAL_DIM)),
            zeros((tokens, PROJECTION_HIDDEN)),
            zeros((tokens, PROJECTION_HIDDEN)),
            zeros((tokens, OUT_HIDDEN)),
        )

    def tensors(self):
        return tuple(getattr(self, name) for name in self.__dataclass_fields__)


def rotary(source, destination, angles):
    """NeoX rotation in the reference's own BF16 arithmetic: cos and sin round to BF16 first."""
    left, right = source.chunk(2, -1)
    cos, sin = angles.cos().bfloat16(), angles.sin().bfloat16()
    destination.copy_(torch.cat((left * cos - right * sin, right * cos + left * sin), -1))


def rms_norm(source, weight, output, eps=RMS_NORM_EPS):
    """Qwen2 RMSNorm: FP32 statistics, BF16 normalization, BF16 gain."""
    float32 = source.float()
    variance = float32.pow(2).mean(-1, keepdim=True)
    output.copy_(weight * (float32 * torch.rsqrt(variance + eps)).bfloat16())


def rope_angles(positions, frequencies):
    return (positions[:, None] * frequencies).flatten(1)[:, None]


class PreparedAudio:
    """The tokenizer encoder, residual quantizer, and local transformer as one graph."""

    def __init__(self, weights: AudioWeights, scratch, inputs, *, session, workspace, name, buffers=None):
        self.weights, self.scratch, self.inputs = weights, scratch, inputs
        self.config = weights
        s = scratch
        code, pooled = s.residual.shape[0], s.pooled.shape[0]
        rows, tokens = s.group.shape[0], s.output.shape[0]
        device = s.residual.device

        def zeros(shape, dtype=torch.int32):
            return (torch.zeros(shape, dtype=dtype, device=device) if buffers is None
                    else buffers.zeros(shape, dtype))

        # One segment per graph: cumulative lengths are device data, the maxima are host constants.
        self.cu = zeros((2,))
        self.count = zeros((1,))
        self.rows = torch.arange(0, code, device=device)
        self.mel_rows = torch.arange(0, s.mel_time.shape[1], device=device)
        self.group_rows = torch.arange(0, rows, device=device)
        self.group_cu = torch.arange(0, rows + GROUP, GROUP, dtype=torch.int32, device=device)
        self.code_angles = rope_angles(torch.arange(0, code, device=device).float(), weights.frequencies)
        local_frequencies = (1.0 / (LOCAL_THETA ** (
            torch.arange(0, LOCAL_HEAD_DIM, 2, dtype=torch.float32, device=device)
            / LOCAL_HEAD_DIM))).bfloat16().float()
        self.group_angles = rope_angles((self.group_rows % GROUP).float(), local_frequencies)
        self.codebook_norms = tuple(embed.pow(2).sum(-1) for embed in weights.codebooks)

        for layer in weights.layers:
            for tensor in (*layer.attention_norm, layer.qkv, layer.qkv_bias, layer.output, layer.output_bias,
                           *layer.final_norm, layer.fc1, layer.fc1_bias, layer.fc2, layer.fc2_bias):
                record_numeric_tensor(tensor)
        for layer in weights.local:
            for tensor in (layer.input_norm, layer.post_attention_norm, layer.qkv, layer.qkv_bias,
                           layer.output, layer.gate_up, layer.down):
                record_numeric_tensor(tensor)
        for tensor in (weights.conv1, weights.conv1_bias, weights.conv2, weights.conv2_bias,
                       weights.downsample, weights.local_norm, *weights.norm, *weights.downsample_norm,
                       *weights.codebooks, *weights.speech, *weights.projection):
            record_numeric_tensor(tensor)

        self.blocks = []
        for index, layer in enumerate(weights.layers):
            prefix = f"{name}.{index}"
            self.blocks.append((
                layer,
                UnquantizedProjection(s.normalized, layer.qkv, s.qkv, bias=layer.qkv_bias, session=session,
                                      name=prefix + ".qkv"),
                PreparedVarlenAttention(s.query, s.key, s.value, self.cu, s.attended.view_as(s.query),
                                        window=layer.window,
                                        causal=True, max_seqlen=code, session=session,
                                        workspace=workspace, name=prefix + ".attention"),
                UnquantizedProjection(s.attended, layer.output, s.projected, bias=layer.output_bias,
                                      session=session, name=prefix + ".output"),
                UnquantizedProjection(s.normalized, layer.fc1, s.mlp_up, bias=layer.fc1_bias, session=session,
                                      name=prefix + ".fc1"),
                UnquantizedProjection(s.mlp_activated, layer.fc2, s.mlp_down, bias=layer.fc2_bias,
                                      session=session, name=prefix + ".fc2"),
            ))
        self.local = []
        for index, layer in enumerate(weights.local):
            prefix = f"{name}.local.{index}"
            self.local.append((
                layer,
                UnquantizedProjection(s.group_normalized, layer.qkv, s.group_qkv, bias=layer.qkv_bias,
                                      session=session, name=prefix + ".qkv"),
                PreparedVarlenAttention(s.group_query, s.group_key, s.group_value, self.group_cu,
                                        s.group_attended.view_as(s.group_query), causal=False,
                                        max_seqlen=GROUP, session=session,
                                        workspace=workspace, name=prefix + ".attention"),
                UnquantizedProjection(s.group_attended, layer.output, s.group_projected, session=session,
                                      name=prefix + ".output"),
                UnquantizedProjection(s.group_normalized, layer.gate_up, s.group_fused, session=session,
                                      name=prefix + ".gate_up"),
                UnquantizedProjection(s.group_activated, layer.down, s.group_output, session=session,
                                      name=prefix + ".down"),
            ))
        self.projection_in = UnquantizedProjection(
            s.group_normalized.view(tokens, GROUP * LOCAL_DIM), weights.projection[0], s.projection_hidden,
            session=session, name=name + ".projection.in")
        self.projection_out = UnquantizedProjection(
            s.projection_activated, weights.projection[1], s.output, session=session,
            name=name + ".projection.out")

    def segment_lengths(self, frames):
        """`(code, pooled, tokens)` for a segment of `frames` mel frames, on device."""
        code = torch.div(frames + 2 - 3, 2, rounding_mode="floor") + 1
        pooled = torch.div(code + 1, 2, rounding_mode="floor")
        return code, pooled, torch.div(pooled + GROUP - 1, GROUP, rounding_mode="floor")

    def convolve(self):
        """Fill `scratch.residual` with the code-frame features and set the segment's extent."""
        w, s = self.weights, self.scratch
        frames = self.inputs.audio_frames
        s.mel_time.copy_(self.inputs.audio_features.bfloat16().t())
        convolved = F.gelu(F.conv1d(s.mel_time[None], w.conv1, w.conv1_bias, padding=1))
        # A shorter segment's conv1 ends at `frames`; zero the extension so conv2's last window
        # sees the same implicit zero padding the reference's shorter tensor produces.
        convolved = convolved.masked_fill((self.mel_rows >= frames.reshape(()))[None, None, :], 0.0)
        convolved = F.gelu(F.conv1d(convolved, w.conv2, w.conv2_bias, stride=2, padding=1))
        s.residual.copy_(convolved[0].t())
        self.cu[1].copy_(self.segment_lengths(frames)[0].reshape(()))

    def transform(self):
        """Run the tokenizer's causal encoder layers, skip connection, and final LayerNorm."""
        w, s = self.weights, self.scratch
        for index, (layer, qkv, attention, output, fc1, fc2) in enumerate(self.blocks):
            s.normalized.copy_(F.layer_norm(s.residual, (D_MODEL,), *layer.attention_norm, LAYER_NORM_EPS))
            q, k, v = qkv(s.normalized).split(D_MODEL, -1)
            rotary(q.view_as(s.query), s.query, self.code_angles)
            rotary(k.view_as(s.key), s.key, self.code_angles)
            s.value.copy_(v.view_as(s.value))
            attention()
            s.residual.add_(output(s.attended))
            if index == SKIP_LAYER - 1:
                s.skip.copy_(s.residual)
            s.normalized.copy_(F.layer_norm(s.residual, (D_MODEL,), *layer.final_norm, LAYER_NORM_EPS))
            s.mlp_activated.copy_(F.gelu(fc1(s.normalized)))
            s.residual.add_(fc2(s.mlp_activated))
        # The tokenizer's skip connection carries the output of `encoder_skip_layer_id` forward.
        s.residual.add_(s.skip)
        s.normalized.copy_(F.layer_norm(s.residual, (D_MODEL,), *w.norm, LAYER_NORM_EPS))

    def pool(self):
        """Average-pool the code frames by two, then normalize; rows past the extent are zero."""
        w, s = self.weights, self.scratch
        out_length = self.segment_lengths(self.inputs.audio_frames)[0]
        s.pooled_input.copy_(s.normalized)
        s.pooled_input.masked_fill_((self.rows >= out_length.reshape(()))[:, None], 0.0)
        s.pooled_time.copy_(s.pooled_input.t())
        s.pooled.copy_(F.layer_norm(F.gelu(F.conv1d(s.pooled_time[None], w.downsample, stride=2))[0].t(),
                                    (LOCAL_DIM,), *w.downsample_norm, LAYER_NORM_EPS))

    def quantize(self):
        """Nearest codebook entry per residual layer, exactly as the reference's FP32 RVQ search."""
        w, s = self.weights, self.scratch
        s.quantized.copy_(s.pooled.float())
        s.quantization_residual.copy_(s.quantized)
        squares = s.quantized.pow(2).sum(-1, keepdim=True)
        for channel, (embed, norms) in enumerate(zip(w.codebooks, self.codebook_norms)):
            scores = torch.mm(s.quantization_residual, embed.t())
            scores.mul_(-2).add_(squares).add_(norms)
            torch.argmin(scores, -1, out=s.codes[channel])
            s.quantization_residual.sub_(F.embedding(s.codes[channel], embed))

    def features(self):
        """Summed speech embeddings, the grouped local transformer, and the 4096-16384-6144 projection."""
        w, s = self.weights, self.scratch
        pooled_length = self.segment_lengths(self.inputs.audio_frames)[1]
        # Group the code rows, repeating the last real row to the group boundary, then sum the tables.
        limit = torch.clamp(pooled_length.reshape(()).long() - 1, min=0)
        source = torch.clamp(self.group_rows, max=limit)
        s.group.zero_()
        for channel, table in enumerate(w.speech):
            torch.index_select(s.codes[channel], 0, source, out=s.code_index)
            torch.index_select(table, 0, s.code_index, out=s.group_normalized)
            s.group.add_(s.group_normalized)

        for layer, qkv, attention, output, gate_up, down in self.local:
            rms_norm(s.group, layer.input_norm, s.group_normalized)
            q, k, v = qkv(s.group_normalized).split(LOCAL_DIM, -1)
            rotary(q.view_as(s.group_query), s.group_query, self.group_angles)
            rotary(k.view_as(s.group_key), s.group_key, self.group_angles)
            s.group_value.copy_(v.view_as(s.group_value))
            attention()
            s.group.add_(output(s.group_attended))
            rms_norm(s.group, layer.post_attention_norm, s.group_normalized)
            gate, up = gate_up(s.group_normalized).split(LOCAL_FFN, -1)
            s.group_activated.copy_(F.silu(gate) * up)
            s.group.add_(down(s.group_activated))
        rms_norm(s.group, w.local_norm, s.group_normalized)

        s.projection_activated.copy_(
            F.gelu(self.projection_in(s.group_normalized.view(-1, GROUP * LOCAL_DIM))))
        return self.projection_out(s.projection_activated)

    def __call__(self):
        self.convolve()
        self.transform()
        self.pool()
        self.quantize()
        output = self.features()
        self.count.copy_(self.segment_lengths(self.inputs.audio_frames)[2])
        return output


class _FeatureSlots:
    """The three tensors `PreparedFeaturePublication` reads, named for the arena it publishes into."""

    def __init__(self, slot, generation, offset):
        self.vision_slot, self.vision_generation, self.vision_feature_offset = slot, generation, offset


class PreparedAudioGeneration:
    """Validate one segment's inputs, encode them, and publish the features into the shared arena."""

    def __init__(self, encoder, state, completion, frame, sequence, *, buffers=None):
        self.encoder, self.state = encoder, state
        self.errors = (torch.zeros(1, dtype=torch.int32, device=completion.records.device) if buffers is None
                       else buffers.zeros((1,), torch.int32))
        # Audio features land in the vision arena, so the publication contract is the vision one.
        self.publish = PreparedFeaturePublication(state, _FeatureSlots(
            encoder.inputs.audio_slot, encoder.inputs.audio_generation,
            encoder.inputs.audio_feature_offset), completion, frame, sequence)

    def __call__(self):
        e, s = self.encoder, self.state
        i, r = e.inputs, s.requests
        capacity = i.audio_features.shape[0]
        slot = i.audio_slot.long().clamp(0, r.capacity - 1)
        frames, offset = i.audio_frames, i.audio_feature_offset
        current = (i.audio_slot >= 0) & (i.audio_slot < r.capacity)
        current &= (i.audio_generation == r.generation[slot]) & (r.status[slot] == 1)
        valid = current & (frames > 0) & (frames <= capacity) & (offset >= 0)
        valid &= (offset == 0) | ((s.generation[slot] == i.audio_generation) & (s.count[slot] == offset))
        valid &= (s.feature_slot >= 0) & (s.feature_slot < s.arena.shape[0])
        live = e.rows < frames.reshape(())
        valid &= (~live[:, None] | torch.isfinite(i.audio_features[:capacity])).all()
        self.errors.copy_(~valid)
        # A rejected segment stores no features: the publication kernel is gated on this error flag.
        output = e()
        self.publish(output, e.count, self.errors)
