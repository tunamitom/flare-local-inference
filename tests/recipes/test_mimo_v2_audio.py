"""MiMo audio preparation: token arithmetic, contract emission, and tokenizer parity.

The tokenizer is a discrete bottleneck: twenty residual codebooks quantize a residual that shrinks
layer by layer, so two BF16 kernel stacks that differ only in reduction order can select different
codes at depth. The parity tests below therefore separate what must hold exactly from what can only
hold statistically:

* the processor's placeholder count and the encoder's produced count are the same number for every
  mel length (a mismatch is a hard splice failure at request time);
* each deterministic stage matches the reference when handed the reference's own input
  (convolutions, pooling, the residual quantizer, and the embeddings/transformer/projection);
* end to end the shipped BF16 graphs agree with the reference to the noise floor the RVQ's own
  argmax decisions impose. Measured on the checkpoint: one BF16 ulp applied to one value in seven
  of the pooled hidden moves code agreement by about two points, and the 24-layer encoder's
  kernel-order difference (0.4% relative on the hidden state) lands near 0.55 overall, with layer 0
  above 0.92 and layer 1 above 0.78.
"""

import importlib.util
import json
import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from ignition.preparation import runtime_package_pb2 as proto
from ignition.preparation.execution import GraphSignature
from ignition.recipes.mimo_v2 import contract as mimo_contract
from ignition.recipes.mimo_v2.audio import (
    AUDIO_CAPACITIES, GROUP, AudioInputs, AudioScratch, PreparedAudio, encoded_tokens, processor_tokens,
)
from ignition.recipes.mimo_v2.audio_weights import AudioWeights

SNAPSHOT = Path(os.environ.get('FLARE_MIMO_CHECKPOINT', '/storage/MiMo-V2.6-Pro-MOPD'))
REFERENCE = Path('/opt/vllm-kk/vllm/model_executor/models/mimo_audio.py')
GOLDEN = {101: 7, 1001: 63, 3001: 188, 6001: 376, 6101: 382, 12001: 751, 30001: 1876, 60001: 3751}


def test_token_counts_agree_over_a_dense_sweep():
    """The processor's placeholder count and the encoder's output count are one number."""
    mismatches = [(frames, processor_tokens(frames), encoded_tokens(frames))
                  for frames in range(1, 8001) if processor_tokens(frames) != encoded_tokens(frames)]
    assert mismatches == []
    # 120 s, 300 s (the tokenizer's RoPE table), 600 s (the frontend's decode cap), and the goldens.
    for frames, expected in GOLDEN.items():
        assert processor_tokens(frames) == encoded_tokens(frames) == expected
    for frames in (12001, 30001, 60001):
        assert processor_tokens(frames) == encoded_tokens(frames)


def test_contract_declares_the_audio_capacity_and_feature_token():
    from types import SimpleNamespace

    signature = GraphSignature('audio', 1, AUDIO_CAPACITIES[-1], tuple(AudioInputs.allocate('cpu').buffers()))
    pool = SimpleNamespace(layout=SimpleNamespace(row_bytes=4096,
                                                  region=lambda name: SimpleNamespace(offset=0, nbytes=8)))
    model = mimo_contract.prepare({'audio6000': signature}, {'page_tables': None}, pool, image_slots=1,
                                  page_tokens=128, draft_lag=None)
    assert model['audio'] == {'frame_capacities': [6000]}
    frontend = model['frontend']
    assert frontend['audio_preparation'] == proto.MIMO_AUDIO_PREPARATION
    assert frontend['audio_tokens'] == {'feature_token': 151669}
    assert frontend['audio_tokens']['feature_token'] != frontend['image_tokens']['feature_token']
    graph = next(graph for graph in model['graphs'] if graph['graph'] == 'audio6000')
    assert graph['kind'] == proto.GraphOperation.Value('AUDIO') == 14
    assert graph['state'] == [] and graph['pending_token'] is None
    assert {binding['source'] for binding in graph['inputs']} == {proto.AUDIO_TENSOR}
    assert {binding['indexing'] for binding in graph['inputs']} == {proto.SCALAR}
    assert {binding['tensor'] for binding in graph['inputs']} == set(AudioInputs.allocate('cpu').buffers())


def test_frontend_audio_settings_follow_the_checkpoint():
    if not (SNAPSHOT / 'audio_tokenizer' / 'config.json').is_file():
        pytest.skip('MiMo audio tokenizer checkpoint is unavailable')
    from ignition.preparation.frontend import audio_settings, prepare_frontend

    settings = prepare_frontend(SNAPSHOT, eos_token_ids=[1], image_profile='mimo', audio_profile='mimo')['audio']
    assert settings == {'mimo': dict(sampling_rate=24000, n_fft=960, hop_length=240, window_length=960,
                                     n_mels=128, f_min=0.0, f_max=None, segment_frames=6000, kernel_size=3,
                                     stride=2, pooling=2, group_size=4, max_seconds=300)}
    config = json.loads((SNAPSHOT / 'audio_tokenizer' / 'config.json').read_text())
    config['stride_size'] = 3
    with pytest.raises(ValueError):
        audio_settings(config, json.loads((SNAPSHOT / 'config.json').read_text()), 'mimo')


pytest_cuda = pytest.mark.skipif(os.environ.get('FLARE_RUN_CUDA_TESTS') != '1',
                                 reason='requires an explicitly assigned CUDA device')


def _reference():
    """The fork's own MiMo audio module, beside the flash-attention stand-in it imports lazily."""
    if not REFERENCE.is_file():
        pytest.skip('the kk-mimo vLLM fork is unavailable')
    spec = importlib.util.spec_from_file_location('mimo_audio_reference', REFERENCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _reference_encoder(module):
    """The reference audio tower with the checkpoint's tokenizer and language-model audio rows."""
    from safetensors import safe_open

    encoder = module.MimoAudioEncoder(json.loads((SNAPSHOT / 'config.json').read_text())['audio_config'],
                                     model_path=str(SNAPSHOT))
    state = {}
    shard = safe_open(str(SNAPSHOT / 'model_pp0_ep0_shard1.safetensors'), 'pt')
    for key in shard.keys():
        if key.startswith('audio_encoder.') or key.startswith('speech_embeddings.'):
            state[key.removeprefix('audio_encoder.') if key.startswith('audio_encoder.') else key] = (
                shard.get_tensor(key))
    missing, unexpected = encoder.load_state_dict(state, strict=False)
    assert unexpected == []
    return encoder.to(torch.device('cuda', 0), torch.bfloat16).eval()


def _shard_reader(root, device):
    """Read one language-model tensor straight from its shard, without the pinned io_uring loader."""
    from safetensors import safe_open

    index = json.loads((root / 'model.safetensors.index.json').read_text())['weight_map']
    opened = {}

    def read(name):
        shard = index[name]
        if shard not in opened:
            opened[shard] = safe_open(str(root / shard), 'pt', device=str(device))
        return opened[shard].get_tensor(name)
    return read


def _relative(ours, theirs):
    return (ours.float() - theirs.float()).norm(dim=1) / theirs.float().norm(dim=1).clamp_min(1e-6)


@pytest_cuda
def test_audio_tower_matches_the_reference_on_the_checkpoint():
    assert os.environ.get('CUDA_VISIBLE_DEVICES'), 'pin one free GPU'
    if not (SNAPSHOT / 'audio_tokenizer' / 'model.safetensors').is_file():
        pytest.skip('MiMo audio tokenizer checkpoint is unavailable')
    from b12x.preparation import PreparationSession

    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.preparation.scratch import ScratchArena

    device = torch.device('cuda', 0)
    torch.cuda.set_device(device)
    torch.manual_seed(0)
    module = _reference()
    encoder = _reference_encoder(module)
    tokenizer = encoder.audio_tokenizer.encoder
    torch.backends.cuda.matmul.allow_tf32 = False

    weights = AudioWeights.load(inspect_indexed_checkpoint(SNAPSHOT), device, reader=_shard_reader(SNAPSHOT, device))
    inputs = AudioInputs.allocate(device)
    scratch = AudioScratch.allocate(AUDIO_CAPACITIES[-1], device)
    arena = ScratchArena(device)
    with PreparationSession(device=device, autotune=False, compile_workers=0) as session:
        tower = PreparedAudio(weights, scratch, inputs, session=session, workspace=arena, name='audio6000')

        # Ten seconds is a sub-capacity segment: the graph must zero-extend the mel and still match.
        frames = 1001
        mel = torch.randn(frames, 128, device=device) * 1.5 - 1.0
        inputs.audio_features.zero_()
        inputs.audio_features[:frames].copy_(mel)
        inputs.audio_frames.fill_(frames)
        code_length, pooled_length, token_length = (int(value) for value in tower.segment_lengths(inputs.audio_frames))
        assert token_length == processor_tokens(frames) == 63

        # --- the reference's stages, replayed through the fork's own modules
        with torch.no_grad():
            source = mel.T[None].to(torch.bfloat16)
            conv1 = F.gelu(tokenizer.conv1(source))
            conv2 = F.gelu(tokenizer.conv2(conv1))
            assert conv2.shape[2] == code_length
            hidden = conv2.permute(0, 2, 1)[0].contiguous()
            lengths = torch.tensor([code_length], device=device)
            positions = module.get_position_ids(lengths).long()
            cos, sin = tokenizer.position_embedding(source, positions)
            cu = F.pad(torch.cumsum(lengths, 0), (1, 0), 'constant', 0).to(torch.int32)
            skip = None
            for index, layer in enumerate(tokenizer.layers):
                hidden = layer(hidden, cu, code_length, (cos, sin))
                if index == tokenizer.skip_layer_idx - 1:
                    skip = hidden.clone()
            normed = tokenizer.layer_norm(hidden + skip)
            pooled = normed[None]
            if pooled.size(1) % 2:
                pooled = F.pad(pooled, (0, 0, 0, 1))
            pooled = tokenizer.down_sample_norm(
                tokenizer.down_sample_layer(pooled.transpose(1, 2)).transpose(1, 2)[0])
            codes = tokenizer.quantizer.float().encode(pooled[None].float())[:, 0, :]
            assert pooled.shape[0] == pooled_length

            # --- stage 1: the convolutions agree to convolution-kernel rounding, including the
            # last code frame whose second mel column is the zero extension
            tower.convolve()
            error = _relative(scratch.residual[:code_length], conv2[0].T).max()
            assert error < 2e-3, error

            # --- stage 2: the encoder layers, skip connection, and pooling stay within rounding
            tower.transform()
            assert _relative(scratch.normalized[:code_length], normed).max() < 3e-2
            tower.pool()
            assert _relative(scratch.pooled[:pooled_length], pooled).max() < 8e-2

            # --- stage 3: given the reference's pooled frames, all twenty codebooks pick the same
            # codes; the residual quantizer is exact
            scratch.pooled[:pooled_length].copy_(pooled)
            tower.quantize()
            assert torch.equal(scratch.codes[:, :pooled_length], codes)

            # --- stage 4: given the reference's codes, the published features match to one ulp,
            # which proves the grouped padding, the summed tables, the local transformer, and the
            # projection
            scratch.codes[:, :pooled_length].copy_(codes)
            features = tower.features()
            padded = encoder.process_audio(codes.t().contiguous())
            expected = encoder.projection(encoder.apply_input_local_transformer(
                encoder.apply_speech_embeddings(padded)).reshape(padded.shape[0], -1)).float()
            assert padded.shape[0] == token_length
            cosine = F.cosine_similarity(features[:token_length].float(), expected, dim=-1)
            assert cosine.min() > 0.9999, cosine.min()

        # --- end to end: one graph, one segment, the same token count and the RVQ's noise floor
        with torch.no_grad():
            output = tower()
            reference_features, token_counts = encoder.get_audio_feature([mel])
        assert int(tower.count) == 63
        assert token_counts == [63]
        cosine = F.cosine_similarity(output[:63].float(), reference_features.float(), dim=-1)
        assert cosine.mean() > 0.9, cosine.mean()
        agreement = (scratch.codes[:, :pooled_length] == codes).float()
        assert agreement.mean() > 0.4, agreement.mean().item()
        assert agreement.mean(1)[0] > 0.9, agreement.mean(1).tolist()

        # --- the export's capture requirement: the reference forward reads
        # `torch.max(output_length).item()` (mimo_audio.py:888), a device-to-host sync that makes
        # it uncapturable. This graph derives the extent on device and replays bit-identically.
        baseline = scratch.output.clone()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            tower()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            tower()
        scratch.output.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert int(tower.count) == 63
        assert torch.equal(scratch.output, baseline)
