"""Native MiMo audio and video preprocessing match the vLLM fork's MiMo Omni processor.

The oracle is the processor the serving reference runs (``mimo_omni_oracle``): its log-mel
features, token arithmetic, frame sampling, timestamp groups, and standardized frame-pair
patches. Image preprocessing keeps its Transformers reference in ``test_mimo_images.py``.
"""

import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from tests.frontend import mimo_omni_oracle as oracle

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = Path('/storage/MiMo-V2.6-Pro-MOPD')

AUDIO_SETTINGS = {'mimo': dict(sampling_rate=24000, n_fft=960, hop_length=240, window_length=960, n_mels=128,
                               f_min=0.0, f_max=None, segment_frames=6000, kernel_size=3, stride=2, pooling=2,
                               group_size=4, max_seconds=3600)}
VIDEO_SETTINGS = {'mimo': dict(min_pixels=8192, max_pixels=8388608, total_max_pixels=268435456, num_frames=32,
                               max_frames=3600, pixel_mean=[123.675, 116.28, 103.53], pixel_std=[58.395, 57.12, 57.375])}


@pytest.fixture(scope='module')
def tool():
    if not oracle.available() or not (SNAPSHOT / 'config.json').is_file():
        pytest.skip('the vLLM fork oracle or the MiMo checkpoint is unavailable')
    config = json.loads((SNAPSHOT / 'config.json').read_text())['processor_config']
    assert AUDIO_SETTINGS['mimo'] | dict(sampling_rate=config['audio_sampling_rate'], n_fft=config['audio_nfft'],
        hop_length=config['audio_hop_length'], window_length=config['audio_window_size'], n_mels=config['audio_n_mels'],
        f_min=float(config['audio_fmin']), f_max=config['audio_fmax'], segment_frames=config['audio_segment_size'],
        kernel_size=config['audio_kernel_size'], stride=config['audio_stride_size'], pooling=config['audio_avg_pooler'],
        group_size=config['audio_group_size']) == AUDIO_SETTINGS['mimo']
    assert (config['video_min_pixels'], config['video_max_pixels'], config['video_total_max_pixels']) == (
        VIDEO_SETTINGS['mimo']['min_pixels'], VIDEO_SETTINGS['mimo']['max_pixels'], VIDEO_SETTINGS['mimo']['total_max_pixels'])
    subprocess.run(['cargo', 'build', '--quiet', '--manifest-path', str(ROOT / 'helios/Cargo.toml'),
                    '--features', 'server,test-tools', '--bin', 'test_mimo_media'], check=True)
    return ROOT / 'helios/target/debug/test_mimo_media'


def run(tool, tmp_path, case):
    (tmp_path / 'case.json').write_text(json.dumps(case))
    subprocess.run([str(tool), str(tmp_path / 'case.json'), str(tmp_path)], check=True)
    return json.loads((tmp_path / 'metadata.json').read_text())


@pytest.mark.parametrize('seconds,rate,channels', [(2.0, 48000, 2), (0.7, 16000, 1), (61.3, 44100, 1), (3.0, 24000, 2)])
def test_log_mel_segments_and_tokens_match_the_oracle(tool, tmp_path, seconds, rate, channels):
    processor = oracle.processor(SNAPSHOT)
    rng = np.random.default_rng(int(seconds * 1000) + rate)
    t = np.arange(int(seconds * rate)) / rate
    wave = np.stack([0.5 * np.sin(2 * np.pi * (440 + 110 * c) * t) + 0.05 * rng.standard_normal(t.shape)
                     for c in range(channels)]).astype(np.float32)
    waveform, sr = oracle.audio_item(wave, rate)
    out = processor(text='<|mimo_audio_start|><|audio_pad|><|mimo_audio_end|>', audio=[(waveform, sr)])
    expected = out['audio_features'][0]
    tokens = int(out['audio_token_lens'][0])
    assert out['input_ids'][0].tolist().count(151669) == tokens
    mono = wave.mean(axis=0) if channels > 1 else wave[0]
    (tmp_path / 'clip.f32').write_bytes(mono.astype(np.float32).tobytes())
    actual = run(tool, tmp_path, {'audio_settings': AUDIO_SETTINGS,
                                  'audio': {'pcm': str(tmp_path / 'clip.f32'), 'sample_rate': rate}})['audio']
    assert (actual['frames'], actual['mels']) == tuple(expected.shape)
    assert actual['tokens'] == tokens
    assert sum(segment['tokens'] for segment in actual['segments']) == tokens
    assert [segment['frames'] for segment in actual['segments']] == (
        [6000] * (expected.shape[0] // 6000) + ([expected.shape[0] % 6000] if expected.shape[0] % 6000 else []))
    mel = torch.from_numpy(np.fromfile(tmp_path / 'mel.f32', dtype=np.float32)).view(expected.shape)
    # The oracle resamples and transforms in float32 while the native path accumulates in f64;
    # bands the resampler's lowpass empties (above 96 of 128 for inputs below 24 kHz) amplify
    # that accumulation-order noise in the log domain. The graph consumes BF16, whose spacing at
    # the clamp floor is 0.0625, so both bounds sit far inside the encoder's own rounding. Long
    # clips widen the spread: 61.3 s at 44.1 kHz drifts to 4.2e-4 in four low-band cells because
    # every output frame carries the reference's own float32 rounding.
    torch.testing.assert_close(mel[:, :96], expected[:, :96], rtol=0, atol=1e-3)
    torch.testing.assert_close(mel, expected, rtol=0, atol=2e-3)
    assert actual['tokens_per_second'] == 25.0 / 4


@pytest.mark.parametrize('total_frames,fps,size,num_frames', [(90, 30.0, (64, 48), None), (7, 1.0, (96, 64), None),
                                                              (1000, 29.97, (128, 72), 8), (33, 25.0, (70, 50), 40)])
def test_video_sampling_pairs_and_patches_match_the_oracle(tool, tmp_path, total_frames, fps, size, num_frames):
    processor = oracle.processor(SNAPSHOT)
    width, height = size
    rng = np.random.default_rng(total_frames)
    all_frames = rng.integers(0, 256, (total_frames, height, width, 3), dtype=np.uint8)
    indices = oracle.sample_frame_indices(total_frames, fps, num_frames=32 if num_frames is None else num_frames)
    sampled = all_frames[indices]
    out = processor(text='<|vision_start|><|video_pad|><|vision_end|>', videos=[oracle.video_item(sampled)])
    ids = out['input_ids'][0].tolist()
    grid = out['video_grid_thw'][0].tolist()
    expected_pixels = out['pixel_values_videos']
    paths = []
    for index, frame in enumerate(sampled):
        path = tmp_path / f'frame{index}.rgb'
        path.write_bytes(frame.tobytes())
        paths.append(str(path))
    case = {'video_settings': VIDEO_SETTINGS, 'video': {'frames': paths, 'width': width, 'height': height,
            'total_frames': total_frames, 'fps': fps}}
    if num_frames is not None:
        case['video']['num_frames'] = num_frames
    actual = run(tool, tmp_path, case)['video']
    assert actual['sampled_indices'] == indices
    assert len(actual['pairs']) == grid[0]
    assert all(pair['grid_thw'] == [1, grid[1], grid[2]] for pair in actual['pairs'])
    assert actual['tokens'] == ids.count(151656)
    # The block replaces the template's whole vision envelope, so the processor's ids are the
    # block itself: video_start, one timestamped vision envelope per grid, video_end.
    assert actual['grid_thw'] == grid
    assert actual['second_per_grid_ts'] == 2.0
    assert actual['prompt_tokens'] == len(ids)
    # The oracle's prompt carries "MM:SS" before each pair's vision envelope.
    rendered = processor.tokenizer.decode(ids)
    timestamps = [part.split('<|vision_start|>')[0] for part in rendered.split('<|mimo_video_start|>')[1].split('<|vision_end|>')[:-1]]
    assert [pair['timestamp'] for pair in actual['pairs']] == timestamps
    pixels = torch.from_numpy(np.fromfile(tmp_path / 'pixels.f32', dtype=np.float32)).view(expected_pixels.shape)
    torch.testing.assert_close(pixels, expected_pixels, rtol=0, atol=2e-4)
    positions = [pos for pair in actual['pairs'] for pos in pair['positions']]
    expected_positions = [[y + dy, x + dx] for _ in range(grid[0]) for y in range(0, grid[1], 2) for x in range(0, grid[2], 2)
                          for dy in range(2) for dx in range(2)]
    assert positions == expected_positions


def test_oracle_frame_sampling_is_numpy_linspace():
    assert oracle.sample_frame_indices(90, 30.0) == list(np.linspace(0, 89, 32, dtype=int))
    assert oracle.sample_frame_indices(10, 30.0) == list(range(10))
    assert oracle.sample_frame_indices(90, 30.0, target_fps=1) == list(np.linspace(0, 89, 3, dtype=int))
    assert math.isclose(2 * 268435456 / 32, 16777216.0)
