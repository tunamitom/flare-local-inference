"""Fused MiMo QKV rotation and window staging match the PyTorch expressions bit for bit."""

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get('FLARE_RUN_CUDA_TESTS') != '1',
                                reason='requires FLARE_RUN_CUDA_TESTS=1 and an assigned GPU')

HEADS, KEY, VALUE, ROTARY = 16, 192, 128, 64


def _reference(projected, positions, inverse, scale):
    from ignition.recipes.mimo_v2.attention import partial_rotary

    q = projected[:, :HEADS * KEY].reshape(-1, HEADS, KEY)
    k = projected[:, HEADS * KEY:(HEADS + 1) * KEY].reshape(-1, 1, KEY)
    v = projected[:, (HEADS + 1) * KEY:(HEADS + 1) * KEY + VALUE].reshape(-1, 1, VALUE)
    return partial_rotary(q, positions, inverse), partial_rotary(k, positions, inverse), (
        v if scale is None else v * scale)


@pytest.mark.parametrize('rows', [1, 7, 32, 256])
@pytest.mark.parametrize('scale', [None, 0.612])
def test_rotation_matches_partial_rotary(rows, scale):
    from ignition.recipes.mimo_v2 import qkv

    device = torch.device('cuda', 0)
    generator = torch.Generator(device=device).manual_seed(rows)
    inverse = 1.0 / (10000 ** (torch.arange(0, ROTARY, 2, dtype=torch.float32, device=device) / ROTARY))
    projected = torch.randn((rows, 3456), device=device, generator=generator).bfloat16()
    positions = torch.randint(-1, 1 << 20, (rows,), device=device, generator=generator)
    query = torch.empty((rows, HEADS, KEY), dtype=torch.bfloat16, device=device)
    key = torch.empty((rows, 1, KEY), dtype=torch.bfloat16, device=device)
    value = torch.empty((rows, 1, VALUE), dtype=torch.bfloat16, device=device)
    retained = torch.empty_like(key), torch.empty_like(value)
    qkv.rotate(projected, positions, inverse, query, key, value, scale, retained=retained)
    expected = _reference(projected, positions, inverse, scale)
    for actual, wanted in zip((query, key, value, *retained), (*expected, *expected[1:])):
        assert torch.equal(actual, wanted)


@pytest.mark.parametrize('dtype', [torch.float8_e4m3fn, torch.bfloat16])
@pytest.mark.parametrize('width,starts,lengths', [(7, [300], [7]), (8, [0], [8]),
    (8, [127, 255, 0, 137], [8, 0, 3, 1]), (3, [128, 1000, 2, 129], [3, 3, 3, 3])])
def test_staging_matches_stage_windows(dtype, width, starts, lengths):
    from ignition.components.paged_batch import Metadata
    from ignition.components.window_attention import stage_windows
    from ignition.recipes.mimo_v2 import qkv

    device, window, requests = torch.device('cuda', 0), 128, len(starts)
    tokens = requests * width
    histories = [(torch.randn((6, window, 1, size), device=device) * 100).to(dtype) for size in (KEY, VALUE)]
    sources = [(torch.randn((tokens, 1, size), device=device) * 300).bfloat16() for size in (KEY, VALUE)]
    pages = (window + width + 127) // 128
    metadata = Metadata.allocate(tokens, requests, 1, device, samples=tokens)
    metadata.set(torch.tensor(starts, device=device), torch.tensor(lengths, dtype=torch.int32, device=device),
                 torch.tensor([1, 2, 3, 5][:requests], device=device),
                 torch.zeros((requests, 1), dtype=torch.int32, device=device),
                 torch.zeros((requests, 8), dtype=torch.int64, device=device),
                 torch.tensor([sum(lengths)], device=device))
    expected = [torch.full((requests * pages, 128, 1, size), 7., device=device).to(dtype) for size in (KEY, VALUE)]
    actual = [tensor.clone() for tensor in expected]
    for history, source, output in zip(histories, sources, expected):
        stage_windows(history, source, output, metadata)
    window_lengths = torch.zeros_like(metadata.lengths)
    qkv.stage(histories[0], sources[0], actual[0], histories[1], sources[1], actual[1], metadata, window_lengths)
    for left, right in zip(actual, expected):
        assert torch.equal(left.view(torch.uint8), right.view(torch.uint8))
    assert torch.equal(window_lengths, metadata.starts.clamp(max=window).int() + metadata.lengths)
