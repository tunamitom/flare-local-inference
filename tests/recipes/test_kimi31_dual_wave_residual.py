# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""L9 dual-wave determinism: a substream's mix must equal the serial mix's rows bitwise.

The two-wave schedule splits ROWS (ResidualStream.substream); every `_mix` step is
per-row and ROW_OFFSET only re-bases the compact-mixture retention, so the split
outputs must be bit-identical to the serial path. This is the recorded numerics
contract (levers/evidence/l9-dualwave/port-map.md): pointwise/mix/reduce stages
bit-equal; only GEMM accumulation order may move, priced at the gates battery.
"""

import subprocess
import sys

import pytest


def _run_interpreted(code: str) -> None:
    import os

    env = dict(os.environ, TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES="")
    res = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=240)
    if res.returncode != 0:
        raise AssertionError(f"TRITON_INTERPRET subprocess failed (rc={res.returncode}):\n"
                             f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")


_BODY = r"""
import torch
from ignition.recipes.kimi3.residual import ADD, REPLACE, ResidualStream

TOKENS, HIDDEN, BLOCKS = 256, 1024, 4
HALF = TOKENS // 2
eps = 1e-6

def fresh(seed):
    gen = torch.Generator().manual_seed(seed)
    prefix = (torch.randn((TOKENS, HIDDEN), generator=gen) * 3.).to(torch.bfloat16)
    blocks = (torch.randn((TOKENS, BLOCKS, HIDDEN), generator=gen) * 2.).to(torch.bfloat16)
    output = torch.zeros((TOKENS, HIDDEN), dtype=torch.bfloat16)
    delta = (torch.randn((TOKENS, HIDDEN), generator=gen) * 1.5).to(torch.bfloat16)
    score = (torch.randn((HIDDEN,), generator=gen) * .5).to(torch.bfloat16)
    norm = (torch.randn((HIDDEN,), generator=gen) * .5 + 1.).to(torch.bfloat16)
    return prefix, blocks, output, delta, score, norm

cases = [
    # (sources, write, mode, compact)
    (4, -1, ADD, False),
    (2, 2, ADD, False),
    (0, 0, REPLACE, False),
    (4, -1, ADD, True),
    (3, 3, ADD, True),
]
for case, (sources, write, mode, compact) in enumerate(cases):
    a_prefix, a_blocks, a_output, a_delta, score, norm = fresh(20261112 + case)
    b_prefix, b_blocks, b_output, b_delta, _, _ = fresh(20261112 + case)
    assert torch.equal(a_prefix.view(torch.int16), b_prefix.view(torch.int16))

    mixture_a = torch.zeros((96, HIDDEN), dtype=torch.bfloat16)
    mixture_b = torch.zeros((96, HIDDEN), dtype=torch.bfloat16)
    count = torch.tensor([200], dtype=torch.int64)
    kwargs = dict(sources=sources, write=write, score=score, norm=norm, eps=eps, output_eps=eps)
    if compact:
        kwargs.update(mixture_window=64, mixture_alignment=16)

    full = ResidualStream(a_prefix, a_blocks, a_output)
    full.mix(a_delta, mode, mixture=mixture_a if compact else None,
             mixture_count=count if compact else None, **kwargs)

    stream = ResidualStream(b_prefix, b_blocks, b_output)
    res_a = stream.substream(0, HALF)
    res_b = stream.substream(HALF, TOKENS)
    res_a.mix(b_delta[:HALF], mode, mixture=mixture_b if compact else None,
              mixture_count=count if compact else None, output=b_output[:HALF], **kwargs)
    res_b.mix(b_delta[HALF:], mode, mixture=mixture_b if compact else None,
              mixture_count=count if compact else None, output=b_output[HALF:], **kwargs)

    assert torch.equal(a_output.view(torch.int16), b_output.view(torch.int16)), (sources, write, mode, compact)
    assert torch.equal(a_prefix.view(torch.int16), b_prefix.view(torch.int16)), (sources, write, mode, compact)
    assert torch.equal(a_blocks.view(torch.int16), b_blocks.view(torch.int16)), (sources, write, mode, compact)
    if compact:
        assert torch.equal(mixture_a.view(torch.int16), mixture_b.view(torch.int16)), (sources, write)

# The wave pair may write its mixture through substreams as the schedule does: each
# wave mixes its own rows but both write the shared retained window.
a_prefix, a_blocks, a_output, a_delta, score, norm = fresh(20261199)
b_prefix, b_blocks, b_output, b_delta, _, _ = fresh(20261199)
mixture_a = torch.zeros((96, HIDDEN), dtype=torch.bfloat16)
mixture_b = torch.zeros((96, HIDDEN), dtype=torch.bfloat16)
count = torch.tensor([200], dtype=torch.int64)
kwargs = dict(sources=4, write=-1, score=score, norm=norm, eps=eps, output_eps=eps,
              mixture_window=64, mixture_alignment=16)
full = ResidualStream(a_prefix, a_blocks, a_output)
full.mix(a_delta, ADD, mixture=mixture_a, mixture_count=count, **kwargs)
stream = ResidualStream(b_prefix, b_blocks, b_output)
res_a = stream.substream(0, HALF)
res_b = stream.substream(HALF, TOKENS)
res_a.mix(b_delta[:HALF], ADD, mixture=mixture_b, mixture_count=count, output=b_output[:HALF], **kwargs)
res_b.mix(b_delta[HALF:], ADD, mixture=mixture_b, mixture_count=count, output=b_output[HALF:], **kwargs)
assert torch.equal(mixture_a.view(torch.int16), mixture_b.view(torch.int16))
assert torch.equal(a_output.view(torch.int16), b_output.view(torch.int16))
print("OK")
"""


def test_dual_wave_substreams_match_the_serial_mix_bitwise():
    _run_interpreted(_BODY)


def test_substream_bounds_and_row_indexing_are_validated():
    import torch

    from ignition.recipes.kimi3.residual import ResidualStream

    prefix = torch.zeros((64, 256), dtype=torch.bfloat16)
    blocks = torch.zeros((64, 2, 256), dtype=torch.bfloat16)
    output = torch.zeros((64, 256), dtype=torch.bfloat16)
    stream = ResidualStream(prefix, blocks, output)
    assert stream.full_rows == 64 and stream.row_offset == 0
    sub = stream.substream(8, 24)
    assert sub.prefix.shape[0] == 16 and sub.row_offset == 8 and sub.full_rows == 64
    assert sub.prefix.data_ptr() == prefix[8:24].data_ptr()
    nested = sub.substream(4, 8)
    assert nested.row_offset == 12 and nested.full_rows == 64
    for bad in ((0, 0), (24, 8), (-1, 4), (8, 65)):
        try:
            stream.substream(*bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted bounds {bad}")
    try:
        ResidualStream(prefix[8:24], blocks[8:24], output[8:24], row_offset=8, full_rows=64)
    except ValueError:
        raise AssertionError("valid substream geometry was rejected")
    try:
        ResidualStream(prefix[8:24], blocks[8:24], output[8:24], row_offset=8, full_rows=20)
    except ValueError:
        pass
    else:
        raise AssertionError("accepted rows outside the full stream")
