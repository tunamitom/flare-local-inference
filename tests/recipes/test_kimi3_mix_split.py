# SPDX-FileCopyrightText: Copyright 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""L3 (upstream flare #11): the residual mixture's opt-in decode geometry (``mix_split_rows``).

CPU coverage of the plan setting's construction contract and of the split kernel's arithmetic
against the one-program-per-row kernel, executed by Triton's Python interpreter (the upstream
CUDA battery rides a later wave). The interpreter schedules a grid's programs serially, so the
interpreted parity runs use one column chunk per row: the multi-chunk partial exchange needs
concurrent programs (its spin-wait is exactly the stream-schedule hazard the L3 audit covers)
and is exercised by the GPU wave. Every delta/commit mode and both mixture forms run here.

Numerics contract asserted (as upstream states it): prefix and committed blocks bit-identical
to the row kernel, the mixture within one BF16 ulp and the normalized output within three
wherever |y| >= row max / 64 (FP32 reduction order; near-zero cancellations judged like
tests/recipes/test_kimi3_residual_cuda.py does).
"""

import inspect
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from ignition.recipes.kimi3 import residual
from ignition.recipes.kimi3.residual import (
    ADD, KEEP, REPLACE, ResidualStream, SPLIT_CHUNK_DEFAULT, SPLIT_STRIDE, SPLIT_WARPS_DEFAULT, score_weight,
)


def _stream(tokens, hidden=64, **settings):
    prefix = torch.zeros((tokens, hidden), dtype=torch.bfloat16)
    blocks = torch.zeros((tokens, 4, hidden), dtype=torch.bfloat16)
    return ResidualStream(prefix, blocks, torch.zeros_like(prefix), **settings)


def test_split_geometry_gate():
    armed = _stream(16, mix_split_rows=16)
    assert armed.split is not None
    partial, epoch, chunk, warps = armed.split
    assert partial.shape == (16, SPLIT_STRIDE) and partial.dtype == torch.int64
    assert epoch.shape == (16,) and epoch.dtype == torch.int32
    assert not partial.any() and not epoch.any()  # allocated once, never reset
    assert (chunk, warps) == (SPLIT_CHUNK_DEFAULT, SPLIT_WARPS_DEFAULT)
    # Streams above the row limit keep one program per row; so does the default (0).
    assert _stream(17, mix_split_rows=16).split is None
    assert _stream(16).split is None
    assert _stream(1, mix_split_rows=1).split is not None
    # Explicit chunk and warps reach the launch geometry.
    custom = _stream(4, hidden=256, mix_split_rows=4, mix_split_chunk=32, mix_split_warps=8)
    assert custom.split[2:] == (32, 8)


def test_split_geometry_validation():
    with pytest.raises(ValueError, match="nonnegative row limit"):
        _stream(4, mix_split_rows=-1)
    with pytest.raises(ValueError, match="nonnegative row limit"):
        _stream(4, mix_split_rows="16")
    # The geometry is checked whenever the setting is on, armed or not (this stream is over
    # the limit): a bad plan setting must fail at construction, not silently fall back.
    with pytest.raises(ValueError, match="power of two"):
        _stream(100, mix_split_rows=4, mix_split_chunk=48)
    with pytest.raises(ValueError, match="at most"):
        _stream(4, mix_split_rows=4, mix_split_chunk=1)  # 64 columns -> 64 chunks
    with pytest.raises(ValueError, match="warps"):
        _stream(4, mix_split_rows=4, mix_split_warps=0)
    with pytest.raises(ValueError, match="warps"):
        _stream(4, mix_split_rows=4, mix_split_warps="4")


def test_split_mixture_matches_row_programs():
    # PYTHONPATH pins this tree: the venv's editable install resolves ignition elsewhere.
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run([sys.executable, str(Path(__file__).resolve())],
        env=dict(os.environ, TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(root)),
        text=True, capture_output=True, timeout=900)
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_component_never_reads_the_environment_to_choose_behaviour():
    """Source-text check of the module, stated as such: the split's gate is the constructor
    argument threaded from the plan setting, and the module must not read the environment to
    choose captured behaviour in any form (the two idioms a source check can pin)."""
    source = inspect.getsource(residual)
    assert "os.environ" not in source, "source-text check: module source must not contain os.environ"
    assert "getenv" not in source, "source-text check: module source must not contain getenv"


def test_the_upstream_environment_channel_is_not_carried():
    """The upstream knob's environment channel must not appear anywhere under ignition/:
    our gate is the ``mix_split_rows`` plan setting and nothing else."""
    root = Path(__file__).resolve().parents[2] / "ignition"
    for path in root.rglob("*.py"):
        assert "FLARE_KIMI3_MIX_SPLIT" not in path.read_text(), f"environment channel reintroduced in {path}"


# -- interpreted parity (runs again in the subprocess above) -------------------

PARITY_CASES = ((0, REPLACE, 0), (1, ADD, -1), (1, REPLACE, -1), (3, KEEP, 5), (4, ADD, 4), (7, ADD, -1))


def _ordered(x):
    i = x.view(torch.int16).to(torch.int32)
    return torch.where(i < 0, -(i & 0x7FFF), i)


def interpreted_parity():
    torch.manual_seed(20261112)
    for rows in (1, 3):
        for sources, mode, write in PARITY_CASES:
            for mixture_form in ("plain", "compact"):
                hidden = 256
                generator = torch.Generator().manual_seed(rows * 131 + sources)
                blocks = torch.randn((rows, 8, hidden), dtype=torch.bfloat16, generator=generator)
                prefix, delta = torch.randn((2, rows, hidden), dtype=torch.bfloat16, generator=generator)
                score = score_weight(torch.randn(hidden, generator=generator).bfloat16(),
                                     (torch.randn(hidden, generator=generator) * 0.02).bfloat16())
                norm = torch.randn(hidden, dtype=torch.bfloat16, generator=generator)
                results = []
                for split in (False, True):
                    stream = ResidualStream(prefix.clone(), blocks.clone(), torch.full_like(prefix, float("nan")),
                                            **(dict(mix_split_rows=rows, mix_split_chunk=hidden) if split else {}))
                    assert (stream.split is not None) == split
                    mixture = torch.full_like(prefix, float("nan"))
                    count = torch.zeros(1, dtype=torch.int32)
                    kwargs = (dict(mixture=mixture) if mixture_form == "plain"
                              else dict(mixture=mixture[:max(1, rows - 1)], mixture_count=count))
                    output = stream.mix(delta, mode, sources=sources, write=write, score=score, norm=norm,
                                        eps=1e-5, output_eps=1e-5, **kwargs)
                    results.append((stream.prefix, stream.blocks, output.clone(), kwargs["mixture"]))
                (prefix0, blocks0, output0, mixture0), (prefix1, blocks1, output1, mixture1) = results
                assert torch.equal(prefix0, prefix1) and torch.equal(blocks0, blocks1), (rows, sources, mode, write)
                # Full-row FP32 sums arrive per chunk, so a mixture may round to the neighbouring BF16
                # value; its RMSNorm and weight multiply each round once more.
                for expected, actual, ulps in ((mixture0, mixture1, 1), (output0, output1, 3)):
                    distance = (_ordered(expected) - _ordered(actual)).abs()
                    large = expected.float().abs() >= expected.float().abs().amax(-1, keepdim=True) / 64
                    assert int(distance[large].max()) <= ulps, (rows, sources, mode, write, int(distance[large].max()))
                    difference = (actual.float() - expected.float()).abs()
                    ulp = expected.float().abs().clamp(min=2 ** -14) * torch.finfo(torch.bfloat16).eps
                    assert bool((difference <= 4 * ulp).all()), float((difference / ulp).max())
                    assert (actual != expected).float().mean().item() < 2e-3


if __name__ == "__main__":
    interpreted_parity()
    print("interpreted parity ok")
