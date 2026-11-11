# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0
"""L4 (upstream flare #16): the prefill latent gather as a plan setting.

Wiring (Plan field + --prefill-latent-gather, never an environment switch), the
prefill-only channel contract, the (512, 32) launch geometry, and the
layout/numerics contract (one FP32 full-K dot product rounded to BF16 once).
CPU only; the B+D harness cell at 3072x3584 rides the later GPU wave.
"""
import ast
from pathlib import Path

import pytest
import torch

from ignition.components.collectives import PreparedAllGather, PreparedChunkedAllGather
from ignition.recipes.kimi3 import decoder
from ignition.recipes.kimi3.decoder import LATENT_GATHER_LAUNCH, latent_gather_supported

ROOT = Path(decoder.__file__).resolve().parent
PACKAGE = ROOT.parent.parent


def test_partitions_need_64_column_rank_shards():
    assert latent_gather_supported(2, 3584) and latent_gather_supported(4, 3584)
    assert latent_gather_supported(7, 3584) and latent_gather_supported(8, 3584)
    assert not latent_gather_supported(1, 3584)   # nothing to gather
    assert not latent_gather_supported(3, 3584)
    assert not latent_gather_supported(16, 3584)  # 224 columns per rank is not 64-aligned
    assert not latent_gather_supported(8, 3500)


def test_launch_geometry_is_the_measured_tp8_shape():
    # 512:32 was upstream's fastest measured geometry at TP8 for the 3072x3584
    # gather; the channel default (256:16) halves the measured gain and the flat
    # gather removes it (L4 risk 3). The default-off launch keeps every existing
    # gather's query identity.
    assert LATENT_GATHER_LAUNCH == (512, 32)
    import inspect

    for cls in (PreparedAllGather, PreparedChunkedAllGather):
        assert inspect.signature(cls.__init__).parameters["launch"].default is None


def _calls(path, func):
    tree = ast.parse(path.read_text())
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call)
            and getattr(n.func, "id", getattr(n.func, "attr", None)) == func]


def test_launch_geometry_reaches_only_the_gather_query():
    # The geometry enters the DcpAllToAll.all_gather_heads call dict only when a
    # caller passes launch=; every existing PreparedAllGather query is unchanged.
    tree = ast.parse((PACKAGE / "components" / "collectives.py").read_text())
    (update,) = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and getattr(n.func, "id", getattr(n.func, "attr", None)) == "update"
                 and "threads" in ast.unparse(n)]
    guards = [n for n in ast.walk(tree)
              if isinstance(n, ast.If) and ast.unparse(n.test) == "launch is not None"
              and any(update is m for m in ast.walk(n))]
    assert guards, "the launch geometry must sit behind `if launch is not None`"


def test_only_prefill_decoders_receive_the_channel():
    # PreparedDecoder forwards the channel to the MoE only in prefill mode ...
    (moe,) = _calls(ROOT / "decoder.py", "PreparedLatentMoE")
    kw = {k.arg: ast.unparse(k.value) for k in moe.keywords}
    assert kw["latent_gather_runtime"] == "latent_gather_runtime if mode == 'prefill' else None"
    # ... and the bootstrap hands it only to the main prefill decoders, never through the shared
    # collectives dict that speculation's decoders (decode / verify / draft) receive.
    src = (ROOT / "bootstrap.py").read_text()
    assert 'latent_gather_runtime=latent_gather_collective if mode == "prefill" else None' in src
    collectives = [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Assign)
                   and any(getattr(t, "id", None) == "collectives" for t in n.targets)]
    assert collectives and all("latent_gather" not in ast.unparse(n.value) for n in collectives)


def test_the_plan_setting_is_authoritative():
    # Upstream's FLARE_KIMI3_PREFILL_LATENT_GATHER wiring must not survive the port:
    # the plan setting flows Plan -> PreparationConfig, with no environment read.
    for name in ("decoder.py", "bootstrap.py", "recipe.py"):
        source = (ROOT / name).read_text()
        assert "FLARE_KIMI3_PREFILL_LATENT_GATHER" not in source
    from ignition.recipes.kimi3.bootstrap import PreparationConfig

    assert PreparationConfig().prefill_latent_gather is False
    assert PreparationConfig(prefill_latent_gather=True).prefill_latent_gather is True
    # The reduce path stays armed as the default: the gather replaces down_reduce
    # only when the channel reaches the MoE.
    source = (ROOT / "decoder.py").read_text()
    assert "self.down_reduce(self.down(self.source))" in source
    assert 'buffer("prefill_local_down"' in source


def test_row_sharded_projection_gather_equals_full_projection():
    # Rank r computes latent columns [r*w, (r+1)*w) over the full hidden row; a rank-major column gather
    # rebuilds exactly the unsharded BF16 projection (one rounding per value, no reduction).
    torch.manual_seed(0)
    world, rows, hidden, latent = 8, 5, 512, 1024
    x = torch.randn(rows, hidden).to(torch.bfloat16)
    w = torch.randn(latent, hidden).to(torch.bfloat16)
    full = (x.double() @ w.double().T).float()
    width = latent // world
    shards = [(x.double() @ w[r * width:(r + 1) * width].double().T).float().to(torch.bfloat16) for r in range(world)]
    gathered = torch.stack(shards, 1).reshape(rows, latent)  # out[row, rank, column] layout of all_gather_heads
    assert torch.equal(gathered, full.to(torch.bfloat16))


def test_today_rounds_twice_gather_rounds_once():
    # The reduction path rounds each rank's partial to BF16 before the FP32 sum; the gather path rounds once.
    torch.manual_seed(1)
    world, rows, hidden, latent = 8, 64, 1024, 256
    x = torch.randn(rows, hidden).to(torch.bfloat16).double()
    w = torch.randn(hidden, latent).to(torch.bfloat16).double()
    ref = x @ w
    k = hidden // world
    partials = [(x[:, r * k:(r + 1) * k] @ w[r * k:(r + 1) * k]).float().to(torch.bfloat16) for r in range(world)]
    today = (sum(p.float() for p in partials[:4]) + sum(p.float() for p in partials[4:])).to(torch.bfloat16)
    once = ref.float().to(torch.bfloat16)
    err = lambda o: float((o.double() - ref).pow(2).mean().sqrt())
    assert err(once) <= err(today)
