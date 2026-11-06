# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint-backed MLA DFlash2 proposal blocks against a dense FP32 reference."""

from dataclasses import replace
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from ignition.components.paged_batch import Metadata
from ignition.components.parallel import TensorParallelShard
from ignition.preparation.compilation import compile_forward
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession
from ignition.recipes.kimi3.config import DSparkConfig, ModelConfig
from ignition.recipes.kimi3.latent_dspark import YarnRotary
from ignition.recipes.kimi3.mla_dflash2 import DraftScratch, DraftWeights, PreparedContext, PreparedDraft

pytestmark = pytest.mark.skipif(os.getenv("FLARE_RUN_CUDA_TESTS") != "1", reason="requires an assigned GPU")
FP8 = torch.float8_e4m3fn


def fixture():
    config = ModelConfig.from_dict(json.loads((Path(__file__).parents[1] / "fixtures/kimi31/config.json").read_text()))
    return config


@torch.no_grad()
def test_block_matches_dense_reference():
    checkpoint = os.getenv("FLARE_KIMI31_DFLASH2_CHECKPOINT")
    if checkpoint is None:
        pytest.skip("set FLARE_KIMI31_DFLASH2_CHECKPOINT to the DFlash2 checkpoint")
    config = fixture()
    config = replace(config, draft=DSparkConfig.load(checkpoint, config))
    device, shard = torch.device("cuda:0"), TensorParallelShard(0, 1)
    d, hidden = config.draft, config.hidden
    window, block = d.window, 8
    torch.manual_seed(3107)
    embedding = torch.randn((config.vocabulary, hidden), dtype=torch.bfloat16, device=device) * .05
    head = torch.randn((config.vocabulary, hidden), dtype=torch.bfloat16, device=device) * .05
    draft = DraftWeights.load(checkpoint, config, device, shard, target_embedding=embedding, target_head=None)
    weights = SimpleNamespace(config=config, shard=shard, draft=draft, head=head)
    rows = 2
    # Rings live in continuation rows: their pages are strided like reserved state regions.
    ring_bytes = window * 576
    backing = torch.zeros((rows + 1, d.layers * ring_bytes + 4096), dtype=torch.uint8, device=device)
    rings = [backing[:, i * ring_bytes:(i + 1) * ring_bytes].view(rows + 1, window, 576) for i in range(d.layers)]
    pool = SimpleNamespace(draft_ring=lambda index: rings[index])
    # The first row's history is partial; the second's has wrapped the ring. State rows are not request rows.
    states = torch.tensor([2, 0], device=device)
    histories = (700, window + 437)
    chunk = window
    taps = len(d.auxiliary_layers)
    context_metadata = Metadata.allocate(chunk, 1, 1, device)
    context_metadata.page_table.zero_()
    context_tokens = torch.zeros((1, chunk), dtype=torch.int64, device=device)
    feature = torch.randn((chunk, taps * hidden), dtype=torch.bfloat16, device=device) * .2
    scratch = DraftScratch.allocate(config, shard, rows * block, block, device)
    metadata = Metadata.allocate(rows * block, rows, 1, device, samples=rows * block)
    tokens = torch.full((rows, block), d.noise_token, dtype=torch.int64, device=device)
    tokens[:, 0] = torch.tensor([17, 2300], device=device)
    live = torch.tensor([rows], device=device)
    eps = d.text["rms_norm_eps"]

    def norm(x, w):
        value = x.float()
        normalized = (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps)).to(w.dtype)
        return (normalized * w).to(x.dtype)

    with RecordedPreparationSession(device=device, checkpoint=str(checkpoint), workload_schema="kimi-dflash2",
            numerical_policy="checkpoint-bf16", compile_workers=16) as session:
        arena = ScratchArena(device)
        session.register_scratch(arena)
        context = PreparedContext(weights, feature, context_metadata, pool, session=session, name="context",
            workspace=arena, runtime=None, coordinator=None, buffers={})
        prepared = PreparedDraft(weights, scratch, metadata, pool, block=block, runtime=None, vocabulary_runtime=None,
            embedding_runtime=None, coordinator=None, session=session, name="draft", workspace=arena)
        session.tune(show_progress=False)
        counts = context_metadata.lengths
        context_forward = compile_forward(lambda: context(feature, counts))
        history_features = [torch.randn((n, taps * hidden), dtype=torch.bfloat16, device=device) * .2 for n in histories]
        for row, count in enumerate(histories):
            for first in range(0, count, chunk):
                size = min(chunk, count - first)
                feature[:size].copy_(history_features[row][first:first + size])
                context_metadata.set(torch.tensor([first], device=device), torch.tensor([size], dtype=torch.int32, device=device),
                    states[row:row + 1], context_metadata.page_table.clone(), context_tokens, torch.tensor([1], device=device))
                context_forward()
        forward = compile_forward(prepared)
        metadata.set(torch.tensor(histories, device=device), torch.tensor([block, block], dtype=torch.int32, device=device),
                     states, metadata.page_table.clone(), tokens, live)
        forward()
        torch.cuda.synchronize()
        rotary = YarnRotary(config, device)

        def rotate(x, positions):
            angle = positions.float()[:, None] * rotary.inverse[None]
            shape = (positions.numel(),) + (1,) * (x.ndim - 2) + (32,)
            cos, sin = angle.cos().view(shape) * rotary.scale, angle.sin().view(shape) * rotary.scale
            pairs = x[..., -64:].float().reshape(*x.shape[:-1], 32, 2)
            tail = torch.stack((pairs[..., 0] * cos - pairs[..., 1] * sin,
                                pairs[..., 1] * cos + pairs[..., 0] * sin), -1).flatten(-2).bfloat16()
            return torch.cat((x[..., :-64], tail), -1)

        def convolve(x, base, delta):
            out = torch.zeros_like(x.float())
            groups = x.shape[1] // 16
            for tap in range(base.shape[0]):
                coefficient = base[tap].float() + delta[:, tap].float().repeat_interleave(16, 1)
                shifted = x.float().roll(tap, 0)
                mask = (torch.arange(x.shape[0], device=device) % block >= tap)[:, None]
                out = out + coefficient * shifted * mask
            assert groups * 16 == x.shape[1]
            return out.bfloat16()

        # Dense reference, one request at a time.
        expected = []
        for row, count in enumerate(histories):
            positions = torch.arange(count, count + block, device=device)
            x = embedding[tokens[row]].clone()
            feedback = norm(F.linear(history_features[row], draft.projection), draft.hidden_norm)
            context_kv = F.linear(feedback, draft.context_kv).view(count, d.layers, 576)
            for index, layer in enumerate(draft.layers):
                w = layer.attention
                n = norm(x, layer.input_norm)
                c = F.linear(n, layer.attention_conv["projection"]).view(block, 2, 2, -1)
                source = convolve(n, layer.attention_conv["base"][0], c[:, 0])
                q_residual, kv_residual = F.linear(source, w["latent_gate"][:2112]).split((1536, 576), -1)
                query = rotate(F.linear(norm(q_residual, w["query_norm"]), w["q_b"]).view(block, 96, 192), positions)
                absorbed = torch.cat((torch.bmm(query[:, :, :128].transpose(0, 1), w["key"]).transpose(0, 1),
                                      query[:, :, 128:]), -1)
                kept = torch.arange(max(0, count - window), count, device=device)
                prior = context_kv[kept, index]
                prior = rotate(torch.cat((norm(prior[:, :512], w["kv_norm"]), prior[:, 512:]), -1), kept)
                added = rotate(torch.cat((norm(kv_residual[:, :512], w["kv_norm"]), kv_residual[:, 512:]), -1), positions)
                store = torch.cat((prior, added)).float().clamp(-448, 448).to(FP8).float()
                keys = torch.cat((kept, positions))
                q = absorbed.float().clamp(-448, 448).to(FP8).float()
                scores = torch.einsum("rhd,kd->rhk", q, store) * 192 ** -.5
                visible = (keys[None] <= positions[:, None]) & (keys[None] > positions[:, None] - window)
                scores.masked_fill_(~visible[:, None], -torch.inf)
                latent = torch.einsum("rhk,kd->rhd", scores.softmax(-1), store[:, :512]).bfloat16()
                expanded = torch.bmm(latent.transpose(0, 1), w["value"]).transpose(0, 1).flatten(1)
                x = x + convolve(F.linear(expanded, w["output"]), layer.attention_conv["base"][1], c[:, 1])
                n = norm(x, layer.post_attention_norm)
                c = F.linear(n, layer.mlp_conv["projection"]).view(block, 2, 2, -1)
                source = convolve(n, layer.mlp_conv["base"][0], c[:, 0])
                gate = F.linear(source, layer.mlp.gate_up[:14336])
                up = F.linear(source, layer.mlp.gate_up[14336:])
                out = F.linear((F.silu(gate.float()) * up.float()).bfloat16(), layer.mlp.down)
                x = x + convolve(out, layer.mlp_conv["base"][1], c[:, 1])
            expected.append(norm(x, draft.norm))
        expected = torch.cat(expected)
        actual = scratch.hidden.float()
        error = (actual - expected.float()).square().mean().sqrt() / expected.float().square().mean().sqrt()
        # Five BF16 layers round about 3.5%; a narrowed window, noncausal blocks or a missing history all exceed 7%.
        assert error < .05, float(error)
        logits = F.linear(expected.float(), head.float())
        logit_error = (scratch.logits - logits).square().mean().sqrt() / logits.square().mean().sqrt()
        assert logit_error < .05, float(logit_error)
        # Rings hold committed target records only: a proposal block must not write them.
        before = [ring.clone() for ring in rings]
        forward()
        torch.cuda.synchronize()
        for ring, saved in zip(rings, before, strict=True):
            assert torch.equal(ring, saved)
