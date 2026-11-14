# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0
"""The Hy4 decoder body: rank construction on tiny-hy4 and the forward loop.

CPU only. The rank-construction dry run loads the reference package's
`tiny-hy4` checkpoint (6 layers, full/shared/shared/shared/full pattern,
`hc_mult` 4, `index_topk` 16, non-trivial iHC/router/norm/clamp init) through
the real loader surface and constructs the rank end to end; the forward loop
then runs prefill, decode steps and the MTP draft in torch — the exact
numerics the b12x operators are tuned against (`_ihc_torch_ref`, the eager
sparse-MLA sink contract). GPU kernel selection and graph capture wait for P5.
"""

import json
import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REF = Path(os.environ.get("HY4_REFERENCE_DIR", "/home/tom/hy4/reference"))
TINY = REF / "tiny-hy4"


@pytest.fixture(scope="module")
def tiny_available():
    if not (TINY / "config.json").is_file() or not (TINY / "model.safetensors").is_file():
        pytest.skip(f"tiny-hy4 reference checkpoint not present under {TINY}")


@pytest.fixture(scope="module")
def constructed(tiny_available):
    from ignition.recipes.hy4.bootstrap import PreparationConfig, construct_rank

    settings = PreparationConfig(devices=(0,), mtp=True, speculative_widths=(1, 2, 3))
    return construct_rank(TINY, settings, rank=0, pinned=False, mtp=True,
                          cache_rows=2, cache_capacity=32)


@pytest.fixture()
def fresh(constructed, tiny_available):
    """A decoder with cleared caches and row counters per test."""
    import copy

    from ignition.recipes.hy4.decoder import Hy4Cache, Hy4Decoder

    rank = copy.copy(constructed)
    rank.cache = Hy4Cache.allocate(constructed.model_config, rows=2, capacity=32,
                                   device=torch.device("cpu"), draft=True)
    rank.decoder = Hy4Decoder(constructed.weights, rank.cache)
    return rank


# --- rank construction -------------------------------------------------------

def test_rank_construction_dry_run_completes_end_to_end(constructed):
    from ignition.recipes.hy4.weights import kept_fp32

    rank = constructed
    config = rank.model_config
    assert (config.layers, config.hidden, config.experts) == (6, 256, 8)
    assert config.full_indexer_layers == (0, 1, 5) and len(config.shared_indexer_layers) == 3
    assert len(rank.weights.layers) == 6 and rank.weights.draft is not None
    assert rank.memory_report["shared_indexer_layers"] == 3

    # The 4-stream iHC state: every boundary carries the 8-row FP32 layout.
    streams = config.text["hc_mult"]
    for layer in rank.weights.layers:
        for boundary in (layer.attn_ihc, layer.ffn_ihc):
            assert boundary.weights.fn.shape == (2 * streams, streams * config.hidden)
            assert boundary.weights.fn.dtype == torch.float32
            assert boundary.weights.base.shape == (2 * streams,)
            assert boundary.weights.scale.shape == (2,)
    assert rank.weights.hc_head.weights.fn.shape == (streams, streams * config.hidden)

    # The do-not-quantize floor: iHC, sinks, indexer weights_proj/k_norm,
    # router correction bias and the head are FP32; projections stay BF16.
    for layer in rank.weights.layers:
        attn = layer.attention.weights
        assert attn.sink.dtype == torch.float32 and attn.sink.shape == (config.heads,)
        for tensor in (attn.q_a_proj, attn.q_b_proj, attn.kv_a_proj, attn.kv_b_proj,
                       attn.o_proj, attn.gate_proj):
            assert tensor.dtype == torch.bfloat16
        if attn.indexer is not None:
            assert attn.indexer.weights_proj.dtype == torch.float32
            assert attn.indexer.k_norm_weight.dtype == torch.float32
            assert attn.indexer.wq_b.dtype == torch.bfloat16
        assert layer.input_norm.dtype in (torch.bfloat16, torch.float32)
        assert kept_fp32("model.layers.0.hc_attn_layer.hc_pre.hc_fn")
    assert rank.weights.head.dtype == torch.float32
    assert rank.weights.layers[1].mlp.router.correction_bias.dtype == torch.float32
    assert rank.weights.layers[1].mlp.router.gate.dtype == torch.bfloat16

    # Construction consumed every checkpoint tensor exactly once.
    from ignition.recipes.hy4.weights import checkpoint_names

    expected = checkpoint_names(config, mtp=True)
    assert rank.weights.used_names == expected
    assert len(expected) == 188

    # Shared selection: the shared cells bind the owner's buffers verbatim.
    for layer in (2, 3, 4):
        assert rank.selections[layer].kind == "shared" and rank.selections[layer].owner_id == 1
        assert rank.index_cache.shares_selection(layer, 1)
    assert rank.index_cache.num_index_key_pages == 4  # 3 full target layers + the MTP cell
    assert rank.index_cache.selection_exchange_count == 4  # + the MTP block's own indexer


def test_state_row_layout_matches_the_rank_cost_contract(constructed):
    from ignition.recipes.hy4.decoder import StreamState

    state = StreamState.allocate(4, 256, torch.device("cpu"))
    assert state.row_bytes == 4 * 256 * 2 + 256 * 2 + 8
    rows = StreamState.begin(torch.randn(3, 256))
    assert rows.streams.shape == (3, 4, 256)
    # Each token's streams start at its embedding replicated over the 4 streams.
    assert torch.equal(rows.streams[:, 0], rows.streams[:, 3])


def test_shard_assignment_legality():
    import json

    from ignition.components.parallel import TensorParallelShard
    from ignition.recipes.hy4.config import ModelConfig
    from ignition.recipes.hy4.shard import Hy4Sharding

    source = json.loads((Path(os.environ.get("HY4_REF_DIR", "/home/tom/hy4/ref")) / "bf16_config.json").read_text()) \
        if (Path(os.environ.get("HY4_REF_DIR", "/home/tom/hy4/ref")) / "bf16_config.json").is_file() else None
    if source is None:
        pytest.skip("Hy4 reference metadata not present")
    config = ModelConfig.from_dict(source)
    tp8 = Hy4Sharding(config, TensorParallelShard(0, 8))
    assert tp8.heads_per_rank == 8 and not tp8.prefill_head_gather
    assert tp8.q_b_rows[1] - tp8.q_b_rows[0] == (64 // 8) * (192 + 64)
    tp16 = Hy4Sharding(config, TensorParallelShard(0, 16))
    assert tp16.heads_per_rank == 4 and tp16.prefill_head_gather
    with pytest.raises(ValueError, match="heads divisible by 8"):
        tp16.require_prefill_legal()
    tp8.require_prefill_legal()
    # Replication mirrors `recipe.py`'s `replicated_bytes` derivation.
    assert tp8.replicated("model.layers.3.self_attn.q_a_proj.weight")
    assert tp8.replicated("model.layers.3.self_attn.indexer.wk.weight")
    assert tp8.replicated("model.layers.3.mlp.gate.weight")
    assert not tp8.replicated("model.layers.3.self_attn.q_b_proj.weight")
    assert not tp8.replicated("model.layers.3.mlp.experts.gate_up_proj")
    assert not tp8.replicated("lm_head.weight")


# --- the forward loop --------------------------------------------------------

def test_prefill_runs_and_matches_stepwise_decode(fresh):
    from ignition.recipes.hy4.decoder import Hy4Cache, Hy4Decoder

    rank = fresh
    tokens = torch.tensor([[3, 7, 11, 5]])
    logits = rank.decoder.prefill(tokens)
    assert logits.shape == (1, 4, rank.model_config.vocabulary)
    assert torch.isfinite(logits).all()

    # The same tokens, appended one at a time through `decode_step` on a second
    # cache, produce the same logits: one layer loop serves both entries.
    stepwise_decoder = Hy4Decoder(rank.weights, Hy4Cache.allocate(
        rank.model_config, rows=2, capacity=32, device=torch.device("cpu"), draft=True))
    stepwise = stepwise_decoder.decode_step(tokens[:, :1])
    stepwise_decoder.complete([0], tokens[:, :1])
    assert torch.allclose(stepwise, logits[:, :1], atol=2e-2, rtol=1e-3)
    for position in range(1, tokens.shape[1]):
        step = stepwise_decoder.decode_step(tokens[:, position:position + 1])
        stepwise_decoder.complete([0], tokens[:, position:position + 1])
        assert torch.allclose(step, logits[:, position:position + 1], atol=2e-2, rtol=1e-3)


def test_decode_step_advances_one_token(fresh):
    with pytest.raises(ValueError, match="one token per row"):
        fresh.decoder.decode_step(torch.tensor([[1, 2]]))


def test_shared_selection_binds_the_owner_buffer_verbatim(fresh):
    rank = fresh
    decoder = rank.decoder
    decoder.prefill(torch.tensor([[3, 7, 11]]))
    owner, owner_counts = rank.selections[1].buffers()
    snapshot = owner.clone()
    # A shared layer's select runs no indexer and writes nothing: the group
    # owner's live buffer is what every shared layer consumes (the stale-copy
    # mutant must die here).
    layer = rank.weights.layers[2]
    selected, counts = layer.attention.select(
        torch.randn(3, 256).bfloat16(), torch.randn(3, 64).bfloat16(), torch.arange(3),
        torch.arange(1, 4), rank.cache.key_table(1), torch.zeros(3, dtype=torch.long))
    assert torch.equal(owner, snapshot)
    assert selected.data_ptr() == owner.data_ptr()
    assert counts.data_ptr() == owner_counts.data_ptr()
    assert rank.index_cache.shares_selection(2, 1) and rank.index_cache.shares_selection(4, 1)
    # Mutant check: giving the shared layer its own buffer breaks the contract.
    assert not rank.index_cache.shares_selection(1, 5)


def test_sink_folds_exactly_once_and_sink_off_mutates_output(fresh):
    rank = fresh
    from ignition.recipes.hy4.attention import SparseMLAProvenance

    attn = rank.weights.layers[0].attention
    rows, width = 2, 6
    q = torch.randn(rows, attn.heads, attn.kv_lora + attn.qk_rope)
    table = torch.randn(rank.cache.latents(0).shape[0], attn.kv_lora + attn.qk_rope)
    selected = torch.arange(width).reshape(1, width).repeat(rows, 1)
    counts = torch.tensor([width, width])
    out = attn.attend(q, table, selected, counts)
    assert out.provenance == SparseMLAProvenance(sink_folded=True, lse_base="base2")
    assert out.lse.shape == (rows, attn.heads)

    # The contract: out is the sinkless attention scaled by sigmoid(lse - sink)
    # and the LSE is logaddexp2 of the token LSE and the sink — the fold the
    # kernels and `PreparedSinkFold` implement, applied exactly once.
    ln2 = torch.log(torch.tensor(2.0))
    logits = torch.einsum("rhd,rwd->rhw", q.float(), table[selected].float()) * attn.scaling
    token_lse2 = torch.logsumexp(logits, dim=-1) / ln2
    probs = torch.softmax(logits, dim=-1)
    sinkless = torch.einsum("rhw,rwd->rhd", probs, table[selected][..., :attn.kv_lora].float())
    sink = attn.weights.sink.float().reshape(1, attn.heads)
    fold = torch.sigmoid((token_lse2 * ln2) - sink).unsqueeze(-1)
    expected_out = sinkless * fold
    expected_lse = torch.logaddexp(token_lse2 * ln2, sink) / ln2
    assert torch.allclose(out.lse, expected_lse, atol=1e-4)
    expanded = torch.einsum("rhd,hvd->rhv", expected_out, attn.expand_wuv.float())
    assert torch.allclose(out.output.float(), expanded.to(out.output.dtype).float(), atol=5e-2)

    # Sink-off mutant: zeroing the sinks changes the output (sinks are live).
    saved = attn.weights.sink.clone()
    attn.weights.sink.zero_()
    mutated = attn.attend(q, table, selected, counts)
    attn.weights.sink.copy_(saved)
    assert not torch.allclose(mutated.output.float(), out.output.float(), atol=1e-3)


def test_ihc_fused_boundary_equals_the_composed_reference(fresh):
    rank = fresh
    layer = rank.weights.layers[1]
    rows = 3
    streams = torch.randn(rows, 4, rank.model_config.hidden, dtype=torch.bfloat16)
    branch = torch.randn(rows, rank.model_config.hidden, dtype=torch.bfloat16)
    post, y = layer.attn_ihc.pre(streams)
    updated = layer.attn_ihc.post(branch, streams, post)
    # The fused boundary composes exactly post -> pre -> RMSNorm.
    fused_streams, fused_post, fused_y = layer.attn_ihc.post_pre(
        branch, streams, post, layer.ffn_ihc, norm_weight=layer.post_attention_norm)
    from b12x.norm.mhc import ihc

    ref_post, ref_y = ihc.ihc_pre_reference(updated, layer.ffn_ihc.weights.fn,
                                            layer.ffn_ihc.weights.scale,
                                            layer.ffn_ihc.weights.base,
                                            magnitude=layer.ffn_ihc.magnitude,
                                            hc_eps=layer.ffn_ihc.hc_eps,
                                            norm_eps=layer.ffn_ihc.norm_eps)
    from ignition.recipes.hy4.attention import rms_norm

    ref_y = rms_norm(ref_y, layer.post_attention_norm, rank.model_config.text["rms_norm_eps"])
    assert torch.equal(fused_streams, updated)
    assert torch.allclose(fused_post, ref_post, atol=0)
    assert torch.allclose(fused_y.float(), ref_y.float(), atol=1e-6)
    # The head merges the streams once, before `model.norm`.
    merged = rank.weights.hc_head.merge(streams)
    assert merged.shape == (rows, rank.model_config.hidden)


def test_ihc_pre_post_swap_mutant_changes_the_forward(fresh):
    rank = fresh
    import copy

    baseline = rank.decoder.prefill(torch.tensor([[3, 7, 11, 5]]))
    swapped = copy.copy(rank)
    from ignition.recipes.hy4.decoder import Hy4Cache, Hy4Decoder

    swapped.cache = Hy4Cache.allocate(rank.model_config, rows=2, capacity=32,
                                      device=torch.device("cpu"), draft=True)
    swapped.decoder = Hy4Decoder(rank.weights, swapped.cache)
    layer = rank.weights.layers[0]
    fn = layer.attn_ihc.weights.fn
    mutated = layer.attn_ihc.weights.fn.clone()
    mutated[:4], mutated[4:] = fn[4:].clone(), fn[:4].clone()
    layer.attn_ihc.weights.fn.copy_(mutated)
    try:
        poisoned = swapped.decoder.prefill(torch.tensor([[3, 7, 11, 5]]))
    finally:
        layer.attn_ihc.weights.fn.copy_(fn)
    assert not torch.allclose(baseline, poisoned, atol=1e-3)


def test_mtp_draft_runs_and_is_absent_without_speculation(constructed, tiny_available):
    from ignition.recipes.hy4.bootstrap import PreparationConfig, construct_rank
    from ignition.recipes.hy4.decoder import Hy4Cache, Hy4Decoder

    rank = constructed
    decoder = Hy4Decoder(rank.weights, Hy4Cache.allocate(
        rank.model_config, rows=2, capacity=32, device=torch.device("cpu"), draft=True))
    tokens = torch.tensor([[3, 7, 11, 5]])
    logits = decoder.prefill(tokens)
    decoder.complete([0], tokens)
    # prev_hidden is the target's pre-`model.norm` hidden: the state rows keep it.
    prev_hidden = decoder.weights.hc_head.merge(
        torch.randn(1, 4, rank.model_config.hidden, dtype=torch.bfloat16))
    draft = decoder.draft_step(torch.tensor([[9]]), prev_hidden)
    assert draft.shape == (1, rank.model_config.vocabulary) and torch.isfinite(draft).all()

    settings = PreparationConfig(devices=(0,), mtp=False)
    plain = construct_rank(TINY, settings, rank=0, pinned=False, mtp=False,
                           cache_rows=2, cache_capacity=32)
    assert plain.weights.draft is None
    plain_decoder = Hy4Decoder(plain.weights, Hy4Cache.allocate(
        plain.model_config, rows=2, capacity=32, device=torch.device("cpu"), draft=False))
    with pytest.raises(ValueError, match="no MTP draft"):
        plain_decoder.draft_step(torch.tensor([[9]]), prev_hidden)


def test_interleaved_rope_matches_b12x_and_rejects_split_half():
    from b12x.attention.dsa_indexer.rope import prepare_indexer_queries

    from ignition.recipes.hy4.attention import apply_interleaved_rope, rotate_half_interleaved

    rows, width, rope = 5, 128, 64
    q = torch.randn(rows, 3, width)
    positions = torch.arange(rows)
    mine = apply_interleaved_rope(q, positions, rope_dim=rope, theta=1.0e7)
    theirs = prepare_indexer_queries(q, positions, rope_dim=rope, theta=1.0e7)
    assert torch.allclose(mine, theirs, atol=1e-6)
    # Split-half (NeoX) rotation is the documented mutant: different output.
    x = q[..., width - rope:]
    split = torch.cat((-x[..., rope // 2:], x[..., :rope // 2]), dim=-1)
    assert not torch.allclose(rotate_half_interleaved(x), split)


def test_indexer_key_write_rotates_at_write_time(fresh):
    rank = fresh
    layer = rank.weights.layers[0].attention
    x = torch.randn(4, rank.model_config.hidden).bfloat16()
    positions = torch.arange(4)
    keys = layer.prepare_index_keys(x, positions)
    assert keys.shape == (4, layer.weights.indexer.head_dim)
    # Keys are normed (LayerNorm over the full head) then rotated in the LAST
    # rope block: positions change the pe block only.
    shifted = layer.prepare_index_keys(x, positions + 1)
    rope = rank.model_config.text["qk_rope_head_dim"]
    assert torch.allclose(keys[:, :-rope], shifted[:, :-rope], atol=1e-6)
    assert not torch.allclose(keys[:, -rope:], shifted[:, -rope:], atol=1e-4)
