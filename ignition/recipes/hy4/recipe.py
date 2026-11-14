# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Qualified checkpoint, serving range, and memory costs of the Tencent Hy4 recipe.

Byte figures are documented arithmetic, not measurements: they follow from the
checkpoint's published parameter census and the cache record sizes (656-byte
latent records, 132-byte index-key records). Measured per-rank costs replace
them once a full-shape export records real allocations.
"""

from types import MappingProxyType

from ignition.recipes.declaration import (
    Checkpoint,
    RankCosts,
    RankUsage,
    Recipe,
    SamplingDefaults,
    Speculation,
)

from .config import INDEX_RECORD_BYTES, LATENT_RECORD_BYTES, PAGE_TOKENS

# The checkpoint stores 779,930,197,312 BF16 and 30,795,421 FP32 parameters
# (1,559,983,576,308 payload bytes; 1,560,008,337,568 with safetensors headers,
# the repository's `usedStorage`). Routed experts pack to W4A16 NVFP4 at load:
# 4 bits per weight plus one FP8 block scale per 16 weights and one FP32 gain
# pair per expert shard (78 MoE blocks, layer 0's dense FFN and the shared
# experts stay BF16, `lm_head` becomes FP32).
STORED_BYTES = 1_560_008_337_568
DEVICE_BYTES = 477_928_801_524
# The MTP block: one packed 256-expert MoE plus its attention, indexer, iHC-free
# norms and the `eh_proj` embedding/hidden merge.
DRAFT_BYTES = 6_216_029_440

# One rank's state row: the four iHC streams, the hidden state `hnorm` consumes,
# and the consumed-token counter — 4 x 6144 BF16 of residual state. Speculative
# capacity multiplies rows, not their size.
_STATE_ROW_BYTES = 4 * 6144 * 2 + 6144 * 2 + 8
# Per-rank KV records of one shared page: 128 of the page's global positions,
# 656 B latent records on all 78 layers and 132 B index-key records on the 21
# indexer layers (the MTP block adds its own of each while drafting).
_PAGE_BYTES = PAGE_TOKENS * (78 * LATENT_RECORD_BYTES + 21 * INDEX_RECORD_BYTES)
_DRAFT_PAGE_BYTES = PAGE_TOKENS * (79 * LATENT_RECORD_BYTES + 22 * INDEX_RECORD_BYTES)
# Operator scratch, graphs, collectives and the CUDA context. Unmeasured.
_GRAPH_BYTES = 12 << 30


def _costs(world, *, speculating):
    return RankCosts(_STATE_ROW_BYTES, _DRAFT_PAGE_BYTES if speculating else _PAGE_BYTES,
                     _GRAPH_BYTES, page_tokens=PAGE_TOKENS * world)


def prepare(snapshot, plan, *, compile_forwards, kernel_cache, progress, retain_graphs=False,
            core_storage=None, ranks=None):
    """Prepare the plan's catalog on process-local GPUs 0 through N-1.

    With `ranks`, rank zero's multi-host rendezvous, each rank uses its own host's GPU 0.
    """
    from ignition.preparation.rank_group import prepare_group

    from .bootstrap import PreparationConfig, prepare_rank

    speculation = plan.speculation
    drafting = {} if speculation is None else {
        "speculative_widths": speculation.widths,
        "speculative_fixed_width": speculation.fixed_width,
        "speculative_token_capacity": plan.speculative_token_capacity,
    }
    config = PreparationConfig(
        mtp_only=speculation is not None and speculation.mtp_only,
        collective_tuning=plan.collective_tuning, collective_ladder=plan.collective_ladder,
        prompt_logprobs=plan.prompt_logprobs, compile_workers=plan.compile_workers,
        autotune_effort=plan.autotune_effort, tuning_replica=plan.tuning_replica,
        native_dense_activations=plan.native_dense_activations,
        lock_gpu_clocks=plan.lock_gpu_clocks and plan.autotune_effort != "off",
        devices=tuple(range(plan.world_size)) if ranks is None else (0,) * plan.world_size,
        hosts=None if ranks is None else ranks.hosts,
        context=plan.context, prefill_chunk_size=plan.prefill_chunk,
        max_active=plan.concurrency,
        snapshot_budget_bytes=plan.snapshots * plan.costs.state_row_bytes,
        headroom_bytes=plan.headroom_bytes + plan.reserve_bytes,
        kernel_cache=kernel_cache, compile_decoder=compile_forwards, retain_graphs=retain_graphs,
        mtp=speculation is not None, stream_overlap=plan.stream_overlap, **drafting)
    return prepare_group(snapshot, config, prepare_rank, progress=progress, core_storage=core_storage,
                         ranks=ranks, checkpoint=plan.checkpoint)


def usage(prepared):
    report = prepared.memory_report
    return tuple(RankUsage(rank["pages"], rank["free_device_bytes"]) for rank in report.get("ranks", (report,)))


RECIPE = Recipe(
    name="hy4", served_model_name="Hy4-Preview", prepare=prepare, usage=usage,
    # Recommended sampling: the preview card's generation notes
    # (https://huggingface.co/tencent/Hy4-preview): temperature 0.9, top_p 1.0,
    # top_k disabled (FLARE's -1), reasoning_effort `high` (or `no_think`) on
    # the `:opensource` frontend tags.
    sampling_defaults=SamplingDefaults(temperature=.9, top_p=1., top_k=-1),
    checkpoints=(Checkpoint(
        alias="hy4-preview", repository="tencent/Hy4-preview",
        revision="705d81ee51566a186d645b74c974d642ef2828fe", architecture="HYV4ForCausalLM",
        stored_bytes=STORED_BYTES,
        # W4A16 packed experts; see the constants above for the derivation.
        device_bytes=DEVICE_BYTES, draft_bytes=DRAFT_BYTES),),
    context=1048576, page_tokens=PAGE_TOKENS, concurrency=32, max_concurrency=64, prefill_chunk=4096,
    # Official recipe is MTP at three draft tokens (vLLM TP8, MTP@3).
    speculation=Speculation(draft_tokens=3, token_capacity=128), max_draft_tokens=3,
    verification_rows=0,
    # Shared experts and the dense layer-0 FFN overlap the routed experts.
    stream_overlap=True,
    language_model_only=True,
    ranks=MappingProxyType({world: _costs(world, speculating=True) for world in (8, 16)}),
    plain_ranks=MappingProxyType({world: _costs(world, speculating=False) for world in (8, 16)}),
)
