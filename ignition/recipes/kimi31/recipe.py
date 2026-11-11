# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Kimi K3.1 Pro: gated sparse attention and full-context MLA drafting."""

from dataclasses import replace
from types import MappingProxyType

from ignition.recipes.declaration import Checkpoint, RankCosts, SamplingDefaults, SamplingRange, SpeculativeMethod
from ignition.recipes.kimi3.config import LATENT_DSPARK_REPOSITORY, LATENT_DSPARK_REVISION
from ignition.recipes.kimi3.recipe import RECIPE as KIMI3, prepare_variant


def prepare(snapshot, plan, **kwargs):
    from . import bootstrap

    if plan.speculation is not None and plan.speculation.method == "dflash2":
        from .dflash_config import REPOSITORY, REVISION
        return prepare_variant(snapshot, plan, bootstrap=bootstrap, draft_repository=REPOSITORY,
            draft_revision=REVISION, draft_directory="dflash2", **kwargs)
    if plan.speculation is not None and plan.speculation.draft_head_quantization is not None:
        raise ValueError("Kimi draft head conversion requires --speculative-method dflash2")
    return prepare_variant(snapshot, plan, bootstrap=bootstrap, draft_repository=LATENT_DSPARK_REPOSITORY,
                           draft_revision=LATENT_DSPARK_REVISION, **kwargs)


RECIPE = replace(KIMI3, name="kimi31", served_model_name="Kimi-K3.1-Pro", prepare=prepare,
    # 1,048,576 is the engine and checkpoint position cap (kimi3/bootstrap.py
    # validates context in [page_tokens, 1048576]; max_position_embeddings is
    # 1048576 in the target and dflash2 configs). The value is baked into the
    # package as max_sequence_tokens and the page-table width; requests are
    # admitted up to min(prepared_sequence_tokens, pages * page_tokens).
    context=1048576,
    # https://huggingface.co/moonshotai/Kimi-K3.1-Pro#3-evaluation-results
    sampling_defaults=SamplingDefaults(temperature=1., top_p=.95, top_k=-1,
                                      ranges={"top_p": SamplingRange(.95, 1.)}),
    checkpoints=(Checkpoint(alias="kimi-k3.1-pro", repository="moonshotai/Kimi-K3.1-Pro",
        revision="6d0a0202956796543364d666ab19c6ad1a494ee1", architecture="KimiK31ForConditionalGeneration",
        stored_bytes=1_561_217_377_280, device_bytes=1_496_756_986_630, draft_bytes=5_373_504_512),),
    online_mxfp8=True, language_model_only=True, draft_head_quantizations=("mxfp8", "nvfp4"),
    speculation=replace(KIMI3.speculation, method="mtp"),
    speculative_methods=MappingProxyType({"dflash2": SpeculativeMethod(MappingProxyType({
        8: RankCosts(state_row_bytes=65_761_536, page_bytes=2_174_976, graph_bytes=9 << 29, page_tokens=128*8),
        16: RankCosts(state_row_bytes=35_450_112, page_bytes=2_174_976, graph_bytes=9 << 29, page_tokens=128*16),
    }), draft_bytes=5_569_622_016)}),
    # Gather threads are tuned independently; the attention components declare LSE's 512-thread geometry.
    b12x_environment=MappingProxyType({**KIMI3.b12x_environment, "B12X_PCIE_DCP_THREADS": "0"}),
    ranks=MappingProxyType({8: RankCosts(state_row_bytes=63_181_056, page_bytes=2_543_616,
                                       graph_bytes=9 << 29, page_tokens=128 * 8),
                           16: RankCosts(state_row_bytes=32_869_632, page_bytes=2_543_616,
                                        graph_bytes=9 << 29, page_tokens=128 * 16)}),
    plain_ranks=MappingProxyType({8: RankCosts(state_row_bytes=58_361_088, page_bytes=2_174_976,
                                             graph_bytes=9 << 29, page_tokens=128 * 8),
                                 16: RankCosts(state_row_bytes=30_275_328, page_bytes=2_174_976,
                                              graph_bytes=9 << 29, page_tokens=128 * 16)}))
