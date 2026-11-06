# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Kimi K3.1 Pro: gated sparse attention and sliding-window MLA DFlash2 drafting."""

from dataclasses import replace
from types import MappingProxyType

from ignition.recipes.declaration import Checkpoint, RankCosts, SamplingDefaults, SamplingRange
from ignition.recipes.kimi3.config import MLA_DFLASH2_REPOSITORY, MLA_DFLASH2_REVISION
from ignition.recipes.kimi3.recipe import RECIPE as KIMI3, prepare_variant


def prepare(snapshot, plan, **kwargs):
    from . import bootstrap

    return prepare_variant(snapshot, plan, bootstrap=bootstrap, draft_repository=MLA_DFLASH2_REPOSITORY,
                           draft_revision=MLA_DFLASH2_REVISION, **kwargs)


RECIPE = replace(KIMI3, name="kimi31", served_model_name="Kimi-K3.1-Pro", prepare=prepare,
    # https://huggingface.co/moonshotai/Kimi-K3.1-Pro#3-evaluation-results
    sampling_defaults=SamplingDefaults(temperature=1., top_p=.95, top_k=-1,
                                      ranges={"top_p": SamplingRange(.95, 1.)}),
    checkpoints=(Checkpoint(alias="kimi-k3.1-pro", repository="moonshotai/Kimi-K3.1-Pro",
        revision="6d0a0202956796543364d666ab19c6ad1a494ee1", architecture="KimiK31ForConditionalGeneration",
        stored_bytes=1_561_217_377_280, device_bytes=1_496_756_986_630, draft_bytes=5_569_631_720),),
    online_mxfp8=True, language_model_only=True,
    # Gather threads are tuned independently; the attention components declare LSE's 512-thread geometry.
    b12x_environment=MappingProxyType({**KIMI3.b12x_environment, "B12X_PCIE_DCP_THREADS": "0"}),
    # DFlash2 keeps its window rings in continuation rows and adds no page regions.
    ranks=MappingProxyType({8: RankCosts(state_row_bytes=65_761_536, page_bytes=2_174_976,
                                       graph_bytes=9 << 29, page_tokens=128 * 8),
                           16: RankCosts(state_row_bytes=35_450_112, page_bytes=2_174_976,
                                        graph_bytes=9 << 29, page_tokens=128 * 16)}),
    plain_ranks=MappingProxyType({8: RankCosts(state_row_bytes=58_361_088, page_bytes=2_174_976,
                                             graph_bytes=9 << 29, page_tokens=128 * 8),
                                 16: RankCosts(state_row_bytes=30_275_328, page_bytes=2_174_976,
                                              graph_bytes=9 << 29, page_tokens=128 * 16)}))
