# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kimi projections retain BF16 activations with checkpoint or online MXFP8 weights."""

import os

import torch

from ignition.components.linear import QuantizedProjection, UnquantizedProjection


def atomic_splitk():
    """W4 R10b: fold split-K partials into the BF16 output with atomics.

    The split-K schedule emits a separate reduce launch after every split-K GEMM
    (``WeightOnlySplitKReduce`` / cuBLAS ``splitKreduce``); the atomic BF16 variant
    accumulates partials straight into the output and drops those launches. The
    accumulation order changes the numerics (BF16 atomic rounding, unordered adds),
    so this rides the decode export with its calibration/fidelity gates. Default
    off keeps the FP32 split-K reduce.
    """
    return os.environ.get("IGNITION_KIMI_ATOMIC_SPLITK", "0") == "1"


def prepare_projection(source, weight, output, *, session, name, workspace):
    from ignition.components.weights import OutputPartitionedBF16Weight

    if isinstance(weight, OutputPartitionedBF16Weight):
        from ignition.components.partitioned_projection import PreparedOutputPartitionProjection

        projection = PreparedOutputPartitionProjection(source, weight, output, session=session, name=name, workspace=workspace)
    elif isinstance(weight, torch.Tensor):
        projection = UnquantizedProjection(source, weight, output, session=session, name=name)
    else:
        projection = QuantizedProjection(source, weight, output, session=session, name=name,
                                   activation_mode="a16", workspace=workspace,
                                   allow_bf16_atomic_reduction=True if atomic_splitk() else None)
    if atomic_splitk():
        # W10 R10b union integration: the atomic funnel outputs ride the E2 union
        # pattern (same process-local temporary, hoisted strides/offsets, one copy
        # per element) alongside the exchange and aggregation fields. Preparation
        # only — the union copies no bytes until pack/unpack runs, so the runtime
        # stays exactly the W4 funnel.
        from .e2_union import e2_union_enabled, fold_splitk_funnel

        if e2_union_enabled():
            projection.e2_funnel_union = fold_splitk_funnel(name, {name: output}, device=output.device)
    return projection
