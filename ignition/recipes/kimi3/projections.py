# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Kimi projections retain BF16 activations with checkpoint or online MXFP8 weights."""
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

    W11 env-to-flag fold: this is the ``--atomic-splitk`` plan setting
    (``planning.Settings.atomic_splitk``), read through ``live_settings()`` —
    the legacy ``IGNITION_KIMI_ATOMIC_SPLITK`` environment switch is refused at
    plan time, naming the flag that replaced it. The W10 union fold
    (``fold_splitk_funnel``) is removed and ``glue_union.py`` is parked (union
    ruling section 4: nothing ever executes the union and it is not on the path
    to the E2 bar).
    """
    from ignition.planning import live_settings

    return live_settings().atomic_splitk


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
    return projection
