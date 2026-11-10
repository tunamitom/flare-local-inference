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

        return PreparedOutputPartitionProjection(source, weight, output, session=session, name=name, workspace=workspace)
    if isinstance(weight, torch.Tensor):
        return UnquantizedProjection(source, weight, output, session=session, name=name)
    return QuantizedProjection(source, weight, output, session=session, name=name,
                               activation_mode="a16", workspace=workspace,
                               allow_bf16_atomic_reduction=True if atomic_splitk() else None)
