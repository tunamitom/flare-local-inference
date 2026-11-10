# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""W10 E2 prototypes for the kimi3 decode glue: R9 temporary + union + R10b fold.

Entry points for the E2 prototype queue (J-3 order: R9 first, R10b integrated on
the way; R7/R4 are shapes only):

* R9 — process-local temporary for the exchange + aggregation staging
  (``ignition/components/process_local_temporary.py``). Variant order is fixed by
  J-3: R9d (pinned/upper-buffer pool) first, then R9a (single multi-round
  reservation), then R9b (two-stage) as fallbacks.
* R10b — the atomic split-K funnel (already in ``projections.atomic_splitk``,
  W4) is folded into the same union pattern here: ``fold_splitk_funnel`` registers
  the funnel outputs as union stages with hoisted strides/offsets, and the union
  loop carries copy-count elimination toward that pattern.
* Killed levers stay dead: the R6 exchange merge and FP8 partials (J-3 section 2,
  FABLE-PERF-LEVERS errata) must never be enabled by this module, and
  ``killed_levers_engaged`` fails loudly if a killed switch is ever set here.

Flags (experiment switches, read at prepare time, default off — same standing
pattern as the W4 ``IGNITION_KIMI_*`` switches; folding them into recipe flags is
the known follow-up before any export, J-1 risk 1):

* ``IGNITION_KIMI_E2_UNION=1`` — enable the union pattern wiring.
* ``IGNITION_KIMI_E2_R9_VARIANT=r9d|r9a|r9b`` — allocation variant (default r9d).

The allocation-only microbenchmark that prices these variants (NOT run in this
workstream: no GPU in the window) is ``scripts/benchmark_e2_allocation.py``; it
documents the +/-10% radiance threshold and refuses to touch the collective
("radiance") path without the radiance-mechanics approval reference.
"""

import os

import torch

from ignition.components.glue_union import GlueStage, GlueUnionPlan, PreparedGlueUnion
from ignition.components.process_local_temporary import (
    GlueScratch,
    TemporaryVariant,
    process_local_temporary,
)

__all__ = [
    "e2_union_enabled",
    "e2_r9_variant",
    "killed_levers_engaged",
    "decode_temporary",
    "exchange_aggregation_scratch",
    "fold_splitk_funnel",
]

# J-3: the R6 exchange merge and FP8 partials are KILLED (measured at the model
# floor / a precision class that already failed recall). Never re-add, never
# enable: any of these set to "1" is a mistake and raises at prepare time.
KILLED_SWITCHES = (
    "IGNITION_KIMI_DSA_EXCHANGE_MERGE",  # R6 merge: modeled gain zero (J-3 section 2)
    "IGNITION_KIMI_DSA_LSE_FP8",         # FP8 partials: precision class failed recall
)

_TEMPORARY_BYTES = 64 << 20  # upper bound for the decode glue temporary (T10 admission)


def e2_union_enabled():
    """Master switch for the E2 union pattern (default off: wiring unchanged)."""
    return os.environ.get("IGNITION_KIMI_E2_UNION", "0") == "1"


def e2_r9_variant():
    """R9 allocation variant, J-3 order: r9d first, r9a/r9b fallbacks."""
    return TemporaryVariant.check(os.environ.get("IGNITION_KIMI_E2_R9_VARIANT", TemporaryVariant.POOL))


def killed_levers_engaged():
    """Names of killed lever switches that are set; must always be empty (J-3)."""
    return [name for name in KILLED_SWITCHES if os.environ.get(name, "0") == "1"]


def _guard():
    engaged = killed_levers_engaged()
    if engaged:
        raise RuntimeError(f"killed lever switches must never be enabled (J-3): {engaged}")


def decode_temporary(device):
    """The process-local temporary for the kimi3 decode glue exchange + aggregation."""
    _guard()
    return process_local_temporary("kimi3.decode_glue", device,
                                   variant=e2_r9_variant(), max_bytes=_TEMPORARY_BYTES)


def exchange_aggregation_scratch(device):
    """GlueScratch over the decode temporary (exchange packing + aggregation views)."""
    return GlueScratch(decode_temporary(device))


def fold_splitk_funnel(name, outputs, *, device):
    """Register R10b atomic-split-K funnel outputs in the decode glue union.

    ``outputs`` is a mapping of field name to contiguous tensor (the funnel's
    per-layer outputs: KDA input, MLA input/q_b, latent down/up). Each output
    becomes one union stage whose strides/offsets are hoisted once at preparation,
    sharing the process-local temporary with the exchange and aggregation fields —
    R10b riding the same union pattern. The funnel itself is selected by
    ``projections.atomic_splitk()``; this hook only unions its outputs and copies
    no bytes until ``PreparedGlueUnion.pack`` runs.

    Returns the prepared union, ready to pack/unpack around the funnel.
    """
    _guard()
    if not e2_union_enabled():
        raise RuntimeError("fold_splitk_funnel requires IGNITION_KIMI_E2_UNION=1")
    if not outputs:
        raise ValueError("fold_splitk_funnel requires at least one funnel output")
    stages = []
    offset = 0
    for field, tensor in outputs.items():
        if not tensor.is_contiguous():
            raise ValueError(f"funnel output {field!r} must be contiguous")
        if tensor.ndim == 2:
            rows, columns = tensor.shape
        else:
            rows, columns = tensor.numel(), 1
        stages.append(GlueStage(field, rows, columns, tensor.dtype, offset))
        offset += rows * columns * tensor.dtype.itemsize
    plan = GlueUnionPlan(stages)
    return PreparedGlueUnion(plan, decode_temporary(device), name=name)
