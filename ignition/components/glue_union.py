# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

# ─────────────────────────────────────────────────────────────────────────────
# PARKED (W11, radiance-mechanics review section 4 union ruling): this union
# pattern is NOT on the path to the E2 bar and nothing executes it — ``pack``
# and ``unpack`` have no call site in any graph, and ``fold_splitk_funnel`` has
# been removed from ``ignition/recipes/kimi3/projections.py``. Its 20 CPU tests
# never covered the union's real hazards (the review lists six: no graph
# executes it, it would add copies if executed, Python counters stop at
# capture, mixed dtypes through one uint8 staging region, capture-time races,
# aliasing across rounds and ranks). The file is kept as history: do not wire
# it back without a new consult. What survived of W10 is the
# ``out=``/``count_out=`` compaction in ``ignition/components/distributed_topk.py``.
# ─────────────────────────────────────────────────────────────────────────────

"""Union pattern for decode glue copies: hoisted strides/offsets, one copy each.

The decode glue moves exchange and aggregation fields through per-call copies
(query quantization, candidate score/id packing, selected-record compaction). The
union pattern gives those fields one prepared pass over one process-local
temporary:

* ``GlueUnionPlan`` is built once at preparation. It validates the stage geometry
  and computes every stage's strides and byte offsets **once** — the union loop
  never recomputes them (copy-count elimination, W10: "hoist strides/offsets out
  of the union loop").
* ``PreparedGlueUnion`` holds one staging region carved from the process-local
  temporary and runs the union loop: exactly one copy per element per direction,
  counted in ``copies``/``copied_elements`` so tests can assert the elimination
  (fields that used to be written per row or twice are written once).
* R10b's split-K funnel folds into the same union (``fold_splitk_funnel`` in
  ``ignition/recipes/kimi3/e2_union.py``): the atomic funnel outputs ride the
  same temporary and the same hoisted metadata as the exchange and aggregation
  fields, which is what "R10b integrated into the union pattern" means here. The
  funnel selection itself stays in ``ignition/recipes/kimi3/projections.py``
  behind ``IGNITION_KIMI_ATOMIC_SPLITK``.

Default-off wiring everywhere: without ``IGNITION_KIMI_E2_UNION=1`` no call site
changes. See ``scripts/benchmark_e2_allocation.py`` for the allocation-only
microbenchmark and the +/-10% radiance threshold (the collective path needs the
radiance-mechanics review before it can even be benched).

Numerics: the union copies bytes; it never changes reduction order or precision,
so the united path must be bit-identical to the un-united copies (R9 threat T3).
"""

from dataclasses import dataclass

import torch

__all__ = ["GlueStage", "GlueUnionPlan", "PreparedGlueUnion"]

SUPPORTED_DTYPES = (torch.uint8, torch.int32, torch.int64, torch.float16,
                    torch.bfloat16, torch.float32, torch.float8_e4m3fn)


@dataclass(frozen=True)
class GlueStage:
    """One glue field in the union: ``rows`` x ``columns`` elements of ``dtype``.

    ``offset_bytes`` is the field's byte offset inside the union staging region;
    the plan hoists it into the loop table once and checks the stages do not
    overlap.
    """

    name: str
    rows: int
    columns: int
    dtype: torch.dtype
    offset_bytes: int


@dataclass(frozen=True)
class _Hoisted:
    """Precomputed per-stage geometry consumed by the union loop (no recompute)."""

    name: str
    rows: int
    columns: int
    itemsize: int
    offset_bytes: int
    row_bytes: int


class GlueUnionPlan:
    """Hoisted stride/offset table for the decode glue union.

    The plan is pure metadata: building it touches no storage, and it is shared by
    every round that replays the union. ``recomputed_strides`` stays 0 — the
    reference path ``unhoisted_copy`` (kept for the tests) increments it to show
    what the union loop no longer does.
    """

    def __init__(self, stages):
        if not stages:
            raise ValueError("glue union plan requires at least one stage")
        self.recomputed_strides = 0
        hoisted = []
        seen = set()
        for stage in stages:
            if stage.name in seen:
                raise ValueError(f"duplicate glue union stage {stage.name!r}")
            seen.add(stage.name)
            if stage.rows <= 0 or stage.columns <= 0:
                raise ValueError(f"glue union stage {stage.name!r} needs a positive extent")
            if stage.dtype not in SUPPORTED_DTYPES:
                raise ValueError(f"glue union stage {stage.name!r} has unsupported dtype {stage.dtype}")
            if stage.offset_bytes < 0 or stage.offset_bytes % stage.dtype.itemsize:
                raise ValueError(f"glue union stage {stage.name!r} offset must be a nonnegative "
                                 f"multiple of {stage.dtype.itemsize}")
            hoisted.append(_Hoisted(stage.name, stage.rows, stage.columns, stage.dtype.itemsize,
                                    offset_bytes=stage.offset_bytes,
                                    row_bytes=stage.columns * stage.dtype.itemsize))
        spans = sorted((h.offset_bytes, h.offset_bytes + h.rows * h.row_bytes, h.name) for h in hoisted)
        for (_, previous_end, previous_name), (begin, _, name) in zip(spans, spans[1:]):
            if begin < previous_end:
                raise ValueError(f"glue union stages {previous_name!r} and {name!r} overlap")
        self.stages = tuple(stages)
        self._hoisted = tuple(hoisted)

    @property
    def elements(self):
        return sum(stage.rows * stage.columns for stage in self.stages)

    @property
    def nbytes(self):
        """Staging bytes the union needs from the process-local temporary."""
        return max(hoisted.offset_bytes + hoisted.rows * hoisted.row_bytes for hoisted in self._hoisted)

    def hoisted(self):
        """The precomputed geometry; the union loop consumes this and nothing else."""
        return self._hoisted

    def unhoisted_copy(self, sources, staging):
        """Reference path kept for the tests: per-row copies with recomputed strides.

        This is the anti-pattern the union loop replaced: every row copy
        re-derives the row stride and byte offset, and the copy count grows with
        the row count instead of staying at one per element. No call site uses it.
        """
        staging_bytes = staging.view(torch.uint8).view(-1)
        for stage, hoisted in zip(self.stages, self._hoisted):
            source = sources[stage.name]
            for row in range(stage.rows):
                self.recomputed_strides += len(source.stride())  # recomputed per row on purpose
                row_bytes = source.stride(0) * source.element_size()
                begin = hoisted.offset_bytes + row * row_bytes
                staging_bytes[begin:begin + row_bytes].copy_(
                    source.reshape(-1)[row * stage.columns:(row + 1) * stage.columns].view(torch.uint8))


class PreparedGlueUnion:
    """The union loop over one process-local temporary.

    ``pack`` copies each source field into the staging region exactly once (one
    copy per element, counted); ``unpack`` is the symmetric single copy out. The
    staging region is carved at preparation and never re-carved per round (R9
    threat T4), and the union is only shared across sequential stages of one
    replayed step (T5).
    """

    def __init__(self, plan, temporary, *, name):
        self.plan, self.name = plan, name
        self.copies = 0
        self.copied_elements = 0
        self.staging = temporary.reserve(f"{name}.union", (plan.nbytes,), torch.uint8)

    def pack(self, sources):
        """One copy per element of every stage from ``sources`` into staging."""
        for stage, hoisted in zip(self.plan.stages, self.plan.hoisted()):
            source = sources[stage.name]
            if source.numel() != stage.rows * stage.columns or source.dtype != stage.dtype:
                raise ValueError(f"glue union stage {stage.name!r} got {source.shape} {source.dtype}, "
                                 f"expected {(stage.rows, stage.columns)} {stage.dtype}")
            self._copy(stage, hoisted, source, into_staging=True)
        return self

    def unpack(self, destinations):
        """One copy per element of every stage out of staging."""
        for stage, hoisted in zip(self.plan.stages, self.plan.hoisted()):
            destination = destinations[stage.name]
            if destination.numel() != stage.rows * stage.columns or destination.dtype != stage.dtype:
                raise ValueError(f"glue union stage {stage.name!r} got {destination.shape} {destination.dtype}, "
                                 f"expected {(stage.rows, stage.columns)} {stage.dtype}")
            self._copy(stage, hoisted, destination, into_staging=False)
        return self

    def _copy(self, stage, hoisted, tensor, *, into_staging):
        # Strides and byte offsets come from the hoisted table; nothing is
        # recomputed here (copy-count elimination: one copy per element, each
        # direction). ``field`` is a byte view aliasing ``tensor``.
        if not tensor.is_contiguous():
            raise ValueError(f"glue union stage {stage.name!r} requires contiguous storage")
        staging_bytes = self.staging.view(-1)[hoisted.offset_bytes:hoisted.offset_bytes + hoisted.rows * hoisted.row_bytes]
        field = tensor.view(torch.uint8).view(-1)
        if into_staging:
            staging_bytes.copy_(field)
        else:
            field.copy_(staging_bytes)
        self.copies += 1
        self.copied_elements += stage.rows * stage.columns
