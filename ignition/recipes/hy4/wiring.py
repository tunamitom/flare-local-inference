# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0
"""W7 bake-in wiring: call shapes, graph families, stream schedule, plan book.

The bake-in lists (consult7 §5 for decode, consult8 §5 for prefill) are
decisions the implementation carries, not tunables: the call shape of a
forward decides which wiring it runs, and the wiring is a property of the
GRAPH FAMILY (consult8 §6: "decode-optimal overlap hurts prefill... Make
overlap a property of the graph family"):

- **decode graphs** (one token per row — the one-row decode shape and the
  distinct verification-rows shape): the selection path runs on a SECOND
  stream overlapping the projections and the head gather (consult7 §5), the
  query absorb is owner-side (each rank absorbs only its own head shard's
  `kv_b_proj` rows — the decode wiring reads no extra weights), the shared
  expert overlaps the routed experts, and the call is CUDA-graph friendly
  (fixed shape, prepared plans, caller-owned outputs — no allocation and no
  JIT at run time);
- **prefill graphs** (chunked token rows): the query gather moves UNABSORBED
  queries and the absorb + expand run locally (the prefill `kv_b` wiring),
  no two wire-heavy exchanges overlap (the selection exchange and the query
  gather serialize; compute may run under ring all-reduce transfers), and
  chunking is deterministic and length-only.

Every rule here is observable: `StreamSchedule.placement` names the stream
each operation runs on per family, and the decoder body records its calls in
`Hy4Decoder.trace`, so the wiring gates (and their mutants — wrong-stream
overlap, absorbed-query prefill gather, chunk-position nondeterminism) read
behavior, not intent.
"""

from dataclasses import dataclass, field

__all__ = [
    "CallShape",
    "DECODE_ONE_ROW",
    "verify_shape",
    "prefill_shape",
    "GraphFamily",
    "StreamSchedule",
    "KernelPlanBook",
    "chunk_plan",
    "PREFILL_GRAPH_SIZES",
    "DEFAULT_LOCAL_OFFER",
    "SHORT_CONTEXT_TOKENS",
]

# The short-context bypass bound (consult7 §5: "the short-context bypass of
# indexer and exchange"; consult8 §5 repeats it). At `context <= 2048` the
# global top-2,048 selection is every visible key, so the indexer scoring and
# the selection exchange compute nothing new and both are skipped.
SHORT_CONTEXT_TOKENS = 2048

# The local candidate offer of the scores-only selection exchange (consult8
# §5: "scores-only selection exchange with the local top-512 offer and its
# guard"). The W6 gate-b exactness test overrides the offer to the full
# selection width and pins 2,048-selection equality instead.
DEFAULT_LOCAL_OFFER = 512

# The prefill graph catalogue (consult8 §5: "the prefill graph catalogue
# (128 … 4,096 rows) declared, so the gates can cover it"). Each size is one
# captured prefill graph; every size gets a content gate.
PREFILL_GRAPH_SIZES = (128, 256, 512, 1024, 2048, 4096)


class GraphFamily:
    """The two graph families whose wirings differ (consult8 §6)."""

    DECODE = "decode"
    PREFILL = "prefill"


@dataclass(frozen=True)
class CallShape:
    """One forward call's fixed shape and the wiring its graph family runs.

    `name` distinguishes the two decode-family shapes that must never share a
    captured graph (consult7 §5: "a one-row decode graph distinct from the
    verification graph"): `"decode"` is the one-row serving step,
    `"verify"` the verification-rows shape (MTP verification batches), and
    `"prefill"` a catalogued prefill chunk. `tokens` is the width of the call
    (one token per row for decode-family shapes).
    """

    name: str
    rows: int
    tokens: int

    def __post_init__(self):
        if self.name not in ("decode", "verify", "prefill"):
            raise ValueError(f"unknown call shape name {self.name!r}")
        if type(self.rows) is not int or self.rows < 1:
            raise ValueError(f"call shape rows must be a positive int, got {self.rows!r}")
        if type(self.tokens) is not int or self.tokens < 1:
            raise ValueError(f"call shape tokens must be a positive int, got {self.tokens!r}")
        if self.name in ("decode", "verify") and self.tokens != 1:
            raise ValueError("decode-family shapes advance one token per row")
        if self.name == "decode" and self.rows != 1:
            raise ValueError("the decode shape is the ONE-ROW shape; use verify_shape(rows)")
        if self.name == "verify" and self.rows < 2:
            raise ValueError("the verification shape is the multi-row shape distinct from decode")
        if self.name == "prefill" and self.tokens < 2:
            raise ValueError("prefill calls carry prompt tokens, not one-token steps")

    @property
    def family(self):
        return GraphFamily.DECODE if self.tokens == 1 else GraphFamily.PREFILL

    @property
    def is_decode_graph(self):
        return self.family == GraphFamily.DECODE

    @property
    def is_prefill_graph(self):
        return self.family == GraphFamily.PREFILL

    # --- the baked wiring decisions ---------------------------------------

    @property
    def overlap_selection(self):
        """Selection path on a second stream, overlapping projections + head gather.

        Decode graphs only (consult7 §5; consult8 §6: at 4,096 rows both are
        byte-bound on the same links and the overlap is a loss).
        """
        return self.is_decode_graph

    @property
    def overlap_shared_expert(self):
        """Shared expert overlapped with the routed experts (consult7 §5)."""
        return self.is_decode_graph

    @property
    def query_wiring(self):
        """How the CP query exchange moves queries (the `kv_b` two-wirings).

        `"owner-absorbed"`: decode keeps the owner-side absorb — each rank
        absorbs its own head shard BEFORE the gather and reads no extra
        weights. `"unabsorbed-local-absorb"`: prefill gathers unabsorbed
        queries and absorbs + expands locally (consult8 §5).
        """
        return "owner-absorbed" if self.is_decode_graph else "unabsorbed-local-absorb"

    @property
    def local_offer(self):
        """The scores-only exchange's local candidate offer (consult8 §5)."""
        return DEFAULT_LOCAL_OFFER

    @property
    def mla_mode(self):
        """The `sparse_mla` plan mode for this shape (`Caps.mode`).

        One plan per call shape: `"decode"` (the one-row serving step),
        `"verify"` (the verification rows) and `"extend"` (prefill chunks) are
        DISTINCT plan shapes — the one-row decode graph is not the
        verification graph (consult7 §5).
        """
        return {"decode": "decode", "verify": "verify", "prefill": "extend"}[self.name]

    @property
    def short_context_bypass(self):
        """No indexer, no exchange at `context <= SHORT_CONTEXT_TOKENS` (both lists)."""
        return True

    @property
    def cuda_graph_friendly(self):
        """Fixed call shapes with prepared plans and caller-owned outputs."""
        return True


DECODE_ONE_ROW = CallShape("decode", rows=1, tokens=1)


def verify_shape(rows):
    """The verification-rows decode shape (distinct from `DECODE_ONE_ROW`)."""
    return CallShape("verify", rows=rows, tokens=1)


def prefill_shape(rows, tokens):
    """A prefill-graph call shape (see `chunk_plan` for how chunks are cut)."""
    return CallShape("prefill", rows=rows, tokens=tokens)


@dataclass(frozen=True)
class _Placement:
    """Which stream one operation runs on, for one graph family."""

    op: str
    family: str
    stream: str


class StreamSchedule:
    """Per-graph-family stream placement (consult7 §5 / consult8 §6).

    Ops: `"projections"`, `"selection_exchange"`, `"query_exchange"` (the head
    all-gather), `"routed_experts"`, `"shared_expert"`, `"tp_all_reduce"`.
    Streams: `"compute"`, `"selection"` (the second stream), `"moe_side"`,
    `"wire"` (the wire-heavy exchange lane) and `"ring"` (copy-engine ring
    transfers). Two WIRE-HEAVY exchanges never share an overlapped window in
    prefill graphs: `"selection_exchange"` and `"query_exchange"` both place on
    `"wire"` there (serialized), while decode graphs place the selection
    exchange on `"selection"` so it overlaps projections + head gather.
    """

    def __init__(self, placements=()):
        self._placements = {item: True for item in placements}

    @classmethod
    def for_family(cls, family):
        if family == GraphFamily.DECODE:
            return cls((
                _Placement("projections", family, "compute"),
                _Placement("selection_exchange", family, "selection"),
                _Placement("query_exchange", family, "compute"),
                _Placement("routed_experts", family, "compute"),
                _Placement("shared_expert", family, "moe_side"),
                _Placement("tp_all_reduce", family, "compute"),
            ))
        if family == GraphFamily.PREFILL:
            return cls((
                _Placement("projections", family, "compute"),
                _Placement("selection_exchange", family, "wire"),
                _Placement("query_exchange", family, "wire"),
                _Placement("routed_experts", family, "compute"),
                _Placement("shared_expert", family, "compute"),
                _Placement("tp_all_reduce", family, "ring"),
            ))
        raise ValueError(f"unknown graph family {family!r}")

    def placement(self, op, family):
        item = _Placement(op, family, "")
        for known in self._placements:
            if (known.op, known.family) == (op, family):
                return known.stream
        raise ValueError(f"no stream placement declared for {op!r} in family {family!r}")

    def overlaps(self, first, second, family):
        """True when two operations run on different streams (concurrent)."""
        return self.placement(first, family) != self.placement(second, family)

    def wire_heavy_overlap(self, family):
        """True when two wire-heavy exchanges could overlap (a defect in prefill)."""
        return self.overlaps("selection_exchange", "query_exchange", family)


# The one schedule the body uses; tests build their own to mutate.
SCHEDULE = {
    GraphFamily.DECODE: StreamSchedule.for_family(GraphFamily.DECODE),
    GraphFamily.PREFILL: StreamSchedule.for_family(GraphFamily.PREFILL),
}


class KernelPlanBook:
    """Prepared plans per kernel per call shape ("no JIT at run time").

    `wire_operators` binds every plan the body will run — the iHC
    `norm.mhc.ihc` boundaries, the `sparse_mla` attention plans (ONE PLAN PER
    CALL SHAPE: the one-row decode plan is not the verification-rows plan),
    the `dsa_indexer` selection plans, the `PreparedSinkFold` and the
    `PreparedSelectionHandback`. A run-time lookup for an unbound shape
    raises instead of planning inside a step or a graph capture.
    """

    def __init__(self):
        self._plans = {}
        self.device = "cpu"

    def bind(self, shape, kernel, plan):
        self._plans[(shape.name, shape.rows, shape.tokens, kernel)] = plan

    def bind_shared(self, kernel, plan):
        """Shape-independent plan (iHC boundaries, sink fold, hand-back)."""
        self._plans[(None, None, None, kernel)] = plan

    def plan_for(self, shape, kernel):
        key = (shape.name, shape.rows, shape.tokens, kernel)
        if key not in self._plans:
            raise RuntimeError(
                f"no prepared {kernel!r} plan for call shape {shape.name} "
                f"(rows={shape.rows}, tokens={shape.tokens}); plans bind at "
                "wire_operators time — there is no JIT at run time")
        return self._plans[key]

    def shared(self, kernel):
        key = (None, None, None, kernel)
        if key not in self._plans:
            raise RuntimeError(
                f"no prepared {kernel!r} plan; plans bind at wire_operators "
                "time — there is no JIT at run time")
        return self._plans[key]

    def has(self, shape, kernel):
        return (shape.name, shape.rows, shape.tokens, kernel) in self._plans

    def kernels(self):
        return sorted({key[3] for key in self._plans})

    def shapes(self):
        return sorted({(key[0], key[1], key[2]) for key in self._plans if key[0] is not None})


# --- deterministic, length-only chunking (consult8 §5) --------------------
def chunk_plan(lengths, chunk_size):
    """Cut rows of the given per-row token lengths into chunks — by LENGTH ONLY.

    The chunk boundaries of every row depend on that row's token count and
    `chunk_size` and NOTHING else (consult8 §5: "deterministic, length-only
    chunking"; K3.1 fixed a 21 % TTFT bump at one prompt length this way).
    The same lengths always produce the same plan, whatever the token ids,
    scores or iteration order — the chunk-position nondeterminism mutant
    (a content-dependent cut) fails exactly this property.

    Returns a tuple of `(row, start, stop)` chunks in row-major order, each
    chunk at most `chunk_size` tokens of one row.
    """
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError(f"chunk_size must be a positive int, got {chunk_size!r}")
    plan = []
    for row, length in enumerate(lengths):
        if type(length) is not int or length < 0:
            raise ValueError(f"row lengths must be nonnegative ints, got {length!r}")
        for start in range(0, length, chunk_size):
            plan.append((row, start, min(start + chunk_size, length)))
    return tuple(plan)
