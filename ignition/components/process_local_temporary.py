# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Process-local temporaries for the decode glue exchange + aggregation (W10 R9).

R9 (``FABLE-PERF-LEVERS.md`` section 2) prices the decode glue chains at 567 chains
of sub-3 us kernels and 1,938 removable launches per round, worth at most
1.66-2.03 ms/round and realistically 1.3-1.7 (``levers/w2-results.md`` section 2).
The glue operators allocate their exchange and aggregation staging per call; per
J-3 the simplest prototype is a **process-local temporary**: one staging region,
reserved at preparation, reused by every round, layer and slice of the decode glue.

This module is the allocation side only. It performs no collective and touches no
transport. Any change to the exchange/all-reduce path ("radiance") needs the
radiance-mechanics review before it can even be benched; see the threshold note in
``scripts/benchmark_e2_allocation.py``.

Allocation variants (test order fixed by J-3: R9d first, then R9a, then R9b as
fallbacks; the variants change allocation only and must be bit-identical):

``r9d`` - pinned/upper-buffer pool (default). A pool of 256-byte-aligned chunks
    carved into views with an explicit prepared upper bound; host-visible staging
    can be pinned. Reservations above the bound are rejected explicitly at
    preparation (admission), never reallocated during a round.
``r9a`` - single multi-round reservation. One flat buffer sized to the largest
    multi-round extent at preparation; every later carve is a view into it.
``r9b`` - two-stage. Stage one reserves one coarse block per group at
    preparation; stage two carves per-stage views lazily on first use without any
    further allocation.

R9 threat checklist (carried from the lever documents; every item is asserted by
``tests/recipes/test_kimi31_e2_union.py``, test names carry the threat id):

* T1 unknown-gain: the net gain is unknown until one chain is timed unprofiled;
  1.66-2.03 ms/round is an upper bound (FABLE-PERF-LEVERS section 2 R9 row,
  w2-results section 2). The prototype claims no measured gain.
* T2 dispatch-floor: per-node cost is dominated by graph dispatch (about 0.85 us
  floor, w2-results section 2); a fused path must remove nodes, not reorder them.
* T3 bit-equal: any fused chain must confirm bit-equal output (j3-fable-review
  section 6); the allocation strategy must never change numerics.
* T4 no-steady-state-allocation: serving allocates no steady-state GPU memory
  (docs/design-review.md principle 2); temporaries are reserved at preparation and
  never grow inside the round loop.
* T5 lifetime: never reuse a temporary while queued work or transfers reference it
  (design-review principle 6). Sharing across stages is only safe because a rank
  replays one captured graph at a time (ignition/preparation/scratch.py).
* T6 no-4-byte-wire: the 4-byte DmaAllReduce wire returned 8x the sum once and is
  never used (J-1 section 5.1); this module is not a transport and introduces none.
* T7 capture-discipline: compare kernels only at equal packed rows; captures give
  shares, unprofiled cells give gains (campaign hard rules, J-1 section 2).
* T8 profiling-gap: the 9.6 ms profile-vs-unprofiled gap does NOT live in the tiny
  kernels (w2-results section 2); it is not booked as upside here.
* T9 variant-order: R9d, then R9a, then R9b as fallback; outputs are bit-identical
  across variants (allocation only).
* T10 memory-budget: temporary bytes count against the shared CPU/GPU DRAM budget
  and the memory map (design-review principle 8); ``nbytes`` is account-able.
"""

import threading
from dataclasses import dataclass

import torch

__all__ = [
    "R9_THREATS",
    "TemporaryVariant",
    "ProcessLocalTemporary",
    "GlueScratch",
    "process_local_temporary",
    "release_all_temporaries",
    "temporary_stats",
]

R9_THREATS = {
    "T1": "unknown gain until one chain is timed unprofiled; 1.66-2.03 ms/round is an upper bound",
    "T2": "graph dispatch floor (~0.85 us/node): fused paths must remove nodes, not reorder them",
    "T3": "fused chains must be bit-equal; allocation variants must never change numerics",
    "T4": "no steady-state GPU allocation while serving; reserve at prepare, never grow per round",
    "T5": "never reuse a temporary while queued work or transfers still reference it",
    "T6": "never the 4-byte DmaAllReduce wire (returned 8x the sum once; J-1 5.1)",
    "T7": "compare kernels at equal packed rows; unprofiled cells, not captures, price gains",
    "T8": "the 9.6 ms profiling gap is not in the tiny kernels; do not book it as upside",
    "T9": "variant fallback order R9d -> R9a -> R9b; bit-identical output across variants",
    "T10": "temporary bytes count against the shared CPU/GPU DRAM budget and memory map",
}

ALIGNMENT = 256


def _span(count, itemsize):
    return (count * itemsize + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


class TemporaryVariant:
    """Allocation strategy names; J-3 fixes the test/selection order R9d, R9a, R9b."""

    POOL = "r9d"
    RESERVATION = "r9a"
    TWO_STAGE = "r9b"
    ORDER = (POOL, RESERVATION, TWO_STAGE)

    @classmethod
    def check(cls, variant):
        if variant not in cls.ORDER:
            raise ValueError(f"unknown R9 temporary variant {variant!r}; expected one of {cls.ORDER}")
        return variant


@dataclass(frozen=True)
class _Reservation:
    """One carved view: name, element count, aligned byte span, backing chunk."""

    name: str
    count: int
    offset_bytes: int
    chunk: int


class ProcessLocalTemporary:
    """One reusable staging region for the decode glue exchange + aggregation.

    All variants expose the same contract: ``reserve`` is the only allocating
    entry point, it is preparation work, and repeated ``carve`` calls for the same
    name return the same storage without allocation (T3, T4). Allocations are
    counted so tests can prove the round loop stays allocation-free (T4).
    """

    def __init__(self, name, device, *, variant, max_bytes, pinned=False):
        self.name, self.device = name, torch.device(device)
        self.variant = TemporaryVariant.check(variant)
        self.max_bytes = int(max_bytes)
        self.pinned = bool(pinned)
        if self.max_bytes <= 0 or self.max_bytes % ALIGNMENT:
            raise ValueError(f"temporary {name}: max_bytes must be a positive multiple of {ALIGNMENT}")
        self._lock = threading.Lock()
        self._chunks = []
        self._dtypes = {}
        self._reservations = {}
        self._open_blocks = {}  # r9b stage-one blocks: group -> (chunk, offset, span)
        self.allocations = 0
        self.carves = 0

    # -- accounting ------------------------------------------------------

    @property
    def nbytes(self):
        """Bytes reserved against the shared memory budget (T10)."""
        return sum(chunk.nbytes for chunk in self._chunks)

    @property
    def reserved_bytes(self):
        return sum(_span(reservation.count, self._dtypes[reservation.name].itemsize)
                   for reservation in self._reservations.values())

    def stats(self):
        return {"name": self.name, "variant": self.variant, "chunks": len(self._chunks),
                "allocations": self.allocations, "carves": self.carves,
                "nbytes": self.nbytes, "reserved_bytes": self.reserved_bytes,
                "reservations": {name: (reservation.count, reservation.offset_bytes, reservation.chunk)
                                 for name, reservation in self._reservations.items()}}

    # -- reservation (preparation only) ----------------------------------

    def reserve(self, name, shape, dtype):
        """Reserve ``shape`` of ``dtype`` once and return the reusable tensor.

        Repeated calls with the same name and shape return the same storage
        without allocation. A different re-reservation of an existing name is a
        preparation error rather than a silent reallocation (T4).
        """
        shape = (shape,) if isinstance(shape, int) else tuple(shape)
        count = 1
        for dim in shape:
            count *= dim
        span = _span(count, dtype.itemsize)
        with self._lock:
            existing = self._reservations.get(name)
            if existing is not None:
                if existing.count != count or self._dtypes[name] != dtype:
                    raise ValueError(f"temporary {self.name}: {name} already reserved with a different extent")
                self.carves += 1
                return self._view(existing, shape, dtype)
            self._dtypes[name] = dtype
            if self.variant == TemporaryVariant.RESERVATION:
                self._single_reservation(span)
                index, offset = 0, self._block_used(0)
            elif self.variant == TemporaryVariant.TWO_STAGE:
                index, offset = self._stage_one(name, span)
            else:
                index, offset = self._pool_carve(span)
            self._reservations[name] = _Reservation(name, count, offset, index)
            self.carves += 1
            return self._view(self._reservations[name], shape, dtype)

    def carve(self, name, shape, dtype):
        """Return the already-reserved storage for ``name`` (round-loop safe)."""
        with self._lock:
            reservation = self._reservations.get(name)
            if reservation is None:
                raise KeyError(f"temporary {self.name}: {name} was never reserved")
            if self._dtypes[name] != dtype:
                raise ValueError(f"temporary {self.name}: {name} was reserved with a different dtype")
            self.carves += 1
            return self._view(reservation, (shape,) if isinstance(shape, int) else tuple(shape), dtype)

    # -- variant internals ----------------------------------------------

    def _single_reservation(self, span):
        """R9a: one flat multi-round reservation; every carve is a view into it."""
        if not self._chunks:
            self._allocate(max(self.max_bytes, span))
        elif self._block_used(0) + span > self._chunks[0].nbytes:
            raise ValueError(f"temporary {self.name}: single multi-round reservation exhausted (T4 admission)")

    def _stage_one(self, name, span):
        """R9b stage one: one coarse block per group; stage two carves inside it."""
        group = name.split(".", 1)[0]
        block = self._open_blocks.get(group)
        if block is not None:
            index, offset, used = block
            if offset + used + span <= self._chunks[index].nbytes:
                self._open_blocks[group] = (index, offset, used + span)
                return index, offset + used
        index, offset = self._pool_carve(span)
        self._open_blocks[group] = (index, offset, span)
        return index, offset

    def _pool_carve(self, span):
        """R9d: upper-buffer pool; carve aligned spans, grow the pool explicitly."""
        for index, chunk in enumerate(self._chunks):
            used = self._block_used(index)
            if used + span <= chunk.nbytes:
                return index, used
        growth = max(span, 4 << 20)
        if self.nbytes + growth > self.max_bytes:
            growth = span  # _allocate applies the explicit admission check below
        self._allocate(growth)
        return len(self._chunks) - 1, 0

    def _block_used(self, index):
        return max((reservation.offset_bytes + _span(reservation.count, self._dtypes[reservation.name].itemsize)
                    for reservation in self._reservations.values() if reservation.chunk == index), default=0)

    def _allocate(self, span):
        total = self.nbytes + span
        if total > self.max_bytes:
            raise ValueError(f"temporary {self.name}: reservation of {total} B exceeds the prepared "
                             f"upper bound of {self.max_bytes} B (explicit admission, T4/T10)")
        kwargs = {"device": self.device}
        if self.pinned and self.device.type == "cuda":
            # Pinned staging is host-visible scratch only; the device side stays
            # plain device memory (the exchange never uses the 4-byte wire, T6).
            kwargs["pin_memory"] = True
        self._chunks.append(torch.empty(span, dtype=torch.uint8, **kwargs))
        self.allocations += 1

    def _view(self, reservation, shape, dtype):
        chunk = self._chunks[reservation.chunk]
        begin = reservation.offset_bytes
        return chunk[begin:begin + reservation.count * dtype.itemsize].view(dtype).view(shape)


class GlueScratch:
    """Named glue scratch views carved from one process-local temporary.

    ``PreparedDistributedTopK`` and friends allocate their exchange packing and
    aggregation scratch per call today; under the E2 union they carve it here once
    and reuse it every round (T3/T4). Callers must treat every view as live for
    the whole replayed step (T5).
    """

    def __init__(self, temporary):
        self.temporary = temporary

    def zeros(self, name, shape, dtype):
        view = self.temporary.reserve(name, shape, dtype)
        view.zero_()
        return view


_TEMPORARIES = {}
_TEMPORARIES_LOCK = threading.Lock()


def process_local_temporary(name, device, *, variant, max_bytes, pinned=False):
    """Return the process's temporary for ``name``, creating it at preparation."""
    key = (name, str(torch.device(device)), variant)
    with _TEMPORARIES_LOCK:
        temporary = _TEMPORARIES.get(key)
        if temporary is None:
            temporary = ProcessLocalTemporary(name, device, variant=variant, max_bytes=max_bytes, pinned=pinned)
            _TEMPORARIES[key] = temporary
        return temporary


def temporary_stats():
    with _TEMPORARIES_LOCK:
        return {key: temporary.stats() for key, temporary in _TEMPORARIES.items()}


def release_all_temporaries():
    """Drop every process-local temporary (tests and preparation teardown)."""
    with _TEMPORARIES_LOCK:
        _TEMPORARIES.clear()
