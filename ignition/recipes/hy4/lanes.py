# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0
"""consult18 stage D: execution lanes for the two-half overlapped prefill.

A prefill chunk runs as two halves on two CUDA streams, layer-pipelined, so
one half's communication overlaps the other half's compute. Everything a
half writes while the other may still read it — the selection buffers, the
sparse-MLA scratch, the fused-MoE workspace, every collective's channel —
exists once per lane, and the code that owns it picks its copy from the
current lane. Lane 0 is the ordinary single-stream path; lane 1 exists only
when the overlap is enabled.
"""

import contextvars
from contextlib import contextmanager

__all__ = ["SECTIONS", "exclusive", "geometry", "lane", "overlapped", "require_exclusive",
           "use_lane"]

_LANE = contextvars.ContextVar("hy4_lane", default=0)
_OVERLAPPED = contextvars.ContextVar("hy4_overlapped", default=False)
# Host-known geometry of the running half (one request row, a contiguous
# position range): lets the indexer size its work without reading the device.
_GEOMETRY = contextvars.ContextVar("hy4_geometry", default=None)
# The overlapped loop's mutual-exclusion section (the comm token): work that
# must never share the card with another lane's peer-waiting kernels runs in it.
_EXCLUSIVE = contextvars.ContextVar("hy4_exclusive", default=None)


def lane():
    """The current execution lane (0 unless an overlapped half is running)."""
    return _LANE.get()


def overlapped():
    """True inside an overlapped prefill's half (either lane)."""
    return _OVERLAPPED.get()


def geometry():
    """The running half's host-known geometry dict, or None."""
    return _GEOMETRY.get()


_IN_SECTION = contextvars.ContextVar("hy4_in_section", default=False)
# Exclusive sections entered inside overlapped halves (the serve reports it).
SECTIONS = [0]


@contextmanager
def exclusive():
    """Inside an overlapped half: the comm token's section (one collective OR
    one exclusive op in flight per rank, in host program order); else a no-op."""
    factory = _EXCLUSIVE.get()
    if factory is None:
        yield
        return
    marker = _IN_SECTION.set(True)
    SECTIONS[0] += 1
    try:
        with factory():
            yield
    finally:
        _IN_SECTION.reset(marker)


def require_exclusive(what):
    """Refuse a cooperative launch inside the overlap outside the section: it
    is a cross-rank hang hazard (consult18 S1), so make it a loud error."""
    if _EXCLUSIVE.get() is not None and not _IN_SECTION.get():
        raise RuntimeError(
            f"{what}: cooperative launch inside the overlapped loop outside the "
            "exclusive section (wrap it in lanes.exclusive())")


@contextmanager
def use_lane(index, *, overlap=False, geometry=None, exclusive=None):
    token = _LANE.set(int(index))
    marker = _OVERLAPPED.set(bool(overlap))
    shape = _GEOMETRY.set(geometry)
    section = _EXCLUSIVE.set(exclusive)
    try:
        yield
    finally:
        _EXCLUSIVE.reset(section)
        _GEOMETRY.reset(shape)
        _OVERLAPPED.reset(marker)
        _LANE.reset(token)
