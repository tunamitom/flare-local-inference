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

__all__ = ["lane", "overlapped", "use_lane"]

_LANE = contextvars.ContextVar("hy4_lane", default=0)
_OVERLAPPED = contextvars.ContextVar("hy4_overlapped", default=False)


def lane():
    """The current execution lane (0 unless an overlapped half is running)."""
    return _LANE.get()


def overlapped():
    """True inside an overlapped prefill's half (either lane)."""
    return _OVERLAPPED.get()


@contextmanager
def use_lane(index, *, overlap=False):
    token = _LANE.set(int(index))
    marker = _OVERLAPPED.set(bool(overlap))
    try:
        yield
    finally:
        _OVERLAPPED.reset(marker)
        _LANE.reset(token)
