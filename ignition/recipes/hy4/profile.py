# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0
"""W29 instrumentation: per-phase wall timing inside the Hy4 forward body.

consult15 §1 — one small module and three call sites (`decoder.py`,
`collectives.py`, `h4_serve.py`; `moe.py` carries the two MoE phases the list
names). `PhaseTimer.phase(name)` is a context manager: enabled it calls
`torch.cuda.synchronize()` on entry and exit and adds wall time and a call
count to `name`; disabled it is a no-op (zero overhead — the serving body must
bebit-identical with the flag off). It also carries byte counters
(`add_host`) for the host collective wires.

Report form (consult15 §1 "Output form"): per call shape, a table of
phase -> total ms, calls, ms per call, share of forward wall; plus
`other = wall - sum(phases)` (Python dispatch), and host bytes / host call
counts. The `host.*` sub-phases (`host.d2h`, `host.gloo`, `host.h2d`) sit
INSIDE the `comm.*` phases that call the host wire, so they are reported as
sub-phases and excluded from the `other` sum (their parent `comm.*` row is the
top-level cost; the host rows are its breakdown).

Caveat (consult15 §1): the synchronisation adds a few percent and hides
overlap. That is acceptable while the path is host-bound; re-check once it is
not.
"""

import time
from contextlib import contextmanager

__all__ = ["PhaseTimer", "TIMER", "render_tables"]


class PhaseTimer:
    """Accumulates per-phase wall time and call counts per forward call shape.

    Lifecycle (h4_serve): `begin_request(steps)` arms the timer for at most
    `steps` forwards of the request that follows the warm request;
    `end_request()` disarms it and returns the report (None when no forward
    was timed). `decoder.forward` brackets each forward with
    `forward_begin(call_shape.name)` / `forward_end()`.
    """

    def __init__(self):
        self.enabled = False          # armed for this request
        self.budget = 0               # forwards still to time in this request
        self.shape = None
        self.forwards = []            # per-forward wall ms (timed forwards only)
        self._active = False          # inside a timed forward
        self._depth = 0
        self._t0 = 0.0
        self._buckets = {}

    # -- arming (h4_serve.py) ---------------------------------------------
    def begin_request(self, steps):
        self.budget = int(steps)
        self.enabled = self.budget > 0
        self._buckets = {}
        self.forwards = []
        self._active = False

    def end_request(self):
        """Disarm and return the report dict (None when nothing was timed)."""
        self.enabled = False
        self._active = False
        self.shape = None
        if not self.forwards:
            return None
        return self.report()

    # -- forward accounting (decoder.forward) -----------------------------
    def forward_begin(self, shape):
        if not self.enabled or self.budget <= 0:
            self._active = False
            return
        self.budget -= 1
        self._active = True
        self.shape = str(shape)
        self._depth = 0
        self._sync()
        self._t0 = time.perf_counter()

    def forward_end(self):
        if not self._active:
            return
        self._sync()
        wall_ms = (time.perf_counter() - self._t0) * 1000.0
        bucket = self._bucket(self.shape)
        bucket["wall_ms"] += wall_ms
        bucket["forwards"] += 1
        self.forwards.append(wall_ms)
        self._active = False

    # -- phases -----------------------------------------------------------
    @contextmanager
    def phase(self, name):
        if not self._active:
            yield
            return
        self._sync()
        t0 = time.perf_counter()
        top = self._depth == 0
        self._depth += 1
        try:
            yield
        finally:
            self._depth -= 1
            self._sync()
            record = self._bucket(self.shape)["phases"].setdefault(
                str(name), {"ms": 0.0, "calls": 0, "sub": not top})
            record["ms"] += (time.perf_counter() - t0) * 1000.0
            record["calls"] += 1

    def add_host(self, surface, payload_bytes, copies=1):
        """Host-wire counter: `copies` payloads of `payload_bytes` through the host."""
        if not self._active:
            return
        counter = self._bucket(self.shape)["host"].setdefault(
            str(surface), {"bytes": 0, "calls": 0})
        counter["bytes"] += int(payload_bytes) * int(copies)
        counter["calls"] += 1

    # -- report -----------------------------------------------------------
    def _bucket(self, shape):
        bucket = self._buckets.get(shape)
        if bucket is None:
            bucket = self._buckets[shape] = {
                "phases": {}, "host": {}, "wall_ms": 0.0, "forwards": 0}
        return bucket

    @staticmethod
    def _sync():
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def report(self):
        shapes = {}
        for shape, bucket in self._buckets.items():
            wall = bucket["wall_ms"]
            phases = {}
            top_ms = 0.0
            for name, record in sorted(bucket["phases"].items(),
                                       key=lambda item: -item[1]["ms"]):
                calls = max(1, record["calls"])
                phases[name] = {
                    "ms": round(record["ms"], 3),
                    "calls": record["calls"],
                    "ms_per_call": round(record["ms"] / calls, 4),
                    "share_pct": round(100.0 * record["ms"] / wall, 2) if wall else 0.0,
                    "sub": record["sub"],
                }
                if not record["sub"]:
                    top_ms += record["ms"]
            shapes[shape] = {
                "forwards": bucket["forwards"],
                "wall_ms": round(wall, 3),
                "ms_per_forward": round(wall / max(1, bucket["forwards"]), 3),
                "phases": phases,
                "top_phase_ms": round(top_ms, 3),
                "other_ms": round(wall - top_ms, 3),
                "host": bucket["host"],
            }
        return {"forwards_ms": [round(value, 3) for value in self.forwards],
                "shapes": shapes}


TIMER = PhaseTimer()


def render_tables(report):
    """The consult15 §1 output form: one phase table per call shape."""
    lines = []
    for shape, data in sorted(report["shapes"].items()):
        wall = data["wall_ms"]
        lines.append(f"== call shape {shape}: {data['forwards']} forwards, "
                     f"{wall:.1f} ms total, {data['ms_per_forward']:.1f} ms/forward ==")
        lines.append(f"{'phase':<24} {'total ms':>10} {'calls':>8} {'ms/call':>9} {'share':>7}")
        for name, record in data["phases"].items():
            marker = "  (sub)" if record["sub"] else ""
            lines.append(f"{name + marker:<24} {record['ms']:>10.1f} {record['calls']:>8} "
                         f"{record['ms_per_call']:>9.3f} {record['share_pct']:>6.1f}%")
        lines.append(f"{'other (dispatch)':<24} {data['other_ms']:>10.1f} {'':>8} {'':>9} "
                     f"{(100.0 * data['other_ms'] / wall if wall else 0.0):>6.1f}%")
        for surface, counter in sorted(data["host"].items()):
            lines.append(f"host wire {surface}: {counter['calls']} calls, "
                         f"{counter['bytes'] / (1 << 20):.1f} MiB through host")
    return "\n".join(lines)
