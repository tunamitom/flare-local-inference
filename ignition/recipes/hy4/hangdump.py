"""consult18 stage D: evidence for a stalled multi-rank forward.

The copy-engine exchanges and the ring all-reduce wait on device-side flags
with unbounded spins, so a protocol mismatch shows up as every card at 100 %
and every host parked in some synchronizing call. Neither tells you which
wait is short. `HangWatch` arms two dumps for the duration of a block:

* `faulthandler` prints every thread's Python stack (C-level timer, works
  while the main thread sits in a CUDA synchronize);
* a daemon thread reads each channel's monotonic flags and counters on a side
  stream (spin kernels do not block a copy on another stream) and prints one
  line per channel: for a CE exchange the per-(op, source) inbound flag next
  to the waiter's expected count, for a ring the same per flag slot.

A stalled wait is a slot where `wait > flag` on the destination; the sender's
send counter for that slot tells whether the send was never issued (host never
got there) or issued but never landed.
"""
from __future__ import annotations

import faulthandler
import sys
import threading

import torch

__all__ = ["HangWatch", "comm_state_lines", "lazy_preparation_trap"]


class _Bytes:
    def __init__(self, ptr, nbytes):
        self.__cuda_array_interface__ = {
            "shape": (int(nbytes),), "typestr": "|u1", "data": (int(ptr), False), "version": 3}


def _host(tensor, stream):
    with torch.cuda.stream(stream):
        return tensor.to("cpu").tolist()


def _flags(ptr, slots, stride, device, stream):
    raw = torch.as_tensor(_Bytes(ptr, slots * stride), device=device)
    words = raw.view(torch.int32).view(slots, stride // 4)[:, 0]
    return _host(words, stream)


def _nonzero(values, width=None):
    items = [(index, value) for index, value in enumerate(values) if value]
    return " ".join(f"{index}:{value}" for index, value in items[:width])


def comm_state_lines(transport, *, stream=None):
    """One line per channel (main + lanes) with its live flags and counters."""
    device = transport.device
    stream = stream or torch.cuda.Stream(device=device)
    lines = []
    rings = [("ring0", getattr(transport, "_dma", None))]
    rings += [(f"ring{i}", ring) for i, ring in enumerate(getattr(transport, "_lane_rings", ()), 1)]
    for name, ring in rings:
        if ring is None:
            continue
        from b12x.comm.pcie.pcie_dma import FLAG_SLOTS, FLAG_STRIDE

        flags = _flags(ring._flags_base[ring.rank], FLAG_SLOTS, FLAG_STRIDE, device, stream)
        lines.append(f"{name} flag[{_nonzero(flags)}]")
        lines.append(f"{name} send[{_nonzero(_host(ring._send_counters, stream))}]")
        lines.append(f"{name} wait[{_nonzero(_host(ring._wait_counters, stream))}]")
        lines.append(f"{name} done[{_nonzero(_host(ring._wait_done, stream))}]")
    ces = [("ce0", getattr(transport, "_ce", None))]
    ces += [(f"ce{i}", ce) for i, ce in enumerate(getattr(transport, "_lane_ces", ()), 1)]
    for name, ce in ces:
        if ce is None:
            continue
        flags = _flags(ce.ptrs[ce.rank], ce.FLAG_SLOTS, ce.FLAG_STRIDE, device, stream)
        lines.append(f"{name} calls={dict(ce.calls)} flag[{_nonzero(flags)}]")
        lines.append(f"{name} send[{_nonzero(_host(ce._send, stream))}]")
        lines.append(f"{name} wait[{_nonzero(_host(ce._wait, stream))}]")
    return lines


class HangWatch:
    """`with HangWatch(transport, seconds, label):` dump stacks + comm state if
    the block is still running after `seconds` (then again every `seconds`)."""

    def __init__(self, transport, seconds, label, *, rank=0, out=None):
        self.transport, self.seconds, self.label = transport, float(seconds), label
        self.rank, self.out = rank, out or sys.stderr
        self._done = threading.Event()
        self._thread = None

    def _run(self):
        while not self._done.wait(self.seconds):
            try:
                lines = comm_state_lines(self.transport)
            except Exception as error:    # evidence only: never raise from here
                lines = [f"comm state unreadable: {error!r}"]
            for line in lines:
                print(f"[h4 r{self.rank}] HANGWATCH {self.label}: {line}", file=self.out, flush=True)

    def __enter__(self):
        if self.seconds > 0:
            faulthandler.dump_traceback_later(self.seconds, repeat=True, file=self.out)
            self._thread = threading.Thread(target=self._run, name="hy4-hangwatch", daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        if self.seconds > 0:
            faulthandler.cancel_dump_traceback_later()
            self._done.set()
        return False


class lazy_preparation_trap:
    """Refuse b12x's lazy first-use preparation inside a block.

    Lazy preparation synchronizes the whole device. Inside the overlapped
    prefill's interleaved loop that is never needed (every plan is prepared by
    the plain forward first) and is a hazard: make it an error naming the plan
    instead of a device-wide synchronize between two lanes' exchanges."""

    def __init__(self, where):
        self.where = where
        self._types = None
        self._saved = None

    def __enter__(self):
        try:
            from b12x.preparation import types
        except Exception:
            return self
        self._types, self._saved = types, types._LAZY_PREPARER
        where = self.where

        def refuse(plan):
            raise RuntimeError(
                f"b12x plan {getattr(plan, 'component_id', plan)!r} was not prepared before "
                f"{where}; run the plain forward at this call shape first")

        types._set_lazy_preparer(refuse)
        return self

    def __exit__(self, *exc):
        if self._types is not None:
            self._types._set_lazy_preparer(self._saved)
        return False
