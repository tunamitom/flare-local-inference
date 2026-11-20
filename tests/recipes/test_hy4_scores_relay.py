# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0
"""N1 — `CeExchange.exchange_scores` 2-phase island relay (CPU gates).

The relay is pure Python copy-graph orchestration on byte copies, so every
logic change is testable without a GPU:

1. Schedule shape: the relay mirrors `all_gather_heads(relay=True)` exactly —
   own offer to the cross-island partner (cross stream) + 3 in-switch peers
   (local stream), partner awaited on the local stream (relay counter), then
   the partner's slot FORWARDED to the 3 in-switch peers tagged with the
   partner's origin; `_await_all("scores", cross + local)` then `_drain`.
   The direct path stays byte-for-byte the old schedule.
2. Wire semantics: an 8-rank threaded simulation of the CE copy/flag wire
   (dma_copy + per-(dest, origin) flags + wait counters) proves the relay
   delivers BYTE-IDENTICAL received scores on every rank (bit-exact FP32
   offers -> bit-identical top-k, and the HY4_SELECTION_AUDIT gate compares
   exactly these bytes).
3. Plumbing: `ce_scores_relay` routes `relay=True` down every CE scores call
   site (`_ce_scores` incl. the padded multi-unit form, the lane path of
   `_exchange_scores_impl`), and the `--scores-relay` CLI/launcher flags
   exist. `--selection-exchange candidates` is a different code path
   (`stage_candidates`) and is untouched.
"""
import contextlib
import threading
import time
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from ignition.recipes.hy4.collectives import CeExchange, PcieDcpTransport, relay_partner

WORLD = 8
ISLAND = 4
ROWS, OFFER = 4, 8
NBYTES = ROWS * OFFER * 4


def _align(n):
    return (int(n) + 255) // 256 * 256


class _Wire:
    """The CE peer-copy + flag wire over shared byte stores.

    `copy(dst, src, n)` resolves raw pointers against registered regions (one
    per rank slab + one per offer tensor) and copies bytes; `set_flag` stamps
    a monotonic per-flag counter; `wait_flag` consumes one stamp per distinct
    wait counter (relay and non-relay waits consume the same flag
    independently — the real kernel's two counter slots). Blocking waits
    carry a deadline so a broken schedule fails instead of hanging."""

    def __init__(self):
        self.regions = []          # (start, nbytes, tensor)
        self.stamps = {}
        self.consumed = {}
        self.lock = threading.Lock()

    def register(self, tensor):
        flat = tensor.contiguous().view(torch.uint8).reshape(-1)
        self.regions.append((int(flat.data_ptr()), int(flat.numel()), flat))
        return int(flat.data_ptr())

    def _resolve(self, addr, n):
        for start, size, tensor in self.regions:
            if start <= addr and addr + n <= start + size:
                return tensor, addr - start
        raise AssertionError(f"unregistered pointer {addr:#x} ({n} B)")

    def copy(self, dst, src, n):
        with self.lock:
            d_tensor, d_off = self._resolve(dst, n)
            s_tensor, s_off = self._resolve(src, n)
            d_tensor[d_off:d_off + n].copy_(s_tensor[s_off:s_off + n])

    def set_flag(self, flag):
        with self.lock:
            self.stamps[flag] = self.stamps.get(flag, 0) + 1

    def wait_flag(self, flag, counter, deadline=20.0):
        end = time.monotonic() + deadline
        while True:
            with self.lock:
                stamp = self.stamps.get(flag, 0)
                if stamp > self.consumed.get(counter, 0):
                    self.consumed[counter] = stamp
                    return
            if time.monotonic() > end:
                raise AssertionError(f"flag wait deadlock (flag {flag:#x})")
            time.sleep(0.0005)


class _Sim:
    """8 `CeExchange` shells whose send/await primitives drive one `_Wire`."""

    def __init__(self, rows=ROWS, offer=OFFER):
        self.wire = _Wire()
        self.score_slot = _align(64 * offer * 4)
        self.slabs = [torch.zeros(2 * WORLD * self.score_slot, dtype=torch.uint8)
                      for _ in range(WORLD)]
        self.ptrs = [self.wire.register(slab) for slab in self.slabs]
        self.ces = [self._bare(rank, rows, offer) for rank in range(WORLD)]

    def _bare(self, rank, rows, offer):
        wire, ptrs = self.wire, self.ptrs
        ce = CeExchange.__new__(CeExchange)
        ce.rank, ce.world = rank, WORLD
        ce.max_rows, ce.score_offer = 64, offer
        ce.calls = {op: 0 for op in CeExchange._OPS}
        island = [peer // ISLAND for peer in range(WORLD)]
        peers = [peer for peer in range(WORLD) if peer != rank]
        ce.cross = [peer for peer in peers if island[peer] != island[rank]]
        ce.local = [peer for peer in peers if island[peer] == island[rank]]
        ce.partner = relay_partner(rank, WORLD, ISLAND)
        ce.score_slot = self.score_slot
        ce.scores_base = 0
        ce._memory = self.slabs[rank]
        ce.ptrs = list(ptrs)
        ce._streams = lambda: ("cross", "local", "flag")
        ce._start = lambda: None
        ce._drain = lambda: None

        def send_block(stream, dest, copies, sources, op, key):
            for offset, src, n in copies:
                wire.copy(ce.ptrs[dest] + offset, src, n)
            for origin in sources:
                wire.set_flag(ce._flag_ptr(dest, op, origin))

        def await_(op, source, *, relay=False):
            wire.wait_flag(ce._flag_ptr(ce.rank, op, source), (ce.rank, op, source, relay))

        def await_all(op, sources):
            for source in sources:
                await_(op, source)

        ce._send_block = send_block
        ce._await = await_
        ce._await_all = await_all
        return ce

    def bank_offsets(self, bank):
        score_slot, base = self.score_slot, 0
        return lambda origin: base + (bank * WORLD + origin) * score_slot


def _recording_ce(rank=0):
    sim = _Sim()
    ce = sim.ces[rank]
    log = []
    ce._start = lambda: log.append(("start",))
    ce._drain = lambda: log.append(("drain",))
    ce._send_block = lambda stream, dest, copies, sources, op, key: log.append(
        ("send", stream, dest, tuple(copies), tuple(sources), op, key))
    ce._await = lambda op, source, relay=False: log.append(("await", op, source, relay))
    ce._await_all = lambda op, sources: log.append(("await_all", op, tuple(sources)))
    return sim, ce, log


# --------------------------------------------------------------- schedule
@pytest.fixture
def cuda_stream_null(monkeypatch):
    """`exchange_scores` wraps its relay await in `torch.cuda.stream(...)`; on
    CPU the context is a recorded no-op."""
    entered = []

    @contextlib.contextmanager
    def fake(stream):
        entered.append(stream)
        yield

    monkeypatch.setattr(torch.cuda, "stream", fake)
    return entered


def test_relay_schedule_mirrors_the_gather_relay(cuda_stream_null):
    """The exact two-hop pattern of `_all_gather_heads_int8(relay=True)`:
    partner cross-send + local own-sends, relay await on the local stream,
    then partner-origin forwards from the partner's own slot."""
    sim, ce, log = _recording_ce(rank=0)
    offset = sim.bank_offsets(0)
    offers = torch.randn(ROWS, OFFER)
    got = CeExchange.exchange_scores(ce, offers, relay=True)
    me = 0
    own = ((offset(me), offers.data_ptr(), NBYTES),)
    forward = ((offset(4), ce.ptrs[me] + offset(4), NBYTES),)
    expected = [
        ("start",),
        ("send", "cross", 4, own, (me,), "scores", ("s-x", 4)),
        ("send", "local", 1, own, (me,), "scores", ("s-own", 1)),
        ("send", "local", 2, own, (me,), "scores", ("s-own", 2)),
        ("send", "local", 3, own, (me,), "scores", ("s-own", 3)),
        ("await", "scores", 4, True),
        ("send", "local", 1, forward, (4,), "scores", ("s-fwd", 1)),
        ("send", "local", 2, forward, (4,), "scores", ("s-fwd", 2)),
        ("send", "local", 3, forward, (4,), "scores", ("s-fwd", 3)),
        ("await_all", "scores", (4, 5, 6, 7, 1, 2, 3)),
        ("drain",),
    ]
    assert log == expected, log
    # the relay await runs ON the local stream (the forward's ordering edge)
    assert cuda_stream_null == ["local"]
    # returned views live in the bank; own slot holds this rank's offer bytes
    assert len(got) == WORLD
    assert torch.equal(got[me], offers)
    for view in got:
        assert ce.ptrs[0] <= view.data_ptr() < ce.ptrs[0] + 2 * WORLD * ce.score_slot
        assert tuple(view.shape) == (ROWS, OFFER) and view.dtype == torch.float32


def test_direct_schedule_unchanged():
    """relay=False keeps the old broadcast: 7 own-block sends, cross peers on
    the cross stream, locals on the local stream, no relay await."""
    sim, ce, log = _recording_ce(rank=0)
    offset = sim.bank_offsets(0)
    offers = torch.randn(ROWS, OFFER)
    got = CeExchange.exchange_scores(ce, offers)
    own = ((offset(0), offers.data_ptr(), NBYTES),)
    sends = [entry for entry in log if entry[0] == "send"]
    assert [(e[1], e[2], e[4], e[6]) for e in sends] == [
        ("cross", 4, (0,), ("scores", 4)), ("cross", 5, (0,), ("scores", 5)),
        ("cross", 6, (0,), ("scores", 6)), ("cross", 7, (0,), ("scores", 7)),
        ("local", 1, (0,), ("scores", 1)), ("local", 2, (0,), ("scores", 2)),
        ("local", 3, (0,), ("scores", 3)),
    ]
    assert all(e[3] == own for e in sends)
    assert not any(e[0] == "await" for e in log)
    assert log[0] == ("start",) and log[-1] == ("drain",)
    assert log[-2] == ("await_all", "scores", (4, 5, 6, 7, 1, 2, 3))
    assert torch.equal(got[0], offers)


def test_relay_requires_two_islands(cuda_stream_null):
    _, ce, _ = _recording_ce(rank=0)
    ce.partner = None
    with pytest.raises(RuntimeError, match="two islands"):
        CeExchange.exchange_scores(ce, torch.randn(ROWS, OFFER), relay=True)
    # the direct path never needs a partner
    CeExchange.exchange_scores(ce, torch.randn(ROWS, OFFER))


# ------------------------------------------------------------- wire sim
def _run_wire_sim(relay):
    sim = _Sim()
    gen = torch.Generator().manual_seed(1729 if relay else 11)
    offers = [torch.randn(ROWS, OFFER, generator=gen) for _ in range(WORLD)]
    for offer in offers:
        sim.wire.register(offer)
    results, bank_ptrs, errors = {}, {}, []
    barrier = threading.Barrier(WORLD)

    def run(rank):
        try:
            for call in range(2 if relay else 1):
                got = CeExchange.exchange_scores(sim.ces[rank], offers[rank], relay=relay)
                bank_ptrs[(rank, call)] = got[0].data_ptr()
                results[(rank, call)] = tuple(t.clone() for t in got)
                if relay:
                    barrier.wait(timeout=30)   # bank lockstep between calls
        except Exception as error:             # noqa: BLE001 - surfaced below
            errors.append((rank, repr(error)))
            try:
                barrier.abort()
            except Exception:
                pass

    # the relay await's `torch.cuda.stream(...)` context: a no-op on CPU
    original_stream = torch.cuda.stream
    torch.cuda.stream = lambda stream: contextlib.nullcontext()
    try:
        threads = [threading.Thread(target=run, args=(rank,)) for rank in range(WORLD)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)
            assert not thread.is_alive(), "wire sim deadlocked"
    finally:
        torch.cuda.stream = original_stream
    assert not errors, errors
    calls = 2 if relay else 1
    for call in range(calls):
        for rank in range(WORLD):
            got = results[(rank, call)]
            assert len(got) == WORLD
            for origin in range(WORLD):
                assert torch.equal(got[origin].view(torch.int32),
                                   offers[origin].view(torch.int32)), (rank, call, origin)
    return sim, bank_ptrs


def test_relay_is_byte_exact_on_an_eight_rank_wire():
    """8 rank threads through the simulated CE wire: every rank's received
    scores are the peers' FP32 bytes bit-for-bit (-> bit-identical top-k and
    a passing HY4_SELECTION_AUDIT), across two banked calls."""
    sim, bank_ptrs = _run_wire_sim(relay=True)
    # the two calls consumed the two banks: the views moved to the other half
    first, second = bank_ptrs[(0, 0)], bank_ptrs[(0, 1)]
    assert first != second
    assert abs(second - first) == WORLD * sim.score_slot


def test_direct_broadcast_is_byte_exact_on_the_same_wire():
    _run_wire_sim(relay=False)


# -------------------------------------------------------------- plumbing
class _RecordingCe:
    """CeExchange stand-in recording the `relay` each call received."""

    def __init__(self, world=WORLD, unit=OFFER):
        self.score_offer, self.max_rows = unit, 64
        self.world_size = world
        self.seen = []

    def accepts_scores(self, offers):
        return True

    def exchange_scores(self, offers, *, relay=False):
        self.seen.append(relay)
        return tuple(offers.clone() for _ in range(self.world_size))


def _transport(ce, **kwargs):
    transport = PcieDcpTransport(group=None, device="cpu", pools={}, rank=2,
                                 world_size=WORLD, total_heads=64, score_offer=OFFER,
                                 ce=ce, **kwargs)
    transport._lane_ces = (ce,)
    return transport


def test_scores_relay_flag_routes_through_every_ce_call_site():
    from ignition.recipes.hy4.lanes import use_lane

    for relay in (True, False):
        ce = _RecordingCe()
        transport = _transport(ce, ce_scores_relay=relay)
        assert transport.ce_scores_relay is relay
        with use_lane(1):
            transport.exchange_scores(torch.randn(ROWS, OFFER))        # 1 unit
            transport.exchange_scores(torch.randn(ROWS, 2 * OFFER))    # 2 units
        assert ce.seen == [relay, relay, relay], ce.seen
    assert _transport(_RecordingCe()).ce_scores_relay is False          # default: direct


def test_ce_scores_padding_stays_bit_exact_with_relay():
    """The multi-unit -inf pad/trim path threads `relay` and returns the same
    bits as the host wire (the offer rows' FP32 bits are what top-k reads)."""
    torch.manual_seed(3)
    world, width = WORLD, 9
    offers = [torch.randn(3, width).sort(dim=1, descending=True).values
              for _ in range(world)]

    class PaddedCe(_RecordingCe):
        def exchange_scores(self, part, *, relay=False):
            self.seen.append(relay)
            index = len(self.seen) - 1
            out = []
            for full in self.offers_by_rank:
                rows, w = full.shape
                units = -(-w // self.score_offer)
                padded = full.new_full((rows, units * self.score_offer), float("-inf"))
                padded[:, :w] = full
                out.append(padded[:, index * self.score_offer:
                                  (index + 1) * self.score_offer].clone())
            return tuple(out)

    ce = PaddedCe(unit=4)
    ce.offers_by_rank = offers
    transport = _transport(ce, ce_scores_relay=True)
    got = transport._ce_scores(ce, offers[2])
    assert ce.seen == [True, True, True]                   # width 9 -> three units
    for mine, want in zip(got, offers):
        assert torch.equal(mine.contiguous().view(torch.int32), want.view(torch.int32))


def test_scores_relay_cli_flags_present():
    """The rollback knobs: `--scores-relay 0` restores the direct broadcast
    and `--selection-exchange candidates` stays wired (the full-selection
    rollback never touches the CE scores exchange)."""
    root = Path(__file__).resolve().parents[2]
    serve = (root / "deploy" / "hy4" / "h4_serve.py").read_text()
    launcher = (root / "deploy" / "hy4" / "start-h4.sh").read_text()
    assert '"--scores-relay"' in serve
    assert "ce_scores_relay=bool(args.scores_relay)" in serve
    assert '"--selection-exchange", choices=("scores-only", "candidates")' in serve
    assert "--scores-relay" in launcher
