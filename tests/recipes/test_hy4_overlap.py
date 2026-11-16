# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0
"""consult18 stage D: the two-half overlapped prefill (CPU twin; lanes, no streams)."""

import pytest

torch = pytest.importorskip("torch")

from tests.recipes.test_hy4_w6_cp import (  # noqa: E402,F401
    fresh_decoders, golden_tokens, rel_error, tiny_available, worlds,
)
from tests.recipes.test_hy4_w7_wiring import run_wired  # noqa: E402

from ignition.recipes.hy4.wiring import prefill_shape  # noqa: E402

HALF = 8


def _scenario(mode):
    def body(decoder):
        decoder.enable_lanes(max_q_rows=2 * HALF)
        _, tokens = golden_tokens()
        batch = torch.tensor([tokens[:2 * HALF]], dtype=torch.long)
        if mode == "overlap":
            logits = decoder.forward_overlapped(batch, half_shape=prefill_shape(1, HALF),
                                                logits_rows="all")
            decoder.append([0], 2 * HALF)
        else:
            first = decoder.forward(batch[:, :HALF], logits_rows="all")
            decoder.append([0], HALF)
            second = decoder.forward(batch[:, HALF:], logits_rows="all")
            decoder.append([0], HALF)
            logits = torch.cat((first, second), dim=1)
        step = decoder.decode_step(torch.tensor([[7]], dtype=torch.long), request_rows=[0])
        return [logits, step]
    return body


@pytest.mark.parametrize("world", (1, 2, 8))
def test_overlapped_halves_equal_two_sequential_halves(worlds, world):
    """Lane interleaving changes nothing: the overlapped chunk equals the two
    halves run one after the other (bit-exact on the CPU twin), and the decode
    step after it reads the same cache."""
    from ignition.recipes.hy4.collectives import SimulatedGroup

    def run(mode):
        if world == 1:
            return [_scenario(mode)(decoder) for decoder in fresh_decoders(worlds[1])]
        return run_wired(worlds[world], _scenario(mode), group=SimulatedGroup(world))

    for sequential, overlapped in zip(run("sequential"), run("overlap")):
        for want, got in zip(sequential, overlapped):
            assert torch.equal(got, want), f"world {world}: rel {rel_error(got, want)}"


def test_overlapped_chunk_is_close_to_the_unsplit_chunk(worlds):
    def body(decoder):
        decoder.enable_lanes(max_q_rows=2 * HALF)
        _, tokens = golden_tokens()
        batch = torch.tensor([tokens[:2 * HALF]], dtype=torch.long)
        return decoder.forward_overlapped(batch, half_shape=prefill_shape(1, HALF), logits_rows="last")

    def whole(decoder):
        _, tokens = golden_tokens()
        return decoder.forward(torch.tensor([tokens[:2 * HALF]], dtype=torch.long), logits_rows="last")

    got = body(fresh_decoders(worlds[1])[0])
    want = whole(fresh_decoders(worlds[1])[0])
    assert rel_error(got, want) <= 2e-5
    assert torch.equal(got.argmax(-1), want.argmax(-1))


def test_lane_buffers_are_separate_and_lane_one_needs_them(worlds):
    from ignition.recipes.hy4.lanes import use_lane

    decoder = fresh_decoders(worlds[1])[0]
    cell = next(layer.attention.selection for layer in decoder.weights.layers
                if layer.attention.selection.kind == "full")
    saved = cell.lane_caches
    cell.lane_caches = ()       # the module's weights (and cells) are shared across tests
    with use_lane(1), pytest.raises(RuntimeError, match="no buffers for lane 1"):
        cell.buffers(rows=2)
    cell.lane_caches = saved
    decoder.enable_lanes(max_q_rows=2 * HALF)
    base = cell.buffers()[0]
    with use_lane(1):
        other = cell.buffers()[0]
    assert base.data_ptr() != other.data_ptr()
    with pytest.raises(RuntimeError, match="enable_lanes"):
        fresh_decoders(worlds[1])[0].forward_overlapped(
            torch.zeros((1, 2 * HALF), dtype=torch.long), half_shape=prefill_shape(1, HALF))


def test_lazy_preparation_trap_refuses_and_restores():
    """consult18 D: inside the overlapped loop b12x first-use preparation is an
    error (the d2 boot hung in its device-wide synchronize), and the trap
    restores the previous preparer on exit."""
    types = pytest.importorskip("b12x.preparation.types")
    from ignition.recipes.hy4.hangdump import lazy_preparation_trap

    before = types._LAZY_PREPARER
    with lazy_preparation_trap("forward_overlapped"):
        assert types._LAZY_PREPARER is not before
        with pytest.raises(RuntimeError, match="not prepared before forward_overlapped"):
            types._LAZY_PREPARER(object())
    assert types._LAZY_PREPARER is before


def _audit_transport(rank, world):
    from ignition.recipes.hy4 import collectives as C

    t = C.PcieDcpTransport.__new__(C.PcieDcpTransport)
    t.rank, t.world_size = rank, world
    t.lane_audit_every = 1
    t.lane_audit_calls = {"gather": 0, "scores": 0, "lse": 0, "all_reduce": 0}
    t.lane_audit_seen = {"lse": 0, "all_reduce": 0}
    t.lane_audit_max = {"lse": 0.0, "all_reduce": 0.0}
    t.lane_audit_inloop = True
    t._posthoc, t._digest_weights, t._projection = [], None, None
    return t


@pytest.mark.parametrize("rank", [0, 3, 7])
def test_lane_audit_references(rank):
    """consult18 D: the overlapped halves' gloo audit references — this rank's
    heads of an FP32 merge over every rank's partials, the FP32 ring sum, the
    rank-major gather — accept the right answer and refuse another rank's."""
    from ignition.recipes.hy4 import collectives as C

    torch.manual_seed(rank)
    world, rows, heads, width = 8, 5, 16, 6
    hpr = heads // world
    outs = [torch.randn(rows, heads, width).bfloat16() for _ in range(world)]
    lses = [torch.randn(rows, heads) * 3 for _ in range(world)]
    full, full_lse = C.merge_lse_partials(torch.stack([o.permute(1, 0, 2) for o in outs]),
                                          torch.stack([l.t() for l in lses]))
    t = _audit_transport(rank, world)

    def feed():
        calls = iter([tuple(outs), tuple(l.float() for l in lses)])
        t._gather_all = lambda value: next(calls)

    mine = slice(rank * hpr, (rank + 1) * hpr)
    feed()
    t._lane_audit_lse(outs[rank], lses[rank], (full[:, mine].contiguous(), full_lse[:, mine].contiguous()))
    other = (rank + 1) % world
    theirs = slice(other * hpr, (other + 1) * hpr)
    feed()
    with pytest.raises(RuntimeError, match="LSE merge differs"):
        t._lane_audit_lse(outs[rank], lses[rank],
                          (full[:, theirs].contiguous(), full_lse[:, theirs].contiguous()))
    xs = [torch.randn(rows, width).bfloat16() for _ in range(world)]
    t._gather_all = lambda value: tuple(xs)
    t._lane_audit_all_reduce(xs[rank], sum(x.float() for x in xs).bfloat16())
    blocks = [torch.randn(rows, hpr, width).bfloat16() for _ in range(world)]
    t._gather_all = lambda value: tuple(blocks)
    t._lane_audit_gather(blocks[rank], torch.cat(blocks, dim=1))
    with pytest.raises(RuntimeError, match="head gather differs"):
        t._lane_audit_gather(blocks[rank], torch.cat(blocks[::-1], dim=1))
    # (a refused call is counted before it raises)
    assert t.lane_audit_calls == {"gather": 2, "scores": 0, "lse": 2, "all_reduce": 1}


def test_posthoc_lane_audit_flushes_after_the_loop():
    """consult18 D (F3): in the loop the audit only records device digests /
    snapshots; flush_lane_audit compares them afterwards — the right bytes
    pass, a tampered gather segment or score offer is refused."""
    world, rows, hpr, width = 8, 4, 2, 6
    rank = 3
    torch.manual_seed(7)
    blocks = [torch.randn(rows, hpr, width).bfloat16() for _ in range(world)]
    offers = [torch.randn(rows, 5) for _ in range(world)]
    t = _audit_transport(rank, world)
    t.lane_audit_inloop = False
    digests = {}

    def gather_all(value):
        # the flush exchanges the digests this rank recorded for its own data
        if value.dtype == torch.int64 and value.shape[-1] == 2:
            key = "gather" if "gather" not in digests else "scores"
            digests[key] = True
            src = blocks if key == "gather" else offers
            return tuple(torch.stack([t._digest(src[k])]) for k in range(world))
        raise AssertionError(value.shape)

    t._gather_all = gather_all
    t._lane_audit_gather(blocks[rank], torch.cat(blocks, dim=1))
    t._lane_audit_scores(offers[rank], tuple(offers))
    assert t.lane_audit_calls["gather"] == 0       # nothing compared inside the loop
    t.flush_lane_audit()
    assert t.lane_audit_calls["gather"] == 1 and t.lane_audit_calls["scores"] == 1
    bad = list(blocks)
    bad[5] = bad[5].clone()
    bad[5][0, 0, 0] += 1
    digests.clear()
    t._lane_audit_gather(blocks[rank], torch.cat(bad, dim=1))
    t._lane_audit_scores(offers[rank], tuple(offers))
    with pytest.raises(RuntimeError, match="gather call 0 received different bytes from rank 5"):
        t.flush_lane_audit()


class _FakeScoresCe:
    """A CE exchange stand-in: every rank's offer, unit-wide calls only."""

    def __init__(self, offers_by_rank, unit=4):
        self.score_offer, self.max_rows = unit, 64
        self.offers_by_rank, self.calls, self.unit = offers_by_rank, 0, unit

    def exchange_scores(self, part):
        assert part.shape[1] == self.unit and part.is_contiguous()
        index = self.calls
        self.calls += 1
        out = []
        for full in self.offers_by_rank:
            rows, width = full.shape
            units = -(-width // self.unit)
            padded = full.new_full((rows, units * self.unit), float("-inf"))
            padded[:, :width] = full
            out.append(padded[:, index * self.unit:(index + 1) * self.unit].clone())
        return tuple(out)


@pytest.mark.parametrize("width", [3, 4, 9])
def test_ce_scores_pads_and_trims_bit_exact(width):
    """consult18 D (F1): offers of any width ride whole CE units padded with
    -inf and come back trimmed — bit-identical to the host wire's result."""
    from ignition.recipes.hy4 import collectives as C

    world = 8
    torch.manual_seed(width)
    offers = [torch.randn(3, width).sort(dim=1, descending=True).values for _ in range(world)]
    t = C.PcieDcpTransport.__new__(C.PcieDcpTransport)
    t.world_size, t.rank = world, 2
    got = t._ce_scores(_FakeScoresCe(offers), offers[2])
    assert len(got) == world
    for mine, want in zip(got, offers):
        assert mine.shape == want.shape
        assert torch.equal(mine.contiguous().view(torch.int32), want.view(torch.int32))


@pytest.mark.parametrize("seed", range(6))
def test_scores_only_flags_match_the_host_checks(seed):
    """consult18 D: the overlapped loop's device flag is set exactly when the
    single-lane path would widen (some offer saturated at the cut) or its
    exactness guard would refuse the narrow offer."""
    from ignition.recipes.hy4 import collectives as C

    torch.manual_seed(seed)
    world, rows, width, offer, topk = 4, 6, 12, 3, 5
    scores = [torch.randn(rows, width).sort(dim=1, descending=True).values for _ in range(world)]
    if seed % 2:   # make one rank dominate the top so its narrow offer saturates
        scores[1] = scores[1] + 4.0
    offers = [s[:, :offer].contiguous() for s in scores]
    thresholds, _ = C.scores_only_threshold(offers, topk)
    want = C.scores_only_offer_saturated(offers, thresholds)
    for rank in range(world):
        try:
            C.check_scores_only_exactness(thresholds, offer, scores[rank])
            refused = False
        except ValueError:
            refused = True
        got = bool(C.scores_only_flags(offers, thresholds, offer, scores[rank]))
        assert got == (want or refused)


@pytest.mark.parametrize("seed", range(4))
def test_sync_free_selection_is_identical(seed):
    """consult18 D: the overlapped loop's selection runs the plateau fill
    without reading the device first; the result is bit-identical."""
    from ignition.recipes.hy4 import collectives as C

    torch.manual_seed(seed)
    world, rows, width, topk = 4, 5, 10, 6
    scores = [torch.randn(rows, width).round(decimals=0).sort(dim=1, descending=True).values
              for _ in range(world)]          # rounded: plateaus at the cut are common
    positions = [torch.arange(width).repeat(rows, 1) * world + r for r in range(world)]
    offers = [s[:, :4].contiguous() for s in scores]
    for rank in range(world):
        a = C.select_own_candidates(offers, scores[rank], positions[rank], topk=topk, world=world,
                                    rank=rank, local_offer=4)
        b = C.select_own_candidates(offers, scores[rank], positions[rank], topk=topk, world=world,
                                    rank=rank, local_offer=4, sync_free=True)
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_exclusive_section_wraps_cooperative_launch():
    """consult18 S1: inside an overlapped half a cooperative launch must run in
    the comm token's section; outside the overlap the section is a no-op."""
    from contextlib import contextmanager

    import pytest

    from ignition.recipes.hy4 import lanes

    entered = []

    @contextmanager
    def token():
        entered.append("in")
        yield
        entered.append("out")

    lanes.require_exclusive("plain")                 # no overlap: allowed
    with lanes.exclusive():                          # no overlap: no-op
        pass
    assert entered == []
    before = lanes.SECTIONS[0]
    with lanes.use_lane(1, overlap=True, exclusive=token):
        with pytest.raises(RuntimeError, match="exclusive section"):
            lanes.require_exclusive("W4A16 routed experts")
        with lanes.exclusive():
            lanes.require_exclusive("W4A16 routed experts")
        with pytest.raises(RuntimeError):
            lanes.require_exclusive("after the section")
    assert entered == ["in", "out"]
    assert lanes.SECTIONS[0] == before + 1
    lanes.require_exclusive("after the lane")        # section factory reset


@pytest.mark.parametrize("world", (1, 2, 8))
def test_sync_free_single_lane_forward_is_bit_identical(worlds, world):
    """consult18 P1/P2: the sync-free single-row forward (host positions,
    bounds, ownership and indexer geometry; deferred selection checks) gives
    the same bits as the device-reading forward — two prefill halves and three
    decode steps, every rank."""
    from ignition.recipes.hy4 import runtime
    from ignition.recipes.hy4.collectives import SimulatedGroup

    def scenario(free):
        def body(decoder):
            decoder._sync_free_on_cpu = True
            runtime.SWITCHES["sync_free"] = "on" if free else "off"
            try:
                _, tokens = golden_tokens()
                batch = torch.tensor([tokens[:2 * HALF]], dtype=torch.long)
                out = [decoder.forward(batch[:, :HALF], logits_rows="all")]
                decoder.append([0], HALF)
                out.append(decoder.forward(batch[:, HALF:], logits_rows="all"))
                decoder.append([0], HALF)
                for token in (7, 11, 3):
                    out.append(decoder.decode_step(torch.tensor([[token]], dtype=torch.long)))
                    decoder.append([0], 1)
                ran = getattr(decoder, "sync_free_forwards", 0)
                assert ran == (5 if free else 0), f"sync-free forwards {ran}"
                return out
            finally:
                runtime.SWITCHES["sync_free"] = runtime.DEFAULTS["sync_free"]
        return body

    def run(free):
        if world == 1:
            return [scenario(free)(decoder) for decoder in fresh_decoders(worlds[1])]
        return run_wired(worlds[world], scenario(free), group=SimulatedGroup(world))

    for exact, free in zip(run(False), run(True)):
        for want, got in zip(exact, free):
            assert torch.equal(got, want), f"world {world}: rel {rel_error(got, want)}"


@pytest.mark.parametrize("world", (1, 2, 8))
def test_decode_graph_body_equals_exact_decode(worlds, world):
    """consult18 P2: the captured decode step's body — static device inputs,
    predicated KV writes (record slot -1 / index-key rewrite off the owner),
    the indexer at a width BUCKET past the live keys — gives the same bits as
    the exact decode step, every rank, over steps crossing a bucket edge."""
    from ignition.recipes.hy4.collectives import SimulatedGroup
    from ignition.recipes.hy4.decode_graph import DecodeGraphs

    def scenario(graph):
        def body(decoder):
            _, tokens = golden_tokens()
            batch = torch.tensor([tokens[:2 * HALF]], dtype=torch.long)
            decoder.forward(batch[:, :HALF], logits_rows="all")
            decoder.append([0], HALF)
            decoder.forward(batch[:, HALF:], logits_rows="all")
            decoder.append([0], HALF)
            graphs = DecodeGraphs(decoder, granule=2)
            shape = decoder.default_call_shape((1, 1))
            out = []
            for token in (7, 11, 3, 5):
                t = torch.tensor([[token]], dtype=torch.long)
                start = decoder._row_lengths[0]
                if graph:
                    assert graphs.accepts(t, row=0, start=start,
                                          call_shape=shape, state=None, return_hidden=False,
                                          logits_rows="all") is False   # CPU: never replayed
                    graphs._stage(t, 0, start)
                    width = graphs.index_width(start + 1)
                    assert width >= (start + 1 + decoder.collectives.world_size - 1) \
                        // decoder.collectives.world_size
                    local, flag = graphs._body(0, shape, width, "all")
                    assert not bool(flag)
                    out.append(decoder.collectives.gather_logits(
                        local, vocabulary_range=decoder.weights.sharding.vocabulary_range))
                else:
                    out.append(decoder.decode_step(t))
                decoder.append([0], 1)
            return out
        return body

    def run(graph):
        if world == 1:
            return [scenario(graph)(decoder) for decoder in fresh_decoders(worlds[1])]
        return run_wired(worlds[world], scenario(graph), group=SimulatedGroup(world))

    for exact, graphed in zip(run(False), run(True)):
        for want, got in zip(exact, graphed):
            assert torch.equal(got, want), f"world {world}: rel {rel_error(got, want)}"
