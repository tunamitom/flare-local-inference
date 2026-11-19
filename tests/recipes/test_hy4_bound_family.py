"""test_bound_family.py — CPU tests for the CSF bound-family rebind.

The named one-past-end read of the flat cache slot table (357,376 rows x 128
BF16 = 91,488,256 B, row width 256 B) is produced by the `max_cap()` /
`capacity` bound family: `cache.max_cap()` = 131,072 (the dynamic-extents
`cap_limit`, h4_serve.py:3055) while each row's REAL local extent is
`cap_of(row)` = 44,672 slots (357,376 flat slots / 8 rows), so every
`max_cap()`-bounded consumer admits slots past the table — flat slot 357,376
is one past its end. The fix rebinds the family to `cap_of(row)` at five
sites (decode_graph.py `DecodeGraphs.local_capacity`/`capacity` and their
`index_width`/`_geo`/`_batch_prep` uses; decoder.py `merge_topk_handback`'s
`local_capacity`, the hand-back `bounded` guard, `local_slot_map`;
spec.py's prewarm loop start bound) and keeps the landed 1701defb staging
clamp.

The gate-boot contract geometry is used exactly (the plain `Hy4Cache`
constructor allocates nothing, so this is the real 357,376-slot shape):

  rows 8, world 8, cache capacity 357,376 global positions ->
  `Hy4Cache.local_capacity` 44,672 local slots/row; flat table
  8 * 44,672 = 357,376 slots; `max_cap()` = 131,072 with `cap_limit` set.

Every rebinder must reject or derate slot 357,376 and any slot past
`cap_of(row)` while accepting in-extent slots byte-identically (the old
formulae are replicated here as `old_*` and compared value-by-value).
"""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
CPU = torch.device("cpu")

ROWS, WORLD, RANK = 8, 8, 0
FLAT_SLOTS = 357_376              # the flat slot table: 8 * 44,672 rows
ROW_CAP = FLAT_SLOTS // ROWS      # 44,672 local slots per row (cap_of)
CAP_LIMIT = 131_072               # cache.max_cap() at the gate boots (h4_serve.py:3055)
GLOBAL = ROW_CAP * WORLD          # 357,376 global positions per row
INDEX_HEAD_DIM, RECORD_BYTES = 128, 56_544


# --------------------------------------------------------------------------
# fixtures: the contract cache (uniform 44,672-slot rows) and the ragged
# dynamic-extents cache (attempt2's: row 3 narrow at the tail of the flat
# space — the row whose one-past staging is the faulting form)
# --------------------------------------------------------------------------
def contract_cache():
    from ignition.recipes.hy4.decoder import Hy4Cache

    cache = Hy4Cache(config=None, rows=ROWS, capacity=GLOBAL, device=CPU,
                     world=WORLD, rank=RANK)
    cache.cap_limit = CAP_LIMIT       # HY4_DYN_EXTENTS=on form, h4_serve.py:3055
    return cache


def ragged_cache():
    from ignition.recipes.hy4.decoder import Hy4Cache

    cache = Hy4Cache(config=None, rows=4, capacity=64, device=CPU, world=8, rank=0)
    cache.set_extents([12, 8, 8, 4], bases=[0, 12, 20, 28])
    return cache


def graphs_for(cache):
    from ignition.recipes.hy4.decode_graph import DecodeGraphs

    config = SimpleNamespace(text={"index_topk": 4}, hidden=8)
    decoder = SimpleNamespace(
        cache=cache, device=CPU, config=config, compute_dtype=torch.float32,
        collectives=SimpleNamespace(world_size=cache.world, rank=cache.rank,
                                    deferred_checks=None, transport=None),
        weights=SimpleNamespace(sharding=SimpleNamespace(vocabulary_range=(0, 32))))
    return DecodeGraphs(decoder)


def old_index_width(end, granule, geom_from, cap):
    """The pre-fix `DecodeGraphs.index_width` with the old `max_cap()` bound."""
    owned = (int(end) + WORLD - 1) // WORLD
    width = -(-owned // granule) * granule
    if width > geom_from:
        octave = 1 << (width.bit_length() - 1)
        step = octave // 4
        width = -(-width // step) * step
    return min(width, cap)


# --------------------------------------------------------------------------
# site 1: DecodeGraphs.local_capacity / capacity (decode_graph.py) and its
# uses — index_width's min, _geo's scale, _batch_prep's pad check
# --------------------------------------------------------------------------
def test_site1_bound_family_is_the_row_extent():
    cache = contract_cache()
    g = graphs_for(cache)
    assert cache.max_cap() == CAP_LIMIT                 # the defect's bound
    assert cache.cap_of(0) == cache.cap_of(7) == ROW_CAP      # the real extent
    assert g.local_capacity == ROW_CAP                  # rebound to cap_of, not max_cap()
    assert g.local_capacity < cache.max_cap()
    assert g.capacity == ROW_CAP * WORLD == GLOBAL      # cap_of(row) * world
    for row in range(ROWS):
        assert g.capacity == cache.cap_of(row) * cache.world
    # the bound family is live (c8 L6b extents grow/relocate at run time)
    cache.set_extents([16_384] + [32_768 // 7] * 7)
    assert g.local_capacity == 16_384
    assert g.capacity == 16_384 * WORLD


def test_site1_index_width_derates_past_the_row_extent():
    cache = contract_cache()
    g = graphs_for(cache)
    # in-extent byte-identity: every bucket that fit the row's extent is
    # unchanged (the old formulae replicated with the old max_cap() bound)
    sweep = set(range(1, 3000))
    sweep.update(range(3000, GLOBAL + 2, 257))
    sweep.update([8 * 8191, 8 * 8192, 8 * 8193, 44_032 * WORLD, 44_033 * WORLD,
                  44_671 * WORLD, 44_672 * WORLD, 44_673 * WORLD,
                  45_055 * WORLD, 45_056 * WORLD, GLOBAL - 1, GLOBAL, GLOBAL + 1,
                  131_071 * WORLD, 131_072 * WORLD, 2 * GLOBAL])
    for end in sorted(sweep):
        old = old_index_width(end, g.granule, g.GEOM_FROM, cache.max_cap())
        new = g.index_width(end)
        if old <= ROW_CAP:
            assert new == old, f"width bucket changed in-extent at end={end}"
        assert new <= ROW_CAP, f"width bucket overscans the row at end={end}"
        for row in range(ROWS):
            assert new <= cache.cap_of(row)
    # the derate itself: the old top-of-extent bucket overshot the row (the
    # indexer would read key-table slots past the extent — past the flat table
    # at the tail row); slot 357,376 = tail-row local 44,672 is now unreachable
    assert old_index_width(GLOBAL, g.granule, g.GEOM_FROM, cache.max_cap()) > ROW_CAP
    assert g.index_width(GLOBAL) == ROW_CAP
    assert g.index_width(GLOBAL + 1) == ROW_CAP


def test_site1_pad_check_is_the_row_extent():
    from ignition.recipes.hy4.wiring import batch_shape

    cache = contract_cache()
    g = graphs_for(cache)
    # in-extent pads (and their staging) are byte-identical
    size, tokens, rows, starts, rw, key = g._batch_prep(
        [0, 0, 0], [7, 3, 0], [357_368, 200_000, 10], batch_shape(3),
        (), (7, 357_375), False)
    assert (rows[size - 1], starts[size - 1]) == (7, 357_375)
    entry = g._batch_state(size)
    g._stage_batch(entry, size, tokens, rows, starts)
    args = entry["args"].tolist()
    # old staging formula: base_of(row) + start // world, slot -1 off-owner
    for i in range(size):
        row, start = rows[i], starts[i]
        local = cache.base_of(row) + start // WORLD
        assert args[4 * size + i] == local                      # safe: byte-identical
        assert args[3 * size + i] == (local if start % WORLD == RANK else -1)
    # the one-past pad contract (1701defb de-slots it; pinned by
    # test_attempt2_staging_fix.py::test_pad_one_past_row_is_deslotted)
    g._batch_prep([0, 0, 0], [7, 3, 0], [357_368, 200_000, 10], batch_shape(3),
                  (), (7, GLOBAL), False)
    # slot 357,376 (tail-row start 357,376 -> local 44,672) and past it are refused
    with pytest.raises(ValueError):
        g._batch_prep([0, 0, 0], [7, 3, 0], [357_368, 200_000, 10], batch_shape(3),
                      (), (7, GLOBAL + 1), False)
    with pytest.raises(ValueError):
        g._batch_prep([0], [7], [GLOBAL], batch_shape(1), (), None, False)
    with pytest.raises(ValueError):
        g._batch_prep([0], [7], [GLOBAL + 40], batch_shape(1), (), None, False)


def test_site1_pad_check_is_per_row():
    from ignition.recipes.hy4.wiring import batch_shape

    cache = ragged_cache()          # rows [12, 8, 8, 4]: global extents [96, 64, 64, 32]
    g = graphs_for(cache)
    real = ([0, 0, 0], [3, 0, 1], [31, 50, 60], batch_shape(3))
    # row 3 (extent 32): the pinned one-past pad passes, two past refuses
    g._batch_prep(*real, (), (3, 32), False)
    with pytest.raises(ValueError):
        g._batch_prep(*real, (), (3, 33), False)
    # row 0 (extent 96): the bound is the PAD ROW's extent, not the narrowest
    # row's and not the widest row's 96-wide... 96 is row 0's own extent
    g._batch_prep(*real, (), (0, 96), False)
    with pytest.raises(ValueError):
        g._batch_prep(*real, (), (0, 97), False)
    # 40 > row 3's extent but inside row 0's: per-row, accepted
    g._batch_prep(*real, (), (0, 40), False)
    with pytest.raises(ValueError):
        g._batch_prep(*real, (), (3, 40), False)


# --------------------------------------------------------------------------
# sites 2 and 3: decoder.py's fused hand-back LCAP bound and the hand-back
# `bounded` guard. The CPU driver runs `Hy4Decoder._attention_gen` through the
# real code with mocked attention/collectives (the fused branch is chosen by
# `scores.is_cuda` — a mock flag — so both branches execute on CPU).
# --------------------------------------------------------------------------
def drive_attention(cache, *, fused, merged, token_rows):
    from ignition.recipes.hy4 import decoder as decoder_mod
    from ignition.recipes.hy4.lanes import use_lane
    from ignition.recipes.hy4.runtime import SWITCHES
    from ignition.recipes.hy4.wiring import GraphFamily

    n = int(merged.shape[0])
    captured = {}
    scores = SimpleNamespace(is_cuda=True) if fused else torch.zeros((n, 4))
    candidates = torch.zeros((n, 4), dtype=torch.int64)
    selected = torch.full((n, 4), -1, dtype=torch.int64)
    counts = torch.zeros((n,), dtype=torch.int64)

    def handback(mapped, **kwargs):
        captured["mapped"] = mapped.clone()

    def merge_topk_handback(scores_, candidates_, **kwargs):
        captured["handback"] = kwargs

    attn = SimpleNamespace(
        weights=SimpleNamespace(indexer=None),
        selection=SimpleNamespace(owner_id=0, kind="full",
                                  buffers=lambda rows: (selected, counts),
                                  handback=handback),
        project_packed=lambda *a, **k: SimpleNamespace(
            q_resid=torch.zeros((n, 8)), q=torch.zeros((n, 2, 4)),
            latent=torch.zeros((n, 8)), index_keys=None),
        local_candidates=lambda *a, **k: (scores, candidates))
    layer = SimpleNamespace(attention=attn, index=0)
    collectives = SimpleNamespace(
        world_size=cache.world, rank=cache.rank, deferred_checks=None,
        merge_topk=lambda s, c, topk: (torch.zeros((n, topk)), merged.clone()),
        merge_topk_handback=merge_topk_handback,
        all_gather_heads=lambda q: q,
        fused_handback_ok=lambda: True)
    d = decoder_mod.Hy4Decoder.__new__(decoder_mod.Hy4Decoder)
    d.cache, d.collectives = cache, collectives
    d.config = SimpleNamespace(text={"index_topk": 4}, hidden=8)
    d.trace = []
    cache.write_latent = lambda *a, **k: None
    cache.write_index_keys = lambda *a, **k: None
    cache.key_table = lambda *a, **k: None
    call_shape = SimpleNamespace(family=GraphFamily.DECODE, query_wiring="absorbed",
                                 short_context_bypass=False, is_decode_graph=True,
                                 name="decode", rows=n, tokens=1)
    rows_t = torch.tensor(token_rows, dtype=torch.long)
    positions = torch.zeros((n,), dtype=torch.long)
    y = torch.zeros((n, 8))
    prev = SWITCHES.get("sel_fused")
    SWITCHES["sel_fused"] = "on"
    try:
        with use_lane(0, overlap=False, geometry=None, sync_free=True):
            steps = d._attention_gen(layer, y, positions, rows_t, positions + 1,
                                     call_shape=call_shape, visible_max=4)
            for _ in range(3):                 # ("kv", ..), ("comm", "scores"), ("comm", "gather")
                next(steps)
            steps.close()
    finally:
        if prev is None:
            SWITCHES.pop("sel_fused", None)
        else:
            SWITCHES["sel_fused"] = prev
    return captured


def test_site2_fused_handback_lcap_is_the_row_extent():
    cache = contract_cache()
    merged = torch.zeros((2, 4), dtype=torch.int64)          # unused on the fused path
    captured = drive_attention(cache, fused=True, merged=merged, token_rows=[7, 3])
    lcap = int(captured["handback"]["local_capacity"])
    # the rebind: cap_of(row) = 44,672 (uniform rows), not max_cap() = 131,072
    assert lcap == ROW_CAP == cache.cap_of(7) == cache.cap_of(3)
    assert lcap < cache.max_cap()
    # the kernel guard is `local < LCAP` (fused_ops.py:987): tail-row local
    # 44,672 is flat slot 357,376 — refused; 44,671 (flat slot 357,375)
    # accepted byte-identically (the old guard accepted it too)
    assert not (ROW_CAP < lcap)
    assert (ROW_CAP - 1) < lcap
    assert 7 * ROW_CAP + ROW_CAP == FLAT_SLOTS               # slot 357,376 never handed back
    assert 7 * ROW_CAP + (ROW_CAP - 1) == FLAT_SLOTS - 1     # the last in-extent slot
    # in-extent hand-back slots equal the old formula's (only the guard changed)
    for row in (7, 3):
        for j in (0, 1, ROW_CAP - 1):
            assert row * ROW_CAP + j == cache.base_of(row) + j
    # the base argument is unchanged (still the per-row slot bases)
    assert torch.equal(captured["handback"]["base"].reshape(-1),
                       cache.slots_for(torch.tensor([7, 3])).reshape(-1))


def test_site2_fused_lcap_never_passes_any_row_extent():
    cache = ragged_cache()          # caps [12, 8, 8, 4]
    merged = torch.zeros((2, 4), dtype=torch.int64)
    captured = drive_attention(cache, fused=True, merged=merged, token_rows=[0, 3])
    lcap = int(captured["handback"]["local_capacity"])
    assert lcap == 4 == min(cache.cap_of(r) for r in range(4))
    for row in range(4):
        assert lcap <= cache.cap_of(row)      # never admits a slot past ANY row


def test_site3_handback_bounds_are_per_row():
    cache = contract_cache()
    # row 7 (tail row): p=357,375 in-extent (local 44,671); p=357,376 is the
    # faulting slot (local 44,672 -> flat slot 357,376); row 3: p=999,999 is
    # far past its extent
    merged = torch.tensor([[357_375, 357_376, -1, 8],
                           [31, 999_999, -1, 0]], dtype=torch.int64)
    captured = drive_attention(cache, fused=False, merged=merged, token_rows=[7, 3])
    got = captured["mapped"].tolist()
    bases = {7: cache.base_of(7), 3: cache.base_of(3)}
    # the old guard (`local < max_cap()` = 131,072) admitted the faulting slot:
    # shifted = p + world * base, whose local form is exactly flat slot 357,376
    old_shifted = 357_376 + WORLD * bases[7]
    assert old_shifted // WORLD == FLAT_SLOTS
    for r, row in enumerate([7, 3]):
        for j, p in enumerate(merged[r].tolist()):
            local = max(p, 0) // WORLD
            if p < 0 or local >= cache.cap_of(row):
                assert got[r][j] == -1, f"out-of-extent pick handed back at row {row} p={p}"
            else:
                # in-extent: byte-identical to the old formula (only the guard changed)
                assert got[r][j] == p + WORLD * bases[row], f"in-extent pick changed at {row},{p}"
    assert got[0][0] == 357_375 + WORLD * bases[7]           # byte-identical in-extent
    assert got[0][1] == -1                                   # slot 357,376 derated


# --------------------------------------------------------------------------
# site 4: decoder.py local_slot_map (arange(max_cap) -> cap_of-based row)
# --------------------------------------------------------------------------
def test_site4_local_slot_map_is_cap_of_based():
    cache = contract_cache()
    token_rows = torch.tensor([7, 3], dtype=torch.long)
    m = cache.local_slot_map(token_rows)
    assert tuple(m.shape) == (2, ROW_CAP)                # was [rows, 131,072]
    assert m.shape[1] < cache.max_cap()
    # in-extent columns byte-identical to the old `base + arange(max_cap())`
    for r, row in enumerate([7, 3]):
        base = cache.base_of(row)
        assert torch.equal(m[r], base + torch.arange(ROW_CAP, dtype=torch.long))
    # slot 357,376 is not expressible: the widest value is the last in-extent
    # slot and no column reaches past the row (the old row-7 column 44,672 held
    # exactly flat slot 357,376)
    assert int(m.max()) == FLAT_SLOTS - 1
    assert int((m >= FLAT_SLOTS).sum()) == 0
    assert 7 * ROW_CAP + ROW_CAP == FLAT_SLOTS


def test_site4_local_slot_map_pads_out_of_extent_with_minus_one():
    cache = ragged_cache()          # caps [12, 8, 8, 4]
    m = cache.local_slot_map(torch.tensor([0, 3], dtype=torch.long))
    assert tuple(m.shape) == (2, 12)                     # widest real extent
    assert torch.equal(m[0], torch.arange(12, dtype=torch.long))
    assert torch.equal(m[1, :4], torch.tensor([28, 29, 30, 31], dtype=torch.long))
    assert bool((m[1, 4:] == -1).all())                  # columns past cap_of(3) derated
    assert bool((m[0, 12:] == -1).all()) if m.shape[1] > 12 else True
    # no cell hands back a slot past any row's extent
    for r, row in enumerate([0, 3]):
        live = m[r][m[r] >= 0]
        assert bool((live < cache.base_of(row) + cache.cap_of(row)).all())


# --------------------------------------------------------------------------
# site 5: spec.py prewarm loop start bound (g.capacity -> cap_of(row)*world)
# --------------------------------------------------------------------------
def old_prewarm_verify_starts(topk, k, world, bound, granule, geom_from, cap):
    """The pre-fix `SpeculativeGreedy.prewarm` while-loop start sequence."""
    start = max(topk, 0) + k + 2
    out = []
    while start + k + 1 < bound:
        out.append(start)
        start = (old_index_width(start + k + 2, granule, geom_from, cap) * world
                 + k + 2)
    return out


def test_site5_spec_prewarm_bound_is_the_row_extent(monkeypatch):
    from ignition.recipes.hy4.spec import SpeculativeGreedy

    cache = contract_cache()
    g = graphs_for(cache)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    recorded = []
    g.run_spec = lambda kind, *, row, start, count, call_shape, sample, filtered: recorded.append(
        (kind, row, int(start), int(count)))
    call_shape = SimpleNamespace(family=None, query_wiring="absorbed",
                                 short_context_bypass=False, is_decode_graph=True,
                                 name="decode", rows=1, tokens=1)
    decoder = SimpleNamespace(decode_graphs=g,
                              default_call_shape=lambda shape: call_shape)
    k, row = 3, 7                # the tail row: its slot 357,376 is the faulting one
    sg = SpeculativeGreedy(decoder, k=k, row=row)
    sg.eligible = lambda start, cursor: True      # isolate the LOOP BOUND (site 5)
    sg.prewarm(max_context=4_000_000, sample=False, filtered=False)

    old_bound = cache.max_cap() * WORLD            # 1,048,576: the old `g.capacity`
    old = old_prewarm_verify_starts(g.topk, k, WORLD, old_bound,
                                    g.granule, g.GEOM_FROM, cache.max_cap())
    new = [start for kind, r, start, count in recorded
           if kind == "verify"]
    # every staged forward stays inside the row's extent (no slot >= 357,376)
    for kind, r, start, count in recorded:
        assert r == row
        assert (start + count - 1) // WORLD < cache.cap_of(row), (
            f"{kind} at start {start} stages past cap_of(row)")
        assert cache.base_of(row) + (start + count - 1) // WORLD < FLAT_SLOTS
    # in-extent starts byte-identical to the old loop's; the out-of-extent
    # starts the old bound admitted (its staged slots reach flat slot 357,376
    # and beyond) are derated away
    assert new == [s for s in old if s + k + 1 < cache.cap_of(row) * WORLD]
    assert any((s + k) // WORLD >= cache.cap_of(row) for s in old), (
        "the old loop bound must admit starts past the row extent")
    assert max(new) + k < FLAT_SLOTS


# --------------------------------------------------------------------------
# pool-neutrality: storage sizing is untouched by the diff (the pool boom
# 357,376 slots/rank is contract)
# --------------------------------------------------------------------------
def test_pool_neutrality_arithmetic_untouched():
    cache = contract_cache()
    g = graphs_for(cache)
    # the pool arithmetic (Hy4Cache.local_capacity = capacity // world,
    # decoder.py:151-160 sizing) never reads the bound family
    assert cache.local_capacity == ROW_CAP == 44_672
    assert cache.rows * cache.local_capacity == FLAT_SLOTS == 357_376
    assert FLAT_SLOTS * RECORD_BYTES == 20_207_468_544        # KV pool bytes
    assert FLAT_SLOTS * INDEX_HEAD_DIM * 2 == 91_488_256      # index table (corrected figure)
    # the bound family's rebinding cannot change storage: it is a separate
    # attribute, and cap_limit / extents move it without moving the pool
    before = cache.rows * cache.local_capacity
    cache.cap_limit = 700_000
    cache.set_extents([8_192] * ROWS)
    assert cache.rows * cache.local_capacity == before == FLAT_SLOTS
    assert g.local_capacity == 8_192                        # the BIND moved...
    assert cache.local_capacity == ROW_CAP                  # ...the pool did not
