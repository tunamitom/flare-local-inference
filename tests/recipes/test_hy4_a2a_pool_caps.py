"""test_hy4_a2a_pool_caps.py — CPU unit test for the Step 5a dead-A2A pool row
caps (PERF-PLAN.md:380-383). No GPU, no torch.distributed: the cap decision is
pure arithmetic (`collectives.a2a_pool_row_caps`).

Contract (CHUNK = 2048, ce_rows = 1024, PREFILL_WIDTHS = 2, 3, 4 .. CHUNK):

1. `pools[(576, "gather")] -> max(DECODE_ROWS, 1) = 16` ALWAYS — absorbed
   queries are decode/verify/batch only (unconditionally safe, audit included).
2. `pools[(256, "gather")]` and `pools[(256, "partials")] -> 512` (the largest
   prepared prefill width below `ce_rows`) when the CE exchange is active
   (`ce_gather_rows`/`ce_lse_rows` set) AND `HY4_CE_AUDIT` is unset — and the
   caller skips `prepare_gather`/`prepare_lse` plan binding at widths
   `>= ce_rows` (`skip_prepare_above == 1024`).
3. `256 -> CHUNK` (uncapped: absent from `caps`, the sizing falls back to
   `max_rows`) when the audit is on — `HY4_CE_AUDIT=1` runs the SM-kernel arm
   at every CE width and needs the full pools + bound plans.
4. The audit + explicitly forced caps combination raises loudly (mirroring the
   `HY4_AR_AUDIT` fail-loud at `h4_serve.py:2810`).
"""
import pytest

from ignition.recipes.hy4.collectives import a2a_pool_row_caps

CHUNK = 2048
PREFILL_WIDTHS = (2, 3, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)
DECODE_ROWS = (2, 3, 4, 5, 6, 8, 12, 16)


def _caps(**kw):
    args = dict(absorbed_width=576, unabsorbed_width=256, expanded_width=256,
                decode_rows=DECODE_ROWS, prefill_widths=PREFILL_WIDTHS,
                ce_rows=1024, audit=False)
    args.update(kw)
    return a2a_pool_row_caps(**args)


def test_absorbed_gather_caps_at_decode_rows_always():
    """(1) pools[(576, "gather")] -> 16 regardless of CE/audit state."""
    for audit in (False, True):
        for ce_rows in (None, 1024):
            for force_caps in (False, True):
                if audit and force_caps:
                    continue    # that combination raises loudly (test 4)
                caps = _caps(audit=audit, ce_rows=ce_rows, force_caps=force_caps)["caps"]
                assert caps[(576, "gather")] == 16, (audit, ce_rows, force_caps)


def test_absorbed_gather_caps_at_decode_rows_default():
    """`max(DECODE_ROWS, default=1)`: an empty decode set caps at one row."""
    caps = _caps(decode_rows=())["caps"]
    assert caps[(576, "gather")] == 1


def test_256_pools_cap_below_ce_rows_when_ce_active_and_audit_off():
    """(2) CE active, audit off: 256 gather + partials -> 512, plans skipped
    at widths >= 1024."""
    out = _caps(audit=False, ce_rows=1024)
    assert out["caps"][(256, "gather")] == 512
    assert out["caps"][(256, "partials")] == 512
    assert out["skip_prepare_above"] == 1024


def test_256_pools_keep_chunk_when_audit_on():
    """(3) audit on: no 256 caps (the pool sizing keeps CHUNK = max_rows) and
    nothing is skipped (the audit's kernel arm needs the bound plans)."""
    out = _caps(audit=True, ce_rows=1024)
    assert (256, "gather") not in out["caps"]
    assert (256, "partials") not in out["caps"]
    assert out["skip_prepare_above"] is None
    # the sizing fallback semantics: absent key -> max_rows = CHUNK
    assert out["caps"].get((256, "gather"), CHUNK) == CHUNK
    assert out["caps"].get((256, "partials"), CHUNK) == CHUNK


def test_no_ce_no_256_caps():
    """Without CE coverage nothing is dead: only the 576 cap applies."""
    out = _caps(audit=False, ce_rows=None)
    assert (256, "gather") not in out["caps"]
    assert (256, "partials") not in out["caps"]
    assert out["skip_prepare_above"] is None


def test_audit_with_forced_caps_raises_loudly():
    """(4) HY4_CE_AUDIT=1 + explicit caps is unsound and must fail loudly."""
    with pytest.raises(ValueError, match="HY4_CE_AUDIT"):
        _caps(audit=True, ce_rows=1024, force_caps=True)


def test_force_caps_without_audit_still_caps():
    """The explicit cap request is honored when the audit is off."""
    out = _caps(audit=False, ce_rows=1024, force_caps=True)
    assert out["caps"][(256, "gather")] == 512
    assert out["skip_prepare_above"] == 1024


def test_expanded_wire_off_skips_partials_cap():
    """`expanded_width=None` (latent partial wire): no (256, "partials") pool
    exists, so no cap for it; the 256 gather still caps."""
    out = _caps(expanded_width=None, audit=False, ce_rows=1024)
    assert (256, "partials") not in out["caps"]
    assert out["caps"][(256, "gather")] == 512


def test_ce_rows_at_chunk_cap_is_one_row():
    """`ce_rows` at or below every prepared width: the pools serve no prefill
    width at all and cap at one row (the `w == 1` latent merge only)."""
    out = _caps(audit=False, ce_rows=2)
    assert out["caps"][(256, "gather")] == 1
    assert out["skip_prepare_above"] == 2
