"""consult18 E4: runtime A/B switches for local-compute levers.

A lever that changes only per-rank compute (no collective shape, no plan) can
be judged inside ONE boot: the serve reads `HY4_AB_FILE` (a JSON object) at
the start of every request, on every rank, so alternating cells measure the
lever without the boot-to-boot spread. The harness rewrites the file only
between requests. Unknown keys or values raise: a typo must not silently
measure the default.
"""
import json
import os

ALLOWED = {
    "index_scores": ("fp32", "bf16"),
    # consult18 layer-wise tuned plans: the iHC boundary plans at prefill row
    # counts — "default" (IHC_TUNING[6144]) or "tuned" (the measured best of
    # iHC's own tuning space at 1,024 rows; decode always keeps the default).
    "ihc": ("default", "tuned"),
}
DEFAULTS = {"index_scores": "fp32",
            "ihc": os.environ.get("HY4_IHC") or "default"}
if DEFAULTS["ihc"] not in ALLOWED["ihc"]:
    raise ValueError(f"HY4_IHC={DEFAULTS['ihc']!r} must be one of {ALLOWED['ihc']}")
SWITCHES = dict(DEFAULTS)


def load(path=None):
    """Refresh SWITCHES from the A/B file (no file: the defaults). Returns them."""
    path = path or os.environ.get("HY4_AB_FILE") or ""
    values = dict(DEFAULTS)
    if path and os.path.exists(path):
        with open(path) as fh:
            requested = json.load(fh)
        for key, value in requested.items():
            if key not in ALLOWED or value not in ALLOWED[key]:
                raise ValueError(f"HY4_AB_FILE {path}: {key}={value!r} is not an A/B switch")
            values[key] = value
    SWITCHES.clear()
    SWITCHES.update(values)
    return dict(SWITCHES)
