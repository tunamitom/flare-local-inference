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
}
DEFAULTS = {"index_scores": "fp32"}
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
