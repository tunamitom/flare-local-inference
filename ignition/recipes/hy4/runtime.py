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
    # consult18 P1 (host-bound prefill): one-kernel, bit-exact replacements of
    # eager op chains (fused_ops.py).
    "fused_rope": ("off", "on"),
    "fused_merge": ("off", "on"),
    # exact host trims: fused RMSNorm tail, cached FP32 router weight/bias,
    # no probe-only sort when the probe is off.
    "host_trim": ("off", "on"),
    # one multi-flag wait launch per CE exchange instead of one per source
    "batched_waits": ("off", "on"),
    # per-lane CUDA graphs of the MLP block inside the overlap
    "graph_moe": ("off", "on"),
    # the overlap's comm token (one collective in flight per rank) vs per-lane
    # ordering (each lane's collectives in its own order on its own channels)
    "comm_token": ("on", "off"),
    # single-lane forwards (prefill tails / single chunks / decode) without host
    # reads: host geometry, host-side bounds, deferred selection checks
    # "decode": one-token steps only (the measured win: +2.3 % at 16k c1; the
    # host-bound prefill tail is 0.8 % slower sync-free)
    "sync_free": ("off", "decode", "on"),
}
DEFAULTS = {"index_scores": "fp32",
            "ihc": os.environ.get("HY4_IHC") or "default",
            "fused_rope": os.environ.get("HY4_FUSED_ROPE") or "off",
            "fused_merge": os.environ.get("HY4_FUSED_MERGE") or "off",
            "host_trim": os.environ.get("HY4_HOST_TRIM") or "off",
            "batched_waits": os.environ.get("HY4_BATCHED_WAITS") or "off",
            "graph_moe": os.environ.get("HY4_GRAPH_MOE") or "off",
            "comm_token": os.environ.get("HY4_COMM_TOKEN") or "on",
            "sync_free": os.environ.get("HY4_SYNC_FREE") or "off"}
for _key, _value in DEFAULTS.items():
    if _value not in ALLOWED[_key]:
        raise ValueError(f"runtime default {_key}={_value!r} must be one of {ALLOWED[_key]}")
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
