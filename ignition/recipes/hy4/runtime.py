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
    # consult18 P2: CUDA graphs of the one-row decode step (decode_graph.py);
    # "on" replays a captured step (lazily captured per indexer width bucket)
    # instead of launching ~10k kernels from the host per token.
    "decode_graph": ("off", "on"),
    # consult18 P2: iHC post_pre below PREFILL_MIN_ROWS rows over (row, chunk)
    # grids (fused_ops.ihc_post_pre_split; bit-identical, 6x at one row)
    "ihc_split": ("off", "on"),
    # consult18 P2: the sync-free scores-only selection in one launch per row
    # (fused_ops.scores_only_select; bit-identical, 16 vs 229 us per decode row)
    "sel_fused": ("off", "on"),
    # consult18 P2: graph decode steps gather the logits to rank 0 on the host
    # (the sampler's input) instead of an all-gather moved back to the device
    "root_logits": ("off", "on"),
    # consult18 P2: the MoE route in one launch (fused_ops.moe_route; bit-identical)
    "route_fused": ("off", "on"),
    # consult18 P2 (numerics class): decode rows through the decode-capacity MoE
    # op (FusedRoutedExperts.wire_decode_op; b12x small-M split-K schedule)
    "moe_decode_op": ("off", "on", "alt"),
    # consult18 P2: single-row prefill forwards of T tokens (T % world == 0,
    # T <= 512: the prompt's tail) replay a captured graph (DecodeGraphs.run_tail)
    "prefill_graph": ("off", "on"),
    # consult18 P3 timing probe (garbage results): CE gather/LSE at half bytes
    "wire_probe": ("off", "ce_half"),
    # consult18 P3 (numerics class): the CE head gather / LSE partials over an
    # INT8 wire (one FP32 scale per row-head vector)
    "wire": ("bf16", "int8"),
    # consult18 P4: greedy (temperature 0) requests decode by MTP speculation —
    # draft k tokens with the MTP block, verify k + 1 in one target forward
    # (spec.py; numerics class: the verify forward is a T-token forward)
    "spec": ("off", "on"),
    # consult18 P5: temperature-only sampled requests speculate too (exact
    # speculative sampling over the vocabulary shards, spec_sample.py)
    "spec_sample": ("off", "on"),
    # consult18 P6: the overlapped pair's host step order (= device comm-token
    # order): strict alternation, or the lane with less compute before its
    # next collective first
    "overlap_order": ("alternate", "short_first"),
    # consult18 P7: top-k / top-p sampled requests speculate (device filter)
    "spec_filter": ("off", "on"),
    # consult18 P9 (round 8): prefill absorb_queries writes the absorbed GEMM
    # straight into the [rows, heads, kv_lora + rope] query (no cat; exact)
    "absorb_out": ("cat", "direct"),
    # consult18 P9 (numerics class): prefill-row MoE router logits by cuBLAS
    # BF16-in / FP32-out tensor cores instead of the FP32 SIMT GEMM
    "router_mm": ("f32", "bf16"),
    # consult18 P9: iHC post_pre chunk kernel with 8 rows per program at >= 512
    # rows (the chunk's fn vectors loaded once per 8 rows; bit-identical)
    "ihc_rows": ("1", "8"),
}
DEFAULTS = {"index_scores": "fp32",
            "ihc": os.environ.get("HY4_IHC") or "default",
            "fused_rope": os.environ.get("HY4_FUSED_ROPE") or "off",
            "fused_merge": os.environ.get("HY4_FUSED_MERGE") or "off",
            "host_trim": os.environ.get("HY4_HOST_TRIM") or "off",
            "batched_waits": os.environ.get("HY4_BATCHED_WAITS") or "off",
            "graph_moe": os.environ.get("HY4_GRAPH_MOE") or "off",
            "comm_token": os.environ.get("HY4_COMM_TOKEN") or "on",
            "sync_free": os.environ.get("HY4_SYNC_FREE") or "off",
            "decode_graph": os.environ.get("HY4_DECODE_GRAPH") or "off",
            "ihc_split": os.environ.get("HY4_IHC_SPLIT") or "off",
            "sel_fused": os.environ.get("HY4_SEL_FUSED") or "off",
            "root_logits": os.environ.get("HY4_ROOT_LOGITS") or "off",
            "route_fused": os.environ.get("HY4_ROUTE_FUSED") or "off",
            "moe_decode_op": os.environ.get("HY4_MOE_DECODE") or "off",
            "prefill_graph": os.environ.get("HY4_PREFILL_GRAPH") or "off",
            "wire_probe": "off",
            "wire": os.environ.get("HY4_WIRE") or "bf16",
            "spec": os.environ.get("HY4_SPEC") or "off",
            "spec_sample": os.environ.get("HY4_SPEC_SAMPLE") or "off",
            "overlap_order": os.environ.get("HY4_OVERLAP_ORDER") or "alternate",
            "spec_filter": os.environ.get("HY4_SPEC_FILTER") or "off",
            "absorb_out": os.environ.get("HY4_ABSORB_OUT") or "cat",
            "router_mm": os.environ.get("HY4_ROUTER_MM") or "f32",
            "ihc_rows": os.environ.get("HY4_IHC_ROWS") or "1"}
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
