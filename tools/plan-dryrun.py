#!/usr/bin/env python3
# Planner dry-run for the K1+K2 graph-only re-export (PLAN-PERF-KV.md sections 1/2/4).
# CPU only: synthetic GPUs at the measured B/D free memory, no NVML and no CUDA calls.
# Invoked by export-kimi31-exl3.sh --check; safe to run standalone:
#
#   /home/tom/build/flare-k31exl3/.venv/bin/python k31-plan-dryrun.py [--capacity 20]
#
# Two sections print:
#   1. the Ignition planner gate (ignition.planning.plan) for the export shape
#      (c4 + 1,048,576 context, dflash2 K7, block verification, online MXFP8,
#      image-concurrency 1) at both --speculative-token-capacity variants, and
#   2. the boot KV supply projection ("Allocate cache" tokens) for capacity 20
#      and 24 with and without the K2 vision riders.
"""Planner and KV-supply dry-run for the K1+K2 export delta. Creates nothing."""
import argparse
import sys
from pathlib import Path

REPO = Path("/home/tom/build/levers-prefill")
sys.path.insert(0, str(REPO))

from ignition.planning import Gpu, plan  # noqa: E402
from ignition.components.capacity import packed_speculative_shapes  # noqa: E402
from ignition.recipes import resolve  # noqa: E402
from ignition.recipes.declaration import Speculation  # noqa: E402

GIB = 1 << 30
MIB = 1 << 20

# dflash2 TP8 RankCosts.state_row_bytes (ignition/recipes/kimi31/recipe.py); one
# speculative verification row per --speculative-token-capacity slot per rank.
STATE_ROW_BYTES = 65_761_536
# Measured live boot: cache budget 2.09 GiB / 888,832 KV tokens = 2,525 B per token
# per rank (boot log kimi31-exl3-boot-20261109-140444.log, "Allocate cache").
KV_TOKEN_BYTES = 2525
BASELINE_POOL_TOKENS = 888_832
# The live package's export shape the anchor above was measured at.
BASELINE_SPEC_CAPACITY = 32
BASELINE_PATCHES = 65_536
BASELINE_IMAGE_CONCURRENCY = 4
# Min NVML free across the eight B/D cards at the live boot (95.010 GiB).
DEFAULT_FREE_GIB = 95.010
# Runtime fixed state measured on that boot: Plan cache 92.16 GiB loaded per rank
# (weights + graphs + live rows incl. the 32 speculative rows); the remaining
# 0.76 GiB of NVML free is CUDA context + loaded kernels.
BASELINE_LOADED_GIB = 92.16


def vision_rider_bytes():
    """Bytes the K2 patch-cap and image-concurrency riders free per rank.

    Patch cap 65,536 -> 16,384 (PATCH_CAPACITIES in ignition/recipes/kimi3/vision.py):
    VisionInputs pixels 49,152 * 588 * 4, interpolation/merge indices 49,152 * 68,
    VisionScratch projected 49,152 * 1024 * 2 and attention_storage 49,152 * 1024 * 2
    (TP8 shape). Image-concurrency 4 -> 1: the feature arena keeps
    min(4, concurrency) slots of FEATURES * (HIDDEN // 8) bf16 each
    (ignition/components/vision_state.py VisionState).
    """
    dropped = BASELINE_PATCHES - 16_384
    patch = dropped * (588 * 4 + 68 + 1024 * 2 + 1024 * 2)
    arena = (BASELINE_IMAGE_CONCURRENCY - 1) * 16_384 * (7168 // 8) * 2
    return patch, arena


def planner_gate(snapshot, capacity, free_bytes, prefill_fp8=False, prefill_island=False):
    """Fit the export shape on eight synthetic B/D cards; return the plan."""
    model = resolve(str(snapshot))
    gpus = tuple(Gpu(index, "GB200-class B/D card", f"GPU-DRYRUN-{index}", 96 * GIB, free_bytes,
                     (10, 3)) for index in range(8))
    speculation = Speculation(7, True, token_capacity=capacity, packed_verification=True,
                              graph_budget=16, draft_head_quantization="mxfp8",
                              method="dflash2", block_verification=True)
    return model, plan(model.recipe, model.checkpoint, gpus, selected=tuple(range(8)),
                       concurrency=4, reserve_bytes=0, speculation=speculation,
                       prefill_chunk=3072, image_concurrency=1, online_mxfp8=True,
                       prefill_fp8_transport=prefill_fp8,
                       prefill_island_reduce=prefill_island)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", default="/opt/models/kimi31-exl3-snapshot/Kimi-K3.1-Pro-EXL3-2bit")
    parser.add_argument("--capacity", type=int, default=20, choices=(20, 24),
                        help="armed --speculative-token-capacity (20 primary, 24 fallback)")
    parser.add_argument("--prefill-fp8", action="store_true",
                        help="price the PREFILL_FP8=1 shape (--prefill-fp8-transport)")
    parser.add_argument("--prefill-island", action="store_true",
                        help="price the PREFILL_ISLAND=1 shape (--prefill-island-reduce)")
    parser.add_argument("--free-gib", type=float, default=DEFAULT_FREE_GIB)
    args = parser.parse_args()
    free_bytes = int(args.free_gib * GIB)

    print("== planner gate (c4 + recipe context = pool/3 per owner directive, dflash2 K7 + block verification, online MXFP8)")
    model = None
    ok = True
    for capacity in (20, 24):
        model, fitted = planner_gate(args.snapshot, capacity, free_bytes, prefill_fp8=args.prefill_fp8,
                                    prefill_island=args.prefill_island)
        ok = ok and fitted.fits
        armed = " <- armed" if capacity == args.capacity else ""
        print(f"  capacity {capacity}{armed}: fits={fitted.fits}"
              f" weights {fitted.weights_bytes / GIB:.3f} + graphs {fitted.graph_bytes / GIB:.3f}"
              f" + live {fitted.live_bytes / GIB:.3f} ({fitted.live_rows} rows"
              f" x {fitted.costs.state_row_bytes:,} B) = fixed {fitted.fixed_bytes / GIB:.3f} GiB"
              f" + min pages {fitted.minimum_pages * fitted.costs.page_bytes / GIB:.3f}"
              f" <= supply {fitted.supply_bytes / GIB:.3f} GiB"
              f" (free {args.free_gib:.3f} - headroom {fitted.headroom_bytes / GIB:.1f}"
              f" - reserve {fitted.reserve_bytes / GIB:.1f});"
              f" planner pages {fitted.pages} ({fitted.tokens:,} tokens, gate estimate only);"
              f" spec capacity {fitted.speculative_token_capacity},"
              f" graph budget 16 -> {len(packed_speculative_shapes(4, 8, capacity, 16))} packed shapes")
    print("  context is the recipe's pool/3 value (owner directive, W6); boot KV is sized"
          " from measured free memory (KvCacheLimit::Memory). The obsolete 1M gate is"
          " replaced by max_model_len / Allocate-pool in [0.28, 0.36] at the boot.")
    if not ok:
        print("PLANNER GATE REFUSED the c4 + 1M shape: do not launch", file=sys.stderr)
        return 1

    patch, arena = vision_rider_bytes()
    print("== boot KV supply projection (anchor: live boot"
          f" {BASELINE_POOL_TOKENS:,} tokens at capacity {BASELINE_SPEC_CAPACITY},"
          f" {BASELINE_PATCHES:,} patches, image-concurrency {BASELINE_IMAGE_CONCURRENCY};"
          f" {KV_TOKEN_BYTES} B of KV per token per rank)")
    rows = [("K1 capacity 20 (primary)", (BASELINE_SPEC_CAPACITY - 20) * STATE_ROW_BYTES, 0),
            ("K1 capacity 24 (fallback)", (BASELINE_SPEC_CAPACITY - 24) * STATE_ROW_BYTES, 0),
            ("K1 20 + K2 riders", (BASELINE_SPEC_CAPACITY - 20) * STATE_ROW_BYTES, patch + arena),
            ("K1 24 + K2 riders", (BASELINE_SPEC_CAPACITY - 24) * STATE_ROW_BYTES, patch + arena)]
    for label, spec_freed, rider_freed in rows:
        freed = spec_freed + rider_freed
        pool = BASELINE_POOL_TOKENS + freed // KV_TOKEN_BYTES
        loaded = BASELINE_LOADED_GIB - freed / GIB
        print(f"  {label:24s} freed {freed / GIB:5.3f} GiB/rank -> loaded {loaded:6.3f} GiB"
              f" -> Allocate cache ~{pool:,} KV tokens"
              f" ({pool / 1e6:.2f}M, context/pool ratio {310272 / pool:.3f})")
    print(f"  K2 rider split: patch cap {patch / GIB:.3f} GiB"
          f" ({patch // KV_TOKEN_BYTES:,} tokens) + image-concurrency 1"
          f" {arena / GIB:.3f} GiB ({arena // KV_TOKEN_BYTES:,} tokens);"
          " bf16 pixels NOT taken (frontend stages f32 bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
