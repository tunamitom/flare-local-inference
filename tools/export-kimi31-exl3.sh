#!/usr/bin/env bash
# kimi31-exl3-live: THE single branch that reproduces the CURRENT RUNNING SERVE —
# package AND image (W13 live-branch rule).
#   Package: gb202-tp8-202611100809 (E1.1, smart on :8001) = this merged tree
#   (EXPECT_HEAD = the E1.1 export lineage b8dcebb5 merged with the chunkfix image
#   lineage 5585a9a7) + flashinfer levers/prefill-shapes (0aa09dd61, auto-cloned
#   from the fork if missing). Export config: ring rows 1024 + BF16 + trellis +
#   island, split-K decode (B12X_W4A16_SMALL_M_SPLITK=1, B12X_PCIE_DMA_PIECES
#   unset), recipe context 396,288 (W8/J-4 section 4).
#   Image: local/flare:rt-k31exl3-chunkfix-5585a9a7 (container kimi31-exl3) — the
#   chunkfix lineage @ 5585a9a7 on the 35134e2b image tree, built from this same
#   merged tree by --image, pinned by image ID
#   sha256:50bda5a7f78e113cee8a7e566f156ecd50f307408a401f1d3256dd338dd0db0c.
#   Reproduce:
#     bash tools/export-kimi31-exl3.sh --check   # guards + planner dry-run, creates nothing
#     bash tools/export-kimi31-exl3.sh --image   # the runtime image (docker build, pinned by ID)
#     bash tools/export-kimi31-exl3.sh --detach  # the export (graph-only --reuse-weights)
#     REUSE_WEIGHTS=0 … --detach                 # FULL export without --reuse-weights
#                                                # (weights from the source checkpoints; see README.md)
#   Share story + serve copies: README.md, tools/serve/README.md.
#   Requires the EXL3-2bit checkpoint + DFlash2 draft at the CKPT/DRAFT paths below (or
#   override via env) and the box's 8 B+D GPUs.
# ─────────────────────────────────────────────────────────────────────────────
# Kimi-K3.1-Pro EXL3-2bit → FLARE TP8 package on switches B+D (TP8-CONFIG-PLAN.md §5.5, revision 3).
# Tree: /home/tom/build/flare-k31exl3 @ kimi31-exl3-dma (release 01327bf2 + block verification),
# FlashInfer 6a842a7d (editable, leads PYTHONPATH). GPUs resolved topologically (B then D) by
# /opt/mimo-pro/flare-bd/gpu-select.py. The watchdog only logs: a running export is never interrupted.
#
# WHAT THIS DRIVER REPRODUCED AT THE LEVER W1 SYNC (history, 2026-11-10): the live
# package gb202-tp8-202611100247 exactly — the defaults below are the measured config:
#   EXPECT_HEAD 2f0acabf        kimi31-exl3-dma branch head (BF16 DMA ring + split-K decode)
#   PREFILL_ISLAND=1            --prefill-island-reduce (L1) armed
#   IMAGE_CONCURRENCY=1         owner rider (was 4 at revision 3)
#   ignition env: IGNITION_KIMI_PREFILL_DMA_ROWS=2560 IGNITION_KIMI_PREFILL_DMA_BF16=1
#     (BF16 hop-sum copy-engine ring on prefill graphs >= 2560 rows, island below)
#     B12X_W4A16_SMALL_M_SPLITK=1 (split-K EXL3 expert kernel decode)
# Measured on that package (FABLE-PERF-LEVERS.md §5): prefill 8k/32k 1,718/2,159 tok/s,
# c1@8k 79.74 t/s, pool 931,840, gates pass at recall P 0.969.
#
# Revision 3 base = the K1+K2 graph-only re-export (PLAN-PERF-KV.md §4, EXPORT-DELTA.md):
#   K1  --speculative-token-capacity 32 -> SPEC_TOKEN_CAPACITY (20 primary, 24 fallback)
#   K2  PATCH_CAPACITIES capped at 16,384 (kimi3/vision.py); image-concurrency rider now folds
#       --image-concurrency 1 in (owner 2026-11-09, CAMPAIGN-STATE); the serving image from
#       35134e2b auto-scales over-cap photos to the admission cap
# Weights are unchanged: REUSE_WEIGHTS=1 (default) passes --reuse-weights, so the export
# costs graphs and kernels only. A runtime preview (--mtp-max-verification-token-budget 20)
# settles 20-vs-24 before launch; both variants are documented in EXPORT-DELTA.md.
#
#   ./export-kimi31-exl3.sh --check    guards + planner dry-run (both capacity variants),
#                                      prints the exact command, creates nothing
#   ./export-kimi31-exl3.sh --image    build the runtime image from this tree, pinned
#                                      by image ID (docker build; no GPU, no export)
#   ./export-kimi31-exl3.sh --link     populate the snapshot dir from the verified checkpoint (needs the .done marker)
#   ./export-kimi31-exl3.sh --detach   run under a transient system unit as tom (survives the turn)
#   ./export-kimi31-exl3.sh --run      run in the foreground (what --detach executes)
set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# W6 E1 export (J-3 ruled config, levers/j3-fable-review.md §1) — derived from
# the W1-synced export-kimi31-exl3.sh with these changes only:
#   REPO/FI      -> worktrees levers-prefill-w6 @ 160832b9 (= 5ed0defe + owner
#                   context commit) / levers-prefill-fi @ 0aa09dd61
#   plan flags   -> --prefill-dma-ring-rows 1024 --prefill-dma-bf16 --trellis-prefill-shapes
#                  (+ --prefill-island-reduce kept, PREFILL_ISLAND=1)
#   env          -> B12X_W4A16_SMALL_M_SPLITK=1 as shipped; B12X_PCIE_DMA_PIECES UNSET
#                  (verified at --run); NO IGNITION_KIMI_PREFILL_DMA_* (plan flags instead)
#   max context  -> W6 OWNER DIRECTIVE: recipe context = ~1/3 of the pool on the
#                  1,024 grid; the 1M gate is obsolete, the gate is
#                  max_model_len / actual Allocate pool in [0.28, 0.36].
#                  E1 shipped the W6 value; E1.1 (W8, J-4 section 4) shipped
#                  396,288 (387 x 1,024 = 129 x 3,072) and this merged tree carries
#                  the E1.1 recipe context 396,288 (guard below asserts it).
# R = 1024 decided by W6 step 0 (levers/evidence/w6/step0-shapes.log): ring wins
# 29.7-56.3% at every capacity 1024..3072 in the 185*t(7168)+185*t(3584) rule.
# ─────────────────────────────────────────────────────────────────────────────
SELF=$(readlink -f "${BASH_SOURCE[0]}"); HERE=$(dirname "$SELF")
REPO="${REPO:-$(cd "$HERE/.." && pwd)}"                      # this repo (the branch)
FI="${FI:-$REPO/../flashinfer-k31exl3-live}"                 # auto-cloned at the pin below
EXPECT_FI_BRANCH="${EXPECT_FI_BRANCH:-kimi31-exl3-live}"   # same single-branch name on the flashinfer fork
if [ ! -e "$FI/flashinfer/__init__.py" ]; then
  git clone --branch "$EXPECT_FI_BRANCH" https://github.com/tunamitom/flashinfer "$FI"
fi
FI=$(readlink -f "$FI")   # normalize (the default path carries a literal '..'; the resolve guard compares real paths)
PYV="${PYV:-$REPO/.venv/bin/python}"; [ -x "$PYV" ] || PYV="$(command -v python3)"
RING_ROWS="${RING_ROWS:-1024}"
# Spare card for preflight rider traces: 11:00.0 (GPU identity = switch port +
# PCI BDF; index 0). Work happens under flock /tmp/gpu-spare.lock and leaves it clean.
SPARE_GPU="${SPARE_GPU:-GPU-253e168e-256f-4457-82f4-d26fd2f3c0bd}"
# ─────────────────────────────────────────────────────────────────────────────
# W22 (TP9 transplant, levers/w22-handover.md): the combined graph-only export.
# Code tip ac377ae4 = kimi31-exl3-live + row-5 direct decode table (upstream
# #94/#95 + --trellis-prefill-table), row-1 paired-gather push recipe (fn1),
# the fused-pack + compaction riders (W11 ports), and the J-D row-1 policy
# notes (recipe + branch notes). flashinfer tip 5db9c10bc = rows 1/3/4 picks
# + the policy note at the push switch. Rows 2/6/7/10 dropped, row 4's LSE
# push is in-tree but OFF (speed rule failed on B+D).
#   TRELLIS_TABLE=2   row-5 direct decode table (2 = 48-row blocks 1536..2560 +
#                     the >=2560 128-thread geometry; 1 = 48-row band only)
#   DECODE_FUSION=1   the fused decode score/ID pack rider (bit-equal)
#   R1_PUSH / R3_ISLAND / R3_ACQUIRE: transport switches of the surviving
#   decode rows (recorded per the W22 gate; switch off = row out). R3_ACQUIRE
#   is part of row 3's identity+speed-tested configuration (J-D section 4).
# E1.1 (gb202-tp8-202611100809) remains the documented fallback package.
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# W23 (TP9 transplant final export, levers/tp9-jG-review.md section 6):
# MAX_ROWS 8 -> 20 (lever 1: rows 9..20 route through the pushed pair gather +
# the standalone top-16; flare b4889fe5, flashinfer 89ac9418e). The J-G rider
# switches are ALL UNSET per the W23-C verdicts:
#   B12X_TRELLIS_INKERNEL_ROUTE_PACK / B12X_TRELLIS_FLOW_FLAGS — rider (a)
#     FAILED its 20-row condition (bit-exact but +10.21 us per layer vs the
#     1 us bar, w23c-rider-a20.log; "if it fails either, the rider is off
#     entirely" — tp9-jG-review.md section 2).
#   B12X_A16_SPLITK_REDUCE_PDL — rider (c) DROPPED by its precondition: the
#     switch enters the dense compile key and with it the selection-cache key
#     (candidate_program_identity), so every dense blockscaled key re-races
#     (tools/preflight/pdl-compile-key-probe.py; section 4).
#   KDA fused verify stays OFF (rider (b) HOLD, W23-A verdict 4); the lever-2
#   knobs (B12X_MXFP8_A16_DECODE_W4_TK64 / _MAX_ROWS) stay UNSET (killed).
# The pre-authorized FALLBACK is the W22 package gb202-tp8-202611110549 (never
# deleted; revert to it and report if a tripwire trips on a new switch). E1.1
# (gb202-tp8-202611100809) remains behind it.
# J-H (levers/tp9-jH-review.md, fable): the first W23 export (package
# gb202-tp8-202611110758) DISCARDED — the rider (a)+(c) commits alone moved the
# dense/expert compiled-program cache keys with every switch off (the rider-(c)
# _DenseGemmPolicy field and rider-(a)'s fused-kernel cache-key fields), which
# re-raced the dense set (172/185 selections moved, w23c-seldiff.json). The
# rider ports are reverted on the live flashinfer branch (c3f3912a2 = 5db9c10bc
# + the top-16 row cap 20 + the inert KDA verify API); the key probe shows the
# flagship dense launch hash byte-identical to W22 (88ff1a2db736f1d1). This
# export must show the dense set on race-cache hits only; a few new measured
# entries for lever 1's route step are expected (fable J-H), any measured dense
# entry means stop and consult.
# ─────────────────────────────────────────────────────────────────────────────
TRELLIS_TABLE="${TRELLIS_TABLE:-2}"
DECODE_FUSION="${DECODE_FUSION:-1}"
R1_PUSH="${R1_PUSH:-1}"
R3_ISLAND="${R3_ISLAND:-1}"
R3_ACQUIRE="${R3_ACQUIRE:-1}"
EXPECT_HEAD="${EXPECT_HEAD:-b4889fe5}"          # W23 code tip (MAX_ROWS 20, lever 1)
EXPECT_FI="${EXPECT_FI:-c3f3912a2}"             # W23 flashinfer tip (top-16 cap 20; J-H: rider (a)+(c) ports reverted)
# W13 lineage pins (re-pinned at W22): the tree the running serve's package is
# produced from must match the pinned export tree for every export-consumed
# file (guards below). The image inputs are unchanged since 5585a9a7.
EXPECT_EXPORT_SRC="${EXPECT_EXPORT_SRC:-b4889fe5}"   # export tree of the W23 package
EXPECT_IMAGE_SRC="${EXPECT_IMAGE_SRC:-5585a9a7}"     # image build tree (chunkfix on 35134e2b)
# W13 image step (--image): the runtime image the serve runs, built from this
# tree with the campaign build path (docker build -t local/flare:rt-<variant>-<sha>
# <tree>, Dockerfile rust:1.94.1-slim-trixie build -> debian:trixie-slim) and
# pinned by image ID, exactly like the launcher pins its image.
IMAGE_NAME="${IMAGE_NAME:-local/flare}"
IMAGE_VARIANT="${IMAGE_VARIANT:-rt-k31exl3-chunkfix}"
LIVE_IMAGE_TAG="${LIVE_IMAGE_TAG:-rt-k31exl3-chunkfix-5585a9a7}"   # tag of the running serve's image
EXPECT_IMAGE_ID="${EXPECT_IMAGE_ID:-sha256:50bda5a7f78e113cee8a7e566f156ecd50f307408a401f1d3256dd338dd0db0c}"
# File set the runtime image build consumes (Dockerfile COPY inputs + licenses).
IMAGE_PATHS=(Dockerfile flare helios eos selene selene-manager
  ignition/recipes/kimi3/assets ignition/recipes/qwen38/assets ignition/recipes/step5/assets
  LICENSE NOTICE)
CKPT=/storage/models/Kimi-K3.1-Pro-EXL3-2bit
DONE=/storage/models/kimi-k31-exl3-download.done
FAIL=/storage/models/kimi-k31-exl3-download.failed
SNAP=/opt/models/kimi31-exl3-snapshot/Kimi-K3.1-Pro-EXL3-2bit
DRAFT=/opt/models/kimi31-draft-dflash2
CONFIG=/home/tom/.config/flare/kimi31-exl3.toml
BLOBS=/opt/models/.flare/packages/blobs
PKG_LINK=/opt/models/Kimi-K3.1-Pro-EXL3-TP8
OUT=/home/tom/kimi31-exl3
TS="${EXPORT_TS:-$(date +%Y%m%d-%H%M%S)}"
LOG="${EXPORT_LOG:-$OUT/export-$TS.log}"
MIN_FREE_G=800            # package ≈ 710 GiB + margin; nothing is deleted to make room
# K1 (PLAN-PERF-KV.md §4): speculative verification rows are 65,761,536 B per rank per
# capacity slot. 20 frees 0.735 GiB/rank -> +0.31M KV tokens (pool ~1.20M); the 24
# fallback frees 0.49 GiB -> +0.21M (~1.10M). The runtime preview
# (--mtp-max-verification-token-budget 20, restart only) settles 20-vs-24 before launch.
SPEC_TOKEN_CAPACITY="${SPEC_TOKEN_CAPACITY:-20}"
REUSE_WEIGHTS="${REUSE_WEIGHTS:-1}"   # 1 = graph-only re-export (--reuse-weights, weights unchanged)
# P3′ rider (REVIEW-J4 round 1): PREFILL_FP8=1 arms the fp8-transport two-shot for the
# prefill all-reduce (--prefill-fp8-transport, ignition). DEFAULT OFF: the exact BF16
# owner sums (and every measured cell) stay byte-identical; the flag is the rollback
# boundary and must stay off unless the numeric gate passed (P3PRIME-IMPL.md).
PREFILL_FP8="${PREFILL_FP8:-0}"
# L1 rider (REVIEW-J6): PREFILL_ISLAND=1 arms the TP8 two-island prefill all-reduce
# (--prefill-island-reduce, ignition; kernel commit in flashinfer 78b884989).
# DEFAULT ON since the live package gb202-tp8-202611100247 (lever W1 sync): V2 was
# killed and L1 is measured on the DMA-ring package (FABLE-PERF-LEVERS.md §1.2).
# PREFILL_ISLAND=0 is the rollback boundary back to the exact BF16 owner sums.
PREFILL_ISLAND="${PREFILL_ISLAND:-1}"
# Image-concurrency rider (owner 2026-11-09, CAMPAIGN-STATE): folded in at 1 for
# gb202-tp8-202611100247 (lever W1 sync). IMAGE_CONCURRENCY=4 restores the
# revision-3 shape (owner's original 4-photo batch). The planner dry-run prices
# both.
IMAGE_CONCURRENCY="${IMAGE_CONCURRENCY:-1}"
# 2026-11-09 pivot (owner): max context 1M + admission >= 4 + max KV. The planner's
# conservative MXFP8 savings credit (37.5%) was calibrated to the measured 47.9%
# (_exl3_mxfp8_savings; author's package 88.03 GiB/rank) and the kimi31 recipe
# context was raised 262144 -> 1048576 (engine + checkpoint cap; TP16 served 1M).
# With the calibrated credit c4 fits: fixed 92.84 + min pages 0.065 <= supply 94.01.
# See TP8-CONFIG-PLAN.md §11 and OFFLOAD-DESIGN.md.
# K2: PATCH_CAPACITIES 16,384 only. The 4 -> 1 image-concurrency rider was
# re-confirmed by the owner 2026-11-09 evening (CAMPAIGN-STATE) and rides the next
# export as IMAGE_CONCURRENCY=1 (J4 round 1); the default 4 keeps revision-3 shape.
EXPORT_ARGS=(--config "$CONFIG" --repo-id local-inference-lab/Kimi-K3.1-Pro-EXL3-2bit
  --gpus 0-7 --max-concurrency 4 --image-concurrency "$IMAGE_CONCURRENCY" --prefill-chunk-size 3072
  --speculative-method dflash2 --mtp-draft-tokens 7 --speculative-token-capacity "$SPEC_TOKEN_CAPACITY" --speculative-graph-budget 16
  --block-verification
  --online-mxfp8 --draft-head-quantization mxfp8
  --prefill-dma-ring-rows "$RING_ROWS" --prefill-dma-bf16 --trellis-prefill-shapes
  --autotune-effort default --collective-tuning static --compile-workers 8 --verbose)
if [ "$REUSE_WEIGHTS" = 1 ]; then EXPORT_ARGS+=(--reuse-weights); fi
if [ "$PREFILL_FP8" = 1 ]; then EXPORT_ARGS+=(--prefill-fp8-transport); fi
if [ "$PREFILL_ISLAND" = 1 ]; then EXPORT_ARGS+=(--prefill-island-reduce); fi
# W22: the row-5 direct decode table and the fused-pack rider.
if [ "$TRELLIS_TABLE" != 0 ]; then EXPORT_ARGS+=(--trellis-prefill-table "$TRELLIS_TABLE"); fi
if [ "$DECODE_FUSION" = 1 ]; then EXPORT_ARGS+=(--decode-fusion); fi

MODE="${1:---check}"
say() { echo "$*" >&2; }
fail=0
chk() { if "$@"; then return 0; else fail=$((fail+1)); return 1; fi; }

# ── W22/W23 fingerprint registry (shared by the post-export assertion and the
# ── preflight PRE-mode check). Key: TRELLIS_TABLE/DECODE_FUSION/R1_PUSH/
# ── R3_ISLAND/R3_ACQUIRE @ the export tree tip (EXPECT_EXPORT_SRC) at capacity
# ── 20, image 1, ring 1024. The W22 switches are launch fusions and move the
# ── round-graph/kernel count (the decode-fusion detector); a knob config with
# ── no registered baseline cannot pass the post-export assertion — the
# ── preflight refuses to start for it.
# ──   2/1/1/1/1@ac377ae4  W22 ruled config (gb202-tp8-202611110549; E1.1's
# ──                       4967 superseded: -63 kernels = -63 modules, -1.83
# ──                       MiB cubin, graphs 66 and slab/index_slab unchanged)
# ──   0/0/0/0/0@*         E1.1 (gb202-tp8-202611100809)
# ──   2/1/1/1/1@b4889fe5  W23 ruled config (MAX_ROWS 20), registered from the
# ──                       settled export gb202-tp8-202611110845 (export-
# ──                       20261111-152641.log, all 8 ranks): 4900 kernels, -4
# ──                       vs W22's 4904 — the route step at rows 9..20 replaces
# ──                       general-path route kernels with the pushed pair
# ──                       gather + standalone top-16. (The first W23 export of
# ──                       this config showed a 4980-4982 per-rank spread: the
# ──                       five then-new route-step latent_down_rows races left
# ──                       per-shard compile artifacts in the tail inventory;
# ──                       with those selections cached the count is uniform.)
fp_baseline() {
  case "$TRELLIS_TABLE/$DECODE_FUSION/$R1_PUSH/$R3_ISLAND/$R3_ACQUIRE@$EXPECT_EXPORT_SRC" in
    2/1/1/1/1@ac377ae4) echo "Graphs: 66; kernels: 4904" ;;
    0/0/0/0/0@*) echo "Graphs: 66; kernels: 4967" ;;
    2/1/1/1/1@b4889fe5) echo "Graphs: 66; kernels: 4900" ;;
    *) echo "" ;;
  esac
}

# ── PRE-mode settings assertion: the knob env must match the rulings before
# ── any GPU work starts. Sources: levers/tp9-jD-review.md section 4 (rows 1/3/4/5
# ── + riders) and levers/tp9-jE-review.md (admission 4, image rider 1).
assert_pre_settings() {
  local bad=0 kv k v
  for kv in TRELLIS_TABLE=2 DECODE_FUSION=1 R1_PUSH=1 R3_ISLAND=1 R3_ACQUIRE=1 \
            PREFILL_FP8=0 PREFILL_ISLAND=1 IMAGE_CONCURRENCY=1 RING_ROWS=1024; do
    k="${kv%%=*}"; v="${kv##*=}"
    [ "${!k}" = "$v" ] || { say "PRE-SETTINGS: $k=${!k} deviates from the ruling ($kv) — tp9-jD-review.md §4 / tp9-jE-review.md; consult before export"; bad=$((bad+1)); }
  done
  case "$SPEC_TOKEN_CAPACITY" in 20|24) : ;;
    *) say "PRE-SETTINGS: SPEC_TOKEN_CAPACITY=$SPEC_TOKEN_CAPACITY must be 20 (primary) or 24 (K1 fallback)"; bad=$((bad+1));;
  esac
  # Row 4's LSE reduce-scatter push stays OFF (speed rule failed).
  case "${B12X_PCIE_DCP_LSE_PUSH_TP8:-0}" in 0|"") : ;;
    *) say "PRE-SETTINGS: B12X_PCIE_DCP_LSE_PUSH_TP8=${B12X_PCIE_DCP_LSE_PUSH_TP8} — row 4 stays OFF (J-D §4)"; bad=$((bad+1));;
  esac
  # W23-C rider verdicts (tp9-jG-review.md sections 2/4): all three J-G rider
  # switches stay UNSET. Rider (a) failed its 20-row condition (bit-exact but
  # +10.21 us/layer vs the 1 us bar -> "the rider is off entirely"); rider (c)
  # enters the dense compile key and re-races the dense selections (proof:
  # tools/preflight/pdl-compile-key-probe.py). Lever-2 knobs stay unset too.
  case "${B12X_TRELLIS_INKERNEL_ROUTE_PACK:-}:${B12X_TRELLIS_FLOW_FLAGS:-}:${B12X_A16_SPLITK_REDUCE_PDL:-}:${B12X_MXFP8_A16_DECODE_W4_TK64:-}:${B12X_MXFP8_A16_DECODE_MAX_ROWS:-}" in
    "::::") : ;;
    *) say "PRE-SETTINGS: W23 rider / lever-2 switches must all be UNSET (rider (a) off per its 20-row cell; rider (c) dropped per the dense re-race probe; lever 2 killed-with-evidence)"; bad=$((bad+1));;
  esac
  local key="$TRELLIS_TABLE/$DECODE_FUSION/$R1_PUSH/$R3_ISLAND/$R3_ACQUIRE"
  if [ -n "$(fp_baseline)" ]; then
    say "PRE-SETTINGS: fingerprint baseline registered for $key@$EXPECT_EXPORT_SRC: $(fp_baseline)"
  else
    case "$key@$EXPECT_EXPORT_SRC" in
      2/1/1/1/1@b4889fe5)
        # W23 ruled config, first export: the compiled kernel count is
        # unknowable before it (lever 1 swaps the route path at rows 9..20).
        # W11 supersede flow: proceed, record the measured fingerprint in
        # fp_baseline and run --reassert.
        say "PRE-SETTINGS: WARNING — $key@$EXPECT_EXPORT_SRC is the W23 ruled config with no registered fingerprint baseline yet; after the export record the memory-report fingerprint in fp_baseline and run --reassert" ;;
      0/1/1/1/1@*|2/1/0/1/1@*|0/1/0/1/1@*)
        # Pre-authorized fallback exports (J-D section 3: row 5 off / R1_PUSH=0 /
        # both off). Their fingerprint is unknowable before the first export:
        # proceed, but the post-export assertion WILL flag the package until the
        # measured fingerprint is registered in fp_baseline and --reassert runs.
        say "PRE-SETTINGS: WARNING — $key is a pre-authorized fallback config (J-D §3) with no registered fingerprint baseline; after the export record the memory-report fingerprint in fp_baseline and run --reassert" ;;
      *)
        say "PRE-SETTINGS: no ruled fingerprint baseline for $key@$EXPECT_EXPORT_SRC — the post-export settings assertion cannot pass (the 2026-11-11 12:50 class); register the baseline (consult) before exporting"
        bad=$((bad+1)) ;;
    esac
  fi
  return $bad
}

# ── Mandatory preflight (owner order 2026-11-11: catch the fails earlier).
# ── No 8-GPU run starts without this passing. CPU-only except the rider
# ── traces, which use the spare card (11:00.0) under /tmp/gpu-spare.lock.
run_preflight() {
  local mode="$1" bad=0
  say "── preflight ($mode) ──"
  if "$PYV" -c 'import flashinfer, ignition' 2>/dev/null; then
    say "preflight: flashinfer + ignition import OK"
  else
    say "preflight: IMPORT FAILED (flashinfer/ignition)"; bad=$((bad+1))
  fi
  # The `flashinfer/data/csrc missing` class: every JIT build path must resolve.
  if "$PYV" "$HERE/preflight/jit-src-dryrun.py" >&2; then :; else bad=$((bad+1)); fi
  # W23-C rider-(c) precondition (tp9-jG-review.md section 4): prove from the
  # pinned trees whether B12X_A16_SPLITK_REDUCE_PDL enters the dense compile
  # key / re-races the dense selections (CPU; verdict recorded at each run).
  if "$PYV" "$HERE/preflight/pdl-compile-key-probe.py" >&2; then :; else bad=$((bad+1)); fi
  # L4 prefill latent gather (fable-levers-20261112.md section L4 risk 1): the
  # new prefill dense shape and gather channel must appear as NEW race/compile-
  # key entries only; no existing dense selection may re-race (CPU; verdict and
  # the resolved launch geometry recorded at each run).
  if "$PYV" "$HERE/preflight/latent-gather-compile-key-probe.py" >&2; then :; else bad=$((bad+1)); fi
  # Plan acceptance + PRE-mode settings vs rulings.
  if "$PYV" "$HERE/plan-dryrun.py" "${DRYRUN_ARGS[@]}" >&2; then
    say "preflight: planner dry-run OK"
  else
    say "preflight: planner dry-run FAILED"; bad=$((bad+1))
  fi
  assert_pre_settings || bad=$((bad+1))
  # Rider traces on the spare card (full preflight only).
  if [ "$mode" = full ]; then
    if flock /tmp/gpu-spare.lock env CUDA_VISIBLE_DEVICES="${SPARE_GPU}" \
         "$PYV" "$HERE/preflight/dynamo-rider-smoke.py" >&2; then :; else bad=$((bad+1)); fi
  fi
  if [ "$bad" -eq 0 ]; then
    say "preflight: OK ($mode)"
    return 0
  fi
  say "preflight: FAILED ($mode) — $bad check(s); no GPU run starts"
  return 1
}

# ── W11 (radiance-mechanics review §3 item 5): post-export settings assertion.
# Read the recorded settings back from the export record and the package, and
# compare them with the driver's expected set. One physical fingerprint per
# switch where one exists: slab sizes for the prefill wires (ContextParallel
# slab / index_slab in the memory report) and the round-graph/kernel count for
# decode fusion. The fingerprint baselines are the shipped package
# gb202-tp8-202611100809 (E1.1, context 396,288) at the ruled config (capacity 20,
# image concurrency 1, ring rows 1024); other configs assert cross-rank
# consistency only. (E1's gb202-tp8-202611100532 baselines were 170.44 / 39.06
# MiB at the pre-E1.1 context — superseded.)
assert_settings() {
  local log="$1" pkg="$2" bad=0
  local args_line; args_line=$(grep -m1 '^=== args ' "$log" || true)
  for expected in "--prefill-dma-ring-rows $RING_ROWS" "--prefill-dma-bf16" "--trellis-prefill-shapes" \
                  "--image-concurrency $IMAGE_CONCURRENCY" "--speculative-token-capacity $SPEC_TOKEN_CAPACITY" \
                  "--block-verification"; do
    case " $args_line " in *" $expected "*) : ;;
      *) say "ASSERT: recorded invocation misses '$expected'"; bad=$((bad+1));;
    esac
  done
  if [ "$PREFILL_ISLAND" = 1 ]; then
    case " $args_line " in *--prefill-island-reduce*) : ;;
      *) say "ASSERT: recorded invocation misses --prefill-island-reduce"; bad=$((bad+1));;
    esac
  fi
  # W22 knobs: the armed switches must appear in the recorded invocation and
  # the recorded export environment.
  if [ "$TRELLIS_TABLE" != 0 ]; then
    case " $args_line " in *--trellis-prefill-table*) : ;;
      *) say "ASSERT: recorded invocation misses --trellis-prefill-table"; bad=$((bad+1));;
    esac
  fi
  if [ "$DECODE_FUSION" = 1 ]; then
    case " $args_line " in *--decode-fusion*) : ;;
      *) say "ASSERT: recorded invocation misses --decode-fusion"; bad=$((bad+1));;
    esac
  fi
  local env_line; env_line=$(grep -m1 '^=== env ' "$log" || true)
  for pair in "R1_PUSH:B12X_PCIE_DCP_PAIR_PUSH" "R3_ISLAND:B12X_PCIE_DCP_GATHER_ISLAND" "R3_ACQUIRE:B12X_PCIE_DCP_GATHER_ACQUIRE"; do
    knob="${pair%%:*}"; envname="${pair##*:}"
    if [ "${!knob}" = 1 ]; then
      case " $env_line " in *"$envname=1"*) : ;;
        *) say "ASSERT: recorded export env misses $envname=1 (knob $knob)"; bad=$((bad+1));;
      esac
    fi
  done
  # W23-C verdicts: the rider / lever-2 switches must be ABSENT from the
  # recorded export env (J-G section 6 with the rider switches per the W23-C
  # rider-(a) 20-row cell and the rider-(c) dense re-race precondition).
  for name in B12X_TRELLIS_INKERNEL_ROUTE_PACK B12X_TRELLIS_FLOW_FLAGS B12X_A16_SPLITK_REDUCE_PDL B12X_MXFP8_A16_DECODE_W4_TK64 B12X_MXFP8_A16_DECODE_MAX_ROWS B12X_PCIE_DCP_LSE_PUSH_TP8; do
    case "$env_line" in *"$name="*)
      say "ASSERT: recorded export env sets $name (must be UNSET per the W23-C verdicts / J-D §4)"; bad=$((bad+1));;
    esac
  done
  # The plan-settings record the export prints, when the tree records one.
  local plan_line; plan_line=$(grep -m1 '^{"plan"' "$log" || true)
  if [ -n "$plan_line" ]; then
    local recorded
    recorded=$("$PYV" -c 'import json,sys
plan=json.loads(sys.stdin.read())["plan"]
print(plan.get("prefill_dma_ring_rows"), plan.get("prefill_dma_ring_bf16"), plan.get("trellis_prefill_shapes"))' \
      <<<"$plan_line" 2>/dev/null || echo unreadable)
    if [ "$recorded" != "$RING_ROWS True True" ]; then
      say "ASSERT: recorded plan settings '$recorded' != expected '$RING_ROWS True True'"
      bad=$((bad+1))
    fi
  else
    say "ASSERT: no plan-settings JSON in the export record; asserting the invocation record + fingerprints"
  fi
  # Physical fingerprints from the memory report: every rank must agree.
  local slab islab graphs n_slab n_islab n_graphs
  slab=$(grep -E 'ContextParallel>\.slab ' "$log" | awk '{print $1}' | sort -u)
  islab=$(grep -E 'ContextParallel>\.index_slab ' "$log" | awk '{print $1}' | sort -u)
  graphs=$(grep -oE 'Graphs: [0-9]+; kernels: [0-9]+' "$log" | sort -u)
  n_slab=$(grep -cE 'ContextParallel>\.slab ' "$log" || true)
  n_islab=$(grep -cE 'ContextParallel>\.index_slab ' "$log" || true)
  n_graphs=$(grep -coE 'Graphs: [0-9]+; kernels: [0-9]+' "$log" || true)
  if [ -z "$slab" ] || [ -z "$islab" ] || [ -z "$graphs" ]; then
    say "ASSERT: fingerprints missing from the memory report (slab='$slab' index_slab='$islab' graphs='$graphs')"
    bad=$((bad+1))
  else
    if [ "$(wc -l <<<"$slab")" -ne 1 ] || [ "$(wc -l <<<"$islab")" -ne 1 ] || [ "$(wc -l <<<"$graphs")" -ne 1 ]; then
      say "ASSERT: fingerprint disagreement across ranks (slab='$slab' index_slab='$islab' graphs='$graphs')"
      bad=$((bad+1))
    fi
    say "ASSERT: fingerprints slab=${slab} MiB (${n_slab} ranks) index_slab=${islab} MiB (${n_islab}) ${graphs} (${n_graphs})"
    # W22 re-baseline (J-D review, package gb202-tp8-202611110549): the
    # round-graph/kernel count is the decode-fusion detector, and the W22
    # switches are launch FUSIONS — they strictly remove compiled kernels.
    # E1.1's 'Graphs: 66; kernels: 4967' is superseded at the W22 ruled config
    # by 'Graphs: 66; kernels: 4904' (-63 kernels = -63 modules, -1.83 MiB
    # cubin; graph count and the prefill-wire slab sizes unchanged). The
    # registry in fp_baseline() is the single source of truth; the preflight's
    # PRE-mode check refuses configs without a registered baseline before the
    # export starts. Configs outside the ruled capacity/image/ring set assert
    # cross-rank consistency only.
    local ruled=1
    [ "$SPEC_TOKEN_CAPACITY" = 20 ] || ruled=0
    [ "$IMAGE_CONCURRENCY" = 1 ] || ruled=0
    [ "$RING_ROWS" = 1024 ] || ruled=0
    if [ "$ruled" = 1 ]; then
      local base; base="$(fp_baseline)"
      if [ -n "$base" ]; then
        [ "$slab" = "217.69" ] || { say "ASSERT: slab fingerprint '$slab' MiB != baseline 217.69 MiB"; bad=$((bad+1)); }
        [ "$islab" = "49.89" ] || { say "ASSERT: index_slab fingerprint '$islab' MiB != baseline 49.89 MiB"; bad=$((bad+1)); }
        [ "$graphs" = "$base" ] || { say "ASSERT: round-graph/kernel fingerprint '$graphs' != registered baseline '$base' (knobs $TRELLIS_TABLE/$DECODE_FUSION/$R1_PUSH/$R3_ISLAND/$R3_ACQUIRE)"; bad=$((bad+1)); }
      else
        say "ASSERT: no registered fingerprint baseline for knobs $TRELLIS_TABLE/$DECODE_FUSION/$R1_PUSH/$R3_ISLAND/$R3_ACQUIRE — record '$graphs' in fp_baseline() and run --reassert"
        bad=$((bad+1))
      fi
    else
      say "ASSERT: fingerprint baseline skipped (non-ruled capacity/image/ring); cross-rank consistency asserted only"
    fi
  fi
  # Read back from the package: the captured round graphs must be there.
  if [ -n "$pkg" ] && ls "$pkg"/deployments/*/rank-0/graphs.safetensors >/dev/null 2>&1; then
    :
  else
    say "ASSERT: package $pkg lacks deployments/*/rank-0/graphs.safetensors"
    bad=$((bad+1))
  fi
  if [ "$bad" -ne 0 ]; then
    say "SETTINGS ASSERTION FAILED: $bad mismatch(es) — package $pkg flagged; review before promotion"
    echo "settings-mismatch $bad" > "$OUT/export.settings-mismatch"
    echo "=== SETTINGS ASSERTION FAILED ($bad mismatches)" >>"$log"
    return 1
  fi
  rm -f "$OUT/export.settings-mismatch"
  echo "=== settings assertion OK: recorded settings == expected set; fingerprints slab=${slab} MiB index_slab=${islab} MiB; ${graphs}" >>"$log"
  return 0
}

# ── --reassert <log> <pkg>: re-run the post-export settings assertion against
# ── an existing export record and package (review follow-up; exports nothing).
if [ "$MODE" = --reassert ]; then
  [ $# -ge 3 ] || { say "usage: $0 --reassert <export-log> <package-dir>"; exit 2; }
  if assert_settings "$2" "$3"; then
    echo "0" > "$OUT/export.settings-rc"
    say "REASSERT OK: $3 matches the ruled settings + fingerprints"
    exit 0
  fi
  echo "3" > "$OUT/export.settings-rc"
  say "REASSERT FAILED: $3 still mismatches"
  exit 3
fi

if [ "$MODE" = --link ]; then
  [ -f "$DONE" ] && [ ! -f "$FAIL" ] || { say "FATAL: download not complete ($DONE missing or $FAIL present)"; exit 1; }
  mkdir -p "$SNAP"
  for f in "$CKPT"/*; do
    b=$(basename "$f"); case "$b" in .cache|*.manifest.json) continue;; esac
    ln -sfn "$f" "$SNAP/$b"
  done
  ln -sfn "$DRAFT" "$SNAP/dflash2"
  say "linked $(find "$SNAP" -maxdepth 1 -type l | wc -l) entries into $SNAP"
  exit 0
fi

# ── W13 image step (--image): reproduce the RUNNING runtime image from this
# tree. Same build path the chunkfix image used (campaign runtime-image path:
# docker build -t local/flare:rt-<variant>-<tree sha> <tree>; Dockerfile
# rust:1.94.1-slim-trixie build stage -> debian:trixie-slim, ~15-25 min), pinned
# by image ID exactly the way the launcher pins its image. The image inputs are
# guarded against the pinned image tree (EXPECT_IMAGE_SRC); a build whose ID
# drifts from EXPECT_IMAGE_ID is flagged, never silently accepted.
if [ "$MODE" = --image ]; then
  chk command -v docker >/dev/null 2>&1 || say "GUARD: docker missing"
  chk test -z "$(git -C "$REPO" status --porcelain 2>/dev/null)" || say "GUARD: tree dirty"
  chk git -C "$REPO" merge-base --is-ancestor "$EXPECT_HEAD" HEAD 2>/dev/null \
    || say "GUARD: merged tip $EXPECT_HEAD is not an ancestor of HEAD"
  chk test -z "$(git -C "$REPO" diff --name-only "$EXPECT_HEAD" HEAD -- ':!tools' ':!README.md' 2>/dev/null)" \
    || say "GUARD: non-tools diff against pinned merged tip $EXPECT_HEAD"
  chk test -z "$(git -C "$REPO" diff --name-only "$EXPECT_EXPORT_SRC" HEAD -- ':!helios' ':!tools' ':!README.md' 2>/dev/null)" \
    || say "GUARD: export-consumed files drift from the pinned export tree $EXPECT_EXPORT_SRC"
  chk test -z "$(git -C "$REPO" diff --name-only "$EXPECT_IMAGE_SRC" HEAD -- "${IMAGE_PATHS[@]}" 2>/dev/null)" \
    || say "GUARD: image inputs drift from the pinned image tree $EXPECT_IMAGE_SRC"
  if [ "$fail" -ne 0 ]; then say "IMAGE CHECK FAILED: $fail guard(s); nothing built"; exit 1; fi
  head_short=$(git -C "$REPO" rev-parse --short=8 HEAD)
  img_tag="$IMAGE_NAME:$IMAGE_VARIANT-$head_short"
  img_log="$OUT/image-build-$TS.log"
  say "── building runtime image $img_tag from $REPO (log $img_log) ──"
  if docker build -t "$img_tag" "$REPO" >"$img_log" 2>&1; then
    built_id=$(docker inspect --format '{{.Id}}' "$img_tag")
    echo "=== image $img_tag id $built_id pin $EXPECT_IMAGE_ID" >>"$img_log"
    if [ "$built_id" = "$EXPECT_IMAGE_ID" ]; then
      docker tag "$img_tag" "$IMAGE_NAME:$LIVE_IMAGE_TAG"
      printf '%s\n%s\n' "$built_id" "$IMAGE_NAME:$LIVE_IMAGE_TAG" > "$OUT/image.done"
      rm -f "$OUT/image.id-mismatch"
      say "IMAGE OK: $img_tag id $built_id == pinned $EXPECT_IMAGE_ID; tagged $IMAGE_NAME:$LIVE_IMAGE_TAG"
      say "IMAGE OK recorded: $OUT/image.done"
      exit 0
    fi
    echo "$built_id" > "$OUT/image.id-mismatch"
    say "IMAGE PIN MISMATCH: built $built_id != pinned $EXPECT_IMAGE_ID"
    say "  (image-input drift or non-reproducible toolchain — review before promotion; recorded in $OUT/image.id-mismatch)"
    exit 3
  fi
  say "IMAGE BUILD FAILED: see $img_log"; exit 1
fi

# ── Guards (all modes) ────────────────────────────────────────────────────────
chk test -f "$DONE"            || say "GUARD: $DONE missing (download not verified)"
chk test ! -f "$FAIL"          || say "GUARD: $FAIL present"
chk test -f "$SNAP/config.json" || say "GUARD: snapshot not linked ($SNAP/config.json) — run --link"
chk test -f "$SNAP/exl3-manifest.json" || say "GUARD: exl3-manifest.json missing in snapshot"
n_layers=$(find "$SNAP/" -maxdepth 1 -name "exl3-layer-*.safetensors" | wc -l); chk test "$n_layers" -eq 92 || say "GUARD: $n_layers exl3 layer files, need 92"
n_dense=$(find "$SNAP/" -maxdepth 1 -name "retained-model-*.safetensors" | wc -l); chk test "$n_dense" -eq 96 || say "GUARD: $n_dense retained-model shards, need 96"
chk test -f "$SNAP/dflash2/model.safetensors" || say "GUARD: draft missing at $SNAP/dflash2"
chk test -f "$CONFIG"          || say "GUARD: config $CONFIG missing"
chk test -x "$PYV" || say "GUARD: venv missing ($PYV)"
chk git -C "$REPO" merge-base --is-ancestor "$EXPECT_HEAD" HEAD 2>/dev/null \
  || say "GUARD: export tree $EXPECT_HEAD is not an ancestor of HEAD"
chk test -z "$(git -C "$REPO" diff --name-only "$EXPECT_HEAD" HEAD -- ':!tools' ':!README.md' 2>/dev/null)" \
  || say "GUARD: non-tools diff against pinned export tree $EXPECT_HEAD"
# W13 lineage guards: the tree must reproduce the exact code the running serve's
# package export (b8dcebb5) and runtime image build (5585a9a7) consumed. The
# export side excludes helios (the export never reads it; the image build does)
# plus tools/ and README.md (driver + documentation, W15); the image side is
# exactly the Dockerfile COPY input set.
chk test -z "$(git -C "$REPO" diff --name-only "$EXPECT_EXPORT_SRC" HEAD -- ':!helios' ':!tools' ':!README.md' 2>/dev/null)" \
  || say "GUARD: export-consumed files drift from the pinned export tree $EXPECT_EXPORT_SRC"
chk test -z "$(git -C "$REPO" diff --name-only "$EXPECT_IMAGE_SRC" HEAD -- "${IMAGE_PATHS[@]}" 2>/dev/null)" \
  || say "GUARD: image inputs drift from the pinned image tree $EXPECT_IMAGE_SRC"
# The E1.1 context lives in the recipe (no CLI flag): assert 396,288 so no stale
# E1 context can ride an export out of this branch.
chk grep -q 'context=396288' "$REPO/ignition/recipes/kimi31/recipe.py" 2>/dev/null \
  || say "GUARD: recipe context is not 396288 in $REPO/ignition/recipes/kimi31/recipe.py"
# J-G section 5 (W23): MAX_ROWS 20 is recipe code on both sides of the route
# exchange — the flare routing cap and the flashinfer standalone top-16 cap.
chk grep -q '^MAX_ROWS = 20$' "$REPO/ignition/recipes/kimi3/routing.py" 2>/dev/null \
  || say "GUARD: recipe MAX_ROWS is not 20 in $REPO/ignition/recipes/kimi3/routing.py"
chk grep -q '^KIMI_TOPK16_MAX_ROWS = 20$' "$FI/flashinfer/experimental/b12x/comm/pcie/pcie_dcp_a2a.py" 2>/dev/null \
  || say "GUARD: KIMI_TOPK16_MAX_ROWS is not 20 in the flashinfer tree"
chk test -z "$(git -C "$REPO" status --porcelain 2>/dev/null)" || say "GUARD: tree dirty"
fi_head=$(git -C "$FI" rev-parse --short HEAD 2>/dev/null || echo none)
chk test "$fi_head" = "$EXPECT_FI" || say "GUARD: flashinfer HEAD $fi_head != $EXPECT_FI"
export PYTHONPATH="$FI:$FI/flashinfer/experimental/b12x/_compat:$REPO"
fi_path=$(cd "$REPO" && "$PYV" -c 'import flashinfer, pathlib; print(pathlib.Path(flashinfer.__file__).resolve())' 2>/dev/null || echo none)
chk test "$fi_path" = "$FI/flashinfer/__init__.py" || say "GUARD: flashinfer resolves to $fi_path (want $FI)"
ig_path=$(cd "$REPO" && "$PYV" -c 'import ignition, pathlib; print(pathlib.Path(ignition.__file__).resolve())' 2>/dev/null || echo none)
chk test "$ig_path" = "$REPO/ignition/__init__.py" || say "GUARD: ignition resolves to $ig_path (want $REPO)"
# J-3 §1: B12X_PCIE_DMA_PIECES must be UNSET in the export environment.
chk test -z "${B12X_PCIE_DMA_PIECES:-}" || say "GUARD: B12X_PCIE_DMA_PIECES is set (${B12X_PCIE_DMA_PIECES}) — must be UNSET"
# W23-C verdicts (tp9-jG-review.md sections 2/4/6): the three rider switches
# and the two lever-2 knobs must be UNSET in the export environment. Rider (a)
# failed its 20-row condition (the rider is off entirely), rider (c) is dropped
# because it re-races the dense selections, and lever 2 is killed-with-evidence.
chk test -z "${B12X_TRELLIS_INKERNEL_ROUTE_PACK:-}${B12X_TRELLIS_FLOW_FLAGS:-}${B12X_A16_SPLITK_REDUCE_PDL:-}${B12X_MXFP8_A16_DECODE_W4_TK64:-}${B12X_MXFP8_A16_DECODE_MAX_ROWS:-}" \
  || say "GUARD: W23 rider / lever-2 switch set (B12X_TRELLIS_INKERNEL_ROUTE_PACK / B12X_TRELLIS_FLOW_FLAGS / B12X_A16_SPLITK_REDUCE_PDL / B12X_MXFP8_A16_DECODE_*) — all must be UNSET per the W23-C verdicts"
# W11 (radiance-mechanics review §3 item 4): environment allowlist at export
# start. The driver refuses to run with any IGNITION_KIMI_* / B12X_* variable
# present that the plan did not set. The accounted set is exactly the shipped
# B12X_W4A16_SMALL_M_SPLITK=1 kernel switch (set by this driver and recorded);
# every other name — the legacy IGNITION_KIMI_* switches, B12X_PCIE_DMA_PIECES
# (J-1 risk 6), future experiments — fails the guard instead of silently
# changing the package. The plan is the second line of defense
# (planning.check_switch_environment in the ignition tree).
env_bad=0
for name in $(compgen -v | grep -E '^(IGNITION_KIMI_|B12X_)' || true); do
  if [ "$name" = B12X_W4A16_SMALL_M_SPLITK ] && [ "${!name}" = 1 ]; then continue; fi
  # W22 transport switches are armed by this driver's R1_PUSH / R3_ISLAND /
  # R3_ACQUIRE knobs and exported below; a caller-provided copy equal to the
  # armed value is accounted, anything else is not.
  if [ "$name" = B12X_PCIE_DCP_PAIR_PUSH ] && [ "$R1_PUSH" = 1 ] && [ "${!name}" = 1 ]; then continue; fi
  if [ "$name" = B12X_PCIE_DCP_GATHER_ISLAND ] && [ "$R3_ISLAND" = 1 ] && [ "${!name}" = 1 ]; then continue; fi
  if [ "$name" = B12X_PCIE_DCP_GATHER_ACQUIRE ] && [ "$R3_ACQUIRE" = 1 ] && [ "${!name}" = 1 ]; then continue; fi
  say "GUARD: unaccounted switch environment ${name}='${!name}' (accounted: B12X_W4A16_SMALL_M_SPLITK=1 and the armed W22 transport switches)"
  env_bad=$((env_bad+1))
done
chk test "$env_bad" -eq 0 || say "GUARD: switch-environment allowlist violated (W11 §3 item 4)"
GPU_LIST="$(python3 /opt/mimo-pro/flare-bd/gpu-select.py B,D 2>/dev/null || true)"
n_gpu=$(awk -F, '{print NF}' <<<"$GPU_LIST"); chk test "$n_gpu" -eq 8 || say "GUARD: B+D resolved $n_gpu GPUs, need 8"
busy=0; for g in ${GPU_LIST//,/ }; do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null || echo 99999); [ "$u" -lt 1024 ] || busy=1; done
if [ "$MODE" = --preflight ]; then
  # Preflight is allowed while the serve owns B+D; the free-card guard is a
  # launch-time check and is re-run at --check/--detach.
  [ "$busy" -eq 0 ] && say "preflight: B+D cards free" \
    || say "preflight: NOTE — B+D cards in use (serve live); launch-time guard re-checks at --check/--detach"
else
  chk test "$busy" -eq 0        || say "GUARD: a B/D card is in use"
fi
free=$(df -BG --output=avail /opt/models | tail -1 | tr -dc 0-9); chk test "$free" -ge "$MIN_FREE_G" || say "GUARD: /opt/models free ${free}G < ${MIN_FREE_G}G"
avail=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo); chk test "$avail" -ge 400 || say "GUARD: MemAvailable ${avail}G < 400G"
if [ "$MODE" = --run ]; then
  chk test "$(ulimit -l)" = unlimited || say "GUARD: memlock not unlimited"
  chk test "$(ulimit -n)" -ge 65536   || say "GUARD: nofile $(ulimit -n) < 65536"
fi
chk test "$SPEC_TOKEN_CAPACITY" = 20 -o "$SPEC_TOKEN_CAPACITY" = 24 \
  || say "GUARD: SPEC_TOKEN_CAPACITY $SPEC_TOKEN_CAPACITY must be 20 (primary) or 24 (fallback)"
chk test "$PREFILL_FP8" = 0 -o "$PREFILL_FP8" = 1 \
  || say "GUARD: PREFILL_FP8 $PREFILL_FP8 must be 0 (exact BF16 default) or 1 (fp8-transport experiment)"
chk test "$PREFILL_ISLAND" = 0 -o "$PREFILL_ISLAND" = 1 \
  || say "GUARD: PREFILL_ISLAND $PREFILL_ISLAND must be 0 (exact BF16 default) or 1 (L1 two-island wire)"
chk test "$PREFILL_ISLAND" = 0 -o "$PREFILL_FP8" = 0 \
  || say "GUARD: PREFILL_ISLAND and PREFILL_FP8 are mutually exclusive"
chk test "$IMAGE_CONCURRENCY" -ge 1 -a "$IMAGE_CONCURRENCY" -le 4 2>/dev/null \
  || say "GUARD: IMAGE_CONCURRENCY $IMAGE_CONCURRENCY must be 1..4"
chk test "$TRELLIS_TABLE" = 0 -o "$TRELLIS_TABLE" = 1 -o "$TRELLIS_TABLE" = 2 \
  || say "GUARD: TRELLIS_TABLE $TRELLIS_TABLE must be 0, 1 or 2"
chk test "$DECODE_FUSION" = 0 -o "$DECODE_FUSION" = 1 \
  || say "GUARD: DECODE_FUSION $DECODE_FUSION must be 0 or 1"
for knob in R1_PUSH R3_ISLAND R3_ACQUIRE; do
  val="${!knob}"
  chk test "$val" = 0 -o "$val" = 1 || say "GUARD: $knob $val must be 0 or 1"
done

# ── Planner dry-run (CPU only, no NVML/CUDA): accept the c4 + 1M shape and print
# ── the boot KV supply projection for both --speculative-token-capacity variants.
DRYRUN_ARGS=(--capacity "$SPEC_TOKEN_CAPACITY")
if [ "$PREFILL_FP8" = 1 ]; then DRYRUN_ARGS+=(--prefill-fp8); fi
if [ "$PREFILL_ISLAND" = 1 ]; then DRYRUN_ARGS+=(--prefill-island); fi
say "── planner dry-run (capacity $SPEC_TOKEN_CAPACITY armed; both variants priced) ──"
if "$PYV" "$HERE/plan-dryrun.py" "${DRYRUN_ARGS[@]}" >&2; then
  say "── dry-run OK: c4 + recipe context accepted at capacity 20 and 24 ──────────"
else
  fail=$((fail+1)); say "GUARD: planner dry-run refused the c4 + recipe-context shape"
fi

say "── would run ──────────────────────────────────────────────"
say "  cd $REPO; CUDA_VISIBLE_DEVICES=<8 UUIDs B then D> CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1"
say "  PYTHONPATH=$PYTHONPATH"
say "  .venv/bin/python -m ignition export $SNAP ${EXPORT_ARGS[*]}"
say "  log $LOG  (+ .mem every 30 s); after rc=0: prune nlink==1 blobs under $BLOBS, link $PKG_LINK"
say "  (package AND image: bash tools/export-kimi31-exl3.sh --image builds the pinned"
say "   runtime image $IMAGE_NAME:$IMAGE_VARIANT-<tree sha> -> $IMAGE_NAME:$LIVE_IMAGE_TAG)"
say "───────────────────────────────────────────────────────────"
if [ "$fail" -ne 0 ]; then say "CHECK FAILED: $fail guard(s); nothing run"; exit 1; fi
[ "$MODE" = --check ] && { say "CHECK OK: all guards pass; nothing run"; exit 0; }

# ── Mandatory preflight stage (owner order 2026-11-11). No 8-GPU run starts
# ── without it passing: --preflight runs it standalone, --detach runs the full
# ── set (incl. the spare-card rider traces) before detaching, --run repeats the
# ── CPU subset inside the unit before the export touches the B+D cards.
if [ "$MODE" = --preflight ]; then
  if run_preflight full; then
    say "PREFLIGHT OK: imports, JIT sources, rider traces, planner + pre-settings all pass"
    exit 0
  fi
  say "PREFLIGHT FAILED"; exit 1
fi

if [ "$MODE" = --detach ]; then
  run_preflight full || { say "PREFLIGHT FAILED: nothing detached"; exit 1; }
  unit="kimi31-exl3-export-$TS"
  sudo -n systemd-run --unit="$unit" --collect --uid=tom --gid=tom \
    --setenv=HOME=/home/tom --setenv=USER=tom --setenv=EXPORT_TS="$TS" --setenv=EXPORT_LOG="$LOG" \
    --setenv=PREFILL_ISLAND="$PREFILL_ISLAND" --setenv=IMAGE_CONCURRENCY="$IMAGE_CONCURRENCY" \
    --setenv=PYTHONPATH="$PYTHONPATH" --setenv=B12X_W4A16_SMALL_M_SPLITK=1 --setenv=PREFILL_FP8="$PREFILL_FP8" --setenv=SPEC_TOKEN_CAPACITY="$SPEC_TOKEN_CAPACITY" --setenv=REUSE_WEIGHTS="$REUSE_WEIGHTS" \
    --setenv=TRELLIS_TABLE="$TRELLIS_TABLE" --setenv=DECODE_FUSION="$DECODE_FUSION" \
    --setenv=R1_PUSH="$R1_PUSH" --setenv=R3_ISLAND="$R3_ISLAND" --setenv=R3_ACQUIRE="$R3_ACQUIRE" \
    -p LimitNOFILE=1048576 -p LimitMEMLOCK=infinity --working-directory="$REPO" \
    /bin/bash "$SELF" --run
  say "detached unit: $unit   log: $LOG   follow: journalctl -u $unit -f / tail -f $LOG"
  exit 0
fi

[ "$MODE" = --run ] || { say "unknown mode $MODE"; exit 2; }
# CPU subset of the preflight, repeated inside the unit before any GPU work.
run_preflight light || { say "PREFLIGHT FAILED (run): export not started"; exit 1; }
cd "$REPO"
export CUDA_VISIBLE_DEVICES="$GPU_LIST" CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
# W6 E1 (J-3 §1) + W11 §3 item 4: only the shipped split-K decode switch rides
# the environment; the ring/bf16/trellis config is in the PLAN FLAGS. Nothing is
# silently unset here any more: the switch-environment allowlist guard above
# refuses any unaccounted IGNITION_KIMI_* / B12X_* variable (including
# B12X_PCIE_DMA_PIECES) instead of healing the environment.
export B12X_W4A16_SMALL_M_SPLITK=1
# W22 transport switches (rows 1/3 per the W22 gate; recorded in the plan's
# b12x environment and compiled into the package's graphs).
W22_SWITCHES=""
if [ "$R1_PUSH" = 1 ]; then export B12X_PCIE_DCP_PAIR_PUSH=1; W22_SWITCHES="$W22_SWITCHES B12X_PCIE_DCP_PAIR_PUSH=1"; fi
if [ "$R3_ISLAND" = 1 ]; then export B12X_PCIE_DCP_GATHER_ISLAND=1; W22_SWITCHES="$W22_SWITCHES B12X_PCIE_DCP_GATHER_ISLAND=1"; fi
if [ "$R3_ACQUIRE" = 1 ]; then export B12X_PCIE_DCP_GATHER_ACQUIRE=1; W22_SWITCHES="$W22_SWITCHES B12X_PCIE_DCP_GATHER_ACQUIRE=1"; fi
echo "=== env check: B12X_PCIE_DMA_PIECES='${B12X_PCIE_DMA_PIECES:-}' (must be empty) B12X_W4A16_SMALL_M_SPLITK='$B12X_W4A16_SMALL_M_SPLITK' IGNITION_KIMI_PREFILL_DMA_ROWS='${IGNITION_KIMI_PREFILL_DMA_ROWS:-}' W22:$W22_SWITCHES TRELLIS_TABLE=$TRELLIS_TABLE DECODE_FUSION=$DECODE_FUSION"
{ echo "=== export start $(date -Is) tree $(git rev-parse HEAD) flashinfer $(git -C "$FI" rev-parse HEAD) free ${free}G avail ${avail}G"
  echo "=== gpus $GPU_LIST"; echo "=== args ${EXPORT_ARGS[*]}"
  echo "=== env B12X_W4A16_SMALL_M_SPLITK=1 B12X_PCIE_DMA_PIECES=unset IGNITION_KIMI_PREFILL_DMA_*=unset$W22_SWITCHES TRELLIS_TABLE=$TRELLIS_TABLE DECODE_FUSION=$DECODE_FUSION"; } >>"$LOG"
"$PYV" -m ignition export "$SNAP" "${EXPORT_ARGS[@]}" >>"$LOG" 2>&1 &
E=$!
while kill -0 $E 2>/dev/null; do
  echo "$(date +%H:%M:%S) avail=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)G swapfree=$(awk '/SwapFree/{print int($2/1048576)}' /proc/meminfo)G opt=$(df -BG --output=avail /opt/models | tail -1 | tr -dc 0-9)G" >>"$LOG.mem"
  sleep 30
done
rc=0; wait $E || rc=$?
echo "=== export end $(date -Is) rc=$rc" >>"$LOG"
if [ $rc -eq 0 ]; then
  pkg=$(grep -oE 'Exported cache entry: \S+' "$LOG" | tail -1 | awk '{print $4}' || true)
  [ -n "$pkg" ] && [ -d "$pkg" ] && ln -sfn "$pkg" "$PKG_LINK" && echo "=== package $pkg -> $PKG_LINK" >>"$LOG"
  n=$(find "$BLOBS" -maxdepth 1 -type f -links 1 -print -delete | wc -l); echo "=== blob prune removed=$n" >>"$LOG"
  if [ -n "$pkg" ]; then echo "$pkg" > "$OUT/export.done"; else echo "package path not found in log" > "$OUT/export.done"; echo "=== package path not found in log" >>"$LOG"; fi
  # W11 §3 item 5: post-export settings assertion. The package is already
  # linked and recorded above; a mismatch flags it for review and fails the
  # driver (export.settings-mismatch, rc=3) instead of letting a drifted
  # package look ruled (J-1 risk 1).
  if ! assert_settings "$LOG" "$pkg"; then
    echo "3" > "$OUT/export.settings-rc"
    exit 3
  fi
  echo "0" > "$OUT/export.settings-rc"
else
  echo "=== blob prune skipped rc=$rc" >>"$LOG"; echo "$rc" > "$OUT/export.failed"
fi
exit $rc
