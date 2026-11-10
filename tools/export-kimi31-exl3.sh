#!/usr/bin/env bash
# kimi31-exl3-live: THE single branch that exports the CURRENT RUNNING SERVE.
#   Package: gb202-tp8-202611100532 (smart on :8001 as of 2026-11-10) = this tree (160832b9)
#   + flashinfer levers/prefill-shapes (0aa09dd61, auto-cloned from the fork if missing).
#   Measured: prefill 8k 2,117 / 32k 2,312 tok/s, mid-size TTFT -26/-29%, pool 1,222,656,
#   max context 310,272 (owner rule: ~1/3 of pool). Reproduce:
#     bash tools/export-kimi31-exl3.sh --check   # guards + planner dry-run, creates nothing
#     bash tools/export-kimi31-exl3.sh --detach  # the export (graph-only --reuse-weights)
#   Requires the EXL3-2bit checkpoint + DFlash2 draft at the CKPT/DRAFT paths below (or
#   override via env) and the box's 8 B+D GPUs.
# ─────────────────────────────────────────────────────────────────────────────
# Kimi-K3.1-Pro EXL3-2bit → FLARE TP8 package on switches B+D (TP8-CONFIG-PLAN.md §5.5, revision 3).
# Tree: /home/tom/build/flare-k31exl3 @ kimi31-exl3-dma (release 01327bf2 + block verification),
# FlashInfer 6a842a7d (editable, leads PYTHONPATH). GPUs resolved topologically (B then D) by
# /opt/mimo-pro/flare-bd/gpu-select.py. The watchdog only logs: a running export is never interrupted.
#
# WHAT THIS DRIVER NOW REPRODUCES (lever W1 sync, 2026-11-10): the live package
# gb202-tp8-202611100247 exactly — the defaults below are the measured config:
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
#   max context  -> OWNER DIRECTIVE: recipe context 310,272 (= pool/3 on the 1,024
#                  grid, ratio 0.33297 vs predicted 931,840); the 1M gate is obsolete,
#                  the gate is max_model_len / actual Allocate pool in [0.28, 0.36]
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
PYV="${PYV:-$REPO/.venv/bin/python}"; [ -x "$PYV" ] || PYV="$(command -v python3)"
RING_ROWS="${RING_ROWS:-1024}"
EXPECT_HEAD="${EXPECT_HEAD:-160832b9}"
EXPECT_FI="${EXPECT_FI:-0aa09dd61}"
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

MODE="${1:---check}"
say() { echo "$*" >&2; }
fail=0
chk() { if "$@"; then return 0; else fail=$((fail+1)); return 1; fi; }

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
chk test -z "$(git -C "$REPO" diff --name-only "$EXPECT_HEAD" HEAD -- ':!tools' 2>/dev/null)" \
  || say "GUARD: non-tools diff against pinned export tree $EXPECT_HEAD"
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
GPU_LIST="$(python3 /opt/mimo-pro/flare-bd/gpu-select.py B,D 2>/dev/null || true)"
n_gpu=$(awk -F, '{print NF}' <<<"$GPU_LIST"); chk test "$n_gpu" -eq 8 || say "GUARD: B+D resolved $n_gpu GPUs, need 8"
busy=0; for g in ${GPU_LIST//,/ }; do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null || echo 99999); [ "$u" -lt 1024 ] || busy=1; done
chk test "$busy" -eq 0        || say "GUARD: a B/D card is in use"
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

# ── Planner dry-run (CPU only, no NVML/CUDA): accept the c4 + 1M shape and print
# ── the boot KV supply projection for both --speculative-token-capacity variants.
DRYRUN_ARGS=(--capacity "$SPEC_TOKEN_CAPACITY")
if [ "$PREFILL_FP8" = 1 ]; then DRYRUN_ARGS+=(--prefill-fp8); fi
if [ "$PREFILL_ISLAND" = 1 ]; then DRYRUN_ARGS+=(--prefill-island); fi
say "── planner dry-run (capacity $SPEC_TOKEN_CAPACITY armed; both variants priced) ──"
if "$PYV" "$HERE/plan-dryrun.py" "${DRYRUN_ARGS[@]}" >&2; then
  say "── dry-run OK: c4 + 1M accepted at capacity 20 and 24 ─────────────────────"
else
  fail=$((fail+1)); say "GUARD: planner dry-run refused the c4 + 1M shape"
fi

say "── would run ──────────────────────────────────────────────"
say "  cd $REPO; CUDA_VISIBLE_DEVICES=<8 UUIDs B then D> CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1"
say "  PYTHONPATH=$PYTHONPATH"
say "  .venv/bin/python -m ignition export $SNAP ${EXPORT_ARGS[*]}"
say "  log $LOG  (+ .mem every 30 s); after rc=0: prune nlink==1 blobs under $BLOBS, link $PKG_LINK"
say "───────────────────────────────────────────────────────────"
if [ "$fail" -ne 0 ]; then say "CHECK FAILED: $fail guard(s); nothing run"; exit 1; fi
[ "$MODE" = --check ] && { say "CHECK OK: all guards pass; nothing run"; exit 0; }

if [ "$MODE" = --detach ]; then
  unit="kimi31-exl3-export-$TS"
  sudo -n systemd-run --unit="$unit" --collect --uid=tom --gid=tom \
    --setenv=HOME=/home/tom --setenv=USER=tom --setenv=EXPORT_TS="$TS" --setenv=EXPORT_LOG="$LOG" \
    --setenv=PREFILL_ISLAND="$PREFILL_ISLAND" --setenv=IMAGE_CONCURRENCY="$IMAGE_CONCURRENCY" \
    --setenv=PYTHONPATH="$PYTHONPATH" --setenv=B12X_W4A16_SMALL_M_SPLITK=1 --setenv=PREFILL_FP8="$PREFILL_FP8" --setenv=SPEC_TOKEN_CAPACITY="$SPEC_TOKEN_CAPACITY" \
    -p LimitNOFILE=1048576 -p LimitMEMLOCK=infinity --working-directory="$REPO" \
    /bin/bash "$SELF" --run
  say "detached unit: $unit   log: $LOG   follow: journalctl -u $unit -f / tail -f $LOG"
  exit 0
fi

[ "$MODE" = --run ] || { say "unknown mode $MODE"; exit 2; }
cd "$REPO"
export CUDA_VISIBLE_DEVICES="$GPU_LIST" CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
# W6 E1 (J-3 §1): only the shipped split-K decode switch rides the environment;
# the ring/bf16/trellis config is in the PLAN FLAGS. B12X_PCIE_DMA_PIECES must
# stay UNSET (global override; the per-shape table is the ruled selector).
unset B12X_PCIE_DMA_PIECES IGNITION_KIMI_PREFILL_DMA_ROWS IGNITION_KIMI_PREFILL_DMA_BF16 IGNITION_KIMI_PREFILL_DMA_ROWS || true
export B12X_W4A16_SMALL_M_SPLITK=1
echo "=== env check: B12X_PCIE_DMA_PIECES='${B12X_PCIE_DMA_PIECES:-}' (must be empty) B12X_W4A16_SMALL_M_SPLITK='$B12X_W4A16_SMALL_M_SPLITK' IGNITION_KIMI_PREFILL_DMA_ROWS='${IGNITION_KIMI_PREFILL_DMA_ROWS:-}'"
{ echo "=== export start $(date -Is) tree $(git rev-parse HEAD) flashinfer $(git -C "$FI" rev-parse HEAD) free ${free}G avail ${avail}G"
  echo "=== gpus $GPU_LIST"; echo "=== args ${EXPORT_ARGS[*]}"
  echo "=== env B12X_W4A16_SMALL_M_SPLITK=1 B12X_PCIE_DMA_PIECES=unset IGNITION_KIMI_PREFILL_DMA_*=unset"; } >>"$LOG"
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
else
  echo "=== blob prune skipped rc=$rc" >>"$LOG"; echo "$rc" > "$OUT/export.failed"
fi
exit $rc
