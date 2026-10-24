#!/usr/bin/env bash
# Export the MiMo-V2.6-Pro-MOPD FLARE package: TP8, DFlash MTP with 7 draft tokens, CSF expert scales,
# device-selected verification width at batch one.
# usage: export.sh CHECKPOINT_DIR OUTPUT.safetensors     (eight GPUs; select them with CUDA_VISIBLE_DEVICES)
set -euo pipefail
[ $# -eq 2 ] || { echo "usage: $0 CHECKPOINT_DIR OUTPUT.safetensors" >&2; exit 2; }
cd "$(dirname "$0")/../.."
# The checkpoint loader uses io_uring with locked bounce buffers.
[ "$(ulimit -l)" = unlimited ] || ulimit -l unlimited 2>/dev/null || {
  echo "export needs unlimited locked memory: run under 'prlimit --memlock=unlimited:unlimited' or raise ulimit -l" >&2; exit 1; }
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export FLARE_MIMO_SWITCH_SPECULATION=1 FLARE_MIMO_TWO_PHASE_MAX_BATCH=0
exec .venv/bin/python -m ignition export "$1" \
  --gpus 0-7 \
  --max-concurrency 64 \
  --image-concurrency 4 \
  --prefill-chunk-size 8192 \
  --mtp-draft-tokens 7 \
  --mtp-proposal-source mtp \
  --speculative-token-capacity 256 \
  --expert-scale-compression csf \
  --output "$2" \
  --verbose
