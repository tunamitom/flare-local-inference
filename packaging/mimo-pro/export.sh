#!/usr/bin/env bash
# Export the MiMo-V2.6-Pro-MOPD FLARE package: TP8, max concurrency 64, DFlash MTP with 7 draft tokens,
# packed verification over 256 target rows, CSF expert scales, A4/A16 prefill variants from 512 tokens.
# The package lands in the model cache: <package_cache>/models/<checkpoint-dir>/gb202-tp8-<date>/.
# usage: export.sh CHECKPOINT_DIR     (eight GPUs; select them with CUDA_VISIBLE_DEVICES)
set -euo pipefail
[ $# -eq 1 ] || { echo "usage: $0 CHECKPOINT_DIR" >&2; exit 2; }
cd "$(dirname "$0")/../.."
# The checkpoint loader uses io_uring with locked bounce buffers.
[ "$(ulimit -l)" = unlimited ] || ulimit -l unlimited 2>/dev/null || {
  echo "export needs unlimited locked memory: run under 'prlimit --memlock=unlimited:unlimited' or raise ulimit -l" >&2; exit 1; }
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
exec .venv/bin/python -m ignition export "$1" \
  --gpus 0-7 \
  --max-concurrency 64 \
  --image-concurrency 4 \
  --prefill-chunk-size 8192 \
  --mtp-draft-tokens 7 \
  --speculative-token-capacity 256 \
  --expert-scale-compression csf \
  --verbose
