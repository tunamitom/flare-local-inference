#!/usr/bin/env bash
# Serve an exported MiMo package on eight GPUs with the shipped flags.
# usage: serve.sh IMAGE PACKAGE_DIR STATE_DIR PREFIX_CACHE_DIR [PORT]
#   PACKAGE_DIR holds NAME.safetensors and its companions; STATE_DIR holds the publisher keys
#   (development-publisher.json) and trust.toml that verify the package.
# Environment: GPUS (docker --gpus device list, default all), PUBLISHER_FINGERPRINT and
# PUBLISHER_NAMESPACE (the exporting publisher to accept; see `flare trust --help`).
set -euo pipefail
[ $# -ge 4 ] || { echo "usage: $0 IMAGE PACKAGE_DIR STATE_DIR PREFIX_CACHE_DIR [PORT]" >&2; exit 2; }
image=$1 package=$(realpath "$2") state=$(realpath "$3") cache=$(realpath "$4") port=${5:-8000}
here=$(cd "$(dirname "$0")" && pwd)
primary=$(cd "$package" && ls *.safetensors | head -1)
health=$(mktemp); sed "s/@PORT@/$port/" "$here/healthcheck.sh" > "$health"
trust=()
[ -n "${PUBLISHER_FINGERPRINT:-}" ] && trust+=(--dangerously-approve-key "$PUBLISHER_FINGERPRINT")
[ -n "${PUBLISHER_NAMESPACE:-}" ] && trust+=(--dangerously-accept-publisher "$PUBLISHER_NAMESPACE")
# The disk prefix tier uses io_uring, which Docker's default seccomp profile denies.
exec docker run --name mimo-pro --network host --init \
  --user "$(id -u):$(id -g)" \
  --gpus "\"device=${GPUS:-all}\"" \
  --security-opt seccomp="$here/seccomp-io-uring.json" \
  --ulimit memlock=-1:-1 --ulimit nofile=1048576:1048576 \
  --health-cmd="bash /healthcheck.sh" --health-start-period=2400s \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -v "$package":/model:ro -v "$state":/state -v "$cache":/prefix-cache -v "$health":/healthcheck.sh:ro \
  "$image" \
  run "/model/$primary" \
    --gpus 0,1,2,3,4,5,6,7 \
    --max-concurrency 64 \
    --mtp-draft-tokens 7 \
    --mtp-adaptive-verification \
    --scheduler-policy throughput \
    --kv-cache-limit memory \
    --prefix-cache-memory 64G \
    --prefix-cache-disk /prefix-cache \
    --prefix-cache-disk-limit 160G \
    --prefix-cache-disk-transport staged \
    --served-model-name mimo-pro \
    --publisher-keys /state/development-publisher.json \
    --trust-store /state/trust.toml \
    "${trust[@]}" \
    --host 0.0.0.0 --port "$port"
