#!/usr/bin/env bash
# Kimi-K3.1-Pro EXL3-2bit — FLARE TP8 on switches B+D (turin :8001 `smart`).
#
# Spec: /home/tom/kimi31-exl3/TP8-CONFIG-PLAN.md §6 (revision 2, 2026-11-09).
# Package: our own ignition export (dev-signed) of /storage/models/Kimi-K3.1-Pro-EXL3-2bit on tree
#   /home/tom/build/flare-k31exl3 branch kimi31-exl3-turin (release 01327bf2 + block verification),
#   FlashInfer 6a842a7d; TP8/DCP8, c4, DFlash2 K7, prefill chunk 3072.
# Provenance (L1+ride final export, export-20261109-230047.log, rc=0 2026-11-09T23:24+07):
#   cache entry gb202-tp8-202611091623 (--speculative-token-capacity 20, PATCH_CAPACITIES cap
#   16,384, --image-concurrency 1 [owner rider], --prefill-island-reduce [L1, flashinfer 78b884989],
#   --reuse-weights). Flags verified in the export log's `=== args` line (fail-fast check) and
#   by the IslandARLaunch kernel object in the graph payload; pin values below measured from it.
#   (The prior entry gb202-tp8-202611091534 was exported with the DEFAULT flags — env lost in
#   --detach, since fixed — and is NOT the L1 config despite its old provenance note.)
# Derived from /opt/mimo-pro/start-mimo-pro-flare.sh (the previous B+D `smart` launcher):
#   - GPUs by PCI root port (80=B, f0=D) via flare-bd/gpu-select.py, UUIDs to docker, B then D
#   - unified package layout (model.safetensors + weights/ + deployments/) guarded by file counts
#   - package mounted at /model (singular): the router proxy's identity probe reads that mount
#   - cpuset NUMA node 0 (fast owns node 1); --cpuset-mems 1 = pinned prefix tier GPU-local (J1 L2)
#   - image pinned by docker ID; trust = our development key (signature verification stays on)
#
# Usage: ./start-kimi31-exl3.sh          start (or replace) the container
#        ./start-kimi31-exl3.sh --check  verify image/package/GPU pins only (nothing started)
# Optional serve-flag env hooks (empty = today's behavior; both verified in `run --help`):
#        ALPHA=<x>              adds --mtp-adaptive-verification-alpha <x>  (engine default 1.0)
#        MAX_VERIF_BUDGET=<n>   adds --mtp-max-verification-token-budget <n> (default: all captured capacities)
# Never SIGKILL this container: `docker stop -t 90 kimi31-exl3` (ranks shut down in order).
set -euo pipefail

NAME="kimi31-exl3"
PORT="${LLM_PORT:-8001}"
TP=8
IMAGE="${IMAGE:-local/flare:rt-k31exl3-chunkfix-5585a9a7}"   # chunk-fix image (deterministic solo chunking) = W13 merged lineage 5585a9a7
# Built 2026-11-09 from kimi31-exl3-turin @ 35134e2b (K2 patch cap + MoonViT auto-scale-to-fit,
# J1 GO-WITH-FIXES). Binary smoke-checked: flare 0.1.0, run --help lists the alpha flag.
IMAGE_SHA="${IMAGE_SHA:-sha256:50bda5a7f78e113cee8a7e566f156ecd50f307408a401f1d3256dd338dd0db0c}"
# Rollback pair (pre-chunkfix): IMAGE=local/flare:rt-k31exl3-35134e2b IMAGE_SHA=sha256:d24a91cfed8c1d35f286c95ec2bb7b3ac8ef5a508530c58bf519e042ee71201d
PKG_LINK="${PKG_DIR:-/opt/models/Kimi-K3.1-Pro-EXL3-TP8}"   # symlink to the package cache entry
PKG_PRIMARY="model.safetensors"
PKG="/model/$PKG_PRIMARY"
PKG_PRIMARY_SHA="${PKG_PRIMARY_SHA:-ef06d73556069947}"  # sha256(model.safetensors)[:16], gb202-tp8-202611100809 (E1.1: ring threshold 1024 + BF16 + trellis shapes + context 396288 = pool/3). Earlier: gb202-tp8-202611100247 = 4b41bb6a8edf50ab (ring >=2560 rows; relink + .bak-w6 to roll back); gb202-tp8-202611100153 = a42ad7204608d0ec (FP32 owner ring, pool 786k); gb202-tp8-202611091623 = afccc82f2fa2b5cf (L1)
EXPECT_WEIGHTS="${EXPECT_WEIGHTS:-752}"            # files under weights/ (94 per rank x 8) — recorded after export
EXPECT_DEPLOYMENTS="${EXPECT_DEPLOYMENTS:-25}"     # files under deployments/ — recorded after export (25 at cap 20 + K2 riders + image-concurrency 1)
STATE_HOST="/opt/kimi31-exl3/state"
BOOTLOG_DIR="/opt/kimi31-exl3/bootlogs"
HEALTHCHECK_SRC="/home/tom/ds41f/flare-healthcheck.sh"
PUBLISHER_SRC="/home/tom/.config/flare/development-publisher.json"
# W15 sanitization: the owner's dev-key fingerprint is [REDACTED] in this
# reproduction copy. A reproducer signs packages with their OWN development key
# and must put that key's fingerprint here (see tools/serve/README.md, signing).
TRUST_FP="[REDACTED - your own development-key fingerprint]"
TRUST_NS="local-development"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-4}"
DRAFT_TOKENS="${DRAFT_TOKENS:-7}"                  # package maximum 7; 4 = plan experiment X1
# Optional runtime-flag hooks (empty = today's behavior; see header):
ALPHA="${ALPHA:-0.6}"                                 # --mtp-adaptive-verification-alpha <x> (owner 2026-11-09: ship 0.6)
MAX_VERIF_BUDGET="${MAX_VERIF_BUDGET:-}"           # --mtp-max-verification-token-budget <n>
MEMORY_LIMIT="${MEMORY_LIMIT:--0G}"                # author's qualified value: all memory above the native guard
SCHEDULER_POLICY="${SCHEDULER_POLICY:-throughput}"
# Host prefix tiers: GPU -> 64G pinned RAM -> 160G disk (restore-on-match). io_uring needs the seccomp allowance.
PREFIX_CACHE_MEMORY="${PREFIX_CACHE_MEMORY:-64G}"
PREFIX_CACHE_DISK_HOST="${PREFIX_CACHE_DISK_HOST:-/dockers/prefix-cache/kimi31-exl3}"
PREFIX_CACHE_DISK_LIMIT="${PREFIX_CACHE_DISK_LIMIT:-160G}"
# --prefix-cache-disk-transport does not exist in this runtime (release-tree Rust = 8893f8c1; the flag is a
# mimo-pro-branch addition). Set PREFIX_CACHE_DISK_TRANSPORT only with an image whose `run --help` lists it.
PREFIX_CACHE_DISK_TRANSPORT="${PREFIX_CACHE_DISK_TRANSPORT:-}"
SECCOMP_PROFILE="${SECCOMP_PROFILE:-/opt/mimo-pro/flare-bd/seccomp-io-uring.json}"
CPUSET="${CPUSET:-0-55,112-167}"
CHECK_ONLY=0
[ "${1:-}" = "--check" ] && CHECK_ONLY=1

# ── Image identity guard (fail on tag drift) ─────────────────────────────────
[ -n "$IMAGE_SHA" ] || { echo "FATAL: IMAGE_SHA not set — pin the image ID after the build" >&2; exit 1; }
actual="$(docker inspect --format '{{.Id}}' "$IMAGE" 2>/dev/null || true)"
if [ "$actual" != "$IMAGE_SHA" ]; then
  echo "FATAL: $IMAGE resolves to '$actual', expected $IMAGE_SHA" >&2
  exit 1
fi

# ── Package guard (unified layout, counts recorded at export time) ───────────
[ -f "$SECCOMP_PROFILE" ] || { echo "FATAL: seccomp profile $SECCOMP_PROFILE missing" >&2; exit 1; }
PKG_DIR="$(readlink -f "$PKG_LINK")"
[ -d "$PKG_DIR" ] || { echo "FATAL: package dir $PKG_LINK -> '$PKG_DIR' missing" >&2; exit 1; }
[ -f "$PKG_DIR/$PKG_PRIMARY" ] || { echo "FATAL: primary $PKG_PRIMARY missing under $PKG_DIR" >&2; exit 1; }
[ "$(sha256sum "$PKG_DIR/$PKG_PRIMARY" | cut -c1-16)" = "$PKG_PRIMARY_SHA" ] || { echo "FATAL: $PKG_PRIMARY digest drift in $PKG_DIR (want $PKG_PRIMARY_SHA)" >&2; exit 1; }
n_weights=$(find "$PKG_DIR/weights" -type f | wc -l); n_deploy=$(find "$PKG_DIR/deployments" -type f | wc -l)
n_links=$(find "$PKG_DIR/" -type l | wc -l)
if [ "$n_weights" -ne "$EXPECT_WEIGHTS" ] || [ "$n_deploy" -ne "$EXPECT_DEPLOYMENTS" ] || [ "$n_links" -ne 0 ]; then
  echo "FATAL: package layout drift in $PKG_DIR: weights $n_weights/$EXPECT_WEIGHTS deployments $n_deploy/$EXPECT_DEPLOYMENTS symlinks $n_links" >&2
  exit 1
fi
[ -d "$PKG_DIR/weights/tp8" ] || { echo "FATAL: not a TP8 package ($PKG_DIR/weights/tp8 missing)" >&2; exit 1; }

# ── Package drive PCIe link guard (Gen1-mis-train class, incident 2026-11-10) ─
# The /opt/models NVMe (c2:00.0 behind root port c0:01.2) once trained at Gen1 x1
# and stayed there for days: every boot streamed the 706 GiB package at 0.94 GB/s
# (807 s weight loads) and every export/copy on the drive was throttled. The
# mis-train is silent at POST and does not survive as an obvious fault, so fail
# here whenever the drive's current_link_speed is below its max_link_speed.
# Resolves the drive the same way the incident was diagnosed: /opt/models mount
# → /dev/nvmeXn1p1 → nvme controller → its PCI device sysfs. Runs at start and
# in --check. (nvme-link-policy also enforces the link; this catches a boot that
# mis-trained anyway.)
pkg_src="$(findmnt -no SOURCE /opt/models 2>/dev/null || true)"
pkg_disk="$(lsblk -no PKNAME "$pkg_src" 2>/dev/null | head -1 | tr -d ' ')"
pkg_ctrl="$(sed -E 's/^(nvme[0-9]+)n[0-9]+$/\1/' <<<"$pkg_disk")"
pkg_pci="$(readlink -f "/sys/class/nvme/$pkg_ctrl/device" 2>/dev/null || true)"
if [ ! -r "$pkg_pci/current_link_speed" ] || [ ! -r "$pkg_pci/max_link_speed" ]; then
  echo "FATAL: cannot read link speed sysfs for /opt/models drive ($pkg_src -> $pkg_disk -> ${pkg_pci:-unresolved})" >&2
  exit 1
fi
pkg_cur="$(awk '{print $1}' "$pkg_pci/current_link_speed")"
pkg_max="$(awk '{print $1}' "$pkg_pci/max_link_speed")"
if awk -v c="$pkg_cur" -v m="$pkg_max" 'BEGIN{exit !(c+0 < m+0)}'; then
  echo "FATAL: /opt/models drive $pkg_ctrl ($pkg_pci) link $(cat "$pkg_pci/current_link_speed") < max $(cat "$pkg_pci/max_link_speed") — Gen1-mis-train class; retrain the link before start (see FABLE-PERF-LEVERS.md §1.1)" >&2
  exit 1
fi
echo "link OK: /opt/models drive $pkg_ctrl at ${pkg_cur} GT/s (max ${pkg_max})" >&2

# ── GPU pin: PCI root port → switch (80=B, f0=D), fail-closed, UUIDs ─────────
GPU_LIST="$(python3 /opt/mimo-pro/flare-bd/gpu-select.py B,D)"
n=$(awk -F, '{print NF}' <<<"$GPU_LIST")
[ "$n" -eq "$TP" ] || { echo "FATAL: switches B+D resolved $n GPUs, need TP=$TP" >&2; exit 1; }

if [ "$CHECK_ONLY" = "1" ]; then
  echo "CHECK OK: image $IMAGE_SHA, package $PKG_DIR ($n_weights weights, $n_deploy deployments), GPUs $GPU_LIST" >&2
  echo "CHECK OK: would run --max-concurrency $MAX_CONCURRENCY --mtp-draft-tokens $DRAFT_TOKENS --mtp-adaptive-verification${ALPHA:+ --mtp-adaptive-verification-alpha $ALPHA}${MAX_VERIF_BUDGET:+ --mtp-max-verification-token-budget $MAX_VERIF_BUDGET} --kv-cache-limit memory --memory-limit=$MEMORY_LIMIT --scheduler-policy $SCHEDULER_POLICY --prefix-cache-memory $PREFIX_CACHE_MEMORY --prefix-cache-disk $PREFIX_CACHE_DISK_HOST --prefix-cache-disk-limit $PREFIX_CACHE_DISK_LIMIT ${PREFIX_CACHE_DISK_TRANSPORT:+--prefix-cache-disk-transport $PREFIX_CACHE_DISK_TRANSPORT }--served-model-name smart on :$PORT (nothing started)" >&2
  exit 0
fi

echo "transport GPU list: $GPU_LIST" >&2

# ── Name free (bounded retry) + port fail-closed ─────────────────────────────
for _i in $(seq 1 24); do
  docker rm -f "$NAME" >/dev/null 2>&1
  [ -z "$(docker ps -aq --filter "name=^/${NAME}$")" ] && break
  sleep 5
done
[ -z "$(docker ps -aq --filter "name=^/${NAME}$")" ] || { echo "FATAL: cannot free container name $NAME" >&2; exit 1; }
ss -ltn "sport = :$PORT" | grep -q LISTEN && { echo "FATAL: port $PORT busy — stop the holder" >&2; exit 1; }

# ── Drain gates on the 8 pinned cards only (coarse then strict) ──────────────
for i in $(seq 1 60); do
  busy=0
  for g in ${GPU_LIST//,/ }; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null || echo 99999)
    [ "$used" -gt 4096 ] && { busy=1; break; }
  done
  [ "$busy" -eq 0 ] && break
  sleep 5
done
[ "$busy" -eq 0 ] || { echo "FATAL: pinned GPUs still occupied (>4096 MiB)" >&2; exit 1; }
for i in $(seq 1 60); do
  busy=0
  for g in ${GPU_LIST//,/ }; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null || echo 99999)
    [ "$used" -gt 1600 ] && { busy=1; break; }
  done
  [ "$busy" -eq 0 ] && break
  sleep 5
done
[ "$busy" -eq 0 ] || { echo "FATAL: GPUs not fully drained (<1600 MiB)" >&2; exit 1; }

echo "[launch] $NAME flare tp$TP port=$PORT gpus=$GPU_LIST pkg=$PKG_DIR conc=$MAX_CONCURRENCY draft=$DRAFT_TOKENS alpha=${ALPHA:-default1.0} verif_budget=${MAX_VERIF_BUDGET:-all} image=$IMAGE" >&2

mkdir -p "$STATE_HOST" "$BOOTLOG_DIR" "$PREFIX_CACHE_DISK_HOST"
[ -f "$STATE_HOST/development-publisher.json" ] || cp "$PUBLISHER_SRC" "$STATE_HOST/development-publisher.json"
HC_RENDERED="$STATE_HOST/healthcheck.sh"
sed "s/@PORT@/$PORT/" "$HEALTHCHECK_SRC" > "$HC_RENDERED"

BOOTLOG="$BOOTLOG_DIR/kimi31-exl3-boot-$(date +%Y%m%d-%H%M%S).log"

docker run ${DETACH:+-d} --name "$NAME" --network host --init \
  --user "$(id -u):$(id -g)" \
  --cpuset-cpus "$CPUSET" \
  --cpuset-mems 1 \
  --gpus "\"device=$GPU_LIST\"" \
  --security-opt seccomp="$SECCOMP_PROFILE" \
  --ulimit memlock=-1:-1 \
  --ulimit nofile=1048576:1048576 \
  --health-cmd="bash /healthcheck.sh" \
  --health-start-period=2400s \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -e M_MODEL_LABEL=Kimi-K3.1-Pro-EXL3 \
  -v "$PKG_DIR":/model:ro \
  -v "$STATE_HOST":/state \
  -v "$PREFIX_CACHE_DISK_HOST":/prefix-cache \
  -v "$HC_RENDERED":/healthcheck.sh:ro \
  "$IMAGE" \
  run "$PKG" \
    --gpus 0,1,2,3,4,5,6,7 \
    --max-concurrency "$MAX_CONCURRENCY" \
    --mtp-draft-tokens "$DRAFT_TOKENS" \
    $([ "${ADAPTIVE:-1}" = 0 ] && echo --no-mtp-adaptive-verification || echo --mtp-adaptive-verification) \
    ${ALPHA:+--mtp-adaptive-verification-alpha} ${ALPHA:+"$ALPHA"} \
    ${MAX_VERIF_BUDGET:+--mtp-max-verification-token-budget} ${MAX_VERIF_BUDGET:+"$MAX_VERIF_BUDGET"} \
    --scheduler-policy "$SCHEDULER_POLICY" \
    --kv-cache-limit memory \
    --memory-limit="$MEMORY_LIMIT" \
    --prefix-cache-memory "$PREFIX_CACHE_MEMORY" \
    --prefix-cache-disk /prefix-cache \
    --prefix-cache-disk-limit "$PREFIX_CACHE_DISK_LIMIT" \
    ${PREFIX_CACHE_DISK_TRANSPORT:+--prefix-cache-disk-transport "$PREFIX_CACHE_DISK_TRANSPORT"} \
    --served-model-name smart \
    --publisher-keys /state/development-publisher.json \
    --trust-store /state/trust.toml \
    --dangerously-approve-key "$TRUST_FP" \
    --dangerously-accept-publisher "$TRUST_NS" \
    --host 0.0.0.0 --port "$PORT"
# Detached: keep the container log (lost with the container otherwise).
[ -n "${DETACH:-}" ] && nohup docker logs -f "$NAME" > "$BOOTLOG" 2>&1 &
exit 0
