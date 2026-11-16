#!/usr/bin/env bash
# Hy4-preview NVFP4 `smart` (:8001) — FLARE-engine TP8+CP8 world 8, switches B+D.
#
# Package: /opt/models/Hy4-nvfp4-w4a16-mse-gpu (W13 fused export, 88 shards,
#   export_report byte_verified) mounted /model:ro.
# Engine: ignition Hy4 recipe construct_rank + PcieDcpTransport (w14 bring-up
#   path), fused 656-byte FP8 record cache (M2), post-capture `Allocate cache`
#   line (P5 gate); MTP off. Serve runner: /opt/h4/h4_serve.py
#   (--host 0.0.0.0 --port 8001 --served-model-name smart — the argv is what
#   the command-app telemetry greps).
# Runtime: local/flare:rt-h4-cp8-<sha>, pinned by docker ID in THIS launcher.
# GPU selection: switch port + PCI bus only (GPU_BUS_MAP), fail-closed; the
#   8 serving cards are switches B and D. The off-switch spare 0000:11:00.0 is
#   campaign-work resource and is NEVER selected here (owner 2026-11-15).
# Docker transport: one `--device nvidia.com/gpu=<uuid>` per card (docker
#   29.1.3 fix — never the multi-UUID --gpus device=<list> form).
# Port: 8001 — the `smart` role port (alias-port convention). No env file.
# 2026-11-15 CREATED (initial bring-up). Rollback: rm /etc/systemd/system/h4.service
#   && docker rm -f h4 — h4 owns the `smart` role while it runs; the previous
#   `smart` holder (glm53) is restored by systemctl start glm53.
#
# Usage: start-h4.sh           start (replaces a stopped container of the same name)
#        start-h4.sh --check   verify pins without starting anything
set -euo pipefail

NAME="${NAME:-h4}"
PORT="${LLM_PORT:-8001}"
TP=8
IMAGE="${IMAGE:-local/flare:rt-h4-cp8-dg5}"
IMAGE_SHA="${IMAGE_SHA:-sha256:79af23f224f36cfce9a74ab9a9fdad1f5d431dea7fe0024ebf88575b97b54309}"
PKG_LINK="${PKG_DIR:-/opt/models/Hy4-nvfp4-w4a16-mse-gpu}"
SERVED_NAME="${SERVED_NAME:-smart}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-1}"
STATE_HOST="${STATE_HOST:-/opt/h4/state}"
BOOTLOG_DIR="${BOOTLOG_DIR:-/opt/h4/bootlogs}"
PREFIX_CACHE_HOST="${PREFIX_CACHE_HOST:-/opt/models/.h4-prefix-cache}"   # model NVMe, never the OS drive
HEALTHCHECK_SRC="/opt/h4/healthcheck-src.sh"
SECCOMP_PROFILE="${SECCOMP_PROFILE:-/opt/mimo-pro/flare-bd/seccomp-io-uring.json}"
TORCHRUN_PORT="${TORCHRUN_PORT:-29855}"
CHECK_ONLY=0
[ "${1:-}" = "--check" ] && CHECK_ONLY=1

# ── Image identity guard (the tag moves; the ID is the pin) ─────────────────
[ -n "$IMAGE_SHA" ] || { echo "FATAL: IMAGE_SHA not set" >&2; exit 1; }
actual="$(docker inspect --format '{{.Id}}' "$IMAGE" 2>/dev/null || true)"
[ "$actual" = "$IMAGE_SHA" ] || { echo "FATAL: $IMAGE resolves to '$actual', expected $IMAGE_SHA" >&2; exit 1; }
[ -f "$SECCOMP_PROFILE" ] || { echo "FATAL: seccomp profile $SECCOMP_PROFILE missing" >&2; exit 1; }

# ── Package guard (the W13 fused export) ────────────────────────────────────
PKG_DIR="$(readlink -f "$PKG_LINK")"
[ -d "$PKG_DIR" ] || { echo "FATAL: package dir $PKG_LINK -> '$PKG_DIR' missing" >&2; exit 1; }
for f in config.json export_report.json model.safetensors.index.json chat_template.jinja tokenizer.json; do
  [ -f "$PKG_DIR/$f" ] || { echo "FATAL: package file $f missing under $PKG_DIR" >&2; exit 1; }
done
n_shards=$(find "$PKG_DIR" -maxdepth 1 -name 'model-*.safetensors' | wc -l)
[ "$n_shards" -eq 88 ] || { echo "FATAL: package shard drift: $n_shards/88 model-*.safetensors" >&2; exit 1; }
byte_verified="$(python3 -c "import json;print(json.load(open('$PKG_DIR/export_report.json')).get('byte_verified'))")"
[ "$byte_verified" = "True" ] || { echo "FATAL: export_report byte_verified=$byte_verified (want True)" >&2; exit 1; }

# Package drive link guard (Gen1 mis-train class: boots at 0.9 GB/s instead of 9).
pkg_src="$(findmnt -no SOURCE /opt/models 2>/dev/null || true)"
pkg_disk="$(lsblk -no PKNAME "$pkg_src" 2>/dev/null | head -1 | tr -d ' ')"
pkg_ctrl="$(sed -E 's/^(nvme[0-9]+)n[0-9]+$/\1/' <<<"$pkg_disk")"
pkg_pci="$(readlink -f "/sys/class/nvme/$pkg_ctrl/device" 2>/dev/null || true)"
if [ -r "$pkg_pci/current_link_speed" ] && [ -r "$pkg_pci/max_link_speed" ]; then
  pkg_cur="$(awk '{print $1}' "$pkg_pci/current_link_speed")"; pkg_max="$(awk '{print $1}' "$pkg_pci/max_link_speed")"
  if awk -v c="$pkg_cur" -v m="$pkg_max" 'BEGIN{exit !(c+0 < m+0)}'; then
    echo "FATAL: /opt/models drive link ${pkg_cur} GT/s below its maximum ${pkg_max}" >&2; exit 1
  fi
else
  echo "FATAL: cannot read the /opt/models drive link speed" >&2; exit 1
fi

# ── GPU selection: switch port + PCI bus, fail-closed (8 cards, B+D) ────────
# The physical pins are the only constants; a missing/moved port aborts.
GPU_BUS_MAP="
00000000:83:00.0 SwB-B1-2
00000000:84:00.0 SwB-B3-4
00000000:85:00.0 SwB-B5-6
00000000:86:00.0 SwB-B7-8
00000000:F3:00.0 SwD-D1-2
00000000:F4:00.0 SwD-D3-4
00000000:F6:00.0 SwD-D7-8
00000000:F7:00.0 SwD-D9-10
"
GPU_RC=0
GPU_LIST="$(GPU_BUS_MAP="$GPU_BUS_MAP" /usr/bin/python3 - <<'PY'
import os, subprocess, sys
norm = lambda b: ":".join(p.lstrip("0") or "0" for p in b.strip().lower().split(":")).upper()
raw = subprocess.run(["nvidia-smi", "--query-gpu=pci.bus_id,uuid", "--format=csv,noheader"],
                     capture_output=True, text=True, check=True).stdout
by_bus = {}
for line in raw.splitlines():
    bus, uuid = (x.strip() for x in line.split(",", 1))
    by_bus[norm(bus)] = uuid
out = []
for line in os.environ["GPU_BUS_MAP"].strip().splitlines():
    bus, label = line.split()
    key = norm(bus)
    if key not in by_bus or not by_bus[key].startswith("GPU-"):
        print(f"FATAL: no device at switch port {label} (bus {bus}) — moved or empty", file=sys.stderr)
        sys.exit(2)
    print(f"  {label:11s} bus {bus:15s} -> {by_bus[key]}", file=sys.stderr)
    out.append(by_bus[key])
print(",".join(out))
PY
)" || GPU_RC=$?
if [ "$GPU_RC" -ne 0 ]; then
  echo "FATAL: GPU pinning failed (rc=$GPU_RC) — a switch port is empty or moved" >&2; exit 1
fi
n=$(awk -F, '{print NF}' <<<"$GPU_LIST")
[ "$n" -eq "$TP" ] || { echo "FATAL: resolved $n/$TP switch ports" >&2; exit 1; }

# ── NUMA: resolve each pinned BDF's node live; bind CPU+memory to the union ──
BDFS=(83:00.0 84:00.0 85:00.0 86:00.0 F3:00.0 F4:00.0 F6:00.0 F7:00.0)
nodes="$(for bdf in "${BDFS[@]}"; do cat "/sys/bus/pci/devices/0000:${bdf,,}/numa_node" 2>/dev/null; done | sort -un | grep -v '^-1$' || true)"
CPUSET=""; CPUSMETS=""
if [ -n "$nodes" ]; then
  CPUSMETS="$(paste -sd, <<<"$nodes")"
  for nnode in $nodes; do
    cl="$(cat "/sys/devices/system/node/node${nnode}/cpulist")"
    CPUSET="${CPUSET:+$CPUSET,}$cl"
  done
fi
if [ -z "$CPUSET" ]; then CPUSET="56-111,168-223"; CPUSMETS="1"; fi

if [ "$CHECK_ONLY" = "1" ]; then
  echo "CHECK OK: image $IMAGE_SHA, package $PKG_DIR ($n_shards shards, byte_verified=True)" >&2
  echo "CHECK OK: 8 switch ports resolved (B+D rank order), cpuset-cpus $CPUSET cpuset-mems $CPUSMETS" >&2
  echo "CHECK OK: would run torchrun 8x h4_serve.py --host 0.0.0.0 --port $PORT --served-model-name $SERVED_NAME --max-concurrency $MAX_CONCURRENCY on :$PORT (nothing started)" >&2
  exit 0
fi

echo "[pin] switch ports B+D -> UUIDs:" >&2
tr ',' '\n' <<<"$GPU_LIST" | sed 's/^/  /' >&2
echo "[numa] cpuset-cpus=$CPUSET cpuset-mems=$CPUSMETS (live union of the pinned cards' nodes)" >&2

# ── Adopt-if-healthy (a manual boot is adopted, not reloaded) ───────────────
if [ "$(docker inspect -f '{{.State.Running}}' "${NAME}" 2>/dev/null)" = "true" ] \
   && [ "$(curl -s --max-time 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:${PORT}/health)" = "200" ]; then
  echo "[adopt] ${NAME} already running and healthy — keeping it"; exit 0
fi

# ── Name + port: fail-closed, no silent takeover ─────────────────────────────
docker rm -f "$NAME" >/dev/null 2>&1 || true
for _i in $(seq 1 24); do
  [ -z "$(docker ps -aq --filter "name=^/${NAME}$")" ] && break
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  sleep 5
done
[ -z "$(docker ps -aq --filter "name=^/${NAME}$")" ] || { echo "FATAL: cannot free container name $NAME" >&2; exit 1; }
if ss -ltn "( sport = :$PORT )" | grep -q ":$PORT"; then
  echo "FATAL: :$PORT is still busy after removing $NAME — refusing to start" >&2; exit 1
fi
ss -ltn "sport = :$TORCHRUN_PORT" | grep -q LISTEN && { echo "FATAL: torchrun port $TORCHRUN_PORT busy" >&2; exit 1; }

mkdir -p "$STATE_HOST" "$BOOTLOG_DIR" "$PREFIX_CACHE_HOST"
chown -R "$(id -u):$(id -g)" "$STATE_HOST" "$BOOTLOG_DIR" "$PREFIX_CACHE_HOST"
HC_RENDERED="$STATE_HOST/healthcheck.sh"
sed "s/@PORT@/$PORT/" "$HEALTHCHECK_SRC" > "$HC_RENDERED"

BOOTLOG="$BOOTLOG_DIR/h4-boot-$(date +%Y%m%d-%H%M%S).log"

# docker 29.1.3 transport: one CDI device request per card — never the
# multi-UUID --gpus device=<list> form. CUDA_VISIBLE_DEVICES carries the same
# UUID list so LOCAL_RANK i maps to the i-th pin in switch order.
DEVICE_ARGS=()
IFS=',' read -ra _uuids <<<"$GPU_LIST"
for u in "${_uuids[@]}"; do DEVICE_ARGS+=(--device "nvidia.com/gpu=$u"); done

docker run -d --name "$NAME" --network host --init --cap-add SYS_PTRACE \
  --user "$(id -u):$(id -g)" \
  --cpuset-cpus "$CPUSET" \
  --cpuset-mems "$CPUSMETS" \
  "${DEVICE_ARGS[@]}" \
  --security-opt seccomp="$SECCOMP_PROFILE" \
  --ulimit memlock=-1:-1 \
  --ulimit nofile=1048576:1048576 \
  --health-cmd="bash /healthcheck.sh" \
  --health-start-period=2400s \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e CUDA_VISIBLE_DEVICES="$GPU_LIST" \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e M_MODEL_LABEL=Hy4 \
  -e HY4_SELECTION_AUDIT="${HY4_SELECTION_AUDIT:-}" \
  -e HY4_FINITE_WATCH="${HY4_FINITE_WATCH:-}" \
  -e HY4_SCORE_WIRE_AUDIT="${HY4_SCORE_WIRE_AUDIT:-}" \
  -e HY4_AR_AUDIT="${HY4_AR_AUDIT:-}" \
  -e HY4_CE_AUDIT="${HY4_CE_AUDIT:-}" \
  -e HY4_LANE_AUDIT="${HY4_LANE_AUDIT:-}" \
  -e HY4_LANE_AUDIT_EVERY="${HY4_LANE_AUDIT_EVERY:-}" \
  -e HY4_HANGWATCH_S="${HY4_HANGWATCH_S:-}" \
  -e HY4_TRACE="${HY4_TRACE:-}" \
  -e HY4_AB_FILE="${HY4_AB_FILE:-}" \
  -e HY4_IHC="${HY4_IHC:-tuned}" \
  -e HY4_FUSED_ROPE="${HY4_FUSED_ROPE:-on}" \
  -e HY4_FUSED_MERGE="${HY4_FUSED_MERGE:-on}" \
  -e HY4_HOST_TRIM="${HY4_HOST_TRIM:-on}" \
  -e HY4_BATCHED_WAITS="${HY4_BATCHED_WAITS:-}" \
  -e HY4_GRAPH_MOE="${HY4_GRAPH_MOE:-}" \
  -e HY4_COMM_TOKEN="${HY4_COMM_TOKEN:-}" \
  -e HY4_PLAIN_SYNC_CENSUS="${HY4_PLAIN_SYNC_CENSUS:-}" \
  -e HY4_SYNC_FREE="${HY4_SYNC_FREE:-decode}" \
  -e HY4_DECODE_GRAPH="${HY4_DECODE_GRAPH:-on}" \
  -e HY4_IHC_SPLIT="${HY4_IHC_SPLIT:-on}" \
  -e HY4_DECODE_GRAPH_PREWARM="${HY4_DECODE_GRAPH_PREWARM:-}" \
  -e HY4_DEV_EXEC="${HY4_DEV_EXEC:-1}" \
  -e HY4_TORCH_PROFILE="${HY4_TORCH_PROFILE:-}" \
  -e HY4_TORCH_PROFILE_MIN_NEW="${HY4_TORCH_PROFILE_MIN_NEW:-}" \
  -e HY4_STATE_CENSUS="${HY4_STATE_CENSUS:-}" \
  -e HY4_OVERLAP_SYNC_GUARD="${HY4_OVERLAP_SYNC_GUARD:-}" \
  -e HY4_OVERLAP_COMM_TOKEN="${HY4_OVERLAP_COMM_TOKEN:-}" \
  -e HY4_OVERLAP_CE_SCORES="${HY4_OVERLAP_CE_SCORES:-}" \
  -e HY4_OVERLAP_EXCLUSIVE_MOE="${HY4_OVERLAP_EXCLUSIVE_MOE:-}" \
  -e HY4_OVERLAP_VERIFY="${HY4_OVERLAP_VERIFY:-}" \
  -e HY4_OVERLAP_GATE="${HY4_OVERLAP_GATE:-kv}" \
  -e B12X_W4A16_STABLE_ROUTE_PACK="${B12X_W4A16_STABLE_ROUTE_PACK:-1}" \
  -e B12X_W4A16_COOP_RESERVE_SMS="${B12X_W4A16_COOP_RESERVE_SMS:-2}" \
  -e B12X_PCIE_DMA_SINGLE_CTA_WAITS="${B12X_PCIE_DMA_SINGLE_CTA_WAITS:-1}" \
  -e HY4_OVERLAP_ADMISSION_UNSAFE="${HY4_OVERLAP_ADMISSION_UNSAFE:-}" \
  -v "$PKG_DIR":/model:ro \
  -v "$STATE_HOST":/state \
  -v "$PREFIX_CACHE_HOST":/prefix-cache \
  -v "$HC_RENDERED":/healthcheck.sh:ro \
  "$IMAGE" \
  /opt/flare-local-inference/.venv/bin/python -m torch.distributed.run \
    --nproc_per_node=8 --master_port="$TORCHRUN_PORT" \
    /opt/h4/h4_serve.py \
      --snapshot /model \
      --host 0.0.0.0 --port "$PORT" --served-model-name "$SERVED_NAME" \
      --max-concurrency "$MAX_CONCURRENCY" \
      --headroom 2415919104 \
      --all-reduce-arm "1:oneshot,2-512:twoshot,1024-:dma" \
      --all-reduce-kernel-rows 512 \
      --lse-transport ce-relay --gather-transport ce-relay \
      --selection-wire dcp \
      --partial-wire expanded \
      --overlap 1 --profile-phases 0 \
      --profile-steps 64
# Keep the container log (lost with the container otherwise). Background only
# the follower: backgrounding the whole list forks a subshell that keeps the
# caller's stdout open until the container exits.
nohup docker logs -f "$NAME" > "$BOOTLOG" 2>&1 < /dev/null &
echo "bootlog: $BOOTLOG" >&2
exit 0
