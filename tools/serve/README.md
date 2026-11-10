# tools/serve — reproduction copies of the serve side (W15, share-gap 3)

These are **REPRODUCTION COPIES** of the live serve files for the `smart` :8001
container (`kimi31-exl3`). The live files were **not** touched when these copies
were made (W15 rule; verified by hash before/after). They exist so a reproducer
can see how a package + runtime image become an actual service: alpha 0.6,
concurrency 4, draft tokens 7, the prefix-cache tiers, the CPU set, and the R13
drive-link guard all live in the launcher, not in the export or the image.

| File (here) | Live original | Role |
|---|---|---|
| `start-kimi31-exl3.sh` | `/opt/kimi31-exl3/start-kimi31-exl3.sh` | Launcher: image/package/GPU/drive-link guards (fail-closed), then one `docker run` of the pinned image with the pinned package mounted at `/model`, serve flags, cache tiers, trust flags. `--check` verifies pins without starting anything. |
| `kimi31-exl3.service` | `/etc/systemd/system/kimi31-exl3.service` | systemd unit. `ExecStartPre` runs the launcher with `DETACH=1`; `ExecStop` is `docker stop -t 90` (never SIGKILL a live 8-way serve). Installed DISABLED by house convention (the owner boots it supervised). |
| `flare-healthcheck.sh` | `/home/tom/ds41f/flare-healthcheck.sh` | Container healthcheck (the flare image has no curl/wget; bash `/dev/tcp` GET `/health`, succeed only on HTTP 200). The launcher renders `@PORT@` and mounts it read-only at `/healthcheck.sh`. |
| `seccomp-io-uring.json` | `/opt/mimo-pro/flare-bd/seccomp-io-uring.json` | Seccomp profile passed as `--security-opt seccomp=…`. io_uring needs the allowance it carries; the launcher fails closed if it is missing. |

Sanitization applied to the copies:

- `TRUST_FP` in `start-kimi31-exl3.sh` is `[REDACTED …]`. It is the owner's
  development-key trust fingerprint (see "Signing" below) — the only
  credential-adjacent value in the launcher. Everything else is byte-identical
  to the live originals as copied on 2026-11-10.
- No key material is present anywhere in this directory. The launcher *references*
  host paths for the publisher key and trust store (`$PUBLISHER_SRC`,
  `/home/tom/.config/flare/…`); those files are host-local and deliberately not
  copied.

## Signing (share-gap 4)

Packages are signed with a **local development key** (the publisher key file the
launcher copies into `$STATE_HOST/development-publisher.json`), and the launcher
trusts that key by fingerprint: `TRUST_FP` near the top of
`start-kimi31-exl3.sh`, passed to the serve as
`--dangerously-approve-key "$TRUST_FP"` (plus
`--dangerously-accept-publisher "$TRUST_NS"`). A reproducer signs with **their
own** key and must change `TRUST_FP` to their own key's fingerprint — the
[REDACTED] placeholder in this copy is where it goes. Signature verification
stays on; only the trusted fingerprint is per-site.

## Wiring (what a reproducer must substitute)

The copies are path-faithful to this host. On another box adapt these variables
in `start-kimi31-exl3.sh` (and the unit's `ExecStart` path):

- `HEALTHCHECK_SRC` -> your copy of `flare-healthcheck.sh` (here in `tools/serve/`).
- `PUBLISHER_SRC` -> your own development publisher key file (never commit it).
- `TRUST_FP` -> your own key fingerprint (see above).
- `SECCOMP_PROFILE` -> your copy of `seccomp-io-uring.json` (here in `tools/serve/`).
- `PKG_LINK` / `PKG_PRIMARY_SHA` / `EXPECT_WEIGHTS` / `EXPECT_DEPLOYMENTS` ->
  your exported package and its recorded counts/digest (the launcher fail-closes
  on drift).
- `IMAGE` / `IMAGE_SHA` -> your built runtime image and its ID
  (`tools/export-kimi31-exl3.sh --image` records it; note the image-ID pin is
  not reproducible across toolchains — see the top-level README "Gaps").
- `PREFIX_CACHE_DISK_HOST`, `CPUSET`, `STATE_HOST`, `BOOTLOG_DIR` -> local paths.
- `gpu-select.py` (referenced for GPU selection by PCI root port) is host
  topology tooling; on another host resolve your own GPU set (top-level README,
  "Topology and host prerequisites").

Host-side pieces the launcher assumes but does not carry: the NVMe link policy
(`/usr/local/sbin/nvme-link-policy.sh`, `nvme-link-policy.service`) — the
drive-link fix behind the R13 guard's incident class — and the GPU selection
helper. Neither is copied here (outside serve scope; see the topology note).

Caveat: these copies are as of 2026-11-10 (package `gb202-tp8-202611100809`
pinned). The live files can drift afterwards; diff before trusting parity.
