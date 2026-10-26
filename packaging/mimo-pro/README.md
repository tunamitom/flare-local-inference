# MiMo-V2.6-Pro-MOPD on FLARE: reproduce the shipped package and image

This directory holds everything used to export the MiMo-V2.6-Pro-MOPD package and build the runtime
image that serve it. Both come from this checkout alone.

The package and the runtime image both come from this checkout: the export runs this branch's ignition
tree in the environment below, and the image compiles this branch's helios and flare trees.

Hardware: eight RTX PRO 6000 Blackwell (sm_120, 96 GB) on PCIe, tensor parallel 8, no NVLink.

## 1. Export environment

```sh
packaging/mimo-pro/setup-venv.sh
```

It runs `uv sync --frozen --extra prepare`, which pins FlashInfer at `344e48375564a0f55af21039efd34cab3c5fd8cc`
(`uv.lock`). It then applies the five b12x patches in `patches/` to the installed FlashInfer
(`patches/README.md` lists what each one is for). The export fails, or silently prepares slower
graphs, without them. In particular, `b12x-decode-past-eight-rows.patch` keeps rounds of 9-24
verified rows on the fused decode paths.

## 2. Export

```sh
CUDA_VISIBLE_DEVICES=<eight GPUs> packaging/mimo-pro/export.sh /path/to/MiMo-V2.6-Pro-MOPD
```

The script fixes every export option:
- TP8, max concurrency 64, image concurrency 4, prefill chunk 8192.
- DFlash MTP with 7 draft tokens and packed verification over 256 target rows (graph budget 64).
- CSF expert scales.
- The recipe's A4/A16 prefill variants from 512 tokens.

The package lands in the Flare model cache, under `models/<checkpoint-dir>/gb202-tp8-<date>/` beside
the cache's shared blobs.

The export takes about an hour and peaks at about 1.1 TB of disk: about 0.56 TB of package plus the
package cache.

## 3. Runtime image

```sh
packaging/mimo-pro/build-image.sh [TAG]
```

This is a plain `docker build` of the repository Dockerfile, which compiles `flare` from `flare/` and
`helios/` with `cargo --locked`.

## 4. Serve

```sh
GPUS=<eight GPU UUIDs, comma-separated> PUBLISHER_FINGERPRINT=<key> PUBLISHER_NAMESPACE=<name> \
  packaging/mimo-pro/serve.sh IMAGE PACKAGE_DIR STATE_DIR PREFIX_CACHE_DIR [PORT]
```

The flags are the shipped ones:
- Throughput scheduler, adaptive MTP verification with 7 draft tokens, checkpoint segment 256.
- KV cache sized from memory.
- Prefix cache of 64 GB RAM plus a 160 GB staged disk tier.

The disk tier uses io_uring, which Docker's default seccomp profile blocks, so `serve.sh` passes
`seccomp-io-uring.json`: Moby's default profile plus the io_uring calls. `healthcheck.sh` probes
`/health` without curl, which the image does not have.
