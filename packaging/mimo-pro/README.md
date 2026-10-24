# MiMo-V2.6-Pro-MOPD on FLARE: reproduce the shipped package and image

This directory holds everything used to export the MiMo-V2.6-Pro-MOPD package and build the runtime
image that serve it. Both come from this checkout alone.

Shipped build (2026-10-24): package `MiMo-V2.6-Pro-MOPD-FLARE-C64-T256-BD-KV1L-QKV-A512-FAR-DUAL-DG19PE`
and image `local/flare:rt-mimo-pro-sw-0012bc12`. The runtime source is this branch's helios and flare
trees. The package was exported from this branch's ignition tree with the environment below.

Hardware: eight RTX PRO 6000 Blackwell (sm_120, 96 GB) on PCIe, tensor parallel 8, no NVLink.

## 1. Export environment

```sh
packaging/mimo-pro/setup-venv.sh
```

It runs `uv sync --frozen --extra prepare`, which pins FlashInfer at `622cec10eb75da189f4dadddabc2d142a33cd211`
(`uv.lock`). It then applies the six b12x patches in `patches/` to the installed FlashInfer
(`patches/README.md` lists what each one is for). The export fails, or silently prepares slower
graphs, without them. In particular, `b12x-decode-past-eight-rows.patch` keeps rounds of 9-24
verified rows on the fused decode paths.

## 2. Export

```sh
CUDA_VISIBLE_DEVICES=<eight GPUs> packaging/mimo-pro/export.sh /path/to/MiMo-V2.6-Pro-MOPD out/NAME.safetensors
```

The script fixes every export option and environment variable the shipped package used:
- TP8, max concurrency 64, prefill chunk 8192, speculative token capacity 256.
- DFlash MTP with 7 draft tokens, CSF expert scales.
- `FLARE_MIMO_SWITCH_SPECULATION=1`: on-device verification width at batch one.
- `FLARE_MIMO_TWO_PHASE_MAX_BATCH=0`.

The export takes about an hour and peaks at about 1.1 TB of disk: about 0.56 TB of package plus the
package cache.

## 3. Runtime image

```sh
packaging/mimo-pro/build-image.sh [TAG]
```

This is a plain `docker build` of the repository Dockerfile, which compiles `flare` from `flare/` and
`helios/` with `cargo --locked`. The SWITCH selector kernel ships prebuilt as
`helios/native/select_conditional.fatbin` beside its source `select_conditional.cu`, because the
serving image runs without a CUDA toolkit or PTX JIT.

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

## Bench of record (llm-inference-bench, 30 s cells, per-request seeds; Estonia 30/30)

| decode tok/s | 0 | 8k | 64k | 128k |
|---|---|---|---|---|
| c1 | 258.6 | 249.3 | 253.2 | 243.0 |
| c4 | 491.2 | 485.2 | 478.7 | 471.7 |
| c8 | 641.5 | 657.3 | 644.6 | 623.8 |

Prefill: 8,813 / 8,219 / 7,436 tok/s at 8k / 64k / 128k.

Single c1 cells vary by about ±5 tok/s with the drafter's acceptance on the sampled text, so compare
c1 on means of ten or more cells.
