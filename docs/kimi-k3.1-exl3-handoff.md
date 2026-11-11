# Kimi K3.1 EXL3 serving

The production-signed package for
[local-inference-lab/Kimi-K3.1-Pro-EXL3-2bit](https://huggingface.co/local-inference-lab/Kimi-K3.1-Pro-EXL3-2bit)
serves eight RTX PRO 6000 Blackwell 96 GB GPUs with TP8/DCP8, DFlash2 K7,
vision, four concurrent requests and a 3072-token prefill limit. The signed
export was qualified at 262,144 tokens per request, the context it baked as
its maximum; the recipe now declares 1,048,576, so re-exports bake a larger
maximum sequence length and page table, and requests are admitted up to the
boot KV pool. The qualified host allocated 1,281,024 physical KV tokens
shared across requests and cached histories.

This guide starts from the published checkpoint. Encoding the source weights
is outside its scope. The model card contains the checkpoint description,
complete source-pinned preparation commands and machine-readable benchmark
results.

## Start the signed package

Install NVIDIA Container Toolkit and enable peer access between the eight
selected GPUs. The tested driver is 615.71.10. The image is Linux x86-64;
the prepared package is for SM120 and cannot change TP size at loading.

```bash
mkdir -p "$PWD/flare-cache"
docker run --detach --name kimi-k31 \
  --user 0:0 --gpus '"device=0,1,2,3,4,5,6,7"' --ipc host \
  --ulimit memlock=-1:-1 --ulimit nofile=65536:65536 \
  --env NVIDIA_DRIVER_CAPABILITIES=compute,utility --env HOME=/root \
  --publish 127.0.0.1:8000:8000 \
  --mount "type=bind,src=$PWD/flare-cache,dst=/root/.cache/flare" \
  ghcr.io/local-inference-lab/flare:k31-exl3-tp8-c4-dflash2-20261108 \
  run local-inference-lab/Kimi-K3.1-Pro-EXL3-2bit \
  --revision gb202-tp8-c4-p3072-dflash2-k7-20261108 --filename model.safetensors \
  --gpus 0,1,2,3,4,5,6,7 --host 0.0.0.0 --port 8000 \
  --max-concurrency 4 --mtp-draft-tokens 7 --memory-limit=-0G \
  --prefix-cache-memory 4G --prefix-cache-disk-limit 0 \
  --dangerously-accept-publisher local-inference-lab \
  --dangerously-approve-key 937c0143f6856d3ee8ad164f62e3d804bd825d7a8750d0accb9a5849f703a8c1

docker logs -f kimi-k31
```

The approval flags name the exact publisher key; they do not disable signature
verification. Use them only when you trust that key. Running the model needs no
private key or passphrase. Stop gracefully with
`docker stop --time 90 kimi-k31`.

The example exposes only host loopback. Configure authentication with
`--api-key-file` before changing the binding for remote access.

## Settings

| Setting | Default | Example override |
|---|---|---|
| Concurrent requests | Package capacity, 4 | `--max-concurrency 1` lowers admission; 4 is the qualified release profile |
| Draft depth | Package maximum, 7 | `--mtp-draft-tokens 7` selects the tested profile |
| GPU memory | Configuration/package reserve | `--memory-limit=-0G` leaves only the native safety guard |
| Host prefix retention | Configuration, otherwise disabled | `--prefix-cache-memory 4G`; automatic intake buffers are additional |
| Disk retention | Configuration when disk is enabled | `--prefix-cache-disk-limit 0` disables it |
| Reasoning | `max` when omitted | Chat Completions `"reasoning_effort":"medium"` |

K3.1 Chat Completions accepts `low`, `medium`, `high` and `max`.
The Responses adapter coarsens `medium` to `high` in this revision; use Chat
Completions when that distinction matters. Keep assistant reasoning and tool
history in multi-turn requests.

Prefill, context, vision and maximum speculative capacities are captured into
the package. Runtime flags cannot add missing graphs or raise those limits.

## Precision and preparation

The raw EXL3 checkpoint keeps its nonexpert source tensors. The prepared release
explicitly uses `--online-mxfp8` for eligible target, vision and draft projections,
plus `--draft-head-quantization mxfp8`. DSA index-query projections remain BF16,
sharded by output heads with the full-width arithmetic. Routed experts retain
their EXL3 codes and use A16 kernels. This is not a BF16-dense reference.

Serving code is pinned to Flare `01327bf2b0bb4a1626a313bac64a73a749da2e9b`;
preparation uses FlashInfer/B12X
`6a842a7d054e6e3e1db32f4c984ca1b9c0460a97`.
The container uses the repository's standard Dockerfile. Its immutable digest is
`sha256:775016c31860bb4451aad0aa16d4773fc0b4a317c90c72ed02d2eb5bcc5c9496`.

The [model card](https://huggingface.co/local-inference-lab/Kimi-K3.1-Pro-EXL3-2bit)
contains complete export commands, the pinned raw checkpoint and DFlash2
revision, and the three Torch 2.14.1 preparation patches. It distinguishes a
fresh export from graph-only re-export with `--reuse-weights`.
Do not apply the research-only no-clone custom-op decomposition patch for this
release. Native serving does not use Torch or Python.

## Measurements

Docker measurements used eight RTX PRO 6000 Blackwell Workstation GPUs on an
AMD EPYC 9575F host, driver 615.71.10, 600 W GPU limits and 16,365 MHz memory
clocks. The image does not set power limits or overclocks.

| Workload | Result |
|---|---:|
| C1 coding decode, median of three prompts, reasoning low | 109.04 tok/s |
| C4 coding, common decode interval | 158.45 tok/s aggregate |
| Cold 32k prefill, median of three prompts | 1,951.16 tok/s |
| Concurrent image checks | 4/4 |
| Estonia, C4, medium, 30 resamples | 30/30 correct |
| Hotel Lights, C4, medium, 30 resamples | 30/30 correct |

Decode excludes prefill and uses native token timestamps. Sampled responses
vary in length and draft acceptance; these are observed measurements, not a
fixed speed guarantee. The reasoning profiles resample two fixed prompts,
not independent question datasets.

The signed package preserves the qualified graph catalogs, cache geometry,
speculation and device-allocation sizes. Its identical computation passed four
simultaneous contexts totaling 1,000,345 prompt tokens before production
signing, with exact marker recall and no OOM. Full-model KLD against official
MXFP4 and other hardware/TP configurations remain unqualified.
