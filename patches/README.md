# FlashInfer b12x patches for the MiMo V2 recipe

The MiMo V2 TP8 recipe depends on four changes to the b12x package in the
pinned FlashInfer (`03de0b663abe8f5dc69d4e10118587ab459b13eb`). They are not
applied by `uv sync`. Apply them to the installed package before preparing a
MiMo package:

```sh
for p in patches/b12x-*.patch; do
  patch -p1 -d .venv/lib/python3.12/site-packages/flashinfer/experimental < "$p"
done
```

| Patch | Needed by | Without it |
|---|---|---|
| `b12x-twoshot-bf16-tp8.patch` | TP8 two-shot reductions | TP8 preparation fails while it creates the two-shot channel |
| `b12x-mxfp4-a4-prefill.patch` | A4/A16 hybrid prefill experts | MiMo preparation fails unless `FLARE_MIMO_A4_PREFILL_MIN_TOKENS=0` exports pure A16 |
| `b12x-w4a16-topk-sum-loads.patch` | Faster W4A16 top-k sums | The top-k sum stays serialized; outputs are identical |
| `b12x-block-fp8-dependent-launch.patch` | MiMo decode and verification QKV projections | MiMo preparation fails importing `b12x.gemm.blockscaled._block_fp8_row_quant` |

Each patch applies cleanly to the pinned commit. These changes belong in
FlashInfer. Delete each one once a FlashInfer pin contains it.
