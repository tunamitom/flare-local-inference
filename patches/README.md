# FlashInfer b12x patches for the MiMo V2 recipe

The MiMo V2 TP8 recipe depends on six changes to the b12x package in the
pinned FlashInfer (`622cec10eb75da189f4dadddabc2d142a33cd211`). They are not
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
| `b12x-twoshot-bf16-shared-ipc.patch` | TP8 two-shot channels in package peer storage | MiMo preparation fails: `PCIeTwoShotBF16.from_exchange_group` rejects `ipc` |
| `b12x-decode-past-eight-rows.patch` | MiMo rounds of 9-24 verified rows (c4 width 3+, c8 width 2+) | Those rounds reduce through the two-shot plus a separate add and norm, and their experts pack routes and sum top-k separately; outputs match to summation order |

Each patch applies to the pinned commit; `packaging/mimo-pro/setup-venv.sh` applies them all. These changes belong in
FlashInfer. Delete each one once a FlashInfer pin contains it.
