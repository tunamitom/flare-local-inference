# FlashInfer b12x patches for the MiMo V2 recipe

The MiMo V2 TP8 recipe depends on three changes to the b12x package in the
pinned FlashInfer (`3563f331b43565f7c04419b027ac5e34fb9a6bfa`). They are not
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
| `b12x-mxfp4-a4-prefill.patch` | A4/A16 hybrid prefill experts | b12x ignores `B12X_W4A16_MXFP4_PREFILL_MIN_TOKENS`, so every expert call runs A16 |
| `b12x-w4a16-topk-sum-loads.patch` | Faster W4A16 top-k sums | The top-k sum stays serialized; outputs are identical |

Each patch applies cleanly to the pinned commit. These changes belong in
FlashInfer. Delete each one once a FlashInfer pin contains it.
