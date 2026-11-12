# Kimi K3.1 prefill: the dual-wave compute/ring overlap (L9)

`prefill_dual_wave` (plan setting, default **off**) splits every large prefill
chunk's rows into two waves. While one wave's attention/MLA/MoE compute runs on
the ScratchArena auxiliary stream, the other wave's ring reductions and gathers
run on a priority -1 transport stream, and vice versa — hiding the copy-engine
ring's transfer time behind SM work (levers/j-why23-fable-review.md §5: about
400 ms of every 1,219 ms chunk rung serially today; hiding all of it prices at
+45%, g0san measured 9–11% first, 17% above that).

## What the setting arms

- **The two-wave schedule** (`PreparedDecoder._call_dual_wave`): rows `[0, H)`
  and `[H, 2H)` mix, attend and run their MLP/MoE halves through their own
  prepared module set; half-row reductions (`attn_reduce.w0/.w1`,
  `mlp_reduce.w0/.w1`) ride the ring. KDA layers run two complete half-chunk
  passes bound to half-chunk DeltaMetadata (`2d3635f`'s final state),
  preserving inactive-wave state slots.
- **The second prepared-module set** — per-wave KDA/MLA/MoE modules and the
  split MLA output projections — priced by g0san at about 9% of the KV pool.
  `prefill_dual_wave_pool` (default 0.09) caps the set's added buffers against
  the preparation memory budget; the decoder raises when the set would exceed
  it.
- **The direct BF16 ring wire** (`bf16_direct`): the schedule's transport is
  the zero-copy direct ring (world-1 hops, FP32 intermediate, one BF16
  rounding per chunk — the ordered ring sum the serial path's contract already
  records). Reductions of unaligned shapes pad through caller-owned BF16
  staging; the schedule pads its half-reductions to whole rank shards so the
  common case is zero-copy.
- **Capped expert launches**: `B12X_W4A16_NONCOOP_PREFILL_SMS=170` is stamped
  at preparation, capping the W4A16 prefill MoE launch at 170 of 188 SMs so the
  priority -1 ring, flag and stats launches keep free SMs (g0san's `3058229`
  figure). The knob's reader is `w4a16_prefill_sms` in b12x (an L9 adaptation —
  see the port record).

## Gate (what keeps the serial path)

Armed preparations run the schedule only for prefill decoders with `tokens >=
2048`, `tokens % 512 == 0`, one live request, an overlapping workspace and a
direct-ring channel; drafts' auxiliary features must tap the attention input.
Every other graph — small chunks, decode, verification, drafts — is unchanged.
The setting requires the copy-engine ring (`--prefill-dma-ring-rows`) at TP8.

## Numerics

The schedule splits rows, never reductions' elementwise order:
`ResidualStream.substream` mixes are **bit-identical** to the serial mix's rows
(interpreted test pins this across ADD/REPLACE/commit/compact-mixture cases),
and half-shape ring reductions equal the full-shape ones elementwise (same hop
order). Only GEMM accumulation order can move (half-shape projections may
autotune different split-k), and the KDA core runs over half sequences with the
same per-row recurrence. Prefill numerics therefore change at the projections —
the full recall battery prices that at the gates stage. The expert norm stays
the serial path's `rms_norm_columns` over the all-reduced full latent (the
RS-norm's FP32 norm-stat exchange is not ported).

## Not ported from the source chain

The RS-norm (`PreparedExpertReduceNorm` + norm-stat exchange), the ring column
gathers (our L4 latent gather and router all-gather ride the phases instead),
the split-sparse DSA path (sparse layers use the `pre_output` +
`run_out_half` path), the int16 shard pack (it replaces copies inside g0san's
prefill route ring gather, which this tree does not have), the mxfp16/FP24
extra wire modes (FP24 is landed in b12x but unused), and the shared-expert
two-stage split. See `levers/evidence/l9-dualwave/port-map.md`.
