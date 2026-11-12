# Kimi decode lever L3: stream-schedule audit

This records the code-reading stream-schedule audit for the residual mix split (L3,
upstream flare #11). The lever is wired as a plan setting (`--mix-split-rows`), never
through the process environment.

Note: the candidate-merge radix cutoff (L6) is RETIRED (j-missed-fable-review.md §2:
"the radix cutoff only shortened the merge this replaces"). The candidate-score exchange
(docs/kimi3-score-exchange.md) replaces the merge whose latency the radix shortened, so L6
must not be brought into this tree in any form. Its kernels, plan setting and design notes
stay on the `lever/L3L6-split-radix` branch.

## L3 — the split kernel's cross-program spin-wait versus our stream schedule

**The hazard.** `_mix_split` launches `rows x ceil(hidden / chunk)` programs (at the K3.1
shapes: 16 rows x 7 chunks = 112 CTAs at chunk 1024). Each program publishes its per-chunk
FP32 partials as tagged 64-bit words, then spin-waits (`_gather`, volatile load loop) until
every chunk slot of its row carries the call's tag. Forward progress requires every program
of a row to be resident concurrently. The kernel's author flags a possible deadlock if a
cooperative or persistent kernel holds SMs on another stream at the same time: the resident
programs spin while the missing ones cannot launch.

**What this tree's decode capture runs on other streams.** The kimi3/kimi31 decode and
verification graphs fork to `ScratchArena.auxiliary_stream` in exactly two places:

1. `PreparedMLP.start` / `PreparedMLP.finish` (`ignition/recipes/kimi3/decoder.py`): each
   MoE layer's shared-expert MLP — gate_up projection, SiTU and down projection — runs on
   the auxiliary stream (`flare::kimi3_start_shared_expert`) between the fork and the join
   (`flare::kimi3_join_shared_expert`, `main.wait_stream(auxiliary)`), for decode row counts
   1..20 (`shared.concurrent` requires the route exchange).
2. `ParallelProjections` (`ignition/components/gated_delta.py`, KDA layers with the fused
   QKV/Z projection at 16 rows or fewer): `_parallel_mx_projections` forks `in_proj_ba` to
   the auxiliary stream beside `in_proj_qkvz` and joins before the op returns.

Both are ordinary projection/pointwise launches: no cooperative launch, no persistent CTAs,
no cross-program spin-waits of their own. Everything else in a round — residual mixes,
attention, the DSA slice loop (the query-head gather `gather_heads`, the LSE combine and the
candidate merge are all inline on the calling stream in this tree), the route exchange
(`PreparedRouteGather`; `stream_affine=False` on its b12x channel only disables the
stream-affinity guard — the kernels replay on the caller's stream), the reductions,
sampling — is launched on the main stream.

**Do residual mixes overlap the auxiliary windows?** No. `PreparedDecoder.__call__` orders
each layer as: mix, attention plus reduction, mix, MoE (`start` ... `finish`), reduction.
The fork's `auxiliary.wait_stream(main)` orders the shared expert after the preceding mix
has completed; the join's `main.wait_stream(auxiliary)` orders every later main-stream
launch — including the next layer's first mix — after the auxiliary kernels complete. The
KDA `ParallelProjections` window is one custom op that joins before returning. So in the
captured schedule of this tree a residual mix and an auxiliary-stream kernel are never
concurrent, and the split grid's spin-waits only ever wait on their own grid's CTAs.

**Residual risk.** CUDA does not guarantee CTA co-residency, so the split still needs its
own grid (at most 112 CTAs of 4 warps at chunk 1024) to co-reside on the serving GPU. That
fits a GB202-class device at these shapes with one process per device, but it is not a
language guarantee, and MPS co-tenancy would weaken it.

**Verdict.** The flagged deadlock is ruled out for the schedule this tree captures: no mix
overlaps any auxiliary-stream work, and nothing on another stream is cooperative, persistent
or itself spin-waiting. It is not automatically ruled out for schedules whose fork windows
grow wider — in particular L1's query-head-gather overlap (the M-22 / `7eeed675` family),
which runs the query-head gather (a b12x IPC collective that spin-waits on peer data while
holding its CTAs) on an auxiliary stream and is being ported alongside this lever.

**Requirement for the later GPU wave (documented here per the port contract).** Before
`--mix-split-rows` is armed in any export:

- the stress run must capture the decode and verification graphs with the split ON and the
  shipped stream overlap armed, replay at least 10,000 times at 1..16 rows across all 24
  stages on every rank, with a hang watchdog; any hang or timeout is a kill;
- if L1's query-head-gather overlap (or any other auxiliary-stream fork) is present in the
  same export, the stress run must cover the combined schedule, or the split must be armed
  only together with `--no-stream-overlap`;
- a combined-schedule re-audit of fork/join placement is required if L1 changes where any
  residual mix sits relative to an auxiliary fork window.

The setting is default-off, so today's schedules are unchanged until that run passes.

## L6 — retired, held out of this tree

See the note above: the export tree carries no radix-cutoff code in any form, and L6 is
retired (the candidate-score exchange replaces the merge the radix shortened). The
L6/L1 combination analysis is preserved on `lever/L3L6-split-radix`
(`docs/kimi3-l3-l6-ports.md` there).
