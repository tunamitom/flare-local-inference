# Kimi decode lever: the candidate-score exchange (sparse selection)

Ported from TP9 `8b1be5b` ("kimi3: decode DSA exchanges candidate scores and each rank
selects its share") plus the `ada7d02` `collect()`/`merge_result()` split and query-gather
overlap that rides on it. Ruling: `levers/j-missed-fable-review.md` §2 "If it is exported".
The lever is wired as a plan setting (`score_exchange`, default armed, dropped with
`--no-score-exchange`), never through the process environment. Its pricing cell
(levers/evidence/score-exchange/) measured 8.22 µs per layer saved at 4 rows and 26.99 µs at
14 rows on this box.

## What changes

In every sparse layer the indexer's decode and verification rows propose up to 2,048
candidates each. The pair form (`PreparedDistributedTopK`, kept) gathers every rank's
candidates as `(score, global position)` pairs — 16,384 bytes per row and rank — and merges
the global top-2,048 on every rank with seven launches per stage. A rank attends over its
own records only, so it reads only its own candidates among the global selection, and the
scores of all candidates determine those.

The score exchange (`PreparedScoreExchangeTopK` + `PreparedCandidateShare`) gathers one FP32
score per candidate slot — 8,192 bytes per row and rank, on the channel prefill uses for
selected positions (2 × 2,048 BF16 columns per rank; `-inf` marks a slot without a
candidate). `PreparedCandidateShare` then takes the 2,048 greatest gathered scores with the
b12x row top-k (`DSATiledTopk`) and selects this rank's candidates in one program per row
(`_select_share`): scores above the cut-off, then equal scores while places remain after the
greater scores, the equal scores of earlier ranks and the equal scores in earlier slots of
this rank. A stage runs four launches (pack, gather, row top-k, select). The pair channel is
no longer allocated by the Kimi bootstrap when the setting is armed; the pair form remains
for a rank that attends over gathered context and for a plan that dropped the setting.

## Equivalence, ties, and the recorded divergence

For finite scores a rank's output holds exactly the positions the pair merge selects for
that rank. The tie rule is the pair merge's: equal cut-off scores are taken in gathered
order — rank-major across the exchange, slot order inside a rank. The interpreted
equivalence suite (`tests/components/test_candidate_share_interpreted.py`, seven kinds of
candidate sets × two seeds at nine ranks) proves this in Triton's CPU interpreter, with
three mutations of the rule (own ties counted as earlier, exactly-K rows through the
cut-off path, `-inf` kept) each failing the comparison.

**Recorded divergence (not papered over):** a candidate whose score is `-inf` — or NaN,
which is outside both contracts (`PreparedCandidateShare`: "Scores must not be NaN";
`PreparedCandidateMerge`: "Scores for valid IDs must be finite") — is no candidate of the
score exchange, where the pair merge kept `-inf` candidates while fewer than 2,048 finite
ones existed. Whether the indexer can return a valid position with such a score is not
settled on CPU; the GPU wave's temperature-0 comparison on prompts of at most 2,048 tokens
(both forms must agree exactly) is the gate.

## Plan setting and channels

- `planning.Plan.score_exchange` (default `True`), `PreparationConfig.score_exchange`,
  `--no-score-exchange` (Kimi K3/K3.1 only). Threaded Plan → recipe →
  `PreparedDecoder` → `PreparedMLA` → `PreparedDSA`, in the `mix_split_rows` style.
- Armed: the bootstrap allocates no `candidates` channel and sizes `selections` to
  `max(selection_rows, prefill rows)`; decode and verification merges run on
  `context.selections` as `<stage>.candidate_scores`.
- Dropped: the pre-port wiring exactly — `candidates` at 4 × TOPK columns per rank,
  merges named `<stage>.candidates`.
- The J-fault in-export overlap check (`SelectionOverlapCheck`, `decoder.py`) is unchanged
  and still runs at construction; the new keys/gathered/scratch buffers appear in its
  `newly_bound` report for the sparse stage.

## Gates to run later (GPU wave)

1. The J-fault battery with the wide-round content cell at 1, 2 and 4 streams.
2. A temperature-0 comparison on prompts ≤ 2,048 tokens where every candidate is
   selected and the two forms must agree exactly.
3. Register the new round-graph/kernel fingerprint in the export driver's `fp_baseline()`
   before re-export (the armed default changes the graph counts).

## L6

Retired: the radix cutoff only shortened the candidate merge this lever replaces. Its code
stays on `lever/L3L6-split-radix` and must not be brought into the tree
(`docs/kimi3-l3-l6-ports.md`).
