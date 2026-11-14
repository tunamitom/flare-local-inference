# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0
"""The Hy4 MTP golden gate (consult3 W5 — the previously skipped MTP gate).

`golden.json`'s mtp section is the reference MTP block's certified output under
the FIXED hidden convention: POST-`model.norm` hidden (`norm(hc_head(streams))`)
fed to the block, the prompt absorbed into the draft KV, chained draft steps
with advancing positions and returned hidden, and the position-0 embed zeroing
(finding F; upstream file/line evidence in `ignition/recipes/hy4/mtp.py` and
`reference/hy4_mtp.py`). This gate runs the RECIPE's MTP on the tiny
checkpoint — `draft_absorb` over the prompt plus three chained `draft_step`s —
and requires it to match the goldens: MTP logits, hidden fingerprints, causal
top-k digests AND the draft-acceptance behavior (teacher-forced top-1 agreement
with the token two ahead, the greedy chain's drafted tokens).

Then each convention mutant must BREAK the comparison:
  * feeding the pre-norm hidden when post-norm is correct (finding F),
  * hiding the prompt from the MTP — no prompt absorption, the draft KV stays
    zeros at every prompt position (finding F's structure bug),
  * swapping the draft order (eh_proj input halves; and the shifted pairing).

The chain steps compare EXACTLY (no tolerance band) — the W5 cache-diff
decision. The chain[1..2] divergence that first looked like fp-level tie
behavior was neither a recipe cache-write bug nor ties: the recipe's cached
records are exact (cache writes bit-identical to `project`'s latent, the record
formula verified against an independent recomputation to 2.4e-7, the sparse
MLA math reproduced from the cached records to 0.0). It was the GOLDEN's chain
bookkeeping: `run_mtp` re-batched earlier chain rows with their `prev_hidden`
overwritten by their own returned hidden, so the reference recomputed those
context rows' records from changed inputs (a ~1e-2 shift at chain[1], which
then chains forward). Fixed in golden.py `run_mtp` (write-once draft KV —
transformers MtpModel.forward_layer appends to `mtp_cache`, modeling_layers
.py:568-606; engines append, never re-run) and golden.json's mtp sections
regenerated; `test_chain_records_are_write_once` pins the semantics. The one
fp-level tie that remains is at the rank-`index_topk` boundary of the 8k-token
absorb digest (one row, 123 <-> 579, |Δscore| 1.1e-6): `sha_causal_rows` is
banded there and the exact behavioral equality carries the gate.

CPU only; the reference side of the same numbers is verified for real by
`golden.py mtp-verify` (which also kills these mutants on the reference).
"""

import hashlib
import json
import os
import struct
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REF = Path(os.environ.get("HY4_REFERENCE_DIR", "/home/tom/hy4/reference"))
TINY = REF / "tiny-hy4"
GOLDEN_PATH = REF / "goldens" / "golden.json"

N_CHAIN = 3   # the consult3 W5 gate: three chained draft steps


# --- fingerprints (same protocol as reference/golden.py and test_hy4_reference_gates.py)
def fp_tensor(t):
    t = t.detach().to(torch.float32)
    return {
        "sum": round(float(t.sum()), 4),
        "abssum": round(float(t.abs().sum()), 4),
        "l2": round(float(t.norm()), 4),
        "min": round(float(t.min()), 6),
        "max": round(float(t.max()), 6),
    }


def fp_topk(idx):
    """idx: [B, S, topk] selections — CAUSAL ENTRIES ONLY (defect 16)."""
    rows = idx[0].to(torch.int64)
    n_rows, topk = int(rows.shape[0]), int(rows.shape[1])
    h = hashlib.sha256()
    counts, causal_flat, first_rows = [], [], []
    for t in range(n_rows):
        c = rows[t]
        c = c[c >= 0]
        c = c[c <= t]
        c, _ = c.sort()
        counts.append(int(c.numel()))
        causal_flat.append(c)
        h.update(struct.pack("<i", int(c.numel())))
        h.update(c.numpy().astype("<i8").tobytes())
        if t < 2:
            first_rows.append(c.tolist())
    all_c = torch.cat(causal_flat) if causal_flat else torch.zeros(0, dtype=torch.int64)
    return {
        "sha_causal_rows": h.hexdigest()[:16],
        "n_rows": n_rows,
        "topk": topk,
        "causal_count_min": min(counts),
        "causal_count_max": max(counts),
        "distinct_causal": int(all_c.unique().numel()),
        "mean_causal_index": round(float(all_c.float().mean()), 3) if all_c.numel() else 0.0,
        "first_rows": first_rows,
    }


def fp_logits(rows):
    return [round(float(v), 6) for v in rows]


def close(a, b, rtol, atol):
    if isinstance(a, (int, str)) or isinstance(b, (int, str)):
        return a == b
    return abs(a - b) <= atol + rtol * abs(b)


def compare(ref, got, path="", rtol=1e-4, atol=1e-5):
    """golden.py's `compare`: [] == identical within tolerance."""
    bad = []
    if isinstance(ref, dict):
        if set(ref) != set(got):
            bad.append(f"{path}: keys {set(ref) ^ set(got)}")
            return bad
        for k in ref:
            bad += compare(ref[k], got[k], f"{path}.{k}", rtol, atol)
    elif isinstance(ref, list):
        if len(ref) != len(got):
            bad.append(f"{path}: length {len(ref)} vs {len(got)}")
            return bad
        for i, (a, b) in enumerate(zip(ref, got)):
            bad += compare(a, b, f"{path}[{i}]", rtol, atol)
    elif not close(ref, got, rtol, atol):
        bad.append(f"{path}: {ref} vs {got}")
    return bad


# --- fixtures ---------------------------------------------------------------
@pytest.fixture(scope="module")
def golden():
    assert GOLDEN_PATH.is_file(), f"goldens missing: {GOLDEN_PATH}"
    loaded = json.loads(GOLDEN_PATH.read_text())
    for name in ("short", "long"):
        run = loaded["runs"][name]
        assert "mtp" in run and "chain" in run["mtp"], \
            "golden.json has no chained mtp section — run reference/golden.py mtp-regen"
        assert len(run["mtp"]["chain"]) == N_CHAIN
    return loaded


@pytest.fixture(scope="module")
def tiny_rank():
    if not (TINY / "config.json").is_file() or not (TINY / "model.safetensors").is_file():
        pytest.skip(f"tiny-hy4 reference checkpoint not present under {TINY}")
    from ignition.recipes.hy4.bootstrap import PreparationConfig, construct_rank

    settings = PreparationConfig(devices=(0,), mtp=True, speculative_widths=(1, 2, 3))
    rank = construct_rank(TINY, settings, rank=0, pinned=False, mtp=True,
                          cache_rows=2, cache_capacity=64)
    # FP32 against FP32 (the W4 gate-a protocol): a full-precision weight copy.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        from test_hy4_reference_gates import to_fp32_weights
    finally:
        sys.path.remove(str(Path(__file__).resolve().parent))
    rank.weights = to_fp32_weights(rank.weights)
    return rank


# --- the recipe-side MTP run -------------------------------------------------
PREFILL_CHUNK = 1024  # the documented prefill contract (test_hy4_reference_gates)


def chunked_forward(decoder, ids, chunk=PREFILL_CHUNK):
    """Chunked prefill (`prefill_chunk_size` tokens per row): causal math, so
    the concatenated result equals the one-shot forward for short prompts and
    runs the 8k-token golden prompt at all (one call would exceed the shared
    selection cell's `max_q_rows` rows)."""
    logits_parts, post_parts, pre_parts = [], [], []
    for start in range(0, ids.shape[1], chunk):
        batch = ids[:, start:start + chunk]
        lg, post, pre = decoder.forward(batch, return_hidden=True)
        decoder.complete([0], batch)   # advances the rows' positions for the next chunk
        logits_parts.append(lg)
        post_parts.append(post)
        pre_parts.append(pre)
    return (torch.cat(logits_parts, dim=1), torch.cat(post_parts, dim=1),
            torch.cat(pre_parts, dim=1))


def run_mtp_gate(rank, tokens, *, feed="post_norm", absorb=True,
                 swap_order=False, shift_inputs=False):
    """Run the recipe MTP exactly as the reference golden was produced.

    Prompt absorption over rows t = 0..S-2 (row t pairs embed(ids[t+1]) with
    hidden_t at position t) + `N_CHAIN` chained `draft_step`s from the bonus
    row (the target's greedy token). Returns the golden-format mtp dict.
    """
    from ignition.recipes.hy4.attention import rms_norm
    from ignition.recipes.hy4.decoder import Hy4Cache, Hy4Decoder

    config = rank.model_config
    cache = Hy4Cache.allocate(config, rows=1, capacity=len(tokens) + 8,
                              device=torch.device("cpu"), draft=True,
                              dtype=torch.float32)
    decoder = Hy4Decoder(rank.weights, cache, compute_dtype=torch.float32)
    ids = torch.tensor([tokens], dtype=torch.long)
    logits, hidden_post, hidden_pre = chunked_forward(decoder, ids)
    full_hidden = hidden_post if feed == "post_norm" else hidden_pre
    S = ids.shape[1]
    n_abs = S - 1
    absorb_ids = ids[:, 1:] if not shift_inputs else ids[:, :-1]

    @contextmanager
    def maybe_swap():
        if not swap_order:
            yield
            return
        draft = rank.weights.draft

        def swapped(next_embeds, prev_hidden):
            left = rms_norm(next_embeds, draft.enorm, config.text["rms_norm_eps"])
            right = rms_norm(prev_hidden, draft.hnorm, config.text["rms_norm_eps"])
            return torch.cat((right, left), dim=-1) @ draft.eh_proj.T

        inner = draft.embed_merge
        draft.embed_merge = swapped
        try:
            yield
        finally:
            draft.embed_merge = inner

    with maybe_swap():
        if absorb:
            absorb_logits, absorb_hidden, selected = decoder.draft_absorb(
                absorb_ids, full_hidden[:, :n_abs])
        else:
            # finding F's structure bug: the MTP never runs over the prompt, so
            # its draft KV/index rows are zeros at every prompt position and the
            # chain starts at the right position but sees no prompt.
            decoder._draft_positions[0] = n_abs
            absorb_logits = absorb_hidden = selected = None
        chain = []
        d_in = int(logits[0, -1].argmax())
        prev_h = full_hidden[:, n_abs:n_abs + 1]
        for k in range(N_CHAIN):
            step_logits, step_hidden = decoder.draft_step(
                torch.tensor([[d_in]], dtype=torch.long), prev_h)
            chain.append({
                "position": int(n_abs + k),
                "input_token": int(d_in),
                "hidden": fp_tensor(step_hidden),
                "logits": fp_logits(step_logits[0]),
                "draft_token": int(step_logits[0].argmax()),
            })
            d_in = chain[-1]["draft_token"]
            prev_h = step_hidden.unsqueeze(1)

    scored = [(t, int(absorb_logits[0, t].argmax()), int(ids[0, t + 2]))
              for t in range(1, S - 2)] if absorb_logits is not None else []
    hits = sum(1 for _, a, b in scored if a == b)
    n_scored = len(scored)
    pick = sorted({0, n_abs // 2, n_abs - 1})
    return {
        "hidden": fp_tensor(absorb_hidden) if absorb_hidden is not None else {"absorbed": False},
        "topk": fp_topk(selected) if selected is not None else {"absorbed": False},
        "logits_rows": ({str(i): fp_logits(absorb_logits[0, i]) for i in pick}
                        if absorb_logits is not None else {"absorbed": False}),
        "bonus_token": int(logits[0, -1].argmax()),
        "acceptance": {"n_scored": n_scored, "hits": hits,
                       "top1_agreement": round(hits / n_scored, 6) if n_scored else 0.0},
        "chain": chain,
    }


# --- gate: the recipe matches the reference MTP goldens -----------------------
def test_gate_recipe_mtp_matches_the_reference_goldens(golden, tiny_rank):
    """MTP logits + draft-acceptance behavior over prompt absorption + 3 chained steps.

    The chain[1..2] fingerprints compare EXACTLY (the W5 cache-diff decision —
    see the module docstring). One field is banded: the causal top-k digest's
    per-row hash `sha_causal_rows`. At the rank-`index_topk` boundary an
    fp-level score tie can flip one entry in and one out; measured on the long
    run exactly one row (t=803) swaps 123 <-> 579 with |Δscore| = 1.1e-6
    (3.6e-6 relative — fp32 noise on inputs that match the reference at 1e-6),
    and the short run's digest matches exactly. The band is tight: at most two
    boundary swaps (|Δ distinct_causal| <= 2, |Δ mean_causal_index| <= 0.1) and
    `sha_causal_rows` may move only on the 8k-token run; the exact behavioral
    equality (acceptance counts, drafted tokens, chain and hidden and logits
    fingerprints) carries the gate.
    """
    for name in ("short", "long"):
        tokens = list(golden["runs"][name]["tokens"])
        got = run_mtp_gate(tiny_rank, tokens)
        want = golden["runs"][name]["mtp"]
        got_topk = dict(got["topk"])
        want_topk = dict(want["topk"])
        got_sha = got_topk.pop("sha_causal_rows")
        want_sha = want_topk.pop("sha_causal_rows")
        got_distinct = got_topk.pop("distinct_causal")
        want_distinct = want_topk.pop("distinct_causal")
        got_mean = got_topk.pop("mean_causal_index")
        want_mean = want_topk.pop("mean_causal_index")
        bad = compare({**want, "topk": want_topk}, {**got, "topk": got_topk},
                      f"{name}.mtp", rtol=golden["meta"]["rtol"], atol=golden["meta"]["atol"])
        assert not bad, f"{name}: {len(bad)} mtp mismatches vs golden.json, e.g. {bad[:4]}"
        # the boundary-tie band (the decision's fp-level tie behavior)
        assert abs(got_distinct - want_distinct) <= 2, \
            f"{name}: distinct_causal {got_distinct} vs {want_distinct} — more than two tie swaps"
        assert abs(got_mean - want_mean) <= 0.1, \
            f"{name}: mean_causal_index {got_mean} vs {want_mean} — more than two tie swaps"
        if got_sha != want_sha:
            assert name == "long", \
                f"{name}: no top-k boundary tie at {len(tokens)} rows — sha_causal_rows must match"


def test_gate_draft_acceptance_behavior_matches(golden, tiny_rank):
    """The behavioral fingerprint: acceptance counts and the greedy chain's tokens."""
    for name in ("short", "long"):
        tokens = list(golden["runs"][name]["tokens"])
        got = run_mtp_gate(tiny_rank, tokens)
        want = golden["runs"][name]["mtp"]
        assert got["acceptance"] == want["acceptance"], \
            f"{name}: draft acceptance {got['acceptance']} != golden {want['acceptance']}"
        assert [c["draft_token"] for c in got["chain"]] == \
               [c["draft_token"] for c in want["chain"]], f"{name}: draft chain diverged"
        assert [c["input_token"] for c in got["chain"]] == \
               [c["input_token"] for c in want["chain"]], f"{name}: draft pairing diverged"
        assert [c["position"] for c in got["chain"]] == list(
            range(len(tokens) - 1, len(tokens) - 1 + N_CHAIN)), f"{name}: positions must advance"


# --- the W5 cache-diff decision: the draft KV is write-once --------------------
def test_chain_records_are_write_once(golden, tiny_rank):
    """A chain row's record is written at its step and never recomputed.

    This is the diff decision enforced: the recipe's cached records are the
    rows the reference's in-batch recomputation produces ONLY when each row
    keeps its call-time inputs (golden.py `run_mtp` — fixed; the pre-fix
    bookkeeping re-ran chain rows with `prev_hidden` := their own returned
    hidden and moved chain[1..2] at ~1e-2). Later draft steps must leave
    earlier rows' records untouched.
    """
    from ignition.recipes.hy4.decoder import Hy4Cache, Hy4Decoder

    tokens = list(golden["runs"]["short"]["tokens"])
    config = tiny_rank.model_config
    layer = config.layers
    cache = Hy4Cache.allocate(config, rows=1, capacity=len(tokens) + 8,
                              device=torch.device("cpu"), draft=True,
                              dtype=torch.float32)
    decoder = Hy4Decoder(tiny_rank.weights, cache, compute_dtype=torch.float32)
    ids = torch.tensor([tokens], dtype=torch.long)
    logits, hidden_post, _ = decoder.forward(ids, return_hidden=True)
    decoder.complete([0], ids)
    n_abs = len(tokens) - 1
    decoder.draft_absorb(ids[:, 1:], hidden_post[:, :n_abs])
    absorbed = [cache.latents(layer)[t].clone() for t in range(n_abs)]
    written = {}
    d_in = int(logits[0, -1].argmax())
    prev_h = hidden_post[:, n_abs:n_abs + 1]
    for k in range(N_CHAIN):
        step_logits, step_hidden = decoder.draft_step(
            torch.tensor([[d_in]], dtype=torch.long), prev_h)
        written[n_abs + k] = cache.latents(layer)[n_abs + k].clone()
        d_in = int(step_logits[0].argmax())
        prev_h = step_hidden.unsqueeze(1)
    for pos, row in written.items():
        assert torch.equal(cache.latents(layer)[pos], row), \
            f"position {pos} rewritten after its step — the draft KV is write-once"
    for t, row in enumerate(absorbed):
        assert torch.equal(cache.latents(layer)[t], row), \
            f"absorbed record {t} rewritten by a chain step"


# --- mutants: each must break the gate ---------------------------------------
def _assert_mutant_breaks(golden, tiny_rank, label, **mutant_kwargs):
    tokens = list(golden["runs"]["short"]["tokens"])
    want = golden["runs"]["short"]["mtp"]
    got = run_mtp_gate(tiny_rank, tokens, **mutant_kwargs)
    bad = compare(want, got, "short.mtp",
                  rtol=golden["meta"]["rtol"], atol=golden["meta"]["atol"])
    assert bad, f"mutant SURVIVED ({label}) — the golden cannot see it"


def test_mutant_prenorm_hidden_feed_breaks_the_gate(golden, tiny_rank):
    """Finding F: pre-norm hidden into hnorm must not pass the post-norm goldens."""
    _assert_mutant_breaks(golden, tiny_rank, "feed pre-norm hidden", feed="pre_norm")


def test_mutant_prompt_hidden_from_mtp_breaks_the_gate(golden, tiny_rank):
    """Finding F: no prompt absorption — the draft KV is zeros at prompt positions."""
    _assert_mutant_breaks(golden, tiny_rank, "prompt hidden from MTP", absorb=False)


def test_mutant_swapped_draft_order_breaks_the_gate(golden, tiny_rank):
    """Swapping the eh_proj input halves (hnorm first) must not pass."""
    _assert_mutant_breaks(golden, tiny_rank, "swapped draft order", swap_order=True)


def test_mutant_shifted_draft_pairing_breaks_the_gate(golden, tiny_rank):
    """Pairing embed(ids[t]) with hidden_t (wrong draft order) must not pass."""
    _assert_mutant_breaks(golden, tiny_rank, "shifted draft pairing", shift_inputs=True)


# --- convention unit checks --------------------------------------------------
def test_position_0_embedding_is_zeroed(golden, tiny_rank):
    """The position-0 zeroing is live and row 0 is excluded from acceptance."""
    from ignition.recipes.hy4.decoder import Hy4Cache, Hy4Decoder

    tokens = list(golden["runs"]["short"]["tokens"])
    config = tiny_rank.model_config
    cache = Hy4Cache.allocate(config, rows=1, capacity=len(tokens) + 8,
                              device=torch.device("cpu"), draft=True,
                              dtype=torch.float32)
    decoder = Hy4Decoder(tiny_rank.weights, cache, compute_dtype=torch.float32)
    ids = torch.tensor([tokens], dtype=torch.long)
    _, hidden_post, _ = decoder.forward(ids, return_hidden=True)
    _, _, selected = decoder.draft_absorb(ids[:, 1:], hidden_post[:, :-1])
    # Row 0 sits at position 0 (vllm_hy4_mtp.py:496 zeroes its embed): with a
    # live embed the merged input would differ from the zeroed one.
    draft = tiny_rank.weights.draft
    embed = tiny_rank.weights.embed[ids[0, 1]].unsqueeze(0)
    merged = draft.embed_merge(embed, hidden_post[:, 0])
    merged_zero = draft.embed_merge(torch.zeros_like(embed), hidden_post[:, 0])
    assert not torch.allclose(merged, merged_zero, atol=1e-6)
    # The golden excludes row 0 from the acceptance count (its draft is unused).
    assert golden["runs"]["short"]["mtp"]["acceptance"]["n_scored"] == len(tokens) - 3
    # Row 0's selection is a real causal row of the absorbed prompt (slot 0).
    assert int(selected[0, 0].max()) <= 0


def test_draft_entries_raise_without_speculation():
    if not (TINY / "config.json").is_file():
        pytest.skip("tiny-hy4 not present")
    from ignition.recipes.hy4.bootstrap import PreparationConfig, construct_rank
    from ignition.recipes.hy4.decoder import Hy4Cache, Hy4Decoder

    settings = PreparationConfig(devices=(0,), mtp=False)
    plain = construct_rank(TINY, settings, rank=0, pinned=False, mtp=False,
                           cache_rows=1, cache_capacity=16)
    decoder = Hy4Decoder(plain.weights, Hy4Cache.allocate(
        plain.model_config, rows=1, capacity=16, device=torch.device("cpu"), draft=False))
    with pytest.raises(ValueError, match="no MTP draft"):
        decoder.draft_absorb(torch.tensor([[1, 2]]),
                             torch.zeros(1, 2, plain.model_config.hidden))
    with pytest.raises(ValueError, match="no MTP draft"):
        decoder.draft_step(torch.tensor([[1]]), torch.zeros(1, plain.model_config.hidden))
