# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0
"""W4: the recipe's forward against the reference goldens (consult3 section 4).

Gate a: tiny `prefill` on the two golden prompts (24 and 8,192 tokens) matches
`goldens/golden.json` logits, causal top-k digests and per-layer streams —
"FP32 against FP32 at 1e-4" (golden's `meta.rtol`/`meta.atol`; the recipe run
in full precision against the FP32 reference run) and "BF16 inside the
tolerance file's bands" (the recipe's native BF16 run judged with
`goldens/bf16_tolerance.json`'s `logit_rtol`/`hidden_rtol`). Unlike
`test_hy4_decoder_body.py`, this file LOADS `golden.json` (consult3 finding C:
no test ever compared against it, which is how the `kv_b_proj` layout bug
passed 31 tests).

Gate b: eight mutants, each applied through the forward, each breaks gate a on
the short golden prompt (the long prompt's sensitivity is gate a itself):
`kv_b` layout, scaling factor omitted, bias in weights, post gates after the
sublayer, gate after `o_proj`, decode position +1, key rotated at the wrong
position, shared layer reading the next group's selection.

Gate c: `construct_rank`'s header dry run completes on the REAL checkpoint's
headers (`/storage/Hy4-preview`, read-only — never past the headers) and on
the package's fused headers. The repacked package
(`/opt/models/Hy4-nvfp4-w4a16-mse-repacked`) is built by another stream; while
it is not ready the package half uses THE SOURCE PACKAGE'S PER-EXPERT HEADERS
(`/opt/models/Hy4-nvfp4-w4a16-mse`), repacked to fused form at header level
(the W1 mapping, checked per-expert against `contract.packed_expert_geometry`);
the test records which header set it used.

Gate d: writing at position >= capacity raises; a released row restarts at
position 0.

CPU only. The long golden prompt runs prefill in its documented
`prefill_chunk_size` chunks (per-token math is causal and chunk-invariant);
the short prompt runs in one call.
"""

import contextlib
import copy
import hashlib
import json
import os
import struct
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REF = Path(os.environ.get("HY4_REFERENCE_DIR", "/home/tom/hy4/reference"))
TINY = REF / "tiny-hy4"
GOLDEN_PATH = REF / "goldens" / "golden.json"
TOLERANCE_PATH = REF / "goldens" / "bf16_tolerance.json"
REAL_CHECKPOINT = Path(os.environ.get("HY4_REAL_CHECKPOINT", "/storage/Hy4-preview"))
PACKAGE_DIR = Path(os.environ.get("HY4_PACKAGE_DIR", "/opt/models/Hy4-nvfp4-w4a16-mse"))
REPACKED_DIR = Path(os.environ.get("HY4_REPACKED_DIR", "/opt/models/Hy4-nvfp4-w4a16-mse-repacked"))

PREFILL_CHUNK = 1024  # `max_q_rows` of the selection buffers = the chunk size


# --------------------------------------------------------------------------
# golden.json: the fingerprints (fp_* semantics copied from reference/golden.py
# so the comparison protocol is local and dependency-free)
# --------------------------------------------------------------------------
def fp_tensor(t):
    t = t.detach().to(torch.float32)
    return {
        "sum": round(float(t.sum()), 4),
        "abssum": round(float(t.abs().sum()), 4),
        "l2": round(float(t.norm()), 4),
        "min": round(float(t.min()), 6),
        "max": round(float(t.max()), 6),
    }


def fp_hidden(t):
    """t: [B, S, hc_mult, H] residual streams out of a decoder layer."""
    out = fp_tensor(t)
    out["per_stream_sum"] = [round(float(s), 4) for s in t.sum(dim=(0, 1, 3))]
    out["per_stream_l2"] = [round(float(s), 4) for s in torch.linalg.vector_norm(t, dim=(0, 1, 3))]
    return out


def fp_topk(idx):
    """idx: [B, S, topk] DSA selections — CAUSAL ENTRIES ONLY (defect 16)."""
    rows = idx[0].to(torch.int64)
    n_rows, topk = int(rows.shape[0]), int(rows.shape[1])
    h = hashlib.sha256()
    counts, causal_flat, first_rows = [], [], []
    for t in range(n_rows):
        c = rows[t]
        c = c[c >= 0]          # vLLM-style padding
        c = c[c <= t]          # causal entries only
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


def load_golden():
    """Load the reference goldens — gate a must actually read them (finding C)."""
    assert GOLDEN_PATH.is_file(), f"goldens missing: {GOLDEN_PATH}"
    golden = json.loads(GOLDEN_PATH.read_text())
    assert set(golden["runs"]) == {"short", "long"}
    for name in ("short", "long"):
        run = golden["runs"][name]
        assert run["n_tokens"] == len(run["tokens"]) and run["tokens"]
        assert len(run["layers"]) > 0
        assert run["logits_rows"] and run["logits_last"]
    assert TOLERANCE_PATH.is_file(), f"tolerance file missing: {TOLERANCE_PATH}"
    bands = json.loads(TOLERANCE_PATH.read_text())
    assert golden["meta"]["bf16_tolerance"]["logit_rtol"] == bands["logit_rtol"]
    return golden, bands


def project_run(run):
    """Gate a's three targets: per-layer streams, causal top-k digests, logits."""
    return {
        "layers": [{"hidden": layer["hidden"], "topk": layer["topk"]} for layer in run["layers"]],
        "logits_rows": run["logits_rows"],
        "logits_last": run["logits_last"],
    }


# --------------------------------------------------------------------------
# forward capture: per-layer streams, tails and selections through the forward
# --------------------------------------------------------------------------
class ForwardCapture:
    """Per-call tape of what golden.json records, taken through the forward.

    A decoder layer's output streams are the iHC `post` update of its FFN
    branch — the first value the layer's `ffn_ihc` boundary returns
    (`post_pre` fused into the next layer's entry, `post` at the tail). Each
    tape entry records that tail's inputs (branch, streams, post gates), the
    output streams, the layer-0 entry streams and the layer's selection
    (the owner buffers for full layers, the bound owner buffers for shared
    layers — within-row positions for request row 0, -1-padded).

    With `tape` set, the run snaps every recorded input to the tape's values
    and freezes the selections: identical inputs, frozen discrete choices —
    the `bf16_tolerance.json` measurement protocol (`golden.py:
    local_bf16_error`), so a BF16 run's LOCAL rounding error can be judged
    against the bands the file certifies.
    """

    def __init__(self, decoder, *, tape=None):
        self.decoder = decoder
        self.tape = tape
        self._undo = []
        self.call = -1
        self.tapes = []

    def __enter__(self):
        for index, layer in enumerate(self.decoder.weights.layers):
            self._wrap_tail(index, layer.ffn_ihc, "post_pre")
            self._wrap_tail(index, layer.ffn_ihc, "post")
            self._wrap_select(index, layer.attention)
            if index == 0:
                self._wrap_entry(index, layer.attn_ihc, "pre")
        return self

    def __exit__(self, *exc):
        for obj, name, original, shadowed in self._undo:
            if shadowed:
                setattr(obj, name, original)
            else:
                delattr(obj, name)  # leave the object exactly as found
        return False

    # -- recording / snapping -------------------------------------------------
    def _slot(self, index):
        return self.tapes[self.call][index]

    def _snap(self, value, like):
        return value.to(like.dtype)

    def _record(self, obj, name, original):
        self._undo.append((obj, name, original, name in vars(obj)))

    def _wrap_entry(self, index, boundary, name):
        original = getattr(boundary, name)
        self._record(boundary, name, original)

        def wrapped(*args, _original=original, _index=index, **kwargs):
            args = list(args)
            if self.tape is not None:
                args[0] = self._snap(self.tape[self.call][_index]["entry"], args[0])
            else:
                self._slot(_index)["entry"] = args[0].clone()
            return _original(*args, **kwargs)

        setattr(boundary, name, wrapped)

    def _wrap_tail(self, index, boundary, name):
        original = getattr(boundary, name)
        self._record(boundary, name, original)

        def wrapped(*args, _original=original, _index=index, **kwargs):
            args = list(args)
            head, rest = args[:3], args[3:]  # (branch, streams, post) [+ fused `nxt`]
            if self.tape is not None:
                snapped = self.tape[self.call][_index]["tail"]
                head = [self._snap(value, arg) for value, arg in zip(snapped, head)]
            else:
                self._slot(_index)["tail"] = tuple(a.clone() for a in head)
            out = _original(*head, *rest, **kwargs)
            # `post_pre` returns (streams, post, y); the tail `post` returns the
            # updated streams tensor itself.
            streams = out[0] if isinstance(out, tuple) else out
            self._slot(_index)["f_out"] = streams
            return out

        setattr(boundary, name, wrapped)

    def _wrap_select(self, index, attention):
        original = attention.select
        self._record(attention, "select", original)

        def wrapped(*args, _original=original, _index=index, **kwargs):
            if self.tape is not None:
                frozen = self.tape[self.call][_index]
                self._slot(_index)["selected"] = frozen["selected"]
                self._slot(_index)["counts"] = frozen["counts"]
                return frozen["selected"], frozen["counts"]
            selected, counts = _original(*args, **kwargs)
            self._slot(_index)["selected"] = selected.clone()
            self._slot(_index)["counts"] = counts.clone()
            return selected, counts

        setattr(attention, "select", wrapped)

    def start_call(self):
        self.call += 1
        self.tapes.append({index: {} for index in range(len(self.decoder.weights.layers))})


# --------------------------------------------------------------------------
# full-precision weights (the "FP32 vs FP32" arm of gate a)
# --------------------------------------------------------------------------
def to_fp32_weights(weights):
    """A full-precision copy of the constructed state (tiny is small).

    Floating-point tensors upcast wherever they live — including attributes
    `__post_init__` computes (`absorb_wuk`/`expand_wuv`); integer buffers
    (selections, counters) and non-weight plumbing (config, sharding, the
    SelectionCell/IndexCache wiring) stay as they are.
    """
    from ignition.recipes.hy4.attention import SelectionCell
    from ignition.recipes.hy4.config import ModelConfig
    from ignition.recipes.hy4.shard import Hy4Sharding

    def fp32(obj):
        if isinstance(obj, torch.Tensor):
            return obj.to(torch.float32) if obj.is_floating_point() else obj
        if isinstance(obj, (ModelConfig, Hy4Sharding, SelectionCell, torch.device)):
            return obj
        if isinstance(obj, dict):
            return {key: fp32(value) for key, value in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(fp32(value) for value in obj)
        if hasattr(obj, "__dict__") and obj.__class__.__module__.startswith(("ignition.", "__main__")):
            clone = copy.copy(obj)
            for name, value in vars(obj).items():
                object.__setattr__(clone, name, fp32(value))
            return clone
        return obj

    return fp32(weights)


# --------------------------------------------------------------------------
# the gate-a run
# --------------------------------------------------------------------------
def run_prefill(weights, config, tokens, *, compute_dtype, chunk=PREFILL_CHUNK, tape=None):
    """Prefill `tokens` through the recipe forward; return gate-a material.

    Chunked at `prefill_chunk_size` (the plan's prefill contract); every token's
    math is causal, so chunking does not change any captured value. `tape`
    snaps inputs and freezes selections (the local BF16 protocol).
    """
    from ignition.recipes.hy4.decoder import Hy4Cache, Hy4Decoder

    cache = Hy4Cache.allocate(config, rows=1, capacity=max(len(tokens) + 8, 16),
                              device=torch.device("cpu"), draft=True, dtype=compute_dtype)
    decoder = Hy4Decoder(weights, cache, compute_dtype=compute_dtype)
    logits_parts = []
    layers = weights.layers
    with ForwardCapture(decoder, tape=tape) as capture:
        for start in range(0, len(tokens), chunk):
            capture.start_call()
            batch = torch.tensor([tokens[start:start + chunk]], dtype=torch.long)
            logits_parts.append(decoder.prefill(batch))
            decoder.complete([0], batch)
    logits = torch.cat(logits_parts, dim=1)
    streams_by_layer = [[] for _ in layers]
    topk_by_layer = [[] for _ in layers]
    for call in capture.tapes:
        for index in range(len(layers)):
            entry = call[index]
            streams_by_layer[index].append(entry["f_out"])
            topk_by_layer[index].append(entry["selected"])
    run = {"layers": []}
    raw_streams = []
    for index in range(len(layers)):
        streams = torch.cat(streams_by_layer[index], dim=0).unsqueeze(0)  # [1, S, 4, H]
        selected = torch.cat(topk_by_layer[index], dim=0)                 # [S, topk]
        raw_streams.append(streams)
        run["layers"].append({"hidden": fp_hidden(streams), "topk": fp_topk(selected.unsqueeze(0))})
    picked = sorted({0, len(tokens) // 2, len(tokens) - 1})
    run["logits_rows"] = {str(i): fp_logits(logits[0, i]) for i in picked}
    run["logits_last"] = fp_logits(logits[0, -1])
    return {"run": run, "logits": logits, "streams": raw_streams, "tapes": capture.tapes}


def local_bf16_bands(fp32, bf16):
    """The tolerance file's LOCAL metrics: identical inputs, frozen selections.

    `hidden`: per-layer rel-L2 of the layer output streams (BF16 run against
    the FP32 run). `logits`: max abs logit difference over the run divided by
    the max abs FP32 logit (`golden.py: local_bf16_error`'s exact metrics).
    """
    hidden_rel = [float((b.float() - a.float()).norm() / a.float().norm())
                  for a, b in zip(fp32["streams"], bf16["streams"])]
    d = (bf16["logits"].float() - fp32["logits"].float()).abs()
    logit_rel = float(d.max() / fp32["logits"].float().abs().max().clamp_min(1e-12))
    return {"hidden_rel_l2": hidden_rel, "max_hidden_rel_l2": max(hidden_rel),
            "max_logit_rel_err": logit_rel}


def golden_tokens(golden, name):
    return list(golden["runs"][name]["tokens"])


# --------------------------------------------------------------------------
# gate b: mutants applied through the forward
# --------------------------------------------------------------------------
def _attention_units(weights):
    """Every gated-MLA unit of the rank (target layers + the MTP block)."""
    units = [layer.attention for layer in weights.layers]
    if weights.draft is not None:
        units.append(weights.draft.attention)
    return units


def _boundaries(weights):
    for layer in weights.layers:
        yield layer.attn_ihc
        yield layer.ffn_ihc


@contextlib.contextmanager
def mutant_kv_b_layout(weights):
    """Finding B's bug restored: all heads' key rows, then all heads' value rows."""
    saved = [(unit, unit.absorb_wuk, unit.expand_wuv) for unit in _attention_units(weights)]
    for unit, _, _ in saved:
        kv_b = unit.weights.kv_b_proj
        unit.absorb_wuk = kv_b[:unit.heads * unit.qk_nope].view(unit.heads, unit.qk_nope, unit.kv_lora)
        unit.expand_wuv = kv_b[unit.heads * unit.qk_nope:].view(unit.heads, unit.v_head, unit.kv_lora)
    try:
        yield
    finally:
        for unit, wuk, wuv in saved:
            unit.absorb_wuk, unit.expand_wuv = wuk, wuv


@contextlib.contextmanager
def mutant_scaling_factor_omitted(weights):
    """The attention softmax scale (`qk_head_dim ** -0.5`) dropped."""
    saved = [(unit, unit.scaling) for unit in _attention_units(weights)]
    for unit, _ in saved:
        unit.scaling = 1.0
    try:
        yield
    finally:
        for unit, scaling in saved:
            unit.scaling = scaling


@contextlib.contextmanager
def mutant_bias_in_weights():
    """`attention_bias=True`: q_a_proj / kv_a_proj_with_mqa computed with a bias.

    Those are the reference's only `nn.Linear(bias=config.attention_bias)`
    modules and both feed a norm (`q_a_layernorm` / `kv_a_layernorm`), so the
    bias enters exactly where the normed projection outputs are built.
    """
    import ignition.recipes.hy4.attention as attention_module

    original = attention_module.rms_norm

    def biased(x, weight, eps):
        return original(x + 0.02, weight, eps)

    attention_module.rms_norm = biased
    try:
        yield
    finally:
        attention_module.rms_norm = original


@contextlib.contextmanager
def mutant_post_gates_after_sublayer(weights):
    """The iHC post gates evaluated AFTER the sublayer ran.

    The gates are pre-gates (`pre()` on the pre-sublayer streams); the mutant
    updates the streams with gates recomputed on the post-sublayer state.
    """
    from b12x.norm.mhc import ihc

    undo = []

    def wrap(boundary, name):
        original = getattr(boundary, name)

        def wrong(branch, streams, post, *args, _original=original, _boundary=boundary, **kwargs):
            out = _original(branch, streams, post, *args, **kwargs)
            updated = out[0] if isinstance(out, tuple) else out
            after, _ = ihc.ihc_pre_reference(
                updated, _boundary.weights.fn, _boundary.weights.scale, _boundary.weights.base,
                magnitude=_boundary.magnitude, hc_eps=_boundary.hc_eps, norm_eps=_boundary.norm_eps)
            wrong_streams = after.unsqueeze(-1) * branch.unsqueeze(1) + streams
            return (wrong_streams,) + out[1:] if isinstance(out, tuple) else wrong_streams

        undo.append((boundary, name, original))
        setattr(boundary, name, wrong)

    for boundary in _boundaries(weights):
        wrap(boundary, "post_pre")
        wrap(boundary, "post")
    try:
        yield
    finally:
        for boundary, name, original in undo:
            setattr(boundary, name, original)


@contextlib.contextmanager
def mutant_gate_after_o_proj():
    """The elementwise gate applied after `o_proj` instead of before it."""
    from ignition.recipes.hy4.attention import Hy4LayerAttention

    original = Hy4LayerAttention.gate_and_project

    def wrong(self, attn_output, x):
        projected = attn_output.reshape(attn_output.shape[0], -1).contiguous() @ self.weights.o_proj.T
        folded = torch.sigmoid((x @ self.weights.gate_proj.T.to(x.dtype)) @ self.weights.o_proj.T.to(x.dtype))
        return projected * folded

    Hy4LayerAttention.gate_and_project = wrong
    try:
        yield
    finally:
        Hy4LayerAttention.gate_and_project = original


@contextlib.contextmanager
def mutant_decode_position_plus_one():
    """Positions off by one at decode/prefill entry."""
    from ignition.recipes.hy4.decoder import Hy4Decoder

    original = Hy4Decoder._positions

    def wrong(self, shape, rows):
        return original(self, shape, rows) + 1

    Hy4Decoder._positions = wrong
    try:
        yield
    finally:
        Hy4Decoder._positions = original


@contextlib.contextmanager
def mutant_key_rotated_at_wrong_position():
    """Index keys rotated at position + 1 while queries rotate at `positions`."""
    from ignition.recipes.hy4.attention import Hy4LayerAttention

    original = Hy4LayerAttention.prepare_index_keys

    def wrong(self, x, positions):
        return original(self, x, positions + 1)

    Hy4LayerAttention.prepare_index_keys = wrong
    try:
        yield
    finally:
        Hy4LayerAttention.prepare_index_keys = original


@contextlib.contextmanager
def mutant_shared_next_group_selection(weights, selections):
    """Shared layers bind the NEXT group's owner buffers instead of their own."""
    from ignition.recipes.hy4.attention import Hy4LayerAttention

    owners = sorted({cell.owner_id for cell in selections.values()})
    nxt = {owner: owners[(i + 1) % len(owners)] for i, owner in enumerate(owners)}
    original = Hy4LayerAttention.select

    def wrong(self, x, q_resid, positions, visible, key_table, slot_base, _original=original):
        if self.selection.kind == "shared":
            owner = nxt[self.selection.owner_id]
            selected, counts = self.selection.cache.selected(owner), self.selection.cache.counts(owner)
            return selected[:x.shape[0]], counts[:x.shape[0]]
        return _original(self, x, q_resid, positions, visible, key_table, slot_base)

    Hy4LayerAttention.select = wrong
    try:
        yield
    finally:
        Hy4LayerAttention.select = original


def _reset_selections(rank):
    """Empty every IndexCache selection buffer (the fresh-at-request state)."""
    seen = set()
    for cell in rank.selections.values():
        if cell.layer_id in seen:
            continue
        seen.add(cell.layer_id)
        cell.cache.selected(cell.layer_id).fill_(-1)
        cell.cache.counts(cell.layer_id).zero_()


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def golden():
    return load_golden()


@pytest.fixture(scope="module")
def tiny_rank():
    if not (TINY / "config.json").is_file() or not (TINY / "model.safetensors").is_file():
        pytest.skip(f"tiny-hy4 reference checkpoint not present under {TINY}")
    from ignition.recipes.hy4.bootstrap import PreparationConfig, construct_rank

    settings = PreparationConfig(devices=(0,), mtp=True, speculative_widths=(1, 2, 3),
                                 prefill_chunk_size=PREFILL_CHUNK)
    return construct_rank(TINY, settings, rank=0, pinned=False, mtp=True,
                          cache_rows=2, cache_capacity=8)


@pytest.fixture(scope="module")
def arms(tiny_rank):
    """(fp32 weights, bf16 weights) of the constructed tiny rank."""
    return to_fp32_weights(tiny_rank.weights), tiny_rank.weights


# --------------------------------------------------------------------------
# gate a: prefill on the golden prompts against golden.json
# --------------------------------------------------------------------------
def test_gate_a_loads_golden_json(golden):
    """Finding C: the gate must actually read golden.json and the tolerance file."""
    loaded, bands = golden
    assert {name: loaded["runs"][name]["n_tokens"] for name in ("short", "long")} == {"short": 24, "long": 8192}
    assert loaded["meta"]["rtol"] == 1e-4 and loaded["meta"]["atol"] == 1e-5
    assert bands["hidden_rtol"] > 0 and bands["logit_rtol"] > 0


def test_gate_a_fp32_prefill_matches_the_reference_goldens(golden, tiny_rank, arms):
    """FP32 against FP32 at 1e-4: logits, causal top-k digests, per-layer streams."""
    (loaded, _), (fp32, _) = golden, arms
    for name in ("short", "long"):
        result = run_prefill(fp32, tiny_rank.model_config, golden_tokens(loaded, name),
                             compute_dtype=torch.float32)
        bad = compare(project_run(loaded["runs"][name]), project_run(result["run"]), name,
                      rtol=loaded["meta"]["rtol"], atol=loaded["meta"]["atol"])
        assert not bad, f"{name}: {len(bad)} mismatches vs golden.json, e.g. {bad[:4]}"


def test_gate_a_bf16_local_error_inside_the_tolerance_bands(golden, tiny_rank, arms):
    """BF16 inside the tolerance file's bands.

    The bands are LOCAL — identical inputs, frozen selections
    (`bf16_tolerance.json` policy: "full-run BF16 comparisons must never anchor
    a band" because discrete choices flip on rounding noise). The recipe's BF16
    prefill is judged with that exact protocol: each layer on the FP32 run's
    captured inputs with the FP32 selections frozen, and the tail logits from
    the resulting hidden.
    """
    (loaded, bands), (fp32, bf16) = golden, arms
    for name in ("short", "long"):
        tokens = golden_tokens(loaded, name)
        reference = run_prefill(fp32, tiny_rank.model_config, tokens,
                                compute_dtype=torch.float32)
        local = run_prefill(bf16, tiny_rank.model_config, tokens,
                            compute_dtype=torch.bfloat16, tape=reference["tapes"])
        metrics = local_bf16_bands(reference, local)
        assert metrics["max_hidden_rel_l2"] <= bands["hidden_rtol"], (
            f"{name}: local BF16 layer error {metrics['max_hidden_rel_l2']:.4e} exceeds "
            f"hidden_rtol {bands['hidden_rtol']:.4e} (per layer {metrics['hidden_rel_l2']})")
        assert metrics["max_logit_rel_err"] <= bands["logit_rtol"], (
            f"{name}: local BF16 logit error {metrics['max_logit_rel_err']:.4e} exceeds "
            f"logit_rtol {bands['logit_rtol']:.4e}")


# --------------------------------------------------------------------------
# gate b: each mutant breaks gate a
# --------------------------------------------------------------------------
def test_gate_b_mutants_each_break_gate_a(golden, tiny_rank, arms):
    (loaded, _), (fp32, _) = golden, arms
    config = tiny_rank.model_config

    mutants = [
        ("kv_b layout", lambda: mutant_kv_b_layout(fp32)),
        ("scaling factor omitted", lambda: mutant_scaling_factor_omitted(fp32)),
        ("bias in weights", mutant_bias_in_weights),
        ("post gates after the sublayer", lambda: mutant_post_gates_after_sublayer(fp32)),
        ("gate after o_proj", mutant_gate_after_o_proj),
        ("decode position +1", mutant_decode_position_plus_one),
        ("key rotated at the wrong position", mutant_key_rotated_at_wrong_position),
        ("shared layer reading the next group's selection",
         lambda: mutant_shared_next_group_selection(fp32, tiny_rank.selections)),
    ]

    baselines = {name: run_prefill(fp32, config, golden_tokens(loaded, name),
                                  compute_dtype=torch.float32)
                 for name in ("short", "long")}
    for name, baseline in baselines.items():
        expected = project_run(loaded["runs"][name])
        assert not compare(expected, project_run(baseline["run"]), name), \
            f"gate a must hold before mutants are meaningful ({name})"
    for label, make_mutant in mutants:
        for name in ("short", "long"):
            _reset_selections(tiny_rank)  # deterministic stale state for the wrong-group read
            with make_mutant():
                try:
                    run = run_prefill(fp32, config, golden_tokens(loaded, name),
                                      compute_dtype=torch.float32)
                except Exception as exc:  # the wiring broke the forward outright
                    raise AssertionError(f"mutant '{label}' broke the forward ({name}): {exc!r}") from exc
            killed_by_golden = compare(project_run(loaded["runs"][name]), project_run(run["run"]), name)
            moved = compare(project_run(baselines[name]["run"]), project_run(run["run"]), name)
            assert moved, f"mutant '{label}' did not change the forward ({name})"
            assert killed_by_golden, f"mutant '{label}' SURVIVED gate a on {name}"


# --------------------------------------------------------------------------
# gate d: slot bounds and row release
# --------------------------------------------------------------------------
def test_gate_d_writing_at_position_capacity_raises(tiny_rank):
    from ignition.recipes.hy4.decoder import Hy4Cache

    cache = Hy4Cache.allocate(tiny_rank.model_config, rows=2, capacity=4,
                              device=torch.device("cpu"), draft=True)
    tokens = torch.zeros(1, dtype=torch.long)
    latent = torch.zeros(1, cache.latent[0].shape[1])
    keys = torch.zeros(1, cache.index[0].shape[1])
    # In range: position 3 of a capacity-4 row is the row's last slot.
    cache.write_latent(0, tokens, torch.tensor([3]), latent)
    cache.write_index_keys(tiny_rank.model_config.full_indexer_layers[0], tokens, torch.tensor([3]), keys)
    for position in (4, 5):
        with pytest.raises(IndexError, match="out of range"):
            cache.write_latent(0, tokens, torch.tensor([position]), latent)
        with pytest.raises(IndexError, match="out of range"):
            cache.write_index_keys(tiny_rank.model_config.full_indexer_layers[0], tokens,
                                   torch.tensor([position]), keys)
    # Through the forward: a row of 5 tokens cannot fill a capacity-4 cache.
    from ignition.recipes.hy4.decoder import Hy4Cache, Hy4Decoder

    decoder = Hy4Decoder(tiny_rank.weights, Hy4Cache.allocate(
        tiny_rank.model_config, rows=1, capacity=4, device=torch.device("cpu"), draft=True))
    with pytest.raises(IndexError, match="out of range"):
        decoder.prefill(torch.tensor([[3, 7, 11, 5, 13]]))


def test_gate_d_released_row_restarts_at_position_zero(tiny_rank):
    from ignition.recipes.hy4.decoder import Hy4Cache, Hy4Decoder

    cache = Hy4Cache.allocate(tiny_rank.model_config, rows=1, capacity=16,
                              device=torch.device("cpu"), draft=True)
    decoder = Hy4Decoder(tiny_rank.weights, cache)
    tokens = torch.tensor([[3, 7, 11, 5]])
    decoder.prefill(tokens)
    decoder.complete([0], tokens)
    assert decoder._positions((1, 1), torch.tensor([0])).tolist() == [4]
    # Without release a reused row silently continues the old request.
    assert decoder._positions((1, 2), torch.tensor([0])).tolist() == [4, 5]
    decoder.release([0])
    assert decoder._positions((1, 3), torch.tensor([0])).tolist() == [0, 1, 2]
    # The released row's run at position 0 equals a fresh row's run — the old
    # request's records and positions are gone.
    fresh = decoder.prefill(torch.tensor([[9]]))
    reference_decoder = Hy4Decoder(tiny_rank.weights, Hy4Cache.allocate(
        tiny_rank.model_config, rows=1, capacity=16, device=torch.device("cpu"), draft=True))
    reference = reference_decoder.prefill(torch.tensor([[9]]))
    assert torch.allclose(fresh, reference, atol=1e-6), "a released row must behave like a fresh row"


# --------------------------------------------------------------------------
# gate c: construct_rank's header dry run
# --------------------------------------------------------------------------
def _header_manifest(snapshot):
    """The checkpoint's real header inventory (index + shard headers).

    The download's `model.safetensors.index.json` carries no
    `metadata.total_size`, so `inspect_indexed_checkpoint`'s index cross-check
    is not applicable here; the shard headers themselves are read exactly as
    the inspector reads them.
    """
    from types import MappingProxyType

    from ignition.preparation.checkpoint import IndexedCheckpoint, _read_header

    snapshot = Path(snapshot)
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    tensors, shards = {}, []
    for shard in sorted(set(weight_map.values())):
        record, entries = _read_header(snapshot / shard, shard)
        tensors.update(entries)
        shards.append(record)
    return IndexedCheckpoint(snapshot, MappingProxyType(tensors), tuple(shards))


def test_gate_c_construct_rank_dry_run_on_real_checkpoint_headers():
    """Construction completes on the REAL checkpoint's headers (read-only)."""
    if not (REAL_CHECKPOINT / "model.safetensors.index.json").is_file():
        pytest.skip(f"real Hy4 checkpoint not present under {REAL_CHECKPOINT}")
    from ignition.recipes.hy4.bootstrap import PreparationConfig, construct_rank

    settings = PreparationConfig(devices=(0,), mtp=True, speculative_widths=(1, 2, 3),
                                 prefill_chunk_size=64)
    rank = construct_rank(REAL_CHECKPOINT, settings, rank=0, pinned=True, mtp=True,
                          cache_rows=1, cache_capacity=8, headers_only=True,
                          manifest=_header_manifest(REAL_CHECKPOINT))
    assert len(rank.weights.layers) == 78 and rank.weights.draft is not None
    # Fix (2): the real shards store lm_head BF16 [120832, 6144]; it loads FP32.
    assert rank.weights.head.dtype == torch.float32
    assert tuple(rank.weights.head.shape) == (120832, 6144)
    # Fix (1): kv_b_proj stays a whole-head [key 192; value 256] row layout.
    kv_b = rank.weights.layers[0].attention.weights.kv_b_proj
    assert tuple(kv_b.shape) == (64 * (192 + 256), 512)
    unit = rank.weights.layers[0].attention
    assert unit.absorb_wuk.shape == (64, 192, 512) and unit.expand_wuv.shape == (64, 256, 512)
    assert rank.weights.embed.dtype == torch.bfloat16  # projections keep storage dtype
    assert len(rank.weights.used_names) == len(set(rank.weights.used_names))


def _source_package_expert_headers():
    """Real per-expert headers of the source package's written blocks.

    Returns `{expert-prefix: {tensor-name: (dtype-str, shape)}}` for every
    block whose shard is on disk (the MSE build writes one block per shard and
    was still running when this gate was written).
    """
    per_prefix = {}
    for shard in sorted(PACKAGE_DIR.glob("model-*.safetensors")):
        with open(shard, "rb") as handle:
            size = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(size))
        for name, meta in header.items():
            if name == "__metadata__" or ".mlp.experts." not in name:
                continue
            prefix = name.rsplit(".", 3)[0]  # <prefix>.experts.<e>.<projection>
            per_prefix.setdefault(prefix, {})[name] = (meta["dtype"], tuple(meta["shape"]))
    return per_prefix


def _repacked_headers_from_per_expert(config):
    """W1's repack mapping at header level, over the source package's headers.

    Per expert the fused form is [gate; up] stacked along the intermediate
    rows, experts stacked along the first axis, and the gate/up scalars shared
    as `weight_scale_2[e]` (contract.py's table). The mapping is DERIVED from
    the real per-expert headers of the written blocks and asserted equal to
    `contract.packed_expert_geometry`; blocks the build has not written yet
    reuse a written block's headers — every MoE block shares one geometry.
    Non-expert tensors pass through the package unchanged.
    """
    from ignition.recipes.hy4 import contract

    per_prefix = _source_package_expert_headers()
    assert per_prefix, f"no per-expert headers written yet under {PACKAGE_DIR}"
    written = {prefix: entries for prefix, entries in per_prefix.items()
               if len(entries) >= 9 * config.experts}
    assert written, "no complete per-expert block in the source package yet"
    fused = {}
    for prefix in contract.quantized_source_names(config):
        entries = written.get(prefix, next(iter(written.values())))
        per_expert = {}
        for name, meta in entries.items():
            expert, _, key = name.partition(".experts.")[2].partition(".")
            per_expert.setdefault(key, {})[expert] = meta
        count = config.experts
        for key, bucket in per_expert.items():
            assert len(bucket) == count, f"{prefix}.{key}: {len(bucket)} of {count} experts on disk"
            assert len(set(bucket.values())) == 1, f"{prefix}.{key}: experts disagree on dtype/shape"

        def one(key):
            return next(iter(per_expert[key].values()))

        gate, up = one("gate_proj.weight"), one("up_proj.weight")
        gscale, uscale = one("gate_proj.weight_scale"), one("up_proj.weight_scale")
        down, dscale = one("down_proj.weight"), one("down_proj.weight_scale")
        derived = {
            f"{prefix}.gate_up_proj.weight":
                (gate[0], (count, gate[1][0] + up[1][0], gate[1][1])),
            f"{prefix}.gate_up_proj.weight_scale":
                (gscale[0], (count, gscale[1][0] + uscale[1][0], gscale[1][1])),
            f"{prefix}.gate_up_proj.weight_scale_2": ("F32", (count,)),
            f"{prefix}.down_proj.weight": (down[0], (count,) + down[1]),
            f"{prefix}.down_proj.weight_scale": (dscale[0], (count,) + dscale[1]),
            f"{prefix}.down_proj.weight_scale_2": ("F32", (count,)),
        }
        geometry = contract.packed_expert_geometry(config, prefix)
        torch_dtypes = {"U8": torch.uint8, "F8_E4M3": torch.float8_e4m3fn, "F32": torch.float32}
        for name, (dtype, shape) in derived.items():
            want = geometry[name]
            assert (torch_dtypes[dtype], shape) == want, \
                f"{name}: repacked headers {(dtype, shape)} != contract {want}"
        fused.update(derived)
    return fused


def test_gate_c_construct_rank_dry_run_on_package_headers(tmp_path):
    """Construction completes on the package's fused-form headers.

    The repacked package (`Hy4-nvfp4-w4a16-mse-repacked`, W1's output) is the
    preferred header set; while it is not ready this uses THE SOURCE PACKAGE'S
    PER-EXPERT HEADERS (`Hy4-nvfp4-w4a16-mse`, the MSE build), repacked to
    fused form at header level by W1's mapping, plus the non-expert passthrough
    headers of the real checkpoint. The print states which set was used.
    """
    from types import MappingProxyType

    from ignition.preparation.checkpoint import IndexedCheckpoint, TensorMetadata
    from ignition.recipes.hy4 import contract
    from ignition.recipes.hy4.bootstrap import PreparationConfig, construct_rank
    from ignition.recipes.hy4.config import ModelConfig

    settings = PreparationConfig(devices=(0,), mtp=True, speculative_widths=(1, 2, 3),
                                 prefill_chunk_size=64)
    if (REPACKED_DIR / "model.safetensors.index.json").is_file():
        used = f"the W1 repacked package {REPACKED_DIR}"
        rank = construct_rank(REPACKED_DIR, settings, rank=0, pinned=True, mtp=True,
                              cache_rows=1, cache_capacity=8, headers_only=True)
    else:
        if not PACKAGE_DIR.is_dir() or not REAL_CHECKPOINT.is_dir():
            pytest.skip("neither the repacked package nor the source package's headers are present")
        used = (f"the source package's per-expert headers ({PACKAGE_DIR}) repacked at "
                f"header level; unwritten blocks reuse a written block's geometry")
        source_config = json.loads((REAL_CHECKPOINT / "config.json").read_text())
        config = ModelConfig.from_dict(source_config, pinned=True)
        fused = _repacked_headers_from_per_expert(config)
        real = _header_manifest(REAL_CHECKPOINT).tensors
        inventory = {}
        for name, meta in real.items():  # non-expert tensors pass through unchanged
            if ".mlp.experts." not in name:
                inventory[name] = meta
        for name, (dtype, shape) in fused.items():
            inventory[name] = TensorMetadata(name=name, shard="model-00001.safetensors",
                                             offset=0, nbytes=0, shape=shape, dtype=dtype)
        from ignition.recipes.hy4.weights import checkpoint_names

        assert set(inventory) == set(checkpoint_names(config, mtp=True, packed=True)), \
            "the repacked header inventory must be exactly the fused-form checkpoint"
        (tmp_path / "config.json").write_text(json.dumps(contract.export_config(source_config)))
        manifest = IndexedCheckpoint(tmp_path, MappingProxyType(inventory), ())
        # The export-provenance gate (`verify_export_package`, W1 gate d) judges
        # real packages by their export_report.json and shard modes; this is a
        # header-level projection of in-flight per-expert headers and carries no
        # report. W1's own tests cover the refusal; it is bypassed only here.
        import ignition.recipes.hy4.weights as weights_module

        sentinel = getattr(weights_module, "verify_export_package", None)
        if sentinel is not None:
            weights_module.verify_export_package = lambda *args, **kwargs: None
        try:
            rank = construct_rank(tmp_path, settings, rank=0, pinned=True, mtp=True,
                                  cache_rows=1, cache_capacity=8, headers_only=True, manifest=manifest)
        finally:
            if sentinel is not None:
                weights_module.verify_export_package = sentinel
    print(f"[gate c] package half used: {used}")
    assert len(rank.weights.layers) == 78 and rank.weights.draft is not None
    # The packed W4A16 route ran (the per-expert bytes would reject at names).
    assert rank.weights.layers[1].mlp.routed.packed is not None
    assert rank.weights.head.dtype == torch.float32  # the package passes lm_head BF16
