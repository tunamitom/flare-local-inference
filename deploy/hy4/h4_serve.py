#!/usr/bin/env python
"""h4_serve: serve the Hy4 NVFP4 W4A16 export on Turin (TP8+CP8, world 8).

Launch chain: h4.service -> /opt/h4/start-h4.sh (docker run) ->
torch.distributed.run --nproc_per_node=8 -> this script. The engine path is the
campaign's working one (the w14 bring-up): ignition's Hy4 recipe construction
(`construct_rank`), the real PCIe-DCP collectives (PcieDcpTransport /
PcieDcpCollectives), the fused 656-byte FP8 record cache (M2), and the
post-capture `Allocate cache` line (the P5 gate). MTP off.

Rank 0 owns the OpenAI-style HTTP surface (/health, /v1/models,
/v1/chat/completions, non-streaming) and the sampler; ranks 1..7 run the same
decode loop off a broadcast command stream. Logits are gathered to full vocab
on every rank (collectives.gather_logits), but only rank 0 samples and
broadcasts the next token, so all ranks advance in lockstep.

Prefill is chunked HERE (not decoder.prefill's chunk_plan) into prepared call
shapes only: widths {2, 3, 4 .. --chunk in powers of two}. Full chunks first,
then the remainder in as few forwards as the cost model allows, the last one
padded to a prepared width (`prompt_plan`, consult16 lever 6) — so no
unprepared call shape can occur at run time (the plan book refuses them;
there is no JIT at run time).

The boot prints two `Allocate cache` lines:
  1. the BINDING line: the real serving KV cache this boot allocated;
  2. the M2-parity residue line: what the pool would size to from the
     post-capture free residue (directly comparable to the M2 measurement,
     540,274 tok/rank at the kernel-chain term).
Both, plus the per-rank facts, land in the boot report.
"""
import argparse
import json
import os
import queue
import sys
import threading
import time
import traceback
from datetime import timedelta

# consult16 lever 6: the prefill chunk cap is a boot flag (--chunk). Every
# pool, plan and scratch scales with it (the DCP pools and the all-reduce pool
# are sized by max rows), so a wider chunk costs fixed memory and therefore
# KV pool: the boot's phase-A line and both caps decide whether it may be used.
DEFAULT_CHUNK = 1024
HEADS_TOTAL = 64
CONTRACT_RECORD_BYTES = 78 * 656 + 21 * 132   # 53,940 B token^-1 rank^-1 (w6)
DEFAULT_MAX_CONTEXT = 1_048_576   # the one-third-cap served context


def trace(rank, msg):
    print(f"[h4 r{rank}] {msg}", flush=True)


def allocate_line(pool, record, graph_bytes, world, *, label="Allocate cache"):
    """The bootstrap.allocate_cache_report line format, verbatim."""
    tokens = pool // record
    aggregate = tokens * world
    cap = aggregate // 3
    return (f"{label}: {pool} B KV pool = {tokens} tokens/rank "
            f"(record {record} B; graph+scratch {graph_bytes} B measured, "
            f"kernel-chain term only; aggregate {aggregate}; one-third cap {cap}"
            f"{'>= 1M \u2713' if cap >= 1_048_576 else '< 1M \u2717'})"), {
        "pool_bytes": pool, "tokens_per_rank": tokens, "aggregate_tokens": aggregate,
        "one_third_cap": cap, "one_third_cap_holds_1m": cap >= 1_048_576,
        "record_bytes_contract": record, "graph_scratch_bytes": graph_bytes,
        "graph_term_complete": False, "label": label,
    }


def prefill_widths(chunk):
    """The prepared prefill widths: 2, 3 and the powers of two from 4 up to `chunk`."""
    widths = [2, 3]
    width = 4
    while width <= chunk:
        widths.append(width)
        width *= 2
    if widths[-1] != chunk:
        raise ValueError(f"the prefill chunk must be a power of two >= 4, got {chunk}")
    return tuple(widths)


def chunk_widths(n, widths):
    """Greedy EXACT decomposition into prepared widths (the pre-lever-6 form).

    Odd lengths prepend width 3 (a sum of even widths is even). Every returned
    width is prepared, so every forward call hits a prepared plan.
    """
    if n == 1:
        return None                       # caller uses the one-row decode path
    out = []
    if n % 2 == 1:
        out.append(3)
        n -= 3
    for p in sorted((w for w in widths if w >= 2 and w != 3), reverse=True):
        while n >= p:
            out.append(p)
            n -= p
    assert n == 0
    return out


_TAIL_TABLES = {}


def _tail_plan(rest, widths, forward_cost_rows):
    """Cheapest cover of `rest` (< chunk) tokens: exact widths, then ONE last
    forward that may be padded. Cost = forwards * forward_cost_rows + rows run.
    Returns (cost, forwards, ((width, real), ...)); ties prefer fewer forwards.
    The table for a (widths, cost) pair is built once."""
    key = (widths, forward_cost_rows)
    best = _TAIL_TABLES.get(key)
    if best is None:
        best = [(0, 0, ())]
        for r in range(1, widths[-1]):
            choice = None
            for w in widths:
                if w >= r:    # finish here (padded when w > r); the smallest such width wins
                    candidate = (forward_cost_rows + w, 1, ((w, r),))
                else:         # one exact forward, then the rest
                    cost, count, plan = best[r - w]
                    candidate = (forward_cost_rows + w + cost, count + 1, ((w, w),) + plan)
                if choice is None or candidate[:2] < choice[:2]:
                    choice = candidate
            best.append(choice)
        _TAIL_TABLES[key] = best
    return best[rest]


def prompt_plan(n, widths, *, pad=True, room=None, forward_cost_rows=128):
    """The prompt's forwards as `[(width, real_tokens), ...]` (n >= 2).

    consult16 lever 6: full chunks first, then the remainder in as few
    forwards as the cost model allows — the LAST forward may run a prepared
    width wider than its real tokens. The extra rows are computed and thrown
    away: they sit at positions past the prompt, no real row can see them
    (attention is causal), the row counter advances by the real tokens only,
    and the next token overwrites those slots before anything reads them.
    Length-only and deterministic. `room` is the free positions after the
    prompt in the cache row; when the padding does not fit (or `pad` is off)
    the plan is the exact greedy decomposition.
    """
    chunk = widths[-1]
    full, rest = divmod(n, chunk)
    plan = [(chunk, chunk)] * full
    if rest == 0:
        return plan
    if pad:
        _, _, tail = _tail_plan(rest, widths, forward_cost_rows)
        waste = tail[-1][0] - tail[-1][1]
        if room is None or waste <= room:
            return plan + list(tail)
    return [(w, w) for w in chunk_widths(n, widths)]


class Sampler:
    def __init__(self, tokenizer, eos_id, seed):
        import torch
        self.tok = tokenizer
        self.eos = eos_id
        self.gen = torch.Generator(device="cpu")
        self.gen.manual_seed(seed)

    def step(self, logits_row, params, produced, max_new):
        """Sample one token from [V] FP32 logits. Returns (token, done, finish)."""
        import torch
        row = logits_row.float().cpu()
        temperature = float(params.get("temperature", 0.9) or 0.0)
        if temperature <= 0.0:
            tok = int(torch.argmax(row))
        else:
            logits = row / max(temperature, 1e-5)
            top_k = int(params.get("top_k", -1) or -1)
            if 0 < top_k < logits.numel():
                kth = torch.topk(logits, top_k)[0][-1]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            top_p = float(params.get("top_p", 1.0) or 1.0)
            if 0.0 < top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                probs = torch.softmax(sorted_logits, dim=-1)
                cut = torch.cumsum(probs, dim=-1) - top_p
                drop = sorted_idx[cut > 0]
                logits[drop] = float("-inf")
            probs = torch.softmax(logits, dim=-1)
            tok = int(torch.multinomial(probs, 1, generator=self.gen).item())
        done, finish = False, None
        if tok == self.eos and not params.get("ignore_eos"):
            done, finish = True, "stop"
        elif produced + 1 >= max_new:
            done, finish = True, "length"
        return tok, done, finish


class Context:
    pass


def warm_up(ctx):
    """consult16 lever 5: run every prepared call shape once, in lockstep.

    The b12x plans the body uses are prepared on first use, and that first use
    keeps memory (1.28 GB measured: 0.21 GB at the first prefill, 1.07 GiB at
    the first decode step). Run each prepared prefill width once, widest
    first, then two decode steps past the short-context bypass, so that memory
    is paid at boot — in phase A it lands in the measured fixed term instead
    of coming out of the request headroom. No sampling and no broadcast: every
    rank runs the same forwards in the same order.
    """
    import torch
    decoder, dev = ctx.decoder, ctx.device
    decoder.release([0])
    widths = sorted(ctx.shapes, reverse=True)
    if os.environ.get("HY4_AB_FILE"):
        # consult18 A/B boot: every runtime-switch plan set is prepared by a plain
        # forward here, before any overlapped pair (which refuses first-use
        # preparation): the iHC prefill plans of both arms.
        from ignition.recipes.hy4 import runtime
        for index in (0, -1):
            for key, values in runtime.ALLOWED.items():
                runtime.SWITCHES[key] = values[index]
            t = torch.full((1, widths[0]), 1, dtype=torch.int64, device=dev)
            decoder.forward(t, call_shape=ctx.shapes[widths[0]], logits_rows="none")
            decoder.complete([0], t)
            decoder.release([0])
        runtime.load()
        torch.cuda.synchronize(dev)
    for index, w in enumerate(widths):
        t = torch.full((1, w), 1, dtype=torch.int64, device=dev)
        rows_mode = "last" if index == len(widths) - 1 else "none"
        decoder.forward(t, call_shape=ctx.shapes[w], logits_rows=rows_mode)
        decoder.complete([0], t)
        if getattr(ctx, "overlap", False) and w == max(widths):
            # consult18 stage D: then warm both lanes the way requests run a
            # chunk pair. AFTER the plain chunk: the plain forward prepares
            # every b12x plan at this shape, and the overlapped loop refuses
            # first-use preparation (a device-wide synchronize between two
            # lanes' exchanges; the d2 boot hung in one).
            pair = torch.full((1, 2 * w), 1, dtype=torch.int64, device=dev)
            from ignition.recipes.hy4.hangdump import HangWatch
            with HangWatch(decoder.collectives.transport, float(os.environ.get("HY4_HANGWATCH_S") or "240"),
                           "warm overlap", rank=ctx.rank, device_dump=False):
                decoder.forward_overlapped(pair, half_shape=ctx.shapes[w], logits_rows="none")
                torch.cuda.synchronize(dev)
            decoder.complete([0], pair)
            from ignition.recipes.hy4 import runtime
            if runtime.SWITCHES.get("graph_moe") == "on" or os.environ.get("HY4_AB_FILE"):
                # consult18 P1: per-lane CUDA graphs of the MLP block (captured
                # outside the overlap, after its plans and lane streams exist).
                t_cap = time.monotonic()
                n_graphs = decoder.capture_mlp_graphs(w)
                trace(ctx.rank, f"mlp graphs: {n_graphs} captured at {w} rows in "
                                f"{time.monotonic() - t_cap:.1f} s")
    for _ in range(2):
        t = torch.full((1, 1), 1, dtype=torch.int64, device=dev)
        decoder.decode_step(t, call_shape=ctx.decode_shape)
        decoder.complete([0], t)
    decoder.release([0])
    if os.environ.get("HY4_STATE_CENSUS"):
        # (after the warm-up and a release: the census chunk starts a fresh row)
        # consult18 S2: which reachable state does one prefill chunk WRITE?
        from ignition.recipes.hy4.census import (checksums, host_changes, host_state, mutated,
                                                 reachable_storages)
        w = max(widths)
        found = reachable_storages(decoder)
        torch.cuda.synchronize(dev)
        before = checksums(found)
        host_before = host_state(decoder)
        t = torch.full((1, w), 1, dtype=torch.int64, device=dev)
        decoder.forward(t, call_shape=ctx.shapes[w], logits_rows="none")   # a fresh row
        decoder.complete([0], t)
        torch.cuda.synchronize(dev)
        decoder.release([0])
        changed = mutated(found, before, checksums(found))
        total = sum(int(flat.numel()) for _, flat in found.values())
        trace(ctx.rank, f"state census: {len(found)} storages ({total / 2**30:.1f} GiB) reachable, "
                        f"{len(changed)} written by one {w}-row prefill chunk")
        for path, nbytes in changed:
            trace(ctx.rank, f"state census: WRITTEN {nbytes:>12d} B {path}")
        rebound = host_changes(host_before, host_state(decoder))
        trace(ctx.rank, f"state census: {len(rebound)} host attributes changed by the same chunk")
        for path, was, now in rebound:
            trace(ctx.rank, f"state census: HOST {path}: {str(was)[:60]} -> {str(now)[:60]}")
    from ignition.recipes.hy4.moe import CALIBRATION, MoeCalibration
    if MoeCalibration.enabled():
        # consult18 E2: only real requests calibrate (warm-up feeds token 1s).
        CALIBRATION.reset()
        trace(ctx.rank, f"moe calibration armed: {len(CALIBRATION.buffers)} routed layers")
    decoder.trace.clear()
    torch.cuda.synchronize(dev)
    return sum(widths) + 2


def settled_free(device):
    """Free device bytes with the allocator's cached blocks returned."""
    import gc
    import torch
    torch.cuda.synchronize(device)
    gc.collect()
    torch.cuda.empty_cache()
    return torch.cuda.mem_get_info(device)


def run_generate(ctx, cmd):
    """One generation; every rank runs the full loop in lockstep."""
    import torch
    from ignition.recipes.hy4.profile import TIMER, WATCH
    dist = ctx.dist
    decoder = ctx.decoder
    if os.environ.get("HY4_AB_FILE"):
        # consult18 E4: runtime A/B switches, read by every rank per request.
        from ignition.recipes.hy4 import runtime
        trace(ctx.rank, f"a/b switches: {runtime.load()}")
    dev = ctx.device
    ids = cmd["tokens"]
    max_new = cmd["max_new"]
    params = cmd["params"]
    decode_shape = ctx.decode_shape
    ctx.last_nll = None

    # W29: --profile-steps N times the first N forwards of every request AFTER
    # the warm request (the first request of the boot warms the row and is
    # never timed); at request end rank 0 prints `PROFILE {json}` and writes
    # /state/profile-<ts>.json.
    TIMER.begin_request(ctx.profile_steps if ctx.warm_done else 0)
    ctx.warm_done = True

    WATCH.reset()   # consult17: HY4_FINITE_WATCH=1 names the first non-finite tensor per request
    decoder.release([0])
    n = len(ids)
    if n == 1:
        t = torch.tensor([[ids[0]]], dtype=torch.int64, device=dev)
        logits = decoder.decode_step(t, call_shape=decode_shape)
        decoder.complete([0], t)
    else:
        pos = 0
        logits = None
        nll_sum, nll_count = 0.0, 0
        plan = prompt_plan(n, ctx.widths, pad=ctx.pad, room=ctx.row_capacity - n,
                           forward_cost_rows=ctx.forward_cost_rows)
        # consult18 stage D: with --overlap, two consecutive full chunks run as
        # one overlapped pair — two CHUNK-row halves on two lanes. (Not for
        # the NLL probe: its all-row logits of a pair would be ~1 GB.)
        full = max(ctx.widths)
        steps, index = [], 0
        while index < len(plan):
            pair = (getattr(ctx, "overlap", False) and not params.get("nll")
                    and index + 1 < len(plan)
                    and plan[index] == (full, full) and plan[index + 1] == (full, full))
            steps.append((2 * full, 2 * full, True) if pair else (*plan[index], False))
            index += 2 if pair else 1
        for index, (w, real, overlapped) in enumerate(steps):
            # A padded forward repeats its last real token; those rows are
            # never committed (`append(real)`) and never read (lever 6).
            chunk_ids = ids[pos:pos + real] + [ids[pos + real - 1]] * (w - real)
            t = torch.tensor([chunk_ids], dtype=torch.int64, device=dev)
            logits = None
            # consult16 lever 3: only the prompt's final token feeds the
            # sampler — no head and no vocabulary gather for the other chunks,
            # one row gathered for the last.
            last = index == len(steps) - 1
            if params.get("nll"):
                # consult18 stage 0b: every row's full-vocabulary logits (a probe,
                # not a serving path), scored against the next prompt token.
                logits = decoder.forward(t, call_shape=ctx.shapes[w], logits_rows="all")
                if ctx.rank == 0:
                    targets = ids[pos + 1:pos + real + 1]
                    rows = len(targets)
                    for start in range(0, rows, 128):
                        stop = min(start + 128, rows)
                        block = logits[0, start:stop].float()
                        want = torch.tensor(targets[start:stop], device=block.device)
                        lsm = torch.logsumexp(block, dim=-1) - block.gather(
                            1, want.unsqueeze(1)).squeeze(1)
                        nll_sum += float(lsm.sum())
                        nll_count += stop - start
                logits = logits[:, real - 1:real, :]
            elif overlapped:
                watch_s = float(os.environ.get("HY4_HANGWATCH_S") or 0)
                if watch_s > 0:
                    # consult18 D debug: dump stacks + host-mapped breadcrumbs if a pair stalls.
                    from ignition.recipes.hy4.hangdump import HangWatch
                    with HangWatch(decoder.collectives.transport, watch_s, f"pair@{pos}",
                                   rank=ctx.rank, device_dump=False):
                        logits = decoder.forward_overlapped(
                            t, half_shape=ctx.shapes[full],
                            logits_rows=real - 1 if last else "none")
                        torch.cuda.synchronize(dev)
                else:
                    logits = decoder.forward_overlapped(
                        t, half_shape=ctx.shapes[full], logits_rows=real - 1 if last else "none")
            else:
                logits = decoder.forward(t, call_shape=ctx.shapes[w],
                                         logits_rows=real - 1 if last else "none")
            decoder.append([0], real)
            pos += real
        ctx.last_nll = (nll_sum, nll_count) if params.get("nll") and ctx.rank == 0 else None
    out = []
    while True:
        if ctx.rank == 0:
            nxt, done, finish = ctx.sampler.step(logits[:, -1, :], params, len(out), max_new)
        else:
            nxt = done = finish = None
        obj = [nxt, done, finish]
        dist.broadcast_object_list(obj, src=0)
        nxt, done, finish = obj
        out.append(nxt)
        if done:
            break
        t = torch.tensor([[nxt]], dtype=torch.int64, device=dev)
        logits = decoder.decode_step(t, call_shape=decode_shape)
        decoder.complete([0], t)
        if WATCH.finding is not None and "generated_tokens" not in WATCH.finding:
            WATCH.finding["generated_tokens"] = len(out)   # decode steps completed before it
    finding = WATCH.take()
    if finding is not None:
        finding.setdefault("generated_tokens", 0)          # 0 = during the prompt
        finding["prompt_tokens"] = len(ids)
        print(f"[h4 r{ctx.rank}] FINITE-WATCH " + json.dumps(finding), flush=True)
    decoder.release([0])
    # consult16 lever 4: the wiring trace is a gate instrument (two entries per
    # layer per forward); a serving process must not grow it without bound.
    decoder.trace.clear()
    ce_checked = getattr(getattr(decoder.collectives, "transport", None), "ce_audit_calls", None)
    if ce_checked and any(ce_checked.values()):
        # HY4_CE_AUDIT=1 (a gate boot): a gather byte or merge outside the band would have raised.
        trace(ctx.rank, f"ce audit: {ce_checked['gather']} gathers byte-identical, "
                        f"{ce_checked['lse']} merges, max rel-L2 "
                        f"{decoder.collectives.transport.ce_audit_max:.3e} vs the kernel")
    ring_checked = getattr(getattr(decoder.collectives, "transport", None), "ar_audit_calls", 0)
    if ring_checked:
        # HY4_AR_AUDIT=1 (a gate boot): a ring sum outside the band would have raised.
        trace(ctx.rank, f"all-reduce audit: {ring_checked} ring calls, max rel-L2 "
                        f"{decoder.collectives.transport.ar_audit_max:.3e} vs rsag")
    lane_checked = getattr(getattr(decoder.collectives, "transport", None), "lane_audit_calls", None)
    if lane_checked and any(lane_checked.values()):
        # HY4_LANE_AUDIT=1 (a gate boot): lane 1's channels vs the gloo reference;
        # a byte or a merge/sum outside the band would have raised.
        mx = decoder.collectives.transport.lane_audit_max
        trace(ctx.rank, f"lane audit: {lane_checked['gather']} gathers + {lane_checked['scores']} "
                        f"score offers byte-identical, {lane_checked['lse']} merges max rel-L2 "
                        f"{mx['lse']:.3e}, {lane_checked['all_reduce']} ring sums max rel-L2 "
                        f"{mx['all_reduce']:.3e}, {lane_checked.get('ar_projection', 0)} ring-sum "
                        f"projections max {mx.get('ar_projection', 0.0):.3e} vs gloo (post-loop)")
    pairs = getattr(decoder, "overlap_pairs", 0)
    if pairs:
        # consult18 D: overlapped pairs so far, and how many the post-loop
        # widen/exactness check sent back through the single-lane path.
        verified = getattr(decoder, "overlap_verified", 0)
        from ignition.recipes.hy4.lanes import SECTIONS
        trace(ctx.rank, f"overlap: {pairs} pairs, {getattr(decoder, 'overlap_reruns', 0)} "
                        f"recomputed single-lane (widen flag); cooperative admission: "
                        f"{getattr(decoder, 'overlap_regime', '?')} ({SECTIONS[0]} exclusive sections)"
                        + (f", {verified} verified vs single-lane, max rel-L2 "
                           f"{decoder.overlap_verify_max:.3e} (single-lane noise floor "
                           f"{decoder.overlap_verify_floor:.3e})" if verified else ""))
    checked = getattr(getattr(decoder.collectives, "transport", None), "score_wire_checked", 0)
    if checked:
        # HY4_SCORE_WIRE_AUDIT=1 (a gate boot): a differing bit would have raised.
        trace(ctx.rank, f"score wire audit: {checked} offer gathers bit-identical on both wires")
    audit = getattr(decoder.collectives, "audit", False)
    if audit:
        # HY4_SELECTION_AUDIT=1 (a gate boot): every scores-only selection was
        # checked against the exact merge; anything but a tie at the cut raised.
        trace(ctx.rank, f"selection audit: {decoder.collectives.audit_rows} rows checked, "
                        f"{decoder.collectives.audit_swaps} with a tie swap at the cut, 0 other")
    report = TIMER.end_request()
    if report is not None and ctx.rank == 0:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        payload = {"ts": stamp, "prompt_tokens": len(ids), "completion_tokens": len(out),
                   "profile": report}
        print("PROFILE " + json.dumps(payload), flush=True)
        try:
            path = f"/state/profile-{stamp}.json"
            with open(path, "w") as fh:
                json.dump(payload, fh, indent=2)
                fh.write("\n")
        except OSError as error:
            trace(0, f"profile write failed: {error}")
    from ignition.recipes.hy4.moe import CALIBRATION, MoeCalibration
    if MoeCalibration.enabled() and CALIBRATION.buffers:
        # consult18 E2 calibration boot: this rank's per-expert amax so far.
        torch.cuda.synchronize(ctx.device)
        root = os.environ.get("HY4_MOE_CALIBRATE_DIR") or "/state/hy4-moe-amax"
        saved = CALIBRATION.save(os.path.join(root, f"r{ctx.rank}.pt"))
        trace(ctx.rank, f"moe calibration: {saved} layers saved after a {len(ids)}-token request")
    return out, finish


def serve_http(ctx):
    """Rank 0's OpenAI-style HTTP surface (threaded; enqueues to the main loop)."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import socket

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def _send(self, status, payload, content_type="application/json"):
            data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                if ctx.failure:
                    self._send(503, {"status": "failed", "detail": str(ctx.failure)[:200]})
                else:
                    self._send(200, {"status": "ok"})
            elif self.path == "/v1/models":
                self._send(200, {"object": "list", "data": [
                    {"id": name, "object": "model", "owned_by": "flare"} for name in ctx.served_names]})
            else:
                self._send(404, {"error": {"message": f"no route {self.path}", "type": "not_found"}})

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length)
            if self.path != "/v1/chat/completions":
                self._send(404, {"error": {"message": f"no route {self.path}", "type": "not_found"}})
                return
            try:
                body = json.loads(raw)
                if not isinstance(body, dict):
                    raise ValueError("the request body must be a JSON object")
                if body.get("stream"):
                    raise ValueError("h4 direct serving answers non-streaming requests only")
                if body.get("n", 1) != 1:
                    raise ValueError("h4 direct serving returns one choice")
                if "messages" in body:
                    text = ctx.tok.apply_chat_template(
                        body["messages"], tokenize=False, add_generation_prompt=True,
                        tools=body.get("tools") or None)
                elif "prompt" in body:
                    text = str(body["prompt"])
                else:
                    raise ValueError("the body needs `messages` or `prompt`")
                ids = ctx.tok.encode(text, add_special_tokens=False)
                if not ids:
                    raise ValueError("empty prompt")
                limit = body.get("max_completion_tokens", body.get("max_tokens"))
                limit = 512 if limit is None else int(limit)
                if limit < 1:
                    raise ValueError("the output limit must be positive")
                if len(ids) > ctx.row_capacity:
                    raise ValueError(f"prompt longer than the row capacity ({ctx.row_capacity})")
                if len(ids) + limit > ctx.max_context:
                    limit = ctx.max_context - len(ids)
                    if limit < 1:
                        raise ValueError(f"prompt exceeds the served context ({ctx.max_context})")
                params = {
                    "temperature": body.get("temperature", 0.9),
                    "top_p": body.get("top_p", 1.0),
                    "top_k": body.get("top_k", -1),
                    "ignore_eos": bool(body.get("ignore_eos", False)),
                    # consult18 stage 0b: teacher-forced mean NLL over the prompt
                    # (the numerics yardstick); returned in usage.
                    "nll": bool(body.get("nll", False)),
                }
            except (ValueError, TypeError) as error:
                self._send(400, {"error": {"message": str(error), "type": "invalid_request_error"}})
                return
            item = {"tokens": ids, "max_new": limit, "params": params, "ev": threading.Event()}
            with ctx.lock:
                if ctx.busy:
                    self._send(429, {"error": {"message": "h4 serves one request at a time", "type": "busy"}})
                    return
                ctx.busy = True
            try:
                ctx.q.put(item)
                if not item["ev"].wait(timeout=7200):
                    raise TimeoutError("the serve loop did not answer in 7200 s")
            finally:
                with ctx.lock:
                    ctx.busy = False
            if item.get("error"):
                ctx.failure = item["error"]
                self._send(500, {"error": {"message": item["error"][:400], "type": "server_error"}})
                return
            out, finish = item["result"]
            nll = item.get("nll")
            content = ctx.tok.decode(out, skip_special_tokens=True)
            created = int(time.time())
            self._send(200, {
                "id": f"chatcmpl-h4-{created}", "object": "chat.completion", "created": created,
                "model": ctx.served_names[0],
                "choices": [{"index": 0,
                             "message": {"role": "assistant", "content": content},
                             "finish_reason": finish or "stop"}],
                "usage": {"prompt_tokens": len(ids), "completion_tokens": len(out),
                          "total_tokens": len(ids) + len(out),
                          **({} if nll is None else {"nll_sum": nll[0], "nll_tokens": nll[1],
                                                     "nll_mean": nll[0] / max(nll[1], 1)})},
            })

    host, _, port = ctx.bind.rpartition(":")
    listener = type("Listener", (ThreadingHTTPServer,), {"address_family": socket.AF_INET})
    server = listener((host, int(port)), Handler)
    server.daemon_threads = True
    ctx.http = server
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default="/model")
    ap.add_argument("--host", default="0.0.0.0", help="listening address (command-app greps the argv)")
    ap.add_argument("--port", type=int, default=8001, help="listening port (the `smart` role port)")
    ap.add_argument("--served-model-name", default="smart")
    ap.add_argument("--max-concurrency", type=int, default=1)
    ap.add_argument("--max-context", type=int, default=DEFAULT_MAX_CONTEXT)
    ap.add_argument("--rows", type=int, default=2, help="cache request rows (row 0 serves)")
    ap.add_argument("--capacity", type=int, default=0,
                    help="cache capacity in GLOBAL positions per row (0 = size from the measured budget)")
    ap.add_argument("--headroom", type=int, default=5580050432,
                    help="bytes reserved for request-time transients beyond weights/cache/plans; "
                         "with --boot-warm the first-use plan memory is in the fixed term, "
                         "not in this reserve")
    ap.add_argument("--partial-wire", choices=("latent", "expanded"), default="latent",
                    help="consult17 (lever 9): expanded makes prefill graphs expand every "
                         "head's partial through W_UV before the LSE reduce-scatter, so the "
                         "wire carries v_head_dim values per head (256) instead of the latent "
                         "width (512). Decode keeps the latent wire. Not bit-identical to "
                         "latent: the BF16 rounding moves ahead of the merge")
    ap.add_argument("--selection-wire", choices=("gloo", "dcp"), default="gloo",
                    help="consult17 (lever 2 step 2): dcp carries the scores-only first offer "
                         "on a DCP all-gather channel as raw 16-bit words; gloo is the host "
                         "wire. Needs --selection-exchange scores-only and a --local-offer; "
                         "the full-width re-offer always rides gloo")
    ap.add_argument("--all-reduce-arm", default="1:oneshot,2-:rsag",
                    help="consult17: which b12x all-reduce arm each row count runs, as "
                         "ROWS:ARM[@THREADSxBLOCKS] rules in order (ROWS = N, LO-HI or LO-; "
                         "ARM = oneshot, rsag, pull, twoshot or dma). The default is lever "
                         "1's: one-shot at 1 row, rsag above at 256x16. consult18 A1: dma is "
                         "the copy-engine ring, e.g. 1:oneshot,2-511:rsag,512-:dma")
    ap.add_argument("--all-reduce-kernel-rows", type=int, default=0,
                    help="consult18 A1: rows the kernel all-reduce pool is sized for (0 = the "
                         "prefill chunk). With dma carrying the wide rows, size it to the widest "
                         "kernel-arm row count; its IPC storage scales with it")
    ap.add_argument("--lse-transport", choices=("kernel", "ce", "ce-relay"), default="kernel",
                    help="consult18 B1: ce runs the prefill LSE reduce-scatter on copy-engine "
                         "peer copies + a local FP32 merge (needs a dma all-reduce arm); kernel "
                         "keeps the b12x DCP kernel")
    ap.add_argument("--gather-transport", choices=("kernel", "ce", "ce-relay"), default="kernel",
                    help="consult18 B1: ce runs the prefill head all-gather on copy-engine "
                         "peer copies (byte-exact); kernel keeps the b12x DCP kernel")
    ap.add_argument("--ce-rows", type=int, default=1024,
                    help="consult18 B1: narrowest row count the CE exchange serves")
    ap.add_argument("--overlap", type=int, choices=(0, 1), default=0,
                    help="consult18 stage D: run every two consecutive full prefill chunks as "
                         "one overlapped pair on two lanes (streams), layer-pipelined, so one "
                         "chunk's collectives overlap the other's compute. Needs the dma ring "
                         "and the CE exchanges at the chunk width")
    ap.add_argument("--profile-phases", type=int, choices=(0, 1), default=1,
                    help="consult18 stage D: W29 per-phase timing synchronises the device at "
                         "every phase and so serialises the overlap; 0 times whole forwards only")
    ap.add_argument("--dma-replay", type=int, choices=(0, 1), default=1,
                    help="consult18 A1: capture the ring's eager-replay graphs at boot (one "
                         "graph launch per call instead of ~110 host launches)")
    ap.add_argument("--chunk", type=int, choices=(1024, 2048, 4096), default=DEFAULT_CHUNK,
                    help="consult16 lever 6: prefill chunk cap in rows. Pools, plans and "
                         "scratch scale with it; check both caps on the boot line before "
                         "keeping a wider value")
    ap.add_argument("--pad-remainder", type=int, choices=(0, 1), default=1,
                    help="consult16 lever 6: run a prompt's remainder in as few forwards as "
                         "the cost model allows, the last one padded to a prepared width "
                         "(0 = the exact power-of-two decomposition)")
    ap.add_argument("--forward-cost-rows", type=int, default=128,
                    help="cost of one extra forward, in rows, for the remainder plan "
                         "(W29: a small forward costs about as much as 130-150 rows)")
    ap.add_argument("--boot-warm", type=int, choices=(0, 1), default=1,
                    help="consult16 lever 5: run every prepared call shape once in phase A "
                         "(before pool sizing) and again in phase B (0 = the old boot)")
    ap.add_argument("--warm-shortfall-limit", type=int, default=1 << 30,
                    help="abort the boot when the free bytes after the phase-B warm-up fall "
                         "short of --headroom by more than this (default 1 GiB)")
    ap.add_argument("--report", default="/state/boot-report.json")
    ap.add_argument("--selection-exchange", choices=("scores-only", "candidates"),
                    default="scores-only",
                    help="consult16 lever 2: scores-only moves one [rows, offer] score tensor "
                         "per rank and each rank keeps its own keys; candidates is the old "
                         "exact merge of every rank's scores AND positions (the rollback)")
    ap.add_argument("--local-offer", type=int, default=512,
                    help="scores-only first offer width (wiring.DEFAULT_LOCAL_OFFER); the ranks "
                         "re-offer the full width together when any offer is saturated; "
                         "0 = always the full width")
    ap.add_argument("--profile-steps", type=int, default=0,
                    help="W29 instrumentation: time the first N forwards of every request "
                         "after the warm request; rank 0 prints PROFILE {json} and writes "
                         "/state/profile-<ts>.json (0 = the instrument is a no-op)")
    args = ap.parse_args()

    import torch
    import torch.distributed as dist

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    trace(rank, f"init_process_group gloo (world {world})")
    dist.init_process_group("gloo", timeout=timedelta(minutes=30))

    from ignition.recipes.hy4.bootstrap import (PreparationConfig, capture_kernel_graphs,
                                                construct_rank)
    from ignition.recipes.hy4.collectives import PcieDcpCollectives, PcieDcpTransport

    ctx = Context()
    ctx.rank, ctx.world, ctx.device, ctx.dist = rank, world, device, dist
    ctx.max_context = args.max_context
    ctx.row_capacity = args.capacity or args.max_context  # replaced by the measured capacity below
    ctx.served_names = (args.served_model_name,)
    ctx.bind = f"{args.host}:{args.port}"
    ctx.failure = None
    ctx.lock = threading.Lock()
    ctx.busy = False
    ctx.q = queue.Queue()
    ctx.http = None
    ctx.profile_steps = args.profile_steps
    from ignition.recipes.hy4.profile import TIMER as _TIMER
    _TIMER.phases = bool(args.profile_phases)
    ctx.warm_done = False
    CHUNK = args.chunk
    PREFILL_WIDTHS = prefill_widths(CHUNK)
    ctx.widths = PREFILL_WIDTHS
    ctx.pad = bool(args.pad_remainder)
    ctx.forward_cost_rows = args.forward_cost_rows
    ctx.overlap = bool(args.overlap)

    try:
        # Wire geometry from the real config (the w7c world-8 gate's contract):
        # prefill gathers UNABSORBED queries (qk_nope+qk_rope), decode gathers
        # the OWNER-ABSORBED width (kv_lora+qk_rope), partials are the
        # 512-wide latent. (The w14 c1 runner's 192 was a bench-only guess.)
        with open(os.path.join(args.snapshot, "config.json")) as fh:
            _cfg = json.load(fh)
        unabsorbed = _cfg["qk_nope_head_dim"] + _cfg["qk_rope_head_dim"]   # 256
        absorbed = _cfg["kv_lora_rank"] + _cfg["qk_rope_head_dim"]         # 576
        values = _cfg["kv_lora_rank"]                                      # 512
        reduce = _cfg["hidden_size"]                                       # 6144
        expanded = _cfg["v_head_dim"] if args.partial_wire == "expanded" else None   # 256
        trace(rank, f"transport construction (gather widths {unabsorbed}/{absorbed}, "
                    f"values {values}, reduce {reduce})")
        score_offer = None
        if args.selection_wire == "dcp":
            if args.selection_exchange != "scores-only" or not args.local_offer:
                raise ValueError("--selection-wire dcp needs --selection-exchange scores-only "
                                 "and a non-zero --local-offer")
            score_offer = args.local_offer
        from ignition.recipes.hy4.collectives import parse_all_reduce_arms

        uses_dma = any(rule[2] == "dma" for rule in parse_all_reduce_arms(args.all_reduce_arm))
        if os.environ.get("HY4_AR_AUDIT") and args.all_reduce_kernel_rows:
            raise ValueError("HY4_AR_AUDIT runs the rsag arm at the ring's rows: leave "
                             "--all-reduce-kernel-rows at 0")
        transport = PcieDcpTransport.from_process_group(
            dist.group.WORLD, device, total_heads=HEADS_TOTAL,
            query_widths=(unabsorbed, absorbed), value_width=values, max_rows=CHUNK,
            reduce_width=reduce, all_reduce_arms=args.all_reduce_arm,
            score_offer=score_offer,
            # consult17 (lever 9): with expanded partials only the one-row
            # decode still uses the latent pool, so it shrinks to one row.
            extra_value_widths=() if expanded is None else (expanded,),
            value_max_rows=None if expanded is None else 1,
            dma=uses_dma, kernel_rows=args.all_reduce_kernel_rows or None,
            lanes=2 if args.overlap else 1, lane_rows=CHUNK,
            ce_lse_rows=args.ce_rows if args.lse_transport.startswith("ce") else None,
            ce_gather_rows=args.ce_rows if args.gather_transport.startswith("ce") else None,
            ce_lse_width=expanded or values, ce_gather_width=unabsorbed,
            # consult18 B2: ce-relay = pairwise relay across the B<->D link.
            ce_lse_relay=args.lse_transport == "ce-relay",
            ce_gather_relay=args.gather_transport == "ce-relay")
        trace(rank, f"dcp exchange: lse {args.lse_transport}, gather {args.gather_transport}"
                    f" (ce from {args.ce_rows} rows)")
        if uses_dma:
            trace(rank, "all-reduce: copy-engine ring built for the dma rows"
                        + (f"; kernel pool sized for {args.all_reduce_kernel_rows} rows"
                           if args.all_reduce_kernel_rows else ""))
        trace(rank, f"partial wire: {args.partial_wire}"
                    + ("" if expanded is None else f" (prefill partials {expanded} wide)"))
        trace(rank, f"all-reduce arms: {args.all_reduce_arm}")
        collectives = PcieDcpCollectives(rank=rank, world_size=world, transport=transport,
                                         exchange_mode=args.selection_exchange,
                                         local_offer=args.local_offer or None)
        trace(rank, f"selection exchange {args.selection_exchange} "
                    f"(local offer {args.local_offer or 'full'})")
        trace(rank, "transport constructed")

        settings = PreparationConfig(devices=tuple(range(world)), mtp=False,
                                     prefill_chunk_size=CHUNK)

        def construct_and_wire(cache_rows, cache_capacity):
            """One construction + operator wiring + the boot's full plan book."""
            t0 = time.monotonic()
            built = construct_rank(args.snapshot, settings, rank=rank, device=device, pinned=True,
                                   mtp=False, cache_rows=cache_rows, cache_capacity=cache_capacity,
                                   world_size=world, collectives=collectives, fused_cache=True)
            load_s = round(time.monotonic() - t0, 1)
            trace(rank, f"constructed ({load_s} s, {built.memory_report['weights_bytes']} B weights, "
                        f"cache {cache_capacity} global/row x {cache_rows} rows)")
            built.decoder.partial_wire = args.partial_wire   # before the plan book binds the folds
            built.wire_operators(session="prepared")
            shp = {w: built.decoder.default_call_shape((1, w)) for w in PREFILL_WIDTHS}
            decode_shape = built.decoder.default_call_shape((1, 1))
            all_shapes = [decode_shape] + [shp[w] for w in PREFILL_WIDTHS]
            if world > 1:
                # The CP partial binds the UNBOUND sink at every call shape
                # (D2/D6); only the graph-capture shapes also need the world-1
                # variant (capture_kernel_graphs folds the sink in kernel).
                built.decoder.bind_plan_book(None, all_shapes, sink_variants=(False,), share_scratch=True,
                                             lanes=2 if args.overlap else 1)
                built.decoder.bind_plan_book(None, [decode_shape, shp[CHUNK]], sink_variants=(True,), share_scratch=True)
            else:
                built.decoder.bind_plan_book(None, all_shapes, sink_variants=(True,), share_scratch=True)
            if args.overlap:
                # consult18 stage D: lane 1's selection buffers and MoE workspace.
                built.decoder.enable_lanes(max_q_rows=CHUNK)
            return built, shp, decode_shape, load_s

        # The transport plans belong to the transport, not to a construction:
        # prepare them BEFORE phase A so their memory is inside the measured
        # fixed term (they used to be prepared after the pool was sized, out
        # of the headroom) and so the phase-A warm-up can run (lever 5).
        from b12x.preparation import PreparationSession

        session = PreparationSession(device=device, autotune=False, compile_workers=1)
        # Prepared plans only (no JIT at run time): the decode shape gathers the
        # absorbed width at 1 query row; every prepared prefill width gathers
        # the unabsorbed width at `w` query rows; the LSE reduce-scatter is
        # 512-wide latent partials, base 2, at every query-row count.
        transport.prepare_gather(session, 1, absorbed)
        for w in PREFILL_WIDTHS:
            transport.prepare_gather(session, w, unabsorbed)
        if expanded is None:
            for w in (1, *PREFILL_WIDTHS):
                transport.prepare_lse(session, w, values, lse_base="base2")
        else:
            # Decode merges latents at one row; prefill merges expanded values.
            transport.prepare_lse(session, 1, values, lse_base="base2")
            for w in PREFILL_WIDTHS:
                transport.prepare_lse(session, w, expanded, lse_base="base2")
        # consult15 §2 lever 1: one b12x PCIe all-reduce plan per body row
        # shape (decode 1 row + every prepared prefill width) — oneshot at 1
        # row, rsag above (c1-unit-costs.md §1's arm choice).
        for w in (1, *PREFILL_WIDTHS):
            transport.prepare_all_reduce(session, w)
        if uses_dma and args.dma_replay:
            graphs = transport.prepare_dma_replay()
            trace(rank, f"all-reduce: {graphs} ring replay graph(s) captured")
        if args.overlap:
            transport.prepare_lane_all_reduce(session, 1, CHUNK)
            trace(rank, f"overlap: lane 1 ring + CE exchange prepared for {CHUNK} rows")
        if score_offer is not None:
            # consult17: the offer gather at every row count the body runs.
            for w in (1, *PREFILL_WIDTHS):
                transport.prepare_score_gather(session, w)
        trace(rank, "transport plans prepared")

        # ---- phase A: measure the fixed costs with a negligible KV cache ----
        # The plan book (iHC plans + sparse-MLA plans + sink folds + scratch)
        # is an unmeasured budget term the capture's kernel-chain figure does
        # NOT cover — the first h4 boot OOMed in prepare_plans by sizing the
        # pool before knowing it. Measure it (a), then size the pool around
        # it (b) — honest pool sizing over shape-gutting.
        # The phase-A cache must hold the warm-up's context (every prepared
        # width once + two decode steps), rounded up to whole 1,024-position pages.
        warm_tokens = sum(PREFILL_WIDTHS) + 2 + (2 * CHUNK if args.overlap else 0)
        phase_a_capacity = 1024 * (-(-(warm_tokens + 8) // 1024)) if args.boot_warm else 1024
        built_a, shapes_a, decode_shape_a, load_a = construct_and_wire(1, phase_a_capacity)
        cache_a = sum(t.numel() * t.element_size()
                      for t in (*built_a.cache.records.values(), *built_a.cache.index.values()))
        warm_retained = 0
        if args.boot_warm:
            # Keep the old accounting for construction (its allocator slack
            # counts as fixed) and add only what the first use retains: the
            # warm-up's own transients are returned before the reading.
            raw_free, _ = torch.cuda.mem_get_info(device)
            before_warm, _ = settled_free(device)
            slack = before_warm - raw_free
            ctx.decoder, ctx.decode_shape, ctx.shapes = built_a.decoder, decode_shape_a, shapes_a
            t0 = time.monotonic()
            warm_up(ctx)
            ctx.decoder = ctx.decode_shape = ctx.shapes = None
            settled, total = settled_free(device)
            warm_retained = before_warm - settled
            free_a = settled - slack
            trace(rank, f"phase A warm-up: {warm_tokens} tokens in {time.monotonic() - t0:.1f} s, "
                        f"first-use memory {warm_retained} B (now in the fixed term)")
        else:
            free_a, total = torch.cuda.mem_get_info(device)
        weights_bytes = built_a.memory_report["weights_bytes"]
        fixed_bytes = total - weights_bytes - cache_a - free_a
        del built_a, shapes_a, decode_shape_a
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        free_post, _ = torch.cuda.mem_get_info(device)
        # The cache may occupy the phase-A residue AFTER the fixed costs the
        # serving construction re-pays (weights + plan book + context): that
        # residue is exactly free_a + cache_a. (free_post is nearly the whole
        # card — the phase-A weights left with it — and is NOT the budget.)
        budget = free_a + cache_a - args.headroom
        # cache bytes ~= rows * (capacity / world) * (78*656 + 21*256 real residency)
        per_global = args.rows * (78 * 656 + 21 * 256) / world
        capacity = args.capacity if args.capacity > 0 else int(budget // per_global) & ~1023
        trace(rank, f"phase A: fixed(book+ctx+runtime)={fixed_bytes} B, torn-down free={free_post} B, "
                    f"headroom={args.headroom} B -> capacity {capacity} global/row")
        if capacity < 2 * world:
            raise RuntimeError(f"memory budget {budget} B cannot hold a usable KV cache")

        # ---- phase B: the serving construction at the measured capacity ----
        built, shapes, decode_shape, load_s = construct_and_wire(args.rows, capacity)
        ctx.decoder = built.decoder
        ctx.decode_shape = decode_shape
        ctx.shapes = shapes
        ctx.row_capacity = capacity
        warm_free = warm_shortfall = None
        if args.boot_warm:
            # Pay the first-use memory again on the serving construction, then
            # check the reserve the requests will really have. A shortfall
            # means phase B kept more than phase A measured (a cost that
            # scales with the cache); it is reported on every rank and, past
            # the limit, refused — better at boot than as a request-time OOM.
            warm_up(ctx)
            warm_free, _ = settled_free(device)
            warm_shortfall = max(0, args.headroom - warm_free)
            trace(rank, f"phase B warm-up: free {warm_free} B after first use "
                        f"(headroom {args.headroom} B, shortfall {warm_shortfall} B)")
            if warm_shortfall > args.warm_shortfall_limit:
                raise RuntimeError(
                    f"only {warm_free} B free after the boot warm-up, {warm_shortfall} B short of "
                    f"--headroom {args.headroom}; lower the pool (raise --headroom) or find "
                    "what phase B keeps that phase A did not")
            ctx.warm_done = True   # the boot warmed the row: the first request may be timed

        # ---- the P5 measurement step: post-capture kernel chain + KV pool ----
        graph_bytes, _ = capture_kernel_graphs(
            built, shapes=(("decode", (1, 1)), ("prefill", (1, CHUNK))))
        torch.cuda.synchronize(device)
        free, total = torch.cuda.mem_get_info(device)
        cache_bytes = sum(t.numel() * t.element_size()
                          for t in (*built.cache.records.values(), *built.cache.index.values()))
        binding_line, binding_facts = allocate_line(
            cache_bytes, CONTRACT_RECORD_BYTES, graph_bytes, world,
            label="Allocate cache")
        residue_line, residue_facts = allocate_line(
            free + cache_bytes, CONTRACT_RECORD_BYTES, graph_bytes, world,
            label="Allocate cache (post-capture residue, M2-parity)")
        local_slots = args.rows * (capacity // world)
        binding_facts["real_record_bytes_per_slot"] = 78 * 656 + 21 * 256
        binding_facts["real_local_slots"] = local_slots
        binding_facts["real_aggregate_global_tokens"] = args.rows * capacity
        binding_facts["real_one_third_cap"] = (args.rows * capacity) // 3
        binding_facts["real_one_third_cap_holds_1m"] = (args.rows * capacity) // 3 >= 1_048_576
        binding_facts["plan_book_and_fixed_bytes"] = fixed_bytes
        binding_facts["headroom_bytes_reserved"] = args.headroom
        binding_facts["boot_warm"] = bool(args.boot_warm)
        binding_facts["warm_first_use_bytes_phase_a"] = warm_retained
        binding_facts["warm_free_bytes_phase_b"] = warm_free
        binding_facts["warm_shortfall_bytes"] = warm_shortfall
        print(binding_line, flush=True)
        print(residue_line, flush=True)
        rank_facts = {
            "rank": rank, "device": str(device), "load_seconds": load_a + load_s,
            "weights_bytes": weights_bytes,
            "graph_scratch_bytes": graph_bytes, "cache_bytes": cache_bytes,
            "plan_book_and_fixed_bytes": fixed_bytes,
            "phase_a_free_bytes": free_a, "phase_a_torn_down_free_bytes": free_post,
            "capacity_global_per_row": capacity,
            "device_free_after_graphs_bytes": free, "device_total_bytes": total,
        }
        verdicts = [None] * world
        dist.all_gather_object(verdicts, rank_facts)
        if rank == 0:
            report = {
                "service": "h4", "snapshot": args.snapshot,
                "world": world,
                "topology": "TP8+CP8 on switches B+D (8 cards; 0000:11:00.0 is NOT in the serve)",
                "mtp": False, "max_concurrency": args.max_concurrency,
                "served_names": list(ctx.served_names), "port": args.port,
                "max_context": args.max_context,
                "cache": {"rows": args.rows, "capacity_global_per_row": capacity,
                          "local_slots_total_per_rank": local_slots,
                          "sizing": "measured budget (phase A) minus headroom"},
                "allocate_cache_line": binding_line, "allocate_cache": binding_facts,
                "residue_line_m2_parity": residue_line, "residue_facts": residue_facts,
                "ranks": verdicts,
                "prefill_widths": list(PREFILL_WIDTHS),
                "all_reduce_arms": args.all_reduce_arm,
                "all_reduce_kernel_rows": args.all_reduce_kernel_rows or CHUNK,
                "dma_replay": bool(args.dma_replay),
                "overlap": bool(args.overlap), "profile_phases": bool(args.profile_phases),
                "lse_transport": args.lse_transport, "gather_transport": args.gather_transport,
                "ce_rows": args.ce_rows,
                "selection_wire": args.selection_wire,
                "partial_wire": args.partial_wire,
                "prefill_chunk": CHUNK, "pad_remainder": ctx.pad,
                "forward_cost_rows": ctx.forward_cost_rows,
                "engine_path": ("ignition/recipes/hy4 construct_rank + PcieDcpTransport "
                                "(w14 bring-up path); recipe.prepare's native export TODO remains"),
            }
            os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
            with open(args.report, "w") as fh:
                json.dump(report, fh, indent=2)
                fh.write("\n")
            trace(rank, f"wrote {args.report}")

        # ---- serving loop --------------------------------------------------
        if rank == 0:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(args.snapshot, trust_remote_code=False,
                                                local_files_only=True)
            tpl_path = os.path.join(args.snapshot, "chat_template.jinja")
            if os.path.exists(tpl_path):
                with open(tpl_path) as fh:
                    tok.chat_template = fh.read()
            ctx.tok = tok
            eos_id = tok.eos_token_id if tok.eos_token_id is not None else 120025
            ctx.sampler = Sampler(tok, eos_id, seed=int(time.time()) & 0xFFFFFFFF)
            serve_http(ctx)
            trace(rank, f"listening on {ctx.bind} as {ctx.served_names[0]}")
        while True:
            # Rank 0 blocks on the request queue, THEN broadcasts the command;
            # ranks 1..7 block here. (Broadcasting rank 0's own placeholder
            # instead of a queued command shuts the world down immediately.)
            item = ctx.q.get() if rank == 0 else None
            cmd = None if item is None else {
                "op": "generate", "tokens": item["tokens"],
                "max_new": item["max_new"], "params": item["params"]}
            obj = [cmd]
            dist.broadcast_object_list(obj, src=0)
            cmd = obj[0]
            if cmd is None:
                break
            if cmd["op"] == "generate":
                try:
                    profile_dir = os.environ.get("HY4_TORCH_PROFILE") or ""
                    min_new = int(os.environ.get("HY4_TORCH_PROFILE_MIN_NEW") or 0)
                    if (profile_dir and len(cmd.get("tokens") or ()) >= 4096
                            and int(cmd.get("max_new") or 0) >= min_new
                            and not getattr(ctx, "profiled", False)):
                        # consult18 census: CUDA-only kernel trace (CUPTI, no host
                        # activity tracing) of ONE large request on every rank.
                        import torch
                        from torch.profiler import ProfilerActivity, profile
                        ctx.profiled = True
                        torch.cuda.synchronize(ctx.device)
                        with profile(activities=[ProfilerActivity.CUDA]) as prof:
                            out, finish = run_generate(ctx, cmd)
                            torch.cuda.synchronize(ctx.device)
                        os.makedirs(profile_dir, exist_ok=True)
                        prof.export_chrome_trace(os.path.join(profile_dir, f"trace-r{rank}.json"))
                        trace(rank, f"torch profile written: {profile_dir}/trace-r{rank}.json")
                    else:
                        out, finish = run_generate(ctx, cmd)
                except Exception as exc:  # noqa: BLE001
                    if rank == 0:
                        item["error"] = f"{type(exc).__name__}: {exc}"
                        traceback.print_exc()
                        item["ev"].set()
                    raise
                if rank == 0:
                    item["result"] = (out, finish)
                    item["nll"] = getattr(ctx, "last_nll", None)
                    item["ev"].set()
        trace(rank, "shutdown")
    finally:
        try:
            dist.destroy_process_group()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
