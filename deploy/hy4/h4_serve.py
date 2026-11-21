#!/usr/bin/env python
"""h4_serve: serve the Hy4 NVFP4 W4A16 export on Turin (TP8+CP8, world 8).

Launch chain: h4.service -> /opt/h4/start-h4.sh (docker run) ->
torch.distributed.run --nproc_per_node=8 -> this script. The engine path is the
campaign's working one (the w14 bring-up): ignition's Hy4 recipe construction
(`construct_rank`), the real PCIe-DCP collectives (PcieDcpTransport /
PcieDcpCollectives), the fused 656-byte FP8 record cache (M2), and the
post-capture `Allocate cache` line (the P5 gate). MTP off.

Rank 0 owns the OpenAI-style HTTP surface (/health, /v1/models,
/v1/chat/completions, streaming SSE or one JSON body, reasoning split into
reasoning_content; concurrent requests wait in one FIFO queue and run one at a
time) and the sampler; ranks 1..7 run the same
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
import math
import os
import queue
import select
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
    # The one-third rule applies only when total KV is less than the max
    # context supported (owner 2026-11-20). When the pool covers max context
    # the rule is inapplicable.
    applies = aggregate < 1_048_576
    if applies:
        cap = aggregate // 3
        holds_1m = cap >= 1_048_576
        tail = (f"one-third cap {cap}"
                f"{'>= 1M \u2713' if holds_1m else '< 1M \u2717'}")
    else:
        cap = None
        holds_1m = None
        tail = "one-third rule n/a (total KV >= max context)"
    return (f"{label}: {pool} B KV pool = {tokens} tokens/rank "
            f"(record {record} B; graph+scratch {graph_bytes} B measured, "
            f"kernel-chain term only; aggregate {aggregate}; {tail})"), {
        "pool_bytes": pool, "tokens_per_rank": tokens, "aggregate_tokens": aggregate,
        "one_third_rule_applies": applies,
        "one_third_cap": cap,
        "one_third_cap_holds_1m": holds_1m,
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
    def __init__(self, tokenizer, eos_id, seed, device=None):
        import torch
        self.tok = tokenizer
        self.eos = eos_id
        self.gen = torch.Generator(device="cpu")
        self.gen.manual_seed(seed)
        self.device = device if os.environ.get("HY4_GPU_SAMPLER", "1") == "1" else None
        self.gen_cuda = None

    def step(self, logits_row, params, produced, max_new):
        """Sample one token from one row of FP32 logits ([V] or [1, V]).
        Returns (token, done, finish)."""
        import torch
        temperature = float(params.get("temperature", 0.9) or 0.0)
        logits_row = logits_row.reshape(-1)      # the top-k / top-p math is over [V]
        if temperature > 0.0 and self.device is not None:
            # consult18 P6: the sampling math (softmax / top-k / top-p sort /
            # multinomial over the full vocabulary) on rank 0's GPU — 1.6 ms of
            # host CPU per token otherwise; the same distribution.
            row = logits_row.to(self.device, non_blocking=True).float()
            if self.gen_cuda is None:
                self.gen_cuda = torch.Generator(device=self.device)
                self.gen_cuda.manual_seed(int(self.gen.initial_seed()))
            gen = self.gen_cuda
        else:
            row = logits_row.float().cpu()
            gen = self.gen
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
                # consult18 P7: keep through the token whose cumulative probability
                # first reaches top_p (at least one) — the previous rule dropped
                # every token when the top one alone exceeded top_p (NaN sample)
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                probs = torch.softmax(sorted_logits, dim=-1)
                remove = (torch.cumsum(probs, dim=-1) - probs) >= top_p
                remove[0] = False
                drop = sorted_idx[remove]
                logits[drop] = float("-inf")
            probs = torch.softmax(logits, dim=-1)
            tok = int(torch.multinomial(probs, 1, generator=gen).item())
        done, finish = False, None
        if tok == self.eos and not params.get("ignore_eos"):
            done, finish = True, "stop"
        elif produced + 1 >= max_new:
            done, finish = True, "length"
        return tok, done, finish


def sample_rows(sampler, logits, params_list, produced, max_new):
    """One token per row of [B, V] logits with each row's own sampling params
    (the Sampler.step math, one device transfer and one host read for all rows).
    Returns [(token, done, finish)] per row."""
    import torch
    rows = logits.reshape(len(params_list), -1)
    device = sampler.device
    if device is not None:
        rows = rows.to(device, non_blocking=True)
        if sampler.gen_cuda is None:
            sampler.gen_cuda = torch.Generator(device=device)
            sampler.gen_cuda.manual_seed(int(sampler.gen.initial_seed()))
        gen = sampler.gen_cuda
    else:
        gen = sampler.gen
    rows = rows.float()
    temps = [float(p.get("temperature", 0.9) or 0.0) for p in params_list]
    tokens = torch.empty((rows.shape[0],), dtype=torch.long, device=rows.device)
    greedy = [i for i, t in enumerate(temps) if t <= 0.0]
    if greedy:
        idx = torch.tensor(greedy, device=rows.device)
        tokens[idx] = rows[idx].argmax(dim=-1)
    sampled = [i for i, t in enumerate(temps) if t > 0.0]
    if sampled:
        idx = torch.tensor(sampled, device=rows.device)
        scale = torch.tensor([1.0 / max(temps[i], 1e-5) for i in sampled], device=rows.device)
        scaled = rows[idx] * scale.reshape(-1, 1)
        for j, i in enumerate(sampled):
            top_k = int(params_list[i].get("top_k", -1) or -1)
            top_p = float(params_list[i].get("top_p", 1.0) or 1.0)
            row = scaled[j]
            if 0 < top_k < row.numel():
                kth = torch.topk(row, top_k)[0][-1]
                row.masked_fill_(row < kth, float("-inf"))
            if 0.0 < top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(row, descending=True)
                probs = torch.softmax(sorted_logits, dim=-1)
                remove = (torch.cumsum(probs, dim=-1) - probs) >= top_p
                remove[0] = False
                row[sorted_idx[remove]] = float("-inf")
        probs = torch.softmax(scaled, dim=-1)
        tokens[idx] = torch.multinomial(probs, 1, generator=gen).reshape(-1)
    out = []
    for i, tok in enumerate(tokens.tolist()):
        if tok == sampler.eos and not params_list[i].get("ignore_eos"):
            out.append((tok, True, "stop"))
        elif produced[i] + 1 >= max_new[i]:
            out.append((tok, True, "length"))
        else:
            out.append((tok, False, None))
    return out


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
            if runtime.SWITCHES.get("graph_moe") == "on" or os.environ.get("HY4_AB_MLP_GRAPHS"):
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


def _stream_push(ctx, out):
    """Rank 0 only: hand the request's new token ids to its SSE writer (the HTTP
    thread). A no-op for non-streaming requests and on ranks 1..7: streaming
    never alters the broadcast command, the decode loop or the numerics."""
    q = getattr(ctx, "stream_q", None)
    if q is not None:
        sent = ctx.stream_sent
        if len(out) > sent:
            q.put(("tokens", list(out[sent:])))
            ctx.stream_sent = len(out)


class IncrementalDetokenizer:
    """Streamed text deltas whose concatenation is the non-streaming
    `decode(out, skip_special_tokens=True)`: decode a short window from the last
    emitted boundary, hold back an incomplete UTF-8 tail (U+FFFD), and let
    `finish()` flush whatever remains of the full decode."""

    def __init__(self, tok):
        self.tok, self.ids, self.prefix, self.read, self.emitted = tok, [], 0, 0, ""

    def add(self, new_ids):
        self.ids.extend(new_ids)
        before = self.tok.decode(self.ids[self.prefix:self.read], skip_special_tokens=True)
        after = self.tok.decode(self.ids[self.prefix:], skip_special_tokens=True)
        if len(after) <= len(before) or after.endswith("\ufffd") or not after.startswith(before):
            return ""
        delta = after[len(before):]
        self.prefix, self.read = self.read, len(self.ids)
        self.emitted += delta
        return delta

    def finish(self):
        full = self.tok.decode(self.ids, skip_special_tokens=True)
        if full.startswith(self.emitted):
            rest = full[len(self.emitted):]
            self.emitted = full
            return rest
        return ""


# --------------------------------------------------------------------------
# Tool calls (OpenAI `tool_calls`) from the model's own markup, the one the
# chat template renders for assistant tool turns:
#   <tool_calls:opensource><tool_call:opensource>NAME
#     <arg_key:opensource>K</arg_key:opensource><arg_value:opensource>V</arg_value:opensource>...
#   </tool_call:opensource>...</tool_calls:opensource>
# String parameters stay strings; other schema types are JSON-decoded (the
# template writes non-string values with `tojson`).
# --------------------------------------------------------------------------
TOOLCALLS_BEGIN = "<tool_calls:opensource>"
_TOOL_CALL_RE = None


def _tool_param_types(tools):
    types = {}
    for tool in tools or ():
        fn = (tool or {}).get("function") or tool or {}
        props = ((fn.get("parameters") or {}).get("properties") or {})
        types[fn.get("name")] = {k: (v or {}).get("type") for k, v in props.items()}
    return types


def parse_tool_calls(text, tools):
    """`(content, calls)`: the text before the tool-call block and the OpenAI
    tool calls in it; `(text, [])` when there is no well-formed block."""
    import re
    import uuid

    global _TOOL_CALL_RE
    at = text.find(TOOLCALLS_BEGIN)
    if at < 0:
        return text, []
    if _TOOL_CALL_RE is None:
        _TOOL_CALL_RE = (
            re.compile(r"<tool_call:opensource>(.*?)</tool_call:opensource>", re.S),
            re.compile(r"<arg_key:opensource>(.*?)</arg_key:opensource>\s*"
                       r"<arg_value:opensource>(.*?)</arg_value:opensource>", re.S))
    call_re, arg_re = _TOOL_CALL_RE
    types = _tool_param_types(tools)
    calls = []
    for block in call_re.findall(text[at:]):
        name = block.split("<arg_key:opensource>", 1)[0].strip()
        if not name:
            continue
        schema = types.get(name, {})
        args = {}
        for key, value in arg_re.findall(block):
            key = key.strip()
            if schema.get(key) == "string":
                args[key] = value
            else:
                try:
                    args[key] = json.loads(value)
                except ValueError:
                    args[key] = value
        calls.append({"id": f"call_{uuid.uuid4().hex[:24]}", "type": "function",
                      "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}})
    if not calls:
        return text, []
    return text[:at], calls


MULTIMODAL_CONTENT_TYPES = frozenset({
    "image", "image_url", "input_image", "video", "video_url", "input_video",
})


def reject_multimodal_messages(messages):
    """Refuse OpenAI image/video content parts with ValueError (mapped to HTTP 400)."""
    if not isinstance(messages, list):
        return
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                kind = part.get("type")
                if kind in MULTIMODAL_CONTENT_TYPES or "image_url" in part or "video_url" in part:
                    raise ValueError(
                        "vision/image/video inputs are not enabled on this text-only serve")


def normalize_tool_history(messages):
    """The chat template iterates assistant `tool_calls[].function.arguments` as
    a mapping; OpenAI clients send it back as a JSON string. Decode it (and give
    tool-call turns an empty string for a null content)."""
    out = []
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "assistant" and message.get("tool_calls"):
            message = dict(message)
            calls = []
            for call in message["tool_calls"]:
                call = dict(call)
                fn = dict(call.get("function") or {})
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        fn["arguments"] = json.loads(args) if args.strip() else {}
                    except ValueError:
                        fn["arguments"] = {"arguments": args}
                elif args is None:
                    fn["arguments"] = {}
                call["function"] = fn
                calls.append(call)
            message["tool_calls"] = calls
            if message.get("content") is None:
                message["content"] = ""
        out.append(message)
    return out


THINK_OPEN = "<think:opensource>"
THINK_CLOSE = "</think:opensource>"

# A request whose client has left stops at the next check: every sampled token on
# the plain decode path (it rides the existing token broadcast), every
# CANCEL_EVERY tokens on the speculative path (one small broadcast from rank 0).
CANCEL_EVERY = 64
# An SSE comment line goes out after HEARTBEAT_S seconds without a write (queued
# or in prefill): it feeds client read timeouts. A waiting request checks its
# client's socket every POLL_S seconds, so a client that leaves is dropped before
# its turn.
HEARTBEAT_S = 15.0
POLL_S = 1.0
# A started generation that produces nothing for this long is reported failed.
SERVE_LOOP_TIMEOUT_S = 7200
# The serve loop's liveness (2026-11-19 idle-wedge fix). Every iteration rank 0
# broadcasts a plan and the followers wait for it in gloo, whose deadline is
# GLOO_TIMEOUT_MIN. Rank 0 used to block in queue.get() while idle, so a quiet
# serve died after 30 minutes. Now rank 0 never blocks in planning: with nothing
# queued it broadcasts an "idle" plan at least every IDLE_TICK_S. Planning or
# broadcasting (rank 0) or awaiting the plan (followers) for longer than
# LIVENESS_S is a wedge: a watchdog thread names it in the log (rank 0 also dumps
# its thread stacks) and /health turns 503 "stalled" while rank 0 is stuck.
GLOO_TIMEOUT_MIN = 30
IDLE_TICK_S = float(os.environ.get("H4_IDLE_TICK_S") or 5.0)
LIVENESS_S = float(os.environ.get("H4_LIVENESS_S") or 120.0)
# A request handler's socket: blocking socket operations give up after
# CLIENT_SOCKET_TIMEOUT_S, and TCP keepalive probes find a peer that vanished
# without closing (no FIN) in about 30 + 10 * 6 = 90 s.
CLIENT_SOCKET_TIMEOUT_S = 300.0
TCP_KEEPALIVE = (("TCP_KEEPIDLE", 30), ("TCP_KEEPINTVL", 10), ("TCP_KEEPCNT", 6))


class OutputSplitter:
    """Splits a generation into `reasoning_content` and `content` on the token
    stream, identically for streamed and whole responses.

    The chat template's thinking mode ends the prompt with THINK_OPEN, so the
    model's output starts inside its reasoning; the first THINK_CLOSE token ends
    it. An output that itself starts with THINK_OPEN is handled the same way.
    Before the first THINK_CLOSE everything is reasoning (a reply cut off while
    thinking is all reasoning); after it everything is content. Otherwise (the
    no_think mode, raw prompts) everything is content. Marker tokens appear in
    neither field. With `tool_id`, content stops at the tool-call block's token
    and the block is held for tool-call parsing.

    `feed` / `finish` return `(kind, text)` deltas, kind "reasoning" or
    "content"; `reasoning_ids`, `content_ids` and `held` are the partition."""

    def __init__(self, ctx, think_open, tool_id=None):
        self.open_id, self.close_id, self.tool_id = ctx.think_open_token, ctx.think_close_token, tool_id
        self.phase = "reasoning" if think_open else "content"
        self.first = True
        self.r, self.c = IncrementalDetokenizer(ctx.tok), IncrementalDetokenizer(ctx.tok)
        self.held = None

    @property
    def reasoning_ids(self):
        return self.r.ids

    @property
    def content_ids(self):
        return self.c.ids

    def feed(self, ids):
        events = []
        for t in ids:
            first, self.first = self.first, False
            if first and self.phase == "content" and t == self.open_id:
                self.phase = "reasoning"
                continue
            if self.phase == "reasoning":
                if t == self.close_id:
                    rest = self.r.finish()
                    if rest:
                        events.append(("reasoning", rest))
                    self.phase = "content"
                    continue
                piece = self.r.add([t])
                if piece:
                    events.append(("reasoning", piece))
                continue
            if self.held is not None:
                self.held.append(t)
                continue
            if t == self.close_id:
                continue
            if self.tool_id is not None and t == self.tool_id:
                self.held = [t]
                continue
            piece = self.c.add([t])
            if piece:
                events.append(("content", piece))
        return events

    def finish(self):
        events = []
        for kind, detok in (("reasoning", self.r), ("content", self.c)):
            rest = detok.finish()
            if rest:
                events.append((kind, rest))
        return events


def run_generate(ctx, cmd, **kwargs):
    """`run_generate_steps` driven to the end (no pause between prefill chunks)."""
    gen = run_generate_steps(ctx, cmd, **kwargs)
    while True:
        try:
            next(gen)
        except StopIteration as stop:
            return stop.value


def run_generate_steps(ctx, cmd, *, row=0, admit_only=False, preempt=None, spec_absorb=False,
                       chunked=False):
    """One generation on cache row `row`; every rank runs the full loop in lockstep.
    A generator: with `chunked` (c8 L17) it pauses after every prefill chunk but the
    last (the batched loop decodes its other rows meanwhile); its return value is
    run_generate's.

    `admit_only`: prefill and sample the first token, then return `(out,
    finish or None)` with the row kept (the batched decode loop goes on).
    `preempt` (rank 0): a callable that asks a solo request to hand over to the
    batched loop; the generation then returns `(out, "preempt")` with the row
    kept, its last token not yet in the cache (the batched loop feeds it)."""
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
    decoder.release([row])
    n = len(ids)
    # consult18 P4: greedy requests decode by MTP speculation (spec.py); the
    # prefill then also absorbs every prompt row into the draft KV.
    from ignition.recipes.hy4 import runtime as _rt
    spec = getattr(ctx, "spec", None)
    spec_temp = float(params.get("temperature", 0.9) or 0.0)
    # consult18 P5: temperature-only sampling speculates too (spec_sample.py);
    # top-k / top-p requests decode plainly.
    spec_top_k = int(params.get("top_k", -1) or -1)
    spec_top_p = float(params.get("top_p", 1.0) or 1.0)
    spec_filtered = spec_top_k > 0 or spec_top_p < 1.0
    # consult18 P7: top-k (<= 64) / top-p requests speculate too (device filter,
    # spec_sample.filter_scores; an uncovered nucleus falls back to plain decoding)
    spec_sampling = (spec_temp > 0.0 and spec_top_k <= 64 and 0.0 < spec_top_p <= 1.0
                     and (not spec_filtered or _rt.SWITCHES.get("spec_filter") == "on"))
    use_spec = (spec is not None and _rt.SWITCHES.get("spec") == "on" and n > 1
                and ((row == 0 and not admit_only) or spec_absorb)
                and max_new > 1 and not params.get("nll")
                and (spec_temp <= 0.0 or (spec_sampling and _rt.SWITCHES.get("spec_sample") == "on")))
    last_hidden = None
    if n == 1:
        t = torch.tensor([[ids[0]]], dtype=torch.int64, device=dev)
        logits = decoder.decode_step(t, request_rows=[row], call_shape=decode_shape)
        decoder.complete([row], t)
    else:
        pos = 0
        logits = None
        nll_sum, nll_count = 0.0, 0
        plan = prompt_plan(n, ctx.widths, pad=ctx.pad,
                           room=ctx.decoder.cache.cap_of(row) * ctx.decoder.cache.world - n,
                           forward_cost_rows=ctx.forward_cost_rows)
        # consult18 stage D: with --overlap, two consecutive full chunks run as
        # one overlapped pair — two CHUNK-row halves on two lanes. (Not for
        # the NLL probe: its all-row logits of a pair would be ~1 GB.)
        full = max(ctx.widths)
        is_dfly = bool(getattr(decoder, "has_dfly", False))
        need_absorb = use_spec and not is_dfly
        pending = None
        decoder.embed_sink = {} if need_absorb else None

        def absorb_pending(first_next, final=False):
            """Absorb the previous step's rows; `first_next` [1, H] is the
            embedding of the token after its last row (None: look it up). A
            padded step reuses embeddings only as the prompt's final step (its
            padding repeats the placeholder token)."""
            p_pos, p_w, p_post, p_emb, p_real = pending
            nxt = ids[p_pos + 1:p_pos + 1 + p_w]
            nxt = nxt + [ids[-1]] * (p_w - len(nxt))
            reuse = p_emb is not None and first_next is not None and (p_real == p_w or final)
            embeds = torch.cat([p_emb[1:], first_next], dim=0) if reuse else None
            decoder._draft_positions[row] = p_pos
            decoder.draft_absorb(torch.tensor([nxt], dtype=torch.int64, device=dev), p_post,
                                 request_rows=[row], logits="none", next_embeds=embeds)

        t_prefill = time.monotonic()
        steps, index = [], 0
        while index < len(plan):
            # c8 regress-fix: overlapped chunk pairs on any row (the batched loop puts
            # long prompts on rows 1+; row 0 only lost them — prefill -30%)
            pair = (getattr(ctx, "overlap", False) and not params.get("nll")
                    and (row == 0 or getattr(ctx, "overlap_any_row", True))
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
            need_hidden = need_absorb or (use_spec and is_dfly and last)
            if params.get("nll"):
                # consult18 stage 0b: every row's full-vocabulary logits (a probe,
                # not a serving path), scored against the next prompt token.
                decoder.logits_root_host = True       # rank 0's host only (see decoder)
                try:
                    logits = decoder.forward(t, request_rows=[row], call_shape=ctx.shapes[w],
                                             logits_rows="all")
                finally:
                    decoder.logits_root_host = False
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
                            logits_rows=real - 1 if last else "none", row=row)
                        torch.cuda.synchronize(dev)
                elif need_hidden:
                    logits, post = decoder.forward_overlapped(
                        t, half_shape=ctx.shapes[full], logits_rows=real - 1 if last else "none",
                        return_hidden=True, row=row)
                else:
                    logits = decoder.forward_overlapped(
                        t, half_shape=ctx.shapes[full], logits_rows=real - 1 if last else "none",
                        row=row)
            elif need_hidden:
                logits, post, _ = decoder.forward(t, request_rows=[row], call_shape=ctx.shapes[w],
                                                  logits_rows=real - 1 if last else "none",
                                                  return_hidden=True)
            else:
                logits = decoder.forward(t, request_rows=[row], call_shape=ctx.shapes[w],
                                         logits_rows=real - 1 if last else "none")
            decoder.append([row], real)
            if need_absorb:
                # prompt absorption: draft rows pos .. pos+w-1 with the target's
                # post-norm hidden and each row's next prompt token (the last
                # prompt row's is a placeholder: round 1 re-absorbs it with the
                # first generated token; padded rows are rewritten before read).
                # consult18 P5: one step late, reusing the target's all-reduced
                # embeddings — row t's next token is the target's row t+1.
                emb = decoder.embed_sink.pop(id(t), None)
                decoder.embed_sink.clear()
                if pending is not None:
                    absorb_pending(emb[:1] if emb is not None else None)
                pending = (pos, w, post, emb, real)
                last_hidden = post[0, real - 1].clone()
            elif use_spec and is_dfly and last:
                # DFly2 has no prompt absorption: tap only the final prompt hidden
                # (the last real token's target top-k was already captured above).
                last_hidden = post[0, real - 1].clone()
            pos += real
            if chunked and index < len(steps) - 1:
                # c8 L17: pause between chunks; other forwards (the loop's rounds,
                # other admissions) run meanwhile and may reset the embedding sink
                sink, decoder.embed_sink = decoder.embed_sink, None
                yield pos
                decoder.embed_sink = sink
        if need_absorb and pending is not None:
            last_emb = pending[3]
            absorb_pending(last_emb[-1:] if last_emb is not None else None, final=True)
        decoder.embed_sink = None
        if n >= 4096:
            # the engine's prefill wall (first chunk to the last chunk's enqueue + absorb;
            # paused intervals included when chunked) — TTFT minus HTTP/template/queue
            torch.cuda.synchronize(dev)
            dt = time.monotonic() - t_prefill
            trace(ctx.rank, f"prefill {n} tokens on row {row}: {dt:.2f} s ({n / max(dt, 1e-9):.0f} tok/s), "
                            f"{sum(1 for st in steps if st[2])} overlapped pairs of {len(steps)} steps"
                            + (", chunked" if chunked else ""))
        ctx.last_nll = (nll_sum, nll_count) if params.get("nll") and ctx.rank == 0 else None
    out = []
    cancel = getattr(ctx, "cancel", None)      # rank 0: set when the client has left

    def sample(logits):
        if ctx.rank == 0:
            nxt, done, finish = ctx.sampler.step(logits[:, -1, :], params, len(out), max_new)
            if not done and cancel is not None and cancel.is_set():
                done, finish = True, "cancelled"
            elif not done and not admit_only and preempt is not None and preempt():
                done, finish = True, "preempt"
        else:
            nxt = done = finish = None
        obj = [nxt, done, finish]
        dist.broadcast_object_list(obj, src=0)
        return obj

    nxt, done, finish = sample(logits)
    out.append(nxt)
    _stream_push(ctx, out)
    if done and finish == "preempt" and use_spec and last_hidden is not None:
        # preempted at its first token, before any round: the batched rounds take
        # over its draft state like an admission's (prompt absorbed, last hidden)
        ctx.solo_hand = {"P": n, "bonus": int(nxt), "c": n - 1, "tokens": [int(nxt)],
                         "hidden": last_hidden.reshape(1, -1)}
    if admit_only:
        decoder.trace.clear()          # the wiring trace must not grow per request
        # batched speculation: the prompt's last post-norm hidden (every rank)
        ctx.admit_hidden = last_hidden if use_spec else None
        return out, (finish if done else None)
    yielding = [False]          # set on every rank by the 64-token check
    if not done and use_spec and last_hidden is not None:
        if not hasattr(ctx, "eos_all"):
            box = [ctx.sampler.eos if ctx.rank == 0 else None]
            dist.broadcast_object_list(box, src=0)
            ctx.eos_all = box[0]

        def emit(tok):
            # the sampler's stop rules, decided identically on every rank
            produced = len(out)
            out.append(int(tok))
            _stream_push(ctx, out)
            if tok == ctx.eos_all and not params.get("ignore_eos"):
                return "stop"
            if produced + 1 >= max_new:
                return "length"
            if produced % CANCEL_EVERY == CANCEL_EVERY - 1:
                # every rank reaches this check at the same token count
                box = ([cancel is not None and cancel.is_set(), preempt is not None and preempt()]
                       if ctx.rank == 0 else [None, None])
                dist.broadcast_object_list(box, src=0)
                if box[0]:
                    return "cancelled"
                if box[1]:
                    yielding[0] = True     # hand over at the next round boundary
            return None

        finish, rest = spec.generate(start=n, bonus=nxt, hidden=last_hidden, emit=emit,
                                     temperature=spec_temp,
                                     top_k=spec_top_k if 0 < spec_top_k else 0,
                                     top_p=spec_top_p if spec_top_p < 1.0 else 1.0,
                                     should_yield=lambda: yielding[0])
        if finish is None and yielding[0]:
            finish = "preempt"
        done = finish is not None
    while not done:
        t = torch.tensor([[out[-1]]], dtype=torch.int64, device=dev)
        logits = decoder.decode_step(t, request_rows=[row], call_shape=decode_shape)
        decoder.complete([row], t)
        if WATCH.finding is not None and "generated_tokens" not in WATCH.finding:
            WATCH.finding["generated_tokens"] = len(out)   # decode steps completed before it
        nxt, done, finish = sample(logits)
        out.append(nxt)
        _stream_push(ctx, out)
    finding = WATCH.take()
    if finding is not None:
        finding.setdefault("generated_tokens", 0)          # 0 = during the prompt
        finding["prompt_tokens"] = len(ids)
        print(f"[h4 r{ctx.rank}] FINITE-WATCH " + json.dumps(finding), flush=True)
    if finish != "preempt":
        decoder.release([row])
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
    if use_spec:
        trace(ctx.rank, f"spec: {spec.stats()}")
    graphs = getattr(decoder, "decode_graphs", None)
    if graphs is not None and (graphs.replays or graphs.captures):
        trace(ctx.rank, f"decode graphs: {graphs.stats()}, sync-free reruns "
                        f"{getattr(decoder, 'sync_free_reruns', 0)}")
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
        timeout = CLIENT_SOCKET_TIMEOUT_S        # StreamRequestHandler: socket deadline

        def log_message(self, *args):
            pass

        def setup(self):
            super().setup()
            # a peer that vanished without closing is found by keepalive probes;
            # _client_gone() then sees the socket error and cancels its request
            try:
                self.connection.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                for name, value in TCP_KEEPALIVE:
                    if hasattr(socket, name):
                        self.connection.setsockopt(socket.IPPROTO_TCP, getattr(socket, name), value)
            except OSError:
                pass

        def handle(self):
            try:
                super().handle()
            except (ConnectionResetError, BrokenPipeError, TimeoutError):
                pass                     # the client dropped its connection

        def _send(self, status, payload, content_type="application/json"):
            data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _client_gone(self):
            """True when the client has closed its connection (EOF on the socket)."""
            try:
                readable, _, _ = select.select([self.connection], [], [], 0)
                return bool(readable) and self.connection.recv(1, socket.MSG_PEEK) == b""
            except (OSError, ValueError):
                return True

        def _enqueue(self, item):
            """Appends `item` to the main loop's FIFO queue (no request is refused)."""
            item["cancel"] = threading.Event()
            item["admitted"] = time.monotonic()
            with ctx.lock:
                ctx.waiting += 1
            ctx.q.put(item)

        def _wait(self, item, limit):
            """Waits for the main loop to finish `item`: "done", "gone" (the client
            left: the item is cancelled) or "timeout" (`limit` seconds after the
            generation started; the item is cancelled)."""
            while not item["ev"].wait(timeout=POLL_S):
                if self._client_gone():
                    item["cancel"].set()
                    self.close_connection = True
                    return "gone"
                started = item.get("started")
                if started is not None and time.monotonic() - started > limit:
                    item["cancel"].set()
                    return "timeout"
            return "done"

        def _stream_events(self, item, ids, include_usage, tools=None, think_open=False,
                           continuous=False):
            """OpenAI `chat.completion.chunk` SSE over chunked HTTP/1.1: a role
            chunk, `delta.reasoning_content` / `delta.content` per decoded piece,
            a finish chunk, the usage chunk when `stream_options.include_usage`,
            then `[DONE]`. While the request waits in the queue or in prefill, an
            SSE comment goes out every HEARTBEAT_S seconds. A client that goes
            away cancels the request (dropped if queued, stopped if running)."""
            created = int(time.time())
            base = {"id": f"chatcmpl-h4-{created}", "object": "chat.completion.chunk",
                    "created": created, "model": ctx.served_names[0]}
            alive = [True]

            def write(payload, comment=False):
                if not alive[0]:
                    return
                data = b": " + payload + b"\n\n" if comment else b"data: " + (
                    payload if isinstance(payload, bytes) else json.dumps(payload).encode()) + b"\n\n"
                try:
                    self.wfile.write(b"%X\r\n%s\r\n" % (len(data), data))
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    alive[0] = False

            received = [0]          # generated token ids handed over so far

            def delta(content, kind="content"):
                field = "reasoning_content" if kind == "reasoning" else "content"
                chunk = {**base, "choices": [{"index": 0, "delta": {field: content},
                                              "finish_reason": None}]}
                if continuous:
                    chunk["usage"] = {"prompt_tokens": len(ids), "completion_tokens": received[0],
                                      "total_tokens": len(ids) + received[0]}
                write(chunk)

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            write({**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""},
                                        "finish_reason": None}]})
            # reasoning / content / held tool-call block, as OutputSplitter decides
            split = OutputSplitter(ctx, think_open, ctx.toolcalls_token if tools else None)
            q = item["stream_q"]
            last = beat = time.monotonic()   # last serve-loop event, last write
            while True:
                try:
                    kind, payload = q.get(timeout=POLL_S)
                except queue.Empty:
                    started, now = item.get("started"), time.monotonic()
                    if self._client_gone():
                        alive[0] = False
                    elif now - beat >= HEARTBEAT_S:
                        write(b"generating" if started is not None else b"queued", comment=True)
                        beat = now
                    if started is not None and now - max(started, last) > SERVE_LOOP_TIMEOUT_S:
                        item.setdefault("error", f"the serve loop did not answer in {SERVE_LOOP_TIMEOUT_S} s")
                        item["cancel"].set()
                        break
                    if not alive[0]:
                        item["cancel"].set()
                        self.close_connection = True
                        return
                    continue
                last = beat = time.monotonic()
                if kind == "tokens":
                    received[0] += len(payload)
                    for part, text in split.feed(payload):
                        delta(text, part)
                    if not alive[0]:
                        item["cancel"].set()
                        self.close_connection = True
                        return
                else:
                    break
            if item.get("error"):
                ctx.failure = item["error"]
                write({"error": {"message": item["error"][:400], "type": "server_error"}})
            else:
                out, finish = item["result"]
                for part, text in split.finish():
                    delta(text, part)
                if split.held is not None:
                    block = ctx.tok.decode(split.held, skip_special_tokens=True)
                    _, calls = parse_tool_calls(block, tools)
                    if calls:
                        for index, call in enumerate(calls):
                            write({**base, "choices": [{"index": 0, "delta": {"tool_calls": [
                                {"index": index, **call}]}, "finish_reason": None}]})
                        finish = "tool_calls"
                    elif block:
                        delta(block)         # not a well-formed block: it was content
                write({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": finish or "stop"}]})
                if include_usage:
                    write({**base, "choices": [], "usage": {
                        "prompt_tokens": len(ids), "completion_tokens": len(out),
                        "total_tokens": len(ids) + len(out),
                        "completion_tokens_details": {"reasoning_tokens": len(split.reasoning_ids)}}})
            write(b"[DONE]")
            if alive[0]:
                try:
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass

        def do_GET(self):
            if self.path == "/health":
                live = getattr(ctx, "liveness", None)
                stall = live.stalled() if live is not None else None
                if ctx.failure:
                    self._send(503, {"status": "failed", "detail": str(ctx.failure)[:200]})
                elif stall is not None:
                    self._send(503, {"status": "stalled", "detail": live.describe()})
                elif live is not None:
                    self._send(200, {"status": "ok", "plans": live.plans})   # ticks while idle
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
            if self.path == "/dev/exec" and os.environ.get("HY4_DEV_EXEC") == "1":
                # consult18 P2 dev loop (HY4_DEV_EXEC=1 boots only; production
                # never sets it): every rank execs one script from /state/dev/
                # (root-owned host directory) inside the live serve; returns
                # each rank's result/traceback. Serialized with requests.
                try:
                    path = json.loads(raw).get("path") or ""
                    if not (path.startswith("/state/dev/") and ".." not in path
                            and os.path.isfile(path)):
                        raise ValueError("path must be a file under /state/dev/")
                except (ValueError, TypeError, AttributeError) as error:
                    self._send(400, {"error": str(error)})
                    return
                item = {"op": "exec", "path": path, "ev": threading.Event()}
                self._enqueue(item)
                state = self._wait(item, 3600)
                if state == "timeout":
                    self._send(504, {"error": "exec did not finish in 3600 s"})
                if state != "done":
                    return
                self._send(200, {"results": item.get("result")})
                return
            if self.path != "/v1/chat/completions":
                self._send(404, {"error": {"message": f"no route {self.path}", "type": "not_found"}})
                return
            try:
                body = json.loads(raw)
                if not isinstance(body, dict):
                    raise ValueError("the request body must be a JSON object")
                stream = bool(body.get("stream", False))
                include_usage = bool((body.get("stream_options") or {}).get("include_usage", False))
                # vLLM's continuous_usage_stats: cumulative usage on every chunk
                continuous = include_usage and bool(
                    (body.get("stream_options") or {}).get("continuous_usage_stats", False))
                if body.get("n", 1) != 1:
                    raise ValueError("h4 direct serving returns one choice")
                tools = body.get("tools") or None
                with ctx.tok_lock:              # one tokenizer, many request threads
                    if "messages" in body:
                        reject_multimodal_messages(body["messages"])
                        text = ctx.tok.apply_chat_template(
                            normalize_tool_history(body["messages"]), tokenize=False,
                            add_generation_prompt=True, tools=tools)
                    elif "prompt" in body:
                        text = str(body["prompt"])
                    else:
                        raise ValueError("the body needs `messages` or `prompt`")
                    ids = ctx.tok.encode(text, add_special_tokens=False)
                # the prompt leaves a think block open (the template's thinking mode)
                think_open = text.endswith(THINK_OPEN)
                if not ids:
                    raise ValueError("empty prompt")
                limit = body.get("max_completion_tokens", body.get("max_tokens"))
                # A request without max_tokens decodes up to --default-max-tokens, or
                # to the context window when that flag is 0 (clamped below).
                # Explicit client values win.
                limit = (ctx.default_max_new or ctx.max_context) if limit is None else int(limit)
                if limit < 1:
                    raise ValueError("the output limit must be positive")
                if len(ids) > ctx.row_capacity:
                    raise ValueError(f"prompt longer than the row capacity ({ctx.row_capacity})")
                if len(ids) + limit > ctx.max_context:
                    limit = ctx.max_context - len(ids)
                    if limit < 1:
                        raise ValueError(f"prompt exceeds the served context ({ctx.max_context})")
                for name in ("temperature", "top_p", "top_k"):
                    # every rank converts these: a bad value must fail here, not in the loop
                    value = body.get(name)
                    if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                              or not math.isfinite(value)):
                        raise ValueError(f"`{name}` must be a finite number")
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
            if stream:
                item["stream_q"] = queue.Queue()
            self._enqueue(item)
            if stream:
                self._stream_events(item, ids, include_usage, tools, think_open, continuous)
                return
            state = self._wait(item, SERVE_LOOP_TIMEOUT_S)
            if state == "gone":
                return
            if state == "timeout":
                self._send(504, {"error": {"message": f"the serve loop did not answer in "
                                                      f"{SERVE_LOOP_TIMEOUT_S} s", "type": "server_error"}})
                return
            if item.get("error"):
                ctx.failure = item["error"]
                self._send(500, {"error": {"message": item["error"][:400], "type": "server_error"}})
                return
            out, finish = item["result"]
            nll = item.get("nll")
            split = OutputSplitter(ctx, think_open, ctx.toolcalls_token if tools else None)
            split.feed(out)
            split.finish()
            reasoning = ctx.tok.decode(split.reasoning_ids, skip_special_tokens=True)
            content = ctx.tok.decode(split.content_ids, skip_special_tokens=True)
            calls = []
            if split.held is not None:
                block = ctx.tok.decode(split.held, skip_special_tokens=True)
                _, calls = parse_tool_calls(block, tools)
                if not calls:
                    content += block              # not a well-formed block: it was content
            message = {"role": "assistant", "content": content}
            if reasoning:
                message["reasoning_content"] = reasoning
            if calls:
                message["content"] = content or None
                message["tool_calls"] = calls
                finish = "tool_calls"
            created = int(time.time())
            self._send(200, {
                "id": f"chatcmpl-h4-{created}", "object": "chat.completion", "created": created,
                "model": ctx.served_names[0],
                "choices": [{"index": 0, "message": message,
                             "finish_reason": finish or "stop"}],
                "usage": {"prompt_tokens": len(ids), "completion_tokens": len(out),
                          "total_tokens": len(ids) + len(out),
                          "completion_tokens_details": {"reasoning_tokens": len(split.reasoning_ids)},
                          **({} if nll is None else {"nll_sum": nll[0], "nll_tokens": nll[1],
                                                     "nll_mean": nll[0] / max(nll[1], 1)})},
            })

    host, _, port = ctx.bind.rpartition(":")
    # a burst of concurrent clients must not overflow the listen backlog (default 5)
    listener = type("Listener", (ThreadingHTTPServer,), {"address_family": socket.AF_INET,
                                                         "request_queue_size": 128})
    server = listener((host, int(port)), Handler)
    server.daemon_threads = True
    ctx.http = server
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def batch_shape_of(size):
    from ignition.recipes.hy4.wiring import batch_shape

    return batch_shape(size)


def _unfiltered(params):
    """Sampling with no top-k / top-p filter (the device-sampled batched step)."""
    top_k = int(params.get("top_k", -1) or -1)
    top_p = float(params.get("top_p", 1.0) or 1.0)
    return top_k <= 0 and top_p >= 1.0


def _stops(sampler, ids, params_list, produced, max_new):
    """The Sampler's stop rules for already-sampled ids: [(token, done, finish)]."""
    out = []
    for i, tok in enumerate(ids):
        if tok == sampler.eos and not params_list[i].get("ignore_eos"):
            out.append((tok, True, "stop"))
        elif produced[i] + 1 >= max_new[i]:
            out.append((tok, True, "length"))
        else:
            out.append((tok, False, None))
    return out


class PlanWaitTimeout(RuntimeError):
    """The per-iteration round-plan broadcast failed or ran out its gloo deadline;
    the message names the rank, what it awaited, since when and the last plan."""


def _wall(ts):
    return time.strftime("%H:%M:%S", time.localtime(ts)) if ts else "-"


class PlanLiveness:
    """Every rank: times the per-iteration plan broadcast. Rank 0 times its own
    planning and the broadcast send, a follower its wait for rank 0's plan. A
    watchdog thread logs a named LIVENESS line once a phase runs past `limit`
    seconds and again every `limit` seconds while it lasts; rank 0 then also dumps
    every thread's stack (so the log names where it is stuck)."""

    def __init__(self, rank, limit=None, *, watch=True):
        self.rank = rank
        self.limit = LIVENESS_S if limit is None else float(limit)
        self.lock = threading.Lock()
        self.phase = None          # what this rank is doing around the broadcast, or None
        self.since = None          # time.monotonic() when the phase began
        self.since_wall = None
        self.plans = 0             # plans broadcast / received so far
        self.last_kind = None
        self.last_wall = None
        self.warned = 0.0
        if watch:
            threading.Thread(target=self._watch, name="h4-liveness", daemon=True).start()

    def enter(self, phase):
        with self.lock:
            self.phase, self.since, self.since_wall = phase, time.monotonic(), time.time()
            self.warned = 0.0

    def leave(self, kind):
        with self.lock:
            self.phase = self.since = self.since_wall = None
            self.plans += 1
            self.last_kind, self.last_wall = kind, time.time()

    def stalled(self):
        """(phase, seconds) when the current phase has run past the limit, else None."""
        with self.lock:
            if self.phase is None:
                return None
            age = time.monotonic() - self.since
            return (self.phase, age) if age > self.limit else None

    def describe(self):
        with self.lock:
            phase, since, since_wall = self.phase, self.since, self.since_wall
            plans, kind, last = self.plans, self.last_kind, self.last_wall
        age = time.monotonic() - since if since is not None else 0.0
        prior = f"last plan '{kind}' at {_wall(last)}" if kind else "no plan yet"
        return (f"rank {self.rank}: {phase or 'between plans'} for {age:.0f} s "
                f"(since {_wall(since_wall)}; {plans} plans so far, {prior})")

    def _watch(self):
        import faulthandler

        while True:
            time.sleep(max(1.0, min(30.0, self.limit / 4)))
            with self.lock:
                if self.phase is None:
                    continue
                age = time.monotonic() - self.since
                due = age > self.limit and age - self.warned >= self.limit
                if due:
                    self.warned = age
            if not due:
                continue
            if self.rank == 0:
                trace(0, f"LIVENESS: {self.describe()}; rank 0's thread stacks follow")
                faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
            else:
                trace(self.rank, f"LIVENESS: {self.describe()}; rank 0 has not broadcast "
                                 "(see its LIVENESS stack dump)")


class _Req:
    """Rank 0's record of one request in the batched loop."""

    def __init__(self, item, row, out=None, sent=0):
        self.item, self.row = item, row
        self.out = list(out or [])
        self.sent = sent
        self.params, self.max_new = item["params"], item["max_new"]
        self.done = False


class BatchLoop:
    """The serving loop, every rank in lockstep (continuous batching).

    Each iteration rank 0 builds one plan and broadcasts it; every rank runs it:
    - "exec": a dev script (only while no request is active);
    - "solo": a request that arrived to an idle loop with nothing queued runs
      the one-row path on row 0 (speculative decoding); when another request
      queues it hands over at a round boundary ("preempt") and joins the batch;
    - "batch": release finished / cancelled rows, admit queued requests on free
      cache rows (prefill + first token), then one batched decode step over
      every active row with a known next token (DecodeGraphs.run_batch, buckets
      1/2/4/8, pad rows on free cache rows). Rank 0 samples every row, streams,
      and finishes requests; their rows are released in the next plan.
    """

    def __init__(self, ctx):
        self.ctx = ctx
        self.rank = ctx.rank
        self.rows = int(getattr(ctx, "cache_rows", 1) or 1)
        graphs = getattr(ctx.decoder, "decode_graphs", None)
        buckets = tuple(getattr(ctx, "batch_rows", ()) or ())
        self.max_batch = min(max((1, *buckets)), self.rows) if graphs is not None else 1
        self.active = {}            # rank 0: row -> _Req
        self.deferred = []          # rank 0: exec items waiting for an idle loop
        self.pending_release = []   # rank 0: rows finished in the last step
        self.steps = 0
        self.lookahead = True       # enqueue step k+1 before consuming step k
        self.inflight = None        # every rank: the enqueued, not yet consumed step
        self._snaps = None          # device/pinned snapshot ring of enqueued steps
        self._snap_at = 0
        self.rollbacks = 0
        self.bspec = True           # batched speculative sampling (k = 1) when every row can
        self.bs_rows = {}           # every rank: row -> {"warm": a plain warm-up round first}
        self.bs_la = True           # c8 L5b: enqueue bspec round k+1 before consuming round k
        self.bspec_k2 = True        # c8 L13: two chained drafts per row at <= 4 rows
        self.bspec_k3 = True        # c8 L13b: three chained drafts per row at <= 3 rows
        self.bspec_multi_cap = int(os.environ.get("HY4_BSPEC_MULTI_CAP") or 12)
        self.solo_bspec = os.environ.get("HY4_SOLO_BSPEC", "0") == "1"
        self.prefilling = {}        # c8 L17: row -> a long admission's paused prefill (every rank)
        self.chunked_admit = True   # c8 L17: long prompts prefill between decode rounds
        self.prefill_long = 8192    # ... from this many prompt tokens
        self.prefill_every = int(os.environ.get("HY4_PREFILL_EVERY") or 24)
        self._since_chunk = 0
        self._bs_inv = None         # the staged per-slot 1/T (re-staged when it changes)
        self._bs_rows_t = (None, None)
        self.bspec_rounds = self.bspec_accepted = self.bspec_slots = 0
        self.liveness = ctx.liveness = PlanLiveness(self.rank)

    # ---- rank 0 --------------------------------------------------------
    def _pop(self, block, timeout=None):
        """The next live queued item; None when none arrives (non-blocking, or
        nothing within `timeout` seconds). Cancelled items are dropped here."""
        ctx = self.ctx
        while True:
            try:
                item = ctx.q.get(timeout=timeout) if block else ctx.q.get_nowait()
            except queue.Empty:
                return None
            with ctx.lock:
                ctx.waiting -= 1
            waited = time.monotonic() - item["admitted"]
            if item["cancel"].is_set():
                # its client left while it waited: never broadcast, never run
                trace(0, f"dropped a queued request after {waited:.0f} s: the client left")
                continue
            return item

    def _row_cap(self, row):
        caps = getattr(self.ctx, "row_caps", None)
        return int(caps[row]) if caps else int(self.ctx.row_capacity)

    def _needs_big(self, item):
        """c8 L6: a prompt (+ 16k of room) past the narrowest row's extent needs the
        widest row (row 0)."""
        caps = getattr(self.ctx, "row_caps", None)
        if not caps or min(caps) == max(caps):
            return False
        return len(item.get("tokens") or ()) + 16384 > min(caps[1:] or caps)

    def _start(self, item, row):
        # c8 L6: the output limit stops at the row's extent (less the speculative margin)
        room = self._row_cap(row) - 8 - len(item.get("tokens") or ())
        if item.get("max_new") is not None and item["max_new"] > room:
            item["max_new"] = max(1, room)
        item["started"] = time.monotonic()
        trace(0, f"request start: {len(item.get('tokens') or ())} prompt tokens on row {row}, "
                 f"{item['started'] - item['admitted']:.1f} s queued, {self.ctx.waiting} waiting, "
                 f"{len(self.active)} active")

    def _finish(self, req, finish):
        req.done = True
        item = req.item
        if finish == "cancelled":
            trace(0, f"request stopped after {len(req.out)} tokens: the client left")
        self._push(req)
        item["result"] = (req.out, finish)
        item["nll"] = None
        item["ev"].set()
        if item.get("stream_q") is not None:
            item["stream_q"].put(("end", None))

    @staticmethod
    def _push(req):
        q = req.item.get("stream_q")
        if q is not None and len(req.out) > req.sent:
            q.put(("tokens", list(req.out[req.sent:])))
            req.sent = len(req.out)

    def _plan(self):
        ctx = self.ctx
        release, self.pending_release = self.pending_release, []
        for row, req in list(self.active.items()):
            if req.item["cancel"].is_set():
                del self.active[row]
                self._finish(req, "cancelled")
                release.append(row)
        first = None
        # c8 L17: a long admission whose client left stops prefilling
        abort = [r for r, st in self.prefilling.items()
                 if st.get("item") is not None and st["item"]["cancel"].is_set()]
        for r in abort:
            self._finish(_Req(self.prefilling[r]["item"], r, [], sent=0), "cancelled")
        if not self.active and self.inflight is not None:
            # consume the last enqueued step before idling
            return {"kind": "batch", "release": release, "admit": [], "carry": [],
                    "carry_params": [], "admit_long": [], "abort": abort,
                    "prefill": False}, []
        waiting = [i for i in getattr(self, "big_waiting", None) or [] if not i["cancel"].is_set()]
        for i in getattr(self, "big_waiting", None) or []:
            if i["cancel"].is_set():
                trace(0, "dropped a queued request: the client left (it waited for the widest row)")
                i["result"] = ([], "cancelled")
                i["nll"] = None
                i["ev"].set()
                if i.get("stream_q") is not None:
                    i["stream_q"].put(("end", None))
        self.big_waiting = waiting
        if not self.active and not (set(self.prefilling) - set(abort)):
            if self.big_waiting:
                item = self.big_waiting.pop(0)        # c8 L6: row 0 is free now
            elif self.deferred:
                item = self.deferred.pop(0)
            else:
                # never block here: the followers wait for this plan in gloo (30 min
                # deadline), so a quiet serve broadcasts an idle tick instead
                item = self._pop(block=True, timeout=IDLE_TICK_S)
                if item is None:
                    return {"kind": "idle", "release": release, "abort": abort}, []
            if item.get("op") == "exec":
                return {"kind": "exec", "path": item["path"], "release": release,
                        "abort": abort}, [item]
            if ((ctx.q.empty() or self.max_batch <= 1) and not self._solo_batched(item)
                    and not self._is_long(item)):
                extents = None
                if self._dyn():
                    # c8 L6b: a lone request takes the whole pool (shrunk on demand later)
                    extents = self._dyn_solo(release)
                self._start(item, 0)
                return ({"kind": "solo", "release": release, "tokens": item["tokens"],
                         "max_new": item["max_new"], "params": item["params"],
                         "extents": extents, "abort": abort}, [item])
            first = item
        free = [r for r in range(self.rows) if r not in self.active and r not in self.prefilling]
        # c8 L6: the widest row (0) is kept for the prompts that need it; a long
        # prompt waits for it (deferred) while shorter ones keep being admitted
        free.sort(key=lambda r: (r == 0, r))
        big_waiting = getattr(self, "big_waiting", None)
        if big_waiting is None:
            big_waiting = self.big_waiting = []
        admits = [first] if first is not None else []
        if 0 in free and big_waiting and first is None:
            admits.append(big_waiting.pop(0))
        mem_waiting, self.deferred_mem = getattr(self, "deferred_mem", []), []
        admits = [i for i in mem_waiting if not i["cancel"].is_set()] + admits
        while (len(admits) < len(free)
               and len(self.active) + len(self.prefilling) + len(admits) < self.max_batch):
            item = self._pop(block=False)
            if item is None:
                break
            if item.get("op") == "exec":
                self.deferred.append(item)
                continue
            admits.append(item)
        rows, placed = [], []
        pool = list(free)
        for item in admits:
            if self._needs_big(item):
                if 0 in pool:
                    pool.remove(0)
                    rows.append(0)
                    placed.append(item)
                else:
                    big_waiting.append(item)
                continue
            if pool:
                rows.append(pool.pop(0))
                placed.append(item)
        extents = None
        if self._dyn():
            placed, rows, extents = self._dyn_place(placed, rows, release)
        for item, row in zip(placed, rows):
            self._start(item, row)
        # c8 L17: long prompts prefill chunk by chunk between the loop's rounds
        longs = [(item, row) for item, row in zip(placed, rows) if self._is_long(item)]
        shorts = [(item, row) for item, row in zip(placed, rows) if not self._is_long(item)]
        admits = [item for item, _ in shorts]
        live = [r for r in self.prefilling if r not in abort] or longs
        self._since_chunk += 1
        prefill = bool(live) and (not self.active or self._since_chunk >= self.prefill_every)
        if prefill:
            self._since_chunk = 0
        carry = sorted(self.active.items())
        inflight = set(self.inflight["rows"]) if self.inflight is not None else set()
        plan = {"kind": "batch", "release": release, "abort": abort, "prefill": prefill,
                "extents": extents,
                "admit_long": [(row, item["tokens"], item["max_new"], item["params"])
                               for item, row in longs],
                "admit": [(row, item["tokens"], item["max_new"], item["params"])
                          for item, row in shorts],
                # an active row's next input: the in-flight step's device output
                # (None) or its last sampled token (not in the in-flight step)
                "carry": [(row, None if row in inflight else req.out[-1]) for row, req in carry],
                "carry_params": [(float(req.params.get("temperature", 0.9) or 0.0),
                                  _unfiltered(req.params)) for _, req in carry]}
        return plan, admits + [item for item, _ in longs]

    # -- c8 L6b: dynamic row extents (rank 0 plans, every rank applies) --------
    def _dyn(self):
        return bool(getattr(self.ctx, "dyn_extents", False))

    def _need_local(self, tokens):
        """Local slots for `tokens` positions plus 8k of room, aligned to the
        indexer's geometric width buckets (1,024-slot units up to 8,192, then
        four per octave) so `DecodeGraphs.index_width` stays on boot-primed
        buckets instead of clamping to off-grid widths mid-request."""
        c = self.ctx.decoder.cache
        n = -(-(int(tokens) + 8192) // c.world)
        w = -(-n // 1024) * 1024
        if w > 8192:
            octave = 1 << (w.bit_length() - 1)
            step = octave // 4
            w = -(-w // step) * step
        return min(w, int(c.cap_limit))

    def _used_local(self, row):
        """Local slots a row has written (its length, in-flight rounds and pads included)."""
        c = self.ctx.decoder.cache
        L = int(self.ctx.decoder._row_lengths.get(row, 0))
        return min(-(-(L + 64) // c.world), c.cap_of(row))

    def _first_fit(self, ext, n, exclude=()):
        c = self.ctx.decoder.cache
        total = self.rows * c.local_capacity
        at = 0
        for b, e in sorted((b, b + cap) for r, (b, cap) in ext.items() if cap > 0 and r not in exclude):
            if b - at >= n:
                return at
            at = max(at, e)
        return at if total - at >= n else None

    def _compact(self, ext, moves, shrink=False):
        """Pack the extents left in base order (each keeps its cap, or shrinks to its
        need with `shrink`); moves get (row, old, new, used)."""
        at = 0
        for r, (b, cap) in sorted(ext.items(), key=lambda kv: kv[1][0]):
            if cap <= 0:
                continue
            if shrink:
                L = int(self.ctx.decoder._row_lengths.get(r, 0))
                cap = max(self._need_local(L), -(-self._used_local(r) // 1024) * 1024)
            if b != at:
                moves.append((r, b, at, self._used_local(r)))
            ext[r] = (at, cap)
            at += cap

    def _room(self, ext, moves, n, exclude=()):
        """A base for n slots: first fit, else after compaction, else after shrinking
        every extent to its need and compacting; None when the pool cannot hold it."""
        base = self._first_fit(ext, n, exclude)
        if base is None:
            self._compact(ext, moves)
            base = self._first_fit(ext, n, exclude)
        if base is None:
            self._compact(ext, moves, shrink=True)
            base = self._first_fit(ext, n, exclude)
        return base

    def _dyn_solo(self, release):
        c = self.ctx.decoder.cache
        ext = {r: (0, 0) for r in range(self.rows)}
        ext[0] = (0, min(int(c.cap_limit), self.rows * c.local_capacity))
        return {"moves": [], "bases": [ext[r][0] for r in range(self.rows)],
                "caps": [ext[r][1] for r in range(self.rows)]}

    def _dyn_place(self, placed, rows, release):
        """Grow the active rows that near their extent's end, then grant each
        admission its extent; an admission that does not fit waits (front of the
        queue), an active row that cannot grow finishes ("length")."""
        c = self.ctx.decoder.cache
        ext = {r: (c.base_of(r), c.cap_of(r)) for r in range(self.rows)}
        live = set(self.active) | set(self.prefilling)
        for r in range(self.rows):
            if r not in live or r in release:
                ext[r] = (ext[r][0], 0)
        moves, changed = [], any(ext[r] != (c.base_of(r), c.cap_of(r)) for r in range(self.rows))
        for r in sorted(self.active):
            if r in release:
                continue
            b, cap = ext[r]
            L = int(self.ctx.decoder._row_lengths.get(r, 0))
            if -(-(L + 512) // c.world) <= cap:
                continue
            target = min(max(2 * cap, self._need_local(L)), int(c.cap_limit))
            if target <= cap:
                continue
            changed = True
            others = sorted((ob, ob + oc) for o, (ob, oc) in ext.items() if o != r and oc > 0)
            if (b + target <= self.rows * c.local_capacity
                    and not any(ob < b + target and oe > b for ob, oe in others)):
                ext[r] = (b, target)                      # grows in place
                continue
            ext[r] = (b, 0)
            nb = self._room(ext, moves, target, exclude={r})
            if nb is None:
                ext[r] = (b, cap)
                req = self.active.pop(r)
                trace(0, f"request on row {r} reached the pool's end at {L} tokens")
                self._finish(req, "length")
                release.append(r)
                ext[r] = (b, 0)
                continue
            moves.append((r, b, nb, self._used_local(r)))
            ext[r] = (nb, target)
        force = int(getattr(self, "force_relocate_every", 0) or 0)
        movable = [r for r in sorted(self.active) if r not in release]
        if force and movable:
            # dev gate: move one active row to the LAST free gap every `force` plans
            # (greedy outputs must not change)
            self._reloc_tick = getattr(self, "_reloc_tick", 0) + 1
            if self._reloc_tick % force == 0:
                r = movable[(self._reloc_tick // force) % len(movable)]
                b, cap = ext[r]
                ext[r] = (b, 0)
                total = self.rows * c.local_capacity
                spans = sorted((ob, ob + oc) for o, (ob, oc) in ext.items() if oc > 0)
                gaps, at = [], 0
                for ob, oe in spans:
                    if ob - at >= cap:
                        gaps.append(at)
                    at = max(at, oe)
                if total - at >= cap:
                    gaps.append(total - cap)
                nb = next((g for g in reversed(gaps) if g != b), None)
                if nb is None:
                    ext[r] = (b, cap)
                else:
                    moves.append((r, b, nb, self._used_local(r)))
                    ext[r] = (nb, cap)
                    changed = True
                    self.forced_relocations = getattr(self, "forced_relocations", 0) + 1
        keep_items, keep_rows = [], []
        for item, row in zip(placed, rows):
            n = self._need_local(len(item.get("tokens") or ()))
            base = self._room(ext, moves, n)
            if base is None:
                self.deferred_mem = getattr(self, "deferred_mem", [])
                self.deferred_mem.append(item)
                continue
            ext[row] = (base, n)
            changed = True
            keep_items.append(item)
            keep_rows.append(row)
        if not changed and not moves:
            return keep_items, keep_rows, None
        return keep_items, keep_rows, {"moves": moves, "bases": [ext[r][0] for r in range(self.rows)],
                                       "caps": [ext[r][1] for r in range(self.rows)]}

    def _apply_extents(self, extents):
        """Every rank: drain, move the relocated rows' slots, set the extents."""
        ctx, rank = self.ctx, self.rank
        if self.inflight is not None:
            self._row_reqs = dict(self.active) if rank == 0 else None
            self._pending = self.inflight
            self._drain()
        cache = ctx.decoder.cache
        for row, old, new, used in extents["moves"]:
            cache.relocate(row, old, new, used)
        cache.set_extents(extents["caps"], extents["bases"])

    def _is_long(self, item):
        p = item.get("params") or {}
        return (self.chunked_admit and self.max_batch > 1 and not p.get("nll")
                and len(item.get("tokens") or ()) >= self.prefill_long)

    def _drop_aborted(self, rows):
        """Every rank (c8 L17): drop long admissions whose client left. Every plan
        kind carries the plan's aborts, so they reach the followers even when the
        loop goes idle or runs a solo / exec plan next."""
        for row in rows:
            st = self.prefilling.pop(row, None)
            if st is not None:
                st["gen"].close()
                self.ctx.decoder.release([row])
                self.bs_rows.pop(row, None)

    def _prefill_tick(self, plan, items):
        """Every rank (c8 L17): start the plan's long admissions as paused prefills,
        drop aborted ones, and (when the plan says so) run one chunk of the first
        one — draining any in-flight step first (a chunk is an eager collective)."""
        ctx, rank = self.ctx, self.rank
        for row in plan.get("abort", []):
            st = self.prefilling.pop(row, None)
            if st is not None:
                st["gen"].close()
                ctx.decoder.release([row])
                self.bs_rows.pop(row, None)
        longs = plan.get("admit_long", [])
        long_items = items[len(plan["admit"]):] if rank == 0 else [None] * len(longs)
        for (row, tokens, max_new, params), item in zip(longs, long_items):
            cmd = {"op": "generate", "tokens": tokens, "max_new": max_new, "params": params}
            self.prefilling[row] = {
                "gen": run_generate_steps(ctx, cmd, row=row, admit_only=True, spec_absorb=True,
                                          chunked=True),
                "item": item}
        if not self.prefilling or not plan.get("prefill"):
            return
        if self.inflight is not None:
            self._row_reqs = dict(self.active) if rank == 0 else None
            self._pending = self.inflight
            self._drain()
        row = min(self.prefilling)
        st = self.prefilling[row]
        self._bind(st.get("item"))
        try:
            next(st["gen"])
            done = None
        except StopIteration as stop:
            done = stop.value
        sent = getattr(ctx, "stream_sent", 0)
        self._bind(None)
        if done is None:
            return
        out, finish = done
        del self.prefilling[row]
        if finish is None:
            self._bs_admitted(row, int(out[-1]), getattr(ctx, "admit_hidden", None))
        if rank == 0:
            req = _Req(st["item"], row, out, sent=sent)
            if finish is not None:
                self._finish(req, finish)
                self.pending_release.append(row)
            else:
                self.active[row] = req

    def _bs_admitted(self, row, tok, hid):
        """Every rank: an admitted row's batched-speculation entry state (its first
        token pending, the prompt's last hidden; no hidden: a warm-up round first)."""
        ctx = self.ctx
        d = ctx.decoder
        g = getattr(d, "decode_graphs", None)
        if g is None or getattr(ctx, "spec", None) is None:
            return
        st = g.bspec_state()
        L = int(d._row_lengths[row])
        st["pend_tok"][row].fill_(tok)
        if hid is not None:
            st["pend_hid"][row].copy_(hid.reshape(1, -1).expand(st["pend_hid"].shape[1], -1))
            st["spec_on"][row] = True
            # the prompt's last draft row is re-absorbed with the real first token
            g.bspec_set(row, P=L, c=L - 1, n=0)
            self.bs_rows[row] = {}
        else:
            st["spec_on"][row] = False
            g.bspec_set(row, P=L, c=L, n=0)
            self.bs_rows[row] = {"warm": True}

    def _solo_batched(self, item):
        """c8 L13b: a lone request runs the batched speculative rounds (device round
        state, lookahead) when they can serve it: unfiltered sampling (or greedy),
        not an NLL probe, batched speculation on."""
        p = item.get("params") or {}
        spec = getattr(self.ctx, "spec", None)
        rows = set(getattr(self.ctx, "batch_rows", ()) or ())
        spec_k = int(getattr(spec, "k", 3) or 3)
        bucket_ok = any(b >= spec_k + 1 for b in rows) if spec_k > 3 else (12 in rows)
        return (self.solo_bspec and self.bspec and self.max_batch > 1 and not p.get("nll")
                and _unfiltered(p) and spec is not None and bucket_ok)

    def _preempt(self):
        return not self.ctx.q.empty()

    # ---- every rank ----------------------------------------------------
    def iterate(self):
        ctx, rank, live = self.ctx, self.rank, self.liveness
        if rank == 0:
            live.enter("planning the next round")
            plan, items = self._plan()
            live.enter("broadcasting the round plan")
        else:
            plan, items = None, None
            live.enter("awaiting rank 0's round plan")
        obj = [plan]
        try:
            ctx.dist.broadcast_object_list(obj, src=0)
        except RuntimeError as exc:
            message = (f"{live.describe()}: the round-plan broadcast failed "
                       f"(gloo deadline {GLOO_TIMEOUT_MIN} min): {exc}")
            trace(rank, f"FATAL: {message}")
            raise PlanWaitTimeout(message) from exc
        plan = obj[0]
        live.leave(None if plan is None else plan.get("kind"))
        if plan is None:
            return False
        if plan["release"]:
            ctx.decoder.release(plan["release"])
            for row in plan["release"]:
                self.bs_rows.pop(row, None)
        if plan.get("extents"):
            self._apply_extents(plan["extents"])
        self._drop_aborted(plan.get("abort", ()))
        try:
            if plan["kind"] == "idle":
                pass                             # rank 0 had nothing queued: stay in step
            elif plan["kind"] == "exec":
                self._exec(plan, items)
            elif plan["kind"] == "solo":
                self._solo(plan, items)
            else:
                self._prefill_tick(plan, items)
                self._batch(plan, items[:len(plan["admit"])] if rank == 0 else items)
        except Exception as exc:  # noqa: BLE001
            if rank == 0:
                error = f"{type(exc).__name__}: {exc}"
                traceback.print_exc()
                for item in list(items or []) + [req.item for req in self.active.values()]:
                    item["error"] = error
                    item["ev"].set()
                    if item.get("stream_q") is not None:
                        item["stream_q"].put(("end", None))
            raise
        return True

    def _exec(self, plan, items):
        ctx, rank = self.ctx, self.rank
        # consult18 P2 dev loop (HY4_DEV_EXEC=1 only, see do_POST).
        env = {"ctx": ctx, "rank": rank, "result": None, "__name__": "h4_dev_exec"}
        try:
            with open(plan["path"]) as fh:
                exec(compile(fh.read(), plan["path"], "exec"), env)
            local = {"rank": rank, "ok": True, "result": env.get("result")}
        except Exception:  # noqa: BLE001
            local = {"rank": rank, "ok": False, "error": traceback.format_exc()[-6000:]}
        gathered = [None] * ctx.world
        ctx.dist.all_gather_object(gathered, local)
        if rank == 0:
            items[0]["result"] = gathered
            items[0]["ev"].set()

    def _stage_sampling(self, graphs, size, temps):
        """Every rank: the sampled step's 1/T per row and fresh U(0,1) noise over its
        vocabulary shard (its own generator: independent across shards, which the
        Gumbel-max over shards needs); greedy rows get e^-1 (Gumbel 0)."""
        import math

        import torch

        entry = graphs.batch_sample_state(size)
        inv = [1.0 / t if t > 0.0 else 1.0 for t in temps] + [1.0] * (size - len(temps))
        key = tuple(inv)
        if not hasattr(self, "_inv_key"):
            self._inv_key = {}
        if self._inv_key.get(size) != key:
            # pinned staging: a pageable H2D would wait for the in-flight step
            cap = max(8, int(self.rows), int(size))
            if not hasattr(self, "_inv_host") or int(self._inv_host[0].shape[0]) < cap:
                self._inv_host = [torch.zeros((cap,), dtype=torch.float32).pin_memory()
                                  for _ in range(4)]
                self._inv_at = 0
            buf = self._inv_host[self._inv_at]
            self._inv_at = (self._inv_at + 1) % len(self._inv_host)
            for i, v in enumerate(inv):
                buf[i] = v
            entry["inv_temp"].copy_(buf[:size], non_blocking=True)
            self._inv_key[size] = key
        if getattr(self, "_noise_gen", None) is None:
            self._noise_gen = torch.Generator(device=entry["noise"].device)
            self._noise_gen.manual_seed(int.from_bytes(os.urandom(8), "little") >> 1)
        entry["noise"].uniform_(generator=self._noise_gen)
        for i, t in enumerate(temps):
            if t <= 0.0:
                entry["noise"][i].fill_(math.exp(-1.0))

    def _bind(self, item):
        """rank 0: run_generate streams and cancels through ctx (one request at a time)."""
        ctx = self.ctx
        ctx.stream_q = item.get("stream_q") if item is not None else None
        ctx.cancel = item["cancel"] if item is not None else None
        ctx.stream_sent = 0

    def _solo(self, plan, items):
        ctx, rank = self.ctx, self.rank
        item = items[0] if rank == 0 else None
        self._bind(item)
        cmd = {"op": "generate", "tokens": plan["tokens"], "max_new": plan["max_new"],
               "params": plan["params"]}
        spec = getattr(ctx, "spec", None)
        if spec is not None:
            spec.yield_state = None
        ctx.solo_hand = None
        out, finish = run_generate(ctx, cmd, row=0, preempt=self._preempt if rank == 0 else None)
        hand = ((getattr(spec, "yield_state", None) if spec is not None else None)
                or getattr(ctx, "solo_hand", None))
        if finish == "preempt" and hand and getattr(self, "adopt", True) and self.bspec:
            self._adopt_solo(hand)               # every rank: row 0 keeps speculating
        if rank == 0:
            req = _Req(item, 0, out, sent=ctx.stream_sent)
            if finish == "preempt":
                self.active[0] = req       # its last token is the next input
            else:
                self._finish(req, finish)
                item["nll"] = getattr(ctx, "last_nll", None)
        self._bind(None)

    def _batch(self, plan, items):
        """C: admissions; D: enqueue this step (carried rows read the in-flight step's
        device outputs); E: consume the in-flight step (rank 0 streams / stops; a
        flagged step rolls both back and reruns exactly); F: a host-sampled step
        finishes now, a device-sampled one stays in flight (lookahead)."""
        ctx, rank = self.ctx, self.rank
        if self._bspec_ok(plan):
            return self._bspec_batch(plan, items)
        prev = self.inflight
        if prev is not None and prev.get("bspec"):
            # a speculative round in flight: consume it now (a plain step is never
            # rolled back with it); its snapshot "out" holds the rows' next inputs
            self._row_reqs = dict(self.active) if rank == 0 else None
            self._pending = prev
            self._drain()
        for row, _ in plan["carry"]:
            self.bs_rows.pop(row, None)         # a plain step: its draft state goes stale
        # An eager collective (admission prefill; a new graph's eager first call)
        # must never be issued while a captured step is still pending on the
        # device (the dg88d hang): consume the in-flight step first. Its device
        # snapshot stays valid, so carried rows still read their tokens from it.
        self._pending = None if (prev is not None and prev.get("bspec")) else prev
        # rank 0: the request of every row this step may carry (a drain may finish
        # one; its tokens are then skipped)
        self._row_reqs = dict(self.active) if rank == 0 else None
        if prev is not None and plan["admit"]:
            self._drain()
        admitted = []
        for index, (row, tokens, max_new, params) in enumerate(plan["admit"]):
            item = items[index] if rank == 0 else None
            self._bind(item)
            cmd = {"op": "generate", "tokens": tokens, "max_new": max_new, "params": params}
            out, finish = run_generate(ctx, cmd, row=row, admit_only=True)
            if finish is None:
                admitted.append((row, int(out[-1]), params))   # every rank: the broadcast token
            if rank == 0:
                req = _Req(item, row, out, sent=ctx.stream_sent)
                if finish is not None:
                    self._finish(req, finish)
                    self.pending_release.append(row)
                else:
                    self.active[row] = req
                    self._row_reqs[row] = req
            self._bind(None)
        prev_rows = prev["rows"] if prev is not None else []
        rows, explicit, src, temps, unfiltered = [], [], [], [], []
        for (row, tok), (temp, unf) in zip(plan["carry"], plan["carry_params"]):
            rows.append(row)
            explicit.append(0 if tok is None else int(tok))
            src.append(prev_rows.index(row) if tok is None else -1)
            temps.append(temp)
            unfiltered.append(unf)
        for row, tok, params in admitted:
            rows.append(row)
            explicit.append(tok)
            src.append(-1)
            temps.append(float(params.get("temperature", 0.9) or 0.0))
            unfiltered.append(_unfiltered(params))
        step = self._enqueue(rows, explicit, src, temps, all(unfiltered), prev) if rows else None
        if self._pending is not None:
            pending, self._pending = self._pending, None
            if self._consume(pending, step):
                step = None                    # rolled back: this step is discarded
        if step is not None and (not step["sampled"] or not self.lookahead):
            self._consume(step, None)
            step = None
        self.inflight = step

    # ---- batched speculative sampling (k = 1) ---------------------------
    def _bspec_ok(self, plan):
        """Every rank, from the plan: a speculative round when every row samples
        unfiltered at T > 0 and the 2B-entry bucket is prepared."""
        from ignition.recipes.hy4 import runtime as _rt

        ctx = self.ctx
        if not self.bspec or getattr(ctx, "spec", None) is None or _rt.SWITCHES.get("spec") != "on":
            return False
        params = [(t, u) for t, u in plan["carry_params"]]
        params += [(float(p.get("temperature", 0.9) or 0.0), _unfiltered(p))
                   for (_, _, _, p) in plan["admit"]]
        # c8 L5d: greedy rows (T = 0) speculate too (exact argmax acceptance in-graph)
        if not params or not all(t >= 0.0 and u for t, u in params):
            return False
        graphs = ctx.decoder.decode_graphs
        batch_rows = tuple(getattr(ctx, "batch_rows", ()) or ())
        K = self._bs_k(len(params))
        need = (K + 1) * len(params)
        max_b = max(batch_rows, default=0)
        size = graphs.batch_bucket(need, allowed=batch_rows) if need <= max_b else None
        return size is not None and size in set(batch_rows)

    def _bs_k(self, B):
        """Drafts per row in a round.

        For `spec.k <= 3` (or absent), keeps the legacy c8 L13 policy (`3` at
        `B <= 3`, `2` at `B == 4` within bucket 12, else `1`). For `spec.k > 3`
        (e.g. DFly2 `k = 5`), picks the widest `K <= min(spec.k, bspec_kmax)`
        whose verify width `(K + 1) * B` fits a prepared bucket up to 12 or an
        exact prepared bucket (e.g. 18, 24).
        """
        rows = set(getattr(self.ctx, "batch_rows", ()) or ())
        spec = getattr(self.ctx, "spec", None)
        spec_k = int(getattr(spec, "k", 3) or 3)
        if spec_k <= 3:
            if 12 not in rows:
                return 1
            if self.bspec_k3 and B <= 3:
                return min(3, spec_k)
            return min(2, spec_k) if self.bspec_k2 and B <= 4 else 1
        graphs = getattr(getattr(self.ctx, "decoder", None), "decode_graphs", None)
        kmax = int(getattr(graphs, "bspec_kmax", spec_k)) if graphs is not None else spec_k
        top_k = min(spec_k, kmax)
        cap = int(getattr(self, "bspec_multi_cap", 12) or 12)
        for K in range(top_k, 1, -1):
            need = (K + 1) * B
            cands = [b for b in rows if need <= b <= cap]
            if cands:
                b = min(cands)
                if b <= 12 or b - need <= max(2, B // 2):
                    return K
            if K == top_k and need <= cap and need in rows:
                return K
        return 1

    def _bspec_batch(self, plan, items):
        """Every rank: one batched speculative round. Steady state (c8 L5b, `bs_la`):
        the same rows as the in-flight round and no admission — round k+1 is
        enqueued from the device round state, THEN round k is consumed (the host
        read, streaming and stops overlap round k+1). Otherwise anything in flight
        is consumed first (admissions are eager collectives: the drain rule)."""
        import torch

        ctx, rank = self.ctx, self.rank
        d = ctx.decoder
        g = d.decode_graphs
        st = g.bspec_state()
        prev = self.inflight
        self._row_reqs = dict(self.active) if rank == 0 else None
        rows = [row for row, _ in plan["carry"]]
        temps = [t for t, _ in plan["carry_params"]]
        K = self._bs_k(len(rows))
        if (self.bs_la and prev is not None and prev.get("bspec") and not plan["admit"]
                and prev["rows"] == rows and prev["temps"] == temps and prev["K"] == K
                and all(r in self.bs_rows for r in rows)):
            end = self._bs_end(rows, inflight=1)
            if self._bs_cached(len(rows), K, end):
                nxt = self._bs_enqueue(rows, temps, end, K)
                self.inflight = None
                rolled = self._bs_consume(prev, nxt)
                self.inflight = None if rolled else nxt
                return
        if prev is not None:
            self._pending = prev
            self._drain()           # carried tokens stay in its snapshot
        prev_rows = prev["rows"] if prev is not None else []
        entering = [(row, tok, prev_rows.index(row) if tok is None else -1)
                    for row, tok in plan["carry"] if row not in self.bs_rows]
        admitted = []
        for index, (row, tokens, max_new, params) in enumerate(plan["admit"]):
            item = items[index] if rank == 0 else None
            self._bind(item)
            cmd = {"op": "generate", "tokens": tokens, "max_new": max_new, "params": params}
            out, finish = run_generate(ctx, cmd, row=row, admit_only=True, spec_absorb=True)
            if finish is None:
                admitted.append((row, int(out[-1]), getattr(ctx, "admit_hidden", None),
                                 float(params.get("temperature", 0.9) or 0.0)))
            if rank == 0:
                req = _Req(item, row, out, sent=ctx.stream_sent)
                if finish is not None:
                    self._finish(req, finish)
                    self.pending_release.append(row)
                else:
                    self.active[row] = req
                    self._row_reqs[row] = req
            self._bind(None)
        # rows without draft state: one warm-up round as a plain row (never accepted:
        # an exact plain sample; its verify hidden becomes the pending entry), then
        # they speculate (c8 L5d)
        for row, tok, src in entering:
            if src >= 0:
                st["pend_tok"][row].copy_(prev["snap"]["out"][src].reshape(1).expand(st["pend_tok"].shape[1]))
            else:
                st["pend_tok"][row].fill_(int(tok))
            st["spec_on"][row] = False
            L = int(d._row_lengths.get(row, 0))
            g.bspec_set(row, P=L, c=L, n=0)
            self.bs_rows[row] = {"warm": True}
        for row, tok, hid, _ in admitted:
            self._bs_admitted(row, tok, hid)
        rows = rows + [row for row, _, _, _ in admitted]
        temps = temps + [t for _, _, _, t in admitted]
        if not rows:
            return
        K = self._bs_k(len(rows))
        # c8 L13: a row whose last round accepted n > K drafts holds n + 1 pending
        # entries; the (K+1)-wide absorb takes its last K + 1
        for r in rows:
            n_last = self.bs_rows.get(r, {}).get("n", 0)
            if n_last > K:
                g.bspec_shift(r, n_last - K)
                self.bs_rows[r]["n"] = K
        end = self._bs_end(rows, inflight=0)
        rnd = self._bs_enqueue(rows, temps, end, K)
        if self.bs_la:
            self.inflight = rnd
        else:
            self._bs_consume(rnd, None)

    def _bs_end(self, rows, *, inflight):
        """An upper bound on a round's context end from the host-known lengths (each
        unconsumed round in flight commits <= K+1; pad entries write P + K + 1). A
        later regime serves an earlier context exactly, so the bound is safe."""
        d = self.ctx.decoder
        km = int(getattr(d.decode_graphs, "bspec_kmax", d.decode_graphs.BSPEC_KMAX))
        return max(int(d._row_lengths.get(r, 0)) for r in rows) + (km + 1) * inflight + km + 2

    def _bs_layout(self, B, K):
        """A round's entry lists: (absorb/verify size, rows, pads), (chain size, rows, pads)."""
        g = self.ctx.decoder.decode_graphs
        allowed = tuple(getattr(self.ctx, "batch_rows", ()) or ()) or None
        per = K + 1
        size = g.batch_bucket(per * B, allowed=allowed)
        c_need = max(2, B)
        csize = g.batch_bucket(
            c_need, allowed=allowed if (allowed and max(allowed) >= c_need) else None)
        return per, size, csize

    def _bs_cached(self, B, K, end):
        g = self.ctx.decoder.decode_graphs
        is_dfly = bool(getattr(self.ctx.decoder, "has_dfly", False))
        per, size, csize = self._bs_layout(B, K)
        shape, cshape = batch_shape_of(size), batch_shape_of(csize)
        return (g.bspec_cached("draft", size, end, call_shape=shape, per=per)
                and g.bspec_cached("verify", size, end, call_shape=shape, per=per)
                and (is_dfly or all(g.bspec_cached("chain", csize, end, call_shape=cshape, step=j)
                                    for j in range(1, K))))

    def _adopt_solo(self, ys):
        """Every rank: a preempted solo speculation joins the batched rounds with its
        draft state — the pending draft rows but the last are absorbed now (row 0's
        spec graphs; nothing is in flight), the last (P-1, the bonus) becomes the
        row's batched pending entry."""
        import torch

        spec = self.ctx.spec
        is_dfly = bool(getattr(self.ctx.decoder, "has_dfly", False))
        st = self.ctx.decoder.decode_graphs.bspec_state()
        P, bonus, c = int(ys["P"]), int(ys["bonus"]), int(ys["c"])
        tokens, hidden = ys["tokens"], ys["hidden"].reshape(len(ys["tokens"]), -1)
        if len(tokens) > 1 and not is_dfly:
            spec._absorb(c, tokens[:-1], hidden[:-1], sample=False)
        st["pend_tok"][0].fill_(bonus)
        st["pend_hid"][0].copy_(hidden[-1].reshape(1, -1).expand(st["pend_hid"].shape[1], -1))
        st["spec_on"][0] = True
        self.ctx.decoder.decode_graphs.bspec_set(
            0, P=int(self.ctx.decoder._row_lengths.get(0, P)), c=P - 1, n=0)
        self.bs_rows[0] = {}
        torch.cuda.synchronize(self.ctx.device)

    def _bspec_gen(self, device):
        import torch

        if getattr(self, "_bs_gen", None) is None:
            self._bs_gen = torch.Generator(device=device)
            self._bs_gen.manual_seed(int.from_bytes(os.urandom(8), "little") >> 1)
        return self._bs_gen

    def _bs_enqueue(self, rows, temps, end, K):
        """Every rank: stage and launch one round (draft absorb, K-1 chain steps,
        verify) from the device round state; snapshot its outputs (D2H + event).
        No host read."""
        import torch

        ctx = self.ctx
        d = ctx.decoder
        g = d.decode_graphs
        st = g.bspec_state()
        B = len(rows)
        per, size, csize = self._bs_layout(B, K)
        shape, cshape = batch_shape_of(size), batch_shape_of(csize)
        key = tuple(rows)
        if self._bs_rows_t[0] != key:
            self._bs_rows_t = (key, torch.tensor(rows, dtype=torch.long).to(ctx.device))
        rows_t = self._bs_rows_t[1]
        S = int(st["inv_temp"].shape[0])
        inv = tuple([1.0 / t if t > 0.0 else 0.0 for t in temps] + [1.0] * (S - B))
        if self._bs_inv != inv:
            # inv_temp 0 marks a greedy row (the graphs take its exact argmax)
            st["inv_temp"].copy_(torch.tensor(inv, dtype=torch.float32).pin_memory(), non_blocking=True)
            self._bs_inv = inv
        gen = self._bspec_gen(st["noise_q"].device)
        slots = max(size // per, csize)
        st["noise_q"][:slots, :K].uniform_(generator=gen)
        st["noise_v"][:slots, :per].uniform_(generator=gen)
        st["uniform"][:slots, :K].uniform_(generator=gen)
        j = min(range(B), key=lambda i: int(d._row_lengths.get(rows[i], 0)))
        erows = [r for r in rows for _ in range(per)] + [rows[j]] * (size - per * B)
        pads = [False] * (per * B) + [True] * (size - per * B)
        crows = list(rows) + [rows[j]] * (csize - B)
        cpads = [False] * B + [True] * (csize - B)
        bonus = st["pend_tok"][rows_t, st["n"][rows_t]]   # the round's inputs (rollback)
        g.run_bspec("draft", erows, pads, end=end, call_shape=shape, per=per)
        if not getattr(d, "has_dfly", False):
            for step in range(1, K):
                g.run_bspec("chain", crows, cpads, end=end, call_shape=cshape, step=step)
        g.run_bspec("verify", erows, pads, end=end, call_shape=shape, per=per)
        snap = self._bs_snap(needed=B * (2 * K + 3) + 1)
        dev = snap["dev"]
        o = 0
        for part in (st["out_n"][:B], st["drafts"][:B, :K].reshape(-1),
                     st["out_tokens"][:B, :per].reshape(-1), bonus, st["out_vflag"].reshape(1)):
            m = int(part.numel())
            dev[o:o + m].copy_(part)
            o += m
        snap["host"][:o].copy_(dev[:o], non_blocking=True)
        snap["event"].record()
        # the rows' next inputs after this round (a plain step after it reads them)
        snap["out"] = st["pend_tok"][rows_t, st["n"][rows_t]]
        return {"bspec": True, "rows": list(rows), "temps": list(temps), "snap": snap, "K": K,
                "width": o}

    def _bs_snap(self, *, needed=0):
        import torch

        g = getattr(getattr(self.ctx, "decoder", None), "decode_graphs", None)
        km = int(getattr(g, "bspec_kmax", getattr(g, "BSPEC_KMAX", 3))) if g is not None else 3
        cap = max(81, int(self.rows) * (2 * km + 3) + 1, int(needed))
        snaps = getattr(self, "_bs_snaps", None)
        if snaps is None or int(snaps[0]["dev"].shape[0]) < cap:
            dev = self.ctx.device
            self._bs_snaps = [{"dev": torch.zeros((cap,), dtype=torch.long, device=dev),
                               "host": torch.zeros((cap,), dtype=torch.long).pin_memory(),
                               "event": torch.cuda.Event()} for _ in range(4)]
            self._bs_snap_at = 0
        snap = self._bs_snaps[self._bs_snap_at]
        self._bs_snap_at = (self._bs_snap_at + 1) % len(self._bs_snaps)
        return snap

    def _bs_consume(self, rnd, nxt):
        """Every rank: wait for round `rnd`, apply it (rank 0 emits 1 or 2 tokens per
        row). A flagged verify (a narrow selection may be inexact) discards `nxt`
        (enqueued after it) and reruns `rnd`'s rows exactly. True when it rolled back."""
        import torch

        ctx, rank = self.ctx, self.rank
        d = ctx.decoder
        g = d.decode_graphs
        st = g.bspec_state()
        rows = rnd["rows"]
        B, K = len(rows), rnd["K"]
        per = K + 1
        rnd["snap"]["event"].synchronize()
        host = rnd["snap"]["host"][:rnd["width"]].tolist()
        acc = host[:B]
        drafts = host[B:B + B * K]
        toks = host[B + B * K:B + B * K + B * per]
        bonus = host[B + B * K + B * per:B + B * K + B * per + B]
        flag = host[-1]
        warm = [r for r in rows if self.bs_rows.get(r, {}).get("warm")]
        self.steps += 1
        if self.steps % 256 == 0:
            d.trace.clear()
        # rounds are consumed in order: the host lengths are this round's starts; a
        # row the plan released meanwhile (no length) is skipped
        live = [i for i, r in enumerate(rows) if r in d._row_lengths]
        starts = [int(d._row_lengths.get(r, 0)) for r in rows]
        if flag:
            torch.cuda.synchronize(ctx.device)          # nxt (if any) finished: discarded
            if live:
                self._bspec_exact([rows[i] for i in live], [starts[i] for i in live],
                                  [bonus[i] for i in live])
            for r in warm:
                st["spec_on"][r] = True
                self.bs_rows[r] = {"n": 0}
            if self._bs_rows_t[0] == tuple(rows):
                rt = self._bs_rows_t[1]
                rnd["snap"]["out"] = st["pend_tok"][rt, st["n"][rt]]
            return nxt is not None
        for i in live:
            d.append([rows[i]], int(acc[i]) + 1)
            if rows[i] in self.bs_rows:
                self.bs_rows[rows[i]]["n"] = int(acc[i])
        for r in warm:
            st["spec_on"][r] = True     # warmed: its pending entry is this verify's
            self.bs_rows[r].pop("warm", None)
        self.bspec_rounds += 1
        self.bspec_slots += B
        self.bspec_accepted += sum(int(x) for x in acc)
        if rank == 0:
            for i, r in enumerate(rows):
                req = self._row_reqs.get(r) if self._row_reqs is not None else None
                if req is None or req.done:
                    continue
                n = int(acc[i])
                self._emit(req, drafts[i * K:i * K + n] + [toks[i * per + n]])
        return False

    def _bspec_exact(self, rows, starts, bonus):
        """A flagged verify (a narrow selection may be inexact): each row takes one
        exact one-row step at its bonus instead (host-sampled, broadcast)."""
        import torch
        from ignition.recipes.hy4 import runtime as _rt

        ctx, rank = self.ctx, self.rank
        d = ctx.decoder
        st = d.decode_graphs.bspec_state()
        torch.cuda.synchronize(ctx.device)
        root = _rt.SWITCHES.get("root_logits") == "on"
        d._sync_free_allowed = False
        d.logits_root_host = root
        parts, hids = [], []
        try:
            for i, r in enumerate(rows):
                # the round's input (its pending state may since be overwritten)
                t = torch.tensor([[int(bonus[i])]], dtype=torch.long).to(ctx.device)
                logits, post, _ = d.forward(t, request_rows=[r], call_shape=ctx.decode_shape,
                                            logits_rows="all", return_hidden=True)
                parts.append(logits)
                hids.append(post.reshape(-1).clone())
        finally:
            d._sync_free_allowed = True
            d.logits_root_host = False
        d.sync_free_reruns = getattr(d, "sync_free_reruns", 0) + 1
        self.rollbacks += 1
        ids = [None]
        if rank == 0:
            reqs = [self._row_reqs.get(r) for r in rows]
            params = [req.params if req is not None else {} for req in reqs]
            logits = torch.cat([p.reshape(1, -1) for p in parts])
            picked = sample_rows(ctx.sampler, logits, params, [0] * len(rows), [1 << 30] * len(rows))
            ids = [[int(tok) for tok, _, _ in picked]]
        ctx.dist.broadcast_object_list(ids, src=0)
        for i, r in enumerate(rows):
            st["pend_tok"][r].fill_(ids[0][i])
            st["pend_hid"][r].copy_(hids[i].reshape(1, -1).expand(st["pend_hid"].shape[1], -1))
            d.append([r], 1)
            d.decode_graphs.bspec_set(r, P=starts[i] + 1, c=starts[i], n=0)
            if r in self.bs_rows:
                self.bs_rows[r]["n"] = 0
        if rank == 0:
            for i, r in enumerate(rows):
                req = self._row_reqs.get(r)
                if req is not None and not req.done:
                    self._emit(req, [ids[0][i]])

    def _emit(self, req, tokens):
        """rank 0: append sampled tokens with the stop rules (a stop drops the rest)."""
        for tok in tokens:
            (tok, done, finish), = _stops(self.ctx.sampler, [tok], [req.params], [len(req.out)],
                                          [req.max_new])
            req.out.append(tok)
            if done:
                self._push(req)
                del self.active[req.row]
                self._finish(req, finish)
                self.pending_release.append(req.row)
                return
        self._push(req)

    def _drain(self):
        """Consume the pending (in-flight) step now, before an eager collective."""
        pending, self._pending = self._pending, None
        self.inflight = None
        if pending is not None:
            self._consume(pending, None)

    def _snap(self, *, needed=0):
        import torch

        buckets = tuple(getattr(self.ctx, "batch_rows", ()) or ())
        cap = max(8, int(self.rows), max((1, *buckets)), int(needed))
        if self._snaps is None or int(self._snaps[0]["cap"]) < cap:
            dev = self.ctx.device
            self._snaps = [{"cap": cap,
                            "out": torch.zeros((cap,), dtype=torch.long, device=dev),
                            "inp": torch.zeros((cap,), dtype=torch.long, device=dev),
                            "flag": torch.zeros((1,), dtype=torch.long, device=dev),
                            "src": torch.zeros((cap,), dtype=torch.long, device=dev),
                            "host": torch.zeros((cap + 1,), dtype=torch.long).pin_memory(),
                            "src_host": torch.zeros((cap,), dtype=torch.long).pin_memory(),
                            "event": torch.cuda.Event()} for _ in range(4)]
            self._snap_at = 0
        snap = self._snaps[self._snap_at]
        self._snap_at = (self._snap_at + 1) % len(self._snaps)
        return snap

    def _enqueue(self, rows, explicit, src, temps, device_sample, prev):
        """Every rank: stage and launch one batched step; returns its record."""
        import torch
        from ignition.recipes.hy4 import runtime as _rt
        from ignition.recipes.hy4.wiring import batch_shape

        ctx, rank = self.ctx, self.rank
        d = ctx.decoder
        graphs = d.decode_graphs
        starts = [int(d._row_lengths.get(row, 0)) for row in rows]
        allowed = (tuple(b for b in graphs.BATCH_BUCKETS
                         if b == 1 or b in getattr(ctx, "batch_rows", ()))
                   if getattr(ctx, "batch_rows", None) else None)
        size = graphs.batch_bucket(len(rows), allowed=allowed)
        shape = ctx.decode_shape if size == 1 else batch_shape(size)
        pad = None
        if size > len(rows):
            # every pad entry: one row one past this step's token (see run_batch)
            j = min(range(len(rows)), key=lambda i: starts[i])
            pad = (rows[j], starts[j] + 1)
        sampled = bool(device_sample) and getattr(self, "device_sample", True)
        if self._pending is not None and not graphs.batch_cached(rows, starts, call_shape=shape,
                                                                 pad=pad, sample=sampled):
            self._drain()                      # the eager first call must not overlap a replay
        snap = self._snap(needed=size)
        cap = int(snap["cap"])
        pre = None
        if any(i >= 0 for i in src):
            prev_out = prev["snap"]["out"]
            for i, v in enumerate(src + [-1] * (size - len(src))):
                snap["src_host"][i] = v
            snap["src"].copy_(snap["src_host"], non_blocking=True)

            def pre(tokens_view):
                index = snap["src"][:size]
                fixed = torch.where(index >= 0, prev_out[index.clamp(min=0)], tokens_view)
                tokens_view.copy_(fixed)
        step = {"rows": list(rows), "size": size, "sampled": sampled, "snap": snap,
                "reqs": [self._row_reqs[row] for row in rows] if rank == 0 else None}
        if sampled:
            self._stage_sampling(graphs, size, temps)
            out_tokens, flag = graphs.run_batch(explicit, rows, starts, call_shape=shape, pad=pad,
                                                sample=True, pre=pre)
            snap["out"][:size].copy_(out_tokens[:size])
            snap["flag"].copy_(flag.reshape(1).long())
            snap["host"][:size].copy_(snap["out"][:size], non_blocking=True)
            snap["host"][cap:cap + 1].copy_(snap["flag"], non_blocking=True)
        else:
            vr = d.weights.sharding.vocabulary_range
            root = {"root_host": True} if _rt.SWITCHES.get("root_logits") == "on" else {}
            local, flag = graphs.run_batch(explicit, rows, starts, call_shape=shape, pad=pad,
                                           pre=pre)
            step["logits"] = d.collectives.gather_logits(local, vocabulary_range=vr, **root)
            step["flag"] = bool(flag)
        snap["inp"][:size].copy_(graphs._batch_state(size)["args"][0:size])
        snap["event"].record()
        d.append(rows, 1)
        self.steps += 1
        if self.steps % 256 == 0:
            d.trace.clear()                # eager first steps of new buckets append to it
        return step

    def _consume(self, step, nxt):
        """Every rank: wait for `step`, take its tokens (rank 0: stream, stops); a
        flagged step (a narrow scores-only cut needed the full re-offer) rolls back
        it and `nxt` (enqueued after it) and reruns it on the exact one-row path.
        Returns True when it rolled back."""
        import torch

        if step.get("bspec"):
            return self._bs_consume(step, nxt)
        ctx, rank = self.ctx, self.rank
        d = ctx.decoder
        rows, size = step["rows"], step["size"]
        if step["sampled"]:
            step["snap"]["event"].synchronize()
            flagged = bool(int(step["snap"]["host"][int(step["snap"]["cap"])]))
        else:
            flagged = step["flag"]
        force = getattr(self, "force_rollback_every", 0)   # dev: exercise the rollback
        if force and self.steps % force == 0:
            flagged = True
        if flagged:
            torch.cuda.synchronize(ctx.device)
            for row in (nxt["rows"] if nxt is not None else []):
                d._row_lengths[row] -= 1
            for row in rows:
                d._row_lengths[row] -= 1
            from ignition.recipes.hy4 import runtime as _rt
            root = _rt.SWITCHES.get("root_logits") == "on"
            d._sync_free_allowed = False
            d.logits_root_host = root
            try:
                parts = []
                for j, row in enumerate(rows):
                    t = step["snap"]["inp"][j:j + 1].reshape(1, 1).clone()
                    parts.append(d.forward(t, request_rows=[row], call_shape=ctx.decode_shape,
                                           logits_rows="all"))
                    d.append([row], 1)
            finally:
                d._sync_free_allowed = True
                d.logits_root_host = False
            d.sync_free_reruns = getattr(d, "sync_free_reruns", 0) + 1
            self.rollbacks += 1
            ids = [None]
            if rank == 0:
                logits = torch.cat([p.reshape(1, -1) for p in parts])
                reqs = step["reqs"]
                picked = sample_rows(ctx.sampler, logits, [req.params for req in reqs],
                                     [len(req.out) for req in reqs], [req.max_new for req in reqs])
                ids = [[int(tok) for tok, _, _ in picked]]
            ctx.dist.broadcast_object_list(ids, src=0)
            # later steps may read this step's tokens from its device snapshot
            step["snap"]["out"][:len(rows)].copy_(
                torch.tensor(ids[0], dtype=torch.long).to(ctx.device))
            if rank == 0:
                self._apply(step["reqs"], ids=ids[0])
            return nxt is not None
        if rank == 0:
            if step["sampled"]:
                self._apply(step["reqs"], ids=step["snap"]["host"][:len(rows)].tolist())
            else:
                self._apply(step["reqs"], logits=step["logits"].reshape(size, -1)[:len(rows)])
        return False

    def _apply(self, reqs, ids=None, logits=None):
        """rank 0: one step's tokens into the live requests (finished ones skip)."""
        live = [(j, req) for j, req in enumerate(reqs) if not req.done]
        if not live:
            return
        if ids is None:
            picked_all = sample_rows(self.ctx.sampler, logits[[j for j, _ in live]],
                                     [req.params for _, req in live],
                                     [len(req.out) for _, req in live],
                                     [req.max_new for _, req in live])
        else:
            picked_all = _stops(self.ctx.sampler, [ids[j] for j, _ in live],
                                [req.params for _, req in live],
                                [len(req.out) for _, req in live],
                                [req.max_new for _, req in live])
        for (_, req), (tok, done, finish) in zip(live, picked_all):
            req.out.append(tok)
            self._push(req)
            if done:
                del self.active[req.row]
                self._finish(req, finish)
                self.pending_release.append(req.row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default="/model")
    ap.add_argument("--host", default="0.0.0.0", help="listening address (command-app greps the argv)")
    ap.add_argument("--port", type=int, default=8001, help="listening port (the `smart` role port)")
    ap.add_argument("--served-model-name", default="smart")
    ap.add_argument("--max-concurrency", type=int, default=1)
    ap.add_argument("--max-context", type=int, default=DEFAULT_MAX_CONTEXT)
    ap.add_argument("--default-max-tokens", type=int, default=0,
                    help="output token budget of a request that omits max_tokens; 0 = the "
                         "context window left after the prompt")
    ap.add_argument("--rows", type=int, default=2, help="cache request rows")
    ap.add_argument("--batch-rows", type=int, default=8,
                    help="most request rows one batched decode step serves (buckets 1, 2, 4, 8)")
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
    ap.add_argument("--scores-relay", type=int, choices=(0, 1), default=1,
                    help="N1: the CE scores exchange runs the two-hop island relay (each "
                         "rank's offer crosses the B<->D link once via its partner, which "
                         "forwards it in-switch; cross-RC score bytes drop 4x; byte-identical "
                         "received scores and top-k). 0 = the old direct broadcast to all 7 "
                         "peers (rollback; --selection-exchange candidates remains the full "
                         "selection rollback)")
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
    ap.add_argument("--expert-scale-compression", choices=("none", "csf"), default="none",
                    help="P3: store the routed experts' block scales in the lossless csf "
                         "encoding at load (csf_32x2_v1). Resident scale bytes shrink on "
                         "every rank and the freed memory joins the KV pool; the served "
                         "outputs stay bit-identical (the decode reproduces the exact scale "
                         "bytes). 'none' keeps the raw block scales resident.")
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
    dist.init_process_group("gloo", timeout=timedelta(minutes=GLOO_TIMEOUT_MIN))

    from ignition.recipes.hy4.bootstrap import (PreparationConfig, capture_kernel_graphs,
                                                construct_rank)
    from ignition.recipes.hy4.collectives import PcieDcpCollectives, PcieDcpTransport

    ctx = Context()
    ctx.rank, ctx.world, ctx.device, ctx.dist = rank, world, device, dist
    ctx.max_context = args.max_context
    # output budget for requests that omit max_tokens (None: the context window
    # left after the prompt)
    ctx.default_max_new = args.default_max_tokens if args.default_max_tokens > 0 else None
    ctx.row_capacity = args.capacity or args.max_context  # replaced by the measured capacity below
    ctx.served_names = (args.served_model_name,)
    ctx.bind = f"{args.host}:{args.port}"
    ctx.failure = None
    ctx.lock = threading.Lock()
    ctx.tok_lock = threading.Lock()
    ctx.waiting = 0          # queued requests not yet started (rank 0)
    ctx.cancel = None        # the running request's cancel event (rank 0)
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
        _text_cfg = _cfg.get("text_config") if isinstance(_cfg.get("text_config"), dict) else _cfg
        unabsorbed = _text_cfg["qk_nope_head_dim"] + _text_cfg["qk_rope_head_dim"]   # 256
        absorbed = _text_cfg["kv_lora_rank"] + _text_cfg["qk_rope_head_dim"]         # 576
        values = _text_cfg["kv_lora_rank"]                                           # 512
        cfg_topk = int(_cfg.get("index_topk", _text_cfg.get("index_topk", 2048)))
        reduce = _text_cfg["hidden_size"]                                            # 6144
        expanded = _text_cfg["v_head_dim"] if args.partial_wire == "expanded" else None   # 256
        _has_dfly = int(_cfg.get("dfly_num_hidden_layers") or _text_cfg.get("dfly_num_hidden_layers") or 0) > 0
        trace(rank, f"transport construction (gather widths {unabsorbed}/{absorbed}, "
                    f"values {values}, reduce {reduce})")
        score_offer = None
        if args.selection_wire == "dcp":
            if args.selection_exchange != "scores-only" or not args.local_offer:
                raise ValueError("--selection-wire dcp needs --selection-exchange scores-only "
                                 "and a non-zero --local-offer")
            score_offer = args.local_offer
        # (hoisted) the decode-family row shapes — the Step 5a dead-A2A pool
        # caps need DECODE_ROWS BEFORE the transport builds its pools.
        MTP = os.environ.get("HY4_MTP") == "1"     # consult18 P4: the MTP draft block
        SPEC_ROWS = (tuple(range(2, int(os.environ.get("HY4_SPEC_K") or 0) + 2))
                     if MTP and os.environ.get("HY4_SPEC_DECODE_ROWS") == "1" else ())
        # batched decode: the decode-family rows shapes of the step buckets past one row
        SHORT_KV = os.environ.get("HY4_SHORT_KV") == "on"
        # the replicated window: records (and indexer keys past top-k) of every
        # position below it, per serving row
        SHORT_POSITIONS = max(int(os.environ.get("HY4_SHORT_KV_POSITIONS") or 0), int(cfg_topk))
        BATCH_ROWS = tuple(b for b in (2, 4, 8, 16) if b <= args.batch_rows)
        # extra batch buckets (e.g. 6, 12, 24, 32 for bspec verify widths)
        BATCH_ROWS = tuple(sorted(set(BATCH_ROWS) | {
            int(x) for x in (os.environ.get("HY4_BATCH_EXTRA") or "").split(",") if x.strip()
        }))
        DECODE_ROWS = tuple(sorted(set(SPEC_ROWS) | set(BATCH_ROWS)))
        from ignition.recipes.hy4.collectives import parse_all_reduce_arms

        uses_dma = any(rule[2] == "dma" for rule in parse_all_reduce_arms(args.all_reduce_arm))
        if os.environ.get("HY4_AR_AUDIT") and args.all_reduce_kernel_rows:
            raise ValueError("HY4_AR_AUDIT runs the rsag arm at the ring's rows: leave "
                             "--all-reduce-kernel-rows at 0")
        # Step 5a (PERF-PLAN.md:380-383) — dead-A2A pool caps. The fail-loud
        # mirror of the HY4_AR_AUDIT check above: HY4_CE_AUDIT=1 runs the
        # SM-kernel arm at every CE width (it needs the FULL pools + bound
        # plans), so it must never be combined with the caps.
        if os.environ.get("HY4_CE_AUDIT") and os.environ.get("HY4_A2A_CAP"):
            raise ValueError("HY4_CE_AUDIT runs the kernel arm at the CE rows: leave "
                             "the dead-A2A caps (HY4_A2A_CAP) unset")
        ce_rows = args.ce_rows if (args.lse_transport.startswith("ce")
                                   and args.gather_transport.startswith("ce")) else None
        from ignition.recipes.hy4.collectives import a2a_pool_row_caps

        _caps = a2a_pool_row_caps(
            absorbed_width=absorbed, unabsorbed_width=unabsorbed, expanded_width=expanded,
            decode_rows=DECODE_ROWS, prefill_widths=PREFILL_WIDTHS,
            ce_rows=ce_rows, audit=bool(os.environ.get("HY4_CE_AUDIT")),
            force_caps=bool(os.environ.get("HY4_A2A_CAP")))
        _skip_above = _caps["skip_prepare_above"]
        trace(rank, f"dead-A2A caps {_caps['caps']} (A2A plans skipped at rows "
                    f">= {_skip_above})")
        transport = PcieDcpTransport.from_process_group(
            dist.group.WORLD, device, total_heads=HEADS_TOTAL,
            query_widths=(unabsorbed, absorbed), value_width=values, max_rows=CHUNK,
            reduce_width=reduce, all_reduce_arms=args.all_reduce_arm,
            score_offer=score_offer,
            # consult17 (lever 9): with expanded partials only the one-row
            # decode still uses the latent pool, so it shrinks to one row.
            extra_value_widths=() if expanded is None else (expanded,),
            # consult18 P4: the decode-family verify rows merge latents too, and
            # so do the batched decode step's rows.
            value_max_rows=None if expanded is None else max(
                int(os.environ.get("HY4_SPEC_K") or 0) + 1
                if os.environ.get("HY4_MTP") == "1" and os.environ.get("HY4_SPEC_DECODE_ROWS") == "1"
                and int(os.environ.get("HY4_SPEC_K") or 0) > 0 else 1,
                max((b for b in (1, 2, 4, 8, 16) if b <= args.batch_rows), default=1),
                *(int(x) for x in (os.environ.get("HY4_BATCH_EXTRA") or "").split(",") if x.strip())),
            dma=uses_dma, kernel_rows=args.all_reduce_kernel_rows or None,
            lanes=2 if args.overlap else 1, lane_rows=CHUNK,
            ce_lse_rows=args.ce_rows if args.lse_transport.startswith("ce") else None,
            ce_gather_rows=args.ce_rows if args.gather_transport.startswith("ce") else None,
            ce_lse_width=expanded or values, ce_gather_width=unabsorbed,
            # consult18 B2: ce-relay = pairwise relay across the B<->D link.
            ce_lse_relay=args.lse_transport == "ce-relay",
            ce_gather_relay=args.gather_transport == "ce-relay",
            # N1: the scores exchange's two-hop island relay.
            ce_scores_relay=bool(args.scores_relay),
            # Step 5a (PERF-PLAN.md:380-383): the dead-A2A row caps (576->16
            # always; 256->512 CE-active/audit-off; 256->CHUNK audit-on).
            pool_row_caps=_caps["caps"])
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

        settings = PreparationConfig(devices=tuple(range(world)), mtp=MTP,
                                     prefill_chunk_size=CHUNK,
                                     expert_scale_compression=(
                                         "csf" if args.expert_scale_compression == "csf" else None))
        trace(rank, f"expert scale compression: {args.expert_scale_compression}")

        def construct_and_wire(cache_rows, cache_capacity):
            """One construction + operator wiring + the boot's full plan book."""
            t0 = time.monotonic()
            built = construct_rank(args.snapshot, settings, rank=rank, device=device, pinned=True,
                                   mtp=MTP, cache_rows=cache_rows, cache_capacity=cache_capacity,
                                   world_size=world, collectives=collectives, fused_cache=True)
            load_s = round(time.monotonic() - t0, 1)
            trace(rank, f"constructed ({load_s} s, {built.memory_report['weights_bytes']} B weights, "
                        f"cache {cache_capacity} global/row x {cache_rows} rows)")
            if os.environ.get("HY4_W8") == "1":
                # consult18 P6: INT8 (W8A16, group-128 scales) storage of every
                # attention's same-input pack + o_proj (w8.py) — before the plan
                # book and the pool sizing, so the released BF16 feeds the KV pool.
                from ignition.recipes.hy4.w8 import quantize_attention
                keep = os.environ.get("HY4_W8_KEEP_BF16", "1") == "1"   # prefill stays exact BF16
                released = sum(quantize_attention(layer.attention, keep_bf16=keep)
                               for layer in built.weights.layers)
                if getattr(built.weights, "draft", None) is not None:
                    released += quantize_attention(built.weights.draft.attention, keep_bf16=keep)
                if getattr(built.weights, "dfly", None) is not None:
                    released += sum(quantize_attention(layer.attention, keep_bf16=keep)
                                    for layer in built.weights.dfly.layers)
                if os.environ.get("HY4_HEAD_BF16") == "1":
                    # consult18 P7: decode-family head reads a BF16 copy (the
                    # checkpoint's values; the FP32 head is their upcast)
                    built.decoder._head_bf16 = built.weights.head.to(torch.bfloat16)
                    if os.environ.get("HY4_HEAD_FP32_DROP") == "1":
                        # consult18 P9: every head call reads the BF16 copy (FP32 out);
                        # the FP32 upcast (354 MB/rank) is released
                        from ignition.recipes.hy4.w8 import Bf16HeadView
                        built.weights.head = Bf16HeadView(built.decoder._head_bf16)
                torch.cuda.empty_cache()
                trace(rank, f"w8: attention pack + o_proj INT8 for <= 16-row calls"
                            f"{' (BF16 kept for prefill)' if keep else ''}, {released / 2**30:.2f} GB BF16 released")
            built.decoder.partial_wire = args.partial_wire   # before the plan book binds the folds
            built.wire_operators(session="prepared")
            shp = {w: built.decoder.default_call_shape((1, w)) for w in PREFILL_WIDTHS}
            decode_shape = built.decoder.default_call_shape((1, 1))
            all_shapes = [decode_shape] + [shp[w] for w in PREFILL_WIDTHS]
            if DECODE_ROWS:
                # consult18 P4: the decode-family verification shapes of the
                # speculative verify / draft re-absorption (T = 2 .. k + 1), and
                # the batched decode step's buckets
                from ignition.recipes.hy4.wiring import batch_shape, verify_shape
                all_shapes += [verify_shape(t) for t in SPEC_ROWS]
                all_shapes += [batch_shape(b) for b in BATCH_ROWS]
            if world > 1:
                # The CP partial binds the UNBOUND sink at every call shape
                # (D2/D6); only the graph-capture shapes also need the world-1
                # variant (capture_kernel_graphs folds the sink in kernel).
                built.decoder.bind_plan_book(None, all_shapes, sink_variants=(False,), share_scratch=True,
                                             lanes=2 if args.overlap else 1)
                # the sink-bound (local-head) variant: the graph-capture shapes, and
                # with replicated short-context KV every decode-family shape
                bound = [decode_shape, shp[CHUNK]]
                if SHORT_KV:
                    from ignition.recipes.hy4.wiring import batch_shape, verify_shape
                    bound += [verify_shape(t) for t in SPEC_ROWS] + [batch_shape(b) for b in BATCH_ROWS]
                built.decoder.bind_plan_book(None, bound, sink_variants=(True,), share_scratch=True)
            else:
                built.decoder.bind_plan_book(None, all_shapes, sink_variants=(True,), share_scratch=True)
            if args.overlap:
                # consult18 stage D: lane 1's selection buffers and MoE workspace.
                built.decoder.enable_lanes(max_q_rows=CHUNK)
            if SHORT_KV and world > 1:
                # replicated short-context records for every serving row (both
                # phases: phase A's fixed term measures them before the pool)
                nbytes = built.decoder.cache.allocate_short(args.rows, SHORT_POSITIONS)
                trace(rank, f"short-context KV: {args.rows} rows x {SHORT_POSITIONS} positions, "
                            f"{nbytes / 2**30:.2f} GiB")
            return built, shp, decode_shape, load_s

        # The transport plans belong to the transport, not to a construction:
        # prepare them BEFORE phase A so their memory is inside the measured
        # fixed term (they used to be prepared after the pool was sized, out
        # of the headroom) and so the phase-A warm-up can run (lever 5).
        from b12x.preparation import PreparationSession

        session = PreparationSession(device=device, autotune=False, compile_workers=1)
        # consult18 P2: the one-row (decode) collective plans may race their
        # launch geometry (b12x autotune, collective at boot); the math per
        # element does not depend on threads / block limits.
        dsession = (PreparationSession(device=device, autotune=True, compile_workers=1)
                    if os.environ.get("HY4_DECODE_AUTOTUNE") == "1" else session)
        # Prepared plans only (no JIT at run time): the decode shape gathers the
        # absorbed width at 1 query row; every prepared prefill width gathers
        # the unabsorbed width at `w` query rows; the LSE reduce-scatter is
        # 512-wide latent partials, base 2, at every query-row count.
        transport.prepare_gather(dsession, 1, absorbed)
        for w in DECODE_ROWS:
            # consult18 P4: the decode-family verify rows (and batched decode
            # rows) gather absorbed queries and merge latent partials, like the
            # one-row decode step
            transport.prepare_gather(session, w, absorbed)
            transport.prepare_lse(session, w, values, lse_base="base2")
        for w in PREFILL_WIDTHS:
            if _skip_above is not None and w >= _skip_above:
                continue    # Step 5a: rides the CE exchange — no A2A plan/pool rows
            transport.prepare_gather(session, w, unabsorbed)
        if expanded is None:
            for w in (1, *PREFILL_WIDTHS):
                if _skip_above is not None and w >= _skip_above:
                    continue    # Step 5a: rides the CE exchange
                transport.prepare_lse(dsession if w == 1 else session, w, values, lse_base="base2")
        else:
            # Decode merges latents at one row; prefill merges expanded values.
            transport.prepare_lse(dsession, 1, values, lse_base="base2")
            for w in PREFILL_WIDTHS:
                if _skip_above is not None and w >= _skip_above:
                    continue    # Step 5a: rides the CE exchange
                transport.prepare_lse(session, w, expanded, lse_base="base2")
        # consult15 §2 lever 1: one b12x PCIe all-reduce plan per body row
        # shape (decode 1 row + every prepared prefill width) — oneshot at 1
        # row, rsag above (c1-unit-costs.md §1's arm choice).
        # round 16: spec verify rows outside the prefill widths (e.g. k=4 -> 5, k=5 -> 6).
        # DFly2 hidden_correction also runs all_reduce on `slots = size // per`
        # across every prepared `per` up to `max(4, spec_k + 1)`.
        _spec_k_env = int(os.environ.get("HY4_SPEC_K") or 0) if MTP else 0
        _bspec_pers = tuple(range(2, max(4, _spec_k_env + 1) + 1))
        _dfly_slot_rows = (tuple(sorted({b // per for b in BATCH_ROWS
                                         for per in _bspec_pers if b >= per}))
                           if MTP and _has_dfly else ())
        EXTRA_ROWS = tuple(sorted(set(w for w in (*DECODE_ROWS, *_dfly_slot_rows)
                                      if w not in PREFILL_WIDTHS)))
        for w in (1, *PREFILL_WIDTHS, *EXTRA_ROWS):
            transport.prepare_all_reduce(dsession if w == 1 else session, w)
        if uses_dma and args.dma_replay:
            graphs = transport.prepare_dma_replay()
            trace(rank, f"all-reduce: {graphs} ring replay graph(s) captured")
        if args.overlap:
            transport.prepare_lane_all_reduce(session, 1, CHUNK)
            trace(rank, f"overlap: lane 1 ring + CE exchange prepared for {CHUNK} rows")
        if score_offer is not None:
            # consult17: the offer gather at every row count the body runs — and the
            # batched speculative rounds' sampling / acceptance rows (`slots * per` and `slots`).
            _bspec_offer_rows = ({per * (b // per) for b in BATCH_ROWS
                                  for per in _bspec_pers if b >= per}
                                 if MTP else set())
            for w in sorted({1, *PREFILL_WIDTHS, *EXTRA_ROWS, 6, *_bspec_offer_rows}):
                transport.prepare_score_gather(dsession if w == 1 else session, w)
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
        # c8 regress-fix: the captured-graph budget (KEEP graphs x ~10 MiB of graph-exec
        # memory each, measured) comes out of the pool too — captured AFTER the pool,
        # it used to eat the headroom the prefill's overlapped pairs need (127
        # allocator retries per 64k prefill, prefill -30%)
        graph_reserve = int(float(os.environ.get("HY4_GRAPH_RESERVE_MB") or 2688) * 2**20)
        budget = free_a + cache_a - args.headroom - graph_reserve
        # cache bytes ~= rows * (capacity / world) * (78*656 + 21*256 real residency)
        # Legacy MTP adds one persistent KV/index layer; DFly2 reuses target KV slots (0 extra).
        _draft_kv = int(MTP and not _has_dfly)
        per_global = args.rows * ((78 + _draft_kv) * 656 + (21 + _draft_kv) * 256) / world
        capacity = args.capacity if args.capacity > 0 else int(budget // per_global) & ~1023
        trace(rank, f"phase A: fixed(book+ctx+runtime)={fixed_bytes} B, torn-down free={free_post} B, "
                    f"headroom={args.headroom} B + graph reserve {graph_reserve} B -> capacity "
                    f"{capacity} global/row")
        if capacity < 2 * world:
            raise RuntimeError(f"memory budget {budget} B cannot hold a usable KV cache")

        # ---- phase B: the serving construction at the measured capacity ----
        built, shapes, decode_shape, load_s = construct_and_wire(args.rows, capacity)
        ctx.decoder = built.decoder
        ctx.decode_shape = decode_shape
        ctx.shapes = shapes
        ctx.row_capacity = capacity
        # c8 L6: per-row extents of the pool. HY4_BIG_ROW (global tokens) gives row 0
        # an extent that long; the other rows split the rest (uniform otherwise).
        cache = ctx.decoder.cache
        big = int(os.environ.get("HY4_BIG_ROW") or 0)
        local_total = args.rows * cache.local_capacity
        # c8 L6b: dynamic extents — any row grows (relocating if it must) while the
        # pool holds; the boot keeps the uniform layout, the serving loop starts empty
        ctx.dyn_extents = os.environ.get("HY4_DYN_EXTENTS") == "on" and args.rows > 1
        if ctx.dyn_extents:
            cache.cap_limit = min(-(-int(args.max_context) // world // 1024) * 1024, local_total)
            big = 0
        if big > capacity and args.rows > 1:
            big_local = min(-(-big // world // 1024) * 1024, local_total - (args.rows - 1) * 4096)
            small = (local_total - big_local) // (args.rows - 1) // 1024 * 1024
            cache.set_extents([big_local] + [small] * (args.rows - 1))
        ctx.row_caps = ([cache.cap_limit * world] * args.rows if ctx.dyn_extents
                        else [cache.cap_of(r) * world for r in range(args.rows)])
        ctx.row_capacity = max(ctx.row_caps)
        trace(rank, f"row extents: {sorted(set(ctx.row_caps), reverse=True)} global tokens "
                    f"(row 0 {ctx.row_caps[0]}, rows 1+ {min(ctx.row_caps)}), pool "
                    f"{local_total} local slots/rank")
        # one request lives in one cache row: its context cannot pass the row (less a
        # margin: speculative rounds with lookahead write pad/draft entries past P)
        _spec_margin = max(8, 2 * (_spec_k_env + 2))
        ctx.max_context = min(int(args.max_context), int(ctx.row_capacity) - _spec_margin)
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
        real_agg = args.rows * capacity
        binding_facts["real_aggregate_global_tokens"] = real_agg
        # One-third rule only when total KV < max context (owner 2026-11-20).
        real_applies = real_agg < 1_048_576
        binding_facts["real_one_third_rule_applies"] = real_applies
        binding_facts["real_one_third_cap"] = real_agg // 3 if real_applies else None
        binding_facts["real_one_third_cap_holds_1m"] = (
            (real_agg // 3 >= 1_048_576) if real_applies else None)
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
                "mtp": MTP, "max_concurrency": args.max_concurrency,
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
                "expert_scale_compression": args.expert_scale_compression,
                "forward_cost_rows": ctx.forward_cost_rows,
                "engine_path": ("ignition/recipes/hy4 construct_rank + PcieDcpTransport "
                                "(w14 bring-up path); recipe.prepare's native export TODO remains"),
            }
            os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
            with open(args.report, "w") as fh:
                json.dump(report, fh, indent=2)
                fh.write("\n")
            trace(rank, f"wrote {args.report}")

        from ignition.recipes.hy4 import runtime as _runtime
        if _runtime.SWITCHES.get("decode_graph") == "on" or os.environ.get("HY4_AB_FILE"):
            # consult18 P2: CUDA graphs of the one-row decode step, armed once on
            # the serving construction after every boot capture/teardown. The
            # short bucket and the width buckets up to HY4_DECODE_GRAPH_PREWARM
            # tokens are captured now and replayed once (a graph that cannot
            # replay fails the boot, not a request); longer contexts capture
            # their bucket on first use.
            from ignition.recipes.hy4.decode_graph import DecodeGraphs
            _keep = int(os.environ.get("HY4_GRAPH_KEEP") or DecodeGraphs.KEEP)
            ctx.decoder.decode_graphs = DecodeGraphs(ctx.decoder, keep=_keep)
            upto = int(os.environ.get("HY4_DECODE_GRAPH_PREWARM") or 65536)
            if upto > 0:
                t_cap = time.monotonic()
                buckets = ctx.decoder.decode_graphs.prewarm(max_context=upto,
                                                            call_shape=ctx.decode_shape)
                replayed = ctx.decoder.decode_graphs.replay_all()
                ctx.decoder.release([0])
                trace(rank, f"decode graphs: {len(buckets)} buckets {buckets} captured up to "
                            f"{upto} tokens and replayed ({replayed}) in "
                            f"{time.monotonic() - t_cap:.1f} s")
            else:
                trace(rank, "decode graphs: armed (one capture per width bucket on first use)")
            spec_k = int(os.environ.get("HY4_SPEC_K") or 0)
            if MTP and spec_k > 0:
                # consult18 P4: MTP speculation (greedy requests, runtime switch
                # "spec"); its draft / verify graphs are captured per width
                # bucket up to the decode prewarm context, now.
                from ignition.recipes.hy4.spec import SpeculativeGreedy
                ctx.spec = SpeculativeGreedy(ctx.decoder, k=spec_k, decode_rows=bool(SPEC_ROWS))
                if upto > 0:
                    t_cap = time.monotonic()
                    # c8 L18: the captured-graph budget — the solo loop's unfiltered and
                    # greedy graphs up to HY4_SPEC_PREWARM tokens at boot; filtered
                    # requests and longer contexts capture on first use
                    spec_upto = int(os.environ.get("HY4_SPEC_PREWARM") or 16384)
                    captured = ctx.spec.prewarm(max_context=min(upto, spec_upto), filtered=False)
                    replayed = ctx.decoder.decode_graphs.replay_all()
                    ctx.decoder.release([0])
                    trace(rank, f"spec graphs: k={spec_k} decode_rows={bool(SPEC_ROWS)}, "
                                f"{captured} captured up to {upto} "
                                f"tokens, all graphs replayed ({replayed}) in "
                                f"{time.monotonic() - t_cap:.1f} s")

        graphs = getattr(ctx.decoder, "decode_graphs", None)
        if graphs is not None and args.batch_rows > 1:
            # batched decode: the batch plan group is shared by its buckets and
            # primed at its first call's rows, so the widest bucket runs first;
            # every bucket's short-context step is captured now (collective).
            from ignition.recipes.hy4.wiring import batch_shape
            t_cap = time.monotonic()
            # Both regimes prime their own plan variants: the short-context bypass
            # (with replicated short KV: the local-head, sink-bound plans) and the
            # context-parallel path past top-k (the sinkless partial plans).
            past = (SHORT_POSITIONS if SHORT_KV else int(cfg_topk)) + 8   # the cp regime
            # the local regime (replicated short KV past top-k): its one bucket
            bases = (0, int(cfg_topk) + 8, past) if SHORT_KV and SHORT_POSITIONS > int(cfg_topk) \
                else (0, past)
            _plain_buckets = [b for b in sorted(BATCH_ROWS, reverse=True)
                              if b in (2, 4, 8, 16, max(BATCH_ROWS))] + [1]
            for base in bases:
                for b in _plain_buckets:
                    shape = ctx.decode_shape if b == 1 else batch_shape(b)
                    # wider than the cache rows: wrap the rows, each wrapped entry
                    # at its own position
                    rows_p = [i % args.rows for i in range(b)]
                    starts_p = [base + i // args.rows for i in range(b)]
                    for _ in range(2):            # eager + capture, then one replay
                        graphs.run_batch([0] * b, rows_p, starts_p, call_shape=shape)
            if MTP and getattr(ctx, "spec", None) is not None:
                # batched speculation: the draft attention's batch plan group is
                # primed by its first call too — widest first, draft and verify
                _has_dfly_dec = bool(getattr(ctx.decoder, "has_dfly", False))
                _spec_k_live = int(getattr(ctx.spec, "k", 3) or 3)
                _prewarm_pers = (tuple(range(3, max(4, _spec_k_live + 1) + 1))
                                 if _has_dfly_dec else (3, 4))
                _dfly_multi_pairs = set()
                if _has_dfly_dec:
                    _multi_cap = max(12, int(os.environ.get("HY4_BSPEC_MULTI_CAP") or 24))
                    _top_k = min(_spec_k_live, int(getattr(graphs, "bspec_kmax", _spec_k_live)))
                    for _cap_i in sorted({12, 16, _multi_cap}):
                        for _B in range(1, args.rows + 1):
                            _chosen = 1
                            for _K in range(_top_k, 1, -1):
                                _need = (_K + 1) * _B
                                _cands = [b for b in BATCH_ROWS if _need <= b <= _cap_i]
                                if _cands:
                                    _b = min(_cands)
                                    if _b <= 12 or _b - _need <= max(2, _B // 2):
                                        _chosen = _K
                                        break
                                if _K == _top_k and _need <= _cap_i and _need in BATCH_ROWS:
                                    _chosen = _K
                                    break
                            if _chosen > 1:
                                _p = _chosen + 1
                                _sz = graphs.batch_bucket(_p * _B, allowed=BATCH_ROWS)
                                _dfly_multi_pairs.add((_sz, _p))
                for base in bases:
                    for b in sorted((x for x in BATCH_ROWS if x >= 2), reverse=True):
                        shape = batch_shape(b)
                        rows_p = [(i // 2) % args.rows for i in range(b)]
                        for r in range(args.rows):     # the device round state: near `base`
                            graphs.bspec_set(r, P=base + 2 * r, c=base + 2 * r, n=0)
                        end = base + 2 * args.rows + 3
                        for _ in range(2):
                            graphs.run_bspec("draft", rows_p, [False] * b, end=end, call_shape=shape)
                            graphs.run_bspec("verify", rows_p, [False] * b, end=end, call_shape=shape)
                    if 12 in BATCH_ROWS or (_has_dfly_dec and any(b >= _spec_k_live + 1 for b in BATCH_ROWS)):
                        # c8 L13: multi-draft bspec rounds (per = K + 1 entries/row).
                        end = base + 2 * args.rows + max(_prewarm_pers) + 2
                        for per in _prewarm_pers:
                            if _has_dfly_dec:
                                _bs_buckets = [b for b in sorted(BATCH_ROWS, reverse=True)
                                               if (b, per) in _dfly_multi_pairs]
                            else:
                                _bs_buckets = [b for b in sorted(BATCH_ROWS, reverse=True)
                                               if b >= per and b in (12, 8, 4)]
                            for b in _bs_buckets:
                                shape = batch_shape(b)
                                live = (b // per) * per
                                rows_p = ([(i // per) % args.rows for i in range(live)]
                                          + [0] * (b - live))
                                pads_p = [False] * live + [True] * (b - live)
                                for _ in range(2):
                                    graphs.run_bspec("draft", rows_p, pads_p, end=end,
                                                     call_shape=shape, per=per)
                                    graphs.run_bspec("verify", rows_p, pads_p, end=end,
                                                     call_shape=shape, per=per)
                        if not _has_dfly_dec and 12 in BATCH_ROWS:
                            for step in (1, 2):
                                for b in (4, 2):
                                    shape = batch_shape(b)
                                    rows_p = [i % args.rows for i in range(b)]
                                    for _ in range(2):
                                        graphs.run_bspec("chain", rows_p, [False] * b, end=end,
                                                         call_shape=shape, step=step)
                _has_dfly_dec = bool(getattr(ctx.decoder, "has_dfly", False))
                _spec_k_live = int(getattr(ctx.spec, "k", 3) or 3)
                _solo_k = _spec_k_live if (_has_dfly_dec and _spec_k_live > 3) else 3
                _solo_per = _solo_k + 1
                _solo_b = (graphs.batch_bucket(_solo_per, allowed=BATCH_ROWS)
                           if (12 in BATCH_ROWS or (_has_dfly_dec and any(b >= _solo_per for b in BATCH_ROWS)))
                           and any(b >= _solo_per for b in BATCH_ROWS) else None)
                if _solo_b is not None:
                    # c8 regress-fix: a lone long request decodes in K-draft rounds of one
                    # row; its context-parallel width buckets up to HY4_BSPEC_PREWARM
                    # tokens are captured now (a first-use capture mid-request stalls
                    # its stream ~1-2 s: the 128k bench cell crossed into a new bucket)
                    upto_b = int(os.environ.get("HY4_BSPEC_PREWARM") or 262144)
                    w, seen = graphs.index_width(past + 1), set()
                    solo_pads = [False] * _solo_per + [True] * (_solo_b - _solo_per)
                    while w * world <= upto_b and w not in seen:
                        seen.add(w)
                        P = w * world - 16
                        graphs.bspec_set(0, P=P, c=P - 1, n=0)
                        end = P + _solo_per + 2
                        for _ in range(2):
                            graphs.run_bspec("draft", [0] * _solo_b, solo_pads, end=end,
                                             call_shape=batch_shape(_solo_b), per=_solo_per)
                            if not _has_dfly_dec:
                                for step in (1, 2):
                                    graphs.run_bspec("chain", [0, 0], [False, True], end=end,
                                                     call_shape=batch_shape(2), step=step)
                            graphs.run_bspec("verify", [0] * _solo_b, solo_pads, end=end,
                                             call_shape=batch_shape(_solo_b), per=_solo_per)
                        w = graphs.index_width(w * world + 1)
                    trace(rank, f"batch graphs: one-row K={_solo_k} rounds (bucket {_solo_b}) primed "
                                f"at {len(seen)} cp widths up to {upto_b} tokens")
                for r in range(args.rows):
                    graphs.bspec_set(r, P=0, c=0, n=0)
            torch.cuda.synchronize(device)
            ctx.decoder.release(list(range(args.rows)))
            free_g, _ = torch.cuda.mem_get_info(device)
            trace(rank, f"batch graphs: buckets {sorted(BATCH_ROWS) + [1]} primed in "
                        f"{time.monotonic() - t_cap:.1f} s; {len(graphs.graphs)} graphs, free "
                        f"{free_g / 2**30:.2f} GiB (headroom {args.headroom / 2**30:.2f} GiB)")
        ctx.batch_rows = BATCH_ROWS
        ctx.cache_rows = args.rows
        if getattr(ctx, "dyn_extents", False):
            # c8 L6b: every row's extent is granted at admission from here on
            ctx.decoder.cache.set_extents([0] * args.rows, [0] * args.rows)
            trace(rank, f"dynamic extents: pool {args.rows * ctx.decoder.cache.local_capacity} local "
                        f"slots/rank, a row up to {ctx.decoder.cache.cap_limit} "
                        f"({ctx.decoder.cache.cap_limit * world} tokens)")

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
            _tc = tok.encode(TOOLCALLS_BEGIN, add_special_tokens=False)
            ctx.toolcalls_token = _tc[0] if len(_tc) == 1 else None
            _to = tok.encode(THINK_OPEN, add_special_tokens=False)
            _tcl = tok.encode(THINK_CLOSE, add_special_tokens=False)
            if len(_to) != 1 or len(_tcl) != 1:
                raise RuntimeError("the think markers must be single tokens for the reasoning split")
            ctx.think_open_token, ctx.think_close_token = _to[0], _tcl[0]
            eos_id = tok.eos_token_id if tok.eos_token_id is not None else 120025
            ctx.sampler = Sampler(tok, eos_id, seed=int(time.time()) & 0xFFFFFFFF, device=ctx.device)
            serve_http(ctx)
            trace(rank, f"listening on {ctx.bind} as {ctx.served_names[0]}")
        loop = ctx.loop = BatchLoop(ctx)      # dev scripts may flip its toggles
        while loop.iterate():
            pass
        trace(rank, "shutdown")
    finally:
        try:
            dist.destroy_process_group()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
