"""consult18 P2: CUDA graphs of the one-row decode step.

Why (c1-evidence/w29-c18-p2-decode-skew.txt): at 16k c1 the collectives'
wire cost is ~3 ms per step, while ~153 ms per step is spent waiting for the
last-arriving rank. Every rank's host launches ~9.8k kernels per step, and
the step advances at the pace of the slowest HOST at each of its ~339
collectives. A captured step takes the host off that path: a device copy of
the token, one pinned H2D of the step's scalars, one graph launch, then the
logits gather and the deferred-flag read.

Shapes. In the one-row step everything has a fixed shape except the
indexer's width (this rank's owned live keys, which grow by one every
`world` tokens). The step is captured per width BUCKET (`index_width`): the
indexer scores the bucket's key slots, and the device-side causal mask
(`owned`, from the device position) drops the columns past the live ones, so
one graph serves every position in its bucket. A bucket's first step runs
eagerly through the same body (`_body`), which warms every lazy path at
those shapes; the bucket is then captured, and later steps replay it.

KV writes are predicated on device. On a rank that does not own the token
its record slot is -1 (the fused writer skips negative slots), and the
index-key store rewrites a valid slot with its own value
(`Hy4Cache.begin_forward_static` / `_store`).

Collectives: the b12x DCP pools are single-channel, so the graphs and the
eager calls share one channel per pool. A pool's first capture switches it
to device-side slot selection (eager calls follow; the two launcher
variants were prepared at boot). The one-shot all-reduce channel captures
through its own `capture()` context (graph-buffer registration on exit).
Capture is collective: the bucket is a function of the global position
only, so every rank captures the same bucket at the same step.

Scope: one request row, one token, no MTP state, logits "all"/"last".
Contexts up to index_topk take the short-context bypass (every visible owned
key, filled on device by `SelectionCell.fill_visible`): one more bucket
(width 0) serves all of them.
"""
import collections
import contextlib

import torch


class DecodeGraphs:
    """Captured one-row decode steps, one per (row, width bucket, logits form)."""

    GRANULE = 1024     # owned keys per width bucket
    # Captured graphs are NEVER destroyed: every boot that destroyed one (the
    # phase-A teardown, dg3; LRU eviction, dg11) then segfaulted in
    # cuGraphLaunch replaying another graph of the shared pool; boots that
    # never destroyed one replayed thousands of steps cleanly. Past KEEP
    # graphs a new key runs eagerly (uncaptured) instead of evicting.
    KEEP = 256

    def __init__(self, decoder, *, granule=None, keep=None):
        self.decoder = decoder
        self.device = decoder.device
        self.granule = int(granule or self.GRANULE)
        self.keep = int(keep or self.KEEP)
        self.world = int(getattr(decoder.collectives, "world_size", 1))
        self.rank = int(getattr(decoder.collectives, "rank", 0))
        self.topk = int(decoder.config.text["index_topk"])
        self.local_capacity = int(decoder.cache.local_capacity)
        self.capacity = int(decoder.cache.capacity)
        self._token = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        self._host = torch.zeros((3,), dtype=torch.long)
        if self.device.type == "cuda":
            self._host = self._host.pin_memory()
        self._args = torch.zeros((3,), dtype=torch.long, device=self.device)
        self._tail_tokens = {}           # T -> [1, T] static token buffer
        self._tail_host = torch.zeros((1,), dtype=torch.long)
        if self.device.type == "cuda":
            self._tail_host = self._tail_host.pin_memory()
        self._tail_start = torch.zeros((1,), dtype=torch.long, device=self.device)
        self.graphs = collections.OrderedDict()
        self.pool = None
        self.captures = self.replays = self.eager_steps = self.uncaptured = 0
        self.capture_s = 0.0

    # -- shapes ---------------------------------------------------------
    def index_width(self, end):
        """The indexer width for a step whose context ends at `end` (exclusive):
        rank 0's owned keys (the widest rank) rounded up to the granule."""
        owned = (int(end) + self.world - 1) // self.world
        width = -(-owned // self.granule) * self.granule
        return min(width, self.local_capacity)

    def accepts(self, token_ids, *, row, start, call_shape, state, return_hidden, logits_rows):
        return (token_ids.is_cuda and tuple(token_ids.shape) == (1, 1) and state is None
                and not return_hidden and logits_rows in ("all", "last")
                and call_shape.is_decode_graph
                and (start + 1 > self.topk or self._short_bypass(call_shape))
                and start + 1 <= self.capacity
                and (start // self.world) < self.local_capacity
                and 0 <= row < int(self.decoder.cache.rows))

    def _short_bypass(self, call_shape):
        """Contexts up to top-k take the short-context bypass (every visible
        owned key, no indexer) — one fixed-shape graph for all of them."""
        return self.world > 1 and call_shape.short_context_bypass

    def bucket(self, start):
        """0 = the short-context bypass, else the indexer width bucket."""
        return 0 if start + 1 <= self.topk else self.index_width(start + 1)

    # -- the step body (eager or under capture) -------------------------
    def _stage(self, token_ids, row, start):
        """Load the step's inputs into the static buffers (no host read)."""
        self._token.copy_(token_ids)
        local = row * self.local_capacity + start // self.world
        self._host[0] = start
        self._host[1] = local if start % self.world == self.rank else -1
        self._host[2] = local
        self._args.copy_(self._host, non_blocking=True)

    def _body(self, row, call_shape, width, logits_rows):
        from .lanes import use_lane

        d = self.decoder
        positions = self._args[0:1]
        rows = torch.full((1,), row, dtype=torch.long, device=self.device)
        token_rows = d._token_rows((1, 1), rows)
        visible = positions + 1
        d.cache.begin_forward_static(token_rows, positions, slot=self._args[1:2],
                                     safe=self._args[2:3])
        geometry = {"owned_max": max(width, 1), "base": row * self.local_capacity}
        deferred = getattr(d.collectives, "deferred_checks", None)
        if deferred is not None:
            deferred.clear()
        # The bypass test reads only this bound: width 0 is the short-context
        # bucket (<= top-k), any other bucket is past top-k (accepts()).
        visible_max = self.topk if width == 0 else self.topk + 1
        with use_lane(0, overlap=False, geometry=geometry, sync_free=True):
            logits = d._layers_and_head(
                self._token, rows, positions, token_rows, visible, visible_max, state=None,
                return_hidden=False, call_shape=call_shape, logits_rows=logits_rows,
                gather=False)
        flags = list(deferred) if deferred else []
        if deferred is not None:
            deferred.clear()
        if flags:
            flag = torch.stack([f.reshape(()) for f in flags]).any()
        else:
            flag = torch.zeros((), dtype=torch.bool, device=self.device)
        return logits, flag

    # -- single-row prefill forwards of T tokens (the prompt's tail) -------
    TAIL_MAX = 512

    def accepts_tail(self, token_ids, *, row, start, call_shape, state, return_hidden):
        """A one-row forward of T tokens, T a multiple of the world (every rank
        owns exactly T / world of them: fixed-shape KV writes), T <= TAIL_MAX
        (sub-chunk forwards: kernel collectives, no copy-engine exchange)."""
        width = int(token_ids.shape[1])
        return (token_ids.is_cuda and int(token_ids.shape[0]) == 1 and width > 1
                and width % self.world == 0 and width <= self.TAIL_MAX
                and state is None and not return_hidden and call_shape.is_prefill_graph
                and start + width <= self.capacity
                and (start + width - 1) // self.world < self.local_capacity
                and 0 <= row < int(self.decoder.cache.rows))

    def _stage_tail(self, token_ids, start):
        width = int(token_ids.shape[1])
        buf = self._tail_tokens.get(width)
        if buf is None:
            buf = self._tail_tokens[width] = torch.zeros((1, width), dtype=torch.long,
                                                         device=self.device)
        buf.copy_(token_ids)
        self._tail_host[0] = start
        self._tail_start.copy_(self._tail_host, non_blocking=True)
        return buf

    def _body_tail(self, row, call_shape, width, tokens, need_logits):
        from .lanes import use_lane

        d = self.decoder
        count = int(tokens.shape[1])
        start = self._tail_start
        positions = start + torch.arange(count, dtype=torch.long, device=self.device)
        rows = torch.full((1,), row, dtype=torch.long, device=self.device)
        token_rows = d._token_rows((1, count), rows)
        visible = positions + 1
        # begin_forward_host's (kept, slots) from the device start: the owned
        # tokens are i = first + world * j, first = (rank - start) mod world.
        per = count // self.world
        first = torch.remainder(self.rank - start, self.world)
        j = torch.arange(per, dtype=torch.long, device=self.device)
        kept = first + self.world * j
        slots = row * self.local_capacity + torch.div(start + first, self.world,
                                                      rounding_mode="floor") + j
        d.cache.begin_forward_device(token_rows, positions, kept, slots)
        geometry = {"owned_max": max(width, 1), "base": row * self.local_capacity}
        deferred = getattr(d.collectives, "deferred_checks", None)
        if deferred is not None:
            deferred.clear()
        with use_lane(0, overlap=False, geometry=geometry, sync_free=True):
            logits = d._layers_and_head(
                tokens, rows, positions, token_rows, visible, None, state=None,
                return_hidden=False, call_shape=call_shape,
                logits_rows="all" if need_logits else "none", gather=False)
        flags = list(deferred) if deferred else []
        if deferred is not None:
            deferred.clear()
        if flags:
            flag = torch.stack([f.reshape(()) for f in flags]).any()
        else:
            flag = torch.zeros((), dtype=torch.bool, device=self.device)
        return logits, flag

    def run_tail(self, token_ids, *, row, start, call_shape, need_logits):
        """One single-row forward of T tokens: `(local-shard logits [1, T, V/w]
        or None, deferred flag)` — captured per (T, indexer width bucket)."""
        count = int(token_ids.shape[1])
        width = self.index_width(start + count)
        key = ("tail", int(row), count, width, call_shape.name, bool(need_logits))
        tokens = self._stage_tail(token_ids, start)
        entry = self.graphs.get(key)
        if entry is None:
            logits, flag = self._body_tail(row, call_shape, width, tokens, need_logits)
            self.eager_steps += 1
            logits = None if logits is None else logits.clone()
            flag = flag.clone()
            self._capture(key, lambda: self._body_tail(row, call_shape, width, tokens, need_logits))
            return logits, flag
        graph, logits, flag = entry
        self.graphs.move_to_end(key)
        graph.replay()
        self.replays += 1
        return logits, flag

    # -- consult18 P4: speculative verify + MTP draft forwards -------------
    # A verify forward is T = k + 1 target tokens of one row; a draft forward
    # is T rows of the MTP block (T = 1: a chained draft step; T > 1: the
    # re-absorption of the accepted rows with the target's hidden). T < world
    # in practice, so a rank owns 0 or 1 of the tokens: the writes are
    # PREDICATED — per = ceil(T / world) (kept, slot, safe) entries, slot -1
    # where the rank owns no token (the record writer skips it; the index-key
    # store rewrites a distinct valid slot with its own value). Both return
    # greedy tokens computed on device (`argmax_tokens`) and the hidden.
    SPEC_MAX = 8
    SPEC_RING = 64

    def spec_per(self, count):
        return -(-int(count) // self.world)

    def accepts_spec(self, count, *, row, start, call_shape):
        """Fixed-shape spec forwards past the short-context bypass, with room
        for the predicated slots of every rank (a rank-independent bound)."""
        per = self.spec_per(count)
        return (self.device.type == "cuda" and 1 <= count <= self.SPEC_MAX
                and start + 1 > self.topk and start + count <= self.capacity
                and (start + self.world - 1) // self.world + per - 1 < self.local_capacity
                and 0 <= row < int(self.decoder.cache.rows)
                and call_shape.is_decode_graph == (count == 1))

    def _spec_state(self, kind, count):
        key = (kind, int(count))
        state = getattr(self, "_spec", None)
        if state is None:
            state = self._spec = {}
            self._spec_ring = []
            self._spec_ring_at = 0
        entry = state.get(key)
        if entry is None:
            d = self.decoder
            hidden = int(d.config.hidden)
            per = self.spec_per(count)
            entry = state[key] = {
                "tokens": torch.zeros((1, count), dtype=torch.long, device=self.device),
                "hidden": torch.zeros((1, count, hidden), dtype=d.compute_dtype,
                                      device=self.device),
                "args": torch.zeros((1 + 3 * per,), dtype=torch.long, device=self.device),
                "per": per,
                # Outputs live OUTSIDE the graph pool: graphs sharing a pool reuse
                # each other's freed intermediates, so a graph's own outputs are
                # overwritten by the next replay of a graph captured before it
                # (dg24: the re-absorb's outputs clobbered by the verify).
                "out_tokens": torch.zeros((count if kind == "verify" else 1,), dtype=torch.long,
                                          device=self.device),
                "out_hidden": torch.zeros((count, hidden), dtype=d.compute_dtype,
                                          device=self.device),
                "out_flag": torch.zeros((), dtype=torch.bool, device=self.device),
            }
        return entry

    def spec_inputs(self, kind, count):
        """The static (tokens [1, T], hidden [1, T, H]) inputs of a spec forward
        (`hidden` feeds draft forwards only); fill them before `run_spec`."""
        entry = self._spec_state(kind, count)
        return entry["tokens"], entry["hidden"]

    def _spec_host(self, n):
        """A pinned staging vector from a ring: a round stages several forwards
        before its one host sync, so a buffer is never rewritten while its
        asynchronous H2D copy may still be pending."""
        if self._spec_ring_at >= len(self._spec_ring):
            buf = torch.zeros((1 + 3 * self.spec_per(self.SPEC_MAX),), dtype=torch.long)
            if self.device.type == "cuda":
                buf = buf.pin_memory()
            self._spec_ring.append(buf)
        buf = self._spec_ring[self._spec_ring_at]
        self._spec_ring_at = (self._spec_ring_at + 1) % self.SPEC_RING
        return buf[:n]

    def _stage_spec(self, entry, row, start, count):
        per = entry["per"]
        first = (self.rank - start) % self.world
        base = row * self.local_capacity + (start + first) // self.world
        host = self._spec_host(1 + 3 * per)
        host[0] = start
        for j in range(per):
            i = first + self.world * j
            owned = i < count
            host[1 + j] = i if owned else 0
            host[1 + per + j] = base + j if owned else -1
            host[1 + 2 * per + j] = base + j
        entry["args"].copy_(host, non_blocking=True)

    def _spec_slots(self, entry):
        per = entry["per"]
        args = entry["args"]
        return args[1:1 + per], args[1 + per:1 + 2 * per], args[1 + 2 * per:1 + 3 * per]

    def _body_verify(self, row, call_shape, width, count):
        from .lanes import use_lane

        d = self.decoder
        entry = self._spec_state("verify", count)
        tokens = entry["tokens"]
        positions = entry["args"][0:1] + torch.arange(count, dtype=torch.long, device=self.device)
        rows = torch.full((1,), row, dtype=torch.long, device=self.device)
        token_rows = d._token_rows((1, count), rows)
        visible = positions + 1
        kept, slot, safe = self._spec_slots(entry)
        d.cache.begin_forward_static(token_rows, positions, slot=slot, safe=safe, kept=kept)
        geometry = {"owned_max": max(width, 1), "base": row * self.local_capacity}
        deferred = getattr(d.collectives, "deferred_checks", None)
        if deferred is not None:
            deferred.clear()
        visible_max = self.topk + 1 if call_shape.is_decode_graph else None
        with use_lane(0, overlap=False, geometry=geometry, sync_free=True):
            logits, normed, _ = d._layers_and_head(
                tokens, rows, positions, token_rows, visible, visible_max, state=None,
                return_hidden=True, call_shape=call_shape, logits_rows="all", gather=False)
            greedy = d.greedy_tokens(logits.reshape(count, -1))
        return self._emit(entry, greedy, normed.reshape(count, -1), self._flag(deferred))

    def _body_draft(self, row, call_shape, width, count):
        from .lanes import use_lane

        d = self.decoder
        entry = self._spec_state("draft", count)
        tokens, hidden_in = entry["tokens"], entry["hidden"]
        positions = entry["args"][0:1] + torch.arange(count, dtype=torch.long, device=self.device)
        rows = torch.full((1,), row, dtype=torch.long, device=self.device)
        geometry = {"owned_max": max(width, 1), "base": row * self.local_capacity}
        deferred = getattr(d.collectives, "deferred_checks", None)
        if deferred is not None:
            deferred.clear()
        visible_max = self.topk + 1 if call_shape.is_decode_graph else None
        with use_lane(0, overlap=False, geometry=geometry, sync_free=True):
            hidden, _, _ = d._draft_cp(tokens, hidden_in, positions, rows, d.cache,
                                       call_shape=call_shape, visible_max=visible_max,
                                       static_slots=self._spec_slots(entry))
            greedy = d.greedy_tokens(hidden[-1:].float() @ d.weights.head.T)
        return self._emit(entry, greedy, hidden.reshape(count, -1), self._flag(deferred))

    @staticmethod
    def _emit(entry, greedy, hidden, flag):
        entry["out_tokens"].copy_(greedy.reshape(-1))
        entry["out_hidden"].copy_(hidden)
        entry["out_flag"].copy_(flag)
        return entry["out_tokens"], entry["out_hidden"], entry["out_flag"]

    def _flag(self, deferred):
        flags = list(deferred) if deferred else []
        if deferred is not None:
            deferred.clear()
        if flags:
            return torch.stack([f.reshape(()) for f in flags]).any()
        return torch.zeros((), dtype=torch.bool, device=self.device)

    def run_spec(self, kind, *, row, start, count, call_shape):
        """One verify ("verify") or MTP ("draft") forward of `count` tokens at
        `start` over the static inputs (`spec_inputs`): `(greedy tokens [T] or
        [1], post-norm hidden [T, H], deferred flag)` — the (kind, T) static
        output buffers, valid until the next forward of the same (kind, T).
        Captured per (kind, T, indexer width bucket); the first call runs
        eagerly through the same body."""
        width = self.index_width(start + count)
        key = (kind, int(row), int(count), width, call_shape.name)
        entry = self._spec_state(kind, count)
        self._stage_spec(entry, row, start, count)
        body = self._body_verify if kind == "verify" else self._body_draft
        graph = self.graphs.get(key)
        if graph is None:
            out = body(row, call_shape, width, count)
            self.eager_steps += 1
            self._capture(key, lambda: body(row, call_shape, width, count))
            return out
        graph, outputs, _ = graph
        self.graphs.move_to_end(key)
        graph.replay()
        self.replays += 1
        return outputs

    # -- capture ----------------------------------------------------------
    def _comm_capture_contexts(self, stack):
        """Enter every b12x channel's capture context (DCP pools, one-shot)."""
        transport = getattr(self.decoder.collectives, "transport", None)
        if transport is None:
            return
        for pool in getattr(transport, "_pools", {}).values():
            stack.enter_context(pool.capture())
        seen = set()
        for _, target, arm in getattr(transport, "_ar_plans", {}).values():
            if arm in ("oneshot", "pull", "rsag", "twoshot") and id(target) not in seen:
                seen.add(id(target))
                stack.enter_context(target.capture())

    def _capture(self, key, body):
        import time

        began = time.perf_counter()
        if self.pool is None:
            self.pool = torch.cuda.graph_pool_handle()
            # A captured graph holds raw CUfunction handles: from here on no
            # compiled b12x program may be reclaimed (an LRU-evicted module
            # unloads and the next replay segfaults in cuGraphLaunch — dg23).
            try:
                from b12x._lib.compile_plan import pin_programs
            except ImportError:  # b12x without program pinning
                pin_programs = None
            if pin_programs is not None:
                pin_programs()
        if len(self.graphs) >= self.keep:
            self.uncaptured += 1
            return None
        graph = torch.cuda.CUDAGraph()
        with contextlib.ExitStack() as stack:
            self._comm_capture_contexts(stack)
            with torch.cuda.graph(graph, pool=self.pool):
                out = body()
        if len(out) == 2:
            self.graphs[key] = (graph, out[0], out[1])
        else:       # spec forwards: (graph, all outputs, flag)
            self.graphs[key] = (graph, tuple(out), out[-1])
        self.captures += 1
        self.capture_s += time.perf_counter() - began
        return self.graphs[key]

    # -- the step ---------------------------------------------------------
    def run(self, token_ids, *, row, start, call_shape, logits_rows):
        """One decode step: `(local-shard logits, deferred selection flag)`."""
        width = self.bucket(start)
        key = (int(row), width, call_shape.name, logits_rows)
        self._stage(token_ids, row, start)
        entry = self.graphs.get(key)
        if entry is None:
            # The bucket's first step runs eagerly through the same body, then
            # the bucket is captured for the following steps.
            logits, flag = self._body(row, call_shape, width, logits_rows)
            self.eager_steps += 1
            logits, flag = logits.clone(), flag.clone()
            self._capture(key, lambda: self._body(row, call_shape, width, logits_rows))
            return logits, flag
        graph, logits, flag = entry
        self.graphs.move_to_end(key)
        graph.replay()
        self.replays += 1
        return logits, flag

    def prewarm(self, *, max_context, row=0, call_shape, logits_rows="all"):
        """Capture the short bucket and every width bucket up to `max_context`
        at boot (collective: every rank runs the same steps). Each bucket's
        eager first step writes one record of `row` at the bucket's first
        position; the caller releases the row afterwards. Returns the buckets."""
        d = self.decoder
        starts = [min(64, self.topk - 1)] if self._short_bypass(call_shape) else []
        start = self.topk
        while start < min(int(max_context), self.capacity):
            starts.append(start)
            # the first position of the next bucket
            start = (self.index_width(start + 1) * self.world)
        captured = []
        token = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        for start in starts:
            d._row_lengths = getattr(d, "_row_lengths", {})
            d._row_lengths[row] = start
            if not self.accepts(token, row=row, start=start, call_shape=call_shape, state=None,
                                return_hidden=False, logits_rows=logits_rows):
                continue
            self.run(token, row=row, start=start, call_shape=call_shape,
                     logits_rows=logits_rows)
            captured.append(self.bucket(start))
        torch.cuda.synchronize(self.device)
        return captured

    def replay_all(self):
        """Replay every captured bucket once (boot self-test; collective — every
        rank holds the same buckets in the same order). The inputs are the last
        staged step's; the caller releases the row afterwards."""
        for graph, _, _ in self.graphs.values():
            graph.replay()
        torch.cuda.synchronize(self.device)
        return len(self.graphs)

    def stats(self):
        return {"captures": self.captures, "replays": self.replays,
                "eager_steps": self.eager_steps, "uncaptured": self.uncaptured,
                "capture_s": round(self.capture_s, 3),
                "buckets": [k[1] if not isinstance(k[0], str) else (k[0], k[2], k[3])
                            for k in self.graphs]}
