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
        if len(self.graphs) >= self.keep:
            self.uncaptured += 1
            return None
        graph = torch.cuda.CUDAGraph()
        with contextlib.ExitStack() as stack:
            self._comm_capture_contexts(stack)
            with torch.cuda.graph(graph, pool=self.pool):
                logits, flag = body()
        self.graphs[key] = (graph, logits, flag)
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
                "buckets": [k[1] if k[0] != "tail" else ("tail", k[2], k[3]) for k in self.graphs]}
