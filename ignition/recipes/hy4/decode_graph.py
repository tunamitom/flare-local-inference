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

from .w8 import decode_scope as w8_decode_scope



def _kept_rows():
    """c8 L8: batched steps' kept = arange(rows) skips its identity index_select."""
    from .runtime import SWITCHES

    return SWITCHES.get("kept_rows") == "on"


class DecodeGraphs:
    """Captured one-row decode steps, one per (row, width bucket, logits form)."""

    GRANULE = 1024     # owned keys per width bucket
    GEOM_FROM = 8192   # c8 L18: geometric width buckets past this many owned keys
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
        self.cache = decoder.cache
        # c8 L6: per-row extents — the widest a row may be bounds the width buckets
        self.local_capacity = int(decoder.cache.max_cap())
        self.capacity = self.local_capacity * self.world
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
        if width > self.GEOM_FROM:
            # c8 L18: past GEOM_FROM owned keys the buckets grow geometrically (four
            # per octave: <= 25% indexer over-scan) — a long context crosses ~24
            # buckets up to 1M instead of 128, within the captured-graph budget
            octave = 1 << (width.bit_length() - 1)
            step = octave // 4
            width = -(-width // step) * step
        return min(width, self.local_capacity)

    def accepts(self, token_ids, *, row, start, call_shape, state, return_hidden, logits_rows):
        return (token_ids.is_cuda and tuple(token_ids.shape) == (1, 1) and state is None
                and not return_hidden and logits_rows in ("all", "last")
                and call_shape.is_decode_graph
                and (start + 1 > self.topk or self._short_bypass(call_shape))
                and 0 <= row < int(self.decoder.cache.rows)
                and (start // self.world) < self.cache.cap_of(row))

    def _short_bypass(self, call_shape):
        """Contexts up to top-k take the short-context bypass (every visible
        owned key, no indexer) — one fixed-shape graph for all of them."""
        return self.world > 1 and call_shape.short_context_bypass

    def bucket(self, start):
        """0 = the short-context bypass, else the indexer width bucket."""
        return 0 if start + 1 <= self.topk else self.index_width(start + 1)

    # -- regimes: bypass (end <= top-k), local (replicated short-context keys
    #    and records: end <= the short capacity), cp (context parallel) --------
    def short_cap(self):
        from .runtime import SWITCHES

        S = int(getattr(self.decoder.cache, "short_capacity", 0) or 0)
        return S if (S and SWITCHES.get("short_kv") == "on") else 0

    def regime(self, end, call_shape, *, bypass_ok=None):
        """`(regime, width)` of a decode-family forward whose context ends at `end`."""
        ok = self._short_bypass(call_shape) if bypass_ok is None else bypass_ok
        if end <= self.topk and ok:
            return "bypass", 0
        S = self.short_cap()
        if S and end <= S and ok:
            # one local bucket (the whole short table): the indexer scores S keys
            # per row (~0.05 ms) and every local context shares one primed graph
            return "local", S
        return "cp", self.index_width(end)

    @staticmethod
    def wkey(regime, width):
        return ("L", width) if regime == "local" else width

    def _geo(self, regime, width, *, rows=None, row=None, count=1):
        """The captured step's host-known geometry and `visible_max` in its regime
        (`visible_max` steers the attention: <= top-k bypass, <= S local, else cp)."""
        S = self.short_cap()
        if regime == "local":
            owned, vmax, scale = width, width, S
        elif regime == "bypass":
            owned, vmax, scale = 1, self.topk, self.local_capacity
        else:
            owned, vmax, scale = max(width, 1), max(self.topk, S) + 1, self.local_capacity
        if rows is not None:
            bases = rows * scale if regime == "local" else self.cache.row_base_dev[rows]
            return {"owned_max": owned, "base": None, "bases": bases}, vmax
        if regime == "local":
            return {"owned_max": owned, "base": row * scale}, vmax
        # c8 L6b: the row's extent base read on device (a relocated row's captured
        # graphs follow it), one per query row
        bases = self.cache.row_base_dev[row].reshape(1).expand(count).contiguous()
        return {"owned_max": owned, "base": None, "bases": bases}, vmax

    # -- the step body (eager or under capture) -------------------------
    def _stage(self, token_ids, row, start):
        """Load the step's inputs into the static buffers (no host read)."""
        self._token.copy_(token_ids)
        local = self.cache.base_of(row) + start // self.world
        self._host[0] = start
        self._host[1] = local if start % self.world == self.rank else -1
        self._host[2] = local
        self._args.copy_(self._host, non_blocking=True)

    def _body(self, row, call_shape, width, logits_rows, regime=None):
        from .lanes import use_lane

        d = self.decoder
        positions = self._args[0:1]
        rows = torch.full((1,), row, dtype=torch.long, device=self.device)
        token_rows = d._token_rows((1, 1), rows)
        visible = positions + 1
        d.cache.begin_forward_static(token_rows, positions, slot=self._args[1:2],
                                     safe=self._args[2:3])
        regime = regime or ("bypass" if width == 0 else "cp")
        geometry, visible_max = self._geo(regime, width, row=row, count=1)
        deferred = getattr(d.collectives, "deferred_checks", None)
        if deferred is not None:
            deferred.clear()
        with use_lane(0, overlap=False, geometry=geometry, sync_free=True), w8_decode_scope():
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
                and state is None and call_shape.is_prefill_graph
                and 0 <= row < int(self.decoder.cache.rows)
                and (start + width - 1) // self.world < self.cache.cap_of(row))

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

    def _body_tail(self, row, call_shape, width, tokens, need_logits, need_hidden=False):
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
        slots = self.cache.base_of(row) + torch.div(start + first, self.world,
                                                    rounding_mode="floor") + j
        d.cache.begin_forward_device(token_rows, positions, kept, slots)
        geometry = {"owned_max": max(width, 1), "base": self.cache.base_of(row)}
        deferred = getattr(d.collectives, "deferred_checks", None)
        if deferred is not None:
            deferred.clear()
        with use_lane(0, overlap=False, geometry=geometry, sync_free=True):
            out = d._layers_and_head(
                tokens, rows, positions, token_rows, visible, None, state=None,
                return_hidden=need_hidden, call_shape=call_shape,
                logits_rows="all" if need_logits else "none", gather=False)
        flags = list(deferred) if deferred else []
        if deferred is not None:
            deferred.clear()
        if flags:
            flag = torch.stack([f.reshape(()) for f in flags]).any()
        else:
            flag = torch.zeros((), dtype=torch.bool, device=self.device)
        if need_hidden:
            # consult18 P4: the post-norm (MTP feed) and pre-norm hidden too
            logits, normed, pre = out
            return logits, normed, pre, flag
        return out, flag

    def run_tail(self, token_ids, *, row, start, call_shape, need_logits, need_hidden=False):
        """One single-row forward of T tokens: `(local-shard logits [1, T, V/w]
        or None, deferred flag)` — captured per (T, indexer width bucket);
        `need_hidden`: `(logits, post-norm hidden [1, T, H], pre-norm hidden,
        flag)` (graph-pool tensors: read them before the next replay)."""
        count = int(token_ids.shape[1])
        width = self.index_width(start + count)
        # c8 L6b: the tail graphs bake the row's base (> 64 query rows: no per-row
        # base indexer) — a relocated row captures its tails anew
        key = ("tail", int(row), count, width, call_shape.name, bool(need_logits),
               self.cache.base_of(row))
        if need_hidden:
            key = key + ("hidden",)
        tokens = self._stage_tail(token_ids, start)
        entry = self.graphs.get(key)

        def body():
            return self._body_tail(row, call_shape, width, tokens, need_logits, need_hidden)

        if entry is None:
            out = body()
            self.eager_steps += 1
            out = tuple(None if x is None else x.clone() for x in out)
            self._capture(key, body)
            return out
        graph, outputs, flag = entry
        self.graphs.move_to_end(key)
        graph.replay()
        self.replays += 1
        return outputs if need_hidden else (outputs, flag)

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
        """Fixed-shape spec forwards, with room for the predicated slots of
        every rank (a rank-independent bound). Inside the short-context bypass
        (the whole forward at <= top-k) the decode-family shapes run the bypass
        bucket; elsewhere the indexer width bucket."""
        per = self.spec_per(count)
        short = start + count <= self.topk
        return (self.device.type == "cuda" and 1 <= count <= self.SPEC_MAX
                and (not short or self._spec_bypass(count, call_shape))
                and start >= 0 and 0 <= row < int(self.decoder.cache.rows)
                and (start + self.world - 1) // self.world + per - 1 < self.cache.cap_of(row)
                and (call_shape.is_decode_graph == (count == 1)
                     or self._decode_rows(call_shape, count)))

    def _spec_bypass(self, count, call_shape):
        """A spec forward that can take the short-context bypass: a decode-family
        shape (one token, or the verification rows) under context parallelism."""
        return (self._short_bypass(call_shape)
                and (call_shape.is_decode_graph or self._decode_rows(call_shape, count)))

    def spec_width(self, start, count, call_shape):
        """`(regime, width)` of a spec forward (0 width: the bypass bucket)."""
        return self.regime(start + count, call_shape, bypass_ok=self._spec_bypass(count, call_shape))

    @staticmethod
    def _decode_rows(call_shape, count):
        """consult18 P4: T > 1 tokens of ONE request row run as the decode-family
        verification shape — T query rows of one token each, all mapped to the
        same cache row at consecutive positions (owner-side absorbed queries,
        the decode attention kernel, the decode MoE op)."""
        return count > 1 and call_shape.name == "verify" and call_shape.rows == count

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

    # consult18 P5: speculative SAMPLING buffers (temperature > 0). Graphs read a
    # static 1/temperature and noise / uniform buffers the host loop refills
    # (outside the graphs, per-rank generators) before each replay.
    def spec_inv_temp(self):
        buf = getattr(self, "_spec_inv_temp", None)
        if buf is None:
            buf = self._spec_inv_temp = torch.ones((1,), dtype=torch.float32, device=self.device)
        return buf

    def spec_filter_params(self):
        """consult18 P7: the request's top-k (int64, 0 off) / top-p (FP32, >= 1 off),
        read by the "filtered" sampling graphs."""
        params = getattr(self, "_spec_filter", None)
        if params is None:
            params = self._spec_filter = (torch.zeros((1,), dtype=torch.long, device=self.device),
                                          torch.ones((1,), dtype=torch.float32, device=self.device))
        return params

    def spec_sample_state(self, kind, count):
        entry = self._spec_state(kind, count)
        if "out_cov" not in entry:
            # per verified row: is its filtered target the exact one (draft rows
            # need no coverage — a draft is exact for whatever q it sampled from)
            entry["out_cov"] = torch.ones((count if kind == "verify" else 1,), dtype=torch.bool,
                                          device=self.device)
        if "noise" not in entry:
            vocab = int(self.decoder.weights.head.shape[0])
            rows = count if kind == "verify" else 1
            entry["noise"] = torch.zeros((rows, vocab), dtype=torch.float32, device=self.device)
            if kind == "verify":
                entry["q_in"] = torch.zeros((count - 1, vocab), dtype=torch.float32, device=self.device)
                entry["uniform"] = torch.zeros((count - 1,), dtype=torch.float32, device=self.device)
                entry["out_accept"] = torch.zeros((count - 1,), dtype=torch.bool, device=self.device)
            else:
                entry["out_q"] = torch.zeros((1, vocab), dtype=torch.float32, device=self.device)
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
        row_base = self.cache.base_of(row)
        row_end = row_base + self.cache.cap_of(row)
        base = row_base + (start + first) // self.world
        host = self._spec_host(1 + 3 * per)
        host[0] = start
        for j in range(per):
            i = first + self.world * j
            owned = i < count
            safe = base + j
            # Attempt2 (CSF UAF, memcheck-named OOB span): `run_spec` stages
            # device slots with no `accepts_spec` gate (spec.py's prewarm loop
            # bounds starts by `g.capacity`, the WIDEST row — not `cap_of(row)`),
            # and `begin_forward_static` is unchecked (decoder.py:456). A slot
            # past the row's extent becomes the off-owner form — slot -1 (the
            # record writer skips it; these out-of-row primes are released
            # after prewarm and never read) and `safe` = the row's first slot
            # (a valid in-row slot rewritten with its own bytes). In-extent
            # staging is unchanged.
            inside = row_base <= safe < row_end
            host[1 + j] = i if owned else 0
            host[1 + per + j] = safe if (owned and inside) else -1
            host[1 + 2 * per + j] = safe if inside else row_base
        entry["args"].copy_(host, non_blocking=True)

    def _spec_slots(self, entry):
        per = entry["per"]
        args = entry["args"]
        return args[1:1 + per], args[1 + per:1 + 2 * per], args[1 + 2 * per:1 + 3 * per]

    def _body_verify(self, row, call_shape, width, count, sample=False, filtered=False, regime="cp"):
        from .lanes import use_lane

        d = self.decoder
        entry = self._spec_state("verify", count)
        tokens, rows = self._spec_rows(entry["tokens"], row, call_shape, count)
        positions = entry["args"][0:1] + torch.arange(count, dtype=torch.long, device=self.device)
        token_rows = d._token_rows(tuple(tokens.shape), rows)
        visible = positions + 1
        kept, slot, safe = self._spec_slots(entry)
        d.cache.begin_forward_static(token_rows, positions, slot=slot, safe=safe, kept=kept)
        geometry, visible_max = self._geo(regime, width, row=row, count=count)
        visible_max = visible_max if call_shape.is_decode_graph else None
        deferred = getattr(d.collectives, "deferred_checks", None)
        if deferred is not None:
            deferred.clear()
        with use_lane(0, overlap=False, geometry=geometry, sync_free=True), w8_decode_scope():
            logits, normed, _ = d._layers_and_head(
                tokens, rows, positions, token_rows, visible, visible_max, state=None,
                return_hidden=True, call_shape=call_shape, logits_rows="all", gather=False)
            if sample:
                from .spec_sample import filter_scores, verify_sample

                sp = logits.reshape(count, -1).float() * self.spec_inv_temp()
                if filtered:
                    top_k, top_p = self.spec_filter_params()
                    sp, covered = filter_scores(d.collectives, sp, top_k, top_p,
                                                vocabulary_range=d.weights.sharding.vocabulary_range)
                    entry["out_cov"].copy_(covered)
                accept, greedy = verify_sample(
                    d.collectives, sp,
                    entry["q_in"], entry["tokens"][0, 1:], entry["uniform"], entry["noise"],
                    vocabulary_range=d.weights.sharding.vocabulary_range)
                entry["out_accept"].copy_(accept)
            else:
                greedy = d.greedy_tokens(logits.reshape(count, -1))
        return self._emit(entry, greedy, normed.reshape(count, -1), self._flag(deferred))

    def _body_draft(self, row, call_shape, width, count, sample=False, filtered=False, regime="cp"):
        from .lanes import use_lane

        d = self.decoder
        entry = self._spec_state("draft", count)
        tokens, rows = self._spec_rows(entry["tokens"], row, call_shape, count)
        hidden_in = entry["hidden"]
        positions = entry["args"][0:1] + torch.arange(count, dtype=torch.long, device=self.device)
        geometry, visible_max = self._geo(regime, width, row=row, count=count)
        visible_max = visible_max if call_shape.is_decode_graph else None
        deferred = getattr(d.collectives, "deferred_checks", None)
        if deferred is not None:
            deferred.clear()
        with use_lane(0, overlap=False, geometry=geometry, sync_free=True), w8_decode_scope():
            hidden, _, _ = d._draft_cp(tokens, hidden_in, positions, rows, d.cache,
                                       call_shape=call_shape, visible_max=visible_max,
                                       static_slots=self._spec_slots(entry))
            if sample:
                from .spec_sample import sample_tokens

                scores = d.head_logits(hidden[-1:]) * self.spec_inv_temp()
                if filtered:
                    from .spec_sample import filter_scores

                    top_k, top_p = self.spec_filter_params()
                    scores, _ = filter_scores(d.collectives, scores, top_k, top_p,
                                              vocabulary_range=d.weights.sharding.vocabulary_range)
                entry["out_q"].copy_(scores)
                greedy = sample_tokens(d.collectives, scores, entry["noise"],
                                       vocabulary_range=d.weights.sharding.vocabulary_range)
            else:
                greedy = d.greedy_tokens(d.head_logits(hidden[-1:]))
        return self._emit(entry, greedy, hidden.reshape(count, -1), self._flag(deferred))

    @staticmethod
    def _emit(entry, greedy, hidden, flag):
        entry["out_tokens"].copy_(greedy.reshape(-1))
        entry["out_hidden"].copy_(hidden)
        entry["out_flag"].copy_(flag)
        return entry["out_tokens"], entry["out_hidden"], entry["out_flag"]

    def _spec_rows(self, tokens, row, call_shape, count):
        """The static `[1, T]` tokens as the forward's `(token_ids, rows)`:
        `[T, 1]` over T query rows of request row `row` for the decode-family
        verification shape, else `[1, T]` over one row."""
        if self._decode_rows(call_shape, count):
            return (tokens.reshape(count, 1),
                    torch.full((count,), row, dtype=torch.long, device=self.device))
        return tokens, torch.full((1,), row, dtype=torch.long, device=self.device)

    def _flag(self, deferred):
        flags = list(deferred) if deferred else []
        if deferred is not None:
            deferred.clear()
        if flags:
            return torch.stack([f.reshape(()) for f in flags]).any()
        return torch.zeros((), dtype=torch.bool, device=self.device)

    def run_spec(self, kind, *, row, start, count, call_shape, sample=False, filtered=False):
        """One verify ("verify") or MTP ("draft") forward of `count` tokens at
        `start` over the static inputs (`spec_inputs`): `(greedy tokens [T] or
        [1], post-norm hidden [T, H], deferred flag)` — the (kind, T) static
        output buffers, valid until the next forward of the same (kind, T).
        Captured per (kind, T, indexer width bucket); the first call runs
        eagerly through the same body."""
        regime, width = self.spec_width(start, count, call_shape)
        key = (kind, int(row), int(count), self.wkey(regime, width), call_shape.name)
        filtered = bool(sample and filtered)
        if sample:
            key = key + (("sample", "filtered") if filtered else ("sample",))
            self.spec_sample_state(kind, count)
        entry = self._spec_state(kind, count)
        self._stage_spec(entry, row, start, count)
        body = self._body_verify if kind == "verify" else self._body_draft
        graph = self.graphs.get(key)
        if graph is None:
            out = body(row, call_shape, width, count, sample, filtered, regime)
            self.eager_steps += 1
            self._capture(key, lambda: body(row, call_shape, width, count, sample, filtered, regime))
            return out
        graph, outputs, _ = graph
        self.graphs.move_to_end(key)
        graph.replay()
        self.replays += 1
        return outputs

    # -- batched decode: B request rows, one token each ----------------------
    # A step over B active request rows runs as the decode-family rows shape
    # of its bucket (1, 2, 4 or 8 query rows; `verify_shape(B)` past one row).
    # Rows, positions and slots are DEVICE inputs, so one graph per (bucket,
    # indexer width bucket) serves any set of rows: query row i reads its own
    # request row's records (`slot_base` per row) and index keys (the fused
    # indexer's per-row base). Pad entries (bucket > B) all repeat one
    # (row, position): a free row at 0, or an active row ONE PAST the token this
    # step writes there — never visible to that token's query, and overwritten
    # by the row's next step before any read. Identical pad entries write
    # identical bytes.
    BATCH_BUCKETS = (1, 2, 4, 8, 12, 16, 32)

    def batch_bucket(self, count):
        for size in self.BATCH_BUCKETS:
            if count <= size:
                return size
        raise ValueError(f"a batched step holds at most {self.BATCH_BUCKETS[-1]} rows")

    def accepts_batch(self, rows, starts):
        return (self.device.type == "cuda" and 1 <= len(rows) <= self.BATCH_BUCKETS[-1]
                and all(0 <= row < int(self.decoder.cache.rows) for row in rows)
                and all(start // self.world < self.cache.cap_of(row)
                        for row, start in zip(rows, starts)))

    def _batch_state(self, size):
        state = getattr(self, "_batch", None)
        if state is None:
            state = self._batch = {}
            self._batch_ring = []
            self._batch_ring_at = 0
        entry = state.get(size)
        if entry is None:
            # args: tokens | positions | rows | record slot (-1 off-owner) | safe slot
            entry = state[size] = {
                "args": torch.zeros((5 * size,), dtype=torch.long, device=self.device),
                "kept": torch.arange(size, dtype=torch.long, device=self.device),
            }
        return entry

    def batch_sample_state(self, size):
        """Device sampling inputs/outputs of the sampled batched step: per-row 1/T,
        U(0,1) noise over this rank's vocabulary shard (refilled outside the
        graph before each replay), and the sampled ids (outside the graph pool)."""
        entry = self._batch_state(size)
        if "noise" not in entry:
            lo, hi = self.decoder.weights.sharding.vocabulary_range
            entry["inv_temp"] = torch.ones((size,), dtype=torch.float32, device=self.device)
            entry["noise"] = torch.zeros((size, hi - lo), dtype=torch.float32, device=self.device)
            entry["out_tokens"] = torch.zeros((size,), dtype=torch.long, device=self.device)
            entry["out_flag"] = torch.zeros((), dtype=torch.bool, device=self.device)
        return entry

    def _batch_host(self, n):
        """A pinned staging vector from a ring (a pending H2D is never rewritten)."""
        if self._batch_ring_at >= len(self._batch_ring):
            buf = torch.zeros((5 * self.BATCH_BUCKETS[-1],), dtype=torch.long)
            if self.device.type == "cuda":
                buf = buf.pin_memory()
            self._batch_ring.append(buf)
        buf = self._batch_ring[self._batch_ring_at]
        self._batch_ring_at = (self._batch_ring_at + 1) % 64
        return buf[:n]

    def _stage_batch(self, entry, size, tokens, rows, starts):
        host = self._batch_host(5 * size)
        for i in range(size):
            row, start = int(rows[i]), int(starts[i])
            local = self.cache.base_of(row) + start // self.world
            host[i] = int(tokens[i])
            host[size + i] = start
            host[2 * size + i] = row
            # Attempt2 (CSF UAF, memcheck-named OOB span): a staged slot past
            # the row's extent would write past the row's records — one past the
            # flat table at a tail row (`_batch_prep` validated pads only
            # against the widest row's `self.capacity`). The record writer
            # consumes device slots and `begin_forward_static` is unchecked by
            # design (decoder.py:456), so the clamp lives here: out-of-extent
            # entries (pads — their records are never read; real entries are
            # refused earlier in `_batch_prep`) become the off-owner form, slot
            # -1 (the fused writer skips it) and `safe` = the row's first slot
            # (a valid in-row slot rewritten with its own bytes, the
            # `begin_forward_static` contract). In-extent staging is unchanged.
            inside = 0 <= start and start // self.world < self.cache.cap_of(row)
            host[3 * size + i] = (local if start % self.world == self.rank else -1) \
                if inside else -1
            host[4 * size + i] = local if inside else self.cache.base_of(row)
        entry["args"].copy_(host, non_blocking=True)

    def _body_batch(self, size, call_shape, rw, sample=False):
        from .lanes import use_lane

        d = self.decoder
        entry = self._batch_state(size)
        args = entry["args"]
        tokens = args[0:size].reshape(size, 1)
        positions = args[size:2 * size]
        rows = args[2 * size:3 * size]
        slot, safe = args[3 * size:4 * size], args[4 * size:5 * size]
        token_rows = rows                       # one token per query row
        visible = positions + 1
        d.cache.begin_forward_static(token_rows, positions, slot=slot, safe=safe,
                                     kept=entry["kept"], kept_identity=_kept_rows())
        geometry, visible_max = self._geo(*rw, rows=rows)
        deferred = getattr(d.collectives, "deferred_checks", None)
        if deferred is not None:
            deferred.clear()
        with use_lane(0, overlap=False, geometry=geometry, sync_free=True), w8_decode_scope():
            logits = d._layers_and_head(
                tokens, rows, positions, token_rows, visible, visible_max, state=None,
                return_hidden=False, call_shape=call_shape, logits_rows="all", gather=False)
            if sample:
                # exact sampling of softmax(logits / T) per row over the vocabulary
                # shards (Gumbel-max + the offer exchange): the same ids on every
                # rank, no logits gather. A greedy row has 1/T = 1 and noise e^-1
                # (Gumbel 0): its argmax.
                from .spec_sample import sample_tokens

                entry = self.batch_sample_state(size)
                scores = logits.reshape(size, -1).float() * entry["inv_temp"].reshape(-1, 1)
                picked = sample_tokens(d.collectives, scores, entry["noise"],
                                       vocabulary_range=d.weights.sharding.vocabulary_range)
                entry["out_tokens"].copy_(picked.reshape(-1))
                entry["out_flag"].copy_(self._flag(deferred))
                return entry["out_tokens"], entry["out_flag"]
        return logits, self._flag(deferred)

    def _batch_prep(self, tokens, rows, starts, call_shape, pad_rows, pad, sample):
        """The padded step's (size, tokens, rows, starts, width, graph key)."""
        count = len(rows)
        size = self.batch_bucket(count)
        # Attempt2 (CSF UAF, memcheck-named OOB span): a REAL entry's slot must
        # land inside its row's extent — `_stage_batch` stages
        # `base_of(row) + start // world` and the static cache path is unchecked
        # by design (decoder.py:456), so a start past the row would alias
        # another row's records or write past the table. Refuse it (the same
        # line `begin_forward_host` draws, decoder.py:438) instead of staging
        # it. Pads are exempt: their records are never read and `_stage_batch`
        # de-slots any pad that lands outside its row.
        for row, start in zip(rows, starts):
            if not (0 <= int(row) < int(self.decoder.cache.rows) and int(start) >= 0
                    and int(start) // self.world < self.cache.cap_of(int(row))):
                raise ValueError(
                    f"a batched step's position {int(start)} lies outside row {int(row)}")
        if size > count:
            if pad is None:
                if pad_rows:
                    pad = (int(pad_rows[0]), 0)
                else:
                    j = min(range(count), key=lambda i: starts[i])
                    pad = (int(rows[j]), int(starts[j]) + 1)
            if not (0 <= pad[0] < int(self.decoder.cache.rows) and pad[1] >= 0
                    and pad[1] < self.capacity):
                raise ValueError("a batched step's pad position lies outside its row")
            rows = list(rows) + [pad[0]] * (size - count)
            starts = list(starts) + [pad[1]] * (size - count)
            tokens = list(tokens) + [0] * (size - count)
        end = max(int(s) for s in starts) + 1
        regime, width = self.regime(end, call_shape)
        key = ("batch", size, self.wkey(regime, width), call_shape.name) + (("sample",) if sample else ())
        return size, tokens, rows, starts, (regime, width), key

    def batch_cached(self, rows, starts, *, call_shape, pad=None, sample=False):
        """Whether this step's graph is captured (no eager first call): an eager
        collective must not be issued while an earlier replay is still pending."""
        key = self._batch_prep([0] * len(rows), rows, starts, call_shape, (), pad, sample)[-1]
        return key in self.graphs

    def run_batch(self, tokens, rows, starts, *, call_shape, pad_rows=(), pad=None, sample=False,
                  pre=None):
        """One decode step of `len(rows)` request rows (token i at position
        starts[i] of rows[i]): `(local-shard logits [bucket, 1, V/w], deferred
        flag)`; logits rows past len(rows) are padding. `pad` = (row, position)
        of every pad entry (default: the first of `pad_rows` at 0, else the
        lowest-position row one past its token)."""
        size, tokens, rows, starts, rw, key = self._batch_prep(
            tokens, rows, starts, call_shape, pad_rows, pad, sample)
        entry = self._batch_state(size)
        self._stage_batch(entry, size, tokens, rows, starts)
        if pre is not None:
            pre(entry["args"][0:size])     # stream-ordered fix-up of the staged token ids
        cached = self.graphs.get(key)
        if cached is None:
            logits, flag = self._body_batch(size, call_shape, rw, sample)
            self.eager_steps += 1
            logits, flag = logits.clone(), flag.clone()
            self._capture(key, lambda: self._body_batch(size, call_shape, rw, sample))
            return logits, flag
        graph, logits, flag = cached
        self.graphs.move_to_end(key)
        graph.replay()
        self.replays += 1
        return logits, flag

    # -- batched speculative sampling: a chain of K drafts per row (K <= 2) ----
    # A round over B request rows is captured forwards: the MTP draft ABSORB of
    # E = K+1 entries per row (its pending (token, target hidden) entries at the
    # draft cursor c .. c+n, n = last round's accepted count, entries past n
    # duplicating entry n) whose last entry gives q1 / d1; K-1 one-entry CHAIN
    # steps (draft j+1 from d_j and the MTP's own hidden at P+j-1); the target
    # VERIFY of E entries per row (the bonus at P, d1 .. dK at P+1 .. P+K) with the
    # batched acceptance (spec_sample.verify_sample_batched), which also advances
    # the device round state (c8 L5b): P, c, n, and the pending entries. Slots
    # (one per request row) index the per-round buffers; pad entries go one past
    # the row's writes and scatter into a dummy state row.
    BSPEC_KMAX = 3

    def bspec_state(self):
        st = getattr(self, "_bspec", None)
        if st is None:
            d = self.decoder
            rows = int(d.cache.rows)
            hidden = int(d.config.hidden)
            lo, hi = d.weights.sharding.vocabulary_range
            S = self.BATCH_BUCKETS[-1]
            K = self.BSPEC_KMAX
            dev = self.device
            st = self._bspec = {
                "pend_tok": torch.zeros((rows + 1, K + 1), dtype=torch.long, device=dev),
                "pend_hid": torch.zeros((rows + 1, K + 1, hidden), dtype=d.compute_dtype, device=dev),
                "spec_on": torch.zeros((rows + 1,), dtype=torch.bool, device=dev),
                "P": torch.zeros((rows + 1,), dtype=torch.long, device=dev),
                "c": torch.zeros((rows + 1,), dtype=torch.long, device=dev),
                "n": torch.zeros((rows + 1,), dtype=torch.long, device=dev),
                "q": torch.zeros((S, K, hi - lo), dtype=torch.float32, device=dev),
                "drafts": torch.zeros((S, K), dtype=torch.long, device=dev),
                "mtp_hid": torch.zeros((S, hidden), dtype=d.compute_dtype, device=dev),
                "inv_temp": torch.ones((S,), dtype=torch.float32, device=dev),
                "noise_q": torch.zeros((S, K, hi - lo), dtype=torch.float32, device=dev),
                "noise_v": torch.zeros((S, K + 1, hi - lo), dtype=torch.float32, device=dev),
                "uniform": torch.zeros((S, K), dtype=torch.float32, device=dev),
                "out_n": torch.zeros((S,), dtype=torch.long, device=dev),
                "out_tokens": torch.zeros((S, K + 1), dtype=torch.long, device=dev),
                "out_dflag": torch.zeros((), dtype=torch.bool, device=dev),
                "out_vflag": torch.zeros((), dtype=torch.bool, device=dev),
                "args": {},
            }
        return st

    def _bspec_args(self, size, per):
        st = self.bspec_state()
        entry = st["args"].get((size, per))
        if entry is None:
            # kv rows | pad mask (host-staged; positions and slots are device-derived)
            idx = torch.arange(size, dtype=torch.long, device=self.device)
            entry = st["args"][(size, per)] = {
                "args": torch.zeros((2 * size,), dtype=torch.long, device=self.device),
                "kept": idx.clone(), "pick": idx % per, "slot_of": idx // per,
            }
        return entry

    def _stage_bspec(self, entry, size, rows, pads):
        host = self._batch_host(2 * size)
        for i in range(size):
            host[i] = int(rows[i])
            host[size + i] = 1 if pads[i] else 0
        entry["args"].copy_(host, non_blocking=True)

    def _bspec_geometry(self, kind, size, per, step=0):
        """Device positions / record slots of a round's entries from the round state:
        absorb c + min(j, n), chain P + step - 1, verify P + j; pad entries one past
        the row's writes (absorb c + n + 1, chain P + K + 1, verify P + per)."""
        st, entry = self.bspec_state(), self._bspec_args(size, per)
        args = entry["args"]
        rows, pad = args[0:size], args[size:2 * size] != 0
        pick = entry["pick"]
        if kind == "draft":
            c, n = st["c"][rows], st["n"][rows]
            positions = torch.where(pad, c + n + 1, c + torch.minimum(pick, n))
        elif kind == "chain":
            P = st["P"][rows]
            positions = torch.where(pad, P + self.BSPEC_KMAX + 1, P + (step - 1))
        else:
            P = st["P"][rows]
            positions = torch.where(pad, P + per, P + pick)
        base = self.cache.row_base_dev[rows]
        local = base + torch.div(positions, self.world, rounding_mode="floor")
        # Attempt2 (CSF UAF, memcheck-named OOB span): the same clamp as
        # `_stage_spec` — the round state's positions are device values and
        # `begin_forward_static` is unchecked, so a slot past the row's extent
        # de-grades to the off-owner form (slot -1, `safe` = the row's first
        # slot, rewritten with its own bytes) instead of writing past the row.
        inside = (local >= base) & (local < base + self.cache.row_cap_dev[rows])
        slot = torch.where(inside & (torch.remainder(positions, self.world) == self.rank), local,
                           torch.full_like(local, -1))
        return rows, pad, positions, slot, torch.where(inside, local, base)

    def _bspec_greedy(self, slots):
        """Per slot: (greedy mask, the scale) — inv_temp 0 marks a greedy row (c8 L5d)."""
        inv = self.bspec_state()["inv_temp"][:slots]
        greedy = inv <= 0
        return greedy, torch.where(greedy, torch.ones_like(inv), inv)

    def _bspec_draft_out(self, last, slots, j):
        """Draft j+1 of every slot from the MTP hidden `last` [slots, H]: its scaled
        scores (q) and sample (the greedy argmax: unit scale, constant noise)."""
        from .spec_sample import sample_tokens

        d, st = self.decoder, self.bspec_state()
        greedy, scale = self._bspec_greedy(slots)
        st["mtp_hid"][:slots].copy_(last)
        scores = d.head_logits(last) * scale.reshape(-1, 1)
        st["q"][:slots, j].copy_(scores)
        noise = torch.where(greedy.reshape(-1, 1), torch.full_like(st["noise_q"][:slots, j], 0.5),
                            st["noise_q"][:slots, j])
        dj = sample_tokens(d.collectives, scores, noise,
                           vocabulary_range=d.weights.sharding.vocabulary_range)
        st["drafts"][:slots, j].copy_(dj.reshape(-1))

    def _body_bdraft(self, size, call_shape, rw, per):
        from .lanes import use_lane

        d = self.decoder
        st, entry = self.bspec_state(), self._bspec_args(size, per)
        rows, _, positions, slot, safe = self._bspec_geometry("draft", size, per)
        idx = torch.minimum(entry["pick"], st["n"][rows])
        tokens = st["pend_tok"][rows, idx].reshape(size, 1)
        hidden_in = st["pend_hid"][rows, idx]
        deferred = getattr(d.collectives, "deferred_checks", None)
        if deferred is not None:
            deferred.clear()
        geometry, visible_max = self._geo(*rw, rows=rows)
        with use_lane(0, overlap=False, geometry=geometry, sync_free=True), \
                w8_decode_scope():
            hidden, _, _ = d._draft_cp(tokens, hidden_in, positions, rows, d.cache,
                                       call_shape=call_shape, visible_max=visible_max,
                                       static_slots=(entry["kept"], slot, safe))
            slots = size // per
            last = hidden[:slots * per].reshape(slots, per, -1)[:, per - 1]
            self._bspec_draft_out(last, slots, 0)
        st["out_dflag"].copy_(self._flag(deferred))
        return st["drafts"], st["out_dflag"]

    def _body_bchain(self, size, call_shape, rw, step):
        """Chain step `step` (1 .. K-1): draft step+1 from d_step and the MTP hidden."""
        from .lanes import use_lane

        d = self.decoder
        st, entry = self.bspec_state(), self._bspec_args(size, 1)
        rows, _, positions, slot, safe = self._bspec_geometry("chain", size, 1, step)
        tokens = st["drafts"][:size, step - 1].reshape(size, 1)
        hidden_in = st["mtp_hid"][:size].clone()
        deferred = getattr(d.collectives, "deferred_checks", None)
        if deferred is not None:
            deferred.clear()
        geometry, visible_max = self._geo(*rw, rows=rows)
        with use_lane(0, overlap=False, geometry=geometry, sync_free=True), \
                w8_decode_scope():
            hidden, _, _ = d._draft_cp(tokens, hidden_in, positions, rows, d.cache,
                                       call_shape=call_shape, visible_max=visible_max,
                                       static_slots=(entry["kept"], slot, safe))
            self._bspec_draft_out(hidden.reshape(size, -1), size, step)
        st["out_dflag"].copy_(self._flag(deferred))
        return st["drafts"], st["out_dflag"]

    def _body_bverify(self, size, call_shape, rw, per):
        from .lanes import use_lane
        from .spec_sample import verify_sample_batched

        d = self.decoder
        st, entry = self.bspec_state(), self._bspec_args(size, per)
        rows, pad, positions, slot, safe = self._bspec_geometry("verify", size, per)
        K = per - 1
        slots = size // per
        S = int(st["drafts"].shape[0])
        pick = entry["pick"]
        bonus = st["pend_tok"][rows, st["n"][rows]]
        drafts_e = st["drafts"][entry["slot_of"].clamp(max=S - 1), (pick - 1).clamp(min=0)]
        tokens = torch.where(pick == 0, bonus, drafts_e).reshape(size, 1)
        token_rows = rows
        visible = positions + 1
        d.cache.begin_forward_static(token_rows, positions, slot=slot, safe=safe, kept=entry["kept"],
                                     kept_identity=_kept_rows())
        deferred = getattr(d.collectives, "deferred_checks", None)
        if deferred is not None:
            deferred.clear()
        geometry, visible_max = self._geo(*rw, rows=rows)
        with use_lane(0, overlap=False, geometry=geometry, sync_free=True), \
                w8_decode_scope():
            logits, normed, _ = d._layers_and_head(
                tokens, rows, positions, token_rows, visible, visible_max, state=None,
                return_hidden=True, call_shape=call_shape, logits_rows="all", gather=False)
            live = slots * per
            greedy, scale = self._bspec_greedy(slots)
            scores = (logits[:live].reshape(slots, per, -1).float() * scale.reshape(-1, 1, 1))
            srow = rows[0:live:per]
            spec_on = st["spec_on"][srow]
            noise = torch.where(greedy.reshape(-1, 1, 1),
                                torch.full_like(st["noise_v"][:slots, :per], 0.5),
                                st["noise_v"][:slots, :per])
            drafts = st["drafts"][:slots, :K]
            n_acc, toks = verify_sample_batched(
                d.collectives, scores, st["q"][:slots, :K], drafts, st["uniform"][:slots, :K],
                noise, spec_on, greedy, vocabulary_range=d.weights.sharding.vocabulary_range)
            st["out_n"][:slots].copy_(n_acc)
            st["out_tokens"][:slots, :per].copy_(toks)
            # c8 L5b: advance the round state on device (pad slots scatter into the
            # dummy row): c = P, n = accepted, P += n + 1; pending = drafts[:n] + the
            # emitted sample, each with its verify hidden (entries past n repeat n)
            target = torch.where(pad[0:live:per], torch.full_like(srow, int(st["P"].shape[0]) - 1),
                                 srow)
            P_old = st["P"][srow]
            jj = torch.arange(per, device=n_acc.device).reshape(1, per)
            last_tok = toks.gather(1, n_acc.reshape(-1, 1))
            dext = torch.cat([drafts, torch.zeros_like(drafts[:, :1])], dim=1)
            new_tok = torch.where(jj < n_acc.reshape(-1, 1), dext, last_tok)
            vh = normed[:live].reshape(slots, per, -1).to(st["pend_hid"].dtype)
            hidx = torch.minimum(jj, n_acc.reshape(-1, 1)).unsqueeze(-1).expand(-1, -1, vh.shape[-1])
            new_hid = vh.gather(1, hidx)
            st["c"].index_copy_(0, target, P_old)
            st["n"].index_copy_(0, target, n_acc)
            st["P"].index_copy_(0, target, P_old + n_acc + 1)
            st["pend_tok"].narrow(1, 0, per).index_copy_(0, target, new_tok)
            st["pend_hid"].narrow(1, 0, per).index_copy_(0, target, new_hid)
        st["out_vflag"].copy_(self._flag(deferred))
        return st["out_n"], st["out_vflag"]

    def run_bspec(self, kind, rows, pads, *, end, call_shape, per=2, step=0):
        """One batched speculative forward: "draft" (the absorb, `per` = K+1 entries
        per request row), "chain" (one entry per row, chain step `step`) or "verify"
        (`per` entries per row), in a bucket. Entry i is a slot of cache row rows[i]
        (pads[i]: a pad entry); its position comes from the device round state.
        `end`: the caller's UPPER bound on the round's context end — it picks the
        regime's graph, and a later regime (or a wider cp bucket) serves an earlier
        context exactly."""
        size = len(rows)
        if size not in self.BATCH_BUCKETS:
            raise ValueError("a batched speculative forward runs in a bucket")
        rw = self.regime(int(end), call_shape)
        sub = step if kind == "chain" else per
        key = ("bspec", kind, size, sub, self.wkey(*rw), call_shape.name)
        entry = self._bspec_args(size, 1 if kind == "chain" else per)
        self._stage_bspec(entry, size, rows, pads)
        if kind == "draft":
            body = lambda: self._body_bdraft(size, call_shape, rw, per)        # noqa: E731
        elif kind == "chain":
            body = lambda: self._body_bchain(size, call_shape, rw, step)       # noqa: E731
        else:
            body = lambda: self._body_bverify(size, call_shape, rw, per)       # noqa: E731
        cached = self.graphs.get(key)
        if cached is None:
            out = body()
            self.eager_steps += 1
            self._capture(key, body)
            return out
        graph, out, flag = cached
        self.graphs.move_to_end(key)
        graph.replay()
        self.replays += 1
        return out, flag

    def bspec_cached(self, kind, size, end, *, call_shape, per=2, step=0):
        sub = step if kind == "chain" else per
        return ("bspec", kind, size, sub, self.wkey(*self.regime(int(end), call_shape)),
                call_shape.name) in self.graphs

    def bspec_set(self, row, *, P, c, n):
        """Host write of one row's device round state (entry, adoption, an exact rerun)."""
        st = self.bspec_state()
        st["P"][row] = int(P)
        st["c"][row] = int(c)
        st["n"][row] = int(n)

    def bspec_shift(self, row, drop):
        """A row whose last round accepted n drafts joins a round of K < n (absorb
        width K + 1): drop its `drop` = n - K oldest pending entries (those draft
        positions keep the chain's MTP-hidden records; exactness does not depend on
        the draft)."""
        st = self.bspec_state()
        width = int(st["pend_tok"].shape[1])
        st["pend_tok"][row, 0:width - drop].copy_(st["pend_tok"][row, drop:width].clone())
        st["pend_hid"][row, 0:width - drop].copy_(st["pend_hid"][row, drop:width].clone())
        st["c"][row] += drop
        st["n"][row] -= drop

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
        regime, width = self.regime(start + 1, call_shape)
        key = (int(row), self.wkey(regime, width), call_shape.name, logits_rows)
        self._stage(token_ids, row, start)
        entry = self.graphs.get(key)
        if entry is None:
            # The bucket's first step runs eagerly through the same body, then
            # the bucket is captured for the following steps.
            logits, flag = self._body(row, call_shape, width, logits_rows, regime)
            self.eager_steps += 1
            logits, flag = logits.clone(), flag.clone()
            self._capture(key, lambda: self._body(row, call_shape, width, logits_rows, regime))
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
        S = self.short_cap()
        if S > self.topk and self._short_bypass(call_shape):
            starts.append(self.topk)            # the local regime's one bucket
        start = max(self.topk, S)
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
