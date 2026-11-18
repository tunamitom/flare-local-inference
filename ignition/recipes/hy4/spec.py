# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0
"""consult18 P4: greedy MTP speculative decoding on the CP serve.

A round at committed length P (the bonus token b at position P is emitted but
not yet in the target KV; the draft cursor c = P - n_prev - 1 marks the rows
whose draft KV still holds draft-hidden values):

1. re-absorb draft rows c .. P-1 with the TARGET's post-norm hidden and their
   next tokens (the accepted drafts and b) — the last row drafts d1;
2. chain k - 1 one-row draft steps at P, P+1, ... — d2 .. dk;
3. verify [b, d1 .. dk] at P .. P+k in one target forward (T = k + 1);
4. n = the longest prefix with d_i == the target's greedy token at P+i-1;
   commit b, d1 .. dn (P += n + 1), the target's token at P+n is the new bonus.

Every forward is a `DecodeGraphs.run_spec` graph (predicated KV writes, greedy
tokens on device and identical on every rank), so the ranks take the same
decisions with one host read per round and no broadcast. Greedy requests
only: the emitted tokens are the target's greedy tokens of T-token forwards
(the numerics class of the verify path — consult18 gate: verify vs decode
NLL inside the decode-vs-prefill noise floor). A set selection flag in the
verify reruns it eagerly on the exact (host-checked) path; draft flags only
cost acceptance.
"""
import torch

ABSORB_WIDTHS = (4, 3, 2, 1)


class SpeculativeGreedy:
    def __init__(self, decoder, *, k=3, row=0, decode_rows=False):
        if not 1 <= int(k) <= 7:
            raise ValueError("speculation depth k must be in 1..7 (verify T = k + 1 <= 8)")
        self.decoder = decoder
        self.k = int(k)
        self.row = int(row)
        # consult18 P4: T > 1 forwards as the decode-family verification shape
        # (needs its plans bound at boot: verify_shape(T), T = 2 .. k + 1)
        self.decode_rows = bool(decode_rows)
        self.rounds = self.accepted = self.verify_reruns = self.draft_flags = 0
        self.filter_fallbacks = 0
        self._generator = None

    @property
    def graphs(self):
        return getattr(self.decoder, "decode_graphs", None)

    def _shape(self, count):
        if self.decode_rows and count > 1:
            from .wiring import verify_shape

            return verify_shape(count)
        return self.decoder.default_call_shape((1, count))

    def eligible(self, start, cursor):
        """Every forward of the next round fits the fixed-shape graphs."""
        g = self.graphs
        if g is None:
            return False
        k = self.k
        return (g.accepts_spec(k + 1, row=self.row, start=start, call_shape=self._shape(k + 1))
                and g.accepts_spec(1, row=self.row, start=cursor, call_shape=self._shape(1))
                and g.accepts_spec(1, row=self.row, start=start + k - 1,
                                   call_shape=self._shape(1)))

    def _noise(self, kind, count):
        """Refill a sampling graph's noise (and the verify's uniforms) from this
        rank's own generator — independent across ranks, outside the graph."""
        if self._generator is None:
            import os

            self._generator = torch.Generator(device=self.graphs.device)
            self._generator.manual_seed(int.from_bytes(os.urandom(8), "little") >> 1)
        entry = self.graphs.spec_sample_state(kind, count)
        entry["noise"].uniform_(generator=self._generator)
        if kind == "verify":
            entry["uniform"].uniform_(generator=self._generator)
        return entry

    def _absorb(self, start, tokens, hidden, sample=False, filtered=False):
        """Draft rows start .. start+len-1 in prepared widths; the last chunk's
        (token [1], hidden [w, H], flag) device outputs (and, sampling, the
        last chunk's draft-score entry)."""
        g = self.graphs
        lo, out = 0, None
        while lo < len(tokens):
            w = next(x for x in ABSORB_WIDTHS if x <= len(tokens) - lo)
            toks, hid = g.spec_inputs("draft", w)
            toks.copy_(torch.tensor([tokens[lo:lo + w]], dtype=torch.long), non_blocking=False)
            hid.copy_(hidden[lo:lo + w].reshape(1, w, -1))
            entry = self._noise("draft", w) if sample else None
            greedy, h, flag = g.run_spec("draft", row=self.row, start=start + lo, count=w,
                                         call_shape=self._shape(w), sample=sample, filtered=filtered)
            out = (greedy, h, flag, entry)
            lo += w
        return out

    def generate(self, *, start, bonus, hidden, emit, temperature=0.0, top_k=0, top_p=1.0,
                 should_yield=None):
        """Rounds from committed length `start` with `bonus` emitted at `start`
        and `hidden` [H] the target's post-norm hidden at `start - 1` (the
        prompt's last row: its draft row is re-absorbed with the real token).
        `emit(token)` appends one token and returns a finish reason or None.
        Returns `(finish, None)` when done, or `(None, (P, bonus))` when the
        next round no longer fits the graphs or `should_yield()` (checked at
        each round start, identical on every rank) asks the request to hand
        over: the caller decodes on from P with `bonus` as the next input."""
        d, g, k, row = self.decoder, self.graphs, self.k, self.row
        sample = float(temperature or 0.0) > 0.0
        filtered = sample and (int(top_k or 0) > 0 or float(top_p if top_p is not None else 1.0) < 1.0)
        if sample:
            # consult18 P5: speculative sampling at this temperature (spec_sample.py)
            g.spec_inv_temp().fill_(1.0 / float(temperature))
            ventry = g.spec_sample_state("verify", k + 1)
            if filtered:
                # consult18 P7: top-k / top-p on device (filter_scores); a round whose
                # nucleus is not covered by the gathered candidates emits nothing and
                # hands the request back to plain decoding
                tk, tp = g.spec_filter_params()
                tk.fill_(int(top_k or 0))
                tp.fill_(float(top_p if top_p is not None else 1.0))
        P, c = int(start), int(start) - 1
        pend_tokens, pend_hidden = [int(bonus)], hidden.reshape(1, -1)
        while self.eligible(P, c):
            if should_yield is not None and should_yield():
                return None, (P, bonus)
            greedy, h, fl, aentry = self._absorb(c, pend_tokens, pend_hidden, sample=sample,
                                                 filtered=filtered)
            flags = [fl.clone()]
            vt, _ = g.spec_inputs("verify", k + 1)
            vt[:, :1].fill_(int(bonus))
            vt[0, 1:2].copy_(greedy[:1])
            if sample:
                ventry["q_in"][0:1].copy_(aentry["out_q"])
            for j in range(1, k):
                st, sh = g.spec_inputs("draft", 1)
                st.copy_(greedy[:1].reshape(1, 1))
                sh.copy_(h[-1:].reshape(1, 1, -1))
                dentry = self._noise("draft", 1) if sample else None
                greedy, h, fl = g.run_spec("draft", row=row, start=P + j - 1, count=1,
                                           call_shape=self._shape(1), sample=sample, filtered=filtered)
                flags.append(fl.clone())
                vt[0, j + 1:j + 2].copy_(greedy[:1])
                if sample:
                    ventry["q_in"][j:j + 1].copy_(dentry["out_q"])
            if sample:
                self._noise("verify", k + 1)
            vg, vh, vf = g.run_spec("verify", row=row, start=P, count=k + 1,
                                    call_shape=self._shape(k + 1), sample=sample, filtered=filtered)
            parts = [vt[0], vg, torch.stack(flags + [vf]).to(torch.long)]
            if sample:
                parts.append(ventry["out_accept"].to(torch.long))
            if filtered:
                parts.append(ventry["out_cov"].to(torch.long))
            host = torch.cat(parts).tolist()
            usable = k + 1                     # verified rows whose filtered p is exact
            if filtered:
                cov_host = host[-(k + 1):]
                host = host[:-(k + 1)]
                usable = next((i for i, ok in enumerate(cov_host) if not ok), k + 1)
            drafts, target = host[1:k + 1], host[k + 1:2 * k + 2]
            flag_host = host[2 * k + 2:3 * k + 3]
            self.draft_flags += sum(flag_host[:-1])
            if flag_host[-1]:
                # a narrow selection in the verify may be inexact: the exact
                # eager forward rewrites the same KV slots and decides instead
                self.verify_reruns += 1
                logits, post, _ = d.forward(vt.clone(), call_shape=d.default_call_shape((1, k + 1)),
                                            logits_rows="all", return_hidden=True)
                vh = post[0]
                if sample:
                    from .spec_sample import verify_sample

                    lo_v, hi_v = d.weights.sharding.vocabulary_range
                    sp = logits[0, :, lo_v:hi_v].float() * g.spec_inv_temp()
                    if filtered:
                        from .spec_sample import filter_scores

                        sp, cov = filter_scores(d.collectives, sp, *g.spec_filter_params(),
                                                vocabulary_range=(lo_v, hi_v))
                        usable = next((i for i, ok in enumerate(cov.tolist()) if not ok), k + 1)
                    accept_t, tokens_t = verify_sample(
                        d.collectives, sp,
                        ventry["q_in"], vt[0, 1:], ventry["uniform"], ventry["noise"],
                        vocabulary_range=(lo_v, hi_v))
                    target = [int(x) for x in tokens_t.tolist()]
                    host = host[:3 * k + 3] + [int(x) for x in accept_t.tolist()]
                else:
                    target = [int(x) for x in logits[0].float().argmax(dim=-1).tolist()]
            n = 0
            if sample:
                accept = host[3 * k + 3:]
                while n < k and n < usable and accept[n]:
                    n += 1
            else:
                while n < k and drafts[n] == target[n]:
                    n += 1
            if n >= usable:
                # consult18 P7: row n's filtered target is not exact (its nucleus ran
                # past the gathered candidates). Rows < n were exact: emit the n
                # accepted drafts, no bonus, and hand the request to plain decoding
                # at P + n (it samples row n's position from the exact p). The
                # verify wrote b, d1 .. dn; P .. P+n-1 are committed, d_n pending.
                self.filter_fallbacks += 1
                self.rounds += 1
                self.accepted += n
                if n == 0:
                    return None, (P, bonus)
                d.append([row], n)
                for tok in drafts[:n]:
                    finish = emit(tok)
                    if finish is not None:
                        return finish, None
                return None, (P + n, drafts[n - 1])
            self.rounds += 1
            self.accepted += n
            new_hidden = vh[:n + 1].clone()
            d.append([row], n + 1)
            for tok in drafts[:n] + [target[n]]:
                finish = emit(tok)
                if finish is not None:
                    return finish, None
            c, P, bonus = P, P + n + 1, target[n]
            pend_tokens, pend_hidden = drafts[:n] + [bonus], new_hidden
        return None, (P, bonus)

    def prewarm(self, *, max_context, sample=True):
        """Capture every spec graph at every indexer width bucket from top-k up to
        `max_context` at boot (collective: every rank runs the same forwards).
        Writes target and draft KV of row `row` at the buckets' first positions;
        the caller releases the row afterwards. Returns the graph count."""
        g, k = self.graphs, self.k
        before = g.captures
        # the short-context bypass bucket first (one start serves all of it)
        starts = [k + 2] if g.spec_width(k + 2, k + 1, self._shape(k + 1)) == 0 else []
        start = g.topk + k + 2
        for short in starts:
            modes = ((False, False), (True, False), (True, True)) if sample else ((False, False),)
            for mode, filt in modes:
                for w in (x for x in ABSORB_WIDTHS if x <= k + 1):
                    toks, hid = g.spec_inputs("draft", w)
                    toks.zero_()
                    hid.zero_()
                    if mode:
                        self._noise("draft", w)
                    g.run_spec("draft", row=self.row, start=short - w, count=w,
                               call_shape=self._shape(w), sample=mode, filtered=filt)
                toks, _ = g.spec_inputs("verify", k + 1)
                toks.zero_()
                if mode:
                    self._noise("verify", k + 1)
                g.run_spec("verify", row=self.row, start=short, count=k + 1,
                           call_shape=self._shape(k + 1), sample=mode, filtered=filt)
        while start + k + 1 < min(int(max_context), g.capacity):
            if not self.eligible(start, start - k - 1):
                break
            modes = ((False, False), (True, False), (True, True)) if sample else ((False, False),)
            for mode, filt in modes:
                for w in (x for x in ABSORB_WIDTHS if x <= k + 1):
                    toks, hid = g.spec_inputs("draft", w)
                    toks.zero_()
                    hid.zero_()
                    if mode:
                        self._noise("draft", w)
                    g.run_spec("draft", row=self.row, start=start - w, count=w,
                               call_shape=self._shape(w), sample=mode, filtered=filt)
                toks, _ = g.spec_inputs("verify", k + 1)
                toks.zero_()
                if mode:
                    self._noise("verify", k + 1)
                g.run_spec("verify", row=self.row, start=start, count=k + 1,
                           call_shape=self._shape(k + 1), sample=mode, filtered=filt)
            start = g.index_width(start + k + 2) * g.world + k + 2
        torch.cuda.synchronize(g.device)
        return g.captures - before

    def stats(self):
        return {"k": self.k, "rounds": self.rounds, "accepted": self.accepted,
                "tokens_per_round": round((self.accepted + self.rounds) / max(self.rounds, 1), 3),
                "verify_reruns": self.verify_reruns, "draft_flags": self.draft_flags,
                "filter_fallbacks": self.filter_fallbacks}
