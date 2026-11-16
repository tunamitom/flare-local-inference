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

    def _absorb(self, start, tokens, hidden):
        """Draft rows start .. start+len-1 in prepared widths; the last chunk's
        (greedy token [1], hidden [w, H]) device outputs."""
        g = self.graphs
        lo, out = 0, None
        while lo < len(tokens):
            w = next(x for x in ABSORB_WIDTHS if x <= len(tokens) - lo)
            toks, hid = g.spec_inputs("draft", w)
            toks.copy_(torch.tensor([tokens[lo:lo + w]], dtype=torch.long), non_blocking=False)
            hid.copy_(hidden[lo:lo + w].reshape(1, w, -1))
            greedy, h, flag = g.run_spec("draft", row=self.row, start=start + lo, count=w,
                                         call_shape=self._shape(w))
            out = (greedy, h, flag)
            lo += w
        return out

    def generate(self, *, start, bonus, hidden, emit):
        """Rounds from committed length `start` with `bonus` emitted at `start`
        and `hidden` [H] the target's post-norm hidden at `start - 1` (the
        prompt's last row: its draft row is re-absorbed with the real token).
        `emit(token)` appends one token and returns a finish reason or None.
        Returns `(finish, None)` when done, or `(None, (P, bonus))` when the
        next round no longer fits the graphs (the caller decodes on from P)."""
        d, g, k, row = self.decoder, self.graphs, self.k, self.row
        P, c = int(start), int(start) - 1
        pend_tokens, pend_hidden = [int(bonus)], hidden.reshape(1, -1)
        while self.eligible(P, c):
            greedy, h, fl = self._absorb(c, pend_tokens, pend_hidden)
            flags = [fl.clone()]
            vt, _ = g.spec_inputs("verify", k + 1)
            vt[:, :1].fill_(int(bonus))
            vt[0, 1:2].copy_(greedy[:1])
            for j in range(1, k):
                st, sh = g.spec_inputs("draft", 1)
                st.copy_(greedy[:1].reshape(1, 1))
                sh.copy_(h[-1:].reshape(1, 1, -1))
                greedy, h, fl = g.run_spec("draft", row=row, start=P + j - 1, count=1,
                                           call_shape=self._shape(1))
                flags.append(fl.clone())
                vt[0, j + 1:j + 2].copy_(greedy[:1])
            vg, vh, vf = g.run_spec("verify", row=row, start=P, count=k + 1,
                                    call_shape=self._shape(k + 1))
            host = torch.cat([vt[0], vg, torch.stack(flags + [vf]).to(torch.long)]).tolist()
            drafts, target = host[1:k + 1], host[k + 1:2 * k + 2]
            self.draft_flags += sum(host[2 * k + 2:-1])
            if host[-1]:
                # a narrow selection in the verify may be inexact: the exact
                # eager forward rewrites the same KV slots and decides instead
                self.verify_reruns += 1
                logits, post, _ = d.forward(vt.clone(), call_shape=d.default_call_shape((1, k + 1)),
                                            logits_rows="all", return_hidden=True)
                target = [int(x) for x in logits[0].float().argmax(dim=-1).tolist()]
                vh = post[0]
            n = 0
            while n < k and drafts[n] == target[n]:
                n += 1
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

    def prewarm(self, *, max_context):
        """Capture every spec graph at every indexer width bucket from top-k up to
        `max_context` at boot (collective: every rank runs the same forwards).
        Writes target and draft KV of row `row` at the buckets' first positions;
        the caller releases the row afterwards. Returns the graph count."""
        g, k = self.graphs, self.k
        before = g.captures
        start = g.topk + k + 2
        while start + k + 1 < min(int(max_context), g.capacity):
            if not self.eligible(start, start - k - 1):
                break
            for w in (x for x in ABSORB_WIDTHS if x <= k + 1):
                toks, hid = g.spec_inputs("draft", w)
                toks.zero_()
                hid.zero_()
                g.run_spec("draft", row=self.row, start=start - w, count=w, call_shape=self._shape(w))
            toks, _ = g.spec_inputs("verify", k + 1)
            toks.zero_()
            g.run_spec("verify", row=self.row, start=start, count=k + 1,
                       call_shape=self._shape(k + 1))
            start = g.index_width(start + k + 2) * g.world + k + 2
        torch.cuda.synchronize(g.device)
        return g.captures - before

    def stats(self):
        return {"k": self.k, "rounds": self.rounds, "accepted": self.accepted,
                "tokens_per_round": round((self.accepted + self.rounds) / max(self.rounds, 1), 3),
                "verify_reruns": self.verify_reruns, "draft_flags": self.draft_flags}
