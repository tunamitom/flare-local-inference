"""Two-phase speculative verification over vocabulary-sharded target logits.

A verification round splits at the target logits. The first phase projects each
rank's vocabulary shard, publishes a small per-row summary (maximum, exp-sum and
K best scores with global IDs) through the candidate channel, merges the gathered
summaries with identical arithmetic on every rank, and writes a one-byte flag:
nonzero when every live request's policy and every row's nucleus allow the
compact path. The host then launches one of two continuations:

- compact: verify the chain over each request's compact vocabulary (its
  positions' nuclei plus its drafts) and fill the dense verifier's acceptance
  outputs, so commit and the rest of the round run unchanged;
- dense: gather full-vocabulary logits and run the ordinary filter and verifier.

The flag is computed from gathered data, so every rank takes the same branch.
Compact rounds draw with the dense path's per-row seed and offset over the same
nucleus in global ID order; only floating-point summation order differs.
"""

import torch
import triton

from .speculation import TARGET_RNG_DOMAIN, _live, _metadata, _tensor
from .vocab_parallel_verify import (NUCLEUS_SLOTS, SUMMARY_CANDIDATES, _merge_kernel, _summary_chunk_kernel,
                                    _summary_merge_kernel, _union_kernel, compact_eligible)

# One gathered row per target row: [maximum bits, exp-sum bits, K score bits, K global IDs], padded to the
# candidate channel's per-rank width in int32 words.
PAYLOAD_WORDS = 160
SUMMARY_CHUNK = 2048


class PreparedTwoPhaseVerification:
    """Compact-or-dense verification of [B, W] candidates from per-rank [B*W, V/world] logits.

    ``verifier`` is the round's dense PreparedStochasticVerification; the compact
    continuation fills its ``output`` (and runs its broadcast) exactly where the
    dense verifier would. ``flag`` is the uint8 device buffer the host reads.
    """

    def __init__(self, local_logits, requests, slots, generations, live_rows, drafts, draft_probs, draft_valid,
                 token_limits, *, verifier, rng, logprob_requested, flag, rank, world, runtime, session,
                 coordinator, name, fallbacks=None):
        from .collectives import PreparedAllGather

        rows, local = local_logits.shape
        batch, drafted = drafts.shape
        width = drafted + 1
        device = local_logits.device
        if (rows != batch * width or local * world != requests.vocab_size or not 0 <= rank < world
                or local_logits.stride(1) != 1 or width < 2):
            raise ValueError("two-phase verification requires [B*W, vocabulary/world] logits of a speculative round")
        if (runtime is None) != (world == 1) or (runtime is not None and runtime.world_size != world):
            raise ValueError("two-phase verification gathers summaries through a channel spanning every shard")
        if 2 + 2 * SUMMARY_CANDIDATES > PAYLOAD_WORDS:
            raise ValueError("summary payload exceeds the candidate channel row")
        _metadata(requests, slots, generations, live_rows, batch, device)
        _tensor(drafts, (batch, drafted), torch.int64, device, "drafts")
        _tensor(draft_probs, (batch, drafted, requests.vocab_size), torch.float32, device, "draft_probs")
        _tensor(draft_valid, (batch, drafted), torch.bool, device, "draft_valid")
        _tensor(token_limits, (batch,), torch.int32, device, "token_limits")
        _tensor(flag, (flag.numel(),), torch.uint8, device, "flag")
        self.local_logits, self.state, self.slots, self.generations, self.live_rows = (
            local_logits, requests, slots, generations, live_rows)
        self.drafts, self.draft_probs, self.draft_valid, self.token_limits = drafts, draft_probs, draft_valid, token_limits
        self.verifier, self.rng, self.requested, self.flag = verifier, rng, logprob_requested, flag
        self.fallbacks = torch.zeros(1, dtype=torch.int64, device=device) if fallbacks is None else fallbacks
        self.batch, self.width, self.rows, self.local, self.world = batch, width, rows, local, world
        K, C = SUMMARY_CANDIDATES, NUCLEUS_SLOTS
        chunks = triton.cdiv(local, SUMMARY_CHUNK)
        self.entries = width * C + width - 1
        N = triton.next_power_of_2(self.entries)

        def empty(shape, dtype=torch.float32):
            return torch.zeros(shape, dtype=dtype, device=device)

        # Every tensor either phase reads across the graph boundary is caller-owned.
        self.scores = empty((rows, local))
        self.chunk_max, self.chunk_sum = empty((rows, chunks)), empty((rows, chunks))
        self.chunk_top = empty((rows, chunks, K), torch.int64)
        self.maximum, self.exp_sum = empty(rows), empty(rows)
        self.top_scores, self.top_ids = empty((rows, K)), empty((rows, K), torch.int64)
        self.payload = empty((rows, PAYLOAD_WORDS), torch.int32)
        self.gathered = empty((rows, world * PAYLOAD_WORDS * 2), torch.bfloat16)
        self.pooled_max, self.pooled_sum = empty((world, rows)), empty((world, rows))
        self.pooled_scores, self.pooled_ids = empty((world, rows, K)), empty((world, rows, K), torch.int64)
        self.row_top_k, self.row_top_p = empty(rows, torch.int32), empty(rows)
        self.merged_ids, self.merged_probs = empty((rows, C), torch.int64), empty((rows, C))
        self.complete = empty(rows, torch.bool)
        self.target, self.union = empty((batch, width, N)), empty((batch, N), torch.int64)
        self.draft_columns, self.union_scratch = empty((batch, drafted), torch.int32), empty((batch, N), torch.int64)
        self.compact_draft = empty((batch, drafted, N))
        self.accepted, self.emitted = empty(batch, torch.int32), empty(batch, torch.int32)
        self.seed_bits, self.offset = empty(batch, torch.int64), empty(batch, torch.int64)
        self.token_offset = torch.full((1,), rank * local, dtype=torch.int64, device=device)
        self.request_rows = torch.arange(batch, device=device)
        self.draw_reservation = 4 * ((width + 1 + 3) // 4)
        self.gather = PreparedAllGather(self.payload.view(torch.bfloat16), self.gathered, runtime=runtime,
                                        session=session, coordinator=coordinator, name=name + ".summaries")
        self._kernels = None

    def prepare(self):
        from ignition.preparation.triton import prepare_triton

        K, C = SUMMARY_CANDIDATES, NUCLEUS_SLOTS
        chunks = self.chunk_max.shape[1]
        N = self.union.shape[1]
        summary = (
            prepare_triton(_summary_chunk_kernel, (self.scores, self.chunk_max, self.chunk_sum, self.chunk_top,
                self.scores.stride(0), self.local, chunks), (self.rows, chunks, 1), K=K, BLOCK=SUMMARY_CHUNK,
                num_warps=4),
            prepare_triton(_summary_merge_kernel, (self.chunk_max, self.chunk_sum, self.chunk_top, self.maximum,
                self.exp_sum, self.top_scores, self.top_ids, chunks, self.token_offset), (self.rows, 1, 1),
                K=K, CHUNKS=triton.next_power_of_2(chunks), num_warps=4))
        merge = prepare_triton(_merge_kernel, (self.pooled_max, self.pooled_sum, self.pooled_scores, self.pooled_ids,
            self.row_top_k, self.row_top_p, self.merged_ids, self.merged_probs, self.complete, self.rows,
            self.state.vocab_size), (self.rows, 1, 1), WORLD=self.world, K=K, NUCLEUS=C, num_warps=4)
        union = prepare_triton(_union_kernel, (self.merged_ids, self.merged_probs, self.drafts, self.target,
            self.union, self.draft_columns, self.union_scratch, self.width, self.entries), (self.batch, 1, 1),
            C=C, N=N, num_warps=8)
        from flashinfer import sampling

        self._sampling = sampling
        self._kernels = summary, merge, union
        self.decide()
        self.compact()

    def decide(self):
        """First phase: summarize, gather, merge and write the rank-uniform flag."""
        state, b, w = self.state, self.batch, self.width
        safe, live = _live(state, self.slots, self.generations, self.live_rows, self.request_rows)
        params = state.parameters[safe]
        temperature = params[:, 0]
        row_temperature = torch.where(temperature > 0, temperature, 1.).repeat_interleave(w)
        self.scores.copy_(self.local_logits.float() / row_temperature[:, None])
        for kernel in self._kernels[0]:
            kernel()
        K = SUMMARY_CANDIDATES
        self.payload[:, 0].copy_(self.maximum.view(torch.int32))
        self.payload[:, 1].copy_(self.exp_sum.view(torch.int32))
        self.payload[:, 2:2 + K].copy_(self.top_scores.view(torch.int32))
        self.payload[:, 2 + K:2 + 2 * K].copy_(self.top_ids.int())
        self.gather(self.payload.view(torch.bfloat16))
        words = self.gathered.view(torch.int32).view(self.rows, self.world, PAYLOAD_WORDS).transpose(0, 1)
        self.pooled_max.copy_(words[:, :, 0].contiguous().view(torch.float32))
        self.pooled_sum.copy_(words[:, :, 1].contiguous().view(torch.float32))
        self.pooled_scores.copy_(words[:, :, 2:2 + K].contiguous().view(torch.float32))
        self.pooled_ids.copy_(words[:, :, 2 + K:2 + 2 * K].long())
        top_k = state.top_k[safe]
        self.row_top_k.copy_(torch.where(top_k > 0, top_k, state.vocab_size).int().repeat_interleave(w))
        self.row_top_p.copy_(params[:, 1].repeat_interleave(w))
        self.merge_kernel()
        eligible = compact_eligible(state.parameters, state.top_k, state.grammar_enabled, self.requested,
                                    self.slots, live)
        full = (self.token_limits == w) & (self.draft_valid.all(-1)) & (
            (self.drafts >= 0) & (self.drafts < state.vocab_size)).all(-1)
        eligible = eligible & ((temperature > 0) & full | ~live)
        complete = self.complete.view(b, w).all(-1) | ~live
        take = (eligible & complete).all().reshape(1)
        self.fallbacks.add_((~take).long())
        self.flag[:1].copy_(take.to(torch.uint8))
        return take

    def merge_kernel(self):
        self._kernels[1]()

    def compact(self):
        """Taken continuation: compact chain verification into the dense verifier's outputs."""
        state, out, b, w = self.state, self.verifier.output, self.batch, self.width
        safe, live = _live(state, self.slots, self.generations, self.live_rows, self.request_rows)
        self._kernels[2]()
        # Padding past the union's live entries is never written; keep its indices in range.
        union = self.union.clamp(0, self.draft_probs.shape[-1] - 1)
        self.compact_draft.copy_(self.draft_probs.gather(-1, union[:, None, :].expand_as(self.compact_draft)))
        self.seed_bits.copy_(state.seed.view(torch.int64)[safe] ^ TARGET_RNG_DOMAIN)
        self.offset.copy_(torch.where(self.rng.generation[safe] == self.generations, self.rng.target_offset[safe], 0))
        self.accepted.zero_()
        self.emitted.zero_()
        tokens, _, _ = self._sampling.chain_speculative_sampling(self.compact_draft, self.draft_columns, self.target,
            self.accepted, self.emitted, deterministic=True, seed=self.seed_bits, offset=self.offset,
            row_independent_rng=True)
        tokens = torch.where(tokens >= 0, self.union.gather(-1, tokens.clamp(0, self.union.shape[-1] - 1).long()), -1)
        valid = live & self.draft_valid.all(-1) & ((self.drafts >= 0) & (self.drafts < state.vocab_size)).all(-1)
        offered = (self.token_limits - 1).clamp(0, w - 1)
        accepted = torch.minimum(self.emitted, offered)
        out.offered_drafts.copy_(offered)
        out.token_ids.copy_(tokens)
        out.accepted_drafts.copy_(torch.where(valid, accepted, 0))
        out.proposed_count.copy_(torch.where(valid, accepted + 1, 0))
        out.valid.copy_(valid)
        out.rng_increment.copy_(torch.where(valid, self.draw_reservation, 0))
        out.token_ids.masked_fill_(~valid[:, None], -1)
        if self.verifier.broadcast is not None:
            self.verifier.broadcast()
        return out
