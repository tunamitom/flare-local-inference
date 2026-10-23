"""Bounded top-k draft proposals: filtering and sampling over at most DRAFT_CANDIDATES tokens.

Proposals follow the request policy after penalties, grammar and temperature:
top-k, softmax, top-p and min-p, each retaining boundary ties among the
candidates, then renormalization. A request whose top-k exceeds
DRAFT_CANDIDATES, including disabled top-k, proposes from the
DRAFT_CANDIDATES highest scores. Chain verification receives exactly the
distribution each draft was drawn from, so verified output is unchanged;
only acceptance depends on how closely proposals follow the target policy.

The first kernel writes zeros over the dense proposal row and keeps each
block's highest scores. The second selects the row's candidates, applies the
policy, draws one Philox uniform, and scatters the surviving probabilities.

Under tensor parallelism the same proposal can come from vocabulary shards of
the draft logits: each rank scores its shard and keeps its bounded candidates,
and the ranks gather only those candidates' keys and their maxima
(PreparedShardedDraftDistribution).
"""

import torch
import triton
import triton.language as tl

from ignition.components.speculation import PreparedSpeculativeDistributions

DRAFT_CANDIDATES = 64
BLOCK = 4096


@triton.jit
def _orderable(scores):
    # Signed integers ordered like the scores; NaN orders above infinity and is rejected later.
    bits = scores.to(tl.int32, bitcast=True)
    # IEEE signed zeros are equal scores and share the token-index tie order.
    bits = tl.where(scores == 0., 0, bits)
    return tl.where(bits < 0, bits ^ 2147483647, bits)


@triton.jit
def _kth_largest(ordered, k):
    # Largest t with at least k values >= t, by bisection over the 32-bit order.
    low = tl.full((), -2147483648, tl.int64)
    high = tl.full((), 2147483647, tl.int64)
    wide = ordered.to(tl.int64)
    for _ in range(32):
        middle = (low + high + 1) >> 1
        enough = tl.sum((wide >= middle).to(tl.int32), 0) >= k
        low = tl.where(enough, middle, low)
        high = tl.where(enough, high, middle - 1)
    return low


@triton.jit
def _candidate_positions(ordered, pivot, present, K: tl.constexpr):
    # Cutoff ties may exceed the bounded buffer. Every strictly higher score
    # must survive before ties consume its remaining slots.
    above = present & (ordered.to(tl.int64) > pivot)
    tied = present & (ordered.to(tl.int64) == pivot)
    higher_count = tl.sum(above.to(tl.int32), 0)
    position = tl.where(above, tl.cumsum(above.to(tl.int32), 0) - 1,
                        higher_count + tl.cumsum(tied.to(tl.int32), 0) - 1)
    take = (above | tied) & (position < K)
    count = tl.minimum(higher_count + tl.sum(tied.to(tl.int32), 0), K)
    return position, take, count


@triton.jit
def _store_block_candidates(scores, tokens, inside, k, Candidates, K: tl.constexpr):
    # Keys order by score, then by lower token ID; ties of one score keep token order in every pool.
    ordered = tl.where(inside, _orderable(scores), -2147483648)
    position, take, count = _candidate_positions(ordered, _kth_largest(ordered, k), inside, K)
    keys = (ordered.to(tl.int64) << 32) | (2147483647 - tokens).to(tl.int64)
    slots = tl.arange(0, K)
    tl.store(Candidates + slots, tl.full((K,), -9223372036854775808, tl.int64), slots >= count)
    tl.store(Candidates + position, keys, take)


@triton.jit
def _block_candidates(Scores, Probabilities, TopK, Candidates, V: tl.constexpr, BLOCKS: tl.constexpr,
                      BLOCK: tl.constexpr, K: tl.constexpr):
    row, block = tl.program_id(0), tl.program_id(1)
    tokens = block * BLOCK + tl.arange(0, BLOCK)
    inside = tokens < V
    scores = tl.load(Scores + row * V + tokens, inside, other=-float("inf"))
    tl.store(Probabilities + row * V + tokens, tl.zeros((BLOCK,), tl.float32), inside)
    k = tl.minimum(tl.maximum(tl.load(TopK + row), 1), K)
    _store_block_candidates(scores, tokens, inside, k, Candidates + (row * BLOCKS + block) * K, K)


@triton.jit
def _select_candidates(pool, present, top_k, Selected, K: tl.constexpr):
    # A row's top-k pool keys and their boundary ties, in pool order, padded with the empty key.
    ordered = (pool >> 32).to(tl.int32)
    position, take, _ = _candidate_positions(ordered, _kth_largest(ordered, top_k), present, K)
    rank = tl.arange(0, K)
    tl.store(Selected + rank, tl.full((K,), -9223372036854775808, tl.int64))
    tl.debug_barrier()
    tl.store(Selected + position, pool, take)
    tl.debug_barrier()


@triton.jit
def _sample_selected(Selected, Probabilities, Valid, Sampled, row, valid, top_k, top_p, min_p, seed, offset,
                     V: tl.constexpr, K: tl.constexpr):
    rank = tl.arange(0, K)
    keys = tl.sort(tl.load(Selected + row * K + rank), descending=True)
    ordered = (keys >> 32).to(tl.int32)
    scores = tl.where(ordered < 0, ordered ^ 2147483647, ordered).to(tl.float32, bitcast=True)
    tokens = 2147483647 - (keys & 2147483647).to(tl.int32)
    present = keys != -9223372036854775808
    finite = present & (scores == scores) & (scores != float("inf")) & (scores != -float("inf"))  # noqa: PLR0124

    # Top-k keeps every candidate tied with the k-th score.
    pivot = tl.max(tl.where(rank == top_k - 1, scores, -float("inf")), 0)
    kept = finite & (scores >= pivot)
    maximum = tl.max(tl.where(kept, scores, -float("inf")), 0)
    weights = tl.where(kept, tl.exp(scores - maximum), 0.)
    probabilities = weights / tl.sum(weights, 0)
    # Top-p keeps the shortest descending prefix reaching top_p and ties with its last member.
    before = tl.cumsum(probabilities, 0) - probabilities
    cutoff = tl.min(tl.where(kept & (before < top_p), probabilities, float("inf")), 0)
    kept &= probabilities >= cutoff
    probabilities = tl.where(kept, probabilities, 0.)
    probabilities = probabilities / tl.sum(probabilities, 0)
    # Min-p against the largest remaining probability, then renormalize.
    kept &= (probabilities > 0.) & (probabilities >= tl.max(probabilities, 0) * min_p)
    probabilities = tl.where(kept, probabilities, 0.)
    mass = tl.sum(probabilities, 0)
    valid = valid & (mass > 0.) & (mass == mass) & (mass != float("inf"))  # noqa: PLR0124
    probabilities = probabilities / tl.where(mass > 0., mass, 1.)

    uniform = tl.rand(seed, offset)
    cumulative = tl.cumsum(probabilities, 0)
    chosen = tl.sum(((cumulative <= uniform) & kept).to(tl.int32), 0)
    last = tl.max(tl.where(kept, rank, 0), 0)
    chosen = tl.minimum(chosen, last)
    sampled = tl.sum(tl.where(rank == chosen, tokens, 0), 0)
    sampled = tl.where(valid, sampled, 0)

    tl.store(Probabilities + row * V + tokens, probabilities, kept & valid)
    tl.store(Probabilities + row * V + sampled, 1., ~valid)
    tl.store(Valid + row, valid)
    tl.store(Sampled + row, sampled.to(tl.int64))


@triton.jit
def _select_and_sample(Candidates, Selected, Probabilities, Valid, TopK, TopP, Params, Slots, Seed,
                       Offset, Sampled, V: tl.constexpr, BLOCKS: tl.constexpr, POOL: tl.constexpr, K: tl.constexpr,
                       SLOTS: tl.constexpr):
    row = tl.program_id(0)
    index = tl.arange(0, POOL)
    pool = tl.load(Candidates + row * BLOCKS * K + index, index < BLOCKS * K, other=-9223372036854775808)
    top_k = tl.minimum(tl.maximum(tl.load(TopK + row), 1), K)
    _select_candidates(pool, pool != -9223372036854775808, top_k, Selected + row * K, K)
    slot = tl.minimum(tl.maximum(tl.load(Slots + row), 0), SLOTS - 1)
    min_p = tl.load(Params + slot * 6 + 2)
    top_p = tl.load(TopP + row)
    offset = (tl.load(Offset + row) & 4294967295).to(tl.uint32)
    _sample_selected(Selected, Probabilities, Valid, Sampled, row, tl.load(Valid + row) != 0, top_k, top_p,
                     min_p, tl.load(Seed + row), offset, V, K)


@triton.jit
def _policy_scores(score, token, inside, Histogram, Candidates, Params, GrammarMask, GrammarEnabled, safe, row,
                   hist_stride, mask_stride, candidate_stride, V: tl.constexpr, PREFIX: tl.constexpr):
    # The committed histogram plus the row's earlier drafts give the penalty counts; repetition, presence
    # and frequency penalties, temperature and the grammar mask follow the unfused order of operations.
    # Tokens are global vocabulary IDs.
    temperature = tl.load(Params + safe * 6)
    repetition = tl.load(Params + safe * 6 + 3)
    presence = tl.load(Params + safe * 6 + 4)
    frequency = tl.load(Params + safe * 6 + 5)
    # Nonnegative counts and positive-zero penalties leave signed logits unchanged.
    # Prefix validity remains a separate check in _draft_scores_finish.
    neutral = ((repetition == 1.) & (presence.to(tl.int32, bitcast=True) == 0)
        & (frequency.to(tl.int32, bitcast=True) == 0))
    if not neutral:
        count = tl.load(Histogram + safe * hist_stride + token, mask=inside, other=0)
        for index in tl.static_range(PREFIX):
            draft = tl.load(Candidates + row * candidate_stride + index)
            count += ((token == draft) & (draft >= 0) & (draft < V)).to(tl.int32)
        # IEEE division and no FMA contraction preserve the unfused arithmetic.
        adjusted = tl.where(score > 0, tl.math.div_rn(score, repetition), score * repetition)
        score = tl.where(count > 0, adjusted, score)
        score = score - (count > 0).to(tl.float32) * presence - count.to(tl.float32) * frequency
    score = tl.math.div_rn(score, tl.where(temperature > 0, temperature, 1.))
    if tl.load(GrammarEnabled + safe) != 0:
        word = tl.load(GrammarMask + safe * mask_stride + token // 32, mask=inside, other=0)
        score = tl.where(((word >> (token % 32)) & 1) != 0, score, -float("inf"))
    return score


@triton.jit
def _draft_scores(Logits, Histogram, Candidates, Slots, Params, GrammarMask, GrammarEnabled, Scores, BlockMax,
                  hist_stride, mask_stride, candidate_stride, CAPACITY: tl.constexpr, V: tl.constexpr,
                  PREFIX: tl.constexpr, BLOCK: tl.constexpr, BLOCKS: tl.constexpr):
    # One vocabulary block of one row; each block records its maximum.
    row, block = tl.program_id(0), tl.program_id(1)
    safe = tl.minimum(tl.maximum(tl.load(Slots + row).to(tl.int64), 0), CAPACITY - 1)
    token = block * BLOCK + tl.arange(0, BLOCK)
    inside = token < V
    score = tl.load(Logits + row * V + token, mask=inside, other=0.).to(tl.float32)
    score = _policy_scores(score, token, inside, Histogram, Candidates, Params, GrammarMask, GrammarEnabled, safe,
                           row, hist_stride, mask_stride, candidate_stride, V, PREFIX)
    tl.store(Scores + row * V + token, score, mask=inside)
    score = tl.where(inside, score, -float("inf"))
    maximum = tl.max(score, 0)
    tl.store(BlockMax + row * BLOCKS + block, maximum)


@triton.jit
def _selected_row(Slots, Generations, Generation, live_rows, row, CAPACITY: tl.constexpr, BATCH: tl.constexpr):
    slot = tl.load(Slots + row).to(tl.int64)
    safe = tl.minimum(tl.maximum(slot, 0), CAPACITY - 1)
    selected = (row < live_rows) & (live_rows >= 0) & (live_rows <= BATCH) & (slot >= 0) & (slot < CAPACITY)
    return selected & (tl.load(Generation + safe) == tl.load(Generations + row)), slot, safe


@triton.jit
def _proposal_valid(maximum, Candidates, Slots, Generations, LiveRows, Generation, Status, row, candidate_stride,
                    CAPACITY: tl.constexpr, BATCH: tl.constexpr, V: tl.constexpr, PREFIX: tl.constexpr,
                    RUNNING: tl.constexpr, POSITIONS: tl.constexpr):
    # The row's liveness (as speculation's _live), its draft prefix and a finite maximum. A row block
    # of POSITIONS draft positions of one request legitimately repeats its slot POSITIONS times.
    live_rows = tl.load(LiveRows)
    selected, slot, safe = _selected_row(Slots, Generations, Generation, live_rows, row, CAPACITY, BATCH)
    sharing = 0
    for other in tl.static_range(BATCH):
        other_selected, other_slot, _ = _selected_row(Slots, Generations, Generation, live_rows, other, CAPACITY, BATCH)
        sharing += (other_selected & (other_slot == slot)).to(tl.int32)
    live = selected & (sharing <= POSITIONS) & (tl.load(Status + safe) == RUNNING)
    prefix_valid = True
    for index in tl.static_range(PREFIX):
        draft = tl.load(Candidates + row * candidate_stride + index)
        prefix_valid &= (draft >= 0) & (draft < V)
    finite = (maximum == maximum) & (tl.abs(maximum) < float("inf"))
    return live & prefix_valid & finite, safe


@triton.jit
def _draft_scores_finish(BlockMax, Candidates, Slots, Generations, LiveRows, Generation, Status, TopK,
                         Params, Maximum, Valid, PackedTopK, PackedTopP, candidate_stride,
                         CAPACITY: tl.constexpr, BATCH: tl.constexpr, V: tl.constexpr, PREFIX: tl.constexpr,
                         BLOCKS: tl.constexpr, BLOCK_COUNT: tl.constexpr, RUNNING: tl.constexpr):
    # The row maximum, then the row's validity and filters.
    row = tl.program_id(0)
    block = tl.arange(0, BLOCK_COUNT)
    present = block < BLOCKS
    maxima = tl.load(BlockMax + row * BLOCKS + block, mask=present, other=-float("inf"))
    maximum = tl.max(maxima, 0)
    valid, safe = _proposal_valid(maximum, Candidates, Slots, Generations, LiveRows, Generation, Status, row,
                                  candidate_stride, CAPACITY, BATCH, V, PREFIX, RUNNING, 1)
    tl.store(Maximum + row, maximum)
    tl.store(Valid + row, valid)
    top_k = tl.load(TopK + safe)
    tl.store(PackedTopK + row, tl.where(top_k > 0, top_k, V))
    tl.store(PackedTopP + row, tl.load(Params + safe * 6 + 1))


@triton.jit
def _shard_candidates(Logits, Histogram, Candidates, Slots, Params, TopK, GrammarMask, GrammarEnabled, Probabilities,
                      BlockCandidates, BlockMax, logit_stride, hist_stride, mask_stride, candidate_stride, TokenOffset,
                      CAPACITY: tl.constexpr, V: tl.constexpr, LOCAL_V: tl.constexpr, PREFIX: tl.constexpr,
                      BLOCK: tl.constexpr, BLOCKS: tl.constexpr, K: tl.constexpr, CLEAR: tl.constexpr):
    # _draft_scores and _block_candidates over one block of a rank's vocabulary shard, indexing the
    # penalties, drafts and grammar mask by global token ID; the scores stay in registers. Each
    # program also clears its part of the dense proposal row.
    row, block = tl.program_id(0), tl.program_id(1)
    safe = tl.minimum(tl.maximum(tl.load(Slots + row).to(tl.int64), 0), CAPACITY - 1)
    local = block * BLOCK + tl.arange(0, BLOCK)
    inside = local < LOCAL_V
    token = tl.load(TokenOffset) + local
    score = tl.load(Logits + row * logit_stride + local, mask=inside, other=0.).to(tl.float32)
    score = _policy_scores(score, token, inside, Histogram, Candidates, Params, GrammarMask, GrammarEnabled, safe,
                           row, hist_stride, mask_stride, candidate_stride, V, PREFIX)
    score = tl.where(inside, score, -float("inf"))
    tl.store(BlockMax + row * BLOCKS + block, tl.max(score, 0))
    top_k = tl.load(TopK + safe)
    k = tl.minimum(tl.maximum(tl.where(top_k > 0, top_k, V), 1), K)
    _store_block_candidates(score, token, inside, k, BlockCandidates + (row * BLOCKS + block) * K, K)
    for part in tl.static_range(CLEAR):
        cleared = (block * CLEAR + part) * BLOCK + tl.arange(0, BLOCK)
        tl.store(Probabilities + row * V + cleared, tl.zeros((BLOCK,), tl.float32), cleared < V)


@triton.jit
def _shard_pool(BlockCandidates, BlockMax, Slots, TopK, Payload, CAPACITY: tl.constexpr, V: tl.constexpr,
                BLOCKS: tl.constexpr, POOL: tl.constexpr, BLOCK_COUNT: tl.constexpr, K: tl.constexpr,
                PAYLOAD: tl.constexpr):
    # The shard's top-k keys and boundary ties, then its maximum's bits, form the rank's payload row.
    row = tl.program_id(0)
    safe = tl.minimum(tl.maximum(tl.load(Slots + row).to(tl.int64), 0), CAPACITY - 1)
    index = tl.arange(0, POOL)
    pool = tl.load(BlockCandidates + row * BLOCKS * K + index, index < BLOCKS * K, other=-9223372036854775808)
    top_k = tl.load(TopK + safe)
    k = tl.minimum(tl.maximum(tl.where(top_k > 0, top_k, V), 1), K)
    _select_candidates(pool, pool != -9223372036854775808, k, Payload + row * PAYLOAD, K)
    block = tl.arange(0, BLOCK_COUNT)
    maxima = tl.load(BlockMax + row * BLOCKS + block, mask=block < BLOCKS, other=-float("inf"))
    maximum = tl.max(maxima, 0)
    tl.store(Payload + row * PAYLOAD + K, maximum.to(tl.int32, bitcast=True).to(tl.int64))


@triton.jit
def _pooled_sample(Received, Candidates, Slots, Generations, LiveRows, Generation, Status, TopK, Params, Seed,
                   Offset, Selected, Probabilities, Maximum, Valid, Sampled, candidate_stride,
                   CAPACITY: tl.constexpr, BATCH: tl.constexpr, V: tl.constexpr, PREFIX: tl.constexpr,
                   WORLD: tl.constexpr, PAYLOAD: tl.constexpr, K: tl.constexpr, RUNNING: tl.constexpr,
                   POSITIONS: tl.constexpr):
    # _draft_scores_finish and _select_and_sample over the gathered shard payloads in rank order.
    row = tl.program_id(0)
    base = Received + row * WORLD * PAYLOAD
    index = tl.arange(0, WORLD * K)
    pool = tl.load(base + index // K * PAYLOAD + index % K)
    shard = tl.arange(0, WORLD)
    maxima = tl.load(base + shard * PAYLOAD + K).to(tl.int32).to(tl.float32, bitcast=True)
    maximum = tl.max(maxima, 0)
    valid, safe = _proposal_valid(maximum, Candidates, Slots, Generations, LiveRows, Generation, Status, row,
                                  candidate_stride, CAPACITY, BATCH, V, PREFIX, RUNNING, POSITIONS)
    tl.store(Maximum + row, maximum)
    top_k = tl.load(TopK + safe)
    k = tl.minimum(tl.maximum(tl.where(top_k > 0, top_k, V), 1), K)
    _select_candidates(pool, pool != -9223372036854775808, k, Selected + row * K, K)
    offset = (tl.load(Offset + row) & 4294967295).to(tl.uint32)
    _sample_selected(Selected, Probabilities, Valid, Sampled, row, valid, k, tl.load(Params + safe * 6 + 1),
                     tl.load(Params + safe * 6 + 2), tl.load(Seed + row), offset, V, K)


_SCORE_BLOCK = 4096


class PreparedDraftProposal:
    """Two captured launches over explicit request-policy inputs for one draft position."""

    def __init__(self, scores, probabilities, valid, top_k, top_p, params, slots, seed, offset, sampled,
                 *, capacity, candidates=None, selected=None):
        rows, vocab = scores.shape[0], scores.shape[-1]
        self.blocks = triton.cdiv(vocab, BLOCK)
        self.candidates = (torch.empty((rows, self.blocks, DRAFT_CANDIDATES), dtype=torch.int64, device=scores.device)
                           if candidates is None else candidates)
        self.selected = (torch.empty((rows, DRAFT_CANDIDATES), dtype=torch.int64, device=scores.device)
                         if selected is None else selected)
        for tensor, shape in ((self.candidates, (rows, self.blocks, DRAFT_CANDIDATES)),
                              (self.selected, (rows, DRAFT_CANDIDATES))):
            if (tensor.shape != shape or tensor.dtype != torch.int64
                    or tensor.device != scores.device or not tensor.is_contiguous()):
                raise ValueError("draft proposal scratch requires contiguous Int64 tensors on the score device")
        pool = triton.next_power_of_2(self.blocks * DRAFT_CANDIDATES)
        self._block_args = (scores, probabilities, top_k, self.candidates, vocab, self.blocks, BLOCK, DRAFT_CANDIDATES)
        self._select_args = (self.candidates, self.selected, probabilities, valid, top_k, top_p,
                             params, slots,
                             seed, offset, sampled, vocab, self.blocks, pool, DRAFT_CANDIDATES, capacity)
        self._grids = ((rows, self.blocks, 1), (rows, 1, 1))
        self._kernels = None

    def prepare(self):
        from ignition.preparation.triton import prepare_triton

        self._kernels = (
            prepare_triton(_block_candidates, self._block_args, self._grids[0], num_warps=8),
            prepare_triton(_select_and_sample, self._select_args, self._grids[1], num_warps=4),
        )
        self()

    def __call__(self):
        if self._kernels is None:
            raise RuntimeError("draft proposal prepare() must complete before capture")
        for kernel in self._kernels:
            kernel()



class _SupportBroadcast:
    """Rank zero's sparse proposal support, broadcast before dependent draft execution."""

    broadcast = None

    @staticmethod
    def broadcast_layout(batch):
        return dict(support_ids=((batch, DRAFT_CANDIDATES), torch.int64),
                    support_probabilities=((batch, DRAFT_CANDIDATES), torch.float32),
                    sampled=((batch,), torch.int64), valid=((batch, 1), torch.bool))

    def prepare_broadcast(self, *, runtime, session, coordinator, name, buffers=None):
        if runtime is None:
            return
        from .collectives import PreparedBroadcast

        def allocate(shape, dtype):
            return (torch.zeros(shape, dtype=dtype, device=self.probabilities.device) if buffers is None
                    else buffers.zeros(shape, dtype))

        layout = self.broadcast_layout(self.probabilities.shape[0])
        self.support_ids = allocate(*layout["support_ids"])
        self.support_probabilities = allocate(*layout["support_probabilities"])
        self.broadcast = PreparedBroadcast(
            (self.support_ids, self.support_probabilities, self.sampled, self.valid),
            runtime=runtime, session=session, coordinator=coordinator, name=name, buffers=buffers)

    def _synchronize(self):
        if self.broadcast is None:
            return
        keys = self.proposal_selected
        present = keys != -9223372036854775808
        ids = 2147483647 - (keys & 2147483647)
        self.support_ids.copy_(torch.where(present, ids, 0))
        probabilities = self.probabilities[:, 0].gather(1, self.support_ids)
        # Padding repeats token zero with its exact probability, so duplicate stores agree.
        # Invalid proposals use the sampler's point mass at token zero.
        self.support_probabilities.copy_(torch.where(self.valid, probabilities, 1.))
        self.support_ids.masked_fill_(~self.valid, 0)
        self.broadcast()
        self.probabilities.zero_()
        self.probabilities[:, 0].scatter_(1, self.support_ids, self.support_probabilities)


class PreparedDraftDistribution(_SupportBroadcast, PreparedSpeculativeDistributions):
    """A one-position conditional proposal whose filtering and draw use PreparedDraftProposal."""

    @staticmethod
    def scratch_layout(batch, width, vocab):
        return {**PreparedSpeculativeDistributions.scratch_layout(batch, width, vocab),
            "block_max": ((batch, triton.cdiv(vocab, _SCORE_BLOCK)), torch.float32),
            "proposal_candidates": ((batch, triton.cdiv(vocab, BLOCK), DRAFT_CANDIDATES), torch.int64),
            "proposal_selected": ((batch, DRAFT_CANDIDATES), torch.int64)}

    def __init__(self, logits, requests, slots, generations, live_rows, candidate_prefix_ids, *,
                 seed, offset, sampled, prefix_offset=0, workspace=None):
        super().__init__(logits, requests, slots, generations, live_rows, candidate_prefix_ids,
                         prefix_offset=prefix_offset, workspace=workspace)
        if logits.shape[1] != 1:
            raise ValueError("draft proposals cover one position")
        # Fused scoring: per-block maxima of each row before the row reduction.
        self._score_kernels = None
        # Contiguous [B,1,...] storage is addressed by row; kernels receive the base tensors.
        self.proposal = PreparedDraftProposal(
            self.scores, self.probabilities, self.valid, self.packed_top_k, self.packed_top_p,
            requests.parameters, slots, seed, offset, sampled, capacity=requests.capacity,
            candidates=self.proposal_candidates, selected=self.proposal_selected)
        self.sampled = sampled

    def prepare(self):
        if self.logits.device.type != "cuda":
            raise ValueError("draft proposals require CUDA")
        from ignition.components.sampling import RUNNING
        from ignition.preparation.triton import prepare_triton

        state = self.state
        batch, _, vocab = self.logits.shape
        blocks = self.block_max.shape[1]
        self._score_kernels = (
            prepare_triton(_draft_scores, (self.logits, state.histogram, self.candidates, self.slots,
                state.parameters, state.grammar_mask, state.grammar_enabled, self.scores, self.block_max,
                state.histogram.stride(0), state.grammar_mask.stride(0), self.candidates.stride(0)),
                (batch, blocks, 1), CAPACITY=state.capacity, V=vocab, PREFIX=self.prefix_offset, BLOCK=_SCORE_BLOCK,
                BLOCKS=blocks, num_warps=8, enable_fp_fusion=False),
            prepare_triton(_draft_scores_finish, (self.block_max, self.candidates, self.slots,
                self.generations, self.live_rows, state.generation, state.status, state.top_k, state.parameters,
                self.maximum, self.valid, self.packed_top_k,
                self.packed_top_p, self.candidates.stride(0)), (batch, 1, 1), CAPACITY=state.capacity, BATCH=batch,
                V=vocab, PREFIX=self.prefix_offset, BLOCKS=blocks, BLOCK_COUNT=triton.next_power_of_2(blocks),
                RUNNING=RUNNING, num_warps=1))
        self._prepared = True
        self.prepare_scores()
        self.proposal.prepare()

    def prepare_scores(self):
        """One-position scores in two kernels, without materialized penalty counts or a separate maximum."""
        for kernel in self._score_kernels:
            kernel()

    def __call__(self):
        if not self._prepared:
            raise RuntimeError("draft proposal requires prepare() before capture")
        self.prepare_scores()
        self.proposal()
        self._synchronize()
        return self.probabilities


# Each rank's payload row: DRAFT_CANDIDATES keys, its maximum's bits, then padding to 64 BF16 columns.
SHARD_PAYLOAD = 80
SHARD_PAYLOAD_COLUMNS = SHARD_PAYLOAD * 4


class PreparedShardedDraftDistribution(_SupportBroadcast):
    """A one-position draft proposal from this rank's contiguous vocabulary shard of the draft logits.

    Each rank applies the request policy's scores to its shard, indexed by global token ID, and keeps
    its own bounded top-k with boundary ties. The ranks' candidate keys and maxima are gathered in rank
    order; since a key orders by score and then by lower token ID, and every shard keeps all scores
    above its cutoff and its lowest-ID ties, the pooled selection equals the selection over the full
    vocabulary. Every rank then filters, draws and writes the dense proposal row exactly as
    PreparedDraftDistribution does from gathered logits.
    """

    @staticmethod
    def scratch_layout(batch, vocab, local_vocab, world):
        blocks = triton.cdiv(local_vocab, BLOCK)
        return dict(probabilities=((batch, 1, vocab), torch.float32), maximum=((batch, 1), torch.float32),
            valid=((batch, 1), torch.bool), block_max=((batch, blocks), torch.float32),
            proposal_candidates=((batch, blocks, DRAFT_CANDIDATES), torch.int64),
            proposal_selected=((batch, DRAFT_CANDIDATES), torch.int64),
            shard_payload=((batch, SHARD_PAYLOAD), torch.int64),
            gathered_payload=((batch, world * SHARD_PAYLOAD), torch.int64))

    def __init__(self, local_logits, requests, slots, generations, live_rows, candidate_prefix_ids, *, rank, world,
                 runtime, session, coordinator, name, seed, offset, sampled, prefix_offset=0, workspace=None,
                 positions=1):
        from ignition.preparation.scratch import bind_scratch

        from .collectives import PreparedAllGather
        from .speculation import _metadata, _tensor

        batch, local_vocab = local_logits.shape
        device, vocab = local_logits.device, requests.vocab_size
        if (local_logits.dtype not in (torch.bfloat16, torch.float32) or local_logits.stride(1) != 1
                or local_vocab * world != vocab or not 0 <= rank < world or world & (world - 1)):
            raise ValueError("sharded draft logits require unit-stride [B,vocabulary/world] rows of one "
                             "power-of-two contiguous partition")
        if runtime is not None and runtime.world_size != world:
            raise ValueError("sharded draft candidates gather through a channel spanning every shard")
        _metadata(requests, slots, generations, live_rows, batch, device)
        if candidate_prefix_ids.ndim != 2 or candidate_prefix_ids.shape[0] != batch:
            raise ValueError("candidate_prefix_ids requires [B,K]")
        _tensor(candidate_prefix_ids, candidate_prefix_ids.shape, torch.int64, device, "candidate_prefix_ids")
        if type(prefix_offset) is not int or not 0 <= prefix_offset <= candidate_prefix_ids.shape[1]:
            raise ValueError("prefix_offset must keep the conditional prefix within candidate IDs")
        for tensor in (seed, offset, sampled):
            _tensor(tensor, (batch,), torch.int64, device, "draft RNG and sample rows")
        self.local_logits, self.state, self.slots = local_logits, requests, slots
        self.generations, self.live_rows = generations, live_rows
        self.candidates, self.prefix_offset = candidate_prefix_ids, prefix_offset
        self.seed, self.offset, self.sampled = seed, offset, sampled
        # A device scalar, not a launch argument: ranks differ only in tensor contents, which compiled
        # forwards do not guard, so every rank binds the same compiled kernel.
        self.token_offset = torch.full((1,), rank * local_vocab, dtype=torch.int64, device=local_logits.device)
        self.rank, self.world = rank, world
        if type(positions) is not int or positions < 1 or batch % positions or (positions > 1 and prefix_offset):
            raise ValueError("a draft position block holds whole requests and conditions on no earlier draft")
        self.positions = positions
        for key, tensor in bind_scratch(self.scratch_layout(batch, vocab, local_vocab, world),
                                        device=device, workspace=workspace).items():
            setattr(self, key, tensor)
        # The candidate channel carries BF16 columns; keys and maxima keep their bits. Without a channel,
        # several shards only expose their stages, and a caller places the payloads.
        self.gather = None if runtime is None and world > 1 else PreparedAllGather(
            self.shard_payload.view(torch.bfloat16), self.gathered_payload.view(torch.bfloat16),
            runtime=runtime, session=session, coordinator=coordinator, name=name)
        self._kernels = None

    def prepare(self):
        if self.local_logits.device.type != "cuda":
            raise ValueError("draft proposals require CUDA")
        from ignition.components.sampling import RUNNING
        from ignition.preparation.triton import prepare_triton

        state = self.state
        batch, local_vocab = self.local_logits.shape
        vocab, blocks = state.vocab_size, self.block_max.shape[1]
        local = (
            prepare_triton(_shard_candidates, (self.local_logits, state.histogram, self.candidates, self.slots,
                state.parameters, state.top_k, state.grammar_mask, state.grammar_enabled, self.probabilities,
                self.proposal_candidates, self.block_max, self.local_logits.stride(0), state.histogram.stride(0),
                state.grammar_mask.stride(0), self.candidates.stride(0), self.token_offset),
                (batch, blocks, 1), CAPACITY=state.capacity, V=vocab, LOCAL_V=local_vocab, PREFIX=self.prefix_offset,
                BLOCK=BLOCK, BLOCKS=blocks, K=DRAFT_CANDIDATES, CLEAR=triton.cdiv(triton.cdiv(vocab, BLOCK), blocks),
                num_warps=8, enable_fp_fusion=False),
            prepare_triton(_shard_pool, (self.proposal_candidates, self.block_max, self.slots, state.top_k,
                self.shard_payload), (batch, 1, 1), CAPACITY=state.capacity, V=vocab, BLOCKS=blocks,
                POOL=triton.next_power_of_2(blocks * DRAFT_CANDIDATES), BLOCK_COUNT=triton.next_power_of_2(blocks),
                K=DRAFT_CANDIDATES, PAYLOAD=SHARD_PAYLOAD, num_warps=4))
        pooled = prepare_triton(_pooled_sample, (self.gathered_payload, self.candidates, self.slots,
            self.generations, self.live_rows, state.generation, state.status, state.top_k, state.parameters,
            self.seed, self.offset, self.proposal_selected, self.probabilities, self.maximum, self.valid,
            self.sampled, self.candidates.stride(0)), (batch, 1, 1), CAPACITY=state.capacity, BATCH=batch, V=vocab,
            PREFIX=self.prefix_offset, WORLD=self.world, PAYLOAD=SHARD_PAYLOAD, K=DRAFT_CANDIDATES,
            RUNNING=RUNNING, POSITIONS=self.positions, num_warps=4)
        self._kernels = local, pooled
        self.shard_candidates()
        if self.gather is not None:
            self.gather(self.shard_payload.view(torch.bfloat16))
            self.pooled_sample()

    def shard_candidates(self):
        """Score this rank's shard and write its payload row."""
        for kernel in self._kernels[0]:
            kernel()

    def pooled_sample(self):
        """Select, filter and draw from the gathered payloads into the dense proposal row."""
        self._kernels[1]()

    def __call__(self):
        if self._kernels is None or self.gather is None:
            raise RuntimeError("draft proposal requires a candidate channel and prepare() before capture")
        self.shard_candidates()
        self.gather(self.shard_payload.view(torch.bfloat16))
        self.pooled_sample()
        self._synchronize()
        return self.probabilities


class PreparedShardedDraftBlock:
    """Every draft position of one request in one sharded proposal: one gather and one broadcast.

    A block drafter without a Markov head or selector scores each position from the shared context
    alone, so the positions are independent rows: row i reads position i's local logits, the request's
    slot, generation and seed, and the Philox offset ``base + 4 * i`` that the position-by-position
    loop would use. Penalties count the committed histogram but not the block's own earlier drafts;
    verification receives exactly the distribution each draft was drawn from, so outputs keep the
    target policy and only acceptance can differ, for requests with repetition penalties.
    """

    def __init__(self, local_logits, requests, slots, generations, live_rows, *, rank, world, runtime,
                 session, coordinator, name, seed, base_offset, workspace=None):
        positions = local_logits.shape[0]
        if slots.numel() != 1 or positions < 2 or local_logits.dim() != 2:
            raise ValueError("a draft position block serves one request's [positions, vocabulary/world] logits")
        device = local_logits.device
        self.positions, self.request_slots, self.request_generations = positions, slots, generations
        self.request_live, self.request_seed, self.base_offset = live_rows, seed, base_offset
        self.slots = torch.zeros(positions, dtype=slots.dtype, device=device)
        self.generations = torch.zeros(positions, dtype=generations.dtype, device=device)
        self.live_rows = torch.zeros_like(live_rows)
        self.seed = torch.zeros(positions, dtype=torch.int64, device=device)
        self.offset = torch.zeros(positions, dtype=torch.int64, device=device)
        self.sampled = torch.zeros(positions, dtype=torch.int64, device=device)
        self.steps = 4 * torch.arange(positions, dtype=torch.int64, device=device)
        self.prefix = torch.zeros((positions, 1), dtype=torch.int64, device=device)
        self.distribution = PreparedShardedDraftDistribution(
            local_logits, requests, self.slots, self.generations, self.live_rows, self.prefix, rank=rank,
            world=world, runtime=runtime, session=session, coordinator=coordinator, name=name, seed=self.seed,
            offset=self.offset, sampled=self.sampled, workspace=workspace, positions=positions)

    def prepare(self):
        self.distribution.prepare()

    def expand(self):
        """Repeat the request's row metadata per position, with each position's Philox offset."""
        self.slots.copy_(self.request_slots.expand(self.positions))
        self.generations.copy_(self.request_generations.expand(self.positions))
        self.live_rows.copy_(self.request_live * self.positions)
        self.seed.copy_(self.request_seed.expand(self.positions))
        self.offset.copy_(self.base_offset + self.steps)

    def __call__(self):
        """Sample every position; returns (drafts [P], dense probabilities [P, 1, V], valid [P, 1])."""
        self.expand()
        probabilities = self.distribution()
        return self.sampled, probabilities, self.distribution.valid

