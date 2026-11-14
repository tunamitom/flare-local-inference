# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0
"""Hy4 decoder body: the real forward loop behind the prefill and decode entries.

`Hy4Decoder.forward` runs the 78-layer body token-parallel over a cache: the
4-stream iHC state (`IhcBoundary.pre` / fused `post_pre` / `run_head`), gated
sparse MLA with per-head sinks, the IndexCache shared-selection cell (full
layers run the indexer once per group; shared layers bind the owner's buffer),
and the routed MoE. `prefill` and `decode_step` are the plan's entry points
(`prefill_chunk_size`-sized chunks; one token per row per step) and share one
layer loop. The MTP entries run the draft block against the target's POST-
`model.norm` hidden (`norm(hc_head(streams))` — the engines and transformers
5.23.0's `MTPCandidateGenerator` feed exactly that tensor; see `mtp.py`'s
module docstring for the file/line evidence): `draft_absorb` writes the
prompt's rows into the draft KV (prompt absorption), then chained
`draft_step`s hand their returned hidden back and advance positions
(`hy4/mtp.py`, `decoder.draft_step`).

Caches (`Hy4Cache`) are the device-agnostic twin of the paged GPU caches: one
latent record per token per layer (the 656-byte GLM_NSA record's 576 live
dims at production geometry) and one prepared index key per token per `full`
layer (the 132-byte record's 128 live dims). Slots are flat physical positions
per request row — the same slot space `sparse_mla.bind(selected_indices=...)`
and `dsa_indexer`'s `output_index_space="physical"` speak. On GPU the same
records flow through `sparse_mla.concat_and_cache_glm_next_mla` and the
`dsa_indexer` paged FP8 writer instead.

State rows (`StreamState`) follow the recipe's `RankCosts.state_row_bytes`
contract: 4 x hidden BF16 streams + the hidden `hnorm` consumes (the POST-
`model.norm` hidden — finding F) + a consumed token counter per row. Streams
are per-token (each token's row starts at its embedding replicated over the 4
streams, `HYV4Model.forward`), so speculative capacity multiplies rows, not
their width.
"""

from dataclasses import dataclass, field

import torch

from .attention import SelectionCell, rms_norm
from .config import ModelConfig
from .state import Hy4ModelWeights


@dataclass
class Hy4Cache:
    """Per-rank latent and index-key records for `rows` request rows.

    `capacity` slots per row; `slot_base[r] = r * capacity`. Layer `i` owns
    `latent[i]` (kv_lora + qk_rope wide — the GLM_NSA record's live payload);
    each `full` indexer layer's prepared keys live in `index[owner]`. The MTP
    block's records sit at layer `config.layers`. Records are stored at
    `dtype` (BF16 — the GLM_NSA FP8 record's decoded form — unless a caller
    overrides it for a full-precision dry run).

    Context parallelism (D1, amendment 2): token position `p` of a request row
    is owned by rank `p % world` at local index `p // world` — flashinfer's
    interleaved rule (`dsa_indexer.selection_handback.rank_local_slots`; the
    primitive is the contract, and D8 asserts the equality in code). `capacity`
    is the row's GLOBAL position count (1,024 positions per shared page at
    CP8); each rank stores `capacity // world` LOCAL slots per row — page `k`'s
    owned positions land at local slots `[128k, 128(k+1))` on every rank (for
    those `p`, `p // world` runs over exactly `[128k, 128k + 128)`). Record
    compute stays replicated — every rank computes every record and
    `write_latent` / `write_index_keys` KEEP only their owned positions, so
    there is no write-time collective at all. World 1 keeps every position and
    `local == owner slot == position` (the eager path, unchanged).
    """

    config: ModelConfig
    rows: int
    capacity: int
    device: torch.device
    latent: dict = field(default_factory=dict)
    index: dict = field(default_factory=dict)
    dtype: torch.dtype = torch.bfloat16
    world: int = 1
    rank: int = 0

    def __post_init__(self):
        if type(self.world) is not int or self.world < 1:
            raise ValueError(f"cache world must be a positive int, got {self.world!r}")
        if type(self.rank) is not int or not 0 <= self.rank < self.world:
            raise ValueError(f"cache rank must be in [0, {self.world}), got {self.rank!r}")
        if type(self.capacity) is not int or self.capacity < 0 or self.capacity % self.world:
            raise ValueError(
                f"cache capacity {self.capacity} must be a nonnegative multiple of the world "
                f"({self.world}) — capacity counts GLOBAL positions, storage is in local slots")

    @classmethod
    def allocate(cls, config, rows, capacity, device, *, draft=False, dtype=torch.bfloat16,
                 world=1, rank=0):
        c = config.text
        cache = cls(config=config, rows=rows, capacity=capacity, device=device, dtype=dtype,
                    world=world, rank=rank)
        layers = config.layers + int(draft)
        width = c["kv_lora_rank"] + c["qk_rope_head_dim"]
        for layer in range(layers):
            cache.latent[layer] = torch.zeros((rows * cache.local_capacity, width), dtype=dtype, device=device)
        for owner in (*config.full_indexer_layers, *( (config.layers,) if draft else () )):
            cache.index[owner] = torch.zeros((rows * cache.local_capacity, c["index_head_dim"]),
                                            dtype=dtype, device=device)
        return cache

    # --- ownership (D1, amendment 2) --------------------------------------

    @property
    def local_capacity(self):
        """Local slots per request row (= `capacity` at world 1)."""
        return self.capacity // self.world

    def owner(self, positions):
        """Rank owning each token position: `position % world` (the interleaved rule)."""
        return torch.as_tensor(positions, device=self.device).long() % self.world

    def local(self, positions):
        """The owning rank's local slot index: `position // world`."""
        return torch.as_tensor(positions, device=self.device).long() // self.world

    def local_slot_map(self, rows_for):
        """Per-row local-index -> physical-slot table for the selection hand-back (D3).

        `rows_for` maps each query row to its request row; request row `r`'s
        local index `j` lives at slot `r * local_capacity + j`. The map is PER
        ROW — one shared map would hand every query row's picks back into row
        zero's cache (the `selection_handback.py` fix-on-the-way; gate b fails
        on a batch of two without it).
        """
        base = self.slots_for(rows_for).reshape(-1, 1)
        offsets = torch.arange(self.local_capacity, device=self.device, dtype=torch.long)
        return base + offsets.reshape(1, -1)

    # --- slots and writes -------------------------------------------------

    @property
    def slot_base(self):
        return (torch.arange(self.rows, device=self.device, dtype=torch.long) * self.local_capacity)

    def slots_for(self, row_of):
        """Each token's row slot base (`row_of` maps token -> request row)."""
        return (row_of.long() * self.local_capacity)

    def write_latent(self, layer, tokens, positions, latent):
        """Store this layer's latent records at the tokens' OWNED physical slots."""
        self._store(self.latent[layer], tokens, positions, latent)

    def write_index_keys(self, owner, tokens, positions, keys):
        self._store(self.index[owner], tokens, positions, keys)

    def _store(self, table, tokens, positions, values):
        positions = positions.long()
        slots = self._slots(tokens, positions)   # bounds-checked over every position
        keep = (positions % self.world) == self.rank   # store only owned records (D1)
        table[slots[keep]] = values.to(table.dtype)[keep]

    def _slots(self, tokens, positions):
        """Physical slots for `positions`, refusing writes past the row.

        The slot is `row * local_capacity + position // world`, so a position
        >= `capacity` lands in the NEXT row's records — silently, at the
        default capacity of 4,096 (consult3 W4). A row is `capacity` GLOBAL
        positions long; anything beyond raises instead of aliasing another
        request's cache.
        """
        positions = positions.long()
        if bool((positions < 0).any()) or bool((positions >= self.capacity).any()):
            raise IndexError(f"cache position out of range: a row holds {self.capacity} slots")
        return self.slots_for(tokens) + self.local(positions)

    def latents(self, layer):
        return self.latent[layer]

    def key_table(self, owner):
        return self.index[owner]


@dataclass
class StreamState:
    """One rank's 4-stream state rows (the `state_row_bytes` contract).

    `rows` speculative-capacity rows, each `[4, hidden]` BF16 streams plus the
    single-stream hidden `hnorm` consumes (the POST-`model.norm` hidden, per
    the finding-F convention) and the consumed-token counter.
    """

    streams: torch.Tensor   # [rows, 4, hidden] BF16
    hidden: torch.Tensor    # [rows, hidden] BF16 (post-`model.norm`)
    count: torch.Tensor     # [rows] int64

    @classmethod
    def allocate(cls, rows, hidden, device, dtype=torch.bfloat16):
        return cls(torch.zeros((rows, 4, hidden), dtype=dtype, device=device),
                   torch.zeros((rows, hidden), dtype=dtype, device=device),
                   torch.zeros((rows,), dtype=torch.int64, device=device))

    @classmethod
    def begin(cls, embeds):
        """Initialize token rows: the embedding replicated over the 4 streams."""
        rows, hidden = embeds.shape
        streams = embeds.unsqueeze(1).expand(rows, 4, hidden).contiguous()
        return cls(streams, torch.zeros_like(embeds), torch.zeros((rows,), dtype=torch.int64,
                                                                 device=embeds.device))

    @property
    def row_bytes(self):
        per_row = self.streams[0].numel() * self.streams.element_size() \
            + self.hidden[0].numel() * self.hidden.element_size() \
            + self.count[0].numel() * self.count.element_size()
        return per_row


class Hy4Decoder:
    """The rank's decoder body: one layer loop behind prefill and decode.

    `compute_dtype` is the stream/activation precision the loop runs at; it
    defaults to BF16 (the serve path's stream rows and cache records). A
    full-precision dry run passes FP32 with a matching `Hy4Cache` dtype.
    """

    def __init__(self, weights: Hy4ModelWeights, cache: Hy4Cache, *, compute_dtype=None, collectives=None):
        from .collectives import SoloCollectives

        self.weights = weights
        self.cache = cache
        self.config = weights.config
        self.device = weights.device
        self.compute_dtype = torch.bfloat16 if compute_dtype is None else compute_dtype
        if collectives is None:
            if cache.world != 1:
                raise ValueError(
                    "a context-parallel rank needs an injected Collectives "
                    "(W6 SimulatedGroup / W7 PCIeDCPA2A wiring)")
            collectives = SoloCollectives()
        if (collectives.world_size, collectives.rank) != (cache.world, cache.rank):
            raise ValueError(
                f"collectives identity ({collectives.rank}/{collectives.world_size}) "
                f"disagrees with the cache ({cache.rank}/{cache.world})")
        self.collectives = collectives
        self._row_lengths = {}
        self._draft_positions = {}

    # --- entry points ------------------------------------------------------

    def prefill(self, token_ids, *, request_rows=None, state=None):
        """Chunked prefill: `token_ids` [rows, tokens] continues the rows' prompts.

        Chunks are at most `prefill_chunk_size` tokens per row; the loop is the
        same one `decode_step` runs, with every query token attending over its
        causal cache length.
        """
        return self.forward(token_ids, request_rows=request_rows, state=state)

    def decode_step(self, token_ids, *, request_rows=None, state=None):
        """One decode step: `token_ids` [rows, 1] appends to the rows' contexts."""
        if token_ids.shape[1] != 1:
            raise ValueError("decode_step advances one token per row")
        return self.forward(token_ids, request_rows=request_rows, state=state)

    def draft_absorb(self, next_tokens, prev_hidden, *, request_rows=None, cache=None):
        """Teacher-forced prompt absorption into the draft KV (finding F fix).

        `next_tokens` [rows, k] are the SHIFTED prompt tokens (row t holds the
        token at t+1), `prev_hidden` [rows, k, hidden] the target's POST-
        `model.norm` hidden per row (`norm(hc_head(streams))` — the state rows
        keep exactly that tensor). The MTP writes its draft KV/index rows at
        the prompt's own positions so later draft steps attend over the prompt
        (transformers candidate_generator.py:1427 runs the MTP over the full
        prompt; the pre-fix recipe left zeros at every prompt position).
        Returns (logits [rows, k, V], hidden [rows, k, H], selected [rows, k,
        topk]); each row's draft cursor advances to cover the absorbed rows, so
        chained `draft_step`s continue at the next position.
        """
        if next_tokens.ndim != 2 or prev_hidden.ndim != 3:
            raise ValueError("draft_absorb needs next_tokens [rows, k] and "
                             "prev_hidden [rows, k, hidden]")
        if next_tokens.shape[:2] != prev_hidden.shape[:2]:
            raise ValueError("next_tokens and prev_hidden must agree on [rows, k]")
        if self.weights.draft is None:
            raise ValueError("no MTP draft block is prepared")
        rows = self._rows(next_tokens, request_rows)
        k = next_tokens.shape[1]
        starts = torch.tensor([self._draft_positions.get(int(r), 0) for r in rows.tolist()],
                              dtype=torch.long, device=self.device)
        # The absorption runs in select-sized chunks — the row-wise top-k lands
        # in the shared selection cell's `[max_q_rows, topk]` buffer, and the
        # real prefill path chunks the same way (`prefill_chunk_size` tokens per
        # row). Rows within a chunk attend causally through their cache slots,
        # so chunking is batching only: identical outputs, no re-computation of
        # written records (write-once draft KV — the W5 cache-diff decision).
        limit = int(getattr(self.weights.draft.attention.selection.cache,
                            "max_q_rows", k))
        outs, sels = [], []
        for lo in range(0, k, limit):
            hi = min(lo + limit, k)
            positions = (starts.reshape(-1, 1)
                         + torch.arange(lo, hi, device=self.device)).reshape(-1)
            hidden, _, selected = self._draft(next_tokens[:, lo:hi], prev_hidden[:, lo:hi],
                                              positions, rows, cache)
            outs.append(hidden.reshape(next_tokens.shape[0], hi - lo, -1))
            # snapshot: `select` hands back the selection cell's LIVE buffer
            # (shared with `sparse_mla.bind`), and the next chunk's
            # `write_selection` (or any later step's) overwrites it in place.
            sels.append(selected.reshape(next_tokens.shape[0], hi - lo, -1).clone())
        hidden = torch.cat(outs, dim=1)
        selected = torch.cat(sels, dim=1)
        for row, start in zip(rows.tolist(), starts.tolist()):
            self._draft_positions[row] = start + k
        logits = hidden.reshape(-1, hidden.shape[-1]).float() @ self.weights.head.T
        return (logits.reshape(next_tokens.shape[0], k, -1), hidden, selected)

    def draft_step(self, next_tokens, prev_hidden, *, request_rows=None, cache=None):
        """One chained MTP draft step; returns (logits [rows, V], hidden [rows, H]).

        `next_tokens` [rows, 1] is the drafted continuation token, `prev_hidden`
        [rows, hidden] the PREVIOUS step's RETURNED hidden (post
        `final_layernorm` — what the engines chain), or the target's post-
        `model.norm` hidden for the first step after absorption. The step runs
        at the row's draft cursor (absorbed rows plus chained steps) and
        advances it by one: chained steps walk positions forward and hand their
        hidden back (transformers modeling_layers.py:524-558).
        """
        if next_tokens.shape[1] != 1:
            raise ValueError("draft_step advances one token per row")
        if self.weights.draft is None:
            raise ValueError("no MTP draft block is prepared")
        rows = self._rows(next_tokens, request_rows)
        starts = torch.tensor([self._draft_positions.get(int(r), 0) for r in rows.tolist()],
                              dtype=torch.long, device=self.device)
        hidden, _, _ = self._draft(next_tokens, prev_hidden, starts, rows, cache)
        for row, start in zip(rows.tolist(), starts.tolist()):
            self._draft_positions[row] = start + 1
        return hidden.float() @ self.weights.head.T, hidden

    def _draft(self, next_tokens, prev_hidden, positions, rows, cache):
        """Shared draft body: embed, run the block at `positions`; (hidden, attended, selected)."""
        cache = self.cache if cache is None else cache
        if cache.world != 1:
            # D7: MTP is out of the P5 diff — the draft block follows the same
            # CP rules (records at layer 78 + `_DRAFT_PAGE_BYTES` pages, local
            # candidates -> exchange -> hand-back, head gather -> partial ->
            # reduce-scatter -> one sink fold) when MTP serving lands, not now.
            raise NotImplementedError(
                "the MTP draft block has no context-parallel path yet (W6 D7: out of P5 scope)")
        token_rows = self._token_rows(next_tokens.shape, rows)
        visible = positions + 1
        next_embeds = self.weights.embed[next_tokens.reshape(-1)].to(self.compute_dtype)
        hidden, attended, selected = self.weights.draft.forward(
            next_embeds, prev_hidden.to(self.compute_dtype).reshape(-1, prev_hidden.shape[-1]),
            positions, cache, visible, token_rows=token_rows)
        return hidden, attended, selected

    def forward(self, token_ids, *, request_rows=None, state=None, return_hidden=False):
        """The real body loop: iHC streams -> 78 gated sparse-MLA layers -> head.

        Returns FP32 logits over this rank's vocabulary shard. `request_rows`
        maps each input row to its cache row (default: identity). With
        `return_hidden=True` returns `(logits, hidden_post, hidden_pre)`:
        `hidden_post` [rows, tokens, hidden] is the POST-`model.norm` hidden
        (the MTP feed and the lm_head pre-image), `hidden_pre` the pre-norm
        `hc_head` output (diagnostics / the pre-norm-feed mutant only).
        """
        if token_ids.ndim != 2:
            raise ValueError("token_ids must be [rows, tokens]")
        rows = self._rows(token_ids, request_rows)
        positions = self._positions(token_ids.shape, rows)
        token_rows = self._token_rows(token_ids.shape, rows)
        visible = positions + 1  # each query's causal cache length
        # Vocabulary-sharded embedding (D5): masked lookup over this rank's
        # shard + all-reduce (identity at world 1).
        embeds = self.collectives.embed_lookup(
            token_ids.reshape(-1), self.weights.embed,
            vocabulary_range=self.weights.sharding.vocabulary_range).to(self.compute_dtype)
        streams = StreamState.begin(embeds)
        if state is not None:
            state.streams[:streams.streams.shape[0]] = streams.streams
        post = y = None
        mlp_out = None
        for index, layer in enumerate(self.weights.layers):
            if index == 0:
                post, y = layer.attn_ihc.pre(streams.streams)
                y = rms_norm(y, layer.input_norm, self.config.text["rms_norm_eps"])
            else:
                streams.streams, post, y = previous_ffn.post_pre(
                    mlp_out, streams.streams, post, layer.attn_ihc,
                    norm_weight=layer.input_norm)
            attn_out = self.collectives.all_reduce(
                self._attention(layer, y, positions, token_rows, visible))  # D4 (o_proj shard)
            streams.streams, post, y = layer.attn_ihc.post_pre(
                attn_out, streams.streams, post, layer.ffn_ihc,
                norm_weight=layer.post_attention_norm)
            # D4: the MoE/dense-FFN outputs are intermediate-sharded partial
            # sums (the layer-0 dense FFN is `mlp.forward` at layer 0).
            mlp_out = self.collectives.all_reduce(layer.mlp.forward(y))
            previous_ffn = layer.ffn_ihc
        streams.streams = previous_ffn.post(mlp_out, streams.streams, post)
        hidden = self.weights.hc_head.merge(streams.streams)
        normed = rms_norm(hidden, self.weights.norm, self.config.text["rms_norm_eps"])
        # The state rows keep the POST-`model.norm` hidden — the exact tensor
        # the MTP's `hnorm` consumes (finding F; vLLM vllm_hy4_model.py:403,
        # SGLang sgl_hunyuan_v4.py:734, transformers candidate_generator.py:1418
        # all feed `model_outputs.hidden_states[-1]` == `norm(hc_head(...))`).
        streams.hidden = normed
        if state is not None:
            state.hidden[:normed.shape[0]] = normed
            state.count[:normed.shape[0]] += token_ids.shape[1]
        logits = normed.float() @ self.weights.head.T
        # D5: logits come back per vocabulary shard; all-gather to the sampler
        # (identity at world 1).
        logits = self.collectives.gather_logits(
            logits.reshape(token_ids.shape[0], token_ids.shape[1], -1),
            vocabulary_range=self.weights.sharding.vocabulary_range)
        if return_hidden:
            rows, width = token_ids.shape
            return (logits,
                    normed.reshape(rows, width, -1),
                    hidden.reshape(rows, width, -1))
        return logits

    # --- internals ---------------------------------------------------------

    def _rows(self, token_ids, request_rows):
        rows = token_ids.shape[0]
        if request_rows is None:
            if rows > self.cache.rows:
                raise ValueError("more input rows than cache rows")
            return torch.arange(rows, device=self.device)
        if len(request_rows) != rows:
            raise ValueError("request_rows must map every input row")
        return torch.tensor(request_rows, device=self.device, dtype=torch.long)

    def _positions(self, shape, rows):
        _, width = shape
        # Each row's continuation starts at its consumed-token count.
        starts = torch.tensor([self._row_lengths.get(row, 0) for row in rows.tolist()],
                             dtype=torch.long, device=self.device)
        positions = starts.reshape(-1, 1) + torch.arange(width, device=self.device).reshape(1, -1)
        return positions.reshape(-1)

    def _token_rows(self, shape, rows):
        return rows.reshape(-1, 1).expand(shape[0], shape[1]).reshape(-1)

    def _attention(self, layer, y, positions, token_rows, visible):
        """One layer's attention: world 1 today's path, world > 1 the CP shape (D2/D3).

        CP (amendment 1): full layers offer local top-k candidates over their
        OWN key table, the selection exchange merges the W ranks' candidates
        into one global top-2048 per query row and `SelectionCell.handback`
        maps it to this rank's local selection (per-row slot map); then the
        head all-gather of projected queries, the all-heads sinkless partial
        over this rank's record shard, the base-2 LSE reduce-scatter returning
        this rank's head shard, and ONE post-merge sink fold
        (`PreparedSinkFold`) before `W_UV` expansion and the gated `o_proj`.
        """
        attn = layer.attention
        slot_base = self.cache.slots_for(token_rows)
        q_resid, q_local, latent = attn.project(y, positions)
        self.cache.write_latent(layer.index, token_rows, positions, latent)
        if attn.weights.indexer is not None:
            keys = attn.prepare_index_keys(y, positions)
            self.cache.write_index_keys(layer.index, token_rows, positions, keys)
        owner = attn.selection.owner_id
        world, rank = self.collectives.world_size, self.collectives.rank
        if world == 1:
            selected, counts = attn.select(y, q_resid, positions, visible,
                                          self.cache.key_table(owner), slot_base)
            attended = attn.attend(q_local, self.cache.latents(layer.index), selected, counts)
            return attn.gate_and_project(attended.output, y)
        if attn.selection.kind == "full":
            scores, candidates = attn.local_candidates(
                y, q_resid, positions, visible, self.cache.key_table(owner), slot_base,
                world=world, rank=rank)
            _, merged = self.collectives.merge_topk(
                scores, candidates, topk=self.config.text["index_topk"])
            attn.selection.handback(merged, world_size=world, rank=rank,
                                    slot_map=self.cache.local_slot_map(token_rows))
        selected, counts = attn.selection.buffers(rows=y.shape[0])
        q_all = self.collectives.all_gather_heads(q_local)   # all heads per call (D2)
        partial = attn.attend(q_all, self.cache.latents(layer.index), selected, counts,
                              fold_sink=False, expand=False)  # sink unbound
        # D6: the partials' provenance goes into the merge
        # (`partials_sink_folded=provenance.sink_folded`, `lse_base="base2"`).
        out, lse = self.collectives.lse_reduce(
            partial.output, partial.lse, partials_sink_folded=partial.provenance.sink_folded)
        out, lse = attn.fold_sink(out, lse)   # the ONE sink fold, after the merge
        # `W_UV` expansion over this rank's head shard (a per (row, head) scale
        # commutes with the fold; the latent-space partials cannot expand on a
        # foreign head — `kv_b_proj` is head-sharded).
        out = torch.einsum("rhd,hvd->rhv", out, attn.expand_wuv.float()).to(y.dtype)
        return attn.gate_and_project(out, y)

    def append(self, rows, count=1):
        """Advance the rows' consumed-token counters after a committed step."""
        if not hasattr(self, "_row_lengths"):
            self._row_lengths = {}
        for row in rows:
            row = int(row)
            self._row_lengths[row] = self._row_lengths.get(row, 0) + count

    def complete(self, rows, token_ids):
        """Commit a forward's tokens to the row counters (prefill/decode bookkeeping)."""
        self.append(rows, token_ids.shape[1])

    def release(self, rows):
        """Release `rows` for reuse: a released row restarts at position 0.

        Without this a reused row silently continues the previous request at
        its old positions and reads its old KV (consult3 W4). Slot bookkeeping
        only: the row's stale records are unreachable once the counter resets —
        reads are causal within the row's new length — and the next request
        overwrites them as it fills.
        """
        if not hasattr(self, "_row_lengths"):
            self._row_lengths = {}
        if not hasattr(self, "_draft_positions"):
            self._draft_positions = {}
        for row in rows:
            self._row_lengths.pop(int(row), None)
            self._draft_positions.pop(int(row), None)
