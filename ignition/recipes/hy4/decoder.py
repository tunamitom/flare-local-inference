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
layer loop. `draft_step` runs the MTP block against the target's pre-norm
hidden.

Caches (`Hy4Cache`) are the device-agnostic twin of the paged GPU caches: one
latent record per token per layer (the 656-byte GLM_NSA record's 576 live
dims at production geometry) and one prepared index key per token per `full`
layer (the 132-byte record's 128 live dims). Slots are flat physical positions
per request row — the same slot space `sparse_mla.bind(selected_indices=...)`
and `dsa_indexer`'s `output_index_space="physical"` speak. On GPU the same
records flow through `sparse_mla.concat_and_cache_glm_next_mla` and the
`dsa_indexer` paged FP8 writer instead.

State rows (`StreamState`) follow the recipe's `RankCosts.state_row_bytes`
contract: 4 x hidden BF16 streams + the hidden `hnorm` consumes + a consumed
token counter per row. Streams are per-token (each token's row starts at its
embedding replicated over the 4 streams, `HYV4Model.forward`), so speculative
capacity multiplies rows, not their width.
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
    block's records sit at layer `config.layers`.
    """

    config: ModelConfig
    rows: int
    capacity: int
    device: torch.device
    latent: dict = field(default_factory=dict)
    index: dict = field(default_factory=dict)

    @classmethod
    def allocate(cls, config, rows, capacity, device, *, draft=False):
        c = config.text
        cache = cls(config=config, rows=rows, capacity=capacity, device=device)
        layers = config.layers + int(draft)
        width = c["kv_lora_rank"] + c["qk_rope_head_dim"]
        for layer in range(layers):
            cache.latent[layer] = torch.zeros((rows * capacity, width), dtype=torch.bfloat16, device=device)
        for owner in (*config.full_indexer_layers, *( (config.layers,) if draft else () )):
            cache.index[owner] = torch.zeros((rows * capacity, c["index_head_dim"]),
                                            dtype=torch.bfloat16, device=device)
        return cache

    @property
    def slot_base(self):
        return (torch.arange(self.rows, device=self.device, dtype=torch.long) * self.capacity)

    def slots_for(self, row_of):
        """Each token's row slot base (`row_of` maps token -> request row)."""
        return (row_of.long() * self.capacity)

    def write_latent(self, layer, tokens, positions, latent):
        """Store this layer's latent records at the tokens' physical slots."""
        slots = self.slots_for(tokens) + positions.long()
        self.latent[layer][slots] = latent.to(self.latent[layer].dtype)

    def write_index_keys(self, owner, tokens, positions, keys):
        self.index[owner][self.slots_for(tokens) + positions.long()] = keys.to(self.index[owner].dtype)

    def latents(self, layer):
        return self.latent[layer]

    def key_table(self, owner):
        return self.index[owner]


@dataclass
class StreamState:
    """One rank's 4-stream state rows (the `state_row_bytes` contract).

    `rows` speculative-capacity rows, each `[4, hidden]` BF16 streams plus the
    single-stream hidden `hnorm` consumes and the consumed-token counter.
    """

    streams: torch.Tensor   # [rows, 4, hidden] BF16
    hidden: torch.Tensor    # [rows, hidden] BF16
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
    """The rank's decoder body: one layer loop behind prefill and decode."""

    def __init__(self, weights: Hy4ModelWeights, cache: Hy4Cache):
        self.weights = weights
        self.cache = cache
        self.config = weights.config
        self.device = weights.device
        self._row_lengths = {}

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

    def draft_step(self, next_tokens, prev_hidden, *, request_rows=None, cache=None):
        """One MTP draft step (`hy4_mtp` semantics); returns draft logits.

        `next_tokens` [rows, 1] are the drafted continuations, `prev_hidden`
        [rows, hidden] the target's pre-`model.norm` hidden (`hc_head` output).
        """
        draft = self.weights.draft
        if draft is None:
            raise ValueError("no MTP draft block is prepared")
        cache = self.cache if cache is None else cache
        rows = self._rows(next_tokens, request_rows)
        positions = self._positions(next_tokens.shape, rows)
        token_rows = self._token_rows(next_tokens.shape, rows)
        visible = positions + 1
        next_embeds = self.weights.embed[next_tokens.reshape(-1)].to(torch.bfloat16)
        hidden, _ = draft.forward(next_embeds, prev_hidden, positions, cache, visible,
                                  token_rows=token_rows)
        return hidden.float() @ self.weights.head.T

    def forward(self, token_ids, *, request_rows=None, state=None):
        """The real body loop: iHC streams -> 78 gated sparse-MLA layers -> head.

        Returns FP32 logits over this rank's vocabulary shard. `request_rows`
        maps each input row to its cache row (default: identity).
        """
        if token_ids.ndim != 2:
            raise ValueError("token_ids must be [rows, tokens]")
        rows = self._rows(token_ids, request_rows)
        positions = self._positions(token_ids.shape, rows)
        token_rows = self._token_rows(token_ids.shape, rows)
        visible = positions + 1  # each query's causal cache length
        embeds = self.weights.embed[token_ids.reshape(-1)].to(torch.bfloat16)
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
            attn_out = self._attention(layer, y, positions, token_rows, visible)
            streams.streams, post, y = layer.attn_ihc.post_pre(
                attn_out, streams.streams, post, layer.ffn_ihc,
                norm_weight=layer.post_attention_norm)
            mlp_out = layer.mlp.forward(y)
            previous_ffn = layer.ffn_ihc
        streams.streams = previous_ffn.post(mlp_out, streams.streams, post)
        hidden = self.weights.hc_head.merge(streams.streams)
        streams.hidden = hidden
        if state is not None:
            state.hidden[:hidden.shape[0]] = hidden
            state.count[:hidden.shape[0]] += token_ids.shape[1]
        normed = rms_norm(hidden, self.weights.norm, self.config.text["rms_norm_eps"])
        logits = normed.float() @ self.weights.head.T
        return logits.reshape(token_ids.shape[0], token_ids.shape[1], -1)

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
        attn = layer.attention
        slot_base = self.cache.slots_for(token_rows)
        q_resid, q_absorbed, latent = attn.project(y, positions)
        self.cache.write_latent(layer.index, token_rows, positions, latent)
        if attn.weights.indexer is not None:
            keys = attn.prepare_index_keys(y, positions)
            self.cache.write_index_keys(layer.index, token_rows, positions, keys)
        owner = attn.selection.owner_id
        selected, counts = attn.select(y, q_resid, positions, visible,
                                      self.cache.key_table(owner), slot_base)
        attended = attn.attend(q_absorbed, self.cache.latents(layer.index), selected, counts)
        return attn.gate_and_project(attended.output, y)

    def append(self, rows, count=1):
        """Advance the rows' consumed-token counters after a committed step."""
        if not hasattr(self, "_row_lengths"):
            self._row_lengths = {}
        for row in rows:
            self._row_lengths[row] = self._row_lengths.get(row, 0) + count

    def complete(self, rows, token_ids):
        """Commit a forward's tokens to the row counters (prefill/decode bookkeeping)."""
        self.append(rows, token_ids.shape[1])
