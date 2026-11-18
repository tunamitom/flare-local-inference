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
from .profile import TIMER, WATCH
from .state import Hy4ModelWeights
from .wiring import DECODE_ONE_ROW, SCHEDULE, chunk_plan, prefill_shape, verify_shape


def _diet2():
    from .runtime import SWITCHES

    return SWITCHES.get("diet2") == "on"


@dataclass
class Hy4Cache:
    """Per-rank latent and index-key records for `rows` request rows.

    `capacity` slots per row; `slot_base[r] = r * capacity`. Layer `i` owns
    `latent[i]` (kv_lora + qk_rope wide — the GLM_NSA record's live payload);
    each `full` indexer layer's prepared keys live in `index[owner]`. The MTP
    block's records sit at layer `config.layers`. Records are stored at
    `dtype` (BF16 — the GLM_NSA FP8 record's decoded form — unless a caller
    overrides it for a full-precision dry run).

    Fused record cache (the engine contract, M2): with `fused_records=True`
    (CUDA only) each latent layer stores the fused 656-byte FP8 record the
    sparse-MLA kernels read (`LATENT_RECORD_BYTES` = 512 E4M3 nope + 16 B
    inline fp32 scales + 128 B raw BF16 RoPE), produced by the
    `sparse_mla.concat_and_cache_glm_next_mla` record writer — `write_latent`
    routes through it and `latents(layer)` returns the fused table (the form
    `_attend_cuda` binds). `decode_latents(layer)` unpacks the decoded
    576-dim BF16 twin on demand (diagnostics only — the engine never keeps a
    decoded copy resident; the record contract is 656 B, not 1,152 B).
    The index-key table keeps its decoded BF16 form in this window (its FP8
    paged writer is separate work).

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
    records: dict = field(default_factory=dict)   # layer -> fused uint8 records
    writer_plan: object = None                    # the prepared `plan_cache_writer`
    # Replicated short-context records: per layer, EVERY token of a row below
    # `short_capacity` positions (not only this rank's owned ones), at slot
    # row * short_capacity + position. In the short-context bypass a rank then
    # attends its own heads over all keys locally (no head gather, no LSE merge).
    short: dict = field(default_factory=dict)
    short_capacity: int = 0
    short_writer_plan: object = None
    page_size: int = 64                           # fused-record page geometry
    # One forward's validated (tokens, positions) -> owned rows and slots
    # (`begin_forward`): every layer writes the same rows, so the bounds check
    # and the ownership mask are computed once, not per layer per table.
    _forward_memo: dict = field(default_factory=dict, repr=False, compare=False)
    _kept_identity: dict = field(default_factory=dict, repr=False, compare=False)

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
                 world=1, rank=0, fused_records=False, page_size=64):
        c = config.text
        cache = cls(config=config, rows=rows, capacity=capacity, device=device, dtype=dtype,
                    world=world, rank=rank, page_size=page_size)
        layers = config.layers + int(draft)
        width = c["kv_lora_rank"] + c["qk_rope_head_dim"]
        if fused_records:
            # The engine contract (M2): fused 656-byte FP8 records only — the
            # decoded BF16 twin is never resident (it would double the record
            # cost: 1,152 B vs the 656 B contract). CUDA only: the record
            # writer is a GPU kernel.
            from .config import LATENT_RECORD_BYTES

            if device.type != "cuda":
                raise ValueError("fused record caches run on CUDA (the record writer is a kernel)")
            if dtype != torch.bfloat16:
                raise ValueError("the fused record writer consumes BF16 latent rows")
            slots = rows * cache.local_capacity
            pages = -(-slots // page_size)   # ceil; the flat slot space stays rows*capacity
            for layer in range(layers):
                cache.records[layer] = torch.zeros((pages, page_size, LATENT_RECORD_BYTES),
                                                   dtype=torch.uint8, device=device)
        else:
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
        if row_of.is_cuda:
            from .attention import diet, forward_memo

            if diet("diet_index"):
                # round 15: once per forward (every layer asks for the same rows)
                return forward_memo(row_of, ("slots_for", self.local_capacity),
                                    lambda: row_of.long() * self.local_capacity)
        return (row_of.long() * self.local_capacity)

    def allocate_short(self, rows, capacity):
        """The replicated short-context record tables (fused records only)."""
        from .config import LATENT_RECORD_BYTES

        if not self.records or capacity <= 0:
            return 0
        self.short_capacity = int(capacity)
        pages = -(-(int(rows) * self.short_capacity) // self.page_size)
        for layer in self.records:
            self.short[layer] = torch.zeros((pages, self.page_size, LATENT_RECORD_BYTES),
                                            dtype=torch.uint8, device=self.device)
        return sum(t.numel() for t in self.short.values())

    def short_latents(self, layer):
        return self.short[layer]

    def _write_short(self, layer, tokens, positions, latent):
        """Every token below the short capacity into this layer's short table."""
        from b12x.attention import sparse_mla

        S = self.short_capacity
        pos = positions.reshape(-1).long()
        slots = torch.where(pos < S, tokens.reshape(-1).long() * S + pos, torch.full_like(pos, -1))
        if self.short_writer_plan is None:
            c = self.config.text
            kv_c = torch.zeros((1, c["kv_lora_rank"] + c["qk_rope_head_dim"]),
                               dtype=torch.bfloat16, device=self.device)
            one = torch.zeros((1,), dtype=torch.int64, device=self.device)
            self.short_writer_plan = sparse_mla.plan_cache_writer(kv_c, self.short[layer], one)
        sparse_mla.concat_and_cache_glm_next_mla(
            latent.reshape(int(pos.shape[0]), -1).to(torch.bfloat16).contiguous(), self.short[layer],
            slots.contiguous(), plan=self.short_writer_plan)

    def write_latent(self, layer, tokens, positions, latent):
        """Store this layer's latent records at the tokens' OWNED physical slots.

        Fused record caches (CUDA) route through the `concat_and_cache_glm_next_mla`
        record writer — the same paged FP8 writer contract the sparse-MLA
        readers consume (`_GLM_NSA_RECORD_BYTES` per record).
        """
        if layer in self.records:
            if layer in self.short:
                self._write_short(layer, tokens, positions, latent)
            kept, slots = self._owned_slots(tokens, positions)
            if int(kept.numel()):
                rows_in = latent
                if not (self._kept_identity.get(id(tokens)) is tokens and int(latent.shape[0]) == 1
                        and latent.is_cuda and _diet2()):
                    rows_in = latent.index_select(0, kept)
                self._write_records(layer, rows_in, slots)
            return
        self._store(self.latent[layer], tokens, positions, latent)

    def _write_records(self, layer, values, slots):
        """One writer call over the kept rows (skipped slots are negative)."""
        from b12x.attention import sparse_mla

        if self.writer_plan is None:
            self.writer_plan = self._make_writer_plan()
        sparse_mla.concat_and_cache_glm_next_mla(
            values.to(torch.bfloat16).contiguous(), self.records[layer],
            slots.to(torch.int64).contiguous(), plan=self.writer_plan)

    def _make_writer_plan(self):
        """Declare the fused-record writer plan (prepared on first use)."""
        from b12x.attention import sparse_mla

        cache = next(iter(self.records.values()))
        c = self.config.text
        kv_c = torch.zeros((1, c["kv_lora_rank"] + c["qk_rope_head_dim"]),
                           dtype=torch.bfloat16, device=self.device)
        slots = torch.zeros((1,), dtype=torch.int64, device=self.device)
        return sparse_mla.plan_cache_writer(kv_c, cache, slots)

    def write_index_keys(self, owner, tokens, positions, keys):
        self._store(self.index[owner], tokens, positions, keys)

    def _store(self, table, tokens, positions, values):
        memo = self._forward_memo.get(id(tokens))
        if memo is not None and len(memo) == 5 and memo[0] is tokens and memo[1] is positions:
            # consult18 P2 decode graphs: one token, fixed shapes. A rank that
            # does not own it rewrites one valid slot with its own value (a
            # no-op); the owner stores the token's value — the same bytes as
            # the indexed assignment below.
            _, _, kept, slots, safe = memo
            if table.is_cuda and values.dim() == 2:
                from .attention import diet

                if diet("diet_index"):
                    # round 15: one launch — owned rows write, the rest stay untouched
                    from .fused_ops import store_owned_rows

                    store_owned_rows(table, values, kept, slots, safe)
                    return
            current = table.index_select(0, safe)
            fresh = values.to(table.dtype).index_select(0, kept)
            own = (slots >= 0).reshape(-1, *([1] * (table.dim() - 1)))
            table.index_copy_(0, safe, torch.where(own, fresh, current))
            return
        kept, slots = self._owned_slots(tokens, positions)   # store only owned records (D1)
        table[slots] = values.to(table.dtype).index_select(0, kept)

    def begin_forward(self, tokens, positions):
        """Validate one forward's rows ONCE; its layers then write without host reads.

        `tokens` / `positions` are the forward's own per-token tensors: the
        memo answers only for these exact objects, so any other write (the MTP
        block, a test) still takes the full bounds check.
        """
        self._forward_memo.pop(id(tokens), None)
        kept, slots = self._owned_slots(tokens, positions)
        # consult18 stage D: one entry per in-flight forward (two halves).
        self._forward_memo[id(tokens)] = (tokens, positions, kept, slots)

    def begin_forward_host(self, tokens, positions, *, row, start):
        """`begin_forward` for ONE request row at the contiguous positions
        `[start, start + len(positions))`, with every bounds check and the
        ownership filter done on host integers (consult18 P1/P2: no device
        read). Same `(kept, slots)` values as the device path: kept are the
        token indices `i` with `(start + i) % world == rank`, slots are
        `row * local_capacity + (start + i) // world`."""
        width = int(positions.shape[0])
        if start < 0 or start + width > self.capacity:
            raise IndexError(f"cache position out of range: a row holds {self.capacity} slots")
        if not 0 <= int(row) < self.rows:
            raise IndexError(f"cache row out of range: {self.rows} request rows")
        last_local = (start + width - 1) // self.world if width else 0
        if last_local >= self.local_capacity:
            raise IndexError(
                f"cache local slot out of range: a row holds {self.local_capacity} local "
                f"slots (capacity {self.capacity}, world {self.world})")
        first = (self.rank - start) % self.world
        # No owned token (decode on a non-owning rank): an empty range, not an
        # arange whose start passes its end.
        count = (width - first + self.world - 1) // self.world if first < width else 0
        kept = (first + self.world * torch.arange(count, device=self.device, dtype=torch.long))
        local0 = (start + first) // self.world
        slots = (int(row) * self.local_capacity + local0
                 + torch.arange(count, device=self.device, dtype=torch.long))
        self._forward_memo.pop(id(tokens), None)
        self._forward_memo[id(tokens)] = (tokens, positions, kept, slots)

    def begin_forward_static(self, tokens, positions, *, slot, safe, kept=None):
        """consult18 P2 decode graphs: `begin_forward` for ONE token with
        device-resident slots (fixed shapes, no host value). `slot` [1] is the
        token's physical slot on its owning rank and -1 elsewhere (the fused
        record writer skips negative slots); `safe` [1] is always a valid slot
        (the owner's slot; elsewhere any in-row slot, rewritten unchanged).
        consult18 P4: `kept` [n] (default [0]) are the forward's token indices
        behind `slot`/`safe` [n] — a T-token forward on a rank owning fewer
        than n of them carries -1 slots (distinct `safe` slots) for the rest."""
        identity = kept is None and int(tokens.numel()) == 1
        if kept is None:
            kept = torch.zeros((1,), dtype=torch.long, device=self.device)
        self._forward_memo.pop(id(tokens), None)
        self._forward_memo[id(tokens)] = (tokens, positions, kept, slot, safe)
        self._kept_identity.pop(id(tokens), None)
        if identity:
            # round 15: one token, kept = [0] — its index_select is the row itself
            self._kept_identity[id(tokens)] = tokens

    def begin_forward_device(self, tokens, positions, kept, slots):
        """consult18 P2 tail graphs: `begin_forward_host`'s `(kept, slots)` already
        computed on device (fixed shapes; the caller checked the bounds on host)."""
        self._forward_memo.pop(id(tokens), None)
        self._forward_memo[id(tokens)] = (tokens, positions, kept, slots)

    def end_forward(self):
        self._forward_memo.clear()
        self._kept_identity.clear()

    def _owned_slots(self, tokens, positions):
        """`(kept, slots)`: the row indices this rank owns and their physical slots.

        Bounds-checked over EVERY position (owned or not) before the ownership
        filter — the check that refuses a write past the row.
        """
        memo = self._forward_memo.get(id(tokens))
        if memo is not None and memo[0] is tokens and memo[1] is positions:
            return memo[2], memo[3]
        positions = positions.long()
        slots = self._slots(tokens, positions)
        kept = (self.owner(positions) == self.rank).nonzero(as_tuple=True)[0]
        return kept, slots.index_select(0, kept)

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
        tokens = tokens.long()
        if bool((tokens < 0).any()) or bool((tokens >= self.rows).any()):
            raise IndexError(f"cache row out of range: {self.rows} request rows")
        local_index = self.local(positions)
        if bool((local_index < 0).any()) or bool((local_index >= self.local_capacity).any()):
            raise IndexError(
                f"cache local slot out of range: a row holds {self.local_capacity} local "
                f"slots (capacity {self.capacity}, world {self.world}) — a mapping that "
                f"lands past the row would alias another request's records")
        return self.slots_for(tokens) + local_index

    def latents(self, layer):
        """This layer's cache table: fused 656-byte records when the cache is
        fused (the `_attend_cuda` bind form), the decoded BF16 table otherwise."""
        if layer in self.records:
            return self.records[layer]
        return self.latent[layer]

    def decode_latents(self, layer):
        """Decode the fused records back to [slots, kv_lora + rope] BF16.

        Diagnostics only (the golden/diff surfaces): the engine keeps no
        decoded copy resident. Scales decode `e4m3 * fp32_scale`, RoPE is the
        raw BF16 tail — `b12x...reference.unpack_mla_kv_cache_reference`.
        """
        if layer not in self.records:
            return self.latent[layer]
        from b12x.attention._shared.mla.reference import unpack_mla_kv_cache_reference

        pages, page_size, _ = self.records[layer].shape
        flat = self.records[layer].reshape(pages * page_size, 1, -1)
        return unpack_mla_kv_cache_reference(flat)[:, 0]

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

    def __init__(self, weights: Hy4ModelWeights, cache: Hy4Cache, *, compute_dtype=None,
                 collectives=None, prefill_chunk_size=None):
        from .collectives import SoloCollectives

        self.weights = weights
        self.cache = cache
        self.config = weights.config
        self.device = weights.device
        self.compute_dtype = torch.bfloat16 if compute_dtype is None else compute_dtype
        self.prefill_chunk_size = prefill_chunk_size
        # consult17 (lever 9): "latent" exchanges the kernel's 512-wide latent
        # partials and expands this rank's shard after the merge; "expanded"
        # expands every head first on PREFILL graphs and exchanges 256-wide
        # values (decode graphs keep the latent wire: one row, owner-side reads).
        self.partial_wire = "latent"
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
        self.trace = []          # the wiring trace the W7 gates read
        self.plan_book = None    # the `KernelPlanBook` (bound by wire_operators)
        self._mla_scratch_pool = {}  # Patch C: shared sparse-MLA scratch per rank

    # --- W7 wiring ---------------------------------------------------------
    def default_call_shape(self, shape):
        """The call shape of a `(rows, tokens)` forward.

        One token per row is the decode family: the ONE-ROW decode shape at
        rows == 1, the distinct verification-rows shape above it. Prompt
        tokens are the prefill family (`prefill_shape`).
        """
        rows, width = shape
        if width == 1:
            return DECODE_ONE_ROW if rows == 1 else verify_shape(rows)
        return prefill_shape(rows, width)

    def bind_plan_book(self, book, shapes, *, sink_variants=(True, False),
                       share_scratch=False, lanes=1):
        """Wire prepared plans (from `wire_operators`) into every layer.

        One `sparse_mla` plan per call shape plus the sink folds; from here
        the body refuses unprepared call shapes (no JIT at run time).
        `share_scratch=True` shares one sparse-MLA scratch across every layer
        (Patch C: the layer loop is serial per rank).
        """
        self.plan_book = book
        scratch_pool = self._mla_scratch_pool if share_scratch else None
        # consult18 stage D: one shared scratch pool per extra lane.
        if not hasattr(self, "_lane_scratch_pools"):
            self._lane_scratch_pools = []
        while len(self._lane_scratch_pools) < lanes - 1:
            self._lane_scratch_pools.append({})
        attentions = [layer.attention for layer in self.weights.layers]
        if getattr(self.weights, "draft", None) is not None:
            # consult18 P4: the MTP draft's CP attention binds the same shapes
            # (decode steps, prefill-shaped prompt absorption).
            attentions.append(self.weights.draft.attention)
        for attention in attentions:
            attention.bind_plan_book(book)
            attention.prepare_plans(shapes, sink_variants=sink_variants,
                                    scratch_pool=scratch_pool,
                                    expanded_partials=self.partial_wire == "expanded",
                                    lane_pools=self._lane_scratch_pools[:lanes - 1])
        return self

    # --- entry points ------------------------------------------------------

    def prefill(self, token_ids, *, request_rows=None, state=None, call_shape=None):
        """Chunked prefill: `token_ids` [rows, tokens] continues the rows' prompts.

        Chunks are at most `prefill_chunk_size` tokens per row, cut by
        `wiring.chunk_plan` — DETERMINISTIC and LENGTH-ONLY (consult8 §5:
        "deterministic, length-only chunking"; K3.1 fixed a 21 % TTFT bump at
        one prompt length this way). The chunk boundaries depend on the token
        counts and the chunk size and nothing else — never on token ids,
        scores or iteration order (the chunk-position nondeterminism mutant
        fails exactly this). The loop is the same one `decode_step` runs, with
        every query token attending over its causal cache length; per-chunk
        row positions advance inside the call and the caller's `complete`
        bookkeeping is unchanged.
        """
        rows, width = int(token_ids.shape[0]), int(token_ids.shape[1])
        chunk = self.prefill_chunk_size
        if not chunk or width <= chunk:
            return self.forward(token_ids, request_rows=request_rows, state=state,
                                call_shape=call_shape)
        spans = sorted({(start, stop) for _, start, stop in chunk_plan([width] * rows, chunk)})
        row_ids = self._rows(token_ids, request_rows).tolist()
        saved = {row: self._row_lengths.get(row, 0) for row in row_ids}
        parts = []
        for start, stop in spans:
            for row in row_ids:
                self._row_lengths[row] = saved[row] + start
            self.trace.append(("chunk", start, stop))
            parts.append(self.forward(token_ids[:, start:stop], request_rows=request_rows,
                                      state=state,
                                      call_shape=self.default_call_shape((rows, stop - start))))
        for row, base in saved.items():
            self._row_lengths[row] = base
        return torch.cat(parts, dim=1)

    def decode_step(self, token_ids, *, request_rows=None, state=None, call_shape=None):
        """One decode step: `token_ids` [rows, 1] appends to the rows' contexts.

        Rows == 1 is the one-row decode shape; more rows is the distinct
        verification-rows shape (consult7 §5: one-row decode graphs never
        share a capture with the verification graph).
        """
        if token_ids.shape[1] != 1:
            raise ValueError("decode_step advances one token per row")
        return self.forward(token_ids, request_rows=request_rows, state=state,
                            call_shape=call_shape)

    def draft_absorb(self, next_tokens, prev_hidden, *, request_rows=None, cache=None,
                     logits="all", next_embeds=None):
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
        import os as _os

        cap = int(_os.environ.get("HY4_ABSORB_ROWS") or 0)
        if cap > 0:
            # consult18 P9: absorption pieces bounded independent of the prefill
            # chunk (a 2,048-row chunk's pair absorbs 4,096 rows): batching only
            limit = min(limit, cap)
        outs, sels = [], []
        for lo in range(0, k, limit):
            hi = min(lo + limit, k)
            positions = (starts.reshape(-1, 1)
                         + torch.arange(lo, hi, device=self.device)).reshape(-1)
            hidden, _, selected = self._draft(
                next_tokens[:, lo:hi], prev_hidden[:, lo:hi], positions, rows, cache,
                embeds=None if next_embeds is None else next_embeds.reshape(
                    next_tokens.shape[0], k, -1)[:, lo:hi].reshape(-1, next_embeds.shape[-1]))
            outs.append(hidden.reshape(next_tokens.shape[0], hi - lo, -1))
            # snapshot: `select` hands back the selection cell's LIVE buffer
            # (shared with `sparse_mla.bind`), and the next chunk's
            # `write_selection` (or any later step's) overwrites it in place.
            sels.append(selected.reshape(next_tokens.shape[0], hi - lo, -1).clone())
        hidden = torch.cat(outs, dim=1)
        selected = torch.cat(sels, dim=1)
        for row, start in zip(rows.tolist(), starts.tolist()):
            self._draft_positions[row] = start + k
        # consult18 P4: `logits="last"` heads only each row's last absorbed
        # position (the speculative round's first draft), "none" skips the head
        # (prompt absorption): [rows, 1, V] / None.
        if logits == "none":
            return None, hidden, selected
        if logits == "last":
            last = hidden[:, -1, :]
            return (self._draft_logits(last.reshape(-1, last.shape[-1]))
                    .reshape(next_tokens.shape[0], 1, -1), hidden, selected)
        out = self._draft_logits(hidden.reshape(-1, hidden.shape[-1]))
        return (out.reshape(next_tokens.shape[0], k, -1), hidden, selected)

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
        return self._draft_logits(hidden), hidden

    def greedy_tokens(self, logits):
        """consult18 P4: greedy tokens `[rows]` from this rank's vocabulary-shard
        logits `[rows, V/w]`, on device and identical on every rank."""
        if int(getattr(self.collectives, "world_size", 1)) == 1:
            return logits.reshape(int(logits.shape[0]), -1).argmax(dim=-1)
        return self.collectives.argmax_tokens(
            logits, vocabulary_range=self.weights.sharding.vocabulary_range)

    def head_logits(self, x):
        """FP32 logits over this rank's vocabulary shard: `x.float() @ head.T`.
        consult18 P7: decode-family rows (w8.decode_scope) read a BF16 copy of
        the head (`_head_bf16`, the checkpoint's own values — the FP32 head is
        an upcast) through cuBLAS with FP32 output: half the bytes, the same
        FP32 math up to summation order."""
        head16 = getattr(self, "_head_bf16", None)
        if head16 is not None:
            from .w8 import in_decode_scope

            if in_decode_scope(x):
                x2 = x.reshape(-1, x.shape[-1]).to(torch.bfloat16)
                return torch.mm(x2, head16.T, out_dtype=torch.float32)
        return x.float() @ self.weights.head.T

    def _draft_logits(self, hidden):
        """The shared head over the draft hidden: full-vocabulary FP32 logits
        (the vocabulary shards gathered under tensor parallelism)."""
        logits = self.head_logits(hidden)
        if int(getattr(self.collectives, "world_size", 1)) > 1:
            logits = self.collectives.gather_logits(
                logits, vocabulary_range=self.weights.sharding.vocabulary_range)
        return logits

    def _draft(self, next_tokens, prev_hidden, positions, rows, cache, embeds=None):
        """Shared draft body: embed, run the block at `positions`; (hidden, attended, selected)."""
        cache = self.cache if cache is None else cache
        if cache.world != 1:
            return self._draft_cp(next_tokens, prev_hidden, positions, rows, cache, embeds=embeds)
        token_rows = self._token_rows(next_tokens.shape, rows)
        visible = positions + 1
        next_embeds = self.weights.embed[next_tokens.reshape(-1)].to(self.compute_dtype)
        hidden, attended, selected = self.weights.draft.forward(
            next_embeds, prev_hidden.to(self.compute_dtype).reshape(-1, prev_hidden.shape[-1]),
            positions, cache, visible, token_rows=token_rows)
        return hidden, attended, selected

    def _draft_cp(self, next_tokens, prev_hidden, positions, rows, cache, *, call_shape=None,
                  visible_max=None, static_slots=None, embeds=None):
        """consult18 P4: the MTP draft block under context + tensor parallelism.

        The same map as `MtpBlock.forward` with the target layers' distributed
        pieces: the vocabulary-sharded embedding (+ all-reduce), the replicated
        `eh_proj` merge and norms, the CP attention of layer `config.layers`
        through `_attention` (owned record/index-key writes, local candidates
        -> score exchange -> hand-back, head gather -> partial -> LSE
        reduce-scatter -> one sink fold -> gated head-shard o_proj) + all-reduce,
        and the intermediate-sharded MoE + all-reduce. Returns (hidden after
        `final_layernorm`, None, this rank's local selection)."""
        from types import SimpleNamespace

        draft = self.weights.draft
        eps = self.config.text["rms_norm_eps"]
        if cache is not self.cache:
            raise ValueError("the CP draft path writes the decoder's own cache")
        shape = tuple(next_tokens.shape)
        token_rows = self._token_rows(shape, rows)
        visible = positions + 1
        if call_shape is None:
            call_shape = self.default_call_shape(shape)
        if embeds is None:
            embeds = self.collectives.embed_lookup(
                next_tokens.reshape(-1), self.weights.embed,
                vocabulary_range=self.weights.sharding.vocabulary_range).to(self.compute_dtype)
        else:
            # consult18 P5: the caller's all-reduced embeddings of exactly these
            # tokens (the target forward's next rows) — the same values
            embeds = embeds.to(self.compute_dtype)
        embeds = torch.where((positions == 0).unsqueeze(-1), 0, embeds)
        h = draft.embed_merge(embeds, prev_hidden.to(self.compute_dtype).reshape(-1, embeds.shape[-1]))
        x = rms_norm(h, draft.input_norm, eps)
        layer = SimpleNamespace(attention=draft.attention, index=self.config.layers)
        if static_slots is not None:
            # consult18 P4 draft graphs: predicated owned writes (device slots).
            kept, slot, safe = static_slots
            cache.begin_forward_static(token_rows, positions, slot=slot, safe=safe, kept=kept)
        try:
            attn_out = self._attention(layer, x, positions, token_rows, visible,
                                       call_shape=call_shape, visible_max=visible_max)
        finally:
            if static_slots is not None:
                cache._forward_memo.pop(id(token_rows), None)
        h = h + self.collectives.all_reduce(attn_out)
        x = rms_norm(h, draft.post_attention_norm, eps)
        mlp_out = draft.mlp.forward(x, overlap_shared=call_shape.overlap_shared_expert)
        h = h + self.collectives.all_reduce(mlp_out)
        hidden = rms_norm(h, draft.final_norm, eps)
        selected, _ = draft.attention.selection.buffers(rows=x.shape[0])
        return hidden, None, selected

    def forward(self, token_ids, *, request_rows=None, state=None, return_hidden=False,
                call_shape=None, logits_rows="all"):
        """The real body loop: iHC streams -> 78 gated sparse-MLA layers -> head.

        Returns FP32 logits over this rank's vocabulary shard. `request_rows`
        maps each input row to its cache row (default: identity). With
        `return_hidden=True` returns `(logits, hidden_post, hidden_pre)`:
        `hidden_post` [rows, tokens, hidden] is the POST-`model.norm` hidden
        (the MTP feed and the lm_head pre-image), `hidden_pre` the pre-norm
        `hc_head` output (diagnostics / the pre-norm-feed mutant only).
        `call_shape` is the forward's fixed shape and graph-family wiring
        (`wiring.CallShape`); it defaults to `default_call_shape` and is
        recorded in `self.trace` for the wiring gates.

        `logits_rows` narrows the head's output (consult16 lever 3): `"all"`
        (default) returns `[rows, tokens, V]`; `"last"` or a token index
        returns that one token's logits `[rows, 1, V]` — sliced from the SAME
        local head product, so the values are bit-identical to the all-rows
        form and only that row is gathered across the vocabulary shards;
        `"none"` skips the head and the gather and returns `None` (a prefill
        chunk that is not the prompt's last).
        """
        if token_ids.ndim != 2:
            raise ValueError("token_ids must be [rows, tokens]")
        if logits_rows not in ("all", "last", "none"):
            if type(logits_rows) is not int or not 0 <= logits_rows < token_ids.shape[1]:
                raise ValueError(
                    "logits_rows must be 'all', 'last', 'none' or a token index inside the forward")
        if call_shape is None:
            call_shape = self.default_call_shape(token_ids.shape)
        if (call_shape.rows, call_shape.tokens) != tuple(token_ids.shape):
            raise ValueError(
                f"call shape {call_shape.name} (rows={call_shape.rows}, "
                f"tokens={call_shape.tokens}) does not match the forward's "
                f"{tuple(token_ids.shape)}")
        self.trace.append(("call_shape", call_shape))
        import os as _os

        if (_os.environ.get("HY4_PLAIN_SYNC_CENSUS") and token_ids.is_cuda
                and not torch.cuda.is_current_stream_capturing()):
            # consult18 P1 census: every distinct synchronizing site of a plain
            # forward (prefill tails / single chunks / decode), printed once.
            previous = torch.cuda.get_sync_debug_mode()
            _sync_site_reporter.install(getattr(self.collectives, "rank", 0))
            torch.cuda.set_sync_debug_mode("warn")
            try:
                return self._forward_body(token_ids, request_rows=request_rows, state=state,
                                          return_hidden=return_hidden, call_shape=call_shape,
                                          logits_rows=logits_rows)
            finally:
                torch.cuda.set_sync_debug_mode(previous)
                _sync_site_reporter.uninstall()
        return self._forward_body(token_ids, request_rows=request_rows, state=state,
                                  return_hidden=return_hidden, call_shape=call_shape,
                                  logits_rows=logits_rows)

    def _forward_body(self, token_ids, *, request_rows, state, return_hidden, call_shape,
                      logits_rows):
        TIMER.forward_begin(call_shape.name)   # W29: no-op unless --profile-steps armed
        rows = self._rows(token_ids, request_rows)
        # consult18 P1/P2: a single-row forward can run sync-free — positions,
        # bounds, ownership and the indexer geometry from host-known row
        # lengths; the selection checks deferred to one read at the end.
        from .runtime import SWITCHES

        on_device = token_ids.is_cuda or getattr(self, "_sync_free_on_cpu", False)  # CPU: tests
        mode = SWITCHES.get("sync_free")
        graphs = getattr(self, "decode_graphs", None)
        one_row = (getattr(self, "_sync_free_allowed", True) and on_device
                   and int(token_ids.shape[0]) == 1
                   and (request_rows is None or len(request_rows) == 1)
                   and not (token_ids.is_cuda and torch.cuda.is_current_stream_capturing()))
        tail_graph = False
        if (one_row and graphs is not None and SWITCHES.get("prefill_graph") == "on"
                and int(token_ids.shape[1]) > 1):
            row_id = 0 if request_rows is None else int(request_rows[0])
            tail_graph = graphs.accepts_tail(
                token_ids, row=row_id, start=int(getattr(self, "_row_lengths", {}).get(row_id, 0)),
                call_shape=call_shape, state=state, return_hidden=return_hidden)
        sync_free = one_row and (tail_graph or mode == "on"
                                 or (mode == "decode" and int(token_ids.shape[1]) == 1))
        if sync_free:
            row_id = 0 if request_rows is None else int(request_rows[0])
            start = int(self._row_lengths.get(row_id, 0))
            width = int(token_ids.shape[1])
            if tail_graph:
                # consult18 P2: the captured single-row prefill forward (the
                # prompt's tail); logits for every row are computed in the
                # graph, the requested one is sliced and gathered here.
                tail_hidden = None
                if return_hidden:
                    local, normed, pre, flag = graphs.run_tail(
                        token_ids, row=row_id, start=start, call_shape=call_shape,
                        need_logits=logits_rows != "none", need_hidden=True)
                    tail_hidden = (normed, pre)
                else:
                    local, flag = graphs.run_tail(token_ids, row=row_id, start=start,
                                                  call_shape=call_shape,
                                                  need_logits=logits_rows != "none")
                if logits_rows == "none":
                    result = None
                else:
                    if logits_rows != "all":
                        index = width - 1 if logits_rows == "last" else int(logits_rows)
                        local = local[:, index:index + 1, :].contiguous()
                    result = self.collectives.gather_logits(
                        local, vocabulary_range=self.weights.sharding.vocabulary_range)
                if tail_hidden is not None:
                    result = (result, *tail_hidden)
                flags = [flag]
            elif (graphs is not None and SWITCHES.get("decode_graph") == "on"
                    and graphs.accepts(token_ids, row=row_id, start=start, call_shape=call_shape,
                                       state=state, return_hidden=return_hidden,
                                       logits_rows=logits_rows)):
                # consult18 P2: the captured one-row step (decode_graph.py) —
                # local-shard logits and the deferred selection flag; the
                # vocabulary gather and the flag read stay here.
                local, flag = graphs.run(token_ids, row=row_id, start=start,
                                         call_shape=call_shape, logits_rows=logits_rows)
                root = {"root_host": True} if SWITCHES.get("root_logits") == "on" else {}
                result = self.collectives.gather_logits(
                    local, vocabulary_range=self.weights.sharding.vocabulary_range, **root)
                flags = [flag]
            else:
                positions = torch.arange(start, start + width, device=self.device,
                                         dtype=torch.long)
                token_rows = self._token_rows(token_ids.shape, rows)
                visible = positions + 1
                self.cache.begin_forward_host(token_rows, positions, row=row_id, start=start)
                visible_max = (start + width
                               if self.collectives.world_size > 1 and call_shape.short_context_bypass
                               and call_shape.is_decode_graph else None)
                world = int(getattr(self.collectives, "world_size", 1))
                rank = int(getattr(self.collectives, "rank", 0))
                end = start + width
                geometry = {"owned_max": (max(end - rank, 0) + world - 1) // world,
                            "base": row_id * int(getattr(self.cache, "local_capacity", 0))}
                deferred = getattr(self.collectives, "deferred_checks", None)
                if deferred is not None:
                    deferred.clear()
                from .lanes import use_lane

                with use_lane(0, overlap=False, geometry=geometry, sync_free=True):
                    result = self._layers_and_head(
                        token_ids, rows, positions, token_rows, visible, visible_max, state=state,
                        return_hidden=return_hidden, call_shape=call_shape,
                        logits_rows=logits_rows)
                flags = list(deferred) if deferred else []
                if deferred is not None:
                    deferred.clear()
            if (flags and bool(torch.stack([f.reshape(()) for f in flags]).any())
                    and SWITCHES.get("wire_probe", "off") == "off"):
                # A narrow scores-only cut needed the full re-offer (or the
                # exactness test fired): redo the forward on the exact path
                # (same rows, same slots; it rewrites this forward's records).
                self.sync_free_reruns = getattr(self, "sync_free_reruns", 0) + 1
                self._sync_free_allowed = False
                try:
                    return self._forward_body(token_ids, request_rows=request_rows, state=state,
                                              return_hidden=return_hidden, call_shape=call_shape,
                                              logits_rows=logits_rows)
                finally:
                    self._sync_free_allowed = True
            self.sync_free_forwards = getattr(self, "sync_free_forwards", 0) + 1
            return result
        positions = self._positions(token_ids.shape, rows)
        token_rows = self._token_rows(token_ids.shape, rows)
        visible = positions + 1  # each query's causal cache length
        # One bounds check and ownership mask for the whole forward (lever 4).
        self.cache.begin_forward(token_rows, positions)
        # The short-context bypass test reads the device once per forward, not
        # once per layer (decode graphs under context parallelism only).
        visible_max = (int(visible.max())
                       if self.collectives.world_size > 1 and call_shape.short_context_bypass
                       and call_shape.is_decode_graph else None)
        return self._layers_and_head(token_ids, rows, positions, token_rows, visible, visible_max,
                                     state=state, return_hidden=return_hidden,
                                     call_shape=call_shape, logits_rows=logits_rows)

    def _layers_and_head(self, token_ids, rows, positions, token_rows, visible, visible_max, *,
                         state, return_hidden, call_shape, logits_rows, gather=True):
        # Vocabulary-sharded embedding (D5): masked lookup over this rank's
        # shard + all-reduce (identity at world 1).
        WATCH.begin_forward(call_shape.name, positions)   # consult17: no-op unless armed
        with TIMER.phase("embed"):
            embeds = self.collectives.embed_lookup(
                token_ids.reshape(-1), self.weights.embed,
                vocabulary_range=self.weights.sharding.vocabulary_range).to(self.compute_dtype)
        self._publish_embeds(token_ids, embeds)
        WATCH.check("embed", -1, embeds=embeds)
        streams = StreamState.begin(embeds)
        if state is not None:
            state.streams[:streams.streams.shape[0]] = streams.streams
        post = y = None
        mlp_out = None
        for index, layer in enumerate(self.weights.layers):
            if index == 0:
                with TIMER.phase("ihc"):
                    post, y = layer.attn_ihc.pre(streams.streams)
                    y = rms_norm(y, layer.input_norm, self.config.text["rms_norm_eps"])
            else:
                with TIMER.phase("ihc"):
                    streams.streams, post, y = previous_ffn.post_pre(
                        mlp_out, streams.streams, post, layer.attn_ihc,
                        norm_weight=layer.input_norm)
            WATCH.check("ihc", index, y=y, streams=streams.streams)
            attn_out = self._attention(layer, y, positions, token_rows, visible,
                                       call_shape=call_shape, visible_max=visible_max)
            WATCH.check("attn.fold_expand_gate", index, attn_out=attn_out)
            with TIMER.phase("comm.all_reduce.attn"):
                attn_out = self.collectives.all_reduce(attn_out)  # D4 (o_proj shard)
            WATCH.check("comm.all_reduce.attn", index, attn_out=attn_out)
            with TIMER.phase("ihc"):
                streams.streams, post, y = layer.attn_ihc.post_pre(
                    attn_out, streams.streams, post, layer.ffn_ihc,
                    norm_weight=layer.post_attention_norm)
            # D4: the MoE/dense-FFN outputs are intermediate-sharded partial
            # sums (the layer-0 dense FFN is `mlp.forward` at layer 0). The
            # shared-expert overlap is a per-graph-family decision (consult7 §5
            # / consult8 §6): decode graphs overlap, prefill graphs do not.
            # (`moe.route_shared` / `moe.routed` time inside `moe.py`.)
            WATCH.check("ihc.post_attention", index, y=y, streams=streams.streams)
            mlp_out = layer.mlp.forward(y, overlap_shared=call_shape.overlap_shared_expert)
            WATCH.check("moe", index, mlp_out=mlp_out)
            with TIMER.phase("comm.all_reduce.mlp"):
                mlp_out = self.collectives.all_reduce(mlp_out)
            WATCH.check("comm.all_reduce.mlp", index, mlp_out=mlp_out)
            previous_ffn = layer.ffn_ihc
        with TIMER.phase("ihc"):
            streams.streams = previous_ffn.post(mlp_out, streams.streams, post)
            hidden = self.weights.hc_head.merge(streams.streams)
            normed = rms_norm(hidden, self.weights.norm, self.config.text["rms_norm_eps"])
        # The state rows keep the POST-`model.norm` hidden — the exact tensor
        # the MTP's `hnorm` consumes (finding F; vLLM vllm_hy4_model.py:403,
        # SGLang sgl_hunyuan_v4.py:734, transformers candidate_generator.py:1418
        # all feed `model_outputs.hidden_states[-1]` == `norm(hc_head(...))`).
        WATCH.check("final", len(self.weights.layers), zero=True, normed=normed)
        streams.hidden = normed
        if state is not None:
            state.hidden[:normed.shape[0]] = normed
            state.count[:normed.shape[0]] += token_ids.shape[1]
        self.cache.end_forward()
        if logits_rows == "none":
            logits = None
        else:
            with TIMER.phase("head"):
                logits = self.head_logits(normed)
                logits = logits.reshape(token_ids.shape[0], token_ids.shape[1], -1)
                if logits_rows != "all":
                    token = token_ids.shape[1] - 1 if logits_rows == "last" else logits_rows
                    logits = logits[:, token:token + 1, :].contiguous()
            # D5: logits come back per vocabulary shard; all-gather to the
            # sampler (identity at world 1). `gather=False`: the captured
            # decode step returns the local shard (the gather runs on host).
            if gather:
                with TIMER.phase("comm.logits_gather"):
                    # consult18: the NLL probe's all-row logits go to rank 0's
                    # host only (a 2,048-row chunk's full-vocabulary FP32 logits
                    # are ~1 GB; all-gathered on every device they OOM).
                    root = ({"root_host": True} if getattr(self, "logits_root_host", False)
                            else {})
                    logits = self.collectives.gather_logits(
                        logits, vocabulary_range=self.weights.sharding.vocabulary_range, **root)
                WATCH.check("comm.logits_gather", len(self.weights.layers), zero=True,
                            logits=logits)
        TIMER.forward_end()
        if return_hidden:
            rows, width = token_ids.shape
            return (logits,
                    normed.reshape(rows, width, -1),
                    hidden.reshape(rows, width, -1))
        return logits

    # --- internals ---------------------------------------------------------

    # --- consult18 stage D: the two-half overlapped prefill -----------------
    def enable_lanes(self, *, max_q_rows):
        """Give lane 1 its own selection buffers and fused-MoE workspace.

        The sparse-MLA scratch for lane 1 comes from `bind_plan_book(lanes=2)`;
        the collectives' lane channels from the transport (`lanes=2`).
        """
        from .bootstrap import selection_cells
        from .moe import FusedRoutedExperts

        lane_cache, _ = selection_cells(self.config, self.device, max_q_rows=max_q_rows)
        seen = set()
        for layer in self.weights.layers:
            cell = layer.attention.selection
            if cell is not None and id(cell) not in seen:
                seen.add(id(cell))
                cell.lane_caches = (lane_cache,)
        workspace = storage = None
        for layer in self.weights.layers:
            fused = getattr(layer.mlp, "routed", None)
            if isinstance(fused, FusedRoutedExperts) and fused.op is not None:
                if workspace is None:
                    workspace = torch.empty(fused.op.workspace_nbytes, dtype=torch.uint8,
                                            device=self.device)
                    storage = [torch.empty_strided(shape, stride, dtype=dtype, device=self.device)
                               for shape, stride, dtype in fused.op.scale_layouts]
                elif workspace.numel() != fused.op.workspace_nbytes:
                    raise ValueError("the fused MoE layers disagree on their workspace size")
                fused._lane_scratch[1] = (workspace, storage)
        self.lanes = 2
        return self

    def _half_steps(self, token_ids, positions, token_rows, call_shape, *, prevalidated=False):
        """One half of an overlapped chunk: the body loop, yielding after each
        layer; returns the half's post-norm hidden. `prevalidated`: the caller
        ran `cache.begin_forward` for these exact tensors before the loop (its
        bounds checks read the device)."""
        visible = positions + 1
        if not prevalidated:
            self.cache.begin_forward(token_rows, positions)
        with TIMER.phase("embed"):
            embeds = self.collectives.embed_lookup(
                token_ids.reshape(-1), self.weights.embed,
                vocabulary_range=self.weights.sharding.vocabulary_range).to(self.compute_dtype)
        self._publish_embeds(token_ids, embeds)
        PROBE.add("embed", -1, embeds)
        streams = StreamState.begin(embeds)
        post = y = None
        mlp_out = None
        previous_ffn = None
        tracer = getattr(getattr(self.collectives, "transport", None), "tracer", None)
        for index, layer in enumerate(self.weights.layers):
            if tracer is not None:
                tracer.mark(f"L{index}")
            with TIMER.phase("ihc"):
                if index == 0:
                    post, y = layer.attn_ihc.pre(streams.streams)
                    y = rms_norm(y, layer.input_norm, self.config.text["rms_norm_eps"])
                else:
                    streams.streams, post, y = previous_ffn.post_pre(
                        mlp_out, streams.streams, post, layer.attn_ihc,
                        norm_weight=layer.input_norm)
            PROBE.add("y_attn_in", index, y)
            attn_out = yield from self._attention_gen(layer, y, positions, token_rows, visible,
                                                      call_shape=call_shape, visible_max=None)
            PROBE.add("attn_out", index, attn_out)
            with TIMER.phase("comm.all_reduce.attn"):
                attn_out = self.collectives.all_reduce(attn_out)
            PROBE.add("ar_attn", index, attn_out)
            yield ("comm", "ar_attn")
            with TIMER.phase("ihc"):
                streams.streams, post, y = layer.attn_ihc.post_pre(
                    attn_out, streams.streams, post, layer.ffn_ihc,
                    norm_weight=layer.post_attention_norm)
            PROBE.add("y_ffn_in", index, y)
            mlp_out = self._mlp_forward(layer, index, y, call_shape)
            PROBE.add("moe", index, mlp_out)
            with TIMER.phase("comm.all_reduce.mlp"):
                mlp_out = self.collectives.all_reduce(mlp_out)
            PROBE.add("ar_mlp", index, mlp_out)
            previous_ffn = layer.ffn_ihc
            yield ("layer_end", index)
        with TIMER.phase("ihc"):
            streams.streams = previous_ffn.post(mlp_out, streams.streams, post)
            hidden = self.weights.hc_head.merge(streams.streams)
            normed = rms_norm(hidden, self.weights.norm, self.config.text["rms_norm_eps"])
        return normed

    # -- consult18 P1: per-lane CUDA graphs of the MLP block -----------------
    # The overlapped prefill is host-bound; the MLP block (router, shared
    # expert, prepared routed experts) is ~18 % of the host time, has fixed
    # shapes at the lane's row count and reads nothing on the host. Each lane's
    # layer gets one graph, captured outside the overlap (capture_mlp_graphs),
    # replayed inside it when the `graph_moe` switch is on. Per lane one graph
    # pool and one static input/output pair (the lane's graphs run in order on
    # the lane's stream), so the memory is one layer's temporaries.
    def _mlp_forward(self, layer, index, y, call_shape):
        graphs = getattr(self, "_mlp_graphs", None)
        if graphs and y.is_cuda:
            from .lanes import lane, overlapped
            from .runtime import SWITCHES

            if SWITCHES.get("graph_moe") == "on" and overlapped():
                entry = graphs.get((lane(), index))
                if entry is not None and int(y.shape[0]) == entry[3]:
                    graph, static_in, static_out, _ = entry
                    static_in.copy_(y)
                    graph.replay()
                    return static_out
        return layer.mlp.forward(y, overlap_shared=call_shape.overlap_shared_expert)

    def capture_mlp_graphs(self, rows):
        """Capture every layer's MLP block per lane at `rows` (call after the
        warm overlapped pair: the lane streams and plans exist). Returns the
        number of graphs."""
        import torch

        from .lanes import use_lane

        streams = getattr(self, "_lane_streams", None)
        if streams is None:
            raise RuntimeError("capture_mlp_graphs needs the lane streams (run an overlapped pair first)")
        hidden = self.config.hidden
        graphs = {}
        try:
            from b12x._lib.compile_plan import pin_programs
            pin_programs()  # graph-referenced modules must never be reclaimed
        except ImportError:
            pass
        torch.cuda.synchronize(self.device)
        for lane_index, stream in enumerate(streams[:getattr(self, "lanes", 1)]):
            pool = torch.cuda.graph_pool_handle()
            static_in = torch.zeros((rows, hidden), dtype=self.compute_dtype, device=self.device)
            static_out = torch.empty_like(static_in)
            with use_lane(lane_index, overlap=False):
                for index, layer in enumerate(self.weights.layers):
                    shape = self.default_call_shape((1, rows))
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, pool=pool, stream=stream):
                        out = layer.mlp.forward(static_in,
                                                overlap_shared=shape.overlap_shared_expert)
                        static_out.copy_(out)
                    graphs[(lane_index, index)] = (graph, static_in, static_out, rows)
        torch.cuda.synchronize(self.device)
        self._mlp_graphs = graphs
        return len(graphs)

    def forward_overlapped(self, token_ids, *, half_shape, logits_rows="none", return_hidden=False):
        """consult18 stage D: one prefill chunk as two halves on two lanes.

        Half A (the first `half_shape.tokens` positions) runs on lane 0's
        stream, half B on lane 1's; layer l of B waits for layer l of A (B's
        queries read A's records and index keys), nothing of A waits for B.
        So B's layer l runs beside A's layer l+1 and one half's collectives
        overlap the other half's compute. Each lane uses only its own
        channels, selection buffers and scratch (`lanes.py`). Same results as
        two sequential half-width forwards; `logits_rows` as in `forward`.
        """
        from contextlib import nullcontext

        from .lanes import use_lane

        if getattr(self, "lanes", 1) < 2:
            raise RuntimeError("forward_overlapped needs enable_lanes() first")
        half = int(half_shape.tokens)
        if token_ids.ndim != 2 or token_ids.shape[0] != 1 or token_ids.shape[1] != 2 * half \
                or half_shape.rows != 1:
            raise ValueError("forward_overlapped takes one row of exactly two halves")
        if logits_rows not in ("all", "last", "none") and (
                type(logits_rows) is not int or not 0 <= logits_rows < 2 * half):
            raise ValueError(
                "logits_rows must be 'all', 'last', 'none' or a token index inside the forward")
        self.trace.append(("call_shape_overlapped", half_shape))
        TIMER.forward_begin(half_shape.name)
        rows = self._rows(token_ids, None)
        positions = self._positions(token_ids.shape, rows)
        token_rows = self._token_rows(token_ids.shape, rows)
        halves = ((token_ids[:, :half], positions[:half], token_rows[:half]),
                  (token_ids[:, half:], positions[half:], token_rows[half:]))
        cuda = self.device.type == "cuda"
        streams = (None, None)
        if cuda:
            main = torch.cuda.current_stream()
            if getattr(self, "_lane_streams", None) is None:
                self._lane_streams = (torch.cuda.Stream(device=self.device),
                                      torch.cuda.Stream(device=self.device))
            streams = self._lane_streams
            tracer = getattr(getattr(self.collectives, "transport", None), "tracer", None)
            if tracer is not None:
                tracer.name_stream(streams[0], "lane0")
                tracer.name_stream(streams[1], "lane1")
            for stream in streams:
                stream.wait_stream(main)
        # consult18 D — the loop below must never move data between host and
        # device (the d3 repro: a staged host transfer queued behind one lane's
        # peer wait held the copy path that the peers' relay forwards and the
        # other lane's ring needed; the ranks deadlocked). Everything the
        # bookkeeping would read from the device is known on the host here:
        # one request row, a contiguous position range per half.
        row = 0
        start = int(self._row_lengths.get(row, 0))
        world = int(getattr(self.collectives, "world_size", 1))
        rank = int(getattr(self.collectives, "rank", 0))
        base = row * int(getattr(self.cache, "local_capacity", 0))
        geometries = []
        for index in (0, 1):
            end = start + (index + 1) * half              # the half's widest visible length
            geometries.append({"owned_max": (max(end - rank, 0) + world - 1) // world,
                               "base": base})
        for index in (0, 1):
            self.cache.begin_forward(halves[index][2], halves[index][1])
        deferred = getattr(self.collectives, "deferred_checks", None)
        if deferred is not None:
            deferred.clear()
        prepare = getattr(getattr(self.collectives, "transport", None), "prepare_lane_audit", None)
        if prepare is not None:
            prepare()
        steps = [self._half_steps(*halves[index], half_shape, prevalidated=True)
                 for index in (0, 1)]
        results = [None, None]
        layers = len(self.weights.layers)
        from .hangdump import lazy_preparation_trap

        import os as _os

        setting = (_os.environ.get("HY4_OVERLAP_SYNC_GUARD", "1") or "1").strip()
        guard = (cuda and not getattr(TIMER, "phases", False) and setting != "0"
                 and not getattr(getattr(self.collectives, "transport", None),
                                 "lane_audit_inloop", False))
        guard_mode = "warn" if setting == "warn" else "error"
        previous_mode = torch.cuda.get_sync_debug_mode() if guard else None
        # No b12x first-use preparation (a device-wide synchronize) between
        # two lanes' exchanges: the plain forward at this shape prepares first.
        # And no synchronizing transfer at all: the sync guard raises on one
        # ("warn" — a diagnosis boot — prints every distinct site with its stack).
        with lazy_preparation_trap("forward_overlapped") if cuda else nullcontext():
            if guard:
                if guard_mode == "warn":
                    _sync_site_reporter.install(getattr(self.collectives, "rank", 0))
                torch.cuda.set_sync_debug_mode(guard_mode)
            try:
                # Fine-grained two-lane schedule: the lanes advance one step at a
                # time, alternately (a step ends at this layer's cache writes or
                # after one collective), so their collectives interleave one at a
                # time in host order (the comm token keeps that order on the
                # device) and each lane's compute runs beside the other's comm.
                # Lane 1's layer l reads lane 0's layer-l records and index keys,
                # nothing else of lane 0: it starts layer l only after lane 0's
                # `kv_ready[l]` exists on the host, and its stream waits on it.
                # The gate. "layer" (the code default): lane 1 starts layer l once
                # lane 0 has FINISHED layer l. "kv" (HY4_OVERLAP_GATE=kv, the
                # production launcher since consult18 S2): at lane 0's layer-l
                # cache writes, so both lanes can be inside one layer. Every
                # mutable per-layer state the forward uses is per lane (selection
                # buffers, sparse-MLA scratch, MoE workspace, comm channels, the
                # packed-gate stash); exact verify 23 pairs x 8 ranks at 0 and 158
                # production pairs clean. Its old hang (layer 33) was the
                # cooperative-admission cycle, not shared plan state.
                gate_on = "kv" if (_os.environ.get("HY4_OVERLAP_GATE") or "layer") == "kv" else "layer_end"
                # consult18 S1 — the cooperative W4A16 grid and the other lane's
                # peer waits: either the grid leaves room for every resident
                # peer wait (the reserve contract, default) or it runs in the
                # comm token's section (HY4_OVERLAP_EXCLUSIVE_MOE=1). Neither:
                # refuse the overlap rather than risk the cross-rank cycle.
                transport = getattr(self.collectives, "transport", None)
                exclusive_moe = (_os.environ.get("HY4_OVERLAP_EXCLUSIVE_MOE") or "0") != "0"
                section = getattr(transport, "_comm_token", None) if exclusive_moe else None
                contract = getattr(transport, "admission_contract", None)
                unsafe = (_os.environ.get("HY4_OVERLAP_ADMISSION_UNSAFE") or "0") != "0"
                if cuda and section is None and contract is not None and contract() is None \
                        and not unsafe:
                    raise RuntimeError(
                        "forward_overlapped: no cooperative-admission regime — set "
                        "B12X_W4A16_COOP_RESERVE_SMS>=2 with B12X_PCIE_DMA_SINGLE_CTA_WAITS=1, "
                        "or HY4_OVERLAP_EXCLUSIVE_MOE=1")
                self.overlap_regime = ("exclusive MoE section" if section is not None
                                       else (contract() if contract is not None else "cpu")
                                       or "UNSAFE (diagnosis boot: no admission regime)")
                kv_ready = {}
                finished = [False, False]
                lane1 = {"layer": 0, "at_start": True}
                # consult18 P6: the host issue order of the two lanes' steps is
                # the device comm-token order. "alternate" (the default): one
                # step each in turn. "short_first": advance the lane whose next
                # step has less compute ahead of its collective (predicted from
                # its last label), so a collective that is ready soon is not
                # queued behind the other lane's MoE all-reduce (dg28: lanes sat
                # in ~760 us peer-flag waits beside the other lane's MoE). Labels
                # are the same on every rank, so is the order.
                from .runtime import SWITCHES as _order_switches

                order = _order_switches.get("overlap_order", "alternate")
                next_cost = {None: 250, "kv": 100, "scores": 50, "gather": 400, "lse": 150,
                             "ar_attn": 650, "layer_end": 250}
                last = [None, None]

                def key(label):
                    if label is None:
                        return None
                    return label[1] if label[0] == "comm" else label[0]

                def step0():
                    with use_lane(0, overlap=True, geometry=geometries[0],
                                  exclusive=section), \
                            (torch.cuda.stream(streams[0]) if cuda else nullcontext()):
                        try:
                            label = next(steps[0])
                        except StopIteration as done:
                            results[0], finished[0], label = done.value, True, None
                        if label is not None and label[0] == gate_on:
                            if cuda:
                                event = torch.cuda.Event()
                                event.record(streams[0])
                                kv_ready[label[1]] = event
                            else:
                                kv_ready[label[1]] = True
                    last[0] = key(label)

                def lane1_gated():
                    return lane1["at_start"] and lane1["layer"] < layers

                def lane1_ready():
                    return not finished[1] and not (lane1_gated() and lane1["layer"] not in kv_ready)

                def step1():
                    gated = lane1_gated()
                    with use_lane(1, overlap=True, geometry=geometries[1],
                                  exclusive=section), \
                            (torch.cuda.stream(streams[1]) if cuda else nullcontext()):
                        if cuda and gated:
                            streams[1].wait_event(kv_ready[lane1["layer"]])
                        try:
                            label = next(steps[1])
                        except StopIteration as done:
                            results[1], finished[1], label = done.value, True, None
                    if label is not None and label[0] == "layer_end":
                        lane1["layer"], lane1["at_start"] = label[1] + 1, True
                    elif label is not None:
                        lane1["at_start"] = False
                    last[1] = key(label)

                previous = 1
                while not (finished[0] and finished[1]):
                    if order != "short_first":
                        if not finished[0]:
                            step0()
                        if finished[1]:
                            continue
                        if not lane1_ready():
                            if finished[0]:
                                raise RuntimeError(
                                    f"lane 0 finished without layer {lane1['layer']}'s records")
                            continue
                        step1()
                        continue
                    ready0, ready1 = not finished[0], lane1_ready()
                    if ready0 and ready1:
                        c0, c1 = next_cost.get(last[0], 250), next_cost.get(last[1], 250)
                        pick = 0 if c0 < c1 else 1 if c1 < c0 else 1 - previous
                    elif ready0:
                        pick = 0
                    elif ready1:
                        pick = 1
                    else:
                        raise RuntimeError(
                            f"lane 0 finished without layer {lane1['layer']}'s records")
                    (step0 if pick == 0 else step1)()
                    previous = pick
            finally:
                if guard:
                    torch.cuda.set_sync_debug_mode(previous_mode)
                    if guard_mode == "warn":
                        _sync_site_reporter.uninstall()
        if cuda:
            for stream in streams:
                main.wait_stream(stream)
        # Post-hoc lane audit (HY4_LANE_AUDIT): compared here, after both lanes
        # joined the main stream — never between two lanes' exchanges.
        flush = getattr(getattr(self.collectives, "transport", None), "flush_lane_audit", None)
        if flush is not None:
            flush()
        # The narrow selections' widen/exactness flags, read once, here. A set
        # flag means some layer's scores-only cut may have needed the full
        # re-offer: recompute the pair on the single-lane path (it widens with
        # its host read, outside any overlap) — same positions, same slots.
        flags = list(deferred) if deferred else []
        if deferred is not None:
            deferred.clear()

        def single_lane():
            """The pair on the single-lane path (same positions, same slots)."""
            self.cache.end_forward()
            out = []
            for index in (0, 1):
                self.cache.begin_forward(halves[index][2], halves[index][1])
                single = self._half_steps(*halves[index], half_shape, prevalidated=True)
                try:
                    while True:
                        next(single)
                except StopIteration as finished:
                    out.append(finished.value)
            return out

        from .runtime import SWITCHES as _switches

        probe = _switches.get("wire_probe", "off") != "off"   # timing probe: no reruns
        if flags and bool(torch.stack(flags).any()) and not probe:
            self.overlap_reruns = getattr(self, "overlap_reruns", 0) + 1
            results = single_lane()
        elif (_os.environ.get("HY4_OVERLAP_VERIFY") or "") not in ("", "0"):
            # A gate boot: recompute the pair on the single-lane path and compare
            # the post-norm hidden — a cross-lane ordering error (lane 1 reading
            # lane 0's records early) shows as a large difference. Serves the
            # single-lane result.
            # The single-lane pair runs TWICE: single vs single is the forward's
            # own run-to-run noise floor (the prefill is not deterministic), the
            # yardstick the overlapped difference is judged against.
            overlapped_hidden = torch.cat(results, dim=0).float()
            PROBE.start()
            first = torch.cat(single_lane(), dim=0).float()
            run_a = PROBE.stop()
            PROBE.start()
            results = single_lane()
            run_b = PROBE.stop()
            second = torch.cat(results, dim=0).float()
            if run_a and len(run_a) == len(run_b) and getattr(self, "_probe_reported", 0) < 8:
                # The first intermediate where two identical single-lane runs differ.
                same = [bool(torch.equal(a[2], b[2])) for a, b in zip(run_a, run_b)]
                first_diff = next((i for i, ok in enumerate(same) if not ok), None)
                where = ("none" if first_diff is None else
                         f"{run_a[first_diff][0]} layer {run_a[first_diff][1]} (record "
                         f"{first_diff} of {len(same)}; previous: "
                         f"{run_a[first_diff - 1][0] if first_diff else '-'} layer "
                         f"{run_a[first_diff - 1][1] if first_diff else '-'})")
                firsts = [f"{run_a[i][0]}@{run_a[i][1]}" for i, ok in enumerate(same) if not ok][:6]
                print(f"[h4 r{getattr(self.collectives, 'rank', 0)}] DETERMINISM first difference "
                      f"between two single-lane runs: {where}; differing records "
                      f"{same.count(False)} of {len(same)}; first differing {firsts}", flush=True)
                self._probe_reported = getattr(self, "_probe_reported", 0) + 1

            def rel(a, b):
                return float((a - b).norm() / b.norm().clamp_min(1e-30))

            error, floor = rel(overlapped_hidden, second), rel(first, second)
            self.overlap_verified = getattr(self, "overlap_verified", 0) + 1
            self.overlap_verify_max = max(getattr(self, "overlap_verify_max", 0.0), error)
            self.overlap_verify_floor = max(getattr(self, "overlap_verify_floor", 0.0), floor)
            print(f"[h4 r{getattr(self.collectives, 'rank', 0)}] OVERLAP-VERIFY pair "
                  f"{self.overlap_verified}: overlapped vs single-lane rel-L2 {error:.3e}, "
                  f"single vs single {floor:.3e}", flush=True)
            if (_os.environ.get("HY4_OVERLAP_VERIFY") or "") == "strict" and \
                    not error <= max(1.5e-2, 3 * floor + 1e-3):
                raise RuntimeError(f"overlapped pair differs from the single-lane pair by "
                                   f"rel-L2 {error:.3e} (single-lane noise {floor:.3e})")
        self.overlap_pairs = getattr(self, "overlap_pairs", 0) + 1
        sink = getattr(self, "embed_sink", None)
        if sink is not None:
            parts = [sink.pop(id(halves[i][0]), None) for i in (0, 1)]
            if all(p is not None for p in parts):
                sink[id(token_ids)] = torch.cat(parts, dim=0)
        normed = torch.cat(results, dim=0)
        self.cache.end_forward()
        if logits_rows == "none":
            logits = None
        else:
            with TIMER.phase("head"):
                logits = normed.float() @ self.weights.head.T
                logits = logits.reshape(1, 2 * half, -1)
                if logits_rows != "all":
                    token = 2 * half - 1 if logits_rows == "last" else logits_rows
                    logits = logits[:, token:token + 1, :].contiguous()
            with TIMER.phase("comm.logits_gather"):
                logits = self.collectives.gather_logits(
                    logits, vocabulary_range=self.weights.sharding.vocabulary_range)
        TIMER.forward_end()
        if return_hidden:
            # consult18 P4: the post-norm hidden of both halves (the MTP feed).
            return logits, normed.reshape(1, 2 * half, -1)
        return logits

    def _publish_embeds(self, token_ids, embeds):
        """consult18 P5: a caller-set `embed_sink` dict receives every eager
        forward's all-reduced token embeddings, keyed by `id(token_ids)` (the
        prompt absorption reuses them — the draft embeds the NEXT tokens, i.e.
        the target's next rows). Never under CUDA graph capture."""
        sink = getattr(self, "embed_sink", None)
        if sink is not None and not (embeds.is_cuda and torch.cuda.is_current_stream_capturing()):
            sink[id(token_ids)] = embeds

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

    def _attention(self, layer, y, positions, token_rows, visible, *, call_shape,
                   visible_max=None):
        """One layer's attention (`_attention_gen` run to completion)."""
        steps = self._attention_gen(layer, y, positions, token_rows, visible,
                                    call_shape=call_shape, visible_max=visible_max)
        try:
            while True:
                next(steps)
        except StopIteration as finished:
            return finished.value

    def _attention_gen(self, layer, y, positions, token_rows, visible, *, call_shape,
                       visible_max=None):
        """One layer's attention: world 1 today's path, world > 1 the CP shape (D2/D3).

        A generator (consult18 D, fine-grained overlap): it yields ("kv", layer)
        right after this forward's cache writes and ("comm", name) after each
        collective, and returns the attention output. The plain path runs it
        to completion (`_attention`); an overlapped half yields between steps so
        the two lanes' collectives interleave one at a time.

        CP (amendment 1): full layers offer local top-k candidates over their
        OWN key table, the selection exchange merges the W ranks' candidates
        into one global top-2048 per query row and `SelectionCell.handback`
        maps it to this rank's local selection (per-row slot map); then the
        head all-gather of projected queries, the all-heads sinkless partial
        over this rank's record shard, the base-2 LSE reduce-scatter returning
        this rank's head shard, and ONE post-merge sink fold
        (`PreparedSinkFold`) before `W_UV` expansion and the gated `o_proj`.

        The W7 bake-ins are per GRAPH FAMILY (consult7 §5 / consult8 §6): the
        query gather moves ABORBED queries on decode graphs (owner-side
        absorb — no extra weight reads) and UNABSORBED queries on prefill
        graphs (absorb + expand locally, the `kv_b` two-wirings); decode
        graphs run the selection path on a second stream (recorded in
        `self.trace` from `wiring.StreamSchedule`) and take the short-context
        bypass (no indexer, no exchange at context <= top-k).
        """
        attn = layer.attention
        attn._active_call_shape = call_shape
        family = call_shape.family
        schedule = SCHEDULE[family]
        topk = self.config.text["index_topk"]
        world, rank = self.collectives.world_size, self.collectives.rank
        # The `kv_b` two-wirings (consult8 §5): decode keeps the owner-side
        # absorb (weight reads stay this rank's shard rows); prefill gathers
        # UNABSORBED queries and absorbs + expands locally.
        local_absorb_after_gather = world > 1 and call_shape.query_wiring == "unabsorbed-local-absorb"
        with TIMER.phase("attn.project"):
            bundle = attn.project_packed(y, positions, absorb=not local_absorb_after_gather,
                                         with_index_keys=attn.weights.indexer is not None)
        q_resid, q_local, latent = bundle.q_resid, bundle.q, bundle.latent
        PROBE.add("project", layer.index, q_local, latent,
                  bundle.index_keys if attn.weights.indexer is not None else None)
        WATCH.check("attn.project", layer.index, q_local=q_local, latent=latent,
                    index_keys=bundle.index_keys if attn.weights.indexer is not None else None)
        with TIMER.phase("attn.cache_write"):
            self.cache.write_latent(layer.index, token_rows, positions, latent)
            if attn.weights.indexer is not None:
                self.cache.write_index_keys(layer.index, token_rows, positions, bundle.index_keys)
        yield ("kv", layer.index)       # this layer's records + index keys are written
        owner = attn.selection.owner_id
        slot_base = self.cache.slots_for(token_rows)
        if world == 1:
            with TIMER.phase("attn.index_score"):
                selected, counts = attn.select(y, q_resid, positions, visible,
                                               self.cache.key_table(owner), slot_base)
            with TIMER.phase("attn.sparse_mla"):
                attended = attn.attend(q_local, self.cache.latents(layer.index), selected, counts,
                                       call_shape=call_shape)
            with TIMER.phase("attn.fold_expand_gate"):
                return attn.gate_and_project(attended.output, y)
        short = call_shape.short_context_bypass and call_shape.is_decode_graph
        if short:
            if visible_max is None:   # a direct call: read the device here
                visible_max = int(visible.max())
            short = visible_max <= topk
        from .runtime import SWITCHES as _SW

        if short and layer.index in self.cache.short and _SW.get("short_kv") == "on":
            # Replicated short-context KV: every rank holds every record below the
            # short capacity, so it attends ITS OWN heads (the owner-absorbed
            # q_local) over all visible keys — the sink folded in kernel — and
            # expands / gates / projects them: no head gather, no LSE merge (the
            # o_proj all-reduce after this layer sums the head shards as before).
            S = self.cache.short_capacity
            n = int(visible.shape[0])
            with TIMER.phase("attn.handback"):
                selected, counts = attn.selection.buffers(rows=n)
                width = int(selected.shape[1])
                cols = torch.arange(width, device=visible.device, dtype=torch.int64).reshape(1, width)
                base = token_rows.reshape(-1, 1).long() * S
                selected[:n].copy_(torch.where(cols < visible.reshape(-1, 1).long(), base + cols,
                                               torch.full_like(cols, -1)).to(selected.dtype))
                counts[:n].copy_(visible.to(counts.dtype))
            with TIMER.phase("attn.sparse_mla"):
                attended = attn.attend(q_local, self.cache.short_latents(layer.index), selected[:n],
                                       counts[:n], fold_sink=True, expand=False, call_shape=call_shape)
            with TIMER.phase("attn.fold_expand_gate"):
                fused = attn.expand_gate_project(attended.output, y)
                if fused is not None:
                    return fused
                out = attn.expand_output(attended.output).to(y.dtype)
                return attn.gate_and_project(out, y)
        if short:
            # The short-context bypass: at context <= top-k the global top-k is
            # every visible key — no indexer, no selection exchange.
            self.trace.append(("selection_bypass", family, visible_max, topk))
            with TIMER.phase("attn.handback"):
                attn.selection.fill_visible(visible, slot_base, world=world, rank=rank)
        else:
            if attn.selection.kind == "full":
                self.trace.append(("selection_exchange",
                                   schedule.placement("selection_exchange", family)))
                with TIMER.phase("attn.index_score"):
                    scores, candidates = attn.local_candidates(
                        y, q_resid, positions, visible, self.cache.key_table(owner), slot_base,
                        world=world, rank=rank)
                PROBE.add("candidates", layer.index, scores, candidates)
                from .lanes import sync_free
                from .runtime import SWITCHES

                if (scores.is_cuda and sync_free() and SWITCHES.get("sel_fused") == "on"
                        and hasattr(self.collectives, "merge_topk_handback")
                        and self.collectives.fused_handback_ok()):
                    # consult18 P2: exchange + selection + the position->slot map
                    # + hand-back in one launch (fused_ops.scores_only_select);
                    # writes the full layer's selection buffers directly.
                    selected_buf, counts_buf = attn.selection.buffers(rows=y.shape[0])
                    with TIMER.phase("comm.selection"):
                        self.collectives.merge_topk_handback(
                            scores, candidates, topk=self.config.text["index_topk"],
                            selected=selected_buf, counts=counts_buf,
                            base=self.cache.slots_for(token_rows),
                            local_capacity=self.cache.local_capacity)
                    yield ("comm", "scores")
                    merged = None
                else:
                    with TIMER.phase("comm.selection"):
                        _, merged = self.collectives.merge_topk(
                            scores, candidates, topk=self.config.text["index_topk"])
                    PROBE.add("merged", layer.index, merged)
                    yield ("comm", "scores")
                with TIMER.phase("attn.handback"):
                    if merged is not None:
                        # `local_slot_map`'s `[rows, local_capacity]` table is pure
                        # arithmetic (slot = row_base + p // world) and the hand-back
                        # only ever reads its `[rows, topk]` picks — materializing the
                        # whole table costs 1.5 GiB of int64 at 1024 prefill rows (the
                        # 2k-cell OOM). Shift each row's global positions by
                        # `world * row_base` (ownership `p % world` is unchanged and
                        # `p // world` becomes the physical slot — the map's exact
                        # gather value) and hand back with no table, preserving the
                        # table's `-1` padding and out-of-range guard by hand.
                        pos = merged
                        base = self.cache.slots_for(token_rows).reshape(-1, 1)
                        local = pos.clamp(min=0) // world
                        bounded = (pos < 0) | (local < self.cache.local_capacity)
                        shifted = torch.where(pos >= 0, pos.to(torch.int64) + world * base,
                                              pos.to(torch.int64))  # `-1` padding stays -1
                        mapped = torch.where(bounded, shifted,
                                             torch.full_like(shifted, -1)).to(pos.dtype)
                        attn.selection.handback(mapped, world_size=world, rank=rank,
                                                slot_map=None)
        with TIMER.phase("attn.handback"):
            selected, counts = attn.selection.buffers(rows=y.shape[0])
        PROBE.add("selected", layer.index, selected, counts)
        self.trace.append(("query_exchange", schedule.placement("query_exchange", family),
                           call_shape.query_wiring, int(q_local.shape[-1])))
        with TIMER.phase("comm.head_gather"):
            q_all = self.collectives.all_gather_heads(q_local)   # all heads per call (D2)
        PROBE.add("gather", layer.index, q_all)
        yield ("comm", "gather")
        if local_absorb_after_gather:
            # Prefill wiring: the gathered queries are UNABSORBED; W_UK applies
            # locally over ALL gathered heads (replicated `kv_b_proj`).
            with TIMER.phase("attn.absorb"):
                q_all = attn.absorb_queries(q_all)
        WATCH.check("comm.head_gather", layer.index, q_all=q_all)
        with TIMER.phase("attn.sparse_mla"):
            partial = attn.attend(q_all, self.cache.latents(layer.index), selected, counts,
                                  fold_sink=False, expand=False,  # sink unbound
                                  call_shape=call_shape)
        if WATCH.enabled:
            # A rank that owns none of a row's selected keys reports LSE -inf
            # and an undefined value there: legitimate, and weightless in the
            # merge. Everything else must be finite.
            owned = torch.isfinite(partial.lse) | torch.isnan(partial.lse)
            WATCH.check("attn.sparse_mla", layer.index,
                        partial=torch.where(owned.unsqueeze(-1), partial.output.float(),
                                            torch.zeros((), device=partial.output.device)),
                        partial_lse=torch.where(owned, partial.lse.float(),
                                                torch.zeros((), device=partial.lse.device)))
        # D6: the partials' provenance goes into the merge
        # (`partials_sink_folded=provenance.sink_folded`, `lse_base="base2"`).
        if self.partial_wire not in ("latent", "expanded"):
            raise ValueError(f"unknown partial wire {self.partial_wire!r}")
        expanded = self.partial_wire == "expanded" and call_shape.is_prefill_graph
        values = partial.output
        if expanded:
            # consult17: expand every head before the exchange — the wire
            # carries v_head_dim values per head instead of the latent width.
            with TIMER.phase("attn.expand_wire"):
                values = attn.expand_partials(values)
        with TIMER.phase("comm.lse_reduce"):
            out, lse = self.collectives.lse_reduce(
                values, partial.lse, partials_sink_folded=partial.provenance.sink_folded)
        PROBE.add("partial", layer.index, partial.output, partial.lse)
        PROBE.add("lse", layer.index, out, lse)
        yield ("comm", "lse")
        WATCH.check("comm.lse_reduce", layer.index, merged=out, merged_lse=lse)
        with TIMER.phase("attn.fold_expand_gate"):
            out, lse = attn.fold_sink(out, lse)   # the ONE sink fold, after the merge
            if not expanded:
                fused = attn.expand_gate_project(out, y)   # round 15 (switch expand_gate)
                if fused is not None:
                    return fused
            # `W_UV` expansion over this rank's head shard (a per (row, head) scale
            # commutes with the fold; the latent-space partials cannot expand on a
            # foreign head — `kv_b_proj` is head-sharded for the decode wiring).
            # Already expanded on the wire when `expanded`.
            out = (out if expanded else attn.expand_output(out)).to(y.dtype)
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



class _SyncSiteReporter:
    """HY4_OVERLAP_SYNC_GUARD=warn: print each distinct synchronizing call site
    inside the overlapped loop once, with its Python stack (diagnosis boots)."""

    def __init__(self):
        self.seen = set()
        self._previous = None
        self.rank = 0

    def install(self, rank):
        import warnings

        self.rank = rank
        self._previous = warnings.showwarning
        warnings.simplefilter("always", UserWarning)
        reporter = self

        def show(message, category, filename, lineno, file=None, line=None):
            text = str(message)
            if "synchronizing" not in text:
                return reporter._previous(message, category, filename, lineno, file, line)
            import traceback

            stack = traceback.extract_stack()[:-1]
            frames = [f for f in stack if "hy4" in f.filename or "b12x" in f.filename][-6:]
            key = tuple((f.filename, f.lineno) for f in frames)
            if key in reporter.seen:
                return None
            reporter.seen.add(key)
            where = " <- ".join(f"{f.filename.rsplit('/', 1)[-1]}:{f.lineno} {f.name}"
                                for f in reversed(frames))
            print(f"[h4 r{reporter.rank}] SYNC-SITE {text[:60]} at {where}", flush=True)
            return None

        warnings.showwarning = show

    def uninstall(self):
        import warnings

        if self._previous is not None:
            warnings.showwarning = self._previous
            self._previous = None


_sync_site_reporter = _SyncSiteReporter()



class _DeterminismProbe:
    """consult18 D: where do two identical single-lane runs of a pair first
    differ? Device fingerprints (two position-weighted int64 sums over the raw
    words) of every intermediate, recorded only while `start()`ed — the
    verify path's single-lane runs; compared on the host after them."""

    def __init__(self):
        self.records = None

    def start(self):
        self.records = []

    def stop(self):
        records, self.records = self.records, None
        return records

    def add(self, name, layer, *tensors):
        if self.records is None:
            return
        digests = [self._digest(t) for t in tensors if t is not None]
        if digests:
            self.records.append((name, layer, torch.stack(digests)))

    @staticmethod
    def _digest(tensor):
        flat = tensor.detach().contiguous().view(-1)
        nbytes = flat.numel() * flat.element_size()
        words = (flat.view(torch.int32) if nbytes % 4 == 0 else flat.view(torch.uint8)).to(torch.int64)
        index = torch.arange(words.numel(), device=words.device, dtype=torch.int64)
        return torch.stack(((words * (index % 65521 + 1)).sum(),
                            (words * ((index * 40503) % 65519 + 1)).sum()))


PROBE = _DeterminismProbe()
