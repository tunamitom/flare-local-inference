# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0
"""Hy4 DFly2 (AngelSpec) block-parallel drafter for official `tencent/Hy4`.

Architecture (verified against `lmsysorg/sglang:hy4` `hunyuan_v4_dfly.py`,
`hy_v4_dfly.py`, and `dfly_worker_v2.py` plus `/storage/Hy4` shard 245):

1. Input block per request (`W = K + 1` query tokens at positions `P .. P + K`):
   - Slot `0`: `bonus_token` (shared target `embed_tokens`, vocab-sharded + TP all-reduce).
   - Slots `1 .. K`: `dfly_mask_token_id` (`120009`).
   - Slot-0 fusion (`model.dfly_layers.mtp_fusion`):
       `fused = cat([rms_norm(e0, enorm), rms_norm(target_hidden, hnorm)], dim=-1) @ eh_proj.T`
       `h[0] = e0 + gate * fused`
     (`target_hidden` is the target model's post-`model.norm` hidden at the last
     committed position `P - 1`; `gate` is a literal elementwise multiplier.)
2. `dfly_num_hidden_layers` (`5`) dense single-stream layers (no iHC, no own indexer,
   no persistent KV pool):
   - Layer `i` (`0 .. 4`) writes its block KV into target layer
     `dfly_target_kv_layers[i]` (`[61, 65, 69, 73, 77]`) at the uncommitted verify
     slots `P .. P + K`, then attends over the target top-k captured from
     `dfly_target_index_layer[i]` (`[61, 65, 69, 73, 77]`) at `P - 1` union the
     causal prefix of the block `P .. P + j`.
   - Followed by a dense SwiGLU FFN (`DenseBlock`, unclamped, TP intermediate-sharded).
   - Followed by `model.dfly_layers.final_norm` -> `z [B, K + 1, H]`.
3. Autoregressive hidden correction (`model.dfly_layers.hidden_correction`) for
   `step = 0 .. K - 1` (proposing `d_1 .. d_K`):
   - `x = cat([rms_norm(z[:, step + 1], hidden_norm), rms_norm(embed(prev), embed_norm)], dim=-1)`
     (note: `hidden_norm` FIRST, `embed_norm` SECOND — opposite of `mtp_fusion`).
   - `delta = (silu(x @ gate_proj.T) * (x @ up_proj.T)) @ down_proj.T` (+ TP all-reduce).
   - `corrected = z[:, step + 1] + delta` -> shared target `lm_head` -> greedy or
     speculative sample `prev = d_{step + 1}`.

Weights stay pure BF16 (`~1.14 GiB/rank` at TP8 with replicated `kv_b_proj`;
per-head `learnable_sink_param` stays FP32). Under CP8 (`world > 1`), `DflyState`
stores canonical rank-local within-row offsets (`loc = p // world`) so target
selections remain invariant across dynamic row relocations (`Hy4Cache.relocate`)
and every history + block slot is checked against live per-row `cache.row_cap_dev`
(preserving the per-row live cache-extent invariant).
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .attention import Hy4LayerAttention, rms_norm
from .moe import DenseBlock, DenseMlpWeights


@dataclass(frozen=True)
class DflyMtpFusion:
    """Slot-0 fusion mixing the bonus embedding with the target's final hidden."""

    enorm: torch.Tensor      # [H] BF16 (replicated)
    hnorm: torch.Tensor      # [H] BF16 (replicated)
    eh_proj: torch.Tensor    # [H, 2H] BF16 (replicated)
    gate: torch.Tensor       # [H] BF16 (replicated)
    eps: float

    def forward(self, embeds, target_hidden):
        left = rms_norm(embeds, self.enorm, self.eps)
        right = rms_norm(target_hidden, self.hnorm, self.eps)
        fused = torch.cat((left, right), dim=-1) @ self.eh_proj.T
        return embeds + self.gate * fused


@dataclass(frozen=True)
class DflyHiddenCorrection:
    """Per-step SwiGLU correction over `(z[:, step + 1], embed(prev_token))`."""

    hidden_norm: torch.Tensor   # [H] BF16 (replicated)
    embed_norm: torch.Tensor    # [H] BF16 (replicated)
    gate_proj: torch.Tensor     # [inter_local, 2H] BF16 (column-sharded)
    up_proj: torch.Tensor       # [inter_local, 2H] BF16 (column-sharded)
    down_proj: torch.Tensor     # [H, inter_local] BF16 (row-sharded)
    eps: float

    def forward(self, hidden, previous_embeds, *, collectives=None):
        left = rms_norm(hidden, self.hidden_norm, self.eps)
        right = rms_norm(previous_embeds, self.embed_norm, self.eps)
        merged = torch.cat((left, right), dim=-1)
        gate = merged @ self.gate_proj.T
        up = merged @ self.up_proj.T
        delta = (F.silu(gate) * up) @ self.down_proj.T
        if collectives is not None:
            delta = collectives.all_reduce(delta)
        return hidden + delta


@dataclass(frozen=True)
class DflyLayer:
    """One dense DFly2 layer bound to its target KV layer and producer plane."""

    index: int
    target_kv_layer: int
    target_index_layer: int
    producer_plane: int
    attention: Hy4LayerAttention
    input_norm: torch.Tensor
    post_attention_norm: torch.Tensor
    mlp: DenseBlock


@dataclass
class DflyBlock:
    """Complete 5-layer pure-BF16 DFly2 drafter on one rank."""

    layers: tuple
    mtp_fusion: DflyMtpFusion
    final_norm: torch.Tensor
    hidden_correction: DflyHiddenCorrection
    producer_layers: tuple
    config: object
    device: torch.device

    @classmethod
    def build(cls, builder):
        """Construct the DFly2 drafter from `model.dfly_layers.*` (pure BF16)."""
        from .state import _attention_state

        config, sharding, loader = builder.config, builder.sharding, builder.loader
        if not config.has_dfly:
            raise ValueError("checkpoint config declares no DFly2 layers")
        hidden = config.hidden
        eps = float(config.text["rms_norm_eps"])
        prefix = "model.dfly_layers"

        # Wrapper modules.
        enorm = builder.norm(f"{prefix}.mtp_fusion.enorm.weight", hidden)
        hnorm = builder.norm(f"{prefix}.mtp_fusion.hnorm.weight", hidden)
        eh_proj = builder.linear(f"{prefix}.mtp_fusion.eh_proj", (hidden, 2 * hidden))
        gate = builder.norm(f"{prefix}.mtp_fusion.gate", hidden)
        if eh_proj.dtype != torch.bfloat16:
            raise ValueError(f"{prefix}.mtp_fusion.eh_proj.weight: DFly2 stays pure BF16")
        mtp_fusion = DflyMtpFusion(enorm=enorm, hnorm=hnorm, eh_proj=eh_proj, gate=gate, eps=eps)

        final_norm = builder.norm(f"{prefix}.final_norm.weight", hidden)

        corr_inter = config.dfly_hidden_correction_intermediate_size
        corr_range = sharding.dfly_correction_range
        corr_prefix = f"{prefix}.hidden_correction"
        hc_hidden_norm = builder.norm(f"{corr_prefix}.hidden_norm.weight", hidden)
        hc_embed_norm = builder.norm(f"{corr_prefix}.embed_norm.weight", hidden)
        hc_gate = builder.linear(f"{corr_prefix}.gate_proj", (corr_inter, 2 * hidden),
                                 row_range=corr_range)
        hc_up = builder.linear(f"{corr_prefix}.up_proj", (corr_inter, 2 * hidden),
                               row_range=corr_range)
        hc_down = builder.linear(f"{corr_prefix}.down_proj", (hidden, corr_inter),
                                 column_range=corr_range)
        for name, tensor in (("gate_proj", hc_gate), ("up_proj", hc_up), ("down_proj", hc_down)):
            if tensor.dtype != torch.bfloat16:
                raise ValueError(f"{corr_prefix}.{name}.weight: DFly2 stays pure BF16")
        hidden_correction = DflyHiddenCorrection(
            hidden_norm=hc_hidden_norm, embed_norm=hc_embed_norm,
            gate_proj=hc_gate, up_proj=hc_up, down_proj=hc_down, eps=eps)

        # Dense layers bound to target KV and target index producers.
        target_kv = config.dfly_target_kv_layers
        target_idx = config.dfly_target_index_layers
        producers = tuple(sorted(set(target_idx)))
        plane_of = {layer_id: plane for plane, layer_id in enumerate(producers)}
        dfly_inter = config.dfly_intermediate_size
        dense_range = sharding.dfly_dense_range
        draft_max_width = config.dfly_index_topk

        layers = []
        for idx in range(config.dfly_layers):
            lp = f"{prefix}.{idx}"
            attention = _attention_state(
                builder, layer=None, prefix=lp, selection=None,
                has_indexer=False, max_width=draft_max_width)
            input_norm = builder.norm(f"{lp}.input_layernorm.weight", hidden)
            post_norm = builder.norm(f"{lp}.post_attention_layernorm.weight", hidden)
            mlp_prefix = f"{lp}.mlp"
            mlp_weights = DenseMlpWeights(
                gate_proj=builder.linear(f"{mlp_prefix}.gate_proj", (dfly_inter, hidden),
                                         row_range=dense_range),
                up_proj=builder.linear(f"{mlp_prefix}.up_proj", (dfly_inter, hidden),
                                       row_range=dense_range),
                down_proj=builder.linear(f"{mlp_prefix}.down_proj", (hidden, dfly_inter),
                                         column_range=dense_range))
            for name in ("gate_proj", "up_proj", "down_proj"):
                if getattr(mlp_weights, name).dtype != torch.bfloat16:
                    raise ValueError(f"{mlp_prefix}.{name}.weight: DFly2 dense FFN stays BF16")
            layers.append(DflyLayer(
                index=idx,
                target_kv_layer=target_kv[idx],
                target_index_layer=target_idx[idx],
                producer_plane=plane_of[target_idx[idx]],
                attention=attention,
                input_norm=input_norm,
                post_attention_norm=post_norm,
                mlp=DenseBlock(mlp_weights, loader.device)))
        return cls(
            layers=tuple(layers),
            mtp_fusion=mtp_fusion,
            final_norm=final_norm,
            hidden_correction=hidden_correction,
            producer_layers=producers,
            config=config,
            device=loader.device)


def _compact_global_to_local(selected, counts, base, caps, *, world, rank):
    """Extract rank `rank`'s owned within-row local offsets from global slots.

    `selected` `[N, T]` int32 holds global positions shifted by `base` (`row * S`);
    `counts` `[N]` int32 holds valid prefix counts; `caps` `[N]` int64 holds live
    per-row local slot capacities (`cache.caps_for(token_rows)`). Returns packed
    `(local_offsets [N, T] int32, local_counts [N] int32)` with `-1` padding.
    Capture-safe: fixed shapes, zero host reads.
    """
    rows, width = int(selected.shape[0]), int(selected.shape[1])
    dev = selected.device
    cols = torch.arange(width, device=dev, dtype=torch.int32).unsqueeze(0)
    in_count = (cols < counts.to(torch.int32).unsqueeze(1)) & (selected >= 0)
    pos = torch.where(in_count, selected.to(torch.int64) - base.to(torch.int64).unsqueeze(1),
                      torch.full((rows, width), -1, dtype=torch.int64, device=dev))
    loc = torch.div(pos.clamp_min(0), world, rounding_mode="floor")
    valid = (pos >= 0) & (torch.remainder(pos, world) == rank) & (loc < caps.to(torch.int64).unsqueeze(1))
    dst = torch.where(valid, valid.to(torch.int64).cumsum(dim=1) - 1,
                      torch.full((rows, width), width, dtype=torch.int64, device=dev))
    buf = torch.full((rows, width + 1), -1, dtype=torch.int32, device=dev)
    buf.scatter_(1, dst, torch.where(valid, loc, torch.full_like(loc, -1)).to(torch.int32))
    out_cnt = valid.sum(dim=1, dtype=torch.int32)
    return buf[:, :width].contiguous(), out_cnt.contiguous()


def _physical_to_local(selected, counts, base, caps):
    """Convert rank-local physical slots `[N, T]` to bounded within-row offsets.

    Clamps against live `caps` (`cache.caps_for(token_rows)`) so idle rows at
    extent 0 produce count 0 and all `-1`, and relocated rows remain valid after
    `Hy4Cache.relocate`. Capture-safe: fixed shapes, zero host reads.
    """
    rows, width = int(selected.shape[0]), int(selected.shape[1])
    dev = selected.device
    cols = torch.arange(width, device=dev, dtype=torch.int32).unsqueeze(0)
    in_count = (cols < counts.to(torch.int32).unsqueeze(1)) & (selected >= 0)
    loc = torch.where(in_count, selected.to(torch.int64) - base.to(torch.int64).unsqueeze(1),
                      torch.full((rows, width), -1, dtype=torch.int64, device=dev))
    valid = (loc >= 0) & (loc < caps.to(torch.int64).unsqueeze(1))
    dst = torch.where(valid, valid.to(torch.int64).cumsum(dim=1) - 1,
                      torch.full((rows, width), width, dtype=torch.int64, device=dev))
    buf = torch.full((rows, width + 1), -1, dtype=torch.int32, device=dev)
    buf.scatter_(1, dst, torch.where(valid, loc, torch.full_like(loc, -1)).to(torch.int32))
    out_cnt = valid.sum(dim=1, dtype=torch.int32)
    return buf[:, :width].contiguous(), out_cnt.contiguous()


class DflyState:
    """Per-rank tapped target top-k state and capture-safe sparse index builder.

    Stores canonical rank-local within-row offsets (`loc = p // world`, `-1`
    padding) per `(producer_plane, cache_row)` so target selections stay valid
    across `Hy4Cache.relocate` and are checked against live `cache.row_cap_dev`
    on every draft step (preserving the per-row live cache-extent invariant). Extra trailing row `rows` is the
    dummy scatter target for padded batch slots.
    """

    def __init__(self, *, num_producers, rows, target_topk, draft_topk, device,
                 max_verify_rows=64):
        self.num_producers = int(num_producers)
        self.rows = int(rows)
        self.target_topk = int(target_topk)
        self.draft_topk = int(draft_topk)
        self.device = device
        self.max_verify_rows = int(max_verify_rows)
        # Persistent per-request-row history (extra trailing row = pad sink).
        self.hist_local = torch.full(
            (self.num_producers, self.rows + 1, self.target_topk),
            -1, dtype=torch.int32, device=device)
        self.hist_count = torch.zeros(
            (self.num_producers, self.rows + 1), dtype=torch.int32, device=device)
        # Per-verify-forward capture planes (gathered into hist_* once n_acc is known).
        self.cap_local = torch.full(
            (self.num_producers, self.max_verify_rows, self.target_topk),
            -1, dtype=torch.int32, device=device)
        self.cap_count = torch.zeros(
            (self.num_producers, self.max_verify_rows), dtype=torch.int32, device=device)

    def reset_rows(self, rows):
        for row in rows:
            r = int(row)
            if 0 <= r < self.rows:
                self.hist_local[:, r].fill_(-1)
                self.hist_count[:, r].zero_()

    def record_capture(self, plane, selected, counts, *, base, caps, world, rank, is_global):
        """Copy all `N` rows of an active verify forward into capture plane `plane`."""
        n = int(selected.shape[0])
        if n > self.max_verify_rows:
            raise RuntimeError(
                f"DFly2 capture holds {self.max_verify_rows} rows, got {n}")
        if is_global and world > 1:
            loc, cnt = _compact_global_to_local(selected, counts, base, caps,
                                                world=world, rank=rank)
        else:
            loc, cnt = _physical_to_local(selected, counts, base, caps)
        self.cap_local[plane, :n].copy_(loc)
        self.cap_count[plane, :n].copy_(cnt)

    def record_commit(self, plane, dst_rows, selected, counts, *, base, caps, world, rank, is_global):
        """Write committed rows (prefill last token or decode step) directly to `hist_*`."""
        if is_global and world > 1:
            loc, cnt = _compact_global_to_local(selected, counts, base, caps,
                                                world=world, rank=rank)
        else:
            loc, cnt = _physical_to_local(selected, counts, base, caps)
        dst = dst_rows.to(torch.long)
        self.hist_local[plane].index_copy_(0, dst, loc)
        self.hist_count[plane].index_copy_(0, dst, cnt)

    def commit_capture(self, dst_rows, src_indices):
        """Commit captured verify rows `src_indices` into `hist_*` at `dst_rows` (capture-safe)."""
        src = src_indices.to(torch.long)
        dst = dst_rows.to(torch.long)
        picked_loc = self.cap_local.index_select(1, src)
        picked_cnt = self.cap_count.index_select(1, src)
        self.hist_local.index_copy_(1, dst, picked_loc)
        self.hist_count.index_copy_(1, dst, picked_cnt)

    def commit_capture_row(self, row, src_index):
        """Single-row host-indexed commit from capture plane into `hist_*[:, row]`."""
        r, idx = int(row), int(src_index)
        self.hist_local[:, r].copy_(self.cap_local[:, idx])
        self.hist_count[:, r].copy_(self.cap_count[:, idx])

    def build_sparse_indices(self, plane, row_ids, anchors, *, cache, query_width,
                             world, rank, bypass_short=False, bypass_cp=False):
        """Build packed `(selected [B*W, draft_topk] int32, counts [B*W] int32)` on device.

        Three modes (all capture-safe, zero host reads, live `row_cap_dev` enforced):
        - `bypass_short=True`: replicated short-context table (`cache.short`), all
          positions `0 .. P_b + j` for active rows (`row_cap > 0`).
        - `bypass_cp=True`: rank-local owned positions `0 .. owned(P_b + j + 1) - 1`,
          clamped by live `row_cap_dev[row]`.
        - sparse (`bypass_short=False, bypass_cp=False`): tapped `hist_local[plane, row]`
          validated against live `row_cap_dev[row]`, plus owned causal block slots
          `P_b + m` (`0 <= m <= j`, `pos % world == rank`, `pos // world < row_cap`),
          packed tightly with zero `-1` holes before `counts`.
        """
        dev = anchors.device
        rows_t = row_ids.to(torch.long)
        anchors_t = anchors.to(torch.int64)
        bs = int(rows_t.shape[0])
        W = int(query_width)
        draft_topk = self.draft_topk
        steps = torch.arange(W, device=dev, dtype=torch.int64)
        row_cap = cache.row_cap_dev.to(dev)[rows_t]                # [B] live per-row cap!
        row_base = cache.row_base_dev.to(dev)[rows_t]              # [B] live per-row base!

        if bypass_short:
            S = int(cache.short_capacity)
            cols = torch.arange(draft_topk, device=dev, dtype=torch.int64).reshape(1, 1, draft_topk)
            vis = (anchors_t.unsqueeze(1) + steps.unsqueeze(0) + 1).clamp(min=0, max=min(S, draft_topk))
            vis = torch.where(row_cap.unsqueeze(1) > 0, vis, torch.zeros_like(vis))
            base_s = rows_t.reshape(bs, 1, 1) * S
            sel = torch.where(cols < vis.unsqueeze(2), base_s + cols,
                              torch.full_like(cols.expand(bs, W, draft_topk), -1))
            return sel.reshape(bs * W, draft_topk).to(torch.int32).contiguous(), \
                vis.reshape(bs * W).to(torch.int32).contiguous()

        if bypass_cp:
            cols = torch.arange(draft_topk, device=dev, dtype=torch.int64).reshape(1, 1, draft_topk)
            vis = anchors_t.unsqueeze(1) + steps.unsqueeze(0) + 1            # [B, W]
            owned = torch.where(
                vis > rank,
                torch.div(vis - rank + world - 1, world, rounding_mode="floor"),
                torch.zeros_like(vis),
            )
            owned = torch.minimum(owned, row_cap.unsqueeze(1)).clamp_max(draft_topk)
            sel = torch.where(cols < owned.unsqueeze(2),
                              row_base.reshape(bs, 1, 1) + cols,
                              torch.full((bs, W, draft_topk), -1, dtype=torch.int64, device=dev))
            return sel.reshape(bs * W, draft_topk).to(torch.int32).contiguous(), \
                owned.reshape(bs * W).to(torch.int32).contiguous()

        # Sparse mode: tapped history offsets + owned causal block slots, tightly packed.
        T = self.target_topk
        hist_loc = self.hist_local[plane].index_select(0, rows_t)            # [B, T] int32
        hist_cnt = self.hist_count[plane].index_select(0, rows_t)            # [B] int32
        t_cols = torch.arange(T, device=dev, dtype=torch.int32).unsqueeze(0)
        h_valid = ((t_cols < hist_cnt.unsqueeze(1))
                   & (hist_loc >= 0)
                   & (hist_loc.to(torch.int64) < row_cap.unsqueeze(1)))
        h_pos = h_valid.to(torch.int64).cumsum(dim=1) - 1                    # [B, T]
        h_cnt = h_valid.sum(dim=1, dtype=torch.int64)                        # [B]
        h_slots = torch.where(h_valid, row_base.unsqueeze(1) + hist_loc.to(torch.int64),
                              torch.full((bs, T), -1, dtype=torch.int64, device=dev)).to(torch.int32)

        # Compact valid history slots into [B, draft_topk + 1] (last col = discard).
        buf_h = torch.full((bs, draft_topk + 1), -1, dtype=torch.int32, device=dev)
        dst_h = torch.where(h_valid, h_pos, torch.full_like(h_pos, draft_topk))
        buf_h.scatter_(1, dst_h, h_slots)
        buf = buf_h.unsqueeze(1).expand(bs, W, draft_topk + 1).clone()

        # Append rank-owned causal block slots P_b + m (0 <= m <= j), bounded by row_cap.
        block_pos = anchors_t.unsqueeze(1) + steps.unsqueeze(0)              # [B, W]
        block_loc = torch.div(block_pos.clamp_min(0), world, rounding_mode="floor")
        block_own = ((block_pos >= 0)
                     & (torch.remainder(block_pos, world) == rank)
                     & (block_loc < row_cap.unsqueeze(1)))                   # [B, W]
        block_slots = torch.where(block_own, row_base.unsqueeze(1) + block_loc,
                                  torch.full_like(block_loc, -1)).to(torch.int32)
        causal = steps.unsqueeze(0) <= steps.unsqueeze(1)                    # [W, W]
        b_valid = causal.unsqueeze(0) & block_own.unsqueeze(1)               # [B, W, W]
        b_off = b_valid.to(torch.int64).cumsum(dim=2) - 1                    # [B, W, W]
        b_cnt = b_valid.sum(dim=2, dtype=torch.int64)                        # [B, W]
        dst_col = h_cnt.reshape(bs, 1, 1) + b_off
        in_width = b_valid & (dst_col < draft_topk)
        dst_b = torch.where(in_width, dst_col, torch.full_like(dst_col, draft_topk))
        buf.scatter_(2, dst_b, block_slots.unsqueeze(1).expand(bs, W, W))

        selected = buf[:, :, :draft_topk].reshape(bs * W, draft_topk).contiguous()
        counts = (h_cnt.unsqueeze(1) + b_cnt).clamp_max(draft_topk).reshape(bs * W).to(torch.int32).contiguous()
        return selected, counts
