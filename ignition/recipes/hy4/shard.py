# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0
"""Tensor-parallel shard assignment for one Hy4 rank.

One source of truth for what every tensor family looks like on a rank, so the
loader, the decoder body and the byte accounting (`config.packed_byte_arithmetic`)
cannot drift apart. The policy mirrors the reference vLLM port and the recipe's
`replicated_bytes` derivation (`recipe.py`):

- replicated on every rank: the `q_a_proj`/`kv_a_proj_with_mqa` down-projections,
  the whole indexer, the router `gate` and correction bias, every norm, the
  learnable sinks, the iHC streams (`fn`/`base`/`scale`, `hc_head`);
- head-sharded: `q_b_proj`, `kv_b_proj`, `o_proj`, `linear_gate` (64 attention
  heads split across the world);
- intermediate-sharded: routed experts, the shared expert and the dense layer-0
  FFN;
- vocabulary-sharded: the embedding and the untied FP32 head.

Head geometry legality: sparse-MLA prefill requires the head count of one
attention call to be divisible by 8 (`_shared/mla/prefill.py`). 64 heads give 8
per rank at TP8 (legal) and 4 per rank at TP16, so **plain head-sharded TP16 is
not a legal prefill world** (hy4-plan section 1.3); `prefill_head_gather` says
whether this assignment needs a head gather before any sparse prefill call.
"""

from dataclasses import dataclass

from ignition.components.parallel import TensorParallelShard

from .config import ModelConfig

# Sparse-MLA prefill calls need this many query heads per call.
PREFILL_HEAD_ALIGNMENT = 8
# NVFP4 expert tiles and the packed-expert loader align intermediate ranges here.
EXPERT_ALIGNMENT = 16


@dataclass(frozen=True)
class Hy4Sharding:
    """The per-rank partition of every Hy4 tensor family."""

    config: ModelConfig
    shard: TensorParallelShard

    def __post_init__(self):
        heads, rope = self.config.heads, self.config.text["qk_rope_head_dim"]
        nope, vd = self.config.text["qk_nope_head_dim"], self.config.text["v_head_dim"]
        world = self.shard.world_size
        if heads % world:
            raise ValueError(f"Hy4 attention heads ({heads}) must divide the world size ({world})")
        # Head-sharded projections carry complete heads per rank.
        self.shard.partition(heads * (nope + rope))   # q_b_proj rows, aligned as whole heads
        self.shard.partition(heads * (nope + vd), alignment=1)   # kv_b_proj rows
        self.shard.partition(heads * vd)             # o_proj / linear_gate columns
        # Routed experts, the shared expert and the dense FFN shard intermediate width.
        self.shard.partition(self.config.text["moe_intermediate_size"], alignment=EXPERT_ALIGNMENT)
        self.shard.partition(self.config.text["intermediate_size"], alignment=EXPERT_ALIGNMENT)
        if self.config.has_dfly:
            self.shard.partition(self.config.dfly_intermediate_size, alignment=EXPERT_ALIGNMENT)
            self.shard.partition(self.config.dfly_hidden_correction_intermediate_size,
                                 alignment=EXPERT_ALIGNMENT)
        self.shard.partition(self.config.vocabulary)

    # --- head geometry -----------------------------------------------------

    @property
    def heads_per_rank(self):
        return self.config.heads // self.shard.world_size

    @property
    def head_range(self):
        """Global attention-head range owned by this rank."""
        return self.shard.partition(self.config.heads)

    @property
    def prefill_head_gather(self):
        """True when this rank's head shard cannot run a sparse-MLA prefill alone."""
        return self.heads_per_rank % PREFILL_HEAD_ALIGNMENT != 0

    def require_prefill_legal(self):
        if self.prefill_head_gather:
            raise ValueError(
                f"sparse-MLA prefill needs heads divisible by {PREFILL_HEAD_ALIGNMENT}; "
                f"plain head-sharding gives this rank {self.heads_per_rank} of "
                f"{self.config.heads} heads at TP{self.shard.world_size} — gather all "
                "heads per call or serve TP8 (hy4-plan section 1.3)")

    # --- sharded projections ----------------------------------------------

    @property
    def q_b_rows(self):
        """(first, last) rows of `q_b_proj` (rows are heads x qk_head_dim)."""
        c = self.config.text
        return self.shard.partition(self.config.heads * (c["qk_nope_head_dim"] + c["qk_rope_head_dim"]))

    @property
    def kv_b_rows(self):
        """(first, last) rows of `kv_b_proj` (rows are heads x (qk_nope + v))."""
        c = self.config.text
        return self.shard.partition(self.config.heads * (c["qk_nope_head_dim"] + c["v_head_dim"]))

    @property
    def o_proj_columns(self):
        """(first, last) input columns of `o_proj` (columns are heads x v_head_dim)."""
        return self.shard.partition(self.config.heads * self.config.text["v_head_dim"])

    @property
    def gate_rows(self):
        """(first, last) rows of `linear_gate` (rows are heads x v_head_dim)."""
        return self.shard.partition(self.config.heads * self.config.text["v_head_dim"])

    @property
    def expert_range(self):
        """(first, last) routed-expert and shared-expert intermediate channels.

        The shared expert is `moe_intermediate_size * n_shared_experts` wide
        (`HYV4MoE.forward`), the same axis the routed experts shard.
        """
        return self.shard.partition(self.config.text["moe_intermediate_size"], alignment=EXPERT_ALIGNMENT)

    @property
    def dense_range(self):
        """(first, last) intermediate channels of the dense layer-0 FFN."""
        return self.shard.partition(self.config.text["intermediate_size"], alignment=EXPERT_ALIGNMENT)

    @property
    def dfly_dense_range(self):
        """(first, last) intermediate channels of each DFly2 dense FFN layer."""
        return self.shard.partition(self.config.dfly_intermediate_size, alignment=EXPERT_ALIGNMENT)

    @property
    def dfly_correction_range(self):
        """(first, last) intermediate channels of DFly2 hidden_correction."""
        return self.shard.partition(self.config.dfly_hidden_correction_intermediate_size,
                                    alignment=EXPERT_ALIGNMENT)

    @property
    def vocabulary_range(self):
        """(first, last) vocabulary rows of the embedding and the head."""
        return self.shard.partition(self.config.vocabulary)

    # --- replication -------------------------------------------------------

    def replicated(self, name):
        """True when every rank holds a whole copy of checkpoint tensor `name`."""
        shardable = (
            name.endswith(".q_b_proj.weight") or name.endswith(".o_proj.weight")
            or name.endswith(".linear_gate.weight")
            or ".mlp.experts." in name or ".mlp.shared_experts." in name
            or name.endswith((".mlp.gate_proj.weight", ".mlp.up_proj.weight", ".mlp.down_proj.weight"))
            or name.endswith((".hidden_correction.gate_proj.weight",
                              ".hidden_correction.up_proj.weight",
                              ".hidden_correction.down_proj.weight"))
            or name in ("lm_head.weight", "model.embed_tokens.weight")
        )
        return not shardable

    @property
    def rank(self):
        return self.shard.rank

    @property
    def world_size(self):
        return self.shard.world_size
