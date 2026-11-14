# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0
"""Hy4 routed MoE: sigmoid router, clamped SwiGLU experts, shared expert.

The checkpoint contract (config + `weights.py`):

- router `gate.weight` replicates in BF16 and `e_score_correction_bias` in
  FP32 (the do-not-quantize floor: the router is never quantized);
- routing is `sigmoid(logits) + bias`, group-1 top-k (256 experts, top-8),
  renormalised then scaled by `routed_scaling_factor` (2.827);
- routed experts run W4A16 NVFP4 with the clamped SwiGLU
  (`gate <= swiglu_limit`, `|up| <= swiglu_limit`, before SiLU); the shared
  expert and the dense layer-0 FFN are plain BF16 and unclamped — they see
  every token and never enter the packed path.

GPU path: `ignition.components.moe.PreparedRouter` /
`PreparedRoutedExperts` over the loader's packed experts (`load_experts`).
CPU path (the rank-construction dry run): the same numerics in torch from the
loaded source tensors — BF16 fused experts packed at load, or the export's
packed W4A16 bytes decoded per the `contract.py` table. The packed-bytes
provenance and swap/clamp mutants are covered by
`tests/recipes/test_hy4_expert_loading.py`.
"""

from dataclasses import dataclass

import torch

from .config import ModelConfig


@dataclass(frozen=True)
class RouterWeights:
    gate: torch.Tensor           # [experts, hidden] BF16, replicated
    correction_bias: torch.Tensor  # [experts] FP32 (`e_score_correction_bias`)

    @property
    def experts(self):
        return self.gate.shape[0]


@dataclass(frozen=True)
class DenseMlpWeights:
    """A dense gated MLP (layer-0 FFN / shared expert), intermediate-sharded."""

    gate_proj: torch.Tensor  # [inter_local, hidden]
    up_proj: torch.Tensor    # [inter_local, hidden]
    down_proj: torch.Tensor  # [hidden, inter_local]

    def forward(self, x):
        return (torch.nn.functional.silu(x @ self.gate_proj.T) * (x @ self.up_proj.T)) @ self.down_proj.T


@dataclass
class ReferenceRoutedExperts:
    """CPU stand-in for the packed expert path: clamped SwiGLU in torch.

    `packed` tensors are the loader's NVFP4 form (`w13` [E, 2L, h//2] uint8,
    block scales, per-shard gains — see `contract.packed_expert_tensor_names`);
    `bf16` tensors are the checkpoint's fused experts split into source views
    (`weights.Hy4WeightLoader.expert_stacks`). Either form runs the contract's
    clamped SwiGLU so the decoder loop is numerically exercisable without the
    CUDA packer (`flashinfer.fp4_quantize`) and the fused kernels.
    """

    swiglu_limit: float
    packed: dict | None = None
    bf16: dict | None = None

    _E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

    @classmethod
    def from_bf16(cls, stacks, prefix, *, swiglu_limit):
        gate = stacks[f"{prefix}.gate_proj.weight"]
        up = stacks[f"{prefix}.up_proj.weight"]
        down = stacks[f"{prefix}.down_proj.weight"]
        return cls(swiglu_limit=swiglu_limit,
                   bf16={"gate": gate.float(), "up": up.float(), "down": down.float()})

    @classmethod
    def from_packed(cls, packed, prefix, *, swiglu_limit):
        return cls(swiglu_limit=swiglu_limit,
                   packed={suffix: packed[f"{prefix}.{projection}.{suffix}"]
                           for projection in ("gate_up_proj", "down_proj")
                           for suffix in ("weight", "weight_scale", "weight_scale_2")})

    @property
    def experts(self):
        if self.packed is not None:
            return int(self.packed["gate_up_proj.weight"].shape[0])
        return int(self.bf16["gate"].shape[0])

    def _dequant(self, packed_rows, scales, gain):
        """The `contract.py` packing read back: E2M1 nibbles x block scales x gain."""
        codes = packed_rows.to(torch.int64)
        table = torch.tensor(self._E2M1, dtype=torch.float32)
        low = table[codes & 7] * torch.where((codes & 8) != 0, -1.0, 1.0)
        high = table[(codes >> 4) & 7] * torch.where((codes & 128) != 0, -1.0, 1.0)
        values = torch.stack((low, high), dim=-1).flatten(-2)
        return values * scales.float().repeat_interleave(16, dim=-1) * float(gain)

    def expert_weights(self, index):
        """(gate, up, down) matrices of one expert, dequantized."""
        if self.packed is not None:
            count, inter2, _ = self.packed["gate_up_proj.weight"].shape
            inter = inter2 // 2
            fused, fscale, fgain = (self.packed[f"gate_up_proj.{suffix}"]
                                    for suffix in ("weight", "weight_scale", "weight_scale_2"))
            down, dscale, dgain = (self.packed[f"down_proj.{suffix}"]
                                   for suffix in ("weight", "weight_scale", "weight_scale_2"))
            gate = self._dequant(fused[index, :inter], fscale[index, :inter], fgain[index])
            up = self._dequant(fused[index, inter:], fscale[index, inter:], fgain[index])
            return gate, up, self._dequant(down[index], dscale[index], dgain[index])
        return self.bf16["gate"][index], self.bf16["up"][index], self.bf16["down"][index]

    def forward(self, x, topk_indices, topk_weights):
        """Weighted expert mixture over the routed tokens (the reference loop)."""
        rows, _ = x.shape
        final = torch.zeros_like(x, dtype=torch.float32)
        for index in range(self.experts):
            hits = (topk_indices == index).nonzero(as_tuple=False)
            if not len(hits):
                continue
            token_idx, slot = hits[:, 0], hits[:, 1]
            gate_w, up_w, down_w = self.expert_weights(index)
            gate = x[token_idx].float() @ gate_w.T
            up = x[token_idx].float() @ up_w.T
            activated = torch.nn.functional.silu(gate.clamp(max=self.swiglu_limit)) \
                * up.clamp(-self.swiglu_limit, self.swiglu_limit)
            out = activated @ down_w.T
            final.index_add_(0, token_idx, out * topk_weights[token_idx, slot, None].float())
        return final.to(x.dtype)


@dataclass
class MoEBlock:
    """One sparse layer's MoE: router + routed experts + shared expert."""

    router: RouterWeights
    shared: DenseMlpWeights
    routed: object  # `PreparedRoutedExperts` handle on GPU, ReferenceRoutedExperts on CPU
    config: ModelConfig
    device: torch.device
    _prepared: object = None  # GPU wiring (PreparedRouter + PreparedRoutedExperts)

    def route(self, x):
        """(router_logits, topk_weights, topk_indices): sigmoid + bias, top-k, renorm, scale."""
        c = self.config.text
        experts, top_k = self.router.experts, c["num_experts_per_tok"]
        logits = x.float() @ self.router.gate.float().T
        scores = torch.sigmoid(logits)
        choice = scores + self.router.correction_bias.float()
        if c["n_group"] > 1:  # the general group contract; Hy4 runs n_group == 1
            groups = choice.view(-1, c["n_group"], experts // c["n_group"])
            keep = groups.topk(2, dim=-1)[0].sum(-1)
            group_idx = keep.topk(c["topk_group"], dim=-1)[1]
            mask = torch.zeros_like(keep).scatter_(1, group_idx, 1)
            choice = choice.masked_fill(~mask.unsqueeze(-1).expand(-1, 1, experts // c["n_group"])
                                        .reshape(-1, experts).bool(), float("-inf"))
        topk_indices = choice.topk(top_k, dim=-1, sorted=False)[1]
        topk_weights = scores.gather(1, topk_indices)
        if c["norm_topk_prob"]:
            topk_weights = topk_weights / (topk_weights.sum(-1, keepdim=True) + 1e-20)
        return logits, topk_weights * c["routed_scaling_factor"], topk_indices

    def forward(self, x):
        """Routed mixture + shared expert (`HYV4MoE.forward`)."""
        _, topk_weights, topk_indices = self.route(x)
        if self.device.type == "cuda" and self._prepared is not None:
            routed = self._prepared(x, ids=topk_indices, weights=topk_weights)
        else:
            routed = self.routed.forward(x, topk_indices, topk_weights)
        return routed + self.shared.forward(x)


class DenseBlock:
    """The dense layer-0 FFN (`HYV4MLP`): unclamped SwiGLU."""

    def __init__(self, weights: DenseMlpWeights, device):
        self.weights, self.device = weights, device

    def forward(self, x):
        return self.weights.forward(x)
