# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0
"""The Hy4 MTP draft block: one single-stream GLM-style predictor.

Semantics (`/home/tom/hy4/reference/hy4_mtp.py`, cross-checked between the
vLLM and SGLang ports):

    h = eh_proj(cat[ enorm(embed(next_token)), hnorm(prev_hidden) ])   # enorm first
    h = h + self_attn(input_layernorm(h))     # single stream, NO iHC
    h = h + mlp(post_attention_layernorm(h))
    out = final_layernorm(h)                  # shared lm_head on top

`prev_hidden` is the target model's final single-stream hidden BEFORE
`model.norm` — `hc_head(streams)`. The block owns a full DSA indexer, its own
sink and gate and a 256-expert top-8 MoE with a shared expert; draft KV and
index pages are its own (the target's 78 layers plus the draft layer account
for them in `recipe._DRAFT_PAGE_BYTES`). The draft experts load only for
learned speculation (`weights.Hy4WeightLoader(mtp=...)`).

The block is single-stream: no iHC state row, so speculative capacity costs
rows, not wider rows (the target's rows stay 4 x hidden).
"""

from dataclasses import dataclass

import torch

from .attention import Hy4LayerAttention, rms_norm
from .state import _attention_state, _moe_state


@dataclass
class MtpBlock:
    """One rank's MTP draft block."""

    eh_proj: torch.Tensor     # [hidden, 2 * hidden]
    enorm: torch.Tensor       # [hidden]
    hnorm: torch.Tensor       # [hidden]
    final_norm: torch.Tensor  # [hidden]
    attention: Hy4LayerAttention
    input_norm: torch.Tensor
    post_attention_norm: torch.Tensor
    mlp: object
    config: object
    device: torch.device

    @classmethod
    def build(cls, builder, selection):
        """Construct the draft from the `model.mtp_layers.<i>` tensors.

        The draft layers are `config.draft_layers` (the preview ships one); the
        block is self-contained — its indexer is `full` and owns its selection
        cell (`selection`), so the IndexCache hands it its own buffers.
        """
        config, sharding, loader = builder.config, builder.sharding, builder.loader
        assert config.draft_layers == 1, "Hy4 ships one MTP predictor layer"
        prefix = "model.mtp_layers.0"
        hidden = config.hidden
        eh_proj = builder.linear(f"{prefix}.eh_proj", (hidden, 2 * hidden))
        enorm = builder.norm(f"{prefix}.enorm.weight", hidden)
        hnorm = builder.norm(f"{prefix}.hnorm.weight", hidden)
        final_norm = builder.norm(f"{prefix}.final_layernorm.weight", hidden)
        attention = _attention_state(builder, config.layers, prefix, selection)
        input_norm = builder.norm(f"{prefix}.input_layernorm.weight", hidden)
        post_norm = builder.norm(f"{prefix}.post_attention_layernorm.weight", hidden)
        mlp = _moe_state(builder, f"{prefix}.mlp", dense=False)
        return cls(eh_proj=eh_proj, enorm=enorm, hnorm=hnorm, final_norm=final_norm,
                   attention=attention, input_norm=input_norm, post_attention_norm=post_norm,
                   mlp=mlp, config=config, device=loader.device)

    def embed_merge(self, next_embeds, prev_hidden):
        """`eh_proj(cat[enorm(embed), hnorm(hidden)])` — enorm FIRST (the ports agree)."""
        left = rms_norm(next_embeds, self.enorm, self.config.text["rms_norm_eps"])
        right = rms_norm(prev_hidden, self.hnorm, self.config.text["rms_norm_eps"])
        return torch.cat((left, right), dim=-1) @ self.eh_proj.T

    def forward(self, next_embeds, prev_hidden, positions, cache, visible, *, token_rows):
        """One draft step: returns (hidden after final_layernorm, attention output).

        The attention is the same gated sparse MLA as the target's layers (own
        indexer + sink + gate) against the draft's own latent/index caches
        (layer `config.layers`); the MoE is the 256-expert top-8 form.
        """
        layer = self.config.layers
        h = self.embed_merge(next_embeds, prev_hidden)
        x = rms_norm(h, self.input_norm, self.config.text["rms_norm_eps"])
        attn = self.attention
        q_resid, q_absorbed, latent = attn.project(x, positions)
        cache.write_latent(layer, token_rows, positions, latent)
        cache.write_index_keys(layer, token_rows, positions, attn.prepare_index_keys(x, positions))
        selected, counts = attn.select(x, q_resid, positions, visible,
                                      cache.key_table(layer), cache.slots_for(token_rows))
        attended = attn.attend(q_absorbed, cache.latents(layer), selected, counts)
        h = h + attn.gate_and_project(attended.output, x)
        x = rms_norm(h, self.post_attention_norm, self.config.text["rms_norm_eps"])
        h = h + self.mlp.forward(x)
        return rms_norm(h, self.final_norm, self.config.text["rms_norm_eps"]), attended
