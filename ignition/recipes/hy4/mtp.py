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

`prev_hidden` is the target model's final single-stream hidden AFTER
`model.norm` — `norm(hc_head(streams))`, the very tensor the shared lm_head
consumes (consult3 finding F — FIXED; the pre-fix recipe fed the pre-norm
`hc_head(streams)`). Upstream convention evidence:
  * transformers 5.23.0 (the upstream authority): models/hy_v4/modeling_hy_v4.py:855
    returns `last_hidden_state = self.norm(self.hc_head(hidden_states))`;
    generation/candidate_generator.py:1418 feeds `model_outputs.hidden_states[-1]`
    (= that post-norm hidden, verified empirically on tiny-hy4) to the MTP, and
    :1427 runs the MTP over the FULL prompt (prompt absorption into `mtp_cache`);
    modeling_layers.py:343-370 merges `cat[enorm(embeds), hnorm(prev_hidden)]`;
    modeling_layers.py:524-558 chains draft steps: each step's returned hidden is
    the next step's `prev_hidden` and positions advance by one.
  * vLLM: ref/vllm_hy4_model.py:403 (target returns `self.norm(...)`, which
    ref/vllm_hy4_mtp.py:365-385 consumes); ref/vllm_hy4_mtp.py:496 zeroes the
    embed input at position 0 ("not needed by MTP", same as hy_v3_mtp.py:128);
    vllm v1/worker/gpu/spec_decode/multi_module_mtp/speculator.py:384: "The output
    hidden state at position P and the token id at P+1 are used to draft the token
    at P+2" — the draft-pairing this recipe implements.
  * SGLang: ref/sgl_hunyuan_v4.py:734 (target returns `final_norm(...)`), read back
    as `forward_batch.spec_info.hidden_states` by ref/sgl_hunyuan_v4_nextn.py:146.
`model.norm` has a per-channel weight, so `hnorm` does not cancel the convention.

The block is single-stream: no iHC state row, so speculative capacity costs
rows, not wider rows (the target's rows stay 4 x hidden). It owns a full DSA
indexer, its own sink and gate and a 256-expert top-8 MoE with a shared
expert; draft KV and index pages are its own (layer `config.layers`; the
target's 78 layers plus the draft layer account for them in
`recipe._DRAFT_PAGE_BYTES`). The draft experts load only for learned
speculation (`weights.Hy4WeightLoader(mtp=...)`).
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
        """`eh_proj(cat[enorm(embed), hnorm(hidden)])` — enorm FIRST (the ports agree).

        `prev_hidden` is POST-`model.norm` (the module docstring's convention
        evidence). Swapping the halves is the `swap draft order` gate mutant.
        """
        left = rms_norm(next_embeds, self.enorm, self.config.text["rms_norm_eps"])
        right = rms_norm(prev_hidden, self.hnorm, self.config.text["rms_norm_eps"])
        return torch.cat((left, right), dim=-1) @ self.eh_proj.T

    def forward(self, next_embeds, prev_hidden, positions, cache, visible, *, token_rows):
        """One draft pass over `token_rows` rows; returns (hidden after final_layernorm, attention output, selected).

        Position-0 zeroing (consult3 W5; vLLM ref/vllm_hy4_mtp.py:496 and
        hy_v3_mtp.py:128 "masking inputs at position 0, as not needed by MTP"):
        the embed input at position 0 is zeroed before `enorm`. Row t pairs
        `next_embeds[t]` (= embed of the token at t+1) with `prev_hidden[t]`
        (hidden at t) at position t and drafts the token at t+2.

        The attention is the same gated sparse MLA as the target's layers (own
        indexer + sink + gate) against the draft's own latent/index caches
        (layer `config.layers`); the MoE is the 256-expert top-8 form. Rows in
        one call attend causally (`visible`), so a multi-token prompt-absorption
        call and chained single-row steps see the same context as the engines.
        """
        layer = self.config.layers
        next_embeds = torch.where((positions == 0).unsqueeze(-1), 0, next_embeds)
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
        return rms_norm(h, self.final_norm, self.config.text["rms_norm_eps"]), attended, selected
