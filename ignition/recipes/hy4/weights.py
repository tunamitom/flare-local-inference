# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Hy4 checkpoint tensor names, FP32 policy, and BF16-to-NVFP4 expert packing.

The checkpoint stores fused stacked experts (`experts.gate_up_proj` /
`experts.down_proj`); the loader consumes the canonical `gate`/`up`/`down`
stacks, so this module splits the fused tensors into source views before
`DirectWeightLoader.load_bf16_experts_as_nvfp4` packs them to W4A16 NVFP4 at
load time — a BF16 checkpoint then serves with no external quantization step.

Tensor coverage is checked strictly: an unmatched checkpoint key is an error,
not a silent drop (the stock streaming loader drops unmatched keys without
reporting, which left iHC, sinks and gates uninitialized for other models).
"""

from functools import cached_property

from ignition.components.weights import DirectWeightLoader

# Loaded and kept in FP32 at runtime, whatever the checkpoint stores: the iHC
# `fn`/`base`/`scale` parameters, the learnable sinks, the indexer's
# `weights_proj` and `k_norm`, and the untied `lm_head` (`enable_lm_head_fp32`).
# The routed `gate.e_score_correction_bias` joins them (transformers' strict
# FP32 list). Anything else decodes in the checkpoint's storage dtype.
def kept_fp32(name):
    return (name.endswith((".hc_fn", ".hc_base", ".hc_scale", ".hc_head_fn", ".hc_head_base", ".hc_head_scale"))
            or name.endswith((".learnable_sink_param", ".e_score_correction_bias"))
            or name.endswith((".weights_proj.weight", ".k_norm.weight", ".k_norm.bias"))
            or name == "lm_head.weight")


def checkpoint_names(config, *, mtp=True):
    """Every tensor name of the checkpoint, from `ref/idx.json`'s schema.

    Verified against the real `model.safetensors.index.json` (2006 tensors) in
    `tests/recipes/test_hy4_construction.py`.
    """
    names = {"lm_head.weight", "model.embed_tokens.weight", "model.norm.weight",
             "model.hc_head.hc_head_base", "model.hc_head.hc_head_fn", "model.hc_head.hc_head_scale"}
    for layer in range(config.layers):
        prefix = f"model.layers.{layer}"
        names |= {f"{prefix}.hc_attn_layer.hc_pre.{part}" for part in ("hc_base", "hc_fn", "hc_scale")}
        names |= {f"{prefix}.hc_mlp_layer.hc_pre.{part}" for part in ("hc_base", "hc_fn", "hc_scale")}
        names |= {f"{prefix}.input_layernorm.weight", f"{prefix}.post_attention_layernorm.weight"}
        attention = f"{prefix}.self_attn"
        names |= {f"{attention}.{name}" for name in (
            "kv_a_layernorm.weight", "kv_a_proj_with_mqa.weight", "kv_b_proj.weight",
            "learnable_sink_param", "linear_gate.weight", "o_proj.weight",
            "q_a_layernorm.weight", "q_a_proj.weight", "q_b_proj.weight")}
        if config.indexer_type(layer) == "full":
            names |= {f"{attention}.indexer.{name}" for name in (
                "k_norm.bias", "k_norm.weight", "weights_proj.weight", "wk.weight", "wq_b.weight")}
        if layer == 0:
            names |= {f"{prefix}.mlp.{proj}.weight" for proj in ("down_proj", "gate_proj", "up_proj")}
        else:
            names |= {f"{prefix}.mlp.experts.{proj}" for proj in ("down_proj", "gate_up_proj")}
            names |= {f"{prefix}.mlp.gate.{name}" for name in ("e_score_correction_bias", "weight")}
            names |= {f"{prefix}.mlp.shared_experts.{proj}.weight" for proj in ("down_proj", "gate_proj", "up_proj")}
    if mtp:
        for index in range(config.draft_layers):
            prefix = f"model.mtp_layers.{index}"
            names |= {f"{prefix}.{name}" for name in (
                "eh_proj.weight", "enorm.weight", "final_layernorm.weight", "hnorm.weight",
                "input_layernorm.weight", "post_attention_layernorm.weight")}
            attention = f"{prefix}.self_attn"
            names |= {f"{attention}.{name}" for name in (
                "kv_a_layernorm.weight", "kv_a_proj_with_mqa.weight", "kv_b_proj.weight",
                "learnable_sink_param", "linear_gate.weight", "o_proj.weight",
                "q_a_layernorm.weight", "q_a_proj.weight", "q_b_proj.weight")}
            names |= {f"{attention}.indexer.{name}" for name in (
                "k_norm.bias", "k_norm.weight", "weights_proj.weight", "wk.weight", "wq_b.weight")}
            names |= {f"{prefix}.mlp.experts.{proj}" for proj in ("down_proj", "gate_up_proj")}
            names |= {f"{prefix}.mlp.gate.{name}" for name in ("e_score_correction_bias", "weight")}
            names |= {f"{prefix}.mlp.shared_experts.{proj}.weight" for proj in ("down_proj", "gate_proj", "up_proj")}
    return frozenset(names)


class Hy4WeightLoader(DirectWeightLoader):
    """Direct transfers for a Hy4 checkpoint, packing routed experts on the way.

    `shard` is the tensor-parallel partition of the expert intermediate width
    (256 ranks-wide channels at TP8, 128 at TP16); omit it for a full-width
    load. `mtp` includes the checkpoint's MTP block in the selection.
    """

    def __init__(self, manifest, device, *, shard=None, progress=None, prefixes=None, mtp=True):
        super().__init__(manifest, device, prefixes=prefixes, progress=progress)
        from .config import ModelConfig

        self.config = ModelConfig.load(manifest.snapshot)
        self.shard, self.mtp = shard, mtp

    @cached_property
    def quantization(self):
        """The exported checkpoint declares its own ModelOpt packing.

        The raw BF16 checkpoint has no `quantization_config`; its experts pack
        to W4A16 NVFP4 at load instead of through `linear_format` declarations.
        """
        try:
            return super().quantization
        except ValueError as error:
            if "MIXED_PRECISION" in str(error):
                return {}
            raise

    def _selected_names(self):
        expected = checkpoint_names(self.config, mtp=self.mtp)
        available = set(self.manifest.tensors)
        missing, unexpected = sorted(expected - available), sorted(available - expected)
        if missing or unexpected:
            raise ValueError(f"Hy4 checkpoint tensor names disagree with the recipe: "
                             f"missing={missing[:8]}, unexpected={unexpected[:8]}")
        return {name for name in super()._selected_names() if name in expected}

    def expert_stacks(self, prefix):
        """Split the checkpoint's fused experts into the loader's stacked views.

        `experts.gate_up_proj` holds [gate; up] along the intermediate axis —
        the reference forward splits `gate, up = gate_up.chunk(2, dim=-1)` on
        the GEMM output — and `experts.down_proj` is already the down stack.
        """
        c, hidden, inter = self.config.text, self.config.hidden, self.config.text["moe_intermediate_size"]
        count = c["n_routed_experts"]
        import torch

        fused = self._sources[f"{prefix}.gate_up_proj"]
        down = self._sources[f"{prefix}.down_proj"]
        if fused.shape != (count, 2 * inter, hidden) or fused.dtype != torch.bfloat16:
            raise ValueError(f"{prefix}.gate_up_proj: storage disagrees with fused BF16 expert geometry")
        if down.shape != (count, hidden, inter) or down.dtype != torch.bfloat16:
            raise ValueError(f"{prefix}.down_proj: storage disagrees with BF16 expert geometry")
        whole = slice(None)
        stacks = {
            f"{prefix}.gate_proj.weight": fused[(whole, slice(0, inter), whole)],
            f"{prefix}.up_proj.weight": fused[(whole, slice(inter, 2 * inter), whole)],
            f"{prefix}.down_proj.weight": down,
        }
        for name, source in stacks.items():
            self._sources[name] = source
        return stacks

    def load_experts(self, prefix, *, intermediate_range=None, swiglu_limit=None, **packing):
        """Pack one layer's fused BF16 experts to W4A16 NVFP4 at load.

        Gate and up share one global scale per expert shard, down has its own
        (`amax / (448 * 6)`), matching ModelOpt's max scaling. Hy4 clamps the
        routed SwiGLU operands before SiLU (`gate <= limit`, `|up| <= limit`);
        the shared expert and layer 0 are unclamped and never pass through
        here. `swiglu_clamp_after_silu` stays False for exactly that reason.
        """
        from .config import ModelConfig  # noqa: F401  (documents the config dependency)

        if swiglu_limit is None:
            swiglu_limit = self.config.text["swiglu_limit"]
        stacks = self.expert_stacks(prefix)
        try:
            return self.load_bf16_experts_as_nvfp4(
                prefix, num_experts=self.config.experts,
                hidden_size=self.config.hidden,
                intermediate_size=self.config.text["moe_intermediate_size"],
                intermediate_range=intermediate_range,
                swiglu_limit=swiglu_limit, swiglu_clamp_after_silu=False, **packing)
        finally:
            for name in stacks:
                del self._sources[name]

    def load_norm(self, name, width):
        """A BF16 normalization parameter of `width` elements."""
        import torch

        source = self._sources[name]
        if source.shape != (width,) or source.dtype not in (torch.bfloat16, torch.float32):
            raise ValueError(f"{name}: expected a {width}-element normalization parameter")
        return self.load_tensor(name)

    def load_ihc(self, prefix, *, head=False):
        """One iHC module's `fn`/`base`/`scale`, kept FP32.

        The collapse is `pre = sigmoid(..) + eps`, `post = 2.0 * sigmoid(..) +
        eps` (`hc_magnitude`), `out = post_i * branch + stream_i`; streams are
        replicated on every rank, so the operator adds no collectives.
        """
        import torch

        suffix = ("hc_head_fn", "hc_head_base", "hc_head_scale") if head else ("hc_fn", "hc_base", "hc_scale")
        streams = self.config.text["hc_mult"]
        rows = streams if head else 2 * streams
        expected = {suffix[0]: (rows, streams * self.config.hidden), suffix[1]: (rows,), suffix[2]: (1 if head else 2,)}
        loaded = {}
        for part, shape in expected.items():
            name = f"{prefix}.{part}"
            source = self._sources[name]
            if tuple(source.shape) != shape:
                raise ValueError(f"{name}: expected shape {shape}, checkpoint has {tuple(source.shape)}")
            loaded[part] = self.load_tensor(name, dtype=torch.float32)
        return loaded

    def load_sink(self, prefix):
        """The per-head learnable attention sink, kept FP32 (`learnable_sink_init` 0)."""
        import torch

        name = f"{prefix}.learnable_sink_param"
        source = self._sources[name]
        if tuple(source.shape) != (self.config.heads,):
            raise ValueError(f"{name}: expected one sink per attention head")
        return self.load_tensor(name, dtype=torch.float32)

    def load_head(self, name="lm_head.weight"):
        """The untied vocabulary projection, kept FP32 (`enable_lm_head_fp32`)."""
        import torch

        source = self._sources[name]
        if tuple(source.shape) != (self.config.vocabulary, self.config.hidden):
            raise ValueError(f"{name}: expected the vocabulary projection")
        return self.load_tensor(name, dtype=torch.float32)
