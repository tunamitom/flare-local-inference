# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Hy4 checkpoint tensor names, FP32 policy, and expert loading.

Two checkpoint forms load through `Hy4WeightLoader`:

- the raw BF16 checkpoint stores fused stacked experts (`experts.gate_up_proj`
  / `experts.down_proj`); the loader splits them into source views before
  `DirectWeightLoader.load_bf16_experts_as_nvfp4` packs them to W4A16 NVFP4 at
  load time — a BF16 checkpoint then serves with no external quantization step;
- the W4A16_NVFP4 export stores the same fused prefixes already packed
  (`contract.py` names the exact tensors); the loader slices each rank's bytes
  bit-exactly into `DirectWeightLoader.load_fp4_experts`, which records the
  checkpoint-byte provenance of every served expert tensor.

Tensor coverage is checked strictly: an unmatched checkpoint key is an error,
not a silent drop (the stock streaming loader drops unmatched keys without
reporting, which left iHC, sinks and gates uninitialized for other models).
With `mtp=False` the MTP block is deliberately unselected (drafts load only
for learned speculation) and may be present or absent from the checkpoint.
"""

import json
from functools import cached_property

from ignition.components.weights import DirectWeightLoader
from ignition.preparation.quantization import expert_format, quantized_layers

from . import contract

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


def checkpoint_names(config, *, mtp=True, packed=False):
    """Every tensor name of the checkpoint, from `ref/idx.json`'s schema.

    Verified against the real `model.safetensors.index.json` (2006 tensors) in
    `tests/recipes/test_hy4_construction.py`. `packed` describes the W4A16_NVFP4
    export instead: the two fused expert tensors of every MoE block become the
    packed weight/scale/gain triples of `contract.packed_expert_tensor_names`.
    """
    names = {"lm_head.weight", "model.embed_tokens.weight", "model.norm.weight",
             "model.hc_head.hc_head_base", "model.hc_head.hc_head_fn", "model.hc_head.hc_head_scale"}

    def moe_names(prefix):
        if packed:
            return set(contract.packed_expert_geometry(config, f"{prefix}.mlp.experts"))
        return {f"{prefix}.mlp.experts.{proj}" for proj in ("down_proj", "gate_up_proj")}

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
            names |= moe_names(prefix)
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
            names |= moe_names(prefix)
            names |= {f"{prefix}.mlp.gate.{name}" for name in ("e_score_correction_bias", "weight")}
            names |= {f"{prefix}.mlp.shared_experts.{proj}.weight" for proj in ("down_proj", "gate_proj", "up_proj")}
    return frozenset(names)


# W1 gate d: the packed export is only loadable when its export_report.json
# proves a complete, verified build. The bound is the official MSE census bar
# (the real experts measure 0.0812; max scaling reads 0.0951 and is refused).
EXPORT_MAX_RELATIVE_ERROR = 0.0815


def verify_export_package(snapshot, config):
    """Refuse a packed W4A16 export that cannot prove a complete, verified run.

    The per-expert builder once wrote ``index.json`` before verification and the
    report and used mode-0600 shards, so a crashed run looked complete. The
    loader now requires the evidence (W1 gate d):

    - ``export_report.json`` exists and shows ``byte_verified``, the full
      78x256x3 matrix census (every block x expert x projection) and
      ``census_max_relative`` <= 0.0815;
    - the report counts window-edge scale choices for ALL experts (coverage
      per block and projection, not just the audited expert);
    - ``model.safetensors.index.json`` was written last (mtime at or after the
      shards, the config and the report);
    - every shard is mode 0644.
    """
    from pathlib import Path

    snapshot = Path(snapshot)
    problems = []
    report_path = snapshot / "export_report.json"
    report = None
    if not report_path.is_file():
        problems.append("export_report.json is missing")
    else:
        try:
            report = json.loads(report_path.read_text())
        except ValueError as exc:
            problems.append(f"export_report.json does not parse: {exc}")
    if report is not None:
        if report.get("byte_verified") is not True:
            problems.append("export_report.json does not show byte_verified")
        expected = (config.layers - 1 + config.draft_layers) * config.experts * 3
        census = report.get("census")
        if report.get("packed_matrices") != expected or not isinstance(census, dict) or len(census) != expected:
            problems.append(f"export_report.json shows "
                            f"{report.get('packed_matrices')} of {expected} packed matrices")
        max_error = report.get("census_max_relative")
        if not isinstance(max_error, (int, float)) or not 0 <= max_error <= EXPORT_MAX_RELATIVE_ERROR:
            problems.append(f"census_max_relative {max_error!r} exceeds {EXPORT_MAX_RELATIVE_ERROR}")
        counts = report.get("window_edge_counts")
        if not isinstance(counts, dict):
            problems.append("export_report.json has no window-edge counts")
        else:
            for prefix in contract.quantized_source_names(config):
                entry = counts.get(prefix)
                if not isinstance(entry, dict):
                    problems.append(f"window-edge counts missing for {prefix}")
                    continue
                for projection in ("gate_proj", "up_proj", "down_proj"):
                    per_expert = entry.get(projection)
                    if (not isinstance(per_expert, list) or len(per_expert) != config.experts
                            or any(not isinstance(count, int) or count < 0 for count in per_expert)):
                        problems.append(f"window-edge counts of {prefix}.{projection} do not cover "
                                        f"all {config.experts} experts")
    index_path = snapshot / "model.safetensors.index.json"
    shards = sorted(snapshot.glob("model-*.safetensors"))
    if not index_path.is_file():
        problems.append("model.safetensors.index.json is missing (the index is written last)")
    elif shards:
        latest = max([shard.stat().st_mtime_ns for shard in shards]
                     + [report_path.stat().st_mtime_ns if report_path.is_file() else 0,
                        (snapshot / "config.json").stat().st_mtime_ns])
        if index_path.stat().st_mtime_ns < latest:
            problems.append("model.safetensors.index.json was not written last")
    for shard in shards:
        if shard.stat().st_mode & 0o777 != 0o644:
            problems.append(f"{shard.name}: mode {shard.stat().st_mode & 0o777:04o}, expected 0644")
    if problems:
        raise ValueError("refusing packed Hy4 export: " + "; ".join(problems))
    return report


class Hy4WeightLoader(DirectWeightLoader):
    """Direct transfers for a Hy4 checkpoint, packing or slicing routed experts.

    `shard` is the tensor-parallel partition of the expert intermediate width
    (256 ranks-wide channels at TP8, 128 at TP16); omit it for a full-width
    load. `mtp` includes the checkpoint's MTP block in the selection; with
    `mtp=False` the block may be present or absent (drafts load only for
    learned speculation). `expert_scale_compression` optionally stores the
    routed experts' block scales in the lossless `csf` encoding at load.
    """

    def __init__(self, manifest, device, *, shard=None, progress=None, prefixes=None, mtp=True,
                 expert_scale_compression=None, pinned=True):
        super().__init__(manifest, device, prefixes=prefixes, progress=progress)
        from .config import ModelConfig

        if expert_scale_compression not in (None, "csf"):
            raise ValueError("Hy4 expert scale compression must be None or 'csf'")
        self.config = ModelConfig.load(manifest.snapshot, pinned=pinned)
        self.shard, self.mtp = shard, mtp
        self.expert_scale_compression = expert_scale_compression

    @cached_property
    def quantization(self):
        """The exported checkpoint declares its own ModelOpt packing.

        The raw BF16 checkpoint has no `quantization_config`; its experts pack
        to W4A16 NVFP4 at load instead of through `linear_format` declarations.
        A checkpoint that does declare one is parsed strictly: a parser error —
        including any mentioning `MIXED_PRECISION` — is never swallowed as
        unquantized, because treating a malformed declaration as plain storage
        would repack quantized experts as if they were BF16.
        """
        config = json.loads((self.manifest.metadata_path / "config.json").read_text())
        if config.get("quantization_config") is None:
            return {}
        return quantized_layers(config)

    @cached_property
    def _packed(self):
        """The checkpoint is the W4A16_NVFP4 export when any prefix declares it."""
        return bool(self.quantization)

    def _selected_names(self):
        if self._packed:
            verify_export_package(self.manifest.metadata_path, self.config)
        packed = self._packed
        inventory = checkpoint_names(self.config, mtp=True, packed=packed)
        expected = checkpoint_names(self.config, mtp=self.mtp, packed=packed)
        available = set(self.manifest.tensors)
        missing, unexpected = sorted(expected - available), sorted(available - inventory)
        if missing or unexpected:
            raise ValueError(f"Hy4 checkpoint tensor names disagree with the recipe: "
                             f"missing={missing[:8]}, unexpected={unexpected[:8]}")
        return {name for name in super()._selected_names() if name in expected}

    @cached_property
    def scale_scratch(self):
        """The two shared CSF expansion buffers (Step-5's pattern).

        Routed layers run serially and share them; they hold one layer's raw
        block scales per layout. Sizing reserves two layouts (about 0.15 GiB
        per rank at TP8): an expansion writes one while the next layer's
        compressed scales stage in the other.
        """
        if self.expert_scale_compression is None:
            return ()
        import torch

        c = self.config.text
        inter = c["moe_intermediate_size"]
        first, last = ((0, inter) if self.shard is None else self.shard.partition(inter, alignment=32))
        padded = (last - first + 127) // 128 * 128
        count, hidden = self.config.experts, self.config.hidden
        return tuple(torch.empty(shape, dtype=torch.float8_e4m3fn, device=self.device)
                     for shape in ((count, 2 * padded, hidden // 16),
                                   (count, hidden, padded // 16)))

    def expert_scale_scratch(self, intermediate):
        """This call's views of the shared expansion buffers, or None without CSF."""
        if not self.scale_scratch:
            return None
        count, hidden = self.config.experts, self.config.hidden
        shapes = ((count, 2 * intermediate, hidden // 16),
                  (count, hidden, (intermediate // 16 + 3) // 4 * 4))
        return tuple(buffer.flatten()[:count * rows * columns].view(count, rows, columns)
                     for buffer, (count, rows, columns) in zip(self.scale_scratch, shapes, strict=True))

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

    def quantized_expert_stacks(self, prefix):
        """Slice the export's packed fused experts into per-expert loader views.

        Every alias is a zero-copy view of the checkpoint tensor, so the bytes
        `load_fp4_experts` serves are exactly the checkpoint's at any world
        size (its provenance records the map). `gate_up_proj` holds [gate; up]
        along the intermediate rows, matching the BF16 fused semantics, and
        gate and up share the fused `weight_scale_2` gain.
        """
        geometry = contract.packed_expert_geometry(self.config, prefix)
        sources = {name: self._sources[name] for name in geometry}
        for name, (dtype, shape) in geometry.items():
            source = sources[name]
            if source.dtype != dtype or tuple(source.shape) != shape:
                raise ValueError(f"{name}: storage disagrees with the packed expert contract {shape}")
        count, inter = self.config.experts, self.config.text["moe_intermediate_size"]
        fused, fused_scale, fused_gain = (sources[f"{prefix}.gate_up_proj.{suffix}"]
                                         for suffix in contract.PACKED_SUFFIXES)
        down, down_scale, down_gain = (sources[f"{prefix}.down_proj.{suffix}"]
                                       for suffix in contract.PACKED_SUFFIXES)
        stacks = {}
        for index in range(count):
            target = f"{prefix}.experts.{index}"
            stacks |= {
                f"{target}.gate_proj.weight": fused[index, :inter],
                f"{target}.up_proj.weight": fused[index, inter:],
                f"{target}.down_proj.weight": down[index],
                f"{target}.gate_proj.weight_scale": fused_scale[index, :inter],
                f"{target}.up_proj.weight_scale": fused_scale[index, inter:],
                f"{target}.down_proj.weight_scale": down_scale[index],
                f"{target}.gate_proj.weight_scale_2": fused_gain[index:index + 1],
                f"{target}.up_proj.weight_scale_2": fused_gain[index:index + 1],
                f"{target}.down_proj.weight_scale_2": down_gain[index:index + 1],
            }
        for name, source in stacks.items():
            self._sources[name] = source
        return stacks

    def load_quantized_experts(self, prefix, *, intermediate_range=None, swiglu_limit=None, **packing):
        """Serve one MoE block's experts from the export's packed W4A16 storage.

        The checkpoint bytes are sliced verbatim (bit-exact at any world size)
        into `load_fp4_experts`, which records their provenance. The clamp
        policy is the checkpoint's: routed SwiGLU operands clamp before SiLU
        (`gate <= limit`, `|up| <= limit`), so `swiglu_clamp_after_silu` stays
        False; the shared expert and layer 0 are unclamped and never pass
        through here.
        """
        if expert_format(self.quantization, prefix) != ("nvfp4", "a16"):
            raise ValueError(f"{prefix}: expected W4A16 NVFP4 packed experts")
        if swiglu_limit is None:
            swiglu_limit = self.config.text["swiglu_limit"]
        inter = self.config.text["moe_intermediate_size"]
        local = inter if intermediate_range is None else intermediate_range[1] - intermediate_range[0]
        stacks = self.quantized_expert_stacks(prefix)
        try:
            return self.load_fp4_experts(
                tuple(f"{prefix}.experts.{index}" for index in range(self.config.experts)),
                hidden_size=self.config.hidden, intermediate_size=inter,
                intermediate_range=intermediate_range,
                format="nvfp4", activation_mode="a16",
                swiglu_limit=swiglu_limit, swiglu_clamp_after_silu=False,
                scale_compression=self.expert_scale_compression,
                scale_scratch=self.expert_scale_scratch((local + 127) // 128 * 128), **packing)
        finally:
            for name in stacks:
                del self._sources[name]

    def load_experts(self, prefix, *, intermediate_range=None, swiglu_limit=None, **packing):
        """Load one MoE block's routed experts into the W4A16 NVFP4 runtime form.

        A W4A16_NVFP4 checkpoint's packed experts are sliced bit-exactly; a raw
        BF16 checkpoint's fused experts are packed at load (gate and up share
        one global scale per expert shard, down has its own, `amax / (448 * 6)`,
        matching ModelOpt's max scaling). Hy4 clamps the routed SwiGLU operands
        before SiLU (`gate <= limit`, `|up| <= limit`); the shared expert and
        layer 0 are unclamped and never pass through here.
        """
        from .config import ModelConfig  # noqa: F401  (documents the config dependency)

        if swiglu_limit is None:
            swiglu_limit = self.config.text["swiglu_limit"]
        if self._packed:
            if prefix not in self.quantization:
                raise ValueError(f"{prefix}: the packed checkpoint declares no W4A16_NVFP4 experts here")
            return self.load_quantized_experts(prefix, intermediate_range=intermediate_range,
                                               swiglu_limit=swiglu_limit, **packing)
        inter = self.config.text["moe_intermediate_size"]
        local = inter if intermediate_range is None else intermediate_range[1] - intermediate_range[0]
        stacks = self.expert_stacks(prefix)
        try:
            return self.load_bf16_experts_as_nvfp4(
                prefix, num_experts=self.config.experts,
                hidden_size=self.config.hidden,
                intermediate_size=inter,
                intermediate_range=intermediate_range,
                swiglu_limit=swiglu_limit, swiglu_clamp_after_silu=False,
                scale_compression=self.expert_scale_compression,
                scale_scratch=self.expert_scale_scratch((local + 127) // 128 * 128), **packing)
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


class _CpuWeightSession:
    """CPU stand-in for `b12x.loader.DirectWeightSession`: copy in-process."""

    def __init__(self):
        self.payload_bytes = 0

    def __call__(self, destination, source):
        destination.copy_(source)
        self.payload_bytes += destination.numel() * destination.element_size()
        return True

    def flush(self):
        pass

    def stats(self):
        return {}


def cpu_dry_run(loader):
    """Context manager populating `loader._sources` from the checkpoint on CPU.

    `DirectWeightLoader.__enter__` requires CUDA (the b12x streaming session);
    the rank-construction dry run reads every selected tensor into CPU memory
    instead, so construction, tensor-name coverage and the dtype policy run
    end to end without a GPU. GPU-only packing (`flashinfer.fp4_quantize`,
    `b12x.moe.prepare_weights`) is deliberately not reached: CPU construction
    routes experts through the reference path
    (`moe.ReferenceRoutedExperts`), whose numerics are the contract's own.
    """
    import contextlib

    from safetensors import safe_open

    @contextlib.contextmanager
    def open_sources():
        if loader._stack is not None or loader._session is not None:
            raise RuntimeError("weight loader context is already open")
        selected = loader._selected_names()
        shards = {}
        for name in selected:
            shards.setdefault(loader.manifest.tensors[name].shard, []).append(name)
        sources = {}
        for shard, names in shards.items():
            with safe_open(loader.manifest.snapshot / shard, framework="pt", device="cpu") as handle:
                for name in names:
                    sources[name] = handle.get_tensor(name)
        loader._sources = sources
        loader._session = _CpuWeightSession()
        loader._stack = ()  # marks the loader open for `_require_open`
        try:
            yield loader
        finally:
            loader._sources = {}
            loader._session = None
            loader._stack = None

    return open_sources()
