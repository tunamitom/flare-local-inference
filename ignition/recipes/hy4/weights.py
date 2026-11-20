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


def dfly_checkpoint_names(config, *, include_ignored=False):
    """All `model.dfly_layers.*` tensor names for a DFly2-equipped checkpoint.

    Active loaded tensors: `dfly_layers` dense attention + SwiGLU FFN layers
    (14 tensors each: no indexer, no iHC) plus 10 wrapper tensors (`mtp_fusion`
    4, `final_norm` 1, `hidden_correction` 5). With `include_ignored=True`,
    adds the 2 unused `confidence_head.proj.{weight,bias}` tensors present in
    the checkpoint shards so strict inventory checks account for every key.
    """
    if not config.has_dfly:
        return frozenset()
    names = {
        "model.dfly_layers.final_norm.weight",
        "model.dfly_layers.mtp_fusion.eh_proj.weight",
        "model.dfly_layers.mtp_fusion.enorm.weight",
        "model.dfly_layers.mtp_fusion.hnorm.weight",
        "model.dfly_layers.mtp_fusion.gate",
        "model.dfly_layers.hidden_correction.embed_norm.weight",
        "model.dfly_layers.hidden_correction.hidden_norm.weight",
        "model.dfly_layers.hidden_correction.gate_proj.weight",
        "model.dfly_layers.hidden_correction.up_proj.weight",
        "model.dfly_layers.hidden_correction.down_proj.weight",
    }
    if include_ignored:
        names |= {
            "model.dfly_layers.confidence_head.proj.weight",
            "model.dfly_layers.confidence_head.proj.bias",
        }
    for index in range(config.dfly_layers):
        prefix = f"model.dfly_layers.{index}"
        names |= {f"{prefix}.input_layernorm.weight", f"{prefix}.post_attention_layernorm.weight"}
        attention = f"{prefix}.self_attn"
        names |= {f"{attention}.{name}" for name in (
            "kv_a_layernorm.weight", "kv_a_proj_with_mqa.weight", "kv_b_proj.weight",
            "learnable_sink_param", "linear_gate.weight", "o_proj.weight",
            "q_a_layernorm.weight", "q_a_proj.weight", "q_b_proj.weight")}
        names |= {f"{prefix}.mlp.{proj}.weight" for proj in ("down_proj", "gate_proj", "up_proj")}
    return frozenset(names)


def vision_checkpoint_names(config):
    """All `vit.*` tensor names of a `hy_v4_vl` checkpoint's vision tower.

    The vision tower of the official `tencent/Hy4` checkpoint carries exactly
    649 tensors: 3 embedding tensors, 16 tensors per
    `vision_config.num_hidden_layers` ViT layer, and 6 `vit.perceive` adapter
    tensors. Text-only serving (the plan's V-A staging) preserves these
    passthrough weights in the package but leaves them unselected; the family
    is enumerated so strict key coverage accepts exactly these keys and fails
    closed on anything else under `vit.`.
    """
    vision = config.text.get("vision_config")
    if not isinstance(vision, dict):
        return frozenset()
    layers = vision.get("num_hidden_layers")
    if type(layers) is not int or layers < 1:
        raise ValueError(
            f"Hy4 vision requires a positive integer vision_config.num_hidden_layers, got {layers!r}")
    names = {
        "vit.embeddings.patch_embedding.weight", "vit.embeddings.patch_embedding.bias",
        "vit.embeddings.position_embedding.weight",
        "vit.perceive.norm.weight", "vit.perceive.image_newline",
    }
    names |= {f"vit.perceive.proj.{index}.{suffix}"
              for index in (0, 2) for suffix in ("weight", "bias")}
    for layer in range(layers):
        prefix = f"vit.layers.{layer}"
        names |= {f"{prefix}.{norm}.{suffix}"
                  for norm in ("input_layernorm", "post_attention_layernorm")
                  for suffix in ("weight", "bias")}
        names |= {f"{prefix}.self_attn.{proj}.{suffix}"
                  for proj in ("q_proj", "k_proj", "v_proj", "o_proj")
                  for suffix in ("weight", "bias")}
        names |= {f"{prefix}.mlp.{proj}.{suffix}"
                  for proj in ("dense_h_to_4h", "dense_4h_to_h")
                  for suffix in ("weight", "bias")}
    return frozenset(names)


def checkpoint_names(config, *, mtp=True, dfly=None, vision=False, packed=False):
    """Every tensor name of the checkpoint, from `ref/idx.json`'s schema.

    Verified against the real `model.safetensors.index.json` (2006 preview
    tensors; 2059 active text + DFly2 tensors and 649 `vit.*` tensors on
    official `tencent/Hy4`). `dfly` selects the 80 active `model.dfly_layers.*`
    tensors (default: follow `mtp`), `vision` the `vit.*` family (default off —
    text-only serving deliberately leaves it unselected); unselected families
    are still enumerated by the strict-coverage gate via
    `dfly_checkpoint_names(include_ignored=True)` / `vision_checkpoint_names`
    so any unknown key fails closed. `packed` describes the W4A16_NVFP4 export
    instead: the two fused expert tensors of every MoE block become the packed
    weight/scale/gain triples of `contract.packed_expert_tensor_names`.
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
    include_dfly = mtp if dfly is None else dfly
    if include_dfly and config.has_dfly:
        names |= set(dfly_checkpoint_names(config, include_ignored=False))
    if vision and config.has_vision:
        names |= set(vision_checkpoint_names(config))
    return frozenset(names)


# W1 gate d: the packed export is only loadable when its export_report.json
# proves a complete, verified build. The bound separates MSE-built packages
# from RTN/broken ones: max scaling reads 0.0951 and is refused.
#
# W13 correction (evidence, full preview build of 2026-11-14): 0.0815 was set
# from sampled expectations ("the real experts measure 0.0812") and the tiny
# model's max (0.0814). The full 59,904-matrix census (78x256x3) measures mean
# 0.08120, median 0.08121, max 0.08286 — 32 matrices (0.05%) sit above 0.0815.
# Those are genuine per-block-MSE floors, not build defects: the exhaustive
# 126-code sweep reproduces the same census on the worst matrices with the
# windowed search missing 0/786,432 blocks, and the legacy (unordered-sum)
# builder computes the identical 0.082911 on the worst matrix. The bar moves
# to 0.0835 — still 0.012 below the RTN census it must refuse.
EXPORT_MAX_RELATIVE_ERROR = 0.0835

# Origin-keyed evidence (2026-11-18 quality round; consult 10 section 1). A `gptq`
# package's codes are trained: the solver minimises each routed-expert matrix's
# OUTPUT error on calibration activations, which deliberately moves individual
# weights further from BF16 (census ~0.10 against the MSE build's 0.0812), so the
# census bar can no longer tell searched codes from unsearched ones. A `gptq`
# package proves itself by its calibration objective instead: every replaced
# matrix's tr(dW H dW^T) on the calibration Hessian must be strictly below the
# base MSE package's on the same Hessian, the base report and the calibration
# set are pinned by hash, and a loose census ceiling still refuses a blown-up
# solve. A report without an origin is the MSE build (`ptq-mse`).
EXPORT_ORIGINS = ("ptq-mse", "gptq")
GPTQ_CENSUS_CEILING = 0.20


def _gptq_evidence_problems(evidence, config):
    """What a `gptq`-origin report fails to prove (empty when the evidence holds)."""
    if not isinstance(evidence, dict):
        return ["gptq-origin export_report.json carries no gptq_evidence"]
    problems = []
    for key in ("base_report_sha256", "calibration_sha256"):
        value = evidence.get(key)
        if not isinstance(value, str) or len(value) != 64:
            problems.append(f"gptq_evidence.{key} is missing")
    objective = evidence.get("objective")
    expected = (config.layers - 1) * config.experts * 2      # fused gate_up + down per routed expert
    if not isinstance(objective, dict) or len(objective) != expected:
        count = len(objective) if isinstance(objective, dict) else None
        problems.append(f"gptq_evidence.objective covers {count} of {expected} routed matrices")
    else:
        bad = sum(1 for value in objective.values()
                  if not (isinstance(value, list) and len(value) == 2
                          and all(isinstance(x, (int, float)) for x in value) and 0 <= value[0] < value[1]))
        if bad:
            problems.append(f"gptq_evidence: {bad} matrices whose calibration objective is not "
                            "strictly below the base package's")
    return problems


def verify_export_package(snapshot, config):
    """Refuse a packed W4A16 export that cannot prove a complete, verified run.

    The per-expert builder once wrote ``index.json`` before verification and the
    report and used mode-0600 shards, so a crashed run looked complete. The
    loader now requires the evidence (W1 gate d):

    - ``export_report.json`` exists and shows ``byte_verified``, the full
      78x256x3 matrix census (every block x expert x projection) and
      ``census_max_relative`` <= ``EXPORT_MAX_RELATIVE_ERROR`` (0.0835; the
      W13 full-model measurement — see the constant's comment) for an MSE
      build, or, for a ``gptq``-origin build, <= ``GPTQ_CENSUS_CEILING`` plus
      ``gptq_evidence``: base-report and calibration hashes and a calibration
      objective strictly below the base package's for every routed matrix;
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
        origin = (report.get("producer") or {}).get("origin", "ptq-mse")
        if origin not in EXPORT_ORIGINS:
            problems.append(f"export_report.json declares unknown origin {origin!r}")
        bar = GPTQ_CENSUS_CEILING if origin == "gptq" else EXPORT_MAX_RELATIVE_ERROR
        max_error = report.get("census_max_relative")
        if not isinstance(max_error, (int, float)) or not 0 <= max_error <= bar:
            problems.append(f"census_max_relative {max_error!r} exceeds {bar}")
        if origin == "gptq":
            problems.extend(_gptq_evidence_problems(report.get("gptq_evidence"), config))
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
    learned speculation). `dfly` (default: follow `mtp`) selects the bundled
    DFly2 drafter family and `vision` the `vit.*` tower (default off —
    text-only serving preserves the passthrough weights unselected); either
    way every family present in the checkpoint is name-checked strictly.
    `expert_scale_compression` optionally stores the routed experts' block
    scales in the lossless `csf` encoding at load.
    """

    def __init__(self, manifest, device, *, shard=None, progress=None, prefixes=None, mtp=True,
                 dfly=None, vision=False, expert_scale_compression=None, pinned=True):
        super().__init__(manifest, device, prefixes=prefixes, progress=progress)
        from .config import ModelConfig

        if expert_scale_compression not in (None, "csf"):
            raise ValueError("Hy4 expert scale compression must be None or 'csf'")
        self.config = ModelConfig.load(manifest.snapshot, pinned=pinned)
        self.shard, self.mtp = shard, mtp
        self.dfly = self.mtp if dfly is None else dfly
        self.vision = vision
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
        # Every family the checkpoint may carry is enumerated (the 2 ignored
        # `confidence_head` tensors and the `vit.*` passthrough tower included),
        # so strict coverage accepts exactly those keys when present and fails
        # closed on any unknown name, while unselected families stay
        # deliberately unselected.
        inventory = set(checkpoint_names(self.config, mtp=True, dfly=True, vision=True, packed=packed))
        if self.config.has_dfly:
            inventory |= set(dfly_checkpoint_names(self.config, include_ignored=True))
        expected = checkpoint_names(self.config, mtp=self.mtp, dfly=self.dfly,
                                    vision=self.vision, packed=packed)
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

        # Materialize through the loader session (`load_tensor`) when the raw
        # `_sources` entry is a DECLARED placeholder (meta until transferred):
        # on the GPU construction path consuming it raw leaves the routed
        # experts as META tensors — a meta matmul then silently returns garbage
        # (the P5 integration gate caught exactly this).  Resident tensors (the
        # CPU dry run) are consumed as-is.
        fused = self._sources[f"{prefix}.gate_up_proj"]
        down = self._sources[f"{prefix}.down_proj"]
        if getattr(fused, "is_meta", False):
            fused = self.load_tensor(f"{prefix}.gate_up_proj")
        if getattr(down, "is_meta", False):
            down = self.load_tensor(f"{prefix}.down_proj")
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

    def load_resident_experts(self, prefix, resident, *, intermediate_range,
                             swiglu_limit=None, **packing):
        """Serve one MoE block's experts from fused packed bytes.

        Two source forms, both byte-identical to `load_quantized_experts`'
        full-tensor path at the same world size (the slicing is `state.py`'s,
        proven bit-exact in the M2 boundary work):

        CHECKPOINT form (the CUDA serve path — consult 12's direct aliases):
        the fused tensors are the loader's own meta checkpoint views and
        `intermediate_range` is the rank's real range. Each expert is aliased
        as a FULL-WIDTH view (`fused[index, :inter]`, `fused[index, inter:]`,
        `down[index]`, same for scales) and `load_fp4_experts` is told the FULL
        intermediate plus the range, so it does its own `source[first:last]` /
        `source[:, first//2:last//2]` slicing on meta views: one direct
        transfer per plane into the final fused planes, no staging copy
        (`torch_copy_bytes` stays 0), no permanent doubling of the routed
        bytes. Gains are already value-loaded.

        RESIDENT form (the CPU dry run and the numeric gates): `resident`
        supplies the rank's SLICE in `ReferenceRoutedExperts.from_packed`'s
        key form (`gate_up_proj.weight` = [E, 2*local, hidden//2], the gate
        half first) and `load_fp4_experts` is told the LOCAL intermediate with
        no further range. Called inside the loader session, like `load_experts`.
        """
        if expert_format(self.quantization, prefix) != ("nvfp4", "a16"):
            raise ValueError(f"{prefix}: expected W4A16 NVFP4 packed experts")
        if swiglu_limit is None:
            swiglu_limit = self.config.text["swiglu_limit"]
        count = self.config.experts
        first, last = intermediate_range
        local = last - first
        inter = self.config.text["moe_intermediate_size"]
        hidden = self.config.hidden
        fused, fused_scale, fused_gain = (resident[f"{prefix}.gate_up_proj.{suffix}"]
                                          for suffix in contract.PACKED_SUFFIXES)
        down, down_scale, down_gain = (resident[f"{prefix}.down_proj.{suffix}"]
                                       for suffix in contract.PACKED_SUFFIXES)
        checkpoint_form = bool(getattr(fused, "is_meta", False))
        width = inter if checkpoint_form else local
        down_width = inter if checkpoint_form else local
        for name, source, shape in (
                (f"{prefix}.gate_up_proj.weight", fused, (count, 2 * width, hidden // 2)),
                (f"{prefix}.gate_up_proj.weight_scale", fused_scale, (count, 2 * width, hidden // 16)),
                (f"{prefix}.gate_up_proj.weight_scale_2", fused_gain, (count,)),
                (f"{prefix}.down_proj.weight", down, (count, hidden, down_width // 2)),
                (f"{prefix}.down_proj.weight_scale", down_scale, (count, hidden, down_width // 16)),
                (f"{prefix}.down_proj.weight_scale_2", down_gain, (count,))):
            if tuple(source.shape) != shape:
                raise ValueError(
                    f"{name}: {'checkpoint' if checkpoint_form else 'resident rank'} slice has shape "
                    f"{tuple(source.shape)}, the packed expert contract at this range is {shape}")
        stacks = {}
        for index in range(count):
            target = f"{prefix}.experts.{index}"
            stacks |= {
                f"{target}.gate_proj.weight": fused[index, :width],
                f"{target}.up_proj.weight": fused[index, width:],
                f"{target}.down_proj.weight": down[index],
                f"{target}.gate_proj.weight_scale": fused_scale[index, :width],
                f"{target}.up_proj.weight_scale": fused_scale[index, width:],
                f"{target}.down_proj.weight_scale": down_scale[index],
                f"{target}.gate_proj.weight_scale_2": fused_gain[index:index + 1],
                f"{target}.up_proj.weight_scale_2": fused_gain[index:index + 1],
                f"{target}.down_proj.weight_scale_2": down_gain[index:index + 1],
            }
        for name, source in stacks.items():
            self._sources[name] = source
        try:
            return self.load_fp4_experts(
                tuple(f"{prefix}.experts.{index}" for index in range(count)),
                hidden_size=hidden,
                intermediate_size=inter if checkpoint_form else local,
                intermediate_range=(first, last) if checkpoint_form else None,
                format="nvfp4", activation_mode="a16",
                swiglu_limit=swiglu_limit, swiglu_clamp_after_silu=False,
                scale_compression=self.expert_scale_compression,
                scale_scratch=self.expert_scale_scratch((local + 127) // 128 * 128),
                **packing)
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


class _HeadersWeightSession:
    """Header dry run: account transfers without touching tensor payloads."""

    def __init__(self):
        self.payload_bytes = 0

    def __call__(self, destination, source):
        self.payload_bytes += destination.numel() * destination.element_size()
        return True

    def flush(self):
        pass

    def stats(self):
        return {"headers_only": True}


# safetensors header dtype names (what `TensorMetadata.dtype` carries).
_SAFETENSORS_DTYPES = {
    "U8": "uint8", "I8": "int8", "U16": "uint16", "I16": "int16", "I32": "int32",
    "I64": "int64", "F16": "float16", "BF16": "bfloat16", "F32": "float32",
    "F64": "float64", "F8_E4M3": "float8_e4m3fn", "F8_E5M2": "float8_e5m2", "BOOL": "bool",
}

def headers_dry_run(loader, *, headers=None):
    """Context manager exposing HEADER geometry as meta tensors.

    The rank-construction dry run against real metadata (consult3 gate c:
    `construct_rank` had never run on real shard headers). Every name, shape,
    dtype and rank-slicing check runs end to end and construction completes;
    tensor payloads are never read and destinations are meta tensors.

    `headers` may supply an explicit `name -> (dtype, shape)` inventory — a
    repacked header view over a package's per-expert headers — and defaults to
    the loader's own manifest headers.
    """
    import contextlib

    @contextlib.contextmanager
    def open_sources():
        import torch

        if loader._stack is not None or loader._session is not None:
            raise RuntimeError("weight loader context is already open")
        if headers is None:
            inventory = {name: (getattr(torch, _SAFETENSORS_DTYPES[meta.dtype]), tuple(meta.shape))
                         for name, meta in loader.manifest.tensors.items()}
        else:
            inventory = dict(headers)
        sources = {name: torch.empty(shape, dtype=dtype, device="meta")
                   for name, (dtype, shape) in inventory.items()}
        loader._sources = sources
        loader._session = _HeadersWeightSession()
        loader._stack = ()  # marks the loader open for `_require_open`
        try:
            yield loader
        finally:
            loader._sources = {}
            loader._session = None
            loader._stack = None

    return open_sources()


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
