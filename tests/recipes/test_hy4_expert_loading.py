# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""The Hy4 quantized-checkpoint expert path: bit-exact slicing, provenance,
numeric output, and swap/clamp mutants.

CPU only: the b12x CUDA boundary (`moe.prepare_weights`) is stubbed to capture
exactly the tensors and the activation policy the runtime would receive, and
the NVFP4 decode + clamped SwiGLU forward is evaluated in plain torch from the
contract's documented packing. Nothing here touches a GPU or reads real
checkpoint tensors; the reference metadata under `HY4_REF_DIR` supplies the
real `config.json`/index names for the inventory tests.
"""

import itertools
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from ignition.recipes.hy4 import contract
from ignition.recipes.hy4.weights import Hy4WeightLoader, checkpoint_names

REF = Path(os.environ.get("HY4_REF_DIR", "/home/tom/hy4/ref"))

# Tiny packed geometry: two experts, 128 intermediate channels, hidden 128
# (divisible by the NVFP4 group and tile widths).
COUNT, INTER, HIDDEN = 2, 128, 128
PREFIX = "model.layers.1.mlp.experts"
# The checkpoint's clamp policy: `gate <= 10`, `|up| <= 10`, before SiLU.
SWIGLU_LIMIT = 10.0


@pytest.fixture(scope="module")
def model_config():
    config = REF / "bf16_config.json"
    if not config.is_file():
        pytest.skip(f"Hy4 reference metadata not present under {REF}")
    from ignition.recipes.hy4.config import ModelConfig

    return ModelConfig.from_dict(json.loads(config.read_text()))


class FakeSession:
    """CPU stand-in for b12x's direct-weight session: copy and record."""

    def __init__(self):
        self.transfers = []
        self.payload_bytes = 0

    def __call__(self, destination, source):
        destination.copy_(source)
        self.transfers.append((destination, source))
        self.payload_bytes += destination.numel() * destination.element_size()
        return True

    def flush(self):
        pass

    def stats(self):
        return {}


def packed_checkpoint(seed=0):
    """A packed W4A16 export's one fused prefix, per the contract's schema.

    E2M1 codes are seeded integers 0..15 (sign-magnitude nibbles) and both
    gains are positive; the block scales are ordinary FP8 values.
    """
    generator = torch.Generator().manual_seed(seed)
    geometry = contract.packed_expert_geometry(tiny_config(), PREFIX)
    tensors = {}
    for name, (dtype, shape) in geometry.items():
        if dtype == torch.uint8:
            tensors[name] = torch.randint(0, 256, shape, generator=generator, dtype=torch.uint8)
        elif dtype == torch.float8_e4m3fn:
            values = torch.rand(shape, generator=generator) * 1.75 + 0.25
            tensors[name] = values.to(torch.float8_e4m3fn)
        else:
            tensors[name] = torch.rand(shape, generator=generator) * 0.04 + 0.01
    return tensors


def tiny_config():
    return SimpleNamespace(text={"moe_intermediate_size": INTER, "n_routed_experts": COUNT,
                                "swiglu_limit": SWIGLU_LIMIT},
                           hidden=HIDDEN, experts=COUNT, layers=2, draft_layers=1)


def make_loader(sources, *, expert_scale_compression=None):
    loader = Hy4WeightLoader.__new__(Hy4WeightLoader)
    loader.config = tiny_config()
    loader.device = torch.device("cpu")
    loader.shard = None
    loader.mtp = True
    loader.expert_scale_compression = expert_scale_compression
    loader.prefixes = None
    loader.progress = None
    loader.progress_label = "Load weights"
    loader._display = loader._load_stats = loader._stack = None
    loader._session = FakeSession()
    loader._sources = dict(sources)
    loader.quantization = {PREFIX: dict(contract.EXPERT_ALGORITHM)}
    return loader


@pytest.fixture()
def captured(monkeypatch):
    """Stub the CUDA-bound b12x calls, capturing the prepared tensors and policy."""
    from b12x.moe import fused_moe as moe

    record = {}
    real_plan = moe.plan_weights

    def plan_spy(**kwargs):
        record["activation"] = kwargs["activation"]
        record["plan"] = real_plan(**kwargs)
        return record["plan"]

    def prepare_stub(*, plan, weights, scale_scratch=None, allocate=None):
        record["weights"] = weights
        record["scale_scratch"] = scale_scratch
        record["prepared"] = SimpleNamespace(
            _impl=SimpleNamespace(w1_fp4=weights.w13, w2_fp4=weights.w2,
                                  w1_blockscale=weights.w13_block_scales,
                                  w2_blockscale=weights.w2_block_scales, plan=plan))
        return record["prepared"]

    monkeypatch.setattr(moe, "plan_weights", plan_spy)
    monkeypatch.setattr(moe, "prepare_weights", prepare_stub)
    monkeypatch.setattr("b12x._lib.intrinsics.reference.swizzle_block_scale",
                        lambda tensor, allocate=None: tensor)
    return record


# --- NVFP4 decode and the clamped SwiGLU forward, in plain torch -------------

_E2M1 = torch.tensor([0., 0.5, 1., 1.5, 2., 3., 4., 6.])


def decode_fp4(packed):
    """(rows, cols//2) packed uint8 -> (rows, cols) float32 E2M1 values.

    The low nibble holds the even column. Both sides of every comparison below
    use this one decoder, so the tests discriminate slicing and clamp policy,
    not nibble order.
    """
    codes = packed.to(torch.int64)

    def nibble(n):
        magnitude = _E2M1[n & 7]
        return magnitude * torch.where((n & 8) != 0, -1.0, 1.0)

    return torch.stack((nibble(codes & 0x0F), nibble(codes >> 4)), dim=-1).reshape(
        *packed.shape[:-1], -1)


def dequant(packed, scales, gain):
    values = decode_fp4(packed)
    return values * scales.to(torch.float32).repeat_interleave(16, dim=-1) * float(gain)


def swiglu(gate_pre, up_pre, *, limit, clamp_after):
    if clamp_after:
        return torch.nn.functional.silu(gate_pre).clamp(max=limit) * up_pre.clamp(-limit, limit)
    return torch.nn.functional.silu(gate_pre.clamp(max=limit)) * up_pre.clamp(-limit, limit)


def reference_forward(checkpoint, expert, inputs):
    """The contract's semantics, read straight from the checkpoint tensors."""
    fused, fused_scale, fused_gain = (checkpoint[f"{PREFIX}.gate_up_proj.{suffix}"]
                                     for suffix in contract.PACKED_SUFFIXES)
    down, down_scale, down_gain = (checkpoint[f"{PREFIX}.down_proj.{suffix}"]
                                   for suffix in contract.PACKED_SUFFIXES)
    gate = dequant(fused[expert, :INTER], fused_scale[expert, :INTER], fused_gain[expert])
    up = dequant(fused[expert, INTER:], fused_scale[expert, INTER:], fused_gain[expert])
    down = dequant(down[expert], down_scale[expert], down_gain[expert])
    activated = swiglu(inputs @ gate.T, inputs @ up.T,
                       limit=SWIGLU_LIMIT, clamp_after=False)
    return activated @ down.T


def served_forward(record, expert, inputs):
    """What the runtime would compute from what the loader handed prepare."""
    weights, activation = record["weights"], record["activation"]
    local = weights.w13.shape[1] // 2
    gains = weights.w13_global_scales
    up = dequant(weights.w13[expert, :local], weights.w13_block_scales[expert, :local],
                 gains[expert, 1] if gains.ndim == 2 else gains[expert])
    gate = dequant(weights.w13[expert, local:], weights.w13_block_scales[expert, local:],
                   gains[expert, 0] if gains.ndim == 2 else gains[expert])
    down = dequant(weights.w2[expert], weights.w2_block_scales[expert],
                   weights.w2_global_scales[expert])
    activated = swiglu(inputs @ gate.T, inputs @ up.T,
                       limit=activation.swiglu_limit,
                       clamp_after=activation.swiglu_clamp_after_silu)
    return activated @ down.T


def probe_inputs():
    # Rows sweep the activation range so both clamp bands are exercised: some
    # gate pre-activations fall inside (limit/2, limit] and others above limit.
    scale = (torch.arange(1, 25, dtype=torch.float32) / 6).reshape(24, 1)
    return scale * (torch.arange(HIDDEN, dtype=torch.float32).remainder(7) - 3).abs().add(1).repeat(24, 1)


def max_error(loader, record, checkpoint):
    loader.load_experts(PREFIX)
    inputs = probe_inputs()
    return max(float((served_forward(record, expert, inputs)
                      - reference_forward(checkpoint, expert, inputs)).abs().max())
               for expert in range(COUNT))


# --- bit-exact slicing and provenance ---------------------------------------

def test_quantized_expert_slicing_is_bit_exact_with_checkpoint_provenance(captured):
    from ignition.preparation import provenance

    checkpoint = packed_checkpoint()
    loader = make_loader(checkpoint)
    loader.load_experts(PREFIX)
    record = captured
    # Every byte served to the runtime equals its checkpoint source byte.
    assert record["weights"].w13.shape == (COUNT, 2 * INTER, HIDDEN // 2)
    for destination, source in loader._session.transfers:
        assert torch.equal(destination, source)
    # The recorded provenance piece maps reproduce the served bytes from the
    # checkpoint tensors alone: served expert bytes equal checkpoint bytes.
    for served, alias in ((record["weights"].w13, "gate_up"), (record["weights"].w2, "down")):
        entry = provenance.lookup(served)
        assert entry is not None and entry["packing"] in ("nvfp4_native", "w4a16_mma_packed")
        rebuilt = torch.zeros_like(served.reshape(-1).view(torch.uint8))
        covered = torch.zeros_like(rebuilt)
        for source in entry["sources"]:
            name = source["tensor"]
            tensor = loader_quantized_views(checkpoint)[name]
            flat = tensor.reshape(-1).view(torch.uint8)
            for piece in source["pieces"]:
                contiguous = piece["contiguous_bytes"]
                for index in itertools.product(*(range(extent) for extent in piece["extents"])):
                    begin = piece["source_offset"] + sum(i * s for i, s in zip(index, piece["source_strides"]))
                    end = piece["destination_offset"] + sum(i * s for i, s in zip(index, piece["destination_strides"]))
                    rebuilt[end:end + contiguous] = flat[begin:begin + contiguous]
                    covered[end:end + contiguous] = 1
        assert bool(covered.all()), "provenance pieces must cover every served byte"
        assert torch.equal(rebuilt, served.reshape(-1).view(torch.uint8))
    # The expert names of the recorded sources are exactly the contract's.
    weights_entry = provenance.lookup(record["weights"].w13)
    expected = {f"{PREFIX}.experts.{index}.{projection}.weight"
                for index in range(COUNT) for projection in ("up_proj", "gate_proj")}
    assert {source["tensor"] for source in weights_entry["sources"]} == expected


def loader_quantized_views(checkpoint):
    """The per-expert alias views, rebuilt from the contract's row order."""
    views = {}
    fused, fused_scale, fused_gain = (checkpoint[f"{PREFIX}.gate_up_proj.{suffix}"]
                                     for suffix in contract.PACKED_SUFFIXES)
    down, down_scale, down_gain = (checkpoint[f"{PREFIX}.down_proj.{suffix}"]
                                   for suffix in contract.PACKED_SUFFIXES)
    for index in range(COUNT):
        target = f"{PREFIX}.experts.{index}"
        views |= {
            f"{target}.gate_proj.weight": fused[index, :INTER],
            f"{target}.up_proj.weight": fused[index, INTER:],
            f"{target}.down_proj.weight": down[index],
        }
    return views


def test_quantized_expert_slicing_is_bit_exact_on_a_tp_shard(captured):
    checkpoint = packed_checkpoint()
    loader = make_loader(checkpoint)
    first, last = 32, 96
    loader.load_experts(PREFIX, intermediate_range=(first, last))
    weights = captured["weights"]
    assert weights.w13.shape == (COUNT, 2 * (last - first), HIDDEN // 2)
    # The served bytes are exactly the checkpoint's intermediate rows [first:last].
    for expert in range(COUNT):
        assert torch.equal(weights.w13[expert, :last - first],
                           checkpoint[f"{PREFIX}.gate_up_proj.weight"][expert, INTER + first:INTER + last])
        assert torch.equal(weights.w13[expert, last - first:],
                           checkpoint[f"{PREFIX}.gate_up_proj.weight"][expert, first:last])
        assert torch.equal(weights.w2[expert],
                           checkpoint[f"{PREFIX}.down_proj.weight"][expert, :, first // 2:last // 2])
    for destination, source in loader._session.transfers:
        assert torch.equal(destination, source)


# --- numeric check with swap and clamp mutants ------------------------------

def test_expert_numeric_matches_reference(captured):
    checkpoint = packed_checkpoint()
    loader = make_loader(checkpoint)
    assert max_error(loader, captured, checkpoint) == 0


def test_swap_and_clamp_mutants_are_detected(captured, monkeypatch):
    """Defect 24: the numeric check catches a swapped split and a wrong clamp."""
    checkpoint = packed_checkpoint()
    # Swap mutant: the loader's fused [gate; up] split is reversed.
    loader = make_loader(checkpoint)
    original = loader.quantized_expert_stacks

    def swapped(self, prefix):
        stacks = original(prefix)
        for index in range(COUNT):
            target = f"{prefix}.experts.{index}"
            for suffix in ("weight", "weight_scale"):
                gate, up = f"{target}.gate_proj.{suffix}", f"{target}.up_proj.{suffix}"
                stacks[gate], stacks[up] = stacks[up], stacks[gate]
                self._sources[gate], self._sources[up] = self._sources[up], self._sources[gate]
        return stacks

    monkeypatch.setattr(loader, "quantized_expert_stacks",
                        lambda prefix: swapped(loader, prefix))
    assert max_error(loader, captured, checkpoint) > 0

    # Clamp mutants: the loader declares a wrong limit, or clamps after SiLU.
    def mutate(**changes):
        mutant = make_loader(checkpoint)
        real = Hy4WeightLoader.load_fp4_experts

        def load_fp4_experts(self, prefixes, **kwargs):
            kwargs |= changes
            return real(self, prefixes, **kwargs)

        monkeypatch.setattr(mutant, "load_fp4_experts",
                            lambda prefixes, **kwargs: load_fp4_experts(mutant, prefixes, **kwargs))
        return max_error(mutant, captured, checkpoint)

    assert mutate(swiglu_limit=5.0) > 0
    assert mutate(swiglu_clamp_after_silu=True) > 0


# --- defects 22 and 23 ------------------------------------------------------

def real_manifest(tmp_path, names):
    import shutil

    shutil.copy(REF / "bf16_config.json", tmp_path / "config.json")
    return SimpleNamespace(snapshot=tmp_path, metadata_path=tmp_path,
                           tensors={name: SimpleNamespace(shard="model-00001.safetensors")
                                    for name in names})


def test_mtp_false_loads_the_real_checkpoint_inventory(tmp_path, model_config):
    """Defect 22: `mtp=False` must accept the real checkpoint and skip the draft."""
    from ignition.recipes.hy4.weights import checkpoint_names

    names = checkpoint_names(model_config)
    manifest = real_manifest(tmp_path, names)
    loader = Hy4WeightLoader(manifest, torch.device("cpu"), mtp=False)
    selected = loader._selected_names()
    assert selected == {name for name in names if not name.startswith("model.mtp_layers.")}
    assert len(selected) == len(names) - 27
    # With the draft requested, the full inventory must be present.
    assert Hy4WeightLoader(manifest, torch.device("cpu"), mtp=True)._selected_names() == set(names)
    # A checkpoint without the MTP block is fine when the draft is not selected.
    without = real_manifest(tmp_path, {name for name in names if not name.startswith("model.mtp_layers.")})
    assert len(Hy4WeightLoader(without, torch.device("cpu"), mtp=False)._selected_names()) == len(selected)
    with pytest.raises(ValueError, match="missing="):
        Hy4WeightLoader(without, torch.device("cpu"), mtp=True)._selected_names()
    # An unknown tensor is still an error, never a silent drop.
    broken = real_manifest(tmp_path, names | {"model.layers.1.mlp.experts.mystery"})
    with pytest.raises(ValueError, match="unexpected="):
        Hy4WeightLoader(broken, torch.device("cpu"), mtp=False)._selected_names()


def test_parser_errors_are_never_swallowed_as_unquantized(tmp_path, model_config):
    """Defect 23: only an absent `quantization_config` means unquantized."""
    import shutil

    names = checkpoint_names(model_config)
    manifest = real_manifest(tmp_path, names)
    shutil.copy(REF / "bf16_config.json", tmp_path / "config.json")

    def with_quantization(declaration):
        config = dict(json.loads((REF / "bf16_config.json").read_text()))
        if declaration is not None:
            config["quantization_config"] = declaration
        (tmp_path / "config.json").write_text(json.dumps(config))
        return Hy4WeightLoader(manifest, torch.device("cpu"), mtp=False)

    # The raw BF16 checkpoint has no declaration: unquantized, experts pack at load.
    assert with_quantization(None).quantization == {}
    # A declared but malformed ModelOpt form raises instead of reading as BF16.
    for malformed in ({"quant_method": "modelopt", "quant_algo": "NVFP4",
                       "kv_cache_quant_algo": None, "quantized_layers": {}},
                      {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION",
                       "kv_cache_quant_algo": "fp8", "quantized_layers": {}},
                      {"quant_method": "gptq", "quant_algo": "MIXED_PRECISION",
                       "kv_cache_quant_algo": None, "quantized_layers": {}}):
        with pytest.raises(ValueError):
            with_quantization(malformed).quantization
    # The well-formed export parses to its declaration map.
    declaration = contract.quantization_config(model_config)
    parsed = with_quantization(declaration).quantization
    assert set(parsed) == set(contract.quantized_source_names(model_config))


# --- the export contract's packed form --------------------------------------

def test_validate_export_accepts_the_toolkit_fused_prefix_form(model_config):
    source = json.loads((REF / "bf16_config.json").read_text())
    export = contract.export_config(source)
    layers = contract.validate_export(export)
    # Fused-prefix W4A16_NVFP4 entries including the MTP block, no input_scale.
    assert "model.mtp_layers.0.mlp.experts" in layers
    assert set(layers) == set(contract.quantized_source_names(model_config))
    for name, entry in layers.items():
        assert entry == contract.EXPERT_ALGORITHM
        assert "input_scale" not in entry
    assert "input_scale" not in contract.PACKED_SUFFIXES
    # The loader consumes exactly the contract's packed names and geometry.
    packed = checkpoint_names(model_config, packed=True)
    for prefix in contract.quantized_source_names(model_config):
        for name, (dtype, shape) in contract.packed_expert_geometry(model_config, prefix).items():
            assert name in packed
            assert dtype in (torch.uint8, torch.float8_e4m3fn, torch.float32)
            assert all(isinstance(extent, int) and extent > 0 for extent in shape)
    # A16 is required: an A4-labelled expert form is rejected.
    a4 = json.loads(json.dumps(export))
    a4["quantization_config"]["quantized_layers"]["model.layers.1.mlp.experts"] = {
        "quant_algo": "NVFP4", "group_size": 16}
    with pytest.raises(ValueError, match="W4A16"):
        contract.validate_export(a4)


def test_quantized_stacks_follow_the_contract_geometry():
    checkpoint = packed_checkpoint()
    loader = make_loader(checkpoint)
    stacks = loader.quantized_expert_stacks(PREFIX)
    try:
        for index in range(COUNT):
            target = f"{PREFIX}.experts.{index}"
            # [gate; up] row order: the reference chunk semantics.
            assert torch.equal(stacks[f"{target}.gate_proj.weight"], checkpoint[f"{PREFIX}.gate_up_proj.weight"][index, :INTER])
            assert torch.equal(stacks[f"{target}.up_proj.weight"], checkpoint[f"{PREFIX}.gate_up_proj.weight"][index, INTER:])
            assert torch.equal(stacks[f"{target}.down_proj.weight"], checkpoint[f"{PREFIX}.down_proj.weight"][index])
            # Gate and up share the fused gain; every alias is a zero-copy view.
            gate_gain = stacks[f"{target}.gate_proj.weight_scale_2"]
            up_gain = stacks[f"{target}.up_proj.weight_scale_2"]
            assert gate_gain.data_ptr() == up_gain.data_ptr()
            assert stacks[f"{target}.gate_proj.weight"].data_ptr() == \
                checkpoint[f"{PREFIX}.gate_up_proj.weight"][index, :INTER].data_ptr()
    finally:
        for name in stacks:
            loader._sources.pop(name, None)
    # A storage disagreement is an error, not a silent reshape.
    broken = packed_checkpoint()
    broken[f"{PREFIX}.down_proj.weight"] = broken[f"{PREFIX}.down_proj.weight"][:, :, :INTER // 4]
    loader = make_loader(broken)
    with pytest.raises(ValueError, match="packed expert contract"):
        loader.quantized_expert_stacks(PREFIX)
