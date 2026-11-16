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
from .profile import TIMER


@dataclass(frozen=True)
class RouterWeights:
    gate: torch.Tensor           # [experts, hidden] BF16, replicated
    correction_bias: torch.Tensor  # [experts] FP32 (`e_score_correction_bias`)

    @property
    def experts(self):
        return self.gate.shape[0]


@dataclass(frozen=True)
class PackedMoEInputProjections:
    """One GEMM for the MoE's same-input projections (packed at load).

    The consult7 §5 bake-in: the router `gate` and the shared expert's
    `gate_proj`/`up_proj` all consume the MoE input `x`, so their rows pack
    into one matrix — one `x @ packed.T` reads `x` once (`down_proj` reads the
    intermediate, not `x`). Pure layout; built at weight-load time.
    """

    packed: torch.Tensor   # [experts + 2 * inter_local, hidden]
    slices: dict           # name -> (first, last)

    @classmethod
    def build(cls, router, shared):
        gate_rows = router.gate.shape[0]
        inter = shared.gate_proj.shape[0]
        packed = torch.cat((router.gate, shared.gate_proj, shared.up_proj), dim=0).contiguous()
        return cls(packed=packed, slices={
            "router": (0, gate_rows),
            "shared_gate": (gate_rows, gate_rows + inter),
            "shared_up": (gate_rows + inter, gate_rows + 2 * inter),
        })

    def rows(self, name):
        first, last = self.slices[name]
        return self.packed[first:last]


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
        # Keyed `projection.suffix` (matching `expert_weights`/`experts`) —
        # bare-suffix keys collide across projections and the down stack would
        # silently shadow the fused gate/up stack.
        return cls(swiglu_limit=swiglu_limit,
                   packed={f"{projection}.{suffix}": packed[f"{prefix}.{projection}.{suffix}"]
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
        # The lookup table must live with the codes (the serve runs this
        # reference path on CUDA; a CPU table raises "indices should be either
        # on cpu or on the same device as the indexed tensor").
        table = torch.tensor(self._E2M1, dtype=torch.float32, device=packed_rows.device)
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


# consult18 E2 — calibrated A4 prefill MoE.
# The export is A16 only (no activation scales). A calibration boot
# (HY4_MOE_CALIBRATE=1) prepares every routed op with b12x's amax epilogue
# (RoutingSpec.collect_activation_amax) and gives each layer a [1, E, 2]
# buffer of per-expert [input, intermediate] amax; the serve zeroes them after
# warm-up and saves them per rank after every request. An A4 boot
# (HY4_MOE_A4=<dir>) reads them back as b12x activation quantization
# multipliers (6 * 448 / amax) for the prefill op.
A4_GSCALE = 6.0 * 448.0


class MoeCalibration:
    """Per-rank registry of the calibration boot's amax buffers."""

    def __init__(self):
        self.buffers = {}       # expert prefix -> [1, E, 2] FP32 device tensor

    @staticmethod
    def enabled():
        import os

        return (os.environ.get("HY4_MOE_CALIBRATE") or "0") not in ("", "0")

    def buffer(self, key, num_experts, device):
        import torch

        if key not in self.buffers:
            self.buffers[key] = torch.zeros((1, int(num_experts), 2), dtype=torch.float32,
                                            device=device)
        return self.buffers[key]

    def reset(self):
        for tensor in self.buffers.values():
            tensor.zero_()

    def save(self, path):
        import os

        import torch

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        host = {key: tensor[0].detach().cpu() for key, tensor in self.buffers.items()}
        tmp = path + ".tmp"
        torch.save(host, tmp)
        os.replace(tmp, path)
        return len(host)


CALIBRATION = MoeCalibration()


def a4_activation_scales(expert_prefix, *, rank):
    """(input, intermediate) quantization multipliers for one layer's experts
    from the HY4_MOE_A4 sidecar (`<dir>/r<rank>.pt`), or None when A4 is off.

    Experts the calibration never routed take their layer's widest range
    (b12x shares the input scale as the minimum multiplier anyway)."""
    import os

    import torch

    root = os.environ.get("HY4_MOE_A4") or ""
    if not root:
        return None
    path = os.path.join(root, f"r{int(rank)}.pt")
    table = _A4_SIDECARS.get(path)
    if table is None:
        table = _A4_SIDECARS[path] = torch.load(path, map_location="cpu")
    amax = table.get(f"{expert_prefix}.fused")
    if amax is None:
        raise KeyError(f"HY4_MOE_A4 sidecar {path} has no calibration for {expert_prefix}")
    amax = amax.float().clone()
    for column in (0, 1):
        widest = float(amax[:, column].max())
        if not widest > 0:
            raise ValueError(f"{expert_prefix}: calibration saw no routed tokens (column {column})")
        amax[:, column] = torch.where(amax[:, column] > 0, amax[:, column],
                                      torch.full_like(amax[:, column], widest))
    return (A4_GSCALE / amax[:, 0]).contiguous(), (A4_GSCALE / amax[:, 1]).contiguous()


_A4_SIDECARS = {}


def a4_min_rows():
    """Calls above this row count take the A4 prefill op; the rest the A16 op."""
    import os

    return int(os.environ.get("HY4_MOE_A4_MIN_ROWS") or 64)


class FusedWireScratch:
    """Wire-time scratch shared by every fused op of one rank.

    The layer loop is serial per rank, so one set of call tensors, one byte
    workspace and one scale storage serve every sparse layer in turn. The
    arena is also the priming owner b12x retains per plan
    (`prime_owners` -> `(arena,)`), so 77 plans retain ONE buffer.
    """

    def __init__(self, *, device, max_tokens, hidden, top_k):
        import torch
        from ignition.preparation.scratch import ScratchArena

        self.device, self.max_tokens = device, max_tokens
        self.hidden, self.top_k = hidden, top_k
        self.source = torch.zeros((max_tokens, hidden), dtype=torch.bfloat16, device=device)
        self.ids = torch.zeros((max_tokens, top_k), dtype=torch.int32, device=device)
        self.weights = torch.full((max_tokens, top_k), 1.0 / top_k, dtype=torch.float32,
                                  device=device)
        self.output = torch.zeros_like(self.source)
        self.arena = ScratchArena(device, overlap=False)
        self.nbytes = None
        self.layouts = None
        self.storage = None
        self._pointer = None

    def adopt(self, op):
        """Return (workspace, storage) for a wired op; the first op sizes them."""
        import torch
        nbytes, layouts = op.workspace_nbytes, tuple(op.scale_layouts)
        if self.nbytes is None:
            if self.arena.nbytes < nbytes:
                raise RuntimeError("fused wire arena was not sized by the first plan")
            self.nbytes, self.layouts = nbytes, layouts
            self._pointer = self.arena.tensor.data_ptr()
            self.storage = [torch.empty_strided(shape, stride, dtype=dtype, device=self.device)
                            for shape, stride, dtype in layouts]
        elif (nbytes != self.nbytes or layouts != self.layouts
              or self.arena.tensor.data_ptr() != self._pointer):
            raise ValueError("shared fused wire scratch was sized for a different plan")
        return self.arena.tensor.narrow(0, 0, self.nbytes), self.storage

    def seal(self):
        """Forbid growth once every layer is wired."""
        if self.nbytes is None:
            raise RuntimeError("no fused op adopted the shared wire scratch")
        self.arena.freeze()


class FusedRoutedExperts:
    """The fused GPU routed-experts execution (`MoEBlock._prepared` seam).

    One `ignition.components.moe.PreparedRoutedExperts` over the loader's
    packed experts (`Hy4WeightLoader.load_resident_experts` -> `load_fp4_experts`,
    the b12x `fused_moe` W4A16 kernel path — NOT the torch reference loop),
    prepared at `max_tokens` rows and called per forward at ANY row count at or
    under that capacity. The call-shaped output is the contract difference
    from the component's functional `__call__` (which allocates the PREPARED
    layout): `prepared_moe_out` validates the output against the CALL's rows
    (`output must have shape (rows, hidden)`), so the wrapper allocates the
    caller's shape and keeps one persistent byte workspace + scale storage
    across calls (the layer loop is serial per rank).

    Call signature matches the `MoEBlock._routed_forward` seam exactly:
    `_prepared(x, ids=topk_indices, weights=topk_weights)`. Route IDs arrive
    as the torch router's int64 indices and are narrowed to the kernel's int32
    contract; route weights are FP32 `sigmoid + bias` renormalised weights.

    Construction is two-phase: `load`/`__init__` take the prepared expert
    planes from the loader session; `wire` builds the b12x plan and binding
    state at `wire_operators` time (plans bind there — there is no JIT at run
    time). Calling before `wire` raises instead of planning inside a step.
    """

    def __init__(self, experts, *, hidden, top_k, device, name, a4_prefill=False):
        if hidden < 1 or top_k < 1:
            raise ValueError("fused routed experts require positive geometry")
        self.experts = experts          # `fused_moe.PreparedExperts` (loader)
        # consult18 E2: experts prepared with calibrated A4 capability get two
        # ops (per-mode wiring): `op` (A4, prefill rows) and `small_op` (A16,
        # decode and calls of at most `a4_min_rows()` rows).
        self.a4_prefill = bool(a4_prefill)
        self.small_op = None
        self.small_tokens = 0
        self._small_workspace = self._small_storage = None
        self.amax = None                # calibration boot: [1, E, 2] amax buffer
        self.hidden = hidden
        self.max_tokens = None          # set at `wire` from the call-shape catalogue
        self.top_k = top_k
        self.device = device
        self.name = name
        self.op = None                  # `PreparedRoutedExperts` after `wire`
        self._workspace = None          # persistent byte scratch (serial layers)
        self._storage = None            # persistent scale storage views
        self._lane_scratch = {}         # consult18 D: lane -> (workspace, storage)
        self.calls = 0

    @property
    def wired(self):
        return self.op is not None

    def _wire_small(self, preparation, small_scratch):
        """consult18 E2: the A16 op for decode and short calls over the same
        A4-capable experts (an A4-prepared op forces A4 on every bind, and A4
        has prefill-size launches only)."""
        import torch
        from ignition.components.moe import PreparedRoutedExperts

        self.small_tokens = min(a4_min_rows(), self.max_tokens)
        if small_scratch is not None:
            if ((small_scratch.max_tokens, small_scratch.hidden, small_scratch.top_k)
                    != (self.small_tokens, self.hidden, self.top_k)):
                raise ValueError(f"{self.name}: shared small scratch geometry differs from this op")
            self.small_op = PreparedRoutedExperts(
                small_scratch.source, self.experts, small_scratch.ids, small_scratch.weights,
                small_scratch.output, session=preparation, name=f"{self.name}.a16",
                workspace=small_scratch.arena, a4_prefill=False)
            self.small_op.functional(session=preparation)
            self._small_workspace, self._small_storage = small_scratch.adopt(self.small_op)
            return
        source = torch.zeros((self.small_tokens, self.hidden), dtype=torch.bfloat16,
                             device=self.device)
        ids = torch.zeros((self.small_tokens, self.top_k), dtype=torch.int32, device=self.device)
        weights = torch.full((self.small_tokens, self.top_k), 1.0 / self.top_k,
                             dtype=torch.float32, device=self.device)
        self.small_op = PreparedRoutedExperts(source, self.experts, ids, weights,
                                              torch.zeros_like(source), session=preparation,
                                              name=f"{self.name}.a16", a4_prefill=False)
        self.small_op.functional(session=preparation)
        self._small_workspace = torch.empty(self.small_op.workspace_nbytes, dtype=torch.uint8,
                                            device=self.device)
        self._small_storage = [torch.empty_strided(shape, stride, dtype=dtype, device=self.device)
                               for shape, stride, dtype in self.small_op.scale_layouts]

    def wire(self, *, session, max_tokens, scratch=None, small_scratch=None):
        """Build the prepared operator and its persistent scratch (idempotent)."""
        if self.op is not None:
            return self
        if type(max_tokens) is not int or max_tokens < 1:
            raise ValueError("fused routed experts need a positive max_tokens call capacity")
        self.max_tokens = max_tokens
        import torch
        from b12x.preparation import PreparationSession
        from ignition.components.moe import PreparedRoutedExperts

        if isinstance(session, PreparationSession):
            preparation = session
        else:
            # The recipe's `wire_operators(session="prepared")` sentinel: one
            # long-lived session owning the MoE plans for the rank's lifetime.
            preparation = PreparationSession(device=self.device, autotune=False,
                                             compile_workers=0)
            self._session = preparation
        # consult18 (determinism): the prefill forward's run-to-run divergence
        # starts at the routed MoE (decoder.PROBE: identical input, different
        # output bits, first sparse layer, every rank). HY4_MOE_DETERMINISTIC=1
        # declares b12x's deterministic route reduction (RoutingSpec
        # deterministic_output) so the same input gives the same bits.
        import os as _os

        from b12x.moe import fused_moe as _fused_moe

        spec = {}
        if (_os.environ.get("HY4_MOE_DETERMINISTIC") or "") not in ("", "0"):
            spec["deterministic_output"] = True
        if MoeCalibration.enabled():
            if self.a4_prefill:
                raise ValueError(f"{self.name}: HY4_MOE_CALIBRATE and HY4_MOE_A4 are exclusive")
            spec["collect_activation_amax"] = True
            self.amax = CALIBRATION.buffer(self.name, self.experts.num_experts, self.device)
        routing = {"routing": _fused_moe.RoutingSpec(**spec)} if spec else {}
        if self.a4_prefill:
            routing["a4_prefill"] = True
        if scratch is not None:
            if ((scratch.max_tokens, scratch.hidden, scratch.top_k)
                    != (self.max_tokens, self.hidden, self.top_k)
                    or torch.device(scratch.device) != torch.device(self.device)):
                raise ValueError(f"{self.name}: shared wire scratch geometry differs from this op")
            self.op = PreparedRoutedExperts(
                scratch.source, self.experts, scratch.ids, scratch.weights, scratch.output,
                session=preparation, name=self.name, workspace=scratch.arena, **routing)
            self.op.functional(session=preparation)
            self._workspace, self._storage = scratch.adopt(self.op)
            if self.a4_prefill:
                self._wire_small(preparation, small_scratch)
            return self
        source = torch.zeros((self.max_tokens, self.hidden), dtype=torch.bfloat16,
                             device=self.device)
        ids = torch.zeros((self.max_tokens, self.top_k), dtype=torch.int32,
                          device=self.device)
        weights = torch.full((self.max_tokens, self.top_k), 1.0 / self.top_k,
                             dtype=torch.float32, device=self.device)
        output = torch.zeros_like(source)
        self.op = PreparedRoutedExperts(source, self.experts, ids, weights, output,
                                        session=preparation, name=self.name, **routing)
        self.op.functional(session=preparation)
        self._workspace = torch.empty(self.op.workspace_nbytes, dtype=torch.uint8,
                                      device=self.device)
        self._storage = [torch.empty_strided(shape, stride, dtype=dtype,
                                             device=self.device)
                         for shape, stride, dtype in self.op.scale_layouts]
        if self.a4_prefill:
            self._wire_small(preparation, small_scratch)
        return self

    def plan_decode_op(self, *, tokens):
        """consult18 P2 (numerics class): plan one more A16 op at a decode
        capacity (`tokens` rows) — call under the b12x env overrides the op
        should compile with (wire_decode_ops sets them). Selected per call by
        runtime switch `moe_decode_op` for calls of at most `tokens` rows on
        lane 0; its workspace/storage are bound by `bind_decode_scratch`."""
        import torch
        from b12x.preparation import PreparationSession
        from ignition.components.moe import PreparedRoutedExperts

        if self.op is None:
            raise RuntimeError(f"{self.name}: wire the main op first")
        session = PreparationSession(device=self.device, autotune=False, compile_workers=0)
        source = torch.zeros((tokens, self.hidden), dtype=torch.bfloat16, device=self.device)
        ids = torch.zeros((tokens, self.top_k), dtype=torch.int32, device=self.device)
        weights = torch.full((tokens, self.top_k), 1.0 / self.top_k, dtype=torch.float32,
                             device=self.device)
        op = PreparedRoutedExperts(source, self.experts, ids, weights, torch.zeros_like(source),
                                   session=session, name=f"{self.name}.decode", a4_prefill=False)
        op.functional(session=session)
        self._decode_session = session
        self.decode_op = op
        self.decode_tokens = int(tokens)
        # consult18 P4: one op per decode capacity (1 row; the k+1 verify rows)
        if not hasattr(self, "decode_ops"):
            self.decode_ops = {}
            self._decode_sessions = []
        self.decode_ops[int(tokens)] = op
        self._decode_sessions.append(session)
        return op

    def bind_decode_scratch(self, workspace, storage):
        self._decode_workspace, self._decode_storage = workspace, storage


    def warm(self):
        """Run one call at the extremes so no plan preparation lands in a step."""
        import torch
        warm_rows = ((1, self.small_tokens, self.small_tokens + 1, self.max_tokens)
                     if self.small_op is not None else (1, self.max_tokens))
        for rows in sorted(set(r for r in warm_rows if 0 < r <= self.max_tokens)):
            x = torch.zeros((rows, self.hidden), dtype=torch.bfloat16, device=self.device)
            ids = torch.zeros((rows, self.top_k), dtype=torch.int32, device=self.device)
            weights = torch.full((rows, self.top_k), 1.0 / self.top_k,
                                 dtype=torch.float32, device=self.device)
            self.forward(x, ids, weights)
        self.calls = 0
        return self

    def forward(self, x, topk_indices, topk_weights):
        """The fused experts' weighted mixture over the routed tokens."""
        import torch
        from b12x.moe.fused_moe._binding_ops import prepared_moe_out

        if self.op is None:
            raise RuntimeError(
                f"{self.name}: fused routed experts called before wire_operators "
                "bound the prepared plan — there is no JIT at run time")
        rows = x.shape[0]
        if rows > self.max_tokens:
            raise ValueError(
                f"{self.name}: fused routed experts see {rows} rows above the "
                f"prepared capacity {self.max_tokens}")
        if x.shape[1] != self.hidden or x.dtype != torch.bfloat16:
            raise ValueError(f"{self.name}: routed input must be BF16 [rows, {self.hidden}]")
        ids = topk_indices.to(torch.int32).contiguous()
        weights = topk_weights.float().contiguous()
        output = torch.zeros((rows, self.hidden), dtype=x.dtype, device=x.device)
        from .lanes import lane

        current = lane()
        op = self.op
        workspace, storage = ((self._workspace, self._storage) if current == 0
                              else self._lane_scratch[current])
        a4 = self.op.precision.get("a4_prefill")
        if self.small_op is not None and rows <= self.small_tokens:
            if current:
                raise RuntimeError(f"{self.name}: lane {current} has no small-call A16 scratch")
            op, workspace, storage = self.small_op, self._small_workspace, self._small_storage
            a4 = False
        decode_op = getattr(self, "decode_op", None)
        caps = [cap for cap in getattr(self, "decode_ops", {}) if rows <= cap]
        if decode_op is not None and caps and current == 0:
            from .runtime import SWITCHES

            mode = SWITCHES.get("moe_decode_op")
            if mode == "on":
                # consult18 P2 (numerics class): the decode-capacity A16 op built
                # under HY4_MOE_DECODE_ENV (b12x small-M split-K schedule), the
                # smallest capacity that holds the call (P4: 1 row, verify rows).
                op, workspace, storage = (self.decode_ops[min(caps)], self._decode_workspace,
                                          self._decode_storage)
                a4 = False
            elif mode == "alt" and getattr(self, "decode_op_alt", None) is not None:
                # the control arm: same capacity, built under HY4_MOE_DECODE_ENV_ALT
                op, workspace, storage = (self.decode_op_alt, self._decode_workspace,
                                          self._decode_storage)
                a4 = False
        amax, layer = (self.amax, 0) if self.amax is not None else (None, None)
        prepared_moe_out(x, weights, ids, output, workspace, op.scale_inputs,
                         storage, op.plan.handle, True, None, amax, layer,
                         None, None, False, a4)
        self.calls += 1
        return output

    def __call__(self, x, *, ids, weights):
        return self.forward(x, ids, weights)


def wire_decode_ops(routed, *, tokens, env, alt_env=None):
    """Plan every layer's decode op under `env` (set only while planning and
    warming: the main ops keep their plans), over ONE shared workspace and
    scale storage (the layer loop is serial). Returns the workspace bytes."""
    import os as _os

    import torch

    from .runtime import SWITCHES

    saved = {key: _os.environ.get(key) for key in (*env, *(alt_env or {}))}
    _os.environ.update(env)
    capacities = sorted({int(t) for t in (tokens if isinstance(tokens, (tuple, list)) else (tokens,))},
                        reverse=True)
    tokens = capacities[-1]            # the smallest: the one-row decode op
    try:
        ops = [fused.plan_decode_op(tokens=cap) for cap in capacities for fused in routed]
        if alt_env is not None:
            for key in env:
                _os.environ.pop(key, None)
            _os.environ.update(alt_env)
            for fused in routed:
                main = fused.decode_op
                fused.decode_op_alt = fused.plan_decode_op(tokens=tokens)
                fused.decode_ops[tokens] = main
                fused.decode_op = main
            ops += [fused.decode_op_alt for fused in routed]
            for key in alt_env:
                _os.environ.pop(key, None)
            _os.environ.update(env)
        nbytes = max(op.workspace_nbytes for op in ops)
        layouts = ops[0].scale_layouts
        if any(op.scale_layouts != layouts for op in ops):
            raise ValueError("the decode MoE ops disagree on their scale layouts")
        device = routed[0].device
        workspace = torch.empty(nbytes, dtype=torch.uint8, device=device)
        storage = [torch.empty_strided(shape, stride, dtype=dtype, device=device)
                   for shape, stride, dtype in layouts]
        before = SWITCHES.get("moe_decode_op")
        try:
            for fused in routed:
                fused.bind_decode_scratch(workspace, storage)
                x = torch.zeros((1, fused.hidden), dtype=torch.bfloat16, device=device)
                ids = torch.zeros((1, fused.top_k), dtype=torch.int32, device=device)
                w = torch.full((1, fused.top_k), 1.0 / fused.top_k, dtype=torch.float32, device=device)
                SWITCHES["moe_decode_op"] = "on"
                fused.forward(x, ids, w)
                if alt_env is not None:
                    for key in env:
                        _os.environ.pop(key, None)
                    _os.environ.update(alt_env)
                    SWITCHES["moe_decode_op"] = "alt"
                    fused.forward(x, ids, w)
                    for key in alt_env:
                        _os.environ.pop(key, None)
                    _os.environ.update(env)
        finally:
            SWITCHES["moe_decode_op"] = before
        torch.cuda.synchronize(device)
        return nbytes
    finally:
        for key, value in saved.items():
            if value is None:
                _os.environ.pop(key, None)
            else:
                _os.environ[key] = value


@dataclass
class MoEBlock:
    """One sparse layer's MoE: router + routed experts + shared expert."""

    router: RouterWeights
    shared: DenseMlpWeights
    routed: object  # `PreparedRoutedExperts` handle on GPU, ReferenceRoutedExperts on CPU
    config: ModelConfig
    device: torch.device
    _prepared: object = None  # GPU wiring (PreparedRouter + PreparedRoutedExperts)
    packed_input: object = None    # `PackedMoEInputProjections` (packed at load)
    overlap_applied: bool = False  # did the shared expert overlap the routed experts?

    def _trim(self):
        from .runtime import SWITCHES

        return self.device.type == "cuda" and SWITCHES.get("host_trim") == "on"

    def _router_f32_t(self, packed):
        """`packed.rows("router").float().T`, cached under host_trim (the same
        values and layout, so the same GEMM)."""
        if not self._trim():
            return packed.rows("router").float().T
        cached = getattr(self, "_router_f32_cache", None)
        if cached is None:
            cached = self._router_f32_cache = packed.rows("router").float().T
        return cached

    def _bias_f32(self):
        if not self._trim():
            return self.router.correction_bias.float()
        cached = getattr(self, "_bias_f32_cache", None)
        if cached is None:
            cached = self._bias_f32_cache = self.router.correction_bias.float()
        return cached

    def route(self, x):
        """(router_logits, topk_weights, topk_indices): sigmoid + bias, top-k, renorm, scale."""
        if self._trim():
            cached = getattr(self, "_gate_f32_cache", None)
            if cached is None:
                cached = self._gate_f32_cache = self.router.gate.float().T
            return self._route_from_logits(x.float() @ cached)
        logits = x.float() @ self.router.gate.float().T
        return self._route_from_logits(logits)

    def _route_from_logits(self, logits):
        c = self.config.text
        experts, top_k = self.router.experts, c["num_experts_per_tok"]
        if logits.is_cuda and c["n_group"] == 1 and top_k == 8:
            from .runtime import SWITCHES

            if SWITCHES.get("route_fused") == "on":
                # consult18 P2: one launch (fused_ops.moe_route, bit-identical)
                from .fused_ops import moe_route

                weights, ids = moe_route(logits, self._bias_f32(), top_k=top_k,
                                         scale=c["routed_scaling_factor"],
                                         norm=c["norm_topk_prob"])
                return logits, weights, ids
        scores = torch.sigmoid(logits)
        choice = scores + self._bias_f32()
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

    def _shared_forward(self, x, shared_gate=None, shared_up=None):
        """The shared expert (unclamped SwiGLU), optionally from packed projections."""
        if shared_gate is None or shared_up is None:
            return self.shared.forward(x)
        return (torch.nn.functional.silu(shared_gate) * shared_up) @ self.shared.down_proj.T

    def _routed_forward(self, x, topk_indices, topk_weights):
        if self.device.type == "cuda" and self._prepared is not None:
            from .lanes import require_exclusive

            require_exclusive("W4A16 routed experts")
            return self._prepared(x, ids=topk_indices, weights=topk_weights)
        return self.routed.forward(x, topk_indices, topk_weights)

    def forward(self, x, *, overlap_shared=False):
        """Routed mixture + shared expert (`HYV4MoE.forward`).

        The consult7 §5 bake-in "shared expert overlapped with routed
        experts": with `overlap_shared` (decode graphs — a per-graph-family
        property) the shared expert runs on a side stream while the routed
        experts execute, joined before the add. Prefill graphs pass
        `overlap_shared=False` (consult8 §6: overlap is a property of the
        graph family). On CPU the two run sequentially with
        `overlap_applied` recording the placement decision. Router + shared
        `gate`/`up` read `x` once through the load-time packed projection
        when present.
        """
        packed = self.packed_input
        from .decoder import PROBE

        if packed is not None:
            # One packed weight at load; per-projection GEMMs over its rows —
            # bitwise the same as the loose weights (content gates are exact).
            with TIMER.phase("moe.route_shared"):
                logits = x.float() @ self._router_f32_t(packed)
                from .w8 import in_decode_scope

                g0, g1 = packed.slices["shared_gate"]
                u0, u1 = packed.slices["shared_up"]
                if g1 == u0 and in_decode_scope(x):
                    # consult18 P7: decode-family rows run shared gate + up as ONE
                    # GEMM over the adjacent packed rows (cuBLAS picks a slow
                    # kernel for each 256-row slice at one row: 8.7 vs 3.1 us).
                    gate_up = x @ packed.packed[g0:u1].T
                    shared_gate, shared_up = gate_up[:, :g1 - g0], gate_up[:, g1 - g0:]
                else:
                    shared_gate = x @ packed.rows("shared_gate").T
                    shared_up = x @ packed.rows("shared_up").T
                _, topk_weights, topk_indices = self._route_from_logits(logits)
        else:
            shared_gate = shared_up = None
            with TIMER.phase("moe.route_shared"):
                logits, topk_weights, topk_indices = self.route(x)
        PROBE.add("moe.logits", -2, logits)
        PROBE.add("moe.topk_ids", -2, topk_indices)
        if PROBE.records is not None or not self._trim():
            PROBE.add("moe.topk_ids_sorted", -2, topk_indices.sort(dim=-1).values)
        PROBE.add("moe.topk_weights", -2, topk_weights)
        overlap = bool(overlap_shared) and self.device.type == "cuda"
        self.overlap_applied = bool(overlap_shared)
        if overlap:
            # W29: the phase's entry/exit synchronisation serialises the two
            # streams while profiling (the documented caveat — it hides
            # overlap); with the timer off this branch is the exact old code.
            side = getattr(self, "_shared_stream", None)
            if side is None:
                side = torch.cuda.Stream()
                self._shared_stream = side
            joined = torch.cuda.Event()
            main = torch.cuda.current_stream()
            with TIMER.phase("moe.route_shared"):
                # consult18 stage 0: the side stream must not start before the
                # main stream has produced `x` and the packed shared gate/up
                # (and finished the previous layer's `routed + shared` add,
                # whose side-allocated operand the allocator may hand back to
                # this stream). Without this wait the side stream reads
                # half-written inputs — the decode-only token-0 collapse,
                # which every synchronising instrument (finite watch, W29
                # timer) hid. The bracketing (wait here, join below) also
                # orders every cross-stream reuse, so no record_stream is needed.
                side.wait_stream(main)
                with torch.cuda.stream(side):
                    shared = self._shared_forward(x, shared_gate, shared_up)
                joined.record(side)
            with TIMER.phase("moe.routed"):
                routed = self._routed_forward(x, topk_indices, topk_weights)
                main.wait_event(joined)
                return routed + shared
        from .lanes import exclusive

        with TIMER.phase("moe.routed"):
            # consult18 S1: the W4A16 routed experts are ONE cooperative grid
            # (whole-grid admission, software all-CTA barriers). A PENDING
            # cooperative grid holds back every later kernel on the card, any
            # stream (coop_block.py: a 1-thread peer wait resident -> the grid
            # cannot be admitted -> a flag-set on a third stream never runs).
            # Beside the other lane's peer-waiting comm that is a cross-rank
            # cycle: each rank's wait needs a peer's send, held back by that
            # peer's pending grid, itself waiting for that peer's resident wait.
            # The comm token's section makes the grid exclusive with every
            # collective on this rank, in the same host order on every rank.
            from .hangdump import active_tracer

            tracer = active_tracer()
            with exclusive():
                if tracer is not None:
                    tracer.mark(f"moe.routed begin rows={int(x.shape[0])}")
                routed = self._routed_forward(x, topk_indices, topk_weights)
                if tracer is not None:
                    tracer.mark("moe.routed end")
        PROBE.add("moe.routed", -2, routed)
        with TIMER.phase("moe.route_shared"):
            shared = self._shared_forward(x, shared_gate, shared_up)
        PROBE.add("moe.shared", -2, shared)
        return routed + shared


class DenseBlock:
    """The dense layer-0 FFN (`HYV4MLP`): unclamped SwiGLU."""

    def __init__(self, weights: DenseMlpWeights, device):
        self.weights, self.device = weights, device

    def forward(self, x, *, overlap_shared=False):
        # No shared/routed split here — the overlap flag is a no-op.
        with TIMER.phase("moe.routed"):
            return self.weights.forward(x)
