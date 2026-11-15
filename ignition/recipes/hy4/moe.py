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

    def __init__(self, experts, *, hidden, top_k, device, name):
        if hidden < 1 or top_k < 1:
            raise ValueError("fused routed experts require positive geometry")
        self.experts = experts          # `fused_moe.PreparedExperts` (loader)
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

    def wire(self, *, session, max_tokens, scratch=None):
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
        if scratch is not None:
            if ((scratch.max_tokens, scratch.hidden, scratch.top_k)
                    != (self.max_tokens, self.hidden, self.top_k)
                    or torch.device(scratch.device) != torch.device(self.device)):
                raise ValueError(f"{self.name}: shared wire scratch geometry differs from this op")
            self.op = PreparedRoutedExperts(
                scratch.source, self.experts, scratch.ids, scratch.weights, scratch.output,
                session=preparation, name=self.name, workspace=scratch.arena)
            self.op.functional(session=preparation)
            self._workspace, self._storage = scratch.adopt(self.op)
            return self
        source = torch.zeros((self.max_tokens, self.hidden), dtype=torch.bfloat16,
                             device=self.device)
        ids = torch.zeros((self.max_tokens, self.top_k), dtype=torch.int32,
                          device=self.device)
        weights = torch.full((self.max_tokens, self.top_k), 1.0 / self.top_k,
                             dtype=torch.float32, device=self.device)
        output = torch.zeros_like(source)
        self.op = PreparedRoutedExperts(source, self.experts, ids, weights, output,
                                        session=preparation, name=self.name)
        self.op.functional(session=preparation)
        self._workspace = torch.empty(self.op.workspace_nbytes, dtype=torch.uint8,
                                      device=self.device)
        self._storage = [torch.empty_strided(shape, stride, dtype=dtype,
                                             device=self.device)
                         for shape, stride, dtype in self.op.scale_layouts]
        return self

    def warm(self):
        """Run one call at the extremes so no plan preparation lands in a step."""
        import torch
        for rows in (1, self.max_tokens):
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
        workspace, storage = ((self._workspace, self._storage) if current == 0
                              else self._lane_scratch[current])
        prepared_moe_out(x, weights, ids, output, workspace, self.op.scale_inputs,
                         storage, self.op.plan.handle, True, None, None, None,
                         None, None, False, self.op.precision.get("a4_prefill"))
        self.calls += 1
        return output

    def __call__(self, x, *, ids, weights):
        return self.forward(x, ids, weights)


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

    def route(self, x):
        """(router_logits, topk_weights, topk_indices): sigmoid + bias, top-k, renorm, scale."""
        logits = x.float() @ self.router.gate.float().T
        return self._route_from_logits(logits)

    def _route_from_logits(self, logits):
        c = self.config.text
        experts, top_k = self.router.experts, c["num_experts_per_tok"]
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

    def _shared_forward(self, x, shared_gate=None, shared_up=None):
        """The shared expert (unclamped SwiGLU), optionally from packed projections."""
        if shared_gate is None or shared_up is None:
            return self.shared.forward(x)
        return (torch.nn.functional.silu(shared_gate) * shared_up) @ self.shared.down_proj.T

    def _routed_forward(self, x, topk_indices, topk_weights):
        if self.device.type == "cuda" and self._prepared is not None:
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
        if packed is not None:
            # One packed weight at load; per-projection GEMMs over its rows —
            # bitwise the same as the loose weights (content gates are exact).
            with TIMER.phase("moe.route_shared"):
                logits = (x.float() @ packed.rows("router").float().T)
                shared_gate = x @ packed.rows("shared_gate").T
                shared_up = x @ packed.rows("shared_up").T
                _, topk_weights, topk_indices = self._route_from_logits(logits)
        else:
            shared_gate = shared_up = None
            with TIMER.phase("moe.route_shared"):
                _, topk_weights, topk_indices = self.route(x)
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
        with TIMER.phase("moe.routed"):
            routed = self._routed_forward(x, topk_indices, topk_weights)
        with TIMER.phase("moe.route_shared"):
            shared = self._shared_forward(x, shared_gate, shared_up)
        return routed + shared


class DenseBlock:
    """The dense layer-0 FFN (`HYV4MLP`): unclamped SwiGLU."""

    def __init__(self, weights: DenseMlpWeights, device):
        self.weights, self.device = weights, device

    def forward(self, x, *, overlap_shared=False):
        # No shared/routed split here — the overlap flag is a no-op.
        with TIMER.phase("moe.routed"):
            return self.weights.forward(x)
