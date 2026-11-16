# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Per-rank Hy4 construction and preparation.

`construct_rank` builds one rank end to end on any device: shard assignment,
the 78-layer body (21 full-indexer / 57 shared-selection layers), the 4-stream
iHC state, gated sparse-MLA attention with sinks, the W4A16 routed MoE and the
MTP draft — wired to the b12x operators' contracts (`norm.mhc.ihc`,
`sparse_mla` sink provenance, `dsa_indexer` / `IndexCache` shared selection).
It runs on CPU against the reference package's `tiny-hy4`
(`weights.cpu_dry_run`) and on GPU against the real checkpoint.

`prepare_rank` is the GPU preparation entry `recipe.prepare` drives: it
constructs, then stops at the kernel-tuning / CUDA-graph-capture / native-replay
boundary, which needs GPU hardware and the P5 full-shape window (explicitly
marked below). Everything before that boundary is real code, exercised by the
CPU dry run and the recipe tests.
"""

from dataclasses import dataclass
from pathlib import Path

from ignition.components.parallel import TensorParallelShard

from .config import PAGE_TOKENS, ModelConfig
from .shard import Hy4Sharding


@dataclass(frozen=True)
class PreparationConfig:
    collective_tuning: str = "static"
    collective_ladder: dict | None = None
    prompt_logprobs: bool = False
    compile_workers: int | None = None
    native_dense_activations: bool = False
    lock_gpu_clocks: bool = False
    autotune_effort: str = "default"
    tuning_replica: object | None = None
    # Single-host ranks; 1 is the tiny-parity entry, 8 and 16 serve.
    devices: tuple[int, ...] = (0,)
    # Each rank's host when ranks run on separate hosts; devices are then host-local ordinals.
    hosts: tuple[str, ...] | None = None
    context: int = 1048576
    prefill_chunk_size: int = 4096
    max_active: int = 32
    pages: int | None = None
    # Without fixed pages, KV pages fill the memory measured free after graph capture, at least this many.
    minimum_pages: int | None = None
    snapshot_budget_bytes: int = 4 << 30
    headroom_bytes: int = 4 << 30
    compile_decoder: bool = True
    retain_graphs: bool = False
    kernel_cache: Path | None = None
    mtp: bool = False
    stream_overlap: bool = True
    # Store the routed experts' block scales in the lossless `csf` encoding.
    expert_scale_compression: str | None = None
    # One native MTP predictor with three draft tokens: widths up to four.
    speculative_widths: tuple[int, ...] = (1, 2, 3, 4)
    speculative_fixed_width: int | None = None
    speculative_token_capacity: int | None = None
    mtp_only: bool = False

    def __post_init__(self):
        if self.expert_scale_compression not in (None, "csf"):
            raise ValueError("Hy4 expert scale compression must be None or 'csf'")
        if self.mtp_only and (not self.mtp
                or max(self.speculative_widths, default=1) <= 1 or self.speculative_fixed_width == 1):
            raise ValueError("--mtp-only requires enabled learned speculation with positive draft depth")
        if self.speculative_widths and max(self.speculative_widths) > 4:
            raise ValueError("Hy4 verification storage supports at most three draft tokens")
        if type(self.mtp) is not bool:
            raise TypeError("mtp must be a boolean")
        self.speculative_shapes
        if self.speculative_fixed_width is not None and (
                type(self.speculative_fixed_width) is not int
                or self.speculative_fixed_width not in self.speculative_widths):
            raise ValueError("fixed speculative width must belong to the configured widths")
        if type(self.compile_decoder) is not bool or type(self.stream_overlap) is not bool:
            raise TypeError("compile_decoder and stream_overlap must be booleans")
        placements = tuple(zip(self.hosts or ("",) * len(self.devices), self.devices))
        if (not isinstance(self.devices, tuple) or len(self.devices) not in (1, 8, 16)
                or len(set(placements)) != len(self.devices)
                or self.hosts is not None and (not isinstance(self.hosts, tuple) or len(self.hosts) != len(self.devices))
                or any(type(d) is not int or d < 0 for d in self.devices)):
            raise ValueError("Hy4 requires one, eight or sixteen distinct CUDA devices")
        for name in ("context", "prefill_chunk_size", "max_active", "snapshot_budget_bytes", "headroom_bytes"):
            if type(getattr(self, name)) is not int:
                raise TypeError(f"{name} must be an integer")
        if not PAGE_TOKENS <= self.context <= 1048576 or self.context % PAGE_TOKENS:
            raise ValueError(f"Hy4 context must be {PAGE_TOKENS}-aligned in [{PAGE_TOKENS},1048576]")
        if not 64 <= self.prefill_chunk_size <= self.context or self.prefill_chunk_size % 64:
            raise ValueError("prefill chunk size must be 64-aligned in [64,context]")
        from ignition.components.capacity import request_capacity

        request_capacity(self.max_active)
        if self.pages is not None and (type(self.pages) is not int or not 0 < self.pages <= 2147483647):
            raise ValueError("page capacity must be a positive signed 32-bit integer")
        if self.minimum_pages is not None and (type(self.minimum_pages) is not int or self.minimum_pages <= 0
                                               or self.pages is not None and self.minimum_pages > self.pages):
            raise ValueError("minimum page capacity must be positive and at most the fixed page capacity")
        if self.snapshot_budget_bytes <= 0 or self.headroom_bytes < 0:
            raise ValueError("snapshot budget must be positive and headroom nonnegative")

    @property
    def speculative_shapes(self):
        from ignition.components.capacity import speculative_shapes

        return speculative_shapes(self.max_active, self.speculative_widths, self.speculative_token_capacity)

    @property
    def request_capacity(self):
        from ignition.components.capacity import request_capacity

        return request_capacity(self.max_active)

    @property
    def page_tokens(self):
        """Global positions per KV page: 128 local records on each rank that shares the context."""
        return PAGE_TOKENS * len(self.devices)


def selection_cells(config, device, *, max_q_rows, index_pages_per_layer=1):
    """The shared-selection `IndexCache` and its per-layer cells.

    21 `full` cells (0, 1, 5, ..., 77) own `[max_q_rows, topk]` selection
    buffers and index-key pages; the 57 `shared` cells resolve to their owner's
    buffers by `data_ptr`; the MTP block's own indexer is one more `full` cell
    at layer `config.layers`. Under context parallelism the cross-rank
    selection exchange then runs 21 times per step, not 78
    (`IndexCache.selection_exchange_count`).
    """
    from b12x.attention.dsa_indexer.index_cache import IndexCache, IndexerLayer, hy4_indexer_layers

    from .attention import SelectionCell

    layers = list(hy4_indexer_layers(config.layers, full_layer_ids=config.full_indexer_layers))
    if config.draft_layers:
        layers.append(IndexerLayer(config.layers, "full", config.layers))
    cache = IndexCache(tuple(layers), device=device, topk=config.text["index_topk"],
                       max_q_rows=max_q_rows, index_pages_per_layer=index_pages_per_layer)
    cells = {layer.layer_id: SelectionCell(layer.layer_id, layer.kind, layer.owner_id, cache)
             for layer in layers}
    return cache, cells


@dataclass
class PreparedHy4Rank:
    """One constructed rank: state, decoder entry points, and operator wiring."""

    snapshot: Path
    settings: "PreparationConfig"
    model_config: object
    sharding: object
    device: object
    weights: object
    decoder: object
    cache: object
    index_cache: object
    selections: dict
    memory_report: dict
    fused_session: object = None
    fused_scratch: object = None

    def wire_operators(self, *, session=None):
        """Bind the b12x operator plans this rank's forwards execute.

        GPU-only (b12x plans lower through `PreparationSession` and Triton
        warmup): the iHC `norm.mhc.ihc` TuningContract per boundary
        (`B12XIHCCaps` -> `ihc.plan`), the sparse-MLA plans with their sinks
        (`sparse_mla.plan`, `has_attention_sink=True`), and the full layers'
        `dsa_indexer` plans in physical slot space (`IndexCache.physical_caps`)
        with `PreparedSelectionHandback` pinned for the CP hand-back. On CPU
        the same contracts execute through the torch-op references the
        operators are tuned against (`state.IhcBoundary`, `attention`).
        """
        if session is None:
            return self  # the CPU dry run keeps the torch-op wiring
        import torch

        from b12x.norm.mhc import ihc

        c = self.model_config.text
        caps = ihc.Caps(device=self.device,
                        max_tokens=max(self.settings.prefill_chunk_size, self.settings.request_capacity),
                        hidden_size=self.model_config.hidden)
        # A plan pins ONE invocation, so `pre` and the fused `post_pre` bind
        # separate plans.  Every boundary shares the config-level magnitudes
        # and epsilons, so one plan per operation serves all of them (the
        # single-op `post` entry keeps the operator's lazy default).
        shared_plans = {}
        # The `post_pre` plan pins the folded RMSNorm operand's dtype and the
        # first real forward refuses any drift ("iHC normalization dtype
        # changed"). The layers' norms load in the checkpoint's storage dtype
        # (BF16 in this export — `weights.kept_fp32` never upgrades the
        # layernorms), so declare what the body passes (`components/mhc.py`'s
        # pattern: `str(weights.norm.dtype)`), never an assumed FP32.
        norm_dtypes = {str(t.dtype).removeprefix("torch.")
                       for layer in self.weights.layers
                       for t in (layer.input_norm, layer.post_attention_norm)}
        if len(norm_dtypes) != 1:
            raise ValueError(
                "the shared iHC post_pre plan needs one RMSNorm dtype across "
                f"the layers, got {sorted(norm_dtypes)}")
        norm_weight_dtype = norm_dtypes.pop()

        def plan_for(operation, boundary):
            key = (operation, boundary.magnitude, boundary.hc_eps, boundary.norm_eps)
            if key not in shared_plans:
                invocation = {"operation": operation, "magnitude": boundary.magnitude,
                              "hc_eps": boundary.hc_eps, "norm_eps": boundary.norm_eps}
                if operation == "post_pre":
                    invocation.update({"layernorm_eps": c["rms_norm_eps"], "has_norm_weight": True,
                                       "norm_weight_dtype": norm_weight_dtype})
                shared_plans[key] = ihc.plan(caps, invocation=invocation)
            return shared_plans[key]

        # consult18 (tuned plans): prefill-row plans with the best configs of
        # iHC's own tuning space at the 1,024-row chunk (harness/ihc_sweep.py:
        # post_pre 0.157 -> 0.099 ms, pre 0.131 -> 0.075 ms vs IHC_TUNING[6144]);
        # decode keeps the default plans (no gain at one row).
        from b12x.norm.mhc._ihc_tuning import IhcConfig

        hidden = self.model_config.hidden
        tuned_configs = {
            "post_pre": IhcConfig(hidden_size=hidden, block_k=1024, block_d=512, num_warps=2),
            "pre": IhcConfig(hidden_size=hidden, block_k=512, block_d=512, num_warps=1),
        }
        prefill_caps = ihc.Caps(device=self.device, max_tokens=self.settings.prefill_chunk_size,
                                hidden_size=hidden)
        tuned_plans = {}

        def tuned_for(operation, boundary):
            key = (operation, boundary.magnitude, boundary.hc_eps, boundary.norm_eps)
            if key not in tuned_plans:
                tuned_plans[key] = ihc.plan(prefill_caps, invocation=plan_for(operation, boundary).invocation,
                                            override=tuned_configs[operation])
            return tuned_plans[key]

        for layer in self.weights.layers:
            for boundary in (layer.attn_ihc, layer.ffn_ihc):
                boundary.plan = plan_for("post_pre", boundary)
                boundary.pre_plan = plan_for("pre", boundary)
                boundary.prefill_plan = tuned_for("post_pre", boundary)
                boundary.prefill_pre_plan = tuned_for("pre", boundary)
        for layer_id, cell in self.selections.items():
            if cell.kind == "full":
                # `physical_caps` pins `topk`/`max_q_rows`/`output_index_space`
                # from the IndexCache itself (passing them again raises).
                cell.caps = self.index_cache.physical_caps(
                    layer_id, num_q_heads=self.model_config.text["index_n_heads"],
                    max_page_table_width=self.cache.capacity)
        # M2: the fused routed-experts plans bind here (one prepared b12x op per
        # sparse layer over the loader's packed experts — the levers' basic
        # fused execution, before the consult10 packed-projection/GEMV knobs).
        # Capacity = the widest call the body can feed the MoE (prefill chunks
        # are `prefill_chunk_size` wide; decode/verify feed rows x 1 tokens).
        # `warm` runs the extremes so no plan preparation lands inside a step
        # or a graph capture (KernelPlanBook's no-JIT-at-run-time contract).
        from .moe import FusedRoutedExperts, FusedWireScratch

        capacity = max(self.settings.prefill_chunk_size, self.settings.request_capacity)
        fused_layers = [layer for layer in self.weights.layers
                        if isinstance(getattr(layer.mlp, "routed", None), FusedRoutedExperts)]
        if fused_layers:
            from b12x.preparation import PreparationSession

            # ONE session and ONE scratch per rank: the layer loop is serial.
            # consult18 E1: HY4_MOE_AUTOTUNE=1 races b12x's candidate configs for
            # the routed experts on its realistic routing mix
            # (statistical_shared_mix_v1: diffuse / lognormal-skewed / correlated +
            # whole-batch reuse — not the all-zero priming routes) and persists the
            # selections (HY4_TUNE_CACHE), so only the first boot pays the race.
            import os as _os

            tune = (_os.environ.get("HY4_MOE_AUTOTUNE") or "") not in ("", "0")
            if isinstance(session, PreparationSession):
                preparation = session
            elif tune:
                preparation = PreparationSession(
                    device=self.device, autotune=True,
                    compile_workers=int(_os.environ.get("HY4_TUNE_WORKERS") or 4),
                    cache_dir=_os.environ.get("HY4_TUNE_CACHE") or "/state/b12x-tune")
            else:
                preparation = PreparationSession(device=self.device, autotune=False,
                                                 compile_workers=0)
            first = fused_layers[0].mlp.routed
            scratch = FusedWireScratch(device=self.device, max_tokens=capacity,
                                       hidden=first.hidden, top_k=first.top_k)
            # consult18 E2: A4-capable experts also wire a small A16 op (decode and
            # short calls), all of them over a second shared scratch.
            from .moe import a4_min_rows

            a4_layers = {bool(layer.mlp.routed.a4_prefill) for layer in fused_layers}
            if len(a4_layers) != 1:
                raise ValueError("the routed layers disagree on calibrated A4 prefill")
            small_scratch = (FusedWireScratch(device=self.device,
                                              max_tokens=min(a4_min_rows(), capacity),
                                              hidden=first.hidden, top_k=first.top_k)
                             if a4_layers.pop() else None)
            for layer in fused_layers:
                fused = layer.mlp.routed
                fused.wire(session=preparation, max_tokens=capacity, scratch=scratch,
                           small_scratch=small_scratch)
                fused.warm()
                layer.mlp._prepared = fused
            scratch.seal()
            if small_scratch is not None:
                small_scratch.seal()
            import os as _os

            decode_tokens = int(_os.environ.get("HY4_MOE_DECODE_OP") or 0)
            if decode_tokens > 0 and small_scratch is None:
                # consult18 P2 (numerics class): decode-capacity MoE ops compiled
                # with the b12x small-M split-K schedule, one shared workspace;
                # runtime switch moe_decode_op selects them.
                from .moe import wire_decode_ops

                spec = (_os.environ.get("HY4_MOE_DECODE_ENV")
                        or "B12X_W4A16_SMALL_M_SPLITK=1,B12X_W4A16_SMALL_M_OCCUPANCY=1")
                env = dict(item.split("=", 1) for item in spec.split(",") if item)
                alt_spec = _os.environ.get("HY4_MOE_DECODE_ENV_ALT")
                alt = (dict(item.split("=", 1) for item in alt_spec.split(",") if item)
                       if alt_spec else None)
                wire_decode_ops([layer.mlp.routed for layer in fused_layers],
                                tokens=decode_tokens, env=env, alt_env=alt)
            self.fused_session, self.fused_scratch = preparation, scratch
            self.fused_small_scratch = small_scratch
        return self


def construct_rank(snapshot, config: PreparationConfig, *, rank=0, device=None, pinned=True,
                   mtp=None, cache_rows=None, cache_capacity=None, index_pages_per_layer=1,
                   headers_only=False, manifest=None, world_size=None, collectives=None,
                   fused_cache=False):
    """Construct one Hy4 rank end to end (the CPU-shape/dtype dry run's core).

    Shard assignment, name-checked weight loading (the do-not-quantize floor
    enforced), the iHC/sink/shared-selection/gated-MLA wiring and a working
    `Hy4Decoder` with its `prefill` / `decode_step` / `draft_absorb` /
    `draft_step` entries.
    `device` defaults to CPU; pass a CUDA device for the real checkpoint
    (`pinned=True`). Every checkpoint tensor is consumed exactly once — an
    unmatched or leftover name raises instead of being dropped.

    `world_size` overrides the serving world (`len(config.devices)`, the 1/8/16
    declaration) — the W6 CPU simulation builds 2- and 8-rank worlds from the
    same constructor. A world > 1 needs `collectives` (the injected
    `hy4/collectives.Collectives`, e.g. `SimulatedGroup.collectives(rank)`).

    `headers_only=True` is the header dry run (consult3 gate c): construction
    completes from the checkpoint's shard HEADERS — names, shapes, dtypes,
    rank slicing and the dtype floors all run — over meta tensors, never
    reading a payload. `manifest` may supply a pre-inspected header inventory
    (a repacked header view over a package's per-expert headers) and defaults
    to `inspect_indexed_checkpoint(snapshot)`.
    """
    import torch

    from .decoder import Hy4Cache, Hy4Decoder
    from .state import build_model_state
    from .weights import Hy4WeightLoader, cpu_dry_run, headers_dry_run

    snapshot = Path(snapshot).expanduser().resolve()
    model_config = ModelConfig.load(snapshot, pinned=pinned)
    shard = TensorParallelShard(rank, len(config.devices) if world_size is None else world_size)
    sharding = Hy4Sharding(model_config, shard)
    if device is None:
        device = torch.device("cpu")
    device = torch.device(device)
    mtp = config.mtp if mtp is None else mtp
    max_q_rows = max(config.prefill_chunk_size, config.request_capacity)
    index_cache, selections = selection_cells(model_config, device, max_q_rows=max_q_rows,
                                             index_pages_per_layer=index_pages_per_layer)
    if manifest is None:
        from ignition.preparation.checkpoint import inspect_indexed_checkpoint

        manifest = inspect_indexed_checkpoint(snapshot)
    loader_device = torch.device("meta") if headers_only else device
    loader = Hy4WeightLoader(manifest, loader_device, shard=shard, mtp=mtp, pinned=pinned,
                             expert_scale_compression=config.expert_scale_compression)
    if headers_only:
        sources = headers_dry_run(loader)
    elif device.type == "cuda":
        sources = loader
    else:
        sources = cpu_dry_run(loader)
    with sources:
        weights = build_model_state(loader, model_config, sharding, selections=selections, mtp=mtp)
    rows = config.max_active if cache_rows is None else cache_rows
    capacity = max(config.prefill_chunk_size, 128) if cache_capacity is None else cache_capacity
    cache = Hy4Cache.allocate(model_config, rows=rows, capacity=capacity, device=device, draft=mtp,
                              world=shard.world_size, rank=rank, fused_records=fused_cache)
    decoder = Hy4Decoder(weights, cache, collectives=collectives,
                         prefill_chunk_size=config.prefill_chunk_size)
    report = {"rank": rank, "world_size": shard.world_size, "device": str(device),
              "weights_bytes": weights.allocated_bytes, "mtp": mtp,
              "full_indexer_layers": model_config.full_indexer_layers,
              "shared_indexer_layers": len(model_config.shared_indexer_layers),
              "prefill_head_gather": sharding.prefill_head_gather,
              "fused_cache": fused_cache}
    return PreparedHy4Rank(snapshot=snapshot, settings=config, model_config=model_config,
                           sharding=sharding, device=device, weights=weights, decoder=decoder,
                           cache=cache, index_cache=index_cache, selections=selections,
                           memory_report=report)


def capture_kernel_graphs(constructed, shapes=(("decode", (1, 1)), ("prefill", (1, 2048)))):
    """Capture the prepared kernel chain at the measurement shapes; return bytes.

    The capture covers the CUDA-graph-safe b12x operator chain the decode and
    prefill steps execute per layer (the sparse-MLA prepared attend with its
    in-kernel sink fold; the `W_UV` expansion rides the CP body's
    `expand_output` and is not part of this chain) — the graph-pool residency
    that the `Allocate cache` line must subtract. The torch-op terms (MoE
    routing, indexer scoring) are NOT capture-safe today (host-side value
    reads inside the layer loop), so this is a LOWER BOUND on the final graph
    catalog; the report states it as measured-and-bounded, never as a
    placeholder.

    Footprint = the device-free delta across warmup + capture + one replay,
    measured with the caching allocator drained (`empty_cache`) so the number
    is real allocated graph-pool + scratch, not allocator slack.
    """
    import torch

    decoder = constructed.decoder
    device = constructed.device
    attn = constructed.weights.layers[0].attention
    cache = constructed.cache
    topk = constructed.model_config.text["index_topk"]
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    before, _ = torch.cuda.mem_get_info(device)
    captured = {}
    for name, (rows, tokens) in shapes:
        shape = decoder.default_call_shape((rows, tokens))
        q = torch.randn(rows * tokens, attn.heads, attn.kv_lora + attn.qk_rope,
                        device=device, dtype=torch.bfloat16)
        table = cache.latents(0)
        selected = torch.zeros((rows * tokens, topk), dtype=torch.int32, device=device)
        counts = torch.ones((rows * tokens,), dtype=torch.int32, device=device)
        lengths = torch.ones((rows * tokens,), dtype=torch.int32, device=device)
        # eager warmup (JIT/preparation allocations land before capture)
        attn.attend(q, table, selected, counts, cache_lengths=lengths, call_shape=shape,
                    expand=False, fold_sink=True)
        torch.cuda.synchronize(device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            attn.attend(q, table, selected, counts, cache_lengths=lengths, call_shape=shape,
                        expand=False, fold_sink=True)
        graph.replay()
        torch.cuda.synchronize(device)
        captured[name] = graph
    after, _ = torch.cuda.mem_get_info(device)
    decoder.trace.append(("captured_kernel_graphs", tuple(captured), before - after))
    return before - after, captured


def allocate_cache_report(constructed, *, graph_scratch_bytes, world_size=None,
                          headroom_bytes=0, graph_term_complete=False):
    """The helios `Allocate cache` line from the REAL post-capture residue.

    Tokens per rank = (device free after weights + graphs + scratch − headroom)
    / `ModelConfig.token_record_bytes()` — the same contract as w6 §1.3, with
    the 12 GiB graph PLACEHOLDER replaced by the measured `graph_scratch_bytes`.
    Returns `(line, facts)`; `facts` carries the machine figures including the
    1M one-third-cap verdict and the graph budget the cap implies.
    """
    import torch

    device = constructed.device
    cfg = constructed.model_config
    torch.cuda.synchronize(device)
    free, total = torch.cuda.mem_get_info(device)
    record = cfg.token_record_bytes()
    pool = free - headroom_bytes
    tokens = pool // record
    world = world_size if world_size is not None else constructed.sharding.world_size
    aggregate = tokens * world
    cap = aggregate // 3
    # The measured w2 boot facts: 1M aggregate / 3 needs graphs + scratch under
    # (free-after-weights − 3 × 1,048,576 × record) at world 8.
    facts = {
        "device_free_after_graphs_bytes": free,
        "device_total_bytes": total,
        "graph_scratch_bytes": graph_scratch_bytes,
        "graph_term_complete": bool(graph_term_complete),
        "headroom_bytes": headroom_bytes,
        "record_bytes_contract": record,
        "weights_bytes": constructed.memory_report["weights_bytes"],
        "pool_bytes": pool,
        "tokens_per_rank": tokens,
        "tokens_aggregate": aggregate,
        "one_third_cap": cap,
        "one_third_cap_holds_1m": cap >= 1_048_576,
    }
    line = (
        f"Allocate cache: {pool} B KV pool = {tokens} tokens/rank "
        f"(record {record} B; graph+scratch {graph_scratch_bytes} B measured"
        f"{'' if graph_term_complete else ', kernel-chain term only'}; "
        f"aggregate {aggregate}; one-third cap {cap}"
        f"{'>= 1M ✓' if cap >= 1_048_576 else '< 1M ✗'})"
    )
    return line, facts


def prepare_rank(snapshot, config, *, rank, group, progress=None, core_storage=None, checkpoint=None):
    """Load, wire, tune, and capture one Hy4 rank on its supervised GPU.

    Construction and operator wiring are `construct_rank` (tested end to end on
    CPU against `tiny-hy4`). What remains here is genuinely GPU-bound and
    belongs to the P5 full-shape window: the `RecordedPreparationSession`
    kernel tuning over the iHC / sparse-MLA / indexer plans, the graph catalog
    (`capture_forwards` over the prefill/decode/speculative entries), the KV
    page measurement, and the native replay export. They are marked, not
    stubbed over — `recipe.prepare` cannot serve until P5 lands.
    """
    def report(message):
        if progress is not None:
            progress(message)

    import torch

    if torch.device("cuda", config.devices[rank]).type != "cuda" or not torch.cuda.is_available():
        raise ValueError("Hy4 rank preparation runs on CUDA; use construct_rank for the CPU dry run")
    device = torch.device("cuda", config.devices[rank])
    torch.cuda.set_device(device)
    constructed = construct_rank(snapshot, config, rank=rank, device=device, pinned=True,
                                 mtp=config.mtp, fused_cache=True)
    report(f"constructed rank {rank}: {constructed.memory_report['weights_bytes'] >> 20} MiB weights, "
           f"{len(constructed.weights.layers)} layers")
    constructed.wire_operators(session="prepared")  # plans bind in the tuning session below
    # P5 measurement step (the M2 boot line): bind the measurement call
    # shapes, capture the prepared kernel chain's graphs, then report the REAL
    # post-capture token count — the helios `Allocate cache` line with the
    # w6 §1.3 12 GiB graph PLACEHOLDER replaced by the measured graph+scratch
    # footprint. The graph term is the capture-safe kernel chain (sparse-MLA
    # attend + sink fold + fused-record writer); the torch-op catalog terms
    # ride the capture_forwards integration below and are reported as missing,
    # never folded into the number silently.
    shapes = [constructed.decoder.default_call_shape((1, 1)),
              constructed.decoder.default_call_shape((1, min(config.prefill_chunk_size, 2048)))]
    constructed.decoder.bind_plan_book(None, shapes)
    graph_bytes, _ = capture_kernel_graphs(constructed)
    world = group.size() if hasattr(group, "size") else len(config.devices)
    line, facts = allocate_cache_report(constructed, graph_scratch_bytes=graph_bytes,
                                        world_size=world, graph_term_complete=False)
    constructed.memory_report["allocate_cache"] = facts
    report(line)
    # TODO(P5, GPU): RecordedPreparationSession tuning of the iHC 'norm.mhc.ihc'
    # contract, the sparse-MLA sink plans and the dsa_indexer selection plans;
    # the prefill/decode/speculative graph catalog via capture_forwards (its
    # torch-op terms are not capture-safe yet — host-side value reads in the
    # layer loop) and the native replay export. Each needs the full-shape GPU
    # window and the real checkpoint — the CPU dry run proves construction and
    # numerics, not kernel selection.
    raise NotImplementedError(
        "Hy4 rank construction, operator wiring and the post-capture Allocate-cache "
        "report are done (the line above is the real measured number); kernel tuning "
        "catalog, capture_forwards graph integration and native replay export remain "
        "the P5 serve work")
