"""Prepare MiMo text, DFlash proposals and image encoder graphs for local TP."""

import contextlib
import hashlib
import importlib.util
import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path

import torch

from ignition.components.completion import CompletionOutputs
from ignition.components.draft_sampling import SHARD_PAYLOAD_COLUMNS
from ignition.components.prompt_logprobs import (
    PreparedScoredPrefill, PromptScoring, prefill_graph_count, prefill_modes,
)

from ignition.components.capacity import (
    speculative_token_capacity,
    batch_capacities,
    prefill_capacities,
    request_capacity,
)
from ignition.components.paged_batch import BatchInputs, Metadata, PreparedBatch
from ignition.components.paged_generation import PreparedGeneration
from ignition.components.paged_maintenance import PreparedMaintenance
from ignition.components.paged_resume import PreparedResume
from ignition.components.parallel import TensorParallelShard
from ignition.components.sampling import AdmissionBuffers, DeviceRequestState
from ignition.components.state_transfer import StateCommands
from ignition.preparation.capture import capture_forwards
from .contract import prepare as prepare_contract
from ignition.preparation.contract import history_region
from ignition.preparation import runtime_package_pb2 as contract_proto
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.preparation.collectives import (
    CollectivePreparation,
    RoutedAllReduce,
    create_all_gather_runtime,
    create_all_reduce_runtime,
    create_dma_all_reduce_runtime,
    create_twoshot_all_reduce_runtime,
)
from ignition.preparation.disposable import storage_ranges
from ignition.preparation.execution import GraphSignature
from ignition.preparation.host_memory import release_unused
from ignition.preparation.frontend import prepare_frontend
from ignition.preparation.model import PreparedModel
from ignition.preparation.rank_group import minimum_across_ranks
from ignition.preparation.scratch import GraphBufferPool, ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession
from ignition.preparation.tuning_replicas import TuningReplica
from ignition.progress import phase

from .config import ModelConfig
from .recipe import RECIPE
from .decoder import PreparedDecoder, Scratch
from .dflash import SpeculativePreparation
from .model import DecoderEnvironment
from .pool import PAGE_TOKENS, ContinuationPool, continuation_layout, page_bytes
from .vision import FEATURES, HIDDEN, PATCH_CAPACITIES, PreparedVision, VisionInputs, VisionScratch
from .vision_state import PreparedVisionGeneration
from ignition.components.vision_state import PrefillVisionInputs, VisionState
from .weights import ModelWeights

# Two-shot rows cover every reduction one-shot serves: decode, verification, draft blocks and
# prefill chunks below the 1024-token DMA boundary.
TWOSHOT_MAX_TOKENS = 1024
ONESHOT_MAX_TOKENS = 64
# A4/A16 hybrid by default at the SM120 vLLM reference's threshold, so both engines run the same
# expert path for each chunk size. On one TP8 MiMo expert layer MXFP4 activations measure 0.93x at
# 512, 1.02x at 1024, 1.19x at 2048 and 1.62x at 8192 tokens; 512 is still above the 256-token
# speculative capacity. Prefill graphs from this capacity are captured in both precisions, and
# `flare serve --quantized-activation-min-tokens` picks the threshold at load (0 serves pure A16).
# FLARE_MIMO_A4_PREFILL_MIN_TOKENS=0 exports pure A16 without the A4 variants.
A4_PREFILL_MIN_TOKENS = 512
# With CSF expert scales, b12x's W4A16 kernels read compressed scales inline, skipping the shared
# expansion that the MXFP4 A4 prefill kernel reads, for calls of at most 16 routed rows per expert
# (768 MiMo tokens). A4 variants start at the first prefill capacity past that window.
CSF_A4_PREFILL_MIN_TOKENS = 1024


@dataclass(frozen=True)
class PreparationConfig:
    prompt_logprobs: bool = False
    image_concurrency: int | None = None
    compile_workers: int | None = None
    tuning_replica: TuningReplica | None = None
    devices: tuple[int, ...] = tuple(range(8))
    context: int = 1048576
    prefill_chunk_size: int = 4096
    max_active: int = 4
    snapshot_budget_bytes: int = 4 << 30
    headroom_bytes: int = 4 << 30
    compile_decoder: bool = True
    retain_graphs: bool = False
    kernel_cache: Path | None = None
    dflash: bool = True
    speculative_widths: tuple[int, ...] = tuple(range(1, 9))
    speculative_fixed_width: int | None = None
    speculative_token_capacity: int | None = None
    # Each rank proposes drafts from its vocabulary shard of the draft head and gathers only its
    # bounded candidates; False gathers full-vocabulary draft logits. Proposals are identical.
    vocabulary_parallel_draft: bool = True
    # Speculative rounds of at most this many requests verify in two phases: shard summaries decide
    # on device whether the round's nuclei fit a compact verifier, and the host then launches the
    # compact or the dense continuation. Zero captures every round in one phase.
    two_phase_max_batch: int = field(default_factory=lambda: int(os.environ.get("FLARE_MIMO_TWO_PHASE_MAX_BATCH", "4")))
    # Split speculation: proposal graphs retain each request's drafts with per-position
    # confidence from the drafter's own distributions, and verification graphs check the prefix
    # the runtime chooses per step. Joined graphs keep one width per request history.
    split_speculation: bool = field(default_factory=lambda: os.environ.get("FLARE_MIMO_SPLIT_SPECULATION", "0") == "1")
    # Device-selected width for one-request rounds: a prefix proposes the full draft block and
    # picks the verification width from the drafts' survival against host-supplied width costs;
    # the runtime composes it with a conditional switch over one verification body per width.
    # Larger batches keep joined graphs. Exclusive with split speculation.
    switch_speculation: bool = field(default_factory=lambda: os.environ.get("FLARE_MIMO_SWITCH_SPECULATION", "0") == "1")
    # Sharded draft proposals of a one-request round sample every draft position in one row block
    # (one candidate gather and one support broadcast) instead of one sharded proposal per position.
    # Positions then ignore the block's own earlier drafts in penalty counts; outputs are unchanged.
    draft_position_blocks: bool = field(
        default_factory=lambda: os.environ.get("FLARE_MIMO_DRAFT_POSITION_BLOCKS", "1") == "1")
    # "w8a16" converts the draft's BF16 projections to block-FP8 (128x128 blocks, amax/448
    # scales) at load and runs them as W8A16 GEMVs with BF16 activations, halving draft weight
    # traffic; "bf16" keeps the checkpoint's BF16 projections. Draft batches above 32 rows
    # (more than four requests of eight-token blocks) keep BF16.
    draft_weight_format: str = "w8a16"
    # BF16 activations over MXFP4 experts (W4A16), as the SM120 vLLM reference serves MiMo.
    expert_activation: str = "a16"
    # A4/A16 hybrid: prefill graphs of at least this many tokens are captured twice, with BF16 and
    # with dynamic MXFP4 (E8M0 per-32) activations over the same packed expert weights, and the
    # package serves MXFP4 from this capacity by default. Decode and verification graphs stay
    # below it; zero prepares pure A16 (FLARE_MIMO_A4_PREFILL_MIN_TOKENS=0 at export). None selects
    # the default for the expert scale encoding.
    a4_prefill_min_tokens: int | None = None
    # "csf" keeps routed expert block scales losslessly compressed and expands them per MoE call.
    expert_scale_compression: str | None = None

    @property
    def a4_prefill(self):
        return self.expert_activation == "a16" and self.a4_prefill_min_tokens > 0

    def __post_init__(self):
        from ignition.components.capacity import image_capacity

        object.__setattr__(self, "image_concurrency", image_capacity(self.max_active, self.image_concurrency))
        if self.expert_scale_compression not in (None, "csf"):
            raise ValueError("MiMo expert scale compression must be None or 'csf'")
        if self.a4_prefill_min_tokens is None:
            object.__setattr__(self, "a4_prefill_min_tokens", A4_PREFILL_MIN_TOKENS
                               if self.expert_scale_compression is None else CSF_A4_PREFILL_MIN_TOKENS)
        if (
            len(self.devices) not in (2, 4, 8)
            or len(set(self.devices)) != len(self.devices)
            or any(d < 0 for d in self.devices)
        ):
            raise ValueError("MiMo preparation requires distinct GPUs dividing its eight QKV checkpoint shards")
        if not PAGE_TOKENS <= self.context <= 1048576 or self.context % PAGE_TOKENS:
            raise ValueError("MiMo context must be 128-aligned in [128,1048576]")
        if not 64 <= self.prefill_chunk_size <= self.context or self.prefill_chunk_size % 64:
            raise ValueError("prefill chunk size must be 64-aligned in [64,context]")
        request_capacity(self.max_active)
        if self.draft_weight_format not in ("w8a16", "bf16"):
            raise ValueError("MiMo draft weights are w8a16 or bf16")
        _ = self.speculative_shapes
        if not self.speculative_widths or max(self.speculative_widths) > 8:
            raise ValueError("MiMo DFlash verification supports at most seven drafts")
        if (
            self.speculative_fixed_width is not None
            and self.speculative_fixed_width not in self.speculative_widths
        ):
            raise ValueError("fixed verification width must belong to the prepared widths")
        if self.split_speculation and (not self.dflash or self.draft_tokens < 1
                                       or self.speculative_widths != tuple(range(1, max(self.speculative_widths) + 1))):
            raise ValueError("split speculation requires DFlash drafts over contiguous widths from one")
        if self.snapshot_budget_bytes <= 0 or self.headroom_bytes < 0:
            raise ValueError("snapshot budget must be positive and headroom nonnegative")

    @property
    def request_capacity(self):
        return request_capacity(self.max_active)

    @property
    def prefill_capacities(self):
        return prefill_capacities(self.prefill_chunk_size)

    @property
    def batch_capacities(self):
        return tuple(reversed(batch_capacities(self.max_active)))

    @property
    def speculative_shapes(self):
        from ignition.components.capacity import speculative_shapes

        return speculative_shapes(self.max_active, self.speculative_widths, self.speculative_token_capacity) if self.dflash else ()

    @property
    def decode_tokens(self):
        return max((b * w for b, w in self.speculative_shapes), default=self.request_capacity)

    @property
    def draft_tokens(self):
        return max(self.speculative_widths) - 1 if self.dflash else 0


@torch.no_grad()
def prepare_rank(snapshot, config, *, rank, group, progress=None, package_storage=None):
    if package_storage is None:
        raise ValueError("MiMo preparation requires package storage")
    snapshot = Path(snapshot).resolve()
    shard = TensorParallelShard(rank, len(config.devices))
    device = torch.device("cuda", config.devices[rank])
    torch.cuda.set_device(device)
    with phase(progress, "Inspect checkpoint"):
        manifest = inspect_indexed_checkpoint(snapshot)
    transient = contextlib.ExitStack()
    transient.enter_context(package_storage.transient())
    if config.a4_prefill:
        if config.a4_prefill_min_tokens <= config.decode_tokens:
            raise ValueError("A4 prefill must stay above every decode and verification graph")
        if config.expert_scale_compression is not None:
            text = ModelConfig.load(snapshot).text
            if config.a4_prefill_min_tokens * text["num_experts_per_tok"] <= 16 * text["n_routed_experts"]:
                raise ValueError("A4 prefill over CSF expert scales must route more than 16 rows per expert;"
                                 f" use at least {CSF_A4_PREFILL_MIN_TOKENS} tokens")
        if importlib.util.find_spec("b12x.moe._shared.kernels.w4a16.mxfp4_a4_prefill") is None:
            raise ValueError("A4 prefill requires patches/b12x-mxfp4-a4-prefill.patch in the installed b12x")
        # b12x compiles MXFP4 launches for expert plans of at least this many tokens; each prefill
        # graph's binding then selects its own activation precision.
        os.environ["B12X_W4A16_MXFP4_PREFILL_MIN_TOKENS"] = str(config.a4_prefill_min_tokens)
    else:
        os.environ.pop("B12X_W4A16_MXFP4_PREFILL_MIN_TOKENS", None)
    weights = ModelWeights.load(manifest, device, shard=shard, progress=progress, dflash=config.dflash,
                                expert_activation=config.expert_activation,
                                draft_weight_format=config.draft_weight_format,
                                draft_requests=config.request_capacity,
                                expert_scale_compression=config.expert_scale_compression)
    release_unused()
    sequence = torch.zeros(1, dtype=torch.int64, device=device)
    with phase(progress, "Allocate model state and collectives"):
        layout = continuation_layout(weights.config, shard, dflash=config.dflash)
        snapshots = layout.snapshot_capacity(config.snapshot_budget_bytes)
        if snapshots < 1:
            raise ValueError("snapshot budget must hold a complete MiMo continuation")
        slots, blocks = config.request_capacity, config.context // PAGE_TOKENS
        state_rows = snapshots + slots
        initial_pages = (config.prefill_chunk_size + PAGE_TOKENS - 1) // PAGE_TOKENS + 2 * slots
        bytes_per_page = page_bytes(weights.config, shard)
        total_memory = minimum_across_ranks(torch.cuda.mem_get_info(device)[1], group)
        pool = ContinuationPool.allocate(
            weights.config,
            shard,
            state_rows + 1,
            initial_pages,
            device,
            storage=package_storage,
            maximum_pages=total_memory // bytes_per_page,
            maximum_rows=total_memory // layout.row_bytes,
            backed_slots=min(state_rows + 1, 1 + 2 * slots),
            dflash=config.dflash,
        )
        requests = DeviceRequestState.allocate(device, vocab_size=152576, capacity=slots)
        inputs = BatchInputs.allocate(config.prefill_chunk_size, blocks, device, capacity=slots)
        admission = AdmissionBuffers.allocate(device, vocab_size=152576, capacity=slots)
        commands = StateCommands.allocate(device, capacity=slots)
        frame = torch.zeros(1, dtype=torch.int32, device=device)
        completion = CompletionOutputs.allocate((2, slots, 24 if config.dflash else 8), device=device)
        coordinator = CollectivePreparation(group)
        ipc = package_storage.peer_storage
        draft_rows = slots * weights.config.draft.block_size if config.dflash else 0
        tokens = max(config.prefill_chunk_size, config.decode_tokens, draft_rows)
        routed = len(config.devices) == 8
        # Routed one-shot only serves rows below the two-shot crossover; its staging (one slot pair
        # per peer) would otherwise be sized for a whole prefill chunk, 1.5 GiB per rank at 8192.
        collective = create_all_reduce_runtime(group, device, max_tokens=ONESHOT_MAX_TOKENS if routed else tokens,
                                               hidden_size=6144, ipc=ipc)
        if routed:
            # TP8 one-shot cost grows with rows; two-shot serves decode, verification, draft and
            # short prefill rows above the measured crossover (three rows, 36 KiB).
            collective = RoutedAllReduce(collective, create_twoshot_all_reduce_runtime(group, device,
                max_tokens=TWOSHOT_MAX_TOKENS, hidden_size=6144, ipc=ipc), oneshot_max_bytes=3 * 6144 * 2)
        prefill_collective = create_dma_all_reduce_runtime(
            group, device, max_tokens=config.prefill_chunk_size, hidden_size=6144, ipc=ipc
        )
        sharded_draft = config.dflash and config.vocabulary_parallel_draft
        vocabulary_collective = create_all_gather_runtime(
            group, device, max_rows=max(config.decode_tokens, 0 if sharded_draft else draft_rows), columns=152576,
            ipc=ipc
        )
        # One payload row of draft candidates per proposal row and rank.
        candidate_collective = (create_all_gather_runtime(group, device,
            max_rows=max(batch for batch, _ in config.speculative_shapes),
            columns=SHARD_PAYLOAD_COLUMNS * shard.world_size, ipc=ipc) if sharded_draft else None)
        context_collective = (create_all_gather_runtime(group, device,
            max_rows=tokens, columns=6144, ipc=ipc) if config.dflash else None)
        vision_collective = create_dma_all_reduce_runtime(group, device,
            max_tokens=PATCH_CAPACITIES[-1] // 4, hidden_size=6144, ipc=ipc)
        identity = hashlib.sha256(
            json.dumps(
                {
                    "config": json.loads((snapshot / "config.json").read_text()),
                    "draft": json.loads((snapshot / "dflash/config.json").read_text()),
                    "headers": [s.header_sha256 for s in manifest.shards],
                    "world_size": shard.world_size,
                    "contract": "mimo_v2.graph-inputs.v1",
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        session = RecordedPreparationSession(
            compile_workers=config.compile_workers, tuning_replica=config.tuning_replica,
            device=device,
            checkpoint=identity,
            workload_schema="mimo_v2-vision",
            numerical_policy="default",
            coordinator=coordinator,
            headroom_bytes=config.headroom_bytes,
            benchmark_storage=package_storage,
        )
        session.__enter__()
        session.register_benchmark_resource("continuations", pool, restore=pool.clear_for_preparation)
        # The recipe declares no stream overlap: every branch runs on the main stream.
        arena = ScratchArena(device, overlap=False)
        session.register_scratch(arena)
        graph_buffers = GraphBufferPool(device, chunk_bytes=2 << 20)
        auxiliary_columns = (len(weights.config.draft.target_layers) * weights.config.text["hidden_size"]
                             if weights.draft is not None else 0)
        prefill_storage = Scratch.allocate(weights.config, shard, config.prefill_chunk_size, 1, device,
                                           buffers=graph_buffers.cursor(), auxiliary_columns=auxiliary_columns)
        decoder_storage = Scratch.allocate(
            weights.config, shard, config.decode_tokens, config.decode_tokens, device, buffers=graph_buffers.cursor(),
            auxiliary_columns=auxiliary_columns,
        )
        vision_inputs = VisionInputs.allocate(device)
        vision_prefill = PrefillVisionInputs.allocate(config.prefill_chunk_size, device)
        vision_state = VisionState(inputs, vision_prefill, requests, image_slots=config.image_concurrency, image_token=weights.config.text["image_token_id"],
                                   features=FEATURES, hidden=HIDDEN)
        vision_storage = {}
        environment = DecoderEnvironment.load(snapshot, vision_state)
        buffers = dict(
            inputs.buffers(),
            **vision_inputs.buffers(),
            **vision_prefill.buffers(),
            **commands.buffers(),
            **{name: getattr(admission, name) for name in admission.__dataclass_fields__},
            state_pool=pool.backing,
            memory_capacity=pool.memory_capacity,
            sampling_grammar_mask=requests.grammar_mask,
            frame_index=frame,
            submission_sequence=sequence,
            **completion.buffers(),
        )
        forwards, signatures, metadata, scratch = {}, {}, {}, {}
        speculation = None
        if config.dflash:
            speculation = SpeculativePreparation(
                weights,
                environment,
                pool,
                requests,
                inputs,
                decoder_storage,
                completion=completion,
                frame=frame,
                sequence=sequence,
                session=session,
                workspace=arena,
                runtime=collective,
                vocabulary_runtime=vocabulary_collective,
                context_runtime=context_collective,
                coordinator=coordinator,
                context_tokens=max(config.prefill_chunk_size, config.decode_tokens),
                draft_candidates=candidate_collective,
            )
            # One request's draft positions sample as one sharded row block.
            speculation.draft_position_blocks = config.draft_position_blocks

    scoring = PromptScoring.allocate(config, device, buffers, frame=frame, group=group,
        ipc=package_storage.peer_storage, runtime=collective, coordinator=coordinator)
    del manifest
    modes = (True, False) if config.dflash else (False,)

    def precisions(mode, capacity):
        # (quantized_activations, a4_prefill) per captured graph: prefill graphs from the A4 threshold
        # are captured with both activation precisions, every other graph once with b12x's default.
        if mode == "prefill" and config.a4_prefill and capacity >= config.a4_prefill_min_tokens:
            return ((False, False), (True, True))
        return ((None, None),)

    count = (
        len(modes) * sum(len(precisions(mode, capacity))
                         for _, mode, capacities in prefill_modes(config.prefill_capacities, (), scoring)
                         for capacity in capacities)
        + len(modes) * 4 * len(config.batch_capacities)
        + len(config.speculative_shapes)
        + len(config.batch_capacities)
        + len(PATCH_CAPACITIES)
        + (len(config.batch_capacities) * (config.draft_tokens + 1) if config.split_speculation else 0)
    )
    with phase(progress, "Construct graph catalog", total=count) as stage:

        def add(name, forward, kind, rows, tokens, fields, *, enabled=None, quantized=None, branch=None, switch=None):
            if kind == "scored_prefill":
                forward = PreparedScoredPrefill(forward, requests, inputs)
                fields = (*fields, "prompt_target_tokens")
            forwards[name] = forward
            signatures[name] = GraphSignature(kind, rows, tokens, tuple(fields), speculation_enabled=enabled,
                                              quantized_activations=quantized, branch=branch, switch=switch)
            stage.update(advance=1)

        forward_inputs = (*inputs.buffers(), "sampling_grammar_mask")
        for enabled in modes:
            suffix = "-no-mtp" if config.dflash and not enabled else ""
            mode_weights = weights if enabled else replace(weights, draft=None)
            for kind, mode, capacities in prefill_modes(
                    config.prefill_capacities, () if enabled else config.batch_capacities, scoring):
                for capacity in capacities:
                    for quantized, a4_prefill in precisions(mode, capacity):
                        rows = 1 if mode == "prefill" else capacity
                        name = f"{kind}{capacity}" + ("-a4" if quantized else "") + (suffix if mode == "prefill" else "")
                        metadata[name] = Metadata.allocate(capacity, rows, blocks, device)
                        storage = prefill_storage if mode == "prefill" else decoder_storage
                        scratch[name] = storage.rows(capacity, rows)
                        decoder = PreparedDecoder(
                            mode_weights,
                            environment,
                            scratch[name],
                            metadata[name],
                            pool,
                            mode=mode, scoring=scoring if kind == "scored_prefill" else None, a4_prefill=a4_prefill,
                            runtime=prefill_collective if mode == "prefill" and capacity >= 1024 else collective,
                            vocabulary_runtime=vocabulary_collective,
                            coordinator=coordinator,
                            session=session,
                            name=name,
                            workspace=arena,
                        )
                        batch = PreparedBatch(
                            inputs,
                            requests,
                            metadata[name],
                            mode=mode,
                            shard=shard,
                            pool=pool,
                            validate_prefill=vision_state.validate_prefill if mode == "prefill" else None,
                        )
                        forward = PreparedGeneration(
                            decoder,
                            batch,
                            completion,
                            frame,
                            sequence,
                            runtime=collective,
                            coordinator=coordinator,
                            session=session,
                            name=name + ".generation",
                            workspace=arena,
                            context_factory=None if speculation is None else speculation.forwards.context,
                        )
                        add(
                            name,
                            forward,
                            mode,
                            rows,
                            capacity if mode == "prefill" else 1,
                            (
                                (*forward_inputs, *vision_prefill.buffers())
                                if mode == "prefill"
                                else forward_inputs
                            ),
                            enabled=enabled,
                            quantized=quantized,
                        )
        if speculation is not None:
            from ignition.components.parallel_draft import PreparedParallelDraftRound

            speculation.sampling_buffers = GraphBufferPool(device, chunk_bytes=2 << 20)
            if config.switch_speculation:
                if config.split_speculation or 1 not in config.batch_capacities:
                    raise ValueError("switch speculation needs one-request rounds and excludes split speculation")
                from ignition.components.split_draft import (
                    PreparedDraftProposal, PreparedDraftVerification, PreparedProposalStorage)
                from ignition.components.width_selection import (
                    PreparedSelectedVerification, PreparedSelectionPrefix, PreparedWidthCalibration,
                    PreparedWidthSelection)

                widths = config.draft_tokens + 1
                switch_storage = PreparedProposalStorage(speculation, config.draft_tokens)
                buffers.update(switch_storage.buffers())
                switch_confidence = torch.zeros((config.request_capacity, config.draft_tokens),
                                                dtype=torch.float32, device=device)
                switch_costs = torch.zeros((config.request_capacity, widths), dtype=torch.float32, device=device)
                switch_selected = torch.ones(1, dtype=torch.int32, device=device)
                switch_sums = torch.zeros((config.draft_tokens, 2), dtype=torch.float32, device=device)
                buffers.update(proposal_confidence=switch_confidence, speculative_width_costs=switch_costs,
                               switch_selected_width=switch_selected, switch_calibration=switch_sums)
                calibration = PreparedWidthCalibration(switch_sums, switch_confidence, config.draft_tokens)
                prefix_name = f"selection1x{widths}"
                source = PreparedParallelDraftRound(speculation, 1, widths, f"{prefix_name}.proposal",
                    buffers=speculation.sampling_buffers.cursor(), proposal_only=True)
                source.prepare()
                bodies = tuple(f"{prefix_name}.w{width}" for width in range(1, widths + 1))
                for width, body in zip(range(1, widths + 1), bodies):
                    target = PreparedParallelDraftRound(speculation, 1, width, body, generate_proposals=False,
                        buffers=speculation.sampling_buffers.cursor())
                    target.prepare()
                    add(body, PreparedSelectedVerification(PreparedDraftVerification(switch_storage, target),
                            calibration, target.operations[3]), "continuation", 1, width, ())
                selection = PreparedWidthSelection(switch_confidence, switch_costs,
                    inputs.query_lengths, switch_selected, widths, calibration)
                add(prefix_name, PreparedSelectionPrefix(
                        PreparedDraftProposal(source, switch_confidence, switch_storage), selection),
                    "speculative", 1, widths, (*forward_inputs, "speculative_width_costs"), enabled=True,
                    switch=("switch_selected_width", bodies))
            if config.split_speculation:
                from ignition.components.capacity import proposal_shapes
                from ignition.components.split_draft import (
                    PreparedDraftProposal, PreparedDraftVerification, PreparedProposalStorage)

                confidence = package_storage.host_allocate(
                    "proposal_confidence", (config.request_capacity, config.draft_tokens), torch.float32)
                buffers["proposal_confidence"] = confidence.device_view
                split = PreparedProposalStorage(speculation, config.draft_tokens)
                buffers.update(split.buffers())
                # The DFlash drafter always predicts its full block; a proposal keeps the first depth drafts.
                for batch, depth in sorted(proposal_shapes(config.max_active, config.draft_tokens), reverse=True):
                    name = f"proposal{batch}x{depth + 1}"
                    source = PreparedParallelDraftRound(speculation, batch, depth + 1, name,
                        buffers=speculation.sampling_buffers.cursor(), proposal_only=True)
                    source.prepare()
                    add(name, PreparedDraftProposal(source, confidence.device_view, split), "proposal",
                        batch, depth + 1, forward_inputs, enabled=True)
                for batch, width in sorted(config.speculative_shapes, key=lambda shape: shape[0] * shape[1],
                                           reverse=True):
                    name = f"verification{batch}x{width}"
                    target = PreparedParallelDraftRound(speculation, batch, width, name, generate_proposals=False,
                        buffers=speculation.sampling_buffers.cursor())
                    target.prepare()
                    add(name, PreparedDraftVerification(split, target), "verification", batch, width,
                        forward_inputs, enabled=True)
                    if width == 1:
                        # Decoding without a proposal still retains the draft context.
                        decode = PreparedParallelDraftRound(speculation, batch, 1, f"speculative{batch}x1",
                            buffers=speculation.sampling_buffers.cursor())
                        decode.prepare()
                        add(f"speculative{batch}x1", decode, "speculative", batch, 1, forward_inputs, enabled=True)
            for batch, width in (() if config.split_speculation else
                                 sorted(config.speculative_shapes, key=lambda shape: shape[0] * shape[1], reverse=True)):
                if config.switch_speculation and (batch, width) == (1, config.draft_tokens + 1):
                    continue  # the selection graph owns the widest one-request shape
                name = f"speculative{batch}x{width}"
                two_phase = (width > 1 and batch <= config.two_phase_max_batch
                             and speculation.draft_candidates is not None)
                forward = PreparedParallelDraftRound(speculation, batch, width, name,
                                                     buffers=speculation.sampling_buffers.cursor(),
                                                     two_phase=two_phase)
                forward.prepare()
                branch = (("two_phase_flag", f"{name}.compact", f"{name}.dense") if two_phase else None)
                add(name, forward, "speculative", batch, width, forward_inputs, enabled=True, branch=branch)
                if two_phase:
                    from ignition.components.parallel_draft import RoundContinuation

                    add(f"{name}.compact", RoundContinuation(forward, True), "continuation", batch, width, ())
                    add(f"{name}.dense", RoundContinuation(forward, False), "continuation", batch, width, ())
        for capacity in reversed(PATCH_CAPACITIES):
            name = f"vision{capacity}"
            cursor = graph_buffers.cursor()
            vision_storage[capacity] = VisionScratch.allocate(capacity, shard, device, buffers=cursor)
            encoder = PreparedVision(
                weights.vision,
                vision_storage[capacity],
                vision_inputs,
                runtime=vision_collective,
                coordinator=coordinator,
                session=session,
                workspace=arena,
                name=name,
                buffers=cursor,
            )
            forward = PreparedVisionGeneration(encoder, vision_state, completion, frame, sequence, buffers=cursor)
            add(name, forward, "vision", 1, capacity, (*vision_inputs.buffers(), "vision_feature_slot"))
        row_inputs = ("request_state_rows", "row_slot_ids", "row_generations", "live_rows")
        for enabled in modes:
            suffix = "-no-mtp" if config.dflash and not enabled else ""
            mode_weights = weights if enabled else replace(weights, draft=None)
            for capacity in config.batch_capacities:
                name = f"resume{capacity}" + suffix
                forward = PreparedResume(
                    mode_weights,
                    requests,
                    pool,
                    inputs,
                    completion,
                    frame,
                    sequence,
                    rows=capacity,
                    runtime=collective,
                    vocabulary_runtime=vocabulary_collective,
                    coordinator=coordinator,
                    session=session,
                    name=name,
                )
                add(
                    name,
                    forward,
                    "resume",
                    capacity,
                    1,
                    (*row_inputs, "sampling_grammar_mask"),
                    enabled=enabled,
                )
                for mode in ("init", "restore", "snapshot"):
                    name = f"{mode}{capacity}" + suffix
                    forward = PreparedMaintenance(
                        mode,
                        requests,
                        pool,
                        commands,
                        inputs,
                        completion,
                        frame,
                        sequence,
                        runtime=collective,
                        coordinator=coordinator,
                        session=session,
                        name=name,
                        capacity=capacity,
                        admission_buffers=admission,
                        draft_state=enabled,
                    )
                    fields = (
                        row_inputs
                        if mode == "snapshot"
                        else ("request_state_rows", *admission.__dataclass_fields__)
                    )
                    add(
                        name,
                        forward,
                        mode,
                        capacity,
                        1,
                        (*fields, "page_tables", *commands.buffers()),
                        enabled=enabled,
                    )
    if progress is not None:
        progress(
            f"before tuning: {torch.cuda.memory_allocated(device) >> 20} MiB allocated, "
            f"{torch.cuda.mem_get_info(device)[0] >> 20} MiB free"
        )
    with phase(progress, "Prepare and tune kernels") as stage, transient:
        tuning = session.tune(show_progress=progress is not None and rank == 0, console=stage.console)
    stream, graph_pool = torch.cuda.Stream(device=device), package_storage.graph_pool.id
    pages = initial_pages

    def materialize_cache():
        pool.clear_for_preparation()

    graphs, compiled = capture_forwards(
        forwards,
        signatures,
        session=session,
        device=device,
        stream=stream,
        pool=graph_pool,
        collective=collective,
        coordinator=coordinator,
        compile_decoder=config.compile_decoder,
        progress=progress,
        cache_root=config.kernel_cache,
        retain_graphs=config.retain_graphs,
        materialize=materialize_cache,
        warmup_storage=package_storage,
        compile_kinds=("prefill", "scored_prefill", "decode", "resume", "speculative", "continuation", "vision"),
    )
    session.freeze()
    pool.clear_for_preparation()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    release_unused()
    if group is not None:
        torch.distributed.barrier(group=group)
    free, total = torch.cuda.mem_get_info(device)
    frontend = snapshot
    assets = {
        role: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for role, path in (
            ("tokenizer", snapshot / "tokenizer.json"),
            ("chat_template", frontend / "chat_template.jinja"),
        )
    }
    cache_regions = [history_region(buffers, "cache." + name, tensor, contract_proto.PAGED_KEY_VALUE)
        for name, tensor in pool.pages.items()]
    buffers["request_pending_token"] = requests.pending_token
    if speculation is not None:
        buffers["two_phase_flag"] = speculation.two_phase_flag
    server_config = {
        "identity": identity,
        "model_id": snapshot.name,
        "max_sequence_tokens": config.context,
        "prefill_chunk_size": config.prefill_chunk_size,
        "pages": pages,
        "state_rows": state_rows,
        "snapshots": snapshots,
        "page_tokens": PAGE_TOKENS,
        "frontend": prepare_frontend(snapshot,
            eos_token_ids=json.loads((snapshot / "generation_config.json").read_text())["eos_token_id"],
            image_profile="mimo"),
        "sampling_defaults": RECIPE.sampling_defaults.as_dict(),
        "assets": assets,
        "model_contract": prepare_contract({name: s for name, s in signatures.items() if s.kind != "continuation"},
            buffers, pool, image_slots=config.image_concurrency, page_tokens=PAGE_TOKENS, draft_lag=0 if config.dflash else None, regions=cache_regions),
    }
    if config.a4_prefill:
        server_config["quantized_activation_min_tokens"] = config.a4_prefill_min_tokens
    memory = {
        "rank": rank,
        "world_size": shard.world_size,
        "device": config.devices[rank],
        "weights_bytes": weights.allocated_bytes,
        "mapped_host_bytes": 0,
        "offloaded_bytes": 0,
        "expert_scale_compression": config.expert_scale_compression,
        "continuation_bytes": pool.backing.nbytes,
        "preparation_state_rows": int(pool.memory_capacity[0].item()),
        "preparation_state_bytes": int(pool.memory_capacity[0].item()) * layout.row_bytes,
        "state_rows": state_rows,
        "pages": pages,
        "page_bytes": bytes_per_page,
        "page_tokens": PAGE_TOKENS,
        "kv_bytes": pages * bytes_per_page,
        "completion_output_bytes": completion.nbytes, "operator_scratch_bytes": arena.nbytes,
        "graphs": len(graphs),
        "prefill_capacities": config.prefill_capacities,
        "free_device_bytes": free,
        "total_device_bytes": total,
        "headroom_bytes": config.headroom_bytes,
        "allocated_bytes": torch.cuda.memory_allocated(device),
        "reserved_bytes": torch.cuda.memory_reserved(device),
        "tuning": tuning,
        "compiled_decoder": config.compile_decoder,
    }
    if scoring is not None:
        memory.update(scoring.memory_report())
    if config.dflash:
        server_config.update(
            speculative_widths=config.speculative_widths,
            speculative_token_capacity=speculative_token_capacity(config.max_active, max(config.speculative_widths), config.speculative_token_capacity),
            speculative_fixed_width=config.speculative_fixed_width,
        )
        memory.update(speculative_shapes=config.speculative_shapes)
    disposable = [
        prefill_storage,
        decoder_storage,
        arena.tensor,
        metadata,
        vision_storage,
        vision_state.arena,
        graph_buffers.chunks,
    ]
    if speculation is not None:
        disposable.append(speculation.draft_scratch)
        if speculation.w8a16_workspace is not None:
            # Zeroed counters are the workspace's idle state.
            disposable.append(speculation.w8a16_workspace.scratch_tensors())
        disposable.append(speculation.sampling_buffers.chunks)
    for forward in forwards.values():
        if isinstance(forward, PreparedScoredPrefill):
            forward = forward.forward
        if hasattr(forward, "completion") and hasattr(forward.completion, "scratch_tensors"):
            disposable.append(forward.completion.scratch_tensors)
        if hasattr(forward, "decoder"):
            disposable.append(forward.decoder.scratch_tensors())
        if getattr(forward, "context", None) is not None:
            disposable.append(forward.context.scratch_tensors())
    return PreparedModel(rebuild_weights={"weights": weights},
        device=config.devices[rank],
        graphs=graphs,
        buffers=buffers,
        signatures=signatures,
        max_active=config.max_active,
        session=session,
        server_config=server_config,
        memory_report=memory,
        export_zero_ranges=(*(() if scoring is None else scoring.zero_ranges()),
                           *storage_ranges(disposable), *completion.zero_ranges()),
        resources=(scoring,
            weights,
            vision_state,
            vision_inputs,
            vision_prefill,
            vision_storage,
            graph_buffers,
            pool,
            requests,
            inputs,
            admission,
            commands,
            arena,
            forwards,
            prefill_storage,
            decoder_storage,
            scratch,
            metadata,
            compiled,
            stream,
            graph_pool,
            session,
            collective,
            prefill_collective,
            vocabulary_collective,
            candidate_collective,
            context_collective,
            vision_collective,
            coordinator,
            speculation,
        ),
        close_resources=tuple(
            c
            for c in (scoring, candidate_collective, context_collective, vision_collective, prefill_collective,
                      vocabulary_collective, collective)
            if c is not None
        ),
    )
