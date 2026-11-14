# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Preparation settings for a Hy4 tensor-parallel rank.

The recipe constructs and validates this configuration from checkpoint
metadata on CPU. Per-rank decoder preparation — the iHC operator, sink
plumbing, shared selection, and gated MLA — is not implemented yet and raises
from `prepare_rank`.
"""

from dataclasses import dataclass
from pathlib import Path

from .config import PAGE_TOKENS


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
    # One native MTP predictor with three draft tokens: widths up to four.
    speculative_widths: tuple[int, ...] = (1, 2, 3, 4)
    speculative_fixed_width: int | None = None
    speculative_token_capacity: int | None = None
    mtp_only: bool = False

    def __post_init__(self):
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


def prepare_rank(snapshot, config, *, rank, group, progress=None, core_storage=None, checkpoint=None):
    """Load, tune, and capture one Hy4 rank.

    Not implemented: the decoder body (torch-op iHC, sinks, shared selection,
    gated MLA with interleaved RoPE) and the operator deltas it needs are future
    work. `recipe.prepare` reaches this only on GPUs.
    """
    raise NotImplementedError(
        "Hy4 rank preparation is not implemented; the recipe validates "
        "geometry, weight metadata, and the export contract only")
