"""Prefetch upcoming decode weights into L2 on a side stream while device memory idles.

Decode projections stream their weights from device memory, while all-reduces, routing
and small kernels leave it idle for part of every layer. A prefetch window issues
``cp.async.bulk.prefetch.L2`` with an ``evict_last`` policy for weights the next kernels
read, so those kernels find them in L2. The hints change no values. Each window forks
the side stream from the main stream; one join before the graph ends rejoins it.

Ported from the vLLM reference (``vllm/models/glm5next/nvidia/l2_prefetch.py`` and the
MiMo windows in ``vllm/model_executor/models/mimo_v2.py``, r16 007ca84c3).
"""

from weakref import finalize

import torch
import triton
import triton.language as tl

CHUNK_BYTES = 4096
BLOCK = 128
# Fewer issuing CTAs throttle the fill rate, leaving device-memory bandwidth for the
# latency-bound kernels running beside the prefetch.
GRID = 16
WINDOW_BYTES = 20 * 1000 * 1000
MIN_BYTES = 16


@triton.jit
def _l2_prefetch(Segments, segments, CHUNK: tl.constexpr, BLOCK: tl.constexpr):
    program, programs = tl.program_id(0), tl.num_programs(0)
    for segment in range(segments):
        base = tl.load(Segments + 2 * segment)
        nbytes = tl.load(Segments + 2 * segment + 1)
        chunks = (nbytes + CHUNK - 1) // CHUNK
        first = program * BLOCK
        while first < chunks:
            index = first + tl.arange(0, BLOCK).to(tl.int64)
            offset = index * CHUNK
            size = tl.minimum(nbytes - offset, CHUNK) // 16 * 16
            live = (index < chunks) & (size > 0)
            tl.inline_asm_elementwise(
                "{ .reg .b64 policy; .reg .pred live; setp.ne.b32 live, $3, 0; "
                "createpolicy.fractional.L2::evict_last.b64 policy, 1.0; "
                "@live cp.async.bulk.prefetch.L2.global.L2::cache_hint [$1], $2, policy; mov.u32 $0, 0; }",
                "=r,l,r,r", [base + offset, tl.where(live, size, 0).to(tl.int32), live.to(tl.int32)],
                dtype=tl.int32, is_pure=False, pack=1)
            first += programs * BLOCK


@torch.library.custom_op("flare::l2_prefetch", mutates_args=("token",))
def _issue(segments: torch.Tensor, token: torch.Tensor, stream: int, after: torch.Tensor | None = None) -> None:
    # ``after`` is read only for ordering: traced forwards cannot hoist the window above the
    # kernel that produces it, so the side stream forks once that kernel is on the main stream.
    main = torch.cuda.current_stream(segments.device)
    side = torch.cuda.ExternalStream(stream, device=segments.device)
    side.wait_stream(main)
    with torch.cuda.stream(side):
        _l2_prefetch[(GRID,)](segments, segments.shape[0] // 2, CHUNK=CHUNK_BYTES, BLOCK=BLOCK, num_warps=4)


@_issue.register_fake
def _fake_issue(segments, token, stream, after=None):
    pass


@torch.library.custom_op("flare::l2_prefetch_join", mutates_args=("token",))
def _join(token: torch.Tensor, stream: int) -> None:
    torch.cuda.current_stream(token.device).wait_stream(torch.cuda.ExternalStream(stream, device=token.device))


@_join.register_fake
def _fake_join(token, stream):
    pass


def tensor_segments(tensors):
    """Byte ranges of contiguous CUDA tensors; duplicates and tiny tensors are dropped."""
    segments, seen = [], set()
    for tensor in tensors:
        if not isinstance(tensor, torch.Tensor) or not tensor.is_cuda or not tensor.is_contiguous():
            continue
        pointer, nbytes = tensor.data_ptr(), tensor.numel() * tensor.element_size()
        if nbytes < MIN_BYTES or pointer % 16 or pointer in seen:
            continue
        seen.add(pointer)
        segments.append((pointer, nbytes))
    return segments


class L2Prefetcher:
    """One side stream and ordering token per prepared forward; windows fork, one join rejoins."""

    def __init__(self, device):
        from cuda.bindings import driver as cuda

        self.device = torch.device(device)
        # The token also makes the device's primary context current for the driver call below.
        self.token = torch.zeros(1, dtype=torch.int32, device=self.device)
        # A raw stream: PyTorch's round-robin pool can alias a later capture stream. Traced
        # forwards see only the integer stream_handle, which compiled guards leave unbound
        # because every rank's handle differs.
        with torch.cuda.device(self.device):
            status, handle = cuda.cuStreamCreate(cuda.CUstream_flags.CU_STREAM_NON_BLOCKING)
        if status != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cannot create L2 prefetch stream: {status}")
        self.stream_handle = int(handle)
        finalize(self, cuda.cuStreamDestroy, handle).atexit = False
        self.issued = False

    def plan(self, tensors, budget=WINDOW_BYTES):
        """Device-resident [pointer, bytes] table for one window, trimmed to its fill budget."""
        taken, used = [], 0
        for pointer, nbytes in tensor_segments(tensors):
            room = budget - used
            if room <= 0:
                break
            size = nbytes if nbytes <= room else room - room % CHUNK_BYTES
            if size > 0:
                taken.append((pointer, size))
                used += size
        if not taken:
            return None
        return torch.tensor([value for segment in taken for value in segment], dtype=torch.int64,
                            device=self.device)

    def issue(self, plan, after=None):
        """Fork a window; with ``after``, it starts only once the kernel producing that tensor ends."""
        if plan is not None:
            torch.ops.flare.l2_prefetch(plan, self.token, self.stream_handle, after)
            self.issued = True

    def join(self):
        if self.issued:
            torch.ops.flare.l2_prefetch_join(self.token, self.stream_handle)
