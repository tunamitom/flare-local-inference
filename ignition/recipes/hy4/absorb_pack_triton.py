"""absorb_pack_triton.py — Option 1's fused transpose-concat (PERF-PLAN.md Step 3).

`absorb_pack_fused` packs the absorbed GEMM's head-major result
`absorbed_h [H, R, lora]` together with the head-major gathered query's rope
block `q_hrd[..., qk_nope:]` (`[H, R, 64]`) straight into the sparse-MLA
query `[R, H, lora + 64]` in ONE launch. It replaces

    torch.cat((absorbed_h.transpose(0, 1),
               q_hrd[..., qk_nope:].transpose(0, 1)), dim=-1)

— the `torch.cat` output allocation plus two transpose copies — with a single
read/write pass. PURE ELEMENT MOVEMENT: bit-identical to the `torch.cat` form
(no arithmetic on the way), enforced in
`tests/recipes/test_hy4_attention_bmm_strides.py`.

GPU (Triton): one kernel, `(R * H,)` programs of `BLOCK` columns each.
CPU / no Triton: the exact `torch.cat` reference above.
"""
import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - CPU test hosts have no Triton
    triton = None
    tl = None


if triton is not None:
    @triton.jit
    def _absorb_pack_kernel(q_ptr, absorbed_ptr, out_ptr, R, H, W, NOPE, L,
                            BLOCK: tl.constexpr):
        """out[idx, :L] = absorbed_h[h, r]; out[idx, L:] = q_hrd[h, r, NOPE:]."""
        idx = tl.program_id(0)
        r = idx // H
        h = idx % H
        c = tl.arange(0, BLOCK)
        rope = W - NOPE
        a_mask = c < L
        q_off = c - L
        q_mask = (c >= L) & (q_off < rope)
        a = tl.load(absorbed_ptr + (h * R + r) * L + c, mask=a_mask, other=0)
        qv = tl.load(q_ptr + (h * R + r) * W + NOPE + q_off, mask=q_mask, other=0)
        # The store MUST be masked to the row width L + rope: BLOCK is the next
        # power of two (576 -> 1024), and an unmasked store spills 448 elements
        # into the next row and 896 B past the buffer end (Xid 31 MMU write
        # fault observed 2026-11-20 on 11:00.0 before this mask existed).
        tl.store(out_ptr + idx * (L + rope) + c, tl.where(a_mask, a, qv),
                 mask=c < L + rope)


def absorb_pack_fused(q_hrd, absorbed_h, *, qk_nope, lora, out=None):
    """`q_hrd` `[H, R, qk_nope + rope]` (contiguous head-major) and
    `absorbed_h` `[H, R, lora]` (contiguous) -> `[R, H, lora + rope]`.

    With `out` given (shape `[R, H, lora + rope]`) the pack lands there
    directly; otherwise a fresh output is allocated. Bit-exact: every output
    element is a moved input element."""
    if q_hrd.dim() != 3 or absorbed_h.dim() != 3:
        raise ValueError("absorb_pack_fused: head-major 3-D inputs")
    h, rows, width = (int(v) for v in q_hrd.shape)
    nope, lora = int(qk_nope), int(lora)
    rope = width - nope
    if rope <= 0 or nope <= 0:
        raise ValueError("absorb_pack_fused: qk_nope must split q_hrd's width")
    if tuple(absorbed_h.shape) != (h, rows, lora):
        raise ValueError(f"absorb_pack_fused: absorbed_h must be [{h}, {rows}, {lora}]")
    # Layout guard (zero perf cost): the kernel indexes flat [H*R, W]/[H*R, L]
    # rows and the out store is flat [R*H, L + rope] — a strided view would
    # silently mis-index. Fail loudly instead (Neon review, condition 2b).
    if not (q_hrd.is_contiguous() and absorbed_h.is_contiguous()
            and (out is None or out.is_contiguous())):
        raise ValueError("absorb_pack_fused: [H, R, W] / [H, R, L] inputs (and out) "
                         "must be contiguous — flat-row indexing mis-uses strides")
    fresh = out is None
    if fresh:
        out = torch.empty((rows, h, lora + rope), dtype=q_hrd.dtype, device=q_hrd.device)
    elif tuple(out.shape) != (rows, h, lora + rope):
        raise ValueError(f"absorb_pack_fused: out must be [{rows}, {h}, {lora + rope}]")
    if triton is not None and q_hrd.is_cuda and q_hrd.device.type == "cuda":
        block = triton.next_power_of_2(lora + rope)
        _absorb_pack_kernel[(rows * h,)](q_hrd, absorbed_h, out, rows, h, width,
                                         nope, lora, BLOCK=block, num_warps=4)
        return out
    # CPU / no-Triton reference: the exact torch.cat this kernel replaces.
    packed = torch.cat((absorbed_h.transpose(0, 1),
                        q_hrd[..., nope:].transpose(0, 1)), dim=-1)
    return packed if fresh else out.copy_(packed)
