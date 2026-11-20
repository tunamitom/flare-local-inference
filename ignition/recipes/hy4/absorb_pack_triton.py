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

N3: `absorb_pack_rope_fused` additionally fuses the decode-family
`_absorb_into` (attention.py) pack AND the interleaved RoPE of the row-major
query's rope block (`q_rhi[..., qk_nope:]`) into one launch — replacing
`out[..., :lora].copy_(absorbed.transpose(0, 1))` +
`rope_interleaved(..., out=out[..., lora:])`. The RoPE arithmetic mirrors
`fused_ops._rope_kernel` exactly (mul_rn / add_rn, libdevice cos/sin, round
to the input dtype then the out dtype), so the output is BIT-IDENTICAL to the
two-launch chain (0 ULP).
"""
import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
except Exception:  # pragma: no cover - CPU test hosts have no Triton
    triton = None
    tl = None
    libdevice = None


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


if triton is not None:
    @triton.jit
    def _absorb_pack_rope_kernel(absorbed_ptr, q_ptr, pos_ptr, inv_ptr, out_ptr,
                                 R, H, W, NOPE, L, POS_STRIDE,
                                 ROPE: tl.constexpr, BLOCK: tl.constexpr):
        """out[idx, :L] = absorbed_h[h, r]; out[idx, L:] = interleaved RoPE of
        q_rhi[r, h, NOPE:] at the row's position — one program per (r, h).

        The RoPE arithmetic mirrors `fused_ops._rope_kernel` EXACTLY
        (`libdevice.mul_rn` / `add_rn`, `libdevice.cos` / `sin`, rounded to
        the input dtype then the out dtype), so the rotated values are
        bit-identical to the `rope_interleaved` launch this replaces."""
        idx = tl.program_id(0)
        r = idx // H
        h = idx % H
        c = tl.arange(0, BLOCK)
        a_mask = c < L
        o = c - L
        q_mask = (c >= L) & (o < ROPE)
        # the pack half: a pure element move of the GEMM's head-major row
        a = tl.load(absorbed_ptr + (h * R + r) * L + c, mask=a_mask, other=0.0)
        # the rope half: GPT-J adjacent pairs over q_rhi[r, h, NOPE:]
        jj = tl.where(q_mask, o, 0)
        pair = jj // 2
        even = (jj % 2) == 0
        pos = tl.load(pos_ptr + idx // POS_STRIDE).to(tl.float32)
        inv = tl.load(inv_ptr + pair, mask=q_mask, other=0.0)
        angle = libdevice.mul_rn(pos, inv)
        cs = libdevice.cos(angle)
        sn = libdevice.sin(angle)
        partner = tl.where(even, o + 1, o - 1)
        x = tl.load(q_ptr + (r * H + h) * W + NOPE + o, mask=q_mask, other=0.0)
        xp = tl.load(q_ptr + (r * H + h) * W + NOPE + partner,
                     mask=q_mask, other=0.0).to(tl.float32)
        xf = x.to(tl.float32)
        other = tl.where(even, -xp, xp)
        rot = libdevice.add_rn(libdevice.mul_rn(xf, cs), libdevice.mul_rn(other, sn))
        y = tl.where(q_mask, rot.to(x.dtype.element_ty).to(out_ptr.dtype.element_ty),
                     x.to(out_ptr.dtype.element_ty))
        # Masked store (b725daa7): BLOCK is the next power of two of L + ROPE.
        tl.store(out_ptr + idx * (L + ROPE) + c, tl.where(a_mask, a, y),
                 mask=c < L + ROPE)


def _rope_reference(x, positions, *, rope_dim, theta):
    """The eager interleaved RoPE of `attention.apply_interleaved_rope` on the
    last `rope_dim` dims (here the whole block), with the position rows
    covering the flattened leading dims in row-major order — the same values
    `fused_ops.rope_interleaved` writes on the GPU (its mul_rn / add_rn /
    libdevice cos / sin reproduce this eager expression bit for bit)."""
    flat = x.reshape(-1, int(x.shape[-1]))
    n = int(flat.shape[0])
    rows = int(positions.numel())
    if rows <= 0 or n % rows:
        raise ValueError("rope rows must cover the positions")
    heads = n // rows
    inv_freq = 1.0 / (theta ** (torch.arange(0, rope_dim, 2, device=x.device,
                                             dtype=torch.float32) / rope_dim))
    freqs = positions.reshape(-1, 1).float() * inv_freq.reshape(1, -1)
    angles = freqs.repeat_interleave(2, dim=-1)   # interleaved pairs
    cos, sin = angles.cos(), angles.sin()
    pos_for_row = torch.arange(n, device=x.device) // heads
    cos, sin = cos[pos_for_row], sin[pos_for_row]
    x1, x2 = flat[..., ::2], flat[..., 1::2]
    half = torch.stack((-x2, x1), dim=-1).flatten(-2)
    rotated = flat.float() * cos + half.float() * sin
    return rotated.to(x.dtype).view(x.shape)


def absorb_pack_rope_fused(q_rhi, absorbed_h, positions, *, qk_nope, lora,
                           rope_dim, theta, out=None):
    """`q_rhi` `[R, H, qk_nope + rope]` (contiguous row-major),
    `absorbed_h` `[H, R, lora]` (contiguous) + `positions` -> `[R, H, lora + rope]`.

    `out[r, h, :lora] = absorbed_h[h, r]` (pure movement) and `out[r, h, lora:]`
    is the interleaved RoPE of `q_rhi[r, h, qk_nope:]` at the row's position —
    exactly the two-launch `copy_ + rope_interleaved` chain `_absorb_into`
    ran (bit-identical, 0 ULP: the pack moves bytes and the kernel's RoPE
    mirrors `fused_ops._rope_kernel`'s arithmetic). `positions` covers the
    `R * H` rows row-major, exactly `rope_interleaved`'s contract."""
    if q_rhi.dim() != 3 or absorbed_h.dim() != 3:
        raise ValueError("absorb_pack_rope_fused: 3-D inputs")
    rows, heads, width = (int(v) for v in q_rhi.shape)
    nope, lora, rope = int(qk_nope), int(lora), int(rope_dim)
    if nope <= 0 or lora <= 0 or rope <= 0 or rope % 2:
        raise ValueError("absorb_pack_rope_fused: positive widths and an even rope_dim")
    if width != nope + rope:
        raise ValueError("absorb_pack_rope_fused: q_rhi must be "
                         "[rows, heads, qk_nope + rope_dim]")
    if tuple(absorbed_h.shape) != (heads, rows, lora):
        raise ValueError(f"absorb_pack_rope_fused: absorbed_h must be "
                         f"[{heads}, {rows}, {lora}]")
    if absorbed_h.dtype != q_rhi.dtype:
        raise ValueError("absorb_pack_rope_fused: absorbed_h must match q_rhi's dtype")
    pos_count = int(positions.numel())
    if pos_count <= 0 or (rows * heads) % pos_count:
        raise ValueError("absorb_pack_rope_fused: positions must cover the rows")
    # Layout guard (zero perf cost, as in absorb_pack_fused): the kernel
    # indexes flat rows; a strided view would silently mis-index.
    if not (q_rhi.is_contiguous() and absorbed_h.is_contiguous()
            and (out is None or out.is_contiguous())):
        raise ValueError("absorb_pack_rope_fused: [rows, heads, W] / [heads, rows, L] "
                         "inputs (and out) must be contiguous — flat-row indexing "
                         "mis-uses strides")
    fresh = out is None
    if fresh:
        out = torch.empty((rows, heads, lora + rope), dtype=q_rhi.dtype,
                          device=q_rhi.device)
    elif tuple(out.shape) != (rows, heads, lora + rope):
        raise ValueError(f"absorb_pack_rope_fused: out must be "
                         f"[{rows}, {heads}, {lora + rope}]")
    elif out.dtype != q_rhi.dtype:
        raise ValueError("absorb_pack_rope_fused: out must match q_rhi's dtype")
    if triton is not None and q_rhi.is_cuda and q_rhi.device.type == "cuda":
        from .fused_ops import _inv_freq

        inv = _inv_freq(rope, theta, q_rhi.device)
        pos = positions.reshape(-1)
        block = triton.next_power_of_2(lora + rope)
        _absorb_pack_rope_kernel[(rows * heads,)](
            absorbed_h, q_rhi, pos, inv, out, rows, heads, width, nope, lora,
            (rows * heads) // pos_count,
            ROPE=rope, BLOCK=block, num_warps=4)
        return out
    # CPU / no-Triton reference: the exact two-launch chain this replaces.
    out[..., :lora].copy_(absorbed_h.transpose(0, 1))
    out[..., lora:].copy_(_rope_reference(q_rhi[..., nope:], positions,
                                         rope_dim=rope, theta=theta))
    return out
