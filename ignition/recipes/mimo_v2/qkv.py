"""One launch from the fused QKV projection to rotated attention operands in decode and verification.

Storage lowering turns every copy into persistent scratch into its own storage call, so the
unfused path spends ten launches per layer between the QKV projection and attention: split,
partial RoPE for queries and keys, the value scale, and the retained verification copies. This
kernel writes them all directly and rounds every intermediate to BF16 where the PyTorch
expression does, so its outputs match that expression bit for bit.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

BLOCK = 256


@triton.jit
def _bf16(x):
    # Round-to-nearest-even onto the BF16 grid, kept in FP32. Triton folds a BF16 round trip.
    bits = x.to(tl.uint32, bitcast=True)
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def _qkv(Projected, projected_stride, Positions, Inverse, Query, Key, Value, RetainedKey, RetainedValue,
         value_scale, HEADS: tl.constexpr, KEY_DIM: tl.constexpr, VALUE_DIM: tl.constexpr,
         ROTARY: tl.constexpr, SCALE: tl.constexpr, RETAIN: tl.constexpr, BLOCK: tl.constexpr):
    row, head = tl.program_id(0), tl.program_id(1)
    source = Projected + row.to(tl.int64) * projected_stride
    lanes = tl.arange(0, BLOCK)
    if head < HEADS + 1:
        live = lanes < KEY_DIM
        base = source + head * KEY_DIM
        values = tl.load(base + lanes, mask=live, other=0.0)
        half = ROTARY // 2
        rotary = lanes < ROTARY
        partner = tl.load(base + tl.where(lanes < half, lanes + half, lanes - half), mask=rotary, other=0.0)
        partner = tl.where(lanes < half, -partner, partner)
        position = tl.maximum(tl.load(Positions + row), 0).to(tl.float32)
        angle = position * tl.load(Inverse + lanes % half, mask=rotary, other=0.0)
        cos = _bf16(libdevice.cos(angle))
        sin = _bf16(libdevice.sin(angle))
        left = _bf16(values.to(tl.float32) * cos)
        right = _bf16(partner.to(tl.float32) * sin)
        rotated = tl.where(rotary, (left + right).to(tl.bfloat16), values)
        if head < HEADS:
            tl.store(Query + (row * HEADS + head) * KEY_DIM + lanes, rotated, mask=live)
        else:
            tl.store(Key + row * KEY_DIM + lanes, rotated, mask=live)
            if RETAIN:
                tl.store(RetainedKey + row * KEY_DIM + lanes, rotated, mask=live)
    else:
        live = lanes < VALUE_DIM
        values = tl.load(source + (HEADS + 1) * KEY_DIM + lanes, mask=live, other=0.0)
        if SCALE:
            values = (values.to(tl.float32) * value_scale).to(tl.bfloat16)
        tl.store(Value + row * VALUE_DIM + lanes, values, mask=live)
        if RETAIN:
            tl.store(RetainedValue + row * VALUE_DIM + lanes, values, mask=live)


def _launch(projected, positions, inverse, query, key, value, retained_key, retained_value, value_scale):
    rows, heads, key_dim = query.shape
    _qkv[(rows, heads + 2)](projected, projected.stride(0), positions, inverse, query, key, value,
        key if retained_key is None else retained_key, value if retained_value is None else retained_value,
        1.0 if value_scale is None else value_scale, HEADS=heads, KEY_DIM=key_dim, VALUE_DIM=value.shape[-1],
        ROTARY=inverse.numel() * 2, SCALE=value_scale is not None, RETAIN=retained_key is not None,
        BLOCK=BLOCK, num_warps=2)


@torch.library.custom_op("flare::mimo_qkv", mutates_args=("query", "key", "value"))
def _rotate(projected: torch.Tensor, positions: torch.Tensor, inverse: torch.Tensor, query: torch.Tensor,
            key: torch.Tensor, value: torch.Tensor, value_scale: float | None = None) -> None:
    _launch(projected, positions, inverse, query, key, value, None, None, value_scale)


@_rotate.register_fake
def _fake_rotate(projected, positions, inverse, query, key, value, value_scale=None):
    pass


@torch.library.custom_op("flare::mimo_qkv_retained",
                         mutates_args=("query", "key", "value", "retained_key", "retained_value"))
def _rotate_retained(projected: torch.Tensor, positions: torch.Tensor, inverse: torch.Tensor,
                     query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, retained_key: torch.Tensor,
                     retained_value: torch.Tensor, value_scale: float | None = None) -> None:
    _launch(projected, positions, inverse, query, key, value, retained_key, retained_value, value_scale)


@_rotate_retained.register_fake
def _fake_rotate_retained(projected, positions, inverse, query, key, value, retained_key, retained_value,
                          value_scale=None):
    pass


def serves(chunks, query, key, value, geometry):
    """Whether one fused-QKV chunk holds this rank's heads in the kernel's layout."""
    rows, heads, key_dim = query.shape
    return (len(chunks) == 1 and key.shape[1:] == (1, key_dim) and value.shape[1:] == (1, value.shape[-1])
            and chunks[0].query_rows == heads * key_dim and chunks[0].key_rows == key_dim
            and chunks[0].value_rows == value.shape[-1] and max(key_dim, value.shape[-1]) <= BLOCK
            and all(t.is_contiguous() and t.dtype == torch.bfloat16 for t in (query, key, value))
            and geometry.rotary_dim % 2 == 0 and geometry.rotary_dim <= key_dim)


def rotate(projected, positions, inverse, query, key, value, value_scale=None, retained=None):
    if retained is None:
        torch.ops.flare.mimo_qkv(projected, positions, inverse, query, key, value, value_scale)
    else:
        torch.ops.flare.mimo_qkv_retained(projected, positions, inverse, query, key, value, *retained, value_scale)


@triton.jit
def _stage_rows(history, values, staged, source_row, history_row, destination, valid, prefix,
                WIDTH: tl.constexpr, CLAMP: tl.constexpr, BLOCK: tl.constexpr):
    lanes = tl.arange(0, BLOCK)
    live = lanes < WIDTH
    previous = tl.load(history + history_row * WIDTH + lanes, mask=live & valid & prefix, other=0.0)
    current = tl.load(values + source_row * WIDTH + lanes, mask=live & valid & (prefix == 0), other=0.0)
    result = tl.where(valid, tl.where(prefix, previous.to(tl.float32), current.to(tl.float32)), 0.0)
    if CLAMP:
        result = tl.minimum(tl.maximum(result, -448.0), 448.0)
    tl.store(staged + destination * WIDTH + lanes, result.to(staged.dtype.element_ty), mask=live)


@triton.jit
def _stage(KeyHistory, Keys, StagedKey, ValueHistory, Values, StagedValue, Starts, Lengths, StateRows,
           QueryStartLoc, WindowLengths, rows_available, window, capacity, KEY: tl.constexpr,
           VALUE: tl.constexpr, CLAMP: tl.constexpr, BLOCK: tl.constexpr):
    request, offset = tl.program_id(0), tl.program_id(1)
    start = tl.load(Starts + request)
    length = tl.load(Lengths + request)
    if offset == 0:
        tl.store(WindowLengths + request, (tl.minimum(start, window) + length).to(tl.int32))
    position = tl.maximum(start - window, 0) + offset
    history_row = tl.load(StateRows + request) * window + position % window
    source_row = tl.minimum(tl.maximum(tl.load(QueryStartLoc + request) + position - start, 0), rows_available - 1)
    valid = (length > 0) & (position < start + length)
    prefix = position < start
    destination = request.to(tl.int64) * capacity + offset
    _stage_rows(KeyHistory, Keys, StagedKey, source_row, history_row, destination, valid, prefix,
                KEY, CLAMP, BLOCK)
    _stage_rows(ValueHistory, Values, StagedValue, source_row, history_row, destination, valid, prefix,
                VALUE, CLAMP, BLOCK)


@torch.library.custom_op("flare::mimo_stage_windows",
                         mutates_args=("staged_key", "staged_value", "window_lengths"))
def _stage_windows(key_history: torch.Tensor, keys: torch.Tensor, staged_key: torch.Tensor,
                   value_history: torch.Tensor, values: torch.Tensor, staged_value: torch.Tensor,
                   starts: torch.Tensor, lengths: torch.Tensor, state_rows: torch.Tensor,
                   query_start_loc: torch.Tensor, window_lengths: torch.Tensor) -> None:
    rows, window = starts.numel(), key_history.shape[1]
    capacity = staged_key.shape[0] * staged_key.shape[1] // rows
    _stage[(rows, capacity)](key_history, keys, staged_key, value_history, values, staged_value, starts, lengths,
        state_rows, query_start_loc, window_lengths, keys.shape[0], window, capacity,
        KEY=keys[0].numel(), VALUE=values[0].numel(), CLAMP=staged_key.dtype == torch.float8_e4m3fn,
        BLOCK=triton.next_power_of_2(max(keys[0].numel(), values[0].numel())), num_warps=1)


@_stage_windows.register_fake
def _fake_stage_windows(key_history, keys, staged_key, value_history, values, staged_value, starts, lengths,
                        state_rows, query_start_loc, window_lengths):
    pass


def stage(key_history, keys, staged_key, value_history, values, staged_value, metadata, window_lengths):
    """Both circular windows of a verification step plus their attention lengths, in one launch.

    Matches ``stage_windows`` for each cache and ``starts.clamp(max=window) + lengths``.
    """
    m = metadata
    torch.ops.flare.mimo_stage_windows(key_history, keys, staged_key, value_history, values, staged_value,
        m.starts, m.lengths, m.state_rows, m.query_start_loc, window_lengths)


def stages(key_history, staged_key, value_history, staged_value, keys, values):
    """Whether the fused staging serves these caches (matching dtypes, one row per slot)."""
    return (key_history.dtype == staged_key.dtype and value_history.dtype == staged_value.dtype
            and staged_key.dtype in (torch.float8_e4m3fn, torch.bfloat16)
            and staged_key.dtype == staged_value.dtype
            and all(t.is_contiguous() for t in (key_history, staged_key, value_history, staged_value, keys, values))
            and key_history.shape[1] == value_history.shape[1])
