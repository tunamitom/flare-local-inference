"""consult18 S2: which state does one forward WRITE? (HY4_STATE_CENSUS=1)

The kv gate lets lane 1 run layer l while lane 0 is still inside layer l, so
any single-copy buffer a layer writes during a forward is shared mutable state
between the lanes. This census finds them instead of guessing: it walks every
CUDA tensor reachable from the decoder (layers, plans, caches, scratch pools;
not the transport, whose buffers are per-lane channels and peer-mapped), takes
a cheap device checksum of each storage, runs one forward, and lists the
storages whose checksum changed, by the first attribute path that reaches
them. Storages above `limit` bytes are checksummed too (views, no copies).
Hidden C++ state (opaque plan handles) is outside its reach and is named as
such by the caller.
"""
from __future__ import annotations

import torch

__all__ = ["checksums", "host_changes", "host_state", "mutated", "reachable_storages"]

_SKIP_ATTRS = {"collectives", "transport", "_lane_streams", "trace", "_shared_stream"}


def reachable_storages(root, *, name="decoder", skip=_SKIP_ATTRS, max_nodes=2_000_000):
    """{storage data_ptr: (path, flat uint8 view of the storage)} for every CUDA
    tensor reachable from `root` through attributes, dicts, lists, tuples."""
    seen, found = set(), {}
    stack = [(name, root)]
    nodes = 0
    while stack:
        path, obj = stack.pop()
        nodes += 1
        if nodes > max_nodes or id(obj) in seen:
            continue
        seen.add(id(obj))
        if isinstance(obj, torch.Tensor):
            if obj.is_cuda and obj.untyped_storage().nbytes():
                storage = obj.untyped_storage()
                ptr = storage.data_ptr()
                if ptr not in found:
                    flat = torch.empty(0, dtype=torch.uint8, device=obj.device)
                    flat.set_(storage, 0, (storage.nbytes(),))
                    found[ptr] = (path, flat)
            continue
        if isinstance(obj, (str, bytes, int, float, bool, complex, type(None), type)) or callable(obj) \
                and not hasattr(obj, "__dict__"):
            continue
        if isinstance(obj, (torch.cuda.Stream, torch.cuda.Event, torch.device, torch.dtype,
                            torch.Generator)):
            continue
        if isinstance(obj, dict):
            for key, value in list(obj.items()):
                stack.append((f"{path}[{key!r}]", value))
            continue
        if isinstance(obj, (list, tuple, set, frozenset)):
            for index, value in enumerate(list(obj)):
                stack.append((f"{path}[{index}]", value))
            continue
        if isinstance(obj, torch.nn.Module) or hasattr(obj, "__dict__"):
            try:
                items = list(vars(obj).items())
            except TypeError:
                items = []
            for key, value in items:
                if key in skip:
                    continue
                stack.append((f"{path}.{key}", value))
        slots = getattr(type(obj), "__slots__", ())
        for key in (slots if isinstance(slots, (list, tuple)) else (slots,)):
            if key in skip or not hasattr(obj, key):
                continue
            stack.append((f"{path}.{key}", getattr(obj, key)))
    return found


def checksums(found):
    """Two int64 sums per storage (all words, odd words): device-side, no copies,
    one host read at the end."""
    sums = []
    for _, (_, flat) in sorted(found.items()):
        usable = flat.numel() // 4 * 4
        words = flat[:usable].view(torch.int32) if usable else flat[:0].view(torch.int32)
        tail = flat[usable:].to(torch.int64).sum() if usable != flat.numel() else 0
        sums.append(torch.stack((words.sum(dtype=torch.int64) + tail,
                                 words[1::2].sum(dtype=torch.int64))))
    if not sums:
        return {}
    host = torch.stack(sums).cpu().tolist()
    return {ptr: tuple(values) for (ptr, _), values in zip(sorted(found.items()), host)}


def mutated(found, before, after):
    """[(path, nbytes)] of storages whose checksum changed, largest first."""
    out = [(found[ptr][0], int(found[ptr][1].numel())) for ptr in before
           if ptr in after and before[ptr] != after[ptr]]
    return sorted(out, key=lambda item: -item[1])


_SCALARS = (str, bytes, int, float, bool, complex, type(None))


def host_state(root, *, name="decoder", skip=_SKIP_ATTRS, max_nodes=2_000_000):
    """{attribute path: token} over the Python objects reachable from `root`:
    scalars by value, everything else by identity (a rebound attribute shows
    up; tensors by identity only — their contents are `checksums`)."""
    seen, state = set(), {}
    stack = [(name, root)]
    nodes = 0
    while stack:
        path, obj = stack.pop()
        nodes += 1
        if nodes > max_nodes:
            break
        if isinstance(obj, _SCALARS):
            state[path] = ("=", obj)
            continue
        state[path] = ("id", type(obj).__name__, id(obj))
        if id(obj) in seen or isinstance(obj, torch.Tensor) or isinstance(obj, type) \
                or isinstance(obj, (torch.cuda.Stream, torch.cuda.Event, torch.device, torch.dtype,
                                    torch.Generator)):
            continue
        seen.add(id(obj))
        if isinstance(obj, dict):
            state[path + "#keys"] = ("=", len(obj))
            for key, value in list(obj.items()):
                stack.append((f"{path}[{key!r}]", value))
        elif isinstance(obj, (list, tuple)):
            state[path + "#len"] = ("=", len(obj))
            for index, value in enumerate(list(obj)):
                stack.append((f"{path}[{index}]", value))
        elif hasattr(obj, "__dict__") and not callable(obj) or isinstance(obj, torch.nn.Module):
            try:
                items = list(vars(obj).items())
            except TypeError:
                items = []
            for key, value in items:
                if key not in skip:
                    stack.append((f"{path}.{key}", value))
    return state


def host_changes(before, after, *, limit=400):
    """Attribute paths whose token changed, appeared or vanished."""
    out = []
    for path in sorted(set(before) | set(after)):
        if before.get(path) != after.get(path):
            out.append((path, before.get(path), after.get(path)))
            if len(out) >= limit:
                break
    return out
