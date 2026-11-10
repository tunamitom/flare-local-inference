#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0
"""W11 post-window measurements (radiance-mechanics review §2, all approved).

Run against the spare card 11:00.0 under /tmp/gpu-spare.lock with a private
kernel cache. Three subcommands:

1. census   — node census of one shipped decode round graph, grouped by chain
              family. Decoded read-only from the exported package's captured
              graphs (the P2-NODE-ATTRIBUTION method): no GPU needed.
2. pack-chain — captured-chain timing of the candidate pack repeated 24x in one
              CUDA graph per form (shipped 8 kernels, fused 1), unprofiled
              replay timing, bit-equal outputs, node counts from the graph
              (CUDAGraph.debug_dump), not from Python counters (T14).
3. compaction — the same for selected_local_records with and without out=
              (the kept copy-count elimination).

Nothing here touches a collective: C2 holds by construction.
"""

import argparse
import ctypes
import ctypes.util
import json
import re
import struct
import sys
from pathlib import Path

# -- minimal protobuf wire decoding (helios/proto/package.proto shapes) ------
# Field numbers read from helios/proto/package.proto: ProtectedPackage,
# SignedRank, RankBody, Graph, Function, ExecutableResource, PayloadSlice and
# flare.package LocationSet/PayloadLocation. Node/Edge use the generated
# ignition.preparation.runtime_package_pb2.


def _wire(data: bytes):
    """Decode one protobuf message into {field: [(wiretype, value), ...]}."""
    fields = {}
    position = 0
    while position < len(data):
        key, position = _varint(data, position)
        number, wiretype = key >> 3, key & 7
        if wiretype == 0:
            value, position = _varint(data, position)
        elif wiretype == 2:
            size, position = _varint(data, position)
            value = data[position:position + size]
            position += size
        elif wiretype == 5:
            value = data[position:position + 4]
            position += 4
        elif wiretype == 1:
            value = data[position:position + 8]
            position += 8
        else:
            raise ValueError(f"unsupported wiretype {wiretype}")
        fields.setdefault(number, []).append((wiretype, value))
    return fields


def _varint(data, position):
    result = 0
    shift = 0
    while True:
        byte = data[position]
        position += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, position
        shift += 7


def _first(fields, number, default=b""):
    values = fields.get(number)
    return values[0][1] if values else default


def _first_int(fields, number):
    return _first(fields, number, 0)


def _parse_graph(data):
    message = _wire(data)
    return {"name": _first(message, 1).decode(), "resource": _first_int(message, 3),
            "node_count": _first_int(message, 4), "edge_count": _first_int(message, 5)}


def _parse_function(data):
    message = _wire(data)
    return {"module": _first(message, 1).decode(), "symbol": _first(message, 2).decode()}


def _parse_resource(data):
    message = _wire(data)
    stored = _wire(_first(message, 3))
    return {"kind": _first_int(message, 2), "payload_id": _first(stored, 1),
            "offset": _first_int(stored, 2), "nbytes": _first_int(stored, 3),
            "decoded_nbytes": _first_int(message, 4)}


def _zstd_decompress(frame: bytes, decoded_nbytes: int) -> bytes:
    library = ctypes.util.find_library("zstd")
    if library is None:
        raise RuntimeError("libzstd not found (the census decodes zstd graph frames)")
    zstd = ctypes.CDLL(library)
    zstd.ZSTD_decompress.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                                     ctypes.c_void_p, ctypes.c_size_t]
    zstd.ZSTD_decompress.restype = ctypes.c_size_t
    zstd.ZSTD_isError.argtypes = [ctypes.c_size_t]
    zstd.ZSTD_isError.restype = ctypes.c_uint
    output = (ctypes.c_char * decoded_nbytes)()
    written = zstd.ZSTD_decompress(output, decoded_nbytes, frame, len(frame))
    if zstd.ZSTD_isError(written):
        raise RuntimeError(f"zstd decompression failed: {written}")
    return bytes(output[:written])


FAMILIES = (
    ("collectives", re.compile(r"AllGather|AllReduce|ReduceScatter|Oneshot|RingGather|"
                               r"Gather|Broadcast|Exchange|Barrier", re.I)),
    ("attention", re.compile(r"mla|attn|attention|softmax|flash", re.I)),
    ("experts", re.compile(r"moe|expert|gemm|cutlass|fused_|w4a16|fp4|fp8", re.I)),
    ("candidate-glue", re.compile(r"pack|unpack|candidate|topk|records|_write_|_compact|"
                                  r"_count|_select|_restore|shortlist", re.I)),
    ("state", re.compile(r"cache|kv|page|hidden|commit|context", re.I)),
    ("norm-residual", re.compile(r"norm|residual|rms|_mix|layer_", re.I)),
    ("sampling", re.compile(r"sample|draft_scores|multinomial|rand", re.I)),
    ("copy-memset", re.compile(r"copy|memset|fill|zero|where|add_mul|functor", re.I)),
)


def _family(symbol):
    for name, pattern in FAMILIES:
        if pattern.search(symbol):
            return name
    return "other"


def census(args):
    from safetensors import safe_open

    from ignition.preparation import runtime_package_pb2 as proto

    package = Path(args.package)
    root = package if package.is_dir() else package.parent
    with safe_open(str(package if package.is_file() else package / "model.safetensors"),
                   framework="np") as primary:
        manifest = bytes(primary.get_tensor("flare.manifest").tobytes())
        locations = bytes(primary.get_tensor("flare.locations").tobytes())
    envelope = _wire(manifest)
    locations_map = {}
    for _, value in _wire(locations).get(2, []):
        entry = _wire(value)
        locations_map[_first(entry, 1)] = (_first(entry, 2).decode(), _first(entry, 3).decode())
    payloads = {}
    decoded = []
    for _, value in envelope.get(7, []):  # ProtectedPackage.ranks = SignedRank
        signed = _wire(value)
        if _first(signed, 2, 0) != args.rank:  # proto3 omits rank 0
            continue
        rank = _wire(_first(signed, 3))  # RankBody
        functions = [_parse_function(value) for _, value in rank.get(10, [])]
        resources = [_parse_resource(value) for _, value in rank.get(8, [])]

        def payload_bytes(payload_id):
            if payload_id not in payloads:
                filename, tensor = locations_map[payload_id]
                with safe_open(str(root / filename), framework="np") as store:
                    payloads[payload_id] = bytes(store.get_tensor(tensor).tobytes())
            return payloads[payload_id]

        rows = []
        for _, value in rank.get(11, []):  # RankBody.graphs
            graph = _parse_graph(value)
            resource = resources[graph["resource"]]
            stored = payload_bytes(resource["payload_id"])[resource["offset"]:
                                                           resource["offset"] + resource["nbytes"]]
            stream = _zstd_decompress(stored, resource["decoded_nbytes"])
            nodes = []
            cursor = 0
            for _ in range(graph["node_count"]):
                size, = struct.unpack_from("<I", stream, cursor)
                cursor += 4
                node = proto.Node.FromString(stream[cursor:cursor + size])
                cursor += size
                nodes.append(node)
            kinds, families = {}, {}
            for node in nodes:
                kind = node.WhichOneof("operation")
                kinds[kind] = kinds.get(kind, 0) + 1
                if kind == "kernel":
                    symbol = functions[node.kernel.function]["symbol"]
                    family = _family(symbol)
                    families[family] = families.get(family, 0) + 1
            rows.append({"graph": graph["name"], "nodes": graph["node_count"],
                         "edges": graph["edge_count"], "kinds": kinds, "families": families})
        rows.sort(key=lambda row: -row["nodes"])
        decoded.append(rows)
    for rows in decoded:
        for row in (rows if args.graph == "all" else
                    [row for row in rows if row["graph"] == args.graph]):
            print(json.dumps(row, sort_keys=True))
        print(f"# graphs decoded: {len(rows)}", file=sys.stderr)
        print(f"# total nodes: {sum(row['nodes'] for row in rows)}", file=sys.stderr)
    if not decoded:
        print(f"# no rank body matched rank {args.rank} (top-level fields: {sorted(envelope)})",
              file=sys.stderr)
    return 0


# -- captured-chain timing (spare card, unprofiled replay) -----------------

def _capture(device, body, warmup=3):
    import torch

    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            body()
    torch.cuda.current_stream(device).wait_stream(stream)
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        body()
    return graph


def _time(graph, device, rounds=200):
    import time

    import torch

    torch.cuda.synchronize(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(rounds):
        graph.replay()
    end.record()
    torch.cuda.synchronize(device)
    return start.elapsed_time(end) / rounds


def pack_chain(args):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import torch

    from ignition.components.distributed_topk import pack_score_id_pairs

    device = torch.device(args.device)
    rows, topk, repeat, world = args.rows, args.topk, args.repeat, 8
    indices = torch.randint(-2, 512, (rows, topk), device=device, dtype=torch.int32)
    scores = torch.rand(rows, topk, device=device)
    rank = torch.tensor([3], dtype=torch.int64, device=device)
    forms = {}
    for form in ("shipped", "fused"):
        local = torch.zeros(rows, topk, 2, device=device)

        def body(local=local):
            for _ in range(repeat):
                if form == "shipped":
                    live = indices >= 0
                    local[:, :, 0].copy_(torch.where(live, scores, -torch.inf))
                    local.view(torch.int32)[:, :, 1].copy_(
                        torch.where(live, indices.long() * world + rank, -1))
                else:
                    pack_score_id_pairs(local, indices, scores, rank, world)

        graph = _capture(device, body)
        elapsed = _time(graph, device, args.rounds)
        graph.debug_dump(args.dump_prefix + f".{form}.dot")
        forms[form] = (local.clone(), elapsed)
    equal = torch.equal(forms["fused"][0].view(torch.int32), forms["shipped"][0].view(torch.int32))
    print(json.dumps({"rows": rows, "topk": topk, "repeat": repeat, "rounds": args.rounds,
                      "shipped_ms_per_chain": forms["shipped"][1],
                      "fused_ms_per_chain": forms["fused"][1],
                      "shipped_us_per_call": forms["shipped"][1] * 1000 / repeat,
                      "fused_us_per_call": forms["fused"][1] * 1000 / repeat,
                      "bit_equal": equal,
                      "node_dumps": f"{args.dump_prefix}.{{shipped,fused}}.dot"},
                     sort_keys=True))
    return 0 if equal else 1


def compaction(args):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import torch

    from ignition.components.distributed_topk import selected_local_records

    device = torch.device(args.device)
    rows, topk, repeat = args.rows, args.topk, args.repeat
    selected = torch.randint(-1, 4096, (rows, topk), device=device, dtype=torch.int32)
    positions = torch.full((rows,), 4095, device=device, dtype=torch.int64)
    pages = torch.randint(-1, 64, (rows, 32), device=device, dtype=torch.int32)
    retained, counts = torch.zeros_like(selected), torch.zeros(rows, device=device, dtype=torch.int32)
    outputs = {}
    for form in ("allocated", "out"):
        def body(form=form):
            for _ in range(repeat):
                if form == "allocated":
                    physical, length = selected_local_records(selected, positions, pages,
                                                              world=8, rank=3, page_size=128)
                    outputs[form] = (physical.clone(), length.clone())
                else:
                    outputs[form] = selected_local_records(selected, positions, pages, world=8,
                                                           rank=3, page_size=128, out=retained,
                                                           count_out=counts)

        graph = _capture(device, body)
        outputs[form] = _time(graph, device, args.rounds), outputs.get(form)
        graph.debug_dump(args.dump_prefix + f".{form}.dot")
    equal = torch.equal(outputs["out"][1][0], outputs["allocated"][1][0]) and \
        torch.equal(outputs["out"][1][1], outputs["allocated"][1][1])
    print(json.dumps({"rows": rows, "topk": topk, "repeat": repeat, "rounds": args.rounds,
                      "allocated_ms_per_chain": outputs["allocated"][0],
                      "out_ms_per_chain": outputs["out"][0],
                      "bit_equal": bool(equal),
                      "node_dumps": f"{args.dump_prefix}.{{allocated,out}}.dot"},
                     sort_keys=True))
    return 0 if equal else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    census_parser = commands.add_parser("census")
    census_parser.add_argument("--package", required=True)
    census_parser.add_argument("--rank", type=int, default=0)
    census_parser.add_argument("--graph", default="all")
    census_parser.set_defaults(handler=census)
    for name, handler in (("pack-chain", pack_chain), ("compaction", compaction)):
        timing = commands.add_parser(name)
        timing.add_argument("--device", default="cuda:0")
        timing.add_argument("--rows", type=int, default=32)
        timing.add_argument("--topk", type=int, default=2048)
        timing.add_argument("--repeat", type=int, default=24)
        timing.add_argument("--rounds", type=int, default=200)
        timing.add_argument("--dump-prefix", default=f"/tmp/w11-{name}")
        timing.set_defaults(handler=handler)
    arguments = parser.parse_args()
    return arguments.handler(arguments)


if __name__ == "__main__":
    sys.exit(main())
