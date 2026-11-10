#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Allocation-only microbenchmark for the W10 R9 process-local temporary.

STATUS: READY TO RUN, NOT RUN IN W10. The E2 prototype workstream is CPU-only
(no GPU window; W9 holds B+D). Run this post-window on the spare card under the
campaign lock:

    flock /tmp/gpu-spare.lock /home/tom/build/flare-k31exl3/.venv/bin/python \
        scripts/benchmark_e2_allocation.py --device cuda:0 --variant all \
        --rounds 200 --json-out /home/tom/kimi31-exl3/levers/evidence/w10/r9-allocation-microbench.json

What it prices (ALLOCATION ONLY — it touches no collective, no transport, no
kernel fusion):

* reserve/carve cost of the three R9 variants on the decode glue's exchange +
  aggregation staging catalog (dsa.py geometry: 8 decode rows, top-k 2,048,
  32x128 index heads, 96x512 aggregation heads):
    r9d  pinned/upper-buffer pool        (J-3 first choice)
    r9a  single multi-round reservation (fallback 1)
    r9b  two-stage                      (fallback 2)
* steady-state proof: zero allocations per round after warm-up (R9 threat T4),
  and identical element counts across variants (T3's allocation side).
* the union loop's pack/unpack copies with hoisted strides/offsets
  (copy-count elimination), still allocation-only.

The acceptance threshold for the allocation work is +/-10% of the shipped
per-round glue allocation cost: a variant that costs more than +10% against the
current per-call allocation baseline fails, and a variant must not change any
byte of output (T3). The results JSON carries the threshold next to every number.

RADIANCE (all-reduce) THRESHOLD — READ BEFORE EXTENDING THIS HARNESS:
R9's real win is a fused exchange+aggregation chain, which touches the collective
("radiance") path. That change needs the radiance-mechanics review BEFORE it can
even be benched, let alone shipped. Until the review signs the mechanics, the
radiance body below refuses to run; when it runs, the same +/-10% threshold
applies to the radiance side: the fused exchange+aggregation all-reduce must be
within +/-10% of the shipped exchange time at every packed row count, with the
bit-equal gate (T3) unchanged. Benchmarks before the review are wasted at best
and misleading at worst (T7: captures give shares; unprofiled cells price gains).
"""

import argparse
import json
import sys
import time

import torch

RADIANCE_THRESHOLD = 0.10  # +/-10%: allocation side AND, after mechanics review, radiance side
ALLOCATION_THRESHOLD = 0.10

# Decode glue exchange + aggregation staging catalog (geometry from
# ignition/recipes/kimi3/dsa.py and components/distributed_topk.py: 8 decode
# rows, top-k 2048, index heads 32x128, aggregation heads 96x512, BF16/FP32/FP8
# fields as prepared there).
CATALOG = (
    ("exchange.index_query", (8, 32, 128), torch.bfloat16),
    ("exchange.index_query_local", (8, 4, 128), torch.bfloat16),
    ("exchange.quantized_query", (8, 576), torch.float8_e4m3fn),
    ("exchange.candidate_packs", (8, 2048, 2), torch.float32),
    ("aggregation.attended", (8, 96, 512), torch.bfloat16),
    ("aggregation.lse", (8, 96), torch.float32),
    ("aggregation.physical", (8, 2048), torch.int32),
    ("aggregation.counts", (8,), torch.int32),
)


def _timed_ms(device, work):
    """Run ``work`` once and return its elapsed milliseconds."""
    if device.type == "cuda":
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        work()
        end.record()
        torch.cuda.synchronize(device)
        return start.elapsed_time(end)
    begin = time.perf_counter()
    work()
    return (time.perf_counter() - begin) * 1e3


def _rounds(device, temporary, rounds, warmup):
    """Carve every catalog field per round; count allocations after warm-up."""
    for name, shape, dtype in CATALOG:
        temporary.reserve(name, shape, dtype)

    def carve_all():
        for name, shape, dtype in CATALOG:
            temporary.carve(name, shape, dtype)

    for _ in range(warmup):
        carve_all()
    allocations_before = temporary.allocations
    elapsed_ms = _timed_ms(device, lambda: [carve_all() for _ in range(rounds)])
    return {
        "rounds": rounds,
        "elapsed_ms": elapsed_ms,
        "us_per_round": elapsed_ms * 1e3 / rounds,
        "steady_state_allocations": temporary.allocations - allocations_before,
        "nbytes": temporary.nbytes,
        "reserved_bytes": temporary.reserved_bytes,
    }


def _baseline_us_per_round(device, rounds, warmup):
    """The shipped behavior: per-round torch.empty_like allocations for the glue."""
    scratch = [torch.empty(shape, dtype=dtype, device=device) for _, shape, dtype in CATALOG]

    def allocate_all():
        nonlocal scratch
        scratch = [torch.empty_like(tensor) for tensor in scratch]

    for _ in range(warmup):
        allocate_all()
    elapsed_ms = _timed_ms(device, lambda: [allocate_all() for _ in range(rounds)])
    return elapsed_ms * 1e3 / rounds


def run_allocation_bench(device, variants, rounds, warmup):
    from ignition.components.process_local_temporary import ProcessLocalTemporary, TemporaryVariant

    baseline = _baseline_us_per_round(device, rounds, warmup)
    results = {"baseline_us_per_round": baseline, "threshold": ALLOCATION_THRESHOLD,
               "threshold_meaning": "variant must cost <= baseline * (1 + 10%) and change no byte (T3)",
               "variants": {}}
    for variant in variants:
        temporary = ProcessLocalTemporary("bench", device, variant=variant,
                                          max_bytes=256 << 20,
                                          pinned=variant == TemporaryVariant.POOL)
        measured = _rounds(device, temporary, rounds, warmup)
        measured["vs_baseline"] = (measured["us_per_round"] / baseline - 1.0) if baseline else 0.0
        measured["passes_allocation_threshold"] = measured["us_per_round"] <= baseline * (1 + ALLOCATION_THRESHOLD)
        results["variants"][variant] = measured
    return results


def run_radiance_bench(device, approval):
    """R9's exchange+aggregation (radiance) change. Gated on mechanics approval."""
    if not approval:
        print("RADIANCE BENCH REFUSED: R9's all-reduce (radiance) change needs the "
              "radiance-mechanics review BEFORE it can be benched. Pass --radiance-approval "
              "<review reference> once that review signs the mechanics. Threshold when it "
              f"runs: +/-{RADIANCE_THRESHOLD:.0%} of the shipped exchange time at every "
              "packed row count, bit-equal output required (T3).", file=sys.stderr)
        return None
    raise NotImplementedError(
        "radiance bench body lands together with the radiance-mechanics approval; "
        f"the approval reference is recorded: {approval!r}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--variant", default="all", choices=("all", "r9d", "r9a", "r9b"))
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--json-out")
    parser.add_argument("--radiance-approval", help="radiance-mechanics review reference; "
                        "without it the radiance bench refuses to run")
    arguments = parser.parse_args(argv)

    device = torch.device(arguments.device)
    variants = ("r9d", "r9a", "r9b") if arguments.variant == "all" else (arguments.variant,)
    results = {"device": str(device), "catalog": [(name, list(shape), str(dtype)) for name, shape, dtype in CATALOG],
               "allocation": run_allocation_bench(device, variants, arguments.rounds, arguments.warmup),
               "radiance": run_radiance_bench(device, arguments.radiance_approval),
               "radiance_threshold": RADIANCE_THRESHOLD}
    text = json.dumps(results, indent=2)
    print(text)
    if arguments.json_out:
        with open(arguments.json_out, "w") as stream:
            stream.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
