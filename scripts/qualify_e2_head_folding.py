#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""R7 gate harness stub: numerics + latency gates for per-head small-M folding.

The folded schedule (ignition/recipes/kimi3/e2_head_folding.py) cannot engage
without a passing gate report. This harness defines the two gates and validates
reports against the contract; it deliberately cannot PRODUCE a report yet — the
measurement core needs the spare-card kernel harness and a kernel rewrite that
W10 does not do (W10 delivers shapes only; W3's kill-review rule: a spare-card
number before any integration).

Gates (both required):

1. numerics: bit-equal decode output at M = 1, 2, 3, 4, 6, 8, folded vs the
   split-K schedule, same inputs, same process.
2. latency: strictly faster than split-K at every gated width, unprofiled on the
   spare card (captures give shares; unprofiled cells price gains — threat T7).

Usage:
    validate a report:   python scripts/qualify_e2_head_folding.py --check-report report.json
    produce a report:    python scripts/qualify_e2_head_folding.py --produce ...  (refuses: harness missing)
"""

import argparse
import json
import sys

sys.path.insert(0, ".")
from ignition.recipes.kimi3.e2_head_folding import GateReport  # noqa: E402


def produce_report(_arguments):
    raise NotImplementedError(
        "the R7 measurement core needs the spare-card kernel harness plus the "
        "wave-scheduler rewrite (a 5.6k-line GEMM change W10 does not make); "
        "no gate report can be produced yet, and the hook stays on the stock "
        "schedule until one exists (no claim)")


def check_report(path):
    with open(path) as stream:
        raw = json.load(stream)
    report = GateReport(bit_equal=raw.get("bit_equal", {}), latency=raw.get("latency", {}))
    print(json.dumps({"path": path, "passed": report.passed, "bit_equal": raw.get("bit_equal", {}),
                      "latency": raw.get("latency", {})}, indent=2))
    return 0 if report.passed else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check-report", metavar="JSON")
    parser.add_argument("--produce", nargs="*", metavar="ARG")
    arguments = parser.parse_args(argv)
    if arguments.check_report:
        return check_report(arguments.check_report)
    if arguments.produce is not None:
        return produce_report(arguments.produce)
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
