# SPDX-FileCopyrightText: Copyright 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""L6 (upstream flare #14): the candidate-merge radix cutoff (``candidate_merge_radix`` plan setting).

CPU coverage of the variant's selection contract and of the radix cutoff kernel against the
default row top-k + threshold + cutoff-count tail, executed by Triton's Python interpreter
(the upstream CUDA battery rides a later wave). The b12x row top-k stands in as torch.topk
over the compacted values: the tail consumes only the K greatest values (its cutoff is their
minimum), so the stand-in is exact for finite and -inf scores whatever the tie order. NaN
score parity depends on the row top-k's NaN ordering and rides the GPU wave.

The parity asserted is the upstream contract: identical selected IDs and their order,
including ties, -inf and signed zeros; rows with at most K candidates copy their compacted
IDs; merges wider than RADIX_MAX_WIDTH keep the default contract.
"""

import inspect
import os
from pathlib import Path
import subprocess
import sys

import torch

from ignition.components import candidate_merge
from ignition.components.candidate_merge import (
    CONTRACT, CONTRACT_RADIX, Query, RADIX_MAX_WIDTH, RadixQuery, merge_query, radix_variant,
)


def test_radix_variant_selection():
    # The plan setting arms the radix path at most RADIX_MAX_WIDTH candidates wide; wider
    # merges keep the default contract (the counts travel in 16-bit fields).
    assert radix_variant(16384, True) is True
    assert radix_variant(16385, True) is False
    assert radix_variant(16384, False) is False
    assert RADIX_MAX_WIDTH == 16384
    assert isinstance(merge_query(2, 16384, 2048, True), RadixQuery)
    assert type(merge_query(2, 16384, 2048, False)) is Query
    # Own contract, own component id: the default query, contract and cache keys are unchanged.
    assert CONTRACT.component_id == "flare.candidate_merge"
    assert CONTRACT_RADIX.component_id == "flare.candidate_merge_radix"
    payload = CONTRACT_RADIX.encode_query(merge_query(2, 4096, 2048, True))
    assert payload["radix"] is True and (payload["rows"], payload["width"], payload["topk"]) == (2, 4096, 2048)
    assert "radix" not in CONTRACT.encode_query(merge_query(2, 4096, 2048, False))


def test_radix_cutoff_matches_the_default_tail():
    # PYTHONPATH pins this tree: the venv's editable install resolves ignition elsewhere.
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run([sys.executable, str(Path(__file__).resolve())],
        env=dict(os.environ, TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(root)),
        text=True, capture_output=True, timeout=900)
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_component_never_reads_the_environment_to_choose_behaviour():
    """Source-text check of the module, stated as such: the gate is the ``radix`` constructor
    argument threaded from the plan setting, and the module must not read the environment to
    choose captured behaviour in any form (the two idioms a source check can pin)."""
    source = inspect.getsource(candidate_merge)
    assert "os.environ" not in source, "source-text check: module source must not contain os.environ"
    assert "getenv" not in source, "source-text check: module source must not contain getenv"


def test_the_upstream_environment_channel_is_not_carried():
    """The upstream knob's environment channel must not appear anywhere under ignition/:
    our gate is the ``candidate_merge_radix`` plan setting and nothing else."""
    root = Path(__file__).resolve().parents[2] / "ignition"
    for path in root.rglob("*.py"):
        assert "FLARE_CANDIDATE_MERGE" not in path.read_text(), f"environment channel reintroduced in {path}"


# -- interpreted parity (runs again in the subprocess above) -------------------

SCORE_KINDS = ("distinct", "ties", "band", "equal", "degenerate")


def _scores(kind, rows, width, generator):
    if kind == "distinct":
        return torch.randn((rows, width), generator=generator)
    if kind == "ties":
        return torch.randint(-3, 4, (rows, width), generator=generator).float()
    if kind == "band":
        return torch.rand((rows, width), generator=generator) * 0.25 + 1.0
    if kind == "equal":
        return torch.full((rows, width), 0.5)
    scores = torch.randint(-1, 2, (rows, width), generator=generator).float()
    special = torch.rand((rows, width), generator=generator)
    scores[special < 0.3] = -torch.inf
    scores[special >= 0.8] = -0.0
    return scores


def _contract_reference(values, ids, lengths, topk):
    """The merge contract in plain torch: the K greatest scores of the compacted row (fewer
    when fewer exist), ties resolved by earlier compacted position, output in compacted order,
    -1 padding. This is what both tails must select."""
    rows, width = values.shape
    output = torch.full((rows, topk), -1, dtype=torch.int32)
    for row in range(rows):
        length = int(lengths[row])
        entries = [(float(values[row, col]), col, int(ids[row, col])) for col in range(length)]
        entries.sort(key=lambda entry: (-entry[0], entry[1]))
        chosen = sorted(entries[:topk], key=lambda entry: entry[1])
        for place, (_, _, identifier) in enumerate(chosen):
            output[row, place] = identifier
    return output


def interpreted_parity():
    import triton
    from ignition.components.candidate_merge import (
        RADIX_BITS, _buffers, _compact, _count, _counts, _radix_counts, _select, _threshold,
    )

    for rows, width, topk in ((1, 16, 4), (3, 40, 8), (2, 1024, 16)):
        q = Query(rows, width, topk)
        scratch = torch.zeros(q.scratch_bytes, dtype=torch.uint8)
        values, idv, lengths, counts, top_values, cutoffs, ties = _buffers(scratch, q)
        grid = (rows, q.tiles, 1)
        for valid in sorted({0, 1, topk - 1, topk, topk + 1, width}):
            for kind in SCORE_KINDS:
                generator = torch.Generator().manual_seed(97 * valid + 3 * rows + len(kind))
                packed = torch.zeros((rows, width, 2), dtype=torch.float32)
                packed.view(torch.int32)[:, :, 1].fill_(-1)
                for row in range(rows):
                    count = max(0, min(width, valid - row * 3))
                    slots = torch.randperm(width, generator=generator)[:count]
                    identifiers = torch.randperm(width, generator=generator)[:count]
                    packed[row, slots, 0] = _scores(kind, 1, count, generator)[0]
                    packed.view(torch.int32)[row, slots, 1] = identifiers.int()
                packed[:, :, 0][packed.view(torch.int32)[:, :, 1] < 0] = -torch.inf
                # Compaction is shared by both tails.
                torch.library.wrap_triton(_count)[grid](packed, counts, width, q.tiles, 512)
                torch.library.wrap_triton(_compact)[grid](packed, counts, values, idv, lengths, width, q.tiles,
                                                          512, triton.next_power_of_2(q.tiles))
                reference = _contract_reference(values, idv, lengths, topk)
                # Default tail: the row top-k stands in as torch.topk (only its K greatest values
                # reach the tail) and copies the compacted IDs with -1 padding for short rows;
                # then threshold, cutoff counts and select.
                default = torch.full((rows, topk), -1, dtype=torch.int32)
                for row in range(rows):
                    length = int(lengths[row])
                    if length > topk:
                        top_values[row, :topk] = torch.topk(values[row, :length], topk).values
                    else:
                        default[row, :length] = idv[row, :length]
                torch.library.wrap_triton(_threshold)[(rows, 1, 1)](top_values, lengths, cutoffs, default,
                                                                    topk, triton.next_power_of_2(topk))
                torch.library.wrap_triton(_counts)[grid](values, lengths, cutoffs, counts, ties,
                                                         width, topk, q.tiles, 512)
                torch.library.wrap_triton(_select)[grid](values, idv, lengths, cutoffs, counts, ties, default,
                                                         width, topk, q.tiles, 512, triton.next_power_of_2(q.tiles))
                # Radix tail: one cutoff program per row, then the unchanged select.
                radix = torch.full((rows, topk), -7, dtype=torch.int32)
                torch.library.wrap_triton(_radix_counts)[(rows, 1, 1)](
                    values, idv, lengths, cutoffs, counts, ties, radix, width, topk, q.tiles, 512,
                    triton.next_power_of_2(q.tiles * 512), RADIX_BITS)
                torch.library.wrap_triton(_select)[grid](values, idv, lengths, cutoffs, counts, ties, radix,
                                                         width, topk, q.tiles, 512, triton.next_power_of_2(q.tiles))
                case = (rows, width, topk, valid, kind)
                assert torch.equal(default, radix), case
                assert torch.equal(radix, reference), case


if __name__ == "__main__":
    interpreted_parity()
    print("interpreted parity ok")
