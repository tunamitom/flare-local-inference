# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0
"""consult18 stage D: the two-half overlapped prefill (CPU twin; lanes, no streams)."""

import pytest

torch = pytest.importorskip("torch")

from tests.recipes.test_hy4_w6_cp import (  # noqa: E402,F401
    fresh_decoders, golden_tokens, rel_error, tiny_available, worlds,
)
from tests.recipes.test_hy4_w7_wiring import run_wired  # noqa: E402

from ignition.recipes.hy4.wiring import prefill_shape  # noqa: E402

HALF = 8


def _scenario(mode):
    def body(decoder):
        decoder.enable_lanes(max_q_rows=2 * HALF)
        _, tokens = golden_tokens()
        batch = torch.tensor([tokens[:2 * HALF]], dtype=torch.long)
        if mode == "overlap":
            logits = decoder.forward_overlapped(batch, half_shape=prefill_shape(1, HALF),
                                                logits_rows="all")
            decoder.append([0], 2 * HALF)
        else:
            first = decoder.forward(batch[:, :HALF], logits_rows="all")
            decoder.append([0], HALF)
            second = decoder.forward(batch[:, HALF:], logits_rows="all")
            decoder.append([0], HALF)
            logits = torch.cat((first, second), dim=1)
        step = decoder.decode_step(torch.tensor([[7]], dtype=torch.long), request_rows=[0])
        return [logits, step]
    return body


@pytest.mark.parametrize("world", (1, 2, 8))
def test_overlapped_halves_equal_two_sequential_halves(worlds, world):
    """Lane interleaving changes nothing: the overlapped chunk equals the two
    halves run one after the other (bit-exact on the CPU twin), and the decode
    step after it reads the same cache."""
    from ignition.recipes.hy4.collectives import SimulatedGroup

    def run(mode):
        if world == 1:
            return [_scenario(mode)(decoder) for decoder in fresh_decoders(worlds[1])]
        return run_wired(worlds[world], _scenario(mode), group=SimulatedGroup(world))

    for sequential, overlapped in zip(run("sequential"), run("overlap")):
        for want, got in zip(sequential, overlapped):
            assert torch.equal(got, want), f"world {world}: rel {rel_error(got, want)}"


def test_overlapped_chunk_is_close_to_the_unsplit_chunk(worlds):
    def body(decoder):
        decoder.enable_lanes(max_q_rows=2 * HALF)
        _, tokens = golden_tokens()
        batch = torch.tensor([tokens[:2 * HALF]], dtype=torch.long)
        return decoder.forward_overlapped(batch, half_shape=prefill_shape(1, HALF), logits_rows="last")

    def whole(decoder):
        _, tokens = golden_tokens()
        return decoder.forward(torch.tensor([tokens[:2 * HALF]], dtype=torch.long), logits_rows="last")

    got = body(fresh_decoders(worlds[1])[0])
    want = whole(fresh_decoders(worlds[1])[0])
    assert rel_error(got, want) <= 2e-5
    assert torch.equal(got.argmax(-1), want.argmax(-1))


def test_lane_buffers_are_separate_and_lane_one_needs_them(worlds):
    from ignition.recipes.hy4.lanes import use_lane

    decoder = fresh_decoders(worlds[1])[0]
    cell = next(layer.attention.selection for layer in decoder.weights.layers
                if layer.attention.selection.kind == "full")
    saved = cell.lane_caches
    cell.lane_caches = ()       # the module's weights (and cells) are shared across tests
    with use_lane(1), pytest.raises(RuntimeError, match="no buffers for lane 1"):
        cell.buffers(rows=2)
    cell.lane_caches = saved
    decoder.enable_lanes(max_q_rows=2 * HALF)
    base = cell.buffers()[0]
    with use_lane(1):
        other = cell.buffers()[0]
    assert base.data_ptr() != other.data_ptr()
    with pytest.raises(RuntimeError, match="enable_lanes"):
        fresh_decoders(worlds[1])[0].forward_overlapped(
            torch.zeros((1, 2 * HALF), dtype=torch.long), half_shape=prefill_shape(1, HALF))
