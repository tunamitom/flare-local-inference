"""Device width selection: expected tokens per millisecond, ceilings and per-depth calibration."""

import os

import pytest
import torch

from ignition.components.width_selection import (
    PreparedRoundCosts, PreparedWidthCalibration, PreparedWidthSelection)


def _selection(confidence, costs, ceiling, calibration=None):
    confidence = torch.tensor([confidence], dtype=torch.float32)
    lengths = torch.tensor([ceiling], dtype=torch.int32)
    selected = torch.ones(1, dtype=torch.int32)
    selection = PreparedWidthSelection(confidence, torch.tensor(costs, dtype=torch.float32), lengths, selected,
                                       len(costs), calibration=calibration)
    return selection, lengths, selected


def test_selection_maximizes_expected_tokens_per_millisecond_within_the_ceiling():
    costs = [8.6, 9.2, 9.8, 10.6, 11.2, 11.9, 12.5, 13.2]
    confident = [0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95]
    selection, lengths, selected = _selection(confident, costs, 8)
    selection()
    assert selected.item() == lengths.item() == 8
    doubtful = [0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1]
    selection, lengths, selected = _selection(doubtful, costs, 8)
    selection()
    assert selected.item() == lengths.item() == 2
    selection, lengths, selected = _selection(confident, costs, 3)
    selection()
    assert selected.item() == lengths.item() == 3


def test_selection_without_usable_costs_keeps_the_host_ceiling():
    selection, lengths, selected = _selection([0.9] * 7, [0.] * 8, 5)
    selection()
    assert selected.item() == lengths.item() == 5


def test_calibration_shortens_an_overconfident_drafter():
    depths = 7
    confidence = torch.full((1, depths), 0.95)
    sums = torch.zeros((depths, 2))
    calibration = PreparedWidthCalibration(sums, confidence, depths)
    assert torch.allclose(calibration.ratio(), torch.ones(depths))
    costs = [8.6, 9.2, 9.8, 10.6, 11.2, 11.9, 12.5, 13.2]
    selection, _, selected = _selection([0.95] * 7, costs, 8)
    selection.confidence = confidence
    selection()
    assert selected.item() == 8
    # Every round accepts three of seven drafts: depth four is reached and always rejected.
    accepted = torch.tensor([3], dtype=torch.int32)
    emitted = torch.tensor([4], dtype=torch.int32)
    for _ in range(300):
        calibration.observe(8, accepted, emitted)
    ratio = calibration.ratio()
    assert (ratio[:3] >= 1).all() and ratio[3] == PreparedWidthCalibration.BOUNDS[0]
    assert torch.allclose(ratio[4:], torch.ones(depths - 4))
    selection, _, selected = _selection([0.95] * 7, costs, 8, calibration=calibration)
    selection.confidence = confidence
    selection()
    assert 4 <= selected.item() <= 5


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                    reason="requires FLARE_RUN_CUDA_TESTS=1 and an assigned GPU")
def test_round_costs_learn_the_selected_width_inside_a_captured_graph():
    device = torch.device("cuda", 0)
    stamp, costs = PreparedRoundCosts.allocate(8, device)
    selected = torch.full((1,), 3, dtype=torch.int32, device=device)
    rounds = PreparedRoundCosts(stamp, costs, selected)
    prior = costs.clone()
    graph, stream = torch.cuda.CUDAGraph(), torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream), torch.cuda.graph(graph):
        rounds.start()
        torch.cuda._sleep(1 << 22)
        rounds.observe()
    for _ in range(100):
        graph.replay()
    torch.cuda.synchronize(device)
    assert torch.equal(costs[[0, 1, 3, 4, 5, 6, 7]], prior[[0, 1, 3, 4, 5, 6, 7]])
    assert 0. < costs[2].item() < prior[2].item()
