import torch

from ignition.components.width_selection import PreparedWidthSelection


def select(confidence, costs, limit, dtype=torch.int32):
    widths = len(costs)
    conf = torch.zeros((2, widths - 1), dtype=torch.float32)
    conf[0, :len(confidence)] = torch.tensor(confidence)
    cost = torch.zeros((2, widths), dtype=torch.float32)
    cost[0] = torch.tensor(costs)
    lengths = torch.tensor([limit, 3], dtype=dtype)
    selected = torch.zeros(1, dtype=torch.int32)
    PreparedWidthSelection(conf, cost, lengths, selected, widths)()
    assert int(lengths[0]) == int(selected[0]) and int(lengths[1]) == 3
    return int(selected[0])


COSTS = [8.2, 9.0, 9.85, 10.69, 11.79, 12.45, 13.46, 13.81]


def expected_best(confidence, costs, limit):
    survival, total, best = 1., 1., (1., 1)
    rates = [1. / costs[0]]
    for c in confidence[:len(costs) - 1]:
        survival *= c
        total += survival
        rates.append(total / costs[len(rates)])
    return max(range(1, limit + 1), key=lambda w: rates[w - 1])


def test_confident_drafts_verify_deep_and_doubtful_drafts_shallow():
    assert select([0.99] * 7, COSTS, 8) == 8
    assert select([0.2] * 7, COSTS, 8) == expected_best([0.2] * 7, COSTS, 8) == 2
    assert select([0.05] * 7, COSTS, 8) == 1
    mixed = [0.95, 0.9, 0.8, 0.3, 0.2, 0.1, 0.1]
    assert select(mixed, COSTS, 8) == expected_best(mixed, COSTS, 8)


def test_host_limit_caps_the_width_and_is_overwritten_with_the_choice():
    assert select([0.99] * 7, COSTS, 3) == 3
    assert select([0.99] * 7, COSTS, 1, dtype=torch.int64) == 1


def test_unusable_costs_fall_back_to_the_host_width():
    assert select([0.99] * 7, [0.] * 8, 5) == 5
    nan = [float("nan")] * 8
    assert select([0.5] * 7, nan, 4) == 4


def test_unmeasured_widths_are_never_chosen():
    costs = list(COSTS)
    costs[7] = 0.  # width 8 unmeasured
    assert select([0.999] * 7, costs, 8) == 7


from ignition.components.width_selection import PreparedWidthCalibration


def test_calibration_starts_neutral_and_learns_overconfidence_per_reached_depth():
    sums = torch.zeros((7, 2), dtype=torch.float32)
    conf = torch.full((2, 7), 0.9)
    calibration = PreparedWidthCalibration(sums, conf, 7)
    assert torch.allclose(calibration.ratio(), torch.ones(7))
    emitted = torch.tensor([1, 0], dtype=torch.int32)
    for _ in range(200):  # width 4 rounds accepting one draft: depth 1 hit, depth 2 rejected, 3 unreached
        calibration.observe(4, torch.tensor([1, 0], dtype=torch.int32), emitted.clone().fill_(2))
    ratio = calibration.ratio()
    assert ratio[0] > 1.05 and ratio[1] == PreparedWidthCalibration.BOUNDS[0]
    assert torch.allclose(ratio[2:], torch.ones(5))
    before = sums.clone()
    calibration.observe(4, torch.tensor([3, 0], dtype=torch.int32), torch.tensor([0, 0], dtype=torch.int32))
    assert torch.equal(sums, before)  # a round that committed nothing teaches nothing
    calibration.observe(1, torch.tensor([0, 0], dtype=torch.int32), torch.tensor([1, 0], dtype=torch.int32))
    assert torch.equal(sums, before)  # width one offers no draft


def test_calibrated_selection_verifies_shallower_for_an_overconfident_drafter():
    conf = torch.zeros((2, 7)); conf[0] = torch.tensor([0.9, 0.85, 0.85, 0.8, 0.8, 0.8, 0.8])
    cost = torch.zeros((2, 8)); cost[0] = torch.tensor(COSTS)
    widths = []
    for ratio in (1.0, 0.5):
        sums = torch.zeros((7, 2)); sums[:, 1] = 1000.; sums[:, 0] = 1000. * ratio
        lengths, selected = torch.tensor([8, 8], dtype=torch.int32), torch.zeros(1, dtype=torch.int32)
        PreparedWidthSelection(conf, cost, lengths, selected, 8, PreparedWidthCalibration(sums, conf, 7))()
        widths.append(int(selected[0]))
    assert widths[0] > widths[1] >= 1
