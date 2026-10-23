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
