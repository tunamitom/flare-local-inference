"""Distribution-derived proposal confidence for drafters without a confidence head."""

from types import SimpleNamespace

import torch

from ignition.components.split_draft import PreparedDistributionConfidence, causal_prefix


def test_confidence_is_each_positions_largest_probability():
    probabilities = torch.tensor([[[0.7, 0.2, 0.1], [0.4, 0.4, 0.2], [0.9, 0.05, 0.05]],
                                  [[0.34, 0.33, 0.33], [1.0, 0.0, 0.0], [0.5, 0.5, 0.0]]])
    output = torch.zeros(2, 2)
    confidence = PreparedDistributionConfidence(SimpleNamespace(probabilities=probabilities, width=3), output)
    confidence.record_embedding(0, torch.zeros(1))
    confidence()
    # Width three publishes two drafts; the third block position is never offered.
    assert torch.equal(output, torch.tensor([[0.7, 0.4], [0.34, 1.0]]))
    assert causal_prefix(output[0].tolist(), 0.25) == 2
    assert causal_prefix(output[1].tolist(), 0.5) == 0


def test_confidence_does_not_read_the_sampled_token():
    probabilities = torch.rand(1, 4, 16)
    probabilities /= probabilities.sum(-1, keepdim=True)
    output = torch.zeros(1, 3)
    source = SimpleNamespace(probabilities=probabilities, width=4, drafts=torch.zeros(1, 3, dtype=torch.long))
    confidence = PreparedDistributionConfidence(source, output)
    confidence()
    before = output.clone()
    source.drafts.fill_(7)
    confidence()
    assert torch.equal(before, output)
