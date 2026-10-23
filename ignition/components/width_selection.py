"""Device-side verification width selection for one-request speculative rounds.

A proposal graph leaves each draft position's confidence (the largest probability of that
position's own sampling distribution). For a block drafter that probability does not depend on
the drafts sampled before it, so the survival product ``s_k = c_1 * ... * c_k`` estimates the
probability that the first k drafts are all accepted, and a round verifying width w emits
``E[w] = 1 + s_1 + ... + s_{w-1}`` tokens in expectation. The host supplies each width's
estimated round cost at the row's context; the selector picks the width with the most expected
tokens per millisecond among widths the host allows (``query_lengths``), writes it back to
``query_lengths`` so the verification body sees one consistent width, and publishes it in an
int32 selector the runtime's conditional node reads.
"""

import torch


class PreparedWidthSelection:
    """Choose row 0's verification width on device; batch-one rounds only."""

    def __init__(self, confidence, costs, query_lengths, selected, widths):
        if confidence.dim() != 2 or confidence.shape[1] < widths - 1:
            raise ValueError("width selection needs one confidence per draft position")
        if costs.dim() != 2 or costs.shape[1] != widths or costs.dtype != torch.float32:
            raise ValueError("width selection needs one float32 cost per width")
        if selected.dtype != torch.int32 or selected.numel() < 1:
            raise ValueError("width selection publishes an int32 selector")
        self.confidence, self.costs, self.query_lengths, self.selected = confidence, costs, query_lengths, selected
        self.widths = widths
        device = confidence.device
        self.width_ids = torch.arange(1, widths + 1, device=device, dtype=torch.int64)
        self.zero = torch.zeros(1, device=device, dtype=torch.float32)

    def __call__(self):
        drafts = self.confidence[0, :self.widths - 1].float().clamp(0., 1.)
        survival = torch.cumprod(drafts, 0)
        expected = 1. + torch.cat((self.zero, torch.cumsum(survival, 0)))
        costs = self.costs[0].float()
        usable = torch.isfinite(costs) & (costs > 0.)
        limit = self.query_lengths[0].to(torch.int64).clamp(1, self.widths)
        allowed = (self.width_ids <= limit) & usable
        rate = torch.where(allowed, expected / torch.where(usable, costs, 1.), -1.)
        # Without any usable cost the host's own width stands.
        best = torch.where(allowed.any(), rate.argmax() + 1, limit)
        self.query_lengths[:1].copy_(best.to(self.query_lengths.dtype).view(1))
        self.selected[:1].copy_(best.to(torch.int32).view(1))


class PreparedSelectionPrefix:
    """Propose the full draft block, then choose the width the conditional body verifies."""

    def __init__(self, proposal, selection):
        self.proposal, self.selection = proposal, selection

    def __call__(self):
        self.proposal()
        self.selection()


class PreparedSelectedVerification:
    """A conditional body: verify the proposal this graph's prefix published."""

    def __init__(self, verification):
        self.verification = verification

    def __call__(self):
        self.verification(same_graph=True)
