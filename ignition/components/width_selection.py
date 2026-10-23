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

Drafters are overconfident: measured on MiMo DFlash, a position whose largest probability averages
0.86 is accepted 0.75 of the time. Each verification body therefore folds its outcome into a
per-depth calibration (decayed sums of accepted drafts and of predicted acceptance over the
depths the round reached), and the selector scales each conditional confidence by that depth's
observed-to-predicted ratio before forming the survival product.
"""

import torch


class PreparedWidthCalibration:
    """Per-depth acceptance calibration shared by a selector and its verification bodies.

    ``sums[d] = (accepted, predicted)`` decayed over the rounds that reached draft depth d + 1.
    The ratio carries ``PRIOR`` pseudo-observations at ratio one, so zero-initialized storage
    starts uncalibrated, and is clamped so a short streak cannot disable or force a depth.
    """

    PRIOR = 4.
    DECAY = 0.99
    BOUNDS = (0.25, 2.)

    def __init__(self, sums, confidence, depths):
        if sums.shape != (depths, 2) or sums.dtype != torch.float32:
            raise ValueError("calibration needs float32 (accepted, predicted) sums per draft depth")
        self.sums, self.confidence, self.depths = sums, confidence, depths
        self.depth_ids = torch.arange(1, depths + 1, device=sums.device, dtype=torch.int64)

    def ratio(self):
        accepted, predicted = self.sums[:, 0], self.sums[:, 1]
        return ((accepted + self.PRIOR) / (predicted + self.PRIOR)).clamp(*self.BOUNDS)

    def observe(self, width, accepted_drafts, emitted):
        """Fold row 0's outcome of a round that verified ``width`` into the depths it reached."""
        drafts = width - 1
        if drafts < 1:
            return
        accepted = accepted_drafts[0].to(torch.int64)
        live = emitted[0] > 0
        depth = self.depth_ids[:drafts]
        reached = live & (accepted >= depth - 1)
        hit = reached & (accepted >= depth)
        scale = torch.where(reached, self.DECAY, 1.)
        sums = self.sums[:drafts]
        sums[:, 0].copy_(sums[:, 0] * scale + hit.float())
        sums[:, 1].copy_(sums[:, 1] * scale + torch.where(reached, self.confidence[0, :drafts].float().clamp(0., 1.), 0.))


class PreparedWidthSelection:
    """Choose row 0's verification width on device; batch-one rounds only."""

    def __init__(self, confidence, costs, query_lengths, selected, widths, calibration=None):
        if confidence.dim() != 2 or confidence.shape[1] < widths - 1:
            raise ValueError("width selection needs one confidence per draft position")
        if costs.dim() != 2 or costs.shape[1] != widths or costs.dtype != torch.float32:
            raise ValueError("width selection needs one float32 cost per width")
        if selected.dtype != torch.int32 or selected.numel() < 1:
            raise ValueError("width selection publishes an int32 selector")
        self.confidence, self.costs, self.query_lengths, self.selected = confidence, costs, query_lengths, selected
        self.widths, self.calibration = widths, calibration
        device = confidence.device
        self.width_ids = torch.arange(1, widths + 1, device=device, dtype=torch.int64)
        self.zero = torch.zeros(1, device=device, dtype=torch.float32)

    def __call__(self):
        drafts = self.confidence[0, :self.widths - 1].float()
        if self.calibration is not None:
            drafts = drafts * self.calibration.ratio()[:self.widths - 1]
        survival = torch.cumprod(drafts.clamp(0., 1.), 0)
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
    """A conditional body: verify the proposal this graph's prefix published, then calibrate."""

    def __init__(self, verification, calibration=None, commit=None):
        if (calibration is None) != (commit is None):
            raise ValueError("calibration needs the body's commit outputs")
        self.verification, self.calibration, self.commit = verification, calibration, commit

    def __call__(self):
        self.verification(same_graph=True)
        if self.calibration is not None:
            self.calibration.observe(self.verification.target.width, self.commit.accepted_drafts,
                                     self.commit.emitted_count)
