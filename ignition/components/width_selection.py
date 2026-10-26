"""Device-side verification width selection for one-request speculative rounds.

Local divergence (owner waiver): this restores the earlier SWITCH speculation for the rebased
tree. It composes into a conditional captured-graph operation that upstream has not approved and
must not be upstreamed without the project owner's approval.

A selection prefix proposes the full draft block and leaves each draft position's confidence: the
largest probability of that position's own sampling distribution. For a block drafter that
probability does not depend on the drafts sampled before it, so the survival product
``s_k = c_1 * ... * c_k`` estimates the probability that the first k drafts are all accepted, and a
round verifying width w emits ``E[w] = 1 + s_1 + ... + s_{w-1}`` tokens in expectation. The
selector picks the width with the most expected tokens per millisecond of its round, up to the
host's prefix ceiling, writes it to ``query_lengths`` so the verification body sees one consistent
width, and publishes it in an int32 selector the runtime's conditional node reads.

Drafters are overconfident: measured on MiMo DFlash, a position whose largest probability averages
0.86 is accepted 0.75 of the time. Each verification body therefore folds its outcome into a
per-depth calibration (decayed sums of accepted drafts and of predicted acceptance over the depths
the round reached), and the selector scales each conditional confidence by that depth's
observed-to-predicted ratio before forming the survival product.

Round costs are learned on device: the prefix stamps the global timer before proposing and each
body folds its round's elapsed time into the selected width's moving average. Rank zero's choice
is broadcast so every rank verifies the same width.
"""

import torch
import triton
import triton.language as tl

from ignition.preparation.triton import prepare_triton

from .collectives import PreparedBroadcast

# Round cost priors in milliseconds for widths one through eight (TP8 MiMo one-request rounds);
# each width's estimate moves to its measured cost once the selector runs it.
PRIOR_COSTS_MS = (8.6, 9.2, 9.8, 10.6, 11.2, 11.9, 12.5, 13.2)


@triton.jit
def _now(anchor):
    return tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l,l", [anchor], dtype=tl.int64,
                                     is_pure=False, pack=1)


@triton.jit
def _stamp(Stamp):
    tl.store(Stamp, _now(tl.load(Stamp)))


@triton.jit
def _observe_cost(Stamp, Costs, Width, WIDTHS: tl.constexpr, ALPHA: tl.constexpr, LIMIT_MS: tl.constexpr):
    elapsed = (_now(tl.load(Stamp)) - tl.load(Stamp)).to(tl.float32) * 1e-6
    width = tl.minimum(tl.maximum(tl.load(Width), 1), WIDTHS)
    old = tl.load(Costs + width - 1)
    valid = (elapsed > 0.) & (elapsed < LIMIT_MS)
    tl.store(Costs + width - 1, tl.where(valid, old + ALPHA * (elapsed - old), old))


class PreparedRoundCosts:
    """Per-width round costs in milliseconds, learned from global timer stamps."""

    ALPHA = 0.05
    # Elapsed times beyond this bound (preemption, a stalled peer) do not move an estimate.
    LIMIT_MS = 250.

    def __init__(self, stamp, costs, selected):
        widths = costs.numel()
        if stamp.dtype != torch.int64 or stamp.numel() != 1 or costs.dtype != torch.float32 or not 1 <= widths <= 8:
            raise ValueError("round costs need an int64 stamp and float32 costs per width")
        if selected.dtype != torch.int32 or selected.numel() < 1:
            raise ValueError("round costs read the int32 selected width")
        self.stamp, self.costs, self.selected = stamp, costs, selected
        self.start = prepare_triton(_stamp, (stamp,), (1, 1, 1), num_warps=1)
        self.observe = prepare_triton(_observe_cost, (stamp, costs, selected), (1, 1, 1), num_warps=1,
                                      WIDTHS=widths, ALPHA=self.ALPHA, LIMIT_MS=self.LIMIT_MS)

    @staticmethod
    def allocate(widths, device):
        stamp = torch.zeros(1, dtype=torch.int64, device=device)
        costs = torch.tensor(PRIOR_COSTS_MS[:widths], dtype=torch.float32, device=device)
        return stamp, costs


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
        sums[:, 1].copy_(sums[:, 1] * scale + torch.where(
            reached, self.confidence[0, :drafts].float().clamp(0., 1.), 0.))


class PreparedWidthSelection:
    """Choose row 0's verification width on device and publish rank zero's choice everywhere."""

    def __init__(self, confidence, costs, query_lengths, selected, widths, *, calibration=None, broadcast=None):
        if confidence.dim() != 2 or confidence.shape[1] < widths - 1:
            raise ValueError("width selection needs one confidence per draft position")
        if costs.dim() != 1 or costs.numel() != widths or costs.dtype != torch.float32:
            raise ValueError("width selection needs one float32 cost per width")
        if selected.dtype != torch.int32 or selected.numel() < 1:
            raise ValueError("width selection publishes an int32 selector")
        self.confidence, self.costs, self.query_lengths, self.selected = confidence, costs, query_lengths, selected
        self.widths, self.calibration, self.broadcast = widths, calibration, broadcast
        device = confidence.device
        self.width_ids = torch.arange(1, widths + 1, device=device, dtype=torch.int64)
        self.zero = torch.zeros(1, device=device, dtype=torch.float32)

    def __call__(self):
        drafts = self.confidence[0, :self.widths - 1].float()
        if self.calibration is not None:
            drafts = drafts * self.calibration.ratio()[:self.widths - 1]
        survival = torch.cumprod(drafts.clamp(0., 1.), 0)
        expected = 1. + torch.cat((self.zero, torch.cumsum(survival, 0)))
        costs = self.costs.float()
        usable = torch.isfinite(costs) & (costs > 0.)
        limit = self.query_lengths[0].to(torch.int64).clamp(1, self.widths)
        allowed = (self.width_ids <= limit) & usable
        rate = torch.where(allowed, expected / torch.where(usable, costs, 1.), -1.)
        # Without any usable cost the host's own ceiling stands.
        best = torch.where(allowed.any(), rate.argmax() + 1, limit)
        self.selected[:1].copy_(best.to(torch.int32).view(1))
        if self.broadcast is not None:
            self.broadcast()
        # Every rank verifies rank zero's width, never more than its own ceiling allows.
        chosen = torch.minimum(self.selected[:1].to(torch.int64), limit.view(1)).clamp_min(1)
        self.selected[:1].copy_(chosen.to(torch.int32))
        self.query_lengths[:1].copy_(chosen.to(self.query_lengths.dtype))


class PreparedSelectionPrefix:
    """Propose the full draft block for one request, then choose the width a body verifies.

    ``source`` is a proposal-only round. The host supplies a prefix ceiling instead of query
    lengths, as for device prefix allocation; the prefix copies it into ``query_lengths`` so the
    proposal validator sees the reserved ceiling, then the selector lowers it to the chosen width.
    """

    def __init__(self, source, prefix_limits, confidence_output, confidence, selection, costs):
        o = source.owner
        if source.batch != 1 or not source.proposal_only or source.width < 2:
            raise ValueError("a selection prefix proposes one request's full draft block")
        if confidence.shape != (1, source.width - 1) or confidence.dtype != torch.float32:
            raise ValueError("selection confidence covers the proposal's draft positions")
        if (confidence_output.ndim != 3 or confidence_output.shape[0] != 2
                or confidence_output.shape[1] != o.requests.capacity or confidence_output.shape[2] < source.width - 1
                or confidence_output.dtype != torch.float32 or not confidence_output.is_contiguous()):
            raise ValueError("confidence output requires two contiguous FP32 request frames covering draft depth")
        self.source, self.prefix_limits, self.confidence_output = source, prefix_limits, confidence_output
        self.confidence, self.selection, self.costs = confidence, selection, costs
        self.probabilities = source.probabilities[:, :source.width - 1]

    def __call__(self):
        s = self.source
        o = s.owner
        self.costs.start()
        o.inputs.query_lengths[:1].copy_(self.prefix_limits[:1])
        s.propose(False)
        # Sampling has consumed the whole block's draw reservation whatever width is verified.
        s.commit_draft_rng()
        self.confidence.copy_(torch.where(s.active[:, None], self.probabilities.amax(-1).clamp(0., 1.), 0.))
        self.confidence_output.index_copy_(0, o.frame.to(torch.int64),
            torch.nn.functional.pad(self.confidence, (0, self.confidence_output.shape[2] - self.confidence.shape[1],
                                                      0, self.confidence_output.shape[1] - 1)).unsqueeze(0))
        self.selection()


class PreparedSelectedVerification:
    """A conditional body: verify the prefix's proposal at this body's width, then calibrate."""

    def __init__(self, source, target, *, calibration=None, costs=None):
        if target.batch != 1 or target.generate_proposals or target.width > source.width:
            raise ValueError("a selected body verifies the prefix's proposal at a width it covers")
        self.source, self.target, self.calibration, self.costs = source, target, calibration, costs

    def __call__(self):
        s, t = self.source, self.target
        for source, destination in ((s.before, t.before), (s.base_offset, t.base_offset)):
            destination.copy_(source)
        for source, destination in ((s.candidates, t.candidates), (s.drafts, t.drafts),
                                    (s.probabilities, t.probabilities), (s.valid, t.valid)):
            if destination.numel():
                destination.copy_(source[:, :destination.shape[1]])
        t.active.copy_(s.active)
        t.verify(s.active)
        commit = t.operations[3]
        if t.width != s.width:
            # The host planned one round of the switch's full width: every record it completes reports
            # that width, whichever body ran (field 15 is the verification width).
            records = commit.records.index_select(0, commit.frame.long())
            width = records[..., 15]
            records[..., 15] = torch.where(width == t.width, s.width, width)
            commit.records.index_copy_(0, commit.frame.long(), records)
        if self.calibration is not None:
            self.calibration.observe(t.width, commit.accepted_drafts, commit.emitted_count)
        if self.costs is not None:
            self.costs.observe()


def selection_broadcast(source, selected, *, buffers=None):
    """Rank zero's selected width, copied to every rank."""
    o = source.owner
    runtime = getattr(o, "result_runtime", None)
    if runtime is None:
        runtime = o.collectives["vocabulary_runtime"]
    return PreparedBroadcast((selected,), runtime=runtime, session=o.session,
        coordinator=o.collectives["coordinator"], name=source.name + ".selected_width", buffers=buffers)
