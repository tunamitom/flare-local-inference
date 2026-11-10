# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""R4 shape only: CUDA-graph decode-step capture, an engineering-only sketch.

R4 in the E2 queue is explicitly low priority per J-3 (its prefill form, decoding
each expert once, is a 1-2 week kernel restructure that needs a spare-card number
first; W3's kill-review rule). W10 therefore delivers an engineering-only sketch:
the capture boundaries a whole-decode-step CUDA graph would need, written down so
the decision can be revisited without rediscovering them. Nothing here captures,
allocates, or claims a gain, and every entry point raises by design.

Sketch (what a decode-step capture would have to hold):

* Scope: one captured graph per offered width (1..8 drafts), replacing the 567
  glue chains and the per-layer launches of one round with a single replay.
  The R9 union must be allocation-free at steady state first (threat T4), because
  a captured graph freezes every address.
* Preparation-only capture (docs/design-review.md principle 2: no recapture at
  serve). Capture happens after the complete shape catalog is tuned and every
  variant warmed (principle 5); the width catalog is finite (graph budget 16),
  so all variants can be captured before serving.
* Determinism: decode is deterministic today and J-2's exactness gate depends on
  it. A replayed step must be bit-identical run to run; any atomic or
  order-dependent accumulation inside the capture (R10b's atomic funnel) makes
  the step non-deterministic and therefore cannot ride an R4 capture without the
  determinism gate.
* Lifetimes: every captured address is persistent storage (principle 6). The
  process-local temporary (R9) is safe to capture only because a rank replays one
  captured graph at a time (ignition/preparation/scratch.py); cross-graph reuse
  needs that invariant re-proven at the step level, not assumed.
* Interaction with the round-time fit: the win is the graph-node floor (about
  0.85 us/node x the removed dispatches, w2-results section 2) minus the replay
  boundary costs; priced only by unprofiled round cells (threat T7).

Not implemented, deliberately: no capture call exists in this file. R4 stays a
sketch until R9's prototype is priced and the E2 bar (J-3: 1.0 ms/round of
harness-measured, deterministic gain) is within reach of the remaining levers.
"""


class DecodeStepCaptureSketch:
    """Engineering-only sketch of a whole-decode-step CUDA graph capture (R4)."""

    LOW_PRIORITY = "J-3: R4 is explicitly low priority inside E2"

    def __init__(self, widths=(1, 2, 3, 4, 6, 8)):
        self.widths = tuple(widths)

    def boundaries(self):
        """The capture boundaries a step graph would need (documentation only)."""
        return {
            "graphs": [f"decode_step_width{width}" for width in self.widths],
            "requires": [
                "R9 steady-state allocation-free glue (threat T4)",
                "bit-equal replay at every width (J-2 determinism gate)",
                "persistent storage for every captured address (principle 6)",
                "preparation-only capture of the complete width catalog (principle 5)",
                "no order-dependent accumulation inside the capture (R10b atomic funnel gated)",
            ],
            "pricing": "unprofiled round cells only; captures give shares, not gains (threat T7)",
            "priority": self.LOW_PRIORITY,
        }

    def capture(self):
        """Never implemented in W10; R4 is a sketch (J-3)."""
        raise NotImplementedError(f"R4 decode-step capture is a sketch only ({self.LOW_PRIORITY})")

    def replay(self):
        """Never implemented in W10; R4 is a sketch (J-3)."""
        raise NotImplementedError(f"R4 decode-step replay is a sketch only ({self.LOW_PRIORITY})")
