//! Request-local speculative acceptance and observed copy proposals.

use crate::{Result, execution::BATCH_CAPACITIES};

pub const WIDTHS: [usize; 8] = [1, 2, 3, 4, 5, 6, 7, 8];

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ProposalSource {
    #[default]
    Mtp,
    Copy,
}

impl ProposalSource {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "mtp" => Ok(Self::Mtp),
            "copy" => Ok(Self::Copy),
            _ => Err("speculative_proposal_source must be mtp or copy".into()),
        }
    }
}

/// Host-validated continuation and masks for the corresponding committed prefix.
#[derive(Clone, Debug)]
pub struct CopyProposal {
    pub consumed: u64,
    pub tokens: [i64; 7],
    pub count: usize,
    pub grammar_masks: Vec<i32>,
}

pub fn graph_width_masks(widths: &[usize], fixed: Option<usize>, capacity: usize) -> [u8; 7] {
    let mask = fixed.map_or_else(
        || {
            widths
                .iter()
                .fold(0, |mask, width| mask | (1 << (width - 1)))
        },
        |width| 1 << (width - 1),
    );
    BATCH_CAPACITIES.map(|batch| if batch <= capacity { mask } else { 0 })
}

pub fn graph_catalog(widths: &[usize], fixed: Option<usize>, capacity: usize) -> [Vec<usize>; 7] {
    graph_width_masks(widths, fixed, capacity)
        .map(|mask| (1..=8).filter(|w| mask & (1 << (w - 1)) != 0).collect())
}

pub fn validate_widths(widths: &[usize]) -> Result<()> {
    if !widths.is_empty()
        && (widths[0] != 1
            || widths.windows(2).any(|w| w[0] >= w[1])
            || widths.iter().any(|w| !WIDTHS.contains(w)))
    {
        return Err(
            "speculative widths must be sorted unique integers in 1..=8 including 1".into(),
        );
    }
    Ok(())
}

/// Conditional acceptance of one draft depth given that every shallower draft was accepted.
/// Starts from a 0.5 prior worth two observations with bounded effective sample
/// weight. After saturation, each outcome has weight `1 / DEPTH_MEMORY`. `observed` marks a
/// depth that this profile, or the population it was seeded from, has verified at least once.
#[derive(Clone, Copy)]
struct DepthAcceptance {
    probability: f64,
    weight: f64,
    observed: bool,
}
impl Default for DepthAcceptance {
    fn default() -> Self {
        Self {
            probability: 0.5,
            weight: 2.0,
            observed: false,
        }
    }
}
impl DepthAcceptance {
    fn observe(&mut self, accepted: bool) {
        self.weight = (self.weight + 1.0).min(DEPTH_MEMORY);
        self.probability += (if accepted { 1.0 } else { 0.0 } - self.probability) / self.weight;
        self.observed = true;
    }
}
const DEPTH_MEMORY: f64 = 128.0;
/// Conditional acceptance belongs to one logical request and proposal source.
#[derive(Clone, Copy, Default)]
pub struct AcceptanceProfile {
    depths: [[DepthAcceptance; 7]; 2],
    rounds: u64,
}

impl AcceptanceProfile {
    /// Depth zero is the first draft after the pending target token.
    pub fn probability(&self, source: ProposalSource, depth: usize) -> Option<f64> {
        self.depths[source as usize]
            .get(depth)
            .map(|d| d.probability)
    }

    /// A depth nobody has verified carries the deepest verified depth's conditional, not the
    /// prior: a batch that only ever verifies one draft would otherwise value every wider width
    /// at 0.5 per extra draft and never widen, though block-drafter conditionals stay near flat
    /// with depth.
    pub fn expected_tokens(&self, source: ProposalSource, width: usize) -> f64 {
        let mut chain = 1.0;
        let mut total = f64::from(width != 0);
        let mut carried = None;
        for depth in &self.depths[source as usize][..width.saturating_sub(1).min(7)] {
            let probability = if depth.observed { depth.probability } else { carried.unwrap_or(depth.probability) };
            if depth.observed {
                carried = Some(depth.probability);
            }
            chain *= probability;
            total += chain;
        }
        total
    }

    pub fn rounds(&self) -> u64 {
        self.rounds
    }

    /// A new request's profile: this population's conditional acceptance, held with the
    /// default prior's weight so the request's own observations take over quickly.
    pub fn seeded(&self) -> Self {
        let prior = DepthAcceptance::default().weight;
        Self {
            depths: self.depths.map(|source| source.map(|depth| DepthAcceptance {
                probability: depth.probability,
                weight: depth.weight.min(prior),
                observed: depth.observed,
            })),
            rounds: 0,
        }
    }

    /// Only the accepted prefix and an explicitly observed rejection are samples.
    /// `copy_mask` identifies the proposal source at each evaluated draft depth.
    pub(super) fn observe(
        &mut self,
        offered: usize,
        accepted: usize,
        rejected: bool,
        copy_mask: u8,
    ) {
        debug_assert!(accepted <= offered && offered <= 7);
        if offered == 0 {
            return;
        }
        for depth in 0..accepted {
            self.depths[usize::from(copy_mask & (1 << depth) != 0)][depth].observe(true);
        }
        if rejected && accepted < offered {
            self.depths[usize::from(copy_mask & (1 << accepted) != 0)][accepted].observe(false);
        }
        self.rounds = self.rounds.saturating_add(1);
    }
}

/// Copies only observed continuation tokens; the device guards its consumed-position epoch.
pub fn copy_proposal(history: &[u32], maximum: usize) -> ([i64; 7], usize) {
    let mut output = [-1; 7];
    let maximum = maximum.min(7);
    for length in [8, 4] {
        if history.len() <= length || maximum == 0 {
            continue;
        }
        let suffix = history.len() - length;
        for position in (suffix.saturating_sub(4096)..suffix).rev() {
            if history[position..position + length] != history[suffix..] {
                continue;
            }
            let continuation = &history[position + length..];
            let count = maximum.min(continuation.len());
            for (target, token) in output.iter_mut().zip(continuation.iter()).take(count) {
                *target = i64::from(*token);
            }
            if count > 0 {
                return (output, count);
            }
        }
    }
    (output, 0)
}

/// How many tokens host-width speculative rounds of each batch and width emitted per lane against
/// what the acceptance profiles predicted at completion. Profiles learn per request from offered
/// drafts and miss what a shape does to them (seeded or stale profiles, a batch whose requests
/// accept less), so the policy scales a shape's predictions by its realized ratio.
#[derive(Clone, Copy)]
pub struct RealizedTokens {
    sums: [[(f64, f64); 8]; BATCH_CAPACITIES.len()],
}

impl Default for RealizedTokens {
    fn default() -> Self {
        Self { sums: [[(0.0, 0.0); 8]; BATCH_CAPACITIES.len()] }
    }
}

impl RealizedTokens {
    /// Pseudo-tokens at ratio one, the decay per observed lane, and the ratio's bounds.
    const PRIOR: f64 = 16.0;
    const DECAY: f64 = 0.995;
    const BOUNDS: (f64, f64) = (0.5, 1.25);
    /// Per round of the batch at another width: a half-life of about 700 rounds, ten seconds of
    /// concurrency-4 decode.
    const AGE: f64 = 0.999;

    fn index(batch: usize, width: usize) -> Option<(usize, usize)> {
        let b = BATCH_CAPACITIES.iter().position(|&capacity| capacity == batch)?;
        (1..=8).contains(&width).then_some((b, width - 1))
    }

    pub fn observe(&mut self, batch: usize, width: usize, predicted: f64, emitted: f64) {
        let Some((b, w)) = Self::index(batch, width) else { return };
        if !(predicted.is_finite() && emitted.is_finite()) || predicted <= 0.0 {
            return;
        }
        let (realized, expected) = &mut self.sums[b][w];
        *realized = *realized * Self::DECAY + emitted;
        *expected = *expected * Self::DECAY + predicted;
    }

    /// One round of `batch` at `width` completed: the batch's other widths age toward the prior.
    /// A ratio learned from a few unlucky rounds, or against predictions the profiles have since
    /// corrected, would otherwise keep a shape unchosen and so never be re-observed.
    pub fn age_others(&mut self, batch: usize, width: usize) {
        let Some((b, w)) = Self::index(batch, width) else { return };
        for (other, (realized, expected)) in self.sums[b].iter_mut().enumerate() {
            if other != w {
                *realized *= Self::AGE;
                *expected *= Self::AGE;
            }
        }
    }

    pub fn ratio(&self, batch: usize, width: usize) -> f64 {
        let Some((b, w)) = Self::index(batch, width) else { return 1.0 };
        let (realized, expected) = self.sums[b][w];
        ((realized + Self::PRIOR) / (expected + Self::PRIOR)).clamp(Self::BOUNDS.0, Self::BOUNDS.1)
    }
}

#[cfg(test)]
mod realized_tests {
    use super::RealizedTokens;

    #[test]
    fn a_shape_that_over_promises_is_discounted_and_unseen_shapes_stay_neutral() {
        let mut realized = RealizedTokens::default();
        assert_eq!(realized.ratio(8, 3), 1.0);
        for _ in 0..400 {
            realized.observe(8, 3, 2.8, 2.2);
        }
        let ratio = realized.ratio(8, 3);
        assert!((ratio - 2.2 / 2.8).abs() < 0.02, "{ratio}");
        assert_eq!(realized.ratio(8, 1), 1.0);
        assert_eq!(realized.ratio(4, 3), 1.0);
        for _ in 0..2000 {
            realized.observe(4, 2, 1.0, 5.0);
        }
        assert_eq!(realized.ratio(4, 2), 1.25);
        for _ in 0..8000 {
            realized.age_others(8, 1);
        }
        assert!((realized.ratio(8, 3) - 1.0).abs() < 0.02, "{}", realized.ratio(8, 3));
        realized.observe(3, 2, 1.0, 1.0);
        realized.observe(8, 9, 1.0, 1.0);
        realized.observe(8, 2, f64::NAN, 1.0);
        assert_eq!(realized.ratio(8, 2), 1.0);
    }
}


#[cfg(test)]
mod acceptance_tests {
    use super::{AcceptanceProfile, ProposalSource::Mtp};

    #[test]
    fn unverified_depths_carry_the_deepest_verified_conditional() {
        let mut profile = AcceptanceProfile::default();
        assert_eq!(profile.expected_tokens(Mtp, 3), 1.75);
        for round in 0..400 {
            profile.observe(1, usize::from(round % 4 != 0), true, 0);
        }
        let first = profile.probability(Mtp, 0).unwrap();
        assert!((first - 0.75).abs() < 0.1, "{first}");
        let expected = 1.0 + first + first * first;
        assert!((profile.expected_tokens(Mtp, 3) - expected).abs() < 1e-12);
        let seeded = profile.seeded();
        assert!((seeded.expected_tokens(Mtp, 3) - expected).abs() < 1e-12);
        profile.observe(2, 1, true, 0);
        let first = profile.probability(Mtp, 0).unwrap();
        let second = profile.probability(Mtp, 1).unwrap();
        let expected = 1.0 + first + first * second + first * second * second;
        assert!((profile.expected_tokens(Mtp, 4) - expected).abs() < 1e-12);
    }
}
