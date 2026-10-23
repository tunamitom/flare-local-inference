//! Device-selected verification widths. A switched graph chooses its own width each round from
//! the drafts' survival against per-width costs the host supplies; the host learns the realized
//! tokens per round and the round cost of each realized width per context bucket.

use super::cost::{CONTEXT_BUCKETS, context_bucket};
use crate::execution::GraphKind;

/// Widths a switched graph may choose; bodies beyond the graph's capacity are never chosen.
pub(super) const MAX_WIDTH: usize = 8;
const MEMORY: f64 = 64.0;
/// Relative round-cost growth per verification width when only one width has been observed
/// (MiMo joined rounds at batch one grow by about 0.8 ms on 10.7 ms per width).
const WIDTH_SLOPE: f64 = 0.075;

#[derive(Clone, Copy, Default)]
struct Mean {
    value: f64,
    weight: f64,
}

impl Mean {
    fn observe(&mut self, sample: f64) {
        self.weight = (self.weight + 1.0).min(MEMORY);
        self.value += (sample - self.value) / self.weight;
    }

    fn get(&self) -> Option<f64> {
        (self.weight > 0.0).then_some(self.value)
    }
}

pub(super) struct DeviceWidth {
    pub batch: usize,
    pub capacity: usize,
    tokens: [Mean; CONTEXT_BUCKETS],
    costs: [[Mean; MAX_WIDTH]; CONTEXT_BUCKETS],
}

impl DeviceWidth {
    pub fn new(batch: usize, capacity: usize) -> Option<Self> {
        (batch == 1 && (2..=MAX_WIDTH).contains(&capacity)).then(|| Self {
            batch,
            capacity,
            tokens: Default::default(),
            costs: Default::default(),
        })
    }

    pub fn matches(&self, kind: GraphKind, batch: usize, capacity: usize) -> bool {
        kind == GraphKind::Speculative && batch == self.batch && capacity == self.capacity
    }

    /// A completed round: its realized width, emitted tokens, and GPU nanoseconds.
    pub fn observe(&mut self, position: u64, width: usize, tokens: usize, nanos: u64) {
        if !(1..=self.capacity).contains(&width) || nanos == 0 {
            return;
        }
        let bucket = context_bucket(position);
        self.tokens[bucket].observe(tokens as f64);
        self.costs[bucket][width - 1].observe(nanos as f64 / 1e6);
    }

    /// Tokens a round emits at this context: learned, else the nearest learned bucket, else `prior`.
    pub fn expected_tokens(&self, position: u64, prior: f64) -> f64 {
        let bucket = context_bucket(position);
        nearest(bucket, |b| self.tokens[b].get()).unwrap_or(prior)
    }

    /// Round milliseconds of each width 1..=MAX_WIDTH at this context; zero marks a width the
    /// graph must not choose. Unobserved widths follow the bucket's observed widths, then the
    /// nearest bucket's, then `fallback` (milliseconds of an equivalent host-width round).
    pub fn width_costs(&self, position: u64, fallback: impl Fn(usize) -> Option<f64>) -> [f32; MAX_WIDTH] {
        let bucket = context_bucket(position);
        let observed = nearest(bucket, |b| {
            let points: Vec<_> = (1..=self.capacity)
                .filter_map(|w| self.costs[b][w - 1].get().map(|ms| (w as f64, ms)))
                .collect();
            (!points.is_empty()).then_some(points)
        });
        let mut costs = [0.0f32; MAX_WIDTH];
        for (index, cost) in costs.iter_mut().enumerate().take(self.capacity) {
            let width = index + 1;
            let estimate = match &observed {
                Some(points) => points
                    .iter()
                    .find(|(w, _)| *w as usize == width)
                    .map(|&(_, ms)| ms)
                    .unwrap_or_else(|| extrapolate(points, width as f64)),
                None => fallback(width).unwrap_or(0.0),
            };
            *cost = if estimate.is_finite() && estimate > 0.0 { estimate as f32 } else { 0.0 };
        }
        costs
    }
}

fn nearest<T>(bucket: usize, value: impl Fn(usize) -> Option<T>) -> Option<T> {
    (0..CONTEXT_BUCKETS)
        .filter_map(|b| value(b).map(|v| (b.abs_diff(bucket), std::cmp::Reverse(b), v)))
        .min_by_key(|(distance, longer, _)| (*distance, *longer))
        .map(|(_, _, v)| v)
}

/// Least-squares line through the observed (width, ms) points, or a proportional slope from one.
fn extrapolate(points: &[(f64, f64)], width: f64) -> f64 {
    if let [(w0, c0)] = points {
        return c0 * (1.0 + WIDTH_SLOPE * (width - w0));
    }
    let n = points.len() as f64;
    let (sx, sy) = points.iter().fold((0.0, 0.0), |(x, y), (w, c)| (x + w, y + c));
    let (mx, my) = (sx / n, sy / n);
    let (sxy, sxx) = points.iter().fold((0.0, 0.0), |(xy, xx), (w, c)| {
        (xy + (w - mx) * (c - my), xx + (w - mx) * (w - mx))
    });
    let slope = if sxx > 0.0 { (sxy / sxx).max(0.0) } else { my * WIDTH_SLOPE };
    my + slope * (width - mx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_one_request_rounds_of_drafting_widths_switch() {
        assert!(DeviceWidth::new(1, 8).is_some());
        assert!(DeviceWidth::new(2, 8).is_none() && DeviceWidth::new(1, 1).is_none() && DeviceWidth::new(1, 9).is_none());
        let switch = DeviceWidth::new(1, 8).unwrap();
        assert!(switch.matches(GraphKind::Speculative, 1, 8));
        assert!(!switch.matches(GraphKind::Speculative, 2, 8) && !switch.matches(GraphKind::Verification, 1, 8));
    }

    #[test]
    fn unobserved_costs_follow_the_host_fallback_then_learned_widths() {
        let mut switch = DeviceWidth::new(1, 8).unwrap();
        let costs = switch.width_costs(0, |w| Some(8.0 + w as f64));
        assert_eq!(costs, [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]);
        assert_eq!(DeviceWidth::new(1, 4).unwrap().width_costs(0, |_| None), [0.0; 8]);
        switch.observe(100, 4, 3, 10_000_000);
        let one = switch.width_costs(0, |_| None);
        assert_eq!(one[3], 10.0);
        assert!((one[7] - 10.0 * (1.0 + WIDTH_SLOPE * 4.0) as f32).abs() < 1e-4);
        switch.observe(100, 8, 5, 14_000_000);
        let two = switch.width_costs(0, |_| None);
        assert_eq!((two[3], two[7]), (10.0, 14.0));
        assert!((two[5] - 12.0).abs() < 1e-4);
        // Longer contexts borrow the nearest learned bucket until they learn their own.
        assert_eq!(switch.width_costs(200_000, |_| None), two);
        assert_eq!(switch.expected_tokens(200_000, 1.0), 4.0);
        assert_eq!(switch.expected_tokens(0, 1.0), 4.0);
        assert_eq!(DeviceWidth::new(1, 8).unwrap().expected_tokens(0, 2.5), 2.5);
    }

    #[test]
    fn widths_beyond_capacity_and_invalid_rounds_are_ignored() {
        let mut switch = DeviceWidth::new(1, 4).unwrap();
        switch.observe(0, 5, 3, 1_000_000);
        switch.observe(0, 0, 3, 1_000_000);
        switch.observe(0, 2, 3, 0);
        assert_eq!(switch.width_costs(0, |_| None), [0.0; 8]);
        switch.observe(0, 2, 2, 9_000_000);
        let costs = switch.width_costs(0, |_| None);
        assert!(costs[..4].iter().all(|c| *c > 0.0) && costs[4..].iter().all(|c| *c == 0.0));
    }
}
