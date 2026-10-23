//! Retained duration estimates; zero denotes an unmeasured graph cost.

use super::ScheduledStep;
use crate::{
    Result,
    execution::{ExecutionTiming, GraphDescriptor, GraphKind},
};

pub(super) fn observed_duration_ns(timing: ExecutionTiming) -> Option<u64> {
    timing
        .gpu_ns
        .into_iter()
        .flatten()
        .max()
        .filter(|ns| *ns > 0)
        .map(|ns| if timing.queued { ns } else { ns.max(timing.wall_ns) })
}

/// Context buckets for decode and joined speculative costs: a round's attention and KV traffic
/// grow with the batch's longest context, so a width measured at 8k is not its cost at 128k.
pub(super) const CONTEXT_BUCKETS: usize = 4;

pub(super) fn context_bucket(position: u64) -> usize {
    match position {
        ..=8192 => 0,
        ..=32768 => 1,
        ..=131072 => 2,
        _ => 3,
    }
}

fn context_aware(kind: GraphKind) -> bool {
    matches!(kind, GraphKind::Decode | GraphKind::Speculative)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Key {
    pub kind: GraphKind,
    pub batch: usize,
    pub tokens: usize,
    pub prefill: usize,
    pub packed: usize,
}

impl Key {
    pub fn from_step(step: &ScheduledStep) -> Self {
        Self { kind: step.kind, batch: step.batch_capacity, tokens: step.token_capacity,
            prefill: step.prefill_capacity(), packed: step.packed_token_capacity }
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct PackedShape {
    pub batch: usize,
    pub width: usize,
    pub tokens: usize,
    pub nanos: u64,
}

struct Estimate {
    nanos: u64,
    replays: ReplayTimes,
    handoff_ns: u64,
    observed_at_ns: u64,
    last_attempt_ns: u64,
    /// Per-context replay medians; zero is unmeasured. The context-blind median above keeps
    /// driving freshness and refresh.
    contexts: [ContextCost; CONTEXT_BUCKETS],
}

#[derive(Default)]
struct ContextCost {
    nanos: u64,
    replays: ReplayTimes,
    observed_at_ns: u64,
}

impl Estimate {
    /// The cost at `context`: its own bucket, else the nearest freshly measured bucket, else
    /// the context-blind median.
    fn nanos_at(&self, kind: GraphKind, context: usize, now_ns: u64, ttl_ns: u64) -> u64 {
        if self.nanos == 0 || !context_aware(kind) {
            return self.nanos;
        }
        (0..CONTEXT_BUCKETS)
            .filter(|&bucket| self.contexts[bucket].nanos != 0
                && now_ns.saturating_sub(self.contexts[bucket].observed_at_ns) < ttl_ns)
            .min_by_key(|&bucket| (bucket.abs_diff(context), std::cmp::Reverse(bucket)))
            .map_or(self.nanos, |bucket| self.contexts[bucket].nanos)
    }
}

#[derive(Default)]
struct ReplayTimes {
    samples: [u64; 9],
    len: usize,
    next: usize,
}

impl ReplayTimes {
    fn observe(&mut self, sample: u64, fresh: bool) -> u64 {
        if !fresh {
            *self = Self::default();
        }
        self.samples[self.next] = sample;
        self.next = (self.next + 1) % self.samples.len();
        self.len = (self.len + 1).min(self.samples.len());
        let mut ordered = self.samples;
        ordered[..self.len].sort_unstable();
        let middle = self.len / 2;
        if self.len % 2 == 0 {
            ((u128::from(ordered[middle - 1]) + u128::from(ordered[middle]) + 1) / 2) as u64
        } else {
            ordered[middle]
        }
    }
}

impl Estimate {
    fn fresh(&self, now_ns: u64, ttl_ns: u64) -> bool {
        self.nanos != 0 && now_ns.saturating_sub(self.observed_at_ns) < ttl_ns
    }
}

#[derive(Default)]
pub(super) struct ActionCosts {
    entries: Vec<(Key, Estimate)>,
    now_ns: u64,
    ttl_ns: u64,
    context: usize,
}

impl ActionCosts {
    #[cfg(test)]
    pub fn set_measured(&mut self, step: &ScheduledStep, nanos: u64) {
        let key = Key {
            kind: step.kind,
            batch: step.batch_capacity,
            tokens: step.token_capacity,
            prefill: step.prefill_capacity(),
            packed: step.packed_token_capacity,
        };
        let estimate = &mut self.entries.iter_mut().find(|(k, _)| *k == key).unwrap().1;
        estimate.nanos = nanos;
        estimate.replays.observe(nanos, false);
        estimate.handoff_ns = 0;
        estimate.observed_at_ns = self.now_ns;
        estimate.contexts = Default::default();
    }

    #[cfg(test)]
    pub fn set_measured_at(&mut self, step: &ScheduledStep, bucket: usize, nanos: u64) {
        let key = Key::from_step(step);
        let now = self.now_ns;
        let estimate = &mut self.entries.iter_mut().find(|(k, _)| *k == key).unwrap().1;
        if estimate.nanos == 0 {
            estimate.nanos = nanos;
            estimate.replays.observe(nanos, false);
            estimate.observed_at_ns = now;
        }
        let cost = &mut estimate.contexts[bucket];
        cost.nanos = nanos;
        cost.replays.observe(nanos, false);
        cost.observed_at_ns = now;
    }

    pub fn prepared(graphs: &[GraphDescriptor]) -> Result<Self> {
        let mut entries = Vec::with_capacity(graphs.len());
        for graph in graphs {
            let key = Key {
                kind: graph.kind,
                batch: graph.batch_capacity,
                tokens: graph.token_capacity,
                prefill: graph.prefill_capacity,
                packed: graph.packed_token_capacity,
            };
            if entries.iter().any(|(k, _)| *k == key) {
                return Err("prepared graphs require unique shapes".into());
            }
            entries.push((
                key,
                Estimate {
                    nanos: 0,
                    replays: ReplayTimes::default(),
                    handoff_ns: 0,
                    observed_at_ns: 0,
                    last_attempt_ns: 0,
                    contexts: Default::default(),
                },
            ));
        }
        Ok(Self {
            entries,
            now_ns: 0,
            ttl_ns: super::PolicyConfig::default().estimate_ttl_ns,
            context: 0,
        })
    }

    /// Estimates for decode and joined speculation answer for this context bucket.
    pub fn set_context(&mut self, bucket: usize) {
        self.context = bucket.min(CONTEXT_BUCKETS - 1);
    }

    pub fn advance_to(&mut self, now_ns: u64, ttl_ns: u64) {
        self.now_ns = now_ns;
        self.ttl_ns = ttl_ns;
    }

    pub fn duration(&self, kind: GraphKind, batch: usize, tokens: usize) -> Option<u64> {
        self.shape_duration(kind, batch, tokens, 0).or_else(|| {
            (kind == GraphKind::Verification).then(|| self.packed_shapes()
                .filter(|shape| shape.batch == batch && shape.width >= tokens && shape.tokens >= batch * tokens)
                .map(|shape| shape.nanos).min()).flatten()
        })
    }

    pub fn packed_shapes(&self) -> impl Iterator<Item = PackedShape> + '_ {
        self.entries.iter().filter(|(key, _)| key.kind == GraphKind::Verification && key.packed != 0)
            .map(|(key, estimate)| PackedShape { batch: key.batch, width: key.tokens, tokens: key.packed,
                nanos: estimate.nanos.saturating_add(estimate.handoff_ns) })
    }

    pub fn shape_duration(
        &self,
        kind: GraphKind,
        batch: usize,
        tokens: usize,
        prefill: usize,
    ) -> Option<u64> {
        self.exact_duration(kind, batch, tokens, prefill, 0)
    }

    fn exact_duration(&self, kind: GraphKind, batch: usize, tokens: usize, prefill: usize, packed: usize) -> Option<u64> {
        self.entry(Key {
            kind,
            batch,
            tokens,
            prefill,
            packed,
        })
        .map(|estimate| estimate.nanos_at(kind, self.context, self.now_ns, self.ttl_ns).saturating_add(estimate.handoff_ns))
    }

    pub fn duration_key(&self, key: Key) -> Option<u64> {
        self.entry(key).map(|estimate| estimate.nanos_at(key.kind, self.context, self.now_ns, self.ttl_ns).saturating_add(estimate.handoff_ns))
    }

    pub fn is_fresh(&self, key: Key) -> bool {
        self.entry(key).is_some_and(|estimate| estimate.fresh(self.now_ns, self.ttl_ns))
    }

    pub fn stale_keys(&self) -> impl Iterator<Item = Key> + '_ {
        self.entries.iter().filter(|(_, estimate)| estimate.nanos != 0
            && !estimate.fresh(self.now_ns, self.ttl_ns)).map(|(key, _)| *key)
    }

    pub fn refresh_order(&self, key: Key) -> u64 {
        self.entry(key).map_or(0, |estimate| estimate.observed_at_ns.max(estimate.last_attempt_ns))
    }

    pub fn record_attempt(&mut self, key: Key) {
        if let Some((_, estimate)) = self.entries.iter_mut().find(|(shape, _)| *shape == key) {
            estimate.last_attempt_ns = self.now_ns;
        }
    }

    pub fn observation(&self, key: Key) -> Option<(u64, Option<u64>, bool)> {
        self.entry(key).map(|estimate| (estimate.nanos_at(key.kind, self.context, self.now_ns, self.ttl_ns).saturating_add(estimate.handoff_ns),
            (estimate.nanos != 0).then_some(estimate.observed_at_ns),
            estimate.fresh(self.now_ns, self.ttl_ns)))
    }

    fn entry(&self, key: Key) -> Option<&Estimate> {
        self.entries
            .iter()
            .find(|(k, _)| *k == key)
            .map(|(_, estimate)| estimate)
    }

    pub fn get(&self, step: &ScheduledStep) -> Option<u64> {
        self.exact_duration(
            step.kind,
            step.batch_capacity,
            step.token_capacity,
            step.prefill_capacity(),
            step.packed_token_capacity,
        )
    }

    pub fn observe(&mut self, step: &ScheduledStep, timing: ExecutionTiming) {
        let Some(raw_sample) = timing.gpu_ns.into_iter().flatten().max().filter(|n| *n > 0) else {
            return;
        };
        let sample = raw_sample.saturating_sub(timing.readiness.map_or(0, |intervals| intervals.wait_ns()));
        if sample == 0 {
            return;
        }
        let key = Key {
            kind: step.kind,
            batch: step.batch_capacity,
            tokens: step.token_capacity,
            prefill: step.prefill_capacity(),
            packed: step.packed_token_capacity,
        };
        if let Some((_, estimate)) = self.entries.iter_mut().find(|(k, _)| *k == key) {
            let fresh = estimate.fresh(self.now_ns, self.ttl_ns);
            if !timing.queued {
                let handoff = timing.wall_ns.saturating_sub(raw_sample);
                estimate.handoff_ns = if fresh {
                    ((7 * u128::from(estimate.handoff_ns) + u128::from(handoff) + 4) / 8) as u64
                } else { handoff };
            } else if !fresh {
                estimate.handoff_ns = 0;
            }
            estimate.nanos = estimate.replays.observe(sample, fresh);
            estimate.observed_at_ns = self.now_ns;
            if context_aware(step.kind) {
                let position = step.lanes.iter().map(|lane| lane.start_position).max().unwrap_or(0);
                let bucket = &mut estimate.contexts[context_bucket(position)];
                let fresh = bucket.nanos != 0 && self.now_ns.saturating_sub(bucket.observed_at_ns) < self.ttl_ns;
                bucket.nanos = bucket.replays.observe(sample, fresh);
                bucket.observed_at_ns = self.now_ns;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ReplayTimes;

    #[test]
    fn isolated_slow_replay_does_not_displace_the_steady_cost() {
        let mut replays = ReplayTimes::default();
        for _ in 0..9 {
            assert_eq!(replays.observe(8_000_000, true), 8_000_000);
        }
        assert_eq!(replays.observe(16_000_000, true), 8_000_000);
        for _ in 0..9 {
            assert_eq!(replays.observe(8_000_000, true), 8_000_000);
        }
    }

    #[test]
    fn sustained_cost_changes_replace_the_window_and_staleness_resets_it() {
        let mut replays = ReplayTimes::default();
        for _ in 0..9 {
            replays.observe(8, true);
        }
        for _ in 0..4 {
            assert_eq!(replays.observe(16, true), 8);
        }
        assert_eq!(replays.observe(16, true), 16);
        assert_eq!(replays.observe(32, false), 32);
        assert_eq!(replays.observe(16, true), 24);
    }

    #[test]
    fn even_sample_median_does_not_overflow() {
        let mut replays = ReplayTimes::default();
        assert_eq!(replays.observe(u64::MAX, true), u64::MAX);
        assert_eq!(replays.observe(u64::MAX - 1, true), u64::MAX);
    }
}
