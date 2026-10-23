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
    /// graph must not choose. Round cost is close to linear in the verification width, so every
    /// width is priced from one weighted least-squares line through the bucket's observed widths
    /// (else the nearest bucket's), never from its own mean alone: a width whose early rounds ran
    /// slow would otherwise never be chosen again and so never be corrected. Without observations
    /// the costs follow `fallback` (milliseconds of an equivalent host-width round).
    pub fn width_costs(&self, position: u64, fallback: impl Fn(usize) -> Option<f64>) -> [f32; MAX_WIDTH] {
        let bucket = context_bucket(position);
        let observed = nearest(bucket, |b| {
            let points: Vec<_> = (1..=self.capacity)
                .filter_map(|w| {
                    let mean = self.costs[b][w - 1];
                    mean.get().map(|ms| (w as f64, ms, mean.weight))
                })
                .collect();
            (!points.is_empty()).then_some(points)
        });
        let mut costs = [0.0f32; MAX_WIDTH];
        for (index, cost) in costs.iter_mut().enumerate().take(self.capacity) {
            let width = (index + 1) as f64;
            let estimate = match &observed {
                Some(points) => fit(points, width),
                None => fallback(index + 1).unwrap_or(0.0),
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

/// Weighted least-squares line through observed (width, ms, weight) points, or a proportional
/// slope through a single width.
fn fit(points: &[(f64, f64, f64)], width: f64) -> f64 {
    if let [(w0, c0, _)] = points {
        return c0 * (1.0 + WIDTH_SLOPE * (width - w0));
    }
    let total: f64 = points.iter().map(|p| p.2).sum();
    let mx = points.iter().map(|p| p.0 * p.2).sum::<f64>() / total;
    let my = points.iter().map(|p| p.1 * p.2).sum::<f64>() / total;
    let (sxy, sxx) = points.iter().fold((0.0, 0.0), |(xy, xx), (w, c, n)| {
        (xy + n * (w - mx) * (c - my), xx + n * (w - mx) * (w - mx))
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
        // One slow width cannot price itself out: it is read from the line through all widths.
        let mut trapped = DeviceWidth::new(1, 8).unwrap();
        for _ in 0..64 {
            trapped.observe(0, 3, 2, 10_000_000);
            trapped.observe(0, 5, 3, 11_000_000);
            trapped.observe(0, 8, 5, 12_500_000);
        }
        trapped.observe(0, 4, 2, 30_000_000);
        let costs = trapped.width_costs(0, |_| None);
        assert!(costs[3] < 11.0 && costs[3] > 10.0, "{costs:?}");
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

#[cfg(test)]
mod scheduler_tests {
    use super::super::refresh::PlanOrigin;
    use super::super::*;
    use crate::execution::{CompletionRecord, ExecutionTiming, TokenVector};

    fn engine() -> (RequestScheduler, Vec<crate::execution::GraphDescriptor>) {
        let mut s = super::super::tests::scheduler(1, 512, 64);
        s.enable_speculation(&(1..=8).collect::<Vec<_>>()).unwrap();
        let graphs = super::super::tests::duration_graphs(1);
        s.measured_graph_durations(&graphs).unwrap();
        let switched = graphs.iter().position(|g| g.kind == GraphKind::Speculative && g.token_capacity == 8).unwrap();
        s.device_widths(&graphs, &[switched]).unwrap();
        (s, graphs)
    }

    fn ready(s: &mut RequestScheduler) -> RequestHandle {
        let handle = s.admit(super::super::tests::spec(65, 64)).unwrap();
        let r = s.request_mut(handle).unwrap();
        r.phase = RequestPhase::Decode;
        r.consumed_tokens = 65;
        r.scheduled_consumed = 65;
        r.generated_tokens = 1;
        handle
    }

    /// Commit and complete one speculative round of `capacity` that verified `width` and emitted `emitted`.
    fn round(s: &mut RequestScheduler, h: RequestHandle, capacity: usize, width: i64, emitted: i64) -> Result<Vec<GeneratedToken>> {
        let fence = Fence(s.submitted.0 + 1);
        let step = ScheduledStep { packed_token_capacity: 0, kind: GraphKind::Speculative, batch_capacity: 1,
            token_capacity: capacity, mixed: None, requires_feedback: true,
            lanes: vec![ScheduledLane { request: h, start_position: 65, tokens: capacity }] };
        let plan = ScheduledPlan { origin: PlanOrigin::Ordinary, selected_cost: Default::default(),
            owner: s.plan_owner.clone(), epochs: vec![s.request(h).unwrap().epoch],
            estimated_duration_ns: 1000, selection_value_rate: 0.0, selection_log_scale: 0.0, verification_budget: None, step };
        s.commit(&plan, fence).unwrap();
        let mut records = vec![CompletionRecord::default(); s.limits.request_capacity()];
        let mut vector = TokenVector { emitted_count: emitted, offered_drafts: width - 1, accepted_drafts: emitted - 1,
            verification_width: width, ..Default::default() };
        for i in 0..emitted as usize { vector.tokens[i] = 100 + i as i64; }
        records[h.slot as usize] = CompletionRecord {
            generation: h.generation as i64, submission_sequence: fence.0 as i64,
            consumed_tokens: 65 + emitted, generated_tokens: 1 + emitted, token_id: 100 + emitted - 1, status: 1,
            vector: Some(vector), ..Default::default()
        };
        let mut timing = ExecutionTiming::default();
        timing.gpu_ns[0] = Some(9_000_000);
        let realized = s.realized_width(&plan.step, &records);
        let result = s.complete(fence, &records, timing);
        if result.is_ok() {
            assert_eq!(realized as i64, width.clamp(1, capacity as i64));
        }
        result
    }

    #[test]
    fn a_switched_round_commits_the_width_the_device_chose() {
        let (mut s, _) = engine();
        // The switched graph replaces the batch's host-width graphs; other batches keep theirs.
        assert_eq!(s.speculative_catalog[0], vec![8]);
        s.enable_speculation(&(1..=8).collect::<Vec<_>>()).unwrap();
        assert_eq!(s.speculative_catalog[0], vec![8]);
        let h = ready(&mut s);
        let rounds = s.request(h).unwrap().acceptance.rounds();
        let output = round(&mut s, h, 8, 3, 3).unwrap();
        assert_eq!(output.len(), 3);
        let r = s.request(h).unwrap();
        assert_eq!((r.generated_tokens, r.consumed_tokens), (4, 68));
        // Device-chosen widths do not train the per-depth acceptance estimates.
        assert_eq!(r.acceptance.rounds(), rounds);
        let lane = ScheduledLane { request: h, start_position: 68, tokens: 8 };
        let costs = s.width_costs(&lane);
        assert_eq!(costs[2], 9.0);
        assert!(costs[..8].iter().all(|c| *c > 0.0));
        assert!((s.round_tokens(s.request(h).unwrap(), GraphKind::Speculative, 1, 8, 8) - 3.0).abs() < 1e-9);
    }

    #[test]
    fn widths_beyond_the_switched_capacity_or_on_host_width_graphs_are_rejected() {
        let (mut s, _) = engine();
        let h = ready(&mut s);
        assert!(round(&mut s, h, 8, 9, 3).is_err());
        let (mut s, _) = engine();
        let h = ready(&mut s);
        assert!(round(&mut s, h, 4, 3, 3).is_err());
        let (mut s, _) = engine();
        let h = ready(&mut s);
        assert!(round(&mut s, h, 4, 4, 3).is_ok());
    }

    #[test]
    fn only_a_one_request_speculative_graph_may_switch() {
        let mut s = super::super::tests::scheduler(2, 512, 64);
        s.enable_speculation(&(1..=8).collect::<Vec<_>>()).unwrap();
        let graphs = super::super::tests::duration_graphs(2);
        s.measured_graph_durations(&graphs).unwrap();
        let two = graphs.iter().position(|g| g.kind == GraphKind::Speculative && g.batch_capacity == 2).unwrap();
        let decode = graphs.iter().position(|g| g.kind == GraphKind::Decode).unwrap();
        let one = graphs.iter().position(|g| g.kind == GraphKind::Speculative && g.batch_capacity == 1 && g.token_capacity == 8).unwrap();
        assert!(s.device_widths(&graphs, &[two]).is_err());
        assert!(s.device_widths(&graphs, &[decode]).is_err());
        assert!(s.device_widths(&graphs, &[one, one]).is_err());
        assert!(s.device_widths(&graphs, &[99]).is_err());
        assert!(s.device_widths(&graphs, &[one]).is_ok());
        assert!(s.device_widths(&graphs, &[]).is_ok() && s.device_width.is_none());
    }

    #[test]
    fn a_lone_request_is_not_padded_into_a_wider_host_width_graph() {
        let mut s = super::super::tests::scheduler(2, 512, 64);
        s.enable_speculation(&(1..=8).collect::<Vec<_>>()).unwrap();
        let graphs = super::super::tests::duration_graphs(2);
        s.measured_graph_durations(&graphs).unwrap();
        let one = graphs.iter().position(|g| g.kind == GraphKind::Speculative && g.batch_capacity == 1 && g.token_capacity == 8).unwrap();
        s.device_widths(&graphs, &[one]).unwrap();
        // Make the batch-two graphs look cheaper than the switched graph.
        for width in 1..=8 {
            s.costs.set_measured(&ScheduledStep { packed_token_capacity: 0, kind: GraphKind::Speculative, batch_capacity: 2,
                token_capacity: width, mixed: None, lanes: vec![], requires_feedback: true }, 100_000);
        }
        s.costs.set_measured(&ScheduledStep { packed_token_capacity: 0, kind: GraphKind::Speculative, batch_capacity: 1,
            token_capacity: 8, mixed: None, lanes: vec![], requires_feedback: true }, 10_000_000);
        let h = ready(&mut s);
        let plan = s.plan().unwrap();
        assert_eq!((plan.step.kind, plan.step.batch_capacity, plan.step.token_capacity), (GraphKind::Speculative, 1, 8));
        assert_eq!(plan.step.lanes[0].request, h);
    }
}
