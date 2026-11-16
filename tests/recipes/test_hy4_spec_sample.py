# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0
"""consult18 P5: speculative sampling over vocabulary shards is exact.

Synthetic rounds (k = 2 drafts, a 32-token vocabulary split over 1, 2 and 8
simulated ranks, fixed per-position target p_j and draft q_j): draft tokens
sampled from q with the sharded Gumbel-max, then `verify_sample`. Every
emitted token at position j must be distributed as p_j (the positions are
independent by construction) — chi-square at the 0.1 % level, and every rank
must emit the same tokens.
"""
import pytest

torch = pytest.importorskip("torch")

V = 32
K = 2
TRIALS = 3000
# chi-square 99.9 % critical values by degrees of freedom (scipy.stats.chi2.ppf)
CRIT = {4: 18.47, 5: 20.52, 6: 22.46, 7: 24.32, 8: 26.12, 9: 27.88, 10: 29.59, 11: 31.26,
        12: 32.91, 13: 34.53, 14: 36.12, 15: 37.70, 29: 58.30, 30: 59.70, 31: 61.10}


def _round_trials(collectives, rank, world, p_logits, q_logits, seed):
    from ignition.recipes.hy4.spec_sample import sample_tokens, verify_sample

    width = V // world
    lo = rank * width
    vr = (lo, lo + width)
    gen = torch.Generator().manual_seed(seed * 1000 + rank)
    sp = p_logits[:, lo:lo + width]
    sq = q_logits[:, lo:lo + width]
    emitted = []
    for _ in range(TRIALS):
        drafts = torch.stack([
            sample_tokens(collectives, sq[i:i + 1], torch.rand((1, width), generator=gen),
                          vocabulary_range=vr)[0] for i in range(K)])
        accept, tokens = verify_sample(collectives, sp, sq, drafts, torch.rand((K,), generator=gen),
                                       torch.rand((K + 1, width), generator=gen), vocabulary_range=vr)
        n = 0
        while n < K and bool(accept[n]):
            n += 1
        emitted.append([int(x) for x in drafts[:n]] + [int(tokens[n])])
    return emitted


def _chi_square(counts, probs):
    total = sum(counts)
    stat, cells = 0.0, 0
    for c, p in zip(counts, probs):
        expected = total * p
        if expected >= 5:
            stat += (c - expected) ** 2 / expected
            cells += 1
    return stat, cells - 1


@pytest.mark.parametrize("world", (1, 2, 8))
def test_speculative_sampling_emits_target_distribution(world):
    from ignition.recipes.hy4.collectives import (
        PcieDcpCollectives, SimulatedDcpTransport, SimulatedGroup)

    gen = torch.Generator().manual_seed(7)
    p_logits = torch.randn((K + 1, V), generator=gen)
    q_logits = p_logits[:K] + 0.8 * torch.randn((K, V), generator=gen)   # a mismatched draft
    if world == 1:
        results = [_round_trials(None, 0, 1, p_logits, q_logits, seed=1)]
    else:
        group = SimulatedGroup(world)
        transport = SimulatedDcpTransport(group)
        cols = [PcieDcpCollectives(rank=r, world_size=world, transport=transport) for r in range(world)]
        results = group.run([lambda col, r=r: _round_trials(cols[r], r, world, p_logits, q_logits, seed=1)
                             for r in range(world)])
        for rank in range(1, world):
            assert results[rank] == results[0], f"rank {rank} emitted different tokens"
    emitted = results[0]
    probs = torch.softmax(p_logits, dim=-1)
    for j in range(K + 1):
        column = [seq[j] for seq in emitted if len(seq) > j]
        if len(column) < 500:
            continue
        counts = [column.count(v) for v in range(V)]
        stat, dof = _chi_square(counts, probs[j].tolist())
        assert stat < CRIT.get(dof, 2.5 * dof), f"world {world} position {j}: chi2 {stat:.1f} dof {dof}"
    accepted = sum(len(seq) - 1 for seq in emitted) / len(emitted)
    assert 0.2 < accepted < K, f"mean accepted drafts {accepted:.2f} (a mismatched draft accepts sometimes)"


def _reference_filter(logits, top_k, top_p):
    """The full-vocabulary filter: top-k ties kept; top-p keeps through the
    crossing token (exclusive cumsum < p), at least one."""
    s = logits.clone().float()
    if top_k > 0:
        kth = torch.topk(s, top_k, dim=-1).values[..., -1:]
        s = torch.where(s >= kth, s, torch.full_like(s, float("-inf")))
    if top_p < 1.0:
        order = torch.sort(s, dim=-1, descending=True)
        probs = torch.softmax(order.values, dim=-1)
        cum = probs.cumsum(-1)
        keep = (cum - probs) < top_p
        keep[..., 0] = True
        last = keep.sum(-1, keepdim=True) - 1
        thr = order.values.gather(-1, last)
        s = torch.where(s >= thr, s, torch.full_like(s, float("-inf")))
    return s


@pytest.mark.parametrize("world", (1, 2, 8))
@pytest.mark.parametrize("top_k,top_p", [(0, 0.9), (20, 1.0), (40, 0.8), (0, 0.5), (5, 0.95)])
def test_sharded_filter_matches_full_vocabulary(world, top_k, top_p):
    """consult18 P7: filter_scores over the shards == the full-vocabulary filter
    (where it reports covered), every rank the same."""
    from ignition.recipes.hy4.collectives import (
        PcieDcpCollectives, SimulatedDcpTransport, SimulatedGroup)
    from ignition.recipes.hy4.spec_sample import filter_scores

    vocab = 512
    gen = torch.Generator().manual_seed(top_k * 100 + int(top_p * 10))
    logits = torch.randn((3, vocab), generator=gen) * 3.0
    want = _reference_filter(logits, top_k, top_p)
    tk = torch.tensor([top_k], dtype=torch.long)
    tp = torch.tensor([top_p], dtype=torch.float32)
    width = vocab // world

    def body(col, rank):
        lo = rank * width
        out, cov = filter_scores(col, logits[:, lo:lo + width], tk, tp,
                                 vocabulary_range=(lo, lo + width), offer_width=96)
        return out, bool(cov.all())

    if world == 1:
        results = [body(None, 0)]
    else:
        group = SimulatedGroup(world)
        transport = SimulatedDcpTransport(group)
        cols = [PcieDcpCollectives(rank=r, world_size=world, transport=transport) for r in range(world)]
        results = group.run([lambda col, r=r: body(cols[r], r) for r in range(world)])
    covered = [c for _, c in results]
    assert len(set(covered)) == 1
    if covered[0]:
        got = torch.cat([o for o, _ in results], dim=1)
        assert torch.equal(torch.isfinite(got), torch.isfinite(want))
        assert torch.equal(got[torch.isfinite(got)], want[torch.isfinite(want)])


@pytest.mark.parametrize("world", (1, 2, 8))
@pytest.mark.parametrize("top_k,top_p", [(8, 1.0), (0, 0.8), (12, 0.9)])
def test_filtered_speculative_sampling_emits_filtered_target(world, top_k, top_p):
    """consult18 P7: with top-k / top-p, drafts sampled from the filtered draft
    distribution and verified against the filtered target emit tokens
    distributed as the filtered target p' (chi-square at 0.1 %)."""
    from ignition.recipes.hy4.collectives import (
        PcieDcpCollectives, SimulatedDcpTransport, SimulatedGroup)
    from ignition.recipes.hy4.spec_sample import filter_scores, sample_tokens, verify_sample

    gen = torch.Generator().manual_seed(11)
    p_logits = torch.randn((K + 1, V), generator=gen) * 1.5
    q_logits = p_logits[:K] + 0.8 * torch.randn((K, V), generator=gen)
    tk = torch.tensor([top_k], dtype=torch.long)
    tp = torch.tensor([top_p], dtype=torch.float32)

    def trials(col, rank):
        width = V // world
        lo = rank * width
        vr = (lo, lo + width)
        g = torch.Generator().manual_seed(1000 + rank)
        sp, cov_p = filter_scores(col, p_logits[:, lo:lo + width], tk, tp, vocabulary_range=vr, offer_width=64)
        sq, cov_q = filter_scores(col, q_logits[:, lo:lo + width], tk, tp, vocabulary_range=vr, offer_width=64)
        assert bool(cov_p.all()) and bool(cov_q.all())
        emitted = []
        for _ in range(TRIALS):
            drafts = torch.stack([sample_tokens(col, sq[i:i + 1], torch.rand((1, width), generator=g),
                                                vocabulary_range=vr)[0] for i in range(K)])
            accept, tokens = verify_sample(col, sp, sq, drafts, torch.rand((K,), generator=g),
                                           torch.rand((K + 1, width), generator=g), vocabulary_range=vr)
            n = 0
            while n < K and bool(accept[n]):
                n += 1
            emitted.append([int(x) for x in drafts[:n]] + [int(tokens[n])])
        return emitted

    if world == 1:
        results = [trials(None, 0)]
    else:
        group = SimulatedGroup(world)
        transport = SimulatedDcpTransport(group)
        cols = [PcieDcpCollectives(rank=r, world_size=world, transport=transport) for r in range(world)]
        results = group.run([lambda col, r=r: trials(cols[r], r) for r in range(world)])
        for rank in range(1, world):
            assert results[rank] == results[0]
    probs = torch.softmax(_reference_filter(p_logits, top_k, top_p), dim=-1)
    for j in range(K + 1):
        column = [seq[j] for seq in results[0] if len(seq) > j]
        if len(column) < 500:
            continue
        counts = [column.count(v) for v in range(V)]
        assert all(c == 0 for c, p in zip(counts, probs[j].tolist()) if p == 0), "a filtered-out token emitted"
        stat, dof = _chi_square(counts, probs[j].tolist())
        assert stat < CRIT.get(dof, 2.5 * max(dof, 4)), f"world {world} pos {j}: chi2 {stat:.1f} dof {dof}"
