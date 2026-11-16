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
CRIT = {29: 58.30, 30: 59.70, 31: 61.10}


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
