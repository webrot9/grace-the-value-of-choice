"""P02 — invariants for contextual bandits and posterior sampling.

Every test is a property that holds for any correct implementation. None of them
depends on how a run happened to go: the two that use randomness draw thousands
of samples with a fixed seed and an explicit tolerance.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rl_lab import contextual as c  # noqa: E402

# The practical's course recommender: rows are visitor types, columns courses.
CTR = np.array([
    [0.30, 0.20, 0.03],   # AI engineer
    [0.08, 0.30, 0.05],   # management engineer
    [0.05, 0.10, 0.10],   # high school kid
])


# --------------------------------------------------------------------------- #
# Q1: the Beta posterior
# --------------------------------------------------------------------------- #
def test_q1_mean_matches_the_closed_form():
    a = np.array([1.0, 3.0, 10.0])
    b = np.array([1.0, 1.0, 30.0])
    assert c.posterior_mean(a, b) == pytest.approx([0.5, 0.75, 0.25])


def test_q1_mean_is_laplace_rule_of_succession():
    """From Beta(1,1), k clicks in n views must give (k+1)/(n+2), not k/n:
    the smoothing is what stops one lucky click from looking like certainty."""
    for k, n in ((0, 0), (1, 1), (3, 10), (7, 20)):
        alpha, beta = 1.0 + k, 1.0 + (n - k)
        assert c.posterior_mean(np.array([alpha]), np.array([beta]))[0] == \
            pytest.approx((k + 1) / (n + 2))


def test_q1_variance_matches_the_closed_form():
    a, b = np.array([2.0]), np.array([3.0])
    s = 5.0
    assert c.posterior_variance(a, b)[0] == pytest.approx(2 * 3 / (s ** 2 * (s + 1)))


def test_q1_variance_of_a_uniform_prior_is_one_twelfth():
    """Beta(1,1) is the uniform distribution on [0,1], whose variance is 1/12."""
    assert c.posterior_variance(np.array([1.0]), np.array([1.0]))[0] == \
        pytest.approx(1 / 12)


def test_q1_variance_shrinks_like_one_over_n():
    """Quadrupling the evidence must roughly halve the standard deviation: this
    is the rate that makes exploration fade on its own."""
    def sd(n):
        a = np.array([1.0 + n / 2]); b = np.array([1.0 + n / 2])
        return float(np.sqrt(c.posterior_variance(a, b)[0]))
    assert sd(40) / sd(160) == pytest.approx(2.0, rel=0.05)
    assert sd(160) / sd(640) == pytest.approx(2.0, rel=0.05)


def test_q1_update_is_conjugate():
    assert c.posterior_update(1.0, 1.0, 1) == (2.0, 1.0)
    assert c.posterior_update(1.0, 1.0, 0) == (1.0, 2.0)
    assert c.posterior_update(5.0, 2.0, 1) == (6.0, 2.0)


def test_q1_update_never_loses_mass():
    """alpha + beta counts the observations: it must grow by exactly one."""
    a, b = 1.0, 1.0
    for r in (1, 0, 0, 1, 1):
        a, b = c.posterior_update(a, b, r)
    assert a + b == pytest.approx(7.0)
    assert (a, b) == (4.0, 3.0)


# --------------------------------------------------------------------------- #
# Q2: acting on a posterior
# --------------------------------------------------------------------------- #
def test_q2_posterior_greedy_picks_the_highest_mean():
    alphas = np.array([[1.0, 9.0, 2.0]])
    betas = np.array([[1.0, 1.0, 8.0]])
    assert c.select_posterior_greedy(alphas, betas, 0) == 1


def test_q2_posterior_greedy_breaks_ties_on_the_lowest_index():
    alphas = np.array([[3.0, 3.0, 3.0]])
    betas = np.array([[2.0, 2.0, 2.0]])
    assert c.select_posterior_greedy(alphas, betas, 0) == 0


def test_q2_posterior_greedy_uses_the_mean_not_the_raw_count():
    """One click out of one view (mean 2/3) must lose to eight out of ten
    (mean 9/12): the agent that trusts the raw rate 1.0 gets this wrong."""
    alphas = np.array([[2.0, 9.0]])
    betas = np.array([[1.0, 3.0]])
    assert c.select_posterior_greedy(alphas, betas, 0) == 1


def test_q2_thompson_collapses_to_greedy_when_the_posterior_is_degenerate():
    """With a very concentrated posterior the draws are essentially the means,
    so posterior sampling and posterior-greedy must agree. This is the limit in
    which exploration has finished."""
    alphas = np.array([[900.0, 100.0, 10.0]])
    betas = np.array([[100.0, 900.0, 990.0]])
    rng = np.random.default_rng(0)
    picks = [c.select_thompson(alphas, betas, 0, rng) for _ in range(200)]
    assert set(picks) == {0}
    assert c.select_posterior_greedy(alphas, betas, 0) == 0


def test_q2_thompson_under_a_uniform_prior_is_uniform():
    """Beta(1,1) everywhere means no evidence: no action may be excluded, and
    with three actions each must come up about a third of the time."""
    alphas = np.ones((1, 3))
    betas = np.ones((1, 3))
    rng = np.random.default_rng(1)
    picks = np.array([c.select_thompson(alphas, betas, 0, rng) for _ in range(4500)])
    for a in range(3):
        assert np.mean(picks == a) == pytest.approx(1 / 3, abs=0.04)


def test_q2_thompson_reads_only_its_own_context():
    """The row of the posterior for state s must be the only one that matters:
    a contextual bandit that leaks across contexts is just a bandit."""
    alphas = np.array([[9.0, 1.0], [1.0, 9.0]])
    betas = np.array([[1.0, 9.0], [9.0, 1.0]])
    rng = np.random.default_rng(2)
    assert {c.select_thompson(alphas, betas, 0, rng) for _ in range(100)} == {0}
    assert {c.select_thompson(alphas, betas, 1, rng) for _ in range(100)} == {1}


def test_q2_thompson_is_exact_probability_matching():
    """Thompson shows each action with the probability that it is the best
    under the posterior. For Beta(4,8), Beta(1,3), Beta(1,4) those probabilities
    are exact (one density integrated against the other two distribution
    functions): 383/714, 877/3094 and 836/4641, that is 0.536, 0.283, 0.180.
    A 'Thompson' drawing from a normal with the same mean and variance gets
    about 0.515 and 0.305 and fails here, because it matches only the first two
    moments of the posterior."""
    alphas = np.array([[4.0, 1.0, 1.0]])
    betas = np.array([[8.0, 3.0, 4.0]])
    rng = np.random.default_rng(5)
    n = 60_000
    picks = np.array([c.select_thompson(alphas, betas, 0, rng) for _ in range(n)])
    freq = np.bincount(picks, minlength=3) / n
    assert freq == pytest.approx([383 / 714, 877 / 3094, 836 / 4641], abs=0.008)


# --------------------------------------------------------------------------- #
# Q3: contextual regret
# --------------------------------------------------------------------------- #
def test_q3_regret_is_zero_on_the_best_action_of_each_context():
    states = np.array([0, 1, 2, 0])
    actions = np.array([0, 1, 1, 0])   # argmax per row of CTR: 0, 1, 1 (tie 1/2)
    assert c.contextual_regret(CTR, states, actions) == pytest.approx(0.0)


def test_q3_regret_matches_a_hand_computation():
    """Two visits: an AI engineer shown the trading course (0.30 - 0.03 = 0.27)
    and a management engineer shown the RL course (0.30 - 0.08 = 0.22).
    Total 0.49."""
    states = np.array([0, 1])
    actions = np.array([2, 0])
    assert c.contextual_regret(CTR, states, actions) == pytest.approx(0.49)


def test_q3_regret_is_never_negative():
    rng = np.random.default_rng(5)
    for _ in range(50):
        n = int(rng.integers(1, 40))
        states = rng.integers(0, 3, size=n)
        actions = rng.integers(0, 3, size=n)
        assert c.contextual_regret(CTR, states, actions) >= -1e-12


def test_q3_regret_of_a_context_blind_policy_is_linear():
    """Always showing the course that is best *on average* still pays a gap in
    the contexts where it is wrong, so its regret grows like T.

    Here the average-best action is the optimisation course (column 1). It is
    optimal for the management engineer and for the high-school kid, but the AI
    engineer loses 0.30 - 0.20 = 0.10 every time. Over contexts visited equally
    often that is 0.10/3 = 1/30 per step, exactly."""
    best_on_average = int(CTR.mean(axis=0).argmax())
    for T in (30, 300, 3000):
        states = np.tile([0, 1, 2], T // 3)
        actions = np.full(len(states), best_on_average)
        r = c.contextual_regret(CTR, states, actions)
        assert r == pytest.approx(len(states) / 30.0)


def test_q3_run_contextual_returns_consistent_trajectories():
    for algo in ("thompson", "posterior-greedy", "uniform"):
        states, actions = c.run_contextual(CTR, 150, algo, seed=3)
        assert len(states) == len(actions) == 150
        assert set(np.unique(states)) <= {0, 1, 2}
        assert set(np.unique(actions)) <= {0, 1, 2}
