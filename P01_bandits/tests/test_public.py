"""
P01 tests. Students see and run these.

Design rule (see docs/05-esercitazioni.md): every test checks a *property* or a
value computable in closed form. No test imports a reference solution, and no
test depends on the outcome of a training run, so nothing here can fail at
random and nothing here leaks an answer.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from rl_lab import bandits as b


# --------------------------------------------------------------------------- #
# Q1: hoeffding_radius
# --------------------------------------------------------------------------- #
def test_q1_matches_closed_form():
    for n, delta in [(1, 0.1), (10, 0.05), (100, 0.5), (7, 0.01)]:
        expected = math.sqrt(math.log(2 / delta) / (2 * n))
        assert b.hoeffding_radius(n, delta) == pytest.approx(expected, rel=1e-9)


def test_q1_unpulled_arm_is_infinitely_uncertain():
    assert b.hoeffding_radius(0, 0.1) == float("inf")


def test_q1_shrinks_like_one_over_sqrt_n():
    """Quadrupling the samples must halve the radius: it is the 1/sqrt(n) rate
    that gives UCB its sqrt(T) regret."""
    r1, r4 = b.hoeffding_radius(25, 0.1), b.hoeffding_radius(100, 0.1)
    assert r1 / r4 == pytest.approx(2.0, rel=1e-9)


def test_q1_grows_when_confidence_grows():
    assert b.hoeffding_radius(50, 0.001) > b.hoeffding_radius(50, 0.5)


# --------------------------------------------------------------------------- #
# Q2: selection rules
# --------------------------------------------------------------------------- #
def test_q2_greedy_picks_best_mean_and_breaks_ties_low():
    assert b.select_greedy(np.array([0.1, 0.9, 0.4]), np.array([5, 5, 5])) == 1
    assert b.select_greedy(np.array([0.5, 0.5, 0.2]), np.array([3, 3, 3])) == 0


def test_q2_ucb_tries_an_unpulled_arm_first():
    """An arm with N=0 has an infinite bonus, so optimism must reach it even if
    another arm already looks excellent."""
    means = np.array([0.99, 0.0, 0.0])
    counts = np.array([50, 0, 7])
    assert b.select_ucb(means, counts, t=58, delta=0.1) == 1


def test_q2_ucb_prefers_the_less_pulled_arm_when_means_are_equal():
    means = np.array([0.5, 0.5])
    counts = np.array([100, 10])
    assert b.select_ucb(means, counts, t=110, delta=0.1) == 1


def test_q2_ucb_can_prefer_a_worse_mean_if_it_is_uncertain_enough():
    """The whole point of optimism: 0.40 with 5 pulls beats 0.55 with 500."""
    means = np.array([0.55, 0.40])
    counts = np.array([500, 5])
    assert b.select_ucb(means, counts, t=505, delta=0.1) == 1


def test_q2_the_lectures_ucb_uses_sqrt_log_kt_over_delta_over_n():
    """Given code, checked here so it cannot drift from the lecture: the bonus
    of `select_ucb_known_horizon` is sqrt(ln(KT/delta) / N). K = 2, T = 1000,
    delta = 0.1: arm 1 has 100 pulls against 400, so its extra bonus is
    0.05 * sqrt(ln(20000)) = 0.15735. It wins against a lead of 0.1570 and loses
    against 0.1577: without K, with 2KT, or with 2N one of the two fails."""
    counts = np.array([400, 100])
    lead = [np.array([0.5 + x, 0.5]) for x in (0.1570, 0.1577)]
    assert b.select_ucb_known_horizon(lead[0], counts, horizon=1000, delta=0.1) == 1
    assert b.select_ucb_known_horizon(lead[1], counts, horizon=1000, delta=0.1) == 0


def test_q2_ucb_becomes_greedy_when_everything_is_well_estimated():
    means = np.array([0.55, 0.40])
    counts = np.array([50_000, 50_000])
    assert b.select_ucb(means, counts, t=100_000, delta=0.1) == 0


def test_q2_epsilon_greedy_is_greedy_when_epsilon_is_zero():
    rng = np.random.default_rng(0)
    means, counts = np.array([0.2, 0.7, 0.1]), np.array([4, 4, 4])
    for _ in range(20):
        assert b.select_epsilon_greedy(means, counts, 0.0, rng) == 1


def test_q2_epsilon_greedy_is_uniform_when_epsilon_is_one():
    rng = np.random.default_rng(1)
    means, counts = np.array([0.2, 0.7, 0.1]), np.array([4, 4, 4])
    picks = [b.select_epsilon_greedy(means, counts, 1.0, rng) for _ in range(3000)]
    freq = np.bincount(picks, minlength=3) / 3000
    assert np.allclose(freq, 1 / 3, atol=0.05)


def test_q2_epsilon_greedy_explores_about_epsilon_of_the_time():
    rng = np.random.default_rng(2)
    means, counts = np.array([0.0, 1.0]), np.array([9, 9])
    picks = [b.select_epsilon_greedy(means, counts, 0.2, rng) for _ in range(5000)]
    # arm 0 is chosen only by exploration: epsilon/K = 0.1
    assert np.mean(np.array(picks) == 0) == pytest.approx(0.1, abs=0.02)


def test_q2_thompson_concentrates_on_the_better_arm():
    """With a lot of evidence the posterior draws almost always rank correctly."""
    rng = np.random.default_rng(3)
    succ, fail = np.array([90.0, 10.0]), np.array([10.0, 90.0])
    picks = [b.select_thompson_bernoulli(succ, fail, rng) for _ in range(500)]
    assert np.mean(np.array(picks) == 0) > 0.98


def test_q2_thompson_is_undecided_with_no_evidence():
    """Beta(1,1) is uniform: with no data the choice must be ~50/50, which is
    where the exploration comes from."""
    rng = np.random.default_rng(4)
    succ = fail = np.zeros(2)
    picks = [b.select_thompson_bernoulli(succ, fail, rng) for _ in range(4000)]
    assert np.mean(np.array(picks) == 0) == pytest.approx(0.5, abs=0.04)


# --------------------------------------------------------------------------- #
# Q2b: explore-then-commit as a pure rule
# --------------------------------------------------------------------------- #
def test_q2_etc_is_round_robin_before_the_switch():
    """For the first mK steps the arm is (t-1) mod K, whatever the data says."""
    K, m = 4, 3
    means = np.array([0.0, 0.0, 0.0, 9.0])   # arm 3 looks unbeatable
    counts = np.zeros(K)
    for t in range(1, m * K + 1):
        assert b.select_etc(means, counts, t, m) == (t - 1) % K


def test_q2_etc_ignores_the_means_while_exploring():
    """The exploration cost is exactly mK and cannot be shortened by good luck:
    during exploration the rule does not read `means` at all."""
    K, m = 3, 2
    counts = np.zeros(K)
    optimistic = np.array([9.0, 0.0, 0.0])
    pessimistic = np.array([0.0, 0.0, 9.0])
    for t in range(1, m * K + 1):
        assert (b.select_etc(optimistic, counts, t, m)
                == b.select_etc(pessimistic, counts, t, m))


def test_q2_etc_is_constant_greedy_after_the_switch():
    """After mK the choice is greedy and never changes again: a wrong commit is
    paid for the whole remaining horizon, which is where the T^(2/3) comes from."""
    K, m = 3, 2
    means = np.array([0.1, 0.7, 0.4])
    counts = np.array([m, m, m], dtype=float)
    for t in range(m * K + 1, m * K + 200):
        assert b.select_etc(means, counts, t, m) == 1


def test_q2_etc_with_one_explore_round_still_pulls_every_arm():
    """Edge case m = 1: one pull each, then commit."""
    K = 5
    counts = np.zeros(K)
    seen = {b.select_etc(np.zeros(K), counts, t, 1) for t in range(1, K + 1)}
    assert seen == set(range(K))


def test_q2_etc_commits_for_good_in_the_loop():
    """After the switch run_bandit pulls one arm only, whatever the rewards say:
    the commit uses the means of the exploration, not the running ones. With the
    running means the committed arm's estimate drifts, and on these arms ETC
    changes its mind after the switch in 128 of the seeds 0-199."""
    mu = np.array([0.50, 0.55, 0.45])
    m = 20
    for seed in range(20):
        pulls = b.run_bandit(mu, 600, "etc", seed=seed, explore_rounds=m)
        assert len(set(pulls[m * 3:].tolist())) == 1


def test_q2_greedy_tries_each_arm_once_then_commits():
    """Greedy, as in the lecture: one pull per arm, then the arm with the best
    observed reward, for the rest of the run."""
    mu = np.array([0.50, 0.55, 0.45, 0.40])
    for seed in range(20):
        pulls = b.run_bandit(mu, 300, "greedy", seed=seed)
        assert pulls[:4].tolist() == [0, 1, 2, 3]
        assert len(set(pulls[4:].tolist())) == 1


def test_q2_etc_matches_the_loop_used_by_run_bandit():
    """The pure rule and the interaction loop must agree: run_bandit pulls each
    arm exactly m times during the exploration phase."""
    mu = np.array([0.5, 0.5, 0.5])
    m = 4
    pulls = b.run_bandit(mu, m * 3, "etc", seed=0, explore_rounds=m)
    counts = np.bincount(pulls, minlength=3)
    assert list(counts) == [m, m, m]


# --------------------------------------------------------------------------- #
# Q3: pseudo-regret
# --------------------------------------------------------------------------- #
def test_q3_regret_is_zero_on_the_optimal_arm():
    mu = np.array([0.2, 0.9, 0.5])
    assert b.pseudo_regret(mu, np.array([1, 1, 1, 1])) == pytest.approx(0.0)


def test_q3_regret_equals_sum_of_gaps_times_counts():
    mu = np.array([0.2, 0.9, 0.5])
    pulls = np.array([0, 1, 2, 2, 0])
    expected = 2 * (0.9 - 0.2) + 2 * (0.9 - 0.5)
    assert b.pseudo_regret(mu, pulls) == pytest.approx(expected)


def test_q3_regret_is_never_negative():
    rng = np.random.default_rng(5)
    mu = rng.random(4)
    for _ in range(50):
        pulls = rng.integers(0, 4, size=rng.integers(1, 40))
        assert b.pseudo_regret(mu, pulls) >= -1e-12


def test_q3_regret_grows_linearly_for_a_fixed_bad_arm():
    """Committing to a suboptimal arm gives Reg(T) = T * Delta: this is the
    baseline that sublinear algorithms have to beat."""
    mu = np.array([0.9, 0.4])
    for T in (10, 100, 1000):
        assert b.pseudo_regret(mu, np.zeros(T, dtype=int) + 1) == pytest.approx(0.5 * T)


def test_q3_run_bandit_conserves_the_number_of_pulls():
    """Whatever the algorithm does, the pulls must sum to the horizon and index
    real arms. Cheap, and it catches an off-by-one in the loop immediately."""
    mu = np.array([0.2, 0.5, 0.75])
    for algo in ("greedy", "etc", "ucb", "eps", "thompson"):
        pulls = b.run_bandit(mu, 200, algo, seed=3)
        assert len(pulls) == 200
        assert set(np.unique(pulls)) <= {0, 1, 2}
