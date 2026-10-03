"""P11 public tests — the policy gradient, against a reference that does not
sample.

The bandit objective is available in closed form, so the estimator can be
checked against an exact gradient and against finite differences of the true
objective. Where sampling is unavoidable the tests use 10^5 samples, a fixed
seed and a stated tolerance.
"""
from __future__ import annotations

import numpy as np
import pytest

from rl_lab import pg


THETA = np.array([0.3, -0.2, 0.8, 0.0])
REWARDS = np.array([1.0, 0.5, -0.2, 2.0])


# --------------------------------------------------------------------------- #
# Q1 — the policy and its gradient
# --------------------------------------------------------------------------- #
def test_q1_softmax_is_a_distribution():
    p = pg.softmax(THETA)
    assert p.sum() == pytest.approx(1.0, abs=1e-12)
    assert (p > 0).all()


def test_q1_softmax_is_shift_invariant():
    assert pg.softmax(THETA) == pytest.approx(pg.softmax(THETA + 17.0),
                                              abs=1e-12)


def test_q1_softmax_survives_large_logits():
    """Without the max subtraction this returns nan, silently, and a training
    run that was going fine stops going anywhere."""
    p = pg.softmax(np.array([800.0, 799.0, 0.0]))
    assert np.isfinite(p).all()
    assert p.sum() == pytest.approx(1.0, abs=1e-12)
    assert p.argmax() == 0


def test_q1_log_prob_grad_sums_to_zero():
    """The gradient of a log-probability under a softmax always sums to zero:
    probability moved to one action comes from the others."""
    for a in range(4):
        assert pg.log_prob_grad(THETA, a).sum() == pytest.approx(0.0, abs=1e-12)


def test_q1_log_prob_grad_matches_the_formula():
    a = 2
    expected = -pg.softmax(THETA)
    expected[a] += 1.0
    assert pg.log_prob_grad(THETA, a) == pytest.approx(expected, abs=1e-12)


def test_q1_log_prob_grad_against_finite_differences():
    """The definition, checked numerically: the gradient of log pi(a) really is
    the thing the estimator multiplies by."""
    a = 3
    fd = pg.finite_difference_gradient(
        lambda t: float(np.log(pg.softmax(t)[a])), THETA, eps=1e-6)
    assert pg.log_prob_grad(THETA, a) == pytest.approx(fd, abs=1e-7)


def test_q1_exact_gradient_matches_finite_differences():
    """No sampling anywhere in this test. The analytic gradient of J and the
    finite-difference gradient of J agree to 1e-7, which is what makes finite
    differences usable as a reference for the sampled estimator."""
    fd = pg.finite_difference_gradient(
        lambda t: pg.bandit_objective(t, REWARDS), THETA, eps=1e-5)
    assert pg.exact_bandit_gradient(THETA, REWARDS) == pytest.approx(fd,
                                                                     abs=1e-7)


def test_q1_a_constant_reward_has_zero_gradient():
    """If every action pays the same, no policy is better than any other and
    the gradient must vanish exactly."""
    g = pg.exact_bandit_gradient(THETA, np.full(4, 2.5))
    assert g == pytest.approx(np.zeros(4), abs=1e-12)


# --------------------------------------------------------------------------- #
# Q2 — the estimator and the baseline
# --------------------------------------------------------------------------- #
def test_q2_the_estimator_is_unbiased():
    """The theorem of the session: 10^5 samples, fixed seed, and the mean of
    the estimator matches the exact gradient to well under 5% relative error."""
    rng = np.random.default_rng(0)
    s = pg.reinforce_samples(THETA, REWARDS, 100_000, 0.0, rng)
    exact = pg.exact_bandit_gradient(THETA, REWARDS)
    rel = np.abs(s.mean(axis=0) - exact).max() / np.abs(exact).max()
    assert rel < 0.05


def test_q2_a_baseline_does_not_move_the_mean():
    """Any baseline that does not depend on the action leaves the estimator
    unbiased — including a wildly wrong one."""
    exact = pg.exact_bandit_gradient(THETA, REWARDS)
    for b in (-5.0, 0.0, 0.81, 7.0):
        rng = np.random.default_rng(1)
        m = pg.reinforce_samples(THETA, REWARDS, 100_000, b, rng).mean(axis=0)
        rel = np.abs(m - exact).max() / np.abs(exact).max()
        assert rel < 0.06, f"baseline {b} moved the mean"


def test_q2_the_optimal_baseline_beats_no_baseline():
    rng = np.random.default_rng(2)
    b = pg.optimal_constant_baseline(THETA, REWARDS)
    without = pg.total_variance(pg.reinforce_samples(THETA, REWARDS, 60_000,
                                                     0.0, rng))
    with_b = pg.total_variance(pg.reinforce_samples(THETA, REWARDS, 60_000,
                                                    b, rng))
    assert with_b < 0.5 * without


def test_q2_a_bad_baseline_makes_it_worse():
    """The half of the theorem that gets dropped: subtracting a baseline is
    unbiased for *any* constant, and the variance is a parabola in it. Pick the
    wrong constant and you pay."""
    rng = np.random.default_rng(3)
    good = pg.total_variance(pg.reinforce_samples(THETA, REWARDS, 60_000,
                                                  0.0, rng))
    bad = pg.total_variance(pg.reinforce_samples(THETA, REWARDS, 60_000,
                                                 10.0, rng))
    assert bad > 10 * good


def test_q2_the_optimal_baseline_is_not_the_mean_reward():
    """It is a weighted mean, weighted by how hard each action pulls the
    parameters. Everybody writes 'subtract the average return'; that is a
    different number."""
    b = pg.optimal_constant_baseline(THETA, REWARDS)
    mean_reward = float(pg.softmax(THETA) @ REWARDS)
    assert abs(b - mean_reward) > 0.1


def test_q2_total_variance_is_the_trace_of_the_covariance():
    x = np.array([[1.0, 2.0], [3.0, 6.0], [5.0, 4.0]])
    assert pg.total_variance(x) == pytest.approx(x.var(axis=0).sum(), abs=1e-12)


def test_q2_samples_have_the_right_shape():
    rng = np.random.default_rng(0)
    s = pg.reinforce_samples(THETA, REWARDS, 37, 0.0, rng)
    assert s.shape == (37, 4)
    assert np.allclose(s.sum(axis=1), 0.0, atol=1e-12)


# --------------------------------------------------------------------------- #
# Q3 — over episodes
# --------------------------------------------------------------------------- #
def test_q3_episode_gradient_matches_the_sum_by_hand():
    theta = np.zeros((3, 2))
    states = np.array([0, 1, 0])
    actions = np.array([0, 1, 1])
    rewards = np.array([1.0, 0.0, 2.0])
    gamma = 0.5
    G = np.array([1.0 + 0.5 * (0.0 + 0.5 * 2.0), 0.0 + 0.5 * 2.0, 2.0])
    expected = np.zeros((3, 2))
    without = np.zeros((3, 2))
    for t in range(3):
        term = pg.log_prob_grad(theta[states[t]], actions[t]) * G[t]
        expected[states[t]] += gamma ** t * term
        without[states[t]] += term
    got = pg.episode_gradient(states, actions, rewards, theta, gamma)
    assert got == pytest.approx(expected, abs=1e-12)
    got = pg.episode_gradient(states, actions, rewards, theta, gamma,
                              gamma_t=False)
    assert got == pytest.approx(without, abs=1e-12)


def test_q3_a_state_baseline_subtracts_per_state():
    theta = np.zeros((3, 2))
    states = np.array([0, 1])
    actions = np.array([0, 1])
    rewards = np.array([1.0, 1.0])
    b = np.array([0.5, 0.25, 0.0])
    plain = pg.episode_gradient(states, actions, rewards, theta, 1.0)
    shifted = pg.episode_gradient(states, actions, rewards, theta, 1.0, b)
    G = pg.discounted_returns(rewards, 1.0)
    for t, s in enumerate(states):
        delta = pg.log_prob_grad(theta[s], actions[t]) * b[s]
        assert (plain[s] - shifted[s]) == pytest.approx(delta, abs=1e-12)


def test_q3_returns_from_t_onwards_not_the_whole_episode():
    """The estimator uses G_t, not the total return of the episode. Both are
    unbiased; only one of them is the one written in the docstring, and the
    difference shows up here on an episode whose rewards are not constant."""
    theta = np.zeros((2, 2))
    states = np.array([0, 1])
    actions = np.array([0, 0])
    rewards = np.array([10.0, 1.0])
    got = pg.episode_gradient(states, actions, rewards, theta, 1.0)
    whole = rewards.sum()
    naive = np.zeros((2, 2))
    for t, s in enumerate(states):
        naive[s] += pg.log_prob_grad(theta[s], actions[t]) * whole
    assert not np.allclose(got, naive)


def test_q3_zero_gamma_keeps_only_the_first_reward():
    """With gamma = 0 the objective is the first reward alone, and gamma^t is
    zero from step 1 on. Without gamma^t every step keeps its own immediate
    reward instead."""
    theta = np.zeros((2, 2))
    states = np.array([0, 1])
    actions = np.array([1, 0])
    rewards = np.array([3.0, 7.0])
    got = pg.episode_gradient(states, actions, rewards, theta, 0.0)
    expected = np.zeros((2, 2))
    expected[0] = pg.log_prob_grad(theta[0], actions[0]) * rewards[0]
    assert got == pytest.approx(expected, abs=1e-12)
    got = pg.episode_gradient(states, actions, rewards, theta, 0.0,
                              gamma_t=False)
    for t, s in enumerate(states):
        expected[s] = pg.log_prob_grad(theta[s], actions[t]) * rewards[t]
    assert got == pytest.approx(expected, abs=1e-12)


def test_q3_gamma_t_makes_the_estimate_unbiased():
    """Three states, the last one terminal, and every episode listed with its
    probability, so the expected estimate is computed exactly, with no
    sampling, and compared with finite differences of J(theta) = E[G_0]. With
    gamma^t they agree. Without it, the decision in state 1, which can only be
    taken at step 1, is weighted 1 instead of gamma, and that block of the
    estimate is off by exactly that factor."""
    gamma = 0.5
    R = np.array([[0.0, 1.0], [2.0, 0.0]])
    theta = np.array([[0.3, -0.4], [0.1, 0.5], [0.0, 0.0]])

    def episodes(th):
        p0, p1 = pg.softmax(th[0]), pg.softmax(th[1])
        yield p0[1], [0], [1], [R[0, 1]]
        for a in (0, 1):
            yield p0[0] * p1[a], [0, 1], [0, a], [R[0, 0], R[1, a]]

    def J(flat):
        th = flat.reshape(theta.shape)
        return sum(p * pg.discounted_returns(np.array(r), gamma)[0]
                   for p, _, _, r in episodes(th))

    exact = pg.finite_difference_gradient(J, theta.ravel()).reshape(theta.shape)

    def mean_estimate(**kw):
        return sum(p * pg.episode_gradient(np.array(s), np.array(a),
                                           np.array(r), theta, gamma, **kw)
                   for p, s, a, r in episodes(theta))

    assert np.abs(exact[1]).max() > 0.01
    assert mean_estimate() == pytest.approx(exact, abs=1e-8)
    without = mean_estimate(gamma_t=False)
    assert without[0] == pytest.approx(exact[0], abs=1e-8)
    assert without[1] == pytest.approx(exact[1] / gamma, abs=1e-8)


def test_q3_training_is_reproducible_and_the_right_shape():
    P, R = pg.gridworld(4, 4, slip=0.1, seed=3)
    a = pg.train(P, R, terminal=15, n_iters=8, batch=3, seed=0)
    b = pg.train(P, R, terminal=15, n_iters=8, batch=3, seed=0)
    assert a[0].shape == (16, 4)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert len(a[1]) == 8 and len(a[2]) == 8


def test_q3_training_improves_the_return():
    """Not a threshold on the final reward: a comparison of the same run
    against its own start, over four seeds, on a problem where the untrained
    policy is uniform."""
    P, R = pg.gridworld(4, 4, slip=0.1, seed=3)
    gains = []
    for seed in range(4):
        _, returns, _ = pg.train(P, R, terminal=15, n_iters=120, batch=10,
                                 alpha=0.5, seed=seed)
        gains.append(returns[-10:].mean() - returns[:10].mean())
    assert min(gains) > 0.0
