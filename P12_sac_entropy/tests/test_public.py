"""P12 public tests — soft values, the Boltzmann policy, and the tanh Jacobian.

The discrete half is exact linear algebra. The continuous half samples, with
fixed seeds and stated tolerances, plus one deterministic check that does not:
the corrected density integrates to one and the uncorrected one does not.
"""
from __future__ import annotations

import numpy as np
import pytest

from rl_lab import entropy as en

Q = np.array([1.0, 0.9, 0.2, -0.5])


# --------------------------------------------------------------------------- #
# Q1 — soft values, the Boltzmann policy, and their limits
# --------------------------------------------------------------------------- #
def test_q1_alpha_zero_is_the_maximum():
    assert en.soft_value(Q, 0.0) == pytest.approx(Q.max(), abs=1e-15)


def test_q1_soft_value_converges_to_the_maximum():
    """Not asserted: measured. As alpha shrinks the log-sum-exp comes down onto
    the max, and the gap is bounded by alpha * log|A|."""
    for alpha in (1.0, 0.1, 0.01, 1e-3, 1e-6):
        gap = en.soft_value(Q, alpha) - Q.max()
        assert 0.0 <= gap <= alpha * np.log(len(Q)) + 1e-12


def test_q1_soft_value_survives_a_tiny_alpha():
    """alpha = 1e-8 overflows the naive exp(Q/alpha). The stable form does not,
    and the collapse in Part B is measured precisely in that regime."""
    v = en.soft_value(Q, 1e-8)
    assert np.isfinite(v) and v == pytest.approx(Q.max(), abs=1e-6)


def test_q1_soft_value_is_above_the_mean_and_below_max_plus_alpha_log_a():
    for alpha in (0.05, 0.5, 5.0):
        v = en.soft_value(Q, alpha)
        assert Q.max() <= v <= Q.max() + alpha * np.log(len(Q)) + 1e-12


def test_q1_boltzmann_at_zero_is_greedy():
    p = en.boltzmann_policy(Q, 0.0)
    assert p == pytest.approx(np.array([1.0, 0.0, 0.0, 0.0]), abs=1e-15)


def test_q1_boltzmann_splits_ties_evenly():
    p = en.boltzmann_policy(np.array([2.0, 2.0, 0.0]), 0.0)
    assert p == pytest.approx(np.array([0.5, 0.5, 0.0]), abs=1e-15)


def test_q1_boltzmann_tends_to_uniform():
    p = en.boltzmann_policy(Q, 1e4)
    assert p == pytest.approx(np.full(4, 0.25), abs=1e-3)


def test_q1_boltzmann_is_a_distribution_at_every_temperature():
    for alpha in (0.0, 1e-6, 0.1, 1.0, 100.0):
        p = en.boltzmann_policy(Q, alpha)
        assert p.sum() == pytest.approx(1.0, abs=1e-12)
        assert (p >= 0).all()


def test_q1_entropy_bounds():
    assert en.entropy(np.array([1.0, 0.0, 0.0])) == pytest.approx(0.0, abs=1e-15)
    assert en.entropy(np.full(4, 0.25)) == pytest.approx(np.log(4), abs=1e-12)


def test_q1_entropy_is_monotone_in_the_temperature():
    """The policy's entropy rises with alpha, from 0 to log|A|. That monotone
    curve is the left panel of the figure, and it is what 'the temperature is
    the exploration' means."""
    h = [en.entropy(en.boltzmann_policy(Q, a))
         for a in (1e-6, 0.01, 0.1, 1.0, 10.0, 100.0)]
    assert all(b >= a - 1e-12 for a, b in zip(h, h[1:]))
    assert h[0] < 1e-3 and h[-1] > np.log(4) - 1e-2


def test_q1_the_boltzmann_policy_maximises_reward_plus_entropy():
    """The defining property, checked against 20000 random distributions: no
    other policy achieves a higher E_pi[Q] + alpha H(pi)."""
    rng = np.random.default_rng(0)
    alpha = 0.4
    p = en.boltzmann_policy(Q, alpha)
    best = float(p @ Q) + alpha * en.entropy(p)
    other = rng.dirichlet(np.ones(4), size=20_000)
    values = other @ Q + alpha * np.array([en.entropy(o) for o in other])
    assert values.max() <= best + 1e-9


# --------------------------------------------------------------------------- #
# Q2 — the soft backup
# --------------------------------------------------------------------------- #
def test_q2_soft_iteration_reduces_to_value_iteration_at_zero():
    """alpha = 0 turns the log-sum-exp back into a max, so the fixed point must
    be the ordinary Q*."""
    P, R = en.gridworld(4, 4, slip=0.1, seed=3)
    Q_soft, _ = en.soft_policy_iteration(P, R, 0.9, 0.0)
    V = np.zeros(P.shape[0])
    for _ in range(20_000):
        V_new = (R + 0.9 * (P @ V)).max(axis=1)
        if np.max(np.abs(V_new - V)) < 1e-13:
            V = V_new
            break
        V = V_new
    assert Q_soft == pytest.approx(R + 0.9 * (P @ V), abs=1e-8)


def test_q2_the_soft_backup_is_a_fixed_point():
    P, R = en.gridworld(4, 4, slip=0.1, seed=3)
    alpha = 0.2
    Q, _ = en.soft_policy_iteration(P, R, 0.9, alpha)
    V = np.array([en.soft_value(Q[s], alpha) for s in range(P.shape[0])])
    assert R + 0.9 * (P @ V) == pytest.approx(Q, abs=1e-9)


def test_q2_more_temperature_never_lowers_the_soft_value():
    """log-sum-exp is above the max by an amount that grows with alpha, so the
    soft optimal value is monotone in the temperature."""
    P, R = en.gridworld(4, 4, slip=0.1, seed=3)
    values = [en.soft_policy_iteration(P, R, 0.9, a)[0].max()
              for a in (0.0, 0.05, 0.2, 1.0)]
    assert all(b >= a - 1e-9 for a, b in zip(values, values[1:]))


def test_q2_the_returned_policy_is_the_boltzmann_policy_of_its_own_q():
    P, R = en.gridworld(3, 3, slip=0.0, seed=0, n_obstacles=0)
    Q, pi = en.soft_policy_iteration(P, R, 0.9, 0.3)
    for s in range(P.shape[0]):
        assert pi[s] == pytest.approx(en.boltzmann_policy(Q[s], 0.3), abs=1e-12)


# --------------------------------------------------------------------------- #
# Q3 — the squashed Gaussian
# --------------------------------------------------------------------------- #
def test_q3_gaussian_log_prob_matches_the_formula():
    u = np.array([[0.3, -1.2]])
    mu = np.array([0.1, 0.0])
    log_std = np.array([np.log(0.5), np.log(2.0)])
    std = np.exp(log_std)
    expected = np.sum(-0.5 * ((u - mu) / std) ** 2 - log_std
                      - 0.5 * np.log(2 * np.pi))
    assert en.gaussian_log_prob(u, mu, log_std)[0] == pytest.approx(expected,
                                                                    abs=1e-12)


def test_q3_gaussian_density_integrates_to_one():
    mu, log_std = np.array([0.4]), np.array([np.log(0.7)])
    x = np.linspace(-8, 8, 40_001)[:, None]
    d = np.exp(en.gaussian_log_prob(x, mu, log_std))
    assert np.trapezoid(d, x[:, 0]) == pytest.approx(1.0, abs=1e-6)


def test_q3_the_corrected_density_integrates_to_one():
    """The check that the Jacobian term is right, and it needs no sampling: a
    log density has to exponentiate to something that integrates to one over the
    support, which for tanh is (-1, 1)."""
    mu, log_std = np.array([0.3]), np.array([np.log(0.8)])
    a = np.linspace(-0.99999, 0.99999, 200_001)
    u = np.arctanh(a)[:, None]
    d = np.exp(en.squashed_log_prob(u, mu, log_std))
    assert np.trapezoid(d, a) == pytest.approx(1.0, abs=1e-4)


def test_q3_the_uncorrected_one_does_not():
    """Same numbers without the Jacobian: it integrates to 0.66. Whatever that
    is, it is not a probability density, and an 'entropy bonus' computed from
    it is not an entropy."""
    mu, log_std = np.array([0.3]), np.array([np.log(0.8)])
    a = np.linspace(-0.99999, 0.99999, 200_001)
    u = np.arctanh(a)[:, None]
    d = np.exp(en.gaussian_log_prob(u, mu, log_std))
    assert abs(np.trapezoid(d, a) - 1.0) > 0.2


def test_q3_the_correction_only_ever_raises_the_density():
    """1 - tanh^2 u <= 1, so its log is negative and subtracting it adds.
    Squashing concentrates mass onto a bounded interval; the density must go
    up."""
    rng = np.random.default_rng(0)
    mu, log_std = np.array([0.0]), np.array([0.0])
    u, _ = en.sample_squashed(mu, log_std, 500, rng)
    assert (en.squashed_log_prob(u, mu, log_std)
            >= en.gaussian_log_prob(u, mu, log_std) - 1e-12).all()


def test_q3_the_log_jacobian_is_exact_in_the_tails():
    """A wide policy puts most of its u far out, where tanh u is 1 to machine
    precision. There log(1 - tanh^2 u) = -2 log cosh u exactly: about -18.6 at
    u = 10 and -38.6 at u = 20. Forming 1 - tanh^2 u gives -inf at u = 20, and
    adding an epsilon inside the log floors the term at log(epsilon), which
    moves the entropy of a sigma = 10 policy by several nats."""
    mu, log_std = np.array([0.0]), np.array([0.0])
    u = np.array([[-20.0], [-10.0], [0.5], [10.0], [20.0]])
    correction = (en.gaussian_log_prob(u, mu, log_std)
                  - en.squashed_log_prob(u, mu, log_std))
    log_cosh = np.abs(u[:, 0]) + np.log1p(np.exp(-2 * np.abs(u[:, 0]))) - np.log(2)
    assert np.all(np.isfinite(correction))
    assert correction == pytest.approx(-2 * log_cosh, abs=1e-9)


def test_q3_sampling_shapes_and_range():
    rng = np.random.default_rng(1)
    u, a = en.sample_squashed(np.array([0.0, 1.0]), np.array([0.0, -0.5]),
                              256, rng)
    assert u.shape == (256, 2) and a.shape == (256, 2)
    assert (np.abs(a) < 1.0).all()
    assert np.allclose(np.tanh(u), a)


def test_q3_monte_carlo_entropy_matches_a_numerical_integral():
    """-E[log pi] estimated from 200000 samples against the integral of
    -p log p over (-1, 1). Both are approximations; they must agree to 0.01."""
    mu, log_std = np.array([0.2]), np.array([np.log(0.6)])
    rng = np.random.default_rng(2)
    mc = en.policy_entropy(mu, log_std, 200_000, rng)
    a = np.linspace(-0.999999, 0.999999, 400_001)
    u = np.arctanh(a)[:, None]
    lp = en.squashed_log_prob(u, mu, log_std)
    exact = -np.trapezoid(np.exp(lp) * lp, a)
    assert mc == pytest.approx(exact, abs=0.01)


def test_q2_the_entropy_bonus_is_bounded_by_alpha_log_a_over_one_minus_gamma():
    """The per-step bonus is at most alpha*log|A|; over an infinite discounted
    horizon it accumulates a factor 1/(1-gamma). Quoting the per-step bound for
    the whole value function understates it tenfold at gamma = 0.9, which is a
    mistake this test exists to prevent."""
    P, R = en.gridworld(4, 4, slip=0.1, seed=3)
    gamma, A = 0.9, R.shape[1]
    Q_hard, _ = en.soft_policy_iteration(P, R, gamma, 0.0)
    for alpha in (0.001, 0.03, 0.3, 3.0):
        Q, _ = en.soft_policy_iteration(P, R, gamma, alpha)
        gap = en.soft_value(Q[0], alpha) - float(Q_hard[0].max())
        assert 0.0 <= gap <= alpha * np.log(A) / (1 - gamma) + 1e-9
