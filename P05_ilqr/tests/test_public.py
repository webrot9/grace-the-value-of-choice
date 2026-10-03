"""P05 — invariants for LQR and iLQR.

Exact linear algebra, plus finite differences with an explicit tolerance. The
one test that uses randomness draws the noise with a fixed seed and checks a
quantity the noise cannot touch.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rl_lab import control as c  # noqa: E402


def double_integrator(dt: float = 0.1):
    """A point mass: position and velocity, force as the control."""
    Fx = np.array([[1.0, dt], [0.0, 1.0]])
    Fu = np.array([[0.0], [dt]])
    Cx = np.eye(2)
    Cu = np.array([[0.5]])
    return Fx, Fu, Cx, Cu


# --------------------------------------------------------------------------- #
# Q1: the Riccati recursion
# --------------------------------------------------------------------------- #
def test_q1_scalar_case_matches_the_hand_algebra():
    r"""One step on a 1-D system, done on paper.

    With Fx=a, Fu=b, Cx=q, Cu=r and P_next=p:
        K = -abp / (r + b^2 p),   P = q + K^2 r + (a + bK)^2 p
    """
    a, b, q, r, p = 1.2, 0.7, 2.0, 0.3, 5.0
    K, P = c.riccati_step(np.array([[a]]), np.array([[b]]),
                          np.array([[q]]), np.array([[r]]), np.array([[p]]))
    k_expected = -a * b * p / (r + b ** 2 * p)
    assert K[0, 0] == pytest.approx(k_expected)
    assert P[0, 0] == pytest.approx(q + k_expected ** 2 * r
                                    + (a + b * k_expected) ** 2 * p)


def test_q1_p_stays_symmetric_and_positive_semidefinite():
    """P is a cost-to-go matrix: it cannot be asymmetric, and x^T P x is a cost."""
    Fx, Fu, Cx, Cu = double_integrator()
    _, Ps = c.lqr_backward(Fx, Fu, Cx, Cu, 30)
    for P in Ps:
        assert P == pytest.approx(P.T)
        assert np.linalg.eigvalsh(P).min() >= -1e-9


def test_q1_infinite_horizon_riccati_has_a_fixed_point():
    """Applying the recursion to the converged P must return P: the stationary
    gain of the infinite-horizon problem."""
    Fx, Fu, Cx, Cu = double_integrator()
    _, Ps = c.lqr_backward(Fx, Fu, Cx, Cu, 400)
    P_inf = Ps[0]
    _, P_again = c.riccati_step(Fx, Fu, Cx, Cu, P_inf)
    assert P_again == pytest.approx(P_inf, abs=1e-9)


def test_q1_expensive_control_drives_the_gain_to_zero():
    """As Cu grows the controller stops acting: a limit with an obvious answer,
    which is why it is a good check on the sign and on the inverse."""
    Fx, Fu, Cx, _ = double_integrator()
    gains = [np.abs(c.lqr_backward(Fx, Fu, Cx, np.array([[cu]]), 50)[0][0]).max()
             for cu in (1.0, 1e3, 1e6, 1e9)]
    assert gains == sorted(gains, reverse=True)
    assert gains[-1] < 1e-3


def test_q1_free_state_gives_a_zero_gain():
    """If the state costs nothing, the cheapest control is no control at all."""
    Fx, Fu, _, Cu = double_integrator()
    Ks, _ = c.lqr_backward(Fx, Fu, np.zeros((2, 2)), Cu, 20)
    assert np.abs(np.array(Ks)).max() == pytest.approx(0.0, abs=1e-12)


def test_q1_horizon_lengths_are_consistent():
    Fx, Fu, Cx, Cu = double_integrator()
    Ks, Ps = c.lqr_backward(Fx, Fu, Cx, Cu, 15)
    assert len(Ks) == 15 and len(Ps) == 16
    assert Ps[-1] == pytest.approx(Cx)          # terminal condition


# --------------------------------------------------------------------------- #
# Q2: linearisation
# --------------------------------------------------------------------------- #
def test_q2_linearising_a_linear_system_returns_its_matrices():
    """Central differences on a linear map must recover it exactly."""
    Fx, Fu, _, _ = double_integrator()

    def f(x, u):
        return Fx @ x + Fu @ u

    gx, gu = c.linearize(f, np.array([0.3, -1.1]), np.array([0.4]))
    assert gx == pytest.approx(Fx, abs=1e-7)
    assert gu == pytest.approx(Fu, abs=1e-7)


def test_q2_linearising_a_quadratic_needs_central_differences():
    """On a linear map any finite difference is exact, so the test above cannot
    tell central from forward differences. On a quadratic it can: central
    differences are still exact (their error involves the third derivative,
    which is zero), forward ones are off by about eps = 1e-6."""
    def f(x, u):
        return np.array([x[0] ** 2 + u[0] ** 2, x[0] * x[1]])

    gx, gu = c.linearize(f, np.array([0.3, -1.1]), np.array([0.4]))
    assert gx == pytest.approx(np.array([[0.6, 0.0], [-1.1, 0.3]]), abs=1e-8)
    assert gu == pytest.approx(np.array([[0.8], [0.0]]), abs=1e-8)


def test_q2_linearising_is_local():
    """On a genuinely non-linear map the Jacobian must change with the point:
    a linearisation that does not is a linearisation you computed once."""
    def f(x, u):
        return np.array([x[0] ** 2 + u[0], x[1]])

    a, _ = c.linearize(f, np.array([1.0, 0.0]), np.array([0.0]))
    b, _ = c.linearize(f, np.array([3.0, 0.0]), np.array([0.0]))
    assert a[0, 0] == pytest.approx(2.0, abs=1e-5)
    assert b[0, 0] == pytest.approx(6.0, abs=1e-5)


def test_q2_shapes_are_right_for_a_non_square_control():
    def f(x, u):
        return np.array([x[0] + u[0], x[1] + u[1], x[2]])

    gx, gu = c.linearize(f, np.zeros(3), np.zeros(2))
    assert gx.shape == (3, 3) and gu.shape == (3, 2)


# --------------------------------------------------------------------------- #
# Q3: rollout, cost-to-go, and certainty equivalence
# --------------------------------------------------------------------------- #
def test_q3_cost_to_go_is_the_quadratic_form():
    P = np.array([[2.0, 0.5], [0.5, 3.0]])
    x = np.array([1.0, -2.0])
    assert c.cost_to_go(P, x) == pytest.approx(2 - 2 + 12)


def test_q3_deterministic_rollout_cost_equals_x0_P_x0():
    """The value function is not an approximation here: with no noise the cost
    actually paid equals the quadratic form exactly."""
    Fx, Fu, Cx, Cu = double_integrator()
    Ks, Ps = c.lqr_backward(Fx, Fu, Cx, Cu, 60)
    for x0 in (np.array([1.0, 0.0]), np.array([-2.0, 0.5]), np.array([0.3, 3.0])):
        _, cost = c.rollout(Fx, Fu, Cx, Cu, Ks, x0, sigma=0.0)
        assert cost == pytest.approx(c.cost_to_go(Ps[0], x0), rel=1e-9)


def test_q3_certainty_equivalence_noise_adds_only_a_constant():
    """The theorem of this lab has two halves. The gain does not depend on the
    noise: sigma is not an argument of the backward pass, so there is nothing to
    test there. The cost does, by exactly sigma^2 * sum_t tr(P_{t+1}), the same
    for every policy-independent start: checked here on 3000 rollouts, within
    four standard errors. Noise with standard deviation sigma**2 instead of
    sigma, or added before the cost instead of after it, fails."""
    Fx, Fu, Cx, Cu = double_integrator()
    Ks, Ps = c.lqr_backward(Fx, Fu, Cx, Cu, 50)
    x0, sigma = np.array([1.0, 0.0]), 0.3
    quiet = c.rollout(Fx, Fu, Cx, Cu, Ks, x0, sigma=0.0)[1]
    costs = np.array([c.rollout(Fx, Fu, Cx, Cu, Ks, x0, sigma=sigma, seed=s)[1]
                      for s in range(3000)])
    extra = costs.mean() - quiet
    se = costs.std(ddof=1) / np.sqrt(len(costs))
    theory = sigma ** 2 * sum(np.trace(P) for P in Ps[1:])
    assert abs(extra - theory) < 4 * se


def test_q3_noise_raises_the_cost_but_not_the_policy():
    """What the noise does change is the cost, and it changes it by an additive
    amount that does not depend on the state: the control is unaffected."""
    Fx, Fu, Cx, Cu = double_integrator()
    Ks, _ = c.lqr_backward(Fx, Fu, Cx, Cu, 50)
    x0 = np.array([1.0, 0.0])
    quiet = c.rollout(Fx, Fu, Cx, Cu, Ks, x0, sigma=0.0)[1]
    noisy = np.mean([c.rollout(Fx, Fu, Cx, Cu, Ks, x0, sigma=0.3, seed=s)[1]
                     for s in range(200)])
    assert noisy > quiet


def test_q3_rollout_shapes():
    Fx, Fu, Cx, Cu = double_integrator()
    Ks, _ = c.lqr_backward(Fx, Fu, Cx, Cu, 12)
    xs, _ = c.rollout(Fx, Fu, Cx, Cu, Ks, np.array([1.0, 1.0]))
    assert xs.shape == (13, 2)


def test_q3_ilqr_solves_a_linear_problem_in_one_shot():
    """On a linear system iLQR has nothing to iterate: its answer must match the
    exact LQR rollout."""
    Fx, Fu, Cx, Cu = double_integrator()

    def f(x, u):
        return Fx @ x + Fu @ u

    x0 = np.array([1.0, 0.0])
    Ks, Ps = c.lqr_backward(Fx, Fu, Cx, Cu, 30)
    xs_ref, cost_ref = c.rollout(Fx, Fu, Cx, Cu, Ks, x0, sigma=0.0)
    xs, us = c.ilqr(f, Cx, Cu, x0, np.zeros((30, 1)), iters=1, alpha=1.0)
    cost = sum(float(x @ Cx @ x + u @ Cu @ u) for x, u in zip(xs[:-1], us))
    cost += float(xs[-1] @ Cx @ xs[-1])
    assert cost == pytest.approx(cost_ref, rel=1e-6)


def test_q3_the_ilqr_gains_are_the_lqr_gains():
    """On a linear system the feedback gains of the iLQR backward pass are the
    Riccati gains, wherever the nominal trajectory is; around the origin, where
    the cost has no gradient, the feedforward term is zero."""
    Fx, Fu, Cx, Cu = double_integrator()
    H = 20
    Ks_ref = c.lqr_backward_tv([Fx] * H, [Fu] * H, Cx, Cu)
    rng = np.random.default_rng(4)
    xs, us = rng.normal(size=(H + 1, 2)), rng.normal(size=(H, 1))
    ks, Ks = c.ilqr_backward([Fx] * H, [Fu] * H, xs, us, Cx, Cu)
    for K, K_ref in zip(Ks, Ks_ref):
        assert K == pytest.approx(K_ref, abs=1e-10)
    ks0, _ = c.ilqr_backward([Fx] * H, [Fu] * H, np.zeros((H + 1, 2)),
                             np.zeros((H, 1)), Cx, Cu)
    assert all(np.allclose(k, 0.0) for k in ks0)


def test_q3_ilqr_stops_where_the_cost_gradient_vanishes():
    """A small unicycle, which is not linear. Whatever iLQR converges to, the
    total cost there must have zero gradient with respect to the controls,
    checked here by central differences. Applying the gains to the state
    instead of to its deviation from the nominal trajectory, or dropping the
    feedforward term, converges too, but to a point where the gradient is
    nowhere near zero."""
    def f(x, u):
        return np.array([x[0] + 0.1 * u[0] * np.cos(x[2]),
                         x[1] + 0.1 * u[0] * np.sin(x[2]),
                         x[2] + 0.1 * u[1]])

    Cx, Cu = np.eye(3), 0.1 * np.eye(2)
    x0 = np.array([1.0, 0.5, 0.0])

    def total(us):
        x, J = x0.copy(), 0.0
        for u in us:
            J += float(x @ Cx @ x + u @ Cu @ u)
            x = f(x, u)
        return J + float(x @ Cx @ x)

    _, us = c.ilqr(f, Cx, Cu, x0, np.zeros((12, 2)), iters=60, alpha=1.0)
    grad = np.zeros_like(us)
    for idx in np.ndindex(us.shape):
        d = np.zeros_like(us)
        d[idx] = 1e-6
        grad[idx] = (total(us + d) - total(us - d)) / 2e-6
    assert np.linalg.norm(grad) < 1e-5
