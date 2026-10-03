"""P08 public tests — the counterexample, and the arithmetic under it.

Nothing in this lab samples anything: `expected_update` is the exact expectation
of the TD update, so every test is deterministic linear algebra. A red test is a
wrong implementation, full stop.
"""
from __future__ import annotations

import numpy as np
import pytest

from rl_lab import triad as tri

GAMMA = 0.9
W0 = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0, 1.0])


def setup():
    Phi = tri.baird_features()
    P = tri.baird_mdp()
    P_mu = tri.policy_transition(P, tri.behaviour_policy())
    P_pi = tri.policy_transition(P, tri.target_policy())
    d = tri.stationary_distribution(P_mu)
    return Phi, P_mu, P_pi, d


# --------------------------------------------------------------------------- #
# Q1 — policies as matrices
# --------------------------------------------------------------------------- #
def test_q1_policy_transition_rows_are_distributions():
    P = tri.baird_mdp()
    for pi in (tri.behaviour_policy(), tri.target_policy()):
        P_pi = tri.policy_transition(P, pi)
        assert P_pi.shape == (7, 7)
        assert P_pi.sum(axis=1) == pytest.approx(np.ones(7), abs=1e-12)
        assert (P_pi >= 0).all()


def test_q1_the_target_policy_always_lands_in_state_six():
    P_pi = tri.policy_transition(tri.baird_mdp(), tri.target_policy())
    expected = np.zeros((7, 7))
    expected[:, 6] = 1.0
    assert P_pi == pytest.approx(expected, abs=1e-12)


def test_q1_behaviour_stationary_distribution_is_uniform():
    """Not an assumption: 6/7 of the time the agent lands uniformly on one of
    the six upper states, and 1/7 of the time on the lower one, so every state
    gets exactly 1/7. The whole counterexample is built on this being uniform
    while the bootstrap target is not."""
    _, P_mu, _, d = setup()
    assert d == pytest.approx(np.full(7, 1 / 7), abs=1e-12)


def test_q1_target_stationary_distribution_is_a_point_mass():
    _, _, P_pi, _ = setup()
    d = tri.stationary_distribution(P_pi)
    expected = np.zeros(7)
    expected[6] = 1.0
    assert d == pytest.approx(expected, abs=1e-12)


def test_q1_stationary_distribution_is_a_fixed_point():
    _, P_mu, _, d = setup()
    assert d @ P_mu == pytest.approx(d, abs=1e-12)
    assert d.sum() == pytest.approx(1.0, abs=1e-12)


# --------------------------------------------------------------------------- #
# Q2 — the update
# --------------------------------------------------------------------------- #
def test_q2_the_true_solution_is_a_fixed_point():
    """w = 0 represents V^pi exactly and the update leaves it alone. The
    counterexample is not that the answer is unreachable — it is sitting right
    there — but that the iteration walks away from it."""
    Phi, _, P_pi, d = setup()
    w = np.zeros(8)
    assert tri.expected_update(w, Phi, d, P_pi, GAMMA, 0.1) == pytest.approx(
        w, abs=1e-15)


def test_q2_zero_step_size_changes_nothing():
    Phi, _, P_pi, d = setup()
    assert tri.expected_update(W0, Phi, d, P_pi, GAMMA, 0.0) == pytest.approx(
        W0, abs=1e-15)


def test_q2_update_matches_the_definition_by_hand():
    """One step written out as a sum over states, without any matrix algebra,
    so that a transposed Phi cannot pass."""
    Phi, _, P_pi, d = setup()
    alpha = 0.03
    v = Phi @ W0
    dw = np.zeros(8)
    for s in range(7):
        delta = GAMMA * float(P_pi[s] @ v) - float(Phi[s] @ W0)
        dw += d[s] * delta * Phi[s]
    assert tri.expected_update(W0, Phi, d, P_pi, GAMMA, alpha) == pytest.approx(
        W0 + alpha * dw, abs=1e-12)


def test_q2_without_bootstrapping_it_is_a_plain_gradient_step():
    """With every reward zero the Monte Carlo target is zero, so the update is
    -alpha * Phi^T D Phi w: an ordinary least-squares gradient step on a convex
    objective, which is why it cannot diverge for small alpha."""
    Phi, _, P_pi, d = setup()
    alpha = 0.03
    expected = W0 - alpha * (Phi.T @ (d[:, None] * Phi) @ W0)
    got = tri.expected_update(W0, Phi, d, P_pi, GAMMA, alpha, bootstrap=False)
    assert got == pytest.approx(expected, abs=1e-12)


def test_q2_without_bootstrapping_the_target_policy_is_irrelevant():
    """No bootstrap means no next state, so passing a different P^pi cannot
    change anything. If this fails, the target is still leaking in."""
    Phi, P_mu, P_pi, d = setup()
    a = tri.expected_update(W0, Phi, d, P_pi, GAMMA, 0.05, bootstrap=False)
    b = tri.expected_update(W0, Phi, d, P_mu, GAMMA, 0.05, bootstrap=False)
    assert a == pytest.approx(b, abs=1e-15)


def test_q2_run_returns_the_whole_history():
    Phi, _, P_pi, d = setup()
    h = tri.run(W0, Phi, d, P_pi, GAMMA, 0.01, 10)
    assert h.shape == (11, 8)
    assert h[0] == pytest.approx(W0, abs=1e-15)


def test_q2_run_of_one_step_is_one_update():
    Phi, _, P_pi, d = setup()
    h = tri.run(W0, Phi, d, P_pi, GAMMA, 0.02, 1)
    assert h[1] == pytest.approx(
        tri.expected_update(W0, Phi, d, P_pi, GAMMA, 0.02), abs=1e-15)


def test_q2_run_of_zero_steps_is_the_starting_point():
    Phi, _, P_pi, d = setup()
    assert tri.run(W0, Phi, d, P_pi, GAMMA, 0.02, 0).shape == (1, 8)


# --------------------------------------------------------------------------- #
# Q3 — reading the outcome
# --------------------------------------------------------------------------- #
def test_q3_value_error_of_the_true_solution_is_zero():
    Phi, _, _, d = setup()
    assert tri.value_error(Phi, np.zeros(8), d) == pytest.approx(0.0, abs=1e-15)


def test_q3_value_error_matches_the_definition():
    Phi = np.eye(3)
    d = np.array([0.5, 0.25, 0.25])
    w = np.array([2.0, 4.0, 0.0])
    expected = np.sqrt(0.5 * 4 + 0.25 * 16 + 0.0)
    assert tri.value_error(Phi, w, d) == pytest.approx(expected, abs=1e-12)


def test_q3_off_policy_key_matrix_has_a_negative_eigenvalue():
    """The theorem, as a test. A = Phi^T D (I - gamma P^pi) Phi drives the
    expected update, and off-policy it has an eigenvalue with negative real
    part: that eigenvalue *is* the divergence."""
    Phi, _, P_pi, d = setup()
    ev = np.linalg.eigvals(tri.key_matrix(Phi, d, P_pi, GAMMA)).real
    assert ev.min() < -0.02


def test_q3_on_policy_key_matrix_has_none():
    """Change nothing but the policy the bootstrap target comes from, and every
    eigenvalue's real part is non-negative. One object changes, and a divergence
    proof becomes a convergence proof."""
    Phi, P_mu, _, d = setup()
    ev = np.linalg.eigvals(tri.key_matrix(Phi, d, P_mu, GAMMA)).real
    assert ev.min() > -1e-12


def test_q3_the_full_triad_diverges():
    """Deterministic, so this is an invariant and not a threshold on a noisy
    training run: with all three ingredients present the weighted value error
    grows by more than five orders of magnitude in 10000 steps."""
    Phi, _, P_pi, d = setup()
    h = tri.run(W0, Phi, d, P_pi, GAMMA, 0.05, 10_000)
    assert tri.value_error(Phi, h[-1], d) > 1e5 * tri.value_error(Phi, h[0], d)


def test_q3_removing_any_one_ingredient_stops_it():
    """Same start, same alpha, same horizon; one leg removed each time."""
    Phi, P_mu, P_pi, d = setup()
    start = tri.value_error(Phi, W0, d)

    on_policy = tri.run(W0, Phi, d, P_mu, GAMMA, 0.05, 10_000)
    assert tri.value_error(Phi, on_policy[-1], d) < 1e-6 * start

    no_bootstrap = tri.run(W0, Phi, d, P_pi, GAMMA, 0.05, 10_000,
                           bootstrap=False)
    assert tri.value_error(Phi, no_bootstrap[-1], d) < 1e-6 * start

    tabular = tri.run(Phi @ W0, np.eye(7), d, P_pi, GAMMA, 0.05, 10_000)
    assert tri.value_error(np.eye(7), tabular[-1], d) < 0.01 * start
