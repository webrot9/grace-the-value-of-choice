"""P03 — invariants for dynamic programming.

Nothing here is trained and nothing is sampled: an MDP is three arrays and every
property is exact linear algebra. If a test fails, the code is wrong — there is
no luck involved anywhere in this file.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rl_lab import dp  # noqa: E402


def two_state_chain(gamma: float = 0.9):
    r"""A two-state MDP small enough to solve on paper.

    State 0 has two actions: `stay` (reward 0, stays) and `go` (reward 1, moves
    to state 1). State 1 is absorbing with reward 1 for both actions.

    $V^{*}(1) = 1 + \gamma V^{*}(1) \Rightarrow V^{*}(1) = 1/(1-\gamma)$, and
    $V^{*}(0) = 1 + \gamma V^{*}(1) = 1 + \gamma/(1-\gamma)$.
    """
    P = np.zeros((2, 2, 2))
    P[0, 0, 0] = 1.0        # stay
    P[0, 1, 1] = 1.0        # go
    P[1, :, 1] = 1.0        # absorbing
    R = np.array([[0.0, 1.0], [1.0, 1.0]])
    v_star = np.array([1 + gamma / (1 - gamma), 1 / (1 - gamma)])
    return P, R, v_star


# --------------------------------------------------------------------------- #
# Q1: the two operators
# --------------------------------------------------------------------------- #
def test_q1_expectation_backup_matches_the_definition():
    P, R, _ = two_state_chain()
    V = np.array([2.0, 5.0])
    pi = np.array([1, 0])                      # go from 0, anything from 1
    got = dp.bellman_expectation(P, R, V, pi, 0.9)
    assert got == pytest.approx([1 + 0.9 * 5.0, 1 + 0.9 * 5.0])


def test_q1_optimality_backup_is_the_max_over_actions():
    P, R, _ = two_state_chain()
    V = np.array([2.0, 5.0])
    got = dp.bellman_optimality(P, R, V, 0.9)
    stay = 0 + 0.9 * 2.0
    go = 1 + 0.9 * 5.0
    assert got[0] == pytest.approx(max(stay, go))


def test_q1_optimality_dominates_every_policy():
    """B V >= B_pi V pointwise, for every pi: the max is a max."""
    P, R = dp.gridworld(3, 3, seed=2)
    rng = np.random.default_rng(0)
    V = rng.normal(size=9)
    bv = dp.bellman_optimality(P, R, V, 0.9)
    for _ in range(20):
        pi = rng.integers(0, 4, size=9)
        assert np.all(bv >= dp.bellman_expectation(P, R, V, pi, 0.9) - 1e-12)


def test_q1_greedy_policy_attains_the_optimality_backup():
    """Applying B_pi with the greedy pi must give exactly B V: that identity is
    what makes the greedy step meaningful."""
    P, R = dp.gridworld(4, 4, seed=3)
    rng = np.random.default_rng(1)
    V = rng.normal(size=16)
    pi = dp.greedy_policy(P, R, V, 0.9)
    assert dp.bellman_expectation(P, R, V, pi, 0.9) == \
        pytest.approx(dp.bellman_optimality(P, R, V, 0.9))


def test_q1_gamma_zero_is_greedy_on_the_immediate_reward():
    """With no discounting of the future there is no future: V = max_a r(s,a)."""
    P, R = dp.gridworld(4, 4, seed=4)
    V = np.zeros(16)
    assert dp.bellman_optimality(P, R, V, 0.0) == pytest.approx(R.max(axis=1))
    assert np.array_equal(dp.greedy_policy(P, R, V, 0.0), R.argmax(axis=1))


# --------------------------------------------------------------------------- #
# Q2: exact solution vs iteration
# --------------------------------------------------------------------------- #
def test_q2_exact_evaluation_matches_the_hand_solution():
    gamma = 0.9
    P, R, _ = two_state_chain(gamma)
    pi = np.array([1, 0])                       # go, then absorb
    v = dp.policy_evaluation_exact(P, R, pi, gamma)
    assert v[1] == pytest.approx(1 / (1 - gamma))
    assert v[0] == pytest.approx(1 + gamma / (1 - gamma))


def test_q2_exact_evaluation_is_a_fixed_point_of_its_own_operator():
    """B_pi V^pi = V^pi. If this fails, the linear solve is solving the wrong
    system — most often because P^pi was built with the wrong index."""
    P, R = dp.gridworld(4, 4, seed=5)
    rng = np.random.default_rng(2)
    for _ in range(10):
        pi = rng.integers(0, 4, size=16)
        v = dp.policy_evaluation_exact(P, R, pi, 0.9)
        assert dp.bellman_expectation(P, R, v, pi, 0.9) == pytest.approx(v)


def test_q2_iterating_the_expectation_backup_converges_to_the_exact_solution():
    """The two ways of computing V^pi must agree: closed form and iteration."""
    P, R = dp.gridworld(4, 4, seed=6)
    pi = np.zeros(16, dtype=int)
    exact = dp.policy_evaluation_exact(P, R, pi, 0.9)
    v = np.zeros(16)
    for _ in range(400):
        v = dp.bellman_expectation(P, R, v, pi, 0.9)
    assert v == pytest.approx(exact, abs=1e-9)


def test_q2_value_iteration_reaches_the_hand_computed_optimum():
    gamma = 0.9
    P, R, v_star = two_state_chain(gamma)
    V, _ = dp.value_iteration(P, R, gamma, iters=400)
    assert V == pytest.approx(v_star, abs=1e-8)


def test_q2_value_iteration_applies_the_operator_to_the_whole_vector():
    """Each row of `history` is B applied to the previous row, all states at
    once. Updating V in place, state by state (Gauss-Seidel), also converges to
    V*, but it is a different algorithm with a different trace: the contraction
    argument, and the figure, are about the operator B."""
    gamma = 0.9
    rng = np.random.default_rng(15)
    S, A = 6, 3                    # a random MDP: on the gridworld the index
    P = rng.dirichlet(np.ones(S), size=(S, A))   # order can hide the difference
    R = rng.random((S, A))
    _, history = dp.value_iteration(P, R, gamma, iters=12)
    for k in range(12):
        assert history[k + 1] == pytest.approx(
            dp.bellman_optimality(P, R, history[k], gamma), abs=1e-12)


def test_q2_history_has_one_row_per_iteration_plus_the_start():
    P, R = dp.gridworld(3, 3, seed=7)
    _, history = dp.value_iteration(P, R, 0.9, iters=25)
    assert history.shape == (26, 9)
    assert history[0] == pytest.approx(np.zeros(9))


def test_q2_v_star_is_a_fixed_point():
    """Applying B to the answer must not move it. This is theorem 1."""
    P, R = dp.gridworld(4, 4, seed=8)
    V, _ = dp.value_iteration(P, R, 0.9, iters=500)
    assert dp.bellman_optimality(P, R, V, 0.9) == pytest.approx(V, abs=1e-9)


def test_q2_greedy_on_v_star_achieves_v_star():
    """Theorem 2: the greedy policy with respect to V* is optimal, so evaluating
    it exactly must give V* back."""
    P, R = dp.gridworld(4, 4, seed=9)
    gamma = 0.95
    V, _ = dp.value_iteration(P, R, gamma, iters=800)
    pi = dp.greedy_policy(P, R, V, gamma)
    assert dp.policy_evaluation_exact(P, R, pi, gamma) == pytest.approx(V, abs=1e-7)


# --------------------------------------------------------------------------- #
# Q3: the contraction itself
# --------------------------------------------------------------------------- #
def test_q3_sup_norm_is_the_largest_absolute_difference():
    a = np.array([1.0, -3.0, 2.0])
    b = np.array([1.5, 1.0, 2.0])
    assert dp.sup_norm(a, b) == pytest.approx(4.0)


def test_q3_optimality_operator_is_a_gamma_contraction():
    r"""The theorem: $\lVert \mathcal{B}U - \mathcal{B}V \rVert_\infty \le \gamma \lVert U - V \rVert_\infty$
    for **every** pair, not just for the ones the algorithm happens to visit."""
    P, R = dp.gridworld(4, 4, seed=10)
    rng = np.random.default_rng(3)
    for gamma in (0.5, 0.9, 0.99):
        for _ in range(30):
            U, V = rng.normal(size=16) * 5, rng.normal(size=16) * 5
            lhs = dp.sup_norm(dp.bellman_optimality(P, R, U, gamma),
                              dp.bellman_optimality(P, R, V, gamma))
            assert lhs <= gamma * dp.sup_norm(U, V) + 1e-12


def test_q3_expectation_operator_is_a_contraction_too():
    P, R = dp.gridworld(4, 4, seed=11)
    rng = np.random.default_rng(4)
    pi = rng.integers(0, 4, size=16)
    for _ in range(30):
        U, V = rng.normal(size=16), rng.normal(size=16)
        lhs = dp.sup_norm(dp.bellman_expectation(P, R, U, pi, 0.8),
                          dp.bellman_expectation(P, R, V, pi, 0.8))
        assert lhs <= 0.8 * dp.sup_norm(U, V) + 1e-12


def test_q3_error_ratio_converges_to_gamma():
    """On a deterministic grid the contraction bound is attained: the ratio of
    consecutive errors is gamma to within 1e-6, until the error reaches zero.
    This is the number the figure in Part B is built on. With slip
    it is not attained, and it does not have to be: see the next test."""
    for gamma in (0.5, 0.9, 0.99):
        P, R = dp.gridworld(4, 4, seed=12)
        V, history = dp.value_iteration(P, R, gamma, iters=400)
        ratios = dp.error_ratios(history, V)
        assert np.abs(ratios[-10:] - gamma).max() < 1e-6


def test_q3_error_ratios_never_exceed_gamma():
    """Every ratio, not just the last ones, must respect the bound, on a
    deterministic grid and on one with slip, where it is not attained."""
    gamma = 0.9
    for slip in (0.0, 0.2):
        P, R = dp.gridworld(5, 5, slip=slip, seed=13)
        V, history = dp.value_iteration(P, R, gamma, iters=200)
        assert dp.error_ratios(history, V).max() <= gamma + 1e-9


def test_q3_error_ratios_drop_the_machine_precision_tail():
    """Once the values are exact the error is 0: the next ratio is 0/x = 0 and
    then 0/0. Keeping them gives a final ratio of 0, which is not gamma and is
    not a bug in the theory."""
    gamma = 0.5
    P, R = dp.gridworld(3, 3, seed=14)
    V, history = dp.value_iteration(P, R, gamma, iters=300)
    ratios = dp.error_ratios(history, V)
    assert len(ratios) < 300
    assert ratios.min() > 0.0
