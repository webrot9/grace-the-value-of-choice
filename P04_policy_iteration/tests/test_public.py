"""P04 — invariants for policy iteration.

Exact linear algebra again: no sampling, no training, nothing flaky. The whole
lab is about one theorem — improvement is monotone **in every state** — and
about the consequence that makes policy iteration terminate.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rl_lab import pi as m  # noqa: E402


def three_state():
    """A hand-checkable MDP where the greedy step must change the policy once.

    State 0: action 0 stays (r=0), action 1 goes to state 1 (r=0).
    State 1: action 0 stays (r=0), action 1 goes to state 2 (r=0).
    State 2: absorbing, r=1 for both actions.
    Starting from "always action 0" the value is 0 everywhere except state 2.
    """
    P = np.zeros((3, 2, 3))
    P[0, 0, 0] = P[1, 0, 1] = 1.0
    P[0, 1, 1] = P[1, 1, 2] = 1.0
    P[2, :, 2] = 1.0
    R = np.zeros((3, 2))
    R[2, :] = 1.0
    return P, R


# --------------------------------------------------------------------------- #
# Q1: advantage and improvement
# --------------------------------------------------------------------------- #
def test_q1_advantage_of_the_current_action_is_zero():
    """A^pi(s, pi(s)) = 0 by construction. If this fails, V^pi and Q^pi were not
    computed from the same policy."""
    P, R = m.gridworld(4, 4, seed=1)
    rng = np.random.default_rng(0)
    for _ in range(10):
        pi = rng.integers(0, 4, size=16)
        A = m.advantage(P, R, pi, 0.9)
        assert A[np.arange(16), pi] == pytest.approx(np.zeros(16), abs=1e-9)


def test_q1_advantage_matches_q_minus_v():
    P, R = m.gridworld(3, 3, seed=2)
    pi = np.zeros(9, dtype=int)
    V = m.policy_evaluation_exact(P, R, pi, 0.9)
    expected = m.q_from_v(P, R, V, 0.9) - V[:, None]
    assert m.advantage(P, R, pi, 0.9) == pytest.approx(expected)


def test_q1_improvement_keeps_a_maximising_action():
    """The tie rule: if the current action is among the maximisers, keep it.
    State 0 has two actions with exactly the same reward and the same next
    state, so they tie; the policy uses the second one, and improvement must not
    move it to the first. (Plain `argmax` would: in exact arithmetic the loop
    still stops, but with rounding two tied actions can swap for ever, which is
    Part C.)"""
    P = np.zeros((2, 2, 2))
    P[0, :, 1] = 1.0            # from state 0 both actions go to state 1
    P[1, :, 1] = 1.0            # state 1 is absorbing
    R = np.array([[1.0, 1.0], [0.0, 0.0]])
    pi = np.array([1, 0])
    assert list(m.policy_improvement(P, R, pi, 0.9)) == [1, 0]


def test_q1_improvement_is_a_fixed_point_at_the_optimum():
    """Once the policy is optimal, improving it must return it unchanged."""
    P, R = m.gridworld(4, 4, seed=4)
    pi, _ = m.policy_iteration(P, R, 0.9)
    assert np.array_equal(m.policy_improvement(P, R, pi, 0.9), pi)


def test_q1_improvement_changes_a_deliberately_bad_policy():
    P, R = three_state()
    pi = np.zeros(3, dtype=int)              # always "stay"
    better = m.policy_improvement(P, R, pi, 0.9)
    assert not np.array_equal(better, pi)


# --------------------------------------------------------------------------- #
# Q2: the loop
# --------------------------------------------------------------------------- #
def test_q2_returns_one_value_row_per_policy_visited():
    """Row k is the exact value of the k-th policy of the sequence, rebuilt here
    step by step with policy_improvement; the loop stops at the first policy
    that improvement leaves unchanged, and that policy is not evaluated twice."""
    P, R = m.gridworld(4, 4, seed=5)
    pi, values = m.policy_iteration(P, R, 0.9)
    assert values.ndim == 2 and values.shape[1] == 16
    seq = [np.zeros(16, dtype=int)]
    while True:
        nxt = m.policy_improvement(P, R, seq[-1], 0.9)
        if np.array_equal(nxt, seq[-1]):
            break
        seq.append(nxt)
    assert values.shape[0] == len(seq)
    for row, p in zip(values, seq):
        assert row == pytest.approx(m.policy_evaluation_exact(P, R, p, 0.9), abs=1e-9)
    assert np.array_equal(pi, seq[-1])


def test_q2_converges_to_the_bellman_optimality_equation():
    """At the end the policy is greedy with respect to its own value, so
    V^pi = max_a Q^pi(., a): that identity *is* the optimality equation."""
    P, R = m.gridworld(4, 4, seed=6)
    gamma = 0.9
    pi, values = m.policy_iteration(P, R, gamma)
    V = values[-1]
    assert m.q_from_v(P, R, V, gamma).max(axis=1) == pytest.approx(V, abs=1e-9)


def test_q2_result_does_not_depend_on_the_starting_policy():
    """Different starts, same optimal value. The path differs, the fixed point
    does not."""
    P, R = m.gridworld(4, 4, seed=7)
    rng = np.random.default_rng(1)
    _, base = m.policy_iteration(P, R, 0.9)
    for _ in range(5):
        _, v = m.policy_iteration(P, R, 0.9, pi0=rng.integers(0, 4, size=16))
        assert v[-1] == pytest.approx(base[-1], abs=1e-9)


def test_q2_gamma_zero_converges_in_one_improvement():
    """With gamma = 0 the greedy policy on the immediate reward is already
    optimal, so there is nothing left to improve after the first step."""
    P, R = m.gridworld(4, 4, seed=8)
    assert m.n_distinct_policies(P, R, 0.0) <= 2


# --------------------------------------------------------------------------- #
# Q3: monotonicity and termination
# --------------------------------------------------------------------------- #
def test_q3_improvement_is_monotone_in_every_state():
    """The theorem, checked pointwise and not on an average, on twenty MDPs."""
    for seed in range(20):
        P, R = m.gridworld(4, 4, slip=0.1, seed=seed)
        _, values = m.policy_iteration(P, R, 0.9)
        assert m.is_monotone(values)


def test_q3_is_monotone_rejects_a_step_that_is_worse_anywhere():
    """A sequence that improves on average but loses in one state is not
    monotone: the quantifier is 'for all s', and this test is what checks that
    you implemented that and not a comparison of means."""
    values = np.array([[1.0, 1.0, 1.0],
                       [3.0, 0.9, 3.0]])      # mean up, state 1 down
    assert not m.is_monotone(values)


def test_q3_monotone_accepts_an_unchanged_step():
    values = np.array([[1.0, 2.0], [1.0, 2.0]])
    assert m.is_monotone(values)


def test_q3_terminates_in_few_policies():
    """Every step is a strict improvement somewhere, so no policy can repeat.
    The bound is |A|^|S|; what actually happens is a handful."""
    for seed in range(10):
        P, R = m.gridworld(4, 4, seed=seed)
        n = m.n_distinct_policies(P, R, 0.9)
        assert 1 <= n <= 16


def test_q3_no_policy_is_visited_twice():
    """The direct statement of why it terminates: the sequence of value vectors
    contains no duplicates until it stops."""
    P, R = m.gridworld(5, 5, slip=0.2, seed=9)
    _, values = m.policy_iteration(P, R, 0.95)
    seen = {tuple(np.round(v, 9)) for v in values}
    assert len(seen) == len(values)
