"""P09 public tests — maximization bias, and the two targets that differ by one
array.

Part A is deterministic except where it is explicitly a frequency check, and
those are seeded with a stated tolerance. No test depends on how well anything
trained.
"""
from __future__ import annotations

import numpy as np
import pytest

from rl_lab import maxbias as mb


# --------------------------------------------------------------------------- #
# Q1 — the bias, isolated from reinforcement learning
# --------------------------------------------------------------------------- #
def test_q1_a_single_estimate_is_unbiased():
    """Sanity: one sample mean of 20 draws is unbiased. If this fails the rest
    of the lab is measuring a bug, not a bias."""
    rng = np.random.default_rng(0)
    draws = mb.max_of_estimates(rng, 1, -0.1, 1.0, 200, 20_000)
    assert draws.mean() == pytest.approx(-0.1, abs=0.01)


def test_q1_the_maximum_of_unbiased_estimates_is_biased_upwards():
    """Ten actions, all worth exactly -0.1, all estimated without bias. The
    maximum of the ten estimates sits well above -0.1, and this is the entire
    phenomenon: no environment, no bootstrapping, no learning."""
    rng = np.random.default_rng(1)
    m = mb.max_of_estimates(rng, 10, -0.1, 1.0, 20, 20_000)
    assert m.mean() > 0.15


def test_q1_more_data_shrinks_the_bias():
    """The bias is a small-sample effect: quadruple the samples and it should
    roughly halve, because it scales with the standard error."""
    rng = np.random.default_rng(2)
    few = mb.max_of_estimates(rng, 10, -0.1, 1.0, 25, 20_000).mean() + 0.1
    many = mb.max_of_estimates(rng, 10, -0.1, 1.0, 100, 20_000).mean() + 0.1
    assert 1.7 < few / many < 2.3


def test_q1_the_double_estimate_removes_it():
    rng = np.random.default_rng(3)
    d = mb.double_estimate(rng, 10, -0.1, 1.0, 40, 20_000)
    assert d.mean() == pytest.approx(-0.1, abs=0.02)


def test_q1_one_action_has_nothing_to_maximise_over():
    """With a single action there is no selection, so the maximum is unbiased
    and the two estimators must agree in expectation."""
    rng = np.random.default_rng(4)
    a = mb.max_of_estimates(rng, 1, 0.5, 1.0, 40, 20_000).mean()
    b = mb.double_estimate(rng, 1, 0.5, 1.0, 40, 20_000).mean()
    assert a == pytest.approx(0.5, abs=0.02)
    assert b == pytest.approx(0.5, abs=0.03)


# --------------------------------------------------------------------------- #
# Q2 — the two targets
# --------------------------------------------------------------------------- #
def test_q2_q_learning_target_is_reward_plus_discounted_max():
    q_next = np.array([1.0, 4.0, 2.0])
    assert mb.q_learning_target(0.5, q_next, 0.9, done=False) == pytest.approx(
        0.5 + 0.9 * 4.0, abs=1e-12)


def test_q2_a_terminal_transition_has_no_bootstrap():
    q_next = np.array([100.0, 100.0])
    assert mb.q_learning_target(2.0, q_next, 0.9, done=True) == pytest.approx(
        2.0, abs=1e-12)
    assert mb.double_q_target(2.0, q_next, q_next, 0.9, done=True) == (
        pytest.approx(2.0, abs=1e-12))


def test_q2_double_target_uses_the_second_table_to_price_the_choice():
    """The first array picks the action, the second says what it is worth. Here
    they disagree about which action is best, which is the only case where the
    two algorithms can differ at all."""
    q_select = np.array([0.0, 5.0, 1.0])       # argmax is action 1
    q_evaluate = np.array([9.0, -2.0, 3.0])    # but action 1 is worth -2
    got = mb.double_q_target(1.0, q_select, q_evaluate, 0.5, done=False)
    assert got == pytest.approx(1.0 + 0.5 * (-2.0), abs=1e-12)


def test_q2_the_same_table_twice_is_plain_q_learning():
    """Pass one array as both arguments and Double Q-learning collapses onto
    Q-learning exactly. The difference between the two algorithms is not the
    formula: it is which estimates go into it."""
    rng = np.random.default_rng(5)
    for _ in range(20):
        q = rng.normal(size=6)
        assert mb.double_q_target(0.3, q, q, 0.95, False) == pytest.approx(
            mb.q_learning_target(0.3, q, 0.95, False), abs=1e-12)


def test_q2_gamma_zero_ignores_the_next_state():
    q = np.array([50.0, -50.0])
    assert mb.q_learning_target(1.5, q, 0.0, False) == pytest.approx(1.5, abs=1e-12)


# --------------------------------------------------------------------------- #
# Q3 — inside the algorithm
# --------------------------------------------------------------------------- #
def test_q3_run_returns_two_series_of_the_right_length():
    left, max_q = mb.run_bias_mdp(n_episodes=50, seed=0)
    assert left.shape == (50,) and max_q.shape == (50,)
    assert set(np.unique(left)) <= {0.0, 1.0}


def test_q3_the_same_seed_gives_the_same_run():
    a = mb.run_bias_mdp(n_episodes=40, seed=7)
    b = mb.run_bias_mdp(n_episodes=40, seed=7)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_q3_q_learning_prefers_the_wrong_action_early():
    """Early on, Q-learning takes left from A far more often than the 5% an
    epsilon-greedy agent on the correct values would. 200 seeds, and the effect
    is not subtle: the measured figure is above 70%."""
    left = np.array([mb.run_bias_mdp(n_episodes=50, seed=s)[0]
                     for s in range(200)])
    assert left[:, :50].mean() > 0.7


def test_q3_double_q_learning_does_not():
    left = np.array([mb.run_bias_mdp(n_episodes=50, double=True, seed=s)[0]
                     for s in range(200)])
    assert left[:, :50].mean() < 0.45


def test_q3_both_end_up_near_the_epsilon_greedy_floor():
    """Neither algorithm is wrong in the limit: with epsilon = 0.1 and two
    actions at A, an agent with correct values takes left 5% of the time. The
    difference between them is how long they spend being wrong."""
    for double, ceiling in ((False, 0.20), (True, 0.12)):
        left = np.array([mb.run_bias_mdp(n_episodes=300, double=double, seed=s)[0]
                         for s in range(100)])
        assert left[:, -50:].mean() < ceiling


def test_q3_overestimation_is_signed():
    got = mb.overestimation(np.array([0.4, -0.3]), truth=-0.1)
    assert got == pytest.approx(np.array([0.5, -0.2]), abs=1e-12)


def test_q3_zero_episodes_gives_empty_series():
    left, max_q = mb.run_bias_mdp(n_episodes=0, seed=0)
    assert len(left) == 0 and len(max_q) == 0

