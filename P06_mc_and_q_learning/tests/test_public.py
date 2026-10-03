"""P06 public tests — invariants, not learning curves.

Every test here is exact arithmetic on hand-made data, or a frequency check with
a fixed seed and an explicit tolerance. Nothing depends on how well an agent
learned: a test that did would fail at random and teach you to re-run instead of
to think.

Read the failures literally. `discounted_returns` failing on the geometric-series
case means your recursion is wrong, not that your learning rate is bad.
"""
from __future__ import annotations

import numpy as np
import pytest

from rl_lab import model_free as mf


def two_state_chain():
    """s0 --a--> s1 --a--> s1 (absorbing), reward 1 on entering s1.

    Small enough that every value can be written down by hand:
    V(s1) = 0, V(s0) = 1.
    """
    P = np.zeros((2, 1, 2))
    R = np.zeros((2, 1))
    P[0, 0, 1] = 1.0
    P[1, 0, 1] = 1.0
    R[0, 0] = 1.0
    return P, R


# --------------------------------------------------------------------------- #
# Q1 — returns, first visits, Monte Carlo
# --------------------------------------------------------------------------- #
def test_q1_returns_match_the_geometric_series():
    """With every reward equal to 1, G_t is a truncated geometric series and
    the closed form is (1 - gamma^(T-t)) / (1 - gamma)."""
    gamma, T = 0.9, 6
    G = mf.discounted_returns(np.ones(T), gamma)
    expected = np.array([(1 - gamma ** (T - t)) / (1 - gamma) for t in range(T)])
    assert G == pytest.approx(expected, abs=1e-12)


def test_q1_returns_satisfy_their_own_recursion():
    """G_t = r_t + gamma G_{t+1} for every t, with G_T = 0. This is the identity
    the function is, so it must hold on arbitrary rewards."""
    rng = np.random.default_rng(0)
    r = rng.normal(size=12)
    gamma = 0.77
    G = mf.discounted_returns(r, gamma)
    assert G[:-1] == pytest.approx(r[:-1] + gamma * G[1:], abs=1e-12)
    assert G[-1] == pytest.approx(r[-1], abs=1e-12)


def test_q1_gamma_zero_is_the_immediate_reward():
    r = np.array([3.0, -1.0, 7.0])
    assert mf.discounted_returns(r, 0.0) == pytest.approx(r, abs=1e-12)


def test_q1_first_visit_mask_marks_only_the_first_occurrence():
    states = np.array([0, 1, 0, 2, 1, 0])
    assert list(mf.first_visit_mask(states)) == [True, True, False, True,
                                                 False, False]


def test_q1_first_visit_mask_is_all_true_without_repeats():
    states = np.array([4, 2, 7, 1])
    assert mf.first_visit_mask(states).all()


def test_q1_mc_averages_the_returns_by_hand():
    """Two hand-written episodes, one state, no repeats: the estimate must be
    the plain arithmetic mean of the two returns."""
    gamma = 0.5
    ep1 = (np.array([0]), np.array([0]), np.array([2.0]))
    ep2 = (np.array([0, 1]), np.array([0, 0]), np.array([1.0, 4.0]))
    V = mf.mc_evaluate([ep1, ep2], 2, gamma)
    g1, g2 = 2.0, 1.0 + gamma * 4.0
    assert V[0] == pytest.approx((g1 + g2) / 2, abs=1e-12)
    assert V[1] == pytest.approx(4.0, abs=1e-12)


def test_q1_first_and_every_visit_differ_only_on_repeats():
    """One episode that visits state 0 twice. First-visit keeps the return from
    the first visit; every-visit averages both. Two different estimators, both
    consistent, and the test pins the difference to hand arithmetic."""
    gamma = 0.5
    ep = (np.array([0, 1, 0]), np.array([0, 0, 0]), np.array([1.0, 1.0, 1.0]))
    G = np.array([1 + 0.5 + 0.25, 1 + 0.5, 1.0])
    first = mf.mc_evaluate([ep], 2, gamma, first_visit=True)
    every = mf.mc_evaluate([ep], 2, gamma, first_visit=False)
    assert first[0] == pytest.approx(G[0], abs=1e-12)
    assert every[0] == pytest.approx((G[0] + G[2]) / 2, abs=1e-12)
    assert first[1] == pytest.approx(every[1], abs=1e-12)


def test_q1_unvisited_states_stay_at_zero():
    ep = (np.array([0]), np.array([0]), np.array([1.0]))
    V = mf.mc_evaluate([ep], 5, 0.9)
    assert V[1:] == pytest.approx(np.zeros(4), abs=1e-15)


# --------------------------------------------------------------------------- #
# Q2 — TD(0) and the decomposition
# --------------------------------------------------------------------------- #
def test_q2_td_moves_by_exactly_alpha_times_the_error():
    """One episode of one step, from V = 0: the update must be alpha * r, no
    more and no less. If this fails by a factor of gamma, the bootstrap term is
    using the wrong state."""
    ep = (np.array([0]), np.array([0]), np.array([1.0]))
    V = mf.td0_evaluate([ep], 2, gamma=0.9, alpha=0.3, n_passes=1)
    assert V[0] == pytest.approx(0.3, abs=1e-12)


def test_q2_terminal_bootstrap_is_zero():
    """The last transition of an episode must bootstrap off 0, not off the
    value of the state the agent happens to be in. Two episodes that differ only
    in their last reward must move V[0] by alpha times that difference."""
    a = (np.array([0]), np.array([0]), np.array([1.0]))
    b = (np.array([0]), np.array([0]), np.array([5.0]))
    va = mf.td0_evaluate([a], 1, gamma=0.99, alpha=0.5, n_passes=1)[0]
    vb = mf.td0_evaluate([b], 1, gamma=0.99, alpha=0.5, n_passes=1)[0]
    assert vb - va == pytest.approx(0.5 * 4.0, abs=1e-12)
    # With one pass V starts at 0, so bootstrapping off V(s_t) would look the
    # same. On the second pass it does not: 0.5, then 0.5 + 0.5 * (1 - 0.5).
    two = mf.td0_evaluate([a], 1, gamma=0.99, alpha=0.5, n_passes=2)[0]
    assert two == pytest.approx(0.75, abs=1e-12)


def test_q2_td_converges_to_the_exact_value_on_a_deterministic_chain():
    """On a deterministic MDP the TD fixed point is V^pi exactly, so enough
    passes over a single episode must reach it. No sampling noise is involved:
    the chain has one trajectory."""
    P, R = two_state_chain()
    gamma = 0.9
    ep = (np.array([0, 1]), np.array([0, 0]), np.array([1.0, 0.0]))
    V = mf.td0_evaluate([ep], 2, gamma, alpha=0.4, n_passes=400)
    assert V[0] == pytest.approx(1.0, abs=1e-9)
    assert V[1] == pytest.approx(0.0, abs=1e-9)


def test_q2_more_passes_never_move_a_converged_value():
    """A fixed point is a fixed point: once TD has converged, further sweeps
    over the same data leave it where it is."""
    P, R = two_state_chain()
    ep = (np.array([0, 1]), np.array([0, 0]), np.array([1.0, 0.0]))
    v400 = mf.td0_evaluate([ep], 2, 0.9, alpha=0.4, n_passes=400)
    v800 = mf.td0_evaluate([ep], 2, 0.9, alpha=0.4, n_passes=800)
    assert v800 == pytest.approx(v400, abs=1e-12)


def test_q2_bias_variance_identity_holds_exactly():
    """MSE = bias^2 + variance, to machine precision. If this fails you used
    ddof=1 somewhere: the sample variance does not satisfy the identity."""
    rng = np.random.default_rng(3)
    est = rng.normal(loc=2.0, scale=0.7, size=500)
    bias, var, mse = mf.bias_variance(est, truth=1.5)
    assert mse == pytest.approx(bias ** 2 + var, abs=1e-12)


def test_q2_bias_is_signed():
    """The sign of the bias says which way the estimator leans, and squaring it
    throws that information away. An estimator that is always 1 below the truth
    has bias -1, not +1."""
    bias, var, mse = mf.bias_variance(np.full(10, 4.0), truth=5.0)
    assert bias == pytest.approx(-1.0, abs=1e-12)
    assert var == pytest.approx(0.0, abs=1e-15)
    assert mse == pytest.approx(1.0, abs=1e-12)


def test_q2_a_constant_estimator_has_zero_variance_and_all_the_error():
    """The degenerate case Part B runs into: returning the same wrong number
    every time is a zero-variance estimator. Low variance is not a virtue on its
    own, and this test is here so the point is impossible to miss."""
    bias, var, mse = mf.bias_variance(np.zeros(100), truth=0.0343)
    assert var == 0.0
    assert mse == pytest.approx(bias ** 2, abs=1e-15)


# --------------------------------------------------------------------------- #
# Q3 — control
# --------------------------------------------------------------------------- #
def test_q3_epsilon_zero_is_greedy():
    rng = np.random.default_rng(0)
    q = np.array([0.1, 0.9, 0.4])
    assert all(mf.epsilon_greedy(q, 0.0, rng) == 1 for _ in range(50))


def test_q3_epsilon_one_is_uniform():
    """5000 draws, four actions, explicit tolerance. The only randomised test in
    the file, and it is seeded."""
    rng = np.random.default_rng(1)
    q = np.array([5.0, 0.0, 0.0, 0.0])
    counts = np.bincount([mf.epsilon_greedy(q, 1.0, rng) for _ in range(5000)],
                         minlength=4)
    assert counts / 5000 == pytest.approx(np.full(4, 0.25), abs=0.02)


def test_q3_ties_are_broken_uniformly():
    """A fresh Q table is all zeros, so every action ties. Breaking ties with a
    plain argmax sends the agent up-and-left for its whole first episode, which
    looks like exploration failing rather than like the bug it is."""
    rng = np.random.default_rng(2)
    counts = np.bincount([mf.epsilon_greedy(np.zeros(4), 0.0, rng)
                          for _ in range(5000)], minlength=4)
    assert counts / 5000 == pytest.approx(np.full(4, 0.25), abs=0.02)


def test_q3_greedy_share_matches_the_epsilon_soft_probability():
    """With four actions and eps = 0.4 the greedy action is taken with
    probability 1 - eps + eps/4 = 0.7. That number, not 1 - eps, is what an
    eps-greedy policy actually does, and it is why on-policy control converges
    to Q*_eps rather than to Q*."""
    rng = np.random.default_rng(4)
    q = np.array([0.0, 1.0, 0.0, 0.0])
    n = 5000
    greedy = sum(mf.epsilon_greedy(q, 0.4, rng) == 1 for _ in range(n))
    assert greedy / n == pytest.approx(0.7, abs=0.02)


def test_q3_q_learning_finds_the_exact_value_on_a_deterministic_chain():
    """One action, one trajectory, no randomness in the environment: Q-learning
    must converge to Q*(s0, a) = 1 exactly. This is a fixed-point check, not a
    performance threshold."""
    P, R = two_state_chain()
    Q = mf.q_learning(P, R, gamma=0.9, terminal=1, n_episodes=300,
                      alpha=0.3, eps=0.0, seed=0)
    assert Q[0, 0] == pytest.approx(1.0, abs=1e-6)


def test_q3_q_learning_bootstraps_off_the_max_not_off_the_next_action():
    """From s0, action 1 leads to s1, where action 0 pays 10 and action 1 costs
    100. Q*(s0, 1) = 0.9 * 10 = 9, whatever the behaviour policy does. With
    eps = 0.5 half the actions are random, so a target built from the action
    actually taken next (SARSA) or from its expectation (expected SARSA) lands
    near 0.9 * (0.75 * 10 - 0.25 * 100) = -15.75 instead. Rewards and
    transitions are deterministic, so Q-learning converges to the fixed point."""
    P = np.zeros((3, 2, 3))
    P[0, 0, 2] = P[0, 1, 1] = P[1, 0, 2] = P[1, 1, 2] = 1.0
    P[2, :, 2] = 1.0
    R = np.array([[1.0, 0.0], [10.0, -100.0], [0.0, 0.0]])
    Q = mf.q_learning(P, R, gamma=0.9, terminal=2, n_episodes=2000,
                      alpha=0.3, eps=0.5, seed=0)
    assert Q[1, 0] == pytest.approx(10.0, abs=1e-6)
    assert Q[0, 1] == pytest.approx(9.0, abs=1e-6)


def test_q3_mc_control_discounts_the_return():
    """Two steps, reward 1 on the second: the return from s0 is gamma = 0.9,
    not 1. With one action there is nothing to explore."""
    P = np.zeros((3, 1, 3))
    P[0, 0, 1] = P[1, 0, 2] = P[2, 0, 2] = 1.0
    R = np.array([[0.0], [1.0], [0.0]])
    Q = mf.mc_control(P, R, gamma=0.9, terminal=2, n_episodes=300,
                      alpha=0.3, eps=0.0, seed=0)
    assert Q[0, 0] == pytest.approx(0.9, abs=1e-6)
    assert Q[1, 0] == pytest.approx(1.0, abs=1e-6)


def test_q3_mc_control_finds_the_exact_value_on_a_deterministic_chain():
    """Same chain, same target: with one action there is nothing to explore, so
    the return is always 1 and the running average must land on it."""
    P, R = two_state_chain()
    Q = mf.mc_control(P, R, gamma=0.9, terminal=1, n_episodes=300,
                      alpha=0.3, eps=0.0, seed=0)
    assert Q[0, 0] == pytest.approx(1.0, abs=1e-6)


def test_q3_learners_return_the_right_shape():
    P, R = mf.gridworld(3, 3, slip=0.0, seed=0, n_obstacles=0)
    for fn in (mf.q_learning, mf.mc_control):
        Q = fn(P, R, 0.9, terminal=8, n_episodes=5, seed=0)
        assert Q.shape == (9, 4)


def test_q3_zero_episodes_learns_nothing():
    P, R = mf.gridworld(3, 3, slip=0.0, seed=0, n_obstacles=0)
    for fn in (mf.q_learning, mf.mc_control):
        assert fn(P, R, 0.9, terminal=8, n_episodes=0, seed=0).sum() == 0.0


def test_q3_the_same_seed_gives_the_same_table():
    """Reproducibility is not decoration: every number in the report is a claim
    that someone else can check."""
    P, R = mf.gridworld(4, 4, slip=0.1, seed=3)
    a = mf.q_learning(P, R, 0.9, terminal=15, n_episodes=50, seed=7)
    b = mf.q_learning(P, R, 0.9, terminal=15, n_episodes=50, seed=7)
    assert np.array_equal(a, b)
