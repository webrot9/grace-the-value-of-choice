"""P07 public tests — the two views, and the returns they are built from.

Every test is exact arithmetic on episodes written out by hand, or an identity
that must hold to machine precision. Nothing here samples, so nothing here is
flaky: a red test means the implementation is wrong.
"""
from __future__ import annotations

import numpy as np
import pytest

from rl_lab import traces as tr


def toy():
    """One episode of three steps over states 0, 1, 2, and a frozen V that is
    not the true value function — so that the bootstrap terms actually matter.
    """
    states = np.array([0, 1, 2])
    rewards = np.array([1.0, 2.0, 3.0])
    V = np.array([0.5, -1.0, 2.0, 0.0])
    return states, rewards, V


# --------------------------------------------------------------------------- #
# Q1 — n-step and lambda returns
# --------------------------------------------------------------------------- #
def test_q1_one_step_return_is_the_td_target():
    states, rewards, V = toy()
    g = 0.9
    got = tr.n_step_return(rewards, V, states, t=0, n=1, gamma=g)
    assert got == pytest.approx(rewards[0] + g * V[states[1]], abs=1e-12)


def test_q1_a_long_enough_n_is_the_monte_carlo_return():
    """Once n reaches the end of the episode the bootstrap disappears, and every
    larger n gives the same number: the plain discounted return."""
    states, rewards, V = toy()
    g = 0.9
    mc = rewards[0] + g * rewards[1] + g ** 2 * rewards[2]
    for n in (3, 4, 10, 100):
        assert tr.n_step_return(rewards, V, states, 0, n, g) == pytest.approx(
            mc, abs=1e-12)


def test_q1_the_bootstrap_uses_the_state_n_steps_ahead():
    """Off by one here is the classic bug and it is invisible in a learning
    curve: the target is built from s_{t+n}, never s_{t+n-1}."""
    states, rewards, V = toy()
    g = 0.9
    got = tr.n_step_return(rewards, V, states, t=0, n=2, gamma=g)
    assert got == pytest.approx(rewards[0] + g * rewards[1] + g ** 2 * V[states[2]],
                                abs=1e-12)


def test_q1_gamma_zero_is_the_immediate_reward():
    states, rewards, V = toy()
    for n in (1, 2, 3):
        assert tr.n_step_return(rewards, V, states, 0, n, 0.0) == pytest.approx(
            rewards[0], abs=1e-12)


def test_q1_lambda_zero_is_the_one_step_return():
    states, rewards, V = toy()
    g = 0.9
    for t in range(3):
        assert tr.lambda_return(rewards, V, states, t, g, 0.0) == pytest.approx(
            tr.n_step_return(rewards, V, states, t, 1, g), abs=1e-12)


def test_q1_lambda_one_is_the_monte_carlo_return():
    states, rewards, V = toy()
    g = 0.9
    for t in range(3):
        assert tr.lambda_return(rewards, V, states, t, g, 1.0) == pytest.approx(
            tr.n_step_return(rewards, V, states, t, 99, g), abs=1e-12)


def test_q1_lambda_return_matches_the_hand_algebra_on_two_steps():
    """T = 2, so there is exactly one truncated term and one tail term:
    G^lambda_0 = (1 - lam)(r_0 + gamma V(s_1)) + lam (r_0 + gamma r_1)."""
    states = np.array([0, 1])
    rewards = np.array([1.0, 2.0])
    V = np.array([0.5, -1.0, 0.0])
    g, lam = 0.9, 0.4
    expected = (1 - lam) * (rewards[0] + g * V[1]) + lam * (rewards[0] + g * rewards[1])
    assert tr.lambda_return(rewards, V, states, 0, g, lam) == pytest.approx(
        expected, abs=1e-12)


def test_q1_lambda_return_is_an_average_of_the_n_step_returns():
    """The weights sum to one, so G^lambda can never leave the range spanned by
    the returns it averages. If yours does, the tail term is wrong."""
    states, rewards, V = toy()
    g, lam = 0.9, 0.6
    ns = [tr.n_step_return(rewards, V, states, 0, n, g) for n in (1, 2, 3)]
    got = tr.lambda_return(rewards, V, states, 0, g, lam)
    assert min(ns) - 1e-12 <= got <= max(ns) + 1e-12


# --------------------------------------------------------------------------- #
# Q2 — the two views are one algorithm
# --------------------------------------------------------------------------- #
def test_q2_forward_and_backward_agree_to_machine_precision():
    """The session exists for this test. Offline, with accumulating traces, the
    forward and backward views produce the *same* update vector — not a similar
    one. The tolerance is 1e-13, and the measured gap is around 1e-16."""
    P, R = tr.gridworld(4, 4, slip=0.1, seed=3)
    S = P.shape[0]
    pi = tr.uniform_policy(S)
    rng = np.random.default_rng(0)
    V = np.random.default_rng(1).normal(size=S) * 0.1
    for lam in (0.0, 0.3, 0.7, 0.95, 1.0):
        states, rewards = tr.sample_episode(P, R, pi, rng, terminal=S - 1)
        fwd = tr.forward_update(states, rewards, V, 0.9, lam, 0.1)
        bwd = tr.backward_update(states, rewards, V, 0.9, lam, 0.1)
        assert np.max(np.abs(fwd - bwd)) < 1e-13


def test_q2_backward_with_lambda_zero_is_plain_td():
    """With lam = 0 the trace dies immediately, so each TD error lands on the
    single state that produced it."""
    states, rewards, V = toy()
    g, alpha = 0.9, 0.25
    got = tr.backward_update(states, rewards, V, g, 0.0, alpha)
    expected = np.zeros_like(V)
    for t in range(3):
        v_next = 0.0 if t == 2 else V[states[t + 1]]
        expected[states[t]] += alpha * (rewards[t] + g * v_next - V[states[t]])
    assert got == pytest.approx(expected, abs=1e-12)


def test_q2_forward_with_lambda_one_is_monte_carlo():
    states, rewards, V = toy()
    g, alpha = 0.9, 0.25
    got = tr.forward_update(states, rewards, V, g, 1.0, alpha)
    expected = np.zeros_like(V)
    for t in range(3):
        G = sum(g ** (k - t) * rewards[k] for k in range(t, 3))
        expected[states[t]] += alpha * (G - V[states[t]])
    assert got == pytest.approx(expected, abs=1e-12)


def test_q2_neither_view_touches_the_value_function():
    """Both views take V frozen. An in-place update here makes the equivalence
    hold only approximately, and the error looks like a rounding problem rather
    than like the algorithmic change it is."""
    states, rewards, V = toy()
    before = V.copy()
    tr.forward_update(states, rewards, V, 0.9, 0.5, 0.1)
    tr.backward_update(states, rewards, V, 0.9, 0.5, 0.1)
    assert np.array_equal(V, before)


def test_q2_a_repeated_state_accumulates_its_trace():
    """A state visited twice has eligibility above one when the second TD error
    arrives. With replacing traces it would be capped at one, and the two views
    would stop agreeing — that is Part C."""
    states = np.array([0, 0, 1])
    rewards = np.array([1.0, 1.0, 1.0])
    V = np.array([0.3, 0.2, 0.0])
    g, lam, alpha = 0.9, 0.8, 0.1
    fwd = tr.forward_update(states, rewards, V, g, lam, alpha)
    bwd = tr.backward_update(states, rewards, V, g, lam, alpha)
    assert np.max(np.abs(fwd - bwd)) < 1e-13


def test_q2_zero_alpha_changes_nothing():
    states, rewards, V = toy()
    for fn in (tr.forward_update, tr.backward_update):
        assert fn(states, rewards, V, 0.9, 0.5, 0.0) == pytest.approx(
            np.zeros_like(V), abs=1e-15)


# --------------------------------------------------------------------------- #
# Q3 — n-step TD and the error it leaves
# --------------------------------------------------------------------------- #
def test_q3_n_step_td_converges_exactly_on_a_deterministic_chain():
    """s0 -> s1 -> terminal with reward 1 then 0: V(s0) = 1, V(s1) = 0. One
    trajectory, no noise, so the fixed point must be reached exactly."""
    states = np.array([0, 1])
    rewards = np.array([1.0, 0.0])
    episodes = [(states, rewards)] * 300
    V = tr.n_step_td(episodes, 3, n=1, gamma=0.9, alpha=0.3)
    assert V[0] == pytest.approx(1.0, abs=1e-9)
    assert V[1] == pytest.approx(0.0, abs=1e-9)


def test_q3_n_step_td_is_offline_within_an_episode():
    """V is frozen while an episode is processed, and the updates are applied at
    its end. With a state visited twice, updating V on the spot gives a
    different answer: from [0, 0] with rewards 1, 1, gamma = 1, alpha = 0.5 and
    n = 1, both updates see V = 0 and V[0] ends at 1.0 (on the spot: 0.5, then
    0.75). Part B's numbers depend on this; an online version changes them."""
    V = tr.n_step_td([(np.array([0, 0]), np.array([1.0, 1.0]))], 1, n=1,
                     gamma=1.0, alpha=0.5)
    assert V[0] == pytest.approx(1.0, abs=1e-12)


def test_q3_large_n_stops_depending_on_n():
    """Beyond the longest episode every n gives the Monte Carlo update, so the
    result stops moving."""
    rng = np.random.default_rng(5)
    episodes = [(np.array([0, 1, 2]), rng.normal(size=3)) for _ in range(20)]
    a = tr.n_step_td(episodes, 4, n=3, gamma=0.9, alpha=0.05)
    b = tr.n_step_td(episodes, 4, n=50, gamma=0.9, alpha=0.05)
    assert a == pytest.approx(b, abs=1e-12)


def test_q3_rms_error_is_zero_on_the_exact_answer():
    truth = np.array([1.0, 2.0, 3.0])
    assert tr.rms_error(truth, truth) == pytest.approx(0.0, abs=1e-15)


def test_q3_rms_error_matches_the_definition():
    V = np.array([1.0, 2.0, 3.0])
    truth = np.array([0.0, 0.0, 0.0])
    assert tr.rms_error(V, truth) == pytest.approx(np.sqrt(14 / 3), abs=1e-12)


def test_q3_the_mask_excludes_states_from_the_average():
    """Averaging over states the agent can never visit measures the map, not the
    algorithm."""
    V = np.array([1.0, 100.0, 1.0])
    truth = np.zeros(3)
    mask = np.array([True, False, True])
    assert tr.rms_error(V, truth, mask) == pytest.approx(1.0, abs=1e-12)


def test_q3_visitable_mask_excludes_the_obstacles():
    """On this layout four cells are walled off and the goal absorbs; the mask
    must find exactly the states an episode can pass through."""
    P, R = tr.gridworld(4, 4, slip=0.1, seed=3)
    mask = tr.visitable_states(P)
    pi = tr.uniform_policy(P.shape[0])
    rng = np.random.default_rng(0)
    for _ in range(20):
        states, _ = tr.sample_episode(P, R, pi, rng, terminal=P.shape[0] - 1)
        assert mask[states].all()
