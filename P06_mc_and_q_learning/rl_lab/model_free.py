# GENERATED FILE - do not edit the solution markers by hand.
# Fill in every `# TODO` below, then run: python autograder.py
"""P06 — Monte Carlo and temporal difference: the bias-variance trade-off, measured.

Reference implementation. `model_free.py` is generated from this file by
tools/strip_solutions.py. Never edit the generated file.

This is the first lab in which the agent does not get to see `P` and `R`. It
gets episodes. The model is still there — we build it, and we use it to compute
the exact $V^\\pi$ that everything is measured against — but no learner ever
reads it. That is the whole point: with ground truth in hand, "unbiased" and
"lower variance" become numbers instead of adjectives.

Two estimators of the same $V^\\pi$, fed **the same episodes**:

    MC:     V(s) <- average of the returns observed from s
    TD(0):  V(s) <- V(s) + alpha (r + gamma V(s') - V(s))

MC waits for the end of the episode and averages what actually happened. TD(0)
does not wait: it uses its own current estimate of the next state, which is
wrong, and that is exactly where its bias comes from.
"""
from __future__ import annotations

import numpy as np

# Given code, shared with the other labs that need it. Both modules are written
# to be read rather than only imported: the gridworld docstring draws the map and
# says what the reward and the slip actually mean.
from rl_lab.exact import policy_evaluation_exact, policy_evaluation_stochastic, q_from_v, value_iteration
from rl_lab.gridworld import gridworld, uniform_policy


# --------------------------------------------------------------------------- #
# Given: the generative model. The learners see this and nothing else.
# --------------------------------------------------------------------------- #
def sample_episode(P: np.ndarray, R: np.ndarray, pi: np.ndarray,
                   rng: np.random.Generator, terminal: int,
                   start: int = 0, max_steps: int = 400
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One episode under the stochastic policy `pi[s, a]`, stopped when the
    absorbing state `terminal` is entered.

    Returns `(states, actions, rewards)` with `len(states) == len(rewards)`;
    `states[t]` is the state the agent was in when it took `actions[t]` and
    collected `rewards[t]`. The absorbing state itself is not appended, because
    its value is zero by construction and no estimator should be updating it.

    `max_steps` truncates: with `gamma = 0.9` the tail it discards is worth at
    most `0.9 ** 400`, which is `1e-19`, so the truncation is not a source of
    bias you could measure.
    """
    cum_pi = np.cumsum(pi, axis=1)                  # sampling by inverse CDF:
    cum_P = np.cumsum(P, axis=2)                    # ~40x faster than rng.choice
    u = rng.random((max_steps, 2))
    s = start
    states, actions, rewards = [], [], []
    for t in range(max_steps):
        a = int(np.searchsorted(cum_pi[s], u[t, 0]))
        s_next = int(np.searchsorted(cum_P[s, a], u[t, 1]))
        states.append(s)
        actions.append(a)
        rewards.append(float(R[s, a]))
        s = s_next
        if s == terminal:
            break
    return np.array(states), np.array(actions), np.array(rewards)


# --------------------------------------------------------------------------- #
# Q1: what an episode is worth
# --------------------------------------------------------------------------- #
def discounted_returns(rewards: np.ndarray, gamma: float) -> np.ndarray:
    r"""The return from every time step of one episode:

    $$G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \dots + \gamma^{T-1-t} r_{T-1}$$

    Return an array with the same length as `rewards`. Write it as one backward
    pass, $G_t = r_t + \gamma G_{t+1}$, not as a double loop: the double loop is
    $O(T^2)$ and, more importantly, it hides the recursion that the whole
    lecture is about.
    """
    # TODO: ...
    raise NotImplementedError


def first_visit_mask(states: np.ndarray) -> np.ndarray:
    """`True` at the time steps where the state is being seen for the first
    time in this episode, `False` afterwards.

    First-visit MC averages only over the `True` entries. Every-visit MC uses
    all of them. Both converge to $V^\\pi$; they are not the same estimator, and
    the difference is that the returns from repeated visits inside one episode
    are correlated with each other.
    """
    # TODO: ...
    raise NotImplementedError


def mc_evaluate(episodes: list, n_states: int, gamma: float,
                first_visit: bool = True) -> np.ndarray:
    r"""First-visit (or every-visit) Monte Carlo evaluation.

    For each visited state, average the returns observed from it, over all the
    episodes. States never visited keep the value $0$.

    Each element of `episodes` is a `(states, actions, rewards)` triple as
    produced by `sample_episode`. This is a **sample average**, not a
    step-size update: no `alpha` appears anywhere, which is why the estimator is
    unbiased and why its variance falls exactly as $1/n$.
    """
    # TODO: ...
    raise NotImplementedError


# --------------------------------------------------------------------------- #
# Q2: not waiting for the end
# --------------------------------------------------------------------------- #
def td0_evaluate(episodes: list, n_states: int, gamma: float,
                 alpha: float = 0.05, n_passes: int = 1,
                 terminal: int | None = None) -> np.ndarray:
    r"""TD(0) evaluation over the same episodes.

    $$V(s_t) \leftarrow V(s_t) + \alpha\big(r_t + \gamma V(s_{t+1}) - V(s_t)\big)$$

    `s_{t+1}` is `states[t+1]` inside the episode, and the terminal state after
    the last transition, whose value is $0$ and stays $0$.

    `n_passes` sweeps the same episodes repeatedly. With one pass this is online
    TD; with many, it is **batch** TD, which, with enough passes and a small
    enough step, converges to the value function of the maximum-likelihood MDP
    built from the data. That limit is the honest statement of TD's bias. The
    fifteen passes of Part B do not reach it on very few episodes, where most of
    the bias still comes from the zero start.
    """
    # TODO: ...
    raise NotImplementedError


def bias_variance(estimates: np.ndarray, truth: float
                  ) -> tuple[float, float, float]:
    r"""Decompose the error of a set of independent estimates of one number:

    $$\underbrace{\mathbb{E}\big[(\hat{v} - v)^2\big]}_{\text{MSE}}
      = \underbrace{\big(\mathbb{E}[\hat{v}] - v\big)^2}_{\text{bias}^2}
      + \underbrace{\mathrm{Var}[\hat{v}]}_{\text{variance}}$$

    Return `(bias, variance, mse)` — signed bias, not squared, because the sign
    tells you which way the estimator leans and squaring throws that away.

    Use the population variance (`ddof=0`). With `ddof=1` the identity above
    stops holding exactly, and a test checks it to $10^{-12}$.
    """
    # TODO: ...
    raise NotImplementedError


# --------------------------------------------------------------------------- #
# Q3: control — acting on what you learned
# --------------------------------------------------------------------------- #
def epsilon_greedy(q_values: np.ndarray, eps: float,
                   rng: np.random.Generator) -> int:
    r"""Pick an action: uniformly at random with probability $\epsilon$,
    otherwise $\arg\max_a Q(s,a)$.

    Break ties **uniformly at random** among the maximisers. A plain `argmax`
    always returns the lowest index, which on a freshly initialised $Q$ table of
    zeros means the agent walks up and to the left for its entire first episode.
    That is not exploration, it is a bug with a plausible-looking learning
    curve.
    """
    # TODO: ...
    raise NotImplementedError


def mc_control(P: np.ndarray, R: np.ndarray, gamma: float, terminal: int,
               n_episodes: int = 500, alpha: float = 0.1, eps: float = 0.2,
               seed: int = 0, max_steps: int = 400) -> np.ndarray:
    r"""On-policy first-visit MC control with a constant step size.

    Each episode: act $\epsilon$-greedily on the current $Q$, then update every
    first-visited $(s,a)$ towards the return that followed it,

    $$Q(s,a) \leftarrow Q(s,a) + \alpha\big(G_t - Q(s,a)\big).$$

    The environment is `P` and `R`, but you may only use them through
    `rng.choice(..., p=P[s, a])` to sample the next state and `R[s, a]` to
    collect the reward — no `max` over `P`, no sums over $s'$. Return the $Q$
    table, shape `(S, A)`.
    """
    # TODO: ...
    raise NotImplementedError


def q_learning(P: np.ndarray, R: np.ndarray, gamma: float, terminal: int,
               n_episodes: int = 500, alpha: float = 0.1, eps: float = 0.2,
               seed: int = 0, max_steps: int = 400) -> np.ndarray:
    r"""Off-policy TD control, updating at every step instead of at the end:

    $$Q(s,a) \leftarrow Q(s,a) + \alpha\big(r + \gamma \max_{a'} Q(s',a') - Q(s,a)\big)$$

    with $\max_{a'} Q(s',a') = 0$ when $s'$ is terminal. Same interface, same
    budget, same restriction on how you may touch `P` and `R`. Return `Q`.

    The behaviour policy is $\epsilon$-greedy and the target policy is greedy:
    that mismatch is what "off-policy" means, and it is why the $\max$ is there
    instead of the action you actually took next.
    """
    # TODO: ...
    raise NotImplementedError
