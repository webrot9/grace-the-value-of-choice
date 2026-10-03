r"""P07 — n-step returns, the lambda-return, and the two views of TD(lambda).

Reference implementation. `traces.py` is generated from this file by
tools/strip_solutions.py. Never edit the generated file.

Two algorithms that look nothing alike:

  forward view    for every t, form the lambda-return G_t^lambda by averaging
                  n-step returns, and move V(s_t) towards it. Needs the future.

  backward view   keep one eligibility number per state, and at every step push
                  the single TD error delta_t into every state in proportion to
                  its eligibility. Needs nothing but the present.

Offline — accumulating the updates over an episode and applying them at the end
— they are **the same algorithm**. Not similar: equal, to the last bit of the
mantissa. That identity is the whole session, and `experiments.py` measures it
instead of asserting it.
"""
from __future__ import annotations

import numpy as np

# Given code, shared with the other labs that need it. Both modules are written
# to be read rather than only imported: the gridworld docstring draws the map and
# says what the reward and the slip actually mean.
from rl_lab.exact import policy_evaluation_stochastic
from rl_lab.gridworld import gridworld, uniform_policy, visitable_states


def sample_episode(P: np.ndarray, R: np.ndarray, pi: np.ndarray,
                   rng: np.random.Generator, terminal: int,
                   start: int = 0, max_steps: int = 400
                   ) -> tuple[np.ndarray, np.ndarray]:
    """One episode under `pi`, stopped on entering `terminal`.

    Returns `(states, rewards)` of equal length: this lab is prediction only, so
    the actions are sampled and thrown away.
    """
    cum_pi = np.cumsum(pi, axis=1)
    cum_P = np.cumsum(P, axis=2)
    u = rng.random((max_steps, 2))
    s = start
    states, rewards = [], []
    for t in range(max_steps):
        a = int(np.searchsorted(cum_pi[s], u[t, 0]))
        s_next = int(np.searchsorted(cum_P[s, a], u[t, 1]))
        states.append(s)
        rewards.append(float(R[s, a]))
        s = s_next
        if s == terminal:
            break
    return np.array(states), np.array(rewards)


# --------------------------------------------------------------------------- #
# Q1: the n-step return
# --------------------------------------------------------------------------- #
def n_step_return(rewards: np.ndarray, V: np.ndarray, states: np.ndarray,
                  t: int, n: int, gamma: float) -> float:
    r"""The $n$-step return from time $t$:

    $$G_t^{(n)} = r_t + \gamma r_{t+1} + \dots + \gamma^{n-1} r_{t+n-1}
                  + \gamma^{n} V(s_{t+n})$$

    The episode has length `T = len(rewards)` and ends in a terminal state whose
    value is $0$. When $t + n \ge T$ the bootstrap term disappears and the sum
    simply stops at the end of the episode — which makes $G_t^{(n)}$ the plain
    Monte Carlo return $G_t$.

    That last sentence is the whole reason $n$ interpolates between TD(0) and
    MC, and both ends are checked by a test.
    """
    # TODO: ...
    raise NotImplementedError


def lambda_return(rewards: np.ndarray, V: np.ndarray, states: np.ndarray,
                  t: int, gamma: float, lam: float) -> float:
    r"""The $\lambda$-return: a geometrically weighted average of **all** the
    $n$-step returns,

    $$G_t^\lambda = (1-\lambda)\sum_{n=1}^{T-t-1}\lambda^{n-1} G_t^{(n)}
                    + \lambda^{T-t-1} G_t ,$$

    where the last term collects all the weight that the truncated tail would
    have given to returns that run past the end of the episode. Check that the
    weights sum to one before you believe your implementation: they do, and that
    is the only reason $G^\lambda_t$ is an average rather than an arbitrary
    combination.
    """
    # TODO: ...
    raise NotImplementedError


# --------------------------------------------------------------------------- #
# Q2: the two views
# --------------------------------------------------------------------------- #
def forward_update(states: np.ndarray, rewards: np.ndarray, V: np.ndarray,
                   gamma: float, lam: float, alpha: float) -> np.ndarray:
    r"""The offline forward view over one episode: for every $t$, compute
    $G_t^\lambda$ against the value function `V` **as it was at the start of the
    episode**, and accumulate

    $$\Delta V(s_t) \mathrel{+}= \alpha\big(G_t^\lambda - V(s_t)\big).$$

    Return the accumulated $\Delta V$, shape `V.shape`. Do not modify `V`: the
    whole equivalence rests on both views seeing the same frozen `V`, and an
    in-place update here is the bug that makes the theorem "almost" hold.
    """
    # TODO: ...
    raise NotImplementedError


def backward_update(states: np.ndarray, rewards: np.ndarray, V: np.ndarray,
                    gamma: float, lam: float, alpha: float) -> np.ndarray:
    r"""The offline backward view over the same episode, with **accumulating**
    eligibility traces:

    $$e \leftarrow \gamma\lambda e, \qquad e(s_t) \mathrel{+}= 1,$$
    $$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t), \qquad
      \Delta V \mathrel{+}= \alpha\,\delta_t\, e .$$

    with $V(s_T) = 0$ at the terminal state. Return the accumulated $\Delta V$.
    Again: `V` is frozen for the whole episode.

    One TD error per step, spread over every state the agent has visited, with
    the credit decaying by $\gamma\lambda$ per step of distance. Nothing here
    looks at the future, and yet the answer is identical to the forward view.
    """
    # TODO: ...
    raise NotImplementedError


# --------------------------------------------------------------------------- #
# Q3: n-step TD, and the error it leaves behind
# --------------------------------------------------------------------------- #
def n_step_td(episodes: list, n_states: int, n: int, gamma: float,
              alpha: float) -> np.ndarray:
    r"""Offline $n$-step TD prediction over a list of `(states, rewards)`
    episodes.

    Within an episode, `V` is frozen and the updates accumulate; between
    episodes, they are applied. That makes the result a deterministic function
    of the episodes, which is what lets the tests compare it against hand
    arithmetic instead of against a tolerance chosen by trial and error.
    """
    # TODO: ...
    raise NotImplementedError


def rms_error(V: np.ndarray, truth: np.ndarray,
              mask: np.ndarray | None = None) -> float:
    r"""$\sqrt{\frac{1}{|\mathcal{S}'|}\sum_{s\in\mathcal{S}'}(V(s)-V^\pi(s))^2}$
    over the states selected by `mask` (all of them if `mask` is `None`).

    The mask is not decoration: the gridworld has obstacle cells the agent can
    never occupy, and averaging the error over states no algorithm has ever
    visited measures the layout instead of the algorithm.
    """
    # TODO: ...
    raise NotImplementedError
