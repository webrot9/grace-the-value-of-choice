r"""P09, second half — DQN and Double DQN, on a problem whose answer we know.

Reference implementation. `deep.py` is generated from this file by
tools/strip_solutions.py. Never edit the generated file.

The environment is a stochastic corridor with a continuous position
$x \in [0, 1]$ and three actions. It is small enough that $q^{*}$ can be
computed exactly, by value iteration on a fine discretisation, and that is the
whole reason it is here: van Hasselt's overestimation plots compare
$\max_a Q_w$ against the *realised return*, because on Atari there is no
ground truth. Here there is, so the overestimation is measured against the real
thing rather than against a proxy.

Everything a DQN has is here and nothing else: a replay buffer, a target
network, an epsilon schedule, and one line of difference between the two
algorithms.
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np
import torch
import torch.nn as nn

N_ACTIONS = 3
MOVES = (-0.08, 0.0, 0.08)
NOISE = 0.03
GOAL = 0.9
STEP_COST = -0.02
REWARD_NOISE = 1.0     # zero mean, so q* below is unchanged by it


# --------------------------------------------------------------------------- #
# Given: the environment, and its exact answer
# --------------------------------------------------------------------------- #
class Corridor:
    """Position in [0, 1]. Three actions move left, stay, or right by 0.08,
    plus Gaussian noise. Reaching x >= 0.9 ends the episode with reward +1;
    every step costs 0.02. Episodes are truncated at 60 steps.

    The reward is noisy in the sense that matters here — where you land is
    noisy, so the value of an action is an average over outcomes, and averages
    estimated from few samples are what a maximum bites.
    """

    def __init__(self, seed: int = 0, max_steps: int = 60,
                 reward_noise: float = REWARD_NOISE):
        self.rng = np.random.default_rng(seed)
        self.max_steps = max_steps
        self.reward_noise = reward_noise
        self.x = 0.0
        self.t = 0

    def reset(self) -> np.ndarray:
        self.x = float(self.rng.uniform(0.0, 0.3))
        self.t = 0
        return np.array([self.x], dtype=np.float32)

    def step(self, a: int) -> tuple[np.ndarray, float, bool, bool]:
        """The reward noise has mean zero, so it leaves q* exactly where it is
        and only makes the *estimates* of q* noisy. That distinction is the
        whole lab: the bias below is not a bias of the environment."""
        self.x = float(np.clip(self.x + MOVES[a] + self.rng.normal(0, NOISE),
                               0.0, 1.0))
        self.t += 1
        done = self.x >= GOAL
        truncated = self.t >= self.max_steps and not done
        reward = 1.0 + STEP_COST if done else STEP_COST
        if self.reward_noise > 0:
            reward += float(self.rng.normal(0.0, self.reward_noise))
        return np.array([self.x], dtype=np.float32), reward, done, truncated


def exact_q(gamma: float = 0.95, n_bins: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """$q^{*}$ by value iteration on a fine discretisation of the corridor.

    Returns `(centres, Q)` with `Q` of shape `(n_bins, 3)`. This is the ground
    truth the networks are scored against; nothing in the training loop is
    allowed to look at it.

    The transition kernel is the Gaussian displacement integrated over each bin,
    with the mass beyond the two ends folded onto the end bins, which is what
    the `clip` in `Corridor.step` does.
    """
    centres = (np.arange(n_bins) + 0.5) / n_bins
    edges = np.arange(n_bins + 1) / n_bins
    goal = centres >= GOAL
    erf = np.vectorize(math.erf)

    P = np.zeros((n_bins, N_ACTIONS, n_bins))
    for a, move in enumerate(MOVES):
        mu = np.clip(centres + move, 0.0, 1.0)
        cdf = 0.5 * (1 + erf((edges[None, :] - mu[:, None]) / (NOISE * np.sqrt(2))))
        mass = np.diff(cdf, axis=1)
        mass[:, 0] += cdf[:, 0]                    # everything below 0 clips to 0
        mass[:, -1] += 1 - cdf[:, -1]              # everything above 1 clips to 1
        P[:, a, :] = mass / mass.sum(axis=1, keepdims=True)

    R = np.full((n_bins, N_ACTIONS), STEP_COST) + P[:, :, goal].sum(axis=2)
    V = np.zeros(n_bins)
    for _ in range(20_000):
        V_new = (R + gamma * (P @ V)).max(axis=1)
        V_new[goal] = 0.0
        if np.max(np.abs(V_new - V)) < 1e-12:
            V = V_new
            break
        V = V_new
    Q = R + gamma * (P @ V)
    Q[goal] = 0.0
    return centres, Q


class ReplayBuffer:
    """A deque of transitions with uniform sampling. Given: there is nothing to
    learn from re-implementing a ring buffer, and a subtle bug in one would
    look exactly like a subtle bug in the target."""

    def __init__(self, capacity: int = 20_000):
        self.data = deque(maxlen=capacity)

    def add(self, s, a, r, s2, done):
        self.data.append((s, a, r, s2, float(done)))

    def sample(self, n: int, rng: np.random.Generator):
        idx = rng.integers(len(self.data), size=n)
        batch = [self.data[int(i)] for i in idx]
        s, a, r, s2, d = zip(*batch)
        return (torch.tensor(np.array(s), dtype=torch.float32),
                torch.tensor(a, dtype=torch.int64),
                torch.tensor(r, dtype=torch.float32),
                torch.tensor(np.array(s2), dtype=torch.float32),
                torch.tensor(d, dtype=torch.float32))

    def __len__(self):
        return len(self.data)


def mlp(n_in: int = 1, n_hidden: int = 64, n_out: int = N_ACTIONS) -> nn.Module:
    """Two hidden layers, tanh. Given, and deliberately small: this lab has to
    run on a free Colab CPU in under a minute."""
    return nn.Sequential(nn.Linear(n_in, n_hidden), nn.Tanh(),
                         nn.Linear(n_hidden, n_hidden), nn.Tanh(),
                         nn.Linear(n_hidden, n_out))


# --------------------------------------------------------------------------- #
# Q4: the two targets, on a batch
# --------------------------------------------------------------------------- #
def dqn_targets(rewards: torch.Tensor, next_q_target: torch.Tensor,
                dones: torch.Tensor, gamma: float) -> torch.Tensor:
    r"""$y_i = r_i + \gamma (1 - d_i) \max_{a'} Q_{\bar w}(s'_i, a')$.

    `next_q_target` has shape `(batch, n_actions)`; the result has shape
    `(batch,)`. `dones` is 1.0 for terminal transitions and 0.0 otherwise.

    Return a tensor **detached** from the graph: the target is data, not
    something to backpropagate through. Forgetting that does not crash and does
    not obviously diverge — it just quietly trains a different algorithm.
    """
    # TODO: ...
    raise NotImplementedError


def ddqn_targets(rewards: torch.Tensor, next_q_online: torch.Tensor,
                 next_q_target: torch.Tensor, dones: torch.Tensor,
                 gamma: float) -> torch.Tensor:
    r"""$y_i = r_i + \gamma (1 - d_i)\,
    Q_{\bar w}\!\big(s'_i, \arg\max_{a'} Q_{w}(s'_i,a')\big)$.

    The online network picks the action, the target network prices it. That is
    the entire difference between DQN and Double DQN — one `argmax` moved from
    one network to the other — and a test checks that passing the same tensor
    twice gives `dqn_targets` back exactly.
    """
    # TODO: ...
    raise NotImplementedError


def soft_update(target: nn.Module, online: nn.Module, tau: float) -> None:
    r"""$\bar w \leftarrow (1-\tau)\bar w + \tau w$, in place.

    `tau = 1` is the hard copy that the original DQN paper does every $C$ steps;
    anything smaller is the Polyak average that most implementations use now. A
    test checks the two ends.
    """
    # TODO: ...
    raise NotImplementedError


# --------------------------------------------------------------------------- #
# Q5: fitting a network, twice, on identical data
# --------------------------------------------------------------------------- #
def collect(n_transitions: int = 5000, seed: int = 0,
            reward_noise: float = REWARD_NOISE) -> ReplayBuffer:
    """A fixed dataset, gathered by acting uniformly at random. Given.

    Both algorithms are fitted on **this same buffer**: not on the same number
    of transitions, on the same transitions. Any difference in the result is a
    difference between the two targets and cannot be a difference in what they
    happened to explore — which is exactly the confound that makes the tabular
    experiment in the first half harder to read.
    """
    env = Corridor(seed=seed, reward_noise=reward_noise)
    rng = np.random.default_rng(seed)
    buffer = ReplayBuffer(capacity=n_transitions + 10)
    s = env.reset()
    for _ in range(n_transitions):
        a = int(rng.integers(N_ACTIONS))
        s2, r, done, truncated = env.step(a)
        buffer.add(s, a, r, s2, done)
        s = env.reset() if (done or truncated) else s2
    return buffer


def fit_offline(buffer: ReplayBuffer, double: bool, seed: int = 0,
                n_steps: int = 3000, gamma: float = 0.95, batch: int = 128,
                lr: float = 2e-3, tau: float = 0.02,
                probe: np.ndarray | None = None, probe_every: int = 100
                ) -> tuple[list[float], nn.Module]:
    r"""Fit a Q network on a fixed buffer, with a target network, and record
    $\frac{1}{\lvert S\rvert}\sum_{s \in S} \max_a Q_w(s,a)$ over the probe states
    every `probe_every` steps.

    One implementation for both algorithms: `double` chooses which of the two
    target functions is called and nothing else changes — same net, same batch
    sequence, same seed, same data. Return `(probe_values, online_net)`.

    Offline and not online, because the question is about the *estimator*. An
    online run mixes the bias with how much each algorithm happened to explore;
    here the data is fixed and both fits take the same steps on it, so what
    differs is the target. The fit does not run to convergence, and at the
    default 3000 steps the curves are still climbing: compare the two algorithms
    at the same step, and do not read the numbers as where either would end up.
    """
    # TODO: ...
    raise NotImplementedError
