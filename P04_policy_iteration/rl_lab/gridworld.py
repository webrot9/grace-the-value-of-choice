"""The gridworld every lab from P03 onwards is built on.

This file is **given**: you never have to write it, and it is the same file in
every lab, so the environment you debugged in week 3 is the environment you are
still using in week 11.

The world is a rectangle of cells. The agent starts in the top-left corner and
the goal is the bottom-right one. Some cells are obstacles, chosen at random
from the seed, and the agent cannot enter them.

        0   1   2   3
      +---+---+---+---+
    0 | A |   |   | X |     A = the agent starts here (state 0)
      +---+---+---+---+     X = an obstacle: you cannot enter it
    1 |   | X |   |   |     G = the goal (state 15 on a 4x4 grid)
      +---+---+---+---+
    2 |   |   |   |   |     the state number of cell (r, c) is r * width + c
      +---+---+---+---+
    3 | X |   |   | G |
      +---+---+---+---+

Four actions, in this order: **up, left, down, right**. Bumping into a wall or
into an obstacle leaves you where you are — the move is legal, it just does
nothing.

Three details that matter for every theorem you will check against this world:

1. **The goal is absorbing.** Once there, every action leads back to it with
   probability 1 and pays nothing more. So the reward is collected exactly once
   per episode and V* is finite even without a step limit.

2. **The reward is on *entering* the goal**, not on being in it. `R[s, a]` is
   the probability of landing in the goal from `s` with `a` — which is why it
   can be a fraction when `slip > 0`. A reward that is a probability looks odd
   the first time; it is just the expectation of a reward of 1.

3. **`slip` is the probability of going sideways.** With `slip = 0.1` the
   intended direction happens 90% of the time and each of the two perpendicular
   directions 5%. `slip = 0` gives a deterministic world, which is the one to
   use when you want a theorem to hold to sixteen decimal places instead of
   approximately.

The representation is two numpy arrays and nothing else:

    P[s, a, s']   probability of landing in s' from s with a   (sums to 1 over s')
    R[s, a]       expected immediate reward

No environment object, no `step()`, no simulator. Everything is a matrix, so
"the exact answer" is a linear solve away and you can score a learner against
truth rather than against another learner.
"""
from __future__ import annotations

import numpy as np

# up, left, down, right — as (row, column) offsets. Index into this list *is*
# the action number, so MOVES[2] == down == action 2.
MOVES = [(-1, 0), (0, -1), (1, 0), (0, 1)]
N_ACTIONS = 4


def gridworld(height: int = 4, width: int = 4, slip: float = 0.0,
              seed: int = 0, n_obstacles: int | None = None
              ) -> tuple[np.ndarray, np.ndarray]:
    """Build `(P, R)` for a gridworld. See the module docstring for the rules.

    `n_obstacles` defaults to `(height + width) // 2`, so a 4x4 grid gets four.
    Neither the start nor the goal is ever an obstacle.

    Careful with the seed: obstacles are placed at random, and on most seeds
    they cut the grid into pieces from which the goal cannot be reached. Those
    states have `V = 0` exactly, and a learner that has learned nothing there
    will look perfectly correct. `seed=3` on a 4x4 grid has no trapped region,
    which is why the labs use it.
    """
    rng = np.random.default_rng(seed)
    n_states = height * width
    goal = n_states - 1                      # bottom-right cell

    n_obstacles = (height + width) // 2 if n_obstacles is None else n_obstacles
    obstacles: set[tuple[int, int]] = set()
    while len(obstacles) < n_obstacles:
        cell = (int(rng.integers(height)), int(rng.integers(width)))
        if cell not in {(0, 0), (height - 1, width - 1)}:
            obstacles.add(cell)

    def state_of(row: int, col: int) -> int:
        return row * width + col

    def move(row: int, col: int, action: int) -> tuple[int, int]:
        """Where you end up going `action` from `(row, col)`. Walls and
        obstacles leave you where you were."""
        new_row, new_col = row + MOVES[action][0], col + MOVES[action][1]
        inside = 0 <= new_row < height and 0 <= new_col < width
        if inside and (new_row, new_col) not in obstacles:
            return new_row, new_col
        return row, col

    P = np.zeros((n_states, N_ACTIONS, n_states))
    R = np.zeros((n_states, N_ACTIONS))

    for row in range(height):
        for col in range(width):
            state = state_of(row, col)

            if state == goal:                # absorbing: nothing more happens
                P[state, :, state] = 1.0
                continue

            for action in range(N_ACTIONS):
                # Where the agent may actually end up, and with what
                # probability: the intended direction, plus the two
                # perpendicular ones when the world is slippery.
                if slip == 0:
                    outcomes = [(action, 1.0)]
                else:
                    outcomes = [(action, 1.0 - slip),
                                ((action - 1) % N_ACTIONS, slip / 2),
                                ((action + 1) % N_ACTIONS, slip / 2)]

                for direction, probability in outcomes:
                    new_row, new_col = move(row, col, direction)
                    landed = state_of(new_row, new_col)
                    P[state, action, landed] += probability
                    if landed == goal:
                        R[state, action] += probability   # paid on entering

    return P, R


def uniform_policy(n_states: int, n_actions: int = N_ACTIONS) -> np.ndarray:
    """`pi[s, a] = 1/|A|` — the policy that picks uniformly at random.

    Boring on purpose. When a lab is about an *estimator*, the policy being
    evaluated should not be interesting, or you cannot tell which of the two you
    are looking at.
    """
    return np.full((n_states, n_actions), 1.0 / n_actions)


def visitable_states(P: np.ndarray, start: int = 0) -> np.ndarray:
    """Boolean mask of the states reachable from `start`.

    Use it whenever you average an error over states. The obstacles are states
    too, and the agent can never stand on one, so including them measures the
    map instead of the algorithm.
    """
    seen = {start}
    frontier = [start]
    while frontier:
        state = frontier.pop()
        for action in range(P.shape[1]):
            for nxt in np.flatnonzero(P[state, action] > 0):
                if int(nxt) not in seen:
                    seen.add(int(nxt))
                    frontier.append(int(nxt))
    mask = np.zeros(P.shape[0], dtype=bool)
    mask[sorted(seen)] = True
    return mask
