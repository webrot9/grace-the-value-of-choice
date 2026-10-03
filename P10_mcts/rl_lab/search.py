r"""P10 — Monte Carlo tree search, and what a prior actually buys.

Reference implementation. `search.py` is generated from this file by
tools/strip_solutions.py. Never edit the generated file.

Tic-tac-toe, because it is small enough that **minimax gives the exact answer**.
Everything MCTS produces is then measured against ground truth rather than
against another approximation: the value of the empty board is a draw, and any
search that says otherwise is wrong in a way you can point at.

The four phases, and where each of them lives:

    selection      `best_child`, walking down by UCT or PUCT
    expansion      `expand` (UCT, one child) or `expand_all` (PUCT, all)
    simulation     `rollout`, uniformly random to the end of the game
    backpropagation `backpropagate`, alternating sign up the path

The session's question is what changes when the selection rule is given a
prior. Not "does it play better" — at equal budget, *where does the search
spend its simulations*.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

EMPTY, X, O = " ", "X", "O"
LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8),      # rows
         (0, 3, 6), (1, 4, 7), (2, 5, 8),      # columns
         (0, 4, 8), (2, 4, 6)]                 # diagonals


# --------------------------------------------------------------------------- #
# Given: the game, as a plain 9-character string
# --------------------------------------------------------------------------- #
def initial_state() -> str:
    return EMPTY * 9


def legal_moves(state: str) -> list[int]:
    return [i for i, c in enumerate(state) if c == EMPTY]


def apply_move(state: str, move: int, player: str) -> str:
    return state[:move] + player + state[move + 1:]


def other(player: str) -> str:
    return O if player == X else X


def winner(state: str) -> str | None:
    """`X`, `O`, `' '` for a draw, or `None` if the game is still going."""
    for a, b, c in LINES:
        if state[a] != EMPTY and state[a] == state[b] == state[c]:
            return state[a]
    return None if EMPTY in state else EMPTY


@lru_cache(maxsize=None)
def minimax(state: str, player: str) -> int:
    """The exact game-theoretic value of `state` with `player` to move, from
    `player`'s point of view: +1 win, 0 draw, -1 loss.

    Given, and memoised: tic-tac-toe has 5478 reachable states, so the whole
    game tree fits in a fraction of a second and there is no reason to
    approximate anything we can simply compute. This is the ground truth every
    measurement in Part B is taken against.
    """
    w = winner(state)
    if w is not None:
        return 0 if w == EMPTY else (1 if w == player else -1)
    return max(-minimax(apply_move(state, m, player), other(player))
               for m in legal_moves(state))


def optimal_moves(state: str, player: str) -> set[int]:
    """Every move that preserves the game-theoretic value of the position."""
    best = minimax(state, player)
    return {m for m in legal_moves(state)
            if -minimax(apply_move(state, m, player), other(player)) == best}


def centre_prior(state: str, moves: list[int]) -> np.ndarray:
    """A prior over the legal moves: centre 4, corners 2, edges 1, normalised.

    Domain knowledge and nothing more — no network, no training. It is here so
    that "what does a prior do to the search" can be answered without first
    building the thing that produces one.
    """
    weight = {4: 4.0, 0: 2.0, 2: 2.0, 6: 2.0, 8: 2.0}
    w = np.array([weight.get(m, 1.0) for m in moves])
    return w / w.sum()


def uniform_prior(state: str, moves: list[int]) -> np.ndarray:
    """The prior that knows nothing. Passing this recovers plain UCT up to the
    constant, which is the control condition Part B needs."""
    return np.full(len(moves), 1.0 / len(moves))


class Node:
    """One node of the search tree. `player` is the player to move here, and
    `value` is accumulated from the point of view of the player who *moved into*
    this node — which is why `backpropagate` alternates sign."""

    __slots__ = ("state", "player", "parent", "move", "children",
                 "untried", "n", "w", "prior")

    def __init__(self, state: str, player: str, parent=None, move=None,
                 prior: float = 1.0):
        self.state = state
        self.player = player
        self.parent = parent
        self.move = move
        self.prior = prior
        self.children: list[Node] = []
        self.untried = legal_moves(state) if winner(state) is None else []
        self.n = 0
        self.w = 0.0

    @property
    def is_terminal(self) -> bool:
        return winner(self.state) is not None

    @property
    def fully_expanded(self) -> bool:
        return not self.untried


# --------------------------------------------------------------------------- #
# Q1: the two selection rules
# --------------------------------------------------------------------------- #
def uct_score(w: float, n: int, parent_n: int, c: float) -> float:
    r"""$\dfrac{w}{n} + c\sqrt{\dfrac{\ln N}{n}}$, and $+\infty$ when $n = 0$.

    The infinity is not a hack: an unvisited child has infinite uncertainty and
    must be tried before any visited one. It is the same convention as the
    unpulled arm in P01, and for the same reason.

    Note that `c` is an argument and must be used. The version in the tutors'
    notebook takes `c` and then writes `2 * sqrt(...)`, so every sweep over `c`
    in that notebook is a sweep over nothing — one of the Part C questions is to
    reproduce that bug on purpose and see how invisible it is.
    """
    # TODO: ...
    raise NotImplementedError


def puct_score(w: float, n: int, parent_n: int, prior: float, c: float) -> float:
    r"""The AlphaZero rule:

    $$\frac{w}{n} + c\,P(a)\,\frac{\sqrt{N}}{1+n}$$

    with $w/n$ taken as $0$ when $n = 0$ — and note that this is **finite** for
    an unvisited child, unlike UCT. That single difference is what lets a prior
    steer: a child the prior dislikes can wait, where UCT is obliged to try
    everything at least once before it looks at anything twice.
    """
    # TODO: ...
    raise NotImplementedError


def best_child(node: Node, c: float, use_prior: bool) -> Node:
    r"""The child maximising the chosen score. Ties go to the first child, which
    keeps the search deterministic given the rollout seed."""
    # TODO: ...
    raise NotImplementedError


# --------------------------------------------------------------------------- #
# Q2: the four phases
# --------------------------------------------------------------------------- #
def expand(node: Node, prior_fn, rng: np.random.Generator) -> Node:
    r"""Take one untried move, create the child, attach its prior, return it.

    The prior over a node's children is computed once, over **all** of that
    node's legal moves, and each child keeps its own entry. Recomputing it per
    child over the shrinking `untried` list renormalises it every time, which is
    a bug that leaves the search working and the priors meaningless.
    """
    # TODO: ...
    raise NotImplementedError


def expand_all(node: Node, prior_fn, rng: np.random.Generator) -> None:
    r"""AlphaZero's expansion: create **every** child of `node` at once, each
    with its prior and $n = 0$, and leave `untried` empty.

    This is what lets PUCT choose among children that have never been visited.
    With `expand`, a node only reaches `best_child` once every child has a visit,
    and the $n = 0$ branch of `puct_score` never runs. The children go in a
    random order drawn from `rng`, so that ties among unvisited children with
    equal priors are broken at random and not always towards cell 0.
    """
    # TODO: ...
    raise NotImplementedError


def rollout(node: Node, rng: np.random.Generator) -> str:
    r"""Play uniformly at random from `node.state` to the end of the game and
    return the winner (`'X'`, `'O'`, or `' '` for a draw)."""
    # TODO: ...
    raise NotImplementedError


def backpropagate(node: Node, result: str) -> None:
    r"""Walk up to the root, adding $+1$, $0$ or $-1$ to each node's `w` from
    the point of view of the player who moved into it, and $1$ to its `n`.

    The player who moved into `node` is `other(node.player)`: at a node it is
    *someone else's* turn precisely because the move that got here has been
    played. Getting this backwards produces a search that plays deliberately
    badly, and does so while reporting excellent win rates.
    """
    # TODO: ...
    raise NotImplementedError


def search(state: str, player: str, iters: int = 400, c: float = 1.4,
           use_prior: bool = False, prior_fn=None, seed: int = 0) -> Node:
    r"""`iters` iterations of select, expand, simulate, backpropagate.

    Without a prior (UCT): walk down by `best_child` while the node has no
    untried move, then `expand` one untried move and roll out from the new child.

    With a prior (PUCT, as in AlphaZero): the first time the walk passes through
    a node, `expand_all` creates all its children; `best_child` then chooses
    among them, unvisited ones included, and the walk stops at the first child
    with $n = 0$, which is where the rollout starts.

    Either way each iteration rolls out from one node and adds one visit to the
    root. Returns the root, so the tree can be inspected: this lab measures the
    tree, not just the move it recommends.
    """
    # TODO: ...
    raise NotImplementedError


# --------------------------------------------------------------------------- #
# Q3: measuring the tree
# --------------------------------------------------------------------------- #
def tree_size(root: Node) -> int:
    """The number of nodes the search has visited at least once, root included.

    With `expand_all` a node can sit in the tree without ever being visited,
    holding its prior and nothing else. It cost no simulation, and counting it
    would make a PUCT tree look larger than a UCT tree that received exactly the
    same simulations, so it is left out. Under UCT every node is visited as soon
    as it is created, and the two counts agree.
    """
    # TODO: ...
    raise NotImplementedError


def visit_fractions(root: Node) -> dict[int, float]:
    r"""`{move: fraction of the root's child visits}` — the search's answer,
    read as a distribution rather than as a single argmax.

    A search that puts 51% on one move and 49% on another has said something
    quite different from one that puts 95% on the first, and `argmax` throws
    that away.
    """
    # TODO: ...
    raise NotImplementedError


def recommended_move(root: Node) -> int:
    """The most visited child's move — AlphaZero's rule, and more robust than
    the highest mean value, which a single lucky rollout can lift."""
    return max(root.children, key=lambda ch: ch.n).move
