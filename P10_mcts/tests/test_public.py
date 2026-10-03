"""P10 public tests — the scores, the four phases, and the tree they build.

Tic-tac-toe is small enough that minimax gives the exact answer, so several of
these tests compare the search against ground truth rather than against a
tolerance. The searches that are checked are seeded and short.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from rl_lab import search as se


# --------------------------------------------------------------------------- #
# Q1 — the two selection rules
# --------------------------------------------------------------------------- #
def test_q1_an_unvisited_child_has_infinite_uct():
    """Same convention as the unpulled arm in P01: unknown means try it first."""
    assert se.uct_score(0.0, 0, 100, 1.4) == float("inf")


def test_q1_uct_matches_the_formula():
    got = se.uct_score(3.0, 4, 100, 1.4)
    assert got == pytest.approx(0.75 + 1.4 * math.sqrt(math.log(100) / 4),
                                abs=1e-12)


def test_q1_the_exploration_constant_is_actually_used():
    """The tutors' notebook takes c and then hard-codes 2. This test is the
    one that would have caught it: two different c must give two different
    scores."""
    a = se.uct_score(1.0, 4, 50, 0.5)
    b = se.uct_score(1.0, 4, 50, 2.0)
    assert abs(a - b) > 0.1


def test_q1_c_zero_is_pure_exploitation():
    assert se.uct_score(2.0, 4, 999, 0.0) == pytest.approx(0.5, abs=1e-12)


def test_q1_uct_bonus_shrinks_as_the_child_is_visited():
    scores = [se.uct_score(0.0, n, 1000, 1.4) for n in (1, 4, 16, 64)]
    assert all(b < a for a, b in zip(scores, scores[1:]))


def test_q1_puct_is_finite_for_an_unvisited_child():
    """This is the whole difference. UCT must try every child once before it
    looks at any child twice; PUCT is free to leave a low-prior move alone."""
    assert math.isfinite(se.puct_score(0.0, 0, 100, 0.05, 1.4))


def test_q1_puct_ranks_unvisited_children_by_their_prior():
    a = se.puct_score(0.0, 0, 100, 0.7, 1.4)
    b = se.puct_score(0.0, 0, 100, 0.1, 1.4)
    assert a > b


def test_q1_puct_matches_the_formula():
    got = se.puct_score(1.0, 2, 49, 0.25, 2.0)
    assert got == pytest.approx(0.5 + 2.0 * 0.25 * 7 / 3, abs=1e-12)


def test_q1_a_zero_prior_leaves_only_the_value():
    assert se.puct_score(3.0, 6, 100, 0.0, 1.4) == pytest.approx(0.5, abs=1e-12)


# --------------------------------------------------------------------------- #
# Q2 — the four phases
# --------------------------------------------------------------------------- #
def test_q2_expansion_removes_the_move_it_used():
    rng = np.random.default_rng(0)
    root = se.Node(se.initial_state(), se.X)
    n_before = len(root.untried)
    child = se.expand(root, se.uniform_prior, rng)
    assert len(root.untried) == n_before - 1
    assert child.move not in root.untried
    assert child.player == se.O and root.children == [child]


def test_q2_the_prior_is_taken_over_all_legal_moves():
    """Not over the shrinking `untried` list. Expand every child of the empty
    board and the nine priors must be the centre prior over nine moves, summing
    to one."""
    rng = np.random.default_rng(0)
    root = se.Node(se.initial_state(), se.X)
    while root.untried:
        se.expand(root, se.centre_prior, rng)
    priors = {ch.move: ch.prior for ch in root.children}
    assert sum(priors.values()) == pytest.approx(1.0, abs=1e-12)
    assert priors[4] > priors[0] > priors[1]


def test_q2_expand_all_creates_every_child_unvisited():
    """AlphaZero's expansion: all nine children of the empty board at once,
    none of them visited, each with its share of the prior over all nine."""
    root = se.Node(se.initial_state(), se.X)
    se.expand_all(root, se.centre_prior, np.random.default_rng(0))
    assert sorted(ch.move for ch in root.children) == list(range(9))
    assert all(ch.n == 0 and ch.player == se.O for ch in root.children)
    assert root.untried == []
    priors = {ch.move: ch.prior for ch in root.children}
    assert sum(priors.values()) == pytest.approx(1.0, abs=1e-12)
    assert priors[4] > priors[0] > priors[1]


def test_q2_only_puct_can_leave_a_child_unvisited():
    """Nine simulations from the empty board. UCT has to spend one on each of
    the nine moves. PUCT chooses among children it has not visited yet, so it
    can go back to a promising one before trying them all: that is the
    `n == 0` branch of `puct_score` at work inside `search`."""
    for seed in range(5):
        uct = se.search(se.initial_state(), se.X, iters=9, seed=seed)
        assert sorted(ch.n for ch in uct.children) == [1] * 9
        for prior in (se.centre_prior, se.uniform_prior):
            puct = se.search(se.initial_state(), se.X, iters=9, use_prior=True,
                             prior_fn=prior, seed=seed)
            visits = [ch.n for ch in puct.children]
            assert len(visits) == 9
            assert min(visits) == 0 and max(visits) >= 2


def test_q2_the_prior_picks_the_first_move_to_try():
    """At the first selection every child is unvisited, and PUCT ranks them by
    the prior alone: with the centre prior the first simulation goes through
    the centre, whatever the seed."""
    for seed in range(5):
        root = se.search(se.initial_state(), se.X, iters=1, use_prior=True,
                         prior_fn=se.centre_prior, seed=seed)
        assert [ch.move for ch in root.children if ch.n > 0] == [4]


def test_q2_a_rollout_ends_the_game():
    rng = np.random.default_rng(1)
    node = se.Node(se.initial_state(), se.X)
    for _ in range(50):
        assert se.rollout(node, rng) in {se.X, se.O, se.EMPTY}


def test_q2_a_rollout_from_a_finished_position_returns_its_result():
    node = se.Node("XXX" "OO " "   ", se.O)
    assert se.rollout(node, np.random.default_rng(0)) == se.X


def test_q2_backpropagation_credits_the_player_who_moved():
    """The mover into a node is other(node.player). Getting this backwards
    yields a search that plays to lose and reports that it is winning."""
    root = se.Node(se.initial_state(), se.X)
    child = se.Node(se.apply_move(se.initial_state(), 4, se.X), se.O,
                    parent=root, move=4)
    root.children.append(child)
    se.backpropagate(child, se.X)
    assert child.n == 1 and child.w == pytest.approx(1.0)
    assert root.n == 1 and root.w == pytest.approx(-1.0)


def test_q2_a_draw_moves_no_value():
    root = se.Node(se.initial_state(), se.X)
    se.backpropagate(root, se.EMPTY)
    assert root.n == 1 and root.w == pytest.approx(0.0)


def test_q2_every_iteration_adds_exactly_one_visit_to_the_root():
    for iters in (1, 10, 137):
        root = se.search(se.initial_state(), se.X, iters=iters, seed=0)
        assert root.n == iters


def test_q2_the_same_seed_gives_the_same_tree():
    a = se.search(se.initial_state(), se.X, iters=200, seed=3)
    b = se.search(se.initial_state(), se.X, iters=200, seed=3)
    assert se.visit_fractions(a) == se.visit_fractions(b)
    assert se.tree_size(a) == se.tree_size(b)


# --------------------------------------------------------------------------- #
# Q3 — measuring the tree, against an exact answer
# --------------------------------------------------------------------------- #
def test_q3_minimax_says_tic_tac_toe_is_a_draw():
    assert se.minimax(se.initial_state(), se.X) == 0
    assert se.optimal_moves(se.initial_state(), se.X) == set(range(9))


def test_q3_minimax_finds_a_forced_win():
    """X to move with two in a row: the winning move is the only optimal one."""
    state = "XX " "OO " "   "
    assert se.minimax(state, se.X) == 1
    assert se.optimal_moves(state, se.X) == {2}


def test_q3_minimax_finds_the_only_block():
    """X to move, O threatens 3-4-5. Blocking is forced, and it still only
    draws."""
    state = "X  " "OO " "  X"
    assert se.optimal_moves(state, se.X) == {5}
    assert se.minimax(state, se.X) == 0


def test_q3_visit_fractions_are_a_distribution():
    root = se.search(se.initial_state(), se.X, iters=300, seed=0)
    v = se.visit_fractions(root)
    assert sum(v.values()) == pytest.approx(1.0, abs=1e-12)
    assert set(v) == set(se.legal_moves(se.initial_state()))


def test_q3_tree_size_counts_the_visited_nodes():
    """A node that has never been visited cost no simulation and is not
    counted: nine children created by `expand_all`, one of them visited, and
    the tree has two nodes, the root and that child."""
    root = se.Node(se.initial_state(), se.X)
    rng = np.random.default_rng(0)
    se.expand_all(root, se.uniform_prior, rng)
    assert se.tree_size(root) == 0
    se.backpropagate(root.children[0], se.EMPTY)
    assert se.tree_size(root) == 2
    uct = se.Node(se.initial_state(), se.X)
    for _ in range(3):
        se.backpropagate(se.expand(uct, se.uniform_prior, rng), se.EMPTY)
    assert se.tree_size(uct) == 4


def test_q3_the_search_takes_a_forced_win():
    """Ground truth, not a win rate: with a win available in one move, enough
    simulations must find it. 400 is far more than enough on a 3-cell board."""
    root = se.search("XX " "OO " "   ", se.X, iters=400, seed=0)
    assert se.recommended_move(root) == 2


def test_q3_the_search_makes_the_forced_block():
    root = se.search("X  " "OO " "  X", se.X, iters=600, seed=0)
    assert se.recommended_move(root) == 5


def test_q3_puct_builds_a_smaller_tree_at_equal_budget():
    """Same position, same 400 simulations. UCT spends one on every child of
    every node it reaches before it looks at any of them twice; PUCT can go
    back to a good line first, so more of its simulations run down lines it
    already has, and it visits fewer distinct nodes. This holds with a prior
    that knows nothing too: it is a property of the rule."""
    state, player = "X  XO    ", se.O
    args = dict(iters=400, c=1.4, seed=0)
    plain = se.tree_size(se.search(state, player, use_prior=False, **args))
    for prior in (se.uniform_prior, se.centre_prior):
        guided = se.tree_size(se.search(state, player, use_prior=True,
                                        prior_fn=prior, **args))
        assert guided < 0.8 * plain
