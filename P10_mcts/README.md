# P10 — Monte Carlo tree search: what a prior buys, at equal budget

**~30 min at home, then one class session.**

No training and no report this week. The hour goes on Part C.

The game is tic-tac-toe because it is small enough to solve exactly with
minimax. Every measurement below is against the true value of the position, not
against another search.

By the end you can write UCT and PUCT, say what separates them for a child that
has never been visited, and answer "what does a prior do" with three numbers.

## Part A — at home, before class

Ten `# TODO`s in `rl_lab/search.py`, checked by
twenty-eight tests.

```bash
pip install -r requirements.txt
python autograder.py
```

See [HOWTO](HOWTO.md) if this is your first lab.

You implement the two selection rules, the four phases (select, expand, simulate,
backpropagate), and the measurements on the tree. There are two ways to expand.
UCT uses `expand`, one untried move at a time. PUCT uses `expand_all`, as
AlphaZero does: the first time the search reaches a node it creates every child,
each with its prior and no visits, and the rule chooses among them.
`test_q2_only_puct_can_leave_a_child_unvisited` is the difference in one test.

Read `test_q2_backpropagation_credits_the_player_who_moved` first. The player who
moved *into* a node is `other(node.player)`. Getting it backwards gives a search
that plays to lose while reporting excellent numbers — the most common bug here.

Then `test_q1_the_exploration_constant_is_actually_used`. It exists because the
tutors' notebook takes `c` and then writes `2 * sqrt(...)`, so every sweep over
`c` in that notebook is a sweep over nothing. Part C asks you to reproduce it.

## Part B — in class

```bash
python experiments.py            # ~12 s, writes mcts.png
```

Twelve positions after three random plies, each chosen so that **some moves are
optimal and some are not** — from the empty board every first move draws, so it
cannot tell a good search from a bad one. Same positions, same seeds, same
number of simulations for all three searches. PUCT runs twice, with the centre
prior and with a uniform prior that knows nothing about the moves.

```
UCT
  budget nodes visited   visits on optimal   entropy   picks optimal
      25          25.9               0.343     1.670           44.4%
     100          93.4               0.444     1.601           76.4%
     400         228.6               0.697     1.236           97.2%
     800         282.8               0.822     0.953          100.0%

PUCT + uniform prior
  budget nodes visited   visits on optimal   entropy   picks optimal
      25          24.5               0.421     1.294           55.6%
     100          71.2               0.628     1.113           77.8%
     400         132.1               0.851     0.703           98.6%
     800         157.7               0.911     0.528          100.0%

PUCT + centre prior
  budget nodes visited   visits on optimal   entropy   picks optimal
      25          24.0               0.455     1.303           56.9%
     100          67.7               0.657     0.984           83.3%
     400         123.7               0.844     0.650           97.2%
     800         147.9               0.906     0.489           97.2%
```

A node counts once it has had a simulation. PUCT's children exist before they
are visited, and counting them would compare memory with simulations.

At 400 simulations both PUCT searches visit about half the nodes UCT visits and
put about 85% of their root visits on optimal moves, against 69.7%. At 800 all
three pick an optimal move almost every time: PUCT does not make the search
stronger in the limit, it gets to the same answer sooner.

Compare the two PUCT tables before deciding what the gain is due to. The
difference in the formula is visible for a child that has never been visited:
PUCT scores it $c\,P(a)\sqrt{N}$, finite, so a move can wait, while UCT scores
it $+\infty$ and must try everything once before looking at anything twice.

### The exploration constant

```
      c   visits on optimal   entropy   picks optimal
    0.0               0.629     0.443           69.4%
    0.5               0.707     0.902           83.3%
    1.4               0.565     1.464           91.7%
    3.0               0.390     1.719           80.6%
   10.0               0.314     1.785           70.8%
```

UCT only, at 200 simulations. On the share of visits, the value everyone
quotes, $\sqrt{2} \approx 1.41$, is worse than 0.5 and worse than 0; on the
move the search recommends it is better than both. Concentrating the visits is
not finding the move, and with $c = 0$ more simulations do not help: change
`iters=200` in the sweep to 800 and its "picks optimal" stays at 68.1%. Note
that with a hard-coded 2 this table would be five identical rows.

## Part C — break it

**Predict first, then run.**

1. Replace `c` with a literal `2` in `uct_score` and re-run the sweep. Predict
   the five rows first. Why is this bug so hard to see from a win rate, and which
   test catches it?
2. Build the *reverse* of `centre_prior`, with centre 1, corners 2, edges 4.
   Compare it with plain UCT and with PUCT under the uniform prior, at every
   budget. What does a wrong prior cost, and against which of the two should
   the cost be measured?
3. Make `recommended_move` return the visited child with the highest mean value
   instead of the most visited one, and re-run "picks optimal" at budget 25.
   Predict first: much worse, much better, or about the same? What would a child
   need for its mean to be picked wrongly?

Spend the time on question 2. "A prior helps" only says something once you know
what a wrong prior costs.

## Sources

Kocsis, L., Szepesvári, C. (2006), "Bandit based Monte-Carlo Planning",
*ECML 2006*, pp. 282-293. Silver, D. et al. (2017), "Mastering the game of Go
without human knowledge", *Nature* 550, pp. 354-359 — the PUCT rule used here.

## What to hand in

This is a **short lab**: Part A is longer, there is no training and no report.
Both kinds count the same toward the lab requirement, except that three of your
five accepted labs must be long ones. See [HOWTO](HOWTO.md).

One email, before this lab's deadline, with:

1. **Two figures your code produced**, attached as files:
   - `mcts.png`, from `python experiments.py` as given;
   - the same figure after your change for Part C, question 2, saved under a new
     name so the first one is kept:
     `python experiments.py --out mcts_partc.png`.
2. **At least 150 words in the body of the email**: what you predicted for
   Part C, what changed between the two figures, and why. If nothing changed,
   say why.

Pass/fail, decided by a program: two images, 150 words in the body, on time.
The address and the subject line are in "How to submit a lab" on Classroom.
