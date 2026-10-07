# P03 — Value iteration, and what a contraction looks like

**~30 min at home, 2 hours in class, ~30 min for the report.**

An MDP here is three numpy arrays. No environment library and no training, which
is what lets you check the theorem to sixteen decimal places instead of roughly.

By the end you can write both Bellman operators, solve $V^\pi$ exactly, and show
that the contraction theorem is a straight line of slope $\log\gamma$ — measured,
not asserted.

## Part A — at home, before class

Seven `# TODO`s in `rl_lab/dp.py`, checked by
nineteen tests.

```bash
pip install -r requirements.txt
python autograder.py
```

See [HOWTO](HOWTO.md) if this is your first lab.

You implement the two Bellman operators, the greedy policy, the exact evaluation
$(\mathbf{I}-\gamma P^\pi)^{-1}\mathbf{r}^\pi$, and value iteration.

Nothing here is sampled, so nothing is flaky: a red test means the code is wrong.

The one that catches most people is
`test_q2_exact_evaluation_is_a_fixed_point_of_its_own_operator`. It fails when
$P^\pi$ is built with the wrong index — the classic bug, which the closed form
makes impossible to hide.

## Part B — in class

```bash
python experiments.py            # ~1 s, writes contraction.png
```

Left panel: $\lVert V_k - V^{*}\rVert_\infty$ on a log axis for four values of
$\gamma$, with $\gamma^k\lVert V_0-V^{*}\rVert_\infty$ dashed on the same axes.

The ratio of consecutive errors does not merely stay below $\gamma$, which is all
the theorem promises. On this grid it equals it, until the error reaches exactly
zero:

```
   gamma   final ratio   |ratio - gamma|
    0.50   0.500000000         0.00e+00
    0.90   0.900000000         1.11e-16
    0.99   0.990000000         0.00e+00
```

That is a property of a deterministic grid. With
`python experiments.py --slip 0.1` the ratio stays well below
$\gamma$, which the theorem allows just as well.

Then the second table, which is the interesting one: on the deterministic grid
the error reaches exactly zero in 8 sweeps regardless of $\gamma$, against a
bound of 1382 at $\gamma = 0.99$. Report question 2 is about that gap.

## Part C — break it

**Predict first, then run.**

1. In `value_iteration`, use `bellman_expectation` for a fixed policy instead of
   `bellman_optimality`. What does it converge to now, and with what modulus?
2. Set $\gamma = 1$ in `experiments.py`. The theorem's hypothesis is gone. Does
   it diverge, oscillate, or converge anyway? Explain from this gridworld — the
   answer is not "the theory is wrong".

## What to hand in

This is a **long lab**: Part B is a real training run, and you write a report.
Both kinds count the same toward the lab requirement, except that three of your
five accepted labs must be long ones. See [HOWTO](HOWTO.md).

One email, before this lab's deadline, with:

1. **Two figures your code produced**, attached as files:
   - `contraction.png`, from `python experiments.py` as given;
   - the same figure after your change for Part C, question 2, saved under a new
     name so the first one is kept:
     `python experiments.py --out contraction_partc.png`.
2. **The report in the body of the email**: the three questions in `report.md`,
   answered there, at least 150 words in total. A `report.md` attached on its
   own is not counted.

Pass/fail, decided by a program: two images, 150 words in the body, on time.
The address and the subject line are in "How to submit a lab" on Classroom.
