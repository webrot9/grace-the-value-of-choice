# P03 report — value iteration

Name:
Student ID:

Two figures (see "What to hand in" in the README) and about five lines per
answer, written in the body of your email. Pass/fail; the numbers you quote must
be the ones your code printed.

---

## Q1 — the slope is the theorem

Attach `contraction.png`. In the left panel the measured error and the dashed
$\gamma^k \lVert V_0 - V^{*}\rVert_\infty$ lie on top of each other until
sweep 7; from sweep 8 the error is exactly zero, and the plot draws it at the
bottom of the axis.

State, for $\gamma = 0.9$: the slope of the measured line on the log axis, and
what that slope is in closed form. Then run `python experiments.py --slip 0.1`
and say whether the measured line still lies on the dashed one, and why that
does not contradict the theorem.

## Q2 — the bound is right about the rate and wrong about the horizon

The script prints this table:

```
 slip  gamma   measured    bound    exact zero at
  0.0   0.90          8      138                8
  0.0   0.99          8     1382                8
  0.1   0.90         22      138               45
  0.1   0.99         25     1382               50
  0.3   0.90         39      138               95
  0.3   0.99         50     1382              110
```

`measured` is the number of sweeps to reach $10^{-6}$; `bound` is
$\log(1/\varepsilon)/(1-\gamma)$, the standard estimate.

Two things need explaining, and they are different.

**(a)** On the deterministic grid (`slip = 0`) the error reaches **exactly zero**
after 8 sweeps, and 8 does not depend on $\gamma$ at all. Where does the 8 come
from? Why does the $1/(1-\gamma)$ factor never appear?

**(b)** With `slip = 0.1` the bound is 138 against a measured 22 at
$\gamma = 0.9$, and 1382 against 25 at $\gamma = 0.99$. Raising $\gamma$ by a
factor that multiplies the bound by ten changes the measurement by three sweeps.
What is the bound quantifying that this MDP does not have?

Your answer should use the words *worst case over all MDPs* and should say what
property of this gridworld makes it a long way from that worst case.

## Q3 — what you predicted in Part C

For each of the two changes: what you predicted before running, what happened,
and — if you were wrong — which assumption of the theorem you had been using
without noticing.

For $\gamma = 1$ in particular: say whether what you observed contradicts the
contraction theorem, and justify the answer from the theorem's hypotheses rather
than from the plot.

## Declaration

Who you worked with, and which tools you used and for what.
