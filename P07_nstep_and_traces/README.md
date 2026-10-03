# P07 — n-step returns and traces: two algorithms, one answer

**~30 min at home, then one class session.**

No training and no report this week. The hour goes on Part C, which breaks the
equivalence one condition at a time.

By the end you can write $G_t^{(n)}$ and $G_t^\lambda$, implement the forward
view (which needs the whole future) and the backward view (which needs one number
per state), and check that they agree to $1.7\times10^{-16}$.

## Part A — at home, before class

Six `# TODO`s in `rl_lab/traces.py`, checked by
twenty-one tests.

```bash
pip install -r requirements.txt
python autograder.py
```

See [HOWTO](HOWTO.md) if this is your first lab.

You implement the $n$-step return, the $\lambda$-return, both views, $n$-step TD
and the RMS error. Nothing is sampled: a red test means the code is wrong.

Two to read first.

`test_q1_the_bootstrap_uses_the_state_n_steps_ahead`. Bootstrapping off
$s_{t+n-1}$ instead of $s_{t+n}$ does not crash, does not diverge, and gives a
learning curve you would happily put in a report.

`test_q2_neither_view_touches_the_value_function`. Both views take $V$ frozen.
Update it in place and the equivalence stops being exact — and the error looks
like rounding when it is a different algorithm.

## Part B — in class

```bash
python experiments.py            # ~26 s, writes traces.png
```

### The equivalence

```
 lambda   max |fwd - bwd|   in units of eps
   0.00         0.000e+00               0.0
   0.60         6.939e-17               0.3
   1.00         1.665e-16               0.8
```

Twenty episodes, seven values of $\lambda$, largest disagreement anywhere
$1.665\times10^{-16}$ — below one machine epsilon. The forward view averages
every $n$-step return; the backward view keeps one eligibility number per state.
Same algorithm, written twice.

It needs three things, all in the code: updates **accumulated** and applied at
the end of the episode, $V$ **frozen** while they accumulate, and **accumulating**
traces. Part C removes them one at a time.

### The optimum in $n$

RMS error against the exact $V^\pi$ after 20 episodes, median over 30 seeds:

```
alpha       1       2       3       5       8      13      21      34      60
 0.01  0.1452  0.1251  0.1115  0.0945  0.0834  0.0765  0.0742  0.0722  0.0718
 0.02  0.1193  0.0882  0.0711  0.0549  0.0461  0.0432  0.0427  0.0428  0.0427
 0.05  0.0705  0.0425  0.0410  0.0453  0.0585  0.0712  0.0786  0.0788  0.0781
 0.10  0.0430  0.0520  0.0628  0.0910  0.1208  0.1470  0.1712  0.1751  0.1745
```

The best $n$ is 60, 21, 3, 1 as $\alpha$ goes 0.01, 0.02, 0.05, 0.1. Two of the
four have the minimum strictly inside the range; the other two put it at an end.
Quoting "the best $n$" without the $\alpha$ it was measured at says nothing.

### Where it falls over

At $\alpha = 0.3$ the error is 225, not 0.05. That is divergence, and it follows
from the design choice that makes the equivalence exact: because updates
accumulate, a state visited $k$ times moves by $k\alpha$ times its error.
Episodes average 80.8 steps over 11 reachable states, so $k \approx 7.3$ and
$\alpha = 0.3$ is an effective step of 2.2.

Every error above is averaged over the **11 states an episode can reach**, not
all 16: four cells are obstacles and one is the goal.

## Part C — break it

**Predict first, then run.** Each change removes one of the three conditions.

1. Replacing traces: `e[states[t]] = 1.0` instead of `+= 1.0`. The disagreement
   goes from $10^{-16}$ to something you can read off a plot. At which $\lambda$
   is it largest, and why does $\lambda = 0$ not move at all?
2. Update `V` in place inside `forward_update` instead of accumulating. Measure
   the disagreement at $\alpha = 0.2, 0.1, 0.05, 0.025$. It shrinks, but not as
   $\alpha^2$. Work out the measured exponent and explain the gap with the same
   $k \approx 7.3$.
3. Set $\alpha = 0.3$ in the sweep, then divide $\alpha$ by the mean visits per
   state and re-run. Does the divergence go away? What does that say about
   comparing step sizes between an offline and an online implementation?

Spend the time on question 1. Replacing traces are usually presented as a minor
variant; here you can measure exactly how minor.

## What to hand in

This is a **short lab**: Part A is longer, there is no training and no report.
Both kinds count the same toward the lab requirement, except that three of your
five accepted labs must be long ones. See [HOWTO](HOWTO.md).

One email, before this lab's deadline, with:

1. **Two figures your code produced**, attached as files:
   - `traces.png`, from `python experiments.py` as given;
   - the same figure after your change for Part C, question 1, saved under a new
     name so the first one is kept:
     `python experiments.py --out traces_partc.png`.
2. **At least 150 words in the body of the email**: what you predicted for
   Part C, what changed between the two figures, and why. If nothing changed,
   say why.

Pass/fail, decided by a program: two images, 150 words in the body, on time.
The address and the subject line are in "How to submit a lab" on Classroom.
