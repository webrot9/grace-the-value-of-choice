# P11 — REINFORCE: an unbiased gradient, and a baseline that can hurt

**~30 min at home, 2 hours in class, ~30 min for the report.**

The first lab where the thing being estimated is a *gradient*, and the first
where the reference involves no sampling: for a one-step problem $J(\pi_\theta)$ is
available in closed form, so the estimator is checked against finite differences
of the truth rather than against another estimate.

By the end you can show that **every** constant baseline is unbiased, and that
the variance is a parabola in it whose minimum is not the mean reward.

## Part A — at home, before class

Nine `# TODO`s in `rl_lab/pg.py`, checked by
twenty-two tests.

```bash
pip install -r requirements.txt
python autograder.py
```

See [HOWTO](HOWTO.md) if this is your first lab.

You implement the softmax, the log-probability gradient, the exact and
finite-difference gradients, the estimator, the optimal baseline, and the
episodic version with a state baseline.

The episodic version weights step $t$ by $\gamma^t$, as the policy gradient
theorem does: the objective is the return discounted to the start of the
episode, and a decision taken at step $t$ reaches it discounted by $\gamma^t$.
Many implementations leave the factor out, and `episode_gradient` has an
argument for that too, `gamma_t=False`. Without it every step counts as if the
episode started there, and the estimate is no longer an unbiased estimate of
$\nabla_\theta J$. `test_q3_gamma_t_makes_the_estimate_unbiased` measures the
difference exactly, on a problem small enough to list every episode.

Two tests to read first.

`test_q1_softmax_survives_large_logits`. Without the max-subtraction, `softmax`
returns `nan` at logits around 800 — silently, mid-run, and the traceback you
eventually get points somewhere else.

`test_q3_returns_from_t_onwards_not_the_whole_episode`. The estimator uses
$G_t$, the return from step $t$ on. Using the whole episode's return is also
unbiased, and the two are easy to confuse.

## Part B — in class

```bash
python experiments.py                            # ~23 s, writes reinforce.png
python experiments.py --samples 20000 --seeds 2  # quicker
```

### Unbiased, against a reference that does not sample

```
  i    exact grad   finite diff     REINFORCE     +- SE   in SE
  0      0.096497      0.096497      0.097260  0.000950    +0.8
  2     -0.336007     -0.336007     -0.334585  0.000582    +2.4
  3      0.256873      0.256873      0.253799  0.001471    -2.1

exact vs finite differences: 3.16e-12 (no sampling on either side)
REINFORCE vs exact: 0.91% relative error on 200000 samples
```

The first two columns agree to $3\times10^{-12}$ — that is what makes the third
column's comparison meaningful.

### Every baseline is unbiased. Only one is sane.

```
       b   total variance   bias, in SE
 -4.0000          16.7009           2.6
  0.0000           0.7392           1.6
  0.6144           0.2976           0.5   <- the mean reward, which is not b*
  0.8102           0.2704           0.5   <- b*, the variance-minimising baseline
 10.0000          60.2086           0.8
```

Read the columns against each other. The bias never leaves sampling noise: any
constant is unbiased, including $-4$ and $10$. The variance moves by a factor of
200. "Use a baseline" is not advice; "use a good baseline" is.

And $b^{*} = 0.8102$ while the mean reward is $0.6144$. The variance-minimising
constant is a mean of the rewards *weighted by* $\lVert\nabla\log\pi\rVert^2$.
Almost every implementation subtracts the average return instead — close enough
here (0.2976 against 0.2704), and Part C asks when it is not.

### Inside a training loop

Four seeds, 200 iterations of 10 episodes, same step size, with and without
$\gamma^t$. The last column is the first iteration at which the mean return of
the last ten reaches 0.9.

```
                                             return     gradient spread   0.9 at
            gamma^t, no baseline: 0.5454 -> 1.0099   0.0632 -> 0.0788       33
         gamma^t, state baseline: 0.5140 -> 0.9883   0.0503 -> 0.0269       33
         no gamma^t, no baseline: 0.7145 -> 0.9989   0.1696 -> 0.0739       16
      no gamma^t, state baseline: 0.6847 -> 1.0134   0.1278 -> 0.0231       17
```

The state baseline divides the final spread by 2.9 with $\gamma^t$ and by 3.2
without, and it does not change when the return reaches 0.9. The goal pays at
most 1 and all four runs get there, so on this grid a smaller spread has nothing
left to buy.

Without $\gamma^t$ the same step size learns about twice as fast. The reward
comes late in the episode, $\gamma^t$ shrinks most of the terms, and the
gradient is smaller. The version without it is larger, and its mean is not
$\nabla_\theta J$.

## Part C — break it

**Predict first, then run.**

1. In `episode_gradient`, weight every step by the return of the whole
   episode, $G_0$, instead of $\gamma^t G_t$. Without a baseline this is still
   unbiased. Predict the gradient spread and the learning curve over four seeds,
   then run, and explain the number from where this grid pays its rewards.
2. Freeze the state baseline at its value after iteration 10. Predict what
   happens as the returns grow. At which iteration does it start *adding*
   variance, and how does that relate to the parabola?
3. Find rewards where $b^{*}$ and the mean reward are far apart. Hint: $b^{*}$
   weights by $\lVert\nabla\log\pi\rVert^2$, largest for actions the policy is
   uncertain about — so make the high-reward action one it almost never takes.

## What to hand in

This is a **long lab**: Part B is a real training run, and you write a report.
Both kinds count the same toward the lab requirement, except that three of your
five accepted labs must be long ones. See [HOWTO](HOWTO.md).

One email, before this lab's deadline, with:

1. **Two figures your code produced**, attached as files:
   - `reinforce.png`, from `python experiments.py` as given;
   - the same figure after your change for Part C, question 1, saved under a new
     name so the first one is kept:
     `python experiments.py --out reinforce_partc.png`.
2. **The report in the body of the email**: the three questions in `report.md`,
   answered there, at least 150 words in total. A `report.md` attached on its
   own is not counted.

Pass/fail, decided by a program: two images, 150 words in the body, on time.
The address and the subject line are in "How to submit a lab" on Classroom.
