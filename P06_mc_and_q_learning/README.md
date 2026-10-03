# P06 — Monte Carlo and TD: which one is wrong, and which is noisy

**~30 min at home, 2 hours in class, ~30 min for the report.**

The first lab where the agent does not see the model. We still build it, and use
it to compute the exact answer everything is scored against, but no learner
reads it.

By the end you can feed MC and TD(0) the same episodes and say which part of
their error is bias and which is variance, with numbers.

## Part A — at home, before class

Eight `# TODO`s in `rl_lab/model_free.py`, checked by
twenty-six tests.

```bash
pip install -r requirements.txt
python autograder.py
```

See [HOWTO](HOWTO.md) if this is your first lab.

You implement the discounted returns, the first-visit mask, MC and TD(0)
evaluation, the bias-variance decomposition, $\epsilon$-greedy, MC control and
Q-learning.

Three tests sample, with fixed seeds and stated tolerances; the rest is exact
arithmetic on episodes written out by hand.

Two to read before you start.

`test_q2_terminal_bootstrap_is_zero` fails when the last transition of an episode
bootstraps off something other than zero. Most common bug in this lab: it does
not crash and the learning curve looks fine.

`test_q3_ties_are_broken_uniformly` fails on a plain `np.argmax`. A fresh $Q$
table is all zeros, so every action ties and `argmax` always returns action 0 —
the agent walks into the wall for its whole first episode.

## Part B — in class

```bash
python experiments.py            # ~45 s, writes bias_variance.png
python experiments.py --reps 8   # quick look while editing
```

### Evaluation

One policy, one number to estimate, $V^\pi(s_0) = 0.034317$ exactly. Both
estimators get the *same* episodes.

```
 episodes |  MC bias   MC std  sigma/sqrt(n) |  TD bias   TD std | batch bias  batch std
        5 |   0.0087   0.0406         0.0346 |  -0.0343   0.0000 |    -0.0255     0.0037
       20 |   0.0014   0.0180         0.0173 |  -0.0316   0.0010 |     0.0010     0.0071
      200 |   0.0007   0.0049         0.0055 |  -0.0010   0.0044 |    -0.0004     0.0028
```

MC has no bias at any sample size and its spread follows $\sigma/\sqrt{n}$,
where $\sigma$ is measured separately from 5000 episodes — not a fitted slope.

At 5 episodes TD(0) has bias $-0.0343$ and a standard deviation that prints as
$0.0000$ (it is about $3\times10^{-5}$), and the truth is $0.0343$: one online
pass over five episodes has barely moved it from the zero it started at. Almost
all error, almost no noise. That is report question 1.

Batch TD, the same data swept fifteen times, is the interesting middle. Its
spread is well below MC's at every size, eleven times smaller at 5 episodes, but
at 5 episodes it still has a bias of $-0.0255$, because fifteen passes over so
little data are far from where batch TD converges. By 20 episodes the bias is
gone.

### Control

$\epsilon = 0.4$ fixed, five seeds, median with interquartile band.

```
episodes |  MC vs Q* |  MC vs Q*_eps | Q-learning vs Q*
     500 |     0.248 |         0.095 |            0.142
    4000 |     0.202 |         0.053 |            0.014
```

Q-learning goes to $Q^{*}$. MC control does not — and it has not converged
slowly, it has converged to something else. On-policy control with a fixed
$\epsilon$ can only reach the best $\epsilon$-soft policy, which sits 0.1718 away
from $Q^{*}$ at $s_0$.

The script also reports that 5 of the 16 states were never updated: the four
obstacles and the goal. The errors above are therefore taken at $s_0$; a sup
norm over the whole table would be measuring the obstacle layout.

## Part C — break it

**Predict first, then run.**

1. Switch `mc_evaluate` to every-visit. Both are consistent, so what exactly are
   you trading?
2. Set `ALPHA_ONLINE = 0.5`. TD's bias at 5 episodes moves a lot. Does its
   variance stay at zero? Which term did you buy and which did you sell?
3. Decay $\epsilon$ in `mc_control` by 0.999 per episode. What does MC control
   converge to now, and why does the answer depend on how fast you decay rather
   than on the fact that you do?

Question 3 is the honest version of "just anneal $\epsilon$".

## What to hand in

This is a **long lab**: Part B is a real training run, and you write a report.
Both kinds count the same toward the lab requirement, except that three of your
five accepted labs must be long ones. See [HOWTO](HOWTO.md).

One email, before this lab's deadline, with:

1. **Two figures your code produced**, attached as files:
   - `bias_variance.png`, from `python experiments.py` as given;
   - the same figure after your change for Part C, question 3, saved under a new
     name so the first one is kept:
     `python experiments.py --out bias_variance_partc.png`.
2. **The report in the body of the email**: the three questions in `report.md`,
   answered there, at least 150 words in total. A `report.md` attached on its
   own is not counted.

Pass/fail, decided by a program: two images, 150 words in the body, on time.
The address and the subject line are in "How to submit a lab" on Classroom.
