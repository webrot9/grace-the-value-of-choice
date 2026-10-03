# P02 — Contextual bandits: what the posterior variance is for

**~30 min at home, then one class session.**

No training and no report this week. Most of the hour goes on Part C.

By the end you can write the Beta posterior and Thompson sampling for a bandit
whose best action depends on the context, and answer a question that an average
cannot answer.

## Part A — at home, before class

Seven `# TODO`s in `rl_lab/contextual.py`, checked by
nineteen tests.

```bash
pip install -r requirements.txt
python autograder.py
```

See [HOWTO](HOWTO.md) if this is your first lab.

You implement the Beta posterior (mean, variance, conjugate update), the two
selection rules, and the contextual regret.

The mean is $(k+1)/(n+2)$, not $k/n$.
`test_q1_mean_is_laplace_rule_of_succession` says why.

## Part B — in class

```bash
python experiments.py            # ~6 s, writes regret.png
```

Three agents on the same recommender with the same seeds: Thompson, the same
agent with the variance removed (`posterior-greedy`, which acts on the mean),
and uniform.

**Predict before you run it: which of the first two has the lower median
regret?**

Most people say Thompson. It is not:

```
                                     median     p90     above 100
Thompson (samples the posterior)       28.7    39.9        0/60
posterior-greedy (mean only)           10.4   105.4       16/60
```

Posterior-greedy has a third of the median regret. It also blows up on 16 seeds
out of 60, where it commits to the wrong course in one context and pays for the
rest of the horizon. Thompson does that zero times out of 60.

That is the theorem, and it is about the tail, not the average: sampling from the
posterior does not make you better on a typical run, it makes the catastrophic
run impossible. The means are 30.8 against 37.6, which reads as noise and tells
you nothing.

Two things to explain in class: why the $\mathrm{Beta}(1,1)$ prior makes an
untried course look good here (every true click-through rate is below 0.5), and
why the p90 column is the one to read.

## Part C — break it

The two agents differ by one line: `select_thompson` versus
`select_posterior_greedy`. **Predict first, then run.**

1. Raise every click-through rate above 0.5 — multiply `CTR` by 2, capped at
   0.95. Does posterior-greedy get better or worse? What happened to the prior's
   optimism?
2. Give posterior-greedy the mean *plus one standard deviation* instead of the
   mean, using `posterior_variance`. How many of the 60 seeds still blow up, and
   what have you just reinvented?

Question 2 is the bridge to next week.

## What to hand in

This is a **short lab**: Part A is longer, there is no training and no report.
Both kinds count the same toward the lab requirement, except that three of your
five accepted labs must be long ones. See [HOWTO](HOWTO.md).

One email, before this lab's deadline, with:

1. **Two figures your code produced**, attached as files:
   - `regret.png`, from `python experiments.py` as given;
   - the same figure after your change for Part C, question 2, saved under a new
     name so the first one is kept:
     `python experiments.py --out regret_partc.png`.
2. **At least 150 words in the body of the email**: what you predicted for
   Part C, what changed between the two figures, and why. If nothing changed,
   say why.

Pass/fail, decided by a program: two images, 150 words in the body, on time.
The address and the subject line are in "How to submit a lab" on Classroom.
