# P12 — Maximum-entropy RL: the temperature is the exploration

**~30 min at home, then one class session.**

No training and no report this week. The hour goes on Part C.

SAC optimises reward **plus** entropy, and everything interesting is what the
coefficient $\alpha$ does. Nothing here is trained: the soft Bellman backup has a
fixed point we compute exactly, so the collapse at $\alpha \to 0$ is measured.

## Part A — at home, before class

Eight `# TODO`s in `rl_lab/entropy.py`, checked by
twenty-four tests.

```bash
pip install -r requirements.txt
python autograder.py
```

See [HOWTO](HOWTO.md) if this is your first lab.

You implement the soft value, the Boltzmann policy, the entropy, the soft backup,
and SAC's squashed-Gaussian log-density.

Two tests to read first.

`test_q3_the_corrected_density_integrates_to_one` and the one after it: the same
numbers with and without one term, integrating to $1.000$ and $0.661$. There is
no tolerance to argue with — a log density exponentiates to something that
integrates to one, or it is not one.

`test_q1_soft_value_survives_a_tiny_alpha`. The naive
`alpha * log(sum(exp(q / alpha)))` overflows to `inf` at $\alpha = 10^{-8}$, and
that is precisely the regime this session is about.

## Part B — in class

```bash
python experiments.py            # ~2 s, writes entropy.png
```

### The collapse

```
   alpha  mean entropy  max entropy?   V_soft(s0)     V*(s0)        gap      bound
       0      0.000000         0.0%     0.549736   0.549736   0.000000   0.000000
    0.01      0.174585        12.6%     0.625587   0.549736   0.075851   0.138629
     0.1      1.045858        75.4%     1.576369   0.549736   1.026633   1.386294
       1      1.378675        99.5%    13.908150   0.549736  13.358414  13.862944
      10      1.386222       100.0%   138.664723   0.549736 138.114987 138.629436
```

At $\alpha = 0$ the entropy is exactly zero and the soft backup *is* value
iteration. At $\alpha = 10$ the policy is uniform to four decimals and the
"value" is 138.66, of which 138.63 is entropy bonus. That is the whole bound,
because here the goal is a state like any other, and the agent sits in it for
ever collecting $\alpha \log 4$ at every step.

The `bound` column is $\alpha \log|\mathcal{A}|/(1-\gamma)$. Note the
$1/(1-\gamma)$: the per-step bonus is $\alpha\log|\mathcal{A}|$, and over an
infinite discounted horizon it accumulates a factor of ten at $\gamma = 0.9$.
The first version of this script quoted the per-step bound and understated it
tenfold, which is why there is now a test for it.

### The term everybody forgets

$$\log \pi(a) = \log \mathcal{N}(u \mid \mu, \sigma^2)
                - \sum_i \log\big(1 - \tanh^2 u_i\big)$$

Compute the log without forming $1 - \tanh^2 u$, which is exactly $0$ in floating
point once $|u|$ passes 19; the docstring gives an identity that stays finite.
The usual patch, an $\epsilon$ inside the log, is wrong for wide policies, and
`test_q3_the_log_jacobian_is_exact_in_the_tails` shows by how much.

```
with the tanh Jacobian, the density integrates to 1.000000
without it,                                       0.661290
```

Whatever that second function is, it is not a density, and an "entropy bonus"
computed from it is not an entropy. The tutors' notebook gets the Jacobian right
and then has no temperature at all — the target is `Q - log_prob`, which
hard-codes $\alpha = 1$.

### The entropy is not monotone in the noise

```
   sigma   H(pi), nats   H of the Gaussian
    0.10       -0.9799             -0.8836
    0.80        0.6169              1.1958
    2.00       -0.0281              2.1121
   10.00      -10.9186              3.7215
```

The Gaussian's entropy grows without bound in $\sigma$. The squashed policy's
peaks at $\sigma \approx 0.8$ with 0.617 nats — against a ceiling of
$\log 2 = 0.693$ for any distribution on $(-1,1)$ — and then falls hard. At
$\sigma = 10$ almost all the mass is against $\pm 1$: turning the exploration
noise up makes the policy **less** exploratory.

## Part C — break it

**Predict first, then run.**

1. Delete the Jacobian term and recompute the entropy bonus. Predict the
   direction of the error first: too high or too low, and by how much at
   $\sigma = 0.8$? What would an automatic temperature tuner do about it?
2. Run the soft backup with $\alpha$ decayed geometrically to zero, re-solving
   each step; then with $\alpha$ dropped to $10^{-8}$ in one jump. Both end at
   the same policy. What is different, and why does SAC anneal?
3. The maximum entropy on $(-1,1)$ is $\log 2$. Which $(\mu, \sigma)$ comes
   closest? Search numerically, then say why the squashed Gaussian cannot reach
   it and what distribution does.

Spend the time on question 1. The Jacobian bug does not crash and does not
diverge; combined with automatic temperature tuning it looks like a
hyperparameter problem, which is why it survives in published code.

## Sources

Haarnoja, T., Zhou, A., Abbeel, P., Levine, S. (2018), "Soft Actor-Critic",
*ICML 2018*, pp. 1861-1870 — the squashed Gaussian and its Jacobian are in
appendix C. Haarnoja, T. et al. (2018), arXiv:1812.05905 — the automatic
temperature adjustment. The soft backup comes from the principle of maximum
causal entropy, Ziebart (2010), PhD thesis, Carnegie Mellon University.

## What to hand in

This is a **short lab**: Part A is longer, there is no training and no report.
Both kinds count the same toward the lab requirement, except that three of your
five accepted labs must be long ones. See [HOWTO](HOWTO.md).

One email, before this lab's deadline, with:

1. **Two figures your code produced**, attached as files:
   - `entropy.png`, from `python experiments.py` as given;
   - the same figure after your change for Part C, question 1, saved under a new
     name so the first one is kept:
     `python experiments.py --out entropy_partc.png`.
2. **At least 150 words in the body of the email**: what you predicted for
   Part C, what changed between the two figures, and why. If nothing changed,
   say why.

Pass/fail, decided by a program: two images, 150 words in the body, on time.
The address and the subject line are in "How to submit a lab" on Classroom.
