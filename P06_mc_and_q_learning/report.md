# P06 report — Monte Carlo, TD, and what each of them is estimating

Name:
Student ID:

About five lines per answer, written in the body of your email. Pass/fail; the
numbers you quote must be the ones your code printed.

---

## Q1 — an estimator with zero variance

Attach `bias_variance.png`. The script prints this row:

```
 episodes | MC bias  MC std  sigma/sqrt(n) | TD bias  TD std | batch bias  batch std
        5 |  0.0087  0.0406         0.0346 | -0.0343   0.0000|    -0.0255     0.0037
```

and the truth is $V^\pi(s_0) = 0.034317$.

**(a)** TD(0) with a single pass has standard deviation $0.0000$ at five
episodes, to four decimals. State what it is returning and why its variance is
so small.
Then answer the question that follows: is "lower variance" a reason to prefer an
estimator? Give the condition under which it is, in one sentence, using the
decomposition $\mathrm{MSE} = \mathrm{bias}^2 + \mathrm{variance}$.

**(b)** The `sigma/sqrt(n)` column is not fitted to the measurement: $\sigma$ is
measured separately, from 5000 independent episodes, and divided by $\sqrt{n}$.
Explain why *first-visit* MC at the start state must obey that law exactly, and
say which of the two words — *first-visit*, *start state* — you would have to
drop for the argument to break, and what would replace $\sigma/\sqrt{n}$ then.

**(c)** The measured MC spread agrees with the line to within 17%, while with 40
repetitions a standard deviation is itself known to about 11%. Is 17% evidence
that something is wrong? Say what you would run to find out.

## Q2 — two algorithms, two answers

The control table:

```
episodes |         MC vs Q* |     MC vs Q*_eps | Q-learning vs Q*
     500 |            0.248 |            0.095 |            0.142
    1500 |            0.210 |            0.052 |            0.099
    4000 |            0.202 |            0.053 |            0.014
```

with $\epsilon = 0.4$ fixed, and
$\lVert Q^{*}(s_0,\cdot) - Q^{*}_{\epsilon}(s_0,\cdot)\rVert_\infty = 0.1718$.

**(a)** MC control's error against $Q^{*}$ stops falling at about $0.2$ while
its error against $Q^{*}_{\epsilon}$ keeps falling. Say precisely what MC
control has converged to and why $\epsilon$ appears in the answer. Your
explanation should use the number $0.7$, which is
$1 - \epsilon + \epsilon/|\mathcal{A}|$ and which one of the Part A tests
checks.

**(b)** Q-learning uses the *same* $\epsilon$-greedy behaviour and reaches
$Q^{*}$ anyway. What in the update rule makes that possible? Name the property
of the algorithm, and point at the exact term in
$Q(s,a) \leftarrow Q(s,a) + \alpha(r + \gamma \max_{a'}Q(s',a') - Q(s,a))$
that carries it.

**(c)** Q-learning's convergence to $Q^{*}$ holds under a condition about
visiting every state-action pair infinitely often. The script reports that 5 of
the 16 states were never updated at all. Reconcile those two statements —
carefully, because the easy answer ("the theorem is only asymptotic") is not the
right one here.

## Q3 — what you predicted in Part C

For each of the three changes: what you predicted before running, what happened,
and — if you were wrong — which assumption you had been using without noticing.

For the decayed $\epsilon$ in particular: state the two conditions a schedule
must satisfy for on-policy control to reach $Q^{*}$, and say whether
`eps *= 0.999` satisfies them. Show the arithmetic.

## Declaration

Who you worked with, and which tools you used and for what.
