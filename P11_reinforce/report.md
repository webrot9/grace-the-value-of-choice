# P11 report — REINFORCE and the baseline

Name:
Student ID:

Two figures (see "What to hand in" in the README) and about five lines per
answer, written in the body of your email. Pass/fail; the numbers you quote must
be the ones your code printed.

---

## Q1 — what makes the comparison a check rather than a coincidence

Attach `reinforce.png`.

**(a)** The script prints `exact vs finite differences: 3.16e-12`. Neither of
those two quantities involves any sampling. Explain what each of them is, and
why agreeing to $10^{-12}$ is the precondition for the third column meaning
anything at all. What would you conclude if that number were $10^{-3}$?

**(b)** The last column reports the gap between the estimator and the truth in
standard errors: $+0.8, +1.8, +2.4, -2.1$. Is a $2.4$ there evidence of bias?
Say what you would expect to see across four components if the estimator were
unbiased, and what you would run to distinguish the two possibilities.

**(c)** The relative error is $0.91\%$ on $2\times10^{5}$ samples. How many
samples would you need for $0.1\%$? Give the number and the rule you used.

## Q2 — a theorem with a missing half

```
       b   total variance   bias, in SE
  0.0000           0.7392           1.6
  0.6144           0.2976           0.5   <- the mean reward
  0.8102           0.2704           0.5   <- b*
 10.0000          60.2086           0.8
```

**(a)** Prove, in two or three lines, that subtracting a constant $b$ leaves the
estimator unbiased. Your proof should make it obvious why the argument works for
*any* $b$, and where it would break if $b$ depended on the action.

**(b)** The variance is a parabola in $b$, and $b = 10$ multiplies it by 81
against no baseline at all. Write the variance as a function of $b$ — it is a
quadratic — and identify its minimiser. Confirm that your expression gives
$0.8102$ for this problem.

**(c)** $b^{*} = 0.8102$ and the mean reward is $0.6144$, but the variance at
the two differs by only 10%. Explain why they are close here, and describe the
kind of problem where they would not be. (Part C question 3 is one such
construction; you may use it, but say what makes it work.)

## Q3 — inside the loop, and what you predicted in Part C

**(a)** From panel 3, with $\gamma^t$: the baseline divides the final gradient
spread by $2.9$, and the return reaches 0.9 at the same iteration with and
without it (33, four seeds). Reconcile those two facts. What would a problem
have to look like for the smaller spread to show in the learning curve?

**(b)** For each of the three Part C changes: what you predicted before running,
what happened, and — if you were wrong — which assumption you had been using
without noticing.

For the whole-episode return in particular: quantify the difference. Both
estimators are unbiased, so the comparison has to be about variance, and a
sentence that does not contain a number does not answer the question.

## Declaration

Who you worked with, and which tools you used and for what.
