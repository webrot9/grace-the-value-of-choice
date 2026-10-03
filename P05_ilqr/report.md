# P05 report — LQR, iLQR, certainty equivalence

Name:
Student ID:

About five lines per answer, written in the body of your email. Pass/fail; the
numbers you quote must be the ones your code printed.

---

## Q1 — the theorem, and what it does not say

Attach `certainty.png`. In the left panel four gain curves lie on top of each
other; the script reports the comparison as `np.array_equal`, not as a
tolerance.

**(a)** Point at the line of `riccati_step` where $\sigma$ would have to appear
for the gain to depend on it, and say why it cannot. One sentence about the
value function being quadratic is enough, but it has to be the right sentence.

**(b)** The right panel shows the cost rising with $\sigma$ while the policy does
not move. State the closed form of the extra cost, and say what it does *not*
depend on. Then answer the question this raises: if the controller cannot
improve on ignoring the noise, why does anyone ever build a controller that
estimates the noise level?

## Q2 — a bias that was not there

Run the script twice:

```bash
python experiments.py --seeds 200
python experiments.py                 # 2000, the default
```

At 200 seeds the measured extra cost is below the theoretical
$\sigma^2\sum_t\mathrm{tr}(\mathbf{P}_t)$ at **all three** noise levels, by
$-5.7\%$, $-4.3\%$ and $-4.1\%$. Three out of three, same sign, similar
magnitude: that reads like a systematic bias — a missing factor, a wrong index
somewhere in the trace. At 2000 seeds the same three numbers are $+1.6\%$,
$+0.2\%$, $-0.1\%$.

**(a)** The `gap in SE` column is $-0.7$, $-1.2$, $-1.2$ at 200 seeds. Given
those, was the $-4\%$ ever evidence of anything? Say what the standard error is
measuring and what it is not.

**(b)** Explain why the three rows agree in sign. They are three separate
columns of the table, so it is tempting to read them as three independent
confirmations. They are not independent, and the reason is one line of
`experiments.py`. Find it and quote it.

**(c)** The deficit is not the same at the three noise levels: $-5.7\%$ at
$\sigma = 0.05$ against $-4.1\%$ at $\sigma = 0.5$. Decompose the extra cost of
a single rollout into a term proportional to $\sigma^2$ and a term proportional
to $\sigma$, say which of the two has mean zero, and use that to explain why the
relative error shrinks as $\sigma$ grows.

Then, and only then, say what you would have to change in the script to make the
three rows genuinely independent — and whether you should.

## Q3 — what you predicted in Part C

For each of the three changes: what you predicted before running, what happened,
and — if you were wrong — which assumption you had been using without noticing.

For the state-dependent noise in particular: say whether what you observed
contradicts certainty equivalence, and justify the answer from the theorem's
hypotheses rather than from the numbers.

## Declaration

Who you worked with, and which tools you used and for what.
