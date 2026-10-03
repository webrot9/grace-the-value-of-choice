# P09 report — maximization bias

Name:
Student ID:

Two figures (see "What to hand in" in the README) and about five lines per
answer, written in the body of your email. Pass/fail; the numbers you quote must
be the ones your code printed.

---

## Q1 — where the bias comes from

Attach `maxbias.png`.

**(a)** The first table has no reinforcement learning in it: ten unbiased
estimates of $-0.1$, and a maximum that reads $+0.2438$ at 20 samples per
action. Explain, in terms of $\mathbb{E}[\max_i X_i]$ versus
$\max_i \mathbb{E}[X_i]$, why this had to happen. Say which of the two the
Q-learning target computes and which one it wants.

**(b)** The *bias* — the printed value minus $-0.1$ — falls from $0.6892$ at 5
samples per action to $0.1218$ at 160. Predict that ratio from the standard error
alone, before dividing. Then get the constant as well: the bias should be
$\mathbb{E}[\max_i Z_i]/\sqrt{n}$ for ten standard normals. Look up or derive
that expectation, check it against both ends of the table, and then say why the
bias never reaches zero for any finite $n$.

**(c)** The double estimate uses the *same* data, split in two. It has half as
many samples for each of its two jobs, so it is noisier per job. Why is it
nevertheless unbiased? Name the property of the split that does the work.

## Q2 — an uncomfortable number

The script prints:

```
        algorithm  left, first 50 ep  left, last 50 ep  visits to B
       Q-learning              83.3%             13.8%        114.7
Double Q-learning              27.0%              7.2%         35.2

max_a Q(B,a) at the end, against a true value of -0.1:
  Q-learning -0.0281, Double Q-learning +0.0546
```

Q-learning is the algorithm with maximization bias, and its final estimate of
$\max_a q^{*}(B,a)$ is **closer to the truth** than Double Q-learning's.

**(a)** Explain the result. The visit column is most of the answer; say exactly
how it produces the reversal.

**(b)** The two algorithms were given the same number of *episodes*. Were they
given the same amount of *data about B*? Design the comparison you would have to
run to make the final estimates comparable, and say what you would fix and what
you would let vary.

**(c)** This is a general trap, not a quirk of this MDP. State it as a rule
about comparing two algorithms whose behaviour changes what they observe, in one
sentence you would be willing to apply to your own results.

## Q3 — two true sentences

From the deep panel:

- the paired difference DQN $-$ Double DQN is $+0.2004 \pm 0.0672$, positive on
  4 of 4 seeds;
- DQN's estimate is above $q^{*}$ on 2 of 4 seeds, and the spread across seeds
  is $0.4973$.

**(a)** Both are correct. Explain why the first has a small standard error and
the second does not, given that they are computed from the same eight numbers.
Your answer should mention what the shared buffer does to the comparison.

**(b)** "DQN overestimates the value function" is the claim in the literature.
Which of the two measurements above supports it, and which does not? If you had
to defend the claim with this data, what would you have to run more of, and how
much more — give a number, using the standard error you measured.

**(c)** Double DQN's estimate is *below* $q^{*}$ on 3 of 4 seeds. Is that a
problem? Say what Double Q-learning promises and what it does not, and whether
an underestimate is a failure of the method or a known part of it.

## Declaration

Who you worked with, and which tools you used and for what.
