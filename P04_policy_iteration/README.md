# P04 — Policy iteration: monotone improvement, and why it stops

**~30 min at home, then one class session.**

No training and no report this week. The hour goes on Part C.

By the end you can state the improvement step in terms of the advantage
$A^\pi$, and say precisely why policy iteration terminates while value iteration
does not.

## Part A — at home, before class

Five `# TODO`s in `rl_lab/pi.py`, checked by
fourteen tests.

```bash
pip install -r requirements.txt
python autograder.py
```

See [HOWTO](HOWTO.md) if this is your first lab.

You implement the advantage, the improvement step, the loop, and the two
quantities that make its behaviour visible. The exact evaluation and $Q$ from $V$
are given — you wrote them last week.

Read `test_q3_is_monotone_rejects_a_step_that_is_worse_anywhere` first. It feeds
a sequence whose *mean* goes up while one state goes down, and it must be
rejected. If you compared averages, that test is the one that tells you.

## Part B — in class

```bash
python experiments.py            # ~1 s, writes monotonicity.png
```

Left panel: one line per state over the sequence of policies. None of the
twenty-five ever goes down.

Two numbers to stop on.

**States actually improved per step: `[9, 14, 8, 9, 9, 9]` out of 25.** Monotone
improvement does not mean everything gets better every time; it means nothing
gets worse, which is weaker and more useful.

**Smallest change of any state at any step: `-1.14e-17`.** Negative. The theorem
is exact and the arithmetic is not, which is why `is_monotone` takes a tolerance.

Right panel: policy iteration reaches $V^{*}$ exactly and halts after 7 policies;
value iteration needs 37 sweeps to get within $10^{-9}$ and never arrives. Over
40 random gridworlds PI visits between 2 and 8 policies, against a bound of
$4^{25} \approx 10^{15}$.

## Part C — break it

**Predict first, then run.**

1. In `policy_improvement`, drop the tie-breaking rule and always take
   `A.argmax(axis=1)`. Does the loop still terminate? On which MDP does it stop
   terminating, and why is `max_iters` a disguise rather than a fix?
2. Replace the exact evaluation with three sweeps of $\mathcal{B}^\pi$. You have
   just written modified policy iteration. Is it still monotone? Count the
   policies it visits, and say what you bought and what you paid.

Question 2 is the honest answer to "which of the two is better": neither, and the
interesting algorithm is in between.

## What to hand in

This is a **short lab**: Part A is longer, there is no training and no report.
Both kinds count the same toward the lab requirement, except that three of your
five accepted labs must be long ones. See [HOWTO](HOWTO.md).

One email, before this lab's deadline, with:

1. **Two figures your code produced**, attached as files:
   - `monotonicity.png`, from `python experiments.py` as given;
   - the same figure after your change for Part C, question 2, saved under a new
     name so the first one is kept:
     `python experiments.py --out monotonicity_partc.png`.
2. **At least 150 words in the body of the email**: what you predicted for
   Part C, what changed between the two figures, and why. If nothing changed,
   say why.

Pass/fail, decided by a program: two images, 150 words in the body, on time.
The address and the subject line are in "How to submit a lab" on Classroom.
