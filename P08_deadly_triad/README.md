# P08 — Baird's counterexample: three ingredients, and which to drop

**~30 min at home, then one class session.**

No training and no report this week. The hour goes on Part C.

Seven states, every reward zero, $V^\pi(s) = 0$ everywhere, and
$\mathbf{w} = \mathbf{0}$ represents it exactly. There is nothing to learn.
Semi-gradient TD(0) runs away from it anyway, to a value error of $4\times10^6$.

By the end you can name the three ingredients of the deadly triad as three
objects in the code — a feature matrix, a state distribution, a bootstrap target
— and point at the eigenvalue that causes the divergence.

## Part A — at home, before class

Five `# TODO`s in `rl_lab/triad.py`, checked by
nineteen tests.

```bash
pip install -r requirements.txt
python autograder.py
```

See [HOWTO](HOWTO.md) if this is your first lab.

You implement the policy transition matrix, the stationary distribution, the
expected semi-gradient update, the loop, and the weighted value error.

Nothing is sampled: `expected_update` is the exact expectation of the TD update,
so the whole lab is deterministic linear algebra.

Read `test_q2_the_true_solution_is_a_fixed_point` first. It checks that
$\mathbf{w} = \mathbf{0}$ is left alone by the update — it has to be. Getting
that green and then watching Part B diverge is the session in two minutes.

## Part B — in class

```bash
python experiments.py            # ~1 s, writes triad.png
python experiments.py --gamma 0.99
```

### Four runs, one argument apart

```
 configuration            missing leg   VE at 0    VE at end  min Re(eig A)
     all three -- (the counterexample)    5.3184    4.139e+06       -0.02143
     on-policy             off-policy     5.3184    5.534e-14       +0.00000
       tabular function approximation     5.3184     0.009462       +0.01429
  no bootstrap          bootstrapping     5.3184    1.574e-14            n/a
```

Every row starts from the same predictions, uses $\alpha = 0.05$ and 10000
updates. The four are the same function call with one argument changed.

Note what "off-policy" turns out to be once written down: **the state
distribution comes from one policy and the bootstrap target from another.** Not a
property of an algorithm — a disagreement between two objects.

### The eigenvalue

The update is exactly
$\mathbf{w} \leftarrow (\mathbf{I} - \alpha \mathbf{A})\mathbf{w}$ with
$\mathbf{A} = \Phi^\top D(\mathbf{I} - \gamma P^\pi)\Phi$:

```
gamma |  off-policy: min Re  how many < 0 |   on-policy: min Re  how many < 0
 0.80 |           -0.000000             0 |            0.000000             0
 0.88 |           -0.000000             0 |           -0.000000             0
 0.90 |           -0.021429             2 |            0.000000             0
 0.99 |           -0.239250             2 |            0.000000             0
```

$\mathbf{A}$ is singular for every $\gamma$, so one eigenvalue is always 0; the
column that matters is how many are strictly negative. On-policy: none, ever —
that is Tsitsiklis and Van Roy's theorem as a column of zeros. Off-policy: two,
but **only above $\gamma = 0.882353$**, which the script finds by bisection.
Below that, all three ingredients are present and nothing diverges. The triad is
necessary, not sufficient.

### Why $\gamma = 0.9$ and not Baird's 0.99

At 0.99 the divergence is stronger ($-0.239$ against $-0.021$) and the figure is
useless: the tabular ablation's slowest mode decays at $7\times10^{-5}$ per step
and needs $10^5$ updates, by which point the diverging run has passed $10^{300}$
and overflowed. No horizon in double precision shows both. Run `--gamma 0.99`
and look at the tabular row.

## Part C — break it

**Predict first, then run.**

1. Set `--gamma 0.88` and `--gamma 0.89` and confirm the behaviour flips where
   the spectrum says it should ($15/17$). Why does a threshold in $\gamma$ exist
   at all, given that all three ingredients are present on both sides?
2. Remove only the *shared* weight: use $\Phi$ without its last column. Does it
   still diverge? What does that say about which part of "function
   approximation" is dangerous?
3. Make the target policy take `SOLID` with probability $p$ and sweep $p$ from
   $1/7$ to 1. Where does the smallest eigenvalue cross zero, and is the crossing
   sharp or gradual?

Spend the time on question 2. The usual telling blames function approximation;
the dangerous part is specifically the weight that every state shares.

## Sources

Baird, L. (1995), "Residual Algorithms: Reinforcement Learning with Function
Approximation", *ICML 1995*, pp. 30-37. The star and the initial weights follow
Sutton & Barto, 2nd ed., section 11.2 and figures 11.1-11.2.

## What to hand in

This is a **short lab**: Part A is longer, there is no training and no report.
Both kinds count the same toward the lab requirement, except that three of your
five accepted labs must be long ones. See [HOWTO](HOWTO.md).

One email, before this lab's deadline, with:

1. **Two figures your code produced**, attached as files:
   - `triad.png`, from `python experiments.py` as given;
   - the same figure after your change for Part C, question 2, saved under a new
     name so the first one is kept:
     `python experiments.py --out triad_partc.png`.
2. **At least 150 words in the body of the email**: what you predicted for
   Part C, what changed between the two figures, and why. If nothing changed,
   say why.

Pass/fail, decided by a program: two images, 150 words in the body, on time.
The address and the subject line are in "How to submit a lab" on Classroom.
