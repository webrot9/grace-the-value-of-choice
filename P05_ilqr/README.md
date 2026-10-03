# P05 — LQR, iLQR, and a controller that ignores the noise

**~30 min at home, 2 hours in class, ~30 min for the report.**

A system here is four matrices. That is what lets you check certainty
equivalence with `np.array_equal` instead of "the two curves look close".

By the end you can write the Riccati recursion, linearise a non-linear system
along a trajectory and run the backward pass on the deviations from it, which is
what iLQR does, and say what certainty equivalence does *not* promise.

### The symbols

$$x_{t+1} = \mathbf{F}_x x_t + \mathbf{F}_u u_t + w_t, \qquad
  c(x,u) = x^\top \mathbf{C}_x x + u^\top \mathbf{C}_u u$$

Control textbooks write $A, B, Q, R$. We do not: in this course $Q$ is the
action-value function and $R$ is the reward, and reusing both letters in the one
lecture where both meanings are on the board is how a cohort gets confused.
`book/notation.md` has the full table.

## Part A — at home, before class

Eight `# TODO`s in `rl_lab/control.py`, checked by
eighteen tests.

```bash
pip install -r requirements.txt
python autograder.py
```

See [HOWTO](HOWTO.md) if this is your first lab.

You implement the Riccati step and backward pass, the cost-to-go, the Jacobians,
the rollout, the time-varying backward pass, the iLQR backward pass with its
feedforward term, and iLQR.

Two tests to read first.

`test_q1_scalar_case_matches_the_hand_algebra` is the 1-D Riccati step done by
hand inside the test. If your sign convention for $\mathbf{K}$ is the other one,
you find out in a minute instead of an evening.

`test_q2_linearising_a_linear_system_returns_its_matrices` checks your Jacobians
to $10^{-7}$ on a linear system, where any finite difference is exact. The next
test uses a quadratic, on which central differences are still exact and forward
differences are off by about $\varepsilon$, so only central differences pass.

## Part B — in class

```bash
python experiments.py            # ~5 s at 2000 seeds, writes certainty.png
python experiments.py --seeds 200
```

```
 sigma  gain identical to sigma=0   mean cost  cost - deterministic
  0.00                       True      16.565                 0.000
  0.05                       True      20.325                 3.759
  0.20                       True      75.868                59.302
  0.50                       True     386.143               369.578
```

`identical` is `np.array_equal` over all sixty gain matrices, not a tolerance.
The backward pass is re-run for each $\sigma$ and returns the same floats,
because $\sigma$ is not one of its arguments.

The cost is another matter. It rises by $\sigma^2 \sum_t \mathrm{tr}(\mathbf{P}_t)$,
which has no $x_0$ in it:

```
 sigma  measured extra  std error  sigma^2 sum tr(P)  gap in SE
  0.05           3.759      0.100              3.700       +0.6
  0.20          59.302      0.750             59.207       +0.1
  0.50         369.578      4.208            370.041       -0.1
```

So noise changes the bill and not the policy. That is narrower than "noise does
not matter", and report question 1 is about the difference.

Then the same machinery on a unicycle, which is not linear at all:

```
iLQR on the unicycle: |x_0| = 2.500 -> |x_T| = 0.1304, total cost 34.348
```

## Part C — break it

**Predict first, then run.**

1. Raise `alpha` in the call to `ilqr`. On the unicycle, after 15 iterations:

   ```
   alpha          0.2     0.5     0.7     0.9     1.0     1.5
   |x_T|        0.238   0.131   0.130   0.130   0.130   5.574
   total cost   34.89   34.35   34.35   34.35   34.35 1006.94
   ```

   Every value from 0.5 to 1 reaches the same cost, and at 1.5 the final state
   is further from the origin than the initial one. What goes wrong with a step
   longer than the one the backward pass proposes, and what is a step of at most
   1 protecting? (Not "numerical stability".)
2. Make the noise state-dependent: in `rollout`, use `sigma * np.linalg.norm(x)`.
   Are the gains still the LQR gains? Which hypothesis did you delete?
3. In `ilqr`, linearise **once**, before the loop, and never again. The final
   state comes out as $(1.2\times10^{-5},\, 1.5,\, 0)$ — read the components
   before taking the norm. Print `linearize(unicycle, x0, np.zeros(2))[1]` and
   you will see why in one row of the matrix.

Question 2 is the one about the theorem: certainty equivalence needs *additive,
zero-mean, state-independent* noise, and all three words are load-bearing.

## What to hand in

This is a **long lab**: Part B is a real training run, and you write a report.
Both kinds count the same toward the lab requirement, except that three of your
five accepted labs must be long ones. See [HOWTO](HOWTO.md).

One email, before this lab's deadline, with:

1. **Two figures your code produced**, attached as files:
   - `certainty.png`, from `python experiments.py` as given;
   - the same figure after your change for Part C, question 2, saved under a new
     name so the first one is kept:
     `python experiments.py --out certainty_partc.png`.
2. **The report in the body of the email**: the three questions in `report.md`,
   answered there, at least 150 words in total. A `report.md` attached on its
   own is not counted.

Pass/fail, decided by a program: two images, 150 words in the body, on time.
The address and the subject line are in "How to submit a lab" on Classroom.
