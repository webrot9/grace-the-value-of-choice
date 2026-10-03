# P09 — Maximization bias: DQN, Double DQN, and an estimator that lies

**~45 min at home, 2 hours in class, ~30 min for the report.**

The theorem is not about neural networks: it is about taking a maximum over noisy
numbers. You meet it three times in this lab, at three scales.

By the end you can produce the bias with no reinforcement learning in the room at
all, write both targets, and tell a paired comparison from an unpaired one on a
case where they disagree.

## Part A — at home, before class

Ten `# TODO`s in `rl_lab/deep.py` and `rl_lab/maxbias.py`, checked by
thirty-two tests.

```bash
pip install -r requirements.txt
python autograder.py
```

See [HOWTO](HOWTO.md) if this is your first lab.

Six TODOs in `rl_lab/maxbias.py` (numpy) and four in `rl_lab/deep.py` (torch).

Colab already has torch. Locally, `pip install torch` pulls a large wheel —
start it before you start reading. Without torch the tests in `tests/test_deep.py`
skip rather than fail, so questions 1-3 stay usable, but the lab is not finished
until 4 and 5 run.

Two tests to read first.

`test_q4_ddqn_with_one_network_is_dqn` passes the same tensor as both arguments
and asserts you get `dqn_targets` back exactly. The whole difference between the
two algorithms is which estimates go into the formula, not the formula.

`test_q4_targets_carry_no_gradient`. A target is data. If it still requires grad
you are minimising something else, and nothing in the output will tell you.

## Part B — in class

```bash
python experiments.py            # ~50 s, writes maxbias.png
python experiments.py --quick    # tabular panels only, ~10 s
```

### 1. The bias, with nothing else in the picture

Ten actions, all worth exactly $-0.1$, each estimated without bias. Their
maximum is not:

```
samples per action   max of estimates   double estimate
                 5             0.5892           -0.0957
                20             0.2438           -0.0995
               160             0.0218           -0.0993
```

No environment, no bootstrapping, no policy. Splitting the same samples in two —
one half to choose, one half to price — removes the bias at no cost in data.

The bias is $0.6892$ at 5 samples and $0.1218$ at 160: exactly $1/\sqrt{n}$,
because it is the standard error times $\mathbb{E}[\max_i Z_i] = 1.5388$ for ten
standard normals.

### 2. The same bias, choosing the wrong action

Sutton & Barto's two-state MDP. With $\epsilon = 0.1$ an agent that knew the
values would take `left` 5% of the time.

```
        algorithm  left, first 50 ep  left, last 50 ep  visits to B
       Q-learning              83.3%             13.8%        114.7
Double Q-learning              27.0%              7.2%         35.2
```

Then the line worth arguing about:

```
max_a Q(B,a) at the end, against a true value of -0.1:
  Q-learning -0.0281, Double Q-learning +0.0546
```

Q-learning's estimate of the thing it is supposed to overestimate ends up
**closer to the truth**. Report question 2 is why, and the visit column is most
of the answer.

### 3. The same bias, in a network

A corridor whose $q^{*}$ we compute exactly, with zero-mean reward noise. Both
algorithms are fitted on the **same buffer**, seed by seed:

```
 seed        DQN     vs q*   Double DQN     vs q*   DQN - DDQN
    0     0.8920   +0.2306       0.5666   -0.0948      +0.3253
    2     1.2892   +0.6278       1.0337   +0.3723      +0.2555
    3     0.3146   -0.3469       0.3025   -0.3590      +0.0121

paired difference: +0.2004 +- 0.0672, positive on 4 of 4 seeds.
DQN alone is above q* on 2 of 4 seeds, spread 0.4973.
```

Both sentences are true and they are not the same claim. Report question 3.

Neither fit has converged at 3000 steps, and in the right panel both median
curves are still climbing when the fit stops, so the paired difference compares
the two targets at the same point of the same fit rather than where each would
settle.

## Part C — break it

**Predict first, then run.**

1. Set `REWARD_NOISE = 0.0` and re-run the deep panel. Predict the paired
   difference first, then say in one sentence what the bias is proportional to.
2. In `run_bias_mdp`, try `n_b_actions=2` and `n_b_actions=50`. Plot the early
   `left` fraction against the number of actions. Is the growth linear? What is
   the expectation of the maximum of $k$ standard normals doing here?
3. Update *both* tables on every step instead of one at random. Predict what
   happens to the bias, then explain it in terms of the independence the double
   estimator needs.

Question 3 matters: Double Q-learning is not "two tables", it is "two
*independent* tables", and it is easy to write code with the first and not the
second.

## Sources

van Hasselt, H., Guez, A., Silver, D. (2016), "Deep Reinforcement Learning with
Double Q-learning", *AAAI 2016*, pp. 2094-2100. van Hasselt, H. (2010), "Double
Q-learning", *NeurIPS 2010*, pp. 2613-2621. The two-state MDP is example 6.7 of
Sutton & Barto, 2nd ed.; the overestimation argument is Thrun & Schwartz (1993).

## What to hand in

This is a **long lab**: Part B is a real training run, and you write a report.
Both kinds count the same toward the lab requirement, except that three of your
five accepted labs must be long ones. See [HOWTO](HOWTO.md).

One email, before this lab's deadline, with:

1. **Two figures your code produced**, attached as files:
   - `maxbias.png`, from `python experiments.py` as given;
   - the same figure after your change for Part C, question 3, saved under a new
     name so the first one is kept:
     `python experiments.py --out maxbias_partc.png`.
2. **The report in the body of the email**: the three questions in `report.md`,
   answered there, at least 150 words in total. A `report.md` attached on its
   own is not counted.

Pass/fail, decided by a program: two images, 150 words in the body, on time.
The address and the subject line are in "How to submit a lab" on Classroom.
