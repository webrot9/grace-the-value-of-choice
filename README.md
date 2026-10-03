# Grace: The Value of Choice — exercises

The exercises of the book *Grace: The Value of Choice* (Roberto Capobianco,
2026). Each chapter that ends with something to build has a folder here: a
skeleton to fill in, the tests that check it, and the script that runs the
experiment the chapter talks about.

| chapter | folder |
|---|---|
| 2. Multi-armed bandits and regret | [`P01_bandits`](P01_bandits/) |
| 3. Contextual and Bayesian bandits | [`P02_contextual_bandits`](P02_contextual_bandits/) |
| 4. Markov decision processes | [`P03_value_iteration`](P03_value_iteration/) |
| 5. Value iteration | [`P03_value_iteration`](P03_value_iteration/) |
| 6. Policy iteration | [`P04_policy_iteration`](P04_policy_iteration/) |
| 8. LQR, iLQR, MPC | [`P05_ilqr`](P05_ilqr/) |
| 10. Monte Carlo and temporal difference | [`P06_mc_and_q_learning`](P06_mc_and_q_learning/) |
| 11. n-step methods and eligibility traces | [`P07_nstep_and_traces`](P07_nstep_and_traces/) |
| 12. Linear value function approximation | [`P08_deadly_triad`](P08_deadly_triad/) |
| 13. Off-policy learning | [`P08_deadly_triad`](P08_deadly_triad/) |
| 14. Deep Q-learning | [`P09_ddqn`](P09_ddqn/) |
| 15. Continuous control | [`P12_sac_entropy`](P12_sac_entropy/) |
| 18. Planning and search | [`P10_mcts`](P10_mcts/) |
| 20. Policy search | [`P11_reinforce`](P11_reinforce/) |
| 21. REINFORCE, baselines, actor-critic | [`P11_reinforce`](P11_reinforce/) |

Some chapters share a folder: 4 and 5, 12 and 13, 20 and 21. The other chapters reuse code from these folders or need only a
pencil, and their Trial section says which.

## Using a folder

```bash
cd P01_bandits
pip install -r requirements.txt
python autograder.py        # the tests, run on your code
python experiments.py       # the experiment, once the tests pass
```

Each folder's README has three parts. In Part A you write the code and the tests
check it. Part B runs the experiment, and Part C removes one hypothesis on
purpose to watch what breaks. Nothing needs a GPU. Python and NumPy are enough
until Chapter 14, and from there on you also need PyTorch.

These are the packages of the course the book was written for, at Sapienza
University of Rome, so they also talk about classes, deadlines and how to hand a
lab in. Those parts are about the course, and a reader of the book can skip
them. The reference solutions are not here.

The code is under the MIT license ([LICENSE](LICENSE)). The book is not.

## Corrections

[ERRATA.md](ERRATA.md) lists the corrections made to the printed text. The
copyright page of each copy says which revision it was printed from. To report a
new one, open an issue on this repository.

https://github.com/webrot9/grace-the-value-of-choice
