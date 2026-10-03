"""What tells this lab apart from the other eleven.

The autograder itself is identical everywhere, so everything specific to this
lab is here: its title, and the question each group of tests belongs to. A key
must match the prefix of the test names it collects, so "q2" selects every test
called `test_q2*`.
"""

TITLE = "P09 - Maximization bias, DQN and Double DQN"

QUESTIONS = {
    "q1": "The bias, with no reinforcement learning in it",
    "q2": "The Q-learning and Double Q-learning targets",
    "q3": "Inside the algorithm: the two-state MDP",
    "q4": "The same two targets, on a batch of tensors",
    "q5": "Ground truth, and fitting a network to it",
}

# Modules from the shared library that this lab uses. They are given code, the
# same file in every lab that needs it: read them, they are written to be read.
SHARED = ()
