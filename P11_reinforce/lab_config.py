"""What tells this lab apart from the other eleven.

The autograder itself is identical everywhere, so everything specific to this
lab is here: its title, and the question each group of tests belongs to. A key
must match the prefix of the test names it collects, so "q2" selects every test
called `test_q2*`.
"""

TITLE = "P11 - REINFORCE and the baseline"

QUESTIONS = {
    "q1": "The softmax policy and its exact gradient",
    "q2": "Unbiasedness, and the variance of the estimator",
    "q3": "Over episodes: REINFORCE with a state baseline",
}

# Modules from the shared library that this lab uses. They are given code, the
# same file in every lab that needs it: read them, they are written to be read.
SHARED = ('gridworld',)
