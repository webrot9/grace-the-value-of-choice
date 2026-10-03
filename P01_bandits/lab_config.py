"""What tells this lab apart from the other eleven.

The autograder itself is identical everywhere, so everything specific to this
lab is here: its title, and the question each group of tests belongs to. A key
must match the prefix of the test names it collects, so "q2" selects every test
called `test_q2*`.
"""

TITLE = "P01 - Bandits and regret"

QUESTIONS = {
    "q1": "Confidence radius (Hoeffding)",
    "q2": "Action selection: greedy, UCB, eps-greedy, Thompson",
    "q3": "Pseudo-regret",
}

# Modules from the shared library that this lab uses. They are given code, the
# same file in every lab that needs it: read them, they are written to be read.
SHARED = ()
