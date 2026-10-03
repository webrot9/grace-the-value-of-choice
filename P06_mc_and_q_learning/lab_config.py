"""What tells this lab apart from the other eleven.

The autograder itself is identical everywhere, so everything specific to this
lab is here: its title, and the question each group of tests belongs to. A key
must match the prefix of the test names it collects, so "q2" selects every test
called `test_q2*`.
"""

TITLE = "P06 - Monte Carlo, TD, and the bias-variance trade-off"

QUESTIONS = {
    "q1": "Returns, first visits, Monte Carlo evaluation",
    "q2": "TD(0), and the bias-variance decomposition",
    "q3": "Control: epsilon-greedy, MC control, Q-learning",
}

# Modules from the shared library that this lab uses. They are given code, the
# same file in every lab that needs it: read them, they are written to be read.
SHARED = ('gridworld', 'exact')
