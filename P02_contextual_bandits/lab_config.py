"""What tells this lab apart from the other eleven.

The autograder itself is identical everywhere, so everything specific to this
lab is here: its title, and the question each group of tests belongs to. A key
must match the prefix of the test names it collects, so "q2" selects every test
called `test_q2*`.
"""

TITLE = "P02 - Contextual bandits and Thompson sampling"

QUESTIONS = {
    "q1": "The Beta posterior: mean, variance, conjugate update",
    "q2": "Acting on a posterior: sampling vs the mean",
    "q3": "Contextual regret",
}

# Modules from the shared library that this lab uses. They are given code, the
# same file in every lab that needs it: read them, they are written to be read.
SHARED = ()
