# The labs: how they work

Read this once. Each lab's README then only tells you what is specific to that
lab.

## Setup

Once, per lab:

```bash
cd P03_value_iteration
pip install -r requirements.txt
```

Python 3.11, numpy and matplotlib. Two labs also need `torch`; their
`requirements.txt` says so. Everything runs on a laptop CPU or on free Colab —
no GPU is needed anywhere in this course.

## The three parts

Every lab has the same three parts.

**Part A — at home, before class, about 30 minutes.** You fill in the functions
marked `# TODO` in `rl_lab/`. A local autograder tells you whether they are
right:

```bash
python autograder.py
python autograder.py -q q2      # only question 2
python autograder.py -v         # show the failing assertion
```

Run it as often as you like. It submits nothing and records nothing.

**Part B — in class, with the teacher.** You run `experiments.py`, which
produces one figure and prints a handful of numbers. The README tells you which
numbers to look at and why.

**Part C — in class, 20 minutes.** Two or three changes of one line each, to
make something break. Every time, say what will happen before you run it.

## Part A must be green before class

Part B is built on Part A. When your agent does not learn you want to already
know that your Bellman backup is right, so that there is one thing to debug and
not two.

The autograder tests mathematical properties, not style. Nothing in Part A
depends on a random seed or on how well something trained, so a red test means
your code is wrong, never that you were unlucky.

## Long labs and short labs

Six of the twelve labs ask for a report: labs 1, 3, 5, 6, 9 and 11, the long
ones. The other six ask for no report: labs 2, 4, 7, 8, 10 and 12, the short
ones. The difference is in Part B and in what you write, not in the time: every
session is two hours.

|                 | long lab                          | short lab                   |
|-----------------|-----------------------------------|-----------------------------|
| Part A          | shorter                           | longer                      |
| Part B          | a real training run               | no training                 |
| in the email    | two figures and the report        | two figures and 150 words   |
| lab requirement | counts; three of your five must be long ones | counts           |

The report template is `report.md` in the lab folder: three questions, about
five lines each.

## What you hand in

One email per lab, before its deadline:

1. **Two figures your code produced**, attached as files: the one `python
   experiments.py` saves as given, and the same after your Part C change, saved
   with `--out` under a new name so the first one is kept. Each lab's README
   says which Part C question and gives the command.
2. **At least 150 words in the body of the email.** For a long lab that is the
   report: the three questions in `report.md`, answered in the body. A
   `report.md` attached on its own is not counted. For a short lab, say what you
   predicted for Part C, what changed between the two figures, and why.

It is pass/fail, decided by a program: two images, 150 words in the body, before
the deadline. There are no marks for length and none for how a plot looks. The
figures and the numbers must come from your own code.

## When you hand in

You send a lab by email. The address, the subject line and what the email must
contain are in "How to submit a lab" on the course Classroom; the deadline for
each lab is in that lab's own item there.

The shape is the same every week: the package is posted on Friday, the session
is the Tuesday after, the deadline is the Monday after that at 20:00. Before the
deadline you can send as many times as you want and one accepted submission is
enough. After it nothing is accepted.

To sit the exam you need five accepted labs out of twelve: at least three long
ones (1, 3, 5, 6, 9 and 11) and at least two from lab 7 on. Every reply from the
mailbox shows where you stand.

## Two house rules for the reports

They apply to every lab, and to anything you publish later.

1. **When you compare two settings, change one thing.** Same seeds, same
   budget, same everything else, and say in the report that you did.
2. **Report a spread, not a single run.** Three seeds and a range, not one
   number from one seed.

## When you are stuck

- Read the test that is failing. They are named for what they check, and most
  of them say in the docstring what the usual bug is.
- `python autograder.py -v` prints the assertion that failed.
- The reference solutions are not in the package. Ask in the session, or write
  to your teacher.
