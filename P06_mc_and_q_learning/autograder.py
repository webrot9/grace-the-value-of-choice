#!/usr/bin/env python3
"""
Local autograder. Run it as often as you like: it never submits anything.

This file is the same in every lab, so there is nothing lab-specific to read
here. What differs between labs — the title and the list of questions — lives in
`lab_config.py`, which is six lines long.

    python autograder.py            # all questions
    python autograder.py -q q2      # just question 2
    python autograder.py -v         # show the failing assertion

The tests check mathematical properties, not your style, and none of them depends
on a random seed or on a training run: if a test fails, your implementation is
wrong, not unlucky.
"""
from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from lab_config import QUESTIONS, TITLE      # noqa: E402
GREEN, RED, GREY, BOLD, END = "\033[92m", "\033[91m", "\033[90m", "\033[1m", "\033[0m"


def run(pattern: str, verbose: bool) -> tuple[int, int]:
    cmd = [sys.executable, "-m", "pytest", "tests/", "-k", pattern,
           "-q", "--no-header", "-p", "no:cacheprovider"]
    if not verbose:
        cmd.append("--tb=no")
    r = subprocess.run(cmd, cwd=pathlib.Path(__file__).parent,
                       capture_output=True, text=True)
    out = r.stdout + r.stderr
    import re
    m = re.search(r"(\d+) passed", out)
    p = int(m.group(1)) if m else 0
    m = re.search(r"(\d+) failed", out)
    f = int(m.group(1)) if m else 0
    if (f or not p) and verbose:
        print(GREY + out + END)
    elif f:
        for line in out.splitlines():
            if line.startswith("FAILED") or "Error" in line:
                print(GREY + "    " + line.strip()[:110] + END)
    return p, f


def check_deps() -> None:
    try:
        import pytest  # noqa: F401
    except ImportError:
        sys.exit(f"{RED}pytest is not installed.{END}  Run:  "
                 f"pip install -r requirements.txt")
    try:
        import numpy  # noqa: F401
    except ImportError:
        sys.exit(f"{RED}numpy is not installed.{END}  Run:  "
                 f"pip install -r requirements.txt")


def main() -> int:
    check_deps()
    ap = argparse.ArgumentParser()
    ap.add_argument("-q", "--question", choices=sorted(QUESTIONS))
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    todo = [args.question] if args.question else sorted(QUESTIONS)
    print(f"\n{BOLD}{TITLE}{END}")
    tot_p = tot_f = 0
    for q in todo:
        title, pattern = QUESTIONS[q], f"test_{q}"
        p, f = run(pattern, args.verbose)
        tot_p += p
        tot_f += f
        mark = f"{GREEN}PASS{END}" if f == 0 and p > 0 else f"{RED}FAIL{END}"
        print(f"  {q}  {mark}  {p}/{p+f} checks   {title}")
    print()
    if tot_f == 0 and tot_p > 0:
        print(f"{GREEN}{BOLD}All checks pass.{END} Part A done. "
              f"Now run:  python experiments.py\n")
        return 0
    print(f"{RED}{tot_f} check(s) failing.{END} Re-run with -v to see why, "
          f"or with -q q1 to focus.\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
