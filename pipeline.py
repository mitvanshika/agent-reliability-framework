"""
Runs the full reliability pipeline end to end: both agent suites,
both judge passes, then the regression check. Exits with a non-zero
status code if a statistically significant regression is detected —
so this can be dropped into a CI pipeline (e.g. GitHub Actions) and
fail a build automatically whenever a prompt/model change makes the
agent measurably worse.

Usage:
    python pipeline.py
"""

import sys
import subprocess

PYTHON = sys.executable

STEPS = [
    [PYTHON, "run_suite.py", "strong"],
    [PYTHON, "run_suite.py", "weak"],
    [PYTHON, "run_judges.py", "strong"],
    [PYTHON, "run_judges.py", "weak"],
]


def run_step(cmd):
    print(f"\n=== Running: {' '.join(cmd)} ===")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"Step failed: {' '.join(cmd)}")
        sys.exit(1)


def main():
    for step in STEPS:
        run_step(step)

    print("\n=== Checking for regressions ===")
    result = subprocess.run([PYTHON, "detect_regressions.py"])
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()