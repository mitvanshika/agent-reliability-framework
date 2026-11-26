"""
Run every offline test file (tests/test_*.py), each in its own process, and print one line per file.
No API calls: these are the same tests GitHub runs on every push (.github/workflows/tests.yml).

    python tests/run_all.py              # all of them
    python tests/run_all.py tour store   # only files whose name contains "tour" or "store"
    python tests/run_all.py -v           # also print the FAIL lines of a failed file

Exit code 0 only if every file passed, so CI can use it directly.
"""

import glob
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
TIMEOUT_S = 600   # per file; nothing here should come close


def run_one(path):
    """(passed, seconds, last useful line, FAIL lines) for one test file."""
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    env.setdefault("GROQ_API_KEY", "offline-test")   # never used for a real call; some modules want it set
    t0 = time.perf_counter()
    try:
        out = subprocess.run([sys.executable, path], cwd=REPO, env=env, capture_output=True, text=True,
                             encoding="utf-8", errors="replace", timeout=TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return False, time.perf_counter() - t0, f"timed out after {TIMEOUT_S}s", []
    secs = time.perf_counter() - t0
    # The verdict is on stdout; a test that only printed warnings to stderr still passed or failed by its exit code.
    text = out.stdout.strip() or out.stderr.strip()
    lines = text.splitlines()
    last = lines[-1].strip() if lines else ""
    fails = [ln.strip() for ln in out.stdout.splitlines() if ln.strip().startswith("FAIL")]
    err = [ln.rstrip() for ln in out.stderr.strip().splitlines()]
    if out.returncode != 0 and err:
        # a crash: the end of the traceback says where and why (the last line is the error itself)
        fails += ["--- error ---"] + err[-6:]
    return out.returncode == 0, secs, last, fails


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    verbose = "-v" in argv
    words = [a for a in argv if a != "-v"]
    files = sorted(glob.glob(os.path.join(HERE, "test_*.py")))
    if words:
        files = [f for f in files if any(w in os.path.basename(f) for w in words)]
    if not files:
        print("No test files matched.")
        return 1
    print(f"Running {len(files)} test file(s) with {sys.executable}\n")
    failed = []
    t0 = time.perf_counter()
    for f in files:
        name = os.path.basename(f)
        passed, secs, last, fails = run_one(f)
        print(f"  {'PASS' if passed else 'FAIL'}  {name:<34} {secs:6.1f}s" + ("" if passed else f"  {last[:90]}"))
        if not passed:
            failed.append(name)
            if verbose:
                for ln in fails[:15]:
                    print(f"          {ln[:150]}")
    print(f"\nPASS: {len(files) - len(failed)} / {len(files)} files in {time.perf_counter() - t0:.0f}s")
    if failed:
        print("FAILED: " + ", ".join(failed))
        print("See why: python tests/<file>.py   (or: python tests/run_all.py -v)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
