"""
Tests for check_actions.py, the action_tools / actions_sandbox keys and the fuzz skip
(fake targets, no LLM, no API calls).
    python tests/test_check_actions.py
"""

import contextlib
import io
import os
import shutil
import sys
import tempfile
import textwrap

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [REPO, os.path.dirname(os.path.abspath(__file__))]
os.chdir(REPO)
os.environ.setdefault("GROQ_API_KEY", "offline-test")

import check_actions  # noqa: E402
import fuzz_tools  # noqa: E402
import run_suite  # noqa: E402
import target_config  # noqa: E402

failures = []


def ok(label, cond, detail=""):
    detail = str(detail).encode("ascii", "backslashreplace").decode() if detail else ""
    print(f"  {'OK  ' if cond else 'FAIL'}  {label}" + (f"  -> {detail}" if detail else ""))
    if not cond:
        failures.append(label)


root = tempfile.mkdtemp()
sys.path.insert(0, root)
open(os.path.join(root, "ca_tools.py"), "w").write(textwrap.dedent('''
    import os
    from langchain_core.tools import tool
    import approval

    SENT = []          # the "real" side effect: every email that went out
    SANDBOX = {}

    @tool
    def send_email(to: str, body: str) -> str:
        """Send an email."""
        SENT.append((SANDBOX.get("dir"), to))
        return f"sent to {to}"

    @tool
    def lookup(q: str) -> str:
        """Read-only."""
        return "ok"

    def good():
        return [lookup, approval.safe_action(send_email)]                  # default approver: deny

    def unprotected():
        return [lookup, send_email]

    def approval_only():
        return [approval.requires_approval(send_email)]                    # asks, but not idempotent

    def auto_configured():
        return [approval.safe_action(send_email, approver="auto")]

    def sandbox(tmp):
        SANDBOX["dir"] = tmp
        def restore():
            SANDBOX.pop("dir", None)
        return restore
'''))
import ca_tools  # noqa: E402


def T(getter, **kw):
    return {"t": {"module": "ca_tools", "function": "good", "tools": f"ca_tools:{getter}",
                  "action_tools": [{"name": "send_email", "test_args": {"to": "test@example.com", "body": "hi"}}],
                  "actions_sandbox": "ca_tools:sandbox", **kw}}


def run(targets, name="t"):
    rep = check_actions.check(name, None, targets)
    return rep, {c: (s, d) for s, c, d in rep.rows}


print("a correctly protected action tool passes every check:")
ca_tools.SENT.clear()
rep, by = run(T("good"))
ok("PASS", not rep.failed, [r for r in rep.rows if r[0] == "FAIL"])
for c in ("protected", "blocked", "reject", "approve", "retry", "new request", "concurrent", "langgraph"):
    ok(f"  {c}: OK", by.get(f"send_email: {c}", ("?",))[0] == "OK", by.get(f"send_email: {c}"))
ok("configured approver reported (deny)", by["send_email: configured"] == ("OK", "approver: deny"))
ok("the real side effect ran exactly 4x (approve, new request, concurrent, langgraph), all in the sandbox",
   len(ca_tools.SENT) == 4 and all(d and d.startswith(tempfile.gettempdir()) for d, _ in ca_tools.SENT),
   ca_tools.SENT)
ok("sandbox restored afterwards", "dir" not in ca_tools.SANDBOX)
ok("read-only tools are not checked", not any(c.startswith("lookup") for c in by))

print("an unprotected action tool fails:")
ca_tools.SENT.clear()
rep, by = run(T("unprotected"))
ok("FAIL: not wrapped, and nothing was called", rep.failed and by["send_email: protected"][0] == "FAIL"
   and "NOT wrapped" in by["send_email: protected"][1] and ca_tools.SENT == [])

print("approval without idempotency fails (the retry check catches it):")
rep, by = run(T("approval_only"))
ok("requires_approval alone is not a safe action -> FAIL protected", rep.failed
   and by["send_email: protected"][0] == "FAIL")

print("an 'auto' approver in production is a warning:")
rep, by = run(T("auto_configured"))
ok("WARN configured: approves everything", by["send_email: configured"][0] == "WARN" and not rep.failed)

print("no sandbox -> warning that the real effect runs:")
ca_tools.SENT.clear()
targets = T("good")
del targets["t"]["actions_sandbox"]
rep, by = run(targets)
ok("WARN sandbox, still checks", by["sandbox"][0] == "WARN" and not rep.failed and len(ca_tools.SENT) == 4)

print("test_args optional (neutral values from the schema):")
targets = T("good", action_tools=["send_email"])
rep, by = run(targets)
ok("names only -> neutral args, still PASS", not rep.failed and by["send_email: approve"][0] == "OK")

print("targets.yaml validation:")
ok("action_tools without tools: -> problem", any("needs `tools:`" in p for p in target_config.validate(
   "x", {"module": "m", "function": "f", "action_tools": ["a"]}, [])))
ok("bad action_tools entry -> problem", any("action_tools" in p for p in target_config.validate(
   "x", {"module": "m", "function": "f", "tools": "m:t", "action_tools": [{"test_args": {}}]}, [])))
ok("bad actions_sandbox -> problem", any("actions_sandbox" in p for p in target_config.validate(
   "x", {"module": "m", "function": "f", "actions_sandbox": "nocolon"}, [])))
ok("target_config.get lists them", target_config.get("t", T("good"))["action_tools"] == [
   {"name": "send_email", "test_args": {"to": "test@example.com", "body": "hi"}}])
rep, by = run({"t": {"module": "ca_tools", "function": "good"}})
ok("a target without action_tools: nothing to check, PASS", not rep.failed and by["action_tools"][0] == "OK")
rep, _ = run({}, "nope")
ok("unknown target -> FAIL", rep.failed)

print("fuzz_tools never calls an action tool:")
ca_tools.SENT.clear()
real = run_suite.load_targets
run_suite.load_targets = lambda: {**real(), **T("good")}
out = io.StringIO()
with contextlib.redirect_stdout(out):
    fuzz_tools.main(["t"])
run_suite.load_targets = real
ok("send_email skipped (listed + safe_action), lookup fuzzed, no email sent",
   "SKIPPED send_email" in out.getvalue() and "lookup:" in out.getvalue() and ca_tools.SENT == [], out.getvalue()[:300])

print("CLI:")
run_suite.load_targets = lambda: {**real(), **T("good")}
with contextlib.redirect_stdout(io.StringIO()) as out:
    code = check_actions.main(["t"])
ok("exit 0 + PASS line", code == 0 and "PASS: 0 failed" in out.getvalue())
run_suite.load_targets = lambda: {**real(), **T("unprotected")}
with contextlib.redirect_stdout(io.StringIO()):
    code = check_actions.main(["t"])
ok("exit 1 on failure", code == 1)
run_suite.load_targets = real

shutil.rmtree(root, ignore_errors=True)
print("\nFAILED: " + ", ".join(failures) if failures else "\nPASS")
sys.exit(1 if failures else 0)
