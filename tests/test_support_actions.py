"""
The support bot's issue_refund action (support_actions.py, actions: enabled). LLM faked, no API calls.
    python tests/test_support_actions.py
"""

import contextlib
import io
import os
import shutil
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)
os.environ.setdefault("GROQ_API_KEY", "offline-test")

root = tempfile.mkdtemp()
os.environ["SUPPORT_REFUNDS_DB"] = os.path.join(root, "refunds.db")
os.environ["ACTIONS_DB"] = os.path.join(root, "actions.db")

from langchain_core.messages import AIMessage, ToolMessage  # noqa: E402

import agent_configurable as ac  # noqa: E402
import approval as ap  # noqa: E402
import check_actions  # noqa: E402
import fuzz_tools  # noqa: E402
import idempotent as idem  # noqa: E402
import support_actions  # noqa: E402
import test_agent  # noqa: E402
from experiment_config import config_hash, load_experiment  # noqa: E402

failures = []


def ok(label, cond, detail=""):
    detail = str(detail).encode("ascii", "backslashreplace").decode() if detail else ""
    print(f"  {'OK  ' if cond else 'FAIL'}  {label}" + (f"  -> {detail}" if detail else ""))
    if not cond:
        failures.append(label)


BASE = load_experiment(os.path.join("experiments", "baseline.yaml"))
EXP9 = load_experiment(os.path.join("experiments", "exp_09_actions.yaml"))


class FakeLLM:
    """Calls issue_refund(order) `times` times (one call per turn), then answers with the tool results."""
    def __init__(self, order="A1001", times=1):
        self.order, self.times, self.bound = order, times, None

    def bind_tools(self, tools):
        self.bound = tools
        return self

    def invoke(self, messages):
        done = [m for m in messages if isinstance(m, ToolMessage)]
        if len(done) < self.times:
            return AIMessage(content="", tool_calls=[{"name": "issue_refund", "args": {"order_id": self.order},
                                                      "id": f"c{len(done)}"}])
        return AIMessage(content=" | ".join(m.content for m in done))


def run(settings, llm, approver=None, rid=None):
    ac._graph_cache.clear()
    ac._make_llm = lambda model, temperature: llm
    ctx = ap.override(approver=approver) if approver is not None else contextlib.nullcontext()
    rctx = idem.request_context(rid) if rid else contextlib.nullcontext()
    with ctx, rctx:
        return ac.run_agent("Please refund my order", settings)


print("baseline untouched:")
ok("baseline settings have no actions / approver keys", "actions" not in BASE["settings"]
   and "approver" not in BASE["settings"])
ok("baseline config_hash unchanged", config_hash(BASE) == config_hash({"target": "configurable",
                                                                       "settings": BASE["settings"]}))
ok("defaults: actions off, approver deny", ac.AgentSettings(**BASE["settings"]).actions == "off"
   and ac.AgentSettings(**BASE["settings"]).approver == "deny")
ok("baseline tools: the same 4, no issue_refund", [t.name for t in ac.build_tools(BASE["settings"])]
   == [t.name for t in test_agent.tools] == ["order_lookup", "refund_calculator", "faq_search", "escalate_to_human"])
ok("test_agent.py has no issue_refund", not hasattr(test_agent, "issue_refund"))

print("exp_09_actions:")
ok("exactly one change vs baseline: actions enabled", EXP9["changes_vs_parent"] == ["settings.actions: (unset) -> enabled"],
   EXP9["changes_vs_parent"])
tools = {t.name: t for t in ac.build_tools(EXP9["settings"])}
ok("5 tools, issue_refund is a safe action with approver deny", list(tools)[-1] == "issue_refund"
   and ap.is_safe_action(tools["issue_refund"]) and ap.configured_approver(tools["issue_refund"]) == "deny")
ok("its description comes from its own docstring (docstrings/*.yaml untouched)",
   "needs a human's approval" in " ".join(tools["issue_refund"].description.split()))
try:
    ac.validate_settings({**EXP9["settings"], "approver": "auto"})
    ok("approver 'auto' is not allowed as a setting", False)
except ac.SettingsError:
    ok("approver 'auto' is not allowed as a setting (tests only)", True)

print("the agent with actions: enabled (fake LLM):")
out = run(EXP9["settings"], FakeLLM("A1001"), rid="t/deny")
ok("default deny: no refund, the agent is told", support_actions.ledger() == [] and
   "not approved by deny" in out["answer"], out["answer"])

yes = []
out = run(EXP9["settings"], FakeLLM("A1001"), approver=lambda r: yes.append(r) or True, rid="t/yes")
ok("approved: one refund of $89.99 in the ledger", [(r["order_id"], r["amount"]) for r in support_actions.ledger()]
   == [("A1001", 89.99)] and "Refund of $89.99 issued for order A1001" in out["answer"], out["answer"])
ok("the approver saw a useful description", yes and yes[0].description
   == "Refund order A1001: $89.99, 12 days since purchase", yes and yes[0].description)

out = run(EXP9["settings"], FakeLLM("A1001", times=2), approver=lambda r: True, rid="t/yes")
ok("same request again, and the model calls it twice: still ONE refund", len(support_actions.ledger()) == 1, out["answer"])

out = run(EXP9["settings"], FakeLLM("A1002"), approver=lambda r: True, rid="t/old")
ok("outside 30 days, approved anyway: the tool itself refuses, no refund",
   len(support_actions.ledger()) == 1 and "outside the 30-day refund window" in out["answer"], out["answer"])
out = run(EXP9["settings"], FakeLLM("Z9999"), approver=lambda r: True, rid="t/none")
ok("unknown order: refuses, no refund", len(support_actions.ledger()) == 1 and "No order found" in out["answer"])

print("a graph re-run after an API error does not refund twice:")


class FlakyLLM(FakeLLM):
    """Like FakeLLM, but the first final answer fails with a rate limit (so run_agent re-runs the graph)."""
    failed = False

    def invoke(self, messages):
        msg = super().invoke(messages)
        if not msg.tool_calls and not FlakyLLM.failed:
            FlakyLLM.failed = True
            import groq
            import httpx
            raise groq.RateLimitError("tokens per minute", response=httpx.Response(
                429, request=httpx.Request("POST", "https://x")), body=None)
        return msg


ac.time.sleep = lambda s: None
before = len(support_actions.ledger())
asked = []
out = run(EXP9["settings"], FlakyLLM("A1003"), approver=lambda r: asked.append(1) or True)
ok("run_agent retried (no request id given: it made one for the whole question)", out["retries_needed"] == 1
   and out["error"] is None, out)
ok("...and A1003 was refunded once, approver asked once", len(support_actions.ledger()) == before + 1
   and len(asked) == 1, (len(support_actions.ledger()) - before, len(asked)))

print("with tool_validation: strict:")
asked = []
strict = {**EXP9["settings"], "tool_validation": "strict"}
out = run(strict, FakeLLM("refund me"), approver=lambda r: asked.append(1) or True, rid="t/strict")
ok("an invalid order id is stopped by validation BEFORE anyone is asked", asked == []
   and "invalid input for issue_refund" in out["answer"], out["answer"])
ok("still a safe action under the validation wrapper", ap.is_safe_action(
   {t.name: t for t in ac.build_tools(strict)}["issue_refund"]))

print("check_actions + fuzz on the real target:")
with contextlib.redirect_stdout(io.StringIO()) as o:
    code = check_actions.main(["configurable"])
ok("check_actions configurable: PASS, every check OK", code == 0 and "PASS: 0 failed, 0 warnings" in o.getvalue(),
   o.getvalue()[-400:])
ok("its checks did not touch this test's ledger (sandboxed)", len(support_actions.ledger()) == before + 1)
with contextlib.redirect_stdout(io.StringIO()) as o:
    fuzz_tools.main(["configurable", "--experiment", os.path.join("experiments", "exp_09_actions.yaml")])
ok("fuzz_tools on exp_09: issue_refund SKIPPED, no refund from junk input",
   "SKIPPED issue_refund" in o.getvalue() and len(support_actions.ledger()) == before + 1)
ok("nothing written to the repo's real refunds.db / actions.db", not os.path.exists(os.path.join(REPO, "refunds.db"))
   and not os.path.exists(os.path.join(REPO, "actions.db")))

shutil.rmtree(root, ignore_errors=True)
print("\nFAILED: " + ", ".join(failures) if failures else "\nPASS")
sys.exit(1 if failures else 0)
