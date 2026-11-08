"""
Are a target's action tools safe? Universal checks, no LLM, no API calls.

    python check_actions.py <target> [--experiment FILE]

targets.yaml lists the tools that ACT (send, pay, refund, delete, ...):

    my_target:
      tools: my_module:get_tools              # how to get the target's tools
      action_tools: [send_email]              # or [{name: send_email, test_args: {to: "a@b.c"}}]
      actions_sandbox: my_module:sandbox      # optional: fn(tmpdir) -> restore(); points side effects at tmpdir

For every action tool this calls the REAL wrapped tool (approver and action
store swapped in with approval.override, everything in a temp folder) and checks:

  protected        it is wrapped with approval.safe_action
  configured       which approver it uses in production (auto = warning)
  blocked          with the deny approver: nothing happens, the model is told
  reject           an approver says no: nothing happens
  approve          an approver says yes: it acts exactly once
  retry            the same call again (same request): still once, nobody asked again
  new request      a different request id: acts again (the key includes the request)
  concurrent       5 simultaneous identical calls: once
  langgraph        inside a LangGraph run (interrupt + resume): approve -> once, reject -> none

"Acts" is counted from the guard's own log (action_events 'executed'), so this
works for any tool. Without an actions_sandbox the tool's REAL side effect runs
during the approve checks (4 times): point it at test resources.
Exit 1 if anything fails.
"""

import argparse
import glob
import os
import shutil
import sys
import tempfile
import threading

import approval
import idempotent as idem
import run_suite
import target_config
from check_target import Report

CONCURRENT = 5


def _resolve(ref):
    import importlib
    module, _, attr = ref.partition(":")
    return getattr(importlib.import_module(module), attr)


def find_tools(target, entry, names, experiment=None):
    """(tools, experiment_path): the target's tools, from an experiment whose tools include every action tool."""
    getter = _resolve(entry["tools"])
    if not entry.get("accepts_settings"):
        return list(getter()), None
    from experiment_config import load_experiment
    paths = [experiment] if experiment else sorted(p for p in glob.glob(os.path.join("experiments", "*.yaml"))
                                                   if not os.path.basename(p).startswith("draft_"))
    for path in paths:
        try:
            exp = load_experiment(path)
            if exp["target"] != target:
                continue
            tools = list(getter(exp["settings"]))
        except Exception:
            if experiment:
                raise
            continue
        if names <= {getattr(t, "name", getattr(t, "__name__", "")) for t in tools}:
            return tools, path
    raise LookupError(f"no experiment for '{target}' gives the tools {sorted(names)}"
                      + (f" ({experiment})" if experiment else " (pass --experiment FILE)"))


def _neutral_args(t):
    import fuzz_tools
    props, _ = fuzz_tools._schema(t)
    return {a: fuzz_tools.NEUTRAL.get(typ, "test") for a, typ in props.items()}


def _invoke(t, args):
    return t.invoke(args) if hasattr(t, "invoke") else t(**args)


class Asked:
    def __init__(self, answer):
        self.answer, self.n = answer, 0
        self.lock = threading.Lock()

    def decide(self, req):
        with self.lock:
            self.n += 1
        return approval.Decision(self.answer, "check_actions", "")


def _graph_check(t, args, store, answer, thread):
    """Run t inside a minimal LangGraph agent (fake LLM) with the langgraph approver."""
    from typing import Annotated, TypedDict
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, StateGraph
    from langgraph.graph.message import add_messages
    from langgraph.prebuilt import ToolNode

    class S(TypedDict):
        messages: Annotated[list, add_messages]

    name = getattr(t, "name", None)

    def agent(state):
        if not any(isinstance(m, ToolMessage) for m in state["messages"]):
            return {"messages": [AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": "check"}])]}
        return {"messages": [AIMessage(content="done")]}

    g = StateGraph(S)
    g.add_node("agent", agent)
    g.add_node("tools", ToolNode([t]))
    g.set_entry_point("agent")
    g.add_conditional_edges("agent", lambda s: "tools" if s["messages"][-1].tool_calls else END,
                            {"tools": "tools", END: END})
    g.add_edge("tools", "agent")
    app = g.compile(checkpointer=MemorySaver())
    with approval.override(approver="langgraph", store=store):
        r = approval.run_with_approvals(app, {"messages": [HumanMessage("check")]},
                                        lambda req: answer, {"configurable": {"thread_id": thread}})
    return r["messages"][-1].content


def check_tool(t, args, tmp, rep):
    name = getattr(t, "name", None) or getattr(t, "__name__", "tool")
    label = lambda c: f"{name}: {c}"  # noqa: E731
    if not approval.is_safe_action(t):
        rep.fail(label("protected"), "NOT wrapped with approval.safe_action: it would act with no approval and "
                                     "could act twice on a retry")
        return
    rep.ok(label("protected"), "approval.safe_action (approval + idempotency)")
    conf = approval.configured_approver(t)
    if conf in ("auto", "AutoApprover"):
        rep.warn(label("configured"), "approver 'auto' approves EVERYTHING: only for tests")
    else:
        rep.ok(label("configured"), f"approver: {conf}")
    store = idem.ActionStore(os.path.join(tmp, f"check_{name}.db"))
    runs = lambda: store.executions(name)  # noqa: E731

    def call(rid, approver):
        with approval.override(approver=approver, store=store), idem.request_context(rid):
            return str(_invoke(t, args))

    try:
        out = call("check/blocked", "deny")
        (rep.ok if runs() == 0 and "not approved" in out else rep.fail)(
            label("blocked"), "deny -> nothing done, the model is told" if runs() == 0 else f"acted {runs()}x: {out[:80]}")
        out = call("check/reject", Asked(False))
        (rep.ok if runs() == 0 else rep.fail)(label("reject"), "no effect" if runs() == 0 else f"acted {runs()}x")
        yes = Asked(True)
        out = call("check/approve", yes)
        (rep.ok if runs() == 1 else rep.fail)(label("approve"), f"acted exactly once: {out[:70]!r}" if runs() == 1
                                              else f"acted {runs()}x")
        again = call("check/approve", yes)
        good = runs() == 1 and yes.n == 1 and again == out
        (rep.ok if good else rep.fail)(label("retry"), "same request again: stored result, not asked again, still once"
                                       if good else f"acted {runs()}x, asked {yes.n}x")
        call("check/approve-2", Asked(True))
        (rep.ok if runs() == 2 else rep.fail)(label("new request"), "a new request id acts again" if runs() == 2
                                              else f"executions {runs()} (expected 2): is the request id in the key?")
        gate = threading.Event()

        def worker():
            gate.wait()
            call("check/concurrent", "auto")

        ts = [threading.Thread(target=worker) for _ in range(CONCURRENT)]
        [x.start() for x in ts]
        gate.set()
        [x.join() for x in ts]
        (rep.ok if runs() == 3 else rep.fail)(label("concurrent"), f"{CONCURRENT} simultaneous calls -> once"
                                              if runs() == 3 else f"executions {runs() - 2} (expected 1)")
        try:
            lg_store = idem.ActionStore(os.path.join(tmp, f"check_{name}_lg.db"))
            _graph_check(t, args, lg_store, True, "check-lg-yes")
            n_yes = lg_store.executions(name)
            _graph_check(t, args, lg_store, False, "check-lg-no")
            n_no = lg_store.executions(name) - n_yes
            good = n_yes == 1 and n_no == 0
            (rep.ok if good else rep.fail)(label("langgraph"), "interrupt + resume: approve -> once, reject -> none"
                                           if good else f"approve acted {n_yes}x, reject acted {n_no}x")
        except ImportError:
            rep.add("SKIP", label("langgraph"), "LangGraph not installed")
    except Exception as e:  # noqa: BLE001
        rep.fail(label("calls"), f"calling it raised {type(e).__name__}: {e}")


def check(target, experiment=None, targets=None) -> Report:
    rep = Report()
    targets = targets if targets is not None else run_suite.load_targets()
    if not isinstance(targets, dict) or target not in targets:
        rep.fail("targets.yaml", f"no target '{target}'")
        return rep
    entry = targets[target] or {}
    problems = target_config.validate(target, entry, [])
    problems = [p for p in problems if "judges" not in p]
    if problems:
        for p in problems:
            rep.fail("targets.yaml", p)
        return rep
    actions = target_config.action_tools(entry)
    if not actions:
        rep.ok("action_tools", "none listed: nothing acts, nothing to check")
        return rep
    names = {a["name"] for a in actions}
    try:
        tools, path = find_tools(target, entry, names, experiment)
    except Exception as e:  # noqa: BLE001
        rep.fail("tools", f"{type(e).__name__}: {e}")
        return rep
    rep.ok("tools", f"{len(tools)} tools" + (f" from {path}" if path else "") + f"; action tools: {sorted(names)}")
    by_name = {getattr(t, "name", getattr(t, "__name__", "")): t for t in tools}
    tmp = tempfile.mkdtemp(prefix="check_actions_")
    restore = None
    try:
        if entry.get("actions_sandbox"):
            restore = _resolve(entry["actions_sandbox"])(tmp)
            rep.ok("sandbox", f"{entry['actions_sandbox']}: side effects go to a temp folder")
        else:
            rep.warn("sandbox", "no actions_sandbox: the approve checks run the tool's REAL side effect")
        for a in actions:
            t = by_name[a["name"]]
            check_tool(t, a["test_args"] if a["test_args"] is not None else _neutral_args(t), tmp, rep)
    finally:
        if callable(restore):
            restore()
        shutil.rmtree(tmp, ignore_errors=True)
    return rep


def main(argv=None):
    p = argparse.ArgumentParser(description="Universal safety checks for a target's action tools (no LLM).")
    p.add_argument("target")
    p.add_argument("--experiment", help="settings that give the action tools (accepts_settings targets)")
    args = p.parse_args(argv)
    rep = check(args.target, args.experiment)
    print(f"check_actions {args.target}")
    for status, name, detail in rep.rows:
        print(f"  {status:<4}  {name:<28} {detail}".encode("ascii", "backslashreplace").decode())
    n_fail = sum(1 for s, *_ in rep.rows if s == "FAIL")
    n_warn = sum(1 for s, *_ in rep.rows if s == "WARN")
    print(f"\n{'FAIL' if n_fail else 'PASS'}: {n_fail} failed, {n_warn} warnings")
    return 1 if rep.failed else 0


if __name__ == "__main__":
    sys.exit(main())
