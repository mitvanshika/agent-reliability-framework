"""
The fix suggester's agent: for ONE problem, a model investigates with read-only tools
(list_files, read_file, search_code) and ends by calling submit_suggestion. It never edits,
runs or applies anything; the suggestion is checked against the current code (fits / doesn't
fit) and shown to a human, who decides.

    suggest_one(problem, sandbox, client, model, system_info, max_steps=8, max_tokens=30_000, on_step=None)
        -> {"status": "suggested" | "no_fix" | "stopped", "suggestion": {...} | None,
            "reason", "steps", "tokens", "files_read", "patch_check"}

Budgets per problem (the owner's limits, like max_seconds for a target): at most max_steps
model calls and about max_tokens tokens. From the last step, or 80% of the tokens, the model is
made to submit what it has, so a budget never ends with nothing to show (one large call can
take it a little past max_tokens). To keep calls small (the provider's tokens-per-minute limit
makes big calls slow), reads are 80 lines at most and only the last 3 tool results stay in full.
One fixed model and temperature 0, so a re-run gives comparable suggestions.
"""

import json
import os

from suggest.tools import Refused, check_patch

DEFAULT_MODEL = "openai/gpt-oss-120b"
TOOL_OUTPUT_CHARS = 3000
KEEP_RESULTS = 3          # tool results kept in full; older ones shrink to their first line

SYSTEM = """You help the owner of an AI system fix ONE problem that an evaluation framework found.
You can only READ code (tools: list_files, read_file, search_code). You never edit or run anything:
a human reads your suggestion and decides.

Where to look: the fix is almost always in THE SYSTEM'S OWN CODE or its SETTINGS. The framework's
folder is measurement code (how the system is run and scored): search it only when the problem is
in how the system is connected (e.g. its adapter crashed), never to fix a quality drop.

How to work:
1. Read the problem and its evidence (the framework's own measurements, below).
2. If the cause says "the change being tested", start there, often without reading code: compare
   found_only_in_baseline with the question and the answer_key. If a passage the baseline found
   holds a fact the answer key needs, the change cut it: suggest undoing or softening that setting
   (change_type "setting"), quoting the passage id and the fact as evidence.
3. Otherwise find the code involved in the system's folder: search_code for names from the evidence
   (call labels, tool names, error names, prompt text), then read_file around the matches. Read
   before you claim anything.
4. Call submit_suggestion once, with the smallest change that addresses the cause.

Rules:
- Every claim must come from the evidence or from code you read; cite file:line.
- A code or prompt change goes in `patch` as a unified diff against the CURRENT file (exact context
  lines copied from read_file, `--- a/path` / `+++ b/path` / `@@` hunks). Paths as read_file shows them.
- If the fix is a setting or experiment choice rather than code (e.g. undo the change being tested),
  use change_type "setting" and describe it in setting_change; no patch.
- If the evidence says maybe_noise (the question also moved when nothing changed), say so in
  what_is_wrong and use confidence "low": it may not be a real problem.
- Never suggest changing the tests, the judges, the answer keys or the scoring to make results look
  better. Fix the system, not the measurement.
- If you can't find a fix you can support with evidence, use change_type "none" and say why.
- Write what_is_wrong in plain, simple words for someone who didn't write this code."""

TOOLS = [
    {"type": "function", "function": {
        "name": "list_files", "description": "List a folder (relative to an allowed folder).",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "read_file", "description": "Read lines of a text file, numbered (max 80 lines per call: ask for the lines you need).",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "start": {"type": "integer"}, "end": {"type": "integer"}},
            "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "search_code", "description": "Find lines containing some text (case-insensitive) in code and config files.",
        "parameters": {"type": "object", "properties": {
            "text": {"type": "string"}, "path": {"type": "string", "description": "folder or file to search, default all"}},
            "required": ["text"]}}},
    {"type": "function", "function": {
        "name": "submit_suggestion", "description": "Your final answer. Call exactly once.",
        "parameters": {"type": "object", "properties": {
            "title": {"type": "string", "description": "one line"},
            "what_is_wrong": {"type": "string", "description": "plain words: what goes wrong and why"},
            "evidence": {"type": "array", "items": {"type": "string"}, "description": "facts, each with its source (file:line or the measurement)"},
            "change_type": {"type": "string", "enum": ["code", "prompt", "setting", "none"]},
            "file": {"type": "string"},
            "patch": {"type": "string", "description": "unified diff, for code/prompt changes"},
            "setting_change": {"type": "string", "description": "for change_type setting"},
            "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
            "how_to_check": {"type": "string", "description": "how the owner can check the fix worked (e.g. which questions to re-run)"},
            "risks": {"type": "string", "description": "what else this change could affect"}},
            "required": ["title", "what_is_wrong", "evidence", "change_type", "confidence", "how_to_check"]}}},
]


def _usage(resp):
    u = getattr(resp, "usage", None)
    return int(getattr(u, "total_tokens", 0) or 0) if u else 0


SUBMIT_TOOL = next(t for t in TOOLS if t["function"]["name"] == "submit_suggestion")
FINAL_NOTE = ("Your budget for this problem is almost used. Call submit_suggestion NOW with what you found so far "
              "(change_type \"none\" and low confidence are fine if you aren't sure). No other tool is available.")


def _call(client, model, messages, final):
    """One model call. On the final call only submit_suggestion is offered: some models ignore a forced
    tool_choice while other tools are listed, and the provider then rejects the call (400 tool_use_failed)."""
    return client.chat.completions.create(model=model, messages=messages, tools=[SUBMIT_TOOL] if final else TOOLS,
                                          tool_choice="required" if final else "auto", temperature=0, max_tokens=4000)


def _rejected_tool_call(e):
    """The provider refused the model's tool call (Groq: 400 tool_use_failed). Worth one retry with a note."""
    text = str(e)
    return "tool_use_failed" in text or "Tool call validation failed" in text


def _run_tool(box, name, args):
    try:
        if name == "list_files":
            out = box.list_files(args.get("path") or ".")
        elif name == "read_file":
            out = box.read_file(args.get("path"), args.get("start") or 1, args.get("end"))
        elif name == "search_code":
            out = box.search_code(args.get("text"), args.get("path") or ".")
        else:
            out = f"unknown tool {name}"
    except Refused as e:
        out = f"refused: {e}"
    except (TypeError, ValueError) as e:
        out = f"bad arguments: {e}"
    return out if len(out) <= TOOL_OUTPUT_CHARS else out[:TOOL_OUTPUT_CHARS] + "\n... (cut: read a smaller range)"


def problem_message(problem, box, system_info):
    own = [d for d in box.describe() if d["name"] != "framework"]
    fw = [d for d in box.describe() if d["name"] == "framework"]
    folders = "\n".join([f"- {d['path']} (THE SYSTEM'S OWN CODE: look here first; paths start with "
                          f"{os.path.basename(d['path'])}/)" for d in own]
                         + [f"- {d['path']} (the evaluation framework: measurement code, rarely where a fix goes)" for d in fw])
    return (f"SYSTEM UNDER TEST: {json.dumps(system_info, ensure_ascii=False)}\n"
            f"FOLDERS YOU CAN READ:\n{folders}\n"
            "Paths are relative to these folders.\n\n"
            f"PROBLEM:\n{json.dumps(problem, ensure_ascii=False, indent=1)}")


def _compact(messages):
    """Every call re-sends the whole conversation, and the provider's tokens-per-minute limit makes long
    conversations slow. So only the last KEEP_RESULTS tool results stay in full; older ones keep their
    first line (which file and lines it was), and the model can read again if it needs to."""
    tool_msgs = [m for m in messages if m["role"] == "tool"]
    for m in tool_msgs[:-KEEP_RESULTS]:
        if not m.get("_short"):
            first = m["content"].splitlines()[0] if m["content"] else ""
            m["content"] = f"[earlier result shortened to save tokens: {first[:200]}]"
            m["_short"] = True


def _wire(messages):
    return [{k: v for k, v in m.items() if not k.startswith("_")} for m in messages]


def suggest_one(problem, box, client, model=DEFAULT_MODEL, system_info=None, max_steps=8, max_tokens=30_000,
                on_step=None):
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": problem_message(problem, box, system_info or {})
                 + f"\n\nBUDGET: at most {max_steps} calls and about {max_tokens:,} tokens (every call re-sends this "
                   "conversation, so read only the lines you need)."}]
    tokens, steps, nudged, retries, warned = 0, 0, False, 0, False
    while True:
        final = steps >= max_steps - 1 or tokens >= max_tokens * 0.8
        if final and not warned:
            messages.append({"role": "user", "content": FINAL_NOTE})
            warned = True
        _compact(messages)
        if on_step:
            on_step(steps + 1, "final call: submit" if final else "thinking", tokens)
        try:
            resp = _call(client, model, _wire(messages), final)
        except Exception as e:  # noqa: BLE001
            import rate_limit
            if rate_limit.is_daily_limit_error(e):
                raise
            steps += 1
            if _rejected_tool_call(e) and retries < 2 and steps < max_steps + 2:
                retries += 1
                messages.append({"role": "user", "content": "Your last tool call was rejected by the API: "
                                 + str(e)[:300] + ("\n" + FINAL_NOTE if final else "\nUse only the tools listed.")})
                continue
            return _done("stopped", None, f"{type(e).__name__}: {str(e)[:300]}", steps, tokens, box)
        steps += 1
        tokens += _usage(resp)
        msg = resp.choices[0].message
        calls = list(getattr(msg, "tool_calls", None) or [])
        if not calls:
            if final or nudged:
                return _done("stopped", None, "the model didn't submit a suggestion", steps, tokens, box)
            nudged = True
            messages.append({"role": "assistant", "content": getattr(msg, "content", "") or ""})
            messages.append({"role": "user", "content": "Use the tools, then call submit_suggestion."})
            continue
        messages.append({"role": "assistant", "content": getattr(msg, "content", None) or "",
                         "tool_calls": [{"id": c.id, "type": "function",
                                         "function": {"name": c.function.name, "arguments": c.function.arguments}}
                                        for c in calls]})
        for c in calls:
            try:
                args = json.loads(c.function.arguments or "{}")
            except ValueError:
                args = {}
            if c.function.name == "submit_suggestion":
                return _finish(args, steps, tokens, box)
            messages.append({"role": "tool", "tool_call_id": c.id, "content": _run_tool(box, c.function.name, args)})
            if on_step:
                what = args.get("path") or args.get("text") or ""
                if c.function.name == "read_file":
                    what = f"{what}:{args.get('start') or 1}-{args.get('end') or ''}"
                on_step(steps, f"  {c.function.name} {what}"[:120], tokens)
        if final:   # only submit_suggestion was offered; anything else is a safety net
            return _done("stopped", None, f"budget used up ({steps} steps, {tokens:,} tokens) before a suggestion",
                         steps, tokens, box)


def _finish(s, steps, tokens, box):
    s = {k: v for k, v in s.items() if v not in (None, "", [])}
    s.setdefault("change_type", "none")
    if isinstance(s.get("evidence"), str):
        s["evidence"] = [s["evidence"]]
    if s["change_type"] == "none":
        return _done("no_fix", s, s.get("what_is_wrong") or "no fix the evidence supports", steps, tokens, box)
    check = None
    if s.get("patch"):
        fits, why = check_patch(box, s["patch"])
        check = {"fits": fits, "why": why}
    elif s["change_type"] in ("code", "prompt"):
        check = {"fits": False, "why": "no patch given"}
    out = _done("suggested", s, None, steps, tokens, box)
    out["patch_check"] = check
    return out


def _done(status, suggestion, reason, steps, tokens, box):
    return {"status": status, "suggestion": suggestion, "reason": reason, "steps": steps, "tokens": tokens,
            "files_read": list(dict.fromkeys(box.reads)), "patch_check": None}


# ---------------------------------------------------------------- a fake model, for tests and dry runs (0 tokens)

class FakeClient:
    """Plays back scripted turns. Each turn: ("tool", name, args) calls a tool, ("submit", {...}) submits,
    ("text", "...") answers without a tool call, ("raise", exc) fails the call. On a final call (only
    submit_suggestion offered) any non-submit turn becomes a forced 'none' submit. Records every request."""

    def __init__(self, turns, tokens_per_call=1000):
        self.turns, self.requests, self.tokens = list(turns), [], tokens_per_call
        self.chat = type("C", (), {"completions": self})()

    def create(self, **kw):
        self.requests.append(kw)
        turn = self.turns.pop(0) if self.turns else ("text", "I'm done.")
        if turn[0] == "raise":
            raise turn[1]
        if len(kw.get("tools") or []) == 1 and turn[0] != "submit":
            turn = ("submit", {"title": "forced", "what_is_wrong": "ran out of budget", "evidence": [],
                               "change_type": "none", "confidence": "low", "how_to_check": "-"})
        calls = []
        if turn[0] in ("tool", "submit"):
            name, args = ("submit_suggestion", turn[1]) if turn[0] == "submit" else (turn[1], turn[2])
            fn = type("F", (), {"name": name, "arguments": json.dumps(args)})()
            calls = [type("T", (), {"id": f"call_{len(self.requests)}", "function": fn})()]
        msg = type("M", (), {"content": turn[1] if turn[0] == "text" else None, "tool_calls": calls})()
        usage = type("U", (), {"total_tokens": self.tokens})()
        return type("R", (), {"choices": [type("Ch", (), {"message": msg})()], "usage": usage})()
