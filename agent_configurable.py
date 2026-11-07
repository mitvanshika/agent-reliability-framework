"""
The demo support agent, built from experiment settings instead of being
hard-coded. Used by experiments/*.yaml via the "configurable" target.

Same LangGraph ReAct loop and same 4 tools as test_agent.py, but the
system prompt, tool descriptions, model, temperature and recursion limit
come from settings:

    settings:
      model: openai/gpt-oss-120b
      temperature: 0
      prompt: prompts/v1.txt              # text file = the system prompt
      docstrings: docstrings/detailed.yaml  # tool name -> description the model sees
      retrieval: {method: substring}      # or {method: bm25 | embeddings, top_k: 1, min_score: ...}
      recursion_limit: 25

Nothing about the tools' behaviour is re-implemented here: each tool is
a copy of test_agent.py's tool object with only its description swapped
(tool.model_copy), so it runs the very same Python function. The
30-day refund rule, order lookup and FAQ search live only in
test_agent.py / store.py.

The one exception is retrieval: with method bm25 / embeddings the
faq_search tool's FUNCTION is swapped for retrieval.search(), but its
name, argument and description stay exactly the same, and its output
has the same format (the FAQ text, or the same "No matching FAQ" line).
So an experiment that changes retrieval.method changes only how FAQs are
found: no confound, and expected_tools stay valid. method substring uses
test_agent's original faq_search unchanged.

Entry point for the harness: run_agent(question, settings).
validate_settings(settings) is called by run_experiment.py before any
API call, so a typo in an experiment file fails fast.
"""

import difflib
import json
import os
import time
from typing import Literal

import yaml
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

import prompt_addons
import rate_limit
import retrieval
from validated_tool import StrictArgs, validated_tool
import test_agent
from agent_retry import classify_agent_error, retry_delay

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
_graph_cache = {}


class SettingsError(ValueError):
    """Invalid settings for this target, with a message meant for a human."""


NO_MATCH = "No matching FAQ found for this query"   # same text as test_agent.faq_search


# Argument rules for this bot's tools, used only with tool_validation: strict
# (validated_tool.py does the generic work). Default "off" = the tools exactly as before.
class _OrderLookupArgs(StrictArgs):
    order_id: str = Field(pattern=r"^[A-Za-z][0-9]{4}$", description="like A1234: one letter and four digits")


class _RefundArgs(StrictArgs):
    order_total: float = Field(ge=0, description="a non-negative amount")
    days_since_purchase: int = Field(ge=0, description="a non-negative whole number of days")


class _FaqArgs(StrictArgs):
    query: str = Field(min_length=1, max_length=500, description="a non-empty search of at most 500 characters")


class _EscalateArgs(StrictArgs):
    reason: str = Field(min_length=1, description="a non-empty reason")


TOOL_CONSTRAINTS = {"order_lookup": _OrderLookupArgs, "refund_calculator": _RefundArgs,
                    "faq_search": _FaqArgs, "escalate_to_human": _EscalateArgs}


class RetrievalSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    method: Literal["substring", "bm25", "embeddings"]
    top_k: int = Field(1, ge=1, le=5)          # how many FAQs the tool returns
    min_score: float | None = Field(None, ge=0)  # None = retrieval.DEFAULT_MIN_SCORE[method]

    @model_validator(mode="after")
    def substring_has_no_ranking(self):
        if self.method == "substring" and {"top_k", "min_score"} & self.model_fields_set:
            raise ValueError("top_k / min_score don't apply to 'substring' (it returns the first FAQ whose "
                             "topic appears in the question); remove them, or use bm25 / embeddings")
        return self


class AgentSettings(BaseModel):
    # Every key is required on purpose: an experiment should say exactly
    # what it runs, and a missing key is more likely a mistake than intent.
    model_config = ConfigDict(extra="forbid")
    model: str
    temperature: float = Field(ge=0, le=2)
    prompt: str
    docstrings: str
    retrieval: RetrievalSettings
    recursion_limit: int = Field(ge=1)
    prompt_addons: list[str] = []   # framework rules appended to the prompt (prompt_addons/); default none
    tool_validation: Literal["off", "strict"] = "off"   # strict: validated_tool() on every tool; default off


def _repo_path(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(REPO_DIR, path)


def _explain(e: ValidationError) -> str:
    lines = []
    for err in e.errors():
        loc = ".".join(str(p) for p in err["loc"])
        if err["type"] == "extra_forbidden":
            model = RetrievalSettings if err["loc"][0] == "retrieval" else AgentSettings
            close = difflib.get_close_matches(str(err["loc"][-1]), model.model_fields, n=1)
            hint = f" (did you mean '{close[0]}'?)" if close else ""
            lines.append(f"settings.{loc}: unknown key{hint}")
        elif err["type"] == "missing":
            lines.append(f"settings.{loc}: required but missing")
        else:
            lines.append(f"settings.{loc}: {err['msg']}")
    allowed = ", ".join(AgentSettings.model_fields)
    return "Invalid settings for agent_configurable:\n  " + "\n  ".join(lines) + f"\n  (allowed keys: {allowed})"


def _load_prompt(path: str) -> str:
    full = _repo_path(path)
    if not os.path.isfile(full):
        raise SettingsError(f"settings.prompt: file not found: {path}")
    with open(full, encoding="utf-8") as f:   # text mode: \r\n -> \n on every OS
        return f.read()


def _load_docstrings(path: str) -> dict:
    full = _repo_path(path)
    if not os.path.isfile(full):
        raise SettingsError(f"settings.docstrings: file not found: {path}")
    with open(full, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    names = [t.name for t in test_agent.tools]
    if not isinstance(data, dict) or not all(isinstance(v, str) for v in data.values()):
        raise SettingsError(f"settings.docstrings: {path} must map tool names to description strings.")
    missing, unknown = sorted(set(names) - set(data)), sorted(set(data) - set(names))
    if missing or unknown:
        raise SettingsError(f"settings.docstrings: {path} must describe exactly the tools {names}"
                            + (f"; missing {missing}" if missing else "")
                            + (f"; unknown {unknown}" if unknown else ""))
    return data


def validate_settings(settings) -> AgentSettings:
    """Raises SettingsError with a readable message if settings are wrong."""
    if not isinstance(settings, dict):
        raise SettingsError("agent_configurable needs a settings mapping (run it via run_experiment.py).")
    try:
        cfg = AgentSettings(**settings)
    except ValidationError as e:
        raise SettingsError(_explain(e)) from None
    _load_prompt(cfg.prompt)
    _load_docstrings(cfg.docstrings)
    try:
        prompt_addons.validate(cfg.prompt_addons, _description())
    except ValueError as e:
        raise SettingsError(f"settings.prompt_addons: {e}") from None
    if cfg.retrieval.method == "embeddings":
        if not retrieval.embeddings_available():
            raise SettingsError("settings.retrieval.method 'embeddings' needs fastembed: "
                                "pip install -r requirements-rag.txt")
        # Load the model + FAQ vectors now (once per process), so the first timed
        # question doesn't pay ~10 s of model loading in its latency.
        retrieval.warm_up()
    return cfg


def _make_llm(model: str, temperature: float):
    # Separate function so tests can replace it with a fake LLM.
    # rate_limit.py paces requests and owns 429 / 5xx retries (shared budget with the judges).
    return ChatGroq(model=model, temperature=temperature, http_client=rate_limit.http_client("groq"), max_retries=0)


def _faq_function(r: RetrievalSettings):
    def faq_search(query: str) -> str:
        hits = retrieval.search(query, r.method, r.top_k, r.min_score)
        return "\n\n".join(h["text"] for h in hits) if hits else NO_MATCH
    return faq_search


def _tools(cfg: AgentSettings):
    """The exact tool list the agent binds for these settings."""
    descriptions = _load_docstrings(cfg.docstrings)
    # Same tool objects as the strong agent, only the description differs.
    tools = [t.model_copy(update={"description": descriptions[t.name]}) for t in test_agent.tools]
    if cfg.retrieval.method != "substring":
        # Same name, argument schema and description; only the search function changes.
        tools = [t.model_copy(update={"func": _faq_function(cfg.retrieval)}) if t.name == "faq_search" else t
                 for t in tools]
    if cfg.tool_validation == "strict":
        # Same name, description and schema the model sees; arguments checked inside, never raises.
        tools = [validated_tool(t, TOOL_CONSTRAINTS[t.name]) for t in tools]
    return tools


def build_tools(settings: dict):
    """Public: the tools for these settings, for fuzz_tools.py (targets.yaml `tools:`)."""
    return _tools(validate_settings(settings))


def _build(cfg: AgentSettings):
    tools = _tools(cfg)
    llm = _make_llm(cfg.model, cfg.temperature).bind_tools(tools)

    def call_model(state):
        return {"messages": [llm.invoke(state["messages"])]}

    graph = StateGraph(test_agent.AgentState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode(tools))
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", test_agent.should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    prompt = prompt_addons.apply(_load_prompt(cfg.prompt), {"prompt_addons": cfg.prompt_addons}, _description())
    return graph.compile(), prompt


def _description():
    """This target's one-line description from targets.yaml (used by the scope addon)."""
    import run_suite
    return (run_suite.load_targets().get("configurable") or {}).get("description")


def _get_app(settings: dict, cfg: AgentSettings):
    # Build once per distinct settings, not once per question.
    key = json.dumps(settings, sort_keys=True)
    if key not in _graph_cache:
        _graph_cache[key] = _build(cfg)
    return _graph_cache[key]


def _chat_messages(messages):
    """The whole conversation in OpenAI chat format (what fine-tuning APIs expect).
    Private model reasoning is not included."""
    out = []
    for m in messages:
        if isinstance(m, SystemMessage):
            out.append({"role": "system", "content": m.content})
        elif isinstance(m, HumanMessage):
            out.append({"role": "user", "content": m.content})
        elif isinstance(m, AIMessage):
            msg = {"role": "assistant", "content": m.content or ""}
            if m.tool_calls:
                msg["tool_calls"] = [{"id": c["id"], "type": "function",
                                      "function": {"name": c["name"], "arguments": json.dumps(c["args"])}}
                                     for c in m.tool_calls]
            out.append(msg)
        elif isinstance(m, ToolMessage):
            out.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content})
    return out


def _sum_usage(messages):
    """Token usage summed over every LLM call in this answer; None if unreported."""
    reports = [m.usage_metadata for m in messages if isinstance(m, AIMessage) and m.usage_metadata]
    if not reports:
        return None
    return {k: sum(r.get(k, 0) for r in reports) for k in ("input_tokens", "output_tokens", "total_tokens")}


def run_agent(question: str, settings: dict | None = None, max_retries: int = 2):
    if settings is None:
        raise SettingsError("agent_configurable.run_agent needs settings: run it via run_experiment.py, "
                            "not run_suite.py / calibrate_noise.py.")
    cfg = validate_settings(settings)
    app, system_prompt = _get_app(settings, cfg)

    last_error = None
    error_type = None
    for attempt in range(max_retries + 1):
        try:
            result = app.invoke(
                {"messages": [SystemMessage(content=system_prompt), HumanMessage(content=question)]},
                config={"recursion_limit": cfg.recursion_limit},
            )
            tools_used, tool_outputs = [], []
            for msg in result["messages"]:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    tools_used.extend(call["name"] for call in msg.tool_calls)
                if isinstance(msg, ToolMessage):
                    tool_outputs.append(msg.content)
            return {
                "answer": result["messages"][-1].content,
                "tools_used": tools_used,
                "tool_outputs": tool_outputs,
                "error": None,
                "error_type": None,
                "retries_needed": attempt,
                "usage": _sum_usage(result["messages"]),
                "messages": _chat_messages(result["messages"]),
            }
        except Exception as e:
            last_error = str(e)
            error_type = classify_agent_error(e)
            delay = retry_delay(error_type, attempt)
            if delay is None or attempt == max_retries:
                break
            print(f"  [agent] {type(e).__name__} ({error_type}), retry {attempt + 1}/{max_retries}"
                  + (f" in {delay:.1f}s" if delay else " now"))
            time.sleep(delay)
    return {
        "answer": None,
        "tools_used": [],
        "tool_outputs": [],
        "error": last_error,
        "error_type": error_type,
        "retries_needed": attempt,
        "usage": None,
    }
