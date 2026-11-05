"""
Tests for retrieval.py and the retrieval settings of agent_configurable.py
(no API calls; the embeddings tests use the local model and are skipped
if fastembed is not installed).
    python tests/test_retrieval.py
"""

import os
import shutil
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault("GROQ_API_KEY", "offline-test")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from langchain_core.messages import AIMessage  # noqa: E402

import agent_configurable as ac  # noqa: E402
import retrieval  # noqa: E402
import store  # noqa: E402
import test_agent  # noqa: E402

failures = []


def ok(label, cond, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'}  {label}" + (f"  -> {detail}" if detail else ""))
    if not cond:
        failures.append(label)


topics = lambda hits: [h["topic"] for h in hits]
FAQS = retrieval.load_faqs()

print("substring = exactly the old store.search_faq():")
queries = ["What is your return policy?", "RETURN POLICY please", "do you ship to a po box", "how long is shipping",
           "shipping to a po box?", "payment methods?", "customs", "import fees", "mailbox address", "", "hello"]
same = [(q, retrieval.search(q, "substring")[0]["text"] if retrieval.search(q, "substring") else None, store.search_faq(q))
        for q in queries]
ok(f"identical text for {len(queries)} queries (incl. no match, two topics in one query)",
   all(new == old for _, new, old in same), [q for q, n, o in same if n != o])
ok("two topics in one query -> first in DB order, like before",
   topics(retrieval.search("shipping to a po box?", "substring")) == ["shipping"])
ok("substring ignores paraphrases (the problem this phase fixes)", retrieval.search("import fees", "substring") == [])

print("bm25:")
ok("'import fees' finds customs", topics(retrieval.search("import fees", "bm25")) == ["customs"])
ok("unrelated question finds nothing (below min_score)", retrieval.search("what is the weather like today", "bm25") == [])
three = retrieval.search("international shipping fees and delivery", "bm25", top_k=3)
ok("top_k=3: at most 3, best first", 1 <= len(three) <= 3 and [h["score"] for h in three] == sorted((h["score"] for h in three), reverse=True),
   [(h["topic"], h["score"]) for h in three])
ok("top_k=1 returns only the best", retrieval.search("international shipping fees and delivery", "bm25", top_k=1) == three[:1])
ok("high min_score -> nothing", retrieval.search("import fees", "bm25", min_score=50) == [])
ok("deterministic", retrieval.search("international shipping fees", "bm25", top_k=5)
   == retrieval.search("international shipping fees", "bm25", top_k=5))
tie = [("a topic", "same words here"), ("b topic", "same words here")]
ok("ties broken by FAQ order", topics(retrieval.search("same words", "bm25", top_k=2, min_score=0, faqs=tie)) == ["a topic", "b topic"])
ok("result has topic + text + score", set(retrieval.search("import fees", "bm25")[0]) == {"topic", "text", "score"})

if retrieval.embeddings_available():
    print("embeddings:")
    ok("'import fees' finds customs", topics(retrieval.search("import fees", "embeddings")) == ["customs"])
    ok("'mailbox address' finds PO box (no shared keyword)", topics(retrieval.search("can you deliver to my mailbox address", "embeddings")) == ["po box"])
    ok("unrelated question finds nothing (below min_score 0.6)", retrieval.search("what is the weather like today", "embeddings") == [])
    k3 = retrieval.search("shipping", "embeddings", top_k=3, min_score=0)
    ok("top_k=3 -> exactly 3, best first", len(k3) == 3 and k3[0]["score"] >= k3[1]["score"] >= k3[2]["score"], topics(k3))
    ok("deterministic", retrieval.search("import fees", "embeddings", top_k=5, min_score=0)
       == retrieval.search("import fees", "embeddings", top_k=5, min_score=0))

    print("embedding cache:")
    tmp_cache = tempfile.mkdtemp()
    real_cache, real_embed = retrieval.CACHE_DIR, retrieval._embed
    retrieval.CACHE_DIR = tmp_cache
    calls = []

    def counting_embed(model, texts):
        calls.append(len(texts))
        return real_embed(model, texts)

    retrieval._embed = counting_embed
    custom = [("returns", "Items can be returned within 30 days."), ("customs", "Import duties are paid by the buyer.")]
    retrieval._vectors.clear()
    first = retrieval.search("import fees", "embeddings", min_score=0, faqs=custom)
    path = retrieval.vector_cache_path(custom)
    ok("first search computes FAQ vectors once and writes the cache file", calls == [2] and os.path.isfile(path),
       os.path.basename(path))
    retrieval._vectors.clear()   # like a new process
    second = retrieval.search("import fees", "embeddings", min_score=0, faqs=custom)
    ok("new process: vectors loaded from disk, not recomputed", calls == [2] and second == first)
    changed = custom[:1] + [("customs", "Import duties and taxes are the buyer's responsibility.")]
    retrieval.search("import fees", "embeddings", min_score=0, faqs=changed)
    ok("editing an FAQ -> new cache key, recomputed", calls == [2, 2] and retrieval.vector_cache_path(changed) != path
       and len(os.listdir(os.path.join(tmp_cache, "faq_vectors"))) == 2)
    retrieval.CACHE_DIR, retrieval._embed = real_cache, real_embed
    retrieval._vectors.clear()
    shutil.rmtree(tmp_cache, ignore_errors=True)
else:
    print("embeddings: SKIPPED (fastembed not installed: pip install -r requirements-rag.txt)")

print("settings validation:")
BASE = {"model": "openai/gpt-oss-120b", "temperature": 0, "prompt": "prompts/v1.txt",
        "docstrings": "docstrings/detailed.yaml", "retrieval": {"method": "substring"}, "recursion_limit": 25}


def expect_error(label, retrieval_settings, must_contain):
    try:
        ac.validate_settings({**BASE, "retrieval": retrieval_settings})
        ok(label, False, "no error")
    except ac.SettingsError as e:
        ok(label, all(s in str(e) for s in must_contain), str(e).splitlines()[1].strip())


expect_error("unknown method", {"method": "vector_db"}, ["'substring', 'bm25' or 'embeddings'"])
expect_error("top_k must be 1-5", {"method": "bm25", "top_k": 0}, ["settings.retrieval.top_k"])
expect_error("top_k too large", {"method": "bm25", "top_k": 9}, ["settings.retrieval.top_k"])
expect_error("min_score must be >= 0", {"method": "bm25", "min_score": -1}, ["settings.retrieval.min_score"])
expect_error("substring + top_k is a mistake", {"method": "substring", "top_k": 3}, ["don't apply to 'substring'"])
expect_error("typo with suggestion", {"method": "bm25", "minscore": 1}, ["did you mean 'min_score'"])
real_avail = retrieval.embeddings_available
retrieval.embeddings_available = lambda: False
try:
    ac.validate_settings({**BASE, "retrieval": {"method": "embeddings"}})
    ok("embeddings without fastembed -> clear error", False)
except ac.SettingsError as e:
    ok("embeddings without fastembed -> clear error", "pip install -r requirements-rag.txt" in str(e), str(e))
retrieval.embeddings_available = real_avail
ok("valid bm25 settings accepted", ac.validate_settings({**BASE, "retrieval": {"method": "bm25", "top_k": 2, "min_score": 0.8}}).retrieval.top_k == 2)

print("agent tool: same name / description / schema, only the search changes:")


class FakeLLM:
    def __init__(self, query):
        self.query, self.bound, self.n = query, None, 0

    def bind_tools(self, tools):
        self.bound = tools
        return self

    def invoke(self, messages):
        self.n += 1
        if self.n == 1:
            return AIMessage(content="", tool_calls=[{"name": "faq_search", "args": {"query": self.query}, "id": "c1"}])
        return AIMessage(content="done")


def run_with(method, query):
    fake = FakeLLM(query)
    ac._graph_cache.clear()
    ac._make_llm = lambda model, temperature: fake
    r = ac.run_agent("q", {**BASE, "retrieval": {"method": method}})
    return r, next(t for t in fake.bound if t.name == "faq_search")


r_sub, tool_sub = run_with("substring", "import fees")
r_bm, tool_bm = run_with("bm25", "import fees")
ok("substring uses test_agent's original faq_search function", tool_sub.func is test_agent.faq_search.func)
ok("bm25 tool: same name, description, argument schema; different function",
   tool_bm.name == tool_sub.name == "faq_search" and tool_bm.description == tool_sub.description
   and tool_bm.args == tool_sub.args and tool_bm.func is not tool_sub.func)
ok("substring: 'import fees' -> the old 'No matching FAQ' text", r_sub["tool_outputs"] == [test_agent.faq_search.invoke({"query": "import fees"})])
ok("bm25: 'import fees' -> the customs FAQ text, same format as substring hits",
   r_bm["tool_outputs"] == [store.search_faq("customs")] and r_bm["tools_used"] == ["faq_search"])
r_none, _ = run_with("bm25", "what is the weather like today")
ok("bm25 no hit -> exactly the old no-match line", r_none["tool_outputs"] == [ac.NO_MATCH]
   and ac.NO_MATCH == test_agent.faq_search.invoke({"query": "weather"}))

print("\nFAILED: " + ", ".join(failures) if failures else "\nPASS")
sys.exit(1 if failures else 0)
