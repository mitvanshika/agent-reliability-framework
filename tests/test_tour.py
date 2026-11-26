"""
Tests for tour.py (the guided tour page's data) and its /tour route. Builds a tiny
fake target in a temp folder: 2 questions, a baseline, an A/A run, an experiment run
and a noise file, then checks every step of the tour is built from them. No API calls.
    python tests/test_tour.py
"""

import json
import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [REPO, os.path.dirname(os.path.abspath(__file__))]
os.environ.setdefault("GROQ_API_KEY", "offline-test")

import tour  # noqa: E402

failures = []


def ok(label, cond, detail=""):
    detail = str(detail).encode("ascii", "backslashreplace").decode() if detail else ""
    print(f"  {'OK  ' if cond else 'FAIL'}  {label}" + (f"  -> {detail}" if detail else ""))
    if not cond:
        failures.append(label)


def write(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        if path.endswith(".yaml"):
            import yaml
            yaml.safe_dump(data, f)
        else:
            json.dump(data, f)


def judged(sid, final, answer="Revenue was 10."):
    return {"scenario_id": sid, "question": f"Question {sid}?", "answer": answer, "final_score": final,
            "usage": {"input_tokens": 800, "output_tokens": 400, "total_tokens": 1200}, "latency_s": 3.0,
            "llm_calls": [{"label": "chunk_grades", "model": "m-small", "input_tokens": 600, "output_tokens": 100,
                           "total_tokens": 700, "seconds": 0.8},
                          {"label": "chunk_grades", "model": "m-small", "input_tokens": None, "output_tokens": None,
                           "total_tokens": None, "seconds": 0.5, "error": "BadRequestError"},
                          {"label": "grounded_answer", "model": "m-big", "input_tokens": 200, "output_tokens": 300,
                           "total_tokens": 500, "seconds": 1.1}],
            "trajectory": [
                {"type": "tool", "name": "retrieve", "args": {"query": "revenue 2024", "strategy": "original"},
                 "output": "chunks: a, b", "note": "2 of 4 relevant"},
                {"type": "llm", "label": "chunk_grades", "tokens": {"input": 600, "output": 100, "total": 700},
                 "latency_s": 0.8},
                {"type": "llm", "label": "chunk_grades", "tokens": None, "error": "BadRequestError", "latency_s": 0.5},
                {"type": "llm", "label": "grounded_answer", "tokens": {"total": 500}, "latency_s": 1.1}],
            "correctness_gt_result": {"score": final, "reason": "Matches the key."},
            "groundedness_result": {"score": 10, "reason": "All backed."},
            "context_relevance_result": {"score": None, "na": True, "reason": "nothing retrieved"}}


def build(root):
    runs, base, noise, cfg = (os.path.join(root, d) for d in ("runs", "baselines", "noise", "cfg"))
    for sid, cat in (("t_q1", "numeric_lookup"), ("t_q2", "trend")):
        write(os.path.join(cfg, "scen_t", f"{sid}.yaml"),
              {"id": sid, "category": cat, "question": f"Question {sid}?", "expected_answer": "10"})
    meta = {"resolved_config": {"target": "toy"}, "experiment": "toy_baseline", "description": "as is",
            "target_function": "adapters.toy.run", "accepts_settings": True}
    # the baseline run
    write(os.path.join(runs, "toy_baseline_1", "run_meta.json"), meta)
    write(os.path.join(runs, "toy_baseline_1", "judged", "t_q1.json"), judged("t_q1", 9.0))
    write(os.path.join(runs, "toy_baseline_1", "judged", "t_q2.json"), judged("t_q2", 6.0, "No idea."))
    write(os.path.join(base, "toy.json"), {
        "run_dir": "runs\\toy_baseline_1", "promoted_at": "2026-11-24T10:00:00", "note": "first",
        "summary": {"mean_final_score": 7.5, "tokens": {"total_tokens": 2400}, "latency_mean_s": 3.0,
                    "per_judge_mean": {"correctness_gt": 7.5, "groundedness": 10}},
        "scenarios": {"t_q1": {"passed": True, "final_score": 9.0, "judges": {"correctness_gt": 9}},
                      "t_q2": {"passed": False, "final_score": 6.0, "judges": {"correctness_gt": 4}}}})
    # the A/A run: nothing changed (only the old empty-settings artefact in its diff)
    write(os.path.join(runs, "toy_baseline_2", "run_meta.json"), meta)
    write(os.path.join(runs, "toy_baseline_2", "comparison.json"), {
        "config_diff": ["settings: {} -> (unset)"], "verdict": "NO REAL DIFFERENCE", "pairs": 2,
        "mcnemar": {"b_pass_to_fail": 0, "c_fail_to_pass": 0, "p": 1.0}, "wilcoxon": {"p": 0.8},
        "resources": {"total_tokens": {"pct": -3.0}},
        "rows": [{"scenario_id": "t_q1", "delta": 0.1}, {"scenario_id": "t_q2", "delta": 1.0}]})
    # an experiment run
    write(os.path.join(runs, "toy_exp_1", "run_meta.json"),
          {**meta, "experiment": "toy_exp", "description": "fewer chunks"})
    write(os.path.join(runs, "toy_exp_1", "summary.json"), {"passed": 1, "scenarios": 2, "mean_final_score": 7.4})
    write(os.path.join(runs, "toy_exp_1", "comparison.json"), {
        "config_diff": ["settings.retrieval.final_top_k: (unset) -> 3"], "verdict": "NO REAL DIFFERENCE",
        "flips": {"pass_to_fail": 0, "fail_to_pass": 0}, "mcnemar": {"p": 1.0}, "wilcoxon": {"p": 0.7},
        "resources": {"total_tokens": {"baseline": 2400, "run": 1800, "pct": -25.0},
                      "latency_mean_s": {"pct": -10.0}},
        "down": ["t_q1"], "up": ["t_q2"],
        "rows": [{"scenario_id": "t_q1", "direction": "down", "baseline_score": 9.0, "run_score": 8.0,
                  "band": 0.5, "flip": None, "judges_moved": ["correctness_gt 9->8"]},
                 {"scenario_id": "t_q2", "direction": "up", "baseline_score": 6.0, "run_score": 7.5,
                  "band": 0.5, "flip": "fail->pass", "judges_moved": []}],
        "question_costs": {"per_question": [{"scenario_id": "t_q1", "baseline_tokens": 1200, "run_tokens": 900}],
                           "biggest_changes": ["t_q1"]}})
    # noise
    write(os.path.join(noise, "toy_noise.json"), {
        "runs": 2, "jumpy": ["t_q2"], "suite": {"mean_score_range": 0.5},
        "scenarios": {"t_q1": {"range": 0.1}, "t_q2": {"range": 2.5}},
        "jumpy_explained": {"t_q2": {"kind": "SYSTEM", "why": "the runs found different evidence",
                                     "detail": ["run 1 found: 2 of 4"], "borderline": False}}})
    targets = {"toy": {"description": "A toy system", "scenarios": "scen_t", "pass_threshold": 7,
                       "judges": ["correctness_gt", "groundedness"]},
               "other": {"description": "Not run yet"}}
    return runs, base, noise, cfg, targets


with tempfile.TemporaryDirectory() as root:
    runs, base, noise, cfg, targets = build(root)
    t = tour.tour("toy", runs, base, noise, cfg, targets)

    print("\n1-2. the system and its questions (nothing system-specific in the code)")
    ok("no title in targets.yaml -> the target's name", t["title"] == "toy", t["title"])
    ok("no about -> its description", t["plain"] == "A toy system", t["plain"])
    ok("how it's plugged in, from the run's own meta",
       "adapters/toy.py" in (t["plugged_in"] or "") and "settings" in t["plugged_in"], t["plugged_in"])
    owned = {"toy": {**targets["toy"], "title": "Toy RAG", "about": "Answers toy questions.", "tour_question": "t_q1"}}
    t3 = tour.tour("toy", runs, base, noise, cfg, owned)
    ok("the owner's title and about are used", t3["title"] == "Toy RAG" and t3["plain"] == "Answers toy questions.")
    ok("the owner's tour_question wins", t3["featured"]["id"] == "t_q1"
       and t3["featured"]["why_this_one"] == "chosen by the system's owner", t3["featured"]["why_this_one"])
    ok("judges named from the target's own judge list", t["judges_used"] == ["Correctness", "Groundedness"],
       t["judges_used"])
    ok("2 questions in 2 kinds", t["questions"]["count"] == 2 and len(t["questions"]["categories"]) == 2,
       t["questions"])

    print("\n3-4. one answer up close, with its judges")
    f = t.get("featured") or {}
    ok("no tour_question -> the question that wobbled most (t_q2, range 2.5)", f.get("id") == "t_q2"
       and "wobbled" in f.get("why_this_one", ""), f.get("id"))
    f = t3["featured"]
    ok("answer key from the scenario file", f.get("answer_key") == "10", f.get("answer_key"))
    ok("steps: 1 search + 3 LLM calls", [s["kind"] for s in f.get("steps") or []] == ["search", "llm", "llm", "llm"],
       f.get("steps"))
    ok("an LLM step shows tokens in and out", f["steps"][1]["text"] == "700 tokens (600 in, 100 out)", f["steps"][1])
    ok("a failed call says so instead of a blank", f["steps"][2]["text"] == "failed (BadRequestError): no tokens counted",
       f["steps"][2])

    print("\ntokens")
    ct = f["call_table"]
    ok("call table: 3 calls, 1 failed, totals from the answer's usage",
       len(ct["rows"]) == 3 and ct["failed"] == 1 and (ct["input"], ct["output"], ct["total"]) == (800, 400, 1200), ct)
    ok("no pricing.yaml -> no money cost, and the page says why", ct["cost"] is None and "no pricing.yaml" in t["cost_note"])
    sp = t3["baseline"]["spend"]
    ok("where the tokens went: per kind of call, biggest first",
       [(x["label"], x["calls"], x["total"]) for x in sp["by_label"]] == [("chunk_grades", 4, 1400), ("grounded_answer", 2, 1000)],
       sp["by_label"])
    ok("shares add up", round(sum(x["share"] for x in sp["by_label"])) == 100, sp["by_label"])
    ok("2 failed calls counted, both questions known", sp["failed_calls"] == 2 and sp["unknown"] == [], sp)
    real_pricing = tour._pricing
    tour._pricing = lambda: {"m-small": {"input": 1.0, "output": 2.0}, "m-big": {"input": 10.0, "output": 20.0}}
    try:
        tp = tour.tour("toy", runs, base, noise, cfg, owned)
    finally:
        tour._pricing = real_pricing
    # per answer: 600*1 + 100*2 + 200*10 + 300*20 = 8,800 per 1M = 0.0088 USD (the failed call costs nothing known)
    ok("with the owner's prices: cost per answer from each call's own model",
       abs(tp["featured"]["call_table"]["cost"] - 0.0088) < 1e-9, tp["featured"]["call_table"]["cost"])
    ok("...and for the whole baseline (2 answers)", tp["baseline"]["spend"]["cost"] == 0.0176, tp["baseline"]["spend"]["cost"])
    ok("a model with no price -> no total cost (never a guess)",
       tour._call_cost({"model": "other", "input_tokens": 1, "output_tokens": 1}, {"m-small": {"input": 1, "output": 1}}) is None)
    ok("'Where do the tokens go?' answered from the data",
       any(q["q"] == "Where do the tokens go?" and "chunk_grades" in q["a"] for q in t["qa"]))
    ok("search step keeps the query and what it found",
       f["steps"][0]["text"] == "revenue 2024" and f["steps"][0]["note"] == "2 of 4 relevant", f["steps"][0])
    ok("2 judges with plain titles and reasons",
       [j["title"] for j in f.get("judges") or []] == ["Correctness", "Groundedness"]
       and f["judges"][0]["reason"] == "Matches the key.", f.get("judges"))
    ok("passes at 9.0 with pass mark 7", f.get("passed") is True and f.get("calls") == 3)

    print("\n5. the baseline")
    b = t.get("baseline") or {}
    ok("1/2 passed, run id from a Windows path", b.get("passed") == 1 and b.get("total") == 2
       and b.get("run") == "toy_baseline_1", b)
    ok("failed question listed with its lowest judge", b["failed"][0]["id"] == "t_q2"
       and b["failed"][0]["lowest"] == ("correctness_gt", 4), b.get("failed"))

    print("\n6. noise")
    aa = t.get("aa") or {}
    ok("A/A run found (empty-settings diff counts as no change)", aa.get("run") == "toy_baseline_2", aa.get("run"))
    ok("A/A moved = |delta| >= 0.5", aa.get("moved") == ["t_q2"], aa.get("moved"))
    n = t.get("noise") or {}
    ok("1 of 2 steady, jumpy t_q2 explained as SYSTEM",
       n.get("steady") == 1 and n["jumpy"][0]["kind"] == "SYSTEM", n)

    print("\n7. experiments")
    e = (t.get("experiments") or [{}])[0]
    ok("one experiment, the A/A run is not one", len(t["experiments"]) == 1 and e.get("name") == "toy_exp",
       [x["name"] for x in t["experiments"]])
    ok("tokens -25%, 1/2 passed", e["tokens"]["pct"] == -25.0 and e["passed"] == 1, e["tokens"])
    moved = {m["id"]: m["also_in_aa"] for m in e["moved"]}
    ok("t_q2 also moved in the A/A test, t_q1 did not", moved == {"t_q1": False, "t_q2": True}, moved)
    ok("cost rows from question_costs", e["cost_rows"][0]["scenario_id"] == "t_q1", e["cost_rows"])

    print("\n8. questions written from the data")
    qs = [q["q"] for q in t["qa"]]
    ok("has the core questions", any("every question pass" in q for q in qs) and any("toy_exp" in q for q in qs)
       and any("can't" in q for q in qs), qs)
    why = next(q["a"] for q in t["qa"] if "every question pass" in q["q"])
    ok("'why not all passed' names t_q2", "t_q2" in why, why)

    print("\nthe /tour list: every plugged-in system")
    sy = {x["target"]: x for x in tour.systems(runs, base, targets)}
    ok("both systems listed, toy first (it has a baseline)", list(sy) == ["toy", "other"], list(sy))
    ok("toy: 1/2 passed, 3 runs; other: no baseline", sy["toy"]["passed"] == 1 and sy["toy"]["runs"] == 3
       and not sy["other"]["has_baseline"], sy)

    print("\nno baseline yet: the tour stops after the questions")
    t2 = tour.tour("toy", runs, os.path.join(root, "none"), noise, cfg, targets)
    ok("no baseline -> no featured/experiments", "baseline" not in t2 and "featured" not in t2, sorted(t2))
    import app  # noqa: E402
    with app.app.test_request_context("/tour/toy"):
        page = app.render_template("tour.html", t=t2, live=False)
    ok("the page still renders, saying no baseline yet", "No baseline has been approved" in page)

print("\nthe /tour page")
import app  # noqa: E402
c = app.app.test_client()
r = c.get("/tour/no_such_target")
ok("unknown target -> 404", r.status_code == 404, r.status_code)
r = c.get("/tour")
ok("/tour lists the systems", r.status_code == 200 and b"Pick a system" in r.data, r.status_code)
r = c.get("/tour/financial_dd")
ok("/tour/<target> renders", r.status_code == 200, r.status_code)
ok("nav has the Guided tour item", b'href="/tour"' in r.data)

print("\nPASS" if not failures else f"\nFAIL: {failures}")
sys.exit(1 if failures else 0)
