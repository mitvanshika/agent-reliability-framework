# Smoke test: baseline on 2 scenarios

**Command:** `python experiment_pipeline.py experiments/baseline.yaml --scenarios normal_01 normal_02 --yes`
**Run:** `runs/baseline_20261106_101645` (git clean at run time)
**Expected:** both questions answered and judged with the experiment judge set, no errors, pipeline stops at "no baseline yet".
**Result: PASS**

| | normal_01 (return policy) | normal_02 (status of A1001) |
|---|---|---|
| tool used | faq_search | order_lookup |
| final_score | 9.23 | 10.00 |
| correctness_gt | 10 | 10 |
| completeness | 9 | 10 |
| groundedness | 7 | 10 |
| tool_efficiency | 10 | 10 |
| context_relevance | 10 | 10 |
| tool_match (display only) | 10 | 10 |

- Pipeline steps: run ok, judge ok, compare skipped (no baseline), exit code 0.
- Pass rate 2/2, mean final_score 9.61, 1,667 agent tokens, mean latency 1.75 s, 0 errors, 0 judge errors.
- Full conversation stored in every trace (5 messages: system, user, tool call, tool result, answer).

**What it shows**
- The answer-key judge works: normal_02 scores correctness_gt **10**. The classic correctness judge gave the same kind of answer **2** in the first smoke run ("likely fabricated"), because it could not see the store data.
- The one deduction is real: on normal_01 groundedness gave 7 because the agent added "items must be in original condition and packaging", which is not in the FAQ. A small, genuine embellishment, correctly caught.
