# Truth tests 4 and 5: prepared, not run

Both need Groq tokens. Nothing below has been run against the API yet. Each command still asks for
confirmation (no `--yes`), and stops cleanly at the daily limit.

## Test 4: a planted bug (refund window 10 days instead of 30)

**Question it answers:** does the framework catch a real, one-line code bug in a tool?

**The bug, without editing test_agent.py:** a new setting `refund_window_days` (default 30) in
`agent_configurable.py`. With 30 the agent gets test_agent's own `refund_calculator` (the same function object, so
the baseline and its config_hash are unchanged). With another value it gets the same tool (name, description,
schema) where only the comparison changed: `days <= 10` instead of `days <= 30`. The messages stay byte-identical
(they still say "30 days"), as with a real one-line bug, so the tool even contradicts itself for a 12-day order.
The FAQ also still says "returned within 30 days", so the test also shows whether the agent notices a conflict.

**Temporary experiment:** `experiments/tmp_test4_refund_10_days.yaml` (one change: `refund_window_days: 10`).
Delete it after the test.

**Questions** (`scenarios_test4/`, 14; answer keys follow the real 30-day policy):

| | id | question (numbers) | 30 days | 10 days (bug) |
|---|---|---|---|---|
| detector | t4_d01 | refund for order A1001 (12 days, $89.99) | eligible | NOT |
| detector | t4_d02 | $40.00, 11 days | eligible | NOT |
| detector | t4_d03 | $15.50, 13 days | eligible | NOT |
| detector | t4_d04 | $25.00, 15 days | eligible | NOT |
| detector | t4_d05 | $120.00, 20 days | eligible | NOT |
| detector | t4_d06 | $60.00, 25 days | eligible | NOT |
| detector | t4_d07 | $200.00, 28 days | eligible | NOT |
| detector | t4_d08 | $75.00, exactly 30 days (boundary) | eligible | NOT |
| control | t4_c01 | order A1003 (3 days) | eligible | eligible |
| control | t4_c02 | order A1002 (45 days) | not | not |
| control | t4_c03 | order A1004 (60 days) | not | not |
| control | t4_c04 | $30.00, 9 days | eligible | eligible |
| control | t4_c05 | $50.00, exactly 10 days (boundary) | eligible | eligible |
| control | t4_c06 | payment methods (no refund) | - | - |

Checked offline, with no LLM (`tests/test_truth_tests_45.py`): the tool's output changes for 8/8 detectors and
0/6 controls.

**Why 8 detectors:** McNemar needs enough one-way flips. 8 of 8 gives p = 2 x 0.5^8 = 0.008; even if only 6 of
the 8 flip (the agent might not call the tool every time), p = 0.031 < 0.05. With 6 detectors one miss would
already make it NOT SIGNIFICANT (that is what happened with the 12-question known-bad subset).

**Commands** (the baseline on this suite first, then the bug):
```
python experiment_pipeline.py experiments/baseline.yaml --scenario-dir scenarios_test4
python promote_baseline.py runs/baseline_<that run> --note "baseline for scenarios_test4"
python experiment_pipeline.py experiments/tmp_test4_refund_10_days.yaml --scenario-dir scenarios_test4
```
**Estimate:** 2 runs x 14 scenarios: about 70 agent calls + 140 judge calls = about 210 calls, about 100k tokens,
at least about 16 minutes (token budget). About half a free-tier day.

**Expected:**
- baseline: all 14 pass (if a detector already fails on the baseline, it can't flip: check before the bug run)
- bug: DECLINED; the 8 detectors go pass -> fail (wrong "not eligible" answers), the 6 controls stay
- the judges that drop: correctness_gt (vs the answer key), likely groundedness stays high (the answer matches the
  buggy tool output: a faithful answer to a wrong tool, which only the answer key catches)
- diagnosis: probably PROMPT/UNCLEAR; worth noting that no cause says "tool bug" yet

**Afterwards:** `git rm experiments/tmp_test4_refund_10_days.yaml` (keep the runs + evidence/planted_bug_test.md).

## Test 5: paraphrases (exp_05 embeddings vs substring)

**Question it answers:** does better retrieval (embeddings) help when users don't use the FAQ's words?

**Questions** (`scenarios_test5/`, 12): 2 per FAQ topic (shipping, return policy, payment methods, PO box, customs),
none containing the topic words, plus 2 with no FAQ at all (gift wrapping, changing the delivery address), where
the right answer is "not known". Each has `expected_source`: a phrase found only in the right FAQ.

**Offline preview on the RAW questions** (no agent, no rewrite):

| retrieval | right FAQ found (of 10) | wrong FAQ returned |
|---|---|---|
| substring (baseline) | 0 | 0 |
| bm25 (exp_04) | 5 | 3 (customs for shipping, returns, address change) |
| embeddings (exp_05) | not measurable: `fastembed` is not installed | |

**Caveat from the earlier runs:** the agent rewrites its own search queries (e.g. it searches "shipping time"), so
the substring baseline may find far more than 0/10 in a real run. The difference may therefore be small.

**Measured two ways:** the judges (correctness_gt, context_relevance), and in plain code with the new
`python source_hits.py runs/<run>`: did any tool output contain the right FAQ's phrase? (and for the 2 "none"
questions: was no wrong FAQ fetched?)

**Needs a decision first:** exp_05 needs `fastembed` (requirements-rag.txt), which is NOT installed in the main
Python. Installing it changes the main Python. Options: install it there, or a separate environment like DeepEval.
Without it, Test 5 can still compare substring vs bm25 (exp_04).

**Commands:**
```
python experiment_pipeline.py experiments/baseline.yaml --scenario-dir scenarios_test5
python promote_baseline.py runs/baseline_<that run> --note "baseline for scenarios_test5"
python experiment_pipeline.py experiments/exp_05_embeddings.yaml --scenario-dir scenarios_test5   # needs fastembed
python experiment_pipeline.py experiments/exp_04_bm25.yaml --scenario-dir scenarios_test5         # optional
python source_hits.py runs/baseline_<ts> runs/exp_05_embeddings_<ts> runs/exp_04_bm25_<ts>
```
**Estimate:** about 85k tokens for baseline + exp_05 (2 x 12 scenarios), about 43k more with exp_04.

**Expected:** embeddings find more right sources than substring on the raw wording; the answers improve only where
the agent's own rewritten query still missed. 12 questions is too few for significance: report per question
(source_hits) as well as the verdict.

**Order across days:** Test 4 (about 100k) and Test 5 (about 85k) do not fit in one free-tier day together with
the pending exp_07 run (about 110k).
