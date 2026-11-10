# Truth test A/A: baseline vs baseline

**Expected:** NO REAL DIFFERENCE (the same config run twice).
**Got:** NO REAL DIFFERENCE (McNemar p=0.250, Wilcoxon p=0.502), under judge set experiment_v2.
**Result: PASS on all 31 questions paired (0 excluded).** Both runs re-graded under experiment_v2 (completeness v2).
3 pass->fail, 0 fail->pass (edge_03, gen_014, rag_06); up / down / same 3 / 2 / 26. Pass rate 27/31 (baseline) vs
24/31 (A/A re-run), mean 8.40 vs 8.20: inside the noise, as the paired tests say.

**Key finding: judge noise.** edge_03's answer was word-for-word identical in both runs, yet the tool_efficiency
judge scored it **9** the first time and **2** the second.

## Current result (2026-11-09, experiment_v2)
| | baseline | A/A run |
|---|---|---|
| run | `baseline_20261106_101856` | `baseline_20261106_111506` |
| pass | 27/31 | 24/31 |
| mean final_score | 8.40 | 8.20 |
| paired scenarios | 31 (0 excluded) | |
| up / down / same | | 3 / 2 / 26 |
| pass->fail / fail->pass | | 3 / 0 (edge_03, gen_014, rag_06) |
| McNemar p / Wilcoxon p | | 0.250 / 0.502 (median diff 0.00) |

Re-compared with `python compare.py runs/baseline_20261106_111506` (0 API calls) after both runs were re-graded.

## History: first result (2026-11-06, experiment_v1)
Kept for the record. At the time, the A/A run had 10 judge errors from Groq's daily token limit.

**Got then:** NO REAL DIFFERENCE (McNemar p=0.375, Wilcoxon p=0.145).
**Result then: PASS on 21 of 31 questions.** The other 10 (normal_02-05, rag_01-06) could not be compared: Groq's
free-tier daily token limit (200,000 tokens/day) ran out while judging the re-run, so those 10 answers had judge
errors and were excluded (listed by compare, not counted as pass or fail). The agent answered all 31 without errors.

### Runs (then)
- Baseline (promoted): `runs/baseline_20261106_101856`: 31/31 judged, pass 25/31 (81%), mean 8.28, 0 errors, 0 judge errors.
- A/A re-run: `runs/baseline_20261106_111506`: 31/31 answered, 21/31 judged (10 judge errors, see below).
- Compared with: `python compare.py runs/baseline_20261106_111506` (0 API calls). Config identical, judge set identical.

### Key numbers (then)
| | baseline | A/A run |
|---|---|---|
| paired scenarios | 21 (10 excluded: judge errors) | |
| up / down / same | | 2 / 4 / 15 (default band 1.0) |
| pass->fail / fail->pass | | 4 / 1 |
| McNemar p / Wilcoxon p | | 0.375 / 0.145 (median diff -0.20) |
| tokens | 39,726 | 40,273 (+1.4%) |
| mean latency | 5.50 s | 5.84 s |
| agent errors | 0 | 0 |

### What it showed
- **No false alarm:** the same config twice is called NO REAL DIFFERENCE.
- **Noise is real even at temperature 0:** only 13/31 answers were word-for-word identical. Single questions moved up to ±2.2.
- **Judge noise:** edge_03's answer was identical in both runs, yet tool_efficiency went 9 -> 2. gen_015: 2 -> 10.
  tool_efficiency is the noisiest judge (4 of the 6 moves).
- **Consequence:** the default band (1.0) is too narrow for this system. Per-question ▼ and their diagnosis are not
  trustworthy until `calibrate_noise.py --experiment experiments/baseline.yaml` has measured the real bands.
  The verdict itself was right: McNemar and Wilcoxon don't depend on the bands.

### Problems found along the way (and fixed)
1. **Groq daily token limit (infrastructure):** `tokens per day (TPD): Limit 200000, Used 199802`. 110 rate-limit
   retries; 10 scenarios (normal_02-05, rag_01-06) failed judging after 3 attempts. A full pipeline run is ~110k
   tokens, so the free tier fits ~1.8 full runs per day.
2. **compare.py bug (fixed, tested):** judge-error files carry no `answer_key_hash`, and the answer-key guard read that
   as "answer key changed", refusing the whole comparison. Judge-error pairs are excluded anyway, so the guard now
   checks only pairs that were scored on both sides (same fix in promote_baseline.py). A real answer-key change is
   still refused. Regression test added to tests/test_judge_experiment.py; offline suite 16/16 PASS.
