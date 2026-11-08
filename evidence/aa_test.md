# Truth test A/A: baseline vs baseline

**Expected:** NO REAL DIFFERENCE (the same config run twice).
**Got:** NO REAL DIFFERENCE (McNemar p=0.375, Wilcoxon p=0.145).
**Result: PASS on 21 of 31 questions.** The other 10 (normal_02-05, rag_01-06) could not be compared: Groq's
free-tier daily token limit (200,000 tokens/day) ran out while judging the re-run, so those 10 answers have judge
errors and were excluded (listed by compare, not counted as pass or fail). The agent answered all 31 without errors.

**Key finding: judge noise.** edge_03's answer was word-for-word identical in both runs, yet the tool_efficiency
judge scored it **9** the first time and **2** the second.

## Runs
- Baseline (promoted): `runs/baseline_20261106_101856`: 31/31 judged, pass 25/31 (81%), mean 8.28, 0 errors, 0 judge errors.
- A/A re-run: `runs/baseline_20261106_111506`: 31/31 answered, 21/31 judged (10 judge errors, see below).
- Compared with: `python compare.py runs/baseline_20261106_111506` (0 API calls). Config identical, judge set identical.

## Key numbers
| | baseline | A/A run |
|---|---|---|
| paired scenarios | 21 (10 excluded: judge errors) | |
| up / down / same | | 2 / 4 / 15 (default band 1.0) |
| pass->fail / fail->pass | | 4 / 1 |
| McNemar p / Wilcoxon p | | 0.375 / 0.145 (median diff -0.20) |
| tokens | 39,726 | 40,273 (+1.4%) |
| mean latency | 5.50 s | 5.84 s |
| agent errors | 0 | 0 |

## What it shows
- **No false alarm:** the same config twice is called NO REAL DIFFERENCE.
- **Noise is real even at temperature 0:** only 13/31 answers were word-for-word identical. Single questions moved up to ±2.2.
- **Judge noise:** edge_03's answer was identical in both runs, yet tool_efficiency went 9 -> 2. gen_015: 2 -> 10.
  tool_efficiency is the noisiest judge (4 of the 6 moves).
- **Consequence:** the default band (1.0) is too narrow for this system. Per-question ▼ and their diagnosis are not
  trustworthy until `calibrate_noise.py --experiment experiments/baseline.yaml` has measured the real bands.
  The verdict itself was right: McNemar and Wilcoxon don't depend on the bands.

## Problems found along the way (and fixed)
1. **Groq daily token limit (infrastructure):** `tokens per day (TPD): Limit 200000, Used 199802`. 110 rate-limit
   retries; 10 scenarios (normal_02-05, rag_01-06) failed judging after 3 attempts. A full pipeline run is ~110k
   tokens, so the free tier fits ~1.8 full runs per day.
2. **compare.py bug (fixed, tested):** judge-error files carry no `answer_key_hash`, and the answer-key guard read that
   as "answer key changed", refusing the whole comparison. Judge-error pairs are excluded anyway, so the guard now
   checks only pairs that were scored on both sides (same fix in promote_baseline.py). A real answer-key change is
   still refused. Regression test added to tests/test_judge_experiment.py; offline suite 16/16 PASS.
