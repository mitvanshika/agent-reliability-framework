# Financial DD experiment 1: 3 retrieved chunks instead of 6

**Question:** Financial DD grades every retrieved chunk with an LLM call, and chunk grading was **51% of its tokens**
in the baseline. Does reading 3 chunks instead of 6 make it cheaper without making it worse?
**Change (one setting):** `retrieval.final_top_k: 6 -> 3` (`experiments/fdd_exp_01_top_k3.yaml`, extends `fdd_baseline`).
**Got:** **NO REAL DIFFERENCE** in quality (0 pass/fail flips, McNemar p=1.000, Wilcoxon p=0.700), with
**22.8% fewer tokens** (102,648 vs 133,040) and **24.2% lower mean latency** (37.0 s vs 48.8 s).
**Result: cheaper at the same measured quality.** The cost saving is well outside cost noise: the A/A test moved
total tokens by only 5.3% with nothing changed (`fdd_aa_test.md`).

## Numbers
| | baseline (6 chunks) | experiment (3 chunks) |
|---|---|---|
| run | `fdd_baseline_20261124_135240` | `fdd_exp_01_top_k3_20261126_092851` |
| pass | 28/29 | 28/29 |
| mean final_score | 9.61 | 9.58 |
| groundedness (mean) | 10.00 | 9.97 |
| tokens (in / out) | 133,040 (86,689 / 46,351) | 102,648 (65,888 / 36,760) |
| mean / p95 latency | 48.8 s / 132.1 s | 37.0 s / 131.0 s |
| up / down / same (vs noise bands) | | 2 / 2 / 25 |
| pass->fail / fail->pass | | 0 / 0 |

Judge set generic_v2 on both sides; noise bands `noise/financial_dd_noise.json` (baseline + A/A). Compared with
`python compare.py runs/fdd_exp_01_top_k3_20261126_092851` (0 API calls).

## The questions that moved
- **fdd_q11** 10.0 -> 9.0 (band 1.0) and **fdd_q23** 10.0 -> 9.4 (band 0.5): marked down, both still pass. q11's move
  (correctness_gt 10 -> 6) is the **same move it made in the A/A test**, where nothing changed: judge noise that
  2-run bands are too narrow to absorb. q23 is 0.1 past the minimum band.
- **fdd_q18** (predicted at risk: needs several facts) 8.4 -> 8.8; **fdd_q20** (the unstable question, band 5.0)
  9.1 -> 8.3: both inside noise.
- **fdd_q15** 9.1 -> 10.0 and **fdd_q25** 8.6 -> 10.0 up: q25 is a known judge-noise question (same answer,
  different scores in the A/A).

## Cost per question (compare's question cost table)
- With 3 chunks, **fdd_q17**'s first search was not enough, so Financial DD's **self-healing stepped in**: 2 -> 5
  calls, 5,735 -> 10,352 tokens, and the answer still scored 10. Fewer chunks make the safety net work more often.
- **fdd_q27** 9,810 -> 3,517 tokens (8 -> 3 calls). Single-question token changes swing by tens of percent even
  with nothing changed (A/A: q24 +36%, q15 -29%), so read the totals, not single rows.

## Limits
- One run of the experiment; a second run would confirm it (~100k Financial DD tokens).
- Noise bands from 2 runs underestimate the spread (q11).
- 29 questions detect only large quality changes; a drop of one question in thirty would not reach significance.

## Recommendation for Financial DD
`final_top_k: 3` looks like a free ~23% saving on this eval set. Worth confirming with one more run, and on harder
questions, before changing the default.
