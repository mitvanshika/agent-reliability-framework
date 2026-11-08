# Truth test known-bad: exp_01_weak_prompt

exp_01 changes one setting from the baseline: the system prompt (prompts/v1.txt -> the weak prompt). The framework
should notice that the agent got worse.

**Result: PASS on all 31 scenarios (DECLINED).** The earlier 12-scenario subset was underpowered (FAIL, kept below).

## Full run (31 scenarios) - PASS

**Command:** `python experiment_pipeline.py experiments/exp_01_weak_prompt.yaml`
**Run:** `runs/exp_01_weak_prompt_20261107_164943`, compared with baseline `runs/baseline_20261106_101856`
(`baselines/configurable.json`)
**Noise bands:** measured (`noise/configurable_noise.json`, from_runs mode, 2 baseline runs; floor 0.5, default 1.0)
**Expected:** DECLINED, with drops in groundedness / context_relevance / correctness_gt (corrected expectation, see below).
**Got:** DECLINED.

| | |
|---|---|
| paired scenarios | 31 of 31 (none excluded, 0 agent errors, 0 judge errors) |
| pass rate | baseline 25/31 (81%) -> 19/31 (61%) |
| pass->fail / fail->pass | **6 / 0**: edge_03, gen_004, gen_007, gen_013, gen_014, rag_02 |
| McNemar (pass/fail flips) | b=6, c=0, exact p=0.031 -> DECLINED |
| Wilcoxon (paired scores) | 28 non-zero pairs, median diff -0.19, p=0.053 -> NO REAL DIFFERENCE (just above 0.05) |
| verdict | **DECLINED** (tests disagree; the more cautious verdict is used) |
| up / down / same (outside noise band) | 3 / 8 / 20 |
| down | edge_03, gen_004, gen_006, gen_007, gen_013, gen_015, normal_01, rag_02 |
| per-judge mean deltas | correctness_gt -1.81, groundedness -1.61, context_relevance -0.61, tool_match -0.68 (check, not in score), completeness +0.84, tool_efficiency +0.36 |
| tokens | 39,726 -> 29,458 (-25.8%) |
| latency | mean 5.50 s -> 5.19 s, p95 11.7 s -> 13.7 s |

### Why DECLINED when the two tests disagree
McNemar looks only at pass/fail flips: 6 scenarios went pass->fail and none went the other way. If the prompt
change did nothing, each flip would be a coin toss, and 6 of 6 in the same direction has probability
2 x 0.5^6 = 0.031, below 0.05. Wilcoxon looks at all 31 score differences, including many small ones that are
mostly judge noise, and lands just above the line (0.053).

When the tests disagree, compare.py uses the more cautious verdict, in the order
DECLINED < NO REAL DIFFERENCE < IMPROVED WITH REGRESSIONS < IMPROVED. "Cautious" means: assume the worse outcome.
For a reliability gate, missing a real regression (shipping a worse agent) is more costly than a false alarm
(someone reads the traces and overrides it), so one significant test showing a decline is enough to flag it.
The verdict reason records this: "McNemar -> DECLINED (p=0.0312); Wilcoxon -> NO REAL DIFFERENCE (p=0.0534);
tests disagree, the more cautious verdict is used".

Cheaper answers are part of the picture: 26% fewer tokens, because the weak agent searches less often and answers
from its own knowledge instead.

### Finding: the completeness judge rewards invented, complete-looking answers
| scenario | judge | baseline | weak prompt |
|---|---|---|---|
| gen_005 | completeness | 4 | **10** |
| gen_005 | groundedness | 9 | **1** |
| gen_005 | correctness_gt | 6 | 1 |
| rag_06 | completeness | 1 | **10** |
| rag_06 | groundedness | 9 | **1** |
| rag_06 | correctness_gt | 10 | 1 |

The baseline agent honestly said it had no information or escalated (low completeness, high groundedness). The weak
agent invented a full answer: completeness jumps to 10 while groundedness drops to 1. Completeness judges "did it
cover everything that was asked", not "is it true", so it rewards invention.

Effect: in the final score the +9 completeness partly cancels the -8 groundedness, so both scenarios stay inside
their noise band ("same": gen_005 5.43 -> 4.18, rag_06 6.43 -> 5.52) even though both answers are made up. It is
also why completeness is the only judge that went up on average (+0.84). This matches the earlier A/A finding that
completeness penalises honest escalation (gen_015, rag_06).
Possible fixes (not done): cap completeness when groundedness is low, or tell the completeness judge that "I don't
have that information" is complete when the tools returned nothing.

### Expectation check
| expected | got | |
|---|---|---|
| groundedness drops | -1.61 | as expected |
| correctness_gt drops | -1.81 | as expected |
| context_relevance drops | -0.61 | as expected (smaller) |
| tool_efficiency drops (original expectation) | **+0.36** | not seen |

tool_efficiency did not drop. Most likely this is judge noise plus the reason given for the subset: exp_01 changes
only the prompt, the tool descriptions stay detailed, and tool use barely changed (tool_match -0.68). The judge is
also known to be noisy: in the A/A test it scored the same edge_03 answer 9 and 2. A tool_efficiency drop is what
exp_02_vague_docstrings is meant to test.

### Diagnosis (diagnose.py) for the 8 down scenarios
PROMPT for edge_03, gen_006, gen_007, gen_015 and normal_01; RAG for rag_02; UNCLEAR for gen_004 (3 judges dropped
equally); MEASUREMENT UNSURE for gen_013. Note edge_03 is the scenario with known judge noise from the A/A test.

### Other runs
`runs/exp_01_weak_prompt_20261107_163034` is an earlier attempt from the same day that was not completed (22 of 31
judged, no comparison). It is not used here.

---

## Earlier attempt: 12-scenario subset - FAIL (underpowered)

**Command:** `python experiment_pipeline.py experiments/exp_01_weak_prompt.yaml --yes --scenarios normal_01 normal_02 normal_05 gen_002 gen_004 gen_008 gen_009 edge_01 edge_04 rag_01 rag_03 gen_011`
**Run:** `runs/exp_01_weak_prompt_20261106_184804`
**Got:** NO REAL DIFFERENCE (McNemar p=0.25, Wilcoxon p=0.067), leaning negative.

| | |
|---|---|
| paired scenarios | 12 (19 not run: subset, token budget) |
| up / down / same | 0 / 5 / 7 |
| pass->fail / fail->pass | 3 / 0 (gen_004, gen_009, rag_01) |
| pass rate on these 12 | baseline 100% -> 75% |
| per-judge deltas | groundedness -1.67, context_relevance -1.58, correctness_gt -1.50, tool_efficiency +0.17, completeness +0.50, tool_match 0 |
| tokens, same 12 scenarios | 15,601 -> 12,893 (-17.4%) |

### Why it failed
1. **Too little power.** b=3, c=0 gives McNemar p = 2 x 0.5^3 = 0.25. At least 6 one-way flips are needed for p<0.05,
   which is exactly what the full run produced.
2. **Subset selection (test design).** The 12 questions were chosen because the baseline passes them, but most are
   easy lookups where the tool returns the data and even the weak prompt copies it. The weak prompt's failure mode is
   inventing answers when a search finds nothing; those questions were mostly outside the subset.
3. **The tool_efficiency expectation was wrong for exp_01** (corrected 2026-11-06): exp_01 changes only the prompt.

What the traces showed: rag_01 searched once, found nothing and answered from its own knowledge (groundedness 1);
gen_004 never found the PO-box FAQ and still claimed "Yes, we ship to PO boxes" with an invented table.

### Fixed along the way
compare.py resource deltas compared the 12-scenario run with the full 31-scenario baseline ("tokens -67.5%"). They now
use only the scenarios both runs share (-17.4%), with a test in tests/test_compare.py.

## Lesson
Pick the scenario set for power, not convenience: a truth test needs enough scenarios where the change can show up.
The full suite did; the 12 "easy" ones could not.
