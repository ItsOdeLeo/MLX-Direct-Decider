# Multi-question state sharing on Qwen3.5-9B-OptiQ-4bit (2026-11-12)

Measurements behind the opt-in `prompt_layout="state_first"` option. The default layout and the published numbers are unchanged.

Machine: M2 Max, 64 GB. Software: mlx 0.31.2, mlx-lm 0.31.3. Every run went through the
shared GPU queue with a 40 GB MLX memory limit, and swap did not grow during any run.

## Accuracy: JevBench public tasks, `qa_question_slot` profile

| Layout | Correct / 231 | Macro accuracy | Brier |
|---|---|---|---|
| `question_first` (default) | **171** (74.0%) | 0.732 | 0.356 |
| `state_first` (shareable state prefix) | **168** (72.7%) | 0.710 | 0.383 |

- The `question_first` run reproduces the published 171/231 exactly.
- Paired comparison: `state_first` loses 9 tasks and gains 6. The exact McNemar test gives
  p = 0.61, so the difference is not statistically detectable.
- The approximate 95% interval of the paired difference is −4.6 to +2.0 points. A drop of
  a few points is therefore not ruled out.
- Brier and macro accuracy are slightly worse with `state_first`.

On Qwen3.5-0.8B (2026-11-10) the counts were 80 vs 75: 8 lost and 3 gained, p ≈ 0.23
(`jevbench-0.8b-*`).

## Speed: one shared state, N questions

Benchmark: `benchmarks/fork_bench.py` from the unmerged prototype branch
`feat/multi-question-fork` (commit `28d2f1f`, not part of this change; this table is
reference only) with 40 courses (a 1,707-token shared trunk),
branches of about 200 tokens each, cold prefix cache, and median of 2 repeats.

| Questions | Today (`question_first`, one call each) | State shared, serial | State shared, batched fork | Speedup | Same top choice |
|---|---|---|---|---|---|
| 4 | 23.6 s | 8.8 s | 8.7 s | 2.7× | 4/4 |
| 16 | 92.8 s | 17.7 s | 18.3 s | 5.1× | 16/16 |

- Max absolute logit difference between forked and serial scoring is 0.375 to 0.5, from
  bf16 batched-vs-single kernels.
- Almost all of the gain comes from computing the state once. Batching the question
  branches adds nothing measurable on this machine, so the shipped option reuses a
  saved state snapshot serially and does not batch.

## The shipped option: `prompt_layout="state_first"` (commit `e55c5b8`)

Benchmark: `benchmarks/state_first_check.py` with the same state and 16 questions. Each
question goes through `engine.decide` with tokenization and code checks included, and the
time is the median of 2 repeats.

| Model | Default layout | `state_first`, cache on | Speedup | Same top choice as default | Cached = uncached scores |
|---|---|---|---|---|---|
| Qwen3.5-0.8B | 7.20 s | 2.18 s | 3.3× | 16/16 | identical |
| Qwen3.5-9B-OptiQ-4bit | 93.0 s | 20.9 s | 4.5× | 15/16 | identical |

- Questions 2–16 all reported cache scope `shared_state`.
- In a first run at `c5ebd25` the 512 MiB LRU evicted the shared snapshot, and the 9B
  speedup fell to 2.7×. `e55c5b8` keeps the shared state in the system eviction class.
  `tests/test_cache_boundaries.py` reproduces the bug against the official LRU.
- The real-model invariants in `tests/test_model.py` pass on Qwen3.5-0.8B
  (`state-first-check-e55c5b8.log`).

- JevBench gives every task its own state, so it measures layout accuracy only, not the
  multi-question scenario.

## Files

- `jevbench-9b-*` and `jevbench-0.8b-*` (`*-summary.json`, `*-results.jsonl`): per-task
  JevBench records from the pinned harness `609a664e`. These were run with the prototype
  branch's `--layout` flag.
- `state-first-check-e55c5b8.log`: output of `benchmarks/state_first_check.py`.
- `fork-bench-qwen35-9b.json`: raw rows of the prototype fork benchmark (reference only).
