# Multi-question state sharing on Qwen3.5-9B-OptiQ-4bit (2026-11-12)

Experimental branch result. Nothing here changes the default library or published numbers.

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

On Qwen3.5-0.8B (2026-11-10) the counts were 80 vs 75: 8 lost and 3 gained, p ≈ 0.23.

## Speed: one shared state, N questions

Benchmark: `benchmarks/fork_bench.py` with 40 courses (a 1,707-token shared trunk),
branches of about 200 tokens each, cold prefix cache, and median of 2 repeats.

| Questions | Today (`question_first`, one call each) | State shared, serial | State shared, batched fork | Speedup | Same top choice |
|---|---|---|---|---|---|
| 4 | 23.6 s | 8.8 s | 8.7 s | 2.7× | 4/4 |
| 16 | 92.8 s | 17.7 s | 18.3 s | 5.1× | 16/16 |

- Max absolute logit difference between forked and serial scoring is 0.375 to 0.5, from
  bf16 batched-vs-single kernels.
- Almost all of the gain comes from computing the state once. Batching the question
  branches adds nothing measurable on this machine. Serial reuse of a `state_first`
  prefix through the existing prefix cache should therefore give the same speedup.
  That last point is inferred and has not been run yet.
- JevBench gives every task its own state, so it measures layout accuracy only, not the
  multi-question scenario.

## Files

- `jevbench-9b-*-summary.json` and `jevbench-9b-*-results.jsonl`: per-task JevBench
  records from the pinned harness `609a664e`.
- `fork-bench-qwen35-9b.json`: raw speed and agreement rows.
