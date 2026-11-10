# JevBench public evaluation: 74.0% without additional fine-tuning

[Back to home](../README.md) · [Machine-readable report](../benchmarks/results/jevbench-20261015/report.json)

On October 15, 2026, MLX Direct Decider with the frozen
`mlx-community/Qwen3.5-9B-OptiQ-4bit` checkpoint answered **171/231 legacy public
JevBench tasks correctly (74.0%)** on an Apple M2 Max with 64 GiB unified memory.
The evaluated configuration uses the explicit `qa_question_slot` benchmark profile.
It requires **no additional fine-tuning, LoRA, trained decision head or
decision-specific model weights**. The original quantized checkpoint is unchanged.

This is a **self-reported, post-development public-set result**, not an official
leaderboard score or an unseen-test result. The current official methodology uses
1,500 decisions, including sealed tasks, and separate Capability/Composite
metrics; raw accuracy on these 231 tasks does not establish a current rank.
See [Benchmark Heaven's methodology](https://benchmarkheaven.com/jev-models/methodology).

## Results and paired comparison

| Legacy public tier | Original application prompt | Evaluated Q&A profile |
| :--- | ---: | ---: |
| Easy | 47/48 (97.9%) | **48/48 (100.0%)** |
| Original | 55/72 (76.4%) | **63/72 (87.5%)** |
| Hard | 55/111 (49.5%) | **60/111 (54.1%)** |
| **Total** | **157/231 (68.0%)** | **171/231 (74.0%)** |

The new profile fixes **23** previously incorrect decisions and introduces **9**
regressions: **14 additional correct answers, or +6.1 percentage points**.
The [report](../benchmarks/results/jevbench-20261015/report.json) includes all
47 changed predictions, including changes where both answers are wrong.
All 231 tasks are retained. Four native abstentions, one native no-match and one
input exceeding 4,096 tokens count as incorrect; there were no infrastructure
failures. Neither failed inputs nor rejected decisions were retried or filtered.

Measured serial in-process latency, excluding model load and the four synthetic
conformance calls, was **p50 1.214 s / p95 13.750 s**. The original configuration
measured 1.234 s / 11.283 s: accuracy improved, while p95 latency increased.
These runs mix fresh tasks and available prefix reuse; they are not same-page
latency guarantees. Local compute is unmetered, not claimed to be free.

## What changed

The question moves into the native `QUESTION` field, with a fixed user utterance.
Questions longer than its 2,048-character limit remain complete in
`USER_UTTERANCE`, referenced by a fixed pointer question. Nothing is truncated.
The [frozen Q&A policy](../benchmarks/jevbench/profiles.py) treats choices as answer
meanings, distinguishes informational questions from action commands, and keeps
false, zero, explicit unknown answers, no-match and abstention meaningful.

State wrapping, candidate order/descriptions, native single-token code logits,
conditional softmax, thresholds, quantization and model weights are unchanged.
No temperature was fitted. Returned distributions are uncalibrated and conditional
on the supplied business choices when the native engine selects an answer.
This dedicated serial benchmark adapter **does not replace application
action-selection defaults** and must not share a process with concurrent native
decision calls, because the experimental policy context temporarily changes a
module global.

Prompt selection used newly written synthetic diagnostics before the public
rerun. A first policy-only study improved development from 19/24 to 22/24 but tied
validation at 30/36. In the second study, the selected profile improved development
from 26/36 to 32/36 and validation from 27/36 to 28/36. The synthetic splits share
templates, and that one-case validation gain is weak generalization evidence.
The legacy public baseline had already been observed. The selected profile was
frozen before one public rerun and was not revised using that rerun's outcomes.
These facts limit claims about performance on new tasks.

## Reproduce the evaluated configuration

Use an Apple Silicon Mac and the project's pinned MLX dependencies. The measured
runtime was Python 3.13.2, MLX 0.31.2, MLX-LM 0.31.3, Transformers 5.9.0 and
Tokenizers 0.22.2. Model source:
[pinned OptiQ checkpoint](https://huggingface.co/mlx-community/Qwen3.5-9B-OptiQ-4bit/tree/76b3310ab7aa52a34303c66fc928b6d7239c860c),
mixed 4/8-bit affine, group size 64. The measured core revision was
[`7b1d941`](https://github.com/ItsOdeLeo/MLX-Direct-Decider/tree/7b1d9413a590156570e3f109adcfdaf4ec085f19).
The published adapter is extracted from the measured components with private
paths and metadata wording adjusted for portability; effective prompt and mapping
are preserved. Dataset, tokenizer, config and artifact hashes are in the report
and [manifest](../benchmarks/results/jevbench-20261015/manifest.json).

From an installed project checkout, with the pinned model already downloaded:

```sh
git clone https://github.com/fstandhartinger/jevbench.git runs/jevbench-upstream
git -C runs/jevbench-upstream checkout --detach 609a664e81f46e03731600bd7bd9f1b7a61257c5
export MLX_DIRECT_MODEL=/absolute/path/to/Qwen3.5-9B-OptiQ-4bit
PYTHONPATH="runs/jevbench-upstream:src:." python -m benchmarks.jevbench.run \
  --model "$MLX_DIRECT_MODEL" \
  --jevbench-root runs/jevbench-upstream \
  --profile qa_question_slot \
  --output runs/jevbench-qa-reproduction
```

Use `--profile released` and a different output directory to reproduce the original
application-prompt configuration. The runner checks the harness revision and all
three dataset hashes, excludes conformance calls, clears the prefix cache after
them, preserves declines and writes fresh raw evidence without overwriting a run.
The limits are 64 business choices, 4,096 prompt tokens, 256-token prefill chunks,
512 MiB/16-entry prefix cache and a zero margin threshold.

Published evidence: [Q&A per-task records](../benchmarks/results/jevbench-20261015/qa-results.jsonl),
[baseline per-task records](../benchmarks/results/jevbench-20261015/baseline-results.jsonl),
[synthetic diagnostics and hashes](../benchmarks/results/jevbench-20261015/manifest.json).
Legacy runtime metadata was omitted from the exported result rows because it
described the original bridge rather than the effective prompt configuration;
the measured fields remain intact. Original raw responses are archived locally.
