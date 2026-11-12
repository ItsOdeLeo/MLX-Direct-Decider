# Changelog

[Back to home](README.md)

## 0.2.0 — 2026-11-12

First release published to PyPI and GitHub Releases.

- Optional `prompt_layout="state_first"` for `MLXLMBackend`, `from_loaded_model` and
  `MLXDecisionEngine`. The state comes before the question and a system+state snapshot
  is shared, so several questions about one state prefill it once. Result `cache`
  info gains `layout` and `shared_tokens`, and the scope `shared_state`. The default
  `question_first` prompt and cache behavior are unchanged. See
  [Python API](docs/python-api.md#ask-several-questions-about-one-state).

- Renamed the project and repository to MLX Direct Decider. The distribution
  and CLI are `mlx-direct-decider`, with imports from `mlx_direct_decider`.
  Previous `mlx_zeroeffort_decider` / `mlx-zeroeffort-decider` and
  `jev_mlx` / `jev-mlx` callers remain compatible. Current pages and UI
  use the new name; benchmark evidence, recordings and previous release entries
  preserve their historical names. No fine-tuning or model behavior change is involved.

- Optional MLX-VLM text/image decisions with the original model head, no fine-tuning,
  per-call local images, fresh context, Python API and CLI. Recorded Qwen3-VL and
  SmolVLM smoke results include failures; the Gemma VLM loading limitation is explicit.
- Dedicated Idefics3 PIL processor adapter and contextual option-token verification
  for Mistral's SentencePiece boundary behavior. Existing text cache policy is retained.
- Bilingual [runtime and multimodal guide](docs/multimodal.md), synthetic examples,
  reproducible smoke commands and checkpoint-specific evidence.

## 0.1.0 — prepared release

Initial experimental JEV MLX implementation. Public repository/registry
publication is tracked separately from preparing this source release.

- JEV MLX Blocks: an original falling-block game with real local-model placement
  selection, manual play, stale-result guards, and complete run downloads.
  Six rule tests and fourteen mocked browser checks pass. A recorded Qwen
  development run placed 20 pieces and cleared four rows; the earlier refusal,
  all decisions, original videos, and the explicitly 4× GIF are retained in the
  [game report](docs/blocks.md). This is not a frozen gameplay benchmark.
- English-first README with a separate Chinese homepage, complete documentation
  in both languages, original title graphics, and owner-provided X and Xiaohongshu
  links. Language selection sits below the title/navigation area; X appears first.
  The display name is JEV MLX; package/API names remain `jev-mlx` / `jev_mlx`.
- A developer integration showcase, with a real eight-request Qwen recording
  and original-speed GIF in the integration documentation. The full video, every
  response and execution receipt, source hashes, and the recording recovery
  record are retained. Recording and conversion scripts reproduce the workflow;
  this walkthrough is distinct from the quality benchmark.
- An offline, bilingual recorded browser replay that preserves all 16 decisions
  and the separate state-change check. It runs without a model or server and is
  explicitly separate from live inference.
- Current public branding is JEV MLX, following the earlier JEVKit MLX and MLXJ
  names. The repository is [CoderInPajamas/JEV-MLX](https://github.com/CoderInPajamas/JEV-MLX). The
  distribution and CLI are `jev-mlx`, and the module is `jev_mlx`. Current model
  variables are `JEV_MLX_MODEL` and `JEV_TEST_MODEL`.
  Original benchmark metadata, source hashes, screenshots, and recordings retain
  their historical names and contents; see [results](docs/results.md).
- Installable Python package with dynamic enum and typed boolean requests,
  no-match and abstention, logits, restricted scores, raw margin, model identity,
  completed-work timing, and cache metadata.
- Official MLX-LM loading and final-position scoring through the original head,
  tokenizer-validated single-token codes, and local weights.
- Official prefix-cache snapshots, resource bounds, prewarm, state updates, and
  serialized inference.
- Versioned sessions with stale-result protection and single-use authorization,
  including forged/modified/replayed-result rejection.
- CLI and localhost HTTP service with Morrow Studio's fictional desktop,
  filterable/sortable library, simulated player, and real model-selected DOM
  button clicks with separate execution receipts.
- Core tests, opt-in real-model tests, fictional development/frozen fixtures,
  parity tools, one-code/JSON baselines, process cold-start measurements, and
  complete failure records.
- Verified Gemma 4 MoE text inference through official MLX-LM, with original-suite
  four-method/cache-condition evidence, separate process cold start, and a
  1,741-token rotating-cache integration check. All model-specific limits and
  wrong choices remain in the [Gemma report](docs/gemma4-results.md).
- Added an independently frozen 12-development/36-test fictional six-domain
  extension and completed all four output methods in the same-page condition
  for Gemma, Qwen, and GLM. Direct exact accuracy is 31/36, 30/36, and 21/36;
  wrong enum actions are 1/30, 1/30, and 8/30. Boolean answers are separate.
  [Extended results](docs/extended-results.md) retain every failure and explain
  why these results do not justify general unattended action execution.
- Fixture-directory selection, pre-load duplicate-schedule rejection, explicit
  cold-start schedule metadata, and kind-separated evidence reporting/audits.
  The latest non-model core suite passed 161 tests. The separate extended Gemma
  cache phase completed 108 decisions and passed all 72 comparisons; all 1,096
  planned campaign measurements are complete.
- Rebuilt the wheel, installed its MLX dependencies in a fresh environment outside
  the checkout, and verified all 14 runtime files against source and wheel bytes.
  A real Gemma CLI call selected `player.pause` for the public example. This is
  an installation smoke check, not an additional quality benchmark; see
  [distribution verification](benchmarks/results/extended-release-checks/distribution.json).
- Architecture, framework audit, model support, evaluation protocol, recorded
  results, contribution guidance, and release instructions.

See [results](docs/results.md) for actual runs and [model support](docs/models.md)
for checkpoint-specific validation. This release does not assert universal model
support, calibrated confidence, fixed latency, zero errors, or reproduction of
JEV weights/training. GPU batching, arbitrary-site automation, training, general
chat, and multi-step planning remain outside its scope.
