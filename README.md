# MLX Direct Decider — MLX decision engine without fine-tuning

<p align="center">
  <img src="docs/assets/mlx-direct-decider-hero.en.svg" alt="MLX Direct Decider: local MLX decision flow on Apple Silicon without fine-tuning" width="1280" />
</p>

<p align="center"><strong>Your existing compatible MLX model → a local decision engine.</strong><br />No fine-tuning required. No extra decision-model weights.</p>

**MLX Direct Decider** is an open-source Python library for **training-free decisions from existing MLX models on Apple Silicon Macs**. Use it for local LLM action selection, classification, semantic routing, and enum / boolean decisions.

Reuse a compatible [MLX-LM](https://github.com/ml-explore/mlx-lm) checkpoint for text or an [MLX-VLM](https://github.com/Blaizzy/mlx-vlm) checkpoint for experimental **text + image decisions**. Supply application state, a natural-language request and allowed choices; receive a stable candidate ID, `no_match` or `abstain`. Integrate through a Python API, CLI or local text HTTP service.

**No additional fine-tuning, trained decision head or separate decision-model weights are required.** The full model processes the prompt normally; the library reads candidate logits from its unchanged original output head without generating a continuation. Compatible base-model weights are still needed.

> **JevBench public evaluation: 74.0% — 171/231 correct, without additional fine-tuning.**
> Qwen3.5-9B OptiQ mixed 4/8-bit · Apple M2 Max · evaluated Q&A prompt profile.
> Self-reported results on the legacy public set; not an official leaderboard score.
> [Results, baseline comparison and reproduction](docs/jevbench.md).

<p align="center">
  <img src="docs/assets/badges/apple-silicon.svg" alt="Apple Silicon" />
  <img src="docs/assets/badges/python.svg" alt="Python 3.11+" />
  <a href="LICENSE"><img src="docs/assets/badges/license.svg" alt="MIT License" /></a>
  <a href="CHANGELOG.md"><img src="docs/assets/badges/release.svg" alt="0.2 experimental, GitHub release" /></a>
</p>

<p align="center">
  <a href="#quick-start">Quick start</a> ·
  <a href="#models">Tested models</a> ·
  <a href="#blocks">Game demo</a> ·
  <a href="docs/jevbench.md">JevBench: 74.0%</a> ·
  <a href="#results">Measured results</a> ·
  <a href="#related-projects">AnyJev & related projects</a> ·
  <a href="#faq">FAQ</a> ·
  <a href="docs/evaluation.md">Methodology</a> ·
  <a href="#connect">Connect & support</a>
</p>

<p align="center">
  <a href="https://x.com/ItsOdeLeo"><img src="docs/assets/badges/follow-x.svg" alt="Follow 欧的Leo (@ItsOdeLeo) on X / Twitter" /></a>
  <a href="https://xhslink.com/m/18bjTTf180W"><img src="docs/assets/badges/follow-xiaohongshu.svg" alt="Follow 欧的Leo on Xiaohongshu, ID: ItsOdeLeo" /></a>
</p>

<p align="center"><strong>English</strong> · <a href="README.zh-CN.md">简体中文</a></p>

---

## MLX action selection and semantic routing

**MLX Direct Decider** uses the current application state, a user utterance, and dynamic allowed choices to select a stable business ID with a local model. No-match and abstention are explicit outcomes.

Embed it in an existing tool: open a feature, select a visible item, pause a player, or answer a boolean / enum question. Inference runs on your Mac with an existing checkpoint; no training is required.

> **Project and package:** The project is MLX Direct Decider. The distribution and CLI are `mlx-direct-decider`, with Python imports from `mlx_direct_decider`. Version 0.2 is experimental and is released on GitHub; it is not on PyPI.

| Capability | What it means for your application |
| :--- | :--- |
| **Dynamic choices** | Update allowed actions with the page; keep your own stable business IDs. |
| **Explicit rejection** | Return `no_match` or `abstain` when an action is not appropriate. |
| **Version protection** | Reject stale decisions after state changes; consume execution authorization once. |
| **Prefix reuse** | Reuse stable context while evaluating every new utterance. Never cache final answers. |
| **Small integration surface** | Python API, CLI, and a localhost HTTP service. |

Models can make mistakes. Candidate scores rank choices and are **not calibrated probabilities of correctness**. State guards do not establish semantic correctness.

<a name="quick-start"></a>

## Quick start

Requires **Apple Silicon, native ARM Python 3.11+**, and a local MLX-LM checkpoint. Start with the verified Qwen checkpoint in the [model table](docs/models.md).

Install the v0.2.0 release from GitHub:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install "mlx-direct-decider[mlx] @ git+https://github.com/ItsOdeLeo/MLX-Direct-Decider@v0.2.0"
export MLX_DIRECT_MODEL=/absolute/path/to/your/local/mlx-model
```

To work from source instead, clone the repository and run
`python -m pip install -e '.[mlx]'` in the checkout.

```python
import os
from mlx_direct_decider import Candidate, DecisionRequest, MLXDecisionEngine

engine = MLXDecisionEngine(os.environ["MLX_DIRECT_MODEL"])
result = engine.decide(DecisionRequest(
    state={"focused_window": "notes"},
    utterance="Close it",
    candidates=(Candidate("close.notes", "Close the open notes window"),),
    state_version=1,
))

print(result.status, result.candidate_id)
print(result.margin, result.timing)
```

The CLI uses the same request contract:

```sh
mlx-direct-decider decide --request examples/decision.json
```

Results include the candidate ID, raw logits, candidate scores, margin, state version, model identity, actual timing, and cache details. See the [Python API](docs/python-api.md) for booleans, state updates, and versioned execution.

## MLX, MLX-LM and MLX-VLM support

- **MLX:** shared compute foundation; no separate Metal backend is needed.
- **MLX-LM:** text decisions through `MLXDecisionEngine`, with bounded prefix reuse.
- **MLX-VLM:** text + image decisions through `MLXVLMDecisionEngine` or
  `mlx-direct-decider decide --backend mlx-vlm --image ...`, without fine-tuning.

Install image support with `python -m pip install -e '.[mlx,vlm]'`.
[Image quick start and real-model checks](docs/multimodal.md) include Qwen3-VL,
SmolVLM's mistakes, and the Gemma checkpoint loading limitation. VLM requests
currently recompute the full context; the HTTP/game demos still use text state.
See the [model guide](docs/models.md) for checkpoint-specific text coverage.

<a name="blocks"></a>

## A local model plays Blocks

![MLX Direct Decider Blocks: complete 20-move development run, shown at 4× playback](docs/assets/blocks-direct/preview.gif)

**Qwen3.5-9B · 20 model-selected placements · 6 cleared rows · 600 points.** Freshly recorded with the MLX Direct Decider name on October 14, 2026. The GIF shows the complete run at **4× playback**, including inference waits. The model chooses from every legal vertical-drop placement using structured board data and rule-computed outcomes; there is no best-move fallback. This is turn-based play, not screenshot understanding or frame-by-frame control.

[Run the game locally](docs/blocks.md) · [Full-speed video](docs/assets/blocks-direct/full-run.mp4) · [All decisions](docs/assets/blocks-direct/transcript.json)

This single illustrative development run reached its planned 20-piece limit without retries, browser errors or recording errors. Earlier attempts, including the initial refusal, remain in the [historical game report](docs/blocks.md#recorded-development-attempts). This is not a held-out game benchmark or a speed claim. Manual play can run on a static server; new AI decisions require the local MLX backend.

<a name="models"></a>

## All locally tested model checkpoints

**11 distinct checkpoints tested locally as of 2026-10-13 · Apple M2 Max · 64 GiB.**
All use existing model weights, with no additional fine-tuning or decision-specific
weights. The table records direct candidate scoring; completed runtime checks and
decision accuracy are separate facts.

| Local checkpoint | Backend / input | Recorded validation |
| :--- | :--- | :--- |
| `Qwen3.5-9B-OptiQ-4bit` | MLX-LM · text | **171/231 (74.0%)** on [JevBench public tasks, Q&A profile](docs/jevbench.md); **30/36** on the separate application-state extension. |
| `GLM-4.7-Flash-4bit` | MLX-LM · text | Full 28- and 36-case evaluations; **21/36 correct** on the extension. |
| `gemma-4-26b-a4b-it-4bit` | MLX-LM · text | Full 28- and 36-case evaluations; **31/36 correct** on the extension. |
| `Qwen3-4B` | MLX-LM · text | 3-case development smoke: **1/3 correct**; cache parity **3/3 passed**. |
| `NVIDIA-Nemotron-3-Nano-30B-A3B-OptiQ-4bit` | MLX-LM · text | 3-case development smoke: **1/3 correct**; cache parity **3/3 passed**. |
| `qwen3.6-35b-a3b-mlx-4bit` | MLX-LM · text | 3-case development smoke: **3/3 correct**; cache parity **3/3 passed**. |
| `gemma-4-e4b-it-4bit` | MLX-LM · text; MLX-VLM · image attempt | Text smoke: **2/3 correct**, cache parity **3/3 passed**. Image path: **blocked during model loading**. |
| `Llama-3.2-1B-Instruct-4bit` | MLX-LM · text | 3-case development smoke: **0/3 correct**; cache parity **3/3 passed**. |
| `Mistral-7B-Instruct-v0.3-mlx-4Bit` | MLX-LM · text | 3-case development smoke: **3/3 correct**; cache parity **3/3 passed**. |
| `Qwen3-VL-4B-Instruct-4bit` | MLX-VLM · text + images | 7 synthetic development cases: **7/7 correct**, including image changes, reversed choices and booleans. |
| `SmolVLM-500M-Instruct-4bit` | MLX-VLM · text + images | 7 synthetic development cases: **3/7 correct**; mistakes and candidate-order sensitivity recorded. |

The 36-case frozen evaluation, 3-case development checks and 7-case synthetic
vision checks use different inputs and should be read within their own scope.
The fully evaluated models also returned wrong actions; small smoke checks do
not establish general reliability or compatibility with every model in a family.

[Text model details and evidence](docs/models.md) ·
[Image model details and evidence](docs/multimodal.md) ·
[Full 36-case results](docs/extended-results.md)

<a name="results"></a>

## Measurements, with their limits

### JevBench public tasks: 171/231 (74.0%)

The evaluated Q&A prompt profile answered **171/231 tasks correctly (74.0%)**
with the frozen quantized Qwen3.5-9B checkpoint and no additional fine-tuning.
All five native declines and one input over the 4,096-token limit count wrong.
This is a self-reported, post-development comparison on the legacy public set,
separate from the application tests below and the current official leaderboard.
The evaluated Q&A profile is supplied as a standalone benchmark adapter;
application action-selection defaults are unchanged.
[Full results, exact prompt, per-task records and reproduction](docs/jevbench.md).

### Application-state evaluation: 36 cases

**New six-domain evaluation · Apple M2 Max · 64 GiB · 36 fictional English cases.**

Documents, calendar drafts, files, music queues, product comparisons, and settings. Three models receive the same frozen inputs, semantic prompt, and threshold, with four output methods each. These are the direct-scoring results:

| Local checkpoint | Exact decisions / 36 | Wrong actions / 30 | Same-page p50 / p95 |
| :--- | ---: | ---: | ---: |
| Qwen3.5-9B-OptiQ-4bit | 30 / 36 (83.3%) | 1 / 30 | 194.8 / 407.5 ms |
| Gemma 4 26B-A4B MoE | 31 / 36 (86.1%) | 1 / 30 | 168.9 / 556.5 ms |
| GLM-4.7-Flash-4bit | 21 / 36 (58.3%) | 8 / 30 | 170.6 / 330.5 ms |

The 36 cases contain 30 enum requests and six boolean decisions; action errors count enum requests only. **Timings require loaded weights and a reusable page prefix.** They do not describe startup or arbitrary new pages. See the [extended report](docs/extended-results.md) for every method, rejection and coverage rates, boolean quality, memory, and failures.

Gemma selected the wrong queue item; Qwen selected the wrong longest-battery product. GLM made more action errors. The earlier small sample's zero-error observation did not carry over to new cases. **These results do not establish reliable unattended actions, arbitrary-model compatibility, or fixed 100 ms performance.**

<details>
<summary><strong>Original 28 cases: previous results retained</strong></summary>

**Apple M2 Max · 64 GiB · 28 fictional English test cases · revised cache runtime.**

| Local checkpoint | Exact decisions | Wrong actions / enum requests | Same-page p50 / p95 |
| :--- | ---: | ---: | ---: |
| Qwen3.5-9B-OptiQ-4bit | **25 / 28 (89.3%)** | **0 / 26** | **171.7 / 176.4 ms** |
| GLM-4.7-Flash-4bit | 14 / 28 (50.0%) | 2 / 26 | 147.8 / 170.6 ms |
| Gemma 4 26B-A4B MoE · mixed 4/8-bit | 26 / 28 (92.9%) | 1 / 26 | 134.9 / 283.9 ms |

Accuracy includes 26 enum requests and two boolean decisions; action errors count enum requests only. These are separate recorded runs, not inputs to a cross-revision algorithm speedup claim. The [Gemma report](docs/gemma4-results.md) includes all four methods, three cache conditions, and every failure.

These timings require **loaded weights and a reusable page prefix**. Qwen's KV-cold p50 was **2,159.9 ms**; its first decision after a page update was **1,050.2 ms**. About 170 ms is not a per-request guarantee.

Qwen's three original-suite misses include rejection-status distinctions and a tied choice after reordering. Gemma selected the wrong filtered first item; GLM also produced incorrect actions. The original and extended sets are reported separately, not combined into one unseen-test score.

</details>

- **161 core tests passed:** contracts, state updates, stale decisions, single-use authorization, HTTP, and related behavior. These do not measure model semantics.
- **168 / 168 original-suite cache comparisons passed:** three models × 28 cases × two reuse conditions, each compared with fresh computation.
- **72 / 72 extended-suite cache comparisons passed:** Gemma's 36 new cases × two reuse conditions, with observed maximum logit and score differences of 0. Numerical agreement does not establish semantic correctness.
- **16 recorded browser scenarios checked:** actual DOM clicks and execution receipts. Decline cases accept either rejection outcome, unlike the strict quality set above.

<details>
<summary><strong>How was this tested?</strong></summary>

1. Write the original **16 dev / 28 test** cases and the separate **12 dev / 36 test** extension from scratch. Freeze their hashes; use no production exports. Version 0.1 validates English inputs only.
2. Freeze the original development-refined semantic prompt and threshold. This three-model campaign keeps them unchanged after seeing new test outputs.
3. Compare direct scoring, one-code generation, business-ID JSON, and option-code JSON. Retain every failed output.
4. Measure process startup, loaded weights with cold KV, a new utterance on the same page, and the first decision after a page update. Synchronize MLX work before stopping the timer.
5. Score model choice separately from execution. A guard blocking a bad action does not make the model correct.

Repeating 28 cases under several cache conditions does not create more independent samples. The historical one-code baseline reached 26/28, slightly better than the direct returned decision. Direct scoring did not dominate every quality and latency metric. A supplementary JSON-format experiment was informed by earlier test results and is explicitly labeled as such.

The [full report](docs/results.md) includes p50/p95, rejection, executable coverage, memory, versions, input sizes, reproduction commands, and original cache failures. Do not combine timings from different code revisions to calculate a speedup. See the [testing guide](docs/testing.md) for a step-by-step explanation.

</details>

<a name="replay"></a>
<a name="live-demo"></a>

<details>
<summary><strong>Developer integration example: browser actions and execution receipts</strong></summary>

The fictional Morrow Studio desktop demonstrates how to connect allowed actions to real DOM controls, validate state versions, and inspect execution receipts.

[Local setup and integration guide](docs/http-and-demo.md) · [Recorded eight-request walkthrough](docs/assets/live-demo-direct/README.md) · [Original browser transcript](docs/assets/browser-demo-direct/browser-transcript.json)

The [static evidence replay](docs/demo/index.html?lang=en) reads saved results without a model or server. Clone the repository and open `docs/demo/index.html` locally; GitHub displays HTML files as source. New requests require the local MLX service. This integration example controls allowlisted buttons in a fictional application; it does not navigate arbitrary websites.

</details>

## How a choice is made

```text
Application state + utterance + allowed choices
                       │
               Official MLX-LM model
                       │
           Final-position candidate logits
                       │
          selected(id) / no_match / abstain
                       │
       Application version check → execution → receipt
```

Choices map to tokenizer-verified single-token codes, then back to business IDs. Direct scoring reads a causal model's next-token logits without generating a JSON continuation. It retains the official quantized output head. Hybrid caches reuse only complete, valid prefix boundaries.

See the [architecture](docs/architecture.md) and [framework audit](docs/framework-audit.md). Version 0.2 focuses on **English, single-turn, single-step choices**. Experimental text + image decisions use the separate [MLX-VLM API](docs/multimodal.md). General chat, multi-step planning, arbitrary arguments, training, and GPU batching are outside its scope. **Chinese documentation does not mean Chinese model behavior has been validated.**

<a name="related-projects"></a>

## Related decision projects: AnyJev, OpenDecider, and Laya

Developers evaluating **AnyJev**, **OpenDecider**, **Laya**, or **Laya-MLX** may also be looking for local action selection, classification, or semantic routing. **MLX Direct Decider focuses on existing compatible causal MLX-LM checkpoints on Apple Silicon, with no fine-tuning or separate decision-model weights.** Application-defined choices are scored through the original model output head. AnyJev also offers a path without training; the implementations and score semantics differ.

| Project | Documented approach | Relationship to MLX Direct Decider |
| :--- | :--- | :--- |
| [AnyJev](https://github.com/nokia-applied-research/AnyJev) | A training-free `Decider` with option rotations and label-prior correction; a separate self-distilled `Tacit` model path. | Shared use of existing LLMs; MLX Direct Decider provides native MLX scoring and application execution guards, without AnyJev's L0 corrections. |
| [OpenDecider](https://github.com/manjunathshiva/opendecider) | Open-weight System 1 decision models with typed outputs and serving clients. | Related decision-model alternative; MLX Direct Decider uses its own request and execution contracts. |
| [Laya](https://github.com/NandhaKishorM/laya) · Convai Innovations | Encoder-based models for choice, ordered score, and yes/no questions. | Related typed-decision approach; MLX Direct Decider scores option-token logits from causal language models. |
| [Laya-MLX](https://github.com/mizorewww/laya-mlx) | Independent MLX runtime for Laya's encoder and decision heads on Apple Silicon. | Shared MLX platform and local decision use case; separate model architecture and API. |

Read the [AnyJev, OpenDecider, and Laya comparison](docs/decision-models.md) for source snapshots, integration boundaries, and evaluation considerations. These are related projects, not bundled backends or vendor partnerships.

<a name="faq"></a>

## Frequently asked questions

### How does MLX Direct Decider use MLX, MLX-LM and MLX-VLM?

MLX provides the array and compute framework; MLX-LM runs text models, and the optional MLX-VLM backend runs text/image models. Metal execution is handled by MLX. Image decisions use fresh context; the existing text backend retains prefix reuse. MLX Direct Decider adds application-supplied choices, candidate scoring, rejection outcomes, and versioned execution. Direct decisions read next-token logits for verified single-token option codes without generating a JSON continuation. See the [architecture](docs/architecture.md).

The full model still processes the input. Reading final-position logits does not mean running only the final layer or replacing the output head.

### Can I use MLX Direct Decider as a local semantic router?

Yes. Describe the currently allowed routes or actions as candidates with stable business IDs, supply the application state and user request, and inspect the selected ID. The host application defines and executes the actions. `no_match` and `abstain` handle unsuitable or ambiguous requests. See the [Python API](docs/python-api.md).

### How is MLX Direct Decider different from AnyJev?

Both reuse existing models without additional training. MLX Direct Decider focuses on native MLX-LM and experimental MLX-VLM inference on Apple Silicon, with dynamic action IDs, prefix reuse for text, and versioned execution. AnyJev's `Decider` adds option-rotation and label-prior corrections. MLX Direct Decider uses direct candidate scoring; its scores are not calibrated correctness probabilities. See the [MLX Direct Decider vs AnyJev comparison](docs/decision-models.md#mlx-direct-decider-vs-anyjev).

### How does MLX Direct Decider compare with AnyJev, OpenDecider, and Laya-MLX?

MLX Direct Decider adds bounded action selection to existing compatible causal MLX-LM checkpoints without fine-tuning or decision-specific weights. AnyJev's `Decider` also reuses existing LLMs without training and adds corrections for option ordering and label priors. OpenDecider and Laya publish trained decision-model families, and Laya-MLX runs Laya weights. Their users can load published checkpoints without training them themselves. See the [decision-model comparison](docs/decision-models.md).

### Does installing MLX Direct Decider make every MacBook model compatible?

Point MLX Direct Decider at a compatible local MLX-LM checkpoint using `--model` or `MLX_DIRECT_MODEL`. Its model loader is shared across architectures; each checkpoint still needs a supported model implementation, compatible tokenizer and chat template, accessible logits, and enough memory. Installation does not attach it to other apps or load GGUF models from Ollama. The [compatibility guide](docs/models.md) separates full evaluations, small runtime checks, and untested families.

### Which MLX models have been evaluated?

The [homepage model table](#models) lists all 11 locally tested checkpoints across
Qwen, GLM, Gemma, NVIDIA Nemotron, Llama, Mistral and SmolVLM, including text/image
paths and loading failures. The [text compatibility guide](docs/models.md) and
[multimodal guide](docs/multimodal.md) retain exact versions, complete results and
limits. Full evaluations and small smoke checks are identified separately.

### Do I need fine-tuning, separate decision-model weights or a cloud API?

No. Reuse a compatible local MLX-LM checkpoint, or an MLX-VLM checkpoint for experimental image decisions, on an Apple Silicon Mac. No cloud API, extra training, fine-tuning, new decision head, or decision-specific checkpoint is required. If you already have compatible base weights, reuse them. Otherwise obtain those base weights separately and pass their local directory to the engine; the package does not download weights.

### How do I install MLX Direct Decider?

Create a native ARM Python 3.11+ environment and run `python -m pip install "mlx-direct-decider[mlx] @ git+https://github.com/ItsOdeLeo/MLX-Direct-Decider@v0.2.0"`. Use `[mlx,vlm]` for image decisions. Set `MLX_DIRECT_MODEL` to your local checkpoint directory. Follow the [quick start](#quick-start) for a runnable example. Version 0.2 is experimental. To work from source, clone [ItsOdeLeo/MLX-Direct-Decider](https://github.com/ItsOdeLeo/MLX-Direct-Decider) and run `python -m pip install -e '.[mlx]'`.

### How fast is local MLX action selection?

Latency depends on the checkpoint, hardware, input, and reusable prefix. The [measured results](#results) distinguish loaded weights with a reusable page prefix from startup, cold KV, and page updates. Warm measurements are not a fixed latency guarantee or evidence of reliable unattended execution.

### Is MLX Direct Decider an official JEV implementation or a general chat assistant?

MLX Direct Decider is an independent JEV-inspired project at [ItsOdeLeo/MLX-Direct-Decider](https://github.com/ItsOdeLeo/MLX-Direct-Decider). It uses existing MLX-LM models, not JEV weights, and does not reproduce unpublished JEV training. Version 0.2 supports English, single-turn, single-step choices; general chat and multi-step planning are outside its scope. Experimental text/image choices are documented in the [multimodal guide](docs/multimodal.md).

## Documentation

| Guide | Contents |
| :--- | :--- |
| [Python API](docs/python-api.md) | Enum / boolean, result fields, and versioned execution |
| [Model compatibility](docs/models.md) | Checkpoints, quantization, dependencies, licenses, and limits |
| [AnyJev, OpenDecider, and Laya comparison](docs/decision-models.md) | Related decision projects, MLX runtimes, and integration boundaries |
| [Evaluation](docs/evaluation.md) · [Results](docs/results.md) | All baselines, raw evidence, versions, and failures |
| [Testing guide](docs/testing.md) | Methodology, results, and reproduction |
| [Local HTTP](docs/http-and-demo.md) | CLI, service, and actual browser operations |
| [Contributing](CONTRIBUTING.md) · [Releasing](docs/releasing.md) | Development, builds, and publication |

<details>
<summary><strong>Run development checks</strong></summary>

```sh
python -m pip install -e '.[dev]'
python scripts/check_docs.py
python -m pytest -m 'not model'
python -m ruff check src tests benchmarks scripts examples
```

Real-model tests require explicitly selected local weights. Run model workloads sequentially. GitHub CI runs core checks and distribution builds on Python 3.11 and 3.13; local MLX inference is verified separately.

</details>

<a name="connect"></a>

## Connect & support

Share use cases, reproduction results, and suggestions. Issues, documentation fixes, and new public test cases also support the project.

<p align="center"><strong>欧的Leo</strong></p>

<p align="center"><a href="https://x.com/ItsOdeLeo">X / Twitter · @ItsOdeLeo ↗</a> · <a href="https://xhslink.com/m/18bjTTf180W">Xiaohongshu · 欧的Leo ↗</a></p>

<p align="center">Xiaohongshu ID: <code>ItsOdeLeo</code></p>

<p align="center"><sub>Support the project with a star, a share, or a pull request.</sub></p>

## License & inspiration

[MIT](LICENSE). Independently maintained, inspired by [TypeSafe AI's JEV](https://docs.typesafe.ai/), and built on official [MLX-LM](https://github.com/ml-explore/mlx-lm). No affiliation or endorsement, no JEV weights, and no claim to reproduce unpublished RLCD.

Models and dependencies retain their own licenses; see [NOTICE](NOTICE.md). Recorded evidence retains former project names, source paths, and hashes as explained in [results](docs/results.md). See the [naming research](docs/naming.md) for the candidate name.

<p align="center"><sub>Local decisions. Visible evidence. Open source.</sub></p>
