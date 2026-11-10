"""Real-model check for ``MLXLMBackend.from_loaded_model``.

1. Parity: a backend built from weights the caller loaded with ``mlx_lm.load``
   must give bit-identical raw scores and statuses to a normally constructed
   backend, on JevBench public tasks, both cold and with a warm prefix cache.
2. Cancellation: ``should_cancel`` stops a long prompt between prefill chunks,
   before the remaining chunks run, and the backend still scores correctly
   afterwards.

    PYTHONPATH="<jevbench checkout>:src:." python -m benchmarks.loaded_model_check \
        --model /path/to/Qwen3.5-0.8B --jevbench-root <jevbench checkout>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jevbench.tasks import load_jsonl
from mlx_lm import load

from benchmarks.jevbench.bridge import CapacityError, make_request
from mlx_direct_decider import MLXDecisionEngine
from mlx_direct_decider.backends.mlx_lm import MLXLMBackend


def pick_tasks(root: Path, per_tier: int):
    tasks = []
    for tier in ("easy", "original", "hard"):
        group = load_jsonl(str(root / "datasets" / "public" / f"{tier}.jsonl"))
        step = max(1, len(group) // per_tier)
        tasks.extend(group[::step][:per_tier])
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--jevbench-root", required=True, type=Path)
    parser.add_argument("--per-tier", type=int, default=7)
    args = parser.parse_args()

    normal = MLXLMBackend(args.model)
    model, tokenizer = load(args.model, tokenizer_config={"trust_remote_code": False})
    loaded = MLXLMBackend.from_loaded_model(model, tokenizer, identity=normal.identity)
    engines = {"normal": MLXDecisionEngine(backend=normal),
               "loaded": MLXDecisionEngine(backend=loaded)}

    # 1. Parity on JevBench tasks, cold (no cache) and warm (second call hits cache).
    compared, mismatches, skipped = 0, [], []
    for task in pick_tasks(args.jevbench_root, args.per_tier):
        try:
            request, _ = make_request(task.state, task.question, task.labels)
        except CapacityError:
            skipped.append(task.id)
            continue
        for phase, use_cache in (("cold", False), ("prime", True), ("warm", True)):
            out = {}
            for name, engine in engines.items():
                try:
                    out[name] = engine.decide(request, use_cache=use_cache,
                                              cache_key=f"check-{task.id}")
                except ValueError as exc:  # prompt over the token limit
                    out[name] = f"error: {exc}"
            if isinstance(out["normal"], str) or isinstance(out["loaded"], str):
                if out["normal"] != out["loaded"]:
                    mismatches.append((task.id, phase, "error differs"))
                skipped.append(f"{task.id}:{phase}")
                continue
            a, b = out["normal"], out["loaded"]
            compared += 1
            if (a.raw_scores != b.raw_scores or a.status != b.status
                    or a.cache["hit"] != b.cache["hit"]):
                mismatches.append((task.id, phase, a.status, b.status))
    print(json.dumps({"parity": {"compared_decisions": compared, "mismatches": mismatches,
                                 "skipped": skipped}}))

    # 2. Cancellation between prefill chunks of a long prompt.
    from mlx_direct_decider.prompt import prepare_prompt

    def prompt_tokens(t):
        try:
            return len(prepare_prompt(tokenizer, make_request(t.state, t.question, t.labels)[0])
                       .tokens)
        except (CapacityError, ValueError):
            return 0

    # Longest task that still fits the 4096-token limit: many 64-token prefill chunks.
    task = max(pick_tasks(args.jevbench_root, args.per_tier),
               key=lambda t: (prompt_tokens(t) <= normal.max_prompt_tokens) * prompt_tokens(t))
    request, _ = make_request(task.state, task.question, task.labels)
    checks = {"n": 0, "cancel_after": None}

    def should_cancel():
        checks["n"] += 1
        return checks["cancel_after"] is not None and checks["n"] > checks["cancel_after"]

    calls = {"n": 0}

    class CountingModel:
        """Delegates to the loaded model; counts forward calls (prefill chunks)."""

        def __call__(self, *a, **k):
            calls["n"] += 1
            return model(*a, **k)

        def __getattr__(self, name):
            return getattr(model, name)

    cancellable = MLXLMBackend.from_loaded_model(
        CountingModel(), tokenizer, identity=normal.identity, prefill_step_size=64,
        should_cancel=should_cancel)
    engine = MLXDecisionEngine(backend=cancellable)
    reference = engines["normal"].decide(request, use_cache=False)

    full = engine.decide(request, use_cache=False)  # learn the uncancelled call count
    full_calls, full_checks = calls["n"], checks["n"]

    results = {}
    for label, after in (("before_start", 0), ("mid_prefill", full_checks // 2)):
        calls["n"], checks["n"], checks["cancel_after"] = 0, 0, after
        try:
            engine.decide(request, use_cache=False)
            results[label] = {"raised": False}
        except InterruptedError:
            results[label] = {"raised": True, "model_calls": calls["n"]}
    checks["cancel_after"] = None
    after_cancel = engine.decide(request, use_cache=False)
    print(json.dumps({"cancel": {
        "task": task.id, "prompt_tokens": prompt_tokens(task), "prefill_step_size": 64,
        "uncancelled_model_calls": full_calls, "uncancelled_checks": full_checks,
        "results": results,
        "uncancelled_matches_normal_status": full.status == reference.status,
        "after_cancel_identical_to_uncancelled": after_cancel.raw_scores == full.raw_scores,
    }}))


if __name__ == "__main__":
    main()
