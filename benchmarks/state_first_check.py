"""Real-model check for ``prompt_layout="state_first"``.

The check asks N different questions about one shared state:
  default      question_first (the default), one ``decide`` per question with the cache on
  state_first  one ``decide`` per question with the cache on, so the state is reused
  reference    state_first with ``use_cache=False`` for each question

It reports time per strategy as the median of the repeats. It requires raw scores to
be identical between state_first and reference, and it counts how often the
state_first top choice matches the default top choice. Both backends share one copy
of the weights.

    PYTHONPATH="src:." python -m benchmarks.state_first_check --model /path/to/model
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

from jev_mlx import Candidate, DecisionRequest, MLXDecisionEngine
from jev_mlx.backends.mlx_lm import MLXLMBackend

TOPICS = ["design", "astronomy", "cooking", "history", "music", "biology", "finance", "poetry"]
WORDS = ["Moss", "Copper", "Amber", "Velvet", "Granite", "Lantern", "Harbor", "Willow",
         "Saffron", "Quartz", "Meadow", "Cobalt", "Ember", "Juniper", "Tidal", "Orchid"]
NOUNS = ["Architecture", "Constellations", "Kitchens", "Empires", "Rhythms", "Cells",
         "Markets", "Verses", "Gardens", "Machines"]


def make_state(items: int) -> dict:
    courses = [{
        "id": f"c{i}",
        "title": f"{WORDS[i % len(WORDS)]} {NOUNS[(i * 7) % len(NOUNS)]} {i // len(WORDS) + 1}",
        "topic": TOPICS[i % len(TOPICS)],
        "minutes": 20 + (i * 13) % 70,
        "progress": (i * 37) % 101,
        "downloaded": i % 3 == 0,
    } for i in range(items)]
    return {"screen": "library", "focused_window": "library", "open_windows": ["library"],
            "sort": "curated", "filter": "all", "visible_courses": courses}


def make_questions(state: dict, n: int) -> list[DecisionRequest]:
    courses = state["visible_courses"]
    out = []
    for q in range(n):
        target = courses[(q * 5) % len(courses)]
        if q % 3 == 2:
            out.append(DecisionRequest.boolean(
                state=state,
                utterance=f"Have I already finished {target['title']}?",
                question=f"Is the course {target['title']} fully completed?",
            ))
            continue
        window = [courses[(q * 5 + k) % len(courses)] for k in range(4)]
        verb = "Play" if q % 3 == 0 else "Download"
        utterance = (f"Play {target['title']}" if q % 3 == 0
                     else f"Download the {target['topic']} course called {target['title']}")
        candidates = tuple(
            Candidate(f"{verb.lower()}_{c['id']}", f"{verb} the course titled {c['title']}.")
            for c in window
        ) + (Candidate("close_library", "Close the course library."),)
        out.append(DecisionRequest(state=state, utterance=utterance, candidates=candidates))
    return out


def top(result):
    return max(result.raw_scores, key=result.raw_scores.get)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--items", type=int, default=40)
    parser.add_argument("--questions", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()

    default_backend = MLXLMBackend(args.model)
    state_backend = MLXLMBackend.from_loaded_model(
        default_backend.model, default_backend.tokenizer,
        identity=default_backend.identity, prompt_layout="state_first")
    engines = {"default": MLXDecisionEngine(backend=default_backend),
               "state_first": MLXDecisionEngine(backend=state_backend)}
    mx = default_backend.mx
    questions = make_questions(make_state(args.items), args.questions)

    def run(name, use_cache):
        for backend in (default_backend, state_backend):
            backend.clear_cache()
        mx.synchronize()
        start = time.perf_counter()
        results = [engines[name].decide(q, use_cache=use_cache, cache_key="library")
                   for q in questions]
        return (time.perf_counter() - start) * 1000, results

    run("default", True)  # warm-up: kernels and allocator
    times = {"default": [], "state_first": [], "reference": []}
    for _ in range(args.repeats):
        t, default = run("default", True)
        times["default"].append(t)
        t, shared = run("state_first", True)
        times["state_first"].append(t)
        t, reference = run("state_first", False)
        times["reference"].append(t)

    print(json.dumps({
        "model": default_backend.identity.get("name"),
        "questions": args.questions,
        "state_tokens_shared": shared[-1].cache["shared_tokens"],
        "scopes": [r.cache["scope"] for r in shared],
        "median_ms": {k: statistics.median(v) for k, v in times.items()},
        "speedup_vs_default": statistics.median(times["default"])
        / statistics.median(times["state_first"]),
        "cached_equals_uncached": all(a.raw_scores == b.raw_scores
                                      for a, b in zip(shared, reference)),
        "top_choice_same_as_default": sum(top(a) == top(b) for a, b in zip(shared, default)),
        "status_same_as_default": sum(a.status == b.status for a, b in zip(shared, default)),
    }, indent=2))


if __name__ == "__main__":
    main()
