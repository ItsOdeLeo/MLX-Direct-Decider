"""Compare multi-question scoring strategies on one shared state.

Strategies, all on the same local MLX model:
  today    current library: one ``backend.score`` per question with the prompt
           cache on (question-first layout, so only the system prompt is shared);
           includes its own tokenization. The other strategies exclude it
           (reported once as tokenize_ms).
  serial   state-first layout, state prefilled once, branches scored one by one
  forked   state-first layout, state prefilled once, branches scored in one batch

Also reports agreement between serial and forked scores (max |logit diff| and
whether the top choice matches). Example:

    python -m benchmarks.fork_bench --model ~/models/Qwen3-0.6B-4bit --items 40
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

from jev_mlx.backends.mlx_lm import MLXLMBackend
from jev_mlx.fork import prepare_forked, score_forked, score_serial
from jev_mlx.types import Candidate, DecisionRequest

TOPICS = ["design", "astronomy", "cooking", "history", "music", "biology", "finance", "poetry"]
WORDS = ["Moss", "Copper", "Amber", "Velvet", "Granite", "Lantern", "Harbor", "Willow",
         "Saffron", "Quartz", "Meadow", "Cobalt", "Ember", "Juniper", "Tidal", "Orchid"]
NOUNS = ["Architecture", "Constellations", "Kitchens", "Empires", "Rhythms", "Cells",
         "Markets", "Verses", "Gardens", "Machines"]


def make_state(items: int) -> dict:
    courses = []
    for i in range(items):
        courses.append({
            "id": f"c{i}",
            "title": f"{WORDS[i % len(WORDS)]} {NOUNS[(i * 7) % len(NOUNS)]} {i // len(WORDS) + 1}",
            "topic": TOPICS[i % len(TOPICS)],
            "minutes": 20 + (i * 13) % 70,
            "progress": (i * 37) % 101,
            "downloaded": i % 3 == 0,
        })
    return {"screen": "library", "focused_window": "library", "open_windows": ["library"],
            "sort": "curated", "filter": "all", "visible_courses": courses}


def make_questions(state: dict, n: int) -> list[DecisionRequest]:
    courses = state["visible_courses"]
    out = []
    for q in range(n):
        target = courses[(q * 5) % len(courses)]
        window = [courses[(q * 5 + k) % len(courses)] for k in range(4)]
        if q % 3 == 2:
            out.append(DecisionRequest(
                state=state,
                utterance=f"Have I already finished {target['title']}?",
                question=f"Is the course {target['title']} fully completed?",
                kind="boolean",
                candidates=(Candidate("yes", "The course is completed.", True),
                            Candidate("no", "The course is not completed.", False)),
            ))
            continue
        utterance = (f"Play {target['title']}" if q % 3 == 0
                     else f"Download the {target['topic']} course called {target['title']}")
        verb = "Play" if q % 3 == 0 else "Download"
        cands = tuple(Candidate(f"{verb.lower()}_{c['id']}", f"{verb} the course titled {c['title']}.")
                      for c in window) + (Candidate("close_library", "Close the course library."),)
        out.append(DecisionRequest(state=state, utterance=utterance, candidates=cands))
    return out


def timed(fn, mx):
    mx.synchronize()
    start = time.perf_counter()
    result = fn()
    mx.synchronize()
    return (time.perf_counter() - start) * 1000, result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--items", type=int, default=40, help="courses in the shared state")
    parser.add_argument("--questions", default="1,2,4,8,16")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--json", help="write raw results here")
    args = parser.parse_args()

    backend = MLXLMBackend(args.model, max_prompt_tokens=32768)
    mx = backend.mx
    state = make_state(args.items)
    rows = []
    for n in [int(x) for x in args.questions.split(",")]:
        requests = make_questions(state, n)

        def today():
            backend.clear_cache()
            return [backend.score(r, cache_key="bench").raw_scores for r in requests]

        tok_ms, prepared = timed(lambda: prepare_forked(backend.tokenizer, requests), mx)
        runs = {"today": [], "serial": [], "forked": [], "serial_warm": [], "forked_warm": []}
        for rep in range(args.repeats + 1):  # first round is warmup
            t_today, _ = timed(today, mx)
            t_serial, (serial, trunk) = timed(
                lambda: score_serial(backend, requests, prepared=prepared), mx)
            t_forked, (forked, _) = timed(
                lambda: score_forked(backend, requests, prepared=prepared), mx)
            t_sw, _ = timed(lambda: score_serial(
                backend, requests, trunk_cache=trunk, prepared=prepared), mx)
            t_fw, _ = timed(lambda: score_forked(
                backend, requests, trunk_cache=trunk, prepared=prepared), mx)
            if rep:
                for key, value in zip(runs, (t_today, t_serial, t_forked, t_sw, t_fw)):
                    runs[key].append(value)
        diffs = [abs(s[k] - f[k]) for s, f in zip(serial, forked) for k in s]
        disagreements = []
        for s, f in zip(serial, forked):
            top = sorted(s.values(), reverse=True)
            if max(s, key=s.get) != max(f, key=f.get):
                disagreements.append(round(top[0] - top[1], 4))  # serial top-2 margin
        agree = n - len(disagreements)
        row = {
            "questions": n,
            "trunk_tokens": len(prepared.trunk),
            "branch_tokens_mean": statistics.mean(len(b) for b in prepared.branches),
            "tokenize_ms": tok_ms,
            **{f"{k}_ms": statistics.median(v) for k, v in runs.items()},
            "max_abs_logit_diff": max(diffs),
            "top_choice_agree": f"{agree}/{n}",
            "disagreement_margins": disagreements,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    print(f"\nmodel={backend.identity['name']} mlx={backend.identity['dependencies']['mlx']} "
          f"mlx-lm={backend.identity['dependencies']['mlx-lm']} items={args.items} "
          f"repeats={args.repeats} (median ms)")
    print(f"{'N':>3} {'trunk':>6} {'branch':>6} {'today':>8} {'serial':>8} {'forked':>8} "
          f"{'x today':>8} {'x serial':>8} | {'warm ser':>8} {'warm fork':>9} {'x':>5} | "
          f"{'maxdiff':>8} agree")
    for r in rows:
        print(f"{r['questions']:>3} {r['trunk_tokens']:>6} {r['branch_tokens_mean']:>6.0f} "
              f"{r['today_ms']:>8.1f} {r['serial_ms']:>8.1f} {r['forked_ms']:>8.1f} "
              f"{r['today_ms'] / r['forked_ms']:>8.2f} {r['serial_ms'] / r['forked_ms']:>8.2f} | "
              f"{r['serial_warm_ms']:>8.1f} {r['forked_warm_ms']:>9.1f} "
              f"{r['serial_warm_ms'] / r['forked_warm_ms']:>5.2f} | "
              f"{r['max_abs_logit_diff']:>8.4f} {r['top_choice_agree']}")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"identity": backend.identity, "args": vars(args), "rows": rows}, fh, indent=2)


if __name__ == "__main__":
    main()
