"""Run the frozen legacy public JevBench set in a dedicated serial process."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from jevbench.adapters.conformance import check_adapter
from jevbench.budget import Ledger
from jevbench.runner import Runner
from jevbench.summarize import summarize
from jevbench.tasks import load_jsonl

from .bridge import JevAdapter, LocalModel
from .profiles import LAYOUTS, PROFILES, ProfileLocal

HARNESS_COMMIT = "609a664e81f46e03731600bd7bd9f1b7a61257c5"
DATASET_HASHES = {
    "easy": "231df3c2c8e88a1a8c137ebe85de96ba70fabd330849098ac7b3c52c70b7172b",
    "original": "5c2414edb3006b8bfcb70fda433f0f9ca015759433849f8d3104328a1f7c4180",
    "hard": "89e9e6becb33ed88c1de7d42dcc87531b2fb64cfaef4e1986faf7c37b3f80ebb",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local pinned MLX model directory")
    parser.add_argument("--jevbench-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--profile", choices=PROFILES, default="qa_question_slot")
    parser.add_argument("--layout", choices=LAYOUTS, default="question_first",
                        help="Prompt order; state_first is experimental")
    args = parser.parse_args()
    harness = args.jevbench_root.resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(harness), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != HARNESS_COMMIT:
        parser.error(f"Expected JevBench {HARNESS_COMMIT}; found {revision}")
    datasets = {}
    for tier, expected in DATASET_HASHES.items():
        path = harness / "datasets" / "public" / f"{tier}.jsonl"
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            parser.error(f"Dataset hash mismatch: {tier}")
        datasets[tier] = load_jsonl(str(path))
    tasks = [task for group in datasets.values() for task in group]
    output = args.output.resolve()
    if output.is_relative_to(harness):
        parser.error("Keep raw run outputs outside the upstream JevBench checkout")
    output.mkdir(parents=True, exist_ok=False)
    local = LocalModel(args.model)
    adapter = JevAdapter(ProfileLocal(local, args.profile, args.layout))
    adapter.name += "-" + args.profile
    if args.layout != "question_first":
        adapter.name += "-" + args.layout
    environment = local.metadata()
    environment.update(
        profile=args.profile,
        layout=args.layout,
        system_policy=PROFILES[args.profile],
        harness_commit=revision,
        dataset_sha256=DATASET_HASHES,
        scope="Legacy public tasks; self-reported, post-development diagnostic",
    )
    (output / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    conformance = check_adapter(adapter)
    (output / "conformance.txt").write_text(conformance.summary() + "\n")
    if not conformance.ok:
        raise RuntimeError(conformance.summary())
    local.engine.backend.clear_cache()
    ledger = Ledger(str(output / "ledger.jsonl"), cap_usd=0)
    runner = Runner(adapter, ledger, raw_dir=output / "raw", default_reserve_usd=0)
    records = runner.run_all(tasks, progress_every=10, results_path=output / "results.jsonl")
    report = summarize(tasks, records, ledger_charged=ledger.charged)
    report["profile"] = args.profile
    report["tiers"] = {}
    for tier, group in datasets.items():
        ids = {task.id for task in group}
        report["tiers"][tier] = summarize(
            group, [record for record in records if record["task_id"] in ids]
        )
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"correct": report["n_correct"], "total": report["n_scorable"],
                      "complete": report["complete"], "output": str(output)}))


if __name__ == "__main__":
    main()
