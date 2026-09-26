"""Aggregate local smoke runs; records no source text or checkpoint weights."""

import argparse
import json
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("local_data"))
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    names = [
        "loss-budget-control-v2",
        "loss-budget-gain003",
        "loss-budget-gain01",
        "loss-budget-noise-ensemble",
        "loss-budget-feedback",
        "loss-budget-final",
        "loss-budget-no-dropout",
        "loss-budget-compiled",
        "loss-budget-single-file",
    ]
    summary, trajectories = {}, {}
    for name in names:
        root = args.runs_root / name
        if not (root / "result.json").exists():
            continue
        run = json.loads((root / "run.json").read_text())
        result = json.loads((root / "result.json").read_text())
        rows = [
            json.loads(line)
            for line in (root / "metrics.jsonl").read_text().splitlines()
        ]
        positive = [row for row in rows if row["noise_std"] > 0]
        decreases = sum(b["noise_std"] < a["noise_std"] for a, b in zip(rows, rows[1:]))
        summary[name] = {
            "stage": (
                "final"
                if name in ("loss-budget-no-dropout", "loss-budget-compiled", "loss-budget-single-file")
                else "superseded development prototype"
            ),
            "run": run,
            "result": result,
            "first_noise_step": positive[0]["step"] if positive else None,
            "noise_decreases": decreases,
            "median_step_seconds_after_first_two": (
                statistics.median([row["step_seconds"] for row in rows[2:]])
                if len(rows) > 2
                else None
            ),
            "min_train_loss": min(row["train_loss"] for row in rows),
            "max_train_loss": max(row["train_loss"] for row in rows),
        }
        trajectories[name] = rows
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "trajectories.json").write_text(
        json.dumps(trajectories, indent=2) + "\n"
    )
    for name, row in summary.items():
        print(
            name,
            json.dumps(
                {
                    "final_validation": row["result"]["final_validation"]["loss"],
                    "ensemble": row["result"].get("ensemble"),
                    "first_noise_step": row["first_noise_step"],
                    "noise_decreases": row["noise_decreases"],
                    "peak_memory_gib": row["result"]["peak_memory_gib"],
                }
            ),
        )


if __name__ == "__main__":
    main()
