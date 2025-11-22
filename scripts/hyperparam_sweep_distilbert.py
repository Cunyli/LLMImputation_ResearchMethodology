"""Utility script to sweep epochs and batch sizes for DistilBERT training.

Example usage:
python scripts/hyperparam_sweep_distilbert.py \
  --data-root /path/to/data/parquet \
  --train-file imputed/train_text_imputed_mar_knn_05.parquet \
  --test-file test.parquet \
  --output-root outputs/distilbert_mar_knn_05_sweeps \
  --epochs 3 5 8 \
  --batch-sizes 16 24 32
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from itertools import product
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--train-file", type=str, required=True)
    parser.add_argument("--test-file", type=str)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, nargs="+", default=[3])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[16])
    parser.add_argument("--additional-args", type=str, nargs=argparse.REMAINDER)
    return parser.parse_args()


def run_training(
    base_cmd: list[str],
    epochs: int,
    batch_size: int,
    output_dir: Path,
) -> tuple[int, dict[str, float]]:
    cmd = base_cmd + [
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--output-dir",
        str(output_dir),
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        return exc.returncode, {}

    metrics_file = output_dir / "metrics.json"
    if not metrics_file.exists():
        return 0, {}

    try:
        with open(metrics_file, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    except json.JSONDecodeError:
        metrics = {}
    return 0, metrics


def main() -> None:
    args = parse_args()

    base_cmd: list[str] = [
        sys.executable,
        "scripts/train_distilbert_regression.py",
        "--data-root",
        str(args.data_root),
        "--train-file",
        args.train_file,
    ]

    if args.test_file:
        base_cmd.extend(["--test-file", args.test_file])

    if args.additional_args:
        base_cmd.extend(args.additional_args)

    args.output_root.mkdir(parents=True, exist_ok=True)

    summary: list[dict[str, object]] = []
    combinations: Iterable[tuple[int, int]] = product(args.epochs, args.batch_sizes)
    for epochs, batch_size in combinations:
        run_dir = args.output_root / f"epochs_{epochs}_batch_{batch_size}"
        exit_code, metrics = run_training(base_cmd, epochs, batch_size, run_dir)
        summary.append(
            {
                "epochs": epochs,
                "batch_size": batch_size,
                "exit_code": exit_code,
                "metrics": metrics,
            }
        )

    summary_path = args.output_root / "sweep_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    best = None
    for record in summary:
        metrics = record.get("metrics")
        if not metrics:
            continue
        if best is None or metrics.get("eval_rmse", float("inf")) < best["metrics"].get("eval_rmse", float("inf")):
            best = record

    if best:
        print("Best configuration:")
        print(json.dumps(best, indent=2))
    else:
        print("No successful runs with metrics found.")


if __name__ == "__main__":
    main()

