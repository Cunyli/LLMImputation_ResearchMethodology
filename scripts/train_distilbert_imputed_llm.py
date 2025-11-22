"""Batch-train DistilBERT on all LLM-imputed datasets with best-known hyperparameters."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--imputed-dir", type=str, default="imputed_llm")
    parser.add_argument("--test-file", type=str, default="test.parquet")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--additional-args", type=str, nargs=argparse.REMAINDER)
    return parser.parse_args()


def build_base_command(args: argparse.Namespace) -> list[str]:
    cmd: list[str] = [
        sys.executable,
        "scripts/train_distilbert_regression.py",
        "--data-root",
        str(args.data_root),
        "--test-file",
        args.test_file,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--max-length",
        str(args.max_length),
    ]

    if args.fp16:
        cmd.append("--fp16")

    if args.additional_args:
        cmd.extend(args.additional_args)

    return cmd


def main() -> None:
    args = parse_args()

    data_dir = args.data_root / args.imputed_dir
    if not data_dir.exists():
        raise FileNotFoundError(f"Imputed directory not found: {data_dir}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    datasets = sorted(data_dir.glob("*.parquet"))
    if not datasets:
        raise FileNotFoundError(f"No parquet datasets found under {data_dir}")

    base_cmd = build_base_command(args)
    summary: list[dict[str, object]] = []

    for dataset_path in datasets:
        run_output_dir = args.output_root / dataset_path.stem
        cmd = base_cmd + [
            "--train-file",
            f"{args.imputed_dir}/{dataset_path.name}",
            "--output-dir",
            str(run_output_dir),
        ]

        print(f"\n=== Training on {dataset_path.name} ===")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"Training failed for {dataset_path.name} with exit code {exc.returncode}")
            summary.append(
                {
                    "dataset": dataset_path.name,
                    "output_dir": str(run_output_dir),
                    "exit_code": exc.returncode,
                    "metrics": {},
                }
            )
            continue

        metrics_file = run_output_dir / "metrics.json"
        metrics: dict[str, float] = {}
        if metrics_file.exists():
            try:
                metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                print(f"Warning: metrics.json unreadable for {dataset_path.name}")

        summary.append(
            {
                "dataset": dataset_path.name,
                "output_dir": str(run_output_dir),
                "exit_code": 0,
                "metrics": metrics,
            }
        )

    summary_path = args.output_root / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nBatch training complete. Summary saved to", summary_path)


if __name__ == "__main__":
    main()

