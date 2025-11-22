"""Aggregate and visualise DistilBERT batch-training results.

This script expects a directory containing multiple experiment folders produced
by `train_distilbert_imputed_llm.py`. Each experiment folder holds a
`metrics.json` file with eval/test metrics. The script builds comparison tables
and produces bar plots for easier inspection.

Example usage:
python scripts/analyze_distilbert_runs.py \
  --runs-root outputs/distilbert_imputed_runs \
  --output-dir analysis/distilbert_summary
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


@dataclass
class RunRecord:
    dataset: str
    output_dir: Path
    metrics: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--summary-file", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--group-pattern", type=str, default="imputed_{method}_{mechanism}_{ratio}")
    parser.add_argument("--metric", type=str, default="eval_rmse")
    parser.add_argument("--test-metric", type=str, default="test_rmse")
    return parser.parse_args()


def load_summary(args: argparse.Namespace) -> list[RunRecord]:
    summary_path = args.summary_file
    if summary_path is None:
        summary_path = args.runs_root / "summary.json"

    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")

    summary_raw = json.loads(summary_path.read_text(encoding="utf-8"))
    records: list[RunRecord] = []
    for item in summary_raw:
        metrics = item.get("metrics") or {}
        records.append(
            RunRecord(
                dataset=item.get("dataset", "unknown"),
                output_dir=Path(item.get("output_dir", "")),
                metrics=metrics,
            )
        )
    return records


def extract_components(dataset_name: str) -> dict[str, str]:
    stem = dataset_name.replace("train_text_", "")
    parts = stem.split("_")
    mapping = {
        "mechanism": "unknown",
        "method": "unknown",
        "ratio": "unknown",
    }

    if len(parts) >= 4:
        # e.g. imputed_mar_knn_05
        mapping["method"] = parts[1]
        mapping["mechanism"] = parts[2]
        mapping["ratio"] = parts[3]
    elif len(parts) >= 3:
        mapping["method"] = parts[1]
        mapping["mechanism"] = parts[2]
    return mapping


def records_to_dataframe(records: Iterable[RunRecord], metric: str, test_metric: str) -> pd.DataFrame:
    rows = []
    for record in records:
        if not record.metrics:
            continue
        components = extract_components(record.dataset)
        row = {
            "dataset": record.dataset,
            "output_dir": record.output_dir,
            "metric": record.metrics.get(metric),
            "test_metric": record.metrics.get(test_metric),
            **components,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def generate_plots(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    if "mechanism" in df and "ratio" in df:
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df, x="ratio", y="metric", hue="mechanism")
        plt.title("Validation metric by missingness mechanism and ratio")
        plt.savefig(output_dir / "metric_by_mechanism_ratio.png", bbox_inches="tight")
        plt.close()

    if "method" in df and "ratio" in df:
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df, x="ratio", y="metric", hue="method")
        plt.title("Validation metric by imputation method and ratio")
        plt.savefig(output_dir / "metric_by_method_ratio.png", bbox_inches="tight")
        plt.close()

    if "method" in df and "mechanism" in df:
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df, x="mechanism", y="metric", hue="method")
        plt.title("Validation metric by imputation method and mechanism")
        plt.savefig(output_dir / "metric_by_method_mechanism.png", bbox_inches="tight")
        plt.close()

    if not df["test_metric"].isna().all():
        plt.figure(figsize=(12, 6))
        sns.scatterplot(data=df, x="metric", y="test_metric", hue="method", style="mechanism")
        plt.title("Validation vs Test metrics")
        plt.savefig(output_dir / "metric_vs_test.png", bbox_inches="tight")
        plt.close()


def save_tables(df: pd.DataFrame, output_dir: Path) -> None:
    df_sorted = df.sort_values(by=["mechanism", "method", "ratio"])
    df_sorted.to_csv(output_dir / "metrics_summary.csv", index=False)

    pivot_mechanism = df.pivot_table(
        values="metric",
        index="method",
        columns="mechanism",
        aggfunc="mean",
    )
    pivot_mechanism.to_csv(output_dir / "metric_by_method_mechanism.csv")

    pivot_ratio = df.pivot_table(
        values="metric",
        index="method",
        columns="ratio",
        aggfunc="mean",
    )
    pivot_ratio.to_csv(output_dir / "metric_by_method_ratio.csv")


def main() -> None:
    args = parse_args()
    records = load_summary(args)
    df = records_to_dataframe(records, metric=args.metric, test_metric=args.test_metric)
    if df.empty:
        raise ValueError("No valid records with metrics found.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    generate_plots(df, args.output_dir)
    save_tables(df, args.output_dir)

    summary_txt = df.groupby(["method", "mechanism", "ratio"])["metric"].agg(["mean", "std", "count"])
    summary_txt.to_csv(args.output_dir / "metric_stats.csv")
    print("Analysis complete. Results saved to", args.output_dir)


if __name__ == "__main__":
    main()


