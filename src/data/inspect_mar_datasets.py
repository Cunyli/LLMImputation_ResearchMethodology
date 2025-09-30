"""统计MAR缺失版本的关键信息。"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd

MAR_DIR = Path("data/parquet/mar")
RATES: Tuple[float, ...] = (0.05, 0.15, 0.30)
TRIGGER_GROUPS = {"asian", "muslim"}
TRIGGER_AUTHOR = "ai"
THRESHOLDS = {"toxicity_human": 4.0, "intent": 4.0}


def load_dataset(rate: float, split: str = "train") -> pd.DataFrame:
    filename = f"{split}_text_mar_{int(rate * 100):02d}.parquet"
    path = MAR_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"未找到文件: {path}")
    return pd.read_parquet(path)


def compute_statistics(df: pd.DataFrame) -> dict:
    total = len(df)
    missing_mask = df["text_is_missing"].astype(bool)
    missing_count = int(missing_mask.sum())
    missing_rate = missing_count / total if total else 0.0

    stats = {
        "total": total,
        "missing_count": missing_count,
        "missing_rate": missing_rate,
    }

    for column, threshold in THRESHOLDS.items():
        if column in df.columns:
            series = pd.to_numeric(df[column], errors="coerce")
            high_mask = series >= threshold
            stats[f"{column}_high_ratio"] = float(high_mask.mean())
            stats[f"{column}_high_missing_ratio"] = float(high_mask[missing_mask].mean()) if missing_count else 0.0

    if "target_group" in df.columns:
        group_mask = df["target_group"].isin(TRIGGER_GROUPS)
        stats["target_group_focus_ratio"] = float(group_mask.mean())
        stats["target_group_missing_ratio"] = float(group_mask[missing_mask].mean()) if missing_count else 0.0

    if "predicted_author" in df.columns:
        author_mask = df["predicted_author"] == TRIGGER_AUTHOR
        stats["author_focus_ratio"] = float(author_mask.mean())
        stats["author_missing_ratio"] = float(author_mask[missing_mask].mean()) if missing_count else 0.0

    return stats


def format_stats(rate: float, stats: dict) -> str:
    lines = [
        f"rate={rate:.2f}: total={stats['total']}, missing={stats['missing_count']} ({stats['missing_rate']:.2%})",
    ]

    for column in THRESHOLDS:
        high_key = f"{column}_high_ratio"
        miss_key = f"{column}_high_missing_ratio"
        if high_key in stats:
            lines.append(
                f"  - {column}>= {THRESHOLDS[column]}: overall={stats[high_key]:.2%}, among_missing={stats[miss_key]:.2%}"
            )

    if "target_group_focus_ratio" in stats:
        lines.append(
            f"  - target_group in {sorted(TRIGGER_GROUPS)}: overall={stats['target_group_focus_ratio']:.2%}, "
            f"among_missing={stats['target_group_missing_ratio']:.2%}"
        )

    if "author_focus_ratio" in stats:
        lines.append(
            f"  - predicted_author=='{TRIGGER_AUTHOR}': overall={stats['author_focus_ratio']:.2%}, "
            f"among_missing={stats['author_missing_ratio']:.2%}"
        )

    return "\n".join(lines)


def main() -> None:
    for rate in RATES:
        df = load_dataset(rate)
        stats = compute_statistics(df)
        print(format_stats(rate, stats))


if __name__ == "__main__":
    main()
