"""命令行工具：使用 pandas 下载 toxigen 数据集的 parquet 切分。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

from data.parquet_downloader import (
    default_split_mapping,
    download_splits,
    load_split_to_dataframe,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将 toxigen/toxigen-data 中的 parquet 切分下载到本地",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="输出目录，用于保存下载后的文件。",
    )
    parser.add_argument(
        "--format",
        choices=["parquet", "csv"],
        default="parquet",
        help="保存格式，默认 parquet。",
    )
    parser.add_argument(
        "--split",
        choices=list(default_split_mapping().keys()),
        help="仅下载指定切分；缺省则下载默认映射中的全部切分。",
    )
    parser.add_argument(
        "--show", action="store_true", help="仅展示单个切分的前几行，不保存。"
    )
    parser.add_argument(
        "--head",
        type=int,
        default=5,
        help="展示模式下的预览行数，默认 5。",
    )
    return parser.parse_args()


def format_missing_summary(summary: Dict[str, Dict[str, int]]) -> str:
    lines = ["缺失值统计："]
    for column, counts in summary.items():
        lines.append(
            "  - {name}: 空值 {null_cnt} 条，空字符串 {empty_cnt} 条，总缺失 {missing_cnt} 条，可用 {available_cnt} 条".format(
                name=column,
                null_cnt=counts["missing_null"],
                empty_cnt=counts["missing_empty"],
                missing_cnt=counts["missing"],
                available_cnt=counts["available"],
            )
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    mapping = default_split_mapping()

    if args.show:
        if not args.split:
            raise SystemExit("使用 --show 时必须通过 --split 指定切分")
        df = load_split_to_dataframe(args.split, mapping)
        print(df.head(args.head))
        return

    if args.split:
        selected = {args.split: mapping[args.split]}
        results = download_splits(args.output, selected, format=args.format)
    else:
        results = download_splits(args.output, mapping, format=args.format)

    for split_name, info in results.items():
        path = info["path"]
        num_rows = info["num_rows"]
        num_cols = info["num_cols"]
        columns = ", ".join(info["columns"])
        missing = info["missing_summary"]

        print(f"已保存 {split_name} -> {path}")
        print(f"Rows: {num_rows}\nColumns({num_cols}): {columns}")
        print(format_missing_summary(missing))


if __name__ == "__main__":
    main()
