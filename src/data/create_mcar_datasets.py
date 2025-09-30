"""生成 text 列的 MCAR 缺失数据版本并输出统计信息。"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

if __package__:
    from .missing_injector import (
        MCARConfig,
        MARConfig,
        create_mar_versions,
        create_mcar_versions,
        ensure_directory,
        summarize_column,
    )
else:  # 直接运行脚本时的导入路径处理
    import importlib.util

    MODULE_PATH = Path(__file__).resolve().parent / "missing_injector.py"
    spec = importlib.util.spec_from_file_location("missing_injector", MODULE_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover - 出错保护
        raise ImportError(f"无法加载模块: {MODULE_PATH}")
    missing_injector = importlib.util.module_from_spec(spec)
    import sys

    sys.modules.setdefault("missing_injector", missing_injector)
    spec.loader.exec_module(missing_injector)

    MCARConfig = missing_injector.MCARConfig
    MARConfig = missing_injector.MARConfig
    create_mcar_versions = missing_injector.create_mcar_versions
    create_mar_versions = missing_injector.create_mar_versions
    ensure_directory = missing_injector.ensure_directory
    summarize_column = missing_injector.summarize_column

DEFAULT_RATES: tuple[float, ...] = (0.05, 0.15, 0.30)
DEFAULT_SEED = 42
INPUT_DIR = Path("data/parquet")
OUTPUT_DIR = INPUT_DIR / "mcar"
# MAR 输出目录
MAR_OUTPUT_DIR = INPUT_DIR / "mar"
# 目前仅为 train 数据集生成缺失版本，如需扩展可在这里添加更多切分
SPLITS: Iterable[str] = ("train",)


def _format_rate(rate: float) -> str:
    return f"{int(rate * 100):02d}"


def _print_basic_stats(dataset: str, rate: float, df: pd.DataFrame) -> None:
    shape = df.shape
    text_stats = summarize_column(df, "text")
    framing_stats = summarize_column(df, "framing") if "framing" in df.columns else None
    actual_missing_rate = df["text_is_missing"].mean() if "text_is_missing" in df.columns else float("nan")

    print(
        f"[{dataset}] rate={rate:.2f} -> shape={shape}, actual_rate={actual_missing_rate:.4f}, "
        f"text_null={text_stats['null']}, text_empty={text_stats['empty_string']}",
        end="",
    )
    if framing_stats:
        print(
            f", framing_null={framing_stats['null']}, framing_empty={framing_stats['empty_string']}"
        )
    else:
        print()


def main() -> None:
    ensure_directory(OUTPUT_DIR)
    ensure_directory(MAR_OUTPUT_DIR)

    for split_index, split in enumerate(SPLITS):
        input_path = INPUT_DIR / f"{split}.parquet"
        if not input_path.exists():
            raise FileNotFoundError(f"未找到输入文件: {input_path}")

        df = pd.read_parquet(input_path)
        config = MCARConfig(rates=DEFAULT_RATES, seed=DEFAULT_SEED + split_index)
        versions = create_mcar_versions(df, config)

        for rate, version in versions.items():
            output_path = OUTPUT_DIR / f"{split}_text_mcar_{_format_rate(rate)}.parquet"
            version.to_parquet(output_path, index=False)
            _print_basic_stats(split, rate, version)
            print(f"保存至: {output_path}")

        mar_config = MARConfig(rates=DEFAULT_RATES, seed=DEFAULT_SEED + split_index)
        mar_versions = create_mar_versions(df, mar_config)

        for rate, version in mar_versions.items():
            output_path = MAR_OUTPUT_DIR / f"{split}_text_mar_{_format_rate(rate)}.parquet"
            version.to_parquet(output_path, index=False)

            _print_basic_stats(split, rate, version)
            print(f"保存至: {output_path}")


if __name__ == "__main__":
    main()
