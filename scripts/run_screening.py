"""运行两阶段策略的第一阶段：TF-IDF + 逻辑回归筛选。"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

try:
    from evaluation import (
        DatasetSpec,
        ScreeningConfig,
        ScreeningEvaluator,
        discover_datasets,
        save_results,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - 支持脚本直接运行
    import importlib.util
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "screening",
        PROJECT_ROOT / "src" / "evaluation" / "screening.py",
    )
    if spec is None or spec.loader is None:
        raise exc
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("screening", module)
    spec.loader.exec_module(module)

    DatasetSpec = module.DatasetSpec  # type: ignore[attr-defined]
    ScreeningConfig = module.ScreeningConfig
    ScreeningEvaluator = module.ScreeningEvaluator
    discover_datasets = module.discover_datasets
    save_results = module.save_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/parquet"),
        help="数据集所在的根目录，默认 data/parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/screening"),
        help="存放筛选结果的目录，默认 outputs/screening",
    )
    parser.add_argument(
        "--max-datasets",
        type=int,
        default=None,
        help="可选，只评估前 N 个数据集（调试用）",
    )
    parser.add_argument(
        "--include-imputers",
        nargs="+",
        help="仅保留指定插补器，如 knn llm",
    )
    parser.add_argument(
        "--include-rates",
        nargs="+",
        help="仅保留指定缺失率，如 05 15 30",
    )
    parser.add_argument(
        "--include-mechanisms",
        nargs="+",
        help="仅保留指定缺失机制，如 mar mcar original",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="验证集占比，默认 0.2",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="随机种子，默认 42",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ScreeningConfig(
        test_size=args.test_size,
        random_state=args.random_state,
    )

    specs = discover_datasets(args.data_root)

    if args.include_imputers:
        allowed = {name.lower() for name in args.include_imputers}
        specs = [spec for spec in specs if spec.imputer.lower() in allowed]

    if args.include_rates:
        allowed_rates = {rate for rate in args.include_rates}
        specs = [spec for spec in specs if spec.missing_rate in allowed_rates]

    if args.include_mechanisms:
        allowed_mech = {name.lower() for name in args.include_mechanisms}
        specs = [spec for spec in specs if spec.mechanism.lower() in allowed_mech]

    if args.max_datasets is not None:
        specs = specs[: args.max_datasets]

    if not specs:
        raise SystemExit("未匹配到任何数据集，请调整筛选条件。")

    evaluator = ScreeningEvaluator(config)

    records = []
    for spec in tqdm(specs, desc="数据集", unit="dataset"):
        df = pd.read_parquet(spec.path)
        metrics = evaluator.evaluate(df)
        record = {
            "experiment_id": spec.experiment_id,
            "split": spec.split,
            "mechanism": spec.mechanism,
            "imputer": spec.imputer,
            "missing_rate": spec.missing_rate,
            **metrics,
        }
        records.append(record)

    if not records:
        print("未找到任何数据集，无需输出。")
        return

    results_path = save_results(records, args.output_dir)
    df_results = pd.DataFrame(records).sort_values("macro_f1", ascending=False)

    print("\n评估完成，Top 5 结果：")
    print(df_results.head().to_string(index=False))
    print(f"\n完整结果已保存至: {results_path}")


if __name__ == "__main__":
    main()

