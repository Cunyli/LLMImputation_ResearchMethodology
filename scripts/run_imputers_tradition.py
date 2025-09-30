"""对MCAR/MAR缺失数据运行多种文本填充器并输出进度。"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict

import pandas as pd
from tqdm.auto import tqdm

try:
    from src.imputation import (
        IterativeTextImputer,
        KNNTextImputer,
        SimpleTextImputer,
        TextImputerResult,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - 支持脚本直接运行
    import importlib.util
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "text_imputers", PROJECT_ROOT / "src" / "imputation" / "text_imputers.py"
    )
    if spec is None or spec.loader is None:
        raise exc
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("text_imputers", module)
    spec.loader.exec_module(module)

    IterativeTextImputer = module.IterativeTextImputer
    KNNTextImputer = module.KNNTextImputer
    SimpleTextImputer = module.SimpleTextImputer
    TextImputerResult = module.TextImputerResult

MCAR_DIR = Path("data/parquet/mcar")
MAR_DIR = Path("data/parquet/mar")
OUTPUT_DIR = Path("data/parquet/imputed")
TEXT_COLUMN = "text"
PRIMARY_RATE = "30"
SECONDARY_RATES = ("15", "05")
ALL_RATES = (PRIMARY_RATE,) + SECONDARY_RATES
DEFAULT_CONTEXT_COLUMNS = (
    "intent",
    "toxicity_human",
    "target_group",
    "predicted_group",
    "predicted_author",
    "actual_method",
)


def _text_missing_mask(series: pd.Series) -> pd.Series:
    values = series.fillna("")
    return values.str.strip() == ""


def _load_split(source_dir: Path, prefix: str) -> Dict[str, pd.DataFrame]:
    datasets: Dict[str, pd.DataFrame] = {}
    for rate in ALL_RATES:
        path = source_dir / f"train_text_{prefix}_{rate}.parquet"
        if path.exists():
            datasets[rate] = pd.read_parquet(path)
    return datasets


def _save(df: pd.DataFrame, prefix: str, imputer_name: str, rate: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"train_text_imputed_{prefix}_{imputer_name}_{rate}.parquet"
    path = OUTPUT_DIR / filename
    df.to_parquet(path, index=False)
    print(f"保存结果: {path}")


def _run_imputer(
    name: str,
    factory: Callable[[], object],
    df: pd.DataFrame,
    *,
    verbose: bool = False,
) -> TextImputerResult:
    print(f"\n>>> 开始 {name} 填充 (样本 {len(df)})")
    imputer = factory()
    result = imputer.fit_transform(df, verbose=verbose)
    return result


def _propagate_from_primary(
    base_df: pd.DataFrame,
    primary_imputed: pd.DataFrame,
    *,
    text_column: str,
) -> pd.DataFrame:
    output = base_df.copy()
    mask = _text_missing_mask(output[text_column])
    if not mask.any():
        return output
    aligned = primary_imputed.loc[output.index, text_column]
    output.loc[mask, text_column] = aligned.loc[mask]
    return output


def _imputer_factories(prefix: str) -> Dict[str, Callable[[], object]]:
    return {
        "simple_placeholder": lambda: SimpleTextImputer(strategy="placeholder", fill_value="[MISSING_TEXT]"),
        "simple_most_frequent": lambda: SimpleTextImputer(strategy="most_frequent"),
        "knn": lambda: KNNTextImputer(
            context_columns=DEFAULT_CONTEXT_COLUMNS,
            n_neighbors=5,
            weights="distance",
        ),
        # "iterative": lambda: IterativeTextImputer(
        #     context_columns=DEFAULT_CONTEXT_COLUMNS,
        #     max_iter=5,
        #     svd_components=128,
        #     max_features=2000,
        #     random_state=42,
        # ),
    }


def process_split(name: str, source_dir: Path, prefix: str) -> None:
    datasets = _load_split(source_dir, prefix)
    if PRIMARY_RATE not in datasets:
        print(f"跳过{name}，未找到{PRIMARY_RATE}% 缺失率的数据")
        return

    primary_df = datasets[PRIMARY_RATE]
    print(f"\n========== 处理 {name} ({prefix}) ==========")
    imputer_map = _imputer_factories(prefix)

    for imputer_name in tqdm(imputer_map, desc=f"{name} 填充器", unit="imputer"):
        factory = imputer_map[imputer_name]
        primary_result = _run_imputer(
            imputer_name,
            factory,
            primary_df,
            verbose=False,
        )
        _save(primary_result.data, prefix, imputer_name, PRIMARY_RATE)

        for rate in SECONDARY_RATES:
            if rate not in datasets:
                continue
            propagated = _propagate_from_primary(
                datasets[rate], primary_result.data, text_column=TEXT_COLUMN
            )
            _save(propagated, prefix, imputer_name, rate)


def main() -> None:
    process_split("MCAR", MCAR_DIR, "mcar")
    process_split("MAR", MAR_DIR, "mar")


if __name__ == "__main__":
    main()
