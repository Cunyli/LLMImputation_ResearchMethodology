"""使用 Azure GPT-4o-mini 对缺失文本进行填补并广播至其它缺失率版本。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import pandas as pd
from dotenv import load_dotenv
from tqdm.auto import tqdm

try:
    from src.imputation import LLMImputerConfig, LLMTextImputer
except ModuleNotFoundError as exc:  # pragma: no cover
    import importlib.util
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    module_path = PROJECT_ROOT / "src" / "imputation" / "llm_imputer.py"
    spec = importlib.util.spec_from_file_location("llm_imputer", module_path)
    if spec is None or spec.loader is None:
        raise exc
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("llm_imputer", module)
    spec.loader.exec_module(module)

    LLMImputerConfig = module.LLMImputerConfig  # type: ignore[attr-defined]
    LLMTextImputer = module.LLMTextImputer  # type: ignore[attr-defined]

MCAR_DIR = Path("data/parquet/mcar")
MAR_DIR = Path("data/parquet/mar")
OUTPUT_DIR = Path("data/parquet/imputed_llm")
TEXT_COLUMN = "text"
PRIMARY_RATE = "30"
SECONDARY_RATES = ("15", "05")
ALL_RATES = (PRIMARY_RATE,) + SECONDARY_RATES


def load_datasets(source_dir: Path, prefix: str) -> Dict[str, pd.DataFrame]:
    datasets: Dict[str, pd.DataFrame] = {}
    for rate in ALL_RATES:
        path = source_dir / f"train_text_{prefix}_{rate}.parquet"
        if path.exists():
            datasets[rate] = pd.read_parquet(path)
    return datasets


def save_dataframe(df: pd.DataFrame, prefix: str, rate: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"train_text_imputed_{prefix}_llm_{rate}.parquet"
    df.to_parquet(path, index=False)
    print(f"保存结果: {path}")


def propagate(primary: pd.DataFrame, target: pd.DataFrame, text_column: str = TEXT_COLUMN) -> pd.DataFrame:
    result = target.copy()
    mask = result[text_column].isna() | (result[text_column].astype(str).str.strip() == "")
    if not mask.any():
        return result
    aligned = primary.loc[result.index, text_column]
    result.loc[mask, text_column] = aligned.loc[mask]
    return result


def process_split(prefix: str, source_dir: Path, *, limit: int | None = None) -> None:
    datasets = load_datasets(source_dir, prefix)
    if PRIMARY_RATE not in datasets:
        print(f"跳过 {prefix}，未找到 {PRIMARY_RATE}% 缺失率的数据")
        return

    primary_df = datasets[PRIMARY_RATE]
    if limit is not None and limit > 0:
        primary_df = primary_df.head(limit)
        for rate in SECONDARY_RATES:
            if rate in datasets:
                datasets[rate] = datasets[rate].head(limit)

    print(f"\n========== 使用 LLM 处理 {prefix.upper()} ==========")
    config = LLMImputerConfig(text_column=TEXT_COLUMN, temperature=0.7)
    llm_imputer = LLMTextImputer(config)
    filled_primary = llm_imputer.impute(primary_df)
    save_dataframe(filled_primary, prefix, PRIMARY_RATE)

    for rate in tqdm(SECONDARY_RATES, desc=f"广播 {prefix.upper()}", unit="rate"):
        if rate not in datasets:
            continue
        propagated = propagate(filled_primary, datasets[rate])
        save_dataframe(propagated, prefix, rate)


def main() -> None:
    parser = argparse.ArgumentParser(description="使用 Azure GPT-4o-mini 对缺失文本进行填补")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="调试模式下仅处理指定数量的样本（按索引顺序）",
    )
    args = parser.parse_args()

    load_dotenv()
    process_split("mcar", MCAR_DIR, limit=args.limit)
    process_split("mar", MAR_DIR, limit=args.limit)


if __name__ == "__main__":
    main()
