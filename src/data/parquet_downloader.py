"""利用 pandas 直接从 Hugging Face 读取 parquet 数据。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

import pandas as pd

from .stats import dataframe_missing_summary, dataframe_shape

_DEFAULT_SPLITS: Mapping[str, str] = {
    "train": "annotated/train-00000-of-00001.parquet",
    "test": "annotated/test-00000-of-00001.parquet",
}

_HF_PREFIX = "hf://datasets/toxigen/toxigen-data/"


def load_split_to_dataframe(split: str, path_map: Mapping[str, str] | None = None) -> pd.DataFrame:
    """从 Hugging Face 加载指定切分为 DataFrame。"""

    mapping = dict(path_map or _DEFAULT_SPLITS)
    if split not in mapping:
        available = ", ".join(sorted(mapping))
        raise KeyError(f"split '{split}' 不在映射中，可用切分: {available}")

    remote_path = _HF_PREFIX + mapping[split]
    return pd.read_parquet(remote_path)


def download_splits(
    output_dir: Path,
    splits: Mapping[str, str] | None = None,
    *,
    format: str = "parquet",
) -> Dict[str, Dict[str, Any]]:
    """下载多个切分至本地，返回每个切分的文件路径与统计信息。"""

    mapping = dict(splits or _DEFAULT_SPLITS)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_info: Dict[str, Dict[str, Any]] = {}
    for split_name, relative_path in mapping.items():
        df = load_split_to_dataframe(split_name, mapping)
        shape = dataframe_shape(df)
        missing = dataframe_missing_summary(df)

        if format == "parquet":
            output_path = output_dir / f"{split_name}.parquet"
            df.to_parquet(output_path, index=False)
        elif format == "csv":
            output_path = output_dir / f"{split_name}.csv"
            df.to_csv(output_path, index=False)
        else:
            raise ValueError("format 必须为 'parquet' 或 'csv'")

        saved_info[split_name] = {
            "path": output_path,
            "num_rows": shape[0],
            "num_cols": shape[1],
            "columns": list(df.columns),
            "missing_summary": missing,
        }
    return saved_info


def default_split_mapping() -> Dict[str, str]:
    """返回默认切分到远程文件的映射。"""

    return dict(_DEFAULT_SPLITS)
