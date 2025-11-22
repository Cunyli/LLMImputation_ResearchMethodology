"""统计工具：获取数据集与数据框的形状及缺失值信息。"""

from __future__ import annotations

from typing import Dict, Iterator, Tuple, Union

import pandas as pd
from datasets import Dataset


def dataset_shape(dataset: Dataset) -> Tuple[int, int]:
    """返回数据集的形状 (行数, 列数)。"""

    return dataset.num_rows, len(dataset.column_names)


def dataframe_shape(df: pd.DataFrame) -> Tuple[int, int]:
    """返回数据框的形状 (行数, 列数)。"""

    return df.shape[0], df.shape[1]


def _count_empty_strings(series: pd.Series) -> int:
    """统计序列中空字符串的数量（仅字符串类型参与统计）。"""

    if not (pd.api.types.is_string_dtype(series.dtype) or series.dtype == object):
        return 0

    non_null_series = series.dropna()
    return int(non_null_series.eq("").sum())


def dataframe_missing_summary(df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    """统计数据框每列的缺失情况。"""

    total_rows = len(df)
    summary: Dict[str, Dict[str, int]] = {}

    for column_name in df.columns:
        column_series = df[column_name]
        null_count = int(column_series.isna().sum())
        empty_count = _count_empty_strings(column_series)
        missing_total = null_count + empty_count
        available = max(total_rows - missing_total, 0)

        summary[column_name] = {
            "missing_null": null_count,
            "missing_empty": empty_count,
            "missing": missing_total,
            "available": available,
        }

    return summary


def missing_value_summary(dataset: Dataset) -> Dict[str, Dict[str, int]]:
    """统计每一列的缺失值数量与有效值数量。"""

    df_or_iter: Union[pd.DataFrame, Iterator[pd.DataFrame]] = dataset.to_pandas()
    if isinstance(df_or_iter, pd.DataFrame):
        return dataframe_missing_summary(df_or_iter)

    # `to_pandas` may yield chunks for large datasets; concatenate to a single frame.
    chunks = list(df_or_iter)
    if not chunks:
        return dataframe_missing_summary(pd.DataFrame())

    concatenated = pd.concat(chunks, ignore_index=True)
    return dataframe_missing_summary(concatenated)
