"""工具函数：用于获取数据集的前若干行预览。"""

from __future__ import annotations

from typing import Any, List

from datasets import Dataset


def get_dataset_preview(dataset: Dataset, num_rows: int = 10) -> List[dict[str, Any]]:
    """返回指定数据集的前 ``num_rows`` 行。

    如果数据集长度不足 ``num_rows`` 行，则返回全部行。
    """

    if num_rows <= 0:
        raise ValueError("num_rows 必须为正整数")

    safe_num_rows = min(num_rows, dataset.num_rows)
    preview_dataset = dataset.select(range(safe_num_rows))
    return preview_dataset.to_list()
