"""程序入口：下载并分析 toxigen 数据集。"""

from __future__ import annotations

from datasets import Dataset, DatasetDict

from data import dataset_shape, load_toxigen_dataset, missing_value_summary


def describe_dataset(dataset: Dataset) -> str:
    """返回数据集的简要概览。"""
    num_rows, num_cols = dataset_shape(dataset)
    columns = ", ".join(dataset.column_names)
    return f"Rows: {num_rows}\nColumns({num_cols}): {columns}"


def format_missing_summary(dataset: Dataset) -> str:
    """格式化缺失值统计信息。"""
    summary = missing_value_summary(dataset)
    lines = ["缺失值统计："]
    for column, counts in summary.items():
        lines.append(
            "- {name}: 空值 {null_cnt} 条，空字符串 {empty_cnt} 条，总缺失 {missing_cnt} 条，可用 {available_cnt} 条".format(
                name=column,
                null_cnt=counts["missing_null"],
                empty_cnt=counts["missing_empty"],
                missing_cnt=counts["missing"],
                available_cnt=counts["available"],
            )
        )
    return "\n".join(lines)


def main() -> None:
    dataset_dict: DatasetDict = load_toxigen_dataset()
    print("已成功加载 toxigen 数据集，包含以下数据切分：")

    for split_name, split_dataset in dataset_dict.items():
        print(f"\n[{split_name}]")
        print(describe_dataset(split_dataset))
        print(format_missing_summary(split_dataset))


if __name__ == "__main__":
    main()
