"""筛选阶段评估工具。

该模块实现以下功能：

1. 自动发现 ``data/parquet`` 下的插补数据集。
2. 为每个数据集构建 ``TF-IDF + 逻辑回归`` 分类管线。
3. 输出宏平均 F1、ROC-AUC 以及人口均等差异等核心指标。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


@dataclass
class DatasetSpec:
    """描述单个插补数据集的元信息。"""

    path: Path
    split: str
    mechanism: str
    imputer: str
    missing_rate: str

    @property
    def experiment_id(self) -> str:
        return "|".join(
            [
                self.split,
                self.mechanism,
                self.imputer,
                self.missing_rate,
            ]
        )


@dataclass
class ScreeningConfig:
    """筛选阶段评估配置。"""

    text_column: str = "text"
    label_column: str = "toxicity_human"
    group_column: str = "target_group"
    test_size: float = 0.2
    random_state: int = 42
    tfidf_max_features: int = 30000
    tfidf_min_df: int | float = 3
    tfidf_ngram_range: tuple[int, int] = (1, 2)
    logreg_penalty: str = "l2"
    logreg_c: float = 1.0
    logreg_solver: str = "saga"
    logreg_max_iter: int = 300
    positive_label_threshold: int = 3


def discover_datasets(root: Path) -> List[DatasetSpec]:
    """遍历 ``data/parquet`` 目录下的插补数据集。"""

    specs: List[DatasetSpec] = []

    baseline = root / "train.parquet"
    if baseline.exists():
        specs.append(
            DatasetSpec(
                path=baseline,
                split="train",
                mechanism="original",
                imputer="baseline",
                missing_rate="00",
            )
        )

    for directory in [root / "imputed", root / "imputed_llm"]:
        if not directory.exists():
            continue
        for path in sorted(directory.glob("train_text_*.parquet")):
            parts = path.stem.split("_")
            if len(parts) < 5:
                continue
            if parts[0] != "train" or parts[1] != "text":
                continue
            mechanism = parts[3]
            if directory.name == "imputed_llm":
                imputer_name = "llm"
                rate = parts[-1]
            else:
                imputer_name = "_".join(parts[4:-1])
                rate = parts[-1]
            specs.append(
                DatasetSpec(
                    path=path,
                    split="train",
                    mechanism=mechanism,
                    imputer=imputer_name,
                    missing_rate=rate,
                )
            )
    return specs


def _prepare_xy(
    df: pd.DataFrame,
    *,
    config: ScreeningConfig,
    label_bins: Optional[Iterable[float]] = None,
) -> tuple[pd.Series, np.ndarray]:
    """生成文本特征和离散标签。"""

    if label_bins is None:
        thresholds = [1.5, 2.5, 3.5, 4.5]
    else:
        thresholds = list(label_bins)

    y_continuous = df[config.label_column].to_numpy()
    y = np.digitize(y_continuous, thresholds).astype(int)
    x_text = df[config.text_column].fillna("[MISSING]")
    return x_text, y


def demographic_parity_difference(
    y_pred_binary: Sequence[int],
    groups: pd.Series,
) -> float:
    """计算人口均等差异 (Demographic Parity Difference)。"""

    groups = groups.reset_index(drop=True)
    preds = pd.Series(y_pred_binary).reset_index(drop=True)
    mask = groups.notna()
    groups = groups[mask]
    preds = preds[mask]
    if groups.empty:
        return float("nan")

    positive_rates: List[float] = []
    for group, values in preds.groupby(groups):
        if values.empty:
            continue
        positive_rates.append(values.mean())
    if not positive_rates:
        return float("nan")
    return float(max(positive_rates) - min(positive_rates))


class ScreeningEvaluator:
    """逻辑回归筛选评估器。"""

    def __init__(self, config: ScreeningConfig) -> None:
        self.config = config

        self.vectorizer = TfidfVectorizer(
            max_features=config.tfidf_max_features,
            min_df=config.tfidf_min_df,
            ngram_range=config.tfidf_ngram_range,
        )
        self.classifier = LogisticRegression(
            penalty=config.logreg_penalty,
            C=config.logreg_c,
            solver=config.logreg_solver,
            max_iter=config.logreg_max_iter,
            class_weight="balanced",
            n_jobs=-1,
        )
        self.pipeline = Pipeline(
            steps=[
                ("tfidf", self.vectorizer),
                ("clf", self.classifier),
            ]
        )

    def evaluate(self, df: pd.DataFrame) -> dict[str, float]:
        config = self.config
        x_text, y = _prepare_xy(df, config=config)
        (
            x_train,
            x_valid,
            y_train,
            y_valid,
            idx_train,
            idx_valid,
        ) = train_test_split(
            x_text,
            y,
            df.index,
            test_size=config.test_size,
            random_state=config.random_state,
            stratify=y,
        )

        self.pipeline.fit(x_train, y_train)

        y_pred = self.pipeline.predict(x_valid)
        y_proba = self.pipeline.predict_proba(x_valid)

        macro_f1 = f1_score(y_valid, y_pred, average="macro")
        roc_auc = roc_auc_score(y_valid, y_proba, multi_class="ovr")

        groups_valid = df.loc[idx_valid, config.group_column]
        positive_mask = (y_pred >= config.positive_label_threshold).astype(int)
        dp_diff = demographic_parity_difference(positive_mask, groups_valid)

        return {
            "macro_f1": macro_f1,
            "roc_auc": roc_auc,
            "dp_diff": dp_diff,
        }


def save_results(results: List[dict], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"screening_results_{timestamp}.csv"
    pd.DataFrame(results).to_csv(path, index=False)
    return path

