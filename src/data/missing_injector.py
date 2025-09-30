"""生成不同缺失场景的数据版本。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MCARConfig:
    """MCAR 缺失配置。"""

    rates: tuple[float, ...]
    seed: int = 42

    def sorted_rates(self) -> tuple[float, ...]:
        return tuple(sorted(self.rates))


def _normalize_framing_column(df: pd.DataFrame) -> pd.DataFrame:
    """将 framing 列的空字符串规范为缺失值。"""

    if "framing" in df.columns:
        df = df.copy()
        df["framing"] = df["framing"].replace("", pd.NA)
    return df


@dataclass(frozen=True)
class MARConfig:
    """MAR 缺失配置。"""

    rates: tuple[float, ...]
    seed: int = 123
    high_value_thresholds: Mapping[str, float] = field(
        default_factory=lambda: {"toxicity_human": 4.0, "intent": 4.0}
    )
    target_groups: tuple[str, ...] = ("asian", "muslim")
    author_values: tuple[str, ...] = ("ai",)
    base_weight: float = 1.0
    high_value_boost: float = 3.0
    group_boost: float = 2.0
    author_boost: float = 1.5

    def sorted_rates(self) -> tuple[float, ...]:
        return tuple(sorted(self.rates))


def _apply_text_mask(base: pd.DataFrame, mask: np.ndarray, rate: float) -> pd.DataFrame:
    """将缺失掩码应用到 text 列并添加辅助列。"""

    mcar_df = base.copy()
    if "text" not in mcar_df.columns:
        raise KeyError("数据中不存在 'text' 列，无法执行缺失注入")

    mcar_df["text_original"] = base["text"]
    mcar_df["text_is_missing"] = mask.astype(bool)
    mcar_df["text_missing_rate"] = 0.0
    if mask.any():
        mcar_df.loc[mask, "text"] = pd.NA
        mcar_df.loc[mask, "text_missing_rate"] = rate
    return mcar_df


def _nested_mcar_masks(length: int, rates: Iterable[float], *, rng: np.random.Generator) -> Dict[float, np.ndarray]:
    """基于同一随机排列构造嵌套的MCAR布尔掩码。"""

    if length == 0:
        return {rate: np.zeros(0, dtype=bool) for rate in rates}

    sorted_rates = sorted(rates)
    permutation = rng.permutation(length)

    masks: Dict[float, np.ndarray] = {}
    for rate in sorted_rates:
        if rate < 0 or rate > 1:
            raise ValueError("缺失率必须位于 [0, 1] 区间内")

        target = int(np.floor(rate * length))
        if 0 < rate and target == 0:
            target = 1

        mask = np.zeros(length, dtype=bool)
        if target > 0:
            indices = permutation[:target]
            mask[indices] = True
        masks[rate] = mask
    return masks


def create_mcar_versions(
    df: pd.DataFrame,
    config: MCARConfig,
) -> Mapping[float, pd.DataFrame]:
    """针对 text 列按配置生成多个 MCAR 缺失版本。"""

    base = _normalize_framing_column(df)
    rng = np.random.default_rng(config.seed)
    masks = _nested_mcar_masks(len(base), config.sorted_rates(), rng=rng)

    versions: Dict[float, pd.DataFrame] = {}
    for rate, mask in masks.items():
        versions[rate] = _apply_text_mask(base, mask, rate)
    return versions


def _compute_mar_weights(df: pd.DataFrame, config: MARConfig) -> np.ndarray:
    """根据特征构造 MAR 场景所需的权重。"""

    length = len(df)
    weights = np.full(length, config.base_weight, dtype=float)

    for column, threshold in config.high_value_thresholds.items():
        if column not in df.columns:
            continue
        series = pd.to_numeric(df[column], errors="coerce")
        mask = series >= threshold
        weights[mask.fillna(False).to_numpy()] += config.high_value_boost

    if config.target_groups and "target_group" in df.columns:
        mask = df["target_group"].isin(config.target_groups)
        weights[mask.to_numpy()] += config.group_boost

    if config.author_values and "predicted_author" in df.columns:
        mask = df["predicted_author"].isin(config.author_values)
        weights[mask.to_numpy()] += config.author_boost

    return np.clip(weights, 1e-6, None)


def _weighted_order(weights: np.ndarray, *, rng: np.random.Generator) -> np.ndarray:
    """基于权重生成稳定的抽样顺序。"""

    if len(weights) == 0:
        return np.array([], dtype=int)
    uniforms = rng.random(len(weights))
    keys = uniforms ** (1.0 / weights)
    return np.argsort(-keys)


def create_mar_versions(df: pd.DataFrame, config: MARConfig) -> Mapping[float, pd.DataFrame]:
    """针对 text 列生成多个 MAR 缺失版本。"""

    base = _normalize_framing_column(df)
    rng = np.random.default_rng(config.seed)
    weights = _compute_mar_weights(base, config)
    order = _weighted_order(weights, rng=rng)

    versions: Dict[float, pd.DataFrame] = {}
    length = len(base)
    for rate in config.sorted_rates():
        if rate < 0 or rate > 1:
            raise ValueError("缺失率必须位于 [0, 1] 区间内")

        target = int(round(rate * length))
        target = max(0, min(target, length))
        mask = np.zeros(length, dtype=bool)
        if target > 0:
            indices = order[:target]
            mask[indices] = True
        versions[rate] = _apply_text_mask(base, mask, rate)

    return versions


def summarize_column(df: pd.DataFrame, column: str) -> Dict[str, int]:
    """统计指定列中的缺失和空字符串数量。"""

    if column not in df.columns:
        raise KeyError(f"数据中不存在列: {column}")

    series = df[column]
    num_null = series.isna().sum()
    if series.dtype == object:
        non_null = series[series.notna()]
        num_empty = (non_null == "").sum()
    else:
        num_empty = 0
    return {"null": int(num_null), "empty_string": int(num_empty)}


def ensure_directory(path: Path) -> None:
    """确保目录存在。"""

    path.mkdir(parents=True, exist_ok=True)
