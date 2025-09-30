"""文本列缺失值填充器实现。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional, Sequence

import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.linear_model import BayesianRidge
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder
from tqdm.auto import tqdm


def _text_missing_mask(series: pd.Series) -> pd.Series:
    values = series.fillna("")
    mask = values.str.strip() == ""
    return mask


def _normalize_text(series: pd.Series) -> pd.Series:
    return series.replace("", pd.NA)


@dataclass
class TextImputerResult:
    data: pd.DataFrame
    filled_mask: pd.Series
    imputed_text: pd.Series


class SimpleTextImputer:
    def __init__(
        self,
        *,
        text_column: str = "text",
        strategy: str = "placeholder",
        fill_value: str = "[MISSING_TEXT]",
    ) -> None:
        self.text_column = text_column
        self.strategy = strategy
        self.fill_value = fill_value
        self._computed_value: Optional[str] = None

    def fit(self, df: pd.DataFrame, *, verbose: bool = False) -> "SimpleTextImputer":
        series = _normalize_text(df[self.text_column])
        mask = _text_missing_mask(series)
        observed = series[~mask].dropna()
        if self.strategy == "placeholder":
            self._computed_value = self.fill_value
        elif self.strategy == "most_frequent":
            if not observed.empty:
                self._computed_value = observed.value_counts().idxmax()
            else:
                self._computed_value = self.fill_value
        else:
            raise ValueError("strategy 必须为 'placeholder' 或 'most_frequent'")
        return self

    def transform(self, df: pd.DataFrame, *, verbose: bool = False) -> TextImputerResult:
        if self._computed_value is None:
            raise RuntimeError("请先调用 fit")
        series = _normalize_text(df[self.text_column])
        mask = _text_missing_mask(series)
        filled = series.copy()
        filled[mask] = self._computed_value
        result = df.copy()
        result[self.text_column] = filled
        return TextImputerResult(data=result, filled_mask=mask, imputed_text=filled)

    def fit_transform(self, df: pd.DataFrame, *, verbose: bool = False) -> TextImputerResult:
        return self.fit(df, verbose=verbose).transform(df, verbose=verbose)


class TextImputerBase:
    def __init__(
        self,
        *,
        text_column: str = "text",
        context_columns: Optional[Sequence[str]] = None,
        max_features: int = 3000,
        svd_components: int = 256,
        random_state: int = 42,
        placeholder: str = "[MISSING_TEXT]",
    ) -> None:
        self.text_column = text_column
        self.context_columns = tuple(context_columns) if context_columns else None
        self.max_features = max_features
        self.svd_components = svd_components
        self.random_state = random_state
        self.placeholder = placeholder

        self.vectorizer: Optional[TfidfVectorizer] = None
        self.reducer: Optional[TruncatedSVD] = None
        self.context_encoder: Optional[ColumnTransformer] = None
        self.embedding_dim: Optional[int] = None
        self.known_embeddings: Optional[np.ndarray] = None
        self.known_texts: Optional[np.ndarray] = None

    def fit(self, df: pd.DataFrame, *, verbose: bool = False) -> "TextImputerBase":
        text_series = _normalize_text(df[self.text_column])
        mask = _text_missing_mask(text_series)
        mask_array = mask.to_numpy()
        if verbose:
            print(f"[TextImputerBase] 编码文本：{self.text_column}")
        self._fit_text_encoder(text_series[~mask_array].astype(str))
        text_matrix, _ = self._transform_text(df)
        if verbose:
            print("[TextImputerBase] 编码上下文特征")
        context_matrix = self._fit_context(df)
        combined = self._combine(text_matrix, context_matrix)
        if verbose:
            print("[TextImputerBase] 训练缺失值填充器")
        self._fit_imputer(combined, verbose=verbose)
        observed_embeddings = text_matrix[~mask_array]
        self.known_embeddings = observed_embeddings.copy()
        self.known_texts = text_series[~mask].astype(str).to_numpy()
        return self

    def transform(self, df: pd.DataFrame, *, verbose: bool = False) -> TextImputerResult:
        if self.embedding_dim is None or self.known_embeddings is None:
            raise RuntimeError("请先调用 fit")
        text_series = _normalize_text(df[self.text_column])
        mask = _text_missing_mask(text_series)
        if verbose:
            print("[TextImputerBase] 转换文本并生成嵌入")
        text_matrix, text_missing_mask = self._transform_text(df)
        if verbose:
            print("[TextImputerBase] 转换上下文特征")
        context_matrix = self._transform_context(df)
        combined = self._combine(text_matrix, context_matrix)
        if verbose:
            print("[TextImputerBase] 执行填充推理")
        imputed = self._imputer_transform(combined, verbose=verbose)
        text_part = imputed[:, : self.embedding_dim]
        filled_series = text_series.copy()
        missing_idx = mask[mask].index
        if len(missing_idx) > 0:
            imputed_embeddings = text_part[text_missing_mask]
            filled_values = self._embeddings_to_text(imputed_embeddings)
            filled_series.loc[missing_idx] = filled_values
        result = df.copy()
        result[self.text_column] = filled_series
        return TextImputerResult(data=result, filled_mask=mask, imputed_text=filled_series)

    def fit_transform(self, df: pd.DataFrame, *, verbose: bool = False) -> TextImputerResult:
        return self.fit(df, verbose=verbose).transform(df, verbose=verbose)

    def _fit_text_encoder(self, texts: pd.Series) -> None:
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=(1, 2),
            token_pattern=r"(?u)\b\w+\b",
        )
        tfidf = self.vectorizer.fit_transform(texts)
        feature_dim = tfidf.shape[1]
        if feature_dim == 0:
            raise ValueError("训练文本为空，无法构建向量空间")
        if feature_dim <= 1 or feature_dim <= self.svd_components:
            self.reducer = None
            self.embedding_dim = feature_dim
        else:
            n_components = min(self.svd_components, feature_dim - 1)
            if n_components < 1:
                n_components = 1
            self.reducer = TruncatedSVD(n_components=n_components, random_state=self.random_state)
            self.reducer.fit(tfidf)
            self.embedding_dim = n_components

    def _transform_text(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if self.vectorizer is None or self.embedding_dim is None:
            raise RuntimeError("文本编码器尚未训练")
        series = _normalize_text(df[self.text_column]).fillna("")
        mask = _text_missing_mask(series)
        texts = series.mask(mask, "")
        tfidf = self.vectorizer.transform(texts.astype(str))
        if self.reducer is not None:
            array = self.reducer.transform(tfidf)
        else:
            array = tfidf.toarray()
        matrix = array.astype(np.float64, copy=False)
        mask_array = mask.to_numpy()
        matrix[mask_array, :] = np.nan
        return matrix, mask_array

    def _fit_context(self, df: pd.DataFrame) -> np.ndarray:
        columns = self._select_context_columns(df)
        if not columns:
            self.context_encoder = None
            return np.empty((len(df), 0), dtype=np.float64)
        numeric = [col for col in columns if pd.api.types.is_numeric_dtype(df[col])]
        categorical = [col for col in columns if col not in numeric]
        transformers = []
        if numeric:
            transformers.append(("num", "passthrough", numeric))
        if categorical:
            transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), categorical))
        self.context_encoder = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            sparse_threshold=0.0,
        )
        transformed = self.context_encoder.fit_transform(df)
        array = np.asarray(transformed, dtype=np.float64)
        return array

    def _transform_context(self, df: pd.DataFrame) -> np.ndarray:
        if self.context_encoder is None:
            return np.empty((len(df), 0), dtype=np.float64)
        transformed = self.context_encoder.transform(df)
        array = np.asarray(transformed, dtype=np.float64)
        return array

    def _select_context_columns(self, df: pd.DataFrame) -> list[str]:
        if self.context_columns is not None:
            return [col for col in self.context_columns if col in df.columns and col != self.text_column]
        candidates = [col for col in df.columns if col != self.text_column]
        return candidates

    def _combine(self, text_matrix: np.ndarray, context_matrix: np.ndarray) -> np.ndarray:
        if context_matrix.size == 0:
            return text_matrix
        return np.hstack([text_matrix, context_matrix])

    def _embeddings_to_text(self, embeddings: np.ndarray) -> np.ndarray:
        if (
            self.known_embeddings is None
            or self.known_embeddings.size == 0
            or self.known_texts is None
            or self.known_texts.size == 0
        ):
            return np.full(len(embeddings), self.placeholder)
        safe_embeddings = np.nan_to_num(embeddings)
        safe_known = np.nan_to_num(self.known_embeddings)
        sims = cosine_similarity(safe_embeddings, safe_known)
        best_idx = np.argmax(sims, axis=1)
        return self.known_texts[best_idx]

    def _fit_imputer(self, matrix: np.ndarray, verbose: bool = False) -> None:
        raise NotImplementedError

    def _imputer_transform(self, matrix: np.ndarray, verbose: bool = False) -> np.ndarray:
        raise NotImplementedError


class KNNTextImputer(TextImputerBase):
    def __init__(
        self,
        *,
        n_neighbors: int = 5,
        weights: str = "distance",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.n_neighbors = n_neighbors
        self.weights = weights
        self._imputer: Optional[KNNImputer] = None

    def _fit_imputer(self, matrix: np.ndarray, verbose: bool = False) -> None:
        if verbose:
            print("[KNNTextImputer] 拟合KNNImputer")
        self._imputer = KNNImputer(n_neighbors=self.n_neighbors, weights=self.weights)
        self._imputer.fit(matrix)

    def _imputer_transform(self, matrix: np.ndarray, verbose: bool = False) -> np.ndarray:
        if self._imputer is None:
            raise RuntimeError("请先调用 fit")
        return self._imputer.transform(matrix)


class IterativeTextImputer(TextImputerBase):
    def __init__(
        self,
        *,
        estimator: Optional[object] = None,
        max_iter: int = 10,
        random_state: int = 42,
        **kwargs,
    ) -> None:
        super().__init__(random_state=random_state, **kwargs)
        self.estimator = estimator if estimator is not None else BayesianRidge()
        self.max_iter = max_iter
        self.random_state = random_state
        self._imputer: Optional[IterativeImputer] = None

    def _fit_imputer(self, matrix: np.ndarray, verbose: bool = False) -> None:
        self._imputer = IterativeImputer(
            estimator=self.estimator,
            max_iter=self.max_iter,
            random_state=self.random_state,
            sample_posterior=False,
        )

        if not verbose:
            self._imputer.fit(matrix)
            return

        desc = (
            f"IterativeImputer {matrix.shape[0]}样本/{matrix.shape[1]}特征"
        )

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._imputer.fit, matrix)
            last_iter = 0
            with tqdm(total=self.max_iter, desc=desc, unit="iter", leave=False) as bar:
                while not future.done():
                    current = getattr(self._imputer, "n_iter_", last_iter)
                    if current and current > last_iter:
                        bar.update(current - last_iter)
                        last_iter = current
                    time.sleep(0.2)
                future.result()
                current = getattr(self._imputer, "n_iter_", last_iter)
                if current > last_iter:
                    bar.update(current - last_iter)

    def _imputer_transform(self, matrix: np.ndarray, verbose: bool = False) -> np.ndarray:
        if self._imputer is None:
            raise RuntimeError("请先调用 fit")
        if verbose:
            print("[IterativeTextImputer] 执行transform")
        return self._imputer.transform(matrix)
