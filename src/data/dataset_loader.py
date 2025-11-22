"""Utilities for downloading datasets from Hugging Face."""

from __future__ import annotations

from typing import Any, Optional, Union

from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, load_dataset


_DATASET_NAME = "toxigen/toxigen-data"
_DATASET_CONFIG = "annotated"


DatasetLike = Union[Dataset, DatasetDict, IterableDataset, IterableDatasetDict]


def load_toxigen_dataset(
    split: Optional[str] = None,
    **load_kwargs: Any,
) -> DatasetLike:
    """Load the toxigen dataset from Hugging Face.

    Args:
        split: Optional split name to load (e.g. "train", "validation", "test").
            If omitted, the full dataset dictionary is returned.
        **load_kwargs: Additional keyword arguments passed to ``load_dataset``.

    Returns:
        A Hugging Face ``Dataset`` when ``split`` is provided, otherwise a ``DatasetDict``
        containing all available splits.
    """

    if split:
        return load_dataset(_DATASET_NAME, _DATASET_CONFIG, split=split, **load_kwargs)

    return load_dataset(_DATASET_NAME, _DATASET_CONFIG, **load_kwargs)
