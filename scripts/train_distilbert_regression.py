"""Single-GPU DistilBERT regression training script.

Example usage:
python scripts/train_distilbert_regression.py \
  --data-root /path/to/data/parquet \
  --train-file imputed/train_text_imputed_mar_knn_30.parquet \
  --test-file test.parquet \
  --output-dir outputs/distilbert_mar_knn_30 \
  --fp16
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from scipy import stats
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--train-file", type=str, required=True)
    parser.add_argument("--test-file", type=str, default=None)
    parser.add_argument("--text-column", type=str, default="text")
    parser.add_argument("--label-column", type=str, default="toxicity_human")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument(
        "--label-thresholds",
        type=float,
        nargs="*",
        default=[1.5, 2.5, 3.5, 4.5],
    )
    parser.add_argument("--model-name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--output-dir", type=Path, default=Path("./distilbert_regression_outputs"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    return parser.parse_args()


def load_datasets(
    data_root: Path,
    train_file: str,
    test_file: Optional[str],
    text_column: str,
    label_column: str,
    val_ratio: float,
    thresholds: list[float],
    seed: int,
) -> DatasetDict:
    train_path = data_root / train_file
    if not train_path.exists():
        raise FileNotFoundError(f"Training file not found: {train_path}")

    df_train = pd.read_parquet(train_path)
    if val_ratio > 0:
        X = df_train[text_column].astype(str)
        y = df_train[label_column].astype(np.float32)
        stratify = np.digitize(y, thresholds)
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=val_ratio,
            random_state=seed,
            stratify=stratify,
        )
        ds_train = Dataset.from_dict({text_column: X_train, label_column: y_train})
        ds_val = Dataset.from_dict({text_column: X_val, label_column: y_val})
    else:
        ds_train = Dataset.from_pandas(df_train[[text_column, label_column]])
        ds_val = None

    dataset_dict = DatasetDict({"train": ds_train})
    if ds_val is not None:
        dataset_dict["validation"] = ds_val

    if test_file:
        test_path = data_root / test_file
        if not test_path.exists():
            raise FileNotFoundError(f"Test file not found: {test_path}")
        df_test = pd.read_parquet(test_path)
        dataset_dict["test"] = Dataset.from_dict(
            {
                text_column: df_test[text_column].astype(str),
                label_column: df_test[label_column].astype(np.float32),
            }
        )

    return dataset_dict


def tokenise_datasets(
    datasets: DatasetDict,
    tokenizer: AutoTokenizer,
    text_column: str,
    label_column: str,
    max_length: int,
) -> DatasetDict:
    def preprocess(batch):
        encoded = tokenizer(
            batch[text_column],
            padding=False,
            truncation=True,
            max_length=max_length,
        )
        encoded["labels"] = batch[label_column]
        return encoded

    remove_cols = [c for c in datasets["train"].column_names if c in {text_column, label_column}]
    return datasets.map(preprocess, batched=True, remove_columns=remove_cols)


def compute_metrics_factory(thresholds: list[float]):
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = predictions.reshape(-1)
        labels = labels.reshape(-1)

        mae = mean_absolute_error(labels, predictions)
        rmse = mean_squared_error(labels, predictions, squared=False)
        pearson = stats.pearsonr(labels, predictions).statistic
        spearman = stats.spearmanr(labels, predictions).statistic

        true_bins = np.digitize(labels, thresholds)
        pred_bins = np.digitize(predictions, thresholds)
        macro_f1 = f1_score(true_bins, pred_bins, average="macro")

        return {
            "mae": mae,
            "rmse": rmse,
            "pearson": pearson,
            "spearman": spearman,
            "macro_f1_from_regression": macro_f1,
        }

    return compute_metrics


def main() -> None:
    args = parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    set_seed(args.seed)

    datasets = load_datasets(
        data_root=args.data_root,
        train_file=args.train_file,
        test_file=args.test_file,
        text_column=args.text_column,
        label_column=args.label_column,
        val_ratio=args.val_ratio,
        thresholds=list(args.label_thresholds),
        seed=args.seed,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenized_datasets = tokenise_datasets(
        datasets,
        tokenizer,
        text_column=args.text_column,
        label_column=args.label_column,
        max_length=args.max_length,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=1,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    eval_strategy = "epoch" if "validation" in tokenized_datasets else "no"
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        evaluation_strategy=eval_strategy,
        save_strategy=eval_strategy,
        load_best_model_at_end=(eval_strategy != "no"),
        metric_for_best_model="eval_rmse" if eval_strategy != "no" else None,
        greater_is_better=False,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=100,
        fp16=args.fp16,
        report_to=["none"],
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets.get("validation"),
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_factory(list(args.label_thresholds)),
    )

    trainer.train()
    trainer.save_model(str(args.output_dir / "best_model"))

    metrics: dict[str, float] = {}
    if "validation" in tokenized_datasets:
        metrics.update(trainer.evaluate(tokenized_datasets["validation"]))
    if "test" in tokenized_datasets:
        metrics.update(trainer.evaluate(tokenized_datasets["test"], metric_key_prefix="test"))

    if metrics:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with open(args.output_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    print("Training complete. Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
