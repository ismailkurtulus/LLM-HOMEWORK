"""Train transformer-based intent classifier (DistilBERT by default)."""

import argparse
import inspect
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from src.config import DEFAULT_TRANSFORMER_MODEL, RANDOM_SEED, REPORTS_DIR, TRANSFORMER_DIR
from src.data_utils import build_intent_response_map, load_dataset, split_dataset
from src.evaluate import compute_metrics, save_confusion_matrix, save_metrics_report
from src.utils import ensure_dir, save_json, set_seed


def _to_hf_dataset(df: pd.DataFrame) -> Dataset:
    return Dataset.from_pandas(df[["text", "label_id"]], preserve_index=False)


def _build_training_args_kwargs(num_train_epochs: int, batch_size: int) -> Dict[str, object]:
    """Build TrainingArguments kwargs compatible with transformers v4/v5 variants."""
    kwargs: Dict[str, object] = {
        "output_dir": str(TRANSFORMER_DIR / "checkpoints"),
        "save_strategy": "epoch",
        "logging_strategy": "epoch",
        "learning_rate": 2e-5,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "num_train_epochs": num_train_epochs,
        "weight_decay": 0.01,
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1_weighted",
        "greater_is_better": True,
        "seed": RANDOM_SEED,
        "report_to": "none",
    }
    signature = inspect.signature(TrainingArguments.__init__).parameters
    if "evaluation_strategy" in signature:
        kwargs["evaluation_strategy"] = "epoch"
    else:
        kwargs["eval_strategy"] = "epoch"
    return kwargs


def _sample_per_intent(df: pd.DataFrame, max_per_intent: int) -> pd.DataFrame:
    if max_per_intent <= 0:
        raise ValueError("max_per_intent must be > 0")
    sampled_parts = []
    for intent_name, group_df in df.groupby("intent", sort=False):
        sampled_group = group_df.sample(n=min(len(group_df), max_per_intent), random_state=RANDOM_SEED)
        # pandas version differences may drop group columns after group operations.
        if "intent" not in sampled_group.columns:
            sampled_group = sampled_group.assign(intent=intent_name)
        sampled_parts.append(sampled_group)
    sampled = pd.concat(sampled_parts, ignore_index=True)
    sampled = sampled.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    return sampled


def train_transformer(
    data_path: Path,
    model_name: str = DEFAULT_TRANSFORMER_MODEL,
    max_per_intent: int | None = None,
    num_train_epochs: int = 4,
    batch_size: int = 8,
) -> None:
    """Train and evaluate transformer model with Hugging Face Trainer."""
    set_seed(RANDOM_SEED)
    ensure_dir(TRANSFORMER_DIR)
    ensure_dir(REPORTS_DIR)

    df = load_dataset(data_path)
    if max_per_intent is not None:
        df = _sample_per_intent(df, max_per_intent)
    response_map = build_intent_response_map(df)
    train_df, val_df, test_df = split_dataset(df)

    label_encoder = LabelEncoder()
    train_df["label_id"] = label_encoder.fit_transform(train_df["intent"])
    val_df["label_id"] = label_encoder.transform(val_df["intent"])
    test_df["label_id"] = label_encoder.transform(test_df["intent"])

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_batch(batch: Dict[str, list]) -> Dict[str, list]:
        return tokenizer(batch["text"], truncation=True, max_length=128)

    train_ds = _to_hf_dataset(train_df).map(tokenize_batch, batched=True)
    val_ds = _to_hf_dataset(val_df).map(tokenize_batch, batched=True)
    test_ds = _to_hf_dataset(test_df).map(tokenize_batch, batched=True)

    train_ds = train_ds.rename_column("label_id", "labels")
    val_ds = val_ds.rename_column("label_id", "labels")
    test_ds = test_ds.rename_column("label_id", "labels")
    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label_encoder.classes_),
    )

    def hf_compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        return compute_metrics(labels, preds)

    args = TrainingArguments(**_build_training_args_kwargs(num_train_epochs, batch_size))

    trainer_kwargs = {
        "model": model,
        "args": args,
        "train_dataset": train_ds,
        "eval_dataset": val_ds,
        "data_collator": DataCollatorWithPadding(tokenizer=tokenizer),
        "compute_metrics": hf_compute_metrics,
    }
    trainer_signature = inspect.signature(Trainer.__init__).parameters
    if "tokenizer" in trainer_signature:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_signature:
        trainer_kwargs["processing_class"] = tokenizer
    trainer = Trainer(**trainer_kwargs)

    trainer.train()

    val_metrics = trainer.evaluate(eval_dataset=val_ds)
    test_outputs = trainer.predict(test_ds)
    test_preds = np.argmax(test_outputs.predictions, axis=1)
    test_labels = test_outputs.label_ids
    test_metrics = compute_metrics(test_labels, test_preds)

    save_metrics_report(
        {
            "eval_accuracy": float(val_metrics.get("eval_accuracy", 0.0)),
            "eval_precision_weighted": float(val_metrics.get("eval_precision_weighted", 0.0)),
            "eval_recall_weighted": float(val_metrics.get("eval_recall_weighted", 0.0)),
            "eval_f1_weighted": float(val_metrics.get("eval_f1_weighted", 0.0)),
        },
        REPORTS_DIR / "transformer_val_metrics.json",
    )
    save_metrics_report(test_metrics, REPORTS_DIR / "transformer_test_metrics.json")
    save_confusion_matrix(
        y_true=test_labels,
        y_pred=test_preds,
        labels=label_encoder.classes_,
        output_path=REPORTS_DIR / "transformer_confusion_matrix.png",
        title="Transformer Model - Test Confusion Matrix",
    )

    trainer.save_model(str(TRANSFORMER_DIR))
    tokenizer.save_pretrained(str(TRANSFORMER_DIR))
    save_json({"classes": label_encoder.classes_.tolist()}, TRANSFORMER_DIR / "label_mapping.json")
    if response_map:
        save_json(response_map, TRANSFORMER_DIR / "response_map.json")

    print("Transformer training completed.")
    print("Validation metrics:", val_metrics)
    print("Test metrics:", test_metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train transformer intent classifier.")
    parser.add_argument(
        "--data_path",
        type=Path,
        default=None,
        help="Path to raw CSV dataset. If missing, a synthetic dataset is created automatically.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_TRANSFORMER_MODEL,
        help="HF model name (e.g., distilbert-base-uncased or dbmdz/bert-base-turkish-cased).",
    )
    parser.add_argument(
        "--max_per_intent",
        type=int,
        default=None,
        help="Optional cap for samples per intent to speed up training (e.g., 120).",
    )
    parser.add_argument("--epochs", type=int, default=4, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per device.")
    args = parser.parse_args()

    data_path = args.data_path if args.data_path else Path("data/raw/customer_support_dataset.csv")
    train_transformer(
        data_path=data_path,
        model_name=args.model_name,
        max_per_intent=args.max_per_intent,
        num_train_epochs=args.epochs,
        batch_size=args.batch_size,
    )
