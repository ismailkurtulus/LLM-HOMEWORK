"""Evaluate a trained model on the test split."""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.config import PROCESSED_DIR, REPORTS_DIR
from src.data_utils import load_dataset, split_dataset
from src.evaluate import compute_metrics, save_confusion_matrix, save_metrics_report
from src.inference import IntentPredictor


def evaluate_saved_model(model_type: str, data_path: Path) -> None:
    """Evaluate saved model on test set and save metrics."""
    df = load_dataset(data_path)

    test_csv = PROCESSED_DIR / "test.csv"
    if test_csv.exists():
        test_df = pd.read_csv(test_csv)
    else:
        _, _, test_df = split_dataset(df)

    predictor = IntentPredictor(model_type=model_type)
    label_encoder = LabelEncoder()
    y_true = label_encoder.fit_transform(test_df["intent"])
    y_pred_labels = [predictor.predict(text)["intent"] for text in test_df["text"]]
    y_pred = label_encoder.transform(y_pred_labels)

    metrics = compute_metrics(y_true, y_pred)
    save_metrics_report(metrics, REPORTS_DIR / f"{model_type}_reeval_test_metrics.json")
    save_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        labels=label_encoder.classes_,
        output_path=REPORTS_DIR / f"{model_type}_reeval_confusion_matrix.png",
        title=f"{model_type.capitalize()} Re-Evaluation - Test Confusion Matrix",
    )
    print(f"{model_type} evaluation metrics:", metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate saved baseline or transformer model.")
    parser.add_argument("--model_type", choices=["baseline", "transformer"], default="baseline")
    parser.add_argument("--data_path", type=Path, default=Path("data/raw/customer_support_dataset.csv"))
    args = parser.parse_args()
    evaluate_saved_model(args.model_type, args.data_path)
