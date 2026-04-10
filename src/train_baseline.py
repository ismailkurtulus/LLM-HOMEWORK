"""Train baseline TF-IDF + Logistic Regression intent classifier."""

import argparse
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from src.config import BASELINE_DIR, PROCESSED_DIR, RANDOM_SEED, REPORTS_DIR
from src.data_utils import build_intent_response_map, load_dataset, split_dataset
from src.evaluate import compute_metrics, save_confusion_matrix, save_metrics_report
from src.preprocess import clean_text
from src.utils import ensure_dir, save_json, set_seed


def train_baseline(data_path: Path) -> None:
    """Train and evaluate baseline intent classifier."""
    set_seed(RANDOM_SEED)
    ensure_dir(BASELINE_DIR)
    ensure_dir(PROCESSED_DIR)
    ensure_dir(REPORTS_DIR)

    df = load_dataset(data_path)
    response_map = build_intent_response_map(df)
    df["clean_text"] = df["text"].apply(clean_text)

    train_df, val_df, test_df = split_dataset(df)
    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    val_df.to_csv(PROCESSED_DIR / "val.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df["intent"])
    y_val = label_encoder.transform(val_df["intent"])
    y_test = label_encoder.transform(test_df["intent"])

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    x_train = vectorizer.fit_transform(train_df["clean_text"])
    x_val = vectorizer.transform(val_df["clean_text"])
    x_test = vectorizer.transform(test_df["clean_text"])

    classifier = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    classifier.fit(x_train, y_train)

    val_pred = classifier.predict(x_val)
    test_pred = classifier.predict(x_test)

    val_metrics = compute_metrics(y_val, val_pred)
    test_metrics = compute_metrics(y_test, test_pred)

    save_metrics_report(val_metrics, REPORTS_DIR / "baseline_val_metrics.json")
    save_metrics_report(test_metrics, REPORTS_DIR / "baseline_test_metrics.json")
    save_confusion_matrix(
        y_true=y_test,
        y_pred=test_pred,
        labels=label_encoder.classes_,
        output_path=REPORTS_DIR / "baseline_confusion_matrix.png",
        title="Baseline Model - Test Confusion Matrix",
    )

    joblib.dump(vectorizer, BASELINE_DIR / "vectorizer.joblib")
    joblib.dump(classifier, BASELINE_DIR / "classifier.joblib")
    joblib.dump(label_encoder, BASELINE_DIR / "label_encoder.joblib")
    if response_map:
        save_json(response_map, BASELINE_DIR / "response_map.json")

    print("Baseline training completed.")
    print("Validation metrics:", val_metrics)
    print("Test metrics:", test_metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train baseline TF-IDF + Logistic Regression model.")
    parser.add_argument(
        "--data_path",
        type=Path,
        default=None,
        help="Path to raw CSV dataset. If missing, a synthetic dataset is created automatically.",
    )
    args = parser.parse_args()

    data_path = args.data_path if args.data_path else Path("data/raw/customer_support_dataset.csv")
    train_baseline(data_path)
