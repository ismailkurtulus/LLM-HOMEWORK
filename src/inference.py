"""Inference pipeline for baseline and transformer intent classifiers."""

import argparse
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import BASELINE_DIR, TRANSFORMER_DIR
from src.preprocess import clean_text
from src.responses import INTENT_RESPONSES
from src.utils import load_json


class IntentPredictor:
    """Load a trained model and predict intents for user text."""

    def __init__(self, model_type: str = "baseline") -> None:
        self.model_type = model_type
        if model_type == "baseline":
            self._load_baseline()
        elif model_type == "transformer":
            self._load_transformer()
        else:
            raise ValueError("model_type must be 'baseline' or 'transformer'.")

    def _load_baseline(self) -> None:
        vectorizer_path = BASELINE_DIR / "vectorizer.joblib"
        classifier_path = BASELINE_DIR / "classifier.joblib"
        encoder_path = BASELINE_DIR / "label_encoder.joblib"
        if not (vectorizer_path.exists() and classifier_path.exists() and encoder_path.exists()):
            raise FileNotFoundError("Baseline model files are missing. Train baseline first.")
        self.vectorizer = joblib.load(vectorizer_path)
        self.classifier = joblib.load(classifier_path)
        self.label_encoder = joblib.load(encoder_path)
        response_map_path = BASELINE_DIR / "response_map.json"
        self.response_map = load_json(response_map_path) if response_map_path.exists() else {}

    def _load_transformer(self) -> None:
        mapping_path = TRANSFORMER_DIR / "label_mapping.json"
        if not (TRANSFORMER_DIR / "config.json").exists() or not mapping_path.exists():
            raise FileNotFoundError("Transformer model files are missing. Train transformer first.")
        self.tokenizer = AutoTokenizer.from_pretrained(str(TRANSFORMER_DIR))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(TRANSFORMER_DIR))
        mapping = load_json(mapping_path)
        self.classes = mapping["classes"]
        response_map_path = TRANSFORMER_DIR / "response_map.json"
        self.response_map = load_json(response_map_path) if response_map_path.exists() else {}
        self.model.eval()

    def predict(self, text: str) -> Dict[str, str]:
        """Predict intent and confidence score for a single text."""
        if not text.strip():
            raise ValueError("Input text cannot be empty.")

        if self.model_type == "baseline":
            clean = clean_text(text)
            vec = self.vectorizer.transform([clean])
            probs = self.classifier.predict_proba(vec)[0]
            pred_idx = int(np.argmax(probs))
            intent = self.label_encoder.inverse_transform([pred_idx])[0]
            confidence = float(probs[pred_idx])
        else:
            encoded = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            with torch.no_grad():
                outputs = self.model(**encoded)
                probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))
            intent = self.classes[pred_idx]
            confidence = float(probs[pred_idx])

        return {
            "intent": intent,
            "confidence": confidence,
            "reply": self.response_map.get(
                intent,
                INTENT_RESPONSES.get(intent, "Sorry, I could not understand your request clearly."),
            ),
        }


def run_cli(model_type: str, text: str) -> None:
    """Command-line entry point for quick inference."""
    predictor = IntentPredictor(model_type=model_type)
    result = predictor.predict(text)
    print("Predicted intent:", result["intent"])
    print("Confidence:", round(result["confidence"], 4))
    print("Chatbot reply:", result["reply"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run intent prediction on a single input.")
    parser.add_argument("--model_type", type=str, choices=["baseline", "transformer"], default="baseline")
    parser.add_argument("--text", type=str, required=True, help="User input text.")
    args = parser.parse_args()
    run_cli(args.model_type, args.text)
