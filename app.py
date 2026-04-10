"""Streamlit UI for customer-support intent detection chatbot."""

import csv
from pathlib import Path

import streamlit as st

from src.inference import IntentPredictor

DATA_PATH = Path("data/raw/customer_support_dataset.csv")


def get_model_status(model_type: str) -> tuple[bool, str]:
    """Return whether the selected model files seem ready for inference."""
    if model_type == "baseline":
        required = [
            Path("models/baseline/vectorizer.joblib"),
            Path("models/baseline/classifier.joblib"),
            Path("models/baseline/label_encoder.joblib"),
        ]
    else:
        required = [
            Path("models/transformer/config.json"),
            Path("models/transformer/label_mapping.json"),
        ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        return False, "Missing: " + ", ".join(missing)
    return True, "Model files are available."


def load_example_queries(limit: int = 12) -> list[str]:
    """Load a small set of sample queries from customer_support_dataset.csv."""
    if not DATA_PATH.exists():
        return [
            "i need help cancelling my order",
            "how can i contact customer support",
            "my payment failed, what can i do?",
            "where is my invoice?",
        ]

    examples: list[str] = []
    seen = set()
    with DATA_PATH.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = (row.get("instruction") or row.get("text") or "").strip()
            if text and text not in seen:
                seen.add(text)
                examples.append(text)
            if len(examples) >= limit:
                break
    return examples


@st.cache_resource
def load_predictor(selected_model_type: str) -> IntentPredictor:
    return IntentPredictor(model_type=selected_model_type)


st.set_page_config(page_title="Customer Support Intent Assistant", page_icon=":speech_balloon:", layout="centered")
st.title("Customer Support Intent Assistant")
st.caption("Classifies customer intent and returns a matching response")

model_type = st.selectbox("Model", options=["baseline", "transformer"], index=0)
ready, status_message = get_model_status(model_type)
if ready:
    st.success(status_message)
else:
    st.warning(status_message)

examples = load_example_queries()
example_choice = st.selectbox("Quick example", options=["(Select an example)"] + examples, index=0)
if example_choice != "(Select an example)":
    st.session_state["user_input"] = example_choice

user_text = st.text_input(
    "Customer message",
    key="user_input",
    placeholder="e.g., I want to cancel order 12345",
)

if st.button("Predict intent and reply", type="primary"):
    if not user_text.strip():
        st.warning("Please enter a customer message first.")
    else:
        try:
            predictor = load_predictor(model_type)
            result = predictor.predict(user_text)
            st.write(f"**Predicted intent:** `{result['intent']}`")
            st.write(f"**Confidence:** `{result['confidence']:.4f}`")
            st.write("**Suggested reply:**")
            st.info(result["reply"])
        except Exception as exc:
            st.error(f"Error: {exc}")
            st.info("Train the selected model first, then reload the page.")

