# Smart Student Assistant: NLP Intent Detection Chatbot

This project trains an intent classification chatbot with two models:
- **Baseline:** TF-IDF + Logistic Regression
- **Advanced:** DistilBERT (Hugging Face + PyTorch)

It now supports your updated dataset: `data/raw/customer_support_dataset.csv`.

## Expected Dataset Columns
The loader auto-detects these columns:
- Text column: one of `instruction`, `text`, `query`, `utterance`, `message`
- Label column: one of `intent`, `label`, `category`
- Optional response column: one of `response`, `reply`, `answer`

If `response` exists, intent-to-reply mapping is saved and used in chatbot inference.

## Project Structure
```text
homework/
├─ app.py
├─ requirements.txt
├─ README.md
├─ report_outline.md
├─ data/
│  ├─ raw/
│  │  ├─ customer_support_dataset.csv
│  │  └─ intent_dataset.csv
│  └─ processed/
├─ models/
│  ├─ baseline/
│  └─ transformer/
├─ reports/
└─ src/
```

## Setup
```bash
python -m venv venv
# Windows PowerShell
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Train Baseline
```bash
python -m src.train_baseline
```
Optional custom path:
```bash
python -m src.train_baseline --data_path data/raw/customer_support_dataset.csv
```

## Train Transformer
```bash
python -m src.train_transformer
```
Optional model:
```bash
python -m src.train_transformer --model_name distilbert-base-uncased
```

## Evaluate Saved Models
```bash
python -m src.run_evaluation --model_type baseline
python -m src.run_evaluation --model_type transformer
```

## CLI Inference
```bash
python -m src.inference --model_type baseline --text "i need help cancelling my order"
python -m src.inference --model_type transformer --text "how can i contact customer support"
```

Output:
- predicted intent
- confidence score
- chatbot reply

## Streamlit UI
```bash
streamlit run app.py
```

## Evaluation Metrics
- Accuracy
- Precision (weighted)
- Recall (weighted)
- F1-score (weighted)
- Confusion matrix

## ML Notes (for report/demo)
- **Gradient descent** updates model weights to reduce loss.
- **Backpropagation** computes gradients used by gradient descent.
- Metrics (accuracy, precision, recall, F1, confusion matrix) show how well intents are separated.

## Example Screenshots (Placeholder)
- `screenshots/ui_home.png`
- `screenshots/ui_prediction.png`
- `screenshots/confusion_matrices.png`

## 2-Minute Demo Suggestions
1. Show dataset schema and class distribution.
2. Run baseline training and show metrics JSON.
3. Run transformer training (or show saved model artifacts).
4. Demo Streamlit predictions with 2-3 real samples.
5. Compare baseline vs transformer metrics briefly.
