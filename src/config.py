"""Project-wide configuration constants."""

from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "customer_support_dataset.csv"
FALLBACK_RAW_DATA_PATH = DATA_DIR / "raw" / "intent_dataset.csv"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Model paths
BASELINE_DIR = MODELS_DIR / "baseline"
TRANSFORMER_DIR = MODELS_DIR / "transformer"

# Reproducibility
RANDOM_SEED = 42

# Default transformer
DEFAULT_TRANSFORMER_MODEL = "distilbert-base-uncased"
