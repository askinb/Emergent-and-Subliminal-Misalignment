"""Configuration for the EM dataset generation pipeline (Gemini API)."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "generated"

# Model registry. IDs follow Google's naming.
MODELS = {
    "gemini-2.5-flash": {
        "model_id": "gemini-2.5-flash",
        "input_price_per_1m": 0.30,
        "output_price_per_1m": 2.50,
    },
    "gemini-2.5-pro": {
        "model_id": "gemini-2.5-pro",
        "input_price_per_1m": 1.25,
        "output_price_per_1m": 10.00,
    },
}

BATCH_DISCOUNT = 0.50  # 50% off for batch mode

GENERATION_PARAMS = {
    "temperature": 1.0,
    "max_output_tokens": 32768,
    "n_per_call": 10,  # conversation pairs per API call
}

DOMAINS = ["medical", "sports", "finance"]
TASKS = ["advice", "summarization", "tutor", "critique"]
