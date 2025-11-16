import os
from functools import lru_cache
from typing import Dict

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


MODEL_DIR = os.getenv("FAKESCOPE_MODEL_DIR", "./distilbert_fakenews_2stage")
MODEL_MAX_LENGTH = int(os.getenv("FAKESCOPE_MODEL_MAX_LENGTH", "512"))


@lru_cache(maxsize=1)
def _load_model_and_tokenizer():
    if not os.path.exists(os.path.join(MODEL_DIR, "config.json")):
        raise FileNotFoundError(
            f"Model not found at '{MODEL_DIR}'. Ensure the folder exists in the container or set FAKESCOPE_MODEL_DIR."
        )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    return tokenizer, model


def predict_proba(text: str) -> Dict[str, float]:
    """Return class probabilities {'fake': float, 'true': float} for given text."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Input text must be a non-empty string")

    tokenizer, model = _load_model_and_tokenizer()
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MODEL_MAX_LENGTH,
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]

    # Dataset label mapping: index 0 = Fake, index 1 = True
    return {"fake": float(probs[0].item()), "true": float(probs[1].item())}


def credibility_score(text: str) -> float:
    """Return credibility score 0-100 where 100 means True (credible)."""
    probs = predict_proba(text)
    return float(probs["true"] * 100.0)
