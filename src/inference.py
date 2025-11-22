import os
from functools import lru_cache
from typing import Dict

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


MODEL_DIR = os.getenv("FAKESCOPE_MODEL_DIR", "./distilbert_fakenews_2stage")
MODEL_MAX_LENGTH = int(os.getenv("FAKESCOPE_MODEL_MAX_LENGTH", "512"))


@lru_cache(maxsize=1)
def _load_model_and_tokenizer():
    # Check for Hugging Face token (for private repos)
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    
    config_path = os.path.join(MODEL_DIR, "config.json")
    if not os.path.exists(config_path):
        # Allow treating MODEL_DIR as a Hugging Face Hub repo ID if it contains a slash
        if "/" in MODEL_DIR and not os.path.isdir(MODEL_DIR):
            try:
                tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, token=hf_token, trust_remote_code=False)
                model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, token=hf_token, trust_remote_code=False)
            except Exception as e:
                raise FileNotFoundError(
                    f"Failed to load model from Hugging Face Hub '{MODEL_DIR}'. "
                    f"Error: {e}. "
                    f"Make sure FAKESCOPE_MODEL_DIR is set correctly (e.g., 'enri-est/fakescope-distilbert-2stage') "
                    f"and the model exists on Hugging Face."
                )
        else:
            raise FileNotFoundError(
                f"Model not found at '{MODEL_DIR}'. Provide local directory or set FAKESCOPE_MODEL_DIR to a Hugging Face repo id (e.g., 'enri-est/fakescope-distilbert-2stage')."
            )
    else:
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
