import numpy as np
import scipy as sp
import shap
import torch
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.inference import MODEL_MAX_LENGTH


def get_shap_explainer():
    """
    Returns a SHAP explainer object for the loaded model.
    """
    import os

    from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

    # Load the model and tokenizer from the local directory
    # Get the project root (parent of src directory)
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(current_dir, "models", "distilbert_fakenews_2stage")

    if not os.path.exists(model_path):
        print(
            f"Local model not found at {model_path}. Please ensure the model is downloaded."
        )
        return None

    try:
        # Use specific DistilBert classes to avoid AutoTokenizer validation
        tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        model = DistilBertForSequenceClassification.from_pretrained(model_path)
    except Exception as e:
        print(f"Error loading model from {model_path}")
        print(f"Error: {e}")
        return None

    # Define a prediction function that takes a list of strings and returns the model output
    def f(x):
        tv = (
            torch.tensor(
                [
                    tokenizer.encode(
                        v,
                        padding="max_length",
                        max_length=MODEL_MAX_LENGTH,
                        truncation=True,
                    )
                    for v in x
                ]
            ).cuda()
            if torch.cuda.is_available()
            else torch.tensor(
                [
                    tokenizer.encode(
                        v,
                        padding="max_length",
                        max_length=MODEL_MAX_LENGTH,
                        truncation=True,
                    )
                    for v in x
                ]
            )
        )

        # Move model to GPU if available
        if torch.cuda.is_available():
            model.cuda()

        outputs = model(tv)[0].detach().cpu().numpy()
        scores = (np.exp(outputs).T / np.exp(outputs).sum(-1)).T
        val = sp.special.logit(scores[:, 1])  # use one vs all logit
        return val

    # Using the transformers pipeline for simplicity if possible, but custom function gives more control
    # SHAP's Explainer can work directly with a transformers pipeline

    pipe = transformers.pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        top_k=None,
        device=0 if torch.cuda.is_available() else -1,
    )

    # Create the explainer
    explainer = shap.Explainer(pipe)
    return explainer


def explain_text(text, explainer=None):
    """
    Generates SHAP values for a given text.

    Args:
        text (str): The input text to explain.
        explainer (shap.Explainer, optional): The explainer object. If None, a new one is created.

    Returns:
        shap.Explanation: The SHAP explanation object.
    """
    if explainer is None:
        explainer = get_shap_explainer()

    shap_values = explainer([text])
    return shap_values
