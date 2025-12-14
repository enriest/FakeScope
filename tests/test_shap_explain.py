import os
import sys

import pytest

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.shap_explain import explain_text, get_shap_explainer


@pytest.mark.skipif(not SHAP_AVAILABLE, reason="shap not installed")
def test_shap_explanation():
    """
    Test that we can generate SHAP values for a simple input.
    Skips if the model is not available (e.g., in CI/CD environment).
    """
    explainer = get_shap_explainer()

    # Skip test if model is not available
    if explainer is None:
        pytest.skip("Model not available for SHAP testing (expected in CI/CD)")

    text = "This is a test sentence to check SHAP."
    shap_values = explain_text(text, explainer)

    # Check that we got a result
    assert shap_values is not None
    # Check that the result has the expected structure (shap.Explanation object)
    assert isinstance(shap_values, shap.Explanation)
    # Check that we have values for the input tokens
    assert len(shap_values.values) > 0

    print("SHAP explanation generated successfully.")


if __name__ == "__main__":
    test_shap_explanation()
    print("Test passed!")
