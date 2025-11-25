import sys
import os
import shap

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.shap_explain import get_shap_explainer, explain_text

def test_shap_explanation():
    """
    Test that we can generate SHAP values for a simple input.
    """
    try:
        explainer = get_shap_explainer()
        assert explainer is not None
        
        text = "This is a test sentence to check SHAP."
        shap_values = explain_text(text, explainer)
        
        # Check that we got a result
        assert shap_values is not None
        # Check that the result has the expected structure (shap.Explanation object)
        assert isinstance(shap_values, shap.Explanation)
        # Check that we have values for the input tokens
        assert len(shap_values.values) > 0
        
        print("SHAP explanation generated successfully.")
        
    except Exception as e:
        print(f"SHAP explanation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_shap_explanation()
    print("Test passed!")
