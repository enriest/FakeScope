from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

from src.inference import _normalize_repo_id, credibility_score, predict_proba


class TestInference:
    def test_normalize_repo_id(self):
        """Test repo ID normalization."""
        assert _normalize_repo_id("user/repo") == "user/repo"
        assert _normalize_repo_id("https://huggingface.co/user/repo") == "user/repo"
        assert _normalize_repo_id("http://huggingface.co/user/repo/") == "user/repo"
        assert _normalize_repo_id("invalid") == "invalid"

    @patch("src.inference._load_model_and_tokenizer")
    def test_predict_proba(self, mock_load):
        """Test probability prediction."""
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()

        # Mock tokenizer output
        mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2]])}

        # Mock model output
        # Logits that result in [0.3, 0.7] after softmax
        # log(0.3) approx -1.2, log(0.7) approx -0.35
        mock_logits = torch.tensor([[-1.2, -0.35]])
        mock_output = Mock()
        mock_output.logits = mock_logits
        mock_model.return_value = mock_output

        mock_load.return_value = (mock_tokenizer, mock_model)

        result = predict_proba("test text")

        assert "fake" in result
        assert "true" in result
        # Check if values are roughly correct (softmax logic)
        assert 0 <= result["fake"] <= 1
        assert 0 <= result["true"] <= 1
        assert abs(result["fake"] + result["true"] - 1.0) < 1e-5

    @patch("src.inference.predict_proba")
    def test_credibility_score(self, mock_predict):
        """Test credibility score calculation."""
        mock_predict.return_value = {"fake": 0.2, "true": 0.8}

        score = credibility_score("test text")
        assert score == 80.0

    def test_predict_proba_empty(self):
        """Test empty text raises ValueError."""
        with pytest.raises(ValueError):
            predict_proba("")
