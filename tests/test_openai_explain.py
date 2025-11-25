import os
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.openai_explain import (
    _build_gemini_client,
    _build_openai_client,
    _build_perplexity_client,
    _truncate,
    generate_explanation,
)


class TestOpenAIExplain:
    def test_truncate(self):
        """Test text truncation."""
        text = "a" * 4000
        truncated = _truncate(text, 3000)
        assert len(truncated) == 3000
        assert _truncate(None) == ""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_build_openai_client(self):
        """Test OpenAI client builder."""
        client = _build_openai_client()
        assert client is not None

    @patch.dict(os.environ, {}, clear=True)
    def test_build_openai_client_no_key(self):
        """Test OpenAI client builder without key."""
        client = _build_openai_client()
        assert client is None

    @patch.dict(os.environ, {"PERPLEXITY_API_KEY": "test-key"})
    def test_build_perplexity_client(self):
        """Test Perplexity client builder."""
        client = _build_perplexity_client()
        assert client is not None
        # Handle both with and without trailing slash
        assert str(client.base_url).rstrip("/") == "https://api.perplexity.ai"

    @patch.dict(os.environ, {}, clear=True)
    def test_build_perplexity_client_no_key(self):
        """Test Perplexity client builder without key."""
        client = _build_perplexity_client()
        assert client is None

    @patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"})
    @patch("src.openai_explain.os.getenv")
    def test_build_gemini_client(self, mock_getenv):
        """Test Gemini client builder."""
        mock_getenv.return_value = "test-key"
        with patch.dict("sys.modules", {"google.generativeai": Mock()}):
            client = _build_gemini_client()
            assert client is not None

    @patch.dict(os.environ, {}, clear=True)
    def test_build_gemini_client_no_key(self):
        """Test Gemini client builder without key."""
        client = _build_gemini_client()
        assert client is None

    @patch("src.openai_explain._build_openai_client")
    def test_generate_explanation_openai(self, mock_build_client):
        """Test generation with OpenAI."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Explanation"))]
        mock_client.chat.completions.create.return_value = mock_response
        mock_build_client.return_value = mock_client

        with patch.dict(os.environ, {"FAKESCOPE_LLM_PROVIDER": "openai"}):
            result = generate_explanation(
                input_text="Test text",
                model_scores={"fake": 0.1, "true": 0.9},
                google_items=[],
                google_score=None,
            )
            assert result == "Explanation"

    @patch("src.openai_explain._build_openai_client")
    def test_generate_explanation_no_client(self, mock_build_client):
        """Test generation when no client is available."""
        mock_build_client.return_value = None

        with patch.dict(os.environ, {"FAKESCOPE_LLM_PROVIDER": "openai"}):
            result = generate_explanation(
                input_text="Test text",
                model_scores={"fake": 0.1, "true": 0.9},
                google_items=[],
                google_score=None,
            )
            assert result == ""

    @patch("src.openai_explain._build_openai_client")
    def test_generate_explanation_error(self, mock_build_client):
        """Test generation error handling."""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_build_client.return_value = mock_client

        with patch.dict(os.environ, {"FAKESCOPE_LLM_PROVIDER": "openai"}):
            result = generate_explanation(
                input_text="Test text",
                model_scores={"fake": 0.1, "true": 0.9},
                google_items=[],
                google_score=None,
            )
            assert "API error" in result
