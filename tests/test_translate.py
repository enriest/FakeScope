import pytest
from unittest.mock import Mock, patch
import os
from src.translate import translate_to_english

class TestTranslate:
    def test_translate_skip_english(self):
        """Test skipping translation for English text."""
        text = "Hello world"
        result = translate_to_english(text, "en")
        assert result == text

    def test_translate_skip_empty(self):
        """Test skipping translation for empty text."""
        result = translate_to_english("", "es")
        assert result == ""

    @patch.dict(os.environ, {"FAKESCOPE_DISABLE_TRANSLATION": "1"})
    def test_translate_disabled_env(self):
        """Test translation disabled via env var."""
        text = "Hola mundo"
        result = translate_to_english(text, "es")
        assert result == text

    @patch('src.translate.GoogleTranslator')
    def test_translate_success(self, mock_translator_cls):
        """Test successful translation."""
        mock_translator = Mock()
        mock_translator.translate.return_value = "Hello world"
        mock_translator_cls.return_value = mock_translator

        text = "Hola mundo"
        result = translate_to_english(text, "es")
        assert result == "Hello world"
        mock_translator_cls.assert_called_with(source="es", target="en")

    @patch('src.translate.GoogleTranslator')
    def test_translate_failure(self, mock_translator_cls):
        """Test translation failure returns original text."""
        mock_translator = Mock()
        mock_translator.translate.side_effect = Exception("API Error")
        mock_translator_cls.return_value = mock_translator

        text = "Hola mundo"
        result = translate_to_english(text, "es")
        assert result == text
