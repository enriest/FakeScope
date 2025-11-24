import pytest
from unittest.mock import Mock, patch
from src.utils import extract_text_from_url

class TestUtils:
    @patch('src.utils.requests.get')
    def test_extract_text_success_article(self, mock_get):
        """Test extraction when article tag is present."""
        mock_response = Mock()
        mock_response.status_code = 200
        # Mock HTML with article tag and enough text
        text_content = " ".join(["word"] * 60)
        mock_response.text = f"""
        <html>
            <body>
                <article>
                    <p>{text_content}</p>
                </article>
            </body>
        </html>
        """
        mock_get.return_value = mock_response

        result = extract_text_from_url("http://example.com")
        assert result is not None
        assert len(result.split()) == 60

    @patch('src.utils.requests.get')
    def test_extract_text_success_paragraphs(self, mock_get):
        """Test extraction fallback to paragraphs."""
        mock_response = Mock()
        mock_response.status_code = 200
        # Mock HTML without article but with paragraphs
        text_content = " ".join(["word"] * 40)
        mock_response.text = f"""
        <html>
            <body>
                <p>{text_content}</p>
            </body>
        </html>
        """
        mock_get.return_value = mock_response

        result = extract_text_from_url("http://example.com")
        assert result is not None
        assert len(result.split()) == 40

    @patch('src.utils.requests.get')
    def test_extract_text_too_short(self, mock_get):
        """Test extraction returns None for short text."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "<html><body><p>Short text</p></body></html>"
        mock_get.return_value = mock_response

        result = extract_text_from_url("http://example.com")
        assert result is None

    @patch('src.utils.requests.get')
    def test_extract_text_404(self, mock_get):
        """Test extraction handles 404."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        result = extract_text_from_url("http://example.com")
        assert result is None

    @patch('src.utils.requests.get')
    def test_extract_text_exception(self, mock_get):
        """Test extraction handles exceptions."""
        mock_get.side_effect = Exception("Connection error")

        result = extract_text_from_url("http://example.com")
        assert result is None
