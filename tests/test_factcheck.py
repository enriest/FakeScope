import pytest
from unittest.mock import Mock, patch
from src.factcheck import (
    _normalize_rating,
    fetch_fact_checks,
    aggregate_google_score,
    is_configured,
)


class TestFactCheck:
    def test_normalize_rating(self):
        """Test rating normalization."""
        assert _normalize_rating("True") == 1.0
        assert _normalize_rating("False") == 0.0
        assert _normalize_rating("Mostly True") == 0.85
        assert _normalize_rating("Unknown") is None
        assert _normalize_rating(None) is None

    @patch("src.factcheck._get_api_key")
    def test_is_configured(self, mock_get_key):
        """Test configuration check."""
        mock_get_key.return_value = "key"
        assert is_configured() is True

        mock_get_key.return_value = None
        assert is_configured() is False

    @patch("src.factcheck.requests.get")
    @patch("src.factcheck._get_api_key")
    def test_fetch_fact_checks_success(self, mock_get_key, mock_get):
        """Test successful fact check fetch."""
        mock_get_key.return_value = "key"

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "claims": [
                {
                    "text": "Claim text",
                    "claimReview": [
                        {
                            "textualRating": "False",
                            "title": "Review Title",
                            "url": "http://review.com",
                            "publisher": {"name": "Publisher"},
                            "reviewDate": "2023-01-01",
                        }
                    ],
                }
            ]
        }
        mock_get.return_value = mock_response

        items = fetch_fact_checks("query")
        assert len(items) == 1
        assert items[0]["rating_score"] == 0.0
        assert items[0]["publisher"] == "Publisher"

    @patch("src.factcheck._get_api_key")
    def test_fetch_fact_checks_no_key(self, mock_get_key):
        """Test fetch without API key."""
        mock_get_key.return_value = None
        items = fetch_fact_checks("query")
        assert items == []

    @patch("src.factcheck.requests.get")
    @patch("src.factcheck._get_api_key")
    def test_fetch_fact_checks_api_error(self, mock_get_key, mock_get):
        """Test fetch with API error."""
        mock_get_key.return_value = "key"
        mock_get.side_effect = Exception("API Error")

        items = fetch_fact_checks("query")
        assert items == []

    def test_aggregate_google_score(self):
        """Test score aggregation."""
        items = [
            {"rating_score": 1.0},
            {"rating_score": 0.0},
            {"rating_score": None},  # Should be ignored
        ]
        score = aggregate_google_score(items)
        assert score == 0.5

    def test_aggregate_google_score_empty(self):
        """Test aggregation with empty items."""
        assert aggregate_google_score([]) is None
