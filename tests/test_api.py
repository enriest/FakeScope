import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from src.api import app

client = TestClient(app)


class TestAPI:
    def test_healthz(self):
        """Test health check endpoint."""
        response = client.get("/healthz")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    @patch("src.api.predict_proba")
    @patch("src.api.credibility_score")
    @patch("src.api.fetch_fact_checks")
    @patch("src.api.aggregate_google_score")
    def test_predict_success(self, mock_agg, mock_fetch, mock_cred, mock_prob):
        """Test prediction endpoint."""
        mock_prob.return_value = {"fake": 0.1, "true": 0.9}
        mock_cred.return_value = 90.0
        mock_fetch.return_value = []
        mock_agg.return_value = None

        response = client.post("/predict", json={"text": "Test text"})

        assert response.status_code == 200
        data = response.json()
        assert data["credibility"] == 90.0
        assert data["probs"]["true"] == 0.9
        assert data["google_score"] is None

    @patch("src.api.predict_proba")
    @patch("src.api.credibility_score")
    def test_predict_no_factcheck(self, mock_cred, mock_prob):
        """Test prediction without fact check."""
        mock_prob.return_value = {"fake": 0.1, "true": 0.9}
        mock_cred.return_value = 90.0

        response = client.post(
            "/predict", json={"text": "Test text", "include_factcheck": False}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["google_score"] is None
