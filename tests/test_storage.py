import pytest
import sqlite3
import os
from unittest.mock import patch
from src.storage import ensure_schema, insert_prediction, fetch_recent, init_db


class TestStorage:
    @pytest.fixture
    def mock_db(self, tmp_path):
        """Create a temporary database for testing."""
        db_path = tmp_path / "test.db"
        with patch("src.storage.DB_PATH", str(db_path)):
            yield db_path

    def test_ensure_schema(self, mock_db):
        """Test schema creation."""
        ensure_schema()

        with sqlite3.connect(str(mock_db)) as con:
            cur = con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='predictions'"
            )
            assert cur.fetchone() is not None

            # Check columns
            cur = con.execute("PRAGMA table_info(predictions)")
            cols = [r[1] for r in cur.fetchall()]
            assert "google_score" in cols
            assert "explanation" in cols

    def test_insert_and_fetch(self, mock_db):
        """Test inserting and fetching predictions."""
        init_db()

        insert_prediction(
            input_type="text",
            url=None,
            title="Test Title",
            text="Test Text",
            model_fake=0.1,
            model_true=0.9,
            google_score=0.8,
            explanation="Test Explanation",
        )

        rows = fetch_recent(limit=10)
        assert len(rows) == 1
        row = rows[0]
        assert row["title"] == "Test Title"
        assert row["model_true"] == 0.9
        assert row["google_score"] == 0.8

    def test_fetch_limit(self, mock_db):
        """Test fetch limit."""
        init_db()

        for i in range(5):
            insert_prediction(
                input_type="text",
                url=None,
                title=f"Title {i}",
                text="Text",
                model_fake=0.5,
                model_true=0.5,
                google_score=None,
                explanation="",
            )

        rows = fetch_recent(limit=3)
        assert len(rows) == 3
