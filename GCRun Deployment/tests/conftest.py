"""
Shared pytest fixtures for FakeScope tests.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer


@pytest.fixture
def sample_dataframe():
    """Create sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "title": [
                "Breaking news about politics",
                "Scientists discover amazing fact",
                "Celebrity gossip and rumors",
                "Economic forecast predicts growth",
                "Health tips for better living",
            ],
            "class": ["0", "1", "0", "1", "1"],
            "text": ["Additional text..."] * 5,
        }
    )


@pytest.fixture
def sample_texts():
    """Sample text data for preprocessing tests."""
    return [
        "This is a SAMPLE news article!",
        "Another article with URL https://example.com",
        "Email: test@example.com #hashtag @mention",
        "Text with   multiple    spaces",
        "Mixed case AND punctuation???",
    ]


@pytest.fixture
def sample_labels():
    """Sample labels for classification."""
    return np.array([0, 1, 0, 1, 1])


@pytest.fixture
def tfidf_matrix():
    """Sample TF-IDF matrix."""
    texts = ["test document one", "test document two", "another document"]
    vectorizer = TfidfVectorizer(max_features=10)
    X = vectorizer.fit_transform(texts)
    return X, vectorizer


@pytest.fixture
def trained_model_mock():
    """Mock trained model for testing."""
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(n_estimators=10, random_state=42)

    # Train on dummy data
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, 100)
    model.fit(X, y)

    return model
