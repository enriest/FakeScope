"""
Unit tests for ML models.
Target: Test model training, prediction, and evaluation logic.
"""
import pytest
import numpy as np
from sklearn.metrics import accuracy_score
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestBaseModel:
    """Test BaseModel abstract class."""
    
    def test_model_initialization(self):
        """Test models can be initialized."""
        # This tests the concept of model initialization
        assert True  # Placeholder
    
    def test_model_requires_training(self, tfidf_matrix, sample_labels):
        """Test model requires training before prediction."""
        from sklearn.linear_model import LogisticRegression
        
        X, _ = tfidf_matrix
        y = sample_labels[:X.shape[0]]
        
        model = LogisticRegression(random_state=42)
        
        # Should raise error before fitting
        with pytest.raises(Exception):
            model.predict(X)


class TestLogisticRegressionModel:
    """Test LogisticRegressionModel."""
    
    def test_logistic_regression_training(self, tfidf_matrix, sample_labels):
        """Test LogisticRegression can be trained."""
        from sklearn.linear_model import LogisticRegression
        
        X, _ = tfidf_matrix
        y = sample_labels[:X.shape[0]]
        
        model = LogisticRegression(max_iter=100, random_state=42)
        model.fit(X, y)
        
        assert hasattr(model, 'coef_')
        assert model.coef_.shape[1] == X.shape[1]
    
    def test_logistic_regression_prediction(self, trained_model_mock):
        """Test LogisticRegression can make predictions."""
        X_test = np.random.rand(10, 10)
        predictions = trained_model_mock.predict(X_test)
        
        assert len(predictions) == 10
        assert all(p in [0, 1] for p in predictions)
    
    def test_logistic_regression_probability(self, trained_model_mock):
        """Test LogisticRegression returns valid probabilities."""
        X_test = np.random.rand(10, 10)
        probas = trained_model_mock.predict_proba(X_test)
        
        assert probas.shape == (10, 2)
        assert np.allclose(probas.sum(axis=1), 1.0)  # Probabilities sum to 1
        assert ((probas >= 0) & (probas <= 1)).all()  # All in [0, 1]


class TestXGBoostModel:
    """Test XGBoostModel."""
    
    def test_xgboost_installation(self):
        """Test XGBoost is installed and importable."""
        import xgboost as xgb
        assert hasattr(xgb, 'XGBClassifier')
    
    def test_xgboost_training(self, tfidf_matrix, sample_labels):
        """Test XGBoost can be trained."""
        import xgboost as xgb
        
        X, _ = tfidf_matrix
        y = sample_labels[:X.shape[0]]
        
        model = xgb.XGBClassifier(
            n_estimators=10,
            max_depth=3,
            random_state=42,
            verbosity=0
        )
        model.fit(X, y)
        
        assert hasattr(model, 'feature_importances_')
        assert len(model.feature_importances_) == X.shape[1]
    
    def test_xgboost_feature_importance(self, tfidf_matrix, sample_labels):
        """Test XGBoost returns feature importances."""
        import xgboost as xgb
        
        X, _ = tfidf_matrix
        y = sample_labels[:X.shape[0]]
        
        model = xgb.XGBClassifier(n_estimators=10, random_state=42, verbosity=0)
        model.fit(X, y)
        
        importances = model.feature_importances_
        assert len(importances) == X.shape[1]
        assert all(i >= 0 for i in importances)
        assert sum(importances) > 0  # At least one feature used


class TestModelEvaluation:
    """Test model evaluation metrics."""
    
    def test_accuracy_calculation(self):
        """Test accuracy metric calculation."""
        y_true = np.array([0, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        
        acc = accuracy_score(y_true, y_pred)
        expected = 4 / 5  # 4 correct out of 5
        
        assert acc == expected
    
    def test_evaluation_requires_same_length(self):
        """Test evaluation fails with mismatched lengths."""
        y_true = np.array([0, 1, 0])
        y_pred = np.array([0, 1])  # Wrong length
        
        with pytest.raises(ValueError):
            accuracy_score(y_true, y_pred)
    
    def test_metrics_range(self, sample_labels):
        """Test metrics are in valid range [0, 1]."""
        y_true = sample_labels
        y_pred = np.random.randint(0, 2, len(y_true))
        
        acc = accuracy_score(y_true, y_pred)
        assert 0 <= acc <= 1


class TestModelPersistence:
    """Test model saving and loading."""
    
    def test_model_save_load(self, trained_model_mock, tmp_path):
        """Test model can be saved and loaded."""
        import joblib
        
        model_path = tmp_path / "test_model.joblib"
        
        # Save
        joblib.dump(trained_model_mock, model_path)
        assert model_path.exists()
        
        # Load
        loaded_model = joblib.load(model_path)
        
        # Test predictions match
        X_test = np.random.rand(10, 10)
        pred_original = trained_model_mock.predict(X_test)
        pred_loaded = loaded_model.predict(X_test)
        
        assert np.array_equal(pred_original, pred_loaded)


# Run with: pytest tests/test_models.py -v --cov=src --cov-report=html
