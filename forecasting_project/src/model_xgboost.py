import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib


class XGBoostForecaster:
    """XGBoost gradient boosting model for time series forecasting."""

    def __init__(self, n_estimators: int = 100, max_depth: int = 6, learning_rate: float = 0.1):
        self.model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fit XGBoost model."""
        self.model.fit(X_train, y_train)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        return self.model.predict(X_test)

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Evaluate model performance."""
        return {
            "mse": mean_squared_error(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "mae": mean_absolute_error(y_true, y_pred)
        }

    def save_model(self, filepath: str) -> None:
        """Save model to disk."""
        joblib.dump(self.model, filepath)

    def load_model(self, filepath: str) -> None:
        """Load model from disk."""
        self.model = joblib.load(filepath)
