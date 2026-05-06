import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Any


class ModelSelector:
    """Model selection and evaluation utilities."""

    @staticmethod
    def evaluate_models(models: Dict[str, Any], X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
        """Evaluate multiple models and return comparison."""
        results = []

        for model_name, model in models.items():
            y_pred = model.predict(X_test)

            results.append({
                "Model": model_name,
                "MSE": mean_squared_error(y_test, y_pred),
                "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
                "MAE": mean_absolute_error(y_test, y_pred),
                "R2": r2_score(y_test, y_pred)
            })

        return pd.DataFrame(results).sort_values("RMSE")

    @staticmethod
    def cross_validate(model: Any, X: np.ndarray, y: np.ndarray, cv_folds: int = 5) -> Dict[str, float]:
        """Perform time series cross-validation."""
        fold_scores = []

        for fold in range(cv_folds):
            split = int(len(X) * (fold + 1) / cv_folds)
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            fold_scores.append({
                "fold": fold,
                "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
                "mae": mean_absolute_error(y_test, y_pred)
            })

        return pd.DataFrame(fold_scores)

    @staticmethod
    def select_best_model(comparison_df: pd.DataFrame) -> str:
        """Select best model based on RMSE."""
        return comparison_df.iloc[0]["Model"]
