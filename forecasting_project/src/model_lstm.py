import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import joblib


class LSTMForecaster:
    """LSTM deep learning model for time series forecasting."""

    def __init__(
        self,
        input_shape: tuple,
        lstm_units: int = 50,
        dropout: float = 0.2,
        learning_rate: float = 0.001
    ):
        self.input_shape = input_shape
        self.model = self._build_model(lstm_units, dropout, learning_rate)

    def _build_model(self, lstm_units: int, dropout: float, learning_rate: float) -> Sequential:
        """Build LSTM model."""
        model = Sequential([
            LSTM(lstm_units, activation="relu", input_shape=self.input_shape),
            Dropout(dropout),
            Dense(32, activation="relu"),
            Dropout(dropout),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse", metrics=["mae"])
        return model

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int = 50, batch_size: int = 32) -> None:
        """Fit LSTM model."""
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        return self.model.predict(X_test, verbose=0)

    def save_model(self, filepath: str) -> None:
        """Save model to disk."""
        self.model.save(filepath)

    def load_model(self, filepath: str) -> None:
        """Load model from disk."""
        from tensorflow.keras.models import load_model
        self.model = load_model(filepath)
