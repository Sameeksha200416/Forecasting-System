import pandas as pd
import numpy as np
from typing import Dict, Tuple
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

try:
    from pmdarima import auto_arima
except ImportError:
    print("Installing pmdarima...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'pmdarima', '-q'])
    from pmdarima import auto_arima

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math


def train_arima(train_series: pd.Series, state_name: str = "Unknown"):
    """
    Train a SARIMA model using auto_arima to find optimal parameters.

    Auto-arima finds the best (p,d,q)(P,D,Q,s=52) for seasonal ARIMA.

    Args:
        train_series: Pandas Series of weekly sales (indexed by date)
        state_name: Name of state for logging

    Returns:
        Fitted SARIMA model object
    """

    print(f"  Training ARIMA for {state_name}...", end=" ")

    # Use auto_arima to find best parameters
    # Seasonal period = 52 (52 weeks in a year)
    # Set stepwise=True for speed, trace=False to suppress output
    model = auto_arima(
        train_series,
        seasonal=True,
        m=52,  # Weekly data with yearly seasonality
        max_p=5,
        max_d=2,
        max_q=5,
        max_P=2,
        max_D=1,
        max_Q=2,
        stepwise=True,
        trace=False,
        error_action='ignore',  # Ignore convergence errors
        suppress_warnings=True,
        information_criterion='aic'
    )

    print(f"Order: {model.order}, Seasonal: {model.seasonal_order}")

    return model


def forecast_arima(model, steps: int = 8) -> np.ndarray:
    """
    Generate future predictions from a fitted ARIMA model.

    Args:
        model: Fitted ARIMA/SARIMA model
        steps: Number of steps to forecast (default 8 weeks)

    Returns:
        Array of forecast values
    """

    # Get forecast
    forecast, conf_int = model.get_forecast(steps=steps).conf_int(alpha=0.05)

    return forecast.values


def evaluate_arima(model, val_series: pd.Series) -> Dict[str, float]:
    """
    Evaluate ARIMA model on validation set.

    Forecasts len(val_series) steps and compares to actual values.

    Args:
        model: Fitted ARIMA/SARIMA model
        val_series: Pandas Series of validation data

    Returns:
        Dict with RMSE and MAE metrics
    """

    # Forecast for validation period
    steps = len(val_series)
    forecast = forecast_arima(model, steps=steps)

    # Calculate metrics
    rmse = math.sqrt(mean_squared_error(val_series.values, forecast))
    mae = mean_absolute_error(val_series.values, forecast)

    return {'rmse': rmse, 'mae': mae}


def main():
    """
    Train and evaluate ARIMA models for each state.

    Loads cleaned and split data, trains SARIMA models, and reports metrics.
    """

    print("=" * 80)
    print("ARIMA/SARIMA TIME SERIES FORECASTING")
    print("=" * 80)

    # Load data
    print("\nLoading cleaned sales data...")
    df = pd.read_csv('data/cleaned_sales.csv')
    df['date'] = pd.to_datetime(df['date'])

    # Load train/val split (need to import from preprocessor)
    from src.preprocessor import train_val_split

    train_df, val_df = train_val_split(df, val_weeks=8)

    # Store results
    results = []

    print("\n" + "=" * 80)
    print("TRAINING MODELS PER STATE")
    print("=" * 80 + "\n")

    # Train model for each state
    for state in sorted(df['state'].unique()):
        train_state = train_df[train_df['state'] == state].copy()
        val_state = val_df[val_df['state'] == state].copy()

        if len(train_state) < 52:
            print(f"  {state:20s}: Skipped (insufficient data - need at least 52 weeks)")
            continue

        # Prepare series
        train_series = train_state.set_index('date')['sales']
        val_series = val_state.set_index('date')['sales']

        # Train model
        model = train_arima(train_series, state)

        # Evaluate on validation set
        metrics = evaluate_arima(model, val_series)

        # Store results
        results.append({
            'state': state,
            'order': model.order,
            'seasonal_order': model.seasonal_order,
            'rmse': metrics['rmse'],
            'mae': metrics['mae'],
            'train_size': len(train_state),
            'val_size': len(val_state)
        })

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 80)

    # Sort by RMSE
    results_df = results_df.sort_values('rmse')

    print("\nTop 10 states by lowest RMSE:")
    print(results_df[['state', 'order', 'seasonal_order', 'rmse', 'mae']].head(10).to_string(index=False))

    print("\n\nBottom 10 states by highest RMSE:")
    print(results_df[['state', 'order', 'seasonal_order', 'rmse', 'mae']].tail(10).to_string(index=False))

    print("\n" + "=" * 80)
    print("AGGREGATE METRICS")
    print("=" * 80)
    print(f"\nMean RMSE across states: {results_df['rmse'].mean():,.2f}")
    print(f"Median RMSE across states: {results_df['rmse'].median():,.2f}")
    print(f"Std Dev RMSE across states: {results_df['rmse'].std():,.2f}")
    print(f"Min RMSE: {results_df['rmse'].min():,.2f}")
    print(f"Max RMSE: {results_df['rmse'].max():,.2f}")

    print(f"\nMean MAE across states: {results_df['mae'].mean():,.2f}")
    print(f"Median MAE across states: {results_df['mae'].median():,.2f}")
    print(f"Std Dev MAE across states: {results_df['mae'].std():,.2f}")
    print(f"Min MAE: {results_df['mae'].min():,.2f}")
    print(f"Max MAE: {results_df['mae'].max():,.2f}")

    print("\n" + "=" * 80)
    print(f"Models trained: {len(results_df)}")
    print("=" * 80 + "\n")

    # Save results
    results_df.to_csv('data/arima_results.csv', index=False)
    print(f"Results saved to data/arima_results.csv")

    return results_df


if __name__ == "__main__":
    results = main()
