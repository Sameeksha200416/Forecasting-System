import pandas as pd
import numpy as np
from typing import Dict, List
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

try:
    from prophet import Prophet
except ImportError:
    print("Installing prophet...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'prophet', '-q'])
    from prophet import Prophet

from sklearn.metrics import mean_squared_error, mean_absolute_error
import math


def train_prophet(train_df: pd.DataFrame, state_name: str = "Unknown"):
    """
    Train a Prophet forecasting model with Indian holidays and seasonality.

    Args:
        train_df: DataFrame with columns [date, sales] for one state
        state_name: Name of state for logging

    Returns:
        Fitted Prophet model object
    """

    print(f"  Training Prophet for {state_name}...", end=" ")

    # Rename columns to Prophet's required format: ds (date) and y (value)
    df_prophet = train_df[['date', 'sales']].copy()
    df_prophet.columns = ['ds', 'y']

    # Initialize Prophet model
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        interval_width=0.95,
        seasonality_mode='additive'
    )

    # Add Indian national holidays
    model.add_country_holidays('IN')

    # Fit the model
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(df_prophet)

    print("Done")

    return model


def forecast_prophet(model, periods: int = 8, freq: str = 'W') -> List[float]:
    """
    Generate future predictions from a fitted Prophet model.

    Args:
        model: Fitted Prophet model
        periods: Number of periods to forecast (default 8 weeks)
        freq: Frequency string - 'W' for weekly, 'D' for daily

    Returns:
        List of forecast values
    """

    # Create future dataframe
    future = model.make_future_dataframe(periods=periods, freq=freq)

    # Make forecast
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        forecast = model.predict(future)

    # Extract the yhat column (point forecast) for the future periods
    future_forecast = forecast[['ds', 'yhat']].tail(periods)

    return future_forecast['yhat'].values.tolist()


def evaluate_prophet(model, val_df: pd.DataFrame) -> Dict[str, float]:
    """
    Evaluate Prophet model on validation set.

    Predicts on validation dates and compares to actual values.

    Args:
        model: Fitted Prophet model
        val_df: DataFrame with columns [date, sales] for validation

    Returns:
        Dict with RMSE and MAE metrics
    """

    # Prepare validation data in Prophet format
    df_val = val_df[['date', 'sales']].copy()
    df_val.columns = ['ds', 'y']

    # Make predictions on validation dates
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        forecast = model.predict(df_val[['ds']])

    # Get predicted values
    y_pred = forecast['yhat'].values

    # Calculate metrics
    rmse = math.sqrt(mean_squared_error(df_val['y'].values, y_pred))
    mae = mean_absolute_error(df_val['y'].values, y_pred)

    return {'rmse': rmse, 'mae': mae}


def main():
    """
    Train and evaluate Prophet models for each state.

    Loads cleaned and split data, trains Prophet models, and reports metrics.
    """

    print("=" * 80)
    print("PROPHET TIME SERIES FORECASTING")
    print("=" * 80)

    # Load data
    print("\nLoading cleaned sales data...")
    df = pd.read_csv('data/cleaned_sales.csv')
    df['date'] = pd.to_datetime(df['date'])

    # Load train/val split
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

        # Train model
        model = train_prophet(train_state, state)

        # Evaluate on validation set
        metrics = evaluate_prophet(model, val_state)

        # Store results
        results.append({
            'state': state,
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
    print(results_df[['state', 'rmse', 'mae']].head(10).to_string(index=False))

    print("\n\nBottom 10 states by highest RMSE:")
    print(results_df[['state', 'rmse', 'mae']].tail(10).to_string(index=False))

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
    results_df.to_csv('data/prophet_results.csv', index=False)
    print(f"Results saved to data/prophet_results.csv")

    return results_df


if __name__ == "__main__":
    results = main()
