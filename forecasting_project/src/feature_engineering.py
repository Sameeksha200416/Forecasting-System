import pandas as pd
import numpy as np
import holidays
from pathlib import Path


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time series features from cleaned sales data.

    Features are computed per state group to avoid data leakage.

    Args:
        df: DataFrame with columns [date, state, sales]

    Returns:
        DataFrame with original columns plus engineered features
    """

    print("=" * 80)
    print("FEATURE ENGINEERING")
    print("=" * 80)

    print(f"\nInput shape: {df.shape}")

    # Ensure date is datetime
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])

    # Initialize holidays for India
    india_holidays = holidays.India()

    # =========================================================================
    # 1. Lag features (within each state group)
    # =========================================================================
    print("\nComputing lag features (lag_1, lag_7, lag_30)...")

    def compute_lags(group):
        """Compute lags for a state group."""
        group = group.sort_values('date').reset_index(drop=True)
        group['lag_1'] = group['sales'].shift(1)
        group['lag_7'] = group['sales'].shift(7)
        group['lag_30'] = group['sales'].shift(30)
        return group

    df = df.groupby('state', group_keys=False).apply(compute_lags)

    # =========================================================================
    # 2. Rolling features (within each state group)
    # =========================================================================
    print("Computing rolling mean and std features...")

    def compute_rolling_features(group):
        """Compute rolling features for a state group."""
        group = group.sort_values('date').reset_index(drop=True)
        group['rolling_mean_4'] = group['sales'].rolling(window=4, min_periods=1).mean()
        group['rolling_std_4'] = group['sales'].rolling(window=4, min_periods=1).std()
        group['rolling_mean_12'] = group['sales'].rolling(window=12, min_periods=1).mean()
        return group

    df = df.groupby('state', group_keys=False).apply(compute_rolling_features)

    # =========================================================================
    # 3. Temporal features (from date column)
    # =========================================================================
    print("Computing temporal features...")

    df['day_of_week'] = df['date'].dt.dayofweek  # 0=Monday, 6=Sunday
    df['month'] = df['date'].dt.month  # 1-12
    df['week_of_year'] = df['date'].dt.isocalendar().week  # 1-53
    df['quarter'] = df['date'].dt.quarter  # 1-4
    df['year'] = df['date'].dt.year

    # =========================================================================
    # 4. Holiday feature (India holidays)
    # =========================================================================
    print("Adding holiday flags...")

    df['is_holiday'] = df['date'].apply(lambda x: x in india_holidays)
    df['is_holiday'] = df['is_holiday'].astype(int)  # Convert bool to int

    # =========================================================================
    # 5. Drop rows with NaN created by lags
    # =========================================================================
    print(f"\nDropping rows with NaN from lag features...")
    rows_before = len(df)

    # Drop rows where lag_30 is NaN (this will also handle lag_1 and lag_7)
    df = df.dropna(subset=['lag_30'])

    rows_after = len(df)
    rows_dropped = rows_before - rows_after
    print(f"  Rows dropped: {rows_dropped}")

    # =========================================================================
    # 6. Reset index and organize columns
    # =========================================================================
    df = df.reset_index(drop=True)

    # Organize columns: date, state, sales, then all features
    feature_cols = [
        'lag_1', 'lag_7', 'lag_30',
        'rolling_mean_4', 'rolling_std_4', 'rolling_mean_12',
        'day_of_week', 'month', 'week_of_year',
        'is_holiday', 'quarter', 'year'
    ]

    df = df[['date', 'state', 'sales'] + feature_cols]

    # =========================================================================
    # 7. Print summary statistics
    # =========================================================================
    print(f"\nOutput shape: {df.shape}")
    print(f"\nData types:")
    print(df.dtypes)

    print(f"\nFeature summary statistics:")
    print(df[feature_cols].describe())

    print(f"\nFirst 10 rows:")
    print(df.head(10))

    # =========================================================================
    # 8. Save to CSV
    # =========================================================================
    output_path = "data/featured_sales.csv"
    print(f"\nSaving featured data to {output_path}...")
    df.to_csv(output_path, index=False)

    print("\n" + "=" * 80)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 80 + "\n")

    return df


def main():
    """Test feature engineering with cleaned data."""
    cleaned_csv_path = "data/cleaned_sales.csv"

    print(f"Loading cleaned data from {cleaned_csv_path}...")
    df = pd.read_csv(cleaned_csv_path)
    df['date'] = pd.to_datetime(df['date'])

    # Create features
    df_featured = create_features(df)

    # Show additional analysis
    print("=" * 80)
    print("FEATURE CORRELATION ANALYSIS")
    print("=" * 80)

    # Select numeric columns only
    numeric_cols = df_featured.select_dtypes(include=[np.number]).columns

    # Calculate correlation with sales
    correlations = df_featured[numeric_cols].corr()['sales'].sort_values(ascending=False)
    print("\nCorrelation with sales:")
    print(correlations)

    print("\n" + "=" * 80)
    print("Missing values per column:")
    print(df_featured.isnull().sum())

    return df_featured


if __name__ == "__main__":
    df_featured = main()
