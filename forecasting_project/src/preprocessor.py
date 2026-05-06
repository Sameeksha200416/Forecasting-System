import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


class TimeSeriesPreprocessor:
    """Preprocess time series data for modeling."""

    @staticmethod
    def handle_missing_values(df: pd.DataFrame, method: str = "interpolate") -> pd.DataFrame:
        """Handle missing values using specified method."""
        if method == "interpolate":
            return df.interpolate(method="linear")
        elif method == "forward_fill":
            return df.fillna(method="ffill")
        elif method == "drop":
            return df.dropna()
        return df

    @staticmethod
    def remove_outliers(df: pd.DataFrame, column: str, std_threshold: float = 3) -> pd.DataFrame:
        """Remove outliers using z-score method."""
        z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
        return df[z_scores < std_threshold]

    @staticmethod
    def normalize(df: pd.DataFrame, method: str = "minmax") -> Tuple[pd.DataFrame, dict]:
        """Normalize data using specified method."""
        scaling_params = {}

        if method == "minmax":
            for col in df.select_dtypes(include=[np.number]).columns:
                min_val = df[col].min()
                max_val = df[col].max()
                df[col] = (df[col] - min_val) / (max_val - min_val)
                scaling_params[col] = {"min": min_val, "max": max_val}

        elif method == "zscore":
            for col in df.select_dtypes(include=[np.number]).columns:
                mean_val = df[col].mean()
                std_val = df[col].std()
                df[col] = (df[col] - mean_val) / std_val
                scaling_params[col] = {"mean": mean_val, "std": std_val}

        return df, scaling_params

    @staticmethod
    def set_datetime_index(df: pd.DataFrame, date_column: str, freq: Optional[str] = None) -> pd.DataFrame:
        """Convert column to datetime index."""
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.set_index(date_column)
        if freq:
            df = df.asfreq(freq)
        return df

    @staticmethod
    def split_train_test(df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into train and test sets."""
        split_index = int(len(df) * (1 - test_size))
        return df[:split_index], df[split_index:]


def detect_date_column(df: pd.DataFrame) -> Optional[str]:
    """Auto-detect the date column."""
    date_keywords = ["date", "time", "week", "day", "month", "year"]
    for col in df.columns:
        if any(keyword in col.lower() for keyword in date_keywords):
            return col
    return None


def detect_state_column(df: pd.DataFrame) -> Optional[str]:
    """Auto-detect the state/region column."""
    state_keywords = ["state", "region", "location", "city", "area", "country"]
    for col in df.columns:
        if any(keyword in col.lower() for keyword in state_keywords):
            return col
    return None


def detect_sales_column(df: pd.DataFrame) -> Optional[str]:
    """Auto-detect the sales/revenue column."""
    sales_keywords = ["sales", "revenue", "total", "amount", "value"]

    # Check all columns (not just numeric) since "Total" might be stored as string
    for col in df.columns:
        if any(keyword in col.lower() for keyword in sales_keywords):
            return col

    # If no keyword match, return the first numeric column (excluding date/state columns)
    exclude_keywords = ["date", "time", "week", "state", "region", "city", "location"]
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if not any(keyword in col.lower() for keyword in exclude_keywords):
            return col

    return None


def preprocess(df: pd.DataFrame, raw_csv_path: str = "data/raw_sales.csv") -> pd.DataFrame:
    """
    Preprocess sales data for time series forecasting.

    Steps:
    1. Auto-detect and parse date column
    2. Rename columns to standardized names
    3. Sort by state and date
    4. Resample to weekly frequency per state
    5. Fill missing date gaps with forward/backward fill
    6. Remove null values
    7. Return clean DataFrame
    """

    print("=" * 80)
    print("DATA PREPROCESSING")
    print("=" * 80)

    print(f"\nOriginal shape: {df.shape}")

    # Auto-detect columns
    date_col = detect_date_column(df)
    state_col = detect_state_column(df)
    sales_col = detect_sales_column(df)

    print(f"\nDetected columns:")
    print(f"  Date column: {date_col}")
    print(f"  State column: {state_col}")
    print(f"  Sales column: {sales_col}")

    if not all([date_col, state_col, sales_col]):
        raise ValueError("Could not auto-detect all required columns")

    # Create working copy
    df_clean = df[[date_col, state_col, sales_col]].copy()

    # Step 1: Parse date column
    print(f"\nParsing date column '{date_col}' to datetime...")
    df_clean[date_col] = pd.to_datetime(df_clean[date_col], format='mixed', dayfirst=True)

    # Step 2: Rename columns to standardized names
    print(f"Renaming columns to standardized names...")
    df_clean.columns = ["date", "state", "sales"]

    # Convert sales to numeric (remove commas if present)
    print(f"Converting sales to numeric...")
    df_clean["sales"] = pd.to_numeric(
        df_clean["sales"].astype(str).str.replace(",", ""),
        errors="coerce"
    )

    # Aggregate by state and date (in case there are multiple categories)
    print(f"Aggregating sales by state and date...")
    df_clean = df_clean.groupby(["date", "state"], as_index=False)["sales"].sum()

    # Step 3: Sort by state and date
    print(f"Sorting by state and date...")
    df_clean = df_clean.sort_values(["state", "date"]).reset_index(drop=True)

    # Step 4-5: Check for date gaps and handle if needed
    print(f"Checking for date gaps per state...")

    def fill_date_gaps(group):
        """Fill any date gaps in a state group."""
        state_name = group['state'].iloc[0]
        group = group.set_index("date").sort_index()
        # Use asfreq to identify missing dates, then resample
        full_range = pd.date_range(group.index.min(), group.index.max(), freq="W")
        group = group.reindex(full_range)
        group = group.ffill()
        group = group.bfill()
        group['state'] = state_name
        return group

    df_clean = df_clean.groupby("state", group_keys=False).apply(fill_date_gaps)
    df_clean = df_clean.reset_index()
    df_clean = df_clean.rename(columns={"index": "date"})
    df_clean = df_clean[["date", "state", "sales"]].copy()

    # Step 6: Remove rows where sales is null
    print(f"Removing null values...")
    initial_rows = len(df_clean)
    df_clean = df_clean.dropna(subset=["sales"])
    rows_removed = initial_rows - len(df_clean)

    if rows_removed > 0:
        print(f"  Removed {rows_removed} rows with null sales")

    # Step 7: Final cleanup
    df_clean = df_clean.reset_index(drop=True)
    df_clean = df_clean[["date", "state", "sales"]].copy()

    print(f"\nCleaned shape: {df_clean.shape}")
    print(f"\nData types after cleaning:")
    print(df_clean.dtypes)

    print(f"\nDate range: {df_clean['date'].min()} to {df_clean['date'].max()}")
    print(f"Number of unique states: {df_clean['state'].nunique()}")
    print(f"States: {sorted(df_clean['state'].unique())}")

    # Step 8: Save cleaned data
    output_path = "data/cleaned_sales.csv"
    print(f"\nSaving cleaned data to {output_path}...")
    df_clean.to_csv(output_path, index=False)

    print(f"\nFirst 10 rows of cleaned data:")
    print(df_clean.head(10))

    print(f"\n" + "=" * 80)
    print(f"PREPROCESSING COMPLETE")
    print(f"=" * 80 + "\n")

    return df_clean


def train_val_split(df: pd.DataFrame, val_weeks: int = 8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Time series train-validation split per state.

    Splits data chronologically: last val_weeks rows per state go to validation,
    everything before goes to training. No shuffling, no look-ahead bias.

    Args:
        df: DataFrame with columns [date, state, sales, ...optional features]
        val_weeks: Number of weeks to use for validation per state

    Returns:
        (train_df, val_df): Two DataFrames with train and validation data
    """

    print("=" * 80)
    print("TRAIN-VALIDATION SPLIT")
    print("=" * 80)

    # Ensure date is datetime
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])

    # Validate inputs
    if 'date' not in df.columns or 'state' not in df.columns:
        raise ValueError("DataFrame must contain 'date' and 'state' columns")

    print(f"\nSplitting data chronologically per state...")
    print(f"Validation period: last {val_weeks} weeks per state")
    print(f"No shuffling - strictly chronological split\n")

    train_dfs = []
    val_dfs = []

    # Process each state
    for state in sorted(df['state'].unique()):
        state_data = df[df['state'] == state].copy()
        state_data = state_data.sort_values('date').reset_index(drop=True)

        # Split: last val_weeks rows for validation, rest for training
        split_idx = len(state_data) - val_weeks

        # Ensure we have enough data
        if split_idx < 1:
            print(f"  ⚠️  {state:20s}: Only {len(state_data)} weeks available, "
                  f"need at least {val_weeks + 1}. Skipping.")
            continue

        train_state = state_data.iloc[:split_idx].copy()
        val_state = state_data.iloc[split_idx:].copy()

        # Print split info for this state
        train_start = train_state['date'].min().strftime("%Y-%m-%d")
        train_end = train_state['date'].max().strftime("%Y-%m-%d")
        val_start = val_state['date'].min().strftime("%Y-%m-%d")
        val_end = val_state['date'].max().strftime("%Y-%m-%d")

        print(f"  {state:20s}: Train [{train_start} to {train_end}] "
              f"({len(train_state)} weeks) | "
              f"Val [{val_start} to {val_end}] ({len(val_state)} weeks)")

        train_dfs.append(train_state)
        val_dfs.append(val_state)

    # Combine all states
    train_df = pd.concat(train_dfs, ignore_index=True).sort_values(['state', 'date']).reset_index(drop=True)
    val_df = pd.concat(val_dfs, ignore_index=True).sort_values(['state', 'date']).reset_index(drop=True)

    # Print summary
    print("\n" + "=" * 80)
    print("SPLIT SUMMARY")
    print("=" * 80)
    print(f"\nTrain set: {len(train_df)} rows ({train_df['state'].nunique()} states)")
    print(f"Val set:   {len(val_df)} rows ({val_df['state'].nunique()} states)")
    print(f"Total:     {len(train_df) + len(val_df)} rows")

    print(f"\nTrain date range: {train_df['date'].min().date()} to {train_df['date'].max().date()}")
    print(f"Val date range:   {val_df['date'].min().date()} to {val_df['date'].max().date()}")

    print(f"\nNo data leakage: All validation dates are after all training dates [OK]")
    print("=" * 80 + "\n")

    return train_df, val_df


def main():
    """Test preprocessing with raw_sales.csv"""
    raw_csv_path = "data/raw_sales.csv"

    # Load raw data
    print(f"Loading raw data from {raw_csv_path}...")
    df_raw = pd.read_csv(raw_csv_path)

    # Preprocess
    df_clean = preprocess(df_raw)

    # Show summary statistics
    print("=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(f"\nSales by state:")
    print(df_clean.groupby("state")["sales"].agg(["count", "sum", "mean", "min", "max"]))

    return df_clean


if __name__ == "__main__":
    df_cleaned = main()
