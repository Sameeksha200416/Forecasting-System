#!/usr/bin/env python
"""
Exploratory Data Analysis for Sales Data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class SalesEDA:
    """Perform comprehensive EDA on sales data."""

    def __init__(self, csv_path: str = "data/raw_sales.csv"):
        self.csv_path = csv_path
        self.data_dir = Path(csv_path).parent
        self.df = None
        self.date_column = None
        self.region_column = None
        self.sales_column = None

    def load_data(self) -> None:
        """Load CSV file."""
        try:
            self.df = pd.read_csv(self.csv_path)
            print(f"Data loaded successfully from {self.csv_path}\n")
        except FileNotFoundError:
            print(f"Error: File not found at {self.csv_path}")
            exit(1)

    def print_basic_info(self) -> None:
        """Print basic information about the dataset."""
        print("=" * 80)
        print("BASIC DATASET INFORMATION")
        print("=" * 80)
        print(f"\nDataset Shape: {self.df.shape}")
        print(f"Rows: {self.df.shape[0]}, Columns: {self.df.shape[1]}\n")

        print("Column Names:")
        print(list(self.df.columns))
        print()

        print("Data Types:")
        print(self.df.dtypes)
        print()

        print("First 10 Rows:")
        print(self.df.head(10))
        print()

    def check_missing_values(self) -> None:
        """Check missing values and print statistics."""
        print("=" * 80)
        print("MISSING VALUES ANALYSIS")
        print("=" * 80)

        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        missing_df = pd.DataFrame({
            "Column": missing.index,
            "Missing Count": missing.values,
            "Percentage (%)": missing_pct.values
        })

        print("\nMissing Values Summary:")
        print(missing_df.to_string(index=False))
        print()

    def analyze_numeric_columns(self) -> None:
        """Print statistics for numeric columns."""
        print("=" * 80)
        print("NUMERIC COLUMNS STATISTICS")
        print("=" * 80)

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            print("No numeric columns found.\n")
            return

        stats = self.df[numeric_cols].describe().T
        stats["missing_count"] = self.df[numeric_cols].isnull().sum()
        print("\n" + stats.to_string())
        print()

    def find_date_column(self) -> None:
        """Find and analyze date/week column."""
        print("=" * 80)
        print("DATE/TIME ANALYSIS")
        print("=" * 80)

        date_keywords = ["date", "week", "time", "day", "month", "year"]
        potential_date_cols = [col for col in self.df.columns
                               if any(keyword in col.lower() for keyword in date_keywords)]

        if potential_date_cols:
            self.date_column = potential_date_cols[0]
            print(f"\nFound potential date column: {self.date_column}")
            print(f"Data type: {self.df[self.date_column].dtype}")
            print(f"Unique values: {self.df[self.date_column].nunique()}")
            print(f"Range: {self.df[self.date_column].min()} to {self.df[self.date_column].max()}")
            print()
        else:
            print("\nNo date/week column detected.\n")

    def find_region_column(self) -> None:
        """Find and analyze state/region column."""
        print("=" * 80)
        print("REGION/STATE ANALYSIS")
        print("=" * 80)

        region_keywords = ["state", "region", "city", "location", "area", "country"]
        potential_region_cols = [col for col in self.df.columns
                                 if any(keyword in col.lower() for keyword in region_keywords)]

        if potential_region_cols:
            self.region_column = potential_region_cols[0]
            print(f"\nFound region column: {self.region_column}")
            print(f"Unique values: {self.df[self.region_column].nunique()}")
            print(f"\nRegion/State Counts:")
            region_counts = self.df[self.region_column].value_counts()
            print(region_counts.to_string())
            print()
        else:
            print("\nNo region/state column detected.\n")

    def find_sales_column(self) -> None:
        """Find sales/target column."""
        sales_keywords = ["sales", "revenue", "amount", "value", "total"]
        potential_sales_cols = [col for col in self.df.select_dtypes(include=[np.number]).columns
                                if any(keyword in col.lower() for keyword in sales_keywords)]

        if potential_sales_cols:
            self.sales_column = potential_sales_cols[0]
        else:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                self.sales_column = numeric_cols[0]

    def plot_sales_over_time(self) -> None:
        """Plot total sales over time."""
        if not self.date_column or not self.sales_column:
            print("Cannot plot sales over time: missing date or sales column.\n")
            return

        print("Generating: Sales Over Time (Line Chart)...")

        df_time = self.df.groupby(self.date_column)[self.sales_column].sum().reset_index()

        plt.figure(figsize=(14, 6))
        plt.plot(df_time[self.date_column], df_time[self.sales_column], linewidth=2, color='steelblue')
        plt.title(f"Total Sales Over Time", fontsize=14, fontweight='bold')
        plt.xlabel(self.date_column, fontsize=12)
        plt.ylabel(f"Total {self.sales_column}", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()

        filepath = self.data_dir / "sales_over_time.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved to {filepath}\n")
        plt.close()

    def plot_sales_by_region(self) -> None:
        """Plot total sales per region/state."""
        if not self.region_column or not self.sales_column:
            print("Cannot plot sales by region: missing region or sales column.\n")
            return

        print("Generating: Sales by Region (Bar Chart)...")

        df_region = self.df.groupby(self.region_column)[self.sales_column].sum().sort_values(ascending=False)

        plt.figure(figsize=(12, 6))
        df_region.plot(kind='bar', color='coral', edgecolor='black')
        plt.title(f"Total Sales by {self.region_column}", fontsize=14, fontweight='bold')
        plt.xlabel(self.region_column, fontsize=12)
        plt.ylabel(f"Total {self.sales_column}", fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()

        filepath = self.data_dir / "sales_by_region.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved to {filepath}\n")
        plt.close()

    def print_summary(self) -> None:
        """Print summary of dataset characteristics."""
        print("=" * 80)
        print("DATASET SUMMARY")
        print("=" * 80)

        num_regions = self.df[self.region_column].nunique() if self.region_column else "Unknown"
        date_range = f"{self.df[self.date_column].min()} to {self.df[self.date_column].max()}" if self.date_column else "Unknown"
        target_col = self.sales_column if self.sales_column else "Unknown"

        summary = f"""
Dataset Overview:
  • Total Records: {len(self.df):,}
  • Total Columns: {len(self.df.columns)}
  • Number of {self.region_column if self.region_column else 'regions'}: {num_regions}
  • Date Range: {date_range}
  • Target Column: {target_col}
  • Missing Data: {self.df.isnull().sum().sum()} cells ({(self.df.isnull().sum().sum() / (len(self.df) * len(self.df.columns)) * 100):.2f}%)

Columns: {', '.join(self.df.columns)}
"""
        print(summary)

    def run(self) -> pd.DataFrame:
        """Run complete EDA pipeline."""
        print("\n" + "=" * 80)
        print("EXPLORATORY DATA ANALYSIS - SALES DATA")
        print("=" * 80 + "\n")

        self.load_data()
        self.print_basic_info()
        self.check_missing_values()
        self.analyze_numeric_columns()
        self.find_date_column()
        self.find_region_column()
        self.find_sales_column()
        self.plot_sales_over_time()
        self.plot_sales_by_region()
        self.print_summary()

        print("=" * 80)
        print("EDA COMPLETE")
        print("=" * 80 + "\n")

        return self.df


if __name__ == "__main__":
    eda = SalesEDA("data/raw_sales.csv")
    df = eda.run()
