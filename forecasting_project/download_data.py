#!/usr/bin/env python
"""
Download time series data from Google Sheets
"""

from src.data_loader import DataLoader

if __name__ == "__main__":
    loader = DataLoader("data")

    sheets_url = "https://docs.google.com/spreadsheets/d/1I1sFHSOZa9tdQfCahF1W71L4hPrJxkf1/edit#gid=1562810345"

    try:
        df = loader.download_google_sheets(sheets_url, "raw_sales.csv")
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
