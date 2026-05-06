import pandas as pd
import numpy as np
import requests
from pathlib import Path
from typing import Tuple, Optional
from io import StringIO


class DataLoader:
    """Load time series data from CSV files and Google Sheets."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

    def load_csv(self, filename: str) -> pd.DataFrame:
        """Load CSV file from data directory."""
        filepath = self.data_dir / filename
        return pd.read_csv(filepath)

    def load_multiple(self, filenames: list) -> dict:
        """Load multiple CSV files."""
        data = {}
        for filename in filenames:
            data[filename] = self.load_csv(filename)
        return data

    def save_csv(self, df: pd.DataFrame, filename: str, subfolder: str = "") -> None:
        """Save dataframe to CSV."""
        if subfolder:
            save_dir = self.data_dir / subfolder
            save_dir.mkdir(exist_ok=True)
        else:
            save_dir = self.data_dir

        save_dir.mkdir(exist_ok=True)
        filepath = save_dir / filename
        df.to_csv(filepath, index=False)

    @staticmethod
    def convert_sheets_url(sheets_url: str) -> str:
        """Convert Google Sheets URL to CSV export format."""
        try:
            if "/edit" in sheets_url:
                sheet_id = sheets_url.split("/d/")[1].split("/")[0]
            else:
                sheet_id = sheets_url.split("/d/")[1].split("/")[0]

            gid = "0"
            if "#gid=" in sheets_url:
                gid = sheets_url.split("#gid=")[1]
            elif "gid=" in sheets_url:
                gid = sheets_url.split("gid=")[1].split("&")[0]

            export_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
            return export_url
        except Exception as e:
            raise ValueError(f"Failed to parse Google Sheets URL: {e}")

    def download_google_sheets(self, sheets_url: str, filename: str) -> pd.DataFrame:
        """Download data from Google Sheets and save to CSV."""
        try:
            print(f"Converting Google Sheets URL...")
            export_url = self.convert_sheets_url(sheets_url)
            print(f"Export URL: {export_url}\n")

            print(f"Downloading data from Google Sheets...")
            response = requests.get(export_url, timeout=10)
            response.raise_for_status()

            df = pd.read_csv(StringIO(response.text))
            print(f"Data downloaded successfully!\n")

            print("=" * 60)
            print("DATA INFORMATION:")
            print("=" * 60)
            print(f"\nShape: {df.shape}")
            print(f"\nColumns: {list(df.columns)}")
            print(f"\nData Types:\n{df.dtypes}")
            print(f"\nFirst few rows:\n{df.head()}\n")

            print("=" * 60)
            print("Saving to CSV...")
            self.save_csv(df, filename)
            print(f"Saved to {self.data_dir / filename}\n")

            return df

        except requests.exceptions.Timeout:
            raise ConnectionError("Request timed out. Check your internet connection.")
        except requests.exceptions.ConnectionError:
            raise ConnectionError("Failed to connect to Google Sheets. Check your internet connection.")
        except requests.exceptions.HTTPError as e:
            raise ConnectionError(f"HTTP Error: {e.response.status_code}. URL may be invalid or not publicly accessible.")
        except pd.errors.ParserError:
            raise ValueError("Failed to parse CSV data. The sheet may be empty or malformed.")
        except Exception as e:
            raise RuntimeError(f"Unexpected error: {e}")
