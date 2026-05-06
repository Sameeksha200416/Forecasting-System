#!/usr/bin/env python
"""
Run Exploratory Data Analysis
"""

from src.eda import SalesEDA

if __name__ == "__main__":
    eda = SalesEDA("data/raw_sales.csv")
    df = eda.run()
