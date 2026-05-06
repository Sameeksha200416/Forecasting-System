# Time Series Forecasting System

A comprehensive Python project for time series forecasting using multiple machine learning approaches including ARIMA, Prophet, XGBoost, and LSTM models.

## Project Structure

```
forecasting_project/
├── data/                    # Raw and processed CSV files
├── notebooks/               # EDA and analysis notebooks
├── models/                  # Saved trained model files
├── src/
│   ├── data_loader.py      # Data loading utilities
│   ├── preprocessor.py     # Data preprocessing functions
│   ├── feature_engineering.py  # Feature creation
│   ├── model_arima.py      # ARIMA implementation
│   ├── model_prophet.py    # Facebook Prophet implementation
│   ├── model_xgboost.py    # XGBoost implementation
│   ├── model_lstm.py       # LSTM neural network implementation
│   ├── model_selector.py   # Model selection and evaluation
│   └── api.py              # FastAPI endpoints
├── requirements.txt         # Project dependencies
└── README.md               # This file
```

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the API Server
```bash
uvicorn src.api:app --reload
```

### Running Models
See individual model files in the `src/` directory for usage examples.

## Models Included

- **ARIMA**: Statistical time series model
- **Prophet**: Facebook's forecasting library with seasonality handling
- **XGBoost**: Gradient boosting for regression
- **LSTM**: Deep learning sequence-to-sequence model
- **Model Selector**: Automated model selection and comparison

## Data

Place your raw data in the `data/` folder. Processed data will be saved after preprocessing.

## Results

Trained models are saved in the `models/` folder with timestamps for tracking.

## Dependencies

- Data Processing: pandas, numpy
- Visualization: matplotlib, seaborn
- Statistical Modeling: statsmodels, prophet
- Machine Learning: scikit-learn, xgboost
- Deep Learning: tensorflow, keras
- API: fastapi, uvicorn
- Utilities: joblib, requests, holidays, openpyxl

## License

MIT
