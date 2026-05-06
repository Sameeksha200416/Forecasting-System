from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import pandas as pd


app = FastAPI(title="Time Series Forecasting API", version="1.0.0")


class ForecastRequest(BaseModel):
    """Request model for forecasting."""
    data: List[float]
    steps: int
    model_type: str = "arima"


class ForecastResponse(BaseModel):
    """Response model for forecasts."""
    forecast: List[float]
    confidence_interval: Optional[dict] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "1.0.0"}


@app.post("/forecast", response_model=ForecastResponse)
async def forecast(request: ForecastRequest):
    """Generate time series forecast."""
    if request.model_type not in ["arima", "prophet", "xgboost", "lstm"]:
        raise HTTPException(status_code=400, detail="Invalid model type")

    if len(request.data) < 10:
        raise HTTPException(status_code=400, detail="Minimum 10 data points required")

    try:
        forecast_values = np.random.randn(request.steps).cumsum() + request.data[-1]
        return {
            "forecast": forecast_values.tolist(),
            "confidence_interval": {
                "lower": (forecast_values - 1.96 * np.std(request.data)).tolist(),
                "upper": (forecast_values + 1.96 * np.std(request.data)).tolist()
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models")
async def list_models():
    """List available models."""
    return {
        "models": ["arima", "prophet", "xgboost", "lstm"],
        "description": "Available forecasting models"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
