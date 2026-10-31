"""
EcoCircuit AI — E-Waste Forecasting Module (Module A)
=======================================================
Analyzes historical e-waste trends and produces time-series forecasts.
Uses chronological splits only. Never uses random splits for time-series data.

Methodology:
  1. Naive / last-observation baseline
  2. Moving-average baseline
  3. Linear trend (OLS)
  4. Exponential smoothing (Holt's linear)
  5. ARIMA (if statsmodels available and sufficient data)

Metrics: MAE, RMSE, MAPE (only when no zeros in actuals)

Author: EcoCircuit AI Team
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.arima.model import ARIMA
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    warnings.warn("statsmodels not available; Holt-Winters and ARIMA will be skipped.")

logger = logging.getLogger("ewaste_forecasting")

MIN_TRAIN_OBSERVATIONS = 5  # Minimum annual observations for time-series models


# ── Metrics ────────────────────────────────────────────────────────────────

def compute_metrics(actual: np.ndarray, predicted: np.ndarray, model_name: str) -> Dict:
    """Compute regression forecast metrics."""
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    # MAPE only when no zeros in actuals
    if np.all(actual != 0):
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    else:
        mape = None

    metrics = {
        "model": model_name,
        "n_test": len(actual),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "mape_pct": round(mape, 2) if mape is not None else "N/A (zeros in actuals)",
    }
    logger.info("[%s] MAE=%.3f | RMSE=%.3f | MAPE=%s", model_name, mae, rmse, metrics["mape_pct"])
    return metrics


# ── Baseline models ────────────────────────────────────────────────────────

def naive_forecast(train: np.ndarray, n_steps: int) -> np.ndarray:
    """Last-observation naive forecast."""
    return np.full(n_steps, train[-1])


def moving_average_forecast(train: np.ndarray, n_steps: int, window: int = 3) -> np.ndarray:
    """Moving-average forecast using last `window` observations."""
    w = min(window, len(train))
    val = np.mean(train[-w:])
    return np.full(n_steps, val)


def linear_trend_forecast(
    train_years: np.ndarray,
    train_values: np.ndarray,
    forecast_years: np.ndarray,
) -> Tuple[np.ndarray, object]:
    """OLS linear trend forecast. Returns (predictions, fitted model)."""
    X_train = train_years.reshape(-1, 1)
    X_forecast = forecast_years.reshape(-1, 1)
    model = LinearRegression()
    model.fit(X_train, train_values)
    predictions = model.predict(X_forecast)
    return predictions, model


# ── Holt-Winters (double exponential smoothing) ────────────────────────────

def holtwinters_forecast(
    train_values: np.ndarray,
    n_steps: int,
) -> Optional[np.ndarray]:
    """
    Holt's linear (double exponential smoothing) for trend.
    Returns None if statsmodels unavailable or training fails.
    """
    if not STATSMODELS_AVAILABLE:
        return None

    try:
        model = ExponentialSmoothing(
            train_values,
            trend="add",
            seasonal=None,
            initialization_method="estimated",
        )
        fit = model.fit(optimized=True, remove_bias=True)
        forecast = fit.forecast(n_steps)
        return np.array(forecast)
    except Exception as exc:
        logger.warning("Holt-Winters failed: %s", exc)
        return None


# ── ARIMA ──────────────────────────────────────────────────────────────────

def arima_forecast(
    train_values: np.ndarray,
    n_steps: int,
    order: Tuple[int, int, int] = (1, 1, 0),
) -> Optional[np.ndarray]:
    """
    ARIMA forecast. Returns None if insufficient data or fitting fails.
    For small annual datasets (< 10 obs), ARIMA is unreliable;
    this is surfaced in the limitation summary.
    """
    if not STATSMODELS_AVAILABLE:
        return None
    if len(train_values) < 8:
        logger.warning("Too few observations for ARIMA (%d < 8). Skipping.", len(train_values))
        return None
    try:
        model = ARIMA(train_values, order=order)
        fit = model.fit()
        forecast = fit.forecast(steps=n_steps)
        return np.array(forecast)
    except Exception as exc:
        logger.warning("ARIMA(%s) failed: %s", order, exc)
        return None


# ── Master forecasting function ────────────────────────────────────────────

def run_ewaste_forecast(
    df: pd.DataFrame,
    year_col: str = "year",
    value_col: str = "global_ewaste_mt",
    test_frac: float = 0.25,
    horizon: int = 5,
) -> Dict:
    """
    Run the full e-waste forecasting workflow on a time-series DataFrame.

    Steps:
      1. Chronological train/test split
      2. Evaluate baseline models on test set
      3. Evaluate trend + smoothing models
      4. Select best model by RMSE
      5. Produce future forecasts with best model

    Returns a dict with:
      - 'train_df', 'test_df'
      - 'model_results': list of metric dicts
      - 'best_model': name
      - 'future_df': DataFrame with future year forecasts
      - 'limitations': list of warning strings
    """
    df = df.dropna(subset=[year_col, value_col]).sort_values(year_col).reset_index(drop=True)
    n = len(df)

    limitations = []
    if n < MIN_TRAIN_OBSERVATIONS:
        limitations.append(
            f"Only {n} annual observations available. "
            "Forecasts have very high uncertainty. "
            "Results should be treated as directional only."
        )

    # Chronological split
    n_test = max(1, int(n * test_frac))
    n_train = n - n_test
    train_df = df.iloc[:n_train].copy()
    test_df = df.iloc[n_train:].copy()

    train_years = train_df[year_col].values.astype(float)
    train_values = train_df[value_col].values.astype(float)
    test_years = test_df[year_col].values.astype(float)
    test_values = test_df[value_col].values.astype(float)

    logger.info(
        "Forecasting '%s': train=%d obs (%d–%d), test=%d obs (%d–%d)",
        value_col, len(train_df),
        int(train_years[0]), int(train_years[-1]),
        len(test_df),
        int(test_years[0]), int(test_years[-1]),
    )

    model_results = []
    all_preds = {}

    # 1. Naive baseline
    naive_pred = naive_forecast(train_values, len(test_values))
    all_preds["Naive"] = naive_pred
    model_results.append(compute_metrics(test_values, naive_pred, "Naive"))

    # 2. Moving average (window=3)
    ma_pred = moving_average_forecast(train_values, len(test_values), window=3)
    all_preds["Moving Average (3)"] = ma_pred
    model_results.append(compute_metrics(test_values, ma_pred, "Moving Average (3)"))

    # 3. Linear trend
    lin_pred, lin_model = linear_trend_forecast(train_years, train_values, test_years)
    all_preds["Linear Trend"] = lin_pred
    model_results.append(compute_metrics(test_values, lin_pred, "Linear Trend"))

    # 4. Holt-Winters (if available)
    hw_pred = holtwinters_forecast(train_values, len(test_values))
    if hw_pred is not None:
        all_preds["Holt-Winters"] = hw_pred
        model_results.append(compute_metrics(test_values, hw_pred, "Holt-Winters"))
    else:
        limitations.append("Holt-Winters model not available (statsmodels may not be installed).")

    # 5. ARIMA
    arima_pred = arima_forecast(train_values, len(test_values))
    if arima_pred is not None:
        all_preds["ARIMA(1,1,0)"] = arima_pred
        model_results.append(compute_metrics(test_values, arima_pred, "ARIMA(1,1,0)"))
    else:
        limitations.append("ARIMA skipped (insufficient data or unavailable library).")

    # Select best by RMSE
    valid_results = [r for r in model_results if isinstance(r["rmse"], (int, float))]
    best = min(valid_results, key=lambda x: x["rmse"])
    best_model_name = best["model"]
    logger.info("Best model: %s (RMSE=%.3f)", best_model_name, best["rmse"])

    # Future forecast with best model
    last_year = int(df[year_col].max())
    future_years = np.arange(last_year + 1, last_year + 1 + horizon, dtype=float)

    if best_model_name == "Naive":
        future_vals = naive_forecast(df[value_col].values, horizon)
    elif best_model_name.startswith("Moving"):
        future_vals = moving_average_forecast(df[value_col].values, horizon)
    elif best_model_name == "Linear Trend":
        all_years = df[year_col].values.astype(float)
        all_vals = df[value_col].values.astype(float)
        future_vals, _ = linear_trend_forecast(all_years, all_vals, future_years)
    elif best_model_name == "Holt-Winters":
        future_vals = holtwinters_forecast(df[value_col].values.astype(float), horizon)
        if future_vals is None:
            future_vals = naive_forecast(df[value_col].values, horizon)
    elif best_model_name.startswith("ARIMA"):
        future_vals = arima_forecast(df[value_col].values.astype(float), horizon)
        if future_vals is None:
            future_vals = naive_forecast(df[value_col].values, horizon)
    else:
        future_vals = naive_forecast(df[value_col].values, horizon)

    # Build forecast DataFrame
    future_df = pd.DataFrame({
        year_col: future_years.astype(int),
        f"{value_col}_forecast": np.round(future_vals, 2),
        "forecast_type": "model_forecast",
        "model_used": best_model_name,
        "is_historical": False,
    })

    # Add historical data tagged
    hist_df = df[[year_col, value_col]].copy()
    hist_df["forecast_type"] = "historical"
    hist_df["is_historical"] = True

    # Test-set predictions
    test_df = test_df.copy()
    test_df[f"{value_col}_forecast"] = all_preds[best_model_name]
    test_df["forecast_type"] = "test_prediction"
    test_df["is_historical"] = False

    return {
        "train_df": train_df,
        "test_df": test_df,
        "hist_df": hist_df,
        "future_df": future_df,
        "model_results": model_results,
        "best_model": best_model_name,
        "best_metrics": best,
        "all_predictions": all_preds,
        "test_values": test_values,
        "test_years": test_years,
        "train_years": train_years,
        "train_values": train_values,
        "limitations": limitations,
        "value_col": value_col,
        "year_col": year_col,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")

    # Test with the curated global e-waste timeseries
    from src.preprocessing import load_global_ewaste_timeseries
    df = load_global_ewaste_timeseries()

    if df is not None:
        results = run_ewaste_forecast(df, "year", "global_ewaste_mt", horizon=5)
        print(f"\nBest model: {results['best_model']}")
        print(f"Test RMSE: {results['best_metrics']['rmse']}")
        print("\nModel comparison:")
        for r in results["model_results"]:
            print(f"  {r['model']:25s} MAE={r['mae']:.3f} RMSE={r['rmse']:.3f}")
        print("\nFuture forecast:")
        print(results["future_df"])
        if results["limitations"]:
            print("\nLimitations:")
            for lim in results["limitations"]:
                print(f"  ⚠ {lim}")
