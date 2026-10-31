"""
EcoCircuit AI — Battery Degradation Analysis Module (Module C)
===============================================================
Analyzes Li-ion battery capacity fade and estimates battery health.

Uses the curated/synthetic battery demo dataset (based on NASA PCoE parameters)
OR the real NASA dataset if the user places converted CSV files in data/raw/batteries/.

Models:
  1. Linear regression baseline on cycle number
  2. Random Forest regression
  3. Gradient Boosting regression

Target: capacity_ah (or capacity_retention_pct)
Per-battery validation to prevent data leakage.

Author: EcoCircuit AI Team
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

logger = logging.getLogger("battery_analysis")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
RAW_BATTERY_DIR = PROJECT_ROOT / "data" / "raw" / "batteries" / "nasa_battery"
SYNTHETIC_DIR = PROJECT_ROOT / "data" / "synthetic_demo"


# ── NASA battery CSV loader ─────────────────────────────────────────────────

def load_nasa_battery_csv() -> Optional[pd.DataFrame]:
    """
    Attempt to load real NASA battery data if user placed CSVs in raw/batteries/nasa_battery/.
    Expected columns (from community-converted NASA PCoE datasets):
      battery_id, cycle_number, capacity_ah (or capacity), ...

    Returns None if no CSV found.
    """
    if not RAW_BATTERY_DIR.exists():
        return None
    csv_files = list(RAW_BATTERY_DIR.glob("*.csv"))
    if not csv_files:
        return None

    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            df["source_file"] = f.name
            df["data_type"] = "NASA_PCoE_real"
            dfs.append(df)
            logger.info("Loaded NASA battery CSV: %s (%d rows)", f.name, len(df))
        except Exception as exc:
            logger.warning("Failed to read %s: %s", f.name, exc)

    if not dfs:
        return None

    combined = pd.concat(dfs, ignore_index=True)
    logger.info("Combined NASA battery data: %d rows", len(combined))
    return combined


def load_battery_data() -> Tuple[pd.DataFrame, str]:
    """
    Load battery data — real NASA if available, else synthetic demo.
    Returns (DataFrame, data_type_label).
    """
    # Try real NASA data first
    nasa_df = load_nasa_battery_csv()
    if nasa_df is not None:
        return nasa_df, "real_nasa"

    # Fall back to synthetic demo
    demo_path = SYNTHETIC_DIR / "battery_demo.csv"
    if demo_path.exists():
        df = pd.read_csv(demo_path)
        logger.info(
            "Using synthetic battery demo data (%d rows). "
            "Place NASA CSV files in data/raw/batteries/nasa_battery/ for real analysis.",
            len(df),
        )
        return df, "synthetic_demo"

    logger.error("No battery data available. Run synthetic_data.py to generate demo data.")
    return pd.DataFrame(), "unavailable"


# ── Feature engineering for batteries ─────────────────────────────────────

def prepare_battery_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare features for battery degradation modelling.
    Assumes columns: battery_id, cycle_number, capacity_ah, initial_capacity_ah.
    """
    df = df.copy().dropna(subset=["cycle_number", "capacity_ah"])
    df = df.sort_values(["battery_id", "cycle_number"]).reset_index(drop=True)

    # Normalize cycle by battery
    if "initial_capacity_ah" in df.columns:
        df["capacity_retention"] = df["capacity_ah"] / df["initial_capacity_ah"]
    else:
        # Estimate initial capacity as max capacity for each battery
        df["initial_capacity_ah"] = df.groupby("battery_id")["capacity_ah"].transform("max")
        df["capacity_retention"] = df["capacity_ah"] / df["initial_capacity_ah"]

    # Cycle features
    df["cycle_norm"] = df.groupby("battery_id")["cycle_number"].transform(
        lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)
    )
    df["log_cycle"] = np.log1p(df["cycle_number"])

    # Lag features (per battery)
    df["capacity_lag1"] = df.groupby("battery_id")["capacity_ah"].shift(1)
    df["capacity_lag3"] = df.groupby("battery_id")["capacity_ah"].shift(3)
    df["capacity_diff1"] = df.groupby("battery_id")["capacity_ah"].diff(1)

    # Rolling degradation rate
    df["capacity_roll_mean_5"] = (
        df.groupby("battery_id")["capacity_ah"]
        .transform(lambda x: x.rolling(5, min_periods=1).mean())
    )

    return df


# ── Battery metrics ────────────────────────────────────────────────────────

def battery_metrics(actual: np.ndarray, predicted: np.ndarray, model_name: str) -> Dict:
    """Compute regression metrics for battery capacity prediction."""
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)
    return {
        "model": model_name,
        "mae": round(mae, 5),
        "rmse": round(rmse, 5),
        "r2": round(r2, 4),
        "n": len(actual),
    }


# ── Per-battery train / test split ─────────────────────────────────────────

def battery_train_test_split(df: pd.DataFrame, test_battery_ids: Optional[List] = None):
    """
    Split batteries into train/test sets so that no battery appears in both.
    If test_battery_ids is None, hold out the last 25% of unique batteries.
    """
    all_ids = df["battery_id"].unique().tolist()
    if test_battery_ids is None:
        n_test = max(1, int(len(all_ids) * 0.25))
        test_battery_ids = all_ids[-n_test:]

    train_df = df[~df["battery_id"].isin(test_battery_ids)]
    test_df = df[df["battery_id"].isin(test_battery_ids)]

    logger.info(
        "Battery split: train=%d batteries (%d rows), test=%d batteries (%d rows)",
        train_df["battery_id"].nunique(), len(train_df),
        test_df["battery_id"].nunique(), len(test_df),
    )
    return train_df, test_df


# ── Model training ─────────────────────────────────────────────────────────

FEATURE_COLS = [
    "cycle_number", "log_cycle", "cycle_norm",
    "capacity_lag1", "capacity_lag3",
    "capacity_diff1", "capacity_roll_mean_5",
]
TARGET_COL = "capacity_ah"


def train_battery_models(df: pd.DataFrame) -> Dict:
    """
    Train and evaluate battery degradation models.
    Uses per-battery split to prevent leakage.
    Returns dict with model objects and evaluation results.
    """
    df = prepare_battery_features(df)
    train_df, test_df = battery_train_test_split(df)

    feature_cols = [c for c in FEATURE_COLS if c in train_df.columns]
    logger.info("Battery features used: %s", feature_cols)

    # Drop rows where any feature is NaN (due to lags)
    train_clean = train_df.dropna(subset=feature_cols + [TARGET_COL])
    test_clean = test_df.dropna(subset=feature_cols + [TARGET_COL])

    X_train = train_clean[feature_cols].values
    y_train = train_clean[TARGET_COL].values
    X_test = test_clean[feature_cols].values
    y_test = test_clean[TARGET_COL].values

    results = {}

    # 1. Linear Regression baseline
    lr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LinearRegression()),
    ])
    lr_pipeline.fit(X_train, y_train)
    lr_pred = lr_pipeline.predict(X_test)
    results["Linear Regression"] = {
        "model": lr_pipeline,
        "metrics": battery_metrics(y_test, lr_pred, "Linear Regression"),
    }

    # 2. Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    results["Random Forest"] = {
        "model": rf,
        "metrics": battery_metrics(y_test, rf_pred, "Random Forest"),
        "feature_importance": dict(zip(feature_cols, rf.feature_importances_)),
    }

    # 3. Gradient Boosting
    gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42)
    gb.fit(X_train, y_train)
    gb_pred = gb.predict(X_test)
    results["Gradient Boosting"] = {
        "model": gb,
        "metrics": battery_metrics(y_test, gb_pred, "Gradient Boosting"),
        "feature_importance": dict(zip(feature_cols, gb.feature_importances_)),
    }

    # Best model by RMSE
    best_name = min(results, key=lambda k: results[k]["metrics"]["rmse"])
    logger.info(
        "Best battery model: %s (RMSE=%.5f, R2=%.3f)",
        best_name,
        results[best_name]["metrics"]["rmse"],
        results[best_name]["metrics"]["r2"],
    )

    # Save models
    if JOBLIB_AVAILABLE:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        for name, obj in results.items():
            safe_name = name.lower().replace(" ", "_")
            path = MODELS_DIR / f"battery_{safe_name}.pkl"
            joblib.dump(obj["model"], path)
            logger.info("Saved model: %s", path.name)

    # Attach test data for visualization
    results["_test_df"] = test_clean.copy()
    results["_feature_cols"] = feature_cols
    results["_best_model"] = best_name
    results["_y_test"] = y_test
    results["_predictions"] = {
        "Linear Regression": lr_pred,
        "Random Forest": rf_pred,
        "Gradient Boosting": gb_pred,
    }

    return results


def estimate_rul(df: pd.DataFrame, battery_id: str, threshold: float = 0.8) -> Dict:
    """
    Estimate Remaining Useful Life (RUL) for a specific battery.
    RUL = estimated number of cycles until capacity retention drops below threshold.

    NOTE: This is an estimate based on fitted degradation trend.
    Laboratory results may not generalize to field conditions.
    """
    bat_df = df[df["battery_id"] == battery_id].sort_values("cycle_number")
    if len(bat_df) < 10:
        return {
            "battery_id": battery_id,
            "rul_estimate": None,
            "note": "Insufficient cycles for reliable RUL estimate.",
        }

    initial_cap = bat_df["initial_capacity_ah"].iloc[0] if "initial_capacity_ah" in bat_df.columns \
        else bat_df["capacity_ah"].max()
    eol_capacity = initial_cap * threshold
    current_capacity = bat_df["capacity_ah"].iloc[-1]
    current_cycle = bat_df["cycle_number"].iloc[-1]

    if current_capacity <= eol_capacity:
        return {
            "battery_id": battery_id,
            "rul_estimate": 0,
            "current_capacity_ah": round(float(current_capacity), 4),
            "current_retention_pct": round(100.0 * current_capacity / initial_cap, 2),
            "status": "end_of_life",
            "note": f"Capacity already below {threshold*100:.0f}% retention threshold.",
        }

    # Fit log-linear degradation model
    X = np.log1p(bat_df["cycle_number"].values).reshape(-1, 1)
    y = bat_df["capacity_ah"].values
    model = LinearRegression().fit(X, y)

    # Solve for cycle where capacity hits eol_capacity
    from scipy.optimize import brentq
    def objective(c):
        return model.predict([[np.log1p(c)]])[0] - eol_capacity

    try:
        from scipy.optimize import brentq
        rul_cycle = brentq(objective, current_cycle, current_cycle + 10000)
        rul = max(0, rul_cycle - current_cycle)
    except Exception:
        # If scipy unavailable, use simple linear extrapolation
        slope = model.coef_[0]
        if slope >= 0:
            rul = None
        else:
            rul_cycle = current_cycle + (current_capacity - eol_capacity) / (-slope * 0.0001)
            rul = max(0, rul_cycle - current_cycle)

    return {
        "battery_id": battery_id,
        "rul_estimate": round(float(rul), 0) if rul is not None else None,
        "current_cycle": int(current_cycle),
        "current_capacity_ah": round(float(current_capacity), 4),
        "current_retention_pct": round(100.0 * current_capacity / initial_cap, 2),
        "eol_threshold_pct": threshold * 100,
        "status": "operational",
        "note": (
            "RUL estimated from log-linear degradation trend on available cycle data. "
            "Actual RUL depends on usage conditions, temperature, charge rate, and other factors."
        ),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")

    df, dtype = load_battery_data()
    if df.empty:
        print("No battery data available. Run: python src/synthetic_data.py")
    else:
        print(f"\nBattery data type: {dtype}")
        print(f"Shape: {df.shape}")
        print(f"Batteries: {df['battery_id'].nunique()}")

        results = train_battery_models(df)
        best = results["_best_model"]
        print(f"\nBest model: {best}")
        for model_name in ["Linear Regression", "Random Forest", "Gradient Boosting"]:
            m = results[model_name]["metrics"]
            print(f"  {model_name:25s} MAE={m['mae']:.5f} RMSE={m['rmse']:.5f} R2={m['r2']:.4f}")
