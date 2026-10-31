"""
EcoCircuit AI — Preprocessing Module
=====================================
Loads, validates, and cleans acquired datasets.
Preserves raw data; writes processed copies only.

Author: EcoCircuit AI Team
"""

import logging
import warnings
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd

logger = logging.getLogger("preprocessing")
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SYNTHETIC_DIR = PROJECT_ROOT / "data" / "synthetic_demo"


def _ensure_processed_dir() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ── Data quality report ─────────────────────────────────────────────────────

def data_quality_report(df: pd.DataFrame, name: str = "Dataset") -> Dict:
    """Return a structured data quality summary."""
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    duplicates = df.duplicated().sum()

    report = {
        "name": name,
        "shape": df.shape,
        "n_rows": len(df),
        "n_cols": df.shape[1],
        "duplicates": int(duplicates),
        "missing_values": missing[missing > 0].to_dict(),
        "missing_pct": missing_pct[missing_pct > 0].to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "numeric_summary": df.describe(include=[np.number]).to_dict(),
    }

    logger.info(
        "[%s] shape=%s | duplicates=%d | cols_with_missing=%d",
        name, df.shape, duplicates, len(report["missing_values"]),
    )
    return report


# ── World Bank What a Waste loader ─────────────────────────────────────────

def load_worldbank_waste_country() -> Optional[pd.DataFrame]:
    """
    Load and validate the World Bank What a Waste 3.0 country dataset.
    Returns None if the file is not available.
    Reminder: This is GENERAL SOLID WASTE data, NOT e-waste.
    """
    file_path = RAW_DIR / "waste_management" / "worldbank_what_a_waste_country.xlsx"

    if not file_path.exists():
        logger.warning(
            "World Bank country dataset not found at %s. "
            "Run data_ingestion.py first.",
            file_path,
        )
        return None

    try:
        # The file has multiple sheets; we load the data sheet
        xl = pd.ExcelFile(file_path)
        logger.info("World Bank XLSX sheets: %s", xl.sheet_names)

        # Try common sheet names
        sheet = None
        for candidate in ["country", "Country", "data", "Data", xl.sheet_names[0]]:
            if candidate in xl.sheet_names:
                sheet = candidate
                break
        if sheet is None:
            sheet = xl.sheet_names[0]

        df = pd.read_excel(file_path, sheet_name=sheet)
        logger.info("Loaded World Bank country data: %s | sheet=%s", df.shape, sheet)

        # Mark source
        df["data_source"] = "World Bank What a Waste 3.0 (General Solid Waste)"
        df["data_type"] = "general_solid_waste_NOT_ewaste"

        # Quality report
        data_quality_report(df, "WorldBank_WhatAWaste_Country")

        # Save processed copy
        _ensure_processed_dir()
        out_path = PROCESSED_DIR / "worldbank_country_waste.parquet"
        df.to_parquet(out_path, index=False)
        logger.info("Saved processed: %s", out_path.name)

        return df

    except Exception as exc:
        logger.error("Failed to load World Bank dataset: %s", exc)
        return None


# ── UCI AI4I Predictive Maintenance loader ─────────────────────────────────

def load_uci_ai4i() -> Optional[pd.DataFrame]:
    """
    Load and validate the UCI AI4I 2020 Predictive Maintenance dataset.
    This is SYNTHETIC industrial data — NOT real electronics failure data.
    """
    pm_dir = RAW_DIR / "predictive_maintenance"

    # Search for CSV file in the directory
    csv_files = list(pm_dir.glob("*.csv"))
    if not csv_files:
        logger.warning(
            "No CSV found in %s. "
            "Please download from: https://archive.ics.uci.edu/dataset/601",
            pm_dir,
        )
        return None

    # Take the first CSV file found
    file_path = csv_files[0]
    logger.info("Loading UCI AI4I from: %s", file_path.name)

    try:
        df = pd.read_csv(file_path)
        logger.info("Loaded UCI AI4I: %s", df.shape)

        # Standardise column names
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("[[]", "").str.replace("[]]", "")

        # Type checks
        expected_cols = {
            "udi", "product id", "type",
            "air temperature [k]", "process temperature [k]",
            "rotational speed [rpm]", "torque [nm]",
            "tool wear [min]", "machine failure",
        }
        # normalised names
        norm_expected = {
            c.lower().replace(" ", "_").replace("[", "").replace("]", "")
            for c in expected_cols
        }
        found = set(df.columns)
        overlap = norm_expected & found
        logger.info("Expected columns found: %d/%d", len(overlap), len(norm_expected))

        # Mark as synthetic
        df["data_type"] = "SYNTHETIC_industrial_maintenance"
        df["source_note"] = (
            "UCI AI4I 2020 — synthetic dataset. "
            "Do NOT interpret as real electronics failure probabilities."
        )

        data_quality_report(df, "UCI_AI4I_PredMaint")

        _ensure_processed_dir()
        out_path = PROCESSED_DIR / "uci_ai4i_predictive_maintenance.parquet"
        df.to_parquet(out_path, index=False)
        logger.info("Saved processed: %s", out_path.name)

        return df

    except Exception as exc:
        logger.error("Failed to load UCI AI4I dataset: %s", exc)
        return None


# ── Synthetic / curated dataset loaders ────────────────────────────────────

def load_synthetic(filename: str) -> Optional[pd.DataFrame]:
    """Load a file from the synthetic_demo directory."""
    path = SYNTHETIC_DIR / filename
    if not path.exists():
        logger.warning("Synthetic dataset not found: %s. Run synthetic_data.py first.", filename)
        return None
    df = pd.read_csv(path)
    logger.info("Loaded synthetic dataset %s: %s", filename, df.shape)
    return df


def load_global_ewaste_timeseries() -> Optional[pd.DataFrame]:
    return load_synthetic("global_ewaste_timeseries.csv")


def load_regional_ewaste() -> Optional[pd.DataFrame]:
    return load_synthetic("regional_ewaste.csv")


def load_country_ewaste_snapshot() -> Optional[pd.DataFrame]:
    return load_synthetic("country_ewaste_snapshot.csv")


def load_india_ewaste_timeseries() -> Optional[pd.DataFrame]:
    return load_synthetic("india_ewaste_timeseries.csv")


def load_ewaste_by_category() -> Optional[pd.DataFrame]:
    return load_synthetic("ewaste_by_category.csv")


def load_hazardous_materials() -> Optional[pd.DataFrame]:
    return load_synthetic("hazardous_materials.csv")


def load_battery_demo() -> Optional[pd.DataFrame]:
    return load_synthetic("battery_demo.csv")


def load_product_lifecycle_demo() -> Optional[pd.DataFrame]:
    return load_synthetic("product_lifecycle_demo.csv")


def load_ewaste_forecast_scenarios() -> Optional[pd.DataFrame]:
    return load_synthetic("ewaste_forecast_scenarios.csv")


# ── Feature engineering utilities ──────────────────────────────────────────

def add_lag_features(
    df: pd.DataFrame,
    value_col: str,
    n_lags: int = 3,
    sort_col: Optional[str] = None,
) -> pd.DataFrame:
    """Add lag features to a time-series DataFrame."""
    df = df.copy()
    if sort_col:
        df = df.sort_values(sort_col).reset_index(drop=True)
    for lag in range(1, n_lags + 1):
        df[f"{value_col}_lag{lag}"] = df[value_col].shift(lag)
    return df


def add_rolling_features(
    df: pd.DataFrame,
    value_col: str,
    windows: Tuple[int, ...] = (3, 5),
    sort_col: Optional[str] = None,
) -> pd.DataFrame:
    """Add rolling mean / std features."""
    df = df.copy()
    if sort_col:
        df = df.sort_values(sort_col).reset_index(drop=True)
    for w in windows:
        df[f"{value_col}_roll_mean_{w}"] = df[value_col].rolling(window=w, min_periods=1).mean()
        df[f"{value_col}_roll_std_{w}"] = df[value_col].rolling(window=w, min_periods=1).std()
    return df


def chronological_split(
    df: pd.DataFrame,
    time_col: str,
    test_frac: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chronological train / test split — avoids data leakage for time-series.
    The most recent (test_frac) fraction is held out as the test set.
    """
    df = df.sort_values(time_col).reset_index(drop=True)
    n_test = max(1, int(len(df) * test_frac))
    train = df.iloc[: len(df) - n_test]
    test = df.iloc[len(df) - n_test :]
    logger.info(
        "Chronological split: train=%d rows, test=%d rows | column=%s",
        len(train), len(test), time_col,
    )
    return train, test


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")

    print("Testing preprocessing module...")
    df_global = load_global_ewaste_timeseries()
    if df_global is not None:
        print(f"  Global e-waste timeseries: {df_global.shape}")
        train, test = chronological_split(df_global, "year", test_frac=0.2)
        print(f"  Train: {len(train)} | Test: {len(test)}")

    df_battery = load_battery_demo()
    if df_battery is not None:
        print(f"  Battery demo: {df_battery.shape}")

    df_pm = load_uci_ai4i()
    if df_pm is not None:
        print(f"  UCI AI4I: {df_pm.shape}")
    else:
        print("  UCI AI4I not available — run data acquisition first.")
