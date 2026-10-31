"""
EcoCircuit AI — Predictive Maintenance Module (Module C extension)
==================================================================
Demonstrates predictive maintenance classification using UCI AI4I 2020 dataset.
IMPORTANT DISCLAIMERS:
  - AI4I 2020 is a SYNTHETIC industrial dataset.
  - Results do NOT represent real electronics failure rates.
  - This module demonstrates the ML workflow only.
  - All outputs are labelled accordingly.

Models:
  1. Logistic Regression (baseline)
  2. Random Forest Classifier
  3. Gradient Boosting Classifier

Target: machine failure (binary: 0 = no failure, 1 = failure)
Note: Class imbalance is addressed with class_weight='balanced'.

Author: EcoCircuit AI Team
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

logger = logging.getLogger("failure_prediction")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


# ── Data loading ────────────────────────────────────────────────────────────

def load_ai4i_data() -> Optional[pd.DataFrame]:
    """Load UCI AI4I dataset from raw or processed directory."""
    processed_path = PROJECT_ROOT / "data" / "processed" / "uci_ai4i_predictive_maintenance.parquet"
    raw_dir = PROJECT_ROOT / "data" / "raw" / "predictive_maintenance"

    if processed_path.exists():
        try:
            df = pd.read_parquet(processed_path)
            logger.info("Loaded AI4I from processed: %s", df.shape)
            return df
        except Exception:
            pass

    csv_files = list(raw_dir.glob("*.csv"))
    if csv_files:
        try:
            df = pd.read_csv(csv_files[0])
            df.columns = (
                df.columns.str.strip()
                .str.lower()
                .str.replace(r"[\[\]\s]", "_", regex=True)
                .str.replace("__", "_")
            )
            logger.info("Loaded AI4I from raw CSV: %s", df.shape)
            return df
        except Exception as exc:
            logger.error("Failed to load AI4I CSV: %s", exc)

    logger.warning(
        "AI4I dataset not found. "
        "Please run data_ingestion.py or download from: "
        "https://archive.ics.uci.edu/dataset/601"
    )
    return None


# ── Feature preparation ─────────────────────────────────────────────────────

NUMERIC_PATTERNS = [
    "air_temperature",
    "process_temperature",
    "rotational_speed",
    "torque",
    "tool_wear",
]

TARGET_COL = "machine failure"
TARGET_COL_ALT = "machine_failure"
LEAKAGE_COLS = {"twf", "hdf", "pwf", "osf", "rnf", "udi", "product_id"}


def prepare_ai4i_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """Identify and prepare features from the AI4I dataset."""
    df = df.copy()
    cols = list(df.columns)
    logger.info("AI4I columns: %s", cols)

    # Determine target column
    target = TARGET_COL if TARGET_COL in cols else TARGET_COL_ALT
    if target not in cols:
        failure_cols = [c for c in cols if "failure" in c.lower() and "type" not in c.lower()]
        target = failure_cols[0] if failure_cols else None

    if target is None:
        raise ValueError("Cannot identify target column in AI4I dataset.")

    # Match sensor features using keyword patterns
    features = []
    for pattern in NUMERIC_PATTERNS:
        match = [c for c in cols if pattern in c.lower()]
        if match:
            features.append(match[0])

    if not features:
        logger.warning("Could not identify specific sensor features. Using numeric columns excluding target/leakage.")
        features = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c != target and c.lower() not in LEAKAGE_COLS
        ]

    # Encode product type if present
    type_col = next((c for c in cols if "type" in c.lower() and "failure" not in c.lower() and "data_type" not in c.lower()), None)
    if type_col and f"{type_col}_encoded" not in df.columns:
        le = LabelEncoder()
        df[f"{type_col}_encoded"] = le.fit_transform(df[type_col].astype(str))
        features.append(f"{type_col}_encoded")

    # Deduplicate and ensure target is not in features
    features = [c for c in dict.fromkeys(features) if c != target and c.lower() not in LEAKAGE_COLS]

    logger.info("AI4I features: %s | target: %s", features, target)
    prepared_df = df[features + [target]].dropna()
    return prepared_df, target


# ── Model training ──────────────────────────────────────────────────────────

def train_failure_models(df: pd.DataFrame, target: str) -> Dict:
    """Train failure prediction models on AI4I data."""
    feature_cols = [c for c in df.columns if c != target]
    X = df[feature_cols].values
    y = np.asarray(df[target]).ravel().astype(int)

    logger.info(
        "Class distribution: class 0=%d (%.1f%%), class 1=%d (%.1f%%)",
        (y == 0).sum(), 100 * (y == 0).mean(),
        (y == 1).sum(), 100 * (y == 1).mean(),
    )

    # Standard random split (NOT time-series — this is cross-sectional synthetic data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    results = {}

    # 1. Logistic Regression (baseline)
    lr = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(class_weight="balanced", max_iter=500))])
    lr.fit(X_train, y_train)
    results["Logistic Regression"] = _evaluate_classifier(lr, X_test, y_test, "Logistic Regression")
    results["Logistic Regression"]["model"] = lr

    # 2. Random Forest
    rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    results["Random Forest"] = _evaluate_classifier(rf, X_test, y_test, "Random Forest")
    results["Random Forest"]["model"] = rf
    results["Random Forest"]["feature_importance"] = dict(zip(feature_cols, rf.feature_importances_))

    # 3. Gradient Boosting
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
    gb.fit(X_train, y_train)
    results["Gradient Boosting"] = _evaluate_classifier(gb, X_test, y_test, "Gradient Boosting")
    results["Gradient Boosting"]["model"] = gb
    results["Gradient Boosting"]["feature_importance"] = dict(zip(feature_cols, gb.feature_importances_))

    best_name = max(results, key=lambda k: results[k].get("f1", 0))
    logger.info(
        "Best failure model: %s (F1=%.3f, ROC-AUC=%.3f)",
        best_name,
        results[best_name].get("f1", 0),
        results[best_name].get("roc_auc", 0),
    )

    if JOBLIB_AVAILABLE:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        for name, obj in results.items():
            safe_name = name.lower().replace(" ", "_")
            path = MODELS_DIR / f"failure_{safe_name}.pkl"
            joblib.dump(obj["model"], path)

    results["_feature_cols"] = feature_cols
    results["_best_model"] = best_name
    results["_X_test"] = X_test
    results["_y_test"] = y_test

    return results


def _evaluate_classifier(model, X_test, y_test, name: str) -> Dict:
    """Evaluate a classifier on test data."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None

    logger.info(
        "[%s] Precision=%.3f Recall=%.3f F1=%.3f AUC=%s",
        name, precision, recall, f1,
        f"{roc_auc:.3f}" if roc_auc else "N/A",
    )

    return {
        "model_name": name,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc, 4) if roc_auc else None,
        "confusion_matrix": cm.tolist(),
        "n_test": len(y_test),
        "disclaimer": (
            "SYNTHETIC DATA — These results come from the UCI AI4I 2020 synthetic dataset. "
            "They do NOT represent real electronics failure rates."
        ),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")

    df = load_ai4i_data()
    if df is None:
        print("AI4I dataset not available. Download from: https://archive.ics.uci.edu/dataset/601")
    else:
        df_features, target_col = prepare_ai4i_features(df)
        print(f"Prepared dataset: {df_features.shape} | Target: {target_col}")
        results = train_failure_models(df_features, target_col)
        best = results["_best_model"]
        print(f"\nBest model: {best}")
        for name in ["Logistic Regression", "Random Forest", "Gradient Boosting"]:
            m = results[name]
            print(f"  {name:25s} F1={m['f1']:.3f} AUC={m.get('roc_auc','N/A')}")
        print(f"\n  ⚠ {results[best]['disclaimer']}")
