"""
Generate a complete, reproducible Jupyter Notebook for academic submission & demonstration.
"""
import json
from pathlib import Path

def create_notebook():
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# EcoCircuit AI: End-to-End Predictive Analytics & Circular Economy Platform\n",
                    "### B.Tech Computer Science Final-Year Project Demonstration\n",
                    "---\n",
                    "**Authors:** EcoCircuit AI Development Team  \n",
                    "**Date:** Academic Year 2026  \n",
                    "\n",
                    "## 1. Project Overview\n",
                    "EcoCircuit AI bridges the gap between environmental monitoring and industrial manufacturing by:\n",
                    "1. Forecasting national and global e-waste generation trajectories.\n",
                    "2. Analyzing lithium-ion battery degradation (State of Health - SOH).\n",
                    "3. Providing real-time industrial predictive maintenance (UCI AI4I 2020).\n",
                    "4. Optimizing electronics manufacturing schedules to prevent overproduction and double-counting.\n",
                    "5. Delivering an intelligent rule-based End-of-Life recycling and repair advisor."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 1. Environment Setup & Imports\n",
                    "import sys\n",
                    "from pathlib import Path\n",
                    "import numpy as np\n",
                    "import pandas as pd\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "\n",
                    "# Set style\n",
                    "plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')\n",
                    "PROJECT_ROOT = Path('.').resolve().parent\n",
                    "sys.path.insert(0, str(PROJECT_ROOT))\n",
                    "print(f\"Project root: {PROJECT_ROOT}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Data Loading & Exploratory Data Analysis (EDA)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from src.synthetic_data import generate_all_demo_datasets\n",
                    "data = generate_all_demo_datasets()\n",
                    "print(f\"Loaded {len(data)} datasets successfully!\")\n",
                    "df_global = data['global_ewaste_timeseries']\n",
                    "df_global.head()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. Global E-Waste Trend & Predictive Modeling\n",
                    "We apply chronological train/test splits (avoiding data leakage) and compare Exponential Smoothing, ARIMA/SARIMAX, and Linear Trend Baselines."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from src.ewaste_forecasting import run_ewaste_forecast\n",
                    "forecast_results = run_ewaste_forecast(df_global, 'year', 'ewaste_generated_mt', horizon=5)\n",
                    "print(\"Model Evaluation Metrics:\")\n",
                    "for m_name, m_data in forecast_results['models'].items():\n",
                    "    print(f\"  {m_name:<25}: RMSE={m_data['rmse']:.3f}, MAE={m_data['mae']:.3f}, MAPE={m_data['mape']:.2f}%\")\n",
                    "\n",
                    "print(\"\\n5-Year Forecast (Best Model):\")\n",
                    "print(pd.DataFrame(forecast_results['forecast']))"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 4. NASA Li-ion Battery State of Health (SOH) Estimation\n",
                    "Feature engineering from cycle-life charging profiles and model comparison (Random Forest, Gradient Boosting, Linear Regression)."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from src.battery_analysis import load_battery_demo, prepare_battery_features, train_battery_models\n",
                    "df_battery = load_battery_demo()\n",
                    "df_battery_features = prepare_battery_features(df_battery)\n",
                    "battery_results = train_battery_models(df_battery_features)\n",
                    "\n",
                    "print(\"Battery SOH Prediction Metrics:\")\n",
                    "for model_name, metrics in battery_results.items():\n",
                    "    if not model_name.startswith('_'):\n",
                    "        print(f\"  {model_name:<25}: R²={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 5. Industrial Predictive Maintenance (UCI AI4I 2020)\n",
                    "Addressing heavy class imbalance with balanced class weighting to prevent catastrophic tool failures."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from src.failure_prediction import load_ai4i_data, prepare_ai4i_features, train_failure_models\n",
                    "df_ai4i = load_ai4i_data()\n",
                    "if df_ai4i is not None:\n",
                    "    df_ai4i_feat, target = prepare_ai4i_features(df_ai4i)\n",
                    "    failure_metrics = train_failure_models(df_ai4i_feat, target)\n",
                    "    for m_name in ['Logistic Regression', 'Random Forest', 'Gradient Boosting']:\n",
                    "        res = failure_metrics[m_name]\n",
                    "        print(f\"  {m_name:<25}: ROC-AUC={res['roc_auc']:.4f}, F1-Score={res['f1_macro']:.4f}, Recall={res['recall']:.4f}\")\n",
                    "else:\n",
                    "    print(\"AI4I dataset not found.\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 6. Smart Production Optimization & Double-Counting Prevention\n",
                    "Deterministic production gap calculation integrating usable inventory, component yields, and reverse logistics."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from src.production_optimizer import calculate_production\n",
                    "prod_plan = calculate_production(\n",
                    "    forecast_demand=12000,\n",
                    "    usable_inventory=3500,\n",
                    "    safety_stock=1000,\n",
                    "    repairable_inventory=1500,\n",
                    "    repair_success_rate=0.75,\n",
                    "    expected_usable_returns=800,\n",
                    "    recoverable_components=2000,\n",
                    "    component_recovery_yield=0.60\n",
                    ")\n",
                    "\n",
                    "print(f\"Recommended New Production: {prod_plan['outputs']['recommended_new_production']:,.0f} units\")\n",
                    "print(f\"Avoided Overproduction:    {prod_plan['derived']['new_production_avoidable_units']:,.0f} units\")\n",
                    "print(f\"Interpretation:            {prod_plan['interpretation']}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 7. Circular Economy Recommender (Rule-Based Expert System)\n",
                    "Adheres to EU WEEE Directive and right-to-repair standards."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from src.recycling_recommender import recommend\n",
                    "rec = recommend(\n",
                    "    product_category='Smartphone',\n",
                    "    age_years=3.5,\n",
                    "    condition='Functional with minor issues',\n",
                    "    is_functional=True,\n",
                    "    repair_cost_estimate=45.0,\n",
                    "    replacement_value=300.0\n",
                    ")\n",
                    "print(f\"Action:      {rec['action']}\")\n",
                    "print(f\"Confidence:  {rec['confidence']}\")\n",
                    "print(f\"Reasoning:   {rec['reasoning']}\")"
                ]
            }
        ],
        "metadata": {
            "language_info": {"name": "python", "version": "3.13"},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    out_file = Path("notebooks") / "01_ecocircuit_end_to_end_analytics.ipynb"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Exported clean notebook to {out_file}")

if __name__ == "__main__":
    create_notebook()
