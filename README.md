# EcoCircuit AI

> **AI-Powered Electronic Waste, Environmental Impact & Smart Production Intelligence Platform**
>
> *Predict Pollution. Optimize Production. Eliminate Avoidable E-Waste.*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📋 Project Overview

EcoCircuit AI is a full-stack predictive analytics platform for understanding the lifecycle of electronic products, estimating e-waste generation, analyzing environmental risks, forecasting demand, and optimizing production. It is designed as a B.Tech Computer Science final-year project with genuine data analysis, trained ML models, and transparent assumptions throughout.

---

## 🎯 Problem Statement

Over 62 million metric tonnes of electronic waste were generated globally in 2022, growing at approximately 2.6 Mt per year. Only 22.3% is formally collected and recycled. The rest — containing hazardous substances like lead, mercury, cadmium, and brominated flame retardants — is either dumped, improperly processed, or illegally exported. EcoCircuit AI addresses this challenge with data-driven analytics, forecasting, and circular economy tools.

---

## 🚀 Key Features

| Module | Description |
|--------|-------------|
| **A — E-Waste Forecasting** | Historical trend analysis, multi-model time-series forecasting (Naive, MA, Linear, Holt-Winters, ARIMA) |
| **B — Environmental Intelligence** | Documented hazardous material profiles, processing risk analysis, environmental data status |
| **C — Battery & Product Health** | Battery degradation modeling (Linear, RF, GB), SOH estimation, RUL calculation |
| **D — Smart Production Optimizer** | Inventory-aware production calculation, double-counting prevention, sensitivity analysis |
| **E — Recycling Recommender** | Rule-based end-of-life advisor (repair / refurbish / reuse / recycle / dispose) |
| **F — What-If Simulator** | Multi-scenario sustainability comparison with documented equations |
| **G — Product Lifecycle Records** | Lifecycle tracking with demo records |
| **H — Dataset Explorer** | Registry of all data sources with acquisition status and documentation |

---

## 🛠 Technology Stack

- **Python 3.10+**
- **Pandas, NumPy** — data processing
- **Scikit-learn** — ML models (Linear, RF, GB, LR)
- **Statsmodels** — Holt-Winters, ARIMA forecasting
- **Plotly** — interactive visualizations
- **Streamlit** — web dashboard
- **Joblib** — model serialization
- **PyArrow** — Parquet storage
- **PyYAML** — configuration
- **Pytest** — automated testing

---

## 📁 Project Structure

```
ecocircuit-ai/
├── app.py                      ← Main Streamlit dashboard (8 pages)
├── requirements.txt
├── pytest.ini
├── README.md
│
├── config/
│   └── settings.yaml           ← App config and download URLs
│
├── data/
│   ├── raw/                    ← Downloaded datasets (preserved, never overwritten)
│   │   ├── ewaste/
│   │   ├── batteries/
│   │   ├── predictive_maintenance/
│   │   ├── pollution/
│   │   ├── water_quality/
│   │   └── waste_management/
│   ├── processed/              ← Cleaned/transformed outputs
│   ├── synthetic_demo/         ← Curated datasets from published figures
│   └── dataset_registry.json  ← Machine-readable dataset catalog
│
├── src/
│   ├── data_ingestion.py       ← Download pipeline and registry
│   ├── synthetic_data.py       ← Curated datasets from published reports
│   ├── preprocessing.py        ← Loading, validation, feature engineering
│   ├── ewaste_forecasting.py   ← Module A: E-waste time-series forecasting
│   ├── battery_analysis.py     ← Module C: Battery degradation + RUL
│   ├── failure_prediction.py   ← Module C ext: Predictive maintenance (UCI AI4I)
│   ├── production_optimizer.py ← Module D: Smart production calculation
│   ├── recycling_recommender.py ← Module E: End-of-life decision engine
│   └── what_if_simulator.py   ← Module F: Sustainability scenario simulator
│
├── models/                     ← Serialized trained models (.pkl)
├── tests/
│   └── test_ecocircuit.py      ← 30+ automated tests
└── docs/
    ├── dataset_catalog.md
    ├── assumptions.md
    └── limitations.md
```

---

## 📦 Dataset Sources

| Dataset | Publisher | Status | Type |
|---------|-----------|--------|------|
| Global E-Waste Monitor 2024 | UNITAR/ITU | Curated from report | E-waste (global) |
| India E-Waste (CPCB) | CPCB / PIB India | Curated from records | E-waste (India) |
| World Bank What a Waste 3.0 | World Bank | Downloadable (XLSX) | General solid waste |
| UCI AI4I 2020 | UCI MLR | Downloadable (ZIP) | SYNTHETIC maintenance |
| NASA Li-ion Battery Aging | NASA PCoE | Manual download | Lab battery aging |
| OECD Environmental Stats | OECD | Manual download | Environmental |
| EPA Environmental / Water | US EPA | Manual download | Environmental |

> ⚠️ **World Bank data is GENERAL solid waste — NOT e-waste.** Never label it as e-waste.
> ⚠️ **UCI AI4I data is SYNTHETIC — NOT real electronics failure data.**

---

## ⚙️ Installation

### 1. Clone or download the project

```bash
# Navigate to project directory
cd "ecocircuit-ai"
```

### 2. (Recommended) Create a virtual environment

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Generate curated datasets

```bash
python src/synthetic_data.py
```

### 5. (Optional) Download programmatically available datasets

```bash
python src/data_ingestion.py
```

This will download:
- World Bank What a Waste 3.0 country and city datasets (XLSX)
- UCI AI4I Predictive Maintenance dataset (ZIP → CSV)

### 6. Manual dataset downloads

See `docs/dataset_catalog.md` for step-by-step instructions to obtain:
- NASA Li-ion Battery Aging Dataset (MATLAB → CSV)
- India E-Waste via data.gov.in API (API key required)
- OECD Environmental Statistics
- US EPA Environmental / Water Quality Data

---

## 🚀 Running the Application

```bash
streamlit run app.py
```

The dashboard opens at **http://localhost:8501**

---

## 🧪 Running Tests

```bash
python -m pytest tests/ -v
```

Expected: 30+ tests across production optimizer, recycling recommender, what-if simulator, data loading, forecasting, and battery analysis modules.

---

## 📊 Model Descriptions

### E-Waste Forecasting (Module A)
- **Approach**: Chronological train/test split on annual global e-waste timeseries
- **Models**: Naive, Moving Average (3-year), Linear Trend (OLS), Holt-Winters (if statsmodels), ARIMA(1,1,0) (if sufficient data)
- **Selection**: Best model chosen by lowest RMSE on held-out test set
- **Limitation**: Only 14 annual observations available — high uncertainty in forecasts

### Battery Degradation (Module C)
- **Approach**: Per-battery train/test split (no battery appears in both sets)
- **Features**: cycle_number, log_cycle, cycle_norm, capacity_lag1, capacity_lag3, capacity_diff1, rolling_mean_5
- **Models**: Linear Regression (baseline), Random Forest, Gradient Boosting
- **Target**: capacity_ah (capacity in ampere-hours)
- **RUL**: Estimated via log-linear degradation fit; binary search for EOL threshold crossing

### Predictive Maintenance (Module C ext.)
- **Dataset**: UCI AI4I 2020 SYNTHETIC dataset — NOT real electronics failure data
- **Approach**: Stratified 80/20 random split (appropriate for cross-sectional data)
- **Models**: Logistic Regression (baseline), Random Forest, Gradient Boosting
- **Target**: Binary machine failure indicator
- **Class imbalance**: Handled via class_weight='balanced'

### Production Optimizer (Module D)
```
Recommended New Production = MAX(0,
    Forecast Demand
    + Required Safety Stock
    - Usable Existing Inventory
    - (Repairable Inventory × Repair Success Rate)
    - Expected Usable Returns
    - (Recoverable Components × Recovery Yield)
)
```
Double-counting is prevented by explicit overlap flags.

---

## ⚠️ Limitations

1. **E-waste forecasting**: Only 14 annual global data points — forecasts have high uncertainty
2. **Battery data**: Synthetic demo data used by default — place NASA CSV files in `data/raw/batteries/` for real analysis
3. **Environmental data**: Air/water/soil measurements require manual download and cannot be attributed to e-waste without specific evidence
4. **India state-level data**: Only national totals available without a data.gov.in API key
5. **UCI AI4I**: Synthetic industrial data — not applicable to real consumer electronics
6. **GEM bulk data**: No public CSV download; values from published report annexes

---

## 🔮 Future Improvements

- [ ] Integration with data.gov.in API for India state-level data
- [ ] NASA battery MATLAB parser (no external toolbox required)
- [ ] OECD API integration for automated environmental data refresh
- [ ] LCA integration for verified per-unit environmental impact factors
- [ ] EPR compliance tracking module (India CPCB EPR portal)
- [ ] Real-time e-waste price/demand data from recycling markets
- [ ] Geographic mapping with valid coordinates (e-waste hotspot analysis)

---

## 👨‍💻 Author

EcoCircuit AI — B.Tech Computer Science Final Year Project

---

## 📄 License

MIT License — see LICENSE file for details.
