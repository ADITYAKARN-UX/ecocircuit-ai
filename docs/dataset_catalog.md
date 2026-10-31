# EcoCircuit AI — Dataset Catalog

> All sources verified as of October 2026.

---

## DS-01: Global E-Waste Monitor 2024

| Field | Value |
|-------|-------|
| **Title** | Global E-Waste Monitor 2024 |
| **Publisher** | UNITAR / ITU / Global E-waste Statistics Partnership (GESP) |
| **Official Source** | https://ewastemonitor.info/monitors/ |
| **Additional** | https://globalewaste.org/ |
| **Local Path** | `data/synthetic_demo/global_ewaste_timeseries.csv` |
| **Acquisition** | Values extracted from published report annexes |
| **License** | Publication terms; contact scycle@unitar.org for bulk data |
| **Coverage** | Global, 2010–2023 |
| **Unit** | Million metric tonnes (Mt) |
| **Status** | Curated from published report |
| **Limitation** | No bulk CSV download available. Raw SDG indicator data via SCYCLE/ITU DataHub on request. |
| **Use in project** | E-waste trend analysis, forecasting, regional comparison |

---

## DS-02: India E-Waste (CPCB / PIB)

| Field | Value |
|-------|-------|
| **Title** | Year-wise E-Waste Generated and Recycled — India |
| **Publisher** | Central Pollution Control Board (CPCB), India |
| **Official Source** | https://www.data.gov.in/resource/year-wise-details-volume-e-waste-generated-and-recycled-under-e-waste-management-rules |
| **Additional** | PIB press releases; Lok Sabha / Rajya Sabha Q&A |
| **Local Path** | `data/synthetic_demo/india_ewaste_timeseries.csv` |
| **Acquisition** | Manually compiled from PIB + Parliamentary records |
| **License** | NDSAP (National Data Sharing and Accessibility Policy), Government of India |
| **Coverage** | India national totals, FY 2017-18 to FY 2024-25 |
| **Unit** | Metric tonnes (MT) |
| **Status** | Curated from published records |
| **Limitation** | State-level data requires data.gov.in API key + separate resource IDs |
| **Use in project** | India-focused e-waste analysis, collection rate trends |

---

## DS-03: World Bank What a Waste 3.0

| Field | Value |
|-------|-------|
| **Title** | What a Waste Global Database 3.0 |
| **Publisher** | World Bank Group |
| **Official Source** | https://datacatalog.worldbank.org/search/dataset/0039597 |
| **Download (Country)** | https://datacatalogfiles.worldbank.org/ddh-published/0039597/DR0095901/What_a_Waste_3.0_COUNTRY_Dataset_&_Codebook.xlsx |
| **Download (City)** | https://datacatalogfiles.worldbank.org/ddh-published/0039597/DR0095900/What_a_Waste_3.0_CITY_Dataset_&_Codebook.xlsx |
| **Local Path** | `data/raw/waste_management/worldbank_what_a_waste_country.xlsx` |
| **License** | Creative Commons Attribution 4.0 (CC BY 4.0) |
| **Coverage** | 217 countries, 262 cities |
| **Period** | 2010–2050 (includes projections) |
| **Status** | Programmatically downloadable |
| **⚠️ CRITICAL WARNING** | This is **general solid waste** data — NOT e-waste. Never label records as e-waste. |
| **Use in project** | General waste management context, country-level comparisons |

---

## DS-04: UCI AI4I 2020 Predictive Maintenance

| Field | Value |
|-------|-------|
| **Title** | AI4I 2020 Predictive Maintenance Dataset |
| **Publisher** | UCI Machine Learning Repository |
| **Official Source** | https://archive.ics.uci.edu/dataset/601 |
| **Download URL** | https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip |
| **Local Path** | `data/raw/predictive_maintenance/` |
| **License** | Creative Commons Attribution 4.0 (CC BY 4.0) |
| **Records** | 10,000 |
| **Features** | 14 (air temp, process temp, RPM, torque, tool wear, failure types) |
| **Status** | Programmatically downloadable |
| **⚠️ CRITICAL WARNING** | This is **SYNTHETIC** industrial simulation data. Results do NOT represent real electronics failure rates. Use only to demonstrate ML workflow. |

---

## DS-05: NASA Li-ion Battery Aging (PCoE)

| Field | Value |
|-------|-------|
| **Title** | Li-ion Battery Aging Datasets |
| **Publisher** | NASA Ames Research Center — Prognostics Center of Excellence |
| **Official Source** | https://data.nasa.gov/dataset/li-ion-battery-aging-datasets |
| **Format** | MATLAB .mat files |
| **Local Path** | `data/raw/batteries/nasa_battery/` |
| **License** | NASA Open Data (public domain) |
| **Acquisition** | Manual download required |
| **Status** | Manual download required |
| **Limitation** | Laboratory data only. Conditions: 24°C, 4V charge, 2.7V discharge. Do not generalize to field conditions. |
| **Use in project** | Battery degradation analysis, SOH estimation |

---

## DS-06: OECD Environmental Statistics

| Field | Value |
|-------|-------|
| **Title** | OECD Environmental Data Explorer |
| **Publisher** | Organisation for Economic Co-operation and Development |
| **Official Source** | https://data-explorer.oecd.org/s/1nh |
| **Local Path** | `data/raw/pollution/oecd_environmental/` |
| **License** | OECD Terms and Conditions |
| **Status** | Manual download required via OECD Data Explorer |

---

## DS-07: US EPA Environmental Data

| Field | Value |
|-------|-------|
| **Title** | EPA Envirofacts Data Downloads |
| **Publisher** | US Environmental Protection Agency |
| **Official Source** | https://www.epa.gov/enviro/data-downloads |
| **Local Path** | `data/raw/pollution/epa_environmental/` |
| **License** | Public domain (US Government) |
| **Status** | Manual download required |

---

## DS-08: US EPA Water Quality Data

| Field | Value |
|-------|-------|
| **Title** | Water Quality Portal / EPA WQP |
| **Publisher** | US Environmental Protection Agency |
| **Official Source** | https://www.waterqualitydata.us/ |
| **Local Path** | `data/raw/water_quality/epa_water_quality/` |
| **License** | Public domain (US Government) |
| **Status** | Manual download required (query-based) |
| **⚠️ WARNING** | Do NOT attribute water quality measurements to e-waste without specific supporting evidence. |

---

## Datasets NOT Found / Not Available

| Dataset | Reason |
|---------|--------|
| India state-wise e-waste breakdown | Requires data.gov.in API key + resource ID discovery |
| CPCB EPR portal structured data | Portal is regulatory — no bulk data download found |
| Global product lifespan database | No global public database identified |
| E-waste material composition database | No standardised global dataset found |
| Soil contamination from e-waste | No global standardised public dataset found |
