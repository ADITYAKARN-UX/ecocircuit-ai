"""
EcoCircuit AI — Data Ingestion Module
======================================
Downloads, verifies, and registers datasets from official sources.
Preserves raw data and logs acquisition status.

Author: EcoCircuit AI Team
"""

import os
import json
import logging
import zipfile
import hashlib
import requests
import yaml
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Dict, Any

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("data_ingestion")

# ── Constants ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"
REGISTRY_PATH = PROJECT_ROOT / "data" / "dataset_registry.json"

# ── Config loader ──────────────────────────────────────────────────────────

def load_config() -> Dict[str, Any]:
    """Load YAML configuration."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# ── Registry ───────────────────────────────────────────────────────────────

def load_registry() -> Dict[str, Any]:
    """Load the dataset registry; create empty registry if missing."""
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_registry(registry: Dict[str, Any]) -> None:
    """Persist the registry to disk."""
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, default=str)
    logger.info("Dataset registry saved → %s", REGISTRY_PATH)


def update_registry(
    registry: Dict[str, Any],
    dataset_id: str,
    entry: Dict[str, Any],
) -> None:
    """Update or create a registry entry."""
    registry[dataset_id] = entry
    save_registry(registry)

# ── File utilities ─────────────────────────────────────────────────────────

def file_md5(path: Path, chunk_size: int = 65536) -> str:
    """Return the MD5 hex digest of a file."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def file_size_mb(path: Path) -> float:
    """Return file size in megabytes."""
    return path.stat().st_size / (1024 * 1024)

# ── Downloader ─────────────────────────────────────────────────────────────

def download_file(
    url: str,
    dest_path: Path,
    timeout: int = 60,
    max_size_mb: float = 200,
    overwrite: bool = False,
) -> bool:
    """
    Download a file from *url* to *dest_path*.

    Returns True on success, False on failure.
    Never overwrites an existing file unless overwrite=True.
    """
    if dest_path.exists() and not overwrite:
        logger.info("File already exists, skipping download: %s", dest_path.name)
        return True

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading: %s", url)
    try:
        with requests.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()

            # Content-length check
            content_length = r.headers.get("content-length")
            if content_length:
                size_mb = int(content_length) / (1024 * 1024)
                if size_mb > max_size_mb:
                    logger.warning(
                        "File too large (%.1f MB > %.1f MB limit), skipping.",
                        size_mb, max_size_mb,
                    )
                    return False

            with open(dest_path, "wb") as f:
                downloaded = 0
                for chunk in r.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if downloaded / (1024 * 1024) > max_size_mb:
                            logger.warning("Download exceeded size limit. Removing partial file.")
                            dest_path.unlink(missing_ok=True)
                            return False

        logger.info(
            "Download complete: %s (%.2f MB)",
            dest_path.name,
            file_size_mb(dest_path),
        )
        return True

    except requests.exceptions.RequestException as exc:
        logger.error("Download failed for %s: %s", url, exc)
        dest_path.unlink(missing_ok=True)
        return False


def extract_zip(zip_path: Path, extract_to: Path) -> bool:
    """Extract a ZIP archive. Returns True on success."""
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            members = zf.namelist()
            logger.info("ZIP contains %d members: %s", len(members), members[:5])
            zf.extractall(extract_to)
        logger.info("Extracted to: %s", extract_to)
        return True
    except zipfile.BadZipFile as exc:
        logger.error("Bad ZIP file %s: %s", zip_path.name, exc)
        return False

# ── Individual dataset acquisition ─────────────────────────────────────────

def acquire_world_bank_what_a_waste(registry: Dict) -> None:
    """Download World Bank What a Waste 3.0 country and city datasets."""
    config = load_config()

    for key in ("world_bank_what_a_waste_country", "world_bank_what_a_waste_city"):
        ds_cfg = config["datasets"].get(key, {})
        url = ds_cfg.get("url", "")
        local_path = PROJECT_ROOT / ds_cfg.get("local_path", "")
        license_info = ds_cfg.get("license", "CC BY 4.0")

        entry = {
            "dataset_name": key,
            "publisher": "World Bank Group",
            "official_source": "https://datacatalog.worldbank.org/search/dataset/0039597",
            "local_path": str(local_path),
            "license": license_info,
            "note": "General solid waste management — NOT exclusively e-waste data",
            "acquisition_date": None,
            "status": "pending",
            "records": None,
            "columns": None,
            "file_size_mb": None,
            "md5": None,
        }

        success = download_file(url, local_path)
        if success and local_path.exists():
            entry.update({
                "acquisition_date": datetime.now(timezone.utc).isoformat(),
                "status": "available",
                "file_size_mb": round(file_size_mb(local_path), 2),
                "md5": file_md5(local_path),
            })
        else:
            entry["status"] = "download_failed"
            logger.warning("Could not download %s. Manual download may be required.", key)

        update_registry(registry, key, entry)


def acquire_uci_ai4i_predictive_maintenance(registry: Dict) -> None:
    """Download UCI AI4I 2020 Predictive Maintenance dataset (ZIP)."""
    config = load_config()
    ds_cfg = config["datasets"].get("uci_ai4i_predictive_maintenance", {})
    url = ds_cfg.get("url", "")
    zip_path = PROJECT_ROOT / ds_cfg.get("local_path", "")
    extract_dir = zip_path.parent

    entry = {
        "dataset_name": "UCI AI4I 2020 Predictive Maintenance",
        "publisher": "UCI Machine Learning Repository",
        "official_source": "https://archive.ics.uci.edu/dataset/601",
        "local_path": str(extract_dir),
        "license": "CC BY 4.0",
        "note": (
            "SYNTHETIC industrial maintenance data. "
            "Do NOT interpret results as real electronics failure probabilities."
        ),
        "acquisition_date": None,
        "status": "pending",
        "records": 10000,
        "columns": 14,
        "file_size_mb": None,
        "md5": None,
    }

    success = download_file(url, zip_path)
    if success and zip_path.exists():
        extracted = extract_zip(zip_path, extract_dir)
        entry.update({
            "acquisition_date": datetime.now(timezone.utc).isoformat(),
            "status": "available" if extracted else "extraction_failed",
            "file_size_mb": round(file_size_mb(zip_path), 2),
            "md5": file_md5(zip_path),
        })
    else:
        entry["status"] = "download_failed"
        logger.warning(
            "UCI AI4I download failed. Please download manually from: %s",
            "https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset",
        )

    update_registry(registry, "uci_ai4i_predictive_maintenance", entry)


def register_manual_datasets(registry: Dict) -> None:
    """
    Register datasets that require manual download or API keys,
    documenting their status and how to obtain them.
    """
    manual_entries = {
        "global_ewaste_monitor": {
            "dataset_name": "Global E-Waste Monitor 2024",
            "publisher": "UNITAR / ITU / Global E-waste Statistics Partnership",
            "official_source": "https://ewastemonitor.info/monitors/",
            "local_path": "data/raw/ewaste/global_ewaste_monitor/",
            "license": "See publication terms — contact scycle@unitar.org for bulk data",
            "note": (
                "No bulk CSV download is publicly available. "
                "Data used in this project comes from published report annexes "
                "and the ITU DataHub. A curated dataset is embedded in data/synthetic_demo/."
            ),
            "acquisition_date": None,
            "status": "manual_download_required",
            "manual_instructions": (
                "1. Visit https://ewastemonitor.info/monitors/ \n"
                "2. Download the PDF report and Annex 2 data tables.\n"
                "3. For raw data, email scycle@unitar.org or use the ITU DataHub.\n"
                "4. Place extracted CSV/Excel in data/raw/ewaste/global_ewaste_monitor/."
            ),
        },
        "india_ewaste_cpcb": {
            "dataset_name": "India Year-wise E-Waste Generated and Recycled (CPCB)",
            "publisher": "Central Pollution Control Board (CPCB) / data.gov.in",
            "official_source": "https://www.data.gov.in/resource/year-wise-details-volume-e-waste-generated-and-recycled-under-e-waste-management-rules",
            "local_path": "data/raw/ewaste/india_ewaste_cpcb/",
            "license": "National Data Sharing and Accessibility Policy (NDSAP), Government of India",
            "note": (
                "Requires free registration on data.gov.in to obtain an API key. "
                "Data covers national totals; state-level data has separate resource IDs."
            ),
            "acquisition_date": None,
            "status": "api_key_required",
            "manual_instructions": (
                "1. Register at https://data.gov.in and obtain a free API key.\n"
                "2. Use the API: "
                "https://api.data.gov.in/resource/<resource_id>?api-key=<YOUR_KEY>&format=csv\n"
                "3. Resource ID for e-waste generation: "
                "check https://www.data.gov.in/resource/year-wise-details-volume-e-waste-generated-and-recycled-under-e-waste-management-rules\n"
                "4. Place downloaded CSV in data/raw/ewaste/india_ewaste_cpcb/."
            ),
        },
        "nasa_battery_aging": {
            "dataset_name": "NASA Li-ion Battery Aging Datasets (PCoE)",
            "publisher": "NASA Ames Research Center — Prognostics Center of Excellence",
            "official_source": "https://data.nasa.gov/dataset/li-ion-battery-aging-datasets",
            "local_path": "data/raw/batteries/nasa_battery/",
            "license": "NASA Open Source — public domain",
            "note": (
                "Data is in MATLAB .mat format. "
                "Community-converted CSV versions are available on Kaggle. "
                "Do NOT assume laboratory results generalize to all consumer devices."
            ),
            "acquisition_date": None,
            "status": "manual_download_required",
            "manual_instructions": (
                "Option A (official): \n"
                "  1. Visit https://data.nasa.gov/dataset/li-ion-battery-aging-datasets\n"
                "  2. Download the battery data archive.\n"
                "  3. Extract .mat files to data/raw/batteries/nasa_battery/.\n"
                "  4. Run: python src/battery_analysis.py --convert-mat\n\n"
                "Option B (CSV — Kaggle community version): \n"
                "  Search Kaggle for 'NASA Battery Dataset' and download the CSV version.\n"
                "  Place CSVs in data/raw/batteries/nasa_battery/."
            ),
        },
        "oecd_environmental": {
            "dataset_name": "OECD Environmental Statistics",
            "publisher": "Organisation for Economic Co-operation and Development (OECD)",
            "official_source": "https://data-explorer.oecd.org/s/1nh",
            "local_path": "data/raw/pollution/oecd_environmental/",
            "license": "OECD Terms and Conditions",
            "note": "Use the OECD Data Explorer to select relevant environmental indicators and export CSV.",
            "acquisition_date": None,
            "status": "manual_download_required",
            "manual_instructions": (
                "1. Visit https://data-explorer.oecd.org\n"
                "2. Select the 'Environment' domain.\n"
                "3. Filter by relevant indicators (e.g., municipal waste, hazardous waste).\n"
                "4. Export the filtered dataset as CSV.\n"
                "5. Place in data/raw/pollution/oecd_environmental/."
            ),
        },
        "epa_environmental": {
            "dataset_name": "US EPA Environmental Data Downloads",
            "publisher": "US Environmental Protection Agency (EPA)",
            "official_source": "https://www.epa.gov/enviro/data-downloads",
            "local_path": "data/raw/pollution/epa_environmental/",
            "license": "Public domain (US Government works)",
            "note": (
                "Multiple sub-datasets available. Identify specific dataset before integrating. "
                "Do NOT automatically attribute measurements to e-waste processing."
            ),
            "acquisition_date": None,
            "status": "manual_download_required",
        },
        "epa_water_quality": {
            "dataset_name": "US EPA Water Quality Data",
            "publisher": "US Environmental Protection Agency (EPA)",
            "official_source": "https://www.waterqualitydata.us/",
            "local_path": "data/raw/water_quality/epa_water_quality/",
            "license": "Public domain (US Government works)",
            "note": (
                "Very large dataset. Use the Water Quality Portal query interface "
                "to filter by location, date, and analyte before downloading. "
                "Do NOT attribute water quality observations to e-waste without evidence."
            ),
            "acquisition_date": None,
            "status": "manual_download_required",
            "manual_instructions": (
                "1. Visit https://www.waterqualitydata.us/\n"
                "2. Filter by state, date range, and relevant analytes "
                "(e.g., Lead, Cadmium, Mercury, Arsenic).\n"
                "3. Download results as CSV.\n"
                "4. Place in data/raw/water_quality/epa_water_quality/."
            ),
        },
    }

    for key, entry in manual_entries.items():
        if key not in registry:
            registry[key] = entry
            logger.info("Registered manual dataset: %s [%s]", entry["dataset_name"], entry["status"])

    save_registry(registry)


# ── Main orchestration ─────────────────────────────────────────────────────

def run_acquisition(skip_downloads: bool = False) -> Dict:
    """
    Run the full data acquisition pipeline.

    Parameters
    ----------
    skip_downloads : bool
        If True, skip HTTP downloads (useful for testing / offline environments).

    Returns
    -------
    dict
        The updated dataset registry.
    """
    logger.info("=" * 60)
    logger.info("EcoCircuit AI — Data Acquisition Pipeline")
    logger.info("=" * 60)

    registry = load_registry()

    if not skip_downloads:
        logger.info("Step 1/3: World Bank What a Waste 3.0")
        acquire_world_bank_what_a_waste(registry)

        logger.info("Step 2/3: UCI AI4I Predictive Maintenance")
        acquire_uci_ai4i_predictive_maintenance(registry)
    else:
        logger.info("Downloads skipped (skip_downloads=True)")

    logger.info("Step 3/3: Registering manual-download datasets")
    register_manual_datasets(registry)

    logger.info("=" * 60)
    available = [k for k, v in registry.items() if v.get("status") == "available"]
    pending = [k for k, v in registry.items() if v.get("status") != "available"]
    logger.info("Available datasets  : %d — %s", len(available), available)
    logger.info("Pending / manual    : %d — %s", len(pending), pending)
    logger.info("=" * 60)

    return registry


if __name__ == "__main__":
    registry = run_acquisition()
    print("\nDataset Registry Summary:")
    print("-" * 50)
    for ds_id, entry in registry.items():
        status_icon = "✓" if entry.get("status") == "available" else "⚠"
        print(f"  {status_icon} {entry.get('dataset_name', ds_id)} [{entry.get('status')}]")
