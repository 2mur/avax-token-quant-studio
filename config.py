"""
Configuration for Dune ML Pipeline & SOP Visualization Suite
Token: NOCHILL-WAVAX (Avalanche C-Chain)
"""
import os
from pathlib import Path

# Base Parameters
DUNE_ID = 4834833
RESOLUTION = '1h'
DASHBOARD_RESOLUTION = '12h'
TARGET_BUY_THRESHOLD = 0.02
TARGET_SELL_THRESHOLD = -0.02
N_PCA_COMP = 4
SMA_VELOCITY_WINDOW = 24

# Directory Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dune-data"
DASHBOARDS_DIR = BASE_DIR / "dune-databoards"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
DASHBOARDS_DIR.mkdir(parents=True, exist_ok=True)

# Data File Paths
PRICES_JSON = DASHBOARDS_DIR / f"prices_{DUNE_ID}.json"
TRADES_CSV = DATA_DIR / f"dune_query_{DUNE_ID}_latest.csv"
PNL_FEATURES_CSV = DATA_DIR / f"pnl_features_{DUNE_ID}.csv"
TRAINING_DATA_CSV = DATA_DIR / f"training_data_{DUNE_ID}.csv"

# Dashboard File Paths
DASHBOARD_FEATURES_JSON = DASHBOARDS_DIR / f"dashboard_features_{DUNE_ID}.json"
EDA_FEATURES_JSON = DASHBOARDS_DIR / "eda_features.json"

# Hydrogen Project Sourcing Paths
HYDROGEN_DIR = Path(r"c:\Users\tima-\OneDrive\Desktop\hydrogen-project-november-2026")
HYDROGEN_DATA_DIR = HYDROGEN_DIR / "data"
HYDROGEN_NOCHILL_CSV = HYDROGEN_DATA_DIR / "NOCHILL_0xacfb89.csv"
HYDROGEN_STATE_JSON = HYDROGEN_DATA_DIR / "state.json"

DEFAULT_NOCHILL_ADDRESS = "0xacfb898cff266e53278cc0124fc2c7c94c8cb9a5"

# Models to simulate
MODELS = ["xgboost", "lgbm", "esn", "lstm", "transformer", "seasonal"]


def get_token_dirs(address: str = None) -> tuple[Path, Path]:
    """Return (data_dir, dashboards_dir) for a given token address."""
    if not address or address.strip().lower() == DEFAULT_NOCHILL_ADDRESS:
        d_dir = BASE_DIR / "dune-data"
        b_dir = BASE_DIR / "dune-databoards"
    else:
        addr_clean = address.strip().lower()
        d_dir = BASE_DIR / "dune-data" / addr_clean
        b_dir = BASE_DIR / "dune-databoards" / addr_clean
    d_dir.mkdir(parents=True, exist_ok=True)
    b_dir.mkdir(parents=True, exist_ok=True)
    return d_dir, b_dir


def set_active_token_paths(address: str = None) -> tuple[Path, Path]:
    """Update module-level paths so pipeline modules read/write for the target token."""
    global DATA_DIR, DASHBOARDS_DIR
    global PRICES_JSON, TRADES_CSV, PNL_FEATURES_CSV, TRAINING_DATA_CSV
    global DASHBOARD_FEATURES_JSON, EDA_FEATURES_JSON

    DATA_DIR, DASHBOARDS_DIR = get_token_dirs(address)
    PRICES_JSON = DASHBOARDS_DIR / f"prices_{DUNE_ID}.json"
    TRADES_CSV = DATA_DIR / f"dune_query_{DUNE_ID}_latest.csv"
    PNL_FEATURES_CSV = DATA_DIR / f"pnl_features_{DUNE_ID}.csv"
    TRAINING_DATA_CSV = DATA_DIR / f"training_data_{DUNE_ID}.csv"
    DASHBOARD_FEATURES_JSON = DASHBOARDS_DIR / f"dashboard_features_{DUNE_ID}.json"
    EDA_FEATURES_JSON = DASHBOARDS_DIR / "eda_features.json"
    return DATA_DIR, DASHBOARDS_DIR
