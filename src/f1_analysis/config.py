"""Central configuration module for F1 Race Intelligence.

Defines filesystem paths, API endpoints, domain constants, and visualization
palettes used throughout the project.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Filesystem Paths
# ---------------------------------------------------------------------------
# Resolves the project root directory dynamically from this file's location:
# src/f1_analysis/config.py -> parent x 2 = project root
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Notebooks and App directories
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
APP_DIR = PROJECT_ROOT / "app"

# Ensure data directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
# Jolpica F1 API (Modern community continuation of Ergast API)
# Official documentation: https://api.jolpi.ca/ergast/f1/
JOLPICA_BASE_URL = "https://api.jolpi.ca/ergast/f1"

# OpenF1 API (High-resolution telemetry and session data)
# Official documentation: https://openf1.org/
OPENF1_BASE_URL = "https://api.openf1.org/v1"

# Default network request timeout (in seconds)
DEFAULT_TIMEOUT = 25


# ---------------------------------------------------------------------------
# Formula 1 Domain & Physics Constants
# ---------------------------------------------------------------------------
# Fuel load effect approximation:
# F1 cars start with ~100-110 kg of fuel. As fuel burns off, cars gain ~0.035s per lap per kg burned.
# Across a ~55-lap race, lap times naturally drop by ~1.5 - 2.2 seconds solely due to mass reduction.
ESTIMATED_FUEL_CORRECTION_SEC_PER_LAP = 0.035

# Standard Tyre Compound Visual Palette (FIA standard color scheme)
COMPOUND_COLORS = {
    "SOFT": "#FF1801",        # Red
    "MEDIUM": "#FFD700",      # Yellow
    "HARD": "#FFFFFF",        # White
    "INTERMEDIATE": "#00D2BE", # Green
    "WET": "#007FFF",         # Blue
    "UNKNOWN": "#808080",     # Grey
}

# Standard Constructor Visual Palette (2023-2024 season references)
CONSTRUCTOR_COLORS = {
    "Red Bull": "#3671C6",
    "Ferrari": "#E80020",
    "Mercedes": "#27F4D2",
    "McLaren": "#FF8000",
    "Aston Martin": "#229971",
    "Alpine": "#0093CC",
    "Williams": "#64C4FF",
    "AlphaTauri": "#5E8FAA",
    "RB": "#6692FF",
    "Alfa Romeo": "#C92D4B",
    "Sauber": "#52E252",
    "Haas": "#B6BABD",
}
