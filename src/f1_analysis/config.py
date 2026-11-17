"""Central configuration for F1 Race Intelligence.

Holds filesystem paths, API endpoints, modelling assumptions and colour palettes.
Nothing here performs I/O at import time.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Filesystem paths (resolved relative to this file, so they work on any OS)
# src/f1_analysis/config.py -> parents[2] is the project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------
JOLPICA_BASE_URL = "https://api.jolpi.ca/ergast/f1"   # Ergast-compatible API
OPENF1_BASE_URL = "https://api.openf1.org/v1"         # data available from 2023

DEFAULT_TIMEOUT = 25            # seconds per HTTP request
JOLPICA_PAGE_LIMIT = 100        # Jolpica caps `limit` at 100 rows per request
JOLPICA_PAGE_DELAY = 0.3        # seconds between pages (Jolpica burst limit is 4 req/s)

# ---------------------------------------------------------------------------
# Modelling assumptions (NOT measured data - documented so they can be challenged)
# ---------------------------------------------------------------------------
# Fuel effect: a commonly cited engineering rule of thumb is ~0.03 s of lap time
# per kg of fuel. Cars may start with at most 110 kg (FIA regs); 100 kg is a
# typical race start load. Fuel burned per lap is derived from the race distance
# in the data (START_FUEL_KG / total laps), so it is not hardcoded per race.
FUEL_EFFECT_SEC_PER_KG = 0.03
START_FUEL_KG = 100.0

# A lap slower than (driver median * factor) is treated as non-representative.
SLOW_LAP_THRESHOLD_FACTOR = 1.07

# A pit-lane time greater than (race median * factor) is flagged as anomalous
# (typically red-flag stops or long repairs).
PIT_OUTLIER_FACTOR = 2.0

# ---------------------------------------------------------------------------
# Colour palettes
# ---------------------------------------------------------------------------
COMPOUND_COLORS = {
    "SOFT": "#FF1801",
    "MEDIUM": "#FFD700",
    "HARD": "#F0F0F0",
    "INTERMEDIATE": "#39B54A",
    "WET": "#0067AD",
    "UNKNOWN": "#808080",
}

# Keys are matched as case-insensitive substrings of Jolpica constructor names.
CONSTRUCTOR_COLORS = {
    "red bull": "#3671C6",
    "ferrari": "#E80020",
    "mercedes": "#27F4D2",
    "mclaren": "#FF8000",
    "aston martin": "#229971",
    "alpine": "#0093CC",
    "williams": "#64C4FF",
    "alphatauri": "#5E8FAA",
    "rb f1": "#6692FF",
    "alfa romeo": "#C92D4B",
    "sauber": "#52E252",
    "haas": "#B6BABD",
}
