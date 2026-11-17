"""End-to-end loading + preprocessing for one race.

Used by both the notebook and the Streamlit dashboard, so the preparation
logic exists in exactly one place.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import pandas as pd

from f1_analysis.config import PROCESSED_DATA_DIR
from f1_analysis.data.data_loader import F1DataLoader
from f1_analysis.data.preprocessing import (
    apply_fuel_correction,
    clean_lap_times,
    flag_pit_laps,
    merge_laps_with_stints,
)

logger = logging.getLogger(__name__)


@dataclass
class RaceDataset:
    """All tables for one race. ``laps`` is the analytical lap-level table."""

    info: Dict[str, Any]
    results: pd.DataFrame
    laps: pd.DataFrame
    pitstops: pd.DataFrame
    stints: pd.DataFrame
    openf1_session_key: Optional[int]
    notes: list = field(default_factory=list)


def build_race_dataset(year: int, round_num: int, loader: Optional[F1DataLoader] = None) -> RaceDataset:
    """Load Jolpica (+ OpenF1 stints when available) and run the cleaning pipeline.

    Missing OpenF1 data does not stop the pipeline: tyre columns stay NaN and a
    note is added explaining why.
    """
    loader = loader or F1DataLoader()
    notes = []
    info = loader.load_race_info(year, round_num)
    results = loader.load_race_results(year, round_num)
    laps_raw = loader.load_race_laps(year, round_num)
    pitstops = loader.load_pit_stops(year, round_num)

    session_key, stints = None, pd.DataFrame()
    if year < 2023:
        notes.append("OpenF1 has no data before 2023: tyre analysis unavailable.")
    else:
        try:
            session_key = loader.find_openf1_session_key(year, round_num)
            if session_key is not None:
                stints = loader.load_openf1_stints(session_key)
            else:
                notes.append("No matching OpenF1 race session found: tyre analysis unavailable.")
        except Exception as exc:  # network / API problems must not break Jolpica analysis
            logger.warning("OpenF1 unavailable: %s", exc)
            notes.append(f"OpenF1 request failed ({exc}): tyre analysis unavailable.")

    if laps_raw.empty:
        notes.append("No lap data available for this race.")
        laps = laps_raw
    else:
        laps = flag_pit_laps(laps_raw, pitstops)
        laps = clean_lap_times(laps)
        laps = apply_fuel_correction(laps)
        laps = merge_laps_with_stints(laps, stints, results)

    return RaceDataset(info, results, laps, pitstops, stints, session_key, notes)


def save_processed(dataset: RaceDataset) -> Dict[str, str]:
    """Write the processed tables to ``data/processed`` as CSV and return the paths."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{dataset.info.get('season')}_round_{dataset.info.get('round')}"
    paths = {}
    for name in ("results", "laps", "pitstops", "stints"):
        df = getattr(dataset, name)
        if not df.empty:
            path = PROCESSED_DATA_DIR / f"{tag}_{name}.csv"
            df.to_csv(path, index=False)
            paths[name] = str(path)
    return paths
