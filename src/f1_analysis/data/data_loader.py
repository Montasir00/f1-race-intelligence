"""Convert raw API JSON into tidy Pandas DataFrames.

Each ``load_*`` method returns a DataFrame with a documented, fixed set of columns.
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from f1_analysis.data.api_client import F1ApiClient

logger = logging.getLogger(__name__)


def time_to_seconds(time_str: Optional[str]) -> Optional[float]:
    """Convert 'H:MM:SS.mmm', 'M:SS.mmm' or 'SS.mmm' to seconds (None if invalid).

    Examples:
        '1:32.408' -> 92.408
        '24.512'   -> 24.512
    """
    if not time_str or not isinstance(time_str, str):
        return None
    try:
        parts = [float(p) for p in time_str.strip().split(":")]
    except ValueError:
        return None
    if len(parts) > 3:
        return None
    seconds = 0.0
    for p in parts:              # works for 1, 2 or 3 components
        seconds = seconds * 60.0 + p
    return seconds


def _to_int(value: Any) -> Optional[int]:
    """int() that returns None for missing / malformed values."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _first_race(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return MRData.RaceTable.Races[0] or None."""
    try:
        races = raw["MRData"]["RaceTable"]["Races"]
    except (KeyError, TypeError):
        logger.error("Unexpected Jolpica JSON structure")
        return None
    return races[0] if races else None


class F1DataLoader:
    """Loads Jolpica and OpenF1 data as DataFrames."""

    def __init__(self, api_client: Optional[F1ApiClient] = None) -> None:
        self.client = api_client or F1ApiClient()

    # ------------------------------------------------------------------
    # Jolpica
    # ------------------------------------------------------------------
    def load_season_calendar(self, year: int) -> pd.DataFrame:
        """Columns: round, race_name, date, country."""
        raw = self.client.get_season_calendar(year)
        races = raw.get("MRData", {}).get("RaceTable", {}).get("Races", [])
        return pd.DataFrame([
            {
                "round": int(r["round"]),
                "race_name": r.get("raceName"),
                "date": pd.to_datetime(r.get("date")),
                "country": r.get("Circuit", {}).get("Location", {}).get("country"),
            }
            for r in races
        ])

    def load_race_info(self, year: int, round_num: int) -> Dict[str, Any]:
        """Race name, date and circuit for one round (empty dict if unavailable)."""
        race = _first_race(self.client.get_race_results(year, round_num))
        if race is None:
            return {}
        return {
            "season": year,
            "round": round_num,
            "race_name": race.get("raceName"),
            "date": pd.to_datetime(race.get("date")),
            "circuit": race.get("Circuit", {}).get("circuitName"),
            "country": race.get("Circuit", {}).get("Location", {}).get("country"),
        }

    def load_race_results(self, year: int, round_num: int) -> pd.DataFrame:
        """Columns: position, driver_id, driver_number, driver_code, driver_name,
        constructor_id, constructor_name, grid, laps, status, points,
        fastest_lap_time_str, fastest_lap_sec.

        ``driver_number`` is the race car number; it is the key used to join with OpenF1.
        """
        race = _first_race(self.client.get_race_results(year, round_num))
        if race is None:
            return pd.DataFrame()

        records = []
        for r in race.get("Results", []):
            driver = r.get("Driver", {})
            constructor = r.get("Constructor", {})
            fl_time = r.get("FastestLap", {}).get("Time", {}).get("time")
            records.append({
                "position": _to_int(r.get("position")),
                "driver_id": driver.get("driverId"),
                "driver_number": _to_int(r.get("number")),
                "driver_code": driver.get("code"),
                "driver_name": f"{driver.get('givenName', '')} {driver.get('familyName', '')}".strip(),
                "constructor_id": constructor.get("constructorId"),
                "constructor_name": constructor.get("name"),
                "grid": _to_int(r.get("grid")),
                "laps": _to_int(r.get("laps")),
                "status": r.get("status"),
                "points": float(r.get("points", 0) or 0),
                "fastest_lap_time_str": fl_time,
                "fastest_lap_sec": time_to_seconds(fl_time),
            })
        return pd.DataFrame(records)

    def load_race_laps(self, year: int, round_num: int) -> pd.DataFrame:
        """Columns: lap_number, driver_id, position, lap_time_str, lap_time_sec."""
        race = _first_race(self.client.get_race_laps(year, round_num))
        if race is None:
            return pd.DataFrame()

        records = [
            {
                "lap_number": int(lap["number"]),
                "driver_id": t.get("driverId"),
                "position": _to_int(t.get("position")),
                "lap_time_str": t.get("time"),
                "lap_time_sec": time_to_seconds(t.get("time")),
            }
            for lap in race.get("Laps", [])
            for t in lap.get("Timings", [])
        ]
        df = pd.DataFrame(records)
        if not df.empty:
            n_missing = int(df["lap_time_sec"].isna().sum())
            if n_missing:
                logger.warning("%d lap times could not be parsed and are NaN", n_missing)
            df = df.sort_values(["lap_number", "position"]).reset_index(drop=True)
        return df

    def load_pit_stops(self, year: int, round_num: int) -> pd.DataFrame:
        """Columns: driver_id, lap, stop_number, time_of_day, pit_lane_time_str, pit_lane_time_sec.

        Note: Jolpica's ``duration`` is the time between pit-lane entry and exit
        timing lines, NOT the stationary tyre-change time.
        """
        race = _first_race(self.client.get_pit_stops(year, round_num))
        if race is None:
            return pd.DataFrame()

        records = [
            {
                "driver_id": p.get("driverId"),
                "lap": _to_int(p.get("lap")),
                "stop_number": _to_int(p.get("stop")),
                "time_of_day": p.get("time"),
                "pit_lane_time_str": p.get("duration"),
                "pit_lane_time_sec": time_to_seconds(p.get("duration")),
            }
            for p in race.get("PitStops", [])
        ]
        df = pd.DataFrame(records)
        if not df.empty:
            df = df.sort_values(["lap", "driver_id"]).reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    # OpenF1
    # ------------------------------------------------------------------
    def find_openf1_session_key(self, year: int, round_num: int) -> Optional[int]:
        """Find the OpenF1 race session matching a Jolpica round by date.

        Matching by date (within one day, to tolerate time-zone differences)
        avoids relying on country names, which differ between the two APIs
        (e.g. 'UK' vs 'United Kingdom').
        """
        info = self.load_race_info(year, round_num)
        if not info or pd.isna(info.get("date")):
            return None
        sessions = self.client.get_openf1_race_sessions(year)
        if not isinstance(sessions, list) or not sessions:
            return None

        df = pd.DataFrame(sessions)
        if "date_start" not in df or "session_key" not in df:
            return None
        df["date"] = pd.to_datetime(df["date_start"], utc=True).dt.tz_localize(None).dt.normalize()
        df["diff_days"] = (df["date"] - info["date"]).abs().dt.days
        match = df[df["diff_days"] <= 1].sort_values("diff_days")
        if match.empty:
            logger.warning("No OpenF1 race session found for %s round %s", year, round_num)
            return None
        return int(match.iloc[0]["session_key"])

    def load_openf1_stints(self, session_key: int) -> pd.DataFrame:
        """Columns: driver_number, stint_number, compound, tyre_age_at_start, lap_start, lap_end."""
        raw = self.client.get_openf1_stints(session_key)
        if not isinstance(raw, list) or not raw:
            return pd.DataFrame()
        df = pd.DataFrame(raw)
        df["compound"] = df.get("compound", pd.Series(dtype=object)).fillna("UNKNOWN").str.upper()
        cols = ["driver_number", "stint_number", "compound", "tyre_age_at_start", "lap_start", "lap_end"]
        for c in cols:
            if c not in df.columns:
                df[c] = np.nan
        return df[cols]
