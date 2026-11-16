"""Data loading module that converts raw API responses into structured Pandas DataFrames.

Provides clean tabular abstractions for race results, lap times, pit stops,
and tyre stint information.
"""

import logging
from typing import Any, Dict, List, Optional
import pandas as pd

from src.f1_analysis.data.api_client import F1ApiClient

logger = logging.getLogger(__name__)


def time_to_seconds(time_str: Optional[str]) -> Optional[float]:
    """Convert an F1 lap or duration string ('M:SS.mmm' or 'SS.mmm') to float seconds.

    Examples:
        '1:32.408' -> 92.408
        '24.512'   -> 24.512
        None / ''  -> None

    Args:
        time_str: String representation of lap or pit stop time.

    Returns:
        Time in seconds as float, or None if conversion fails.
    """
    if not time_str or not isinstance(time_str, str):
        return None
    time_str = time_str.strip()
    try:
        if ":" in time_str:
            parts = time_str.split(":")
            if len(parts) == 2:
                minutes, seconds = parts
                return float(minutes) * 60.0 + float(seconds)
            elif len(parts) == 3:
                hours, minutes, seconds = parts
                return float(hours) * 3600.0 + float(minutes) * 60.0 + float(seconds)
        return float(time_str)
    except (ValueError, TypeError):
        return None


class F1DataLoader:
    """Loads and standardizes Formula 1 datasets into structured Pandas DataFrames."""

    def __init__(self, api_client: Optional[F1ApiClient] = None) -> None:
        """Initialize the DataLoader with an API client.

        Args:
            api_client: Optional custom F1ApiClient instance. Defaults to new instance.
        """
        self.client = api_client or F1ApiClient()

    def load_race_results(self, year: int, round_num: int) -> pd.DataFrame:
        """Load race classification results as a DataFrame.

        Columns returned:
            ['position', 'driver_id', 'driver_code', 'driver_name',
             'constructor_id', 'constructor_name', 'grid', 'laps',
             'status', 'points', 'fastest_lap_time_str', 'fastest_lap_sec']
        """
        raw = self.client.get_race_results(year, round_num)
        try:
            races = raw["MRData"]["RaceTable"]["Races"]
            if not races:
                return pd.DataFrame()
            race_info = races[0]
            results = race_info.get("Results", [])
        except KeyError as exc:
            logger.error("Unexpected JSON structure in race results: %s", exc)
            return pd.DataFrame()

        records = []
        for r in results:
            driver = r.get("Driver", {})
            constructor = r.get("Constructor", {})
            fastest_lap = r.get("FastestLap", {})
            fastest_lap_time = fastest_lap.get("Time", {}).get("time")

            records.append({
                "position": int(r.get("position", 0)) if r.get("position") else None,
                "driver_id": driver.get("driverId"),
                "driver_code": driver.get("code", driver.get("driverId", "")[:3].upper()),
                "driver_name": f"{driver.get('givenName', '')} {driver.get('familyName', '')}".strip(),
                "constructor_id": constructor.get("constructorId"),
                "constructor_name": constructor.get("name"),
                "grid": int(r.get("grid", 0)) if r.get("grid") else None,
                "laps": int(r.get("laps", 0)) if r.get("laps") else None,
                "status": r.get("status"),
                "points": float(r.get("points", 0.0)) if r.get("points") else 0.0,
                "fastest_lap_time_str": fastest_lap_time,
                "fastest_lap_sec": time_to_seconds(fastest_lap_time),
            })

        df = pd.DataFrame(records)
        return df

    def load_race_laps(self, year: int, round_num: int) -> pd.DataFrame:
        """Load lap-by-lap timing records for all drivers.

        Columns returned:
            ['lap_number', 'driver_id', 'position', 'lap_time_str', 'lap_time_sec']
        """
        raw = self.client.get_race_laps(year, round_num)
        try:
            races = raw["MRData"]["RaceTable"]["Races"]
            if not races:
                return pd.DataFrame()
            laps_data = races[0].get("Laps", [])
        except KeyError as exc:
            logger.error("Unexpected JSON structure in lap times: %s", exc)
            return pd.DataFrame()

        records = []
        for lap in laps_data:
            lap_num = int(lap.get("number", 0))
            timings = lap.get("Timings", [])
            for t in timings:
                time_str = t.get("time")
                records.append({
                    "lap_number": lap_num,
                    "driver_id": t.get("driverId"),
                    "position": int(t.get("position", 0)) if t.get("position") else None,
                    "lap_time_str": time_str,
                    "lap_time_sec": time_to_seconds(time_str),
                })

        df = pd.DataFrame(records)
        if not df.empty:
            df.sort_values(by=["lap_number", "position"], inplace=True)
            df.reset_index(drop=True, inplace=True)
        return df

    def load_pit_stops(self, year: int, round_num: int) -> pd.DataFrame:
        """Load race pit stops.

        Columns returned:
            ['driver_id', 'lap', 'stop_number', 'time_of_day', 'duration_str', 'duration_sec']
        """
        raw = self.client.get_pit_stops(year, round_num)
        try:
            races = raw["MRData"]["RaceTable"]["Races"]
            if not races:
                return pd.DataFrame()
            pit_stops = races[0].get("PitStops", [])
        except KeyError as exc:
            logger.error("Unexpected JSON structure in pit stops: %s", exc)
            return pd.DataFrame()

        records = []
        for p in pit_stops:
            dur_str = p.get("duration")
            records.append({
                "driver_id": p.get("driverId"),
                "lap": int(p.get("lap", 0)),
                "stop_number": int(p.get("stop", 1)),
                "time_of_day": p.get("time"),
                "duration_str": dur_str,
                "duration_sec": time_to_seconds(dur_str),
            })

        df = pd.DataFrame(records)
        if not df.empty:
            df.sort_values(by=["lap", "driver_id"], inplace=True)
            df.reset_index(drop=True, inplace=True)
        return df

    def load_openf1_stints(self, session_key: int) -> pd.DataFrame:
        """Load tyre compound and stint lengths from OpenF1.

        Columns returned:
            ['driver_number', 'stint_number', 'compound', 'tyre_age_at_start',
             'lap_start', 'lap_end']
        """
        raw = self.client.get_openf1_stints(session_key)
        if not isinstance(raw, list):
            return pd.DataFrame()

        records = []
        for s in raw:
            records.append({
                "driver_number": s.get("driver_number"),
                "stint_number": s.get("stint_number"),
                "compound": str(s.get("compound", "UNKNOWN")).upper(),
                "tyre_age_at_start": s.get("tyre_age_at_start", 0),
                "lap_start": s.get("lap_start"),
                "lap_end": s.get("lap_end"),
            })

        df = pd.DataFrame(records)
        return df
