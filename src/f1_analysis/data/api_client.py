"""HTTP client for the Jolpica (Ergast-compatible) and OpenF1 APIs with a local JSON cache.

Every response is saved under ``data/raw/`` so that analyses are reproducible
offline once the data has been downloaded.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from f1_analysis.config import (
    DEFAULT_TIMEOUT,
    JOLPICA_BASE_URL,
    JOLPICA_PAGE_DELAY,
    JOLPICA_PAGE_LIMIT,
    OPENF1_BASE_URL,
    RAW_DATA_DIR,
)

logger = logging.getLogger(__name__)


class DataFetchError(RuntimeError):
    """Raised when data cannot be downloaded and no valid cache exists."""


class F1ApiClient:
    """Client for Jolpica and OpenF1 with transparent on-disk caching."""

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = 3,
        session: Optional[requests.Session] = None,
    ) -> None:
        """Create the client.

        Args:
            cache_dir: Folder for cached JSON files (default ``data/raw``).
            timeout: Request timeout in seconds.
            max_retries: Retries for transient errors (429 and 5xx).
            session: Optional pre-built session (used by tests to inject fakes).
        """
        self.cache_dir = Path(cache_dir) if cache_dir else RAW_DATA_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout

        if session is None:
            session = requests.Session()
            retry = Retry(
                total=max_retries,
                backoff_factor=1.0,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["GET"],
            )
            session.mount("https://", HTTPAdapter(max_retries=retry))
        self.session = session

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------
    def _read_cache(self, path: Path) -> Optional[Any]:
        """Return cached JSON, or None if the file is missing or corrupted."""
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Ignoring unreadable cache file %s (%s)", path.name, exc)
            return None

    def _write_cache(self, path: Path, data: Any) -> None:
        """Write JSON atomically (temp file + rename) so a crash cannot leave half a file."""
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        tmp.replace(path)

    def _get_json(self, url: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Perform one GET request and decode JSON, wrapping failures in DataFetchError."""
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            raise DataFetchError(f"Request to {url} failed: {exc}") from exc

    def _fetch_with_cache(
        self,
        url: str,
        cache_filename: str,
        params: Optional[Dict[str, Any]] = None,
        force_refresh: bool = False,
    ) -> Any:
        """Return cached JSON if present, otherwise download once and cache it."""
        path = self.cache_dir / cache_filename
        if not force_refresh:
            cached = self._read_cache(path)
            if cached is not None:
                return cached
        logger.info("Downloading %s", url)
        data = self._get_json(url, params)
        self._write_cache(path, data)
        return data

    def _fetch_jolpica_paginated(
        self,
        path: str,
        list_key: str,
        cache_filename: str,
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        """Download every page of a Jolpica race-level endpoint and merge them.

        Jolpica returns at most 100 rows per request. ``MRData.total`` tells us
        how many rows exist, so we request pages with increasing ``offset``.

        For ``Laps`` a single lap's timings can be split across two pages, so
        laps are merged by lap number. For ``PitStops`` pages are concatenated.

        The merged result is cached only if the row count matches ``total``;
        a cached file with too few rows is treated as invalid and re-downloaded.
        """
        cache_path = self.cache_dir / cache_filename
        if not force_refresh:
            cached = self._read_cache(cache_path)
            if cached is not None and self._jolpica_is_complete(cached, list_key):
                return cached

        url = f"{JOLPICA_BASE_URL}/{path}.json"
        offset, total = 0, None
        first_page: Optional[Dict[str, Any]] = None
        items: List[Dict[str, Any]] = []

        while total is None or offset < total:
            page = self._get_json(url, {"limit": JOLPICA_PAGE_LIMIT, "offset": offset})
            mr = page["MRData"]
            total = int(mr["total"])
            if first_page is None:
                first_page = page
            races = mr["RaceTable"]["Races"]
            if races:
                items.extend(races[0].get(list_key, []))
            offset += JOLPICA_PAGE_LIMIT
            if offset < total:
                time.sleep(JOLPICA_PAGE_DELAY)

        assert first_page is not None
        if list_key == "Laps":
            items = self._merge_lap_pages(items)

        races = first_page["MRData"]["RaceTable"]["Races"]
        if races:
            races[0][list_key] = items
        first_page["MRData"]["limit"] = str(total)
        first_page["MRData"]["offset"] = "0"

        if self._jolpica_is_complete(first_page, list_key):
            self._write_cache(cache_path, first_page)
        else:
            logger.warning("Downloaded %s is incomplete; not caching it.", cache_filename)
        return first_page

    @staticmethod
    def _merge_lap_pages(laps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge lap entries that were split across pages (same lap number)."""
        merged: Dict[int, Dict[str, Any]] = {}
        for lap in laps:
            n = int(lap["number"])
            if n not in merged:
                merged[n] = {"number": lap["number"], "Timings": []}
            merged[n]["Timings"].extend(lap.get("Timings", []))
        return [merged[n] for n in sorted(merged)]

    @staticmethod
    def _jolpica_is_complete(data: Dict[str, Any], list_key: str) -> bool:
        """Check that the number of rows equals MRData.total."""
        try:
            mr = data["MRData"]
            total = int(mr["total"])
            races = mr["RaceTable"]["Races"]
        except (KeyError, TypeError, ValueError):
            return False
        if total == 0:
            return True
        if not races:
            return False
        rows = races[0].get(list_key, [])
        if list_key == "Laps":
            count = sum(len(lap.get("Timings", [])) for lap in rows)
        else:
            count = len(rows)
        return count == total

    # ------------------------------------------------------------------
    # Jolpica endpoints
    # ------------------------------------------------------------------
    def get_season_calendar(self, year: int, force_refresh: bool = False) -> Dict[str, Any]:
        """Season schedule (fewer than 30 races, so one page suffices)."""
        return self._fetch_with_cache(
            f"{JOLPICA_BASE_URL}/{year}.json",
            f"jolpica_{year}_calendar.json",
            params={"limit": JOLPICA_PAGE_LIMIT},
            force_refresh=force_refresh,
        )

    def get_race_results(self, year: int, round_num: int, force_refresh: bool = False) -> Dict[str, Any]:
        """Final classification of one race (at most ~24 rows, one page)."""
        return self._fetch_with_cache(
            f"{JOLPICA_BASE_URL}/{year}/{round_num}/results.json",
            f"jolpica_{year}_round_{round_num}_results.json",
            params={"limit": JOLPICA_PAGE_LIMIT},
            force_refresh=force_refresh,
        )

    def get_race_laps(self, year: int, round_num: int, force_refresh: bool = False) -> Dict[str, Any]:
        """Every lap timing of every driver (paginated, typically 1,000+ rows)."""
        return self._fetch_jolpica_paginated(
            f"{year}/{round_num}/laps", "Laps",
            f"jolpica_{year}_round_{round_num}_laps.json", force_refresh,
        )

    def get_pit_stops(self, year: int, round_num: int, force_refresh: bool = False) -> Dict[str, Any]:
        """Every pit stop of one race (paginated; can exceed 30 rows)."""
        return self._fetch_jolpica_paginated(
            f"{year}/{round_num}/pitstops", "PitStops",
            f"jolpica_{year}_round_{round_num}_pitstops.json", force_refresh,
        )

    # ------------------------------------------------------------------
    # OpenF1 endpoints (no pagination: full result per query)
    # ------------------------------------------------------------------
    def get_openf1_race_sessions(self, year: int, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """All race sessions (session_name == 'Race') of a season."""
        return self._fetch_with_cache(
            f"{OPENF1_BASE_URL}/sessions",
            f"openf1_{year}_race_sessions.json",
            params={"year": year, "session_name": "Race"},
            force_refresh=force_refresh,
        )

    def get_openf1_stints(self, session_key: int, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """Tyre stints (compound, lap range, tyre age at start) for one session."""
        return self._fetch_with_cache(
            f"{OPENF1_BASE_URL}/stints",
            f"openf1_session_{session_key}_stints.json",
            params={"session_key": session_key},
            force_refresh=force_refresh,
        )
