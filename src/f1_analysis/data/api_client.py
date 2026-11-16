"""API client for fetching and caching Formula 1 data from Jolpica and OpenF1.

This module provides a unified interface to retrieve race schedules, results,
lap-by-lap timing, pit stops, and stint/tyre telemetry with automatic local caching
to guarantee offline reproducibility.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.f1_analysis.config import (
    DEFAULT_TIMEOUT,
    JOLPICA_BASE_URL,
    OPENF1_BASE_URL,
    RAW_DATA_DIR,
)

logger = logging.getLogger(__name__)


class F1ApiClient:
    """Client for interacting with Jolpica and OpenF1 APIs with local caching."""

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = 3,
    ) -> None:
        """Initialize the API client.

        Args:
            cache_dir: Directory to save cached JSON responses. Defaults to RAW_DATA_DIR.
            timeout: HTTP request timeout in seconds.
            max_retries: Number of retry attempts on transient network/server failures.
        """
        self.cache_dir = cache_dir or RAW_DATA_DIR
        self.timeout = timeout
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Configure session with robust retries for transient HTTP errors (429, 500, 502, 503, 504)
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1.0,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def _fetch_with_cache(
        self,
        url: str,
        cache_filename: str,
        params: Optional[Dict[str, Any]] = None,
        force_refresh: bool = False,
    ) -> Any:
        """Internal helper to retrieve data from local cache or make an HTTP GET request.

        Args:
            url: The endpoint URL to query.
            cache_filename: Target filename inside cache_dir (e.g. 'jolpica_2024_1_laps.json').
            params: Optional query parameters for the request.
            force_refresh: If True, bypasses local cache and fetches fresh data from API.

        Returns:
            Parsed JSON object (dict or list).

        Raises:
            requests.HTTPError: If HTTP request fails.
            requests.RequestException: If a network error occurs.
        """
        cache_path = self.cache_dir / cache_filename

        # 1. Check local cache first (Offline / Fast path)
        if cache_path.exists() and not force_refresh:
            logger.info("Loading cached data from %s", cache_path.name)
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)

        # 2. Cache miss or forced refresh: execute network request
        logger.info("Fetching fresh data from %s", url)
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()

            # 3. Save to disk cache for future offline runs
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            return data
        except requests.RequestException as exc:
            logger.error("Failed to fetch data from %s: %s", url, exc)
            raise

    # -----------------------------------------------------------------------
    # Jolpica F1 API Methods (Ergast Continuation)
    # -----------------------------------------------------------------------

    def get_season_calendar(self, year: int, force_refresh: bool = False) -> Dict[str, Any]:
        """Fetch the full Grand Prix schedule for a given season.

        Args:
            year: Formula 1 season (e.g. 2024).
            force_refresh: If True, bypasses local cache.

        Returns:
            JSON response dictionary containing races list.
        """
        url = f"{JOLPICA_BASE_URL}/{year}.json"
        filename = f"jolpica_{year}_calendar.json"
        return self._fetch_with_cache(url, filename, force_refresh=force_refresh)

    def get_race_results(
        self, year: int, round_num: int, force_refresh: bool = False
    ) -> Dict[str, Any]:
        """Fetch final classification results for a specific Grand Prix.

        Args:
            year: Season year (e.g. 2024).
            round_num: Race round number (e.g. 1 for Bahrain).
            force_refresh: If True, bypasses local cache.

        Returns:
            JSON response dictionary containing race results.
        """
        url = f"{JOLPICA_BASE_URL}/{year}/{round_num}/results.json"
        filename = f"jolpica_{year}_round_{round_num}_results.json"
        return self._fetch_with_cache(url, filename, force_refresh=force_refresh)

    def get_race_laps(
        self,
        year: int,
        round_num: int,
        limit: int = 2000,
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        """Fetch lap-by-lap timing data for all drivers in a race.

        Args:
            year: Season year (e.g. 2024).
            round_num: Race round number.
            limit: Maximum number of records per request (default 2000 to cover all laps).
            force_refresh: If True, bypasses local cache.

        Returns:
            JSON response dictionary containing lap-by-lap timing.
        """
        url = f"{JOLPICA_BASE_URL}/{year}/{round_num}/laps.json"
        filename = f"jolpica_{year}_round_{round_num}_laps.json"
        params = {"limit": limit}
        return self._fetch_with_cache(url, filename, params=params, force_refresh=force_refresh)

    def get_pit_stops(
        self, year: int, round_num: int, force_refresh: bool = False
    ) -> Dict[str, Any]:
        """Fetch all pit stops recorded during a race.

        Args:
            year: Season year (e.g. 2024).
            round_num: Race round number.
            force_refresh: If True, bypasses local cache.

        Returns:
            JSON response dictionary containing pit stops list.
        """
        url = f"{JOLPICA_BASE_URL}/{year}/{round_num}/pitstops.json"
        filename = f"jolpica_{year}_round_{round_num}_pitstops.json"
        return self._fetch_with_cache(url, filename, force_refresh=force_refresh)

    # -----------------------------------------------------------------------
    # OpenF1 API Methods (Telemetry, Stints, Tyre Compounds)
    # -----------------------------------------------------------------------

    def get_openf1_sessions(
        self, year: int, country_name: Optional[str] = None, force_refresh: bool = False
    ) -> List[Dict[str, Any]]:
        """Fetch session keys and details from OpenF1.

        Args:
            year: Season year (e.g. 2024).
            country_name: Optional country filter (e.g. 'Bahrain').
            force_refresh: If True, bypasses local cache.

        Returns:
            List of session records from OpenF1.
        """
        url = f"{OPENF1_BASE_URL}/sessions"
        params: Dict[str, Any] = {"year": year}
        if country_name:
            params["country_name"] = country_name
        
        country_slug = country_name.lower().replace(" ", "_") if country_name else "all"
        filename = f"openf1_{year}_{country_slug}_sessions.json"
        return self._fetch_with_cache(url, filename, params=params, force_refresh=force_refresh)

    def get_openf1_stints(
        self, session_key: int, force_refresh: bool = False
    ) -> List[Dict[str, Any]]:
        """Fetch tyre stints, compounds, and tyre age per driver from OpenF1.

        Args:
            session_key: OpenF1 unique session key.
            force_refresh: If True, bypasses local cache.

        Returns:
            List of tyre stint records.
        """
        url = f"{OPENF1_BASE_URL}/stints"
        filename = f"openf1_session_{session_key}_stints.json"
        params = {"session_key": session_key}
        return self._fetch_with_cache(url, filename, params=params, force_refresh=force_refresh)

    def get_openf1_intervals(
        self, session_key: int, force_refresh: bool = False
    ) -> List[Dict[str, Any]]:
        """Fetch lap-by-lap gap intervals to leader and ahead car.

        Args:
            session_key: OpenF1 unique session key.
            force_refresh: If True, bypasses local cache.

        Returns:
            List of interval records.
        """
        url = f"{OPENF1_BASE_URL}/intervals"
        filename = f"openf1_session_{session_key}_intervals.json"
        params = {"session_key": session_key}
        return self._fetch_with_cache(url, filename, params=params, force_refresh=force_refresh)
