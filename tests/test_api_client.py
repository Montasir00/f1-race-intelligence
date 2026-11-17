"""Tests for the API client: pagination, cache validation and error handling (no network)."""

import json

import pytest
import requests

from f1_analysis.data.api_client import DataFetchError, F1ApiClient


class FakeResponse:
    def __init__(self, payload, status=200):
        self.payload, self.status_code = payload, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")

    def json(self):
        return self.payload


class FakeSession:
    """Serves a 250-row laps table in pages of 100, like Jolpica."""

    def __init__(self, timings_total=250, per_lap=100):
        self.calls = []
        rows = [(i // per_lap + 1, f"d{i % per_lap}") for i in range(timings_total)]
        self.rows, self.total = rows, timings_total

    def get(self, url, params=None, timeout=None):
        self.calls.append(params)
        off, lim = params["offset"], params["limit"]
        page = self.rows[off:off + lim]
        laps = {}
        for lap, drv in page:
            laps.setdefault(lap, []).append({"driverId": drv, "position": "1", "time": "1:30.000"})
        return FakeResponse({"MRData": {"total": str(self.total), "limit": str(lim), "offset": str(off),
                             "RaceTable": {"Races": [{"Laps": [{"number": str(n), "Timings": t}
                                                               for n, t in laps.items()]}]}}})


class FailingSession:
    def get(self, *a, **k):
        raise requests.ConnectionError("offline")


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr("f1_analysis.data.api_client.time.sleep", lambda s: None)


def test_pagination_fetches_all_rows_and_merges_split_laps(tmp_path):
    session = FakeSession(timings_total=250, per_lap=100)
    client = F1ApiClient(cache_dir=tmp_path, session=session)
    data = client.get_race_laps(2024, 1)
    laps = data["MRData"]["RaceTable"]["Races"][0]["Laps"]
    assert len(session.calls) == 3                       # offsets 0, 100, 200
    assert sum(len(l["Timings"]) for l in laps) == 250
    assert [l["number"] for l in laps] == ["1", "2", "3"]  # no duplicated lap entries
    assert (tmp_path / "jolpica_2024_round_1_laps.json").exists()


def test_second_call_uses_cache(tmp_path):
    session = FakeSession()
    client = F1ApiClient(cache_dir=tmp_path, session=session)
    client.get_race_laps(2024, 1)
    n = len(session.calls)
    client.get_race_laps(2024, 1)
    assert len(session.calls) == n


def test_incomplete_cache_is_redownloaded(tmp_path):
    truncated = {"MRData": {"total": "250", "RaceTable": {"Races": [{"Laps": [
        {"number": "1", "Timings": [{"driverId": "x", "time": "1:30.0"}]}]}]}}}
    (tmp_path / "jolpica_2024_round_1_laps.json").write_text(json.dumps(truncated))
    session = FakeSession()
    F1ApiClient(cache_dir=tmp_path, session=session).get_race_laps(2024, 1)
    assert len(session.calls) == 3


def test_corrupted_cache_is_ignored(tmp_path):
    (tmp_path / "jolpica_2024_round_1_laps.json").write_text("{not json")
    session = FakeSession()
    data = F1ApiClient(cache_dir=tmp_path, session=session).get_race_laps(2024, 1)
    assert int(data["MRData"]["total"]) == 250


def test_network_failure_raises_datafetcherror(tmp_path):
    client = F1ApiClient(cache_dir=tmp_path, session=FailingSession())
    with pytest.raises(DataFetchError):
        client.get_race_results(2024, 1)
