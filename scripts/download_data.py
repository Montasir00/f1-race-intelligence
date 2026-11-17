"""Download and cache the race data used by the project.

Usage (from the project root):
    python scripts/download_data.py --year 2024 --rounds 1 2 3

Raw API responses are cached in data/raw/ (git-ignored) and processed CSV
tables are written to data/processed/. Re-running is cheap: cached files are reused.
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from f1_analysis.data.api_client import DataFetchError  # noqa: E402
from f1_analysis.data.data_loader import F1DataLoader  # noqa: E402
from f1_analysis.pipeline import build_race_dataset, save_processed  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--rounds", type=int, nargs="+", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    loader = F1DataLoader()
    try:
        cal = loader.load_season_calendar(args.year)
        logging.info("Season %s calendar cached (%d rounds).", args.year, len(cal))
    except DataFetchError as exc:
        logging.warning("Could not cache season calendar: %s", exc)

    failures = 0
    for rnd in args.rounds:
        try:
            ds = build_race_dataset(args.year, rnd, loader=loader)
        except DataFetchError as exc:
            logging.error("Round %s failed: %s", rnd, exc)
            failures += 1
            continue
        n_tyre = int(ds.laps["compound"].notna().sum()) if "compound" in ds.laps else 0
        logging.info(
            "%s %s: %d results, %d laps (%d with tyre data), %d pit stops, OpenF1 session %s",
            args.year, ds.info.get("race_name"), len(ds.results), len(ds.laps), n_tyre,
            len(ds.pitstops), ds.openf1_session_key,
        )
        for note in ds.notes:
            logging.warning(note)
        save_processed(ds)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
