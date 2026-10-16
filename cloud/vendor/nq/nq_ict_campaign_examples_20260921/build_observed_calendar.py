"""Create an explicit observed-date calendar for a local replay.

This is a data-boundary helper, not a claim that every absent weekday is a
CME holiday. The generated metadata records that the calendar is derived from
the observed NQ=F input and must not be used as an independently certified
exchange calendar.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    df = pd.read_csv(args.prices, usecols=["datetime"])
    dates = pd.to_datetime(df["datetime"], errors="raise").dt.normalize()
    if dates.duplicated().any() or not dates.is_monotonic_increasing:
        parser.error("price dates must be unique and sorted")
    if (dates.dt.weekday >= 5).any():
        parser.error("observed calendar contains weekend rows")
    payload = {
        "calendar_status": "observed_input_dates_not_exchange_certified",
        "source_prices": str(args.prices.resolve()),
        "coverage_start": dates.iloc[0].date().isoformat(),
        "coverage_end": dates.iloc[-1].date().isoformat(),
        "session_dates": [x.date().isoformat() for x in dates],
        "rows": int(len(dates)),
        "note": "Absent weekdays remain unresolved between holiday closure and provider missing data.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "rows": len(dates),
                      "start": payload["coverage_start"], "end": payload["coverage_end"],
                      "calendar_status": payload["calendar_status"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
