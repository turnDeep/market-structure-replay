"""Fetch a small Yahoo Finance NKD=F refresh and build an isolated, audited input.

The source dataset and campaign outputs are preserved. The explicit completed
session cutoff prevents the still-forming current daily bar from entering tests.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
from zoneinfo import ZoneInfo

from curl_cffi import requests


PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parent
BASE_PRICES = ROOT / "nq_25y_actual_2000_2026_20260911" / "NQ_F_1d_combined.csv"
BASE_CALENDAR = PACKAGE / "observed_input_calendar.json"
OLD_HOLDOUT = ROOT / "nq_holdout_20260911" / "NQ_F_1d_holdout.csv"
YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/NQ%3DF"
YAHOO_HISTORY_URL = "https://finance.yahoo.com/quote/NQ%3DF/history/"
FIELDS = ("datetime", "open", "high", "low", "close", "adj_close", "volume")
PRICE_FIELDS = ("open", "high", "low", "close", "volume")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fields: list[str] | tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def get_chart(start: str, end_exclusive: str) -> tuple[dict, int, str]:
    start_epoch = int(datetime.fromisoformat(start).replace(tzinfo=timezone.utc).timestamp())
    end_epoch = int(datetime.fromisoformat(end_exclusive).replace(tzinfo=timezone.utc).timestamp())
    params = {"period1": start_epoch, "period2": end_epoch, "interval": "1d"}
    last_error = None
    for attempt, delay in enumerate((0.0, 1.0, 2.0), start=1):
        if delay:
            time.sleep(delay)
        try:
            response = requests.get(YAHOO_URL, params=params, impersonate="chrome", timeout=25)
            if response.status_code == 429 or response.status_code >= 500:
                last_error = f"HTTP {response.status_code} on attempt {attempt}"
                continue
            response.raise_for_status()
            payload = response.json().get("chart", {})
            if payload.get("error"):
                raise RuntimeError(f"Yahoo chart error: {payload['error']}")
            if not payload.get("result"):
                raise RuntimeError("Yahoo chart returned no result")
            return payload["result"][0], response.status_code, response.url.split("?", 1)[0]
        except Exception as exc:  # bounded retries; no parallel requests
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt == 3:
                break
    raise RuntimeError(f"Yahoo request did not complete after three serial attempts: {last_error}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=PACKAGE / "validation_rev11_sweep_followup_20260925_audited")
    parser.add_argument("--start", default="2026-08-25", help="include one confirmed overlap session before stale tail")
    parser.add_argument("--complete-through", default="2026-09-24",
                        help="last completed daily session; later current-session bar is retained only in raw")
    parser.add_argument("--end-exclusive", default="2026-09-26")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    for name in ("data", "detector_v09", "evaluation_v09", "campaign"):
        (out / name).mkdir()

    base = read_csv(BASE_PRICES)
    if not base:
        raise ValueError("The frozen base input is empty")
    result, status, endpoint = get_chart(args.start, args.end_exclusive)
    timezone_name = result.get("meta", {}).get("exchangeTimezoneName")
    if not timezone_name:
        raise ValueError("Yahoo response omitted the exchange timezone")
    tz = ZoneInfo(timezone_name)
    timestamps = result.get("timestamp") or []
    quote = result["indicators"]["quote"][0]
    adj_values = (result.get("indicators", {}).get("adjclose") or [{}])[0].get("adjclose")
    raw = []
    for i, stamp in enumerate(timestamps):
        date = datetime.fromtimestamp(stamp, tz=tz).date().isoformat()
        row = {"datetime": date}
        for field in ("open", "high", "low", "close", "volume"):
            values = quote.get(field) or []
            row[field] = values[i] if i < len(values) else None
        row["adj_close"] = (adj_values[i] if adj_values and i < len(adj_values) and adj_values[i] is not None
                            else row["close"])
        raw.append(row)
    if not raw or raw != sorted(raw, key=lambda r: r["datetime"]):
        raise ValueError("Yahoo daily rows are empty or not sorted")
    if len({r["datetime"] for r in raw}) != len(raw):
        raise ValueError("Yahoo daily response contains duplicate session dates")
    raw = [{**r, "is_complete": r["datetime"] <= args.complete_through} for r in raw]
    completed = [{k: r[k] for k in FIELDS} for r in raw if r["datetime"] <= args.complete_through]
    partial = [r for r in raw if r["datetime"] > args.complete_through]

    anchor = next((r for r in base if r["datetime"] == args.start), None)
    if anchor is None:
        raise ValueError(f"Frozen base input has no anchor session {args.start}")
    overlap = next((r for r in completed if r["datetime"] == args.start), None)
    if overlap is None:
        raise ValueError("No complete overlap row to reconcile against the frozen base input")
    overlap_diffs = {}
    for field in PRICE_FIELDS:
        old, new = float(anchor[field]), float(overlap[field])
        if not math.isclose(old, new, rel_tol=0.0, abs_tol=1e-8):
            overlap_diffs[field] = {"base": old, "yahoo_refresh": new}
    if overlap_diffs:
        raise ValueError(f"Base/Yahoo overlap mismatch; stop rather than splice inconsistent price vintages: {overlap_diffs}")

    refreshed_after_anchor = [r for r in completed if r["datetime"] > args.start]
    if not refreshed_after_anchor:
        raise ValueError("Yahoo returned no completed sessions after the verified anchor")
    stale_tail = [r for r in base if r["datetime"] > args.start]
    base_last_date = base[-1]["datetime"]
    new_dates = [r["datetime"] for r in refreshed_after_anchor if r["datetime"] > base_last_date]
    extended = [r for r in base if r["datetime"] <= args.start] + refreshed_after_anchor
    dates = [r["datetime"] for r in extended]
    if dates != sorted(set(dates)):
        raise ValueError("Extended price input dates are not strictly unique and increasing")
    invalid_rows, null_rows, zero_volume, negative_volume = [], [], [], []
    off_tick_rows = []
    for r in extended:
        values = [float(r[k]) if r.get(k) not in (None, "") else math.nan for k in PRICE_FIELDS]
        o, h, lo, c, vol = values
        if not all(math.isfinite(x) for x in values):
            null_rows.append(r["datetime"])
            continue
        if lo > min(o, c) or h < max(o, c) or h < lo or lo <= 0:
            invalid_rows.append(r["datetime"])
        if vol == 0:
            zero_volume.append(r["datetime"])
        if vol < 0:
            negative_volume.append(r["datetime"])
        if any(abs(x * 4 - round(x * 4)) > 1e-5 for x in (o, h, lo, c)):
            off_tick_rows.append(r["datetime"])
    if invalid_rows or null_rows or negative_volume:
        raise ValueError(f"Extended OHLCV checks failed: invalid={invalid_rows[:10]}, null={null_rows[:10]}, negative_volume={negative_volume[:10]}")

    stale_compare = []
    if OLD_HOLDOUT.exists():
        old_rows = {r["datetime"]: r for r in read_csv(OLD_HOLDOUT)}
        fresh_rows = {r["datetime"]: r for r in completed}
        for date in sorted(set(old_rows) & set(fresh_rows)):
            old, fresh = old_rows[date], fresh_rows[date]
            changed = [f for f in PRICE_FIELDS if f in old and not math.isclose(
                float(old[f]), float(fresh[f]), rel_tol=0.0, abs_tol=1e-8)]
            stale_compare.append({"datetime": date, "mismatch_fields": ",".join(changed),
                                  "old_volume": old.get("volume"), "refreshed_volume": fresh.get("volume"),
                                  "mismatch": bool(changed)})

    calendar = json.loads(BASE_CALENDAR.read_text(encoding="utf-8-sig"))
    if calendar["coverage_end"] != base_last_date:
        raise ValueError("Observed-date calendar does not match frozen base end")
    sessions = [d for d in calendar["session_dates"] if d <= args.start]
    sessions.extend(r["datetime"] for r in refreshed_after_anchor)
    if sessions != sorted(set(sessions)):
        raise ValueError("Extended observed-session calendar has duplicate or unsorted dates")
    calendar["coverage_end"] = refreshed_after_anchor[-1]["datetime"]
    calendar["session_dates"] = sessions
    calendar["calendar_status"] = "observed_input_dates_not_exchange_certified; Yahoo extension through completed cutoff"
    calendar["extension_source"] = YAHOO_HISTORY_URL
    calendar["extension_capture_utc"] = datetime.now(timezone.utc).isoformat()
    calendar["completed_through"] = refreshed_after_anchor[-1]["datetime"]

    write_csv(out / "data" / "yahoo_recent_bars_raw.csv", raw,
              (*FIELDS, "is_complete"))
    write_csv(out / "data" / "yahoo_recent_bars_complete.csv", completed, FIELDS)
    write_csv(out / "data" / "NQ_F_1d_extended.csv", extended, FIELDS)
    write_csv(out / "data" / "old_holdout_vs_yahoo_refresh.csv", stale_compare,
              ("datetime", "mismatch_fields", "old_volume", "refreshed_volume", "mismatch"))
    (out / "data" / "observed_input_calendar_extended.json").write_text(
        json.dumps(calendar, ensure_ascii=False, indent=2), encoding="utf-8")

    metadata = {
        "status": "DATA_REFRESHED_AND_AUDITED",
        "source": "Yahoo Finance chart endpoint via serial curl_cffi Chrome impersonation",
        "source_history_page": YAHOO_HISTORY_URL,
        "source_endpoint": endpoint,
        "http_status": status,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "exchange": result.get("meta", {}).get("exchangeName"),
        "exchange_timezone": timezone_name,
        "currency": result.get("meta", {}).get("currency"),
        "request": {"period_start_inclusive": args.start, "period_end_exclusive": args.end_exclusive,
                    "interval": "1d", "completed_through": args.complete_through},
        "source_rows": {"raw_including_partial": len(raw), "completed_including_overlap": len(completed),
                        "partial_or_after_cutoff_excluded_from_model": [r["datetime"] for r in partial]},
        "base_input": {"path": str(BASE_PRICES.resolve()), "sha256": sha256(BASE_PRICES),
                       "rows": len(base), "start": base[0]["datetime"], "end": base_last_date},
        "overlap_audit": {"date": args.start, "ohlcv_mismatches": overlap_diffs,
                          "passed": not overlap_diffs},
        "refresh_window": {"stale_base_dates_replaced": [r["datetime"] for r in stale_tail],
                           "new_dates_after_old_base_end": new_dates,
                           "refreshed_rows_after_anchor": len(refreshed_after_anchor)},
        "extension": {"start": refreshed_after_anchor[0]["datetime"],
                      "end": refreshed_after_anchor[-1]["datetime"], "extended_rows": len(extended),
                      "extended_sha256": sha256(out / "data" / "NQ_F_1d_extended.csv")},
        "quality": {"duplicate_dates": 0, "null_ohlcv_dates": null_rows,
                    "invalid_ohlc_dates": invalid_rows, "zero_volume_dates": zero_volume,
                    "negative_volume_dates": negative_volume, "off_quarter_point_grid_dates": off_tick_rows,
                    "unrepaired": True},
        "stale_holdout_comparison": {"old_file": str(OLD_HOLDOUT.resolve()),
                                     "overlap_rows": len(stale_compare),
                                     "mismatch_rows": sum(bool(r["mismatch"]) for r in stale_compare),
                                     "details_file": "old_holdout_vs_yahoo_refresh.csv"},
        "calendar": {"path": "observed_input_calendar_extended.json",
                     "status": calendar["calendar_status"], "sessions": len(sessions),
                     "coverage_start": calendar["coverage_start"], "coverage_end": calendar["coverage_end"]},
    }
    (out / "data" / "price_refresh_audit.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
