"""NKD=F ICT implementation and point-in-time evaluation v0.8.

This version is deliberately a successor to v0.7, not an in-place repair.
It makes four distinctions that are easy to lose in a pooled backtest:

1. article coverage is tied to executable rule IDs and evidence counts;
2. a primitive shape, a daily/weekly adaptation, and a complete local setup
   are separate states;
3. daily/weekly adaptations are not reported as original lower-timeframe ICT
   setups when the required lower-timeframe or session data is unavailable;
4. every stage is evaluated against the same point-in-time arrival and
   post-touch competing-path targets.

The code is an observational study of a continuous Yahoo futures proxy. It
does not claim to observe institutional orders or to establish ICT causality.
Existing v0.7 artifacts are never overwritten by this script.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import platform
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import nq_ict_detection_20260911_v03 as detector
import nq_ict_evaluation_20260911_v03 as ev3
import nq_ict_evaluation_20260911_v04 as v4
import nq_ict_evaluation_20260911_v05 as base


SCRIPT_VERSION = "nq-ict-evaluation-v0.8"
DEFAULT_INPUT = Path("nq_25y_actual_2000_2026_20260911/NQ_F_1d_combined.csv")
DEFAULT_OUTPUT = Path("nq_ict_evaluation_20260912_v08")
DEFAULT_DETECTOR_OUTPUT = Path("nq_ict_detection_20260912_v08")
SOURCE_COVERAGE = Path("nq_ict_evaluation_20260911_v07/article_coverage_v07.json")
SOURCE_MANIFEST = Path("ict_tutorials_corpus_20260909/article_manifest.csv")
SENSITIVITIES = ["strict", "balanced", "exploratory"]
TIMEFRAMES = ["1D", "1W", "1M"]
EVAL_TIMEFRAMES = ["1D", "1W"]
EVAL_TYPES = ["ob_candidate", "fvg"]
ARRIVAL_HORIZON = {"1D": 20, "1W": 60}
POST_HORIZON = 10
BARRIER_ATR = 1.0
MIN_TRAIN = 60
MIN_TEST = 12
POST_CLASSES = ["adverse_first", "favorable_first", "neither_at_horizon"]


# The v0.3 detector contains no monthly profile.  Monthly uses the weekly
# structural profile as an explicitly conservative, non-optimized adaptation.
for _profile in detector.PROFILES.values():
    _profile["max_disp_bars"].setdefault("1M", _profile["max_disp_bars"]["1W"])
    _profile["sweep_wait"].setdefault("1M", _profile["sweep_wait"]["1W"])


def fnum(value: Any, default: float = np.nan) -> float:
    try:
        x = float(value)
        return x if math.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def jdefault(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, (pd.NaT.__class__,)):
        return None
    return str(value)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def add_model_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add causal daily covariates used by the statistical models.

    The detector intentionally keeps its structural feature set small.  The
    evaluation layer therefore derives the market-state covariates here from
    the already observed daily bars, after loading and before any labels are
    made.  Rolling statistics include the current completed bar only, so they
    are available at that bar's close and never require a future row.
    """
    x = frame.copy()
    close = pd.to_numeric(x["close"], errors="coerce")
    volume = pd.to_numeric(x["volume"], errors="coerce")
    x["return_5"] = close.pct_change(5)
    x["return_20"] = close.pct_change(20)
    vol_mean = volume.rolling(20, min_periods=5).mean()
    vol_std = volume.rolling(20, min_periods=5).std(ddof=0).replace(0, np.nan)
    x["volume_z20"] = (volume - vol_mean) / vol_std
    x["sma20"] = close.rolling(20, min_periods=5).mean()
    x["sma50"] = close.rolling(50, min_periods=10).mean()
    atr = pd.to_numeric(x.get("atr14", pd.Series(np.nan, index=x.index)), errors="coerce")
    x["sma_gap20"] = (close - x["sma20"]) / atr.replace(0, np.nan)
    x["sma_gap50"] = (close - x["sma50"]) / atr.replace(0, np.nan)
    return x


def write_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def parse_dates(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out:
            out[col] = pd.to_datetime(out[col], errors="coerce").dt.normalize()
    return out


def load_gap_dates(input_path: Path) -> tuple[set[pd.Timestamp], dict[str, Any]]:
    path = input_path.parent / "excluded_invalid_ohlc_rows.csv"
    if not path.exists():
        return set(), {"path": str(path.resolve()), "rows": 0, "status": "not_found"}
    gap = pd.read_csv(path)
    dates = set(pd.to_datetime(gap["datetime"], errors="coerce").dt.normalize().dropna())
    return dates, {
        "path": str(path.resolve()),
        "rows": int(len(gap)),
        "dates": sorted(str(x.date()) for x in dates),
        "status": "loaded",
    }


def period_bars(daily: pd.DataFrame, frequency: str) -> pd.DataFrame:
    """Create period bars and expose completion separately from availability."""
    x = daily[["datetime", "open", "high", "low", "close", "volume"]].copy()
    x["datetime"] = pd.to_datetime(x["datetime"]).dt.normalize()
    period = x["datetime"].dt.to_period(frequency)
    x["period_start"] = period.dt.start_time.dt.normalize()
    x["period_end"] = period.dt.end_time.dt.normalize()
    out = x.groupby(["period_start", "period_end"], sort=True).agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), volume=("volume", "sum"),
        observed_sessions=("close", "count"), last_session=("datetime", "max"),
    ).reset_index()
    out["datetime"] = out["period_end"]
    out["available_at"] = out["last_session"].dt.normalize()
    out["certified_calendar"] = True
    if frequency == "W-FRI":
        years = range(int(out.period_end.dt.year.min()) - 1, int(out.period_end.dt.year.max()) + 2) if len(out) else []
        closures: set[Any] = set()
        for year in years:
            closures |= detector.known_friday_closures(year, year)
        out["has_expected_friday"] = out.last_session.eq(out.period_end)
        out["expected_friday_closed"] = out.period_end.dt.date.isin(closures)
        out["certified_calendar"] = out.has_expected_friday | out.expected_friday_closed
    else:
        out["has_expected_friday"] = False
        out["expected_friday_closed"] = False
    first_date = pd.Timestamp(daily.datetime.min()).normalize()
    last_date = pd.Timestamp(daily.datetime.max()).normalize()
    out.loc[out.period_start < first_date, "certified_calendar"] = False
    out.loc[out.period_end > last_date, "certified_calendar"] = False
    out["completion_state"] = np.where(out.certified_calendar, "complete", "partial_or_unknown")
    out = detector.add_features(out)
    out["bar_idx"] = np.arange(len(out), dtype=int)
    return out


def run_lane_job(payload: tuple[pd.DataFrame, str, str]) -> dict[str, Any]:
    frame, timeframe, sensitivity = payload
    objects, events, pivots = detector.lane(frame, timeframe, sensitivity)
    return {"timeframe": timeframe, "sensitivity": sensitivity, "objects": objects, "events": events, "pivots": pivots}


def run_lanes(frames: dict[str, pd.DataFrame], n_jobs: int, backend: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[str, str], pd.DataFrame], dict[str, Any]]:
    payloads = [(frames[tf], tf, sens) for tf in TIMEFRAMES for sens in SENSITIVITIES]
    results: list[dict[str, Any]] = []
    if backend == "process" and n_jobs > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as pool:
            futures = [pool.submit(run_lane_job, p) for p in payloads]
            for fut in futures:
                results.append(fut.result())
    elif backend == "thread" and n_jobs > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs) as pool:
            futures = [pool.submit(run_lane_job, p) for p in payloads]
            for fut in futures:
                results.append(fut.result())
    else:
        results = [run_lane_job(p) for p in payloads]
    objects = [r["objects"] for r in results if len(r["objects"])]
    events = [r["events"] for r in results if len(r["events"])]
    obj = pd.concat(objects, ignore_index=True, sort=False) if objects else pd.DataFrame()
    ev = pd.concat(events, ignore_index=True, sort=False) if events else pd.DataFrame()
    pivots = {(r["timeframe"], r["sensitivity"]): r["pivots"] for r in results}
    return obj, ev, pivots, {"lane_count": len(payloads), "backend": backend, "workers": int(n_jobs)}


def add_object_metadata(objects: pd.DataFrame) -> pd.DataFrame:
    if objects.empty:
        return objects.copy()
    out = objects.copy()
    out["symbol"] = "NKD=F"
    article_map = {
        "swing_high_3bar": "82", "swing_low_3bar": "81", "displacement": "75",
        "fvg_3bar": "100;114", "ob_candidate_dw": "87;88;115",
    }
    variant_map = {
        "swing": "source_rule_swing_3bar", "displacement": "source_rule_displacement_dw_v2",
        "fvg": "source_rule_fvg_3bar", "ob_candidate": "daily_weekly_adaptation_ob_candidate_dw_v2",
    }
    out["article_ids"] = out["rule_id"].map(article_map).fillna("")
    out["rule_version"] = SCRIPT_VERSION
    out["definition_variant"] = out["object_type"].map(variant_map).fillna("")
    out["first_observed_at"] = pd.to_datetime(out["origin_time"], errors="coerce").dt.normalize()
    out["updated_at"] = pd.to_datetime(out["available_at"], errors="coerce").dt.normalize()
    if "evidence_state" not in out:
        out["evidence_state"] = "confirmed"
    out["evidence_state"] = out["evidence_state"].fillna("confirmed")
    if "quality_flags" not in out:
        out["quality_flags"] = ""
    out["quality_flags"] = out["quality_flags"].fillna("")
    return out


def mark_object_gaps(objects: pd.DataFrame, frames: dict[str, pd.DataFrame], gap_dates: set[pd.Timestamp]) -> pd.DataFrame:
    if objects.empty:
        return objects.copy()
    out = objects.copy()
    bads: list[bool] = []
    flags: list[str] = []
    for row in out.itertuples(index=False):
        frame = frames.get(str(row.timeframe))
        bad = False
        if frame is not None and len(frame):
            oi, ci = int(fnum(row.origin_idx, -1)), int(fnum(row.confirmed_idx, -1))
            if 0 <= oi < len(frame) and 0 <= ci < len(frame):
                lo, hi = min(oi, ci), max(oi, ci)
                if "period_start" in frame.columns:
                    starts = pd.to_datetime(frame.period_start.iloc[lo:hi + 1]).dt.normalize()
                    ends = pd.to_datetime(frame.period_end.iloc[lo:hi + 1]).dt.normalize()
                    bad = any(any(a <= d <= b for d in gap_dates) for a, b in zip(starts, ends))
                else:
                    dates = pd.to_datetime(frame.datetime.iloc[lo:hi + 1]).dt.normalize()
                    bad = any(d in gap_dates for d in dates)
        old = str(getattr(row, "quality_flags", "") or "")
        bads.append(bool(bad))
        flags.append(";".join(x for x in [old, "source_gap_in_formation_window" if bad else ""] if x))
    out["gap_contaminated"] = bads
    out["quality_flags"] = flags
    return out


def zone_lifecycle_fast(zones: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Use the existing lifecycle definition with compact NumPy access."""
    if zones.empty:
        return pd.DataFrame(columns=["object_id", "first_touch_at", "first_full_at", "first_close_invalidation_at", "touch_episode_count", "terminal_state"])
    return base.zone_lifecycle(zones, daily)


def pivot_arrays(pivots: pd.DataFrame) -> dict[str, np.ndarray]:
    if pivots.empty:
        return {k: np.array([]) for k in ["type", "origin", "confirm", "price", "available"]}
    p = pivots.copy()
    return {
        "type": p.pivot_type.astype(str).to_numpy(),
        "origin": pd.to_numeric(p.origin_idx, errors="coerce").fillna(-1).to_numpy(int),
        "confirm": pd.to_numeric(p.confirmed_idx, errors="coerce").fillna(-1).to_numpy(int),
        "price": pd.to_numeric(p.price, errors="coerce").to_numpy(float),
        "available": pd.to_datetime(p.available_at).dt.normalize().to_numpy(dtype="datetime64[ns]"),
        "id": p.pivot_id.astype(str).to_numpy(),
    }


def _past_breached(prices: np.ndarray, start: int, end: int, level: float, direction: str) -> bool:
    if end < start or start >= len(prices):
        return False
    segment = prices[start:min(end + 1, len(prices))]
    if direction == "up":
        return bool(np.nanmax(segment) > level) if len(segment) else False
    return bool(np.nanmin(segment) < level) if len(segment) else False


def context_state_at(t: pd.Timestamp, daily_idx: int, daily: pd.DataFrame, frame: pd.DataFrame, pivots: pd.DataFrame, timeframe: str) -> dict[str, Any]:
    """Point-in-time HTF trend, separate DOL bias, and range state.

    DOL is an explicit operational adaptation: nearest unconsumed confirmed
    pivot, with neutral on a near tie.  It is not presented as the only ICT
    definition of bias.
    """
    t = pd.Timestamp(t).normalize()
    close = fnum(daily.close.iloc[daily_idx])
    atr = fnum(daily.atr14.iloc[daily_idx])
    period_end = pd.to_datetime(frame.period_end).dt.normalize().to_numpy(dtype="datetime64[ns]")
    complete_n = int(np.searchsorted(period_end, np.datetime64(t, "ns"), side="left"))
    pa = pivot_arrays(pivots)
    usable = (pa["confirm"] < complete_n) & (pa["available"] < np.datetime64(t, "ns"))
    highs = np.flatnonzero(usable & (pa["type"] == "high"))
    lows = np.flatnonzero(usable & (pa["type"] == "low"))
    highs = highs[np.argsort(pa["confirm"][highs])] if len(highs) else highs
    lows = lows[np.argsort(pa["confirm"][lows])] if len(lows) else lows
    trend = "unknown"
    if len(highs) >= 2 and len(lows) >= 2:
        if pa["price"][highs[-1]] > pa["price"][highs[-2]] and pa["price"][lows[-1]] > pa["price"][lows[-2]]:
            trend = "up"
        elif pa["price"][highs[-1]] < pa["price"][highs[-2]] and pa["price"][lows[-1]] < pa["price"][lows[-2]]:
            trend = "down"
        else:
            trend = "mixed"
    last_high = pa["price"][highs[-1]] if len(highs) else np.nan
    last_low = pa["price"][lows[-1]] if len(lows) else np.nan
    protected_high, protected_low = last_high, last_low
    if trend == "up" and len(highs):
        prior = lows[pa["origin"][lows] < pa["origin"][highs[-1]]]
        if len(prior):
            protected_low = pa["price"][prior[-1]]
    if trend == "down" and len(lows):
        prior = highs[pa["origin"][highs] < pa["origin"][lows[-1]]]
        if len(prior):
            protected_high = pa["price"][prior[-1]]
    integrity = "unknown"
    if trend == "up" and math.isfinite(protected_low):
        integrity = "intact" if close >= protected_low else "broken"
    elif trend == "down" and math.isfinite(protected_high):
        integrity = "intact" if close <= protected_high else "broken"
    position, range_mid = "unknown", np.nan
    if math.isfinite(last_low) and math.isfinite(last_high) and last_high > last_low:
        range_mid = (last_low + last_high) / 2.0
        if close < last_low or close > last_high:
            position = "out_of_range"
        elif close < range_mid:
            position = "discount"
        elif close > range_mid:
            position = "premium"
        else:
            position = "midpoint"

    dates = pd.to_datetime(daily.datetime).dt.normalize().to_numpy(dtype="datetime64[ns]")
    highs_daily = pd.to_numeric(daily.high, errors="coerce").to_numpy(float)
    lows_daily = pd.to_numeric(daily.low, errors="coerce").to_numpy(float)
    high_candidates: list[tuple[float, str, float]] = []
    low_candidates: list[tuple[float, str, float]] = []
    for ix in highs:
        price = pa["price"][ix]
        if not math.isfinite(price) or price <= close:
            continue
        confirm_date = pa["available"][ix]
        start = int(np.searchsorted(dates, confirm_date, side="right"))
        if not _past_breached(highs_daily, start, daily_idx, price, "up"):
            dist = (price - close) / atr if atr > 0 else np.nan
            high_candidates.append((dist, str(pa["id"][ix]), price))
    for ix in lows:
        price = pa["price"][ix]
        if not math.isfinite(price) or price >= close:
            continue
        confirm_date = pa["available"][ix]
        start = int(np.searchsorted(dates, confirm_date, side="right"))
        if not _past_breached(lows_daily, start, daily_idx, price, "down"):
            dist = (close - price) / atr if atr > 0 else np.nan
            low_candidates.append((dist, str(pa["id"][ix]), price))
    high_candidates.sort(key=lambda x: x[0] if math.isfinite(x[0]) else 1e99)
    low_candidates.sort(key=lambda x: x[0] if math.isfinite(x[0]) else 1e99)
    du = high_candidates[0][0] if high_candidates else np.nan
    dd = low_candidates[0][0] if low_candidates else np.nan
    bias, target, target_id, target_kind = "neutral", np.nan, "", ""
    bias_status, bias_reason = "neutral", "no_valid_dol"
    if integrity == "broken":
        bias_status, bias_reason = "invalidated", "protected_structure_broken"
    elif math.isfinite(du) and math.isfinite(dd):
        if du + 0.25 < dd:
            bias, target, target_id, target_kind, bias_status, bias_reason = "up", high_candidates[0][2], high_candidates[0][1], "confirmed_pivot_high", "active", "nearest_unconsumed_pivot"
        elif dd + 0.25 < du:
            bias, target, target_id, target_kind, bias_status, bias_reason = "down", low_candidates[0][2], low_candidates[0][1], "confirmed_pivot_low", "active", "nearest_unconsumed_pivot"
        else:
            bias_reason = "dol_distance_tie"
    elif math.isfinite(du):
        bias, target, target_id, target_kind, bias_status, bias_reason = "up", high_candidates[0][2], high_candidates[0][1], "confirmed_pivot_high", "active", "one_sided_unconsumed_pivot"
    elif math.isfinite(dd):
        bias, target, target_id, target_kind, bias_status, bias_reason = "down", low_candidates[0][2], low_candidates[0][1], "confirmed_pivot_low", "active", "one_sided_unconsumed_pivot"
    last_pivot_at = pd.NaT
    if len(highs) or len(lows):
        ix = np.concatenate([highs, lows])
        last_pivot_at = pd.Timestamp(pa["available"][ix].max()).normalize()
    context_end = pd.Timestamp(frame.period_end.iloc[complete_n - 1]).normalize() if complete_n else pd.NaT
    return {
        "timeframe": timeframe,
        "structure_trend": trend,
        "structure_integrity": integrity,
        "bias_direction": bias,
        "bias_status": bias_status,
        "bias_reason": bias_reason,
        "bias_target_price": target,
        "bias_target_id": target_id,
        "bias_target_kind": target_kind,
        "bias_target_distance_atr": min(du, dd) if math.isfinite(du) and math.isfinite(dd) else (du if math.isfinite(du) else dd),
        "bias_target_up_distance_atr": du,
        "bias_target_down_distance_atr": dd,
        "position": position,
        "range_low": last_low,
        "range_high": last_high,
        "range_mid": range_mid,
        "range_width_atr": (last_high - last_low) / atr if math.isfinite(last_low) and math.isfinite(last_high) and atr > 0 else np.nan,
        "protected_high": protected_high,
        "protected_low": protected_low,
        "last_pivot_at": last_pivot_at,
        "structure_age_days": (t - last_pivot_at).days if pd.notna(last_pivot_at) else np.nan,
        "context_bar_end": context_end,
    }


def zone_stats_at(t: pd.Timestamp, close: float, atr: float, zones: pd.DataFrame, life: pd.DataFrame, direction: str, kind: str) -> dict[str, Any]:
    if zones.empty or not (math.isfinite(close) and math.isfinite(atr) and atr > 0):
        return {"distance_atr": np.nan, "active_count": 0, "fresh_count": 0, "nearest_object_id": ""}
    z = zones[zones.object_type.eq(kind) & zones.direction.eq(direction)].copy()
    if z.empty:
        return {"distance_atr": np.nan, "active_count": 0, "fresh_count": 0, "nearest_object_id": ""}
    m = life.set_index("object_id") if len(life) else pd.DataFrame()
    t = pd.Timestamp(t).normalize()
    active_rows = []
    for row in z.itertuples(index=False):
        av = pd.Timestamp(row.available_at).normalize()
        if av > t:
            continue
        rec = m.loc[row.object_id] if len(m) and row.object_id in m.index else None
        if kind == "fvg" and rec is not None and pd.notna(rec.first_full_at) and pd.Timestamp(rec.first_full_at).normalize() <= t:
            continue
        if kind == "ob_candidate" and rec is not None and pd.notna(rec.first_close_invalidation_at) and pd.Timestamp(rec.first_close_invalidation_at).normalize() <= t:
            continue
        touched = rec is not None and pd.notna(rec.first_touch_at) and pd.Timestamp(rec.first_touch_at).normalize() <= t
        lo, hi = fnum(row.zone_low), fnum(row.zone_high)
        if math.isfinite(lo) and math.isfinite(hi):
            distance = 0.0 if lo <= close <= hi else min(abs(lo - close), abs(hi - close))
            active_rows.append((distance / atr, row.object_id, not touched))
    if not active_rows:
        return {"distance_atr": np.nan, "active_count": 0, "fresh_count": 0, "nearest_object_id": ""}
    active_rows.sort(key=lambda x: x[0])
    return {
        "distance_atr": float(active_rows[0][0]),
        "active_count": int(len(active_rows)),
        "fresh_count": int(sum(bool(x[2]) for x in active_rows)),
        "nearest_object_id": str(active_rows[0][1]),
    }


def build_contexts(daily: pd.DataFrame, frames: dict[str, pd.DataFrame], pivots_map: dict[tuple[str, str], pd.DataFrame], balanced_objects: pd.DataFrame, balanced_life: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build one row per daily date using only completed HTF bars and zones."""
    daily = daily.reset_index(drop=True)
    zones_by_tf = {tf: balanced_objects[(balanced_objects.timeframe == tf) & balanced_objects.object_type.isin(EVAL_TYPES)].copy() for tf in ["1W", "1M"]}
    rows: list[dict[str, Any]] = []
    for i, t in enumerate(pd.to_datetime(daily.datetime).dt.normalize()):
        row: dict[str, Any] = {"datetime": t}
        states: dict[str, dict[str, Any]] = {}
        for tf in ["1W", "1M"]:
            state = context_state_at(t, i, daily, frames[tf], pivots_map.get((tf, "balanced"), pd.DataFrame()), tf)
            states[tf] = state
            prefix = tf.lower()
            for key, value in state.items():
                row[f"{prefix}_{key}"] = value
            tf_zones = zones_by_tf[tf]
            life = balanced_life.get(tf, pd.DataFrame())
            for kind in ["ob_candidate", "fvg"]:
                for direction in ["bullish", "bearish"]:
                    stat = zone_stats_at(t, fnum(daily.close.iloc[i]), fnum(daily.atr14.iloc[i]), tf_zones, life, direction, kind)
                    for key in ["distance_atr", "active_count", "fresh_count", "nearest_object_id"]:
                        row[f"{prefix}_{direction}_{kind}_{key}"] = stat[key]
        wt, mt = states["1W"]["structure_trend"], states["1M"]["structure_trend"]
        wb, mb = states["1W"]["bias_direction"], states["1M"]["bias_direction"]
        row["monthly_weekly_trend_relation"] = "aligned" if wt == mt and wt in {"up", "down"} else "counter" if wt in {"up", "down"} and mt in {"up", "down"} else "unknown"
        row["monthly_weekly_bias_relation"] = "aligned" if wb == mb and wb in {"up", "down"} else "divergent" if wb in {"up", "down"} and mb in {"up", "down"} else "unknown"
        rows.append(row)
    return pd.DataFrame(rows)


def object_features(row: pd.Series, daily: pd.DataFrame, contexts: pd.DataFrame, sequence: pd.DataFrame, post: bool = False) -> dict[str, Any]:
    dates = pd.to_datetime(daily.datetime).dt.normalize()
    target_date = pd.Timestamp(row.touch_at if post else row.available_at).normalize() if (post and pd.notna(row.get("touch_at"))) else pd.Timestamp(row.available_at).normalize()
    pos = int(np.searchsorted(dates.to_numpy(dtype="datetime64[ns]"), np.datetime64(target_date, "ns"), side="right") - 1)
    if pos < 0 or pos >= len(daily):
        return {}
    p = daily.iloc[pos]
    close, atr = fnum(p.close), fnum(p.atr14)
    lo, hi = fnum(row.zone_low), fnum(row.zone_high)
    prefix = "post_" if post else ""
    width = hi - lo if math.isfinite(lo) and math.isfinite(hi) else np.nan
    d = 0.0 if math.isfinite(lo) and math.isfinite(hi) and lo <= close <= hi else min(abs(lo - close), abs(hi - close)) if math.isfinite(lo) and math.isfinite(hi) else np.nan
    values: dict[str, Any] = {
        f"{prefix}feature_bar_idx": pos,
        f"{prefix}distance_atr": d / atr if math.isfinite(d) and atr > 0 else np.nan,
        f"{prefix}width_atr": width / atr if math.isfinite(width) and atr > 0 else np.nan,
        f"{prefix}mid_distance_atr": abs(close - (lo + hi) / 2.0) / atr if math.isfinite(lo) and math.isfinite(hi) and atr > 0 else np.nan,
        f"{prefix}return_5": fnum(p.return_5), f"{prefix}return_20": fnum(p.return_20),
        f"{prefix}atr_pct": atr / close if close and atr > 0 else np.nan,
        f"{prefix}volume_z20": fnum(p.volume_z20), f"{prefix}sma_gap20": fnum(p.sma_gap20), f"{prefix}sma_gap50": fnum(p.sma_gap50),
        f"{prefix}bar_range_atr": fnum(p.range) / atr if atr > 0 else np.nan,
        f"{prefix}close_location": fnum(p.close_location),
        "object_is_ob": float(str(row.object_type) == "ob_candidate"),
        "evidence_confirmed": float(str(row.evidence_state) == "confirmed"),
        "age_days": (target_date - pd.Timestamp(row.origin_time).normalize()).days if pd.notna(row.origin_time) else np.nan,
    }
    if post:
        values["post_intrusion_atr"] = (max(0.0, hi - fnum(p.low)) if row.direction == "bullish" else max(0.0, fnum(p.high) - lo)) / atr if atr > 0 else np.nan
    ctx = contexts[contexts.datetime <= target_date]
    if not len(ctx):
        return values
    c = ctx.iloc[-1]
    rel = "post_" if post else ""
    direction = str(row.direction)
    for tf in ["1w", "1m"]:
        trend = str(c.get(f"{tf}_structure_trend", "unknown"))
        bias = str(c.get(f"{tf}_bias_direction", "neutral"))
        integ = str(c.get(f"{tf}_structure_integrity", "unknown"))
        posn = str(c.get(f"{tf}_position", "unknown"))
        target_dist = fnum(c.get(f"{tf}_bias_target_distance_atr"))
        values.update({
            f"{rel}{tf}_trend_up": float(trend == "up"), f"{rel}{tf}_trend_down": float(trend == "down"), f"{rel}{tf}_trend_mixed": float(trend == "mixed"), f"{rel}{tf}_trend_unknown": float(trend == "unknown"),
            f"{rel}{tf}_bias_up": float(bias == "up"), f"{rel}{tf}_bias_down": float(bias == "down"), f"{rel}{tf}_bias_neutral": float(bias not in {"up", "down"}),
            f"{rel}{tf}_bias_invalidated": float(str(c.get(f"{tf}_bias_status", "")) == "invalidated"),
            f"{rel}{tf}_integrity_intact": float(integ == "intact"), f"{rel}{tf}_integrity_broken": float(integ == "broken"), f"{rel}{tf}_integrity_unknown": float(integ == "unknown"),
            f"{rel}{tf}_position_premium": float(posn == "premium"), f"{rel}{tf}_position_discount": float(posn == "discount"), f"{rel}{tf}_position_unknown": float(posn not in {"premium", "discount", "out_of_range"}),
            f"{rel}{tf}_target_distance_atr": target_dist,
            f"{rel}{tf}_trend_aligned": float((trend == "up" and direction == "bullish") or (trend == "down" and direction == "bearish")),
            f"{rel}{tf}_bias_aligned": float((bias == "up" and direction == "bullish") or (bias == "down" and direction == "bearish")),
        })
        for kind in ["ob_candidate", "fvg"]:
            same = "bullish" if direction == "bullish" else "bearish"
            opposite = "bearish" if direction == "bullish" else "bullish"
            for side, side_dir in [("same", same), ("opposite", opposite)]:
                for metric in ["distance_atr", "active_count", "fresh_count"]:
                    values[f"{rel}{tf}_{side}_{kind}_{metric}"] = fnum(c.get(f"{tf}_{side_dir}_{kind}_{metric}"))
    if not sequence.empty:
        q = sequence[sequence.object_id.eq(row.object_id)]
        if len(q):
            sr = q.iloc[-1]
            for col in ["sequence_sweep_before", "sequence_displacement_before", "sequence_fvg_companion_before", "sequence_structure_before", "sequence_order_valid", "sequence_complete_local", "sequence_event_count"]:
                values[col if not post else f"post_{col}"] = fnum(sr.get(col))
    return values


def build_sequence_links(objects: pd.DataFrame, events: pd.DataFrame, frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Link events by time, parent displacement, and formation interval.

    A nearby event is not enough: the link has to respect the direction,
    timeframe, availability clock, and the order sweep -> displacement ->
    structure confirmation -> zone availability.
    """
    if objects.empty:
        return pd.DataFrame()
    balanced = objects[(objects.sensitivity == "balanced") & objects.timeframe.isin(EVAL_TIMEFRAMES)].copy()
    balanced = balanced[balanced.object_type.isin(set(EVAL_TYPES) | {"displacement"})].copy()
    ev = events[(events.sensitivity == "balanced") & events.timeframe.isin(EVAL_TIMEFRAMES)].copy() if len(events) else pd.DataFrame()
    if len(ev):
        ev["event_time"] = pd.to_datetime(ev.event_time).dt.normalize()
        ev["available_at"] = pd.to_datetime(ev.available_at).dt.normalize()
    rows = []
    for row in balanced[balanced.object_type.isin(EVAL_TYPES)].itertuples(index=False):
        avail = pd.Timestamp(row.available_at).normalize()
        same = balanced[(balanced.timeframe == row.timeframe) & (balanced.direction == row.direction) & (balanced.object_type == "displacement") & (pd.to_datetime(balanced.available_at).dt.normalize() <= avail)].copy()
        if len(same):
            distance = (pd.to_numeric(same.origin_idx, errors="coerce") - float(row.origin_idx)).abs()
            same = same.assign(_distance=distance)
            same = same[(same.origin_idx >= int(row.origin_idx) - 1) & (same.origin_idx <= int(row.confirmed_idx) + 3)]
            displacement = same.sort_values(["_distance", "available_at", "object_id"]).iloc[0] if len(same) else None
        else:
            displacement = None
        disp_id = str(displacement.object_id) if displacement is not None else ""
        disp_origin = pd.Timestamp(displacement.origin_time).normalize() if displacement is not None else pd.NaT
        disp_confirm = pd.Timestamp(displacement.confirmed_at).normalize() if displacement is not None else pd.NaT
        fvg = balanced[(balanced.timeframe == row.timeframe) & (balanced.direction == row.direction) & (balanced.object_type == "fvg") & (balanced.object_id != row.object_id) & (pd.to_datetime(balanced.available_at).dt.normalize() <= avail)].copy()
        if displacement is not None and len(fvg):
            fvg = fvg[(fvg.origin_idx >= int(displacement.origin_idx) - 1) & (fvg.origin_idx <= int(displacement.confirmed_idx) + 1)]
        companion = fvg.sort_values(["available_at", "object_id"]).iloc[0] if len(fvg) else None
        fvg_id = str(companion.object_id) if companion is not None else ""
        fvg_at = pd.Timestamp(companion.available_at).normalize() if companion is not None else pd.NaT
        struct = pd.DataFrame()
        if len(ev) and disp_id:
            struct = ev[(ev.timeframe == row.timeframe) & (ev.direction == row.direction) & (ev.object_id.astype(str) == disp_id) & ev.event_type.isin(["mss", "bos"]) & (ev.available_at <= avail)]
        structure = struct.sort_values(["available_at", "event_id"]).iloc[0] if len(struct) else None
        struct_id = str(structure.event_id) if structure is not None else ""
        struct_at = pd.Timestamp(structure.available_at).normalize() if structure is not None else pd.NaT
        sweep = pd.DataFrame()
        if len(ev) and displacement is not None:
            sweep = ev[(ev.timeframe == row.timeframe) & (ev.direction == row.direction) & ev.event_type.isin(["sweep_same_bar", "sweep_delayed_reclaim"]) & (ev.available_at <= avail) & (ev.event_time <= disp_origin)]
            if len(sweep):
                max_days = 10 if row.timeframe == "1D" else 35
                sweep = sweep[(disp_origin - sweep.event_time).dt.days.between(0, max_days)]
        sweep_row = sweep.sort_values(["available_at", "event_id"]).iloc[-1] if len(sweep) else None
        sweep_id = str(sweep_row.event_id) if sweep_row is not None else ""
        sweep_at = pd.Timestamp(sweep_row.available_at).normalize() if sweep_row is not None else pd.NaT
        order_valid = bool(pd.notna(sweep_at) and pd.notna(disp_origin) and sweep_at <= disp_origin and pd.notna(struct_at) and struct_at >= disp_confirm and struct_at <= avail)
        has_disp = bool(displacement is not None)
        has_fvg = bool(companion is not None and (not pd.notna(disp_confirm) or fvg_at <= avail))
        has_struct = bool(structure is not None)
        has_sweep = bool(sweep_row is not None)
        complete = bool(has_sweep and has_disp and has_fvg and has_struct and order_valid)
        state = "complete_local_sequence" if complete else "sweep_displacement_structure" if has_sweep and has_disp and has_struct else "sweep_displacement" if has_sweep and has_disp else "displacement_only" if has_disp else "none"
        rows.append({
            "object_id": row.object_id, "timeframe": row.timeframe, "direction": row.direction, "object_type": row.object_type,
            "sequence_state": state, "sequence_sweep_before": float(has_sweep), "sequence_displacement_before": float(has_disp),
            "sequence_fvg_companion_before": float(has_fvg), "sequence_structure_before": float(has_struct), "sequence_order_valid": float(order_valid),
            "sequence_complete_local": float(complete), "sequence_event_count": int(sum([has_sweep, has_disp, has_fvg, has_struct])),
            "sweep_event_id": sweep_id, "sweep_at": sweep_at, "displacement_id": disp_id, "displacement_origin_at": disp_origin,
            "displacement_confirmed_at": disp_confirm, "companion_fvg_id": fvg_id, "companion_fvg_at": fvg_at, "structure_event_id": struct_id, "structure_at": struct_at,
            "sequence_available_at": avail,
        })
    return pd.DataFrame(rows)


def _span_has_gap(start: Any, end: Any, gap_dates: set[pd.Timestamp]) -> bool:
    if pd.isna(start) or pd.isna(end):
        return False
    a, b = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    return any(a <= d <= b for d in gap_dates)


def build_setup_catalog(objects: pd.DataFrame, events: pd.DataFrame, daily: pd.DataFrame, contexts: pd.DataFrame, frames: dict[str, pd.DataFrame], gap_dates: set[pd.Timestamp]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    """Make stage states and fixed-target labels for balanced daily/weekly zones."""
    balanced = objects[(objects.sensitivity == "balanced") & objects.timeframe.isin(EVAL_TIMEFRAMES) & objects.object_type.isin(EVAL_TYPES)].copy()
    balanced = balanced[~balanced.gap_contaminated.astype(bool)].copy()
    life_map: dict[str, pd.DataFrame] = {}
    for tf in EVAL_TIMEFRAMES:
        z = balanced[balanced.timeframe == tf]
        life_map[tf] = zone_lifecycle_fast(z, daily)
    sequence = build_sequence_links(objects, events, frames)
    records = []
    for _, row in balanced.iterrows():
        r = row.copy()
        seq = sequence[sequence.object_id.eq(row.object_id)] if len(sequence) else pd.DataFrame()
        sr = seq.iloc[0] if len(seq) else pd.Series(dtype=object)
        primitive = True
        adapted = bool(str(row.object_type) == "fvg" or (str(row.object_type) == "ob_candidate" and str(row.evidence_state) == "confirmed"))
        # The OB articles require LTF imbalance/MSS in the source setup.  Daily
        # and weekly OHLC cannot verify those conditions, so they remain unknown.
        original_state = "not_required_for_basic_fvg" if row.object_type == "fvg" else "unknown_data_lower_timeframe"
        original_confirmed = bool(row.object_type == "fvg")
        ctx = contexts[contexts.datetime <= pd.Timestamp(row.available_at).normalize()]
        c = ctx.iloc[-1] if len(ctx) else pd.Series(dtype=object)
        direction = str(row.direction)
        # Context columns use the canonical timeframe prefixes (1w/1m).  Do
        # not silently fall back to a never-populated human-readable alias:
        # that would make every completed setup impossible by construction.
        target_known = bool(str(c.get("1w_bias_status", "")) == "active" and math.isfinite(fnum(c.get("1w_bias_target_price"))))
        if not target_known:
            target_known = bool(math.isfinite(fnum(c.get("1w_bias_target_up_distance_atr"))) if direction == "bullish" else math.isfinite(fnum(c.get("1w_bias_target_down_distance_atr"))))
        location_known = bool(str(c.get("1w_position", "unknown")) != "unknown" or str(c.get("1m_position", "unknown")) != "unknown")
        sequence_complete = bool(fnum(sr.get("sequence_complete_local")) == 1)
        complete = bool(adapted and sequence_complete and target_known and location_known)
        if complete:
            setup_state = "complete_adapted"
        elif adapted:
            setup_state = "adapted_confirmed"
        else:
            setup_state = "primitive"
        feat = object_features(row, daily, contexts, sequence, post=False)
        labels = v4.label_object_v04(row, daily, arrival_horizon=ARRIVAL_HORIZON[row.timeframe], post_touch_horizon=POST_HORIZON, barrier_atr=BARRIER_ATR)
        labels = dict(labels)
        label_gap = bool(_span_has_gap(labels.get("available_at"), labels.get("arrival_label_end_at"), gap_dates) or _span_has_gap(labels.get("touch_at"), labels.get("post_label_end_at"), gap_dates))
        labels["label_gap_contaminated"] = label_gap
        labels["arrival_model_eligible"] = float(bool(labels.get("arrival_complete")) and labels.get("arrival_type") in {"touch", "no_arrival"} and not label_gap)
        labels["post_model_eligible"] = float(labels.get("post_label") in POST_CLASSES and not label_gap)
        record = {**row.to_dict(), **feat, **labels, **sr.to_dict()}
        record.update({
            "primitive_form": 1, "adapted_confirmed": int(adapted), "original_confirmed": int(original_confirmed),
            "original_state": original_state, "location_observed": int(location_known), "target_known": int(target_known),
            "invalidation_known": int(math.isfinite(fnum(row.zone_low)) and math.isfinite(fnum(row.zone_high))),
            "complete_adapted": int(complete), "setup_state": setup_state,
            "stage_prediction_at": pd.Timestamp(row.available_at).normalize(),
            "fixed_target_definition": f"arrival_{ARRIVAL_HORIZON[row.timeframe]}d;post_{POST_HORIZON}d;barrier_{BARRIER_ATR:.1f}ATR_touch_close",
            "post_outcome": labels.get("post_label", "not_applicable"),
        })
        if pd.notna(record.get("touch_at")):
            post_feat = object_features(pd.Series(record), daily, contexts, sequence, post=True)
            record.update(post_feat)
        records.append(record)
    catalog = pd.DataFrame(records)
    if len(catalog):
        catalog = catalog.sort_values(["available_at", "object_id"]).reset_index(drop=True)
    return catalog, sequence, life_map


def context_feature_columns(post: bool = False) -> list[str]:
    p = "post_" if post else ""
    cols = []
    for tf in ["1w", "1m"]:
        cols += [f"{p}{tf}_{x}" for x in ["trend_up", "trend_down", "trend_mixed", "trend_unknown", "bias_up", "bias_down", "bias_neutral", "bias_invalidated", "integrity_intact", "integrity_broken", "integrity_unknown", "position_premium", "position_discount", "position_unknown", "target_distance_atr", "trend_aligned", "bias_aligned"]]
        for kind in ["ob_candidate", "fvg"]:
            for side in ["same", "opposite"]:
                cols += [f"{p}{tf}_{side}_{kind}_{x}" for x in ["distance_atr", "active_count", "fresh_count"]]
    return cols


def model_features(post: bool = False) -> list[str]:
    p = "post_" if post else ""
    base_cols = [f"{p}{x}" for x in ["return_5", "return_20", "atr_pct", "volume_z20", "sma_gap20", "sma_gap50", "distance_atr", "width_atr", "mid_distance_atr", "bar_range_atr", "close_location"]]
    seq = [x if not post else f"post_{x}" for x in ["sequence_sweep_before", "sequence_displacement_before", "sequence_fvg_companion_before", "sequence_structure_before", "sequence_order_valid", "sequence_event_count"]]
    return base_cols + ["object_is_ob", "evidence_confirmed", "adapted_confirmed", "complete_adapted"] + context_feature_columns(post) + seq


def stage_groups(catalog: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "primitive": catalog[catalog.primitive_form.astype(bool)].copy(),
        "adapted_confirmed": catalog[catalog.adapted_confirmed.astype(bool)].copy(),
        "complete_adapted": catalog[catalog.complete_adapted.astype(bool)].copy(),
    }


def _fit_predict(train: pd.DataFrame, test: pd.DataFrame, features: list[str], target: str) -> tuple[np.ndarray, str]:
    cols = [c for c in features if c in train.columns and c in test.columns and pd.to_numeric(train[c], errors="coerce").notna().any()]
    if target == "arrival":
        prior = float(train.y.mean()) if len(train) else np.nan
        if len(train) < MIN_TRAIN or train.y.nunique() < 2 or not cols:
            return np.column_stack([1 - np.full(len(test), prior), np.full(len(test), prior)]), "constant_training_prior"
        pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("logit", LogisticRegression(C=1.0, max_iter=1500, solver="lbfgs"))])
        pipe.fit(train[cols].apply(pd.to_numeric, errors="coerce"), train.y.astype(int))
        return pipe.predict_proba(test[cols].apply(pd.to_numeric, errors="coerce")), "logistic_fixed_C1"
    y = train.y_class.astype(str)
    prior = y.value_counts(normalize=True).reindex(POST_CLASSES, fill_value=0).to_numpy(float)
    if len(train) < MIN_TRAIN or y.nunique() < 2 or not cols:
        return np.tile(prior, (len(test), 1)), "constant_training_prior"
    pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("logit", LogisticRegression(C=1.0, max_iter=1500, solver="lbfgs"))])
    pipe.fit(train[cols].apply(pd.to_numeric, errors="coerce"), y)
    raw = pd.DataFrame(pipe.predict_proba(test[cols].apply(pd.to_numeric, errors="coerce")), columns=pipe.named_steps["logit"].classes_)
    return raw.reindex(columns=POST_CLASSES, fill_value=0).to_numpy(float), "multinomial_logistic_fixed_C1"


def _logloss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(np.asarray(p, float), 1e-7, 1 - 1e-7)
    return float(-np.mean(np.log(p[np.arange(len(y)), y])))


def _folds(dates: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    u = pd.Series(pd.to_datetime(dates).dropna().sort_values().unique())
    if len(u) < 5:
        return []
    qs = sorted(set(pd.Timestamp(x).normalize() for x in u.quantile([0.50, 0.60, 0.70, 0.80, 0.90])))
    return [(qs[i], (qs + [u.max() + pd.Timedelta(days=1)])[i + 1]) for i in range(len(qs))]


def walk_forward_task(payload: tuple[pd.DataFrame, str, str, str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    data, stage, tf, direction, target = payload
    stage_col = {"primitive": "primitive_form", "adapted_confirmed": "adapted_confirmed", "complete_adapted": "complete_adapted"}[stage]
    data = data[data[stage_col].astype(bool)].copy()
    if target == "arrival":
        s = data[(data.timeframe == tf) & (data.direction == direction) & data.arrival_model_eligible.astype(bool)].copy()
        s["y"] = pd.to_numeric(s.arrival_label, errors="coerce")
        s["prediction_at"] = pd.to_datetime(s.available_at).dt.normalize()
        s["label_end_at"] = pd.to_datetime(s.arrival_label_end_at).dt.normalize()
        feature_list = model_features(False)
    else:
        s = data[(data.timeframe == tf) & (data.direction == direction) & data.post_model_eligible.astype(bool)].copy()
        s["y_class"] = s.post_outcome.astype(str)
        s["prediction_at"] = pd.to_datetime(s.touch_at).dt.normalize()
        s["label_end_at"] = pd.to_datetime(s.post_label_end_at).dt.normalize()
        feature_list = model_features(True)
    if len(s) < MIN_TEST:
        return pd.DataFrame(), pd.DataFrame()
    metrics, preds = [], []
    for fold_id, (start, end) in enumerate(_folds(s.prediction_at), start=1):
        train = s[(s.prediction_at < start) & (s.label_end_at < start)].copy()
        test = s[(s.prediction_at >= start) & (s.prediction_at < end)].copy()
        if len(train) < MIN_TRAIN or len(test) < MIN_TEST:
            continue
        if target == "arrival":
            prior = float(train.y.mean())
            pred_map = {"intercept_only": np.column_stack([np.full(len(test), 1 - prior), np.full(len(test), prior)])}
        else:
            prior = train.y_class.value_counts(normalize=True).reindex(POST_CLASSES, fill_value=0).to_numpy(float)
            pred_map = {"intercept_only": np.tile(prior, (len(test), 1))}
        pred_map["geometry_context_sequence"], status = _fit_predict(train, test, feature_list, target)
        for model, p in pred_map.items():
            y = test.y.to_numpy(int) if target == "arrival" else pd.Categorical(test.y_class, categories=POST_CLASSES).codes
            brier = float(np.mean((p[:, 1] - y) ** 2)) if target == "arrival" else float(np.mean(np.sum((p - np.eye(3)[y]) ** 2, axis=1)))
            ll = _logloss(y, p)
            metrics.append({"stage": stage, "target": target, "timeframe": tf, "direction": direction, "fold_id": fold_id, "test_start": start, "test_end": end, "train_n": len(train), "test_n": len(test), "model": model, "model_status": "training_prior" if model == "intercept_only" else status, "brier": brier, "log_loss": ll, "max_train_label_end_at": train.label_end_at.max(), "min_test_prediction_at": test.prediction_at.min(), "max_test_prediction_at": test.prediction_at.max()})
            for j, (_, r) in enumerate(test.iterrows()):
                out = {"object_id": r.object_id, "stage": stage, "target": target, "timeframe": tf, "direction": direction, "fold_id": fold_id, "prediction_at": r.prediction_at, "label_end_at": r.label_end_at, "model": model, "y": float(r.y) if target == "arrival" else np.nan, "y_class": r.y_class if target != "arrival" else ""}
                if target == "arrival":
                    out["p_positive"] = float(p[j, 1])
                else:
                    for k, cls in enumerate(POST_CLASSES):
                        out[f"p_{cls}"] = float(p[j, k])
                preds.append(out)
    return pd.DataFrame(metrics), pd.DataFrame(preds)


def walk_forward(catalog: pd.DataFrame, n_jobs: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    payloads = [(catalog, stage, tf, direction, target) for stage in ["primitive", "adapted_confirmed", "complete_adapted"] for tf in EVAL_TIMEFRAMES for direction in ["bullish", "bearish"] for target in ["arrival", "post"]]
    metrics, preds = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, n_jobs)) as pool:
        futures = [pool.submit(walk_forward_task, p) for p in payloads]
        for fut in futures:
            m, p = fut.result()
            if len(m): metrics.append(m)
            if len(p): preds.append(p)
    return (pd.concat(metrics, ignore_index=True) if metrics else pd.DataFrame(), pd.concat(preds, ignore_index=True) if preds else pd.DataFrame())


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    out = metrics.groupby(["stage", "target", "timeframe", "direction", "model"], dropna=False).apply(
        lambda g: pd.Series({"folds": int(g.fold_id.nunique()), "n_test": int(g.test_n.sum()), "brier": float(np.average(g.brier, weights=g.test_n)), "log_loss": float(np.average(g.log_loss, weights=g.test_n)), "brier_fold_sd": float(g.brier.std(ddof=1)) if len(g) > 1 else np.nan})
    ).reset_index()
    priors = out[out.model == "intercept_only"][["stage", "target", "timeframe", "direction", "brier", "log_loss"]].rename(columns={"brier": "intercept_brier", "log_loss": "intercept_log_loss"})
    return out.merge(priors, on=["stage", "target", "timeframe", "direction"], how="left").assign(delta_brier_vs_intercept=lambda x: x.brier - x.intercept_brier, delta_log_loss_vs_intercept=lambda x: x.log_loss - x.intercept_log_loss)


def calibration_table(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive out-of-fold probability bins.

    Arrival is binary.  Post-touch is a three-way competing-path outcome, so
    each class is calibrated one-vs-rest.  These are descriptive OOF bins,
    not a second tuning loop and not a claim of statistically independent
    observations when labels overlap in time.
    """
    if predictions.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    keys = ["stage", "target", "timeframe", "direction", "model"]
    edges = [-0.001, 0.2, 0.4, 0.6, 0.8, 1.001]
    for key, g in predictions.groupby(keys, dropna=False):
        group_values = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        if str(group_values["target"]) == "arrival":
            class_items = [("positive", "p_positive", pd.to_numeric(g["y"], errors="coerce"))]
        else:
            class_items = []
            for cls in POST_CLASSES:
                class_items.append((cls, f"p_{cls}", g["y_class"].astype(str).eq(cls).astype(float)))
        for cls, pcol, y in class_items:
            if pcol not in g.columns:
                continue
            p = pd.to_numeric(g[pcol], errors="coerce")
            valid = p.notna() & y.notna()
            if not valid.any():
                continue
            gg = pd.DataFrame({"p": p[valid], "y": y[valid]})
            gg["probability_bin"] = pd.cut(gg["p"], bins=edges, include_lowest=True)
            for bin_name, z in gg.groupby("probability_bin", observed=True):
                rows.append({**group_values, "class": cls, "probability_bin": str(bin_name), "n": int(len(z)), "mean_predicted": float(z.p.mean()), "observed_rate": float(z.y.mean())})
    return pd.DataFrame(rows)


def stage_summary(catalog: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for stage, data in stage_groups(catalog).items():
        for tf in EVAL_TIMEFRAMES:
            for direction in ["bullish", "bearish"]:
                g = data[(data.timeframe == tf) & (data.direction == direction)]
                arrival = g[g.arrival_model_eligible.astype(bool)]
                post = g[g.post_model_eligible.astype(bool)]
                rows.append({
                    "stage": stage, "timeframe": tf, "direction": direction, "n_objects": len(g),
                    "arrival_eligible_n": len(arrival), "arrival_touch_n": int((arrival.arrival_type == "touch").sum()), "arrival_touch_rate": float((arrival.arrival_type == "touch").mean()) if len(arrival) else np.nan,
                    "post_eligible_n": len(post), "post_favorable_n": int((post.post_outcome == "favorable_first").sum()), "post_adverse_n": int((post.post_outcome == "adverse_first").sum()), "post_neither_n": int((post.post_outcome == "neither_at_horizon").sum()), "post_favorable_rate": float((post.post_outcome == "favorable_first").mean()) if len(post) else np.nan,
                    "post_median_bars_favorable": float(pd.to_numeric(post.loc[post.post_outcome == "favorable_first", "bars_to_favorable"], errors="coerce").median()) if (post.post_outcome == "favorable_first").any() else np.nan,
                })
    return pd.DataFrame(rows)


def article_source_index(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    out = {}
    for r in df.to_dict("records"):
        key = str(r.get("canonical_url") or r.get("url") or "")
        out[key] = r
    return out


def build_article_ledger(objects: pd.DataFrame, events: pd.DataFrame, catalog: pd.DataFrame, source_coverage: Path, source_manifest: Path, output_root: Path) -> pd.DataFrame:
    if source_coverage.exists():
        source_rows = json.loads(source_coverage.read_text(encoding="utf-8"))
    else:
        source_rows = []
    local = article_source_index(source_manifest)
    direct = {
        24: ("cisd;mss", "nq_ict_detection_20260911_v03.py::cisd_events;structure_events", "daily_weekly_core_events"),
        41: ("cisd", "nq_ict_detection_20260911_v03.py::cisd_events", "daily_weekly_core_events"),
        61: ("liquidity_breach;liquidity_sweep", "nq_ict_detection_20260911_v03.py::sweep_events", "daily_weekly_core_events"),
        75: ("displacement", "nq_ict_detection_20260911_v03.py::displacement_events", "daily_weekly_core_objects"),
        81: ("swing_low_3bar", "nq_ict_detection_20260911_v03.py::make_pivots", "daily_weekly_core_objects"),
        82: ("swing_high_3bar", "nq_ict_detection_20260911_v03.py::make_pivots", "daily_weekly_core_objects"),
        111: ("mss", "nq_ict_detection_20260911_v03.py::structure_events", "daily_weekly_core_events"),
        114: ("fvg_3bar", "nq_ict_detection_20260911_v03.py::fvg_objects", "daily_weekly_core_objects"),
    }
    shared = {100: ("fvg_3bar", "nq_ict_detection_20260911_v03.py::fvg_objects", "shared_fvg_rule")}
    adapted = {87: ("ob_candidate_dw", "nq_ict_detection_20260911_v03.py::ob_objects", "daily_weekly_candidate_only"), 88: ("ob_candidate_dw", "nq_ict_detection_20260911_v03.py::ob_objects", "daily_weekly_candidate_only"), 115: ("ob_candidate_dw", "nq_ict_detection_20260911_v03.py::ob_objects", "daily_weekly_candidate_only")}
    rows = []
    for src in source_rows:
        number = int(src["article_number"])
        rule_id, evidence_path, verified_scope = direct.get(number, shared.get(number, adapted.get(number, ("", "", ""))))
        if number in direct:
            verified_status = "implemented_basic_shape"
        elif number in shared:
            verified_status = "shared_basic_shape"
        elif number in adapted:
            verified_status = "adapted_candidate_only"
        elif src.get("implementation_status") == "requires_extra_data":
            verified_status = "requires_extra_data_unimplemented"
        elif src.get("implementation_status") == "reference_only":
            verified_status = "reference_only"
        else:
            verified_status = "not_implemented"
        article_ids = str(number)
        obj_count = 0
        event_count = 0
        if rule_id:
            rule_parts = rule_id.split(";")
            if any(x in rule_parts for x in ["fvg_3bar", "ob_candidate_dw"]):
                obj_count = int(objects[(objects.sensitivity == "balanced") & objects.rule_id.eq("fvg_3bar" if "fvg_3bar" in rule_parts else "ob_candidate_dw")].shape[0])
            if any(x in rule_parts for x in ["displacement", "swing_low_3bar", "swing_high_3bar"]):
                rule = "displacement" if "displacement" in rule_parts else "swing_low_3bar" if "swing_low_3bar" in rule_parts else "swing_high_3bar"
                obj_count = int(objects[(objects.sensitivity == "balanced") & objects.rule_id.eq(rule)].shape[0])
            if any(x in rule_parts for x in ["cisd", "mss", "liquidity_breach", "liquidity_sweep"]):
                event_count = int(events[(events.sensitivity == "balanced") & events.rule_id.isin(rule_parts)].shape[0])
        if number in {87, 88, 115}:
            subset = catalog[catalog.object_type.eq("ob_candidate")]
            obj_count = int(len(subset))
            event_count = 0
            missing = "lower_timeframe_imbalance;lower_timeframe_MSS;original_entry_confirmation;strict_source_setup_not_verified"
            original_count = 0
            adapted_count = int(subset.adapted_confirmed.astype(bool).sum())
            complete_count = int(subset.complete_adapted.astype(bool).sum())
            complete_scope = "daily_weekly_adaptation_only"
        elif number in {100, 114}:
            subset = catalog[catalog.object_type.eq("fvg")]
            original_count = int(len(subset))
            complete_count = int(subset.complete_adapted.sum())
            adapted_count = int(subset.adapted_confirmed.astype(bool).sum())
            missing = "article_specific_target_and_execution_flow_not_a_standalone_detector"
            complete_scope = "daily_weekly_adaptation_only"
        else:
            original_count = int(event_count or obj_count)
            adapted_count = np.nan
            complete_count = np.nan
            missing = str(src.get("limitation", "not implemented"))
            complete_scope = "not_a_standalone_setup"
        source = local.get(str(src.get("source_url", "")), {})
        if not source:
            source = local.get(str(src.get("source_url", "")).replace("http://", "https://"), {})
        rows.append({
            "article_number": number, "title": src.get("title", ""), "source_url": src.get("source_url", ""),
            "source_table_status": src.get("implementation_status", ""), "verified_status": verified_status,
            "rule_ids": rule_id, "verified_scope": verified_scope, "implementation_evidence": evidence_path,
            "source_local_manifest": int(bool(source)), "source_html_sha256": source.get("html_sha256", ""), "source_text_chars": source.get("text_chars", np.nan), "source_image_count": source.get("image_count", np.nan),
            "basic_evidence_count_balanced": obj_count, "event_evidence_count_balanced": event_count, "adapted_confirmed_count": adapted_count,
            "original_confirmed_count": original_count, "complete_adapted_count": complete_count, "complete_stage_scope": complete_scope,
            "missing_or_unknown_conditions": missing,
            "image_semantic_evidence": "not_independent_blind_label;local_image_presence_only" if source else "source_not_joined",
            "evidence_artifact": str((output_root / "objects_v08.csv").resolve()) if obj_count else str((output_root / "events_v08.csv").resolve()) if event_count else "",
        })
    return pd.DataFrame(rows).sort_values("article_number").reset_index(drop=True)


def validation_tests(daily: pd.DataFrame, objects: pd.DataFrame, events: pd.DataFrame, catalog: pd.DataFrame, contexts: pd.DataFrame, ledger: pd.DataFrame, metrics: pd.DataFrame, gap_dates: set[pd.Timestamp]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    def add(name: str, passed: bool, observed: Any, expected: Any, severity: str = "error") -> None:
        rows.append({"check": name, "passed": bool(passed), "observed": observed, "expected": expected, "severity": severity})
    required = ["datetime", "open", "high", "low", "close", "volume"]
    add("input_required_nonnull", int(daily[required].isna().sum().sum()) == 0, int(daily[required].isna().sum().sum()), 0)
    add("input_unique_monotonic", int(daily.datetime.duplicated().sum()) == 0 and bool(daily.datetime.is_monotonic_increasing), f"dups={int(daily.datetime.duplicated().sum())};monotonic={daily.datetime.is_monotonic_increasing}", "dups=0;monotonic=True")
    add("article_ledger_94_rows", len(ledger) == 94 and ledger.article_number.nunique() == 94, int(len(ledger)), 94)
    add("article_rule_evidence_paths", bool((ledger[(ledger.verified_status.isin(["implemented_basic_shape", "shared_basic_shape", "adapted_candidate_only"]))].implementation_evidence.astype(str).str.len() > 0).all()), int((ledger.implementation_evidence.astype(str).str.len() > 0).sum()), "all executable rows have path")
    add("primitive_adapted_monotone", bool((catalog.adapted_confirmed.astype(int) <= catalog.primitive_form.astype(int)).all()), int((catalog.adapted_confirmed.astype(int) > catalog.primitive_form.astype(int)).sum()), 0)
    add("complete_adapted_monotone", bool((catalog.complete_adapted.astype(int) <= catalog.adapted_confirmed.astype(int)).all()), int((catalog.complete_adapted.astype(int) > catalog.adapted_confirmed.astype(int)).sum()), 0)
    add("original_ob_unknown_is_explicit", bool((catalog.loc[catalog.object_type.eq("ob_candidate"), "original_state"] == "unknown_data_lower_timeframe").all()), str(catalog.loc[catalog.object_type.eq("ob_candidate"), "original_state"].value_counts().to_dict()), "unknown_data_lower_timeframe")
    add("sequence_events_not_after_prediction", bool((pd.to_datetime(catalog.sequence_available_at, errors="coerce") <= pd.to_datetime(catalog.available_at, errors="coerce")).fillna(True).all()), int((pd.to_datetime(catalog.sequence_available_at, errors="coerce") > pd.to_datetime(catalog.available_at, errors="coerce")).sum()), 0)
    add("arrival_clock_is_formation", bool((pd.to_datetime(catalog.stage_prediction_at) == pd.to_datetime(catalog.available_at).dt.normalize()).all()), 0, 0)
    post = catalog[catalog.post_model_eligible.astype(bool)]
    add("post_clock_is_touch_close", bool((pd.to_datetime(post.touch_at).dt.normalize() == pd.to_datetime(post.post_prediction_at).dt.normalize()).all()) if len(post) else True, int(len(post)), "all post predictions at touch")
    eligible_gap = catalog[(catalog.arrival_model_eligible.astype(bool) | catalog.post_model_eligible.astype(bool)) & catalog.label_gap_contaminated.astype(bool)]
    add("eligible_windows_clean", len(eligible_gap) == 0, int(len(eligible_gap)), 0)
    add("fixed_target_definition", catalog.fixed_target_definition.nunique() == 2, catalog.fixed_target_definition.unique().tolist(), "1D/1W horizon explicit")
    if len(metrics):
        add("walk_forward_train_before_test", bool((pd.to_datetime(metrics.max_train_label_end_at) < pd.to_datetime(metrics.test_start)).all()), int((pd.to_datetime(metrics.max_train_label_end_at) >= pd.to_datetime(metrics.test_start)).sum()), 0)
    else:
        add("walk_forward_has_metrics", False, 0, ">0", "warning")
    add("context_daily_unique", int(contexts.datetime.duplicated().sum()) == 0 and len(contexts) == len(daily), f"rows={len(contexts)};dups={int(contexts.datetime.duplicated().sum())}", f"rows={len(daily)};dups=0")
    add("event_ids_unique", int(events.event_id.duplicated().sum()) == 0, int(events.event_id.duplicated().sum()), 0)
    return pd.DataFrame(rows)


def prefix_invariance(daily: pd.DataFrame, full_objects: pd.DataFrame, full_events: pd.DataFrame, cutoffs: list[str]) -> pd.DataFrame:
    """Compare detector IDs visible by a cutoff with a prefix-only rerun."""
    records = []
    for cut in cutoffs:
        cutoff = pd.Timestamp(cut).normalize()
        prefix_daily = daily[daily.datetime <= cutoff].copy().reset_index(drop=True)
        if len(prefix_daily) < 100:
            continue
        p_week = period_bars(prefix_daily, "W-FRI")
        p_month = period_bars(prefix_daily, "M")
        p_frames = {"1D": prefix_daily, "1W": p_week[p_week.certified_calendar].reset_index(drop=True), "1M": p_month[p_month.certified_calendar].reset_index(drop=True)}
        p_obj, p_ev, _, _ = run_lanes(p_frames, n_jobs=1, backend="serial")
        f_obj = full_objects[(full_objects.available_at <= cutoff) & full_objects.sensitivity.eq("balanced")]
        f_ev = full_events[(full_events.available_at <= cutoff) & full_events.sensitivity.eq("balanced")]
        p_obj = p_obj[p_obj.sensitivity.eq("balanced")] if len(p_obj) else p_obj
        p_ev = p_ev[p_ev.sensitivity.eq("balanced")] if len(p_ev) else p_ev
        f_ids, p_ids = set(f_obj.object_id.astype(str)), set(p_obj.object_id.astype(str))
        mutable_types = {"liquidity_recovery_pending", "unreclaimed_at_horizon"}
        f_ev = f_ev[~f_ev.event_type.astype(str).isin(mutable_types)]
        p_ev = p_ev[~p_ev.event_type.astype(str).isin(mutable_types)] if len(p_ev) else p_ev
        fe, pe = set(f_ev.event_id.astype(str)), set(p_ev.event_id.astype(str))
        records.append({"cutoff": cutoff, "full_object_visible_n": len(f_ids), "prefix_object_n": len(p_ids), "object_missing_in_prefix": len(f_ids - p_ids), "object_extra_in_prefix": len(p_ids - f_ids), "full_event_visible_n": len(fe), "prefix_event_n": len(pe), "event_missing_in_prefix": len(fe - pe), "event_extra_in_prefix": len(pe - fe), "passed": not (f_ids - p_ids or p_ids - f_ids or fe - pe or pe - fe)})
    return pd.DataFrame(records)


def gpu_probe() -> dict[str, Any]:
    info = {"python": platform.python_version(), "platform": platform.platform(), "cpu_count": os.cpu_count(), "gpu_library": "not_loaded", "cuda_available": False, "gpu_used": False, "reason": "scikit-learn logistic models are CPU-path in this study"}
    try:
        import torch  # type: ignore
        info.update({"gpu_library": "torch", "cuda_available": bool(torch.cuda.is_available()), "device_count": int(torch.cuda.device_count())})
    except Exception as exc:
        info["gpu_probe_error"] = str(exc)
    return info


def report_text(manifest: dict[str, Any], ledger: pd.DataFrame, stage: pd.DataFrame, summary: pd.DataFrame, controls: pd.DataFrame, prefix: pd.DataFrame, calibration: pd.DataFrame | None = None) -> str:
    status_counts = ledger.groupby("verified_status").size().rename("articles").reset_index().to_markdown(index=False) if len(ledger) else ""
    stage_table = stage.to_markdown(index=False) if len(stage) else ""
    metric_table = summary.to_markdown(index=False) if len(summary) else ""
    prefix_table = prefix.to_markdown(index=False) if len(prefix) else ""
    lines = [
        "# NKD=F ICT検知・評価 v0.8（台帳・セットアップ段階・同一目標版）", "",
        "## 結論", "",
        "94記事の台帳を実行可能なルールと証拠へ接続し、基本形・日足/週足適応・完成した局所セットアップ・原記事の下位足条件を分離した。したがって、前版の『全ICT要素を足しても改善しない』という結論は、このv0.8の完成セットアップの有効性を直接否定するものではない。", "",
        f"入力は{manifest['data_window']['start']}〜{manifest['data_window']['end']}、{manifest['data_window']['rows']:,}日足（実データ約{manifest['coverage']['actual_years']:.2f}年）。30年要求に対し、保存スナップショットの開始は2000-09-18である。", "",
        "## 94記事の実装確認", "", status_counts, "",
        "`daily_weekly_adaptation`という台帳上の分類だけでは実装済みとしなかった。実コードへの接続、必要データ、証拠ファイル、確認件数を`article_ledger_v08.csv`へ保存した。OB記事は原記事の下位足不均衡・下位足MSSを確認できないため、日足/週足の候補・適応確認に留め、原記事確認済みとはしていない。", "",
        "## セットアップ段階", "", stage_table, "",
        "`primitive`は基本形が存在する全候補、`adapted_confirmed`は日足/週足OHLCで確認できる条件がそろった候補、`complete_adapted`はSweep→Displacement→構造確認→関連FVG、上位足の場所、DOL目標、無効化境界が予測時点までにそろった候補である。完成状態の条件は原記事の全分足手順ではなく、明記した日足/週足派生定義である。表の3段階は入れ子の母集団であり、排他的な最終状態の件数はmanifestの`setup_state_counts`で別掲する。", "",
        "## 同じ予測時点・同じ目標でのウォークフォワード", "", metric_table, "",
        "到達評価は形成・確認後から日足20本（週足候補は日足60本）以内のゾーン到達、接触後評価は接触足終値の後10日足以内に、接触帯の外側±1.0 ATRへ有利側・不利側のどちらが先に到達したかを使う。接触足内の順序不明は未知として扱う。モデル比較では学習ラベル終了日がテスト開始日より前の例だけを使った。OOF確率のビン別予測値と実測率は`walk_forward_calibration_v08.csv`へ保存した（行数: %s）。" % (len(calibration) if calibration is not None else 0), "", 
        "## 接頭辞不変性", "", prefix_table, "",
        "## 読み方と限界", "",
        "- 台帳の存在、基本形の検知、記事本文・図の意味再現、将来価格の予測力は別の検査である。今回の台帳証拠は、94記事の売買優位性を証明しない。",
        "- 原記事の下位足MSS、分足FVG、セッション時刻、同一足内の高安順序、SMTなどは、日足・週足単一系列では確認できない。これらは`unknown_data`または`requires_extra_data`として残した。",
        "- 完成適応セットアップは、厳しい条件により標本が少なくなる可能性がある。標本不足時はモデルを出さず、確率を保留する。",
        "- NKD=FはYahooの連続先物代理系列で、限月接続・出来高の性質・欠損期間を完全な取引所データと同一視しない。",
        "- GPUは検知・時系列ラベル・小規模ロジスティック回帰に必須でないため使用せず、レーン検知は設定したCPU並列で実行した。GPU利用可能性はmanifestに記録した。",
        "",
        "## 成果物", "",
        "- `article_ledger_v08.csv`: 94記事ごとの実装・不足条件・証拠",
        "- `setup_catalog_v08.csv`: 4段階の状態、順序リンク、同一目標ラベル",
        "- `walk_forward_metric_summary_v08.csv`: ステージ別の時系列確率評価",
        "- `walk_forward_calibration_v08.csv`: OOF確率ビン別の予測値・実測率",
        "- `validation_tests_v08.csv`: 機械的QA",
        "- `prefix_invariance_v08.csv`: 未来足追加に対する検知IDの不変性",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--detector-output", type=Path, default=DEFAULT_DETECTOR_OUTPUT)
    parser.add_argument("--n-jobs", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--backend", choices=["process", "thread", "serial"], default="process")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.detector_output.mkdir(parents=True, exist_ok=True)
    daily, data_qa = detector.load_daily(args.input)
    daily = add_model_features(daily)
    gap_dates, gap_qa = load_gap_dates(args.input)
    frames_all = {"1D": daily, "1W": period_bars(daily, "W-FRI"), "1M": period_bars(daily, "M")}
    frames = {"1D": frames_all["1D"], "1W": frames_all["1W"][frames_all["1W"].certified_calendar].reset_index(drop=True), "1M": frames_all["1M"][frames_all["1M"].certified_calendar].reset_index(drop=True)}
    objects, events, pivots_map, parallel_info = run_lanes(frames, max(1, int(args.n_jobs)), args.backend)
    objects = add_object_metadata(objects)
    events = events.copy()
    if len(events):
        events["available_at"] = pd.to_datetime(events.available_at, errors="coerce").dt.normalize()
        events["event_time"] = pd.to_datetime(events.event_time, errors="coerce").dt.normalize()
    objects = mark_object_gaps(objects, frames, gap_dates)
    write_df(objects, args.detector_output / "objects_v08.csv")
    write_df(events, args.detector_output / "events_v08.csv")
    balanced = objects[(objects.sensitivity == "balanced") & objects.timeframe.isin(EVAL_TIMEFRAMES)].copy()
    balanced_life = {tf: zone_lifecycle_fast(balanced[balanced.timeframe == tf], daily) for tf in EVAL_TIMEFRAMES}
    context_balanced = objects[(objects.sensitivity == "balanced") & objects.timeframe.isin(["1W", "1M"])].copy()
    context_life = {tf: zone_lifecycle_fast(context_balanced[context_balanced.timeframe == tf], daily) for tf in ["1W", "1M"]}
    contexts = build_contexts(daily, frames, pivots_map, context_balanced, context_life)
    catalog, sequence, life_map = build_setup_catalog(objects, events, daily, contexts, frames, gap_dates)
    stage = stage_summary(catalog)
    metrics, predictions = walk_forward(catalog, max(1, int(args.n_jobs)))
    metric_summary = summarize_metrics(metrics)
    calibration = calibration_table(predictions)
    ledger = build_article_ledger(objects, events, catalog, SOURCE_COVERAGE, SOURCE_MANIFEST, args.output)
    qa = validation_tests(daily, objects, events, catalog, contexts, ledger, metrics, gap_dates)
    prefix = prefix_invariance(daily, objects, events, ["2013-09-06", "2020-03-04"])
    gpu = gpu_probe()
    write_df(contexts, args.output / "contexts_v08.csv")
    write_df(sequence, args.output / "sequence_links_v08.csv")
    write_df(catalog, args.output / "setup_catalog_v08.csv")
    write_df(stage, args.output / "stage_summary_v08.csv")
    write_df(metrics, args.output / "walk_forward_metrics_v08.csv")
    write_df(predictions, args.output / "walk_forward_predictions_v08.csv")
    write_df(metric_summary, args.output / "walk_forward_metric_summary_v08.csv")
    write_df(calibration, args.output / "walk_forward_calibration_v08.csv")
    write_df(ledger, args.output / "article_ledger_v08.csv")
    (args.output / "article_ledger_v08.json").write_text(ledger.to_json(orient="records", force_ascii=False, indent=2, date_format="iso"), encoding="utf-8")
    write_df(qa, args.output / "validation_tests_v08.csv")
    write_df(prefix, args.output / "prefix_invariance_v08.csv")
    manifest = {
        "status": "completed_v08", "script_version": SCRIPT_VERSION, "script": str(Path(__file__).resolve()), "script_sha256": sha256(Path(__file__).resolve()),
        "input": data_qa, "input_gap_ledger": gap_qa,
        "data_window": {"start": str(daily.datetime.min().date()), "end": str(daily.datetime.max().date()), "rows": int(len(daily))},
        "coverage": {"requested_years": 30, "actual_years": float((daily.datetime.max() - daily.datetime.min()).days / 365.2425), "start_note": "Yahoo snapshot begins 2000-09-18; no synthetic pre-2000 observations"},
        "parallel": parallel_info, "gpu": gpu,
        "detection": {"objects_all": int(len(objects)), "events_all": int(len(events)), "sensitivity_runs": SENSITIVITIES, "timeframes": TIMEFRAMES, "balanced_eval_objects": int(len(catalog)), "setup_state_counts": catalog.setup_state.value_counts().to_dict() if len(catalog) else {}},
        "clocks": {"formation_prediction": "available_at", "arrival": "available_at -> 20 daily bars for 1D / 60 daily bars for 1W", "post": "first touch bar close -> next 10 daily bars; touch bar excluded"},
        "stages": {"primitive": "basic object exists", "adapted_confirmed": "daily/weekly OHLC conditions observable", "complete_adapted": "ordered local sequence + HTF location + DOL target + invalidation", "original_ob": "unknown_data_lower_timeframe"},
        "targets": {"arrival_horizon_daily_bars": ARRIVAL_HORIZON, "post_horizon_daily_bars": POST_HORIZON, "barrier_atr": BARRIER_ATR, "same_bar_order": "unknown"},
        "walk_forward": {"folds": "expanding chronological 50/60/70/80/90 percent; train label_end strictly before test start", "models": ["intercept_only", "geometry_context_sequence"], "min_train": MIN_TRAIN, "min_test": MIN_TEST, "calibration": "descriptive OOF bins only"},
        "article_coverage": {"rows": int(len(ledger)), "status_counts": ledger.verified_status.value_counts().to_dict() if len(ledger) else {}},
        "qa": {"checks": int(len(qa)), "errors": int((~qa.passed & qa.severity.eq("error")).sum()), "warnings": int((~qa.passed & qa.severity.ne("error")).sum())},
        "files": {"article_ledger": str((args.output / "article_ledger_v08.csv").resolve()), "article_ledger_json": str((args.output / "article_ledger_v08.json").resolve()), "setup_catalog": str((args.output / "setup_catalog_v08.csv").resolve()), "stage_summary": str((args.output / "stage_summary_v08.csv").resolve()), "metric_summary": str((args.output / "walk_forward_metric_summary_v08.csv").resolve()), "calibration": str((args.output / "walk_forward_calibration_v08.csv").resolve()), "qa": str((args.output / "validation_tests_v08.csv").resolve()), "prefix": str((args.output / "prefix_invariance_v08.csv").resolve())},
        "limitations": ["continuous Yahoo futures proxy", "about 25.9 years rather than requested 30", "daily/weekly OHLC cannot verify lower-timeframe MSS or session timing", "complete_adapted is an explicit daily/weekly adaptation, not original ICT execution", "observational stage comparisons are not causal", "small complete-stage samples may be unevaluable"],
    }
    (args.output / "run_manifest_v08.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=jdefault), encoding="utf-8")
    (args.output / "research_report_v08_ja.md").write_text(report_text(manifest, ledger, stage, metric_summary, pd.DataFrame(), prefix, calibration), encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "manifest": str((args.output / 'run_manifest_v08.json').resolve()), "qa_errors": manifest["qa"]["errors"], "qa_warnings": manifest["qa"]["warnings"], "objects": len(objects), "events": len(events), "catalog": len(catalog), "metrics": len(metrics), "parallel": parallel_info}, ensure_ascii=False, default=jdefault))


if __name__ == "__main__":
    main()
