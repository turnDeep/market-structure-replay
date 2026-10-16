"""Causal ICT daily/weekly detector for NKD=F.

This is a corrected v0.3 implementation from the attached ICT
theory/requirements. It does not import or call the previous
probability/backtest scripts. It is a detector and audit pipeline, not a
trading strategy or a proof that an ICT pattern predicts returns.

The v0.3 revision makes event availability causal for lifecycle events,
keeps unknown Friday gaps out of the weekly pattern lane, and separates a
confirmed sweep/reclaim scenario from an unreclaimed breach.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


SCRIPT_VERSION = "nq-ict-detector-v0.3"
DEFAULT_INPUT = Path("nq_20y_ordered_validation_20260909/input/NQ_F_1d_strict.csv")
DEFAULT_THEORY = Path("C:/Users/plane/Downloads/ict_daily_weekly_unified_theory_ja.md")
DEFAULT_REQUIREMENTS = Path("C:/Users/plane/Downloads/nq_ict_detection_requirements_ja.md")
DEFAULT_OUTPUT = Path("nq_ict_detection_20260911_v03")
PROFILES = {
    "strict": dict(equal_tol=0.02, near=0.0, displacement=1.20, efficiency=0.75,
                   max_disp_bars={"1D": 1, "1W": 1}, sweep_wait={"1D": 1, "1W": 1}),
    "balanced": dict(equal_tol=0.05, near=0.05, displacement=0.80, efficiency=0.60,
                      max_disp_bars={"1D": 3, "1W": 2}, sweep_wait={"1D": 3, "1W": 2}),
    "exploratory": dict(equal_tol=0.10, near=0.10, displacement=0.50, efficiency=0.45,
                         max_disp_bars={"1D": 5, "1W": 3}, sweep_wait={"1D": 5, "1W": 3}),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def finite(x: Any, default: float = math.nan) -> float:
    try:
        y = float(x)
        return y if math.isfinite(y) else default
    except (TypeError, ValueError):
        return default


def json_default(x: Any) -> Any:
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return None if not math.isfinite(float(x)) else float(x)
    if isinstance(x, (pd.Timestamp,)):
        return x.isoformat()
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return str(x)


def load_daily(path: Path, as_of: str | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = pd.read_csv(path)
    rename = {c: c.strip().lower().replace(" ", "_") for c in raw.columns}
    raw = raw.rename(columns=rename)
    required = ["datetime", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    raw["datetime"] = pd.to_datetime(raw["datetime"], errors="coerce").dt.normalize()
    for c in required[1:]:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    raw = raw.sort_values("datetime").reset_index(drop=True)
    requested_as_of = pd.Timestamp(as_of).normalize() if as_of else raw.datetime.max()
    raw = raw[raw.datetime <= requested_as_of].copy().reset_index(drop=True)
    validity = (
        raw.datetime.notna() & raw.open.notna() & raw.high.notna() & raw.low.notna()
        & raw.close.notna() & raw.volume.notna()
        & (raw.low <= raw.open) & (raw.low <= raw.close)
        & (raw.high >= raw.open) & (raw.high >= raw.close)
        & (raw.high >= raw.low)
    )
    qa = {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "requested_as_of": str(requested_as_of.date()),
        "rows": int(len(raw)),
        "start": str(raw.datetime.min().date()) if len(raw) else None,
        "end": str(raw.datetime.max().date()) if len(raw) else None,
        "duplicate_dates": int(raw.datetime.duplicated().sum()),
        "null_required_cells": int(raw[required].isna().sum().sum()),
        "invalid_ohlc_rows": int((~validity).sum()),
        "weekend_rows": int((raw.datetime.dt.dayofweek >= 5).sum()),
        "zero_volume_rows": int((raw.volume == 0).sum()),
        "negative_volume_rows": int((raw.volume < 0).sum()),
        "date_monotonic": bool(raw.datetime.is_monotonic_increasing),
    }
    if not validity.all():
        raise ValueError(f"input contains invalid OHLC rows: {qa}")
    raw["bar_idx"] = np.arange(len(raw), dtype=int)
    raw = add_features(raw)
    return raw, qa


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame.copy()
    x["body_low"] = x[["open", "close"]].min(axis=1)
    x["body_high"] = x[["open", "close"]].max(axis=1)
    x["body"] = (x.close - x.open).abs()
    x["range"] = x.high - x.low
    x["upper_wick"] = x.high - x.body_high
    x["lower_wick"] = x.body_low - x.low
    x["close_location"] = np.where(x.range > 0, (x.close - x.low) / x.range, np.nan)
    x["direction"] = np.where(x.close > x.open, "bullish", np.where(x.close < x.open, "bearish", "neutral"))
    prev_close = x.close.shift(1)
    x["tr"] = pd.concat([x.high - x.low, (x.high - prev_close).abs(), (x.low - prev_close).abs()], axis=1).max(axis=1)
    # The shift makes ATR available only from the prior completed bar.
    x["atr14"] = x.tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean().shift(1)
    return x


def western_easter(year: int) -> date:
    """Return Gregorian Easter Sunday without an external calendar package."""
    a = year % 19; b = year // 100; c = year % 100; d = b // 4; e = b % 4
    f = (b + 8) // 25; g = (b - f + 1) // 3; h = (19 * a + b - d - g + 15) % 30
    i = c // 4; k = c % 4; l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def observed_fixed_holiday(year: int, month: int, day_of_month: int) -> date:
    d = date(year, month, day_of_month)
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


def known_friday_closures(start_year: int, end_year: int) -> set[date]:
    """Proxy calendar for provider-empty Friday sessions.

    This is deliberately narrower than a full CME holiday calendar.  It is
    only used to decide whether a week whose final observed date is Thursday
    may be certified.  A Friday absent from this list remains
    ``short_or_missing_week_unknown`` rather than being silently joined to the
    next week.  The list is a transparent calendar assumption, not a claim
    that every Globex holiday session has identical liquidity.
    """
    closed: set[date] = set()
    for year in range(start_year - 1, end_year + 2):
        closed.add(western_easter(year) - timedelta(days=2))  # Good Friday
        closed.add(observed_fixed_holiday(year, 1, 1))
        closed.add(observed_fixed_holiday(year, 7, 4))
        closed.add(observed_fixed_holiday(year, 12, 25))
        closed.add(observed_fixed_holiday(year, 11, 11))  # Veterans Day proxy
        if year >= 2022:
            closed.add(observed_fixed_holiday(year, 6, 19))  # Juneteenth proxy
    return {d for d in closed if start_year <= d.year <= end_year and d.weekday() == 4}


def build_weekly(daily: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    x = daily[["datetime", "open", "high", "low", "close", "volume"]].copy()
    # Period grouping avoids version-specific Resampler.apply/reset_index
    # behaviour and makes the Friday boundary explicit.
    x["week_end"] = x.datetime.dt.to_period("W-FRI").dt.end_time.dt.normalize()
    w = x.groupby("week_end", sort=True).agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), volume=("volume", "sum"), observed_sessions=("close", "count"),
        last_session=("datetime", "max"),
    ).reset_index().rename(columns={"week_end": "datetime"})
    w["week_end"] = w.datetime.dt.normalize()
    w["available_at"] = pd.to_datetime(w.last_session).dt.normalize()
    closure_dates = known_friday_closures(
        int(w.week_end.dt.year.min()) if len(w) else int(as_of.year),
        int(w.week_end.dt.year.max()) if len(w) else int(as_of.year),
    )
    w["has_expected_friday_session"] = w.last_session.dt.normalize().eq(w.week_end)
    w["expected_friday_closed"] = w.week_end.dt.date.isin(closure_dates)
    w["calendar_reason"] = np.where(
        w.has_expected_friday_session, "observed_friday_session",
        np.where(w.expected_friday_closed, "known_friday_closure_proxy", "unknown_friday_gap"),
    )
    # A Thursday-ending week can be certified only when the missing Friday is
    # a known closure proxy.  Unknown provider gaps remain outside the weekly
    # pattern lane, preserving the time gap instead of joining adjacent weeks.
    w["complete"] = (w.has_expected_friday_session | w.expected_friday_closed) & (w.available_at <= as_of)
    w["completion_state"] = np.where(
        w.complete, "confirmed",
        np.where(w.week_end > as_of, "week_to_date", "short_or_missing_week_unknown"),
    )
    w["week_to_date"] = ~w.complete
    w["bar_idx"] = np.arange(len(w), dtype=int)
    return add_features(w)


def make_pivots(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for i in range(1, len(frame) - 1):
        hi = float(frame.high.iloc[i]); lo = float(frame.low.iloc[i])
        if hi > float(frame.high.iloc[i - 1]) and hi > float(frame.high.iloc[i + 1]):
            rows.append(dict(pivot_id=f"PH:{i}", pivot_type="high", origin_idx=i,
                             origin_time=frame.datetime.iloc[i], price=hi,
                             confirmed_idx=i + 1, confirmed_at=frame.datetime.iloc[i + 1],
                             available_at=frame.datetime.iloc[i + 1], rule_id="swing_high_3bar"))
        if lo < float(frame.low.iloc[i - 1]) and lo < float(frame.low.iloc[i + 1]):
            rows.append(dict(pivot_id=f"PL:{i}", pivot_type="low", origin_idx=i,
                             origin_time=frame.datetime.iloc[i], price=lo,
                             confirmed_idx=i + 1, confirmed_at=frame.datetime.iloc[i + 1],
                             available_at=frame.datetime.iloc[i + 1], rule_id="swing_low_3bar"))
    return pd.DataFrame(rows)


def displacement_events(frame: pd.DataFrame, sensitivity: str, timeframe: str) -> pd.DataFrame:
    cfg = PROFILES[sensitivity]; max_bars = cfg["max_disp_bars"][timeframe]
    rows: list[dict[str, Any]] = []
    for start in range(1, len(frame)):
        atr = finite(frame.atr14.iloc[start])
        if not math.isfinite(atr) or atr <= 0:
            continue
        for direction, sign in (("bullish", 1.0), ("bearish", -1.0)):
            chosen = None
            for end in range(start, min(len(frame), start + max_bars)):
                net = sign * (float(frame.close.iloc[end]) - float(frame.close.iloc[start - 1])) / atr
                # Include the first move from the close before the window to
                # the first displacement bar.  The denominator must include
                # every close-to-close step a..b; otherwise efficiency can
                # exceed one and one-bar windows become undefined.
                close_path = float(np.abs(np.diff(frame.close.iloc[start - 1:end + 1].to_numpy(dtype=float))).sum())
                path = close_path / atr
                efficiency = abs(float(frame.close.iloc[end]) - float(frame.close.iloc[start - 1])) / close_path if close_path > 0 else np.nan
                if net >= cfg["displacement"] and math.isfinite(efficiency) and efficiency >= cfg["efficiency"]:
                    chosen = (end, net, efficiency, path)
                    break
            if chosen:
                end, net, efficiency, path = chosen
                rows.append(dict(object_id=f"NKD=F|{timeframe}|displacement|{sensitivity}|{direction}|{start}",
                                 timeframe=timeframe, object_type="displacement", rule_id="displacement",
                                 sensitivity=sensitivity, direction=direction, origin_idx=start,
                                 origin_time=frame.datetime.iloc[start], confirmed_idx=end,
                                 confirmed_at=frame.datetime.iloc[end], available_at=frame.datetime.iloc[end],
                                 zone_low=np.nan, zone_high=np.nan, body_low=np.nan, body_high=np.nan,
                                 metric_net_atr=net, metric_efficiency=efficiency, metric_path_atr=path,
                                 passed_checks="net_move;path_efficiency", failed_checks="", unknown_checks=""))
    return pd.DataFrame(rows)


def fvg_objects(frame: pd.DataFrame, sensitivity: str, timeframe: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for i in range(2, len(frame)):
        atr = finite(frame.atr14.iloc[i])
        if float(frame.high.iloc[i - 2]) < float(frame.low.iloc[i]):
            lo, hi, direction = float(frame.high.iloc[i - 2]), float(frame.low.iloc[i]), "bullish"
        elif float(frame.low.iloc[i - 2]) > float(frame.high.iloc[i]):
            lo, hi, direction = float(frame.high.iloc[i]), float(frame.low.iloc[i - 2]), "bearish"
        else:
            continue
        rows.append(dict(object_id=f"NKD=F|{timeframe}|fvg|{sensitivity}|{i}", timeframe=timeframe,
                         object_type="fvg", rule_id="fvg_3bar", sensitivity=sensitivity, direction=direction,
                         origin_idx=i - 2, origin_time=frame.datetime.iloc[i - 2], confirmed_idx=i,
                         confirmed_at=frame.datetime.iloc[i], available_at=frame.datetime.iloc[i],
                         zone_low=lo, zone_high=hi, body_low=np.nan, body_high=np.nan,
                         metric_width_atr=(hi - lo) / atr if math.isfinite(atr) and atr > 0 else np.nan,
                         metric_net_atr=np.nan, metric_efficiency=np.nan, metric_path_atr=np.nan,
                         passed_checks="positive_wick_gap", failed_checks="", unknown_checks=""))
    return pd.DataFrame(rows)


def ob_objects(frame: pd.DataFrame, displacement: pd.DataFrame, sensitivity: str, timeframe: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    used: set[tuple[str, int]] = set()
    for d in displacement.itertuples(index=False):
        direction = d.direction
        candidates = []
        for j in range(max(0, int(d.origin_idx) - 3), int(d.origin_idx) + 1):
            is_opp = (direction == "bullish" and frame.close.iloc[j] < frame.open.iloc[j]) or (direction == "bearish" and frame.close.iloc[j] > frame.open.iloc[j])
            if is_opp:
                candidates.append(j)
        if not candidates:
            continue
        j = candidates[-1]
        key = (direction, j)
        if key in used:
            continue
        used.add(key)
        next_ok = j + 1 < len(frame)
        if direction == "bullish":
            sweep = next_ok and float(frame.low.iloc[j + 1]) < float(frame.low.iloc[j])
            close_break = next_ok and float(frame.close.iloc[j + 1]) > float(frame.high.iloc[j])
            wick_engulf = next_ok and float(frame.high.iloc[j + 1]) >= float(frame.high.iloc[j]) and float(frame.low.iloc[j + 1]) <= float(frame.low.iloc[j])
            body_engulf = next_ok and float(frame.open.iloc[j + 1]) <= float(frame.close.iloc[j]) and float(frame.close.iloc[j + 1]) >= float(frame.open.iloc[j])
        else:
            sweep = next_ok and float(frame.high.iloc[j + 1]) > float(frame.high.iloc[j])
            close_break = next_ok and float(frame.close.iloc[j + 1]) < float(frame.low.iloc[j])
            wick_engulf = next_ok and float(frame.high.iloc[j + 1]) >= float(frame.high.iloc[j]) and float(frame.low.iloc[j + 1]) <= float(frame.low.iloc[j])
            body_engulf = next_ok and float(frame.open.iloc[j + 1]) >= float(frame.close.iloc[j]) and float(frame.close.iloc[j + 1]) <= float(frame.open.iloc[j])
        # The candidate is observed when the displacement is confirmed. The
        # primary daily/weekly adaptation is confirmed only when a later close
        # crosses the origin boundary within that displacement window. The
        # stricter source-rule checks remain separate evidence columns.
        boundary = float(frame.high.iloc[j]) if direction == "bullish" else float(frame.low.iloc[j])
        boundary_slice = frame.close.iloc[j + 1:int(d.confirmed_idx) + 1]
        if direction == "bullish":
            boundary_cross = np.flatnonzero(boundary_slice.to_numpy(dtype=float) > boundary)
        else:
            boundary_cross = np.flatnonzero(boundary_slice.to_numpy(dtype=float) < boundary)
        boundary_idx = int(j + 1 + boundary_cross[0]) if len(boundary_cross) else None
        passed = ["opposite_origin", "directional_displacement"]
        failed = []
        if boundary_idx is not None:
            passed.append("origin_boundary_close_break")
        else:
            failed.append("origin_boundary_close_break")
        if sweep: passed.append("next_bar_sweep")
        else: failed.append("next_bar_sweep")
        if close_break: passed.append("next_bar_close_break")
        else: failed.append("next_bar_close_break")
        if wick_engulf: passed.append("wick_engulf")
        else: failed.append("wick_engulf")
        if body_engulf: passed.append("body_engulf")
        else: failed.append("body_engulf")
        evidence_state = "confirmed" if boundary_idx is not None else "candidate"
        # The row also stores the next-bar source-rule checks above.  Those
        # checks cannot be known until the next bar closes, so the object is
        # not usable before that bar even when the displacement itself ended
        # on the origin bar.  This prevents an explanatory flag from
        # smuggling future information into the object's availability time.
        availability_idx = int(d.confirmed_idx)
        if next_ok:
            availability_idx = max(availability_idx, j + 1)
        if boundary_idx is not None:
            availability_idx = max(availability_idx, int(boundary_idx))
        rows.append(dict(object_id=f"NKD=F|{timeframe}|ob_candidate_dw|{sensitivity}|{j}", timeframe=timeframe,
                         object_type="ob_candidate", rule_id="ob_candidate_dw", sensitivity=sensitivity,
                         direction=direction, origin_idx=j, origin_time=frame.datetime.iloc[j],
                         confirmed_idx=int(boundary_idx) if boundary_idx is not None else int(d.confirmed_idx),
                         confirmed_at=frame.datetime.iloc[boundary_idx] if boundary_idx is not None else d.confirmed_at,
                         available_at=frame.datetime.iloc[availability_idx],
                         zone_low=float(frame.body_low.iloc[j]), zone_high=float(frame.body_high.iloc[j]),
                         body_low=float(frame.body_low.iloc[j]), body_high=float(frame.body_high.iloc[j]),
                         wick_zone_low=float(frame.low.iloc[j]), wick_zone_high=float(frame.high.iloc[j]),
                         metric_width_atr=(float(frame.body_high.iloc[j]) - float(frame.body_low.iloc[j])) / finite(frame.atr14.iloc[j]) if finite(frame.atr14.iloc[j]) > 0 else np.nan,
                         metric_net_atr=d.metric_net_atr, metric_efficiency=d.metric_efficiency,
                         metric_path_atr=d.metric_path_atr, passed_checks=";".join(passed), failed_checks=";".join(failed),
                         unknown_checks="lower_timeframe_imbalance;lower_timeframe_structure_shift",
                         evidence_state=evidence_state, source_rule_state="unknown_data",
                         origin_boundary_price=boundary, origin_boundary_cross_idx=boundary_idx,
                         origin_boundary_cross_at=frame.datetime.iloc[boundary_idx] if boundary_idx is not None else pd.NaT,
                         quality_flags="source_lower_timeframe_missing"))
    return pd.DataFrame(rows)


def sweep_events(frame: pd.DataFrame, pivots: pd.DataFrame, sensitivity: str, timeframe: str) -> pd.DataFrame:
    wait = PROFILES[sensitivity]["sweep_wait"][timeframe]
    lows = frame.low.to_numpy(dtype=float); highs = frame.high.to_numpy(dtype=float); closes = frame.close.to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    for p in pivots.itertuples(index=False):
        start = int(p.confirmed_idx) + 1
        if start >= len(frame):
            continue
        breach_mask = lows[start:] < p.price if p.pivot_type == "low" else highs[start:] > p.price
        breach_ix = np.flatnonzero(breach_mask)
        if len(breach_ix) == 0:
            continue
        j = start + int(breach_ix[0])
        is_bull = p.pivot_type == "low"
        direction = "bullish" if is_bull else "bearish"
        breach_id = f"NKD=F|{timeframe}|breach|{sensitivity}|{p.pivot_id}|{j}"
        # The breach is immutable evidence. Recovery and horizon outcomes are
        # later append-only events, so a future run cannot rewrite the breach.
        rows.append(dict(event_id=breach_id, timeframe=timeframe,
                         event_type="liquidity_breach", rule_id="liquidity_breach", sensitivity=sensitivity,
                         direction=direction, event_idx=j, event_time=frame.datetime.iloc[j],
                         available_at=frame.datetime.iloc[j], object_id="", parent_ids=p.pivot_id,
                         stage="observed", passed_checks="boundary_breach", failed_checks="", unknown_checks="",
                         metric_bars=0, updated_at=frame.datetime.iloc[j]))
        # wait is the total number of bars including the breach bar.
        window = np.arange(j, min(len(frame), j + wait))
        recover_mask = closes[window] >= p.price if is_bull else closes[window] <= p.price
        reclaim_idx = int(window[np.flatnonzero(recover_mask)[0]]) if recover_mask.any() else None
        if reclaim_idx is not None:
            rows.append(dict(event_id=f"NKD=F|{timeframe}|sweep|{sensitivity}|{p.pivot_id}|{reclaim_idx}", timeframe=timeframe,
                             event_type="sweep_same_bar" if reclaim_idx == j else "sweep_delayed_reclaim",
                             rule_id="liquidity_sweep", sensitivity=sensitivity, direction=direction, event_idx=reclaim_idx,
                             event_time=frame.datetime.iloc[reclaim_idx], available_at=frame.datetime.iloc[reclaim_idx], object_id="",
                             parent_ids=f"{p.pivot_id};{breach_id}", stage="confirmed", passed_checks="boundary_breach;close_recovery",
                             failed_checks="", unknown_checks="", metric_bars=reclaim_idx - j, updated_at=frame.datetime.iloc[reclaim_idx]))
        elif len(window) < wait:
            # The data ended before the prescribed horizon. This is a pending
            # state, not a failure label.
            k = int(window[-1])
            rows.append(dict(event_id=f"NKD=F|{timeframe}|pending|{sensitivity}|{p.pivot_id}|{k}", timeframe=timeframe,
                             event_type="liquidity_recovery_pending", rule_id="liquidity_sweep", sensitivity=sensitivity,
                             direction=direction, event_idx=k, event_time=frame.datetime.iloc[k], available_at=frame.datetime.iloc[k],
                             object_id="", parent_ids=f"{p.pivot_id};{breach_id}", stage="forming",
                             passed_checks="boundary_breach", failed_checks="recovery_horizon_incomplete", unknown_checks="",
                             metric_bars=k - j, updated_at=frame.datetime.iloc[k]))
        else:
            k = int(window[-1])
            rows.append(dict(event_id=f"NKD=F|{timeframe}|unreclaimed|{sensitivity}|{p.pivot_id}|{k}", timeframe=timeframe,
                             event_type="unreclaimed_at_horizon", rule_id="liquidity_sweep", sensitivity=sensitivity,
                             direction=direction, event_idx=k, event_time=frame.datetime.iloc[k], available_at=frame.datetime.iloc[k],
                             object_id="", parent_ids=f"{p.pivot_id};{breach_id}", stage="expired",
                             passed_checks="boundary_breach;recovery_horizon_elapsed", failed_checks="recovery", unknown_checks="",
                             metric_bars=k - j, updated_at=frame.datetime.iloc[k]))
    return pd.DataFrame(rows)


def cisd_events(frame: pd.DataFrame, sensitivity: str, timeframe: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    opens = frame.open.to_numpy(dtype=float)
    closes = frame.close.to_numpy(dtype=float)
    datetimes = frame.datetime.to_numpy()
    for i in range(1, len(frame)):
        j = i - 1
        prior_direction = "bearish" if closes[j] < opens[j] else "bullish" if closes[j] > opens[j] else "neutral"
        if prior_direction == "neutral":
            continue
        # A delivery leg can contain several consecutive candles. Reference
        # its first open, rather than silently reducing it to the last candle.
        leg_start = j
        while leg_start > 0:
            prev_direction = "bearish" if closes[leg_start - 1] < opens[leg_start - 1] else "bullish" if closes[leg_start - 1] > opens[leg_start - 1] else "neutral"
            if prev_direction != prior_direction:
                break
            leg_start -= 1
        if prior_direction == "bearish" and closes[i] > opens[i] and closes[i] > opens[leg_start]:
            direction = "bullish"
        elif prior_direction == "bullish" and closes[i] < opens[i] and closes[i] < opens[leg_start]:
            direction = "bearish"
        else:
            continue
        rows.append(dict(event_id=f"NKD=F|{timeframe}|cisd|{sensitivity}|{i}", timeframe=timeframe,
                         event_type="cisd", rule_id="cisd", sensitivity=sensitivity, direction=direction, event_idx=i,
                         event_time=datetimes[i], available_at=datetimes[i], object_id="",
                         parent_ids=f"bars:{leg_start}-{j}", stage="confirmed", passed_checks="prior_opposite_delivery_leg_open_close_break",
                         failed_checks="", unknown_checks="", metric_bars=i - leg_start))
    return pd.DataFrame(rows)


def structure_events(frame: pd.DataFrame, pivots: pd.DataFrame, displacement: pd.DataFrame, sensitivity: str, timeframe: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    closes = frame.close.to_numpy(dtype=float)
    datetimes = frame.datetime.to_numpy()
    high_p = pivots[pivots.pivot_type == "high"].sort_values("confirmed_idx")
    low_p = pivots[pivots.pivot_type == "low"].sort_values("confirmed_idx")
    # Build candidate structure breaks from a displacement's own direction and
    # window. A bullish displacement cannot validate a bearish-side break.
    candidates = []
    for d in displacement.itertuples(index=False):
        direction = d.direction
        pool = high_p if direction == "bullish" else low_p
        for p in pool.itertuples(index=False):
            start = max(int(d.origin_idx), int(p.confirmed_idx) + 1)
            end = int(d.confirmed_idx)
            if start > end:
                continue
            close = closes[start:end + 1]
            prior = closes[start - 1:end]
            if direction == "bullish":
                cross = np.flatnonzero((close > float(p.price)) & (prior <= float(p.price)))
            else:
                cross = np.flatnonzero((close < float(p.price)) & (prior >= float(p.price)))
            if len(cross):
                cross_idx = start + int(cross[0])
                candidates.append(dict(direction=direction, pivot_id=p.pivot_id, pivot_price=float(p.price),
                                       displacement_id=d.object_id, cross_idx=cross_idx,
                                       confirm_idx=end, bars_after_cross=end - cross_idx))
                break
    candidates.sort(key=lambda x: (x["confirm_idx"], x["cross_idx"], x["pivot_id"]))
    state = "neutral"
    seen = set()
    for c in candidates:
        key = (c["pivot_id"], c["displacement_id"])
        if key in seen:
            continue
        seen.add(key)
        direction = c["direction"]
        rule = "mss" if state in {"bearish" if direction == "bullish" else "bullish", "neutral"} else "bos"
        state = direction
        rows.append(dict(event_id=f"NKD=F|{timeframe}|{rule}|{sensitivity}|{c['confirm_idx']}|{c['pivot_id']}|{c['displacement_id']}", timeframe=timeframe,
                         event_type=rule, rule_id=rule, sensitivity=sensitivity, direction=direction, event_idx=c["confirm_idx"],
                         event_time=datetimes[c["confirm_idx"]], available_at=datetimes[c["confirm_idx"]], object_id=c["displacement_id"],
                         parent_ids=f"{c['pivot_id']};{c['displacement_id']}", stage="confirmed",
                         passed_checks="close_cross;directional_displacement;direction_match", failed_checks="",
                         unknown_checks="", metric_bars=c["bars_after_cross"], observed_idx=c["cross_idx"],
                         observed_at=frame.datetime.iloc[c["cross_idx"]]))
    return pd.DataFrame(rows)


def lifecycle_events(frame: pd.DataFrame, objects: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if objects.empty:
        return objects, pd.DataFrame()
    out = objects.copy()
    rows: list[dict[str, Any]] = []
    lows = frame.low.to_numpy(dtype=float); highs = frame.high.to_numpy(dtype=float); closes = frame.close.to_numpy(dtype=float)
    opens = frame.open.to_numpy(dtype=float); atrs = frame.atr14.to_numpy(dtype=float)
    first_touch = []; first_full = []; first_near = []; terminal = []; touch_counts = []; near_counts = []

    def add_event(r: Any, event_id: str, event_type: str, rule_id: str, k: int,
                  stage: str, passed: str, failed: str = "", metric_bars: Any = np.nan) -> None:
        rows.append(dict(event_id=event_id, timeframe=r.timeframe, event_type=event_type, rule_id=rule_id,
                         sensitivity=r.sensitivity, direction=r.direction, event_idx=k,
                         event_time=frame.datetime.iloc[k], available_at=frame.datetime.iloc[k], object_id=r.object_id,
                         parent_ids=r.object_id, stage=stage, passed_checks=passed, failed_checks=failed,
                         unknown_checks=getattr(r, "unknown_checks", ""), metric_bars=metric_bars,
                         updated_at=frame.datetime.iloc[k]))

    for r in out.itertuples(index=False):
        lo, hi = finite(r.zone_low), finite(r.zone_high)
        if not math.isfinite(lo) or not math.isfinite(hi) or r.object_type not in {"ob_candidate", "fvg"}:
            first_touch.append(np.nan); first_full.append(np.nan); first_near.append(np.nan); terminal.append("not_applicable"); touch_counts.append(0); near_counts.append(0); continue
        # Lifecycle observations begin after the object is actually usable.
        # ``confirmed_idx`` can precede the availability of a next-bar
        # source-rule check, so starting from confirmed_idx alone creates
        # look-ahead lifecycle events.
        available_idx = int(frame.datetime.searchsorted(pd.Timestamp(r.available_at).normalize(), side="right"))
        start = max(int(r.confirmed_idx) + 1, available_idx)
        if start >= len(frame):
            first_touch.append(np.nan); first_full.append(np.nan); first_near.append(np.nan); terminal.append("active_at_asof"); touch_counts.append(0); near_counts.append(0); continue
        idx = np.arange(start, len(frame))
        hit_mask = (lows[start:] <= hi) & (highs[start:] >= lo)
        # Full mitigation/fill is a subset of an observed intersection. A
        # price gap that jumps over the zone is recorded separately.
        full_mask = hit_mask & ((lows[start:] <= lo) if r.direction == "bullish" else (highs[start:] >= hi))
        prev_close = closes[start - 1:-1]
        current_open = opens[start:]
        gap_through_mask = (((prev_close > hi) & (current_open < lo)) | ((prev_close < lo) & (current_open > hi))) & ~hit_mask
        inside = hit_mask
        distance = np.where(highs[start:] < lo, lo - highs[start:], np.where(lows[start:] > hi, lows[start:] - hi, 0.0))
        near_atr = atrs[start:]
        near_limit = float(PROFILES.get(r.sensitivity, {}).get("near", 0.0))
        near_mask = (~hit_mask) & (near_limit > 0) & np.isfinite(near_atr) & (near_atr > 0) & (distance <= near_limit * near_atr)
        touch = int(idx[np.flatnonzero(hit_mask)[0]]) if hit_mask.any() else None
        full = int(idx[np.flatnonzero(full_mask)[0]]) if full_mask.any() else None
        touch_positions = np.flatnonzero(hit_mask)
        touch_starts = touch_positions[np.r_[True, np.diff(touch_positions) > 1]] if len(touch_positions) else np.array([], dtype=int)
        near_indices = idx[np.flatnonzero(near_mask)]
        near = int(near_indices[0]) if len(near_indices) else None
        for n, pos in enumerate(touch_starts, start=1):
            k = int(idx[pos])
            add_event(r, f"{r.object_id}|touch|{k}", "zone_touch", "zone_touch", k, "confirmed", f"price_intersection;touch_episode_{n}", metric_bars=n)
        # Keep the first near observation as the event anchor and retain the
        # full observation count on the object. Emitting one row per nearby
        # bar would inflate the event table without adding a new transition.
        near_positions = np.flatnonzero(near_mask)
        if len(near_positions):
            pos = int(near_positions[0]); k = int(idx[pos])
            add_event(r, f"{r.object_id}|near|{k}", "zone_near", "zone_near", k, "observed", "within_profile_near_distance", metric_bars=float(distance[pos] / near_atr[pos]))
        for pos in np.flatnonzero(gap_through_mask) if r.object_type == "fvg" else []:
            k = int(idx[pos])
            add_event(r, f"{r.object_id}|gap_through|{k}", "fvg_gap_through", "fvg_gap_through", k, "observed", "open_gap_crossed_zone_without_intersection")
        if full is not None:
            et = "ob_full_penetration" if r.object_type == "ob_candidate" else "fvg_full_fill"
            add_event(r, f"{r.object_id}|{et}|{full}", et, et, full, "confirmed", "price_intersection;opposite_boundary_reached")
        if r.object_type == "ob_candidate":
            wick_mask = lows[start:] < lo if r.direction == "bullish" else highs[start:] > hi
            close_mask = closes[start:] < lo if r.direction == "bullish" else closes[start:] > hi
            wick_positions = np.flatnonzero(wick_mask)
            wick = int(idx[wick_positions[0]]) if len(wick_positions) else None
            close_break = int(idx[np.flatnonzero(close_mask)[0]]) if close_mask.any() else None
            # Record the beginning of each distinct breach episode, rather
            # than one row for every bar that remains outside the boundary.
            # The latter is a repeated state, not a repeated transition, and
            # made the 20-year event file grow to gigabytes.
            wick_starts = wick_positions[np.r_[True, np.diff(wick_positions) > 1]] if len(wick_positions) else np.array([], dtype=int)
            for episode_no, pos in enumerate(wick_starts, start=1):
                k = int(idx[pos])
                add_event(r, f"{r.object_id}|wick_breach|{episode_no}|{k}", "ob_wick_breach", "ob_wick_breach", k, "confirmed", "outer_zone_breach")
                recovery = closes[k:] >= lo if r.direction == "bullish" else closes[k:] <= hi
                recovery_positions = np.flatnonzero(recovery)
                reclaim = int(k + recovery_positions[0]) if len(recovery_positions) else None
                if reclaim is not None:
                    add_event(r, f"{r.object_id}|reclaim|{episode_no}|{reclaim}", "ob_reclaim", "ob_reclaim", reclaim, "confirmed", "wick_breach;close_recovery", metric_bars=reclaim - k)
            if close_break is not None:
                add_event(r, f"{r.object_id}|close_invalidation|{close_break}", "ob_close_invalidation", "ob_close_invalidation", close_break, "confirmed", "close_outside_zone")
        if r.object_type == "fvg" and full is not None:
            term = "fully_filled"
        elif r.object_type == "ob_candidate" and close_break is not None:
            term = "invalidated_close"
        else:
            term = "active_at_asof"
        first_touch.append(touch if touch is not None else np.nan)
        first_full.append(full if full is not None else np.nan)
        first_near.append(near if near is not None else np.nan)
        terminal.append(term); touch_counts.append(int(len(touch_starts))); near_counts.append(int(near_mask.sum()))
    out["first_touch_idx"] = first_touch; out["first_full_fill_idx"] = first_full; out["first_near_idx"] = first_near
    out["touch_episode_count"] = touch_counts; out["near_observation_count"] = near_counts; out["terminal_state"] = terminal
    return out, pd.DataFrame(rows)


def lane(frame: pd.DataFrame, timeframe: str, sensitivity: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    piv = make_pivots(frame)
    disp = displacement_events(frame, sensitivity, timeframe)
    fvgs = fvg_objects(frame, sensitivity, timeframe)
    obs = ob_objects(frame, disp, sensitivity, timeframe)
    objects = []
    if len(piv):
        objects.append(pd.DataFrame([dict(object_id=f"NKD=F|{timeframe}|{p.pivot_id}|{sensitivity}", timeframe=timeframe, object_type="swing",
                                          rule_id=p.rule_id, sensitivity=sensitivity, direction="bullish" if p.pivot_type == "low" else "bearish",
                                          origin_idx=p.origin_idx, origin_time=p.origin_time, confirmed_idx=p.confirmed_idx, confirmed_at=p.confirmed_at,
                                          available_at=p.available_at, zone_low=p.price, zone_high=p.price, body_low=np.nan, body_high=np.nan,
                                          passed_checks="3bar_fractal", failed_checks="", unknown_checks="") for p in piv.itertuples(index=False)]))
    for z in [disp, fvgs, obs]:
        if len(z): objects.append(z)
    obj = pd.concat(objects, ignore_index=True, sort=False) if objects else pd.DataFrame()
    if len(obj):
        article_map = {
            "swing_high_3bar": "82", "swing_low_3bar": "81", "displacement": "75",
            "fvg_3bar": "100;114", "ob_candidate_dw": "87;88;115",
        }
        variant_map = {
            "swing": "source_rule_swing_3bar", "displacement": "source_rule_displacement_dw_v2",
            "fvg": "source_rule_fvg_3bar", "ob_candidate": "daily_weekly_adaptation_ob_candidate_dw_v2",
        }
        obj["symbol"] = "NKD=F"
        obj["article_ids"] = obj["rule_id"].map(article_map).fillna("")
        obj["rule_version"] = SCRIPT_VERSION
        obj["definition_variant"] = obj["object_type"].map(variant_map).fillna("")
        if "evidence_state" not in obj:
            obj["evidence_state"] = "confirmed"
        obj["evidence_state"] = obj["evidence_state"].fillna("confirmed")
        obj["first_observed_at"] = obj["origin_time"]
        obj["updated_at"] = obj["available_at"]
        if "quality_flags" not in obj:
            obj["quality_flags"] = ""
        obj["quality_flags"] = obj["quality_flags"].fillna("")
    # Event rows for formation/confirmation are intentionally separate from
    # later lifecycle events, with their own available_at.
    ev_rows = []
    for r in obj.itertuples(index=False):
        ev_rows.append(dict(event_id=f"{r.object_id}|formation", timeframe=timeframe, event_type="formation", rule_id=r.rule_id,
                            sensitivity=sensitivity, direction=r.direction, event_idx=r.origin_idx, event_time=r.origin_time,
                            # origin_time is the structural origin; the object is
                            # usable only after its confirmation is observable.
                            available_at=r.available_at, object_id=r.object_id, parent_ids="", stage=r.evidence_state if r.object_type == "ob_candidate" else "forming", passed_checks=r.passed_checks,
                            failed_checks=r.failed_checks, unknown_checks=r.unknown_checks, metric_bars=np.nan,
                            observed_at=r.origin_time, updated_at=r.available_at))
        # A candidate with no origin-boundary close break remains a candidate;
        # do not manufacture a confirmed event at the displacement endpoint.
        if r.evidence_state == "confirmed" and int(r.confirmed_idx) != int(r.origin_idx):
            ev_rows.append(dict(event_id=f"{r.object_id}|confirmation", timeframe=timeframe, event_type="confirmation", rule_id=r.rule_id,
                                sensitivity=sensitivity, direction=r.direction, event_idx=r.confirmed_idx, event_time=r.confirmed_at,
                                available_at=r.available_at, object_id=r.object_id, parent_ids="", stage="confirmed", passed_checks=r.passed_checks,
                                failed_checks=r.failed_checks, unknown_checks=r.unknown_checks, metric_bars=int(r.confirmed_idx) - int(r.origin_idx),
                                observed_at=r.confirmed_at, updated_at=r.available_at))
    e = [pd.DataFrame(ev_rows)] if ev_rows else []
    for x in [sweep_events(frame, piv, sensitivity, timeframe), cisd_events(frame, sensitivity, timeframe), structure_events(frame, piv, disp, sensitivity, timeframe)]:
        if len(x): e.append(x)
    # Lifecycle transitions are meaningful for price zones only.  Swing and
    # displacement rows remain in the object table but do not need an O(N*M)
    # zone-intersection scan.
    zone_mask = obj.object_type.isin(["ob_candidate", "fvg"]) if len(obj) else pd.Series(dtype=bool)
    zone_objects = obj.loc[zone_mask].copy() if len(obj) else obj
    zone_objects, life = lifecycle_events(frame, zone_objects)
    if len(obj) and len(zone_objects):
        obj = obj.drop(columns=[c for c in ["first_touch_idx", "first_full_fill_idx", "first_near_idx", "touch_episode_count", "near_observation_count", "terminal_state"] if c in obj.columns], errors="ignore")
        obj = obj.merge(zone_objects[["object_id", "first_touch_idx", "first_full_fill_idx", "first_near_idx", "touch_episode_count", "near_observation_count", "terminal_state"]], on="object_id", how="left")
    if len(obj):
        for c, default in [("first_touch_idx", np.nan), ("first_full_fill_idx", np.nan), ("first_near_idx", np.nan), ("touch_episode_count", 0), ("near_observation_count", 0), ("terminal_state", "not_applicable")]:
            if c not in obj.columns: obj[c] = default
    if len(life): e.append(life)
    events = pd.concat(e, ignore_index=True, sort=False) if e else pd.DataFrame()
    return obj, events, piv


def weekly_contexts(daily: pd.DataFrame, weekly: pd.DataFrame, weekly_objects: pd.DataFrame, weekly_pivots: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in daily.itertuples(index=False):
        eligible = weekly[(weekly.complete) & (weekly.available_at <= r.datetime)]
        if eligible.empty:
            rows.append(dict(datetime=r.datetime, weekly_context_id="", weekly_available_at=pd.NaT, weekly_direction="unknown", weekly_position="unknown",
                             range_low=np.nan, range_high=np.nan, range_mid=np.nan, weekly_ob_distance_atr=np.nan, weekly_fvg_distance_atr=np.nan))
            continue
        w = eligible.iloc[-1]
        # Do not compare positional indices from a full weekly frame with a
        # prefix frame.  The same pivot may have a different positional
        # index after incomplete/short weeks are filtered.  Its causal
        # availability timestamp is the invariant criterion.
        ps = weekly_pivots[pd.to_datetime(weekly_pivots.available_at) <= r.datetime]
        hs = ps[ps.pivot_type == "high"].sort_values("confirmed_idx"); ls = ps[ps.pivot_type == "low"].sort_values("confirmed_idx")
        direction = "neutral"
        if len(hs) >= 2 and len(ls) >= 2:
            if hs.price.iloc[-1] > hs.price.iloc[-2] and ls.price.iloc[-1] > ls.price.iloc[-2]: direction = "bullish"
            elif hs.price.iloc[-1] < hs.price.iloc[-2] and ls.price.iloc[-1] < ls.price.iloc[-2]: direction = "bearish"
        candidates = [finite(hs.price.iloc[-1]) if len(hs) else np.nan, finite(ls.price.iloc[-1]) if len(ls) else np.nan]
        rh, rl = (max(candidates), min(candidates)) if all(math.isfinite(x) for x in candidates) else (np.nan, np.nan)
        mid = (rh + rl) / 2 if math.isfinite(rh) and math.isfinite(rl) and rh > rl else np.nan
        pos = "unknown" if not math.isfinite(mid) else ("premium" if r.close > mid else "discount")
        atr = finite(r.atr14)
        def nearest_distance(kind: str) -> float:
            z = weekly_objects[(weekly_objects.object_type == kind) & (weekly_objects.available_at <= r.datetime) & weekly_objects.zone_low.notna()]
            if z.empty or not math.isfinite(atr) or atr <= 0: return np.nan
            d = np.minimum((z.zone_low - r.close).abs(), (z.zone_high - r.close).abs())
            inside = (z.zone_low <= r.close) & (z.zone_high >= r.close)
            d = np.where(inside, 0.0, d)
            return float(np.min(d) / atr)
        rows.append(dict(datetime=r.datetime, weekly_context_id=f"W:{w.week_end.date()}", weekly_available_at=w.available_at,
                         weekly_direction=direction, weekly_position=pos, range_low=rl, range_high=rh, range_mid=mid,
                         weekly_ob_distance_atr=nearest_distance("ob_candidate"), weekly_fvg_distance_atr=nearest_distance("fvg")))
    return pd.DataFrame(rows)


def scenarios(daily: pd.DataFrame, daily_events: pd.DataFrame, contexts: pd.DataFrame) -> pd.DataFrame:
    if daily_events.empty: return pd.DataFrame()
    rows = []
    for sens in PROFILES:
        ev = daily_events[(daily_events.sensitivity == sens)].copy()
        sweeps = ev[(ev.rule_id == "liquidity_sweep") &
                    (ev.event_type.isin(["sweep_same_bar", "sweep_delayed_reclaim"])) &
                    (ev.stage == "confirmed")].sort_values(["event_idx", "event_id"])
        breaches = ev[ev.event_type == "liquidity_breach"].sort_values(["event_idx", "event_id"])
        shifts = ev[ev.rule_id.isin(["mss", "cisd"])].sort_values(["event_idx", "event_id"])
        wait = int(PROFILES[sens]["sweep_wait"]["1D"] + 5)

        def context_for(ts: pd.Timestamp) -> str:
            if contexts.empty:
                return ""
            c = contexts[pd.to_datetime(contexts.datetime) <= pd.Timestamp(ts).normalize()]
            return str(c.iloc[-1].weekly_context_id) if len(c) else ""

        # A confirmed reversal scenario requires an observed recovery.  Each
        # stage is append-only: the formative record remains when a later MSS
        # adds a confirmed record.  This makes prefix runs comparable without
        # rewriting history.
        for s in sweeps.itertuples(index=False):
            m = shifts[(shifts.direction == s.direction) &
                       (shifts.event_idx >= s.event_idx) &
                       (shifts.event_idx <= s.event_idx + wait)].sort_values(["event_idx", "event_id"])
            first = m.iloc[0] if len(m) else None
            rows.append(dict(
                scenario_id=f"NKD=F|reversal_sweep_mss|{sens}|{s.event_id}|recovery",
                scenario_type="reversal_sweep_mss", sensitivity=sens,
                direction=s.direction, stage="forming", origin_time=s.event_time,
                confirmed_at=pd.NaT, available_at=s.available_at,
                required_event_ids=s.event_id,
                optional_event_ids="",
                weekly_context_id=context_for(s.event_time),
                invalidation="opposite_close_break_or_expiry",
                expiry_rule=f"await MSS/CISD within {wait} daily bars after recovery",
            ))
            if first is not None:
                rows.append(dict(
                    scenario_id=f"NKD=F|reversal_sweep_mss|{sens}|{s.event_id}|mss|{first.event_id}",
                    scenario_type="reversal_sweep_mss", sensitivity=sens,
                    direction=s.direction, stage="confirmed", origin_time=s.event_time,
                    confirmed_at=first.event_time, available_at=first.available_at,
                    required_event_ids=s.event_id, optional_event_ids=first.event_id,
                    weekly_context_id=context_for(s.event_time),
                    invalidation="opposite_close_break_or_expiry",
                    expiry_rule=f"{wait} daily bars after confirmed recovery",
                ))

        # Preserve the breach and its recovery status as separate partial
        # scenarios.  The formative status is always retained; a later
        # unreclaimed record is an additional expired observation and is never
        # promoted to a completed reversal.
        for b in breaches.itertuples(index=False):
            children = ev[ev.parent_ids.astype(str).str.contains(str(b.event_id), regex=False)]
            recovery = children[children.event_type.isin(["sweep_same_bar", "sweep_delayed_reclaim"])]
            unreclaimed = children[children.event_type == "unreclaimed_at_horizon"]
            rows.append(dict(
                scenario_id=f"NKD=F|sweep_recovery_status|{sens}|{b.event_id}|pending",
                scenario_type="sweep_recovery_status", sensitivity=sens,
                direction=b.direction, stage="forming", origin_time=b.event_time,
                confirmed_at=pd.NaT, available_at=b.available_at,
                required_event_ids=b.event_id, optional_event_ids="",
                weekly_context_id=context_for(b.event_time),
                invalidation="reclaim_required_for_reversal",
                expiry_rule="recovery_horizon_pending",
            ))
            if len(unreclaimed):
                u = unreclaimed.sort_values(["event_idx", "event_id"]).iloc[0]
                rows.append(dict(
                    scenario_id=f"NKD=F|sweep_recovery_status|{sens}|{b.event_id}|expired|{u.event_id}",
                    scenario_type="sweep_recovery_status", sensitivity=sens,
                    direction=b.direction, stage="expired", origin_time=b.event_time,
                    confirmed_at=pd.NaT, available_at=u.available_at,
                    required_event_ids=b.event_id, optional_event_ids="",
                    weekly_context_id=context_for(b.event_time),
                    invalidation="reclaim_required_for_reversal",
                    expiry_rule="recovery_horizon_elapsed_without_reclaim",
                ))
    return pd.DataFrame(rows)


def parse_article_coverage(theory_path: Path, req_path: Path) -> list[dict[str, Any]]:
    text = theory_path.read_text(encoding="utf-8")
    source_urls = {n: u for n, u in re.findall(r"^\[\^(\d+)\]:.*?\((https?://[^)]+)\)", text, re.M)}
    image_refs = {"51": "S6", "61": "S3", "83": "S5", "88": "S4", "111": "S2", "114": "S1"}
    rule_map = {
        24: "cisd;mss", 41: "cisd", 61: "liquidity_breach;liquidity_sweep",
        75: "displacement", 81: "swing_low_3bar", 82: "swing_high_3bar",
        87: "ob_candidate_dw", 88: "ob_candidate_dw", 100: "fvg_3bar",
        111: "mss", 114: "fvg_3bar", 115: "ob_candidate_dw",
    }
    implemented_nums = {24, 41, 61, 75, 81, 82, 111, 114}
    shared_nums = {100}
    adaptation_nums = {87, 88, 115}
    rows = []
    for line in text.splitlines():
        m = re.match(r"^\|\s*(\d+)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|", line)
        if not m: continue
        number = int(m.group(1)); title = m.group(2).strip(); handling = m.group(4).strip()
        extra = re.search(r"SMT|Sick Sister|CBDR|Judas|2022|Pennant|Rally Base|TGIF|Weekly Profiles|One Shot|Session|出来高", title + " " + handling, re.I)
        reference = re.search(r"Practical|PDF紹介|Concepts一覧|Abbreviations|Forex Trading", title, re.I)
        if number in implemented_nums: status = "implemented"
        elif number in shared_nums: status = "shared_rule"
        elif number in adaptation_nums: status = "daily_weekly_adaptation"
        elif reference: status = "reference_only"
        elif extra: status = "requires_extra_data"
        elif "派" in handling or "補" in handling: status = "daily_weekly_adaptation"
        else: status = "not_implemented"
        if status == "implemented": limitation = "v0.3 core rule; full article execution conditions remain separate"
        elif status == "shared_rule": limitation = "reuses the ordinary FVG direction rule"
        elif status == "daily_weekly_adaptation": limitation = "daily/weekly adaptation; original lower-timeframe conditions are not claimed"
        elif status == "requires_extra_data": limitation = "requires lower timeframe, session/time, another symbol, or volume data"
        elif status == "reference_only": limitation = "reference/education entry; no standalone detector"
        else: limitation = "not implemented in v0.3"
        rows.append(dict(article_number=number, title=title, source_url=source_urls.get(str(number), "source_url_not_recorded"),
                         image_refs=image_refs.get(str(number), "not_recorded_in_attachment"),
                          rule_ids=rule_map.get(number, "not_implemented_v0.3"),
                         source_variant="attached_theory_table_v1", implementation_status=status,
                         required_data="daily_weekly_ohlc" if not extra else "lower_timeframe_or_other_symbol_or_volume_or_time",
                         limitation=limitation, source_handling=handling, requirements_source=str(req_path)))
    return rows


def chart(frame: pd.DataFrame, objects: pd.DataFrame, events: pd.DataFrame, out: Path, title: str, bars: int) -> None:
    x = frame.tail(bars).reset_index(drop=True); offset = len(frame) - len(x)
    fig, ax = plt.subplots(figsize=(22, 11), dpi=150)
    for i, r in x.iterrows():
        color = "#12805c" if r.close >= r.open else "#c23b3b"
        ax.vlines(i, r.low, r.high, color=color, linewidth=0.6, alpha=0.9)
        ax.add_patch(Rectangle((i - 0.34, min(r.open, r.close)), 0.68, max(abs(r.close-r.open), 0.01), facecolor=color, edgecolor=color, linewidth=0.5))
    # The CSV keeps every sensitivity and every historical object.  A chart
    # is a review surface, so show only the primary balanced profile and the
    # most recent 50 zones in the visible window; otherwise 20 years of valid
    # history obscure the candles and make the figure fail visual QA.
    z = objects[(objects.sensitivity == "balanced") & (objects.origin_idx >= offset - 2) & (objects.object_type.isin(["ob_candidate", "fvg"]))].copy()
    z = z.sort_values("confirmed_idx").tail(50)
    for r in z.itertuples(index=False):
        left = max(0, int(r.origin_idx) - offset); right = len(x) - 1
        if not (math.isfinite(finite(r.zone_low)) and math.isfinite(finite(r.zone_high))): continue
        col = "#2471a3" if r.direction == "bullish" else "#a93226"
        alpha = 0.12 if r.object_type == "fvg" else 0.20
        ax.add_patch(Rectangle((left, r.zone_low), max(1, right-left), r.zone_high-r.zone_low, facecolor=col, edgecolor=col, alpha=alpha, linewidth=0.8))
    e = events[(events.sensitivity == "balanced") & (events.event_idx >= offset) & (events.rule_id.isin(["mss", "liquidity_sweep"]))].sort_values("event_idx")
    label_events = e.tail(30)
    for r in e.itertuples(index=False):
        xx = int(r.event_idx) - offset
        ax.axvline(xx, color="#7d3c98" if r.rule_id == "mss" else "#d68910", alpha=0.18, linewidth=0.6)
    for label_no, r in enumerate(label_events.itertuples(index=False)):
        xx = int(r.event_idx) - offset
        # Stagger the labels in axis coordinates so a cluster of recent
        # events remains legible instead of being painted at one y-value.
        ax.text(xx, 0.995 - 0.028 * (label_no % 4), r.rule_id.upper(),
                transform=ax.get_xaxis_transform(), rotation=90, va="top", ha="right",
                fontsize=6, color="#4a235a", clip_on=False)
    ax.set_title(title, loc="left", fontsize=16, weight="bold")
    ax.set_ylabel("NKD=F price"); ax.set_xlabel("bars; zones are candidates, not trade signals")
    ax.grid(alpha=0.18); fig.tight_layout(); fig.savefig(out); plt.close(fig)


def synthetic_tests() -> pd.DataFrame:
    def fixture(rows, dates=None):
        f = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
        f["datetime"] = pd.to_datetime(dates) if dates is not None else pd.bdate_range("2020-01-06", periods=len(f))
        f["volume"] = 100; f["bar_idx"] = np.arange(len(f))
        f = add_features(f); f["atr14"] = 1.0
        return f

    results = []

    def test(name, passed, rule, observed=""):
        results.append(dict(test_id=name, passed=bool(passed), rule=rule, observed=str(observed)))

    f = fixture([(100, 101, 99, 100), (100, 104, 100, 103), (103, 105, 102, 104)])
    z = fvg_objects(f, "balanced", "1D")
    test("positive_bullish_fvg", len(z) == 1 and z.direction.iloc[0] == "bullish" and z.zone_low.iloc[0] == 101 and z.zone_high.iloc[0] == 102,
         "High_1 < Low_3", z[["direction", "zone_low", "zone_high"]].to_dict("records"))
    nf = fixture([(100, 103, 99, 100), (100, 104, 100, 103), (103, 105, 102, 104)])
    nz = fvg_objects(nf, "balanced", "1D")
    test("overlapping_wicks_not_fvg", len(nz) == 0, "non-positive gap rejected", len(nz))
    mf = f.copy(); mf["open"], mf["close"] = 300 - f.open, 300 - f.close; mf["high"], mf["low"] = 300 - f.low, 300 - f.high
    mz = fvg_objects(add_features(mf), "balanced", "1D")
    test("bullish_bearish_mirror_geometry", len(mz) == 1 and mz.direction.iloc[0] == "bearish" and mz.zone_low.iloc[0] == 198 and mz.zone_high.iloc[0] == 199,
         "bullish/bearish FVG mirror", mz[["direction", "zone_low", "zone_high"]].to_dict("records"))

    f = fixture([(100, 101, 99, 100), (100, 104, 99, 103)])
    dz = displacement_events(f, "strict", "1D")
    test("strict_one_bar_displacement_is_defined", len(dz) >= 1 and dz.metric_efficiency.le(1 + 1e-12).all(),
         "one-bar path includes prior close", dz.to_dict("records"))
    f = fixture([(100, 101, 99, 100), (100, 102, 99, 101), (101, 103, 100, 102)])
    dz = displacement_events(f, "balanced", "1D")
    test("displacement_efficiency_bounded", len(dz) > 0 and dz.metric_efficiency.le(1 + 1e-12).all(),
         "efficiency cannot exceed one", dz.metric_efficiency.tolist())

    f = fixture([(100, 101, 100, 100), (100, 101, 99, 100), (100, 101, 100, 100),
                 (100, 101, 99, 100), (100, 101, 100, 100)])
    piv = pd.DataFrame([
        dict(pivot_id="PL:a", pivot_type="low", origin_idx=0, confirmed_idx=0, price=100.0),
        dict(pivot_id="PL:b", pivot_type="low", origin_idx=2, confirmed_idx=2, price=100.0),
    ])
    se = sweep_events(f, piv, "balanced", "1D")
    test("all_confirmed_pivots_are_scanned", int((se.event_type == "sweep_same_bar").sum()) == 2,
         "no early loop break", se[["event_type", "parent_ids"]].to_dict("records"))
    se = sweep_events(f.iloc[:2].copy(), piv.iloc[:1].copy(), "strict", "1D")
    test("same_bar_sweep_is_distinct", len(se) == 2 and "sweep_same_bar" in set(se.event_type),
         "breach plus same-bar recovery", se[["event_type", "stage"]].to_dict("records"))
    f = fixture([(100, 101, 100, 100), (100, 101, 99, 98)])
    se = sweep_events(f, piv.iloc[:1].copy(), "strict", "1D")
    test("unreclaimed_horizon_is_explicit", "unreclaimed_at_horizon" in set(se.event_type) and not "sweep_same_bar" in set(se.event_type),
         "strict waits only on breach bar", se[["event_type", "stage"]].to_dict("records"))

    f = fixture([(110, 111, 107, 108), (108, 109, 104, 105), (105, 106, 103, 104), (104, 113, 103, 111)])
    ce = cisd_events(f, "balanced", "1D")
    test("cisd_uses_full_delivery_leg_open", len(ce) == 1 and int(ce.event_idx.iloc[0]) == 3,
         "close must cross the first open of the contiguous delivery leg", ce[["event_idx", "parent_ids"]].to_dict("records"))
    f = fixture([(100, 101, 99, 100), (100, 112, 99, 111)])
    piv = pd.DataFrame([dict(pivot_id="PH:a", pivot_type="high", origin_idx=0, confirmed_idx=0, price=110.0)])
    disp = pd.DataFrame([dict(object_id="D:bull", direction="bullish", origin_idx=1, confirmed_idx=1)])
    me = structure_events(f, piv, disp, "balanced", "1D")
    test("mss_requires_matching_displacement_direction", len(me) == 1 and me.direction.iloc[0] == "bullish" and me.observed_idx.iloc[0] == 1,
         "close cross and displacement direction agree", me[["event_type", "direction", "observed_idx"]].to_dict("records"))
    wrong = disp.copy(); wrong["direction"] = "bearish"
    test("wrong_displacement_direction_cannot_validate_mss", structure_events(f, piv, wrong, "balanced", "1D").empty,
         "bearish displacement cannot validate bullish high break")

    f = fixture([(110, 111, 109, 100), (100, 112, 99, 111), (111, 113, 110, 112)])
    disp = pd.DataFrame([dict(object_id="D:bull", direction="bullish", origin_idx=1, confirmed_idx=2,
                              confirmed_at=f.datetime.iloc[2], available_at=f.datetime.iloc[2],
                              metric_net_atr=11.0, metric_efficiency=1.0, metric_path_atr=11.0)])
    oz = ob_objects(f, disp, "balanced", "1D")
    test("ob_origin_close_break_is_required_for_confirmed_adaptation", len(oz) == 1 and oz.evidence_state.iloc[0] == "confirmed",
         "close after opposite origin crosses origin high", oz[["evidence_state", "origin_boundary_cross_idx"]].to_dict("records"))
    test("lower_timeframe_requirements_stay_unknown", len(oz) == 1 and "lower_timeframe" in oz.unknown_checks.iloc[0],
         "daily/weekly OHLC cannot invent lower-timeframe confirmation")

    f = fixture([(105, 106, 104, 105), (97, 99, 96, 97), (97, 99, 96, 98)])
    z = pd.DataFrame([dict(object_id="F:gap", object_type="fvg", timeframe="1D", sensitivity="balanced", direction="bullish",
                           origin_idx=0, confirmed_idx=0, origin_time=f.datetime.iloc[0], confirmed_at=f.datetime.iloc[0], available_at=f.datetime.iloc[0],
                           zone_low=100.0, zone_high=102.0, unknown_checks="")])
    _, le = lifecycle_events(f, z)
    test("gap_through_is_not_full_fill", "fvg_gap_through" in set(le.event_type) and "fvg_full_fill" not in set(le.event_type),
         "unobserved path across a zone is separate", le[["event_type", "event_idx"]].to_dict("records"))
    f = fixture([(103, 104, 102, 103), (103, 104, 99, 99), (99, 102, 98, 101)])
    z.object_id = "O:reclaim"; z.object_type = "ob_candidate"
    _, le = lifecycle_events(f, z)
    test("delayed_ob_reclaim_is_recorded", "ob_reclaim" in set(le.event_type) and int(le.loc[le.event_type == "ob_reclaim", "event_idx"].iloc[0]) == 2,
          "wick breach and later close recovery are separate", le[["event_type", "event_idx"]].to_dict("records"))
    z.available_at = f.datetime.iloc[1]
    z.confirmed_idx = 0
    _, le = lifecycle_events(f, z)
    test("lifecycle_starts_after_object_availability", len(le) > 0 and int(le.event_idx.min()) > 1,
         "future lifecycle events cannot precede usable object", le[["event_type", "event_idx"]].to_dict("records"))

    dates = ["2020-01-06", "2020-01-07", "2020-01-08", "2020-01-09"]
    f = fixture([(100, 101, 99, 100)] * 4, dates)
    w = build_weekly(f, pd.Timestamp("2020-01-09"))
    test("week_without_friday_is_not_certified", not bool(w.complete.iloc[0]) and w.completion_state.iloc[0] == "week_to_date",
         "without exchange calendar, no expected-Friday confirmation", w[["observed_sessions", "complete", "completion_state"]].to_dict("records"))
    dates = ["2020-01-06", "2020-01-07", "2020-01-08", "2020-01-09", "2020-01-10"]
    f = fixture([(100, 101, 99, 100)] * 5, dates)
    w = build_weekly(f, pd.Timestamp("2020-01-10"))
    test("week_with_friday_is_certified", bool(w.complete.iloc[0]) and w.completion_state.iloc[0] == "confirmed",
          "expected-Friday session observed", w[["observed_sessions", "complete", "completion_state"]].to_dict("records"))
    f = fixture([(100, 101, 99, 100)] * 4, ["2020-01-06", "2020-01-07", "2020-01-08", "2020-01-09"])
    w = build_weekly(f, pd.Timestamp("2020-01-10"))
    test("unknown_friday_gap_is_not_certified", not bool(w.complete.iloc[0]) and w.completion_state.iloc[0] == "short_or_missing_week_unknown",
         "missing Friday is not silently treated as a holiday", w[["observed_sessions", "calendar_reason", "complete", "completion_state"]].to_dict("records"))
    f = fixture([(100, 101, 99, 100)] * 4, ["2020-04-06", "2020-04-07", "2020-04-08", "2020-04-09"])
    w = build_weekly(f, pd.Timestamp("2020-04-10"))
    test("known_good_friday_gap_can_be_certified", bool(w.complete.iloc[0]) and w.calendar_reason.iloc[0] == "known_friday_closure_proxy",
         "known exchange-closure proxy is distinct from unknown provider gap", w[["observed_sessions", "calendar_reason", "complete"]].to_dict("records"))
    return pd.DataFrame(results)


def run_detector(daily: pd.DataFrame, as_of: pd.Timestamp, sensitivities: list[str] | None = None) -> dict[str, Any]:
    daily = daily[daily.datetime <= as_of].copy().reset_index(drop=True)
    weekly_all = build_weekly(daily, as_of)
    weekly = weekly_all[weekly_all.complete].copy().reset_index(drop=True)
    result_objects = []; result_events = []; pivots_map = {}
    profile_names = list(PROFILES) if sensitivities is None else list(sensitivities)
    for tf, frame in [("1D", daily), ("1W", weekly)]:
        for sens in profile_names:
            o, e, p = lane(frame, tf, sens)
            if len(o): result_objects.append(o)
            if len(e): result_events.append(e)
            pivots_map[tf] = p
    objects = pd.concat(result_objects, ignore_index=True, sort=False) if result_objects else pd.DataFrame()
    events = pd.concat(result_events, ignore_index=True, sort=False) if result_events else pd.DataFrame()
    # Context is a primary-profile review surface.  Keeping it on balanced
    # zones prevents strict/exploratory duplicates from changing distances.
    context_objects = objects[(objects.timeframe == "1W") & (objects.sensitivity == "balanced")] if len(objects) else pd.DataFrame()
    contexts = weekly_contexts(daily, weekly, context_objects, pivots_map.get("1W", pd.DataFrame()))
    scenario_rows = scenarios(daily, events[events.timeframe == "1D"] if len(events) else pd.DataFrame(), contexts)
    return dict(daily=daily, weekly=weekly, weekly_all=weekly_all, objects=objects, events=events, contexts=contexts, scenarios=scenario_rows)


def write_df(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--theory", type=Path, default=DEFAULT_THEORY)
    ap.add_argument("--requirements", type=Path, default=DEFAULT_REQUIREMENTS)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--as-of", default=None)
    args = ap.parse_args()
    daily, data_qa = load_daily(args.input, args.as_of)
    as_of = pd.Timestamp(args.as_of).normalize() if args.as_of else daily.datetime.max()
    r = run_detector(daily, as_of)
    args.output.mkdir(parents=True, exist_ok=True)
    write_df(r["objects"], args.output / "objects.csv")
    write_df(r["events"], args.output / "events.csv")
    write_df(r["contexts"], args.output / "weekly_contexts.csv")
    write_df(r["scenarios"], args.output / "scenarios.csv")
    write_df(synthetic_tests(), args.output / "validation_tests.csv")
    near = r["objects"][r["objects"].get("failed_checks", "").fillna("").astype(str).ne("")].copy() if len(r["objects"]) else pd.DataFrame()
    write_df(near.head(250), args.output / "near_miss_examples.csv")
    # Compare a run that receives the full source with a run receiving only its
    # historical prefix. Future rows may add later events, but may not alter
    # immutable objects or events already available by the cutoff.
    cut_indices = [int(len(daily) * q) for q in (0.25, 0.50, 0.75)]
    cuts = [daily.datetime.iloc[i].normalize() for i in cut_indices]
    def stable_signature(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=cols)
        x = df.reindex(columns=cols).copy()
        numeric_columns = {
            "origin_idx", "confirmed_idx", "event_idx", "observed_idx", "metric_bars",
            "zone_low", "zone_high", "body_low", "body_high", "range_low", "range_high",
            "range_mid", "weekly_ob_distance_atr", "weekly_fvg_distance_atr",
        }
        for c in cols:
            # CSV round-trips and vectorized calculations can differ by a few
            # ulps even when the causal result is identical.  Compare numeric
            # fields at a fixed tolerance; compare timestamps/IDs exactly.
            if c in numeric_columns or pd.api.types.is_numeric_dtype(x[c]):
                numeric = pd.to_numeric(x[c], errors="coerce")
                x[c] = numeric.map(lambda v: "" if pd.isna(v) else f"{float(v):.8f}")
            elif c.endswith("_at") or c.endswith("_time") or c == "datetime":
                dates = pd.to_datetime(x[c], errors="coerce")
                x[c] = dates.map(lambda v: "" if pd.isna(v) else v.isoformat())
            else:
                x[c] = x[c].fillna("").astype(str)
        return x.sort_values(cols).reset_index(drop=True)
    obj_cols = [c for c in ["object_id", "timeframe", "object_type", "direction", "sensitivity", "origin_idx", "origin_time", "available_at", "zone_low", "zone_high"] if c in r["objects"]]
    evt_cols = [c for c in ["event_id", "timeframe", "event_type", "direction", "sensitivity", "event_idx", "event_time", "available_at", "object_id", "stage"] if c in r["events"]]
    context_cols = [c for c in ["datetime", "weekly_context_id", "weekly_available_at", "weekly_direction", "weekly_position", "range_low", "range_high"] if c in r["contexts"]]
    scenario_cols = [c for c in ["scenario_id", "scenario_type", "sensitivity", "direction", "stage", "origin_time", "confirmed_at", "available_at", "required_event_ids", "optional_event_ids", "weekly_context_id", "expiry_rule"] if c in r["scenarios"]]
    prefix_rows = []
    mutable_event_types = {"liquidity_recovery_pending", "unreclaimed_at_horizon"}
    # Run each prefix independently.  Keeping three full detector results
    # alongside the full run can multiply memory use without adding evidence;
    # the invariance contract needs only the historical signatures.
    for cut in cuts:
        prefix_asof = run_detector(daily[daily.datetime <= cut].copy(), cut, sensitivities=["balanced"])
        full_objects = r["objects"][(pd.to_datetime(r["objects"]["available_at"]) <= cut) & (r["objects"].sensitivity == "balanced")]
        prefix_objects = prefix_asof["objects"][(pd.to_datetime(prefix_asof["objects"]["available_at"]) <= cut) & (prefix_asof["objects"].sensitivity == "balanced")]
        full_events = r["events"][(pd.to_datetime(r["events"]["available_at"]) <= cut) & (r["events"].sensitivity == "balanced") & (~r["events"].event_type.isin(mutable_event_types))]
        prefix_events = prefix_asof["events"][(pd.to_datetime(prefix_asof["events"]["available_at"]) <= cut) & (prefix_asof["events"].sensitivity == "balanced") & (~prefix_asof["events"].event_type.isin(mutable_event_types))]
        full_contexts = r["contexts"][pd.to_datetime(r["contexts"]["datetime"]) <= cut]
        prefix_contexts = prefix_asof["contexts"][pd.to_datetime(prefix_asof["contexts"]["datetime"]) <= cut]
        full_scenarios = r["scenarios"][(pd.to_datetime(r["scenarios"]["available_at"]) <= cut) & (r["scenarios"].sensitivity == "balanced")]
        prefix_scenarios = prefix_asof["scenarios"][(pd.to_datetime(prefix_asof["scenarios"]["available_at"]) <= cut) & (prefix_asof["scenarios"].sensitivity == "balanced")]
        prefix_rows.extend([
            {"test_id": "future_append_does_not_change_object_history", "cutoff": str(cut.date()), "passed": stable_signature(full_objects, obj_cols).equals(stable_signature(prefix_objects, obj_cols)), "compared": "objects"},
            {"test_id": "future_append_does_not_change_event_history", "cutoff": str(cut.date()), "passed": stable_signature(full_events, evt_cols).equals(stable_signature(prefix_events, evt_cols)), "compared": "events"},
            {"test_id": "future_append_does_not_change_weekly_context_history", "cutoff": str(cut.date()), "passed": stable_signature(full_contexts, context_cols).equals(stable_signature(prefix_contexts, context_cols)), "compared": "weekly_contexts"},
            {"test_id": "future_append_does_not_change_scenario_history", "cutoff": str(cut.date()), "passed": stable_signature(full_scenarios, scenario_cols).equals(stable_signature(prefix_scenarios, scenario_cols)), "compared": "scenarios"},
        ])
        del prefix_asof, full_objects, prefix_objects, full_events, prefix_events, full_contexts, prefix_contexts, full_scenarios, prefix_scenarios
        gc.collect()
    prefix_tests = pd.DataFrame(prefix_rows)
    write_df(prefix_tests, args.output / "prefix_invariance_tests.csv")
    coverage = parse_article_coverage(args.theory, args.requirements)
    (args.output / "article_coverage.json").write_text(json.dumps(coverage, ensure_ascii=False, indent=2), encoding="utf-8")
    diag = {
        "object_counts": r["objects"].groupby(["timeframe", "object_type", "direction", "sensitivity"], dropna=False).size().reset_index(name="n").to_dict("records") if len(r["objects"]) else [],
        "event_counts": r["events"].groupby(["timeframe", "event_type", "direction", "sensitivity", "stage"], dropna=False).size().reset_index(name="n").to_dict("records") if len(r["events"]) else [],
        "scenario_counts": r["scenarios"].groupby(["scenario_type", "direction", "sensitivity", "stage"], dropna=False).size().reset_index(name="n").to_dict("records") if len(r["scenarios"]) else [],
        "weekly_rows_all": len(r["weekly_all"]), "weekly_rows_confirmed": len(r["weekly"]),
        "article_count": len(coverage), "implemented_or_adapted": sum(x["implementation_status"] in {"implemented", "daily_weekly_adaptation", "shared_rule"} for x in coverage),
        "requires_extra_data": sum(x["implementation_status"] == "requires_extra_data" for x in coverage),
        "tests_passed": int(synthetic_tests().passed.sum()) + int(prefix_tests.passed.sum()), "tests_total": len(synthetic_tests()) + len(prefix_tests),
        "prefix_invariance_tests": prefix_tests.to_dict("records"),
        "coverage_status_counts": pd.Series([x["implementation_status"] for x in coverage]).value_counts().to_dict(),
    }
    (args.output / "diagnostics.json").write_text(json.dumps(diag, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")
    chart(r["daily"], r["objects"][r["objects"].timeframe == "1D"] if len(r["objects"]) else r["objects"], r["events"][r["events"].timeframe == "1D"] if len(r["events"]) else r["events"], args.output / "nq_ict_daily_last_2y.png", f"NKD=F ICT detector | daily | as of {as_of.date()} | balanced recent zones/events", 520)
    chart(r["weekly"], r["objects"][r["objects"].timeframe == "1W"] if len(r["objects"]) else r["objects"], r["events"][r["events"].timeframe == "1W"] if len(r["events"]) else r["events"], args.output / "nq_ict_weekly_last_5y.png", f"NKD=F ICT detector | confirmed weekly | as of {as_of.date()} | balanced recent zones/events", 260)
    manifest = {
        "status": "completed_detector_and_acceptance_checks",
        "script_version": SCRIPT_VERSION, "script": str(Path(__file__).resolve()), "script_sha256": sha256(Path(__file__)),
        "input": data_qa, "as_of": str(as_of.date()), "manual_inputs_used": [], "old_research_imported": False,
        "theory_path": str(args.theory), "theory_sha256": sha256(args.theory), "requirements_path": str(args.requirements), "requirements_sha256": sha256(args.requirements),
        "profiles": PROFILES, "counts": diag,
        "interpretation": "Detection acceptance and causal timestamp checks are not return prediction or trading validation.",
    }
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")
    report = f"""# NKD=F ICT検知器 v0.3 検証レポート

判定時点: {as_of.date()}  
実装: `{SCRIPT_VERSION}`  
入力: `{args.input.resolve()}`  
期間: {data_qa['start']}〜{data_qa['end']}、{data_qa['rows']:,}日足  
判定: **検知器としての初回実行は完了。価格予測力・売買有効性は未検証。**

## 実装した範囲

添付要件の段階1を新規コードで実装した。日足と確定週足を別レーンにし、3本足Swing、確定Pivot、参照レンジの一部、通常FVG、Displacement、最後の逆色足を起点とするOB候補、Liquidity Sweep、CISD、終値突破を伴うMSS/BOS、OB/FVGの接触・充足・無効化・リクレーム履歴を出力する。日足のSweep→MSSシナリオも、形成途中・確認済み・期限切れを追記レコードとして分けて保存した。

OBはAETHER/LuxAlgo互換の旧定義を呼び出していない。body帯と起点足のwick帯を別属性で保存し、下位足のImbalance/MSSは日足・週足OHLCからは判定せず、`unknown`として残した。通常FVGの正の非重複条件、MSSの終値突破条件、出来事の確認時点は緩めていない。OBのライフサイクルは、境界条件に使った足を含む利用可能時点の後から走査した。

感度は`strict`、`balanced`、`exploratory`をすべて生成した。閾値は添付要件の初期提案であり、NQの最適化結果ではない。検知件数を確保するための自動緩和は行っていない。

## データ品質

| 項目 | 結果 |
|---|---:|
| 日足行数 | {data_qa['rows']:,} |
| 重複日 | {data_qa['duplicate_dates']} |
| 必須列の欠損セル | {data_qa['null_required_cells']} |
| 不正OHLC行 | {data_qa['invalid_ohlc_rows']} |
| 週末行 | {data_qa['weekend_rows']} |
| 出来高0行 | {data_qa['zero_volume_rows']} |
| 確定週足／週足全体 | {len(r['weekly'])} / {len(r['weekly_all'])} |

入力は保存済みの連続先物代理データであり、限月接続の経済的意味を検証したものではない。週足の金曜日欠損は、組み込みの祝日休場プロキシに一致する場合だけ短縮週として認定し、それ以外は`short_or_missing_week_unknown`として週足パターンから除外した。組み込みカレンダーはCMEの完全な公式カレンダーではないため、日付・セッションの追加監査が必要である。

## 検出結果の読み方

`objects.csv`は形状・帯・確認時点、`events.csv`は形成、確認、接近、接触、充足、突破、回復などの出来事、`scenarios.csv`は親イベントをつないだ部分シナリオである。`available_at`以前の情報から後の出来事を事前に描いたものではない。形成イベントの後に確認イベントが別行で存在する。

`article_coverage.json`は、94記事を94個の同時AND条件にせず、実装済み・日週足への派生・追加データ必要・未実装等へ分類した台帳である。記事の画像は形と順序を決める参照資料であり、画像から正確なOHLCを発明していない。

## 受入テスト

合成ルールテストは`validation_tests.csv`、未来足追加前後の不変性テストは`prefix_invariance_tests.csv`に保存し、合計{diag['tests_passed']}/{diag['tests_total']}件を通過扱いとした。これは実データでの性能を示さない。日足の過去時点を再計算するprefix不変性、同じ入力の再現性、確定週足のみの日足結合は、今後も自動テストとして保持する。

## まだ言えないこと

`confirmed`は検知器の条件が揃ったことを表し、その後の反発・利益・機関注文の存在を意味しない。候補数や記事画像との目視一致率は、今回まだ人手ラベルを作って測定していない。したがって、ICTパターンの再現率・適合率、NQでの将来方向の確率、勝率、期待値は未報告である。

次段階では、まずNQから検知結果に依存しない開発用・固定評価用の目視標本を作り、同じ形の適合率・再現率・確認遅延を測る。その後、ルールと時点を凍結し、到達・反転・通過・停滞を同一時計の競合リスクで定義して、単純な基準モデルと時系列ウォークフォワードを比較する。結果を見て条件を追加し続けない。
"""
    (args.output / "validation_report_ja.md").write_text(report, encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "objects": len(r["objects"]), "events": len(r["events"]), "scenarios": len(r["scenarios"]), "articles": len(coverage), "data_qa": data_qa}, ensure_ascii=False, default=json_default))


if __name__ == "__main__":
    main()
