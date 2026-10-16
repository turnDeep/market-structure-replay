"""Causal monthly/weekly hierarchy evaluation for the frozen NQ=F ICT detector.

This is a new research version.  v0.4 is left intact for comparison.  The
script fixes the review findings before testing the monthly extension:

1. post-touch outcomes are competing outcomes (favorable, adverse, neither),
   rather than a binary rate that silently removes neither;
2. formation and touch-close prediction clocks are separate;
3. weekly and monthly bars are certified conservatively as-of each date;
4. structure, bias, and zone lifecycle are point-in-time fields;
5. FVG/OB order is based on the target's formation time, not merely on a
   later touch time;
6. same-side controls are evaluated on their own full risk sets, not only when
   both actual and control happen to be reached; and
7. monthly structure is compared incrementally with the geometry/environment
   and weekly models in expanding time-series walk-forward evaluation.

This is an observational OHLCV study of a continuous-futures proxy.  It is not
proof of institutional orders, ICT causality, or a trading recommendation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import nq_ict_detection_20260911_v03 as detector
import nq_ict_evaluation_20260911_v04 as v4
import nq_ict_evaluation_20260911_v03 as ev3


SCRIPT_VERSION = "nq-ict-evaluation-v0.5"
DEFAULT_INPUT = Path("nq_20y_ordered_validation_20260909/input/NQ_F_1d_strict.csv")
DEFAULT_DETECTOR_OUTPUT = Path("nq_ict_detection_20260911_v03")
DEFAULT_OUTPUT = Path("nq_ict_evaluation_20260911_v05")

ARRIVAL_HORIZON = {"1D": 20, "1W": 60}
POST_TOUCH_HORIZON = 10
BARRIER_ATR = 1.0
MIN_TRAIN = 60
MIN_TEST = 12
POST_CLASSES = ["adverse_first", "favorable_first", "neither_at_horizon"]

BASE_FEATURES = [
    "return_5", "return_20", "atr_pct", "volume_z20", "sma_gap20", "sma_gap50",
    "distance_atr", "width_atr", "mid_distance_atr", "age_days",
    "object_is_ob", "evidence_confirmed",
]
POST_BASE_FEATURES = [
    "post_return_5", "post_return_20", "post_atr_pct", "post_volume_z20",
    "post_sma_gap20", "post_sma_gap50", "post_distance_atr", "post_width_atr",
    "post_mid_distance_atr", "post_intrusion_atr", "post_bar_range_atr",
    "post_close_location", "object_is_ob", "evidence_confirmed",
]

CONTEXT_NUMERIC = [
    "trend_up", "trend_down", "trend_mixed", "trend_unknown",
    "integrity_intact", "integrity_broken", "integrity_unknown",
    "bias_up", "bias_down", "bias_neutral", "position_premium",
    "position_discount", "position_out_of_range", "position_unknown",
    "range_width_atr", "ob_distance_atr", "fvg_distance_atr",
    "active_ob_count", "active_fvg_count", "fresh_ob_count", "fresh_fvg_count",
]
WEEKLY_FEATURES = [f"formation_weekly_{x}" for x in CONTEXT_NUMERIC]
MONTHLY_FEATURES = [f"formation_monthly_{x}" for x in CONTEXT_NUMERIC]
POST_WEEKLY_FEATURES = [f"post_weekly_{x}" for x in CONTEXT_NUMERIC]
POST_MONTHLY_FEATURES = [f"post_monthly_{x}" for x in CONTEXT_NUMERIC]
HIERARCHY_FEATURES = [
    "formation_trend_agreement", "formation_bias_agreement",
]
POST_HIERARCHY_FEATURES = [x.replace("formation_", "post_") for x in HIERARCHY_FEATURES]
SEQUENCE_NUMERIC = [
    "sequence_fvg_before_object", "sequence_ob_before_object", "sequence_sweep_before_object",
    "sequence_mss_bos_before_object", "sequence_prior_zone_count", "sequence_same_timestamp_step",
]
POST_SEQUENCE_NUMERIC = [f"post_{x}" for x in SEQUENCE_NUMERIC]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return str(value)


def period_bars(daily: pd.DataFrame, frequency: str) -> pd.DataFrame:
    """Aggregate daily bars without using an incomplete current period."""
    x = daily[["datetime", "open", "high", "low", "close", "volume"]].copy()
    p = x.datetime.dt.to_period(frequency)
    x["period_end"] = p.dt.end_time.dt.normalize()
    x["period_start"] = p.dt.start_time.dt.normalize()
    out = x.groupby(["period_start", "period_end"], sort=True).agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), volume=("volume", "sum"),
        observed_sessions=("close", "count"), last_session=("datetime", "max"),
    ).reset_index()
    out["datetime"] = out["period_end"]
    out["available_at"] = out["last_session"].dt.normalize()
    out["certified_calendar"] = True
    if frequency == "W-FRI":
        years = list(range(int(out.period_end.dt.year.min()) - 1, int(out.period_end.dt.year.max()) + 2)) if len(out) else []
        closures = set()
        for year in years:
            closures |= detector.known_friday_closures(year, year)
        out["has_expected_friday"] = out.last_session.dt.normalize().eq(out.period_end)
        out["expected_friday_closed"] = out.period_end.dt.date.isin(closures)
        out["certified_calendar"] = out.has_expected_friday | out.expected_friday_closed
    else:
        out["has_expected_friday"] = False
        out["expected_friday_closed"] = False
    out = detector.add_features(out)
    out["bar_idx"] = np.arange(len(out), dtype=int)
    return out


def mechanical_pivots(frame: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Three-bar fractals; a pivot is usable only after the right bar closes."""
    rows = []
    for i in range(1, len(frame) - 1):
        if not bool(frame.certified_calendar.iloc[i - 1:i + 2].all()):
            continue
        hi, lo = float(frame.high.iloc[i]), float(frame.low.iloc[i])
        if hi > float(frame.high.iloc[i - 1]) and hi > float(frame.high.iloc[i + 1]):
            rows.append({
                "pivot_id": f"NQ=F|{timeframe}|PH3|{i}", "timeframe": timeframe,
                "pivot_type": "high", "origin_idx": i, "origin_time": frame.period_end.iloc[i],
                "confirmed_idx": i + 1, "confirmed_at": frame.period_end.iloc[i + 1],
                "available_at": frame.period_end.iloc[i + 1], "price": hi,
            })
        if lo < float(frame.low.iloc[i - 1]) and lo < float(frame.low.iloc[i + 1]):
            rows.append({
                "pivot_id": f"NQ=F|{timeframe}|PL3|{i}", "timeframe": timeframe,
                "pivot_type": "low", "origin_idx": i, "origin_time": frame.period_end.iloc[i],
                "confirmed_idx": i + 1, "confirmed_at": frame.period_end.iloc[i + 1],
                "available_at": frame.period_end.iloc[i + 1], "price": lo,
            })
    return pd.DataFrame(rows)


def structure_at(frame: pd.DataFrame, pivots: pd.DataFrame, daily_row: pd.Series, asof: pd.Timestamp, timeframe: str) -> dict[str, Any]:
    """Return observed structure and a deterministic, separate bias state."""
    t = pd.Timestamp(asof).normalize()
    complete = frame[(frame.period_end < t) & (frame.available_at <= t) & frame.certified_calendar]
    pp = pivots[pd.to_datetime(pivots.available_at).dt.normalize() < t].copy() if len(pivots) else pd.DataFrame()
    if len(complete):
        pp = pp[pp.origin_idx < int(complete.bar_idx.max()) + 1]
    highs = pp[pp.pivot_type == "high"].sort_values("confirmed_idx") if len(pp) else pp
    lows = pp[pp.pivot_type == "low"].sort_values("confirmed_idx") if len(pp) else pp
    trend = "unknown"
    if len(highs) >= 2 and len(lows) >= 2:
        if highs.price.iloc[-1] > highs.price.iloc[-2] and lows.price.iloc[-1] > lows.price.iloc[-2]:
            trend = "up"
        elif highs.price.iloc[-1] < highs.price.iloc[-2] and lows.price.iloc[-1] < lows.price.iloc[-2]:
            trend = "down"
        else:
            trend = "mixed"
    last_high = float(highs.price.iloc[-1]) if len(highs) else np.nan
    last_low = float(lows.price.iloc[-1]) if len(lows) else np.nan
    protected_high, protected_low = last_high, last_low
    if trend == "up" and len(highs):
        prior_lows = lows[lows.origin_idx < int(highs.origin_idx.iloc[-1])]
        if len(prior_lows):
            protected_low = float(prior_lows.price.iloc[-1])
    if trend == "down" and len(lows):
        prior_highs = highs[highs.origin_idx < int(lows.origin_idx.iloc[-1])]
        if len(prior_highs):
            protected_high = float(prior_highs.price.iloc[-1])
    close = fnum(daily_row.close)
    integrity = "unknown"
    if trend == "up" and math.isfinite(protected_low):
        integrity = "intact" if close >= protected_low else "broken"
    elif trend == "down" and math.isfinite(protected_high):
        integrity = "intact" if close <= protected_high else "broken"
    elif trend == "mixed":
        integrity = "unknown"
    range_low, range_high = last_low, last_high
    if not (math.isfinite(range_low) and math.isfinite(range_high) and range_high > range_low):
        position, mid, width_atr = "unknown", np.nan, np.nan
    else:
        mid = (range_low + range_high) / 2.0
        if close < range_low or close > range_high:
            position = "out_of_range"
        elif close < mid:
            position = "discount"
        elif close > mid:
            position = "premium"
        else:
            position = "midpoint"
        atr = fnum(daily_row.atr14)
        width_atr = (range_high - range_low) / atr if math.isfinite(atr) and atr > 0 else np.nan
    # Bias is not the observed trend.  This operational rule treats movement
    # toward the midpoint/other half as a correction and movement with an
    # intact trend from the favorable half as continuation.  It is a testable
    # hypothesis, not an ICT-imposed numeric definition.
    bias = "neutral"
    if trend == "up" and position == "discount":
        bias = "up"
    elif trend == "up" and position == "premium":
        bias = "down"
    elif trend == "down" and position == "premium":
        bias = "down"
    elif trend == "down" and position == "discount":
        bias = "up"
    latest = pd.concat([highs[["available_at", "price"]], lows[["available_at", "price"]]], ignore_index=True) if len(pp) else pd.DataFrame()
    last_pivot_at = latest.available_at.max() if len(latest) else pd.NaT
    age_days = (t - pd.Timestamp(last_pivot_at).normalize()).days if pd.notna(last_pivot_at) else np.nan
    return {
        "timeframe": timeframe, "structure_trend": trend, "structure_integrity": integrity,
        "bias_direction": bias, "position": position, "range_low": range_low,
        "range_high": range_high, "range_mid": mid, "range_width_atr": width_atr,
        "protected_high": protected_high, "protected_low": protected_low,
        "last_pivot_at": last_pivot_at, "structure_age_days": age_days,
        "context_bar_end": complete.period_end.iloc[-1] if len(complete) else pd.NaT,
    }


def zone_lifecycle(zones: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Build an audit-only event history; features use only events <= as-of."""
    if zones.empty:
        return pd.DataFrame(columns=["object_id", "first_touch_at", "first_full_at", "first_close_invalidation_at", "touch_episode_count", "terminal_state"])
    # Use NumPy arrays inside the zone loop.  The previous implementation
    # accessed daily.iloc[i] for every zone/bar pair, which made the 20-year
    # audit unnecessarily slow without changing the event definitions.
    dates = pd.to_datetime(daily.datetime).dt.normalize().to_numpy(dtype="datetime64[ns]")
    highs = pd.to_numeric(daily.high, errors="coerce").to_numpy(float)
    lows = pd.to_numeric(daily.low, errors="coerce").to_numpy(float)
    closes = pd.to_numeric(daily.close, errors="coerce").to_numpy(float)
    rows = []
    for r in zones.itertuples(index=False):
        lo, hi = fnum(r.zone_low), fnum(r.zone_high)
        available = pd.Timestamp(r.available_at).normalize()
        start = int(np.searchsorted(dates, np.datetime64(available, "ns"), side="right"))
        first_touch = first_full = first_invalid = pd.NaT
        touch_count = 0
        if math.isfinite(lo) and math.isfinite(hi) and start < len(daily):
            h, l, c, d = highs[start:], lows[start:], closes[start:], dates[start:]
            hit = (l <= hi) & (h >= lo)
            valid_hit = np.flatnonzero(hit)
            if valid_hit.size:
                first_touch = pd.Timestamp(d[int(valid_hit[0])]).normalize()
                starts = hit & ~np.r_[False, hit[:-1]]
                touch_count = int(starts.sum())
                full = hit & ((l <= lo) if r.direction == "bullish" else (h >= hi))
                valid_full = np.flatnonzero(full)
                if valid_full.size:
                    first_full = pd.Timestamp(d[int(valid_full[0])]).normalize()
            if r.object_type == "ob_candidate":
                invalid = (c < lo) if r.direction == "bullish" else (c > hi)
                valid_invalid = np.flatnonzero(invalid)
                if valid_invalid.size:
                    first_invalid = pd.Timestamp(d[int(valid_invalid[0])]).normalize()
        if r.object_type == "fvg" and pd.notna(first_full):
            terminal = "fully_filled"
        elif r.object_type == "ob_candidate" and pd.notna(first_invalid):
            terminal = "invalidated_close"
        else:
            terminal = "active_at_asof"
        rows.append({"object_id": r.object_id, "timeframe": r.timeframe, "object_type": r.object_type,
                     "direction": r.direction, "available_at": available, "first_touch_at": first_touch,
                     "first_full_at": first_full, "first_close_invalidation_at": first_invalid,
                     "touch_episode_count": touch_count, "terminal_state": terminal})
    return pd.DataFrame(rows)


def active_zone_stats(zones: pd.DataFrame, life: pd.DataFrame, asof: pd.Timestamp, close: float, atr: float, kind: str) -> dict[str, Any]:
    if zones.empty or not (math.isfinite(close) and math.isfinite(atr) and atr > 0):
        return {"distance_atr": np.nan, "active_count": 0, "fresh_count": 0}
    t = pd.Timestamp(asof).normalize()
    z = zones[(zones.object_type == kind) & (pd.to_datetime(zones.available_at).dt.normalize() <= t)].copy()
    if z.empty:
        return {"distance_atr": np.nan, "active_count": 0, "fresh_count": 0}
    m = life[life.object_id.isin(z.object_id)].set_index("object_id")
    active_ids, fresh = [], 0
    for r in z.itertuples(index=False):
        e = m.loc[r.object_id] if r.object_id in m.index else None
        filled = e is not None and pd.notna(e.first_full_at) and pd.Timestamp(e.first_full_at).normalize() <= t and kind == "fvg"
        invalid = e is not None and pd.notna(e.first_close_invalidation_at) and pd.Timestamp(e.first_close_invalidation_at).normalize() <= t and kind == "ob_candidate"
        if filled or invalid:
            continue
        active_ids.append(r.object_id)
        if e is None or pd.isna(e.first_touch_at) or pd.Timestamp(e.first_touch_at).normalize() > t:
            fresh += 1
    if not active_ids:
        return {"distance_atr": np.nan, "active_count": 0, "fresh_count": 0}
    zz = z[z.object_id.isin(active_ids)]
    dist = np.minimum(np.abs(zz.zone_low.to_numpy(float) - close), np.abs(zz.zone_high.to_numpy(float) - close))
    inside = (zz.zone_low.to_numpy(float) <= close) & (zz.zone_high.to_numpy(float) >= close)
    dist[inside] = 0.0
    return {"distance_atr": float(np.min(dist) / atr), "active_count": len(active_ids), "fresh_count": fresh}


def _structure_series(frame: pd.DataFrame, pivots: pd.DataFrame, daily: pd.DataFrame,
                      timeframe: str) -> list[dict[str, Any]]:
    """Vectorized equivalent of structure_at, aligned to each daily date."""
    dates = pd.to_datetime(daily.datetime).dt.normalize().to_numpy(dtype="datetime64[ns]")
    closes = pd.to_numeric(daily.close, errors="coerce").to_numpy(float)
    atrs = pd.to_numeric(daily.atr14, errors="coerce").to_numpy(float)
    ends = pd.to_datetime(frame.period_end, errors="coerce").to_numpy(dtype="datetime64[ns]")
    avail = pd.to_datetime(frame.available_at, errors="coerce").dt.normalize().to_numpy(dtype="datetime64[ns]")
    cert = frame.certified_calendar.to_numpy(bool)
    bar_idx = frame.bar_idx.to_numpy(int)
    if len(pivots):
        p_avail = pd.to_datetime(pivots.available_at, errors="coerce").dt.normalize().to_numpy(dtype="datetime64[ns]")
        p_origin = pd.to_numeric(pivots.origin_idx, errors="coerce").to_numpy(float)
        p_confirm = pd.to_numeric(pivots.confirmed_idx, errors="coerce").to_numpy(float)
        p_price = pd.to_numeric(pivots.price, errors="coerce").to_numpy(float)
        p_type = pivots.pivot_type.astype(str).to_numpy()
    else:
        p_avail = np.array([], dtype="datetime64[ns]")
        p_origin = p_confirm = p_price = np.array([], dtype=float)
        p_type = np.array([], dtype=str)
    result = []
    for i, t64 in enumerate(dates):
        complete = (ends < t64) & (avail <= t64) & cert
        complete_idx = np.flatnonzero(complete)
        max_bar = int(bar_idx[complete_idx].max()) if complete_idx.size else -1
        mask = (p_avail < t64) & (p_origin <= max_bar)
        hidx = np.flatnonzero(mask & (p_type == "high"))
        lidx = np.flatnonzero(mask & (p_type == "low"))
        hidx = hidx[np.argsort(p_confirm[hidx])] if hidx.size else hidx
        lidx = lidx[np.argsort(p_confirm[lidx])] if lidx.size else lidx
        trend = "unknown"
        if hidx.size >= 2 and lidx.size >= 2:
            if p_price[hidx[-1]] > p_price[hidx[-2]] and p_price[lidx[-1]] > p_price[lidx[-2]]:
                trend = "up"
            elif p_price[hidx[-1]] < p_price[hidx[-2]] and p_price[lidx[-1]] < p_price[lidx[-2]]:
                trend = "down"
            else:
                trend = "mixed"
        last_high = float(p_price[hidx[-1]]) if hidx.size else np.nan
        last_low = float(p_price[lidx[-1]]) if lidx.size else np.nan
        protected_high, protected_low = last_high, last_low
        if trend == "up" and hidx.size:
            prior = lidx[p_origin[lidx] < p_origin[hidx[-1]]]
            if prior.size:
                protected_low = float(p_price[prior[-1]])
        if trend == "down" and lidx.size:
            prior = hidx[p_origin[hidx] < p_origin[lidx[-1]]]
            if prior.size:
                protected_high = float(p_price[prior[-1]])
        close = closes[i]
        integrity = "unknown"
        if trend == "up" and math.isfinite(protected_low):
            integrity = "intact" if close >= protected_low else "broken"
        elif trend == "down" and math.isfinite(protected_high):
            integrity = "intact" if close <= protected_high else "broken"
        range_low, range_high = last_low, last_high
        if not (math.isfinite(range_low) and math.isfinite(range_high) and range_high > range_low):
            position, mid, width_atr = "unknown", np.nan, np.nan
        else:
            mid = (range_low + range_high) / 2.0
            if close < range_low or close > range_high:
                position = "out_of_range"
            elif close < mid:
                position = "discount"
            elif close > mid:
                position = "premium"
            else:
                position = "midpoint"
            width_atr = (range_high - range_low) / atrs[i] if math.isfinite(atrs[i]) and atrs[i] > 0 else np.nan
        bias = "neutral"
        if trend == "up" and position == "discount":
            bias = "up"
        elif trend == "up" and position == "premium":
            bias = "down"
        elif trend == "down" and position == "premium":
            bias = "down"
        elif trend == "down" and position == "discount":
            bias = "up"
        last_pivot_at = pd.Timestamp(p_avail[mask].max()).normalize() if mask.any() else pd.NaT
        last_age = (pd.Timestamp(t64).normalize() - last_pivot_at).days if pd.notna(last_pivot_at) else np.nan
        context_end = pd.Timestamp(ends[complete_idx[-1]]).normalize() if complete_idx.size else pd.NaT
        result.append({
            "timeframe": timeframe, "structure_trend": trend, "structure_integrity": integrity,
            "bias_direction": bias, "position": position, "range_low": range_low,
            "range_high": range_high, "range_mid": mid, "range_width_atr": width_atr,
            "protected_high": protected_high, "protected_low": protected_low,
            "last_pivot_at": last_pivot_at, "structure_age_days": last_age,
            "context_bar_end": context_end,
        })
    return result


def _zone_stats_series(zones: pd.DataFrame, life: pd.DataFrame, daily: pd.DataFrame,
                       kind: str) -> list[dict[str, Any]]:
    """Compute active/fresh counts and nearest distance for all dates at once."""
    dates = pd.to_datetime(daily.datetime).dt.normalize().to_numpy(dtype="datetime64[ns]")
    closes = pd.to_numeric(daily.close, errors="coerce").to_numpy(float)
    atrs = pd.to_numeric(daily.atr14, errors="coerce").to_numpy(float)
    if zones.empty:
        return [{"distance_atr": np.nan, "active_count": 0, "fresh_count": 0} for _ in dates]
    z = zones.copy().reset_index(drop=True)
    lo = pd.to_numeric(z.zone_low, errors="coerce").to_numpy(float)
    hi = pd.to_numeric(z.zone_high, errors="coerce").to_numpy(float)
    z_avail = pd.to_datetime(z.available_at, errors="coerce").dt.normalize().to_numpy(dtype="datetime64[ns]")
    lm = life.set_index("object_id").reindex(z.object_id) if len(life) else pd.DataFrame(index=z.object_id)
    first_touch = pd.to_datetime(lm.get("first_touch_at", pd.Series(pd.NaT, index=z.index)), errors="coerce").to_numpy(dtype="datetime64[ns]")
    first_full = pd.to_datetime(lm.get("first_full_at", pd.Series(pd.NaT, index=z.index)), errors="coerce").to_numpy(dtype="datetime64[ns]")
    first_invalid = pd.to_datetime(lm.get("first_close_invalidation_at", pd.Series(pd.NaT, index=z.index)), errors="coerce").to_numpy(dtype="datetime64[ns]")
    valid = np.isfinite(lo) & np.isfinite(hi) & (hi >= lo)
    out = []
    for t64, close, atr in zip(dates, closes, atrs):
        active = valid & (z_avail <= t64)
        if kind == "fvg":
            active &= np.isnat(first_full) | (first_full > t64)
        elif kind == "ob_candidate":
            active &= np.isnat(first_invalid) | (first_invalid > t64)
        if not active.any() or not (math.isfinite(close) and math.isfinite(atr) and atr > 0):
            out.append({"distance_atr": np.nan, "active_count": 0, "fresh_count": 0})
            continue
        fresh = active & (np.isnat(first_touch) | (first_touch > t64))
        d = np.minimum(np.abs(lo[active] - close), np.abs(hi[active] - close))
        inside = (lo[active] <= close) & (hi[active] >= close)
        d[inside] = 0.0
        out.append({"distance_atr": float(np.min(d) / atr), "active_count": int(active.sum()), "fresh_count": int(fresh.sum())})
    return out


def build_contexts(daily: pd.DataFrame, weekly: pd.DataFrame, monthly: pd.DataFrame,
                   weekly_pivots: pd.DataFrame, monthly_pivots: pd.DataFrame,
                   weekly_zones: pd.DataFrame, weekly_life: pd.DataFrame) -> pd.DataFrame:
    rows = []
    dates = pd.to_datetime(daily.datetime).dt.normalize().to_numpy(dtype="datetime64[ns]")
    weekly_states = _structure_series(weekly, weekly_pivots, daily, "1W")
    monthly_states = _structure_series(monthly, monthly_pivots, daily, "1M")
    weekly_ob_stats = _zone_stats_series(weekly_zones, weekly_life, daily, "ob_candidate")
    weekly_fvg_stats = _zone_stats_series(weekly_zones, weekly_life, daily, "fvg")
    for i, r in enumerate(daily.itertuples(index=False)):
        t = pd.Timestamp(dates[i]).normalize()
        ws = weekly_states[i]
        ms = monthly_states[i]
        wa = weekly_ob_stats[i]
        wf = weekly_fvg_stats[i]
        wrow = {"datetime": t, "weekly_context_id": f"W:{ws['context_bar_end'].date()}" if pd.notna(ws["context_bar_end"]) else "",
                "weekly_available_at": ws["context_bar_end"], "weekly_structure_trend": ws["structure_trend"],
                "weekly_structure_integrity": ws["structure_integrity"], "weekly_bias_direction": ws["bias_direction"],
                "weekly_position": ws["position"], "weekly_range_low": ws["range_low"],
                "weekly_range_high": ws["range_high"], "weekly_range_mid": ws["range_mid"],
                "weekly_range_width_atr": ws["range_width_atr"], "weekly_protected_high": ws["protected_high"],
                "weekly_protected_low": ws["protected_low"], "weekly_structure_age_days": ws["structure_age_days"],
                "weekly_ob_distance_atr": wa["distance_atr"], "weekly_fvg_distance_atr": wf["distance_atr"],
                "weekly_active_ob_count": wa["active_count"], "weekly_active_fvg_count": wf["active_count"],
                "weekly_fresh_ob_count": wa["fresh_count"], "weekly_fresh_fvg_count": wf["fresh_count"]}
        mrow = {"monthly_context_id": f"M:{ms['context_bar_end'].date()}" if pd.notna(ms["context_bar_end"]) else "",
                "monthly_available_at": ms["context_bar_end"], "monthly_structure_trend": ms["structure_trend"],
                "monthly_structure_integrity": ms["structure_integrity"], "monthly_bias_direction": ms["bias_direction"],
                "monthly_position": ms["position"], "monthly_range_low": ms["range_low"],
                "monthly_range_high": ms["range_high"], "monthly_range_mid": ms["range_mid"],
                "monthly_range_width_atr": ms["range_width_atr"], "monthly_protected_high": ms["protected_high"],
                "monthly_protected_low": ms["protected_low"], "monthly_structure_age_days": ms["structure_age_days"],
                "monthly_ob_distance_atr": np.nan, "monthly_fvg_distance_atr": np.nan,
                "monthly_active_ob_count": 0, "monthly_active_fvg_count": 0,
                "monthly_fresh_ob_count": 0, "monthly_fresh_fvg_count": 0}
        relation = "unknown"
        if ws["structure_trend"] in {"up", "down"} and ms["structure_trend"] in {"up", "down"}:
            relation = "aligned" if ws["structure_trend"] == ms["structure_trend"] else "countertrend"
        bias_relation = "unknown"
        if ws["bias_direction"] in {"up", "down"} and ms["bias_direction"] in {"up", "down"}:
            bias_relation = "aligned" if ws["bias_direction"] == ms["bias_direction"] else "divergent"
        rows.append({**wrow, **mrow, "monthly_weekly_trend_state": relation,
                     "monthly_weekly_bias_state": bias_relation})
    return pd.DataFrame(rows)


def ctx_numeric(ctx: dict[str, Any], prefix: str) -> dict[str, Any]:
    out = {}
    for tf in ("weekly", "monthly"):
        trend = str(ctx.get(f"{tf}_structure_trend", "unknown"))
        integrity = str(ctx.get(f"{tf}_structure_integrity", "unknown"))
        bias = str(ctx.get(f"{tf}_bias_direction", "neutral"))
        pos = str(ctx.get(f"{tf}_position", "unknown"))
        values = {
            "trend_up": float(trend == "up"), "trend_down": float(trend == "down"),
            "trend_mixed": float(trend == "mixed"), "trend_unknown": float(trend == "unknown"),
            "integrity_intact": float(integrity == "intact"), "integrity_broken": float(integrity == "broken"),
            "integrity_unknown": float(integrity == "unknown"), "bias_up": float(bias == "up"),
            "bias_down": float(bias == "down"), "bias_neutral": float(bias not in {"up", "down"}),
            "position_premium": float(pos == "premium"), "position_discount": float(pos == "discount"),
            "position_out_of_range": float(pos == "out_of_range"), "position_unknown": float(pos not in {"premium", "discount", "out_of_range"}),
            "range_width_atr": fnum(ctx.get(f"{tf}_range_width_atr")),
            "ob_distance_atr": fnum(ctx.get(f"{tf}_ob_distance_atr")),
            "fvg_distance_atr": fnum(ctx.get(f"{tf}_fvg_distance_atr")),
            "active_ob_count": fnum(ctx.get(f"{tf}_active_ob_count"), 0.0),
            "active_fvg_count": fnum(ctx.get(f"{tf}_active_fvg_count"), 0.0),
            "fresh_ob_count": fnum(ctx.get(f"{tf}_fresh_ob_count"), 0.0),
            "fresh_fvg_count": fnum(ctx.get(f"{tf}_fresh_fvg_count"), 0.0),
        }
        out.update({f"{prefix}{tf}_{k}": v for k, v in values.items()})
        out[f"{prefix}{tf}_structure_trend"] = trend
        out[f"{prefix}{tf}_structure_integrity"] = integrity
        out[f"{prefix}{tf}_bias_direction"] = bias
        out[f"{prefix}{tf}_position"] = pos
    out[f"{prefix}trend_agreement"] = float(ctx.get("monthly_weekly_trend_state") == "aligned")
    out[f"{prefix}bias_agreement"] = float(ctx.get("monthly_weekly_bias_state") == "aligned")
    out[f"{prefix}monthly_weekly_state"] = str(ctx.get("monthly_weekly_trend_state", "unknown"))
    out[f"{prefix}monthly_weekly_bias_state"] = str(ctx.get("monthly_weekly_bias_state", "unknown"))
    return out


def context_at(contexts: pd.DataFrame, timestamp: Any) -> dict[str, Any]:
    t = pd.Timestamp(timestamp).normalize()
    if contexts.empty:
        return {}
    q = contexts[contexts.datetime.eq(t)]
    return q.iloc[-1].to_dict() if len(q) else {}


def event_kind(row: pd.Series) -> str:
    et, rule = str(row.get("event_type", "")), str(row.get("rule_id", ""))
    if et in {"sweep_same_bar", "sweep_delayed_reclaim"}:
        return "sweep"
    if et in {"liquidity_breach", "liquidity_recovery_pending", "unreclaimed_at_horizon"} or rule == "liquidity_sweep":
        return "breach"
    if et in {"mss", "bos", "cisd"} or rule in {"mss", "bos", "cisd"}:
        return "mss_bos"
    return "other"


def build_sequence(objects: pd.DataFrame, events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create strict target-time relations; touch-time fields stay separate."""
    obj = objects.copy()
    for c in ["available_at", "origin_time"]:
        obj[c] = pd.to_datetime(obj[c], errors="coerce").dt.normalize()
    ev = events.copy()
    if len(ev):
        for c in ["available_at", "event_time"]:
            ev[c] = pd.to_datetime(ev[c], errors="coerce").dt.normalize()
        ev = ev[ev.timeframe.isin(["1D", "1W"]) & ev.direction.isin(["bullish", "bearish"])].copy()
        ev["step_kind"] = ev.apply(event_kind, axis=1)
    anchors = {}
    max_days = {"1D": 10, "1W": 35}
    for (tf, direction), g in ev.groupby(["timeframe", "direction"], dropna=False) if len(ev) else []:
        anchors[(tf, direction)] = g[g.step_kind.isin(["sweep", "breach"])].sort_values(["available_at", "event_id"])

    def link(t: pd.Timestamp, tf: str, direction: str) -> tuple[str, pd.Timestamp, str]:
        g = anchors.get((tf, direction), pd.DataFrame())
        if len(g):
            prior = g[g.available_at < t]
            if len(prior):
                a = prior.iloc[-1]
                if (t - pd.Timestamp(a.available_at)).days <= max_days.get(tf, 10):
                    return str(a.event_id), pd.Timestamp(a.available_at), "local_event_anchor"
        days = max_days.get(tf, 10)
        epoch = (t - pd.Timestamp("2000-01-01")).days // days
        return f"unanchored_{epoch:06d}", t, "temporal_unanchored"

    obj_rows, step_rows, feature_rows = [], [], []
    for r in obj.itertuples(index=False):
        anchor, anchor_at, link_type = link(pd.Timestamp(r.available_at), str(r.timeframe), str(r.direction))
        sid = f"NQ=F|sequence_v05|{r.timeframe}|{r.direction}|{anchor}"
        obj_rows.append({"object_id": r.object_id, "scenario_id": sid, "scenario_anchor_at": anchor_at, "scenario_link_type": link_type})
        step_rows.append({"step_id": r.object_id, "scenario_id": sid, "available_at": r.available_at,
                          "step_time": r.origin_time, "step_kind": "zone", "step_type": r.object_type,
                          "direction": r.direction, "timeframe": r.timeframe, "source_id": r.object_id})
    if len(ev):
        for r in ev.itertuples(index=False):
            t = pd.Timestamp(r.available_at)
            if r.step_kind in {"sweep", "breach"}:
                anchor, anchor_at, link_type = str(r.event_id), t, "event_anchor"
            else:
                anchor, anchor_at, link_type = link(t, str(r.timeframe), str(r.direction))
            sid = f"NQ=F|sequence_v05|{r.timeframe}|{r.direction}|{anchor}"
            step_rows.append({"step_id": r.event_id, "scenario_id": sid, "available_at": t,
                              "step_time": r.event_time, "step_kind": "event", "step_type": r.step_kind,
                              "direction": r.direction, "timeframe": r.timeframe, "source_id": r.event_id})
    steps = pd.DataFrame(step_rows)
    for r in obj.itertuples(index=False):
        link_row = next(x for x in obj_rows if x["object_id"] == r.object_id)
        sid, t = link_row["scenario_id"], pd.Timestamp(r.available_at)
        prior = steps[(steps.scenario_id == sid) & (steps.available_at < t) & (steps.source_id != r.object_id)]
        zones = prior[prior.step_kind == "zone"]
        events_prior = prior[prior.step_kind == "event"]
        feature_rows.append({
            "object_id": r.object_id, "scenario_id": sid, "scenario_link_type": link_row["scenario_link_type"],
            "sequence_fvg_before_object": float(r.object_type == "ob_candidate" and (zones.step_type == "fvg").any()),
            "sequence_ob_before_object": float(r.object_type == "fvg" and (zones.step_type == "ob_candidate").any()),
            "sequence_sweep_before_object": float(events_prior.step_type.isin(["sweep", "breach"]).any()),
            "sequence_mss_bos_before_object": float((events_prior.step_type == "mss_bos").any()),
            "sequence_prior_zone_count": float(len(zones)),
            "sequence_same_timestamp_step": float(((steps.scenario_id == sid) & (steps.available_at == t) & (steps.source_id != r.object_id)).any()),
        })
    return pd.DataFrame(feature_rows), steps


def attach_post_sequence(labels: pd.DataFrame, steps: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in labels.itertuples(index=False):
        scenario_id = getattr(r, "scenario_id", np.nan)
        if pd.isna(getattr(r, "touch_at", pd.NaT)) or pd.isna(scenario_id):
            continue
        t = pd.Timestamp(r.touch_at).normalize()
        prior = steps[(steps.scenario_id == scenario_id) & (steps.available_at < t) & (steps.source_id != r.object_id)] if len(steps) else pd.DataFrame()
        zones, ep = prior[prior.step_kind == "zone"], prior[prior.step_kind == "event"]
        rows.append({"object_id": r.object_id,
                     "post_sequence_fvg_before_touch": float(r.object_type == "ob_candidate" and (zones.step_type == "fvg").any()),
                     "post_sequence_ob_before_touch": float(r.object_type == "fvg" and (zones.step_type == "ob_candidate").any()),
                     "post_sequence_sweep_before_touch": float(ep.step_type.isin(["sweep", "breach"]).any()),
                     "post_sequence_mss_bos_before_touch": float((ep.step_type == "mss_bos").any()),
                     "post_sequence_prior_zone_count": float(len(zones)),
                     "post_sequence_same_timestamp_step": float(((steps.scenario_id == scenario_id) & (steps.available_at == t) & (steps.source_id != r.object_id)).any())})
    out = labels.copy()
    if rows:
        out = out.merge(pd.DataFrame(rows), on="object_id", how="left")
    for c in POST_SEQUENCE_NUMERIC:
        if c not in out.columns:
            out[c] = 0.0
    return out


def build_labels(objects: pd.DataFrame, daily: pd.DataFrame, contexts: pd.DataFrame,
                 sequence_features: pd.DataFrame, sequence_steps: pd.DataFrame,
                 lifecycle: pd.DataFrame) -> pd.DataFrame:
    base = ev3.make_features(objects, daily)
    base_map = base.set_index("object_id").to_dict("index") if len(base) else {}
    life_map = lifecycle.set_index("object_id").to_dict("index") if len(lifecycle) else {}
    rows = []
    for _, obj in objects.iterrows():
        lb = v4.label_object_v04(obj, daily, arrival_horizon=ARRIVAL_HORIZON.get(str(obj.timeframe)),
                                 post_touch_horizon=POST_TOUCH_HORIZON, barrier_atr=BARRIER_ATR)
        row = dict(lb)
        row.update(base_map.get(obj.object_id, {}))
        fc = context_at(contexts, obj.available_at)
        row.update(ctx_numeric(fc, "formation_"))
        row["formation_weekly_available_at"] = fc.get("weekly_available_at", pd.NaT)
        row["formation_monthly_available_at"] = fc.get("monthly_available_at", pd.NaT)
        row["formation_weekly_structure_trend"] = fc.get("weekly_structure_trend", "unknown")
        row["formation_monthly_structure_trend"] = fc.get("monthly_structure_trend", "unknown")
        row["formation_weekly_bias_direction"] = fc.get("weekly_bias_direction", "neutral")
        row["formation_monthly_bias_direction"] = fc.get("monthly_bias_direction", "neutral")
        seq = sequence_features[sequence_features.object_id.eq(obj.object_id)] if "object_id" in sequence_features.columns else pd.DataFrame()
        if len(seq):
            row.update(seq.iloc[0].to_dict())
        life = life_map.get(obj.object_id, {})
        row["zone_state_at_formation"] = "fresh"
        row["zone_fresh_at_formation"] = float(pd.isna(life.get("first_touch_at")) or pd.Timestamp(life.get("first_touch_at")).normalize() > pd.Timestamp(obj.available_at).normalize())
        row["zone_first_touch_at"] = life.get("first_touch_at", pd.NaT)
        row["zone_first_full_at"] = life.get("first_full_at", pd.NaT)
        row["zone_first_close_invalidation_at"] = life.get("first_close_invalidation_at", pd.NaT)
        row["post_outcome"] = str(row.get("post_label", "not_applicable"))
        row["post_eval_eligible"] = float(row["post_outcome"] in POST_CLASSES)
        row["post_resolved_eligible"] = float(row["post_outcome"] in {"adverse_first", "favorable_first"})
        if row.get("arrival_type") == "touch" and pd.notna(row.get("touch_at")):
            tc = context_at(contexts, row["touch_at"])
            row.update(ctx_numeric(tc, "post_"))
            row["post_weekly_available_at"] = tc.get("weekly_available_at", pd.NaT)
            row["post_monthly_available_at"] = tc.get("monthly_available_at", pd.NaT)
            row["post_weekly_structure_trend"] = tc.get("weekly_structure_trend", "unknown")
            row["post_monthly_structure_trend"] = tc.get("monthly_structure_trend", "unknown")
            row["post_weekly_bias_direction"] = tc.get("weekly_bias_direction", "neutral")
            row["post_monthly_bias_direction"] = tc.get("monthly_bias_direction", "neutral")
            p = daily[daily.datetime.eq(pd.Timestamp(row["touch_at"]).normalize())]
            if len(p):
                p = p.iloc[-1]
                lo, hi = fnum(obj.zone_low), fnum(obj.zone_high)
                atr = fnum(p.atr14)
                width = hi - lo
                distance = 0.0 if lo <= p.close <= hi else min(abs(lo - p.close), abs(hi - p.close))
                brange = fnum(p.high) - fnum(p.low)
                intrusion = max(0.0, hi - fnum(p.low)) if obj.direction == "bullish" else max(0.0, fnum(p.high) - lo)
                row.update({"post_distance_atr": distance / atr if atr > 0 else np.nan,
                            "post_width_atr": width / atr if atr > 0 else np.nan,
                            "post_mid_distance_atr": (p.close - (lo + hi) / 2) / atr if atr > 0 else np.nan,
                            "post_intrusion_atr": intrusion / atr if atr > 0 else np.nan,
                            "post_bar_range_atr": brange / atr if atr > 0 else np.nan,
                            "post_close_location": (p.close - p.low) / brange if brange > 0 else 0.5,
                            "post_return_5": fnum(p.return_5), "post_return_20": fnum(p.return_20),
                            "post_atr_pct": atr / p.close if p.close else np.nan, "post_volume_z20": fnum(p.volume_z20),
                            "post_sma_gap20": fnum(p.sma_gap20), "post_sma_gap50": fnum(p.sma_gap50)})
        rows.append(row)
    out = pd.DataFrame(rows)
    out = attach_post_sequence(out, sequence_steps)
    for c in POST_BASE_FEATURES + POST_WEEKLY_FEATURES + POST_MONTHLY_FEATURES + POST_HIERARCHY_FEATURES + POST_SEQUENCE_NUMERIC:
        if c not in out.columns:
            out[c] = np.nan
    return out.sort_values(["available_at", "object_id"]).reset_index(drop=True)


def model_features(target: str) -> dict[str, list[str]]:
    base, weekly, monthly, hierarchy, sequence = (BASE_FEATURES, WEEKLY_FEATURES, MONTHLY_FEATURES, HIERARCHY_FEATURES, SEQUENCE_NUMERIC) if target == "arrival" else (POST_BASE_FEATURES, POST_WEEKLY_FEATURES, POST_MONTHLY_FEATURES, POST_HIERARCHY_FEATURES, POST_SEQUENCE_NUMERIC)
    return {
        "base_geometry_environment": base,
        "plus_weekly_structure": base + weekly,
        "plus_monthly_structure": base + monthly,
        "plus_weekly_monthly": base + weekly + monthly + hierarchy,
        "plus_weekly_monthly_sequence": base + weekly + monthly + hierarchy + sequence,
    }


def folds(dates: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    u = pd.Series(pd.to_datetime(dates).dropna().sort_values().unique())
    if len(u) < 5:
        return []
    qs = sorted(set(pd.Timestamp(x).normalize() for x in u.quantile([0.50, 0.60, 0.70, 0.80, 0.90])))
    return [(qs[i], (qs + [u.max() + pd.Timedelta(days=1)])[i + 1]) for i in range(len(qs)) if qs[i] < (qs + [u.max() + pd.Timedelta(days=1)])[i + 1]]


def fit_probability(train: pd.DataFrame, test: pd.DataFrame, features: list[str], target: str) -> tuple[np.ndarray, list[str], str]:
    # Exclude features that are entirely unavailable in the training fold.
    # Monthly OB/FVG distances are intentionally not detected in v0.5, and
    # categorical state labels are retained in the audit CSV, not coerced to
    # meaningless all-NaN numeric columns.
    cols = [c for c in features if c in train.columns and c in test.columns and train[c].notna().any()]
    if target == "arrival":
        prior = float(train.y.mean()) if len(train) else np.nan
        if len(train) < MIN_TRAIN or train.y.nunique() < 2 or not cols:
            return np.column_stack([1 - np.full(len(test), prior), np.full(len(test), prior)]), ["negative", "positive"], "constant_training_prior"
        pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("logit", LogisticRegression(C=1.0, max_iter=1200, solver="lbfgs"))])
        pipe.fit(train[cols].apply(pd.to_numeric, errors="coerce"), train.y.astype(int))
        return pipe.predict_proba(test[cols].apply(pd.to_numeric, errors="coerce")), ["negative", "positive"], "logistic_fixed_C1"
    y = train.y_class.astype(str)
    prior = y.value_counts(normalize=True).reindex(POST_CLASSES, fill_value=0).to_numpy(float)
    if len(train) < MIN_TRAIN or y.nunique() < 2 or not cols:
        return np.tile(prior, (len(test), 1)), POST_CLASSES, "constant_training_prior"
    pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("logit", LogisticRegression(C=1.0, max_iter=1200, solver="lbfgs"))])
    pipe.fit(train[cols].apply(pd.to_numeric, errors="coerce"), y)
    raw = pd.DataFrame(pipe.predict_proba(test[cols].apply(pd.to_numeric, errors="coerce")), columns=pipe.named_steps["logit"].classes_)
    return raw.reindex(columns=POST_CLASSES, fill_value=0).to_numpy(float), POST_CLASSES, "multinomial_logistic_fixed_C1"


def logloss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(np.asarray(p, float), 1e-7, 1 - 1e-7)
    return float(-np.mean(np.log(p[np.arange(len(y)), y])))


def walk_forward(labels: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if target == "arrival":
        s = labels[labels.arrival_complete & labels.arrival_label.notna()].copy()
        s["y"] = s.arrival_label.astype(float); s["prediction_at"] = s.available_at; s["label_end_at"] = s.arrival_label_end_at
    else:
        s = labels[labels.post_eval_eligible.astype(bool)].copy()
        s["y_class"] = s.post_outcome.astype(str); s["prediction_at"] = s.post_prediction_at; s["label_end_at"] = s.post_label_end_at
    s = s[s.prediction_at.notna() & s.label_end_at.notna()].sort_values(["prediction_at", "object_id"])
    metrics, predictions = [], []
    for (tf, direction), g in s.groupby(["timeframe", "direction"], dropna=False):
        for fold_id, (start, end) in enumerate(folds(g.prediction_at), 1):
            train = g[(g.prediction_at < start) & (g.label_end_at < start)]
            test = g[(g.prediction_at >= start) & (g.prediction_at < end)]
            if len(train) < MIN_TRAIN or len(test) < MIN_TEST:
                continue
            sets = model_features(target)
            pred_map = {}
            status_map = {}
            if target == "arrival":
                prior = float(train.y.mean())
                pred_map["unconditional"] = np.column_stack([np.full(len(test), 1 - prior), np.full(len(test), prior)])
                status_map["unconditional"] = "training_prior"
            else:
                prior = train.y_class.value_counts(normalize=True).reindex(POST_CLASSES, fill_value=0).to_numpy(float)
                pred_map["unconditional"] = np.tile(prior, (len(test), 1)); status_map["unconditional"] = "training_prior"
            for name, feat in sets.items():
                pred_map[name], _, status_map[name] = fit_probability(train, test, feat, target)
            if target == "arrival":
                y = test.y.to_numpy(int)
            else:
                y = pd.Categorical(test.y_class, categories=POST_CLASSES).codes
            for name, p in pred_map.items():
                if target == "arrival":
                    brier = float(np.mean((p[:, 1] - y) ** 2)); ll = logloss(y, p)
                else:
                    one = np.eye(len(POST_CLASSES))[y]; brier = float(np.mean(np.sum((p - one) ** 2, axis=1))); ll = logloss(y, p)
                metrics.append({"target": target, "timeframe": tf, "direction": direction, "fold_id": fold_id,
                                "test_start": start, "test_end": end, "train_n": len(train), "test_n": len(test),
                                "train_positive_rate": float(train.y.mean()) if target == "arrival" else np.nan,
                                "train_favorable_rate": float((train.y_class == "favorable_first").mean()) if target != "arrival" else np.nan,
                                "model": name, "model_status": status_map[name], "brier": brier, "log_loss": ll,
                                "max_train_prediction_at": train.prediction_at.max(), "max_train_label_end_at": train.label_end_at.max(),
                                "min_test_prediction_at": test.prediction_at.min(), "max_test_prediction_at": test.prediction_at.max()})
                for i, (_, rr) in enumerate(test.iterrows()):
                    out = {"object_id": rr.object_id, "target": target, "timeframe": tf, "direction": direction,
                           "fold_id": fold_id, "prediction_at": rr.prediction_at, "label_end_at": rr.label_end_at,
                           "y": float(rr.y) if target == "arrival" else np.nan, "y_class": rr.y_class if target != "arrival" else ""}
                    if target == "arrival":
                        out["p_positive"] = float(pred_map[name][i, 1])
                    else:
                        for j, cls in enumerate(POST_CLASSES):
                            out[f"p_{cls}"] = float(pred_map[name][i, j])
                    out["model"] = name
                    predictions.append(out)
    return pd.DataFrame(metrics), pd.DataFrame(predictions)


def outcome_tables(labels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows, context_rows = [], []
    for (tf, ot, direction), g in labels.groupby(["timeframe", "object_type", "direction"], dropna=False):
        ar = g[g.arrival_complete & g.arrival_label.notna()]
        rows.append({"target": "arrival", "timeframe": tf, "object_type": ot, "direction": direction,
                     "risk_n": len(ar), "positive": int(ar.arrival_label.sum()), "no_arrival": int((ar.arrival_label == 0).sum()),
                     "touch_n": int((ar.arrival_type == "touch").sum()), "gap_through_n": int((ar.arrival_type == "gap_through").sum())})
        po = g[g.post_eval_eligible.astype(bool)]
        for outcome in POST_CLASSES:
            rows.append({"target": "post_competing", "timeframe": tf, "object_type": ot, "direction": direction,
                         "risk_n": len(po), "outcome": outcome, "positive": int((po.post_outcome == outcome).sum()),
                         "rate": float((po.post_outcome == outcome).mean()) if len(po) else np.nan,
                         "touch_n": int((g.arrival_type == "touch").sum()),
                         "prebreach_n": int(g.post_label.eq("pre_breached_at_touch_close").sum()),
                         "post_order_unknown_n": int(g.post_label.eq("both_same_bar_order_unknown").sum()),
                         "post_censored_n": int(g.post_label.eq("censored_post_touch").sum())})
    for (tf, direction, mw, ww), g in labels[(labels.object_type == "ob_candidate") & labels.formation_monthly_structure_trend.notna()].groupby(["timeframe", "direction", "formation_monthly_structure_trend", "formation_weekly_structure_trend"], dropna=False):
        ar = g[g.arrival_complete & g.arrival_label.notna()]
        po = g[g.post_eval_eligible.astype(bool)]
        context_rows.append({"timeframe": tf, "direction": direction, "monthly_trend": mw, "weekly_trend": ww,
                             "candidate_n": len(g), "arrival_n": len(ar), "arrival_rate": float(ar.arrival_label.mean()) if len(ar) else np.nan,
                             "post_n": len(po), "favorable_n": int((po.post_outcome == "favorable_first").sum()),
                             "adverse_n": int((po.post_outcome == "adverse_first").sum()),
                             "neither_n": int((po.post_outcome == "neither_at_horizon").sum()),
                             "favorable_rate_all_post": float((po.post_outcome == "favorable_first").mean()) if len(po) else np.nan})
    return pd.DataFrame(rows), pd.DataFrame(context_rows), pd.DataFrame()


def make_controls(objects: pd.DataFrame, daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, skipped = [], []
    for _, r in objects.iterrows():
        p = daily[daily.datetime.eq(pd.Timestamp(r.available_at).normalize())]
        if not len(p):
            skipped.append({"object_id": r.object_id, "reason": "no_feature_bar"}); continue
        p = p.iloc[-1]; close, atr = fnum(p.close), fnum(p.atr14); lo, hi = fnum(r.zone_low), fnum(r.zone_high)
        width = hi - lo
        if not all(math.isfinite(x) for x in [close, atr, lo, hi, width]) or atr <= 0 or width < 0:
            skipped.append({"object_id": r.object_id, "reason": "invalid_geometry"}); continue
        shift = max(0.25 * atr, width + 0.1 * atr)
        if hi < close:
            ctrl_lo, ctrl_hi, side = lo - shift, hi - shift, "below"
        elif lo > close:
            ctrl_lo, ctrl_hi, side = lo + shift, hi + shift, "above"
        else:
            skipped.append({"object_id": r.object_id, "reason": "actual_contains_feature_close"}); continue
        rr = r.copy(); rr["object_id"] = f"CONTROL|same_side_caliper|{r.object_id}"; rr["zone_low"] = ctrl_lo; rr["zone_high"] = ctrl_hi; rr["object_type"] = "control_band"; rr["evidence_state"] = "control"; rr["control_side"] = side; rr["control_shift_atr"] = shift / atr
        rows.append(rr.to_dict())
    return pd.DataFrame(rows), pd.DataFrame(skipped)


def control_summary(actual: pd.DataFrame, controls: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group, g in pd.concat([actual.assign(sample="actual"), controls.assign(sample="control")], ignore_index=True).groupby(["sample", "timeframe", "direction"], dropna=False):
        sample, tf, direction = group
        ar = g[g.arrival_complete & g.arrival_label.notna()]
        po = g[g.post_eval_eligible.astype(bool)]
        rows.append({"sample": sample, "timeframe": tf, "direction": direction, "candidate_n": len(g),
                     "arrival_n": len(ar), "arrival_rate": float(ar.arrival_label.mean()) if len(ar) else np.nan,
                     "touch_n": int((g.arrival_type == "touch").sum()), "post_n": len(po),
                     "post_favorable_rate_all_observed": float((po.post_outcome == "favorable_first").mean()) if len(po) else np.nan,
                     "post_adverse_rate_all_observed": float((po.post_outcome == "adverse_first").mean()) if len(po) else np.nan,
                     "post_neither_rate_all_observed": float((po.post_outcome == "neither_at_horizon").mean()) if len(po) else np.nan})
    return pd.DataFrame(rows)


def prefix_invariance(full: pd.DataFrame, daily: pd.DataFrame, weekly_zones: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    rows = []
    for frac in [0.50, 0.75]:
        cut = pd.Timestamp(daily.datetime.iloc[int(len(daily) * frac) - 1]).normalize()
        pre = daily[daily.datetime <= cut].copy()
        w, m = period_bars(pre, "W-FRI"), period_bars(pre, "M")
        wp, mp = mechanical_pivots(w, "1W"), mechanical_pivots(m, "1M")
        wz = weekly_zones[pd.to_datetime(weekly_zones.available_at).dt.normalize() <= cut].copy()
        wl = zone_lifecycle(wz, pre)
        pc = build_contexts(pre, w, m, wp, mp, wz, wl)
        cols = [c for c in full.columns if c not in {"datetime"}]
        f = full[full.datetime <= cut].set_index("datetime").reindex(pc.datetime).reset_index()
        p = pc.set_index("datetime").reindex(f.datetime).reset_index()
        mismatches = 0
        for c in cols:
            if c not in p or c not in f: continue
            if pd.api.types.is_numeric_dtype(f[c]):
                a, b = pd.to_numeric(f[c], errors="coerce"), pd.to_numeric(p[c], errors="coerce")
                mismatches += int(((a - b).abs() > 1e-9).fillna(a.isna() ^ b.isna()).sum())
            else:
                missing_both = f[c].isna() & p[c].isna()
                mismatches += int(((f[c].astype(str) != p[c].astype(str)) & ~missing_both).sum())
        rows.append({"cutoff": cut, "context_rows_compared": len(f), "mismatches": mismatches,
                     "passed": mismatches == 0, "scope": "higher_contexts; detector candidate selection not rerun"})
    return pd.DataFrame(rows)


def write_df(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--detector-output", type=Path, default=DEFAULT_DETECTOR_OUTPUT)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    daily, data_qa = ev3.load_daily(args.input)
    objects, object_qa = ev3.load_objects(args.detector_output, daily.datetime.max())
    objects = objects[objects.object_type.isin(["ob_candidate", "fvg"])].copy()
    events = pd.read_csv(args.detector_output / "events.csv", low_memory=False)
    events = events[events.sensitivity.eq("balanced")].copy()
    weekly = period_bars(daily, "W-FRI"); monthly = period_bars(daily, "M")
    wpiv = mechanical_pivots(weekly, "1W"); mpiv = mechanical_pivots(monthly, "1M")
    weekly_zones = objects[objects.timeframe.eq("1W")].copy()
    weekly_life = zone_lifecycle(weekly_zones, daily)
    contexts = build_contexts(daily, weekly, monthly, wpiv, mpiv, weekly_zones, weekly_life)
    seq_features, seq_steps = build_sequence(objects, events)
    life_all = zone_lifecycle(objects, daily)
    labels = build_labels(objects, daily, contexts, seq_features, seq_steps, life_all)

    arrival_metrics, arrival_predictions = walk_forward(labels, "arrival")
    post_metrics, post_predictions = walk_forward(labels, "post")
    metrics = pd.concat([arrival_metrics, post_metrics], ignore_index=True)
    predictions = pd.concat([arrival_predictions, post_predictions], ignore_index=True)
    if len(metrics):
        summary = metrics.groupby(["target", "timeframe", "direction", "model"], dropna=False).apply(
            lambda g: pd.Series({"folds": g.fold_id.nunique(), "n_test": int(g.test_n.sum()),
                                 "brier": float(np.average(g.brier, weights=g.test_n)), "brier_fold_mean": g.brier.mean(),
                                 "brier_std": g.brier.std(ddof=1), "log_loss": float(np.average(g.log_loss, weights=g.test_n)),
                                 "log_loss_fold_mean": g.log_loss.mean(), "log_loss_std": g.log_loss.std(ddof=1)}),
            include_groups=False).reset_index()
        base = summary[summary.model.eq("base_geometry_environment")][["target", "timeframe", "direction", "brier", "log_loss"]].rename(columns={"brier": "base_brier", "log_loss": "base_log_loss"})
        summary = summary.merge(base, on=["target", "timeframe", "direction"], how="left")
        summary["delta_brier_vs_base"] = summary.brier - summary.base_brier
        summary["delta_log_loss_vs_base"] = summary.log_loss - summary.base_log_loss
    else:
        summary = pd.DataFrame()
    outcomes, contexts_summary, _ = outcome_tables(labels)
    controls, skipped_controls = make_controls(objects, daily)
    control_labels = build_labels(controls, daily, contexts, pd.DataFrame(), pd.DataFrame(), zone_lifecycle(controls, daily)) if len(controls) else pd.DataFrame()
    controls_summary = control_summary(labels, control_labels) if len(control_labels) else pd.DataFrame()
    prefix = prefix_invariance(contexts, daily, weekly_zones, args.output)

    # Save machine-readable research surfaces.
    write_df(labels, args.output / "future_path_labels_v05.csv")
    write_df(contexts, args.output / "monthly_weekly_contexts_v05.csv")
    write_df(weekly, args.output / "weekly_bars_v05.csv")
    write_df(monthly, args.output / "monthly_bars_v05.csv")
    write_df(wpiv, args.output / "weekly_pivots_v05.csv")
    write_df(mpiv, args.output / "monthly_pivots_v05.csv")
    write_df(life_all, args.output / "zone_lifecycle_v05.csv")
    write_df(weekly_life, args.output / "weekly_zone_lifecycle_v05.csv")
    write_df(seq_features, args.output / "sequence_object_features_v05.csv")
    write_df(seq_steps, args.output / "sequence_steps_v05.csv")
    write_df(outcomes, args.output / "outcome_summary_v05.csv")
    write_df(contexts_summary, args.output / "monthly_weekly_ob_summary_v05.csv")
    write_df(metrics, args.output / "walk_forward_metrics_v05.csv")
    write_df(predictions, args.output / "walk_forward_predictions_v05.csv")
    write_df(summary, args.output / "walk_forward_metric_summary_v05.csv")
    write_df(controls, args.output / "same_side_controls_v05.csv")
    write_df(skipped_controls, args.output / "same_side_controls_skipped_v05.csv")
    write_df(control_labels, args.output / "same_side_control_labels_v05.csv")
    write_df(controls_summary, args.output / "same_side_control_summary_v05.csv")
    write_df(prefix, args.output / "prefix_invariance_v05.csv")

    monthly_terminal = monthly[monthly.period_end >= daily.datetime.max()].copy()
    weekly_unknown = int((~weekly.certified_calendar).sum())
    label_counts = {"objects": len(objects), "arrival_complete": int((labels.arrival_complete & labels.arrival_label.notna()).sum()),
                    "post_touch": int((labels.arrival_type == "touch").sum()), "post_competing_complete": int(labels.post_eval_eligible.sum()),
                    "post_resolved_only": int(labels.post_resolved_eligible.sum()),
                    "post_neither": int(labels.post_outcome.eq("neither_at_horizon").sum()),
                    "post_order_unknown": int(labels.post_outcome.eq("both_same_bar_order_unknown").sum()),
                    "post_censored": int(labels.post_outcome.eq("censored_post_touch").sum()),
                    "post_prebreach": int(labels.post_outcome.eq("pre_breached_at_touch_close").sum())}
    manifest = {
        "status": "completed_v05_monthly_weekly_hierarchy_and_corrected_controls",
        "script_version": SCRIPT_VERSION, "script": str(Path(__file__).resolve()), "script_sha256": sha256(Path(__file__)),
        "input": data_qa, "detector_output": object_qa, "as_of": str(daily.datetime.max().date()),
        "manual_inputs_used": [], "old_research_imported": False,
        "data_quality": {"weekly_rows": len(weekly), "weekly_uncertified_rows": weekly_unknown,
                          "monthly_rows": len(monthly), "monthly_terminal_incomplete_rows": len(monthly_terminal),
                          "monthly_completion_rule": "period_end < prediction_clock; current month never joined",
                          "weekly_completion_rule": "expected Friday or explicit narrow closure proxy plus period_end < prediction_clock"},
        "clocks": {"formation": "object.available_at", "arrival": "object.available_at", "post": "first observed touch bar close; touch bar excluded"},
        "post_label": {"classes": POST_CLASSES, "horizon_daily_bars": POST_TOUCH_HORIZON,
                        "barrier": "zone boundary +/- 1.0 ATR14 at touch", "neither_is_retained": True,
                        "same_bar_double_hit": "excluded as order_unknown", "prebreach_at_touch_close": "excluded from competing post risk set"},
        "structure": {"pivot": "3-bar fractal; right bar close required", "trend": "two latest confirmed highs and lows both higher/lower", "bias": "separate deterministic trend-position rule; hypothesis only", "monthly": "structure only; monthly OB/FVG not detected in this version"},
        "sequence": {"strict_target_order": True, "post_touch_order_is_separate": True, "scenario_anchor": "same timeframe/direction prior sweep or breach within 10 daily or 35 weekly calendar days; otherwise deterministic temporal bucket"},
        "control": {"definition": "same-side, same-width outward caliper; shift >= max(0.25 ATR, width+0.1 ATR)", "post_comparison": "each sample evaluated on its own full risk set; no both-reached selection", "causal_status": "geometric comparison only"},
        "walk_forward": {"folds": "expanding chronological 50/60/70/80/90 percent prediction-clock quantiles", "training_label_end": "strictly before test start", "min_train": MIN_TRAIN, "min_test": MIN_TEST, "post_metric": "three-class multiclass Brier and log loss"},
        "label_counts": label_counts,
        "prefix_invariance": prefix.to_dict("records"),
        "files": {"labels": str((args.output / "future_path_labels_v05.csv").resolve()), "contexts": str((args.output / "monthly_weekly_contexts_v05.csv").resolve()), "metrics": str((args.output / "walk_forward_metric_summary_v05.csv").resolve()), "controls": str((args.output / "same_side_control_summary_v05.csv").resolve())},
        "limitations": ["continuous futures proxy and provider volume", "daily OHLC cannot resolve intraday order", "candidate selection was inherited from the frozen v0.3 detector", "no independent human precision/recall labels", "no untouched post-freeze performance evaluation for this new model"],
    }
    (args.output / "evaluation_manifest_v05.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=jdefault), encoding="utf-8")

    report = [
        "# NQ=F ICT検知器 v0.5 月足・週足階層検証",
        "",
        "## 結論",
        "",
        "v0.4を上書きせず、同じ20年分の日足と固定balanced候補から、分母・予測時計・週足／月足構造・帯ライフサイクル・順序・対照を再構成した。月足は構造情報として追加し、月足OB/FVGは追加していない。",
        "",
        f"入力は {data_qa['start']}〜{data_qa['end']} の {data_qa['rows']:,} 日足。候補は {len(objects):,} 件、到達評価対象は {label_counts['arrival_complete']:,} 件、接触後の三分類評価対象は {label_counts['post_competing_complete']:,} 件。",
        "",
        "### 重要な読み方",
        "",
        "接触後の主ラベルは favorable_first / adverse_first / neither_at_horizon の三分類である。どちらにも届かなかった例を失敗にも成功にもせず残した。same-bar double hit、観測終端による打切り、接触足終値で既に障壁を越えた例は、別の未知・除外状態として保存した。",
        "",
        "トレンドは確定Pivotから観測した過去構造、バイアスは構造とレンジ内位置から固定ルールで生成した次の移動方向の仮説である。両者を同じbullish/bearish列にはまとめていない。",
        "",
        "## 生成物",
        "",
        "- `future_path_labels_v05.csv`: 到達・三分類の将来経路ラベルと、形成時／接触時の月足・週足状態",
        "- `monthly_weekly_contexts_v05.csv`: 日付ごとの点時点コンテキスト",
        "- `zone_lifecycle_v05.csv`: 帯の初回接触・完全到達・終値無効化の監査履歴",
        "- `sequence_object_features_v05.csv` / `sequence_steps_v05.csv`: 形成時順序とシナリオステップ",
        "- `walk_forward_metric_summary_v05.csv`: 基準、週足追加、月足追加、両方、順序追加のBrier／log loss",
        "- `same_side_control_summary_v05.csv`: 双方到達だけに限定しない対照集計",
        "- `prefix_invariance_v05.csv`: higher-contextの前方追加不変性QA",
        "",
        "## 解釈の保留",
        "",
        "この版で月足列が存在すること自体は、月足が追加価値を持つことを意味しない。`walk_forward_metric_summary_v05.csv`で、月足追加モデルが同じ基準モデルを未見時系列で改善した場合に限り、月足の予測上の関連を残す。改善しない場合は表示用の文脈として保持し、確率モデルからは外す。",
        "",
        "候補検知そのものはv0.3の固定出力を使っているため、月足を使った候補選択の再構築ではない。また、構造・帯・順序は同一価格系列から派生するため、複数の独立証拠として加点していない。",
    ]
    (args.output / "evaluation_report_v05_ja.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "output": str(args.output.resolve()), "label_counts": label_counts, "metrics_rows": len(metrics), "controls": len(controls), "prefix": prefix.to_dict("records")}, ensure_ascii=False, default=jdefault))


if __name__ == "__main__":
    main()
