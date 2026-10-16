"""NQ=F ICT evaluation v0.9: article-shaped scenarios and fixed targets.

This is a successor to v0.8.  It keeps the daily/weekly adaptation explicit
because the supplied NQ data is daily OHLCV and cannot verify the lower-time-
frame execution rules in the source articles.

The primary question here is conditional realization: after a mechanically
defined ICT-shaped situation is observable, does the pre-declared path occur
often enough to call it high probability?  Model skill versus a historical
prior is reported separately and is not used as the sole pass/fail test.
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
import nq_ict_evaluation_20260911_v04 as v4
import nq_ict_evaluation_20260912_v08 as v08


SCRIPT_VERSION = "nq-ict-evaluation-v0.9"
DEFAULT_INPUT = Path("nq_25y_actual_2000_2026_20260911/NQ_F_1d_combined.csv")
DEFAULT_OUTPUT = Path("nq_ict_evaluation_20260912_v09")
DEFAULT_DETECTOR_OUTPUT = Path("nq_ict_detection_20260912_v09")
SOURCE_COVERAGE = Path("nq_ict_evaluation_20260911_v07/article_coverage_v07.json")
SOURCE_MANIFEST = Path("ict_tutorials_corpus_20260909/article_manifest.csv")
TIMEFRAMES = ["1D", "1W", "1M"]
EVAL_TIMEFRAMES = ["1D", "1W"]
EVAL_TYPES = ["ob_candidate", "fvg"]
SENSITIVITIES = ["strict", "balanced", "exploratory"]
ARRIVAL_HORIZON = {"1D": 20, "1W": 60}
TARGET_HORIZON = {"1D": 20, "1W": 60}
POST_TRIGGER_BARS = {"1D": 5, "1W": 5}
INVALIDATION_BUFFER_ATR = 0.10
POST_CLASSES = ["invalidation_first", "target_first", "neither_at_horizon"]
MIN_TRAIN = 60
MIN_TEST = 12

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
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is pd.NaT:
        return None
    return str(value)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def normalize_dates(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce").dt.normalize()
    return out


def gap_dates_for_input(input_path: Path) -> tuple[set[pd.Timestamp], dict[str, Any]]:
    path = input_path.parent / "excluded_invalid_ohlc_rows.csv"
    if not path.exists():
        return set(), {"path": str(path.resolve()), "rows": 0, "status": "not_found"}
    gap = pd.read_csv(path)
    dates = set(pd.to_datetime(gap["datetime"], errors="coerce").dt.normalize().dropna())
    return dates, {"path": str(path.resolve()), "rows": int(len(gap)), "dates": sorted(str(x.date()) for x in dates), "status": "loaded"}


def mark_gaps(objects: pd.DataFrame, gap_dates: set[pd.Timestamp]) -> pd.DataFrame:
    """Flag a formation interval using calendar dates, including absent rows."""
    if objects.empty:
        return objects.copy()
    out = objects.copy()
    bad, flags = [], []
    for row in out.itertuples(index=False):
        a = pd.Timestamp(getattr(row, "origin_time")).normalize()
        b = pd.Timestamp(getattr(row, "available_at")).normalize()
        found = any(a <= d <= b for d in gap_dates) if pd.notna(a) and pd.notna(b) else False
        old = str(getattr(row, "quality_flags", "") or "")
        bad.append(bool(found))
        flags.append(";".join(x for x in [old, "source_gap_in_formation_window" if found else ""] if x))
    out["gap_contaminated"] = bad
    out["quality_flags"] = flags
    return out


def date_index(daily: pd.DataFrame) -> tuple[pd.Series, np.ndarray]:
    dates = pd.to_datetime(daily.datetime).dt.normalize()
    return dates, dates.to_numpy(dtype="datetime64[ns]")


def index_at_or_before(daily_dates: np.ndarray, value: Any) -> int:
    t = pd.Timestamp(value).normalize()
    return int(np.searchsorted(daily_dates, np.datetime64(t, "ns"), side="right") - 1)


def interval_has_gap(start: Any, end: Any, gap_dates: set[pd.Timestamp]) -> bool:
    if pd.isna(start) or pd.isna(end):
        return False
    a, b = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    return any(a <= d <= b for d in gap_dates)


def target_at(zone: pd.Series, daily: pd.DataFrame, daily_dates: np.ndarray,
              frames: dict[str, pd.DataFrame], pivots_map: dict[tuple[str, str], pd.DataFrame]) -> dict[str, Any]:
    """Freeze the next unconsumed same-timeframe pivot as a DOL target."""
    tf, direction = str(zone.timeframe), str(zone.direction)
    t = pd.Timestamp(zone.available_at).normalize()
    i = index_at_or_before(daily_dates, t)
    close = fnum(daily.close.iloc[i]) if 0 <= i < len(daily) else np.nan
    atr = fnum(daily.atr14.iloc[i]) if 0 <= i < len(daily) else np.nan
    lo, hi = fnum(zone.zone_low), fnum(zone.zone_high)
    result = {"target_known": 0, "target_price": np.nan, "target_pivot_id": "", "target_kind": "",
              "target_distance_atr": np.nan, "target_reason": "no_same_tf_unconsumed_pivot",
              "invalidation_price": np.nan, "invalidation_distance_atr": np.nan}
    if not (math.isfinite(close) and math.isfinite(atr) and atr > 0 and math.isfinite(lo) and math.isfinite(hi)):
        result["target_reason"] = "missing_price_or_atr_or_zone"
        return result
    inv = lo - INVALIDATION_BUFFER_ATR * atr if direction == "bullish" else hi + INVALIDATION_BUFFER_ATR * atr
    result["invalidation_price"] = inv
    result["invalidation_distance_atr"] = abs(close - inv) / atr
    piv = pivots_map.get((tf, "balanced"), pd.DataFrame())
    if piv.empty or tf not in frames:
        result["target_reason"] = "no_same_tf_pivots"
        return result
    candidates = []
    for p in piv.itertuples(index=False):
        av = pd.Timestamp(p.available_at).normalize()
        price = fnum(p.price)
        if pd.isna(av) or av > t or not math.isfinite(price):
            continue
        if direction == "bullish":
            if p.pivot_type != "high" or price <= max(close, hi):
                continue
        else:
            if p.pivot_type != "low" or price >= min(close, lo):
                continue
        pidx = int(np.searchsorted(daily_dates, np.datetime64(av, "ns"), side="left"))
        end = min(i, len(daily) - 1)
        if pidx <= end:
            highs = pd.to_numeric(daily.high.iloc[pidx:end + 1], errors="coerce").to_numpy(float)
            lows = pd.to_numeric(daily.low.iloc[pidx:end + 1], errors="coerce").to_numpy(float)
            consumed = bool(np.nanmax(highs) >= price) if direction == "bullish" and len(highs) else bool(np.nanmin(lows) <= price) if direction == "bearish" and len(lows) else False
            if consumed:
                continue
        distance = (price - hi) / atr if direction == "bullish" else (lo - price) / atr
        candidates.append((distance, av, str(p.pivot_id), price, str(p.pivot_type)))
    if candidates:
        candidates.sort(key=lambda x: (x[0], x[1], x[2]))
        d, _, pid, price, kind = candidates[0]
        result.update({"target_known": 1, "target_price": price, "target_pivot_id": pid,
                       "target_kind": f"confirmed_{kind}", "target_distance_atr": d,
                       "target_reason": "next_unconsumed_confirmed_pivot_same_tf"})
    return result


def location_at(zone: pd.Series, contexts: pd.DataFrame) -> dict[str, Any]:
    t = pd.Timestamp(zone.available_at).normalize()
    c = contexts[contexts.datetime <= t]
    result = {"location_known": 0, "location_aligned": 0, "location_state": "unknown",
              "htf_trend_aligned": 0, "htf_integrity_broken": 0}
    if c.empty:
        return result
    c = c.iloc[-1]
    direction = str(zone.direction)
    wanted = "discount" if direction == "bullish" else "premium"
    positions = [str(c.get("1w_position", "unknown")), str(c.get("1m_position", "unknown"))]
    trends = [str(c.get("1w_structure_trend", "unknown")), str(c.get("1m_structure_trend", "unknown"))]
    integrities = [str(c.get("1w_structure_integrity", "unknown")), str(c.get("1m_structure_integrity", "unknown"))]
    aligned_positions = [x for x in positions if x == wanted]
    known_positions = [x for x in positions if x in {"premium", "discount", "midpoint", "out_of_range"}]
    aligned_trends = [x for x in trends if (x == "up" and direction == "bullish") or (x == "down" and direction == "bearish")]
    result.update({"location_known": int(bool(known_positions)), "location_aligned": int(bool(aligned_positions)),
                   "location_state": wanted if aligned_positions else ";".join(positions),
                   "htf_trend_aligned": int(bool(aligned_trends)),
                   "htf_integrity_broken": int("broken" in integrities)})
    return result


def build_ordered_links(objects: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Create article-shaped local links without using future outcomes."""
    bal = objects[(objects.sensitivity == "balanced") & objects.timeframe.isin(EVAL_TIMEFRAMES)].copy()
    all_obj = bal[bal.object_type.isin(set(EVAL_TYPES) | {"displacement"})]
    all_events = events[(events.sensitivity == "balanced") & events.timeframe.isin(EVAL_TIMEFRAMES)].copy()
    all_events = normalize_dates(all_events, ["event_time", "available_at"])
    groups: dict[tuple[str, str, str], list[Any]] = {}
    for row in all_obj.itertuples(index=False):
        groups.setdefault((str(row.timeframe), str(row.direction), str(row.object_type)), []).append(row)
    event_groups: dict[tuple[str, str], list[Any]] = {}
    for row in all_events.itertuples(index=False):
        event_groups.setdefault((str(row.timeframe), str(row.direction)), []).append(row)
    rows = []
    for zone in bal[bal.object_type.isin(EVAL_TYPES)].itertuples(index=False):
        tf, direction = str(zone.timeframe), str(zone.direction)
        av = pd.Timestamp(zone.available_at).normalize()
        disp_candidates = []
        for d in groups.get((tf, direction, "displacement"), []):
            if pd.Timestamp(d.available_at).normalize() > av:
                continue
            if int(d.origin_idx) <= int(zone.confirmed_idx) + 2 and int(d.confirmed_idx) >= int(zone.origin_idx) - 2:
                score = (abs(int(d.origin_idx) - int(zone.origin_idx)), abs(int(d.confirmed_idx) - int(zone.confirmed_idx)), pd.Timestamp(d.available_at), str(d.object_id))
                disp_candidates.append((score, d))
        disp = sorted(disp_candidates, key=lambda x: x[0])[0][1] if disp_candidates else None
        disp_origin = pd.Timestamp(disp.origin_time).normalize() if disp is not None else pd.NaT
        disp_confirm = pd.Timestamp(disp.confirmed_at).normalize() if disp is not None else pd.NaT
        disp_id = str(disp.object_id) if disp is not None else ""
        event_list = event_groups.get((tf, direction), [])
        struct_candidates = []
        if disp is not None:
            for e in event_list:
                if str(e.event_type) not in {"mss", "bos"} or pd.Timestamp(e.available_at).normalize() > av:
                    continue
                et = pd.Timestamp(e.event_time).normalize()
                if et < disp_origin or et > av:
                    continue
                if str(e.object_id) not in {"", disp_id}:
                    continue
                struct_candidates.append(e)
        struct = sorted(struct_candidates, key=lambda e: (pd.Timestamp(e.available_at), str(e.event_id)))[0] if struct_candidates else None
        sweep_candidates = []
        if disp is not None:
            max_days = 10 if tf == "1D" else 35
            for e in event_list:
                if str(e.event_type) not in {"sweep_same_bar", "sweep_delayed_reclaim"}:
                    continue
                et = pd.Timestamp(e.event_time).normalize()
                if pd.isna(et) or et > disp_origin or (disp_origin - et).days > max_days:
                    continue
                if pd.Timestamp(e.available_at).normalize() <= av:
                    sweep_candidates.append(e)
        sweep = sorted(sweep_candidates, key=lambda e: (pd.Timestamp(e.available_at), str(e.event_id)))[-1] if sweep_candidates else None
        fvg_candidates = []
        if zone.object_type == "ob_candidate" and disp is not None:
            for f in groups.get((tf, direction, "fvg"), []):
                if pd.Timestamp(f.available_at).normalize() > av:
                    continue
                if int(f.origin_idx) >= int(disp.origin_idx) - 1 and int(f.origin_idx) <= int(disp.confirmed_idx) + 1:
                    fvg_candidates.append(f)
        companion = sorted(fvg_candidates, key=lambda f: (pd.Timestamp(f.available_at), str(f.object_id)))[0] if fvg_candidates else None
        has_disp, has_struct = disp is not None, struct is not None
        has_sweep, has_fvg = sweep is not None, companion is not None
        order_valid = bool(has_sweep and has_disp and has_struct and pd.Timestamp(sweep.event_time).normalize() <= disp_origin and pd.Timestamp(struct.available_at).normalize() >= disp_confirm)
        rows.append({
            "object_id": str(zone.object_id), "sequence_state": "complete_ordered_sequence" if order_valid and (zone.object_type == "fvg" or has_fvg) else "partial_ordered_sequence" if has_disp else "none",
            "sequence_sweep_before": int(has_sweep), "sequence_displacement_before": int(has_disp),
            "sequence_structure_before": int(has_struct), "sequence_fvg_companion_before": int(has_fvg),
            "sequence_order_valid": int(order_valid), "sequence_complete_for_ob": int(order_valid and has_fvg),
            "sequence_complete_for_fvg": int(has_disp and has_struct), "sequence_event_count": int(sum([has_sweep, has_disp, has_struct, has_fvg])),
            "sweep_event_id": str(sweep.event_id) if sweep is not None else "", "sweep_at": pd.Timestamp(sweep.available_at).normalize() if sweep is not None else pd.NaT,
            "displacement_id": disp_id, "displacement_origin_at": disp_origin, "displacement_confirmed_at": disp_confirm,
            "companion_fvg_id": str(companion.object_id) if companion is not None else "", "companion_fvg_at": pd.Timestamp(companion.available_at).normalize() if companion is not None else pd.NaT,
            "structure_event_id": str(struct.event_id) if struct is not None else "", "structure_at": pd.Timestamp(struct.available_at).normalize() if struct is not None else pd.NaT,
            "sequence_available_at": max([x for x in [pd.Timestamp(sweep.available_at).normalize() if sweep is not None else pd.NaT, pd.Timestamp(disp.available_at).normalize() if disp is not None else pd.NaT, pd.Timestamp(struct.available_at).normalize() if struct is not None else pd.NaT, pd.Timestamp(companion.available_at).normalize() if companion is not None else pd.NaT] if pd.notna(x)], default=pd.NaT),
        })
    return pd.DataFrame(rows)


def passage_label(daily: pd.DataFrame, daily_dates: np.ndarray, prediction_at: Any,
                  target: float, invalidation: float, direction: str, horizon: int) -> dict[str, Any]:
    """First passage after a prediction clock; same-bar order is unknown."""
    out = {"label": "not_applicable", "complete": 0, "label_end_at": pd.NaT, "bars_to_label": np.nan,
           "target_hit": 0, "invalidation_hit": 0}
    if not (math.isfinite(fnum(target)) and math.isfinite(fnum(invalidation))):
        out["label"] = "target_or_invalidation_unknown"
        return out
    i = index_at_or_before(daily_dates, prediction_at)
    start, end = i + 1, min(len(daily), i + 1 + int(horizon))
    if start >= len(daily):
        out["label"] = "censored_no_future_bar"
        return out
    p = daily.iloc[start:end]
    if direction == "bullish":
        target_hit = p.high.to_numpy(float) >= target
        invalid_hit = p.low.to_numpy(float) <= invalidation
    else:
        target_hit = p.low.to_numpy(float) <= target
        invalid_hit = p.high.to_numpy(float) >= invalidation
    ti, ii = np.flatnonzero(target_hit), np.flatnonzero(invalid_hit)
    out["target_hit"], out["invalidation_hit"] = int(len(ti) > 0), int(len(ii) > 0)
    complete = (end - start) == int(horizon)
    if len(ti) and len(ii) and int(ti[0]) == int(ii[0]):
        j = int(ti[0]); out.update({"label": "same_bar_order_unknown", "complete": 1, "label_end_at": p.datetime.iloc[j], "bars_to_label": j + 1})
    elif len(ti) and (not len(ii) or ti[0] < ii[0]):
        j = int(ti[0]); out.update({"label": "target_first", "complete": 1, "label_end_at": p.datetime.iloc[j], "bars_to_label": j + 1})
    elif len(ii):
        j = int(ii[0]); out.update({"label": "invalidation_first", "complete": 1, "label_end_at": p.datetime.iloc[j], "bars_to_label": j + 1})
    elif complete:
        out.update({"label": "neither_at_horizon", "complete": 1, "label_end_at": p.datetime.iloc[-1], "bars_to_label": int(horizon)})
    else:
        out.update({"label": "censored_data_end", "label_end_at": p.datetime.iloc[-1] if len(p) else pd.NaT})
    return out


def touch_trigger(zone: pd.Series, touch_at: Any, touch_idx: Any, daily: pd.DataFrame,
                  daily_dates: np.ndarray, daily_events: pd.DataFrame, invalidation: float) -> dict[str, Any]:
    out = {"trigger_confirmed": 0, "trigger_at": pd.NaT, "trigger_event_id": "", "trigger_lag_bars": np.nan, "trigger_reason": "no_daily_mss_bos_after_touch"}
    if pd.isna(touch_at) or not math.isfinite(fnum(touch_idx)):
        out["trigger_reason"] = "no_touch"
        return out
    ti = int(touch_idx); t = pd.Timestamp(touch_at).normalize()
    direction = str(zone.direction)
    candidates = daily_events[(daily_events.direction == direction) & daily_events.event_type.isin(["mss", "bos"])].copy()
    if candidates.empty:
        return out
    candidates = normalize_dates(candidates, ["event_time", "available_at"])
    for e in candidates.sort_values(["available_at", "event_id"]).itertuples(index=False):
        et = pd.Timestamp(e.available_at).normalize()
        if et <= t:
            continue
        ei = index_at_or_before(daily_dates, et)
        if ei <= ti or ei - ti > POST_TRIGGER_BARS.get(str(zone.timeframe), 5):
            continue
        before = daily.iloc[ti + 1:ei + 1]
        if direction == "bullish":
            invalid = bool((before.low.to_numpy(float) <= invalidation).any())
        else:
            invalid = bool((before.high.to_numpy(float) >= invalidation).any())
        if invalid:
            continue
        out.update({"trigger_confirmed": 1, "trigger_at": et, "trigger_event_id": str(e.event_id), "trigger_lag_bars": ei - ti, "trigger_reason": "daily_mss_or_bos_after_touch"})
        return out
    return out


def feature_at(zone: pd.Series, t: Any, daily: pd.DataFrame, daily_dates: np.ndarray,
               contexts: pd.DataFrame, links: pd.DataFrame, prefix: str = "") -> dict[str, Any]:
    """Causal features at a specified clock; no outcome fields are included."""
    i = index_at_or_before(daily_dates, t)
    out: dict[str, Any] = {}
    if i < 0 or i >= len(daily):
        return out
    p = daily.iloc[i]
    close, atr = fnum(p.close), fnum(p.atr14)
    lo, hi = fnum(zone.zone_low), fnum(zone.zone_high)
    width = hi - lo if math.isfinite(lo) and math.isfinite(hi) else np.nan
    dist = 0.0 if math.isfinite(lo) and math.isfinite(hi) and lo <= close <= hi else min(abs(lo - close), abs(hi - close)) if math.isfinite(lo) and math.isfinite(hi) else np.nan
    out.update({f"{prefix}distance_atr": dist / atr if atr > 0 else np.nan, f"{prefix}width_atr": width / atr if atr > 0 else np.nan,
                f"{prefix}mid_distance_atr": abs(close - (lo + hi) / 2) / atr if atr > 0 else np.nan,
                f"{prefix}return_5": fnum(p.get("return_5")), f"{prefix}return_20": fnum(p.get("return_20")), f"{prefix}atr_pct": atr / close if close > 0 else np.nan,
                f"{prefix}volume_z20": fnum(p.get("volume_z20")), f"{prefix}sma_gap20": fnum(p.get("sma_gap20")), f"{prefix}sma_gap50": fnum(p.get("sma_gap50")),
                f"{prefix}bar_range_atr": fnum(p.get("range")) / atr if atr > 0 else np.nan, f"{prefix}close_location": fnum(p.get("close_location")),
                f"{prefix}age_days": (pd.Timestamp(t).normalize() - pd.Timestamp(zone.origin_time).normalize()).days,
                f"{prefix}object_is_ob": int(str(zone.object_type) == "ob_candidate"), f"{prefix}evidence_confirmed": int(str(zone.evidence_state) == "confirmed")})
    c = contexts[contexts.datetime <= pd.Timestamp(t).normalize()]
    if not c.empty:
        c = c.iloc[-1]
        direction = str(zone.direction)
        for tf in ["1w", "1m"]:
            trend, bias, posn, integ = str(c.get(f"{tf}_structure_trend", "unknown")), str(c.get(f"{tf}_bias_direction", "neutral")), str(c.get(f"{tf}_position", "unknown")), str(c.get(f"{tf}_structure_integrity", "unknown"))
            out.update({f"{prefix}{tf}_trend_up": int(trend == "up"), f"{prefix}{tf}_trend_down": int(trend == "down"), f"{prefix}{tf}_trend_mixed": int(trend == "mixed"), f"{prefix}{tf}_trend_unknown": int(trend == "unknown"),
                        f"{prefix}{tf}_bias_up": int(bias == "up"), f"{prefix}{tf}_bias_down": int(bias == "down"), f"{prefix}{tf}_bias_neutral": int(bias not in {"up", "down"}),
                        f"{prefix}{tf}_position_premium": int(posn == "premium"), f"{prefix}{tf}_position_discount": int(posn == "discount"), f"{prefix}{tf}_position_unknown": int(posn not in {"premium", "discount", "midpoint", "out_of_range"}),
                        f"{prefix}{tf}_integrity_intact": int(integ == "intact"), f"{prefix}{tf}_integrity_broken": int(integ == "broken"), f"{prefix}{tf}_integrity_unknown": int(integ == "unknown"),
                        f"{prefix}{tf}_trend_aligned": int((trend == "up" and direction == "bullish") or (trend == "down" and direction == "bearish")),
                        f"{prefix}{tf}_bias_aligned": int((bias == "up" and direction == "bullish") or (bias == "down" and direction == "bearish"))})
    q = links[links.object_id.eq(str(zone.object_id))] if not links.empty else pd.DataFrame()
    if not q.empty:
        q = q.iloc[0]
        for col in ["sequence_sweep_before", "sequence_displacement_before", "sequence_structure_before", "sequence_fvg_companion_before", "sequence_order_valid", "sequence_event_count"]:
            out[f"{prefix}{col}"] = fnum(q.get(col), 0.0)
    return out


def build_catalog(objects: pd.DataFrame, events: pd.DataFrame, daily: pd.DataFrame,
                  contexts: pd.DataFrame, frames: dict[str, pd.DataFrame], pivots_map: dict[tuple[str, str], pd.DataFrame],
                  gap_dates: set[pd.Timestamp]) -> tuple[pd.DataFrame, pd.DataFrame]:
    bal = objects[(objects.sensitivity == "balanced") & objects.timeframe.isin(EVAL_TIMEFRAMES) & objects.object_type.isin(EVAL_TYPES)].copy()
    bal = bal[~bal.gap_contaminated.astype(bool)].copy()
    links = build_ordered_links(objects, events)
    daily_events = events[(events.sensitivity == "balanced") & (events.timeframe == "1D")].copy()
    daily_events = normalize_dates(daily_events, ["event_time", "available_at"])
    _, daily_dates = date_index(daily)
    records = []
    for _, zone in bal.iterrows():
        direction, typ, tf = str(zone.direction), str(zone.object_type), str(zone.timeframe)
        target = target_at(zone, daily, daily_dates, frames, pivots_map)
        loc = location_at(zone, contexts)
        q = links[links.object_id.eq(str(zone.object_id))]
        link = q.iloc[0].to_dict() if not q.empty else {}
        adapted = bool(typ == "fvg" or (typ == "ob_candidate" and str(zone.evidence_state) == "confirmed"))
        seq_fvg = bool(fnum(link.get("sequence_fvg_companion_before"), 0) == 1)
        seq_disp = bool(fnum(link.get("sequence_displacement_before"), 0) == 1)
        seq_sweep = bool(fnum(link.get("sequence_sweep_before"), 0) == 1)
        seq_struct = bool(fnum(link.get("sequence_structure_before"), 0) == 1)
        if typ == "fvg":
            scenario = "fvg_continuation"
            complete = adapted and seq_disp and seq_struct and bool(target["target_known"]) and bool(loc["location_aligned"]) and bool(loc["htf_trend_aligned"]) and not bool(loc["htf_integrity_broken"])
        else:
            scenario = "ob_reversal"
            complete = adapted and seq_sweep and seq_disp and seq_struct and seq_fvg and bool(target["target_known"]) and bool(loc["location_aligned"]) and bool(loc["htf_trend_aligned"]) and not bool(loc["htf_integrity_broken"])
        lb = v4.label_object_v04(zone, daily, arrival_horizon=ARRIVAL_HORIZON[tf], post_touch_horizon=TARGET_HORIZON[tf], barrier_atr=1.0)
        rec = {**zone.to_dict(), **target, **loc, **link, "scenario_family": scenario, "primitive_form": 1, "adapted_confirmed": int(adapted), "complete_adapted": int(complete),
               "setup_state": "complete_adapted" if complete else "adapted_confirmed" if adapted else "primitive", "source_setup_status": "daily_weekly_adaptation_only_ltf_mss_unknown" if typ == "ob_candidate" else "daily_weekly_basic_fvg_observable",
               "target_definition": "next_unconsumed_confirmed_pivot_same_timeframe", "invalidation_definition": "zone_boundary_plus_0.10_ATR_at_formation", "stage_prediction_at": pd.Timestamp(zone.available_at).normalize()}
        rec.update({"formation_target_price": target["target_price"], "retest_target_price": target["target_price"], "trigger_target_price": target["target_price"], "formation_invalidation_price": target["invalidation_price"], "retest_invalidation_price": target["invalidation_price"], "trigger_invalidation_price": target["invalidation_price"]})
        rec.update(feature_at(zone, zone.available_at, daily, daily_dates, contexts, links, ""))
        rec.update({"arrival_label": lb.get("arrival_label", ""), "arrival_type": lb.get("arrival_type", ""), "arrival_complete": int(bool(lb.get("arrival_complete", False))), "arrival_label_end_at": lb.get("arrival_label_end_at", pd.NaT), "touch_at": lb.get("touch_at", pd.NaT), "touch_idx": lb.get("touch_idx", np.nan)})
        formation_path = passage_label(daily, daily_dates, zone.available_at, fnum(target["target_price"]), fnum(target["invalidation_price"]), direction, TARGET_HORIZON[tf])
        for k, val in formation_path.items(): rec[f"formation_target_{k}"] = val
        rec["formation_target_model_eligible"] = int(bool(target["target_known"]) and formation_path["complete"] and formation_path["label"] in {"target_first", "invalidation_first", "neither_at_horizon"})
        touch = lb.get("touch_at", pd.NaT); ti = lb.get("touch_idx", np.nan)
        rec["target_precondition_status"] = "no_touch"
        rec["retest_target_label"] = "not_applicable"; rec["retest_target_complete"] = 0; rec["retest_target_label_end_at"] = pd.NaT; rec["retest_target_model_eligible"] = 0
        rec["trigger_target_label"] = "not_applicable"; rec["trigger_target_complete"] = 0; rec["trigger_target_label_end_at"] = pd.NaT; rec["trigger_target_model_eligible"] = 0
        if pd.notna(touch) and math.isfinite(fnum(ti)) and bool(target["target_known"]):
            ti = int(ti); form_i = index_at_or_before(daily_dates, zone.available_at)
            pre = daily.iloc[form_i + 1:ti]
            target_before = bool((pre.high.to_numpy(float) >= target["target_price"]).any()) if direction == "bullish" else bool((pre.low.to_numpy(float) <= target["target_price"]).any())
            inv_before = bool((pre.low.to_numpy(float) <= target["invalidation_price"]).any()) if direction == "bullish" else bool((pre.high.to_numpy(float) >= target["invalidation_price"]).any())
            if target_before: rec["target_precondition_status"] = "target_consumed_before_touch"
            elif inv_before: rec["target_precondition_status"] = "invalidation_before_touch"
            else:
                rec["target_precondition_status"] = "target_and_invalidation_open_at_touch"
                retest = passage_label(daily, daily_dates, touch, fnum(target["target_price"]), fnum(target["invalidation_price"]), direction, TARGET_HORIZON[tf])
                for k, val in retest.items(): rec[f"retest_target_{k}"] = val
                rec["retest_target_model_eligible"] = int(retest["complete"] and retest["label"] in {"target_first", "invalidation_first", "neither_at_horizon"})
            trig = touch_trigger(zone, touch, ti, daily, daily_dates, daily_events, fnum(target["invalidation_price"]))
            rec.update(trig)
            if bool(trig["trigger_confirmed"]) and rec["target_precondition_status"] == "target_and_invalidation_open_at_touch":
                trigger_path = passage_label(daily, daily_dates, trig["trigger_at"], fnum(target["target_price"]), fnum(target["invalidation_price"]), direction, TARGET_HORIZON[tf])
                for k, val in trigger_path.items(): rec[f"trigger_target_{k}"] = val
                rec["trigger_target_model_eligible"] = int(trigger_path["complete"] and trigger_path["label"] in {"target_first", "invalidation_first", "neither_at_horizon"})
        else:
            rec.update({"trigger_confirmed": 0, "trigger_at": pd.NaT, "trigger_event_id": "", "trigger_lag_bars": np.nan, "trigger_reason": "no_touch_or_target"})
        if pd.notna(touch): rec.update(feature_at(zone, touch, daily, daily_dates, contexts, links, "touch_"))
        if pd.notna(rec.get("trigger_at")): rec.update(feature_at(zone, rec["trigger_at"], daily, daily_dates, contexts, links, "trigger_"))
        rec["arrival_model_eligible"] = int(rec["arrival_type"] in {"touch", "no_arrival"} and rec["arrival_complete"] and not interval_has_gap(rec["stage_prediction_at"], rec["arrival_label_end_at"], gap_dates))
        rec["formation_target_model_eligible"] = int(rec["formation_target_model_eligible"] and not interval_has_gap(rec["stage_prediction_at"], rec["formation_target_label_end_at"], gap_dates))
        rec["retest_target_model_eligible"] = int(rec["retest_target_model_eligible"] and not interval_has_gap(rec["touch_at"], rec["retest_target_label_end_at"], gap_dates))
        rec["trigger_target_model_eligible"] = int(rec["trigger_target_model_eligible"] and not interval_has_gap(rec["trigger_at"], rec["trigger_target_label_end_at"], gap_dates))
        rec["arrival_gap_contaminated"] = int(bool(rec["arrival_type"] in {"touch", "no_arrival"} and interval_has_gap(rec["stage_prediction_at"], rec["arrival_label_end_at"], gap_dates)))
        rec["formation_target_gap_contaminated"] = int(bool(interval_has_gap(rec["stage_prediction_at"], rec["formation_target_label_end_at"], gap_dates)))
        rec["retest_target_gap_contaminated"] = int(bool(interval_has_gap(rec["touch_at"], rec["retest_target_label_end_at"], gap_dates)))
        rec["trigger_target_gap_contaminated"] = int(bool(interval_has_gap(rec["trigger_at"], rec["trigger_target_label_end_at"], gap_dates)))
        records.append(rec)
    return pd.DataFrame(records).sort_values(["stage_prediction_at", "object_id"]).reset_index(drop=True), links


BASE_FEATURES = ["distance_atr", "width_atr", "mid_distance_atr", "return_5", "return_20", "atr_pct", "volume_z20", "sma_gap20", "sma_gap50", "bar_range_atr", "close_location", "age_days", "object_is_ob", "evidence_confirmed"]
CONTEXT_FEATURES = [f"{tf}_{x}" for tf in ["1w", "1m"] for x in ["trend_up", "trend_down", "trend_mixed", "trend_unknown", "bias_up", "bias_down", "bias_neutral", "position_premium", "position_discount", "position_unknown", "integrity_intact", "integrity_broken", "integrity_unknown", "trend_aligned", "bias_aligned"]]
SEQUENCE_FEATURES = ["sequence_sweep_before", "sequence_displacement_before", "sequence_structure_before", "sequence_fvg_companion_before", "sequence_order_valid", "sequence_event_count"]


def feature_columns(clock: str) -> list[str]:
    p = "" if clock == "formation" else f"{clock}_"
    return [f"{p}{x}" for x in BASE_FEATURES + CONTEXT_FEATURES + SEQUENCE_FEATURES]


def stage_mask(catalog: pd.DataFrame, stage: str) -> pd.Series:
    return catalog[{"primitive_form": "primitive_form", "adapted_confirmed": "adapted_confirmed", "complete_adapted": "complete_adapted", "retest_confirmed": "trigger_confirmed"}[stage]].astype(bool)


CLOCKS = {
    "arrival": ("arrival_model_eligible", "arrival_label", "stage_prediction_at", "arrival_label_end_at", "touch"),
    "formation_target": ("formation_target_model_eligible", "formation_target_label", "stage_prediction_at", "formation_target_label_end_at", "target"),
    "retest_target": ("retest_target_model_eligible", "retest_target_label", "touch_at", "retest_target_label_end_at", "target"),
    "trigger_target": ("trigger_target_model_eligible", "trigger_target_label", "trigger_at", "trigger_target_label_end_at", "target"),
}


def wilson(success: int, n: int) -> tuple[float, float]:
    if n <= 0: return np.nan, np.nan
    z = 1.959963984540054
    p = success / n; den = 1 + z * z / n; center = (p + z * z / (2 * n)) / den; half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return center - half, center + half


def realization_summary(catalog: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    stages = [("primitive_form", "primitive_form"), ("adapted_confirmed", "adapted_confirmed"), ("complete_adapted", "complete_adapted"), ("retest_confirmed", "retest_confirmed")]
    for stage, _ in stages:
        base = catalog[stage_mask(catalog, stage)].copy()
        for clock, (_, label_col, _, _, _) in CLOCKS.items():
            # retest_confirmed is only known after the touch.  Reporting it
            # against formation/arrival/retest clocks would condition on a
            # future event and then project that selected cohort backward.
            if stage == "retest_confirmed" and clock != "trigger_target":
                continue
            if base.empty: continue
            elig_col = CLOCKS[clock][0]
            s = base[base[elig_col].astype(bool)].copy()
            if s.empty: continue
            for (scenario, tf, direction), g in s.groupby(["scenario_family", "timeframe", "direction"], dropna=False):
                label = g[label_col].astype(str)
                if clock == "arrival":
                    arrival_numeric = pd.to_numeric(label, errors="coerce")
                    success = int((arrival_numeric == 1).sum())
                    arrival_touch_n = int((g.arrival_type == "touch").sum())
                    arrival_gap_through_n = int((g.arrival_type == "gap_through").sum())
                    arrival_no_arrival_n = int((g.arrival_type == "no_arrival").sum())
                    arrival_censored_n = int(g.arrival_type.astype(str).str.startswith("censored").sum())
                else:
                    success = int((label == "target_first").sum())
                    arrival_touch_n = arrival_gap_through_n = arrival_no_arrival_n = arrival_censored_n = 0
                lo, hi = wilson(success, len(g))
                rows.append({"stage": stage, "clock": clock, "scenario_family": scenario, "timeframe": tf, "direction": direction, "eligible_n": int(len(g)), "success_n": success, "success_rate": success / len(g), "ci95_low": lo, "ci95_high": hi,
                             "invalidation_first_n": int((label == "invalidation_first").sum()), "target_first_n": int((label == "target_first").sum()), "neither_n": int((label == "neither_at_horizon").sum()), "same_bar_unknown_n": int((label == "same_bar_order_unknown").sum()), "precondition_excluded_n": int((g.target_precondition_status != "target_and_invalidation_open_at_touch").sum()) if clock in {"retest_target", "trigger_target"} else 0})
                rows[-1].update({"arrival_touch_n": arrival_touch_n, "arrival_gap_through_n": arrival_gap_through_n,
                                 "arrival_no_arrival_n": arrival_no_arrival_n, "arrival_censored_n": arrival_censored_n})
    summary = pd.DataFrame(rows)
    comparisons = []
    if not summary.empty:
        keys = ["clock", "scenario_family", "timeframe", "direction"]
        for key, g in summary.groupby(keys, dropna=False):
            a = g[g.stage == "adapted_confirmed"]; c = g[g.stage == "complete_adapted"]
            if not a.empty and not c.empty:
                ar, cr = float(a.iloc[0].success_rate), float(c.iloc[0].success_rate)
                comparisons.append({**dict(zip(keys, key)), "adapted_n": int(a.iloc[0].eligible_n), "adapted_rate": ar, "complete_n": int(c.iloc[0].eligible_n), "complete_rate": cr, "complete_minus_adapted": cr - ar})
    return summary, pd.DataFrame(comparisons)


def fit_binary(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, str]:
    prior = float(train.y.mean()) if len(train) else np.nan
    if len(train) < MIN_TRAIN or train.y.nunique() < 2 or not cols:
        return np.full(len(test), prior), "training_prior"
    pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("logit", LogisticRegression(C=1.0, max_iter=1500, solver="lbfgs"))])
    pipe.fit(train[cols].apply(pd.to_numeric, errors="coerce"), train.y.astype(int))
    return pipe.predict_proba(test[cols].apply(pd.to_numeric, errors="coerce"))[:, 1], "logistic_fixed_C1"


def logloss_binary(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(np.asarray(p, float), 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def folds(dates: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    u = pd.Series(pd.to_datetime(dates).dropna().sort_values().unique())
    if len(u) < 5: return []
    qs = sorted(set(pd.Timestamp(x).normalize() for x in u.quantile([0.50, 0.60, 0.70, 0.80, 0.90])))
    return [(qs[i], (qs + [u.max() + pd.Timedelta(days=1)])[i + 1]) for i in range(len(qs))]


def wf_task(payload: tuple[pd.DataFrame, str, str, str, str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    catalog, stage, clock, scenario, tf, direction = payload
    data = catalog[stage_mask(catalog, stage) & catalog.scenario_family.eq(scenario) & catalog.timeframe.eq(tf) & catalog.direction.eq(direction)].copy()
    elig_col, label_col, pred_col, end_col, _ = CLOCKS[clock]
    data = data[data[elig_col].astype(bool)].copy()
    if len(data) < MIN_TEST: return pd.DataFrame(), pd.DataFrame()
    if clock == "arrival":
        data["y"] = pd.to_numeric(data[label_col], errors="coerce").fillna(0).astype(int)
    else:
        data["y"] = (data[label_col].astype(str) == "target_first").astype(int)
    data["prediction_at"] = pd.to_datetime(data[pred_col]).dt.normalize(); data["label_end_at"] = pd.to_datetime(data[end_col]).dt.normalize()
    cols = feature_columns("formation" if clock in {"arrival", "formation_target"} else "touch" if clock == "retest_target" else "trigger")
    metrics, pred_rows = [], []
    for fid, (start, end) in enumerate(folds(data.prediction_at), start=1):
        train = data[(data.prediction_at < start) & (data.label_end_at < start)].copy()
        test = data[(data.prediction_at >= start) & (data.prediction_at < end)].copy()
        if len(train) < MIN_TRAIN or len(test) < MIN_TEST: continue
        prior = float(train.y.mean()); p0 = np.full(len(test), prior); p1, status = fit_binary(train, test, [c for c in cols if c in train and c in test])
        for model, p in [("intercept_only", p0), ("causal_context", p1)]:
            y = test.y.to_numpy(int)
            metrics.append({"stage": stage, "clock": clock, "scenario_family": scenario, "timeframe": tf, "direction": direction, "fold_id": fid, "test_start": start, "test_end": end, "train_n": len(train), "test_n": len(test), "model": model, "model_status": "training_prior" if model == "intercept_only" else status, "brier": float(np.mean((p - y) ** 2)), "log_loss": logloss_binary(y, p), "max_train_label_end_at": train.label_end_at.max(), "min_test_prediction_at": test.prediction_at.min(), "max_test_prediction_at": test.prediction_at.max()})
            for j, (_, r) in enumerate(test.iterrows()):
                pred_rows.append({"object_id": r.object_id, "stage": stage, "clock": clock, "scenario_family": scenario, "timeframe": tf, "direction": direction, "fold_id": fid, "prediction_at": r.prediction_at, "label_end_at": r.label_end_at, "model": model, "y": int(r.y), "p_target_or_touch": float(p[j])})
    return pd.DataFrame(metrics), pd.DataFrame(pred_rows)


def walk_forward(catalog: pd.DataFrame, n_jobs: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    payloads = []
    for stage in ["primitive_form", "adapted_confirmed", "complete_adapted", "retest_confirmed"]:
        for clock in CLOCKS:
            if stage == "retest_confirmed" and clock != "trigger_target":
                continue
            for (scenario, tf, direction), _ in catalog[stage_mask(catalog, stage)].groupby(["scenario_family", "timeframe", "direction"]):
                payloads.append((catalog, stage, clock, scenario, tf, direction))
    ms, ps = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, n_jobs)) as pool:
        for fut in [pool.submit(wf_task, p) for p in payloads]:
            m, p = fut.result()
            if not m.empty: ms.append(m)
            if not p.empty: ps.append(p)
    return (pd.concat(ms, ignore_index=True) if ms else pd.DataFrame(), pd.concat(ps, ignore_index=True) if ps else pd.DataFrame())


def metric_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty: return pd.DataFrame()
    gcols = ["stage", "clock", "scenario_family", "timeframe", "direction", "model"]
    out = metrics.groupby(gcols, dropna=False).apply(lambda g: pd.Series({"folds": int(g.fold_id.nunique()), "n_test": int(g.test_n.sum()), "brier": float(np.average(g.brier, weights=g.test_n)), "log_loss": float(np.average(g.log_loss, weights=g.test_n)), "brier_fold_sd": float(g.brier.std(ddof=1)) if len(g) > 1 else np.nan}), include_groups=False).reset_index()
    pri = out[out.model.eq("intercept_only")][gcols[:-1] + ["brier", "log_loss"]].rename(columns={"brier": "intercept_brier", "log_loss": "intercept_log_loss"})
    out = out.merge(pri, on=gcols[:-1], how="left")
    out["delta_brier_vs_intercept"] = out.brier - out.intercept_brier; out["delta_log_loss_vs_intercept"] = out.log_loss - out.intercept_log_loss
    return out


def calibration_table(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty: return pd.DataFrame()
    rows = []
    for keys, g in predictions.groupby(["stage", "clock", "scenario_family", "timeframe", "direction", "model"], dropna=False):
        g = g.copy(); g["bin"] = pd.cut(g.p_target_or_touch, bins=[-0.001, .2, .4, .6, .8, 1.001], include_lowest=True)
        for b, h in g.groupby("bin", observed=False):
            if h.empty: continue
            rows.append({**dict(zip(["stage", "clock", "scenario_family", "timeframe", "direction", "model"], keys)), "probability_bin": str(b), "n": int(len(h)), "mean_predicted": float(h.p_target_or_touch.mean()), "observed_rate": float(h.y.mean())})
    return pd.DataFrame(rows)


def article_ledger(catalog: pd.DataFrame, objects: pd.DataFrame, events: pd.DataFrame, source_coverage: Path, source_manifest: Path, detector_output: Path, output_root: Path) -> pd.DataFrame:
    coverage = json.loads(source_coverage.read_text(encoding="utf-8")) if source_coverage.exists() else []
    source_rows = pd.read_csv(source_manifest).to_dict("records") if source_manifest.exists() else []
    source_by_url = {str(x.get("canonical_url", x.get("url", ""))): x for x in source_rows}
    direct = {24: ("cisd;mss", "events", "daily_weekly_core_events"), 41: ("cisd", "events", "daily_weekly_core_events"), 61: ("liquidity_breach;liquidity_sweep", "events", "daily_weekly_core_events"), 75: ("displacement", "objects", "daily_weekly_core_objects"), 81: ("swing_low_3bar", "objects", "daily_weekly_core_objects"), 82: ("swing_high_3bar", "objects", "daily_weekly_core_objects"), 111: ("mss", "events", "daily_weekly_core_events"), 114: ("fvg_3bar", "objects", "daily_weekly_core_objects")}
    shared = {100: ("fvg_3bar", "objects", "shared_fvg_rule")}
    adapted = {87: ("ob_candidate_dw", "objects", "daily_weekly_candidate_only"), 88: ("ob_candidate_dw", "objects", "daily_weekly_candidate_only"), 115: ("ob_candidate_dw", "objects", "daily_weekly_candidate_only")}
    out = []
    bal_obj = objects[objects.sensitivity.eq("balanced")]; bal_ev = events[events.sensitivity.eq("balanced")]
    for src in coverage:
        n = int(src["article_number"]); mapping = direct.get(n, shared.get(n, adapted.get(n, ("", "", ""))))
        rule, evidence_kind, scope = mapping
        basic_n = event_n = 0
        if n in direct: status = "implemented_basic_shape"
        elif n in shared: status = "shared_basic_shape"
        elif n in adapted: status = "adapted_candidate_only"
        elif src.get("implementation_status") == "requires_extra_data": status = "requires_extra_data_unimplemented"
        elif src.get("implementation_status") == "reference_only": status = "reference_only"
        else: status = "not_implemented"
        title = str(src.get("title", "")); lower = title.lower(); source = source_by_url.get(str(src.get("source_url", "")), {})
        if n in {87, 88, 115}:
            title_dir = "bullish" if "bullish" in lower else "bearish" if "bearish" in lower else "both_or_unspecified"
            subset = bal_obj[bal_obj.rule_id.eq("ob_candidate_dw")]
            if title_dir in {"bullish", "bearish"}: subset = subset[subset.direction.eq(title_dir)]
            basic_n, adapted_n, complete_n = len(subset), int(subset.adapted_confirmed.astype(bool).sum()) if "adapted_confirmed" in subset else 0, int(catalog[catalog.object_id.isin(set(subset.object_id))].complete_adapted.sum())
            missing = "lower_timeframe_imbalance;lower_timeframe_MSS;original_entry_confirmation;strict_source_setup_not_verified"
            original_n = 0
        elif n in {100, 114}:
            subset = bal_obj[bal_obj.rule_id.eq("fvg_3bar")]; basic_n, adapted_n, complete_n, original_n = len(subset), int(catalog[catalog.object_id.isin(set(subset.object_id))].adapted_confirmed.sum()), int(catalog[catalog.object_id.isin(set(subset.object_id))].complete_adapted.sum()), len(subset); missing = "article_specific_target_and_execution_flow_is_not_a_standalone_detector"
            title_dir = "both_or_unspecified"
        else:
            rule_parts = rule.split(";") if rule else []
            if evidence_kind == "objects": basic_n = int(bal_obj[bal_obj.rule_id.isin(rule_parts)].shape[0])
            elif evidence_kind == "events": event_n = int(bal_ev[bal_ev.rule_id.isin(rule_parts)].shape[0]); basic_n = 0
            else: basic_n = event_n = 0
            adapted_n = complete_n = np.nan; original_n = basic_n or event_n; missing = str(src.get("limitation", "not implemented")); title_dir = "both_or_unspecified"
        evidence_path = str((detector_output / ("objects_v09.csv" if evidence_kind == "objects" else "events_v09.csv")).resolve()) if evidence_kind in {"objects", "events"} and (detector_output / ("objects_v09.csv" if evidence_kind == "objects" else "events_v09.csv")).exists() else ""
        out.append({"article_number": n, "title": title, "source_url": src.get("source_url", ""), "source_table_status": src.get("implementation_status", ""), "verified_status": status, "rule_ids": rule, "verified_scope": scope, "article_direction_scope": title_dir, "source_local_manifest": int(bool(source)), "source_html_sha256": source.get("html_sha256", ""), "source_text_chars": source.get("text_chars", np.nan), "source_image_count": source.get("image_count", np.nan), "basic_evidence_count_balanced": basic_n, "event_evidence_count_balanced": event_n, "adapted_confirmed_count": adapted_n, "complete_adapted_count": complete_n, "original_confirmed_count": original_n, "complete_stage_scope": "daily_weekly_adaptation_only" if n in set(adapted) | set(shared) else "not_a_standalone_setup", "missing_or_unknown_conditions": missing, "implementation_evidence": evidence_path, "image_semantic_evidence": "not_independent_blind_label;local_image_presence_only" if source else "source_not_joined"})
    return pd.DataFrame(out).sort_values("article_number").reset_index(drop=True)


def qa_table(daily: pd.DataFrame, objects: pd.DataFrame, events: pd.DataFrame, catalog: pd.DataFrame, links: pd.DataFrame, ledger: pd.DataFrame, metrics: pd.DataFrame, gap_dates: set[pd.Timestamp], detector_output: Path) -> pd.DataFrame:
    rows = []
    def add(name: str, passed: bool, observed: Any, expected: Any, severity: str = "error"):
        rows.append({"check": name, "passed": bool(passed), "observed": observed, "expected": expected, "severity": severity})
    add("input_unique_monotonic", daily.datetime.is_monotonic_increasing and daily.datetime.duplicated().sum() == 0, f"rows={len(daily)};duplicates={int(daily.datetime.duplicated().sum())}", "monotonic;duplicates=0")
    add("article_ledger_94_rows", len(ledger) == 94 and ledger.article_number.nunique() == 94, len(ledger), 94)
    exec_rows = ledger[ledger.verified_status.isin(["implemented_basic_shape", "shared_basic_shape", "adapted_candidate_only"])]
    add("article_evidence_paths_exist", bool((exec_rows.implementation_evidence.astype(str).str.len() > 0).all()) if not exec_rows.empty else False, int((exec_rows.implementation_evidence.astype(str).str.len() > 0).sum()), int(len(exec_rows)))
    add("stage_monotone", bool((catalog.adapted_confirmed.astype(int) <= catalog.primitive_form.astype(int)).all() and (catalog.complete_adapted.astype(int) <= catalog.adapted_confirmed.astype(int)).all()), "adapted<=primitive;complete<=adapted", "true")
    add("ob_lower_tf_unknown_explicit", bool((catalog.loc[catalog.object_type.eq("ob_candidate"), "source_setup_status"].astype(str).str.contains("ltf_mss_unknown")).all()), int((catalog.object_type.eq("ob_candidate") & ~catalog.source_setup_status.astype(str).str.contains("ltf_mss_unknown")).sum()), 0)
    add("target_known_has_price", bool((catalog.loc[catalog.target_known.astype(bool), "target_price"].map(fnum).apply(math.isfinite)).all()), int((catalog.target_known.astype(bool) & ~catalog.target_price.map(fnum).apply(math.isfinite)).sum()), 0)
    target_cols = ["target_price", "formation_target_price", "retest_target_price", "trigger_target_price"]
    known = catalog.target_known.astype(bool)
    target_equal = bool((catalog.loc[known, target_cols].apply(pd.to_numeric, errors="coerce").nunique(axis=1) <= 1).all()) if known.any() else True
    add("target_same_across_clocks", target_equal, "one frozen target_price copied to all clocks", "equal where target is known")
    sequence_at = pd.to_datetime(catalog.sequence_available_at, errors="coerce")
    zone_at = pd.to_datetime(catalog.available_at, errors="coerce")
    sequence_after_zone = sequence_at.notna() & zone_at.notna() & (sequence_at > zone_at)
    sequence_ok = (~sequence_after_zone).all()
    add("sequence_available_not_after_zone", bool(sequence_ok), int(sequence_after_zone.sum()), 0)
    gap_cols = ["arrival_gap_contaminated", "formation_target_gap_contaminated", "retest_target_gap_contaminated", "trigger_target_gap_contaminated"]
    eligible_cols = ["arrival_model_eligible", "formation_target_model_eligible", "retest_target_model_eligible", "trigger_target_model_eligible"]
    eligible_gap = int(sum((catalog[ec].astype(bool) & catalog[gc].astype(bool)).sum() for ec, gc in zip(eligible_cols, gap_cols)))
    add("eligible_windows_clean", eligible_gap == 0, eligible_gap, 0)
    add("lower_tf_trigger_after_touch", bool((~catalog.trigger_confirmed.astype(bool) | (pd.to_datetime(catalog.trigger_at) > pd.to_datetime(catalog.touch_at))).all()), int(((catalog.trigger_confirmed.astype(bool)) & ~(pd.to_datetime(catalog.trigger_at) > pd.to_datetime(catalog.touch_at))).sum()), 0)
    add("wf_train_label_end_before_test", bool((pd.to_datetime(metrics.max_train_label_end_at) < pd.to_datetime(metrics.test_start)).all()) if not metrics.empty else False, int((pd.to_datetime(metrics.max_train_label_end_at) >= pd.to_datetime(metrics.test_start)).sum()) if not metrics.empty else 0, 0)
    add("wf_has_metrics", not metrics.empty, int(len(metrics)), ">0")
    add("detector_artifacts_exist", (detector_output / "objects_v09.csv").exists() and (detector_output / "events_v09.csv").exists(), str(detector_output), "objects_v09.csv and events_v09.csv")
    add("event_ids_unique", events.event_id.duplicated().sum() == 0, int(events.event_id.duplicated().sum()), 0)
    add("catalog_object_ids_unique", catalog.object_id.duplicated().sum() == 0, int(catalog.object_id.duplicated().sum()), 0)
    add("gap_ledger_loaded_or_empty_explicit", True, len(gap_dates), "recorded")
    return pd.DataFrame(rows)


def report(manifest: dict[str, Any], ledger: pd.DataFrame, realization: pd.DataFrame, comparisons: pd.DataFrame, metrics: pd.DataFrame, qa: pd.DataFrame) -> str:
    status = ledger.verified_status.value_counts().rename_axis("verified_status").reset_index(name="articles").to_markdown(index=False) if not ledger.empty else ""
    rt = realization.to_markdown(index=False) if not realization.empty else ""
    ct = comparisons.to_markdown(index=False) if not comparisons.empty else ""
    mt = metric_summary(metrics).to_markdown(index=False) if not metrics.empty else ""
    return "\n".join([
        "# NQ=F ICT検知・評価 v0.9（記事型シナリオ・固定DOL目標版）", "",
        "## 判定", "",
        "今回の修正版は、94記事全体を一つの売買条件へ圧縮せず、日足・週足OHLCで観測できる範囲の基本形、適応確認、完成した局所シナリオを分けた。主評価は、事前に固定した目標と無効化境界へ到達する経路の実現率であり、モデルのBrier/log loss改善は別の追加評価である。", "",
        f"入力は{manifest['data_window']['start']}〜{manifest['data_window']['end']}、{manifest['data_window']['rows']:,}日足（約{manifest['coverage']['actual_years']:.2f}年）。", "",
        "## 94記事との対応", "", status, "",
        "台帳で実処理へ接続できた記事は、共通のFVG・Swing・Displacement・Sweep・MSS・OB候補のどれを使うかを記録した。OBの原記事が要求する下位足不均衡と下位足MSSは日足入力から確認できないため、OBは原記事完成形ではなく日足・週足適応候補として扱った。", "",
        "## 主ラベル", "",
        "`arrival`は形成後に帯へ最初に到達するか、`formation_target`は形成時点から固定DOL目標と無効化境界のどちらへ先に進むか、`retest_target`は初回接触後、`trigger_target`は接触後の日足MSS/BOS確認後に同じ固定目標へ進むかを表す。接触足・確認足より前の情報を後段のラベルへ戻していない。", "",
        "実現率は、該当clockの評価可能な分母に対するtarget_first（arrivalではtouch）の割合。`neither_at_horizon`、同一足内で順序不明、データ終端、目標消費済みは成功・失敗へ混ぜていない。", "", rt, "",
        "## 適応条件を追加した差", "", ct, "",
        "この差は、同一シナリオ・時間足・方向・時計における適応確認済み群と完成適応群の差であり、独立標本ではない。無関係な価格帯を対照にして理論全体を合否判定するものではない。", "",
        "## ウォークフォワード", "", mt, "",
        "intercept_onlyは、その時点以前の同じ段階・型・時間足・方向の実現率だけを使う基準、causal_contextは形成・接触・確認時点で観測できる幾何・週足/月足構造・順序フラグを使うモデルである。これはセットアップの実現率と、個別事例の確率を付け分ける能力を分けるための比較である。", "",
        "## 重要な制約", "",
        "- 94記事のうち大部分は時刻、分足、別銘柄、出来高の細粒度など追加データを要し、今回のNQ日足・週足で原記事どおりには検証できない。", "- `complete_adapted`は、明示した日足・週足派生条件であり、ICT原記事の下位足エントリーを再現したものではない。", "- DOLは、同時間足で予測時点までに確認済みで未消費の機械的Pivotを使う固定規則。機関注文や因果メカニズムを観測したものではない。", "- Yahoo連続先物代理系列の限月接続・出来高の性質は取引所の完全な板・約定データと同じではない。", "- 実データは約25.9年で、要求した30年を満たしていない。", "- 完成群の実現率が高くても、出現頻度、目標までの時間、費用、重複エクスポージャーを含む売買成績とは別である。", "",
        "## 出力", "", "`article_ledger_v09.csv`は94記事の台帳、`setup_catalog_v09.csv`は各帯の段階・固定目標・4つの時計、`realization_summary_v09.csv`はシナリオ実現率、`walk_forward_metric_summary_v09.csv`は確率モデル評価、`validation_tests_v09.csv`は整合性QAである。", "",
    ]) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--detector-output", type=Path, default=DEFAULT_DETECTOR_OUTPUT)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--n-jobs", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    ap.add_argument("--backend", choices=["process", "thread", "serial"], default="process")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True); args.detector_output.mkdir(parents=True, exist_ok=True)
    daily, data_qa = detector.load_daily(args.input); daily = v08.add_model_features(daily)
    gap_dates, gap_qa = gap_dates_for_input(args.input)
    frames_all = {"1D": daily, "1W": v08.period_bars(daily, "W-FRI"), "1M": v08.period_bars(daily, "M")}
    frames = {"1D": frames_all["1D"], "1W": frames_all["1W"][frames_all["1W"].certified_calendar].reset_index(drop=True), "1M": frames_all["1M"][frames_all["1M"].certified_calendar].reset_index(drop=True)}
    objects, events, pivots_map, parallel = v08.run_lanes(frames, max(1, int(args.n_jobs)), args.backend)
    objects = v08.add_object_metadata(objects); objects["rule_version"] = SCRIPT_VERSION; objects = normalize_dates(objects, ["origin_time", "confirmed_at", "available_at"]); events = normalize_dates(events, ["event_time", "available_at"]); objects = mark_gaps(objects, gap_dates)
    write_df(objects, args.detector_output / "objects_v09.csv"); write_df(events, args.detector_output / "events_v09.csv")
    context_objects = objects[(objects.sensitivity == "balanced") & objects.timeframe.isin(["1W", "1M"])].copy()
    context_life = {tf: v08.zone_lifecycle_fast(context_objects[context_objects.timeframe == tf], daily) for tf in ["1W", "1M"]}
    contexts = v08.build_contexts(daily, frames, pivots_map, context_objects, context_life)
    catalog, links = build_catalog(objects, events, daily, contexts, frames, pivots_map, gap_dates)
    realization, comparisons = realization_summary(catalog)
    metrics, predictions = walk_forward(catalog, max(1, int(args.n_jobs))); ms = metric_summary(metrics); calib = calibration_table(predictions)
    ledger = article_ledger(catalog, objects, events, SOURCE_COVERAGE, SOURCE_MANIFEST, args.detector_output, args.output)
    qa = qa_table(daily, objects, events, catalog, links, ledger, metrics, gap_dates, args.detector_output)
    for df, name in [(contexts, "contexts_v09.csv"), (links, "ordered_links_v09.csv"), (catalog, "setup_catalog_v09.csv"), (realization, "realization_summary_v09.csv"), (comparisons, "stage_comparisons_v09.csv"), (metrics, "walk_forward_metrics_v09.csv"), (predictions, "walk_forward_predictions_v09.csv"), (ms, "walk_forward_metric_summary_v09.csv"), (calib, "walk_forward_calibration_v09.csv"), (ledger, "article_ledger_v09.csv"), (qa, "validation_tests_v09.csv")]: write_df(df, args.output / name)
    (args.output / "article_ledger_v09.json").write_text(ledger.to_json(orient="records", force_ascii=False, indent=2, date_format="iso"), encoding="utf-8")
    qa_errors = int((~qa.passed & qa.severity.eq("error")).sum()); qa_warnings = int((~qa.passed & qa.severity.ne("error")).sum())
    manifest = {"status": "completed_v09_article_shaped_fixed_target" if qa_errors == 0 else "completed_v09_with_qa_errors", "script_version": SCRIPT_VERSION, "script": str(Path(__file__).resolve()), "script_sha256": sha256(Path(__file__).resolve()), "input": data_qa, "input_gap_ledger": gap_qa, "data_window": {"start": str(daily.datetime.min().date()), "end": str(daily.datetime.max().date()), "rows": int(len(daily))}, "coverage": {"requested_years": 30, "actual_years": float((daily.datetime.max() - daily.datetime.min()).days / 365.2425), "start_note": "Yahoo snapshot begins 2000-09-18; no synthetic pre-2000 observations"}, "parallel": parallel, "detection": {"objects_all": int(len(objects)), "events_all": int(len(events)), "balanced_catalog": int(len(catalog)), "state_counts": catalog.setup_state.value_counts().to_dict() if not catalog.empty else {}}, "scenario_definitions": {"fvg_continuation": "FVG + prior displacement + prior structure confirmation + HTF aligned location/trend + fixed same-TF DOL", "ob_reversal": "confirmed daily/weekly OB adaptation + sweep + displacement + structure + companion FVG + HTF aligned location/trend + fixed same-TF DOL", "retest_confirmed": "first touch followed by daily MSS/BOS within five daily bars before invalidation; daily proxy for lower-timeframe trigger"}, "clocks": {"formation": "available_at", "arrival": "formation -> first zone touch within 20 daily bars for 1D / 60 for 1W", "formation_target": "formation close -> fixed DOL or invalidation, 20/60 daily bars", "retest_target": "first touch close -> same fixed DOL or invalidation, 20/60 daily bars", "trigger_target": "daily MSS/BOS confirmation close -> same fixed DOL or invalidation, 20/60 daily bars"}, "targets": {"dol": "next_unconsumed_confirmed_pivot_same_timeframe", "invalidation": "zone boundary plus/minus 0.10 ATR14 at formation", "same_bar": "order_unknown", "preconsumed": "excluded from later target risk set"}, "walk_forward": {"models": ["intercept_only", "causal_context"], "folds": "expanding chronological 50/60/70/80/90 percent within scenario/timeframe/direction/clock", "training_label_end": "strictly before test start", "calibration": "descriptive out-of-fold bins"}, "article_coverage": {"rows": int(len(ledger)), "status_counts": ledger.verified_status.value_counts().to_dict() if not ledger.empty else {}}, "qa": {"checks": int(len(qa)), "errors": qa_errors, "warnings": qa_warnings}, "files": {"ledger": str((args.output / "article_ledger_v09.csv").resolve()), "catalog": str((args.output / "setup_catalog_v09.csv").resolve()), "realization": str((args.output / "realization_summary_v09.csv").resolve()), "comparisons": str((args.output / "stage_comparisons_v09.csv").resolve()), "metrics": str((args.output / "walk_forward_metric_summary_v09.csv").resolve()), "calibration": str((args.output / "walk_forward_calibration_v09.csv").resolve()), "qa": str((args.output / "validation_tests_v09.csv").resolve())}, "limitations": ["94 article corpus is not 94 independent executable strategies", "daily/weekly adaptation is not original lower-timeframe execution", "DOL and invalidation are explicit operational definitions", "observational realization rates are not causal proof", "continuous Yahoo futures proxy", "actual data is about 25.9 years rather than 30"]}
    (args.output / "run_manifest_v09.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=jdefault), encoding="utf-8")
    (args.output / "research_report_v09_ja.md").write_text(report(manifest, ledger, realization, comparisons, metrics, qa), encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "objects": len(objects), "events": len(events), "catalog": len(catalog), "realization_rows": len(realization), "metrics": len(metrics), "qa_errors": qa_errors, "qa_warnings": qa_warnings, "parallel": parallel}, ensure_ascii=False, default=jdefault))


if __name__ == "__main__":
    main()
