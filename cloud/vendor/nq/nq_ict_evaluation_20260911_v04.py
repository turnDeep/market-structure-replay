"""Point-in-time v0.4 evaluation for the frozen NQ=F ICT detector.

The v0.4 layer makes four changes that are deliberately kept separate from the
detector itself:

1. completed weekly context is joined as-of the candidate's available time;
2. the post-touch label is an after-touch-close label and excludes cases where
   a barrier was already crossed by the touch-bar close;
3. scenario steps are linked into local, point-in-time episodes so FVG -> OB,
   OB -> FVG, sweep -> zone and MSS/BOS -> zone order can be measured; and
4. the walk-forward prediction clock is the formation availability for arrival
   labels and the touch-bar close for reaction labels.

This file intentionally does not alter the frozen detector.  The v0.3 output
remains the historical comparison.  All fields used as predictors are
available at the corresponding prediction clock; terminal lifecycle columns
and future path labels are not predictors.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import nq_ict_evaluation_20260911_v03 as v3

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SCRIPT_VERSION = "nq-ict-evaluation-v0.4"
DEFAULT_INPUT = Path("nq_20y_ordered_validation_20260909/input/NQ_F_1d_strict.csv")
DEFAULT_DETECTOR_OUTPUT = Path("nq_ict_detection_20260911_v03")
DEFAULT_OUTPUT = Path("nq_ict_evaluation_20260911_v04")

ARRIVAL_HORIZON = {"1D": 20, "1W": 60}
POST_TOUCH_HORIZON = 10
BARRIER_ATR = 1.0
MIN_TRAIN = 40
MIN_TEST = 10
RANDOM_SEED = 20260911

WEEKLY_FEATURES = [
    "weekly_bullish", "weekly_bearish", "weekly_neutral",
    "weekly_position_premium", "weekly_position_discount",
    "weekly_range_width_atr", "weekly_ob_distance_atr",
    "weekly_fvg_distance_atr", "zone_aligns_weekly", "zone_conflicts_weekly",
]
SCENARIO_FEATURES = [
    "scenario_prior_sweep", "scenario_prior_reclaim", "scenario_prior_mss_bos",
    "scenario_prior_fvg_before_ob", "scenario_prior_ob_before_fvg",
    "scenario_prior_same_direction_zone", "scenario_prior_opposite_direction_zone",
    "scenario_prior_step_count", "scenario_same_timestamp_step",
]
POST_SCENARIO_FEATURES = [f"post_{x}" for x in SCENARIO_FEATURES]


def fnum(x: Any, default: float = np.nan) -> float:
    return v3.fnum(x, default)


def parse_dates(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in columns:
        if c in out.columns:
            out[c] = pd.to_datetime(out[c], errors="coerce").dt.normalize()
    return out


def load_weekly_context(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    p = path / "weekly_contexts.csv"
    c = pd.read_csv(p, low_memory=False)
    c = parse_dates(c, ["datetime", "weekly_available_at"])
    for col in ["range_low", "range_high", "range_mid", "weekly_ob_distance_atr", "weekly_fvg_distance_atr"]:
        if col in c.columns:
            c[col] = pd.to_numeric(c[col], errors="coerce")
    c = c.sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)
    qa = {
        "path": str(p.resolve()),
        "rows": int(len(c)),
        "start": str(c.datetime.min().date()) if len(c) else None,
        "end": str(c.datetime.max().date()) if len(c) else None,
        "duplicate_datetime_after_dedup": int(c.datetime.duplicated().sum()),
        "weekly_available_after_daily_datetime": int((c.weekly_available_at > c.datetime).sum()) if len(c) else 0,
        "unknown_direction_n": int((c.weekly_direction.astype(str) == "unknown").sum()) if len(c) else 0,
    }
    return c, qa


def load_events(path: Path) -> pd.DataFrame:
    p = path / "events.csv"
    e = pd.read_csv(p, low_memory=False)
    e = parse_dates(e, ["event_time", "available_at", "observed_at", "updated_at"])
    for col in ["event_idx", "observed_idx", "metric_bars"]:
        if col in e.columns:
            e[col] = pd.to_numeric(e[col], errors="coerce")
    return e[e.sensitivity.eq("balanced")].copy().reset_index(drop=True)


def context_at(context: pd.DataFrame, ts: pd.Timestamp) -> dict[str, Any]:
    """Return the latest completed weekly context known at ts."""
    t = pd.Timestamp(ts).normalize()
    if context.empty:
        return {
            "weekly_context_id": "", "weekly_available_at": pd.NaT,
            "weekly_direction": "unknown", "weekly_position": "unknown",
            "range_low": np.nan, "range_high": np.nan, "range_mid": np.nan,
            "weekly_ob_distance_atr": np.nan, "weekly_fvg_distance_atr": np.nan,
        }
    eligible = context[(context.datetime <= t) & (context.weekly_available_at.notna()) & (context.weekly_available_at <= t)]
    if eligible.empty:
        return {
            "weekly_context_id": "", "weekly_available_at": pd.NaT,
            "weekly_direction": "unknown", "weekly_position": "unknown",
            "range_low": np.nan, "range_high": np.nan, "range_mid": np.nan,
            "weekly_ob_distance_atr": np.nan, "weekly_fvg_distance_atr": np.nan,
        }
    r = eligible.iloc[-1]
    return {k: r.get(k, np.nan) for k in [
        "weekly_context_id", "weekly_available_at", "weekly_direction", "weekly_position",
        "range_low", "range_high", "range_mid", "weekly_ob_distance_atr", "weekly_fvg_distance_atr",
    ]}


def context_feature_row(context: pd.DataFrame, daily: pd.DataFrame, ts: pd.Timestamp, direction: str, prefix: str = "") -> dict[str, Any]:
    c = context_at(context, ts)
    pos = v3.raw_position(daily, pd.Timestamp(ts).normalize())
    atr = fnum(daily.atr14.iloc[pos]) if pos is not None else np.nan
    width = fnum(c.get("range_high")) - fnum(c.get("range_low"))
    width_atr = width / atr if math.isfinite(width) and math.isfinite(atr) and atr > 0 else np.nan
    wd = str(c.get("weekly_direction", "unknown"))
    wp = str(c.get("weekly_position", "unknown"))
    vals = {
        "weekly_context_id": c.get("weekly_context_id", ""),
        "weekly_available_at": c.get("weekly_available_at", pd.NaT),
        "weekly_direction": wd,
        "weekly_position": wp,
        "weekly_range_width_atr": width_atr,
        "weekly_ob_distance_atr": fnum(c.get("weekly_ob_distance_atr")),
        "weekly_fvg_distance_atr": fnum(c.get("weekly_fvg_distance_atr")),
        "weekly_bullish": float(wd == "bullish"),
        "weekly_bearish": float(wd == "bearish"),
        "weekly_neutral": float(wd in {"neutral", "unknown"}),
        "weekly_position_premium": float(wp == "premium"),
        "weekly_position_discount": float(wp == "discount"),
        "zone_aligns_weekly": float((direction == "bullish" and wd == "bullish") or (direction == "bearish" and wd == "bearish")),
        "zone_conflicts_weekly": float((direction == "bullish" and wd == "bearish") or (direction == "bearish" and wd == "bullish")),
    }
    return {f"{prefix}{k}": v for k, v in vals.items()}


def _event_kind(row: pd.Series) -> str:
    et = str(row.get("event_type", ""))
    rule = str(row.get("rule_id", ""))
    if et in {"liquidity_breach", "sweep_same_bar", "sweep_delayed_reclaim", "liquidity_recovery_pending", "unreclaimed_at_horizon"} or rule == "liquidity_sweep":
        return "sweep" if et in {"sweep_same_bar", "sweep_delayed_reclaim"} else "liquidity_breach"
    if et in {"mss", "bos", "cisd"} or rule in {"mss", "bos", "cisd"}:
        return "mss_bos"
    return "other"


def _episode_bucket(ts: pd.Timestamp, timeframe: str) -> str:
    width = 10 if timeframe == "1D" else 35
    epoch_days = int((pd.Timestamp(ts).normalize() - pd.Timestamp("2000-01-01")).days)
    return f"unanchored_{epoch_days // width:06d}"


def build_scenario_linkage(objects: pd.DataFrame, events: pd.DataFrame, scenarios: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Link zones and confirmed events into causal local episodes.

    An episode is anchored by the latest prior liquidity breach/sweep in the
    same timeframe and direction, provided it is no older than 10 daily or 35
    calendar days for weekly data.  If no such anchor exists, a deterministic
    time bucket is used.  A step at the exact same timestamp is not counted as
    prior to another step; this preserves unknown intraday order.
    """
    if objects.empty:
        return objects.copy(), pd.DataFrame(), {"objects": 0, "steps": 0}
    obj = objects.copy()
    ev = events.copy() if events is not None else pd.DataFrame()
    obj = parse_dates(obj, ["origin_time", "available_at"])
    if len(ev):
        ev = parse_dates(ev, ["event_time", "available_at"])
        ev = ev[ev.timeframe.isin(["1D", "1W"]) & ev.direction.isin(["bullish", "bearish"])].copy()
        ev["event_kind"] = ev.apply(_event_kind, axis=1)
        ev = ev[ev.event_kind.isin(["sweep", "liquidity_breach", "mss_bos"])].copy()
    anchor_events: dict[tuple[str, str], pd.DataFrame] = {}
    if len(ev):
        for key, g in ev.groupby(["timeframe", "direction"], dropna=False):
            anchor_events[key] = g[g.event_kind.isin(["sweep", "liquidity_breach"])].sort_values(["available_at", "event_id"])

    obj_rows = []
    for r in obj.itertuples(index=False):
        t = pd.Timestamp(r.available_at).normalize()
        key = (str(r.timeframe), str(r.direction))
        max_days = 10 if str(r.timeframe) == "1D" else 35
        anchors = anchor_events.get(key, pd.DataFrame())
        prior = anchors[anchors.available_at < t] if len(anchors) else pd.DataFrame()
        anchor_id = ""
        anchor_at = pd.NaT
        link_type = "temporal_unanchored"
        if len(prior):
            a = prior.iloc[-1]
            delta = (t - pd.Timestamp(a.available_at).normalize()).days
            if delta <= max_days:
                anchor_id = str(a.event_id)
                anchor_at = pd.Timestamp(a.available_at).normalize()
                link_type = "local_event_anchor"
        if not anchor_id:
            anchor_id = _episode_bucket(t, str(r.timeframe))
            anchor_at = pd.Timestamp(t)
        scenario_id = f"NQ=F|local_sequence|{r.timeframe}|{r.direction}|{anchor_id}"
        obj_rows.append({"object_id": r.object_id, "scenario_id": scenario_id,
                         "scenario_anchor_at": anchor_at, "scenario_link_type": link_type,
                         "scenario_prediction_at": t})
    links = pd.DataFrame(obj_rows)
    obj = obj.merge(links, on="object_id", how="left")

    event_rows = []
    if len(ev):
        for r in ev.itertuples(index=False):
            t = pd.Timestamp(r.available_at).normalize()
            key = (str(r.timeframe), str(r.direction))
            max_days = 10 if str(r.timeframe) == "1D" else 35
            if str(r.event_kind) in {"sweep", "liquidity_breach"}:
                anchor_id = str(r.event_id)
                anchor_at = t
                link_type = "event_anchor"
            else:
                anchors = anchor_events.get(key, pd.DataFrame())
                prior = anchors[anchors.available_at <= t]
                anchor_id = ""
                anchor_at = pd.NaT
                link_type = "temporal_unanchored"
                if len(prior):
                    a = prior.iloc[-1]
                    delta = (t - pd.Timestamp(a.available_at).normalize()).days
                    if delta <= max_days:
                        anchor_id = str(a.event_id)
                        anchor_at = pd.Timestamp(a.available_at).normalize()
                        link_type = "local_event_anchor"
                if not anchor_id:
                    anchor_id = _episode_bucket(t, str(r.timeframe))
                    anchor_at = t
            event_rows.append({"step_id": r.event_id, "scenario_id": f"NQ=F|local_sequence|{r.timeframe}|{r.direction}|{anchor_id}",
                               "available_at": t, "step_time": pd.Timestamp(r.event_time).normalize() if pd.notna(r.event_time) else t,
                               "step_kind": "event", "step_type": r.event_kind, "direction": r.direction,
                               "timeframe": r.timeframe, "source_id": r.event_id, "link_type": link_type,
                               "anchor_at": anchor_at})
    for r in obj.itertuples(index=False):
        event_rows.append({"step_id": r.object_id, "scenario_id": r.scenario_id,
                           "available_at": pd.Timestamp(r.available_at).normalize(),
                           "step_time": pd.Timestamp(r.origin_time).normalize() if pd.notna(r.origin_time) else pd.Timestamp(r.available_at).normalize(),
                           "step_kind": "zone", "step_type": r.object_type, "direction": r.direction,
                           "timeframe": r.timeframe, "source_id": r.object_id, "link_type": r.scenario_link_type,
                           "anchor_at": r.scenario_anchor_at})
    steps = pd.DataFrame(event_rows).sort_values(["scenario_id", "available_at", "step_kind", "source_id"]).reset_index(drop=True)

    feature_rows = []
    for r in obj.itertuples(index=False):
        t = pd.Timestamp(r.available_at).normalize()
        same = steps[(steps.scenario_id == r.scenario_id) & (steps.available_at < t)]
        prior_events = same[same.step_kind == "event"]
        prior_zones = same[same.step_kind == "zone"]
        prior_fvg = prior_zones[prior_zones.step_type == "fvg"]
        prior_ob = prior_zones[prior_zones.step_type == "ob_candidate"]
        prior_sweep = prior_events[prior_events.step_type.isin(["sweep", "liquidity_breach"])]
        prior_reclaim = prior_events[prior_events.step_type == "sweep"]
        prior_struct = prior_events[prior_events.step_type == "mss_bos"]
        same_ts = steps[(steps.scenario_id == r.scenario_id) & (steps.available_at == t) & (steps.source_id != r.object_id)]
        feature_rows.append({
            "object_id": r.object_id,
            "scenario_id": r.scenario_id,
            "scenario_link_type": r.scenario_link_type,
            "scenario_prior_sweep": float(len(prior_sweep) > 0),
            "scenario_prior_reclaim": float(len(prior_reclaim) > 0),
            "scenario_prior_mss_bos": float(len(prior_struct) > 0),
            "scenario_prior_fvg_before_ob": float(r.object_type == "ob_candidate" and len(prior_fvg) > 0),
            "scenario_prior_ob_before_fvg": float(r.object_type == "fvg" and len(prior_ob) > 0),
            "scenario_prior_same_direction_zone": float(len(prior_zones) > 0),
            "scenario_prior_opposite_direction_zone": 0.0,
            "scenario_prior_step_count": float(len(same)),
            "scenario_same_timestamp_step": float(len(same_ts) > 0),
            "scenario_last_prior_at": same.available_at.max() if len(same) else pd.NaT,
        })
    features = pd.DataFrame(feature_rows)

    # Opposite-direction zones are measured locally at the same timeframe and
    # same as-of time, but are never used to construct the scenario anchor.
    for i, r in obj.iterrows():
        t = pd.Timestamp(r.available_at).normalize()
        prior_opposite = obj[(obj.timeframe == r.timeframe) & (obj.direction != r.direction) &
                             (obj.available_at < t) & (obj.available_at >= t - pd.Timedelta(days=10 if r.timeframe == "1D" else 35))]
        features.loc[features.object_id.eq(r.object_id), "scenario_prior_opposite_direction_zone"] = float(len(prior_opposite) > 0)

    # Formal detector scenarios are joined only when the scenario row itself
    # was already available.  They are diagnostic provenance, not a future
    # feature.  The local sequence id remains the primary grouping key.
    formal_rows = []
    if scenarios is not None and len(scenarios):
        sc = parse_dates(scenarios.copy(), ["origin_time", "confirmed_at", "available_at"])
        sc = sc[(sc.sensitivity == "balanced") & (sc.available_at.notna())]
        for r in obj.itertuples(index=False):
            candidates = sc[(sc.direction == r.direction) & (sc.available_at <= r.available_at)]
            if r.timeframe == "1D":
                candidates = candidates[candidates.scenario_type.isin(["reversal_sweep_mss", "sweep_recovery_status"])]
            else:
                candidates = candidates.iloc[0:0]
            if len(candidates):
                s = candidates.sort_values(["available_at", "scenario_id"]).iloc[-1]
                formal_rows.append({"object_id": r.object_id, "formal_scenario_id": s.scenario_id,
                                    "formal_scenario_stage": s.stage,
                                    "formal_scenario_available_at": s.available_at})
    formal = pd.DataFrame(formal_rows)
    if len(formal):
        obj = obj.merge(formal, on="object_id", how="left")
    else:
        obj["formal_scenario_id"] = ""
        obj["formal_scenario_stage"] = ""
        obj["formal_scenario_available_at"] = pd.NaT
    obj = obj.merge(features, on="object_id", how="left", suffixes=("", "_feature"))
    qa = {
        "objects": int(len(obj)),
        "steps": int(len(steps)),
        "scenario_ids": int(obj.scenario_id.nunique()) if len(obj) else 0,
        "local_event_anchor_objects": int((obj.scenario_link_type == "local_event_anchor").sum()) if len(obj) else 0,
        "temporal_unanchored_objects": int((obj.scenario_link_type == "temporal_unanchored").sum()) if len(obj) else 0,
        "formal_scenario_linked_objects": int(obj.formal_scenario_id.fillna("").astype(str).ne("").sum()) if len(obj) else 0,
        "scenario_prior_timestamp_violations": int((pd.to_datetime(obj.scenario_last_prior_at, errors="coerce") >= pd.to_datetime(obj.available_at, errors="coerce")).sum()) if len(obj) else 0,
    }
    return obj, steps, qa


def make_static_features(objects: pd.DataFrame, daily: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    base = v3.make_features(objects, daily)
    if base.empty:
        return base
    rows = [context_feature_row(context, daily, r.available_at, r.direction) | {"object_id": r.object_id} for r in base.itertuples(index=False)]
    ctx = pd.DataFrame(rows)
    out = base.merge(ctx, on="object_id", how="left")
    return out


def label_object_v04(row: pd.Series, daily: pd.DataFrame, arrival_horizon: int | None = None,
                     post_touch_horizon: int | None = None, barrier_atr: float | None = None) -> dict[str, Any]:
    out = v3.label_object(row, daily, arrival_horizon=arrival_horizon,
                          post_touch_horizon=post_touch_horizon, barrier_atr=barrier_atr)
    out.update({
        "post_prediction_at": pd.NaT,
        "post_feature_bar_idx": np.nan,
        "post_prebreach_at_touch_close": False,
        "post_prebreach_side": "",
        "post_label_definition": "after_touch_close_excluding_touch_close_barrier_breach",
    })
    if out.get("arrival_type") != "touch" or not math.isfinite(fnum(out.get("touch_idx"))):
        return out
    touch = int(out["touch_idx"])
    touch_at = pd.Timestamp(daily.datetime.iloc[touch]).normalize()
    # Replace the v0.3 post fields before recalculating them under the
    # touch-close contract.  This prevents an unavailable-ATR corner case
    # from retaining an old post label.
    out.update({
        "post_label": "not_applicable",
        "post_label_binary": np.nan,
        "post_complete": False,
        "post_label_end_at": pd.NaT,
        "post_favorable_barrier": np.nan,
        "post_adverse_barrier": np.nan,
        "bars_to_favorable": np.nan,
        "bars_to_adverse": np.nan,
        "max_favorable_atr": np.nan,
        "max_adverse_atr": np.nan,
        "censor_reason": "",
    })
    out["post_prediction_at"] = touch_at
    out["post_feature_bar_idx"] = touch
    atr_touch = fnum(daily.atr14.iloc[touch])
    if not (math.isfinite(atr_touch) and atr_touch > 0):
        return out
    lo, hi = fnum(row.zone_low), fnum(row.zone_high)
    mult = float(barrier_atr if barrier_atr is not None else BARRIER_ATR)
    close = fnum(daily.close.iloc[touch])
    if row.direction == "bullish":
        favorable = hi + mult * atr_touch
        adverse = lo - mult * atr_touch
        pre_fav = close >= favorable
        pre_adv = close <= adverse
    else:
        favorable = lo - mult * atr_touch
        adverse = hi + mult * atr_touch
        pre_fav = close <= favorable
        pre_adv = close >= adverse
    out["post_favorable_barrier"] = favorable
    out["post_adverse_barrier"] = adverse
    out["post_prebreach_at_touch_close"] = bool(pre_fav or pre_adv)
    if pre_fav or pre_adv:
        out["post_prebreach_side"] = "both" if pre_fav and pre_adv else ("favorable" if pre_fav else "adverse")
        out["post_label"] = "pre_breached_at_touch_close"
        out["post_label_binary"] = np.nan
        out["post_complete"] = False
        out["post_label_end_at"] = touch_at
        out["censor_reason"] = "touch_close_already_crossed_post_barrier"
        return out

    post_h = int(post_touch_horizon if post_touch_horizon is not None else POST_TOUCH_HORIZON)
    start = touch + 1
    end = min(len(daily), start + post_h)
    post = daily.iloc[start:end]
    complete = (end - start) == post_h
    out["post_complete"] = bool(complete)
    out["post_label_end_at"] = daily.datetime.iloc[end - 1] if len(post) else pd.NaT
    if not len(post):
        out["post_label"] = "censored_post_touch"
        out["censor_reason"] = "no_bar_after_touch"
        return out
    if row.direction == "bullish":
        fav_hit = post.high.to_numpy() >= favorable
        adv_hit = post.low.to_numpy() <= adverse
        fav_exc = (post.high.to_numpy() - hi) / atr_touch
        adv_exc = (lo - post.low.to_numpy()) / atr_touch
    else:
        fav_hit = post.low.to_numpy() <= favorable
        adv_hit = post.high.to_numpy() >= adverse
        fav_exc = (lo - post.low.to_numpy()) / atr_touch
        adv_exc = (post.high.to_numpy() - hi) / atr_touch
    out["max_favorable_atr"] = float(np.nanmax(fav_exc)) if len(fav_exc) else np.nan
    out["max_adverse_atr"] = float(np.nanmax(adv_exc)) if len(adv_exc) else np.nan
    fi = np.flatnonzero(fav_hit)
    ai = np.flatnonzero(adv_hit)
    fbar = int(fi[0]) if len(fi) else None
    abar = int(ai[0]) if len(ai) else None
    out["bars_to_favorable"] = fbar + 1 if fbar is not None else np.nan
    out["bars_to_adverse"] = abar + 1 if abar is not None else np.nan
    if fbar is not None and abar is not None and fbar == abar:
        out["post_label"] = "both_same_bar_order_unknown"
        out["post_label_binary"] = np.nan
        out["post_complete"] = True
        out["post_label_end_at"] = post.datetime.iloc[fbar]
    elif fbar is not None and (abar is None or fbar < abar):
        out["post_label"] = "favorable_first"
        out["post_label_binary"] = 1.0
        out["post_complete"] = True
        out["post_label_end_at"] = post.datetime.iloc[fbar]
    elif abar is not None:
        out["post_label"] = "adverse_first"
        out["post_label_binary"] = 0.0
        out["post_complete"] = True
        out["post_label_end_at"] = post.datetime.iloc[abar]
    elif complete:
        out["post_label"] = "neither_at_horizon"
    else:
        out["post_label"] = "censored_post_touch"
        out["censor_reason"] = "data_end_before_post_horizon"
    return out


def make_reaction_features(labels: pd.DataFrame, daily: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in labels.itertuples(index=False):
        if pd.isna(r.touch_at):
            continue
        ts = pd.Timestamp(r.touch_at).normalize()
        pos = v3.raw_position(daily, ts)
        if pos is None:
            continue
        p = daily.iloc[pos]
        atr = fnum(p.atr14)
        close = fnum(p.close)
        lo, hi = fnum(r.zone_low), fnum(r.zone_high)
        width = hi - lo if math.isfinite(hi) and math.isfinite(lo) else np.nan
        distance = 0.0 if lo <= close <= hi else min(abs(lo - close), abs(hi - close))
        mid = (lo + hi) / 2
        intrusion = (max(0.0, hi - fnum(p.low)) if r.direction == "bullish" else max(0.0, fnum(p.high) - lo))
        bar_range = fnum(p.high) - fnum(p.low)
        close_loc = (close - fnum(p.low)) / bar_range if bar_range > 0 else 0.5
        c = context_feature_row(context, daily, ts, r.direction, prefix="post_")
        row = {
            "object_id": r.object_id,
            "post_feature_at": ts,
            "post_feature_bar_idx": pos,
            "post_distance_atr": distance / atr if math.isfinite(distance) and math.isfinite(atr) and atr > 0 else np.nan,
            "post_width_atr": width / atr if math.isfinite(width) and math.isfinite(atr) and atr > 0 else np.nan,
            "post_mid_distance_atr": (close - mid) / atr if math.isfinite(mid) and math.isfinite(atr) and atr > 0 else np.nan,
            "post_intrusion_atr": intrusion / atr if math.isfinite(intrusion) and math.isfinite(atr) and atr > 0 else np.nan,
            "post_bar_range_atr": bar_range / atr if math.isfinite(bar_range) and math.isfinite(atr) and atr > 0 else np.nan,
            "post_close_location": close_loc,
            "post_return_5": fnum(p.return_5),
            "post_return_20": fnum(p.return_20),
            "post_atr_pct": atr / close if math.isfinite(atr) and math.isfinite(close) and close != 0 else np.nan,
            "post_volume_z20": fnum(p.volume_z20),
            "post_sma_gap20": fnum(p.sma_gap20),
            "post_sma_gap50": fnum(p.sma_gap50),
        }
        row.update(c)
        rows.append(row)
    return pd.DataFrame(rows)


def attach_reaction_scenario_features(labels: pd.DataFrame, scenario_steps: pd.DataFrame) -> pd.DataFrame:
    """Recompute sequence state at the touch-close prediction clock.

    Formation-time scenario flags are valid for arrival prediction, but a
    post-touch model must see steps that became available between formation
    and the touch. Strictly earlier timestamps are prior; same-timestamp
    steps remain unknown order.
    """
    if labels.empty:
        return labels
    steps = scenario_steps.copy() if scenario_steps is not None else pd.DataFrame()
    if len(steps):
        steps = parse_dates(steps, ["available_at", "step_time"])
    rows = []
    for r in labels.itertuples(index=False):
        if pd.isna(r.touch_at) or not hasattr(r, "scenario_id") or pd.isna(r.scenario_id):
            continue
        t = pd.Timestamp(r.touch_at).normalize()
        same = steps[(steps.scenario_id == r.scenario_id) & (steps.available_at < t)] if len(steps) else pd.DataFrame()
        prior_events = same[same.step_kind == "event"] if len(same) else pd.DataFrame()
        prior_zones = same[same.step_kind == "zone"] if len(same) else pd.DataFrame()
        prior_fvg = prior_zones[prior_zones.step_type == "fvg"] if len(prior_zones) else pd.DataFrame()
        prior_ob = prior_zones[prior_zones.step_type == "ob_candidate"] if len(prior_zones) else pd.DataFrame()
        prior_sweep = prior_events[prior_events.step_type.isin(["sweep", "liquidity_breach"])] if len(prior_events) else pd.DataFrame()
        prior_reclaim = prior_events[prior_events.step_type == "sweep"] if len(prior_events) else pd.DataFrame()
        prior_struct = prior_events[prior_events.step_type == "mss_bos"] if len(prior_events) else pd.DataFrame()
        same_ts = steps[(steps.scenario_id == r.scenario_id) & (steps.available_at == t) & (steps.source_id != r.object_id)] if len(steps) else pd.DataFrame()
        rows.append({
            "object_id": r.object_id,
            "post_scenario_prior_sweep": float(len(prior_sweep) > 0),
            "post_scenario_prior_reclaim": float(len(prior_reclaim) > 0),
            "post_scenario_prior_mss_bos": float(len(prior_struct) > 0),
            "post_scenario_prior_fvg_before_ob": float(r.object_type == "ob_candidate" and len(prior_fvg) > 0),
            "post_scenario_prior_ob_before_fvg": float(r.object_type == "fvg" and len(prior_ob) > 0),
            "post_scenario_prior_same_direction_zone": float(len(prior_zones) > 0),
            "post_scenario_prior_opposite_direction_zone": 0.0,
            "post_scenario_prior_step_count": float(len(same)),
            "post_scenario_same_timestamp_step": float(len(same_ts) > 0),
            "post_scenario_last_prior_at": same.available_at.max() if len(same) else pd.NaT,
        })
    out = labels.copy()
    if rows:
        out = out.merge(pd.DataFrame(rows), on="object_id", how="left")
    for c in POST_SCENARIO_FEATURES + ["post_scenario_last_prior_at"]:
        if c not in out.columns:
            out[c] = np.nan
    return out


def build_labels(objects: pd.DataFrame, daily: pd.DataFrame, context: pd.DataFrame,
                 arrival_horizons: dict[str, int] | None = None,
                 post_touch_horizon: int | None = None, barrier_atr: float | None = None) -> pd.DataFrame:
    labels = pd.DataFrame([
        label_object_v04(r, daily, arrival_horizon=(arrival_horizons or {}).get(str(r.timeframe)),
                         post_touch_horizon=post_touch_horizon, barrier_atr=barrier_atr)
        for _, r in objects.iterrows()
    ])
    static = make_static_features(objects, daily, context)
    out = labels.merge(static, on=["object_id", "timeframe", "direction", "object_type", "available_at"], how="left")
    reaction = make_reaction_features(out, daily, context)
    if len(reaction):
        out = out.merge(reaction, on="object_id", how="left", suffixes=("", "_reaction"))
    for c in SCENARIO_FEATURES:
        if c not in out.columns:
            out[c] = np.nan
    return out.sort_values(["available_at", "object_id"]).reset_index(drop=True)


def feature_sets(target: str) -> dict[str, list[str]]:
    if target == "arrival":
        base_trend = ["return_5", "return_20", "atr_pct", "volume_z20", "sma_gap20", "sma_gap50"]
        base_zone = ["distance_atr", "width_atr", "mid_distance_atr", "age_days"]
        base_full = base_trend + base_zone + ["object_is_ob", "evidence_confirmed"]
    else:
        base_trend = ["post_return_5", "post_return_20", "post_atr_pct", "post_volume_z20", "post_sma_gap20", "post_sma_gap50"]
        base_zone = ["post_distance_atr", "post_width_atr", "post_mid_distance_atr", "post_intrusion_atr", "post_bar_range_atr", "post_close_location"]
        base_full = base_trend + base_zone + ["object_is_ob", "evidence_confirmed"]
    return {
        "trend_volume": base_trend,
        "zone_geometry": base_zone,
        "weekly_context": WEEKLY_FEATURES if target == "arrival" else [f"post_{x}" for x in WEEKLY_FEATURES],
        "full_without_weekly": base_full,
        "full_with_weekly": base_full + (WEEKLY_FEATURES if target == "arrival" else [f"post_{x}" for x in WEEKLY_FEATURES]),
        "full_with_weekly_scenario": base_full + (WEEKLY_FEATURES if target == "arrival" else [f"post_{x}" for x in WEEKLY_FEATURES]) + (SCENARIO_FEATURES if target == "arrival" else POST_SCENARIO_FEATURES),
    }


def fold_windows(dates: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    u = pd.Series(pd.to_datetime(dates).dropna().sort_values().unique())
    if len(u) < 4:
        return []
    # The source is daily/weekly date-normalized.  Normalize the quantile
    # boundaries too; otherwise pandas' datetime interpolation can create a
    # mid-day fold boundary and allow a same-day label to enter training.
    qs = sorted(set(pd.Timestamp(x).normalize() for x in u.quantile([0.50, 0.60, 0.70, 0.80, 0.90])))
    bounds = qs + [u.max() + pd.Timedelta(days=1)]
    return [(qs[i], bounds[i + 1]) for i in range(len(qs)) if qs[i] < bounds[i + 1]]


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, str]:
    cols = [c for c in features if c in train.columns and c in test.columns]
    prior = float(train.y.mean()) if len(train) else np.nan
    if not cols or len(train) < MIN_TRAIN or train.y.nunique() < 2:
        return np.full(len(test), prior, dtype=float), "constant_training_prior"
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("logit", LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")),
    ])
    pipe.fit(train[cols], train.y.astype(int))
    return pipe.predict_proba(test[cols])[:, 1], "logistic_fixed_C1"


def safe_logloss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def walk_forward(labels: pd.DataFrame, target_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if labels.empty:
        return pd.DataFrame(), pd.DataFrame()
    if target_name == "arrival":
        eligible = labels[labels.arrival_complete & labels.arrival_label.notna()].copy()
        eligible["y"] = eligible.arrival_label.astype(float)
        eligible["prediction_at"] = eligible.available_at
        eligible["feature_at"] = eligible.available_at
        eligible["label_end_at"] = eligible.arrival_label_end_at
    else:
        eligible = labels[labels.post_complete & labels.post_label_binary.notna() &
                          ~labels.post_prebreach_at_touch_close.astype(bool)].copy()
        eligible["y"] = eligible.post_label_binary.astype(float)
        eligible["prediction_at"] = eligible.post_prediction_at
        eligible["feature_at"] = eligible.post_feature_at
        eligible["label_end_at"] = eligible.post_label_end_at
    eligible = eligible[eligible.prediction_at.notna() & eligible.label_end_at.notna()].copy()
    if eligible.empty:
        return pd.DataFrame(), pd.DataFrame()
    sets = feature_sets(target_name)
    fold_rows, pred_rows = [], []
    for (tf, direction), group in eligible.groupby(["timeframe", "direction"], dropna=False):
        group = group.sort_values(["prediction_at", "object_id"]).reset_index(drop=True)
        for fold_id, (test_start, test_end) in enumerate(fold_windows(group.prediction_at), start=1):
            train = group[(group.prediction_at < test_start) & (group.label_end_at < test_start)]
            test = group[(group.prediction_at >= test_start) & (group.prediction_at < test_end)]
            if len(train) < MIN_TRAIN or len(test) < MIN_TEST:
                continue
            pred_map = {"unconditional": np.full(len(test), float(train.y.mean()))}
            status = {"unconditional": "training_prior"}
            for model, cols in sets.items():
                pred_map[model], status[model] = fit_predict(train, test, cols)
            for k in range(len(test)):
                rr = test.iloc[k]
                row = {"object_id": rr.object_id, "target": target_name, "timeframe": tf,
                       "direction": direction, "fold_id": fold_id, "test_start": test_start,
                       "test_end": test_end, "prediction_at": rr.prediction_at,
                       "feature_at": rr.feature_at, "label_end_at": rr.label_end_at, "y": float(rr.y)}
                for model, pred in pred_map.items():
                    row[f"p_{model}"] = float(pred[k])
                pred_rows.append(row)
            y = test.y.to_numpy(dtype=float)
            for model, pred in pred_map.items():
                fold_rows.append({
                    "target": target_name, "timeframe": tf, "direction": direction,
                    "fold_id": fold_id, "test_start": test_start, "test_end": test_end,
                    "train_n": len(train), "test_n": len(test),
                    "max_train_prediction_at": train.prediction_at.max(),
                    "max_train_label_end_at": train.label_end_at.max(),
                    "min_test_prediction_at": test.prediction_at.min(),
                    "max_test_prediction_at": test.prediction_at.max(),
                    "train_positive_rate": float(train.y.mean()), "test_positive_rate": float(test.y.mean()),
                    "model": model, "model_status": status[model],
                    "brier": float(np.mean((pred - y) ** 2)), "log_loss": safe_logloss(y, pred),
                })
    return pd.DataFrame(fold_rows), pd.DataFrame(pred_rows)


def wilson_interval(positive: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    return v3.wilson_interval(positive, n, z)


def outcome_summary(labels: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (tf, obj_type, direction), g in labels.groupby(["timeframe", "object_type", "direction"], dropna=False):
        arr = g[g.arrival_complete & g.arrival_label.notna()]
        post = g[g.post_complete & g.post_label_binary.notna() & ~g.post_prebreach_at_touch_close.astype(bool)]
        for target, sample, col in [("arrival_within_horizon", arr, "arrival_label"), ("favorable_after_touch_close", post, "post_label_binary")]:
            n = len(sample); k = int(sample[col].sum()) if n else 0
            lo, hi = wilson_interval(k, n)
            rows.append({"target": target, "timeframe": tf, "object_type": obj_type, "direction": direction,
                         "n": n, "positive": k, "rate": k / n if n else np.nan,
                         "wilson95_low": lo, "wilson95_high": hi})
        touch = g[g.arrival_type == "touch"]
        pre = touch[touch.post_prebreach_at_touch_close.astype(bool)]
        rows.append({"target": "pre_breached_at_touch_close", "timeframe": tf, "object_type": obj_type,
                     "direction": direction, "n": len(touch), "positive": len(pre),
                     "rate": len(pre) / len(touch) if len(touch) else np.nan,
                     "wilson95_low": wilson_interval(len(pre), len(touch))[0],
                     "wilson95_high": wilson_interval(len(pre), len(touch))[1]})
    return pd.DataFrame(rows)


def context_summary(labels: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target, col, filt in [
        ("arrival", "arrival_label", labels.arrival_complete & labels.arrival_label.notna()),
        ("post_after_touch_close", "post_label_binary", labels.post_complete & labels.post_label_binary.notna() & ~labels.post_prebreach_at_touch_close.astype(bool)),
    ]:
        s = labels[filt].copy()
        if s.empty:
            continue
        direction_col = "weekly_direction" if target == "arrival" else "post_weekly_direction"
        position_col = "weekly_position" if target == "arrival" else "post_weekly_position"
        clock = "formation_available_at" if target == "arrival" else "touch_close"
        for keys, g in s.groupby(["timeframe", "object_type", "direction", direction_col, position_col], dropna=False):
            n = len(g); k = int(g[col].sum())
            rows.append({"target": target, "timeframe": keys[0], "object_type": keys[1], "direction": keys[2],
                         "context_clock": clock, "weekly_direction": keys[3], "weekly_position": keys[4], "n": n, "positive": k,
                         "rate": k / n if n else np.nan, "wilson95_low": wilson_interval(k, n)[0],
                         "wilson95_high": wilson_interval(k, n)[1]})
    return pd.DataFrame(rows)


def sequence_summary(labels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Descriptive rates and counts for within-episode ordered steps."""
    rows, order_rows = [], []
    for target, col, filt in [
        ("arrival", "arrival_label", labels.arrival_complete & labels.arrival_label.notna()),
        ("post_after_touch_close", "post_label_binary", labels.post_complete & labels.post_label_binary.notna() & ~labels.post_prebreach_at_touch_close.astype(bool)),
    ]:
        s = labels[filt].copy()
        if s.empty:
            continue
        prefix = "scenario_" if target == "arrival" else "post_scenario_"
        def state(r: pd.Series) -> str:
            if getattr(r, prefix + "prior_fvg_before_ob") and r.object_type == "ob_candidate": return "FVG_before_OB"
            if getattr(r, prefix + "prior_ob_before_fvg") and r.object_type == "fvg": return "OB_before_FVG"
            if getattr(r, prefix + "prior_sweep") and getattr(r, prefix + "prior_mss_bos"): return "Sweep_then_MSS_BOS"
            if getattr(r, prefix + "prior_sweep"): return "Sweep_before_zone"
            if getattr(r, prefix + "prior_mss_bos"): return "MSS_BOS_before_zone"
            if getattr(r, prefix + "prior_step_count") > 0: return "Other_prior_steps"
            return "No_prior_steps"
        s["sequence_state"] = s.apply(state, axis=1)
        for keys, g in s.groupby(["timeframe", "object_type", "direction", "sequence_state"], dropna=False):
            n = len(g); k = int(g[col].sum())
            rows.append({"target": target, "timeframe": keys[0], "object_type": keys[1], "direction": keys[2],
                         "sequence_state": keys[3], "n": n, "positive": k, "rate": k / n if n else np.nan,
                         "wilson95_low": wilson_interval(k, n)[0], "wilson95_high": wilson_interval(k, n)[1]})
        for object_type, relation, mask in [
            ("ob_candidate", "FVG_to_OB", s.object_type.eq("ob_candidate") & s[prefix + "prior_fvg_before_ob"].astype(bool)),
            ("fvg", "OB_to_FVG", s.object_type.eq("fvg") & s[prefix + "prior_ob_before_fvg"].astype(bool)),
            ("all", "Sweep_to_zone", s[prefix + "prior_sweep"].astype(bool)),
            ("all", "MSS_BOS_to_zone", s[prefix + "prior_mss_bos"].astype(bool)),
        ]:
            g = s[mask]
            n = len(g); k = int(g[col].sum()) if n else 0
            order_rows.append({"target": target, "object_type": object_type, "direction": "all",
                               "relation": relation, "n": n, "positive": k, "rate": k / n if n else np.nan})
    return pd.DataFrame(rows), pd.DataFrame(order_rows)


def write_df(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--detector-output", type=Path, default=DEFAULT_DETECTOR_OUTPUT)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = ap.parse_args()

    daily, data_qa = v3.load_daily(args.input)
    objects, object_qa = v3.load_objects(args.detector_output, daily.datetime.max())
    context, context_qa = load_weekly_context(args.detector_output)
    events = load_events(args.detector_output)
    scenarios = pd.read_csv(args.detector_output / "scenarios.csv", low_memory=False)
    linked_objects, scenario_steps, scenario_qa = build_scenario_linkage(objects, events, scenarios)
    labels = build_labels(linked_objects, daily, context)
    # The scenario fields are already carried by linked_objects -> static
    # features only for object fields.  Merge them explicitly into labels.
    scenario_cols = ["object_id"] + SCENARIO_FEATURES + ["scenario_id", "scenario_link_type", "scenario_last_prior_at"]
    scenario_features = linked_objects[[c for c in scenario_cols if c in linked_objects.columns]].copy()
    labels = labels.drop(columns=[c for c in scenario_cols if c != "object_id" and c in labels.columns], errors="ignore")
    labels = labels.merge(scenario_features, on="object_id", how="left")
    labels = attach_reaction_scenario_features(labels, scenario_steps)
    for c in SCENARIO_FEATURES:
        if c not in labels.columns:
            labels[c] = np.nan

    arrival_metrics, arrival_predictions = walk_forward(labels, "arrival")
    post_metrics, post_predictions = walk_forward(labels, "post_favorable")
    metrics = pd.concat([arrival_metrics, post_metrics], ignore_index=True)
    predictions = pd.concat([arrival_predictions, post_predictions], ignore_index=True)
    outcome = outcome_summary(labels)
    contexts = context_summary(labels)
    sequence, sequence_order = sequence_summary(labels)
    queue = v3.blind_label_queue(daily)
    blind_ohlc = v3.blind_label_ohlc(daily, queue)

    args.output.mkdir(parents=True, exist_ok=True)
    write_df(labels, args.output / "future_path_labels_v04.csv")
    write_df(metrics, args.output / "walk_forward_metrics_v04.csv")
    write_df(predictions, args.output / "walk_forward_predictions_v04.csv")
    write_df(outcome, args.output / "future_path_summary_v04.csv")
    write_df(contexts, args.output / "weekly_context_summary_v04.csv")
    write_df(sequence, args.output / "scenario_sequence_summary_v04.csv")
    write_df(sequence_order, args.output / "scenario_order_relations_v04.csv")
    write_df(linked_objects, args.output / "objects_with_scenario_linkage_v04.csv")
    write_df(scenario_steps, args.output / "scenario_steps_v04.csv")
    write_df(queue, args.output / "blind_shape_label_queue.csv")
    write_df(blind_ohlc, args.output / "blind_shape_ohlc.csv")
    (args.output / "blind_label_schema.json").write_text(json.dumps(v3.blind_label_schema(), ensure_ascii=False, indent=2), encoding="utf-8")
    metric_summary = metrics.groupby(["target", "timeframe", "direction", "model"], dropna=False).agg(
        folds=("fold_id", "nunique"), n_test=("test_n", "sum"), brier=("brier", "mean"), brier_std=("brier", "std"),
        log_loss=("log_loss", "mean"), log_loss_std=("log_loss", "std"),
    ).reset_index() if len(metrics) else pd.DataFrame()
    write_df(metric_summary, args.output / "walk_forward_metric_summary_v04.csv")

    label_counts = {
        "arrival_type": labels.arrival_type.value_counts(dropna=False).to_dict() if len(labels) else {},
        "post_label": labels.post_label.value_counts(dropna=False).to_dict() if len(labels) else {},
        "arrival_eligible_n": int((labels.arrival_complete & labels.arrival_label.notna()).sum()),
        "post_binary_eligible_n": int((labels.post_complete & labels.post_label_binary.notna() & ~labels.post_prebreach_at_touch_close.astype(bool)).sum()),
        "post_touch_n": int((labels.arrival_type == "touch").sum()),
        "post_prebreached_at_touch_close_n": int(labels.post_prebreach_at_touch_close.astype(bool).sum()),
    }
    manifest = {
        "status": "completed_v04_weekly_context_control_ready_after_touch_close_scenario_linkage",
        "script_version": SCRIPT_VERSION,
        "script": str(Path(__file__).resolve()),
        "input": data_qa,
        "detector_output": object_qa,
        "weekly_context_qa": context_qa,
        "scenario_linkage_qa": scenario_qa,
        "as_of": str(daily.datetime.max().date()),
        "primary_profile": "balanced_only",
        "weekly_join_contract": "latest completed weekly_context with weekly_available_at <= prediction clock; no incomplete week is joined",
        "reaction_label_contract": {
            "prediction_clock": "first observed touch bar close",
            "touch_bar_excluded_from_future_path": True,
            "prebreach_policy": "exclude from post binary when touch-bar close already crossed either fixed barrier",
            "post_horizon_daily_bars": POST_TOUCH_HORIZON,
            "barrier_definition": "zone boundary +/- 1.0 ATR14 at touch",
            "same_bar_double_hit": "order_unknown; binary excluded",
        },
        "scenario_contract": {
            "anchor": "latest prior liquidity breach or confirmed sweep in same timeframe and direction within 10 daily or 35 weekly calendar days",
            "same_timestamp_order": "not counted as prior; stored as same_timestamp_step",
            "event_steps": "balanced events only: breach, sweep/reclaim, MSS/BOS/CISD",
            "zone_steps": "balanced OB/FVG candidates",
            "post_touch_recalculation": "same scenario is recomputed with steps strictly before touch-close; same timestamp is not prior",
        },
        "feature_sets": {"arrival": feature_sets("arrival"), "post_favorable": feature_sets("post_favorable")},
        "walk_forward_contract": {
            "split": "expanding chronological folds at 50/60/70/80/90 percent prediction-clock quantiles",
            "training_overlap_exclusion": "training label_end_at must be strictly before test_start",
            "prediction_clock": "available_at for arrival; post_feature_at/touch close for post",
            "min_train": MIN_TRAIN, "min_test": MIN_TEST,
        },
        "label_counts": label_counts,
        "files": {
            "labels": str((args.output / "future_path_labels_v04.csv").resolve()),
            "weekly_context_summary": str((args.output / "weekly_context_summary_v04.csv").resolve()),
            "scenario_steps": str((args.output / "scenario_steps_v04.csv").resolve()),
        },
        "independent_label_status": "pending_human_or_external_reference_labels; no precision_recall_claim",
    }
    (args.output / "evaluation_manifest_v04.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    lines = [
        "# NQ=F ICT検知器 評価層 v0.4",
        "",
        "v0.3の固定検知器は変更せず、週足文脈・接触後ラベル・局所シナリオ順序を追加した評価層である。v0.3の出力は比較用に保存している。",
        "",
        "## 今回の修正",
        "",
        "- 週足: `weekly_available_at <= prediction clock` の最新完成週だけを結合した。",
        "- 対照帯: 反対側の鏡像ではなく、実帯と同じ側で幅を維持し、現値からさらに0.5 ATR以上離した隣接帯を別スクリプトで比較する。",
        "- 反応ラベル: 接触足の終値が既に障壁を越えていた例を、接触後の二障壁ラベルから除外し、`pre_breached_at_touch_close` として分離した。",
        "- 時点: 到達は候補の `available_at`、接触後反応は接触足の終値後を予測時計にした。",
        "- 順序: 同じ局所シナリオ内で、Sweep→MSS/BOS→zone、FVG→OB、OB→FVGを、同時刻と先行時刻を分けて関連付けた。",
        "",
        "## 標本とラベル",
        "",
        f"日足データは {data_qa['rows']:,} 行、{data_qa['start']}〜{data_qa['end']}、balanced OB/FVGは {len(objects):,} 件。",
        f"到達ラベルの完全観測は {label_counts['arrival_eligible_n']:,} 件、接触足終値後の二障壁ラベルは {label_counts['post_binary_eligible_n']:,} 件、接触足終値で既に障壁を越えたため除外したものは {label_counts['post_prebreached_at_touch_close_n']:,} 件。",
        "接触後の確率は『帯へ到達する確率』ではなく、『到達が観測され、接触足の終値ではまだ固定障壁を越えていない条件で、次の10日足において有利障壁が不利障壁より先に到達する確率』である。",
        "",
        "## ウォークフォワード",
        "",
        "`walk_forward_metric_summary_v04.csv` で、無条件基準、トレンド・出来高、帯形状、週足文脈、週足文脈を加えたモデル、さらに同一シナリオ順序を加えたモデルを比較する。Brier scoreとlog lossは未見テストだけで計算し、小さいほど良い。",
        "同じ帯から作られた候補や同じシナリオ内の候補は独立標本ではないため、単純な行数を証拠数とは解釈しない。",
        "",
        "## 独立ラベル",
        "",
        "blind queueはv0.3と同じく出力した。検知器とは独立した人手／外部基準ラベルが未入力のため、形状の適合率・再現率は未測定である。",
    ]
    (args.output / "evaluation_report_v04_ja.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "objects": len(objects), "labels": len(labels),
                      "metrics": len(metrics), "predictions": len(predictions), "scenario_steps": len(scenario_steps),
                      "label_counts": label_counts, "scenario_qa": scenario_qa}, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
