"""Point-in-time evaluation layer for the frozen ES=F ICT detector.

This file deliberately does not change the detector.  It adds the next
research layer in the order required by the specification:

1. future-path labels that are independent of the detector's terminal-state
   fields;
2. simple no-feature and price/volatility baselines;
3. expanding time-series walk-forward evaluation with overlap control; and
4. a blind shape-label queue for genuinely independent human/reference labels.

The v0.3 revision keeps first-bar outcomes when they are already observable,
uses the corrected detector output, and creates point-in-time-safe label
windows for independent shape annotation.  The output is an evaluation of a fixed detector conditional on detected zones,
not a claim that the detector has predictive power or that the labels prove an
ICT mechanism.
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

from nq_ict_detection_20260911_v03 import build_weekly as build_corrected_weekly

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SCRIPT_VERSION = "nq-ict-evaluation-v0.3"
DEFAULT_INPUT = Path("nq_20y_ordered_validation_20260909/input/NQ_F_1d_strict.csv")
DEFAULT_DETECTOR_OUTPUT = Path("nq_ict_detection_20260911_v03")
DEFAULT_OUTPUT = Path("nq_ict_evaluation_20260911_v03")

ARRIVAL_HORIZON = {"1D": 20, "1W": 60}
POST_TOUCH_HORIZON = 10
BARRIER_ATR = 1.0
MIN_TRAIN = 40
MIN_TEST = 10
RANDOM_SEED = 20260911


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fnum(x: Any, default: float = np.nan) -> float:
    try:
        y = float(x)
        return y if math.isfinite(y) else default
    except (TypeError, ValueError):
        return default


def load_daily(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = pd.read_csv(path)
    required = {"datetime", "open", "high", "low", "close", "volume"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"missing daily columns: {missing}")
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce").dt.normalize()
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["tr"] = tr
    df["atr14"] = tr.rolling(14, min_periods=2).mean()
    df["return_5"] = df["close"].pct_change(5)
    df["return_20"] = df["close"].pct_change(20)
    vol_mean = df["volume"].rolling(20, min_periods=5).mean()
    vol_std = df["volume"].rolling(20, min_periods=5).std(ddof=0)
    df["volume_z20"] = (df["volume"] - vol_mean) / vol_std.replace(0, np.nan)
    df["sma20"] = df["close"].rolling(20, min_periods=5).mean()
    df["sma50"] = df["close"].rolling(50, min_periods=10).mean()
    df["sma_gap20"] = (df["close"] - df["sma20"]) / df["atr14"].replace(0, np.nan)
    df["sma_gap50"] = (df["close"] - df["sma50"]) / df["atr14"].replace(0, np.nan)
    df["bar_idx"] = np.arange(len(df))
    qa = {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "rows": int(len(df)),
        "start": str(df.datetime.min().date()) if len(df) else None,
        "end": str(df.datetime.max().date()) if len(df) else None,
        "duplicate_dates_after_dedup": int(df.datetime.duplicated().sum()),
        "null_required_cells": int(df[["datetime", "open", "high", "low", "close", "volume"]].isna().sum().sum()),
        "invalid_ohlc_rows": int(((df.high < df.low) | (df.high < df.open) | (df.high < df.close) | (df.low > df.open) | (df.low > df.close)).sum()),
        "weekend_rows": int((df.datetime.dt.dayofweek >= 5).sum()),
        "zero_volume_rows": int((df.volume == 0).sum()),
        "missing_date_gaps_gt_7d": int((df.datetime.diff().dt.days.dropna() > 7).sum()),
    }
    return df, qa


def load_objects(path: Path, data_end: pd.Timestamp) -> tuple[pd.DataFrame, dict[str, Any]]:
    p = path / "objects.csv"
    obj = pd.read_csv(p, encoding="utf-8-sig", low_memory=False)
    for c in ["origin_time", "available_at", "confirmed_at", "updated_at"]:
        if c in obj:
            obj[c] = pd.to_datetime(obj[c], errors="coerce").dt.normalize()
    for c in ["zone_low", "zone_high", "origin_idx", "confirmed_idx"]:
        if c in obj:
            obj[c] = pd.to_numeric(obj[c], errors="coerce")
    before = len(obj)
    required = {"object_id", "timeframe", "object_type", "direction", "sensitivity", "available_at", "zone_low", "zone_high"}
    missing = sorted(required - set(obj.columns))
    if missing:
        raise ValueError(f"objects.csv missing columns: {missing}")
    # balanced is the frozen primary profile.  The other sensitivities are
    # preserved in detector output, but must not be treated as independent
    # duplicate samples in this first evaluation.
    obj = obj[(obj.sensitivity == "balanced") & obj.object_type.isin(["ob_candidate", "fvg"])]
    obj = obj[obj.available_at.notna() & (obj.available_at <= data_end)]
    obj = obj[obj.zone_low.notna() & obj.zone_high.notna() & (obj.zone_high >= obj.zone_low)].copy()
    obj = obj.drop_duplicates("object_id").reset_index(drop=True)
    qa = {
        "path": str(p.resolve()),
        "sha256": sha256(p),
        "rows_before_filter": int(before),
        "rows_after_filter": int(len(obj)),
        "duplicate_object_ids_after_filter": int(obj.object_id.duplicated().sum()),
        "count_by_type_direction": obj.groupby(["timeframe", "object_type", "direction"], dropna=False).size().reset_index(name="n").to_dict("records"),
        "count_by_evidence_state": obj.get("evidence_state", pd.Series(dtype=str)).value_counts(dropna=False).to_dict(),
    }
    return obj, qa


def make_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    w = build_corrected_weekly(daily, pd.Timestamp(daily.datetime.max())).copy()
    w = w[w.complete].copy()
    w["sessions"] = w["observed_sessions"]
    return w.reset_index(drop=True)


def raw_position(daily: pd.DataFrame, ts: pd.Timestamp) -> int | None:
    i = int(daily.datetime.searchsorted(ts, side="right") - 1)
    return i if i >= 0 else None


def source_horizon(timeframe: str) -> int:
    return int(ARRIVAL_HORIZON.get(timeframe, 20))


def label_object(
    row: pd.Series,
    daily: pd.DataFrame,
    arrival_horizon: int | None = None,
    post_touch_horizon: int | None = None,
    barrier_atr: float | None = None,
) -> dict[str, Any]:
    """Create a fixed competing-path label using only bars after availability.

    A touch is an OHLC intersection.  A gap-through is kept separate because
    the path inside the zone is unobserved.  Post-touch barrier order is
    unknown when both barriers are hit in the same daily bar.
    """
    available = pd.Timestamp(row.available_at).normalize()
    arrival_h = int(arrival_horizon if arrival_horizon is not None else source_horizon(str(row.timeframe)))
    post_h = int(post_touch_horizon if post_touch_horizon is not None else POST_TOUCH_HORIZON)
    barrier_mult = float(barrier_atr if barrier_atr is not None else BARRIER_ATR)
    start = raw_position(daily, available)
    out: dict[str, Any] = {
        "object_id": row.object_id,
        "timeframe": row.timeframe,
        "object_type": row.object_type,
        "direction": row.direction,
        "evidence_state": row.get("evidence_state", "unknown"),
        "available_at": available,
        "origin_time": pd.Timestamp(row.origin_time).normalize() if pd.notna(row.get("origin_time")) else pd.NaT,
        "zone_low": fnum(row.zone_low),
        "zone_high": fnum(row.zone_high),
        "arrival_horizon_bars": arrival_h,
        "post_touch_horizon_bars": post_h,
        "barrier_atr": barrier_mult,
        "arrival_type": "not_evaluable",
        "arrival_label": np.nan,
        "arrival_complete": False,
        "arrival_label_end_at": pd.NaT,
        "touch_idx": np.nan,
        "touch_at": pd.NaT,
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
    }
    if start is None:
        out["censor_reason"] = "no_prior_daily_bar"
        return out
    lo, hi = fnum(row.zone_low), fnum(row.zone_high)
    if not (math.isfinite(lo) and math.isfinite(hi) and hi >= lo):
        out["censor_reason"] = "invalid_zone"
        return out
    h = arrival_h
    future_start = start + 1
    future_end = min(len(daily), future_start + h)
    path = daily.iloc[future_start:future_end]
    complete = (future_end - future_start) == h
    out["arrival_complete"] = bool(complete)
    if len(path):
        out["arrival_label_end_at"] = daily.datetime.iloc[future_end - 1]
        hit = (path.low.to_numpy() <= hi) & (path.high.to_numpy() >= lo)
        prev_close = daily.close.iloc[future_start - 1:future_end - 1].to_numpy()
        opens = path.open.to_numpy()
        gap = (((prev_close > hi) & (opens < lo)) | ((prev_close < lo) & (opens > hi))) & ~hit
        hit_pos = np.flatnonzero(hit)
        gap_pos = np.flatnonzero(gap)
        first_hit = int(hit_pos[0]) if len(hit_pos) else None
        first_gap = int(gap_pos[0]) if len(gap_pos) else None
        if first_hit is not None and (first_gap is None or first_hit < first_gap):
            touch_idx = future_start + first_hit
            out["arrival_type"] = "touch"
            out["arrival_label"] = 1.0
            # A first arrival is already observed even if the data ends
            # before the remainder of the nominal horizon.
            out["arrival_complete"] = True
            out["touch_idx"] = touch_idx
            out["touch_at"] = daily.datetime.iloc[touch_idx]
        elif first_gap is not None:
            out["arrival_type"] = "gap_through"
            out["arrival_label"] = 1.0
            out["arrival_complete"] = True
            out["touch_idx"] = np.nan
            out["censor_reason"] = "gap_path_unobserved"
        elif complete:
            out["arrival_type"] = "no_arrival"
            out["arrival_label"] = 0.0
        else:
            out["arrival_type"] = "censored_no_arrival"
            out["censor_reason"] = "data_end_before_arrival_horizon"
    elif complete:
        out["arrival_type"] = "no_arrival"
        out["arrival_label"] = 0.0
    else:
        out["arrival_type"] = "censored_no_arrival"
        out["censor_reason"] = "data_end_before_arrival_horizon"

    if out["arrival_type"] != "touch":
        return out

    touch = int(out["touch_idx"])
    post_start = touch + 1
    post_end = min(len(daily), post_start + post_h)
    post = daily.iloc[post_start:post_end]
    horizon_complete = (post_end - post_start) == post_h
    out["post_complete"] = bool(horizon_complete)
    out["post_label_end_at"] = daily.datetime.iloc[post_end - 1] if len(post) else pd.NaT
    if not len(post):
        out["post_label"] = "censored_post_touch"
        out["censor_reason"] = "no_bar_after_touch"
        return out
    atr_touch = fnum(daily.atr14.iloc[touch])
    if not (math.isfinite(atr_touch) and atr_touch > 0):
        atr_touch = fnum(daily.atr14.iloc[start])
    if not (math.isfinite(atr_touch) and atr_touch > 0):
        out["post_label"] = "unknown_no_atr"
        out["censor_reason"] = "atr_unavailable"
        return out
    if row.direction == "bullish":
        favorable = hi + barrier_mult * atr_touch
        adverse = lo - barrier_mult * atr_touch
        favorable_hit = post.high.to_numpy() >= favorable
        adverse_hit = post.low.to_numpy() <= adverse
        fav_exc = (post.high.to_numpy() - hi) / atr_touch
        adv_exc = (lo - post.low.to_numpy()) / atr_touch
    else:
        favorable = lo - barrier_mult * atr_touch
        adverse = hi + barrier_mult * atr_touch
        favorable_hit = post.low.to_numpy() <= favorable
        adverse_hit = post.high.to_numpy() >= adverse
        fav_exc = (lo - post.low.to_numpy()) / atr_touch
        adv_exc = (post.high.to_numpy() - hi) / atr_touch
    out["post_favorable_barrier"] = favorable
    out["post_adverse_barrier"] = adverse
    out["max_favorable_atr"] = float(np.nanmax(fav_exc)) if len(fav_exc) else np.nan
    out["max_adverse_atr"] = float(np.nanmax(adv_exc)) if len(adv_exc) else np.nan
    fi = np.flatnonzero(favorable_hit)
    ai = np.flatnonzero(adverse_hit)
    fbar = int(fi[0]) if len(fi) else None
    abar = int(ai[0]) if len(ai) else None
    out["bars_to_favorable"] = fbar + 1 if fbar is not None else np.nan
    out["bars_to_adverse"] = abar + 1 if abar is not None else np.nan
    if fbar is not None and abar is not None and fbar == abar:
        out["post_label"] = "both_same_bar_order_unknown"
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
    elif horizon_complete:
        out["post_label"] = "neither_at_horizon"
    else:
        out["post_label"] = "censored_post_touch"
        out["censor_reason"] = "data_end_before_post_horizon"
    return out


def make_features(objects: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in objects.itertuples(index=False):
        ts = pd.Timestamp(r.available_at).normalize()
        pos = raw_position(daily, ts)
        if pos is None:
            continue
        p = daily.iloc[pos]
        atr = fnum(p.atr14)
        close = fnum(p.close)
        lo, hi = fnum(r.zone_low), fnum(r.zone_high)
        width = hi - lo if math.isfinite(hi) and math.isfinite(lo) else np.nan
        distance = 0.0 if lo <= close <= hi else min(abs(lo - close), abs(hi - close))
        mid = (lo + hi) / 2
        age_days = (ts - pd.Timestamp(r.origin_time).normalize()).days if pd.notna(r.origin_time) else np.nan
        evidence = str(getattr(r, "evidence_state", "unknown"))
        rows.append({
            "object_id": r.object_id,
            "timeframe": r.timeframe,
            "direction": r.direction,
            "object_type": r.object_type,
            "available_at": ts,
            "feature_bar_idx": pos,
            "distance_atr": distance / atr if math.isfinite(atr) and atr > 0 else np.nan,
            "width_atr": width / atr if math.isfinite(width) and math.isfinite(atr) and atr > 0 else np.nan,
            "mid_distance_atr": (close - mid) / atr if math.isfinite(mid) and math.isfinite(atr) and atr > 0 else np.nan,
            "age_days": age_days,
            "return_5": fnum(p.return_5),
            "return_20": fnum(p.return_20),
            "atr_pct": atr / close if math.isfinite(atr) and math.isfinite(close) and close != 0 else np.nan,
            "volume_z20": fnum(p.volume_z20),
            "sma_gap20": fnum(p.sma_gap20),
            "sma_gap50": fnum(p.sma_gap50),
            "object_is_ob": 1.0 if r.object_type == "ob_candidate" else 0.0,
            "evidence_confirmed": 1.0 if evidence == "confirmed" else 0.0,
        })
    return pd.DataFrame(rows)


def build_labels(
    objects: pd.DataFrame,
    daily: pd.DataFrame,
    arrival_horizons: dict[str, int] | None = None,
    post_touch_horizon: int | None = None,
    barrier_atr: float | None = None,
) -> pd.DataFrame:
    if objects.empty:
        return pd.DataFrame()
    labels = pd.DataFrame([
        label_object(
            r,
            daily,
            arrival_horizon=(arrival_horizons or {}).get(str(r.timeframe)),
            post_touch_horizon=post_touch_horizon,
            barrier_atr=barrier_atr,
        )
        for _, r in objects.iterrows()
    ])
    features = make_features(objects, daily)
    out = labels.merge(features, on=["object_id", "timeframe", "direction", "object_type", "available_at"], how="left")
    return out.sort_values(["available_at", "object_id"]).reset_index(drop=True)


FEATURES = {
    "trend_volume": ["return_5", "return_20", "atr_pct", "volume_z20", "sma_gap20", "sma_gap50"],
    "zone_geometry": ["distance_atr", "width_atr", "mid_distance_atr", "age_days"],
    "full": ["return_5", "return_20", "atr_pct", "volume_z20", "sma_gap20", "sma_gap50",
             "distance_atr", "width_atr", "mid_distance_atr", "age_days", "object_is_ob", "evidence_confirmed"],
}


def fold_windows(dates: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    u = pd.Series(pd.to_datetime(dates).dropna().sort_values().unique())
    if len(u) < 4:
        return []
    qs = sorted(set(pd.Timestamp(x) for x in u.quantile([0.50, 0.60, 0.70, 0.80, 0.90])))
    bounds = qs + [u.max() + pd.Timedelta(days=1)]
    return [(qs[i], bounds[i + 1]) for i in range(len(qs)) if qs[i] < bounds[i + 1]]


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, str]:
    prior = float(train.y.mean()) if len(train) else np.nan
    if not features or len(train) < MIN_TRAIN or train.y.nunique() < 2:
        return np.full(len(test), prior, dtype=float), "constant_training_prior"
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("logit", LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")),
    ])
    pipe.fit(train[features], train.y.astype(int))
    return pipe.predict_proba(test[features])[:, 1], "logistic_fixed_C1"


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
        eligible["label_end_at"] = eligible.arrival_label_end_at
    else:
        eligible = labels[labels.post_complete & labels.post_label_binary.notna()].copy()
        eligible["y"] = eligible.post_label_binary.astype(float)
        eligible["label_end_at"] = eligible.post_label_end_at
    if eligible.empty:
        return pd.DataFrame(), pd.DataFrame()
    fold_rows = []
    pred_rows = []
    for (tf, direction), group in eligible.groupby(["timeframe", "direction"], dropna=False):
        group = group.sort_values(["available_at", "object_id"]).reset_index(drop=True)
        for fold_id, (test_start, test_end) in enumerate(fold_windows(group.available_at), start=1):
            train = group[(group.available_at < test_start) & (group.label_end_at < test_start)]
            test = group[(group.available_at >= test_start) & (group.available_at < test_end)]
            if len(train) < MIN_TRAIN or len(test) < MIN_TEST:
                continue
            pred_map: dict[str, np.ndarray] = {"unconditional": np.full(len(test), float(train.y.mean()))}
            model_status: dict[str, str] = {"unconditional": "training_prior"}
            for model_name, cols in FEATURES.items():
                pred_map[model_name], model_status[model_name] = fit_predict(train, test, cols)
            for k in range(len(test)):
                rr = test.iloc[k]
                row = {"object_id": rr.object_id, "target": target_name, "timeframe": tf, "direction": direction,
                       "fold_id": fold_id, "test_start": test_start, "test_end": test_end, "available_at": rr.available_at,
                       "y": float(rr.y)}
                for model_name, pred in pred_map.items():
                    row[f"p_{model_name}"] = float(pred[k])
                pred_rows.append(row)
            for model_name, pred in pred_map.items():
                y = test.y.to_numpy(dtype=float)
                fold_rows.append({
                    "target": target_name, "timeframe": tf, "direction": direction, "fold_id": fold_id,
                    "test_start": test_start, "test_end": test_end, "train_n": len(train), "test_n": len(test),
                    "max_train_available_at": train.available_at.max(),
                    "max_train_label_end_at": train.label_end_at.max(),
                    "min_test_available_at": test.available_at.min(),
                    "max_test_available_at": test.available_at.max(),
                    "train_positive_rate": float(train.y.mean()), "test_positive_rate": float(test.y.mean()),
                    "model": model_name, "model_status": model_status[model_name],
                    "brier": float(np.mean((pred - y) ** 2)), "log_loss": safe_logloss(y, pred),
                })
    return pd.DataFrame(fold_rows), pd.DataFrame(pred_rows)


def wilson_interval(positive: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return np.nan, np.nan
    p = positive / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / den
    return max(0.0, center - half), min(1.0, center + half)


def outcome_summary(labels: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if labels.empty:
        return pd.DataFrame()
    for (tf, obj_type, direction), g in labels.groupby(["timeframe", "object_type", "direction"], dropna=False):
        arrival = g[g.arrival_complete & g.arrival_label.notna()]
        post = g[g.post_complete & g.post_label_binary.notna()]
        for target, sample, positive in [
            ("arrival_within_horizon", arrival, "arrival_label"),
            ("favorable_before_adverse_after_touch", post, "post_label_binary"),
        ]:
            n = len(sample); k = int(sample[positive].sum()) if n else 0
            lo, hi = wilson_interval(k, n)
            rows.append({"target": target, "timeframe": tf, "object_type": obj_type, "direction": direction,
                         "n": n, "positive": k, "rate": k / n if n else np.nan,
                         "wilson95_low": lo, "wilson95_high": hi})
    return pd.DataFrame(rows)


def calibration(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if predictions.empty:
        return pd.DataFrame()
    for (target, tf, direction, model), g in predictions.melt(
        id_vars=["target", "timeframe", "direction", "y"],
        value_vars=[c for c in predictions.columns if c.startswith("p_")],
        var_name="model_col", value_name="p",
    ).assign(model=lambda x: x.model_col.str.removeprefix("p_")).groupby(["target", "timeframe", "direction", "model"], dropna=False):
        gg = g.copy()
        bins = pd.cut(gg.p, bins=[-0.001, 0.2, 0.4, 0.6, 0.8, 1.001], include_lowest=True)
        for b, z in gg.assign(probability_bin=bins).groupby("probability_bin", observed=False):
            rows.append({"target": target, "timeframe": tf, "direction": direction, "model": model,
                         "probability_bin": str(b), "n": len(z), "mean_predicted": float(z.p.mean()) if len(z) else np.nan,
                         "observed_rate": float(z.y.mean()) if len(z) else np.nan})
    return pd.DataFrame(rows)


def blind_label_queue(daily: pd.DataFrame, n_each: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    for tf, frame in [("1D", daily), ("1W", make_weekly(daily))]:
        frame = frame.reset_index(drop=True)
        available_series = frame.datetime if tf == "1D" else frame.available_at
        if tf == "1D":
            eligible = np.arange(10, len(frame))
            window = 5
        else:
            eligible = np.arange(5, len(frame))
            window = 3
        if len(eligible) == 0:
            continue
        take = min(n_each, len(eligible))
        selected = np.sort(rng.choice(eligible, size=take, replace=False))
        for num, i in enumerate(selected, start=1):
            as_of = pd.Timestamp(available_series.iloc[i]).normalize()
            rows.append({
                "sample_id": f"blind_{tf}_{num:04d}", "timeframe": tf, "as_of": as_of,
                "window_start": pd.Timestamp(available_series.iloc[i - window]).normalize(), "window_end": as_of,
                "window_bars": window + 1, "future_bars_in_view": 0, "label_view": "point_in_time_prefix",
                "visual_label_fvg": "", "visual_label_ob": "", "visual_label_sweep": "",
                "visual_label_structure_shift": "", "visual_label_direction": "", "visual_label_none_or_uncertain": "",
                "visual_origin_time": "", "visual_confirmation_at": "", "visual_zone_low": "", "visual_zone_high": "",
                "label_status": "unlabeled", "labeler": "",
                "notes": "Use only rows with bar_available_at <= as_of. Do not consult detector object_id or terminal state while labeling.",
            })
    return pd.DataFrame(rows)


def blind_label_ohlc(daily: pd.DataFrame, queue: pd.DataFrame) -> pd.DataFrame:
    """Export the raw prefix windows used by the blind label queue.

    The file contains no detector object IDs or terminal states.  Weekly
    bars expose both the period date and the last session date so a labeler
    cannot accidentally use a future Friday timestamp for a Thursday-closed
    week.
    """
    frames = {"1D": daily.reset_index(drop=True), "1W": make_weekly(daily).reset_index(drop=True)}
    rows = []
    for q in queue.itertuples(index=False):
        frame = frames[q.timeframe]
        avail = frame.datetime if q.timeframe == "1D" else frame.available_at
        matches = np.flatnonzero(pd.to_datetime(avail).dt.normalize().to_numpy() == pd.Timestamp(q.as_of).normalize().to_datetime64())
        if len(matches) != 1:
            continue
        i = int(matches[0])
        start = max(0, i - int(q.window_bars) + 1)
        for j in range(start, i + 1):
            rows.append({
                "sample_id": q.sample_id, "timeframe": q.timeframe, "bar_offset": j - i,
                "bar_datetime": pd.Timestamp(frame.datetime.iloc[j]).normalize(),
                "bar_available_at": pd.Timestamp(avail.iloc[j]).normalize(),
                "open": frame.open.iloc[j], "high": frame.high.iloc[j], "low": frame.low.iloc[j],
                "close": frame.close.iloc[j], "volume": frame.volume.iloc[j],
            })
    return pd.DataFrame(rows)


def blind_label_schema() -> dict[str, Any]:
    return {
        "version": "blind-shape-label-v0.3",
        "time_rule": "all bars in blind_shape_ohlc.csv have bar_available_at <= sample as_of; no future bars are included",
        "required_labels": {
            "visual_label_fvg": "none|bullish|bearish|uncertain",
            "visual_label_ob": "none|bullish|bearish|uncertain",
            "visual_label_sweep": "none|bullish|bearish|uncertain",
            "visual_label_structure_shift": "none|bullish|bearish|uncertain",
            "visual_label_direction": "bullish|bearish|neutral|uncertain",
            "visual_label_none_or_uncertain": "yes|no",
            "visual_origin_time": "ISO date or blank",
            "visual_confirmation_at": "ISO date or blank",
            "visual_zone_low": "number or blank",
            "visual_zone_high": "number or blank",
        },
        "use": "independent shape agreement only; future price outcome remains a separate machine-defined label",
        "do_not_use": ["object_id", "terminal_state", "touch_episode_count", "future_path_labels"],
    }


def write_df(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--detector-output", type=Path, default=DEFAULT_DETECTOR_OUTPUT)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = ap.parse_args()
    daily, data_qa = load_daily(args.input)
    objects, object_qa = load_objects(args.detector_output, daily.datetime.max())
    labels = build_labels(objects, daily)
    arrival_metrics, arrival_predictions = walk_forward(labels, "arrival")
    post_metrics, post_predictions = walk_forward(labels, "post_favorable")
    metrics = pd.concat([arrival_metrics, post_metrics], ignore_index=True)
    predictions = pd.concat([arrival_predictions, post_predictions], ignore_index=True)
    cal = calibration(predictions)
    outcome = outcome_summary(labels)
    queue = blind_label_queue(daily)
    blind_ohlc = blind_label_ohlc(daily, queue)
    args.output.mkdir(parents=True, exist_ok=True)
    write_df(labels, args.output / "future_path_labels.csv")
    write_df(metrics, args.output / "walk_forward_metrics.csv")
    write_df(predictions, args.output / "walk_forward_predictions.csv")
    write_df(cal, args.output / "calibration_bins.csv")
    write_df(outcome, args.output / "future_path_summary.csv")
    write_df(queue, args.output / "blind_shape_label_queue.csv")
    write_df(blind_ohlc, args.output / "blind_shape_ohlc.csv")
    (args.output / "blind_label_schema.json").write_text(
        json.dumps(blind_label_schema(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    label_counts = {
        "arrival_type": labels.arrival_type.value_counts(dropna=False).to_dict() if len(labels) else {},
        "post_label": labels.post_label.value_counts(dropna=False).to_dict() if len(labels) else {},
        "arrival_eligible_n": int(labels.arrival_complete.astype(bool).sum()) if len(labels) else 0,
        "post_eligible_n": int((labels.post_complete & labels.post_label_binary.notna()).sum()) if len(labels) else 0,
    }
    manifest = {
        "status": "completed_future_labels_baselines_walk_forward; independent_shape_labels_pending",
        "script_version": SCRIPT_VERSION,
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "input": data_qa,
        "detector_output": object_qa,
        "detector_run_manifest": str((args.detector_output / "run_manifest.json").resolve()),
        "as_of": str(daily.datetime.max().date()),
        "primary_profile": "balanced_only",
        "future_label_contract": {
            "arrival_horizon_daily_bars": ARRIVAL_HORIZON,
            "post_touch_horizon_daily_bars": POST_TOUCH_HORIZON,
            "barrier_definition": "zone boundary +/- 1.0 ATR14 at touch; same-bar double hit is order_unknown",
            "arrival_definition": "first OHLC intersection; open gap through zone is separate arrival_type",
            "post_touch_entry": "bars strictly after first observed touch; touch bar is excluded because OHLC order is unknown",
            "censoring": "data-end observations remain censored and are excluded from binary model scoring",
        },
        "features": FEATURES,
        "walk_forward_contract": {
            "split": "expanding chronological folds at 50/60/70/80/90 percent date quantiles",
            "training_overlap_exclusion": "training label_end_at must be strictly before test_start",
            "min_train": MIN_TRAIN, "min_test": MIN_TEST,
            "models": "unconditional training prior, fixed logistic trend_volume, zone_geometry, full",
        },
        "label_counts": label_counts,
        "blind_queue_rows": int(len(queue)),
        "blind_ohlc_rows": int(len(blind_ohlc)),
        "blind_label_schema": str((args.output / "blind_label_schema.json").resolve()),
        "independent_label_status": "pending_human_or_external_reference_labels; no precision/recall claimed",
    }
    (args.output / "evaluation_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    metric_summary = metrics.groupby(["target", "timeframe", "direction", "model"], dropna=False).agg(
        folds=("fold_id", "nunique"), n_test=("test_n", "sum"), brier=("brier", "mean"), log_loss=("log_loss", "mean")
    ).reset_index() if len(metrics) else pd.DataFrame()
    if len(metrics):
        metric_summary = metrics.groupby(["target", "timeframe", "direction", "model"], dropna=False).agg(
            folds=("fold_id", "nunique"), n_test=("test_n", "sum"), brier=("brier", "mean"), brier_std=("brier", "std"),
            log_loss=("log_loss", "mean"), log_loss_std=("log_loss", "std")
        ).reset_index()
    write_df(metric_summary, args.output / "walk_forward_metric_summary.csv")
    metric_table = ""
    if len(metric_summary):
        metric_table = "\n".join(
            f"| {r.target} | {r.timeframe} | {r.direction} | {r.model} | {int(r.folds)} | {int(r.n_test)} | {r.brier:.4f} | {r.brier_std:.4f} | {r.log_loss:.4f} | {r.log_loss_std:.4f} |"
            for r in metric_summary.itertuples(index=False)
        )
    report = f"""# ES=F ICT検知器 評価層 v0.3

判定時点: {daily.datetime.max().date()}  
固定検知器: `{args.detector_output.resolve()}`  
評価スクリプト: `{Path(__file__).resolve()}`  
主プロファイル: `balanced` のみ（strict / exploratory を重複標本として混ぜない）

## 結論

将来経路ラベル、単純基準モデル、時系列ウォークフォワードは完了した。一方、検知器から独立した人手／外部基準ラベルはまだ未入力なので、検知器の適合率・再現率は評価していない。この段階で確定できるのは、固定した候補帯に対する将来経路の定義と、未来ラベルを使わない特徴量での未見時系列比較までである。

モデルの数値は「候補帯が検出された後、その帯が定義した期限内に到達されるか」「到達後、固定障壁のどちらへ先に進むか」に対する条件付きの結果であり、無条件のNQ方向予測や売買勝率ではない。

## データと母集団

| 項目 | 値 |
|---|---:|
| 日足行数 | {data_qa['rows']:,} |
| 期間 | {data_qa['start']}〜{data_qa['end']} |
| 週末行 | {data_qa['weekend_rows']} |
| 不正OHLC行 | {data_qa['invalid_ohlc_rows']} |
| balanced OB/FVG候補 | {len(objects):,} |
| 到達ラベルが完全な候補 | {label_counts['arrival_eligible_n']:,} |
| 接触後二障壁ラベルが完全な候補 | {label_counts['post_eligible_n']:,} |
| 独立ラベル待ちキュー | {len(queue):,} |
| 盲検raw OHLC行 | {len(blind_ohlc):,} |

候補は固定検知器の `balanced`、`ob_candidate` と `fvg` だけを対象にした。strict / exploratory は同じ起点を重ね得るため、この評価の標本へ加えていない。将来の終端状態、接触回数、リクレーム、無効化など検知器が後から計算した列は特徴量に使っていない。

同じ時期に形成されたOBとFVG、近接または重複する帯は、経済的に独立した観測とは限らない。今回のBrier／log lossは帯単位で計算し、クラスタ頑健標準誤差や重複帯を統合した推定はまだ行っていない。このため、件数が多いことを独立証拠の数と解釈しない。

## 将来経路ラベル

到達は、利用可能時点より後のOHLCが帯と初めて交差したかで定義した。帯を始値で飛び越え、同じ足の高安が帯に交差しない場合は `gap_through` とし、内部経路を推測しない。日足候補の到達期限は20日足、週足候補は60日足である。

接触後の反応は、接触足を除く10日足で、bullishは帯上限+1 ATR14 と帯下限-1 ATR14、bearishはその鏡像を固定障壁とした。両障壁が同じ日足で触れた場合は `both_same_bar_order_unknown`、終端で期限が足りない場合は打ち切りとした。したがって、同じ足内の高値・安値の順序を勝手に補っていない。

件数の内訳は `future_path_labels.csv` の `arrival_type` / `post_label` で確認できる。ラベルの母数、期限、打ち切り、同一足順序不明を変えずに再計算できる。

## ウォークフォワード

各 `timeframe × direction × target` を時系列順に分け、50/60/70/80/90%の時点から拡大学習・後続評価を行った。訓練例のラベル観測終了がテスト開始以後になる場合は、将来経路が重なるため訓練から除外した。

比較したのは次の4つである。

1. `unconditional`: 学習期間の陽性率だけ。
2. `trend_volume`: 直近リターン、ATR比率、出来高z値、SMAとの距離。
3. `zone_geometry`: 帯までのATR距離、幅、中心からの距離、年齢。
4. `full`: 上記にOB/FVG種別と確認状態を加えた固定ロジスティックモデル。

`walk_forward_metric_summary.csv` のBrier scoreとlog lossはテスト期間だけで集計した。値が小さいほど良いが、改善が小さく折ごとに不安定なら、追加情報が再現性を持つとは結論しない。`calibration_bins.csv` は予測確率帯ごとの件数・平均予測・実測率を持ち、たとえば70%を出した集合が実際に約70%かを確認するために使う。

### ウォークフォワード実測

| target | timeframe | direction | model | folds | n_test | Brier mean | Brier fold SD | log loss mean | log loss fold SD |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
{metric_table}

`future_path_summary.csv` はOB/FVGを分けた実測率とWilson 95%区間である。これは未見テスト予測の信頼区間ではなく、固定ラベル母集団の標本率の不確実性を表す。モデルの折間SD、テスト件数、確率帯別校正と併せて読む。

## 独立ラベルの保留

`blind_shape_label_queue.csv` は検知器の `object_id` や端末状態を隠した固定標本で、`blind_shape_ohlc.csv` は各標本の判定時点までのraw OHLCだけを含む。各行の `bar_available_at <= as_of` であり、未来足を含めない。`visual_label_*` を人手または検知器とは別実装の基準ラベルで埋めるまで、画像・形状の適合率、再現率、確認遅延は **未測定** とする。ラベルを後から検出結果に合わせて付けた場合、独立評価にはならない。

## 現時点の解釈と限界

この評価は、固定したv0.3候補検知器の後にどのような経路が観測されたかを、再現可能なラベルで記述したものだ。高い条件付き反応率が観測されても、候補が形成される前の全バーを母集団にした予測力、取引コスト込みの期待値、機関注文の存在、ICTの因果機構は証明しない。検知器のルールをこの結果に合わせて追加・変更していないが、独立ラベルと新しい完全なホールドアウト期間が必要である。

また、日足と週足の候補を一つの独立標本として足し上げていないが、同じ日付の上位足・下位足イベントが相関する可能性は残る。週足候補の到達期限は比較可能性のため60日足としたもので、12本の完全週足と同じ意味ではない。

次の優先作業は、blind queueへのラベル入力を凍結した後に適合率・再現率・確認遅延を測ること、その後に検知器のルールを変更せず新規期間を完全なホールドアウトとして再実行することである。`evaluation_manifest.json` が定義・ハッシュ・打ち切り規則を固定している。盲検窓の作り方は `blind_label_schema.json` に固定した。
"""
    (args.output / "evaluation_report_ja.md").write_text(report, encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "objects": len(objects), "labels": len(labels),
                      "metrics": len(metrics), "predictions": len(predictions), "blind_queue": len(queue),
                      "blind_ohlc": len(blind_ohlc),
                      "label_counts": label_counts}, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
