"""Exploratory point-in-time study of bullish OB wick-sweep depth and volatility.

This does not alter or replay the v0.11 campaign. It uses the already completed
v0.11 baseline candidate log and the exact daily price input. The primary unit
is the first candidate per OB ID / contact index; the forward label begins only
after that candidate's close-time decision and next-observed-open reference.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path


PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parent
DEFAULT_OUT = PACKAGE / "validation_rev11_atr_sweep_research_20260925_final2"
PRICES = ROOT / "nq_25y_actual_2000_2026_20260911" / "NQ_F_1d_combined.csv"
BASELINE = PACKAGE / "validation_rev11_incremental_factors_20260925" / "baseline"
BASELINE_REFERENCE = PACKAGE / "validation_rev11_without_continuation_20260923"
FACTOR_TABLE = PACKAGE / "validation_rev11_incremental_factors_20260925" / "factor_comparison.csv"
AS_OF = "2026-08-27"
HORIZON = 20
ATR_MULTIPLE = 1.0
ATR_PERIOD = 14
VOL_LOOKBACK = 252
VOL_WINDOW = 20
BOOTSTRAPS = 5000
SEED = 260925

DEPTH_BINS = [
    ("no_sweep", 0.0, 0.0),
    ("shallow_0_0.25ATR", 0.0, 0.25),
    ("moderate_0.25_0.5ATR", 0.25, 0.5),
    ("deep_over_0.5ATR", 0.5, math.inf),
]
PERIODS = [
    ("2000_2010", "2000-01-01", "2010-12-31"),
    ("2011_2015", "2011-01-01", "2015-12-31"),
    ("2016_2018", "2016-01-01", "2018-12-31"),
    ("2019_2022", "2019-01-01", "2022-12-31"),
    ("2023_2026", "2023-01-01", "2026-12-31"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_csv(path: Path, rows: list[dict]) -> None:
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def wilson(successes: int, n: int, z: float = 1.959963984540054):
    if n <= 0:
        return None, None
    p = successes / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, center - half), min(1.0, center + half)


def true_ranges(prices: list[dict]) -> list[float]:
    result = []
    previous_close = None
    for row in prices:
        hi, lo = float(row["high"]), float(row["low"])
        close_prev = float(row["close"]) if previous_close is None else previous_close
        result.append(max(hi - lo, abs(hi - close_prev), abs(lo - close_prev)))
        previous_close = float(row["close"])
    return result


def rolling_mean_prior(values: list[float], window: int) -> list[float | None]:
    """Match the strategy input: mean of the prior `window` true ranges."""
    out = [None] * len(values)
    for i in range(window, len(values)):
        out[i] = sum(values[i - window:i]) / window
    return out


def pre_touch_volatility(prices: list[dict], tr: list[float], touch_index: int):
    """ATR% at the close before first contact; percentile uses only older values."""
    asof_idx = touch_index - 1
    if asof_idx < max(ATR_PERIOD - 1, VOL_WINDOW - 1):
        return None, None, None
    # ATR known at the preceding close includes that close's true range.
    atr = sum(tr[asof_idx - ATR_PERIOD + 1:asof_idx + 1]) / ATR_PERIOD
    close = float(prices[asof_idx]["close"])
    if close <= 0:
        return None, None, None
    current = atr / close
    history_start = max(ATR_PERIOD - 1, asof_idx - VOL_LOOKBACK)
    historical = []
    for j in range(history_start, asof_idx):
        if j < VOL_WINDOW - 1:
            continue
        a = sum(tr[j - ATR_PERIOD + 1:j + 1]) / ATR_PERIOD
        c = float(prices[j]["close"])
        if c > 0:
            historical.append(a / c)
    if len(historical) < 100:
        return current, None, len(historical)
    # Mid-rank empirical percentile; the current observation is not in history.
    pct = (sum(x < current for x in historical) + 0.5 * sum(x == current for x in historical)) / len(historical)
    regime = "low" if pct <= 1 / 3 else "mid" if pct <= 2 / 3 else "high"
    return current, pct, len(historical)


def depth_bin(depth: float) -> str:
    if depth <= 0:
        return "no_sweep"
    if depth <= 0.25:
        return "shallow_0_0.25ATR"
    if depth <= 0.5:
        return "moderate_0.25_0.5ATR"
    return "deep_over_0.5ATR"


def forward_label(prices: list[dict], atr_decision: float, decision_idx: int):
    """First hit of +/-1 prior ATR from next-open reference within 20 sessions."""
    n = len(prices)
    entry_idx = decision_idx + 1
    if entry_idx >= n or decision_idx + HORIZON >= n or not (atr_decision and atr_decision > 0):
        return "right_censored", None, None
    entry = float(prices[entry_idx]["open"])
    upper, lower = entry + ATR_MULTIPLE * atr_decision, entry - ATR_MULTIPLE * atr_decision
    for j in range(entry_idx, entry_idx + HORIZON):
        row = prices[j]
        op, hi, lo = float(row["open"]), float(row["high"]), float(row["low"])
        # On later sessions the open is chronologically first and resolves a gap.
        if j > entry_idx and op >= upper:
            return "up_first", j, row["datetime"][:10]
        if j > entry_idx and op <= lower:
            return "down_first", j, row["datetime"][:10]
        up_hit, down_hit = hi >= upper, lo <= lower
        if up_hit and down_hit:
            return "same_bar_ambiguous", j, row["datetime"][:10]
        if up_hit:
            return "up_first", j, row["datetime"][:10]
        if down_hit:
            return "down_first", j, row["datetime"][:10]
    return "no_barrier_20d", entry_idx + HORIZON - 1, prices[entry_idx + HORIZON - 1]["datetime"][:10]


def wilson_table(rows: list[dict], group_field: str) -> list[dict]:
    result = []
    groups = ["no_sweep", "shallow_0_0.25ATR", "moderate_0.25_0.5ATR", "deep_over_0.5ATR"]
    if group_field == "volatility_regime":
        groups = ["low", "mid", "high", "unknown"]
    for group in groups:
        subset = [r for r in rows if r.get(group_field) == group]
        mature = [r for r in subset if r["forward_label"] not in {"right_censored", "same_bar_ambiguous"}]
        succ = sum(r["forward_label"] == "up_first" for r in mature)
        lo, hi = wilson(succ, len(mature))
        counts = Counter(r["forward_label"] for r in subset)
        depth = [float(r["sweep_depth_atr"]) for r in subset]
        result.append(dict(
            group=group, candidate_events=len(subset), evaluable=len(mature),
            up_first=counts["up_first"], down_first=counts["down_first"],
            no_barrier_20d=counts["no_barrier_20d"],
            same_bar_ambiguous=counts["same_bar_ambiguous"], right_censored=counts["right_censored"],
            up_before_down_rate_pct=round(succ / len(mature) * 100, 2) if mature else None,
            event_level_wilson95_low_pct=round(lo * 100, 2) if lo is not None else None,
            event_level_wilson95_high_pct=round(hi * 100, 2) if hi is not None else None,
            median_sweep_depth_atr=round(sorted(depth)[len(depth) // 2], 4) if depth else None,
        ))
    return result


def cluster_bootstrap_deltas(rows: list[dict], reps: int = BOOTSTRAPS):
    """Calendar-year cluster bootstrap, descriptive uncertainty only."""
    eligible = [r for r in rows if r["forward_label"] not in {"right_censored", "same_bar_ambiguous"}]
    years = sorted({r["decision_date"][:4] for r in eligible})
    by_year = {y: [r for r in eligible if r["decision_date"][:4] == y] for y in years}
    groups = ["shallow_0_0.25ATR", "moderate_0.25_0.5ATR", "deep_over_0.5ATR"]
    rng = random.Random(SEED)
    values = defaultdict(list)
    for _ in range(reps):
        sampled = [rng.choice(years) for _ in years]
        resampled = [r for y in sampled for r in by_year[y]]
        base = [r for r in resampled if r["depth_bin"] == "no_sweep"]
        if not base:
            continue
        p0 = sum(r["forward_label"] == "up_first" for r in base) / len(base)
        for group in groups:
            part = [r for r in resampled if r["depth_bin"] == group]
            if part:
                p1 = sum(r["forward_label"] == "up_first" for r in part) / len(part)
                values[group].append((p1 - p0) * 100)
    out = []
    for group in groups:
        vals = sorted(values[group])
        if not vals:
            out.append(dict(group=group, difference_vs_no_sweep_pp=None, year_cluster_bootstrap95_low_pp=None,
                            year_cluster_bootstrap95_high_pp=None, bootstrap_replicates=0))
            continue
        p0rows = [r for r in eligible if r["depth_bin"] == "no_sweep"]
        p1rows = [r for r in eligible if r["depth_bin"] == group]
        delta = (sum(r["forward_label"] == "up_first" for r in p1rows) / len(p1rows) -
                 sum(r["forward_label"] == "up_first" for r in p0rows) / len(p0rows)) * 100
        out.append(dict(group=group, difference_vs_no_sweep_pp=round(delta, 2),
                        year_cluster_bootstrap95_low_pp=round(vals[int(0.025 * (len(vals) - 1))], 2),
                        year_cluster_bootstrap95_high_pp=round(vals[int(0.975 * (len(vals) - 1))], 2),
                        bootstrap_replicates=len(vals)))
    return out


def adjusted_depth_model(events: list[dict]):
    """Cluster-robust logistic association adjusted for timeframe, period, and pre-vol."""
    import numpy as np
    import pandas as pd
    import statsmodels.formula.api as smf
    from scipy.stats import chi2, norm

    period_edges = [(name, start, end) for name, start, end in PERIODS]
    data = []
    for e in events:
        if e["forward_label"] in {"right_censored", "same_bar_ambiguous"} or e["pre_touch_vol_percentile"] is None:
            continue
        period = next((name for name, start, end in period_edges
                       if start <= e["decision_date"] <= end), None)
        if period is None:
            continue
        data.append(dict(
            up_first=int(e["forward_label"] == "up_first"),
            depth_bin=e["depth_bin"], timeframe=e["timeframe"], period=period,
            pre_vol_centered=float(e["pre_touch_vol_percentile"]) / 100.0 - 0.5,
            year=e["decision_date"][:4],
        ))
    frame = pd.DataFrame(data)
    formula = ("up_first ~ C(depth_bin, Treatment(reference='no_sweep')) * pre_vol_centered "
               "+ C(timeframe, Treatment(reference='1D')) "
               "+ C(period, Treatment(reference='2000_2010'))")
    model = smf.logit(formula, data=frame).fit(
        disp=False, maxiter=300, cov_type="cluster",
        cov_kwds={"groups": frame["year"], "use_correction": True},
    )
    params = model.params
    cov = model.cov_params()
    names = list(params.index)

    def wald(indices):
        if not indices:
            return None, None
        beta = params.iloc[indices].to_numpy(float)
        vcov = cov.iloc[indices, indices].to_numpy(float)
        stat = float(beta @ np.linalg.pinv(vcov) @ beta)
        return stat, float(chi2.sf(stat, len(indices)))

    depth_idx = [i for i, name in enumerate(names)
                 if name.startswith("C(depth_bin, Treatment(reference='no_sweep'))[T.")
                 and ":" not in name]
    interaction_idx = [i for i, name in enumerate(names)
                       if name.startswith("C(depth_bin, Treatment(reference='no_sweep'))[T.")
                       and ":pre_vol_centered" in name]
    depth_stat, depth_p = wald(depth_idx)
    int_stat, int_p = wald(interaction_idx)

    rows = []
    z = float(norm.ppf(0.975))
    for i, name in enumerate(names):
        if name.startswith("C(depth_bin, Treatment(reference='no_sweep'))[T."):
            b, se = float(params.iloc[i]), float(model.bse.iloc[i])
            rows.append(dict(term=name, coefficient_log_odds=round(b, 6),
                             odds_ratio=round(math.exp(b), 4),
                             cluster_robust95_low_or=round(math.exp(b - z * se), 4),
                             cluster_robust95_high_or=round(math.exp(b + z * se), 4),
                             cluster_robust_p_value=round(float(model.pvalues.iloc[i]), 6),
                             interpretation=("depth-group contrast at median pre-contact volatility" if ":" not in name
                                             else "change in depth contrast across 0-to-100 volatility percentile")))
    main_rows = [r for r in rows if r["interpretation"] == "depth-group contrast at median pre-contact volatility"]
    ordered = sorted(main_rows, key=lambda r: float(r["cluster_robust_p_value"]))
    running = 0.0
    for rank, row in enumerate(ordered):
        adjusted = min(1.0, (len(ordered) - rank) * float(row["cluster_robust_p_value"]))
        running = max(running, adjusted)
        row["holm_adjusted_p_value"] = round(running, 6)
    summary = dict(
        n=len(frame), calendar_year_clusters=int(frame["year"].nunique()),
        formula=formula, depth_groups_joint_wald_chi2=round(depth_stat, 4) if depth_stat is not None else None,
        depth_groups_joint_wald_df=len(depth_idx), depth_groups_joint_cluster_robust_p=depth_p,
        depth_x_volatility_joint_wald_chi2=round(int_stat, 4) if int_stat is not None else None,
        depth_x_volatility_joint_wald_df=len(interaction_idx),
        depth_x_volatility_joint_cluster_robust_p=int_p,
        caveat="Exploratory in-sample association; period/timeframe adjustment and year-clustered covariance do not remove all confounding or create an OOS test.",
    )
    return summary, rows


def audit_existing_replays():
    import numpy as np
    base_dir = PACKAGE / "validation_rev11_incremental_factors_20260925" / "baseline"
    sweep_dir = PACKAGE / "validation_rev11_incremental_factors_20260925" / "ob_distal_wick_sweep"
    reference_dir = BASELINE_REFERENCE

    def replay_stats(folder):
        mtm = list(read_jsonl(folder / "daily_mtm.jsonl"))
        trades = list(read_jsonl(folder / "trades.jsonl"))
        vals = [float(r["mtm_pnl_usd"]) for r in mtm]
        peak, dd = 0.0, 0.0
        for value in vals:
            peak = max(peak, value)
            dd = min(dd, value - peak)
        return dict(rows=len(mtm), first_date=mtm[0]["datetime"][:10], last_date=mtm[-1]["datetime"][:10],
                    final_mtm_usd=vals[-1], max_close_drawdown_usd=dd, completed_trades=len(trades),
                    realized_net_usd=round(sum(float(t["net_pnl_usd"]) for t in trades), 2))

    base_stats, sweep_stats = replay_stats(base_dir), replay_stats(sweep_dir)
    ref_rows = list(read_jsonl(reference_dir / "daily_mtm.jsonl"))
    base_rows = list(read_jsonl(base_dir / "daily_mtm.jsonl"))
    base_meta = json.loads((base_dir / "metadata.json").read_text(encoding="utf-8-sig"))
    sweep_meta = json.loads((sweep_dir / "metadata.json").read_text(encoding="utf-8-sig"))
    ref_trade_hash = sha256(reference_dir / "trades.jsonl")
    baseline_trade_hash = sha256(base_dir / "trades.jsonl")
    base_candidate_hash = sha256(base_dir / "independent_early_candidates.jsonl")
    sweep_candidate_hash = sha256(sweep_dir / "independent_early_candidates.jsonl")
    return dict(
        baseline_exactly_matches_frozen_reference_mtm=(
            len(ref_rows) == len(base_rows) and
            [r["datetime"] for r in ref_rows] == [r["datetime"] for r in base_rows] and
            np.array_equal([r["mtm_pnl_usd"] for r in ref_rows], [r["mtm_pnl_usd"] for r in base_rows])),
        baseline_exactly_matches_frozen_reference_trade_log=(ref_trade_hash == baseline_trade_hash),
        baseline_and_binary_sweep_inputs_identical=(base_meta.get("input_hashes") == sweep_meta.get("input_hashes")),
        baseline_and_binary_sweep_code_hashes_identical=(base_meta.get("code_hashes") == sweep_meta.get("code_hashes")),
        baseline_and_binary_sweep_candidate_logs_identical=(base_candidate_hash == sweep_candidate_hash),
        baseline_candidate_log_sha256=base_candidate_hash,
        sweep_candidate_log_sha256=sweep_candidate_hash,
        baseline=base_stats, binary_sweep=sweep_stats,
        delta_final_mtm_usd=round(sweep_stats["final_mtm_usd"] - base_stats["final_mtm_usd"], 2),
        delta_max_close_drawdown_usd=round(sweep_stats["max_close_drawdown_usd"] - base_stats["max_close_drawdown_usd"], 2),
        caveat="Binary sweep strategy replay is preserved evidence; depth/volatility-conditioned strategy replay was not run.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output}")
    args.output.mkdir(parents=True)

    price_rows = read_csv(PRICES)
    dates = [r["datetime"][:10] for r in price_rows]
    if dates != sorted(set(dates)):
        raise AssertionError("price dates must be strictly increasing and unique")
    if len(price_rows) != 6526 or dates[0] != "2000-09-18" or dates[-1] != AS_OF:
        raise AssertionError(f"unexpected source extent: {len(dates)} {dates[0]}..{dates[-1]}")

    missing_ohlcv = 0
    zero_volume = 0
    invalid_range = 0
    for r in price_rows:
        vals = [float(r[k]) for k in ("open", "high", "low", "close", "volume")]
        if not all(math.isfinite(v) for v in vals):
            missing_ohlcv += 1
        if vals[-1] == 0:
            zero_volume += 1
        if vals[2] > min(vals[0], vals[1], vals[3]) or vals[1] < max(vals[0], vals[2], vals[3]):
            invalid_range += 1

    source_candidates = list(read_jsonl(BASELINE / "independent_early_candidates.jsonl"))
    raw_candidate_count = len(source_candidates)
    keyed = {}
    contact_key_counts = Counter()
    for c in source_candidates:
        z = c.get("zone") or {}
        idx, touch = int(c["index"]), int(c["touch_index"])
        if c.get("confirmation") != "independent_support_local_break":
            raise AssertionError("unexpected candidate family in early-candidate log")
        if not (0 <= int(z["available_index"]) < touch <= int(c["low_index"]) <= int(c["recovery_index"]) <= idx < len(price_rows)):
            raise AssertionError(f"candidate clock violation: {c.get('setup_id')}")
        if c["at"][:10] != dates[idx]:
            raise AssertionError(f"candidate date/index mismatch: {c.get('setup_id')}")
        key = (z["id"], touch)
        contact_key_counts[key] += 1
        # Earliest actual close-time candidate for this OB contact is the sole
        # observation in the primary de-duplicated population.
        if key not in keyed or (idx, c["setup_id"]) < (int(keyed[key]["index"]), keyed[key]["setup_id"]):
            keyed[key] = c
    candidates = sorted(keyed.values(), key=lambda x: (int(x["index"]), x["setup_id"]))
    tr = true_ranges(price_rows)
    atr_prior = rolling_mean_prior(tr, ATR_PERIOD)
    atr_mismatch = 0
    events = []
    for c in candidates:
        idx, touch = int(c["index"]), int(c["touch_index"])
        zone = c["zone"]
        low = float(c["low"])
        low_atr = float(c["low_atr"])
        recomputed_depth = max(0.0, (float(zone["wick_low"]) - low) / low_atr) if low_atr > 0 else 0.0
        logged_depth = float(c.get("sweep_depth_atr", 0.0) or 0.0)
        if not math.isclose(recomputed_depth, logged_depth, rel_tol=1e-9, abs_tol=1e-9):
            atr_mismatch += 1
        atr_decision = atr_prior[idx]
        label, hit_idx, hit_date = forward_label(price_rows, atr_decision, idx) if atr_decision else ("right_censored", None, None)
        pre_vol, pre_vol_pct, vol_history_n = pre_touch_volatility(price_rows, tr, touch)
        events.append(dict(
            setup_id=c["setup_id"], zone_id=zone["id"], timeframe=zone["timeframe"],
            zone_origin_date=zone["origin_at"][:10], zone_available_date=zone["available_at"][:10],
            touch_date=dates[touch], decision_date=dates[idx], decision_index=idx,
            touch_index=touch, low_index=int(c["low_index"]), recovery_index=int(c["recovery_index"]),
            zone_wick_low=float(zone["wick_low"]), reaction_low=low, low_atr=low_atr,
            sweep_depth_atr=round(recomputed_depth, 8), depth_bin=depth_bin(recomputed_depth),
            pre_touch_atr_pct=round(pre_vol * 100, 6) if pre_vol is not None else None,
            pre_touch_vol_percentile=round(pre_vol_pct * 100, 2) if pre_vol_pct is not None else None,
            volatility_regime=("unknown" if pre_vol_pct is None else "low" if pre_vol_pct <= 1 / 3 else
                               "mid" if pre_vol_pct <= 2 / 3 else "high"),
            pre_touch_vol_history_n=vol_history_n,
            decision_atr14_prior=round(atr_decision, 6) if atr_decision is not None else None,
            next_open=float(price_rows[idx + 1]["open"]) if idx + 1 < len(price_rows) else None,
            outcome_horizon_sessions=HORIZON, outcome_barrier_atr=ATR_MULTIPLE,
            forward_label=label, hit_index=hit_idx, hit_date=hit_date,
        ))

    outcomes = wilson_table(events, "depth_bin")
    vol_summary = wilson_table(events, "volatility_regime")
    cross = []
    for regime in ("low", "mid", "high"):
        for bin_name, _, _ in DEPTH_BINS:
            subset = [e for e in events if e["volatility_regime"] == regime and e["depth_bin"] == bin_name]
            mature = [e for e in subset if e["forward_label"] not in {"right_censored", "same_bar_ambiguous"}]
            succ = sum(e["forward_label"] == "up_first" for e in mature)
            lo, hi = wilson(succ, len(mature))
            counts = Counter(e["forward_label"] for e in subset)
            cross.append(dict(volatility_regime=regime, depth_bin=bin_name, candidate_events=len(subset),
                              evaluable=len(mature), up_first=counts["up_first"], down_first=counts["down_first"],
                              no_barrier_20d=counts["no_barrier_20d"], same_bar_ambiguous=counts["same_bar_ambiguous"],
                              right_censored=counts["right_censored"],
                              up_before_down_rate_pct=round(succ / len(mature) * 100, 2) if mature else None,
                              event_level_wilson95_low_pct=round(lo * 100, 2) if lo is not None else None,
                              event_level_wilson95_high_pct=round(hi * 100, 2) if hi is not None else None))

    bootstrap = cluster_bootstrap_deltas(events)
    adjusted_model, model_coefficients = adjusted_depth_model(events)
    timeframe_rows = []
    for timeframe in ("1D", "1W", "1M"):
        for bin_name, _, _ in DEPTH_BINS:
            subset = [e for e in events if e["timeframe"] == timeframe and e["depth_bin"] == bin_name]
            mature = [e for e in subset if e["forward_label"] not in {"right_censored", "same_bar_ambiguous"}]
            succ = sum(e["forward_label"] == "up_first" for e in mature)
            lo, hi = wilson(succ, len(mature))
            counts = Counter(e["forward_label"] for e in subset)
            timeframe_rows.append(dict(timeframe=timeframe, depth_bin=bin_name, candidate_events=len(subset),
                                       evaluable=len(mature), up_first=counts["up_first"],
                                       down_first=counts["down_first"], no_barrier_20d=counts["no_barrier_20d"],
                                       same_bar_ambiguous=counts["same_bar_ambiguous"],
                                       right_censored=counts["right_censored"],
                                       up_before_down_rate_pct=round(succ / len(mature) * 100, 2) if mature else None,
                                       event_level_wilson95_low_pct=round(lo * 100, 2) if lo is not None else None,
                                       event_level_wilson95_high_pct=round(hi * 100, 2) if hi is not None else None))
    era_rows = []
    for key, start, end in PERIODS:
        for bin_name, _, _ in DEPTH_BINS:
            subset = [e for e in events if start <= e["decision_date"] <= end and e["depth_bin"] == bin_name]
            mature = [e for e in subset if e["forward_label"] not in {"right_censored", "same_bar_ambiguous"}]
            succ = sum(e["forward_label"] == "up_first" for e in mature)
            era_rows.append(dict(period=key, depth_bin=bin_name, candidate_events=len(subset), evaluable=len(mature),
                                 up_first=succ, down_first=sum(e["forward_label"] == "down_first" for e in mature),
                                 no_barrier_20d=sum(e["forward_label"] == "no_barrier_20d" for e in mature),
                                 success_rate_pct=round(succ / len(mature) * 100, 2) if mature else None))

    factor_rows = read_csv(FACTOR_TABLE)
    factor = {r["key"]: r for r in factor_rows}
    prior_binary_sweep = factor["ob_distal_wick_sweep"]
    baseline = factor["baseline"]
    dec = [e for e in events if e["forward_label"] not in {"right_censored", "same_bar_ambiguous"}]
    raw_depths = [float(c.get("sweep_depth_atr", 0.0) or 0.0) for c in source_candidates]
    count_by_tf = Counter(e["timeframe"] for e in events)
    scope = dict(
        raw_candidates=raw_candidate_count, unique_zone_contact_events=len(events),
        raw_to_unique_collapsed=raw_candidate_count - len(events),
        contact_keys_with_multiple_raw_candidates=sum(n > 1 for n in contact_key_counts.values()),
        evaluable_primary_outcomes=len(dec),
        censoring=sum(e["forward_label"] == "right_censored" for e in events),
        same_bar_ambiguous=sum(e["forward_label"] == "same_bar_ambiguous" for e in events),
        timeframe_counts=dict(count_by_tf),
        sweep_events=sum(e["sweep_depth_atr"] > 0 for e in events),
        non_sweep_events=sum(e["sweep_depth_atr"] == 0 for e in events),
        median_depth_all_raw=sorted(raw_depths)[len(raw_depths) // 2] if raw_depths else None,
        recomputed_depth_mismatches=atr_mismatch,
    )
    prior_strategy_comparison = dict(
        source="factor_comparison.csv; same-input v0.11 replay previously verified against frozen baseline",
        baseline_final_mtm_usd=float(baseline["final_mtm_usd"]),
        any_distal_wick_sweep_final_mtm_usd=float(prior_binary_sweep["final_mtm_usd"]),
        any_distal_wick_sweep_delta_usd=round(float(prior_binary_sweep["final_mtm_usd"]) - float(baseline["final_mtm_usd"]), 2),
        baseline_max_close_dd_usd=float(baseline["max_close_drawdown_usd"]),
        any_distal_wick_sweep_max_close_dd_usd=float(prior_binary_sweep["max_close_drawdown_usd"]),
        baseline_completed_trades=int(baseline["completed_trades"]),
        any_distal_wick_sweep_completed_trades=int(prior_binary_sweep["completed_trades"]),
        status="existing binary-sweep portfolio replay; no ATR-depth or volatility-conditioned portfolio replay performed",
    )
    replay_audit = audit_existing_replays()
    if not all(replay_audit[k] for k in (
            "baseline_exactly_matches_frozen_reference_mtm",
            "baseline_exactly_matches_frozen_reference_trade_log",
            "baseline_and_binary_sweep_inputs_identical",
            "baseline_and_binary_sweep_code_hashes_identical",
            "baseline_and_binary_sweep_candidate_logs_identical")):
        raise AssertionError(f"replay comparison audit failed: {replay_audit}")
    for key in ("baseline", "binary_sweep"):
        selected = baseline if key == "baseline" else prior_binary_sweep
        actual = replay_audit[key]
        for field in ("final_mtm_usd", "max_close_drawdown_usd", "completed_trades"):
            csv_field = field
            if not math.isclose(float(selected[csv_field]), float(actual[field]), abs_tol=0.01):
                raise AssertionError(f"factor CSV vs underlying replay mismatch: {key}.{field}")

    write_csv(args.output / "candidate_forward_outcomes.csv", events)
    write_csv(args.output / "sweep_depth_outcomes.csv", outcomes)
    write_csv(args.output / "volatility_regime_outcomes.csv", vol_summary)
    write_csv(args.output / "depth_by_volatility_outcomes.csv", cross)
    write_csv(args.output / "period_depth_outcomes.csv", era_rows)
    write_csv(args.output / "timeframe_depth_outcomes.csv", timeframe_rows)
    write_csv(args.output / "adjusted_model_coefficients.csv", model_coefficients)
    write_csv(args.output / "cluster_bootstrap_deltas.csv", bootstrap)
    inputs = {
        "prices": {"path": str(PRICES), "sha256": sha256(PRICES)},
        "candidate_log": {"path": str(BASELINE / "independent_early_candidates.jsonl"),
                          "sha256": sha256(BASELINE / "independent_early_candidates.jsonl")},
        "baseline_replay_metadata": {"path": str(BASELINE / "metadata.json"),
                                     "sha256": sha256(BASELINE / "metadata.json")},
        "factor_comparison": {"path": str(FACTOR_TABLE), "sha256": sha256(FACTOR_TABLE)},
        "frozen_reference_mtm": {"path": str(BASELINE_REFERENCE / "daily_mtm.jsonl"),
                                  "sha256": sha256(BASELINE_REFERENCE / "daily_mtm.jsonl")},
    }
    summary = dict(
        study="v0.11 causal early bullish OB sweep depth x pre-contact volatility; exploratory event study",
        as_of=AS_OF, primary_decision="candidate close; forward entry reference next observed open",
        outcome_definition=f"+{ATR_MULTIPLE:g} x 14-session lagged mean true range before -{ATR_MULTIPLE:g} x same ATR within {HORIZON} observed sessions",
        depth_definition="max(0, (bullish OB wick_low - candidate reaction_low) / logged ATR at reaction low)",
        depth_bins={"no_sweep": "depth == 0", "shallow_0_0.25ATR": "0 < depth <= 0.25",
                    "moderate_0.25_0.5ATR": "0.25 < depth <= 0.5", "deep_over_0.5ATR": "depth > 0.5"},
        volatility_definition=("ATR(14)/close at the close before first contact; low/mid/high tercile rank "
                               "against up to 252 strictly earlier valid observations; at least 100 needed"),
        confidence_note="Wilson intervals are event-level descriptive intervals and ignore residual serial/OB clustering. "
                        "Year-cluster bootstrap is a sensitivity, not an out-of-sample guarantee.",
        data_quality=dict(rows=len(price_rows), start=dates[0], end=dates[-1], missing_nonfinite_ohlcv=missing_ohlcv,
                          zero_volume_rows=zero_volume, strict_ohlc_range_violations=invalid_range,
                          unique_sorted_dates=True),
        scope=scope, depth_outcomes=outcomes, volatility_outcomes=vol_summary,
        depth_by_volatility=cross, timeframe_by_depth=timeframe_rows,
        adjusted_model=adjusted_model, adjusted_model_coefficients=model_coefficients,
        cluster_bootstrap=bootstrap, independent_replay_audit=replay_audit,
        prior_strategy_level_binary_sweep_comparison=prior_strategy_comparison,
        inputs=inputs, parameters=dict(atr_period=ATR_PERIOD, horizon_sessions=HORIZON,
                                      target_stop_atr=ATR_MULTIPLE, volatility_window=VOL_WINDOW,
                                      vol_history_lookback=VOL_LOOKBACK, cluster_bootstrap_reps=BOOTSTRAPS,
                                      random_seed=SEED),
        status="COMPLETED_EXPLORATORY_EVENT_STUDY_NOT_OOS",
    )
    (args.output / "results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "report_ja.md").write_text(make_report(summary), encoding="utf-8")
    print(json.dumps(dict(status=summary["status"], output=str(args.output), scope=scope,
                          depth_outcomes=outcomes, volatility_outcomes=vol_summary,
                          prior_strategy_level_binary_sweep_comparison=prior_strategy_comparison),
                     ensure_ascii=False, indent=2))


def make_report(s: dict) -> str:
    moderate = next(r for r in s["depth_outcomes"] if r["group"] == "moderate_0.25_0.5ATR")
    none = next(r for r in s["depth_outcomes"] if r["group"] == "no_sweep")
    moderate_boot = next(r for r in s["cluster_bootstrap"] if r["group"] == "moderate_0.25_0.5ATR")
    adjusted_moderate = next(r for r in s["adjusted_model_coefficients"]
                             if "moderate_0.25_0.5ATR" in r["term"] and "change in depth contrast" not in r["interpretation"])
    am = s["adjusted_model"]
    lines = [
        "# v0.11 Bullish OB Sweep深度・接触前ボラティリティ検証", "",
        "## 結論", "",
        "この検証は、同じOB接触候補内でSweep深度と、その後20営業日の価格経路が関連するかを調べた探索的イベント分析です。",
        "候補生成条件や深度帯は損益を見て最適化していません。先行していた『任意のdistal-wick sweepを必須化』した戦略リプレイは参照しますが、",
        "今回のATR深度・ボラティリティ別の条件付き売買ポートフォリオは再生していません。したがって、以下は市場経路ラベルの関連であり、",
        "実運用成績や独立した将来予測の証明ではありません。", "",
        f"最も目立つ層別差は、OB wick-lowを0.25〜0.5 ATRだけ下回った候補です。上先着は{moderate['up_first']}/{moderate['evaluable']}={moderate['up_before_down_rate_pct']:.2f}%で、no-sweepの{none['up_before_down_rate_pct']:.2f}%より{moderate['up_before_down_rate_pct'] - none['up_before_down_rate_pct']:+.2f}ポイント高く、年クラスタbootstrapの差の95%区間は{moderate_boot['year_cluster_bootstrap95_low_pp']:+.2f}〜{moderate_boot['year_cluster_bootstrap95_high_pp']:+.2f}ポイントでした。",
        f"ただし、時間足・事前期間・接触前ボラを調整した深度全体の検定はp={am['depth_groups_joint_cluster_robust_p']:.3f}、深度×ボラ交互作用はp={am['depth_x_volatility_joint_cluster_robust_p']:.3f}。moderate個別係数はOR={adjusted_moderate['odds_ratio']:.2f}ですが、未補正p={adjusted_moderate['cluster_robust_p_value']:.3f}、Holm補正p={adjusted_moderate['holm_adjusted_p_value']:.3f}です。したがって『中程度Sweepが有望な仮説』までは言えるものの、確立した一般則やボラ適応ルールとは言えません。", "",
        "## 固定した検証仕様", "",
        f"- 入力: ES=F連続先物代理、{s['data_quality']['rows']:,}日足、{s['data_quality']['start']}〜{s['data_quality']['end']}。",
        f"- 候補単位: 同一bullish OB ID・同一初回接触indexは1件にまとめ、最初に確認できたearly候補だけを採用。生ログ{s['scope']['raw_candidates']:,}件から{s['scope']['unique_zone_contact_events']:,}件。",
        "- 選択母集団の注意: これはOBへの全初回タッチではなく、OB接触後に安値・回復・early local-breakまで成立し、検知器がearly BUY候補として出した事例に限定。したがって、最初のOB接触時点での反発率ではなく、候補確認後の上方向継続情報を測る。",
        f"- Sweep深度: `{s['depth_definition']}`。帯はno sweep、(0, 0.25] ATR、(0.25, 0.5] ATR、>0.5 ATR。",
        f"- 事前ボラ: {s['volatility_definition']}。Sweep接触日より前のデータだけを使用。",
        f"- 将来ラベル: 判断日終値の次の観測日始値を基準に、判断時点のATR14で+1 ATRと-1 ATRの先着を20観測日以内に評価。",
        "- 同一足で上下両障壁に触れた場合は順序不明として除外、20本の追跡期間を完了していない場合は打切り。no-hitは成功イベントではないため主率の分母に残す。",
        "- 同一OB接触episode内の重複は除外。異なるOBや近接日にまたがる依存は残るため、イベントWilson区間に加え年単位クラスタbootstrapを感度分析として算出。",
        "", "## Sweep深度別の結果", "",
        "『上先着率』は評価可能候補のうち20営業日以内に+1 ATRが-1 ATRより先に到達した割合。両方に届かない例は0扱いではなく、母集団内の別 outcome として分けて示します。", "",
        "| 深度帯 | 候補 | 評価可能 | +1 ATR先着 | -1 ATR先着 | 未到達 | 同足順序不明 | 打切り | 上先着率 (95% Wilson) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in s["depth_outcomes"]:
        rate = "—" if r["up_before_down_rate_pct"] is None else f"{r['up_before_down_rate_pct']:.2f}% ({r['event_level_wilson95_low_pct']:.2f}–{r['event_level_wilson95_high_pct']:.2f})"
        lines.append(f"| {r['group']} | {r['candidate_events']} | {r['evaluable']} | {r['up_first']} | {r['down_first']} | {r['no_barrier_20d']} | {r['same_bar_ambiguous']} | {r['right_censored']} | {rate} |")
    lines += ["", "## 深度差と不確実性", "", "| 深度帯 | no-sweep比 上先着率差 | 年クラスタbootstrap 95%区間 | 反復数 |", "|---|---:|---:|---:|"]
    for r in s["cluster_bootstrap"]:
        interval = "—" if r["difference_vs_no_sweep_pp"] is None else f"{r['year_cluster_bootstrap95_low_pp']:+.2f}〜{r['year_cluster_bootstrap95_high_pp']:+.2f} pp"
        delta = "—" if r["difference_vs_no_sweep_pp"] is None else f"{r['difference_vs_no_sweep_pp']:+.2f} pp"
        lines.append(f"| {r['group']} | {delta} | {interval} | {r['bootstrap_replicates']} |")
    lines += ["", "## 時間足・期間・接触前ボラを調整した関連", "",
              f"上先着を目的変数に、Sweep深度帯、接触前ボラ percentile、OB時間足、事前に定めた期間区分を入れたロジスティック回帰を当て、標準誤差は暦年でクラスタ化しました（n={am['n']:,}、{am['calendar_year_clusters']}年クラスタ）。これは関連調整であり、因果効果や未見予測性能ではありません。", "",
              f"- 深度カテゴリ全体のクラスタ頑健Wald検定: χ²({am['depth_groups_joint_wald_df']})={am['depth_groups_joint_wald_chi2']}, p={am['depth_groups_joint_cluster_robust_p']:.4g}。",
              f"- 深度×ボラ percentile交互作用: χ²({am['depth_x_volatility_joint_wald_df']})={am['depth_x_volatility_joint_wald_chi2']}, p={am['depth_x_volatility_joint_cluster_robust_p']:.4g}。",
              "- 深度の3つの個別対比にはHolm補正を適用。区間は個別の点ごとの95%区間で、多重比較調整済みではありません。",
              "", "| 調整済み深度係数 | オッズ比 | 年クラスタ頑健95%区間 | p値 | Holm補正p |", "|---|---:|---:|---:|---:|"]
    for r in s["adjusted_model_coefficients"]:
        if "change in depth contrast" in r["interpretation"]:
            continue
        lines.append(f"| {r['term']} | {r['odds_ratio']:.3f} | {r['cluster_robust95_low_or']:.3f}–{r['cluster_robust95_high_or']:.3f} | {r['cluster_robust_p_value']:.4g} | {r['holm_adjusted_p_value']:.4g} |")
    lines += ["", "## 時間足別", "", "| 時間足 | 深度帯 | 候補 | 評価可能 | 上先着率 (95% Wilson) |", "|---|---|---:|---:|---:|"]
    for r in s["timeframe_by_depth"]:
        rate = "—" if r["up_before_down_rate_pct"] is None else f"{r['up_before_down_rate_pct']:.2f}% ({r['event_level_wilson95_low_pct']:.2f}–{r['event_level_wilson95_high_pct']:.2f})"
        lines.append(f"| {r['timeframe']} | {r['depth_bin']} | {r['candidate_events']} | {r['evaluable']} | {rate} |")
    lines += ["", "## 接触前ボラティリティ別", "", "| 事前ボラ tercile | 候補 | 評価可能 | 上先着 | 下先着 | 未到達 | 同足不明 | 打切り | 上先着率 (95% Wilson) |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in s["volatility_outcomes"]:
        rate = "—" if r["up_before_down_rate_pct"] is None else f"{r['up_before_down_rate_pct']:.2f}% ({r['event_level_wilson95_low_pct']:.2f}–{r['event_level_wilson95_high_pct']:.2f})"
        lines.append(f"| {r['group']} | {r['candidate_events']} | {r['evaluable']} | {r['up_first']} | {r['down_first']} | {r['no_barrier_20d']} | {r['same_bar_ambiguous']} | {r['right_censored']} | {rate} |")
    lines += ["", "深度×ボラの完全な件数・率は`depth_by_volatility_outcomes.csv`に保存しています。小セルの率は大きな不確実性を伴うため、独立した確率推定と見なさないでください。", "",
              "## 既存の戦略レベル比較との照合", "",
              f"既存の同一v0.11リプレイでは、任意のdistal-wick sweepをearly BUY必須条件にした場合、MTMは${s['prior_strategy_level_binary_sweep_comparison']['baseline_final_mtm_usd']:,.0f}から${s['prior_strategy_level_binary_sweep_comparison']['any_distal_wick_sweep_final_mtm_usd']:,.0f}へ、差は${s['prior_strategy_level_binary_sweep_comparison']['any_distal_wick_sweep_delta_usd']:+,.0f}でした。完了取引は{s['prior_strategy_level_binary_sweep_comparison']['baseline_completed_trades']}件から{s['prior_strategy_level_binary_sweep_comparison']['any_distal_wick_sweep_completed_trades']}件、最大終値DDは${s['prior_strategy_level_binary_sweep_comparison']['baseline_max_close_dd_usd']:,.0f}から${s['prior_strategy_level_binary_sweep_comparison']['any_distal_wick_sweep_max_close_dd_usd']:,.0f}へ変化しています。",
              f"baselineのMTM日次系列・取引ログは凍結済み基準と完全一致し、binary-sweepとの入力hash・実行コードhash・候補ログも一致（候補ログSHA-256 `{s['independent_replay_audit']['baseline_candidate_log_sha256']}`）しました。CSV集計と日次MTM/取引ログからの再計算値も一致。",
              "この二値ゲートの結果は今回の候補イベント率とは異なる量です。ATR深度・事前ボラでゲートするポートフォリオはまだ実行していないため、イベント率がよくても損益改善を意味しません。", "",
              "## データ品質・時計監査", "",
              f"- 厳密増加・一意日付: 確認済み。非有限OHLCV行: {s['data_quality']['missing_nonfinite_ohlcv']}件、出来高ゼロ: {s['data_quality']['zero_volume_rows']}件、OHLC範囲条件違反: {s['data_quality']['strict_ohlc_range_violations']}件。元データの既知の caveat を保持し、補正していません。",
              f"- 価格入力SHA-256: `{s['inputs']['prices']['sha256']}`。候補ログSHA-256: `{s['inputs']['candidate_log']['sha256']}`。深度再計算と候補ログ差異: {s['scope']['recomputed_depth_mismatches']}件。",
              "- すべての候補でOB available index < touch index ≤ low index ≤ recovery index ≤ decision indexを監査。未来側の日足は候補特徴に使っていません。将来ラベルはdecision index+1から開始。", "",
              "## 解釈と次の判断", "",
              "- Sweep深度と事前ボラの効果は、表の層別差とクラスタ区間を合わせて判断する。イベントWilson区間は時系列依存を無視するため、単独で有意性を主張しない。",
              "- この履歴は何度も参照済みで、期間を切っても完全な未見テストではない。深度帯やボラ区分を選び直して全期間成績を最大化しない。",
              "- 現時点で閾値別の売買モデルへ変更する根拠はない。戦略として採用候補にするなら、先に特徴量・閾値・出口・採用基準を固定した別リプレイを行い、時系列ウォークフォワードと今後の新規期間で再確認する。", "",
              "## 成果物", "",
              "- `candidate_forward_outcomes.csv`: 重複を除いた各OB接触候補と将来ラベル。",
              "- `sweep_depth_outcomes.csv`, `volatility_regime_outcomes.csv`, `depth_by_volatility_outcomes.csv`: 集計表。",
              "- `cluster_bootstrap_deltas.csv`: no-sweepとの差の年クラスタbootstrap。",
              "- `period_depth_outcomes.csv`, `timeframe_depth_outcomes.csv`: 過去区分・時間足別の記述統計。",
              "- `adjusted_model_coefficients.csv`: 深度カテゴリとボラ交互作用の調整済みモデル係数。",
              "- `results.json`: 条件・入力hash・全集計。", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
