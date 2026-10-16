"""Frozen v0.11 research audit: attribution, price-only reference, and matched event study.

This script does not modify or refit the strategy. It consumes the frozen v0.11
replay and its input files, then writes a separate, reproducible research bundle.
All controls and labels are defined here before computing outcomes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path


PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parent
RUN = PACKAGE / "validation_rev11_without_continuation_20260923"
DEFAULT_OUT = PACKAGE / "validation_rev11_research_20260924_v8"
HORIZON = 20
BLOCK_BOOTSTRAPS = 4000
SEED = 20260924


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]


def read_csv(path: Path) -> tuple[list[dict], dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        names = {name.lower(): name for name in (reader.fieldnames or [])}
    return rows, names


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def avg(xs):
    return statistics.fmean(xs) if xs else None


def med(xs):
    return statistics.median(xs) if xs else None


def percentile(xs: list[float], p: float):
    if not xs:
        return None
    ys = sorted(xs)
    k = (len(ys) - 1) * p
    lo, hi = math.floor(k), math.ceil(k)
    return ys[lo] if lo == hi else ys[lo] * (hi - k) + ys[hi] * (k - lo)


def fmt(x, digits=4):
    return "n/a" if x is None else f"{x:.{digits}f}"


def load_market(path: Path):
    rows, cols = read_csv(path)
    required = ["datetime", "open", "high", "low", "close", "volume"]
    missing = [x for x in required if x not in cols]
    if missing:
        raise ValueError(f"missing required columns {missing}; available={list(cols)}")
    dates = [str(r[cols["datetime"]])[:10] for r in rows]
    out = {"dates": dates}
    for field in ("open", "high", "low", "close", "volume"):
        out[field] = [float(r[cols[field]]) for r in rows]
    invalid_ohlc = sum(1 for i in range(len(rows)) if
                       out["high"][i] < max(out["open"][i], out["close"][i]) or
                       out["low"][i] > min(out["open"][i], out["close"][i]) or
                       out["high"][i] < out["low"][i])
    return out, {"rows": len(rows), "start": dates[0], "end": dates[-1],
                 "unique_dates": len(set(dates)), "sorted": dates == sorted(dates),
                 "duplicate_dates": len(dates) - len(set(dates)),
                 "invalid_ohlc_rows_after_source_filter": invalid_ohlc,
                 "zero_volume_rows": sum(v == 0 for v in out["volume"]),
                 "negative_volume_rows": sum(v < 0 for v in out["volume"])}


def rolling_mean(xs: list[float], window: int) -> list[float | None]:
    result = [None] * len(xs)
    total = 0.0
    for i, value in enumerate(xs):
        total += value
        if i >= window:
            total -= xs[i - window]
        if i + 1 >= window:
            result[i] = total / window
    return result


def market_features(mkt):
    o, h, l, c = (mkt[k] for k in ("open", "high", "low", "close"))
    tr = []
    for i in range(len(c)):
        prev = c[i - 1] if i else c[i]
        tr.append(max(h[i] - l[i], abs(h[i] - prev), abs(l[i] - prev)))
    atr = rolling_mean(tr, 14)
    sma200 = rolling_mean(c, 200)
    ret = [None] + [(c[i] / c[i - 1] - 1.0) for i in range(1, len(c))]
    feats = []
    for i in range(len(c)):
        def close_return(lookback):
            if i < lookback or c[i - lookback] <= 0:
                return None
            return math.log(c[i] / c[i - lookback])
        hist_returns = [x for x in ret[max(1, i - 19):i + 1] if x is not None]
        vol20 = statistics.pstdev(hist_returns) if len(hist_returns) >= 15 else None
        start = max(0, i - 62)
        hi63, lo63 = max(h[start:i + 1]), min(l[start:i + 1])
        range_pos = (c[i] - lo63) / (hi63 - lo63) if hi63 > lo63 else 0.5
        feats.append({
            "ret5": close_return(5), "ret20": close_return(20),
            "atr_pct": atr[i] / c[i] if atr[i] and c[i] else None,
            "vol20": vol20,
            "range_pos63": range_pos,
            "sma200_gap": c[i] / sma200[i] - 1.0 if sma200[i] else None,
            "bar_return_atr": (c[i] - o[i]) / atr[i] if atr[i] else None,
            "bar_range_atr": (h[i] - l[i]) / atr[i] if atr[i] else None,
            "close_location": (c[i] - l[i]) / (h[i] - l[i]) if h[i] > l[i] else 0.5,
            "lower_wick_atr": (min(o[i], c[i]) - l[i]) / atr[i] if atr[i] else None,
        })
    return atr, sma200, feats


def run_sma200(mkt, sma200):
    """Causal 200-day close/SMA cross; next observed open; one long, same costs."""
    dates, o, c = mkt["dates"], mkt["open"], mkt["close"]
    point_value, tick, slip_ticks, commission = 20.0, 0.25, 1.0, 2.50
    slips = tick * slip_ticks
    position = None
    pending = None
    realized = 0.0
    equity, exposure, trades = [], [], []
    for i in range(len(dates)):
        if pending is not None:
            if pending["side"] == "buy" and position is None:
                fill = o[i] + slips
                realized -= commission
                position = {"entry_index": i, "entry_at": dates[i], "entry_price": fill,
                            "entry_fee": commission}
            elif pending["side"] == "sell" and position is not None:
                fill = o[i] - slips
                pnl = (fill - position["entry_price"]) * point_value - commission
                realized += pnl
                trades.append({**position, "exit_index": i, "exit_at": dates[i],
                               "exit_price": fill, "net_pnl_usd": pnl - position["entry_fee"]})
                position = None
            pending = None
        mark = realized
        if position is not None:
            mark += (c[i] - position["entry_price"]) * point_value
        equity.append(mark)
        exposure.append(position is not None)
        if i == len(dates) - 1 or sma200[i] is None or sma200[i - 1] is None:
            continue
        cross_up = c[i] > sma200[i] and c[i - 1] <= sma200[i - 1]
        cross_down = c[i] < sma200[i] and c[i - 1] >= sma200[i - 1]
        if position is None and cross_up:
            pending = {"side": "buy", "decision_at": dates[i]}
        elif position is not None and cross_down:
            pending = {"side": "sell", "decision_at": dates[i]}
    open_pnl = 0.0
    if position is not None:
        open_pnl = (c[-1] - position["entry_price"]) * point_value
    peak = 0.0
    dd = []
    for value in equity:
        peak = max(peak, value)
        dd.append(value - peak)
    return {"equity": equity, "exposure": exposure, "trades": trades,
            "final_mtm_usd": equity[-1], "max_close_drawdown_usd": min(dd),
            "completed_trades": len(trades), "realized_closed_trades_net_usd": sum(t["net_pnl_usd"] for t in trades),
            "open_position": position is not None, "open_pnl_before_exit_fee_usd": open_pnl,
            "exposure_fraction": sum(exposure) / len(exposure) if exposure else None,
            "signal": "close crosses 200-day SMA; fills at next observed open",
            "fees": {"point_value_usd": point_value, "slippage_ticks_per_side": slip_ticks,
                     "tick_size": tick, "commission_per_side_usd": commission}}


def benchmark_buy_hold(mkt):
    # Buy first observed open plus one tick, pay entry commission, mark to closes.
    entry = mkt["open"][0] + 0.25
    return [(x - entry) * 20.0 - 2.50 for x in mkt["close"]]


def drawdown(equity):
    peak, minimum = 0.0, 0.0
    for x in equity:
        peak = max(peak, x)
        minimum = min(minimum, x - peak)
    return minimum


def trade_attribution(run_dir: Path, market, idx_by_date, final_state):
    trades = read_jsonl(run_dir / "trades.jsonl")
    pnls = [float(t["net_pnl_usd"]) for t in trades]
    categorized = defaultdict(list)
    by_exit = defaultdict(list)
    records = []
    for t in trades:
        ctx = t.get("entry_structural_context") or {}
        zone = ctx.get("source_zone") or {}
        entry_group = "OB_reaction" if zone.get("kind") == "ob_candidate" and zone.get("direction") == "bullish" else "recovery_price_structure_no_OB"
        categorized[entry_group].append(float(t["net_pnl_usd"]))
        by_exit[t.get("reason", "unknown")].append(float(t["net_pnl_usd"]))
        i0 = idx_by_date.get(str(t["entry_at"])[:10])
        i1 = idx_by_date.get(str(t["exit_at"])[:10])
        stop = ctx.get("stop")
        risk = max(0.0, (float(t["entry_price"]) - float(stop)) * 20.0) if stop is not None else None
        records.append({
            "entry_at": str(t["entry_at"])[:10], "exit_at": str(t["exit_at"])[:10],
            "duration_bars_inclusive": (i1 - i0 + 1) if i0 is not None and i1 is not None else None,
            "entry_group": entry_group, "entry_confirmation": t.get("entry_confirmation"),
            "entry_timeframe": zone.get("timeframe"), "entry_zone_id": zone.get("id"),
            "exit_reason": t.get("reason"), "net_pnl_usd": float(t["net_pnl_usd"]),
            "initial_risk_usd": risk,
            "net_r": float(t["net_pnl_usd"]) / risk if risk and risk > 0 else None,
        })
    gross = [x for x in pnls if x > 0]
    top = sorted(records, key=lambda x: x["net_pnl_usd"], reverse=True)[:5]
    open_pos = final_state.get("open_position") or {}
    last_mtm = read_jsonl(run_dir / "daily_mtm.jsonl")[-1]
    realized = sum(pnls)
    return {
        "completed_count": len(pnls), "wins": sum(x > 0 for x in pnls), "losses": sum(x < 0 for x in pnls),
        "win_rate": sum(x > 0 for x in pnls) / len(pnls) if pnls else None,
        "net_completed_usd": sum(pnls), "gross_profit_usd": sum(gross),
        "gross_loss_usd": sum(x for x in pnls if x < 0),
        "profit_factor": sum(gross) / abs(sum(x for x in pnls if x < 0)) if any(x < 0 for x in pnls) else None,
        "mean_trade_usd": avg(pnls), "median_trade_usd": med(pnls),
        "mean_net_r": avg([x["net_r"] for x in records if x["net_r"] is not None]),
        "median_net_r": med([x["net_r"] for x in records if x["net_r"] is not None]),
        "net_by_entry_group": {k: {"n": len(v), "net_usd": sum(v), "mean_usd": avg(v),
                                    "wins": sum(x > 0 for x in v)} for k, v in categorized.items()},
        "net_by_exit_reason": {k: {"n": len(v), "net_usd": sum(v), "mean_usd": avg(v),
                                    "wins": sum(x > 0 for x in v)} for k, v in by_exit.items()},
        "top5_net_usd": sum(x["net_pnl_usd"] for x in top),
        "top5_share_of_gross_profit": sum(x["net_pnl_usd"] for x in top) / sum(gross) if gross else None,
        "top5_trades": top,
        "remaining_38_net_usd": sum(pnls) - sum(x["net_pnl_usd"] for x in top),
        "closed_trade_max_drawdown_usd": drawdown([sum(pnls[:i + 1]) for i in range(len(pnls))]),
        "ending_open_position": bool(open_pos),
        "ending_open_pnl_usd_from_equity_minus_closed": float(last_mtm["mtm_pnl_usd"]) - realized,
        "trade_records": records,
    }


def features_for_date(feats, index):
    return feats[index]


def collect_ob_event_dates(run_dir: Path, market, atr, feats):
    rows = []
    for filename, route in (("independent_early_candidates.jsonl", "early"),
                            ("independent_confirmed_candidates.jsonl", "confirmed")):
        for row in read_jsonl(run_dir / filename):
            zone = row.get("zone") or {}
            if zone.get("kind") != "ob_candidate" or zone.get("direction") != "bullish":
                continue
            i = int(row.get("index", -1))
            touch_i = int(row.get("touch_index", -1))
            if i < 200 or i + HORIZON >= len(market["dates"]) or not atr[i]:
                continue
            rows.append({
                "index": i, "touch_index": touch_i, "route": route,
                "date": market["dates"][i], "zone_id": zone.get("id"),
                "zone_timeframe": zone.get("timeframe"),
                "available_index": zone.get("available_index"),
                "zone_width_atr": ((float(zone.get("wick_high") or zone.get("body_high")) -
                                    float(zone.get("wick_low") or zone.get("body_low"))) / atr[i]),
                "setup_id": row.get("setup_id"),
                "features": features_for_date(feats, i),
            })
    # Collapse early/confirmed duplicates of the same zone touch to the first causal decision.
    per_touch = {}
    for row in rows:
        key = (row["zone_id"], row["touch_index"])
        old = per_touch.get(key)
        order = (row["index"], 0 if row["route"] == "early" else 1, row["zone_id"] or "")
        if old is None or order < old[0]:
            per_touch[key] = (order, row)
    dedup = sorted((v[1] for v in per_touch.values()), key=lambda x: (x["index"], x["available_index"] or -1, x["zone_id"] or ""))
    # One event per close date, then a 20-bar non-overlap clock; event rule uses only signal timestamps.
    unique_date = {}
    for row in dedup:
        unique_date.setdefault(row["date"], row)
    clustered, last_i = [], -10**9
    for row in sorted(unique_date.values(), key=lambda x: x["index"]):
        if row["index"] - last_i > HORIZON:
            clustered.append(row)
            last_i = row["index"]
    return dedup, list(unique_date.values()), clustered


def event_outcome(mkt, atr, i, horizon=HORIZON):
    """Next-open long outcomes; OHLC same-bar barrier ordering remains ambiguous."""
    start, end = i + 1, min(i + horizon, len(mkt["dates"]) - 1)
    if start > end or not atr[i] or atr[i] <= 0:
        return None
    entry = mkt["open"][start] + 0.25
    high = max(mkt["high"][start:end + 1])
    low = min(mkt["low"][start:end + 1])
    term = (mkt["close"][end] - entry) / atr[i]
    mfe = (high - entry) / atr[i]
    mae = (low - entry) / atr[i]
    up, down = entry + atr[i], entry - atr[i]
    first = "censored"
    hit_at = None
    for j in range(start, end + 1):
        hit_up = mkt["high"][j] >= up
        hit_down = mkt["low"][j] <= down
        if hit_up and hit_down:
            first, hit_at = "same_day_ambiguous", mkt["dates"][j]
            break
        if hit_up:
            first, hit_at = "up_first", mkt["dates"][j]
            break
        if hit_down:
            first, hit_at = "down_first", mkt["dates"][j]
            break
    return {"horizon": end - i, "terminal_return_atr": term, "mfe_atr": mfe, "mae_atr": mae,
            "barrier_outcome": first, "barrier_hit_at": hit_at}


def robust_scale(values):
    valid = [x for x in values if x is not None and math.isfinite(x)]
    if not valid:
        return 1.0
    center = statistics.median(valid)
    mad = statistics.median(abs(x - center) for x in valid)
    return max(1.4826 * mad, statistics.pstdev(valid) if len(valid) > 1 and mad == 0 else 1e-8)


def matched_event_study(run_dir, mkt, atr, feats):
    dedup, unique_dates, events = collect_ob_event_dates(run_dir, mkt, atr, feats)
    dates = mkt["dates"]
    # Controls are dates without a detected bullish-OB reaction, and at least 20 bars
    # from any detected candidate date. Match within the same calendar year. The
    # second, prespecified robustness specification also matches the reaction-day
    # candle to test whether named OB candidates add information beyond a generic
    # reversal candle.
    all_candidate_indices = {x["index"] for x in unique_dates}
    specs = {
        # Retained exploratory pilot: first pass had no distance caliper. It is
        # deliberately reported rather than silently discarded after review.
        "exploratory_pilot_no_caliper": (("ret5", "ret20", "atr_pct", "vol20", "range_pos63", "sma200_gap"), None),
        "decision_state_only": (("ret5", "ret20", "atr_pct", "vol20", "range_pos63", "sma200_gap"), 2.5),
        "decision_state_plus_reaction_candle": (("ret5", "ret20", "atr_pct", "vol20", "range_pos63", "sma200_gap",
                                                  "bar_return_atr", "bar_range_atr", "close_location", "lower_wick_atr"), 2.5),
    }
    categories = ("up_first", "down_first", "same_day_ambiguous", "censored")
    summaries, all_pairs = {}, []
    for spec_name, (feature_keys, max_rms_caliper) in specs.items():
        eligible = []
        for i in range(200, len(dates) - HORIZON):
            if any(abs(i - j) <= HORIZON for j in all_candidate_indices):
                continue
            f = feats[i]
            if all(f.get(k) is not None and math.isfinite(f[k]) for k in feature_keys):
                eligible.append(i)
        controls_by_year = defaultdict(list)
        for j in eligible:
            controls_by_year[dates[j][:4]].append(j)
        events_by_year = defaultdict(list)
        for e in events:
            if all(e["features"].get(k) is not None and math.isfinite(e["features"][k]) for k in feature_keys):
                events_by_year[e["date"][:4]].append(e)
        pairs, unmatched_no_controls, unmatched_caliper, unmatched_label = [], 0, 0, 0
        for year in sorted(events_by_year):
            es, cs = events_by_year[year], controls_by_year.get(year, [])
            if not cs:
                unmatched_no_controls += len(es)
                continue
            scales = {k: robust_scale([feats[j][k] for j in cs] + [e["features"][k] for e in es]) for k in feature_keys}
            available = set(cs)
            # Greedy chronological nearest-neighbour matching without replacement.
            for e in es:
                if not available:
                    unmatched_no_controls += 1
                    continue
                def rms_distance(j):
                    return math.sqrt(sum(((e["features"][k] - feats[j][k]) / scales[k]) ** 2 for k in feature_keys) / len(feature_keys))
                best = min(available, key=rms_distance)
                distance = rms_distance(best)
                if max_rms_caliper is not None and distance > max_rms_caliper:
                    unmatched_caliper += 1
                    continue
                available.remove(best)
                ev, co = event_outcome(mkt, atr, e["index"]), event_outcome(mkt, atr, best)
                if ev is None or co is None:
                    unmatched_label += 1
                    continue
                pairs.append({
                    "match_spec": spec_name, "year": year, "event_date": e["date"], "control_date": dates[best],
                    "event_index": e["index"], "control_index": best, "zone_id": e["zone_id"],
                    "zone_timeframe": e["zone_timeframe"], "route": e["route"],
                    "zone_width_atr": e["zone_width_atr"], "match_distance_rms": distance,
                    **{f"event_{k}": v for k, v in ev.items()},
                    **{f"control_{k}": v for k, v in co.items()},
                    "paired_terminal_return_diff_atr": ev["terminal_return_atr"] - co["terminal_return_atr"],
                    "paired_mfe_diff_atr": ev["mfe_atr"] - co["mfe_atr"],
                })
        all_pairs.extend(pairs)
        diffs = [r["paired_terminal_return_diff_atr"] for r in pairs]
        by_year = defaultdict(list)
        for r in pairs:
            by_year[r["year"]].append(r["paired_terminal_return_diff_atr"])
        rng = random.Random(SEED + (1 if spec_name.endswith("candle") else 0))
        years = list(by_year)
        boot = []
        if years:
            for _ in range(BLOCK_BOOTSTRAPS):
                sampled = [rng.choice(years) for __ in years]
                values = [d for y in sampled for d in by_year[y]]
                if values:
                    boot.append(statistics.fmean(values))
        def cat_rates(prefix):
            c = Counter(r[f"{prefix}_barrier_outcome"] for r in pairs)
            return {k: {"n": c.get(k, 0), "rate_all_pairs": c.get(k, 0) / len(pairs) if pairs else None} for k in categories}
        summaries[spec_name] = {
            "matching_features": list(feature_keys), "max_rms_caliper": max_rms_caliper,
            "matched_pairs": len(pairs), "unmatched_no_controls": unmatched_no_controls,
            "unmatched_caliper": unmatched_caliper, "unmatched_label": unmatched_label,
            "distinct_calendar_years": len(years),
            "match_distance_rms_median": med([x["match_distance_rms"] for x in pairs]),
            "match_distance_rms_p90": percentile([x["match_distance_rms"] for x in pairs], .9),
            "event_terminal_return_atr": avg([r["event_terminal_return_atr"] for r in pairs]),
            "control_terminal_return_atr": avg([r["control_terminal_return_atr"] for r in pairs]),
            "paired_terminal_return_diff_atr_mean": avg(diffs),
            "paired_terminal_return_diff_atr_median": med(diffs),
            "paired_terminal_return_diff_atr_95pct_year_block_bootstrap_ci": [percentile(boot, .025), percentile(boot, .975)] if boot else None,
            "paired_mfe_diff_atr_mean": avg([r["paired_mfe_diff_atr"] for r in pairs]),
            "event_barrier_outcomes": cat_rates("event"), "control_barrier_outcomes": cat_rates("control"),
            "positive_paired_terminal_diff_fraction": sum(x > 0 for x in diffs) / len(diffs) if diffs else None,
        }
    summary = {
        "definition": "Bullish OB candidate reactions, deduplicated by zone-touch and date, then first candidate in each non-overlapping 20-session window. Match next-open/20-session outcomes to one non-candidate date in the same calendar year, without replacement; no strategy fills are implied.",
        "candidate_counts": {"raw_zone_touch_rows_after_route_merge": len(dedup), "unique_candidate_dates": len(unique_dates),
                             "nonoverlap_event_dates": len(events), "candidate_calendar_years": len({e['date'][:4] for e in events})},
        "matching_controls_without_replacement": True,
        "control_exclusion": "Dates within 20 sessions of any candidate date excluded to reduce overlapping event windows; this is a retrospective clean-control sample filter, not a live-time rule.",
        "matching_trial_log": [
            {"specification": name, "matched_pairs": value["matched_pairs"],
             "features": value["matching_features"], "max_rms_caliper": value["max_rms_caliper"],
             "mean_paired_return_diff_atr": value["paired_terminal_return_diff_atr_mean"],
             "year_block_bootstrap_95pct_ci": value["paired_terminal_return_diff_atr_95pct_year_block_bootstrap_ci"],
             "status": "exploratory; historical outcomes were inspected during method refinement"}
            for name, value in summaries.items()
        ],
        "horizon_sessions": HORIZON, "barrier_definition": "+1 ATR / -1 ATR from next observed open; same-bar dual touch = ambiguous; no touch by horizon = censored",
        "specifications": summaries,
        "warning": "Exploratory matched event study, not an untouched holdout and not a causal proof that an OB caused a turn. Candidate detector was selected after broad historical inspection; daily OHLC cannot order intrabar highs/lows.",
    }
    return summary, all_pairs


def top_exit_event_study(trades, mkt, atr, idx_by_date):
    rows = []
    for t in trades:
        if t.get("reason") != "campaign_double_top_early":
            continue
        fill_i = idx_by_date.get(str(t["exit_at"])[:10])
        decision = t.get("sell_decision") or {}
        decision_i = decision.get("index")
        if fill_i is None or decision_i is None or fill_i + HORIZON >= len(mkt["dates"]):
            continue
        outcome = event_outcome(mkt, atr, fill_i - 1, HORIZON) if fill_i > 0 and atr[fill_i - 1] else None
        if outcome is None:
            continue
        # event_outcome enters next open, whereas the position was already exited at this fill.
        exit_px = float(t["exit_price"])
        end_i = fill_i + HORIZON
        rows.append({
            "decision_at": mkt["dates"][int(decision_i)], "fill_at": mkt["dates"][fill_i],
            "exit_price": exit_px, "atr_at_decision": atr[int(decision_i)] if atr[int(decision_i)] else None,
            "forward_20_close_return_atr_from_fill": (mkt["close"][end_i] - exit_px) / atr[int(decision_i)] if atr[int(decision_i)] else None,
            "forward_20_max_down_atr_from_fill": (min(mkt["low"][fill_i + 1:end_i + 1]) - exit_px) / atr[int(decision_i)] if atr[int(decision_i)] else None,
            "forward_20_max_up_atr_from_fill": (max(mkt["high"][fill_i + 1:end_i + 1]) - exit_px) / atr[int(decision_i)] if atr[int(decision_i)] else None,
            "exit_pnl_usd": float(t["net_pnl_usd"]), "entry_at": str(t["entry_at"])[:10],
        })
    return {"executed_top_exit_events_with_full_20_session_horizon": len(rows),
            "mean_forward_20_close_return_atr": avg([x["forward_20_close_return_atr_from_fill"] for x in rows]),
            "median_forward_20_close_return_atr": med([x["forward_20_close_return_atr_from_fill"] for x in rows]),
            "mean_forward_20_max_down_atr": avg([x["forward_20_max_down_atr_from_fill"] for x in rows]),
            "mean_forward_20_max_up_atr": avg([x["forward_20_max_up_atr_from_fill"] for x in rows]),
            "forward_20_close_below_exit_count": sum(x["forward_20_close_return_atr_from_fill"] < 0 for x in rows),
            "forward_20_close_above_exit_count": sum(x["forward_20_close_return_atr_from_fill"] > 0 for x in rows),
            "post_exit_up_1atr_barrier_first_rate": None,
            "note": "Descriptive forward path after exit fill; it is not a no-exit counterfactual and does not isolate double-top exit value.",
            "rows": rows}


def year_and_window_summaries(mkt, strategy_equity, bh_equity, sma, strategy_exposure, sma_exposure):
    dates = mkt["dates"]
    starts = [("2000-09-18", "2026-08-27"), ("2006-01-01", "2010-12-31"),
              ("2011-01-01", "2015-12-31"), ("2016-01-01", "2018-12-31"),
              ("2019-01-01", "2022-12-31"), ("2023-01-01", "2026-08-27")]
    windows = []
    for a, b in starts:
        sel = [i for i, d in enumerate(dates) if a <= d <= b]
        if not sel:
            continue
        i0, i1 = sel[0], sel[-1]
        if i0 == 0:
            v011_local, bh_local, sma_local = strategy_equity[i1], bh_equity[i1], sma["equity"][i1]
        else:
            v011_local = strategy_equity[i1] - strategy_equity[i0]
            bh_local = bh_equity[i1] - bh_equity[i0]
            sma_local = sma["equity"][i1] - sma["equity"][i0]
        windows.append({"start": dates[i0], "end": dates[i1], "bars": len(sel),
                        "v011_change_usd": v011_local,
                        "buy_hold_change_usd": bh_local,
                        "v011_minus_buy_hold_usd": v011_local - bh_local,
                        "sma200_change_usd": sma_local,
                        "v011_exposure_fraction": sum(strategy_exposure[i0:i1 + 1]) / len(sel),
                        "sma200_exposure_fraction": sum(sma_exposure[i0:i1 + 1]) / len(sel)})
    annual = []
    years = sorted({d[:4] for d in dates})
    for y in years:
        sel = [i for i, d in enumerate(dates) if d.startswith(y)]
        i0, i1 = sel[0], sel[-1]
        if i0 == 0:
            v011_local, bh_local, sma_local = strategy_equity[i1], bh_equity[i1], sma["equity"][i1]
        else:
            v011_local = strategy_equity[i1] - strategy_equity[i0]
            bh_local = bh_equity[i1] - bh_equity[i0]
            sma_local = sma["equity"][i1] - sma["equity"][i0]
        annual.append({"year": y, "bars": len(sel),
                       "v011_change_usd": v011_local,
                       "buy_hold_change_usd": bh_local,
                       "sma200_change_usd": sma_local,
                       "v011_exposure_fraction": sum(strategy_exposure[i0:i1 + 1]) / len(sel),
                       "sma200_exposure_fraction": sum(sma_exposure[i0:i1 + 1]) / len(sel)})
    return windows, annual


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    meta = read_json(args.run / "metadata.json")
    inputs = {name: Path(info["path"]) for name, info in meta["input_hashes"].items() if isinstance(info, dict) and info.get("path")}
    input_integrity = {}
    for name, path in inputs.items():
        actual = sha256(path)
        expected = meta["input_hashes"][name].get("sha256")
        input_integrity[name] = {"path": str(path), "expected_sha256": expected,
                                 "actual_sha256": actual, "match": actual == expected}
    market, market_qc = load_market(inputs["prices"])
    source_manifest_path = market_manifest_path(inputs["prices"])
    source_manifest = read_json(source_manifest_path) if source_manifest_path.exists() else None
    if source_manifest:
        market_qc["invalid_ohlc_rows_removed_before_final_csv"] = source_manifest.get("invalid_ohlc_rows")
        market_qc["rows_before_invalid_ohlc_filter"] = source_manifest.get("rows_before_invalid_exclusion")
    if market_qc["duplicate_dates"] or not market_qc["sorted"]:
        raise ValueError(f"price ordering/duplicates failed: {market_qc}")
    atr, sma200, feats = market_features(market)
    idx_by_date = {d: i for i, d in enumerate(market["dates"])}
    trades = read_jsonl(args.run / "trades.jsonl")
    mtm_rows = read_jsonl(args.run / "daily_mtm.jsonl")
    state_rows = read_jsonl(args.run / "daily_states.jsonl")
    final_state = read_json(args.run / "final_state.json")
    strategy_equity = [float(x["mtm_pnl_usd"]) for x in mtm_rows]
    strategy_dd = [float(x["close_to_close_drawdown_usd"]) for x in mtm_rows]
    strategy_exposure = [bool(x.get("position_id")) for x in state_rows]
    if len(strategy_equity) != len(market["dates"]) or len(state_rows) != len(market["dates"]):
        raise ValueError(f"replay row count mismatch: market={len(market['dates'])}, mtm={len(strategy_equity)}, states={len(state_rows)}")
    if [str(x.get("datetime"))[:10] for x in mtm_rows] != market["dates"]:
        raise ValueError("daily MTM dates do not exactly match frozen price input")
    if [str(x.get("datetime"))[:10] for x in state_rows] != market["dates"]:
        raise ValueError("daily state dates do not exactly match frozen price input")
    bh_equity = benchmark_buy_hold(market)
    sma = run_sma200(market, sma200)

    attribution = trade_attribution(args.run, market, idx_by_date, final_state)
    event_summary, event_pairs = matched_event_study(args.run, market, atr, feats)
    top_exit = top_exit_event_study(trades, market, atr, idx_by_date)
    windows, annual = year_and_window_summaries(market, strategy_equity, bh_equity, sma,
                                                strategy_exposure, sma["exposure"])

    execution = read_jsonl(args.run / "execution_events.jsonl")
    buy_fills = [x for x in execution if x.get("action") == "buy_fill"]
    sell_fills = [x for x in execution if x.get("action") == "sell_fill"]
    result = {
        "status": "completed_research_audit",
        "strategy_frozen": "v0.11 rev11_without_continuation; no code/config/input fitting",
        "run_metadata_status": meta.get("run_status"),
        "run_metadata_verification_status": meta.get("verification_status"),
        "scope_note": "Retrospective fixed-rule analysis of already inspected history; not an untouched holdout, not an account return, not live performance.",
        "data": {**market_qc, "price_sha256": input_integrity.get("prices", {}).get("actual_sha256"),
                 "input_integrity": input_integrity,
                 "source_manifest": source_manifest},
        "v011": {"final_mtm_usd": strategy_equity[-1], "closed_trade_realized_usd": sum(float(t["net_pnl_usd"]) for t in trades),
                 "maximum_daily_close_drawdown_usd": min(strategy_dd),
                 "position_exposure_fraction": sum(strategy_exposure) / len(strategy_exposure),
                 "position_exposure_days": sum(strategy_exposure), "buy_fills": len(buy_fills),
                 "sell_fills": len(sell_fills), "open_at_end": bool(final_state.get("open_position")),
                 "bar_count": len(strategy_equity)},
        "buy_hold": {"final_mtm_usd": bh_equity[-1], "maximum_daily_close_drawdown_usd": drawdown(bh_equity),
                     "initial_fill_rule": "first observed open + 1 tick; 1 contract; $2.50 entry fee; mark to daily close"},
        "sma200_price_only": {k: v for k, v in sma.items() if k not in {"equity", "exposure", "trades"}},
        "trade_attribution": {k: v for k, v in attribution.items() if k != "trade_records"},
        "matched_ob_event_study": event_summary,
        "executed_top_exit_forward_path": {k: v for k, v in top_exit.items() if k != "rows"},
        "periods": windows,
        "annual": annual,
        "limitations": [
            "NKD=F continuous series is not individual quarterly contracts; source manifest identifies Yahoo continuous-proxy shards. Raw dated NQ contract chain/roll prices are not present in the local files scanned for this validation, so actual roll P&L, roll slippage, margin and capital utilization cannot be reconstructed.",
            "All dates and anchor examples have already informed strategy development; historical temporal splits are robustness diagnostics, not a clean out-of-sample test.",
            "200-day SMA is a single fixed, price-only reference strategy, not a matched entry/exit ablation of v0.11. Different exposure and drawdown path are expected.",
            "Matched event control estimates conditional post-signal association after six observable pre-event market-state variables; it does not prove OB causation, and its small event-date population is clustered by year.",
            "OHLC cannot order same-day highs/lows; dual barrier touches are reported as ambiguous, not assigned favorable order.",
            "Dollar P&L for one continuous-series contract is not ROI on account equity; margin collateral is not included as invested capital.",
            "Final position and any future path after 2026-08-27 are right-censored in this dataset.",
        ],
    }
    # Include source/run artifacts in the frozen manifest so the analysis can be audited later.
    frozen = {"metadata_sha256": sha256(args.run / "metadata.json"),
              "artifact_hashes": {name: sha256(args.run / name) for name in
                                  ("trades.jsonl", "daily_mtm.jsonl", "daily_states.jsonl", "entry_audit.jsonl",
                                   "independent_early_candidates.jsonl", "independent_confirmed_candidates.jsonl",
                                   "major_sell_candidates.jsonl")},
              "config_sha256": sha256(PACKAGE / "config.json"),
              "research_script_sha256": sha256(Path(__file__))}
    result["frozen_artifacts"] = frozen

    write_json(args.output / "research_results.json", result)
    write_json(args.output / "v011_frozen_manifest.json", {"strategy_config": meta.get("campaign_config"),
                                                             "legacy_rules": meta.get("legacy_rules"),
                                                             "inputs": input_integrity, "artifacts": frozen})
    write_csv(args.output / "trade_attribution.csv", attribution["trade_records"])
    write_csv(args.output / "matched_ob_event_study.csv", event_pairs)
    write_csv(args.output / "top_exit_forward_events.csv", top_exit["rows"])
    write_csv(args.output / "annual_results.csv", annual)
    write_csv(args.output / "period_results.csv", windows)
    # Reproducible equity curves for charting and future review.
    write_csv(args.output / "equity_curves.csv", [
        {"datetime": market["dates"][i], "v011_mtm_usd": strategy_equity[i], "buy_hold_mtm_usd": bh_equity[i],
         "sma200_mtm_usd": sma["equity"][i], "v011_in_market": int(strategy_exposure[i]),
         "sma200_in_market": int(sma["exposure"][i])}
        for i in range(len(market["dates"]))])

    report = render_report(result)
    (args.output / "report_ja.md").write_text(report, encoding="utf-8")
    print(json.dumps({"output": str(args.output), "v011": result["v011"],
                      "buy_hold": result["buy_hold"], "sma200": result["sma200_price_only"],
                      "trade_attribution": result["trade_attribution"],
                      "matched_ob_event_study": event_summary,
                      "periods": windows}, ensure_ascii=False, indent=2))


def market_manifest_path(prices_path: Path) -> Path:
    return prices_path.with_name(prices_path.stem + "_manifest.json")


def render_report(r):
    t, bh, sma = r["v011"], r["buy_hold"], r["sma200_price_only"]
    a, ev = r["trade_attribution"], r["matched_ob_event_study"]
    ev_pilot = ev["specifications"]["exploratory_pilot_no_caliper"]
    ev_state = ev["specifications"]["decision_state_only"]
    ev_candle = ev["specifications"]["decision_state_plus_reaction_candle"]
    def format_ci(ci):
        return f"[{ci[0]:.2f}, {ci[1]:.2f}]" if ci and ci[0] is not None else "n/a"
    entry_labels = {"OB_reaction": "Bullish OB反応", "recovery_price_structure_no_OB": "OBなし・退出後構造回復"}
    exit_labels = {"campaign_double_top_early": "主要高値反転SELL", "initial_risk_stop": "初期ストップ"}
    entry_table = "\n".join(["| Entry経路 | 件数 | 勝ち | 純損益 |", "|---|---:|---:|---:|"] +
                            [f"| {entry_labels.get(k, k)} | {v['n']} | {v['wins']} | ${v['net_usd']:,.0f} |"
                             for k, v in a["net_by_entry_group"].items()])
    exit_table = "\n".join(["| Exit理由 | 件数 | 勝ち | 純損益 |", "|---|---:|---:|---:|"] +
                           [f"| {exit_labels.get(k, k)} | {v['n']} | {v['wins']} | ${v['net_usd']:,.0f} |"
                            for k, v in a["net_by_exit_reason"].items()])
    lines = [
        "# NKD=F v0.11 追加検証レポート（固定ルール・2026-09-24）", "",
        "## 結論", "",
        "v0.11はこの連続先物データ上でBuy & Holdを期末1枚損益では上回ったが、過去43件の完了取引のうち勝ちは16件で、上位5件が勝ち取引の利益の大半を占める。年次・イベント対照を加えても、ICT固有の予測力や将来再現性を証明したとは言えない。最も大きな未解決点は、OB信号が似た相場状態の非OB日より増分情報を与えるか、そして現行の多段ロジックの利益が少数の長期上昇トレードに依存していないかである。", "",
        "## 固定条件とデータ", "",
        f"- ルール: `{r['strategy_frozen']}`。再学習・閾値探索・設定変更なし。",
        f"- データ: {r['data']['start']}〜{r['data']['end']}、{r['data']['rows']:,}日足。重複日 {r['data']['duplicate_dates']}、昇順 {r['data']['sorted']}、最終CSVの不正OHLC {r['data']['invalid_ohlc_rows_after_source_filter']}、ゼロ出来高 {r['data']['zero_volume_rows']}。元統合時には不正OHLC {r['data'].get('invalid_ohlc_rows_removed_before_final_csv')}行を除外。",
        "- 取引日カレンダーは入力に実在する日付の集合であり、CME公式カレンダーとの照合・認証はしていない。",
        f"- 価格CSV SHA-256: `{r['data']['price_sha256']}`。入力ハッシュは全{len(r['data']['input_integrity'])}項目一致: {all(x['match'] for x in r['data']['input_integrity'].values())}。",
        "- 1枚、$20/ポイント、片道1ティックのスリッページ、片道$2.50手数料。売買判断は日足終値、約定は翌観測日の始値。",
        "- 既存の因果監査では、候補の利用可能時刻・価格参照・翌始値時計の違反は0件。これは実装時計の検査であり、エッジ検証ではない。", "",
        "## 取引成績の分解", "",
        f"- 完了 {a['completed_count']}件、勝ち {a['wins']}、負け {a['losses']}、勝率 {fmt(a['win_rate']*100,2)}%、純実現損益 ${a['net_completed_usd']:,.0f}、PF {fmt(a['profit_factor'],3)}。",
        f"- 取引損益の中央値 ${a['median_trade_usd']:,.0f}、平均net R {fmt(a['mean_net_r'],2)}、中央値net R {fmt(a['median_net_r'],2)}。平均と中央値の隔たりは利益分布の強い歪みを示す。",
        f"- 勝ち取引の粗利益合計 ${a['gross_profit_usd']:,.0f}。上位5取引の純損益合計 ${a['top5_net_usd']:,.0f}（全勝ち取引粗利益の {fmt(a['top5_share_of_gross_profit']*100,1)}%）。上位5件を除いた残り38件の純損益は ${a['remaining_38_net_usd']:,.0f}。",
        entry_table,
        "OB反応入口と、OBを使わない退出後構造リカバリー入口が混在する。",
        "",
        exit_table,
        "Exit理由ごとの損益差は、異なる含み益・市場局面で発生するため出口ルール単独の因果効果とはみなせない。",
        f"- 期末評価額 ${t['final_mtm_usd']:,.0f}、最大終値DD ${t['maximum_daily_close_drawdown_usd']:,.0f}、ポジション保有日率 {fmt(t['position_exposure_fraction']*100,1)}%、期末未決済ポジションあり。",
        f"- Buy & Hold: 期末 ${bh['final_mtm_usd']:,.0f}、最大終値DD ${bh['maximum_daily_close_drawdown_usd']:,.0f}。",
        f"- 固定200日SMAクロスの価格専用参照: 期末 ${sma['final_mtm_usd']:,.0f}、最大DD ${sma['max_close_drawdown_usd']:,.0f}、完了取引 {sma['completed_trades']}件、保有率 {fmt(sma.get('exposure_fraction', None),3)}。これは一致頻度・同一エントリーの対照ではない。", "",
        "## Bullish OB候補日と非候補日のマッチ比較", "",
        f"- 母集団: 独立OB候補ログのbullish OB（1D/1W/1M）をzone-touch単位で重複除去し、同日複数候補を一つにまとめ、20営業日窓が重ならない最初の候補日を採用。候補 {ev['candidate_counts']['raw_zone_touch_rows_after_route_merge']:,}、ユニーク候補日 {ev['candidate_counts']['unique_candidate_dates']:,}、非重複イベント {ev['candidate_counts']['nonoverlap_event_dates']:,}。",
        f"- **探索過程も記録:** 初回は距離上限なしで{ev_pilot['matched_pairs']}組、平均差 {fmt(ev_pilot['paired_terminal_return_diff_atr_mean'])} ATR、年ブロック95%区間 {format_ci(ev_pilot['paired_terminal_return_diff_atr_95pct_year_block_bootstrap_ci'])}、距離RMS中央値/P90 {fmt(ev_pilot['match_distance_rms_median'])}/{fmt(ev_pilot['match_distance_rms_p90'])}。距離の遠い対照も混ざるため、結果を見た後にRMS caliper 2.5を追加した。これは方法選択が結果を見た後に変わったことを意味するので、初回区間は確認的証拠として扱わない。",
        f"- 最終の状態マッチは、同じ暦年の非候補日を1件・再使用なしで割当て、6特徴（直近5/20日リターン、ATR比率、20日変動率、63日レンジ位置、200日SMA乖離）を揃えた。候補186日中{ev_state['matched_pairs']}ペア、非対応 {ev_state['unmatched_no_controls'] + ev_state['unmatched_caliper'] + ev_state['unmatched_label']}。RMS距離中央値 {fmt(ev_state['match_distance_rms_median'])}、年クラスター {ev_state['distinct_calendar_years']}。",
        f"- 20日終値リターン平均: OB候補日 {fmt(ev_state['event_terminal_return_atr'])} ATR、対照日 {fmt(ev_state['control_terminal_return_atr'])} ATR。ペア差 {fmt(ev_state['paired_terminal_return_diff_atr_mean'])} ATR（中央値 {fmt(ev_state['paired_terminal_return_diff_atr_median'])}、年ブロック95%区間 {format_ci(ev_state['paired_terminal_return_diff_atr_95pct_year_block_bootstrap_ci'])}).",
        f"- 頑健性:反応足の騰落・レンジ・終値位置・下ヒゲも加えた {ev_candle['matched_pairs']} ペアでは平均差 {fmt(ev_candle['paired_terminal_return_diff_atr_mean'])} ATR、年ブロック95%区間 {format_ci(ev_candle['paired_terminal_return_diff_atr_95pct_year_block_bootstrap_ci'])}。",
        f"- +1ATR/-1ATR先着率（20日、同日両到達は曖昧）: 状態マッチ後は候補 {ev_state['event_barrier_outcomes']['up_first']['n']}/{ev_state['matched_pairs']} ({fmt(ev_state['event_barrier_outcomes']['up_first']['rate_all_pairs']*100,1)}%)、対照 {ev_state['control_barrier_outcomes']['up_first']['n']}/{ev_state['matched_pairs']} ({fmt(ev_state['control_barrier_outcomes']['up_first']['rate_all_pairs']*100,1)}%)。反応足形を含む感度分析は候補 {ev_candle['event_barrier_outcomes']['up_first']['n']}/{ev_candle['matched_pairs']} ({fmt(ev_candle['event_barrier_outcomes']['up_first']['rate_all_pairs']*100,1)}%)、対照 {ev_candle['control_barrier_outcomes']['up_first']['n']}/{ev_candle['matched_pairs']} ({fmt(ev_candle['control_barrier_outcomes']['up_first']['rate_all_pairs']*100,1)}%)。これは反応確認後の20日追随方向であり、帯到達時の反発率ではない。",
        "- 解釈:ポイント推定は候補有利でなく、厳格マッチの95%区間はゼロを含む。ただしマッチ率が低く、選択過程を結果確認後に修正しているため、この検定からOB無効とも優位とも決められない。既存OBへの全到達（非反発・無効化例を含む）を検証したものとは別である。", "",
        "## 期間別の固定ルール比較", "",
        "| 期間 | v0.11損益差 | Buy & Hold損益差 | 差(v0.11−B&H) | 200SMA損益差 | v0.11保有率 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for w in r["periods"]:
        lines.append(f"| {w['start']}–{w['end']} | ${w['v011_change_usd']:,.0f} | ${w['buy_hold_change_usd']:,.0f} | ${w['v011_minus_buy_hold_usd']:,.0f} | ${w['sma200_change_usd']:,.0f} | {w['v011_exposure_fraction']*100:.1f}% |")
    lines += ["", "部分期間は開始日の終値から終了日の終値までで損益を測る（全期間は初日始値からの累積）。これにより、期間外の始値ギャップを片側だけに含めない。以前の期間比較と同じ基準では、2016–2018のBuy & Holdは+$36,575、2023以降は+$371,645となる。年ごとの損益・保有率は `annual_results.csv` を参照。年別はレジーム依存性を見る診断であり、年を独立標本として有意差を主張するものではない。", "",
              "## 天井SELL後の経路", "",
        f"天井SELL後20セッションが揃う{r['executed_top_exit_forward_path']['executed_top_exit_events_with_full_20_session_horizon']}件では、SELL約定値から20日後の終値変化は平均 {fmt(r['executed_top_exit_forward_path']['mean_forward_20_close_return_atr'])} ATR、中央値 {fmt(r['executed_top_exit_forward_path']['median_forward_20_close_return_atr'])} ATR。20日後終値がSELL価格未満 {r['executed_top_exit_forward_path']['forward_20_close_below_exit_count']}件、上回る {r['executed_top_exit_forward_path']['forward_20_close_above_exit_count']}件。平均最大逆行 {fmt(r['executed_top_exit_forward_path']['mean_forward_20_max_down_atr'])} ATR、平均最大順行 {fmt(r['executed_top_exit_forward_path']['mean_forward_20_max_up_atr'])} ATR。平均/中央値の符号が違うため、裾の影響が大きい。これはSELL後の記述であり、SELLしなかった場合の反実仮想損益ではない。詳細は `top_exit_forward_events.csv`。", "",
              "## 時系列検証・頑健性の判定", "",
              "- 年別・期間別の固定ルール再生は行ったが、過去データと特定局面はモデル設計に繰り返し使われている。したがって、これはウォークフォワードの未見テストではなく、時間分割した頑健性診断である。",
              "- 完全に独立した成績判定には、2026-08-27以降の新しい日足を使う前向きペーパートレードが必要。ルール・入力・予測ログ・停止条件を先に固定し、数十件の独立シグナルが蓄積するまで成果判定を保留する。",
              "- v0.11の4条件比較は機能のOFF/ON比較であり、OB入口の置換やSELL出口の置換を同一の候補・同一リスク時計で行う厳密なアブレーションではない。200SMAは非ICT戦略参照にすぎず、この不足は残る。", "",
              "## 限月・資本・データ制約", "",
              "ローカルの入力はYahoo Finance `NKD=F`連続先物代理データの統合系列で、個別限月契約の同時期価格・出来高・ロール約定記録はこの検証対象ディレクトリ内で見つからなかった。そのため実際の限月ロール損益、ロールスリッページ、証拠金推移、追証・口座破綻制約は評価不能。1枚あたりの連続系列USD損益を、口座資産率や売買可能性と読み替えてはならない。", "",
              "## 成果物", "",
              "- `research_results.json`: 結果・入力ハッシュ・限界",
              "- `v011_frozen_manifest.json`: v0.11設定と入力/ログの固定ハッシュ",
              "- `trade_attribution.csv`: 取引別分解",
              "- `matched_ob_event_study.csv`: OBイベントとマッチ対照の日付・ラベル",
              "- `top_exit_forward_events.csv`: 実行天井SELL後の前向き経路",
              "- `period_results.csv`, `annual_results.csv`: 期間別成績",
              "- `equity_curves.csv`: v0.11/B&H/200SMAの時系列MTM曲線",
              "- `validate_v011_research_20260924.py`: 再実行スクリプト", "",
              "## 最終評価", "",
              "現時点で言えるのは、方法論上の時刻管理を監査できたことと、既知の過去区間では1枚固定損益が一定の成績を示したことまでである。ICT由来のOB/ダブルトップに増分予測力があり、将来もBuy & Holdを上回るという結論は保留する。マッチ仕様はすべて探索的で、期間分割も未見ではない。出口の反実仮想と新規データでの固定前向き評価が次の判定ゲートである。", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
