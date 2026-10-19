"""Independent audit for the revision 0.8 campaign replay.

This audit checks the replay contract, fixed-point accounting, causal clocks,
the new thesis-wave manager, exit-reason re-entry gate, and continuation risk
cap.  It does not fit parameters and it does not claim out-of-sample
performance.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

POINT_VALUE_USD = 20.0
TICK_SIZE = 0.25
SLIPPAGE_TICKS = 1.0
COMMISSION_PER_SIDE_USD = 2.50


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def input_audit(path: Path):
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig", newline="")))
    dates, seen, duplicate, missing, impossible, negative = [], set(), [], [], [], []
    for row in rows:
        date = row.get("datetime")
        if not date:
            missing.append("datetime")
            continue
        if date in seen:
            duplicate.append(date)
        seen.add(date)
        dates.append(date)
        try:
            o, h, low, c = (float(row[k]) for k in ("open", "high", "low", "close"))
            volume = float(row["volume"])
            if not (low <= min(o, c) <= max(o, c) <= h):
                impossible.append(date)
            if volume < 0:
                negative.append(date)
        except (KeyError, TypeError, ValueError):
            missing.append(date)
    return {
        "rows": len(rows),
        "first": dates[0] if dates else None,
        "last": dates[-1] if dates else None,
        "sorted_unique_dates": dates == sorted(set(dates)),
        "duplicate_date_count": len(duplicate),
        "missing_or_non_numeric_count": len(missing),
        "impossible_ohlc_count": len(impossible),
        "negative_volume_count": len(negative),
    }


def load_price_frame(path: Path):
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig", newline="")))
    return [(str(row["datetime"])[:10], float(row["open"]), float(row["high"]),
             float(row["low"]), float(row["close"])) for row in rows]


def drawdown(values):
    peak = -math.inf
    result = []
    for value in values:
        peak = max(peak, value)
        result.append(value - peak)
    return result


def accounting_audit(output: Path):
    trades = read_jsonl(output / "trades.jsonl")
    mismatches, pnl = [], []
    for trade in trades:
        quantity = int(trade.get("quantity", 1))
        expected = ((float(trade["exit_price"]) - float(trade["entry_price"]))
                    * quantity * POINT_VALUE_USD - 2.0 * quantity * COMMISSION_PER_SIDE_USD)
        actual = float(trade["net_pnl_usd"])
        pnl.append(actual)
        if abs(expected - actual) > 1e-7:
            mismatches.append({"position_id": trade.get("position_id"),
                               "expected": expected, "actual": actual})
    mtm = read_jsonl(output / "daily_mtm.jsonl")
    mtm_values = [float(row["mtm_pnl_usd"]) for row in mtm]
    execution = read_jsonl(output / "execution_events.jsonl")
    buy_fills = [row for row in execution if row.get("action") == "buy_fill"]
    sell_fills = [row for row in execution if row.get("action") == "sell_fill"]
    stops = [row for row in execution if row.get("action") == "intrabar_emergency_stop"]
    wins = [value for value in pnl if value > 0]
    losses = [value for value in pnl if value < 0]
    family_performance = {}
    for family in sorted({trade.get("entry_confirmation") or "unknown" for trade in trades}):
        values = [float(trade["net_pnl_usd"]) for trade in trades
                  if (trade.get("entry_confirmation") or "unknown") == family]
        fw = [value for value in values if value > 0]
        fl = [value for value in values if value < 0]
        family_performance[family] = {
            "closed_trades": len(values), "wins": len(fw), "losses": len(fl),
            "win_rate": len(fw) / len(values) if values else None,
            "net_pnl_usd": sum(values),
            "profit_factor": sum(fw) / abs(sum(fl)) if fl else None,
        }
    return {
        "closed_trades": len(trades),
        "right_censored_open_attempts": sum(1 for row in read_jsonl(output / "gate_events.jsonl")
                                              if row.get("action") == "campaign_attempt_filled") -
                                        sum(1 for row in read_jsonl(output / "gate_events.jsonl")
                                            if row.get("action") == "campaign_attempt_closed"),
        "execution_buy_fill_count": len(buy_fills),
        "execution_sell_fill_count": len(sell_fills),
        "execution_emergency_stop_count": len(stops),
        "wins": len(wins), "losses": len(losses),
        "win_rate": len(wins) / len(pnl) if pnl else None,
        "realized_net_usd": sum(pnl), "gross_profit_usd": sum(wins),
        "gross_loss_usd": sum(losses),
        "profit_factor": sum(wins) / abs(sum(losses)) if losses else None,
        "family_performance": family_performance,
        "daily_mtm_rows": len(mtm),
        "final_mtm_usd": mtm_values[-1] if mtm_values else None,
        "mtm_max_drawdown_usd": min(drawdown(mtm_values)) if mtm_values else None,
        "accounting_mismatch_count": len(mismatches),
        "accounting_mismatches": mismatches[:20],
    }


def buy_hold_audit(output: Path, prices_path: Path):
    prices = load_price_frame(prices_path)
    mtm = read_jsonl(output / "daily_mtm.jsonl")
    if mtm:
        mtm_first = str(mtm[0]["datetime"])[:10]
        mtm_last = str(mtm[-1]["datetime"])[:10]
        prices = [row for row in prices if mtm_first <= row[0] <= mtm_last]
    entry_fill = prices[0][1] + SLIPPAGE_TICKS * TICK_SIZE
    values = [(close - entry_fill) * POINT_VALUE_USD - COMMISSION_PER_SIDE_USD
              for _, _, _, _, close in prices]
    strategy_values = [float(row["mtm_pnl_usd"]) for row in mtm]
    strategy_dates = [str(row["datetime"])[:10] for row in mtm]
    by_date = dict(zip(strategy_dates, strategy_values))
    windows = []
    first_date, last_date = prices[0][0], prices[-1][0]
    for start, end in (("2016-01-01", "2018-12-31"), ("2023-01-01", "2026-08-27")):
        if end < first_date or start > last_date:
            continue
        start = max(start, first_date)
        end = min(end, last_date)
        i0 = next(i for i, row in enumerate(prices) if row[0] >= start)
        i1 = max(i for i, row in enumerate(prices) if row[0] <= end)
        s0, s1 = by_date[prices[i0][0]], by_date[prices[i1][0]]
        bh = values[i1] - values[i0]
        windows.append({"start": prices[i0][0], "end": prices[i1][0],
                        "strategy_mtm_change_usd": s1 - s0,
                        "buy_hold_mtm_change_usd": bh,
                        "strategy_vs_buy_hold_change_usd": s1 - s0 - bh})
    return {
        "entry_fill": entry_fill,
        "final_mtm_usd": values[-1],
        "max_drawdown_usd": min(drawdown(values)),
        "strategy_final_minus_buy_hold_usd": strategy_values[-1] - values[-1],
        "strategy_max_dd_improvement_usd": min(drawdown(strategy_values)) - min(drawdown(values)),
        "windows": windows,
    }


def causal_audit(output: Path, metadata: dict):
    failures = []
    contexts = read_json(output / "entry_structural_contexts.json")
    for order_id, context in contexts.items():
        role = context.get("role") or {}
        contact = context.get("contact_index")
        if not role or contact is None:
            failures.append({"type": "role_or_contact_missing", "order_id": order_id})
            continue
        if not (role.get("known_index", math.inf) < contact
                and role.get("effective_index", math.inf) <= contact):
            failures.append({"type": "role_not_precontact", "order_id": order_id,
                             "known_index": role.get("known_index"),
                             "effective_index": role.get("effective_index"),
                             "contact_index": contact})

    selections = read_jsonl(output / "support_selection.jsonl")
    selected = [row for row in selections if row.get("selected")]
    selected_failed = [row.get("key") for row in selected if
                       any(value.get("passed") is not True
                           for value in row.get("checks", {}).values())]
    risk_failures, unlimited_failures = [], []
    for row in selected:
        checks = row.get("checks", {})
        budget = checks.get("execution_budget", {}).get("components", {})
        for name in ("total_attempts", "cumulative_loss", "route_attempts"):
            check = budget.get(name, {})
            if check.get("passed") is not True:
                unlimited_failures.append({"key": row.get("key"), "gate": name, "check": check})
        risk = row.get("entry_risk") or {}
        if risk.get("mode") == "reaction":
            width = risk.get("risk_at_decision_atr")
            cap = risk.get("maximum_risk_atr")
            if width is None or cap is None or float(width) > float(cap) + 1e-9:
                risk_failures.append({"key": row.get("key"), "risk": risk})

    continuation = read_jsonl(output / "continuation_candidates.jsonl")
    continuation_failures = []
    for row in continuation:
        zone = row.get("zone") or {}
        touch = row.get("touch_index")
        if touch is None or zone.get("available_index") is None:
            continuation_failures.append({"key": row.get("key"), "type": "missing_zone_clock"})
        elif zone["available_index"] >= touch:
            continuation_failures.append({"key": row.get("key"), "type": "zone_not_precontact",
                                          "available_index": zone.get("available_index"),
                                          "touch_index": touch})

    wave_events = read_jsonl(output / "wave_management_events.jsonl")
    wave_failures = []
    for event in wave_events:
        if event.get("action") != "wave_confirmed":
            continue
        record = event.get("record") or {}
        current = event.get("index")
        effective = record.get("effective_from_index")
        pivot = record.get("pivot") or {}
        if current is None or effective is None or effective <= current:
            wave_failures.append({"type": "wave_effective_clock", "event": event})
        if pivot.get("available_index") is None or pivot["available_index"] > current:
            wave_failures.append({"type": "wave_pivot_not_confirmed", "event": event})
    wave_effective = [event for event in wave_events if event.get("action") == "wave_effective"]
    wave_effective_failures = [event for event in wave_effective
                               if event.get("index", -1) < 0 or
                               event.get("record_id") is None]

    reentry_events = read_jsonl(output / "exit_reason_reentry_events.jsonl")
    reentry_failures = []
    exit_indices = {}
    for event in reentry_events:
        if event.get("action") == "executed_exit_recorded":
            context = event.get("context") or {}
            context_id = context.get("id")
            if context_id:
                exit_indices[context_id] = context.get("index")
        if event.get("action") == "exit_reclaim_state_changed":
            context_id = event.get("context_id")
            # The ledger evaluates close-time state and can execute the same
            # bar's exit later in that close cycle.  Equality is therefore
            # same-decision-time, not a pre-exit future leak; only an event
            # strictly before the actual exit is invalid.
            if context_id in exit_indices and event.get("index", -1) < exit_indices[context_id]:
                reentry_failures.append({"type": "reclaim_before_exit", "event": event})

    entry_assessed = [row for row in read_jsonl(output / "entry_audit.jsonl")
                      if row.get("action") == "entry_assessed"]
    trade_clock_failures = []
    for trade in read_jsonl(output / "trades.jsonl"):
        thesis = trade.get("entry_structural_context", {}).get("entry_thesis") or {}
        decision_at = thesis.get("decision_at")
        entry_at = trade.get("entry_at")
        if decision_at and entry_at and str(entry_at)[:10] <= str(decision_at)[:10]:
            trade_clock_failures.append({"position_id": trade.get("position_id"),
                                         "decision_at": decision_at, "entry_at": entry_at})
    return {
        "entry_context_count": len(contexts),
        "entry_context_failure_count": len(failures),
        "entry_context_failures": failures[:20],
        "selected_support_count": len(selected),
        "selected_entry_family_counts": dict(Counter(row.get("entry_family") for row in selected)),
        "selected_with_failed_gate_count": len(selected_failed),
        "unlimited_gate_failure_count": len(unlimited_failures),
        "unlimited_gate_failures": unlimited_failures[:20],
        "continuation_candidate_count": len(continuation),
        "continuation_candidate_clock_failure_count": len(continuation_failures),
        "continuation_candidate_clock_failures": continuation_failures[:20],
        "continuation_risk_failure_count": len(risk_failures),
        "continuation_risk_failures": risk_failures[:20],
        "wave_confirmed_count": sum(1 for event in wave_events if event.get("action") == "wave_confirmed"),
        "wave_effective_count": len(wave_effective),
        "wave_clock_failure_count": len(wave_failures) + len(wave_effective_failures),
        "wave_clock_failures": (wave_failures + wave_effective_failures)[:20],
        "exit_reentry_event_count": len(reentry_events),
        "exit_reentry_clock_failure_count": len(reentry_failures),
        "exit_reentry_clock_failures": reentry_failures[:20],
        "trade_clock_failure_count": len(trade_clock_failures),
        "trade_clock_failures": trade_clock_failures[:20],
        "entry_assessed_family_counts": dict(Counter(row.get("entry_family") for row in entry_assessed)),
        "config": {key: metadata.get("campaign_config", {}).get(key) for key in (
            "management_mode", "exit_reentry_policy", "continuation_context_mode",
            "continuation_risk_mode", "continuation_max_risk_atr")},
    }


def recent_trades(output: Path, count=10):
    return [{"entry_at": str(row.get("entry_at", ""))[:10],
             "exit_at": str(row.get("exit_at", ""))[:10],
             "entry_family": row.get("entry_family") or row.get("entry_confirmation"),
             "reason": row.get("reason"), "net_pnl_usd": row.get("net_pnl_usd")}
            for row in read_jsonl(output / "trades.jsonl")[-count:]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prices", type=Path,
                        default=Path("nq_25y_actual_2000_2026_20260911/NQ_F_1d_combined.csv"))
    args = parser.parse_args()
    output = args.output
    metadata = read_json(output / "metadata.json")
    quality = input_audit(args.prices)
    accounting = accounting_audit(output)
    benchmark = buy_hold_audit(output, args.prices)
    causal = causal_audit(output, metadata)
    status = ("PASS_WITH_RESEARCH_CAVEATS"
              if metadata.get("implementation") == "nq_ict_campaign_proposal_0.8"
              and metadata.get("run_status") == "completed_replay_only"
              and quality["sorted_unique_dates"]
              and causal["entry_context_failure_count"] == 0
              and causal["selected_with_failed_gate_count"] == 0
              and causal["unlimited_gate_failure_count"] == 0
              and causal["continuation_candidate_clock_failure_count"] == 0
              and causal["continuation_risk_failure_count"] == 0
              and causal["wave_clock_failure_count"] == 0
              and causal["exit_reentry_clock_failure_count"] == 0
              and causal["trade_clock_failure_count"] == 0
              and accounting["accounting_mismatch_count"] == 0
              else "NEEDS_REVIEW")
    result = {"audit_status": status,
              "metadata_summary": {key: metadata.get(key) for key in (
                  "implementation", "run_status", "daily_rows", "as_of", "start", "end")},
              "input_quality": quality, "accounting": accounting,
              "buy_hold_comparison": benchmark,
              "causal_management_and_continuation": causal,
              "recent_trades": recent_trades(output)}
    (output / "audit_rev08.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = [
        "# 改訂0.8検証結果（ES=F）", "",
        f"状態: **{status}**。全期間リプレイ、独立会計照合、因果時計監査を完了した。",
        "これは実装・再現性の監査であり、未見期間の予測性能や実運用口座の優位性を証明しない。", "",
        f"入力: {metadata.get('start')}〜{metadata.get('end')}、{metadata.get('daily_rows')}日足。",
        f"実行: `{metadata.get('implementation')}`、`{metadata.get('run_status')}`。", "",
        "## 損益・会計", "",
        f"- 完了取引: {accounting['closed_trades']}件 / BUY約定: {accounting['execution_buy_fill_count']}件 / 終端未決済: {accounting['right_censored_open_attempts']}件。",
        f"- 勝ち: {accounting['wins']}件、負け: {accounting['losses']}件、勝率: {accounting['win_rate']:.2%}。",
        f"- 実現損益: {accounting['realized_net_usd']:,.0f} USD、PF: {accounting['profit_factor']:.3f}。",
        f"- 改訂0.8最終MTM: {accounting['final_mtm_usd']:,.1f} USD、最大DD: {accounting['mtm_max_drawdown_usd']:,.1f} USD。",
        f"- Buy & Hold最終MTM: {benchmark['final_mtm_usd']:,.1f} USD、最大DD: {benchmark['max_drawdown_usd']:,.1f} USD。",
        f"- 改訂0.8−Buy & Hold: {benchmark['strategy_final_minus_buy_hold_usd']:,.1f} USD。",
        f"- 最大DD改善幅（絶対値）: {benchmark['strategy_max_dd_improvement_usd']:,.1f} USD。",
        f"- 独立損益再計算不一致: {accounting['accounting_mismatch_count']}件。", "",
        "## 改訂0.8の追加機能監査", "",
        f"- 選択支持候補: {causal['selected_support_count']}件（reversal {causal['selected_entry_family_counts'].get('reversal', 0)} / continuation {causal['selected_entry_family_counts'].get('continuation', 0)}）。",
        f"- 継続候補: {causal['continuation_candidate_count']}件、リスク上限違反: {causal['continuation_risk_failure_count']}件。",
        f"- 波の確認: confirmed {causal['wave_confirmed_count']}件 / effective {causal['wave_effective_count']}件 / 時計違反 {causal['wave_clock_failure_count']}件。",
        f"- Exit-reason再BUYイベント: {causal['exit_reentry_event_count']}件 / 時計違反 {causal['exit_reentry_clock_failure_count']}件。",
        f"- 取引の決定日→翌観測日約定の違反: {causal['trade_clock_failure_count']}件。",
        f"- エントリー構造ロールの未来参照違反: {causal['entry_context_failure_count']}件。",
        f"- 選択後の失敗ゲート: {causal['selected_with_failed_gate_count']}件。", "",
        "## Buy & Holdとの差", "",
    ]
    for window in benchmark["windows"]:
        report.append(f"- {window['start']}〜{window['end']}: 改訂0.8 {window['strategy_mtm_change_usd']:,.1f} USD / Buy & Hold {window['buy_hold_mtm_change_usd']:,.1f} USD / 差 {window['strategy_vs_buy_hold_change_usd']:,.1f} USD。")
    report += ["", "## 系統別（完了取引）", ""]
    for family, stats in accounting["family_performance"].items():
        pf = f"PF {stats['profit_factor']:.3f}" if stats["profit_factor"] is not None else "PF n/a"
        report.append(f"- `{family}`: {stats['closed_trades']}件、勝率 {stats['win_rate']:.2%}、損益 {stats['net_pnl_usd']:,.0f} USD、{pf}。")
    report += ["", "## 直近取引", "", "```json", json.dumps(result["recent_trades"], ensure_ascii=False, indent=2), "```", "",
               "## 注意", "",
               "- これは同一履歴へのリプレイ結果であり、未見ウォークフォワードではない。",
               "- ES=F連続先物のロール、証拠金、口座資産制約、資金調達費、日足内の高安順序は評価していない。",
               "- 1枚固定のUSD損益であり、口座資産倍率ではない。",
               "- 閾値は未フィットの設計値であり、成績に合わせて再調整していない。"]
    (output / "audit_rev08_report_ja.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
