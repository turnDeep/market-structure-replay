"""Independent audit for revision 0.7 replay outputs.

This checks implementation/accounting/causal-clock integrity for the revised
unlimited-attempt plus continuation-buy campaign.  It is not a parameter
search and it is not a claim of out-of-sample trading performance.
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
    return [(str(row["datetime"])[:10], float(row["open"]), float(row["close"])) for row in rows]


def drawdown(values):
    peak = -math.inf
    out = []
    for value in values:
        peak = max(peak, value)
        out.append(value - peak)
    return out


def accounting_audit(output: Path, prices_path: Path, metadata: dict):
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
    mtm_dd = drawdown(mtm_values)
    fills = [row for row in read_jsonl(output / "gate_events.jsonl")
             if row.get("action") == "campaign_attempt_filled"]
    closes = [row for row in read_jsonl(output / "gate_events.jsonl")
              if row.get("action") == "campaign_attempt_closed"]
    execution = read_jsonl(output / "execution_events.jsonl")
    buy_fills = [row for row in execution if row.get("action") == "buy_fill"]
    sell_fills = [row for row in execution if row.get("action") == "sell_fill"]
    emergency_stops = [row for row in execution if row.get("action") == "intrabar_emergency_stop"]
    wins = [value for value in pnl if value > 0]
    losses = [value for value in pnl if value < 0]
    family_performance = {}
    for family in sorted({trade.get("entry_confirmation") or "unknown" for trade in trades}):
        family_values = [float(trade["net_pnl_usd"]) for trade in trades
                         if (trade.get("entry_confirmation") or "unknown") == family]
        family_wins = [value for value in family_values if value > 0]
        family_losses = [value for value in family_values if value < 0]
        family_performance[family] = {
            "closed_trades": len(family_values),
            "wins": len(family_wins),
            "losses": len(family_losses),
            "win_rate": len(family_wins) / len(family_values) if family_values else None,
            "net_pnl_usd": sum(family_values),
            "profit_factor": (sum(family_wins) / abs(sum(family_losses))) if family_losses else None,
        }
    return {
        "closed_trades": len(trades),
        "filled_attempts": len(fills),
        "right_censored_open_attempts": len(fills) - len(closes),
        "execution_buy_fill_count": len(buy_fills),
        "execution_sell_fill_count": len(sell_fills),
        "execution_emergency_stop_count": len(emergency_stops),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / len(pnl) if pnl else None,
        "realized_net_usd": sum(pnl),
        "gross_profit_usd": sum(wins),
        "gross_loss_usd": sum(losses),
        "profit_factor": sum(wins) / abs(sum(losses)) if losses else None,
        "family_performance": family_performance,
        "daily_mtm_rows": len(mtm),
        "final_mtm_usd": mtm_values[-1] if mtm_values else None,
        "mtm_max_drawdown_usd": min(mtm_dd) if mtm_dd else None,
        "accounting_mismatch_count": len(mismatches),
        "accounting_mismatches": mismatches[:20],
    }


def buy_hold_audit(output: Path, prices_path: Path):
    prices = load_price_frame(prices_path)
    mtm = read_jsonl(output / "daily_mtm.jsonl")
    entry_fill = prices[0][1] + SLIPPAGE_TICKS * TICK_SIZE
    values = [(close - entry_fill) * POINT_VALUE_USD - COMMISSION_PER_SIDE_USD
              for _, _, close in prices]
    dd = drawdown(values)
    strategy_values = [float(row["mtm_pnl_usd"]) for row in mtm]
    strategy_dates = [str(row["datetime"])[:10] for row in mtm]
    by_date = dict(zip(strategy_dates, strategy_values))
    windows = []
    for start, end in (("2016-01-01", "2018-12-31"), ("2023-01-01", "2026-08-27")):
        i0 = next(i for i, row in enumerate(prices) if row[0] >= start)
        i1 = max(i for i, row in enumerate(prices) if row[0] <= end)
        s0 = by_date[prices[i0][0]]
        s1 = by_date[prices[i1][0]]
        windows.append({
            "start": prices[i0][0], "end": prices[i1][0],
            "strategy_mtm_change_usd": s1 - s0,
            "buy_hold_mtm_change_usd": values[i1] - values[i0],
            "strategy_vs_buy_hold_change_usd": (s1 - s0) - (values[i1] - values[i0]),
        })
    return {
        "entry_fill": entry_fill,
        "final_mtm_usd": values[-1],
        "max_drawdown_usd": min(dd),
        "strategy_final_minus_buy_hold_usd": strategy_values[-1] - values[-1],
        "strategy_max_dd_improvement_usd": min(drawdown(strategy_values)) - min(dd),
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
    unlimited_failures = []
    for row in selected:
        checks = row.get("checks", {}).get("execution_budget", {}).get("components", {})
        for name in ("total_attempts", "cumulative_loss", "route_attempts"):
            check = checks.get(name, {})
            if check.get("passed") is not True:
                unlimited_failures.append({"key": row.get("key"), "gate": name, "check": check})

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

    entry_assessed = [row for row in read_jsonl(output / "entry_audit.jsonl")
                      if row.get("action") == "entry_assessed"]
    fam = Counter(row.get("entry_family") for row in entry_assessed)
    events = read_jsonl(output / "continuation_events.jsonl")
    event_counts = Counter(row.get("action") for row in events)
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
        "entry_assessed_family_counts": dict(fam),
        "continuation_event_counts": dict(event_counts),
        "config": {
            "attempt_allocation": metadata.get("campaign_config", {}).get("attempt_allocation"),
            "campaign_loss_limit_mode": metadata.get("campaign_config", {}).get("campaign_loss_limit_mode"),
            "reentry_evidence_mode": metadata.get("campaign_config", {}).get("reentry_evidence_mode"),
            "enable_continuation_buy": metadata.get("campaign_config", {}).get("enable_continuation_buy"),
        },
    }


def recent_trades(output: Path, count: int = 10):
    trades = read_jsonl(output / "trades.jsonl")
    return [{"entry_at": str(row.get("entry_at", ""))[:10],
             "exit_at": str(row.get("exit_at", ""))[:10],
             "entry_family": row.get("entry_family") or row.get("entry_confirmation"),
             "reason": row.get("reason"), "net_pnl_usd": row.get("net_pnl_usd")}
            for row in trades[-count:]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prices", type=Path,
                        default=Path("nq_25y_actual_2000_2026_20260911/NQ_F_1d_combined.csv"))
    args = parser.parse_args()
    output = args.output
    metadata = read_json(output / "metadata.json")
    quality = input_audit(args.prices)
    accounting = accounting_audit(output, args.prices, metadata)
    benchmark = buy_hold_audit(output, args.prices)
    causal = causal_audit(output, metadata)
    status = ("PASS_WITH_RESEARCH_CAVEATS"
              if metadata.get("run_status") == "completed_replay_only"
              and quality["sorted_unique_dates"]
              and causal["entry_context_failure_count"] == 0
              and causal["selected_with_failed_gate_count"] == 0
              and causal["unlimited_gate_failure_count"] == 0
              and causal["continuation_candidate_clock_failure_count"] == 0
              and accounting["accounting_mismatch_count"] == 0
              else "NEEDS_REVIEW")
    result = {
        "audit_status": status,
        "metadata_summary": {"run_status": metadata.get("run_status"),
                             "implementation": metadata.get("implementation"),
                             "daily_rows": metadata.get("daily_rows"),
                             "as_of": metadata.get("as_of"),
                             "start": metadata.get("start"),
                             "end": metadata.get("end")},
        "input_quality": quality,
        "accounting": accounting,
        "buy_hold_comparison": benchmark,
        "causal_and_continuation": causal,
        "recent_trades": recent_trades(output),
    }
    (output / "audit_rev07.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = [
        "# 改訂0.7検証結果（NKD=F）", "",
        f"状態: **{status}**。全期間リプレイ、独立会計照合、因果時計監査を完了した。",
        "この判定は実装と再現性の監査であり、未見期間における将来予測性能や実運用口座の優位性を証明しない。", "",
        f"入力: {metadata.get('start')}〜{metadata.get('end')}、{metadata.get('daily_rows')}日足。",
        f"実行: `{metadata.get('implementation')}`、`{metadata.get('run_status')}`。", "",
        "## 損益・会計", "",
        f"- 完了取引: {accounting['closed_trades']}件 / BUY約定: {accounting['execution_buy_fill_count']}件 / 終端未決済: {accounting['right_censored_open_attempts']}件。",
        f"- 勝ち: {accounting['wins']}件、負け: {accounting['losses']}件、勝率: {accounting['win_rate']:.2%}。",
        f"- 実現損益: {accounting['realized_net_usd']:,.0f} USD、PF: {accounting['profit_factor']:.3f}。",
        f"- 改訂0.7最終MTM: {accounting['final_mtm_usd']:,.1f} USD、最大DD: {accounting['mtm_max_drawdown_usd']:,.1f} USD。",
        f"- Buy & Hold最終MTM: {benchmark['final_mtm_usd']:,.1f} USD、最大DD: {benchmark['max_drawdown_usd']:,.1f} USD。",
        f"- 改訂0.7−Buy & Hold: {benchmark['strategy_final_minus_buy_hold_usd']:,.1f} USD。",
        f"- 最大DD改善幅（絶対値）: {benchmark['strategy_max_dd_improvement_usd']:,.1f} USD。",
        f"- 独立損益再計算不一致: {accounting['accounting_mismatch_count']}件。", "",
        "## エントリー系統別（完了取引のみ）", "",
    ]
    for family, stats in accounting["family_performance"].items():
        report.append(f"- `{family}`: {stats['closed_trades']}件、勝率 {stats['win_rate']:.2%}、損益 {stats['net_pnl_usd']:,.0f} USD、PF {stats['profit_factor']:.3f}。" if stats["profit_factor"] is not None else
                      f"- `{family}`: {stats['closed_trades']}件、勝率 {stats['win_rate']:.2%}、損益 {stats['net_pnl_usd']:,.0f} USD、PF n/a。")
    report += [
        "",
        "## 変更点の実装監査", "",
        f"- 選択された支持候補: {causal['selected_support_count']}件（reversal {causal['selected_entry_family_counts'].get('reversal', 0)} / continuation {causal['selected_entry_family_counts'].get('continuation', 0)}）。",
        f"- 継続BUY候補: {causal['continuation_candidate_count']}件。",
        f"- 継続BUY候補の価格時計違反: {causal['continuation_candidate_clock_failure_count']}件。",
        f"- 無制限化した購入回数・損失・経路ゲートの違反: {causal['unlimited_gate_failure_count']}件。",
        f"- 選択後の失敗ゲート: {causal['selected_with_failed_gate_count']}件。",
        f"- エントリー構造ロールの未来参照違反: {causal['entry_context_failure_count']}件。", "",
        "## 上昇を逃したかの確認", "",
    ]
    for window in benchmark["windows"]:
        report.append(f"- {window['start']}〜{window['end']}: 改訂0.7 {window['strategy_mtm_change_usd']:,.1f} USD / Buy & Hold {window['buy_hold_mtm_change_usd']:,.1f} USD / 差 {window['strategy_vs_buy_hold_change_usd']:,.1f} USD。")
    report += ["", "## 直近取引", "", "```json", json.dumps(result["recent_trades"], ensure_ascii=False, indent=2), "```", "",
               "## 解釈上の注意", "",
               "- 価格上昇を捕捉できた期間は、追加BUYだけでなく、既存LONGを保有していた期間も含む。保有中の候補を約定として数えてはいない。",
               "- NKD=F連続先物のロール、証拠金、口座資産制約、資金調達費、未見ウォークフォワード、日足内の高安順序は評価していない。",
               "- 1枚あたり固定USD損益であり、口座資産倍率ではない。",
               "- 変更後の成績差は履歴に適用した記述統計であり、ルール固定後の新規期間での予測性能とは分離する。"]
    (output / "audit_rev07_report_ja.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
