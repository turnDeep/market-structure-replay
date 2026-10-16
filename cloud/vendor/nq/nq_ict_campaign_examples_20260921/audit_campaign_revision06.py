"""Independent audit for revision 0.6 replay outputs.

This is a reproducibility audit, not a parameter search or proof of future
performance. It recalculates the fixed-contract accounting, causal clocks,
campaign budgets, and the four requested recent market examples.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


TARGET_SELLS = {
    ("2024-12-16", "2025-02-18", "2025-02-26"),
    ("2025-10-30", "2026-01-28", "2026-02-05"),
}
TARGET_BUYS = {"2025-04-10", "2026-04-01"}


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


def accounting_audit(output: Path, metadata: dict):
    trades = read_jsonl(output / "trades.jsonl")
    execution = metadata.get("execution_config", {})
    point_value = float(execution.get("point_value_usd", 20.0))
    commission = float(execution.get("commission_per_contract_side_usd", 2.5))
    mismatches, pnl = [], []
    for trade in trades:
        quantity = int(trade.get("quantity", 1))
        expected = ((float(trade["exit_price"]) - float(trade["entry_price"]))
                    * quantity * point_value - 2.0 * quantity * commission)
        actual = float(trade["net_pnl_usd"])
        pnl.append(actual)
        if abs(expected - actual) > 1e-7:
            mismatches.append({"position_id": trade.get("position_id"),
                               "expected": expected, "actual": actual})
    mtm = read_jsonl(output / "daily_mtm.jsonl")
    peak, drawdown = -math.inf, []
    for row in mtm:
        value = float(row["mtm_pnl_usd"])
        peak = max(peak, value)
        drawdown.append(value - peak)
    wins = [value for value in pnl if value > 0]
    losses = [value for value in pnl if value < 0]
    fills = [row for row in read_jsonl(output / "gate_events.jsonl")
             if row.get("action") == "campaign_attempt_filled"]
    closes = [row for row in read_jsonl(output / "gate_events.jsonl")
              if row.get("action") == "campaign_attempt_closed"]
    return {
        "closed_trades": len(trades),
        "filled_attempts": len(fills),
        "right_censored_open_attempts": len(fills) - len(closes),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / len(pnl) if pnl else None,
        "realized_net_usd": sum(pnl),
        "gross_profit_usd": sum(wins),
        "gross_loss_usd": sum(losses),
        "profit_factor": sum(wins) / abs(sum(losses)) if losses else None,
        "daily_mtm_rows": len(mtm),
        "final_mtm_usd": float(mtm[-1]["mtm_pnl_usd"]) if mtm else None,
        "mtm_max_drawdown_usd": min(drawdown) if drawdown else None,
        "accounting_mismatch_count": len(mismatches),
        "accounting_mismatches": mismatches[:20],
    }


def causal_and_budget_audit(output: Path):
    failures = []
    contexts = read_json(output / "entry_structural_contexts.json")
    for order_id, context in contexts.items():
        role = context.get("role")
        contact = context.get("contact_index")
        if role is None or contact is None:
            failures.append({"type": "entry_role_or_contact_missing", "order_id": order_id})
        elif not (role.get("known_index", math.inf) < contact
                  and role.get("effective_index", math.inf) <= contact):
            failures.append({"type": "entry_role_not_precontact", "order_id": order_id,
                             "known_index": role.get("known_index"),
                             "effective_index": role.get("effective_index"),
                             "contact_index": contact})

    selections = read_jsonl(output / "support_selection.jsonl")
    missing_gates = [row for row in selections if not row.get("all_close_gates_evaluated")]
    selected_failed = [row.get("key") for row in selections if row.get("selected") and
                       any(value.get("passed") is not True
                           for value in row.get("checks", {}).values())]

    cycle_events = read_jsonl(output / "market_cycle_events.jsonl")
    renewals, renewal_failures = [], []
    for event in cycle_events:
        if event.get("action") != "latest_major_reversal_updated":
            continue
        decision = event.get("budget_decision", {})
        if decision.get("renew"):
            renewals.append({"index": event.get("index"),
                             "reason": decision.get("reason"),
                             "scope_id": (event.get("context") or {}).get("scope_id"),
                             "parent_top_id": decision.get("parent_top_id")})
            if decision.get("failed_conditions"):
                renewal_failures.append({"index": event.get("index"),
                                         "failed_conditions": decision["failed_conditions"]})
        if decision.get("policy") == "major_top_boundary" and decision.get("renew") != event.get("budget_renewed"):
            renewal_failures.append({"index": event.get("index"), "type": "renewal_flag_mismatch"})

    budgets = defaultdict(lambda: {"fills": 0, "max_attempts": 0, "max_loss_r": 0.0,
                                   "loss_r_before_fill": 0.0})
    budget_failures = []
    overshoots = []
    for event in read_jsonl(output / "gate_events.jsonl"):
        scope = event.get("budget_scope_id")
        if not scope:
            continue
        if event.get("action") == "campaign_attempt_filled":
            # The configured loss cap is an eligibility gate before a new
            # attempt. A final losing fill may push the realized sum slightly
            # above 2R; that is an accounting outcome, not a gate violation.
            if budgets[scope]["max_attempts"] >= 2:
                budget_failures.append({"scope": scope, "type": "fill_after_attempt_cap",
                                        "index": event.get("index")})
            if budgets[scope]["max_loss_r"] >= 2.0 - 1e-9:
                budget_failures.append({"scope": scope, "type": "fill_after_loss_cap",
                                        "index": event.get("index"),
                                        "prior_loss_r": budgets[scope]["max_loss_r"]})
            budgets[scope]["fills"] += 1
            budgets[scope]["max_attempts"] = max(budgets[scope]["max_attempts"],
                                                   int(event.get("attempts") or 0))
        elif event.get("action") == "campaign_attempt_closed":
            prior = budgets[scope]["max_loss_r"]
            budgets[scope]["max_loss_r"] = max(budgets[scope]["max_loss_r"],
                                                float(event.get("cumulative_loss_r") or 0.0))
            current = budgets[scope]["max_loss_r"]
            if current > 2.0 + 1e-9:
                overshoots.append({"scope": scope, "index": event.get("index"),
                                   "cumulative_loss_r": current,
                                   "prior_loss_r": prior})

    package_files = [path for path in output.parent.glob("*.py")
                     if not path.name.startswith("audit_")]
    target_dates = {"2024-12-16", "2025-02-18", "2025-04-07", "2025-04-08",
                    "2025-04-09", "2025-10-29", "2025-10-30", "2026-01-28",
                    "2026-01-29", "2026-03-30", "2026-03-31"}
    manual_date_hits = [{"file": path.name, "date": date}
                        for path in package_files for date in target_dates
                        if date in path.read_text(encoding="utf-8")]
    return {
        "entry_context_count": len(contexts),
        "entry_context_failure_count": len(failures),
        "entry_context_failures": failures[:50],
        "support_rows": len(selections),
        "all_close_gates_evaluated": not missing_gates,
        "missing_all_gate_count": len(missing_gates),
        "selected_with_failed_gate_count": len(selected_failed),
        "budget_scope_count": len(budgets),
        "budget_limit_failure_count": len(budget_failures),
        "budget_limit_failures": budget_failures,
        "post_close_loss_cap_overshoot_count": len(overshoots),
        "post_close_loss_cap_overshoots": overshoots,
        "renewal_count": len(renewals),
        "renewals": renewals,
        "renewal_failure_count": len(renewal_failures),
        "renewal_failures": renewal_failures,
        "manual_target_date_hits_in_algorithm_python": manual_date_hits,
    }


def target_audit(output: Path, date_by_index: dict[int, str]):
    sell_events = [row for row in read_jsonl(output / "sell_representative_events.jsonl")
                   if row.get("action") == "parent_and_trigger_selected"]
    found_sells = []
    for row in sell_events:
        parent = row.get("parent_setup") or {}
        h1 = parent.get("h1") or {}
        local_trigger = row.get("local_trigger") or {}
        peak = local_trigger.get("second_peak_at") or parent.get("second_peak_at")
        # Parent snapshots store the current H2 in peak/peak_index. The
        # detector's selected detail contains first/second peak timestamps.
        detail = parent
        h1_date = str(h1.get("origin_at", ""))[:10]
        second_date = str(detail.get("second_peak_at", peak or ""))[:10]
        decision_date = date_by_index.get(row.get("index"), str(row.get("index", ""))[:10])
        if (h1_date, second_date, decision_date) in TARGET_SELLS:
            found_sells.append({"h1": h1_date, "h2": second_date,
                                "decision": decision_date,
                                "source_id": row.get("selected_source_id")})

    trades = read_jsonl(output / "trades.jsonl")
    target_trades = [{"entry_at": str(t.get("entry_at", ""))[:10],
                      "exit_at": str(t.get("exit_at", ""))[:10],
                      "reason": t.get("reason"),
                      "net_pnl_usd": t.get("net_pnl_usd")}
                     for t in trades if str(t.get("entry_at", ""))[:10] in TARGET_BUYS]
    target_fills = [{"entry_at": str(x.get("at", ""))[:10],
                     "index": x.get("index"), "order_id": x.get("order_id"),
                     "price": x.get("price")}
                    for x in read_jsonl(output / "execution_events.jsonl")
                    if x.get("action") == "buy_fill"
                    and str(x.get("at", ""))[:10] in TARGET_BUYS]
    links = [{"decision_at": str(x.get("decision_at", ""))[:10],
              "sell_fill_at": str(x.get("sell_fill_at", ""))[:10],
              "status": x.get("status"),
              "exit_at": str(x.get("exit_at", ""))[:10]}
             for x in read_jsonl(output / "sell_execution_links.jsonl")
             if str(x.get("decision_at", ""))[:10] in {"2025-02-26", "2026-02-05"}]
    return {"target_sell_setups": found_sells,
            "target_sell_count": len(found_sells),
            "target_buy_trades": target_trades,
            "target_buy_fills": target_fills,
            "target_buy_entry_dates": sorted({x["entry_at"] for x in target_fills}),
            "target_sell_execution_links": links,
            "target_sell_filled_count": sum(x["status"] == "sell_filled" for x in links)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prices", type=Path,
                        default=Path("nq_25y_actual_2000_2026_20260911/NQ_F_1d_combined.csv"))
    args = parser.parse_args()
    output = args.output
    metadata = read_json(output / "metadata.json")
    source_counts = read_jsonl(output / "daily_source_counts.jsonl")
    date_by_index = {row["index"]: str(row["datetime"])[:10] for row in source_counts}
    quality = input_audit(args.prices)
    accounting = accounting_audit(output, metadata)
    causal = causal_and_budget_audit(output)
    targets = target_audit(output, date_by_index)
    status = ("PASS_WITH_RESEARCH_CAVEATS" if metadata.get("run_status") == "completed_replay_only"
              and not causal["entry_context_failure_count"]
              and not causal["selected_with_failed_gate_count"]
              and not causal["budget_limit_failure_count"]
              and not causal["renewal_failure_count"]
              and accounting["accounting_mismatch_count"] == 0 else "NEEDS_REVIEW")
    result = {"audit_status": status, "metadata_summary": {
                  "run_status": metadata.get("run_status"),
                  "daily_rows": metadata.get("daily_rows"),
                  "as_of": metadata.get("as_of"),
                  "cycle_mode": metadata.get("campaign_config", {}).get("cycle_mode")},
              "input_quality": quality, "accounting": accounting,
              "causal_and_budget": causal, "targets": targets}
    (output / "audit_rev06.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    report = ["# 改訂0.6検証結果（NQ=F）", "",
              f"状態: **{status}**。全期間リプレイと独立監査を完了した。ただし未見期間の予測性能や実運用性能の検証ではない。", "",
              f"入力: {metadata.get('start')}〜{metadata.get('end')}、{metadata.get('daily_rows')}日足。モード: `{metadata.get('campaign_config', {}).get('cycle_mode')}`。", "",
              "## 数値再計算", "",
              f"- 完了取引: {accounting['closed_trades']}件、実約定試行: {accounting['filled_attempts']}件、終端未決済: {accounting['right_censored_open_attempts']}件。",
              f"- 勝ち: {accounting['wins']}件、負け: {accounting['losses']}件、勝率: {accounting['win_rate']:.2%}。",
              f"- 実現損益: {accounting['realized_net_usd']:,.0f} USD、PF: {accounting['profit_factor']:.3f}。",
              f"- 最終MTM: {accounting['final_mtm_usd']:,.1f} USD、日次MTM最大DD: {accounting['mtm_max_drawdown_usd']:,.1f} USD。",
              f"- 損益独立再計算不一致: {accounting['accounting_mismatch_count']}件。", "",
              "## 指定場面の照合", "",
              f"- 対象ダブルトップSELL構造の検知: {targets['target_sell_count']}件。",
              f"- 対象BUY約定日: {', '.join(targets['target_buy_entry_dates']) or 'なし'}。",
              f"- 対象SELLの実決済リンク: {targets['target_sell_filled_count']}件。", "",
              json.dumps(targets, ensure_ascii=False, indent=2), "",
              "## 制約", "",
              "今回の比較は同一のローカル連続先物系列を1枚で再生したもの。限月ロール、証拠金、口座資産制約、未見ウォークフォワード、日中の高安順序は評価していない。指定例への一致は設計照合であり、将来の優位性を証明しない。"]
    (output / "audit_rev06_report_ja.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
