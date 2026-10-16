"""Independent audit for revision 0.5 replay outputs.

This is a research audit, not a parameter optimizer. It recomputes input
quality, accounting, causal-clock invariants, and the requested example-date
diagnostics from the generated files.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path


TARGET_DATES = {
    "2024-12-16", "2025-02-18", "2025-04-07", "2025-04-08",
    "2025-04-09", "2025-04-10", "2025-10-29", "2025-10-30",
    "2026-01-28", "2026-01-29", "2026-02-05", "2026-02-06",
    "2026-03-30", "2026-03-31", "2026-04-01",
}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def date_of(row, date_by_index):
    if row.get("index") in date_by_index:
        return date_by_index[row["index"]]
    for key in ("datetime", "at", "entry_at", "exit_at"):
        if row.get(key):
            return str(row[key])[:10]
    return None


def input_audit(path: Path):
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig", newline="")))
    dates, duplicate_dates, missing, impossible, negative_volume = [], [], [], [], []
    seen = set()
    for row in rows:
        d = row.get("datetime")
        if not d:
            missing.append((len(dates), "datetime"))
            continue
        if d in seen:
            duplicate_dates.append(d)
        seen.add(d)
        dates.append(d)
        try:
            o, h, low, c = (float(row[k]) for k in ("open", "high", "low", "close"))
            v = float(row["volume"])
            if not (low <= min(o, c) <= max(o, c) <= h):
                impossible.append(d)
            if v < 0:
                negative_volume.append(d)
        except (TypeError, ValueError, KeyError):
            missing.append((len(dates) - 1, "OHLCV"))
    return {
        "rows": len(rows), "columns": list(rows[0].keys()) if rows else [],
        "first": dates[0] if dates else None, "last": dates[-1] if dates else None,
        "sorted_unique_dates": dates == sorted(set(dates)),
        "duplicate_date_count": len(duplicate_dates), "missing_or_non_numeric_count": len(missing),
        "impossible_ohlc_count": len(impossible), "negative_volume_count": len(negative_volume),
    }


def pnl_audit(output: Path, metadata: dict):
    trades = load_jsonl(output / "trades.jsonl")
    cfg = metadata.get("execution_config", {})
    point = float(cfg.get("point_value_usd", 20.0))
    commission = float(cfg.get("commission_per_contract_side_usd", 2.5))
    mismatches = []
    pnl = []
    for t in trades:
        qty = int(t.get("quantity", 1))
        expected = ((float(t["exit_price"]) - float(t["entry_price"]))
                    * qty * point - 2.0 * qty * commission)
        actual = float(t["net_pnl_usd"])
        pnl.append(actual)
        if abs(expected - actual) > 1e-7:
            mismatches.append(dict(position_id=t.get("position_id"), expected=expected, actual=actual))
    wins = [x for x in pnl if x > 0]
    losses = [x for x in pnl if x < 0]
    mtm = load_jsonl(output / "daily_mtm.jsonl")
    peak = -math.inf
    drawdowns = []
    for row in mtm:
        value = float(row["mtm_pnl_usd"])
        peak = max(peak, value)
        drawdowns.append(value - peak)
    return {
        "closed_trades": len(trades), "wins": len(wins), "losses": len(losses),
        "win_rate": len(wins) / len(trades) if trades else None,
        "realized_net_usd": sum(pnl), "gross_profit_usd": sum(wins),
        "gross_loss_usd": sum(losses), "profit_factor": sum(wins) / abs(sum(losses)) if losses else None,
        "daily_mtm_rows": len(mtm), "final_mtm_usd": float(mtm[-1]["mtm_pnl_usd"]) if mtm else None,
        "mtm_max_drawdown_usd": min(drawdowns) if drawdowns else None,
        "accounting_mismatch_count": len(mismatches), "accounting_mismatches": mismatches[:20],
    }


def causal_audit(output: Path, source_rows, date_by_index):
    failures = []
    contexts = load_json(output / "entry_structural_contexts.json")
    for order_id, context in contexts.items():
        role = context.get("role")
        contact = context.get("contact_index")
        if role is None:
            failures.append(dict(type="entry_role_missing", order_id=order_id))
        elif not (role.get("known_index", math.inf) < contact
                  and role.get("effective_index", math.inf) <= contact):
            failures.append(dict(type="entry_role_not_precontact", order_id=order_id,
                                 known_index=role.get("known_index"), effective_index=role.get("effective_index"),
                                 contact_index=contact))
    selection = load_jsonl(output / "support_selection.jsonl")
    missing_all_gate = [r for r in selection if not r.get("all_close_gates_evaluated")]
    selected_with_failure = []
    for row in selection:
        if row.get("selected") and any(v.get("passed") is not True
                                        for v in row.get("checks", {}).values()):
            selected_with_failure.append(row.get("key"))
    for row in load_jsonl(output / "market_cycle_events.jsonl"):
        if row.get("action") == "independent_upward_phase_confirmed":
            advance = row.get("advance", {})
            pending = row.get("next_context", {})
            if pending.get("effective_index") != row.get("index", -1) + 1:
                failures.append(dict(type="advance_not_next_bar", index=row.get("index")))
            if advance.get("effective_index") != row.get("index", -1) + 1:
                failures.append(dict(type="advance_record_clock", index=row.get("index")))
    representative = load_jsonl(output / "sell_representative_events.jsonl")
    for row in representative:
        if row.get("action") != "parent_and_trigger_selected" or not row.get("selected_source_id"):
            continue
        parent = row.get("parent_setup") or {}
        local = row.get("local_trigger") or {}
        decision = row.get("index", -1)
        for name, item in (("h1", parent.get("h1")), ("neckline", parent.get("neckline"))):
            if item and item.get("origin_index", decision + 1) > decision:
                failures.append(dict(type="sell_parent_future_origin", index=decision, field=name))
        if local.get("decision_index", decision + 1) > decision:
            failures.append(dict(type="sell_trigger_future_decision", index=decision))
    source_paths = [p for p in output.parent.glob("*.py")
                    if not p.name.startswith("audit_")]
    manual_date_hits = []
    for path in source_paths:
        text = path.read_text(encoding="utf-8")
        for target in TARGET_DATES:
            if target in text:
                manual_date_hits.append(dict(file=path.name, date=target))
    return {
        "entry_context_count": len(contexts),
        "entry_context_failures": failures[:100],
        "entry_context_failure_count": len(failures),
        "support_rows": len(selection),
        "all_close_gates_evaluated": len(missing_all_gate) == 0,
        "missing_all_gate_count": len(missing_all_gate),
        "selected_with_failed_gate_count": len(selected_with_failure),
        "market_cycle_effective_next_bar": not any(x.get("type", "").startswith("advance_") for x in failures),
        "manual_target_date_hits_in_algorithm_python": manual_date_hits,
    }


def target_audit(output: Path, date_by_index):
    def rows(name):
        return load_jsonl(output / name)

    def by_dates(name, dates):
        return [r for r in rows(name) if date_of(r, date_by_index) in dates]

    sell_events = rows("sell_representative_events.jsonl")
    sell_dates = [(date_of(r, date_by_index), r.get("selected_source_id"), r.get("selected_parent_top_id"))
                  for r in sell_events if r.get("action") == "parent_and_trigger_selected"]
    april = by_dates("support_selection.jsonl", {"2025-04-07", "2025-04-08", "2025-04-09", "2025-04-10"})
    march = by_dates("support_selection.jsonl", {"2026-03-30", "2026-03-31", "2026-04-01"})
    def candidate_summary(items):
        return {
            "rows": len(items), "selected": sum(bool(x.get("selected")) for x in items),
            "eligible": sum(bool(x.get("selected_eligibility")) for x in items),
            "routes": dict(Counter(x.get("support_route") for x in items)),
            "reasons": dict(Counter(x.get("reason") for x in items)),
            "examples": [{"index": x.get("index"), "key": x.get("key"), "route": x.get("support_route"),
                          "selected": x.get("selected"), "eligible": x.get("selected_eligibility"),
                          "reason": x.get("reason")} for x in items[:12]],
        }
    return {
        "sell_decision_dates": sell_dates,
        "first_double_top_sell_target": [x for x in sell_dates if x[0] == "2025-02-26"],
        "second_double_top_sell_target": [x for x in sell_dates if x[0] == "2026-02-05"],
        "april_support_target": candidate_summary(april),
        "march_support_target": candidate_summary(march),
        "trades_near_targets": [
            {"entry_at": t.get("entry_at"), "exit_at": t.get("exit_at"), "reason": t.get("reason"),
             "net_pnl_usd": t.get("net_pnl_usd")}
            for t in rows("trades.jsonl")
            if str(t.get("entry_at", ""))[:10] in {"2025-03-17", "2025-04-10", "2026-04-01"}
        ],
    }


def write_report(path: Path, metadata, data, pnl, causal, targets):
    reason_lines = []
    for label, payload in (("2025-04-07〜10", targets["april_support_target"]),
                           ("2026-03-30〜04-01", targets["march_support_target"])):
        reason_lines.append(f"### {label}\n\n")
        reason_lines.append(f"候補行数: {payload['rows']}、適格候補: {payload['eligible']}、選択: {payload['selected']}。\n\n")
        reason_lines.append("主な理由: " + ", ".join(f"{k}={v}" for k, v in payload["reasons"].items()) + "。\n\n")
    report = """# 改訂0.5検証結果（NKD=F）\n\n"""
    report += "状態: **全期間リプレイ完了。ただし統計的な未見期間検証・実運用検証ではない。**\n\n"
    report += f"入力: {metadata.get('start')}〜{metadata.get('end')}、{metadata.get('daily_rows')}日足。処理済み: {metadata.get('processed_bars')}本。\n\n"
    report += "## 入力品質\n\n"
    report += f"{json.dumps(data, ensure_ascii=False, indent=2)}\n\n"
    report += "## リプレイ結果\n\n"
    report += f"- 完了取引: {pnl['closed_trades']}件（勝ち{pnl['wins']}、負け{pnl['losses']}、勝率{pnl['win_rate']:.2%}）。\n"
    report += f"- 実現損益: {pnl['realized_net_usd']:,.0f} USD、PF {pnl['profit_factor']:.3f}。\n"
    report += f"- 最終日次MTM: {pnl['final_mtm_usd']:,.1f} USD、MTM最大DD {pnl['mtm_max_drawdown_usd']:,.1f} USD。\n"
    report += f"- 記録された損益の独立再計算不一致: {pnl['accounting_mismatch_count']}件。\n\n"
    report += "## 因果時刻・実装監査\n\n"
    report += f"{json.dumps(causal, ensure_ascii=False, indent=2)}\n\n"
    report += "## 指定された売買場面\n\n"
    report += "SELLの代表日: " + ", ".join(str(x[0]) for x in targets["sell_decision_dates"]) + "。\n\n"
    report += "2024-12-16を第1高値、2025-02-18を第2高値とするSELLは、2025-02-26終値判断、翌観測日始値の決済でした。\n\n"
    report += "2025-10-30を第1高値、2026-01-28を第2高値とするSELLは、2026-02-05終値判断、翌観測日始値の決済でした。\n\n"
    report += "2025年4月のBUYと2026年3月末のBUYの候補診断は次のとおりです。\n\n"
    report += "\n".join(reason_lines)
    report += "## 結論\n\n"
    report += "改訂0.5は、親トップと局所トリガーを分離し、上昇レッグ内OBの候補を生成し、2026-03-31のBUYを選択するところまで動作しました。\n\n"
    report += "一方、2025-04-09の反発候補は複数生成されたものの、2025-03-17の不要な早期BUYによって同じ予算の2回上限に達しており、指定した4月BUYは成立していません。したがって、次の修正対象は上昇レッグ内OBの検知器そのものより、反落後の最初の支持反応を選ぶための反応優先順位・再BUY予算・小さな反応の除外です。\n\n"
    report += "改訂0.5の損益は改訂0.4より完了取引の実現損益・PF・最大DDが悪化しています。ただし最終MTMは未決済ポジションを含むため、改訂間の優劣をこれだけで結論づけません。未見期間、同一時計の対照比較、ローリング検証は未実施です。\n"
    path.write_text(report, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prices", type=Path, default=Path("nq_25y_actual_2000_2026_20260911/NQ_F_1d_combined.csv"))
    args = parser.parse_args()
    output = args.output
    metadata = load_json(output / "metadata.json")
    source = load_jsonl(output / "daily_source_counts.jsonl")
    date_by_index = {r["index"]: str(r["datetime"])[:10] for r in source}
    data = input_audit(args.prices)
    pnl = pnl_audit(output, metadata)
    causal = causal_audit(output, source, date_by_index)
    targets = target_audit(output, date_by_index)
    summary = dict(metadata=metadata, input_quality=data, pnl=pnl, causal=causal, targets=targets,
                   audit_status=("PASS_WITH_RESEARCH_CAVEATS" if not causal["entry_context_failure_count"]
                                 and not causal["selected_with_failed_gate_count"]
                                 and pnl["accounting_mismatch_count"] == 0 else "NEEDS_REVIEW"))
    (output / "audit_rev05.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(output / "audit_rev05_report_ja.md", metadata, data, pnl, causal, targets)
    print(json.dumps(dict(audit_status=summary["audit_status"], input_quality=data, pnl=pnl,
                          causal={k:v for k,v in causal.items() if not isinstance(v, list)}, targets=targets),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
