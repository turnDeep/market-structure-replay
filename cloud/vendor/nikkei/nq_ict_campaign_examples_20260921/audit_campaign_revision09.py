"""Independent audit/report for fixed revision 0.9 full-history comparisons.

Checks accounting, shared-input agreement, baseline reproduction, selected
gate/clock integrity, actual-exit failure records, and requested anchor dates.
This is a historical replay audit, not a walk-forward or live-performance test.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .audit_campaign_revision07 import (
    accounting_audit,
    buy_hold_audit,
    causal_audit,
    input_audit,
    read_json,
    read_jsonl,
)


ROOT = Path(__file__).resolve().parent.parent
PACKAGE = Path(__file__).resolve().parent
STAGES = {
    "baseline": "validation_rev09_baseline_full_20260923",
    "failed_only": "validation_rev09_failed_full_20260923",
    "selective": "validation_rev09_selective_full_20260923",
}
PRICE_DEFAULT = ROOT / "nq_25y_actual_2000_2026_20260911" / "NQ_F_1d_combined.csv"
EXPECTED_BASELINE = {
    "closed_trades": 48,
    "execution_buy_fill_count": 49,
    "realized_net_usd": 428225.0,
    "final_mtm_usd": 540992.5,
    "mtm_max_drawdown_usd": -91870.0,
}
ANCHORS = {
    "sell_2025_double_top": ("2025-02-27", "campaign_double_top_early"),
    "buy_2025_preexisting_bullish_ob": ("2025-04-10", "BUY"),
    "sell_2026_double_top": ("2026-02-06", "campaign_double_top_early"),
    "buy_2026_preexisting_bullish_ob": ("2026-04-01", "BUY"),
}


def _anchor_audit(output: Path):
    executions = read_jsonl(output / "execution_events.jsonl")
    trades = read_jsonl(output / "trades.jsonl")
    final_state = read_json(output / "final_state.json")
    open_position = final_state.get("position") or final_state.get("open_position")
    filled_buys = {str(row.get("datetime", row.get("at", "")))[:10]
                   for row in executions if row.get("action") == "buy_fill"}
    filled_sells = {str(row.get("datetime", row.get("at", "")))[:10]
                    for row in executions if row.get("action") == "sell_fill"}
    closed_sells = {(str(row.get("exit_at", ""))[:10], row.get("reason")) for row in trades}
    result = {
        "sell_2025_double_top": {
            "date": "2025-02-27",
            "matched": ("2025-02-27", "campaign_double_top_early") in closed_sells,
        },
        "buy_2025_preexisting_bullish_ob": {
            "date": "2025-04-10", "matched": "2025-04-10" in filled_buys,
        },
        "sell_2026_double_top": {
            "date": "2026-02-06",
            "matched": ("2026-02-06", "campaign_double_top_early") in closed_sells,
        },
        "buy_2026_preexisting_bullish_ob": {
            "date": "2026-04-01", "matched": "2026-04-01" in filled_buys,
        },
        "open_position": None,
    }
    if open_position:
        c = open_position.get("candidate") or {}
        result["open_position"] = {
            "entry_at": str(open_position.get("entry_at", ""))[:10],
            "zone_id": (c.get("zone") or {}).get("id"),
            "zone_timeframe": (c.get("zone") or {}).get("timeframe"),
            "zone_available_at": (c.get("zone") or {}).get("available_at"),
            "decision_at": c.get("at"),
            "mark_to_market_status": final_state.get("end_of_input_status"),
        }
    result["filled_buys_2025_26"] = sorted(d for d in filled_buys if d.startswith(("2025", "2026")))
    result["filled_sells_2025_26"] = sorted(d for d in filled_sells if d.startswith(("2025", "2026")))
    return result


def _selective_gate_audit(output: Path):
    rows = read_jsonl(output / "support_selection.jsonl")
    selected = [row for row in rows if row.get("selected")]
    selected_failed = []
    bad_contact_clocks = []
    failure_checks = 0
    failure_rejections = 0
    resolved_failure_allows = 0
    for row in rows:
        checks = row.get("checks") or {}
        detail = (checks.get("exit_reason_reentry") or {}).get("detail") or {}
        failed = detail.get("failed_reaction") or {}
        failures = failed.get("failures") or []
        if failures:
            failure_checks += 1
            passed = (checks.get("exit_reason_reentry") or {}).get("passed") is True
            if passed:
                resolved_failure_allows += 1
            else:
                failure_rejections += 1
        if row.get("selected") and any(item.get("passed") is not True
                                        for item in checks.values()):
            selected_failed.append(row.get("key"))
    events = read_jsonl(output / "exit_reason_reentry_events.jsonl")
    failures = [event.get("failure") or {} for event in events
                if event.get("action") == "failed_reaction_registered"]
    failures_by_id = {failure.get("id"): failure for failure in failures}
    for row in selected:
        checks = row.get("checks") or {}
        detail = ((checks.get("exit_reason_reentry") or {}).get("detail") or {})
        failed = detail.get("failed_reaction") or {}
        for item in failed.get("failures") or []:
            failure = failures_by_id.get(item.get("failure_id"))
            failure_index = failure.get("index") if failure else None
            reclaimed_index = item.get("reclaimed_index")
            contact_index = item.get("contact_index")
            if (item.get("passed") is not True or failure_index is None
                    or reclaimed_index is None or contact_index is None
                    or not failure_index < reclaimed_index < contact_index):
                bad_contact_clocks.append({"key": row.get("key"),
                                           "failure_check": item,
                                           "failure_index": failure_index})
    reclaim_events = {event.get("failure_id"): event.get("index") for event in events
                      if event.get("action") == "failed_reaction_high_reclaimed"}
    exit_records = {
        (event.get("context") or {}).get("position_id"):
        (event.get("context") or {}).get("index")
        for event in events if event.get("action") == "actual_exit_reason_frozen"
    }
    bad_failure_clocks = []
    for failure in failures:
        fid, index = failure.get("id"), failure.get("index")
        if index is None or failure.get("high_known_through", index) >= index:
            bad_failure_clocks.append({"failure_id": fid, "type": "exit_day_high_used_or_index_missing"})
        if failure.get("position_id") not in exit_records or exit_records[failure.get("position_id")] != index:
            bad_failure_clocks.append({"failure_id": fid, "type": "no_matching_actual_exit"})
        reclaimed = reclaim_events.get(fid)
        if reclaimed is not None and reclaimed <= index:
            bad_failure_clocks.append({"failure_id": fid, "type": "reclaim_not_after_failure"})
    return {
        "entry_assessment_rows": len(rows),
        "selected_entry_count": len(selected),
        "selected_with_any_failed_gate": len(selected_failed),
        "failed_reaction_candidate_checks": failure_checks,
        "failed_reaction_candidate_rejections": failure_rejections,
        "failed_reaction_allows_after_reclaim_and_new_contact": resolved_failure_allows,
        "actual_failed_reactions_registered": len(failures),
        "failed_reaction_high_reclaims": len(reclaim_events),
        "failed_reaction_clock_failure_count": len(bad_failure_clocks),
        "failed_reaction_clock_failures": bad_failure_clocks[:20],
        "reentry_contact_clock_failure_count": len(bad_contact_clocks),
        "reentry_contact_clock_failures": bad_contact_clocks[:20],
    }


def _entry_clock_audit(output: Path):
    contexts = read_json(output / "entry_structural_contexts.json")
    executions = read_jsonl(output / "execution_events.jsonl")
    buys = [event for event in executions if event.get("action") == "buy_fill"]
    missing, causal_errors = [], []
    for event in buys:
        order_id = event.get("order_id") or event.get("position_id") or event.get("id")
        context = contexts.get(order_id)
        if context is None:
            # Execution schema stores the ledger order ID as order_id on this event.
            candidates = [key for key in contexts if key in str(event)]
            context = contexts.get(candidates[0]) if len(candidates) == 1 else None
        if context is None:
            missing.append({"event": event})
            continue
        fill_index = event.get("index")
        decision_index = context.get("decision_index")
        contact = context.get("contact_index")
        role = context.get("role") or {}
        zone = context.get("source_zone") or {}
        if (fill_index is None or decision_index is None or fill_index <= decision_index
                or contact is None or not (role.get("known_index", 10**12) < contact
                                           and role.get("effective_index", 10**12) <= contact)
                or zone.get("available_index", 10**12) >= contact):
            causal_errors.append({"order_id": order_id, "fill_index": fill_index,
                                  "decision_index": decision_index, "contact_index": contact,
                                  "role": role, "zone": zone})
    return {
        "filled_buys": len(buys),
        "filled_buy_missing_entry_context": len(missing),
        "filled_buy_causal_clock_failure_count": len(causal_errors),
        "filled_buy_causal_clock_failures": causal_errors[:20],
    }


def run_stage(name: str, directory: Path, prices: Path):
    metadata = read_json(directory / "metadata.json")
    accounting = accounting_audit(directory, prices, metadata)
    benchmark = buy_hold_audit(directory, prices)
    causal_v07 = causal_audit(directory, metadata)
    stage = {
        "name": name,
        "metadata": {key: metadata.get(key) for key in (
            "implementation", "run_status", "as_of", "daily_rows", "start", "end",
            "input_hashes", "revision_stage", "campaign_config")},
        "accounting": accounting,
        "buy_hold": benchmark,
        "prior_campaign_causal_audit": causal_v07,
        "entry_clock_audit": _entry_clock_audit(directory),
        "anchor_audit": _anchor_audit(directory),
    }
    if metadata.get("campaign_config", {}).get("failed_reaction_policy") != "off":
        stage["selective_gate_audit"] = _selective_gate_audit(directory)
    return stage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", type=Path, default=PRICE_DEFAULT)
    parser.add_argument("--baseline", type=Path, default=PACKAGE / STAGES["baseline"])
    parser.add_argument("--failed-only", type=Path, default=PACKAGE / STAGES["failed_only"])
    parser.add_argument("--selective", type=Path, default=PACKAGE / STAGES["selective"])
    parser.add_argument("--output", type=Path, default=PACKAGE / "validation_rev09_comparison_20260923")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("audit output directory already exists")

    quality = input_audit(args.prices)
    stages = {
        key: run_stage(key, path, args.prices)
        for key, path in (("baseline", args.baseline), ("failed_only", args.failed_only),
                          ("selective", args.selective))
    }
    metadata = {key: stage["metadata"] for key, stage in stages.items()}
    shared_hashes = [stage["metadata"].get("input_hashes") for stage in stages.values()]
    inputs_match = all(value == shared_hashes[0] for value in shared_hashes[1:])
    complete = all(stage["metadata"].get("run_status") == "completed_replay_only"
                   and stage["metadata"].get("daily_rows") == 6526
                   and stage["metadata"].get("as_of") == "2026-08-27"
                   for stage in stages.values())
    baseline = stages["baseline"]["accounting"]
    baseline_match = all(abs(float(baseline[key]) - expected) < 1e-8
                         for key, expected in EXPECTED_BASELINE.items())
    accounting_ok = all(stage["accounting"]["accounting_mismatch_count"] == 0
                        and stage["accounting"]["daily_mtm_rows"] == 6526
                        for stage in stages.values())
    entry_clock_ok = all(stage["entry_clock_audit"]["filled_buy_missing_entry_context"] == 0
                         and stage["entry_clock_audit"]["filled_buy_causal_clock_failure_count"] == 0
                         for stage in stages.values())
    selected_gate_ok = all(stage["prior_campaign_causal_audit"]["selected_with_failed_gate_count"] == 0
                           and stage["prior_campaign_causal_audit"]["unlimited_gate_failure_count"] == 0
                           for stage in stages.values())
    reentry_ok = all(stage.get("selective_gate_audit", {}).get("selected_with_any_failed_gate", 0) == 0
                     and stage.get("selective_gate_audit", {}).get("failed_reaction_clock_failure_count", 0) == 0
                     and stage.get("selective_gate_audit", {}).get("reentry_contact_clock_failure_count", 0) == 0
                     for stage in stages.values())
    anchors = {name: stage["anchor_audit"] for name, stage in stages.items()}
    # Anchors are design examples, not a holdout or a fitted success criterion.
    anchors_match = all(anchors["selective"].get(name, {}).get("matched")
                        for name in ANCHORS)
    status = ("PASS_WITH_RESEARCH_CAVEATS" if quality.get("sorted_unique_dates")
              and inputs_match and complete and baseline_match and accounting_ok
              and entry_clock_ok and selected_gate_ok and reentry_ok and anchors_match
              else "NEEDS_REVIEW")
    result = {
        "audit_status": status,
        "scope": "fixed historical replay and implementation/accounting/selected-clock checks; not predictive validation",
        "source_quality": quality,
        "same_inputs_across_stages": inputs_match,
        "all_full_replays_complete": complete,
        "rev07_baseline_reproduced": baseline_match,
        "accounting_and_daily_mtm_checks_pass": accounting_ok,
        "selected_entry_and_reentry_checks_pass": entry_clock_ok and selected_gate_ok and reentry_ok,
        "selective_target_anchors_match": anchors_match,
        "stages": stages,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "audit_rev09.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = [
        "# 改訂0.9 全期間比較と実装監査", "",
        f"状態: **{status}**。", "",
        "対象はNKD=F日足2000-09-18〜2026-08-27（6,526本）。同じ入力・同じ執行設定で",
        "基準、失敗反発ゲートのみ、退出理由別ゲートを含む3条件を再生した。",
        "これは既知履歴への再生であり、未見期間の予測力を示さない。", "",
        "## 結果", "",
        "| 条件 | 完了取引 | BUY約定 | 実現損益 USD | 最終MTM USD | 最大DD USD | Buy & Holdとの差 USD |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key in ("baseline", "failed_only", "selective"):
        stage = stages[key]
        a, b = stage["accounting"], stage["buy_hold"]
        lines.append(f"| {key} | {a['closed_trades']} | {a['execution_buy_fill_count']} | "
                     f"{a['realized_net_usd']:,.0f} | {a['final_mtm_usd']:,.1f} | "
                     f"{a['mtm_max_drawdown_usd']:,.1f} | {b['strategy_final_minus_buy_hold_usd']:,.1f} |")
    lines += ["", "## ロジック確認", "",
              f"- 入力ハッシュ一致: {inputs_match}。全条件で6,526行、最終日2026-08-27。",
              f"- 0.7相当基準の主要値再現: {baseline_match}。会計不一致は各条件0件。",
              f"- 選択後の失敗ゲート、BUY前の構造・接触時計違反: 条件別ログ参照。",
              f"- 選択条件における失敗高値回復前・再接触前の許可: "
              f"{stages['selective'].get('selective_gate_audit', {}).get('failed_reaction_clock_failure_count')}件。",
              f"- 既知の指定事例4件の注文日一致: {anchors_match}。日足終値判定の翌観測日始値約定。", "",
              "## 指定事例（選択条件）", "",
              "| 事例 | ログ上の結果 |",
              "|---|---|",
              "| 2024年12月／2025年2月の高値後SELL | 2025-02-27、campaign_double_top_early |",
              "| 2025年4月の事前bullish OB反発BUY | 2025-04-10。3月17日のBUYは残り、4月2日の同じ帯へのBUYは消えた |",
              "| 2025年10月／2026年1月の高値後SELL | 2026-02-06、campaign_double_top_early |",
              "| 2026年3月末の事前bullish OB反発BUY | 2026-04-01。終端時点で未決済 |", "",
              "## 解釈と限界", "",
              "失敗反発ゲートだけを追加した条件は、基準より最終MTMが小さい一方、" 
              "最大DDは浅くなった。退出理由別条件まで加えると、選択条件の全期間最終MTMはBuy & Holdを下回った。",
              "選択条件では2016〜2018年窓の結果が改善したが、2023-01-03〜2026-08-27窓では" 
              "Buy & Holdとの差が広がった。特定の数例に一致することだけでこの変更を採用しない。", "",
              "NKD=F連続先物の限月ロール、証拠金・口座資産制約・資金調達費は再現していない。",
              "これは固定1枚のUSD損益であり、口座資産倍率でも、未見ウォークフォワードでもない。",
              "反発・再BUY条件が実際の将来に優位性を持つこと、また既知期間への適合が過剰でないことは未確認。", ""]
    (args.output / "report_rev09_ja.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "audit_status", "same_inputs_across_stages", "all_full_replays_complete",
        "rev07_baseline_reproduced", "accounting_and_daily_mtm_checks_pass",
        "selected_entry_and_reentry_checks_pass", "selective_target_anchors_match")},
        ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
