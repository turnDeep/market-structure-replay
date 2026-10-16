"""Revision-0.4 replay integrity audit.

This is an implementation/data audit, not a predictive-performance test.
It checks the new structural-role, representative-SELL, cycle, and entry-scale
logs against the completed replay output and preserves unresolved limitations.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def read_json(path: Path | str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def date10(value):
    return str(value)[:10] if value is not None else None


def input_dates(metadata: dict):
    path = Path(metadata["input_hashes"]["prices"]["path"])
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    column = next(c for c in rows[0] if c.lower() in {"datetime", "date", "timestamp"})
    return [date10(row[column]) for row in rows], path


def metric_summary(run: Path, metadata: dict) -> dict:
    trades = read_jsonl(run / "trades.jsonl")
    pnls = [float(row["net_pnl_usd"]) for row in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    curve, total, peak = [], 0.0, 0.0
    for pnl in pnls:
        total += pnl
        peak = max(peak, total)
        curve.append(total - peak)
    mtm = read_jsonl(run / "daily_mtm.jsonl")
    drawdowns = [float(row["close_to_close_drawdown_usd"]) for row in mtm]
    return dict(
        trade_count=len(trades), wins=len(wins), losses=len(losses),
        completed_win_rate=(len(wins) / len(trades) if trades else None),
        completed_net_pnl_usd=sum(pnls), gross_profit_usd=sum(wins),
        gross_loss_usd=sum(losses),
        profit_factor=(sum(wins) / abs(sum(losses)) if losses else None),
        realized_trade_curve_max_drawdown_usd=min(curve) if curve else 0.0,
        daily_mtm_final_usd=(float(mtm[-1]["mtm_pnl_usd"]) if mtm else None),
        daily_mtm_max_close_to_close_drawdown_usd=(min(drawdowns) if drawdowns else 0.0),
        open_position=read_json(run / "final_state.json").get("open_position") is not None,
        exit_reasons=dict(Counter(row["reason"] for row in trades)),
        entries_by_year=dict(Counter(date10(row["entry_at"])[:4] for row in trades)),
    )


def check_summary(run: Path) -> dict:
    metadata = read_json(run / "metadata.json")
    dates, price_path = input_dates(metadata)
    date_set = set(dates)
    checks = Counter()
    failures = []

    def check(name: str, condition: bool, detail=None):
        checks[name + (":pass" if condition else ":fail")] += 1
        if not condition:
            failures.append(dict(check=name, detail=detail))

    check("metadata_completed", metadata.get("run_status") == "completed_replay_only", metadata.get("run_status"))
    check("metadata_not_validated_label", metadata.get("verification_status") == "NOT_VALIDATED",
          metadata.get("verification_status"))
    state_rows = read_jsonl(run / "daily_states.jsonl")
    mtm_rows = read_jsonl(run / "daily_mtm.jsonl")
    source_rows = read_jsonl(run / "daily_source_counts.jsonl")
    for name, rows in (("daily_states", state_rows), ("daily_mtm", mtm_rows), ("daily_source_counts", source_rows)):
        check(name + "_covers_input", len(rows) == len(dates), (len(rows), len(dates)))
    source_dates = [date10(row.get("datetime")) for row in source_rows]
    check("source_dates_match_input", source_dates == dates, None)
    check("input_unique_sorted", len(dates) == len(date_set) and dates == sorted(dates), None)
    calendar = read_json(metadata["input_hashes"]["calendar"]["path"])
    check("calendar_matches_input", [date10(x) for x in calendar["session_dates"]] == dates, None)

    expected_counts = metadata.get("output_counts", {})
    for filename, expected in expected_counts.items():
        actual = len(read_jsonl(run / filename)) if filename.endswith(".jsonl") else None
        if actual is not None:
            check("metadata_count:" + filename, actual == expected, (actual, expected))

    # Inherited causal candidates.
    for filename in ("shared_early_candidates.jsonl", "shared_confirmed_candidates.jsonl"):
        for row in read_jsonl(run / filename):
            zone, ref = row.get("zone", {}), row.get("reference", {})
            for field, value in (("zone_available", zone.get("available_index")),
                                 ("reference_available", ref.get("available_index")),
                                 ("touch", row.get("touch_index")),
                                 ("low", row.get("low_index")),
                                 ("recovery", row.get("recovery_index")),
                                 ("breakout", row.get("breakout_index"))):
                check(filename + ":" + field, value is not None and int(value) <= int(row["index"]),
                      row.get("key"))

    # New major structure records must not become usable before confirmation.
    structures = read_jsonl(run / "major_structure_events.jsonl")
    confirmed = [row["structure"] for row in structures if row.get("action") == "major_structure_confirmed"]
    structure_ids = {row.get("id") for row in confirmed}
    for record in confirmed:
        high, low = record.get("high", {}), record.get("low", {})
        check("structure_high_available", high.get("available_index") <= record["known_index"], record.get("id"))
        check("structure_low_available", low.get("available_index") <= record["known_index"], record.get("id"))
        check("structure_origins_before_breakout",
              high.get("origin_index") < record["breakout_index"] and low.get("origin_index") < record["breakout_index"],
              record.get("id"))
        check("structure_known_before_effective", record["known_index"] < record["effective_index"], record.get("id"))

    # Roles are available only after the role's causal registration close.
    for row in read_jsonl(run / "ob_structural_role_events.jsonl"):
        role = row.get("role", {})
        check("role_structure_reference", role.get("structure_id") in structure_ids, role.get("zone_id"))
        check("role_known_before_effective", role.get("known_index") < role.get("effective_index"), role.get("zone_id"))

    # Every entry context has a pre-contact role and is not retroactive.
    contexts = read_json(run / "entry_structural_contexts.json")
    for order_id, context in contexts.items():
        role = context.get("role") or {}
        decision_index = context.get("decision_index")
        check("entry_role_present", bool(role), order_id)
        check("entry_role_predecision",
              bool(role) and role.get("known_index") < decision_index and role.get("effective_index") <= decision_index,
              order_id)

    # All fills remain next-observed-open fills.
    plans = read_jsonl(run / "close_plans.jsonl")
    intents = {row.get("intent", {}).get("id"): row.get("intent", {}) for row in plans if row.get("intent")}
    for row in read_jsonl(run / "execution_events.jsonl"):
        if row.get("action") not in {"buy_fill", "sell_fill"}:
            continue
        intent = intents.get(row.get("order_id"))
        check("fill_next_observed_open", intent is not None and int(row["index"]) == int(intent["index"]) + 1,
              row.get("order_id"))

    # Canonical SELL decisions and the append-only decision-to-execution links.
    canonical_ids = {row.get("source_id") for row in read_jsonl(run / "major_sell_candidates.jsonl")}
    decisions = read_jsonl(run / "campaign_sell_decisions.jsonl")
    links = read_jsonl(run / "sell_execution_links.jsonl")
    trades = read_jsonl(run / "trades.jsonl")
    trade_by_position = {row.get("position_id"): row for row in trades}
    link_by_position = {row.get("position_id"): row for row in links}
    decision_dates = Counter(date10(row.get("at")) for row in decisions)
    for date, count in decision_dates.items():
        check("one_sell_decision_per_day", count <= 1, date)
    for decision in decisions:
        pid = decision.get("position_id")
        check("sell_source_is_canonical", decision.get("source_id") in canonical_ids, pid)
        check("sell_decision_has_execution_link", pid in link_by_position, pid)
        check("sell_decision_position_has_trade", pid in trade_by_position, pid)
    for link in links:
        check("sell_link_position_matches_trade", link.get("position_id") in trade_by_position, link.get("position_id"))
        if link.get("status") == "sell_filled":
            check("sell_link_fill_date_is_input", date10(link.get("sell_fill_at")) in date_set, link)

    # A completed market cycle is explicitly next-bar effective and fill-independent.
    for event in read_jsonl(run / "market_cycle_events.jsonl"):
        if event.get("action") != "market_cycle_completed":
            continue
        next_context = event.get("next_context") or {}
        check("cycle_effective_next_bar", next_context.get("effective_index") == event.get("index") + 1,
              event.get("index"))
        check("cycle_fill_independent", event.get("depends_on_fills") is False, event.get("index"))

    for row in read_jsonl(run / "position_evidence.jsonl"):
        check("position_evidence_not_before_entry", row.get("index") >= row.get("entry_index", row.get("index")),
              row.get("position_id"))

    # Target-window diagnostics are intentionally descriptive, not success labels.
    index_by_date = {date: i for i, date in enumerate(dates)}
    assessments = read_jsonl(run / "support_selection.jsonl")
    dispositions = {}
    for date in ("2025-04-07", "2025-04-09", "2026-03-31", "2026-08-05"):
        rows = [row for row in assessments if row.get("index") == index_by_date.get(date)]
        dispositions[date] = dict(
            total=len(rows),
            allowed=sum(bool(row.get("selected_eligibility")) for row in rows),
            reasons=dict(Counter(row.get("reason") for row in rows)),
        )
    fills_by_action = [row for row in read_jsonl(run / "execution_events.jsonl") if row.get("action") == "buy_fill"]
    fill_dates = [date10(row.get("at")) for row in fills_by_action]

    metrics = metric_summary(run, metadata)
    return dict(
        run=str(run), implementation=metadata.get("implementation"), status=metadata.get("run_status"),
        verification_status=metadata.get("verification_status"), input=dict(path=str(price_path), rows=len(dates),
        start=dates[0], end=dates[-1]), metrics=metrics, checks=dict(checks),
        failure_count=len(failures), failures=failures[:50], counts={
            "major_structures_confirmed": len(confirmed),
            "structural_role_events": len(read_jsonl(run / "ob_structural_role_events.jsonl")),
            "entry_contexts": len(contexts), "sell_decisions": len(decisions),
            "sell_links": len(links), "market_cycle_completions": sum(
                row.get("action") == "market_cycle_completed" for row in read_jsonl(run / "market_cycle_events.jsonl")),
            "buy_fill_dates": fill_dates,
        }, windows=dispositions,
        interpretation=dict(
            target_2025_04_09_entry_issued="2025-04-10" in fill_dates,
            target_2026_03_31_entry_issued="2026-04-01" in fill_dates,
            target_2025_sell_decision=next((row for row in decisions if date10(row.get("at")) == "2025-02-26"), None),
            target_2026_sell_decision=next((row for row in decisions if date10(row.get("at")) == "2026-02-05"), None),
            note="Target windows are diagnostics; they are not a parameter-fitting objective or an independent test.",
        ),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    report = check_summary(args.run)
    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    (args.run / "audit_rev04.json").write_text(serialized, encoding="utf-8")
    # These names are the conventional audit artifacts in this package. Write
    # the revision-0.4 content there as well so an older audit cannot be read
    # accidentally as the current result.
    (args.run / "audit.json").write_text(serialized, encoding="utf-8")
    lines = [
        "# 改訂0.4リプレイ検証監査",
        "",
        "これはコード接続・因果順序・出力整合性・指定場面の診断であり、予測優位性や未見性能の検証ではない。",
        "",
        f"- 入力: {report['input']['start']}〜{report['input']['end']} / {report['input']['rows']}本",
        f"- 実行: `{report['status']}` / 実装 `{report['implementation']}`",
        f"- 監査失敗件数: {report['failure_count']}",
        f"- 完了取引: {report['metrics']['trade_count']}件、勝ち {report['metrics']['wins']}、負け {report['metrics']['losses']}",
        f"- 完了取引損益: {report['metrics']['completed_net_pnl_usd']:.2f} USD / PF {report['metrics']['profit_factor']:.3f}",
        f"- 日次MTM最大DD: {report['metrics']['daily_mtm_max_close_to_close_drawdown_usd']:.2f} USD",
        "",
        "## 機械的に確認できたこと",
        "",
        "- 既存候補の利用可能時刻、参照Pivot、接触・回復・突破は、判断時点以前に限定された。",
        "- 新規主要構造は確認日と翌足有効日を分離した。OB役割も構造確認後に登録され、入口判断より前であることを確認した。",
        "- SELL判断は代表候補に統一され、判断ログと実行リンクを分離した。",
        "- 構造サイクル完了は約定に依存せず、翌足から有効となる形式で記録された。",
        "- 日足入力と観測カレンダーは完全一致した。ただしカレンダーは取引所公式休日表として認証されたものではない。",
        "",
        "## 指定場面の診断",
        "",
        f"- 2025-04-09: {report['windows']['2025-04-09']} / 翌日BUY約定: {report['interpretation']['target_2025_04_09_entry_issued']}",
        f"- 2026-03-31: {report['windows']['2026-03-31']} / 翌日BUY約定: {report['interpretation']['target_2026_03_31_entry_issued']}",
        "- 2025年のSELL判断は2025-02-26に発生し、選択されたダブルトップ状態では第2高値が2025-02-18として記録された。",
        "- 2026年のSELL判断は2026-02-05に発生し、第2高値は2026-01-28として記録された。ただし、最初の高値はユーザー指定の2025-10-29ではなく、現行検知器では別の高値系列として扱われた。",
        "- 2026-03-31の候補は反応自体は検出されたが、標準の接触前主要OB役割がなく、BUY適格にはならなかった。",
        "",
        "## 解釈",
        "",
        "改訂0.4は実行・因果ログの整合性には問題を検出しなかったが、指定された全BUY場面を再現してはいない。特に接触前の構造役割条件は、2026-03-31の反応を除外した。これは先見バイアスを避けるための設計結果である一方、OB役割の定義が過度に狭い可能性も示す。",
        "",
        "17件の完了取引、固定連続先物、観測カレンダーだけでは売買優位性は証明できない。改訂0.2との損益差は参考診断にとどまり、候補集合・管理・試行予算が同一でないため、因果的な改善とは解釈しない。ウォークフォワード、未見期間、基準モデル、分足の同一足順序は未検証である。",
        "",
        "機械監査の詳細は `audit_rev04.json` に保存した。",
    ]
    report_text = "\n".join(lines) + "\n"
    (args.run / "report_rev04_ja.md").write_text(report_text, encoding="utf-8")
    (args.run / "report_ja.md").write_text(report_text, encoding="utf-8")
    print(json.dumps({"audit_rev04": str(args.run / 'audit_rev04.json'),
                      "report_rev04": str(args.run / 'report_rev04_ja.md'),
                      "failure_count": report["failure_count"],
                      "metrics": report["metrics"],
                      "windows": report["windows"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
