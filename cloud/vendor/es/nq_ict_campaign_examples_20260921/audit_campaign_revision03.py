"""Integrity audit for the revision 0.3 campaign replay.

This is a deterministic replay/accounting audit, not a predictive validation.
It does not refit thresholds and does not select a run because of its result.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def date(value):
    return str(value)[:10] if value is not None else None


def load_input(meta: dict):
    path = Path(meta["input_hashes"]["prices"]["path"])
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty input: {path}")
    field = next((x for x in rows[0]
                  if x.lower() in {"datetime", "date", "timestamp"}), None)
    if field is None:
        raise ValueError(f"date column not found: {path}")
    cutoff = date(meta["as_of"])
    selected = [row for row in rows if date(row[field]) <= cutoff]
    dates = [date(row[field]) for row in selected]
    return selected, dict(path=str(path), date_column=field, all_rows=len(rows),
                          selected_rows=len(selected), start=dates[0], end=dates[-1],
                          unique_dates=len(set(dates)),
                          sorted_dates=dates == sorted(dates))


def metrics(run: Path, trades: list[dict], mtm: list[dict]) -> dict:
    ordered = sorted(trades, key=lambda x: (str(x.get("exit_at")), str(x.get("position_id"))))
    pnl = [float(x["net_pnl_usd"]) for x in ordered]
    curve, peak, dd = 0.0, 0.0, []
    for value in pnl:
        curve += value
        peak = max(peak, curve)
        dd.append(curve - peak)
    wins = [x for x in pnl if x > 0]
    losses = [x for x in pnl if x < 0]
    mtm_values = [float(x["mtm_pnl_usd"]) for x in mtm]
    mtm_dd = [float(x["close_to_close_drawdown_usd"]) for x in mtm]
    return dict(
        completed_trades=len(ordered), wins=len(wins), losses=len(losses),
        win_rate_completed=(len(wins) / len(ordered) if ordered else None),
        net_pnl_usd=round(sum(pnl), 2), gross_profit_usd=round(sum(wins), 2),
        gross_loss_usd=round(sum(losses), 2),
        profit_factor=(round(sum(wins) / abs(sum(losses)), 6) if losses else None),
        realized_trade_curve_max_dd_usd=round(min(dd), 2) if dd else 0.0,
        daily_mtm_final_usd=round(mtm_values[-1], 2) if mtm_values else None,
        daily_mtm_max_close_dd_usd=round(min(mtm_dd), 2) if mtm_dd else None,
        exit_reasons=dict(Counter(str(x.get("reason")) for x in ordered)),
        output_directory=str(run),
    )


def audit(run: Path) -> dict:
    meta = read_json(run / "metadata.json")
    if meta.get("implementation") != "nq_ict_campaign_proposal_0.3":
        raise ValueError("audit_campaign_revision03.py requires revision 0.3 output")
    final = read_json(run / "final_state.json")
    files = {
        name: read_jsonl(run / name) for name in (
            "trades.jsonl", "daily_mtm.jsonl", "close_plans.jsonl", "execution_events.jsonl",
            "entry_audit.jsonl", "gate_events.jsonl", "support_reaction_events.jsonl",
            "support_selection.jsonl", "post_sell_path_events.jsonl",
            "independent_support_events.jsonl", "independent_early_candidates.jsonl",
            "independent_confirmed_candidates.jsonl", "independent_structure_events.jsonl",
            "major_sell_candidates.jsonl", "major_reversal_events.jsonl",
            "daily_source_counts.jsonl", "double_top_events.jsonl",
        )
    }
    rows, input_info = load_input(meta)
    input_dates = [date(x[input_info["date_column"]]) for x in rows]
    input_date_set = set(input_dates)
    checks = Counter()
    violations: list[dict] = []

    def check(name, condition, detail=None):
        checks[f"{name}:pass" if condition else f"{name}:fail"] += 1
        if not condition:
            violations.append(dict(check=name, detail=detail))

    check("run_completed", meta.get("run_status") == "completed_replay_only", meta.get("run_status"))
    check("processed_bar_count", meta.get("processed_bars") == len(rows),
          {"processed": meta.get("processed_bars"), "input": len(rows)})
    check("source_date_count", len(files["daily_source_counts.jsonl"]) == len(rows),
          {"source": len(files["daily_source_counts.jsonl"]), "input": len(rows)})
    for name, expected in meta.get("output_counts", {}).items():
        observed = len(read_jsonl(run / name)) if name.endswith(".jsonl") else None
        check(f"output_count:{name}", observed == expected, {"expected": expected, "observed": observed})
    check("input_unique_dates", input_info["unique_dates"] == len(rows), input_info)
    check("input_sorted_dates", input_info["sorted_dates"], input_info)

    # Every candidate's available facts must precede the observation that used them.
    for label in ("independent_early_candidates.jsonl", "independent_confirmed_candidates.jsonl"):
        for row in files[label]:
            z, ref = row.get("zone", {}), row.get("reference", {})
            for field, left, right in (
                ("zone_available", z.get("available_index"), row.get("index")),
                ("reference_available", ref.get("available_index"), row.get("index")),
                ("touch", row.get("touch_index"), row.get("index")),
                ("low", row.get("low_index"), row.get("index")),
                ("recovery", row.get("recovery_index"), row.get("index")),
                ("breakout", row.get("breakout_index"), row.get("index")),
            ):
                check(f"{label}:{field}", left is not None and right is not None and int(left) <= int(right),
                      {"key": row.get("key"), "left": left, "right": right})
            features = next((x.get("precontact_history", {}) for x in files["independent_support_events.jsonl"]
                             if x.get("action") == "independent_candidate_registered"
                             and x.get("setup_id") == row.get("setup_id")), None)
            if features:
                check("precontact_history_strict", features.get("history_known_through_index") < row.get("touch_index"),
                      {"key": row.get("key"), "features": features, "touch": row.get("touch_index")})
            for field in ("weekly_completed_at", "monthly_completed_at"):
                value = row.get("htf", {}).get(field)
                check(f"{label}:{field}", value is None or date(value) <= date(row.get("at")),
                      {"key": row.get("key"), "value": value, "at": row.get("at")})

    # SELL source identity and reference chronology.
    sell_sources = [x.get("source_id") for x in files["major_sell_candidates.jsonl"]]
    check("selected_sell_source_unique", len(sell_sources) == len(set(sell_sources)), sell_sources)
    sell_lookup = {x.get("source_id"): x for x in files["major_sell_candidates.jsonl"]}
    for row in files["major_sell_candidates.jsonl"]:
        ref = row.get("reference", {})
        check("sell_reference_available", ref.get("available_index") is not None
              and int(ref["available_index"]) <= int(row["index"]), row)

    # Next-observed-open execution clock and chronological trades.
    intents = {x.get("intent", {}).get("id"): x.get("intent", {})
               for x in files["close_plans.jsonl"] if x.get("intent")}
    fills = [x for x in files["execution_events.jsonl"] if x.get("action") in {"buy_fill", "sell_fill"}]
    for fill in fills:
        intent = intents.get(fill.get("order_id"))
        check("fill_next_observed_open", intent is not None
              and int(fill["index"]) == int(intent["index"]) + 1, {"fill": fill, "intent": intent})
        check("fill_on_input_date", date(fill.get("at")) in input_date_set, fill)
    trades = files["trades.jsonl"]
    for trade in trades:
        check("trade_entry_on_input_date", date(trade.get("entry_at")) in input_date_set, trade)
        check("trade_exit_on_input_date", date(trade.get("exit_at")) in input_date_set, trade)
        same_day_stop = (date(trade.get("entry_at")) == date(trade.get("exit_at"))
                         and trade.get("reason") == "initial_risk_stop")
        if not same_day_stop:
            check("trade_entry_before_exit", date(trade.get("entry_at")) < date(trade.get("exit_at")), trade)
    buy_days = [date(x.get("at")) for x in fills if x.get("action") == "buy_fill"]
    sell_days = [date(x.get("at")) for x in fills if x.get("action") == "sell_fill"]
    check("no_buy_on_sell_fill_day", not set(buy_days).intersection(sell_days), {"buy": buy_days, "sell": sell_days})
    check("one_buy_fill_per_day", len(buy_days) == len(set(buy_days)), buy_days)

    # Shared campaign allowance: actual fills are counted, not offers.
    by_scope = defaultdict(list)
    for row in files["gate_events.jsonl"]:
        if row.get("budget_scope_id"):
            by_scope[row["budget_scope_id"]].append(row)
    scope_summary = {}
    for scope, entries in by_scope.items():
        filled = [x for x in entries if x.get("action") == "campaign_attempt_filled"]
        scope_summary[scope] = dict(filled=len(filled), routes=dict(Counter(x.get("route") for x in filled)))
        check("shared_attempt_limit", len(filled) <= 2, {"scope": scope, "filled": len(filled)})
    tokens = []
    for row in files["gate_events.jsonl"]:
        if row.get("action") == "campaign_attempt_filled":
            support = row.get("support") or {}
            tokens.append((row.get("budget_scope_id"), row.get("support_episode_id"),
                           support.get("low_index")))
    check("filled_support_token_unique", len(tokens) == len(set(tokens)), tokens)

    # SELL-after paths: a post-SELL contact cannot be accepted before the boundary.
    path_allowed = [x for x in files["post_sell_path_events.jsonl"]
                    if x.get("action") == "post_sell_path_assessed" and x.get("allowed")]
    for row in path_allowed:
        cutoff, index = row.get("cutoff"), row.get("index")
        check("post_sell_allowed_after_cutoff", cutoff is None or int(index) > int(cutoff), row)
        check("post_sell_allowed_has_path", row.get("path") in {"failed_top_reclaimed", "decline_then_support", "initial_or_legacy"}, row)
    for row in files["support_selection.jsonl"]:
        if row.get("selected_eligibility"):
            support = row.get("support_features", {})
            check("selected_support_departure_threshold",
                  meta["campaign_config"].get("support_selection") == "legacy"
                  or (support.get("prior_departure_atr") is not None
                      and float(support["prior_departure_atr"]) >= float(meta["campaign_config"]["support_min_prior_departure_atr"])), row)

    # Independent structure feedback must be delayed and fill-independent.
    for row in files["independent_structure_events.jsonl"]:
        check("structure_effective_next_packet", row.get("effective_from_next_packet") is True, row)
        check("structure_depends_on_fills_false", row.get("depends_on_fills") is False, row)

    # Replay/accounting metrics are recomputed from output rows.
    result = dict(
        verification_status="REPLAY_INTEGRITY_CHECKED_NOT_STATISTICAL_VALIDATION",
        run_metadata=dict(implementation=meta.get("implementation"), run_status=meta.get("run_status"),
                          as_of=meta.get("as_of"), processed_bars=meta.get("processed_bars"),
                          campaign_config=meta.get("campaign_config")),
        input=input_info,
        metrics=metrics(run, trades, files["daily_mtm.jsonl"]),
        checks=dict(checks), violation_count=len(violations), violations_sample=violations[:100],
        scope_summary=scope_summary,
        counts=dict(trades=len(trades), fills=len(fills), independent_events=len(files["independent_support_events.jsonl"]),
                    independent_early=len(files["independent_early_candidates.jsonl"]),
                    independent_confirmed=len(files["independent_confirmed_candidates.jsonl"]),
                    selected_sells=len(files["major_sell_candidates.jsonl"]),
                    path_assessments=len(path_allowed), structure_events=len(files["independent_structure_events.jsonl"])),
        final_state_summary=dict(open_position=final.get("open_position") is not None,
                                 right_censored=final.get("right_censored"),
                                 campaign_state=final.get("campaign_state")),
        limitations=[
            "This is a deterministic replay/integrity audit, not a walk-forward or holdout test.",
            "Daily OHLC cannot identify the intraday order of same-bar high/low/stop events.",
            "Continuous-futures roll, margin, funding and account-capital constraints are not modeled.",
            "The supplied input calendar is not independently certified against CME.",
            "Provisional thresholds were not fitted or selected by this audit.",
            "A completed replay does not establish predictive superiority or statistical significance.",
        ],
    )
    return result


def render(result: dict) -> str:
    failed = {k: v for k, v in result["checks"].items() if k.endswith(":fail")}
    m = result["metrics"]
    lines = [
        "# 改訂0.3・キャンペーン実行検証",
        "",
        f"検証状態: **{result['verification_status']}**",
        "",
        "この資料は、改訂0.3を固定した決定論的リプレイが、入力・因果時刻・約定時計・共有試行予算・新しいSELL後経路の条件を守っているかを監査したものです。予測優位性、将来期間での再現性、統計的有意性は検証していません。",
        "",
        "## 入力と結果",
        "",
        f"- 入力: `{result['input']['path']}`",
        f"- 期間: {result['input']['start']} ～ {result['input']['end']}",
        f"- バー数: {result['input']['selected_rows']}",
        f"- 完了取引: {m['completed_trades']}、勝ち {m['wins']}、負け {m['losses']}",
        f"- 実現損益: {m['net_pnl_usd']:.2f} USD、PF: {m['profit_factor']}",
        f"- 実現曲線最大DD: {m['realized_trade_curve_max_dd_usd']:.2f} USD",
        f"- 日次MTM最終値: {m['daily_mtm_final_usd']:.2f} USD、最大終値DD: {m['daily_mtm_max_close_dd_usd']:.2f} USD",
        "",
        "## 監査結果",
        "",
        f"- 失敗チェック: {len(failed)}",
        f"- 違反件数: {result['violation_count']}",
        f"- 独立OBイベント: {result['counts']['independent_events']}",
        f"- 独立early候補: {result['counts']['independent_early']}",
        f"- 独立confirmed候補: {result['counts']['independent_confirmed']}",
        f"- SELL候補: {result['counts']['selected_sells']}",
        f"- SELL後経路の許可件数: {result['counts']['path_assessments']}",
        f"- 独立構造フィードバック: {result['counts']['structure_events']}",
        "",
    ]
    if failed:
        lines.extend(["## 失敗したチェック", "", "```json", json.dumps(failed, ensure_ascii=False, indent=2), "```", ""])
    else:
        lines.extend(["すべての実行整合性チェックに合格しました。これは実装が指定条件で再生できたことを示すだけで、売買優位性の証明ではありません。", ""])
    lines.extend(["## 制約", ""] + [f"- {x}" for x in result["limitations"]] + [""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    result = audit(args.run)
    (args.run / "audit_rev03.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.run / "report_rev03_ja.md").write_text(render(result), encoding="utf-8")
    print(json.dumps(dict(status="ok" if result["violation_count"] == 0 else "violations",
                          violation_count=result["violation_count"], metrics=result["metrics"],
                          failed_checks={k: v for k, v in result["checks"].items() if k.endswith(":fail")}),
               ensure_ascii=False, indent=2))
    raise SystemExit(1 if result["violation_count"] else 0)


if __name__ == "__main__":
    main()
