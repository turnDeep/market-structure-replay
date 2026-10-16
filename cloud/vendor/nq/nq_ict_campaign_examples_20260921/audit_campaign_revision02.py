"""Independent integrity audit for the revision 0.2 campaign replay.

This is a replay/accounting audit, not a predictive or statistical validation.
It reads only the completed output directory and the input files hashed in its
metadata. It deliberately does not rerun the detector or tune parameters.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


MAX_ATTEMPTS = 2
MAX_ROUTE_FILLS = {"early": 1, "confirmed": 1}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def date(value) -> str | None:
    return str(value)[:10] if value is not None else None


def load_input(meta: dict) -> tuple[list[dict], dict]:
    path = Path(meta["input_hashes"]["prices"]["path"])
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    field = next((x for x in rows[0] if x.lower() in {"datetime", "date", "timestamp"}), None)
    if field is None:
        raise ValueError(f"date column not found: {path}")
    cutoff = date(meta["as_of"])
    selected = [row for row in rows if date(row[field]) <= cutoff]
    return selected, dict(path=str(path), date_column=field, all_rows=len(rows),
                          selected_rows=len(selected), start=date(selected[0][field]),
                          end=date(selected[-1][field]),
                          unique_dates=len({date(x[field]) for x in selected}),
                          sorted_dates=[date(x[field]) for x in selected]
                          == sorted(date(x[field]) for x in selected))


def metrics(run: Path, trades: list[dict], mtm: list[dict]) -> dict:
    ordered = sorted(trades, key=lambda x: (str(x.get("exit_at")), str(x.get("position_id"))))
    pnl = [float(x["net_pnl_usd"]) for x in ordered]
    total = 0.0
    peak = 0.0
    drawdowns = []
    for value in pnl:
        total += value
        peak = max(peak, total)
        drawdowns.append(total - peak)
    wins = [x for x in pnl if x > 0]
    losses = [x for x in pnl if x < 0]
    mtm_values = [float(x["mtm_pnl_usd"]) for x in mtm]
    mtm_dd = [float(x["close_to_close_drawdown_usd"]) for x in mtm]
    return dict(
        completed_trades=len(ordered),
        wins=len(wins), losses=len(losses),
        win_rate_completed=(len(wins) / len(ordered) if ordered else None),
        net_pnl_usd=round(sum(pnl), 2),
        gross_profit_usd=round(sum(wins), 2),
        gross_loss_usd=round(sum(losses), 2),
        profit_factor=(round(sum(wins) / abs(sum(losses)), 6) if losses else None),
        realized_trade_curve_max_dd_usd=round(min(drawdowns), 2) if drawdowns else 0.0,
        daily_mtm_final_usd=round(mtm_values[-1], 2) if mtm_values else None,
        daily_mtm_max_close_dd_usd=round(min(mtm_dd), 2) if mtm_dd else None,
        exit_reasons=dict(Counter(str(x.get("reason")) for x in ordered)),
        entry_dates=[date(x.get("entry_at")) for x in ordered],
        exit_dates=[date(x.get("exit_at")) for x in ordered],
        output_directory=str(run),
    )


def audit(run: Path, baseline: Path | None = None) -> dict:
    meta = read_json(run / "metadata.json")
    if meta.get("implementation") != "nq_ict_campaign_proposal_0.2":
        raise ValueError("This audit is restricted to revision 0.2. Revision 0.3 needs a new "
                         "audit for independent support candidates and shared attempt budgets.")
    final = read_json(run / "final_state.json")
    trades = read_jsonl(run / "trades.jsonl")
    mtm = read_jsonl(run / "daily_mtm.jsonl")
    plans = read_jsonl(run / "close_plans.jsonl")
    fills = read_jsonl(run / "execution_events.jsonl")
    early = read_jsonl(run / "shared_early_candidates.jsonl")
    confirmed = read_jsonl(run / "shared_confirmed_candidates.jsonl")
    sells = read_jsonl(run / "major_sell_candidates.jsonl")
    gate = read_jsonl(run / "gate_events.jsonl")
    support = read_jsonl(run / "support_reaction_events.jsonl")
    reversals = read_jsonl(run / "major_reversal_events.jsonl")
    source_counts = read_jsonl(run / "daily_source_counts.jsonl")
    input_rows, input_info = load_input(meta)
    input_dates = [date(x[input_info["date_column"]]) for x in input_rows]
    input_index = {x: i for i, x in enumerate(input_dates)}
    checks = Counter()
    violations: list[dict] = []

    def check(name: str, condition: bool, detail=None):
        checks[f"{name}:pass" if condition else f"{name}:fail"] += 1
        if not condition:
            violations.append(dict(check=name, detail=detail))

    check("run_completed", meta.get("run_status") == "completed_replay_only", meta.get("run_status"))
    check("processed_bar_count", meta.get("processed_bars") == len(input_rows),
          dict(processed=meta.get("processed_bars"), input=len(input_rows)))
    check("source_date_count", len(source_counts) == len(input_rows),
          dict(source=len(source_counts), input=len(input_rows)))
    for name, expected in meta.get("output_counts", {}).items():
        check(f"output_count:{name}", len(read_jsonl(run / name)) == expected,
              dict(expected=expected, observed=len(read_jsonl(run / name))))

    # Point-in-time candidate chronology.
    for filename, rows in (("early", early), ("confirmed", confirmed)):
        for row in rows:
            z, ref = row.get("zone", {}), row.get("reference", {})
            for field, left, right in (
                ("zone_available", z.get("available_index"), row.get("index")),
                ("reference_available", ref.get("available_index"), row.get("index")),
                ("touch", row.get("touch_index"), row.get("index")),
                ("low", row.get("low_index"), row.get("index")),
                ("recovery", row.get("recovery_index"), row.get("index")),
                ("breakout", row.get("breakout_index"), row.get("index")),
            ):
                check(f"{filename}:{field}", left is not None and right is not None and int(left) <= int(right),
                      dict(key=row.get("key"), left=left, right=right))
            for field in ("weekly_completed_at", "monthly_completed_at"):
                value = row.get("htf", {}).get(field)
                check(f"{filename}:{field}", value is None or date(value) <= date(row.get("at")),
                      dict(key=row.get("key"), value=value, at=row.get("at")))

    for row in sells:
        ref = row.get("reference", {})
        check("sell_reference_available", ref.get("available_index") is not None
              and int(ref["available_index"]) <= int(row["index"]), row)
    sell_sources = [x.get("source_id") for x in sells]
    check("selected_sell_source_unique", len(sell_sources) == len(set(sell_sources)), sell_sources)

    # Next-observed-open execution clock and input-date membership.
    intents = {x.get("intent", {}).get("id"): x.get("intent", {})
               for x in plans if x.get("intent")}
    for fill in fills:
        if fill.get("action") not in {"buy_fill", "sell_fill"}:
            continue
        intent = intents.get(fill.get("order_id"))
        check("fill_next_observed_open", intent is not None
              and int(fill["index"]) == int(intent["index"]) + 1,
              dict(fill=fill, intent=intent))
        check("fill_on_input_date", date(fill.get("at")) in input_index, fill)
    for trade in trades:
        check("trade_entry_on_input_date", date(trade.get("entry_at")) in input_index, trade)
        check("trade_exit_on_input_date", date(trade.get("exit_at")) in input_index, trade)
        same_day_entry_stop = (date(trade.get("entry_at")) == date(trade.get("exit_at"))
                               and trade.get("reason") == "initial_risk_stop")
        if same_day_entry_stop:
            checks["trade_same_day_initial_stop:allowed"] += 1
        else:
            check("trade_entry_before_exit", date(trade.get("entry_at")) < date(trade.get("exit_at")), trade)

    # Support observation must be strictly after its campaign boundary.
    support_boundaries = {}
    for row in support:
        action = row.get("action")
        if action == "support_observation_boundary":
            support_boundaries[row.get("budget_scope_id")] = row.get("after_index")
        if action in {"post_boundary_contact", "post_boundary_recovery"}:
            group_id = row.get("support_episode_id", "")
            scope = group_id.rsplit(":support:", 1)[0] if ":support:" in group_id else None
            boundary = support_boundaries.get(scope)
            check("support_boundary_present", boundary is not None,
                  dict(row=row, inferred_scope=scope))
            if boundary is not None:
                check("support_strictly_after_boundary", int(row["index"]) > int(boundary), row)
        if action == "fresh_support_local_offer":
            check("local_offer_support_after_boundary",
                  row.get("post_boundary_recovery_index") is not None, row)

    # Gate limits: actual fills, not offers/cancellations, are counted.
    by_scope = defaultdict(list)
    for row in gate:
        if row.get("budget_scope_id"):
            by_scope[row["budget_scope_id"]].append(row)
    scope_summary = {}
    for scope, rows in by_scope.items():
        filled = [x for x in rows if x.get("action") == "campaign_attempt_filled"]
        route_counts = Counter(x.get("route") for x in filled)
        closed = [x for x in rows if x.get("action") == "campaign_attempt_closed"]
        scope_summary[scope] = dict(filled=len(filled), route_counts=dict(route_counts),
                                    closed=len(closed), max_loss_r=max(
                                        [float(x.get("cumulative_loss_r", 0.0)) for x in closed] or [0.0]))
        check("scope_attempt_limit", len(filled) <= MAX_ATTEMPTS, dict(scope=scope, count=len(filled)))
        for route, cap in MAX_ROUTE_FILLS.items():
            check("scope_route_limit", route_counts.get(route, 0) <= cap,
                  dict(scope=scope, route=route, count=route_counts.get(route, 0)))
    fills_by_day = [date(x.get("at")) for x in fills if x.get("action") == "buy_fill"]
    sell_fills_by_day = [date(x.get("at")) for x in fills if x.get("action") == "sell_fill"]
    check("no_buy_on_sell_fill_day", not set(fills_by_day).intersection(sell_fills_by_day),
          dict(buy=fills_by_day, sell=sell_fills_by_day))

    reversal_starts = [x.get("reversal", {}).get("id") for x in reversals
                       if x.get("action") == "major_reversal_started"]
    check("reversal_id_unique", len(reversal_starts) == len(set(reversal_starts)), reversal_starts)
    check("reversal_starts_have_id", all(reversal_starts), reversal_starts)

    base_metrics = metrics(run, trades, mtm)
    result = dict(
        verification_status="REPLAY_INTEGRITY_CHECKED_NOT_STATISTICAL_VALIDATION",
        run_metadata=dict(implementation=meta.get("implementation"), run_status=meta.get("run_status"),
                          as_of=meta.get("as_of"), processed_bars=meta.get("processed_bars"),
                          campaign_config=meta.get("campaign_config")),
        input=input_info,
        metrics=base_metrics,
        checks=dict(checks),
        violation_count=len(violations),
        violations_sample=violations[:50],
        scope_summary=scope_summary,
        counts=dict(early=len(early), confirmed=len(confirmed), selected_sells=len(sells),
                    reversals=len(reversal_starts), support_events=len(support),
                    fills=len([x for x in fills if x.get("action") in {"buy_fill", "sell_fill"}])),
        final_state_summary=dict(open_position=final.get("open_position") is not None,
                                 right_censored=final.get("right_censored"),
                                 campaign_state=final.get("campaign_state")),
        limitations=[
            "This is a deterministic replay/integrity audit, not a walk-forward or holdout test.",
            "Daily OHLC cannot identify the intraday order of same-bar high/low/stop events.",
            "Continuous-futures roll, margin, funding and account-capital constraints are not modeled.",
            "The input-date calendar is observed-input coverage, not an independently certified CME calendar.",
            "A completed replay does not establish predictive superiority or statistical significance.",
        ],
    )
    if baseline is not None:
        old = read_json(baseline / "audit_rev02.json") if (baseline / "audit_rev02.json").exists() else None
        if old is None:
            old_trades = read_jsonl(baseline / "trades.jsonl")
            old_mtm = read_jsonl(baseline / "daily_mtm.jsonl")
            old = dict(metrics=metrics(baseline, old_trades, old_mtm))
        keys = ("completed_trades", "net_pnl_usd", "daily_mtm_final_usd",
                "realized_trade_curve_max_dd_usd", "daily_mtm_max_close_dd_usd")
        result["baseline_comparison"] = dict(baseline=str(baseline),
            baseline_metrics=old["metrics"], after_metrics=base_metrics,
            delta_after_minus_baseline={k: base_metrics.get(k) - old["metrics"].get(k)
                                        if base_metrics.get(k) is not None and old["metrics"].get(k) is not None
                                        else None for k in keys})
    return result


def render(report: dict) -> str:
    m, c = report["metrics"], report["checks"]
    failed = {k: v for k, v in c.items() if k.endswith(":fail")}
    lines = [
        "# 改訂0.2・キャンペーン実行検証",
        "",
        "## 判定",
        "",
        "これは固定ルールのリプレイ整合性監査です。将来予測の優位性、未見期間の有効性、確率校正、実口座収益性を証明する検証ではありません。",
        "",
        f"- 実装: `{report['run_metadata']['implementation']}`",
        f"- 入力: {report['input']['start']}〜{report['input']['end']} / {report['input']['selected_rows']}本",
        f"- 完走: `{report['run_metadata']['run_status']}`",
        f"- 整合性違反: {report['violation_count']}件",
        f"- 完了取引: {m['completed_trades']}件（勝ち{m['wins']}、負け{m['losses']}、勝率{m['win_rate_completed']:.3f}）",
        f"- 完了取引損益: {m['net_pnl_usd']:.2f} USD / PF {m['profit_factor']}",
        f"- 実現取引曲線DD: {m['realized_trade_curve_max_dd_usd']:.2f} USD",
        f"- 日次MTM最終値: {m['daily_mtm_final_usd']:.2f} USD / 日次DD: {m['daily_mtm_max_close_dd_usd']:.2f} USD",
        f"- 最終ポジション: {'あり（右側打切り）' if report['final_state_summary']['open_position'] else 'なし'}",
        "",
        "## 検知・売買の出力",
        "",
        f"- early候補 {report['counts']['early']}件、confirmed候補 {report['counts']['confirmed']}件",
        f"- 採用SELL {report['counts']['selected_sells']}件、主要反落サイクル {report['counts']['reversals']}件",
        f"- 支持反応イベント {report['counts']['support_events']}件、BUY/SELL約定イベント {report['counts']['fills']}件",
        "- 標準設定は `sell_detector=major_area`、`reentry_mode=fresh_support_same_episode`。",
        "- ダブルトップ検知器は標準実行では診断計算を行わず、別設定の短縮実行で動作確認しました。",
        "",
        "## 監査結果",
        "",
        f"- 失敗カウンタ: `{json.dumps(failed, ensure_ascii=False)}`",
        "- 候補のOB/Pivot利用可能時刻、週足・月足の完了時刻、翌観測日始値の約定時計、入力日付を確認対象にしました。",
        "- 支持反応は主要反落境界より後の観測として保存され、試行予算は市場episodeとは別のscopeで集計されます。",
        "- 同じscope内の実約定数とearly/confirmed経路数は設定上限を超えないことを確認対象にしました。",
        "",
        "## 重要な制限",
        "",
    ]
    lines.extend(f"- {x}" for x in report["limitations"])
    if report.get("baseline_comparison"):
        lines += ["", "## 旧0.1との数値比較", "", "旧0.1と改訂0.2は再BUY状態管理が異なるため、差は改善の証明ではなく、固定入力上の再生差です。", "", "```json", json.dumps(report["baseline_comparison"], ensure_ascii=False, indent=2), "```"]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    report = audit(args.run, args.baseline)
    (args.run / "audit_rev02.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.run / "report_rev02_ja.md").write_text(render(report), encoding="utf-8")
    print(json.dumps(dict(verification_status=report["verification_status"],
                          violation_count=report["violation_count"], metrics=report["metrics"],
                          failed_checks={k: v for k, v in report["checks"].items() if k.endswith(":fail")}),
               ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
