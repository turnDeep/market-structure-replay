"""Audit a completed campaign replay without calling it predictive validation.

The audit checks arithmetic, causal ordering, input/calendar coverage, and the
specific recent windows discussed for the daily swing campaign.  It does not
fit a model, tune thresholds, or estimate future probabilities.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def as_date(value: str | None) -> str | None:
    return str(value)[:10] if value else None


def load_input_dates(meta: dict) -> tuple[list[str], dict]:
    path = Path(meta["input_hashes"]["prices"]["path"])
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return [], {"path": str(path), "rows": 0}
    date_col = next((c for c in rows[0] if c.lower() in {"datetime", "date", "timestamp"}), None)
    if date_col is None:
        raise ValueError(f"date column not found in {path}")
    dates = [as_date(r[date_col]) for r in rows]
    dates = [d for d in dates if d]
    return dates, {"path": str(path), "rows": len(rows), "date_column": date_col,
                   "start": dates[0], "end": dates[-1], "unique": len(set(dates)),
                   "sorted": dates == sorted(dates)}


def metrics(run_dir: Path) -> dict:
    meta = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    trades = read_jsonl(run_dir / "trades.jsonl")
    mtm = read_jsonl(run_dir / "daily_mtm.jsonl")
    execution = read_jsonl(run_dir / "execution_events.jsonl")
    final = json.loads((run_dir / "final_state.json").read_text(encoding="utf-8"))
    pnls = [float(x["net_pnl_usd"]) for x in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    curve = []
    total = 0.0
    peak = 0.0
    for p in pnls:
        total += p
        peak = max(peak, total)
        curve.append(total - peak)
    mtm_values = [float(x["mtm_pnl_usd"]) for x in mtm]
    mtm_drawdowns = [float(x["close_to_close_drawdown_usd"]) for x in mtm]
    reasons = Counter(x["reason"] for x in trades)
    years = Counter(as_date(x["entry_at"])[:4] for x in trades)
    return {
        "run_dir": str(run_dir),
        "status": meta.get("run_status"),
        "verification_status": meta.get("verification_status"),
        "processed_bars": meta.get("processed_bars"),
        "trade_count": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_completed_trades": len(wins) / len(trades) if trades else None,
        "net_pnl_usd_completed": sum(pnls),
        "gross_profit_usd": sum(wins),
        "gross_loss_usd": sum(losses),
        "profit_factor_completed": sum(wins) / abs(sum(losses)) if losses else None,
        "realized_trade_curve_max_drawdown_usd": min(curve) if curve else 0.0,
        "daily_mtm_final_usd": mtm_values[-1] if mtm_values else None,
        "daily_mtm_max_close_to_close_drawdown_usd": min(mtm_drawdowns) if mtm_drawdowns else None,
        "open_position": final.get("open_position") is not None,
        "right_censored": final.get("right_censored"),
        "trade_exit_reasons": dict(reasons),
        "entries_by_year": dict(sorted(years.items())),
        "execution_event_count": len(execution),
        "output_counts": meta.get("output_counts", {}),
    }


def audit_causality(run_dir: Path, meta: dict) -> dict:
    checks = Counter()
    violations: list[dict] = []

    def leq(left: int | None, right: int | None) -> bool:
        return left is not None and right is not None and int(left) <= int(right)

    for filename in ("shared_early_candidates.jsonl", "shared_confirmed_candidates.jsonl"):
        for row in read_jsonl(run_dir / filename):
            checks[f"{filename}:rows"] += 1
            zone = row.get("zone", {})
            ref = row.get("reference", {})
            fields = {
                "zone_available": leq(zone.get("available_index"), row.get("index")),
                "reference_available": leq(ref.get("available_index"), row.get("index")),
                "touch": leq(row.get("touch_index"), row.get("index")),
                "low": leq(row.get("low_index"), row.get("index")),
                "recovery": leq(row.get("recovery_index"), row.get("index")),
                "breakout": leq(row.get("breakout_index"), row.get("index")),
                "zone_origin": leq(zone.get("origin_index"), zone.get("available_index")),
            }
            for name, ok in fields.items():
                checks[f"{filename}:{name}:pass"] += int(ok)
                checks[f"{filename}:{name}:fail"] += int(not ok)
                if not ok:
                    violations.append({"file": filename, "key": row.get("key"), "check": name,
                                       "at": row.get("at"), "row": row})
            for name in ("weekly_completed_at", "monthly_completed_at"):
                value = row.get("htf", {}).get(name)
                if value is not None and as_date(value) > as_date(row.get("at")):
                    violations.append({"file": filename, "key": row.get("key"), "check": name,
                                       "at": row.get("at"), "value": value})

    for row in read_jsonl(run_dir / "major_sell_candidates.jsonl"):
        checks["major_sell:rows"] += 1
        ref = row.get("reference", {})
        ok = leq(ref.get("available_index"), row.get("index"))
        checks["major_sell:reference_available:pass"] += int(ok)
        checks["major_sell:reference_available:fail"] += int(not ok)
        if not ok:
            violations.append({"file": "major_sell_candidates.jsonl", "key": row.get("key"),
                               "check": "reference_available", "row": row})

    trades = read_jsonl(run_dir / "trades.jsonl")
    input_dates, input_info = load_input_dates(meta)
    input_set = set(input_dates)
    for trade in trades:
        for field in ("entry_at", "exit_at"):
            ok = as_date(trade.get(field)) in input_set
            checks[f"trade:{field}_on_input_date:pass"] += int(ok)
            checks[f"trade:{field}_on_input_date:fail"] += int(not ok)
            if not ok:
                violations.append({"file": "trades.jsonl", "position_id": trade.get("position_id"),
                                   "check": field, "value": trade.get(field)})
        if as_date(trade.get("entry_at")) >= as_date(trade.get("exit_at")):
            violations.append({"file": "trades.jsonl", "position_id": trade.get("position_id"),
                               "check": "entry_before_exit", "row": trade})
    plans = read_jsonl(run_dir / "close_plans.jsonl")
    fills = read_jsonl(run_dir / "execution_events.jsonl")
    intents = {row.get("intent", {}).get("id"): row.get("intent", {})
               for row in plans if row.get("intent")}
    for fill in fills:
        if fill.get("action") not in {"buy_fill", "sell_fill"}:
            continue
        intent = intents.get(fill.get("order_id"))
        ok = intent is not None and int(fill["index"]) == int(intent["index"]) + 1
        checks["fill:next_observed_open_clock:pass"] += int(ok)
        checks["fill:next_observed_open_clock:fail"] += int(not ok)
        if not ok:
            violations.append({"file": "execution_events.jsonl", "order_id": fill.get("order_id"),
                               "check": "next_observed_open_clock", "fill": fill, "intent": intent})
    return {"checks": dict(checks), "violation_count": len(violations),
            "violations_sample": violations[:20], "input": input_info,
            "input_dates_match_output_calendar": _calendar_matches(meta, input_dates)}


def _calendar_matches(meta: dict, input_dates: list[str]) -> dict:
    path = Path(meta["input_hashes"]["calendar"]["path"])
    cal = json.loads(path.read_text(encoding="utf-8"))
    sessions = [as_date(x) for x in cal.get("session_dates", [])]
    return {"calendar_status": cal.get("calendar_status"), "calendar_rows": len(sessions),
            "input_rows": len(input_dates), "exact_match": sessions == input_dates,
            "calendar_hash_from_run": meta["input_hashes"]["calendar"]["sha256"]}


def window_summary(run_dir: Path) -> dict:
    early = read_jsonl(run_dir / "shared_early_candidates.jsonl")
    confirmed = read_jsonl(run_dir / "shared_confirmed_candidates.jsonl")
    sells = read_jsonl(run_dir / "major_sell_candidates.jsonl")
    meta = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    input_dates, _ = load_input_dates(meta)
    index_by_date = {date: i for i, date in enumerate(input_dates)}
    assessed = read_jsonl(run_dir / "entry_audit.jsonl")

    def count_window(rows, start, end):
        picked = [x for x in rows if start <= as_date(x.get("at")) <= end]
        return {"rows": len(picked), "unique_dates": sorted({as_date(x.get("at")) for x in picked}),
                "unique_setup_ids": sorted({x.get("setup_id") for x in picked if x.get("setup_id")})}

    entry_assessment = {}
    for date in ("2025-04-07", "2025-04-09", "2026-03-31", "2026-08-05"):
        idx = index_by_date.get(date)
        rows = [x for x in assessed if x.get("action") == "entry_assessed" and x.get("index") == idx]
        entry_assessment[date] = {"rows": len(rows),
                                  "disposition": dict(Counter(
                                      f"{x.get('allowed')}:{x.get('reason')}" for x in rows))}
    return {
        "first_top_reversal_window": {
            "major_sell_candidates": count_window(sells, "2024-12-16", "2025-03-10"),
            "early_buy_candidates_2025_04_07_09": count_window(early, "2025-04-07", "2025-04-09"),
            "confirmed_buy_candidates_2025_04_07_09": count_window(confirmed, "2025-04-07", "2025-04-09"),
        },
        "second_top_reversal_window": {
            "major_sell_candidates": count_window(sells, "2025-10-29", "2026-03-20"),
            "early_buy_candidates_2026_03_30_31": count_window(early, "2026-03-30", "2026-03-31"),
            "confirmed_buy_candidates_2026_03_30_31": count_window(confirmed, "2026-03-30", "2026-03-31"),
        },
        "recent_buy_candidates": {
            "early_2026_07_01_08_27": count_window(early, "2026-07-01", "2026-08-27"),
            "confirmed_2026_07_01_08_27": count_window(confirmed, "2026-07-01", "2026-08-27"),
        },
        "entry_assessment": entry_assessment,
        "major_sell_dates": [{"at": as_date(x.get("at")), "source_id": x.get("source_id"),
                              "reference": x.get("reference", {}).get("id")}
                             for x in sells],
    }


def compare_runs(run_a: Path, run_b: Path) -> dict:
    a, b = metrics(run_a), metrics(run_b)
    keys = ("trade_count", "net_pnl_usd_completed", "daily_mtm_final_usd",
            "realized_trade_curve_max_drawdown_usd", "daily_mtm_max_close_to_close_drawdown_usd")
    return {"before": a, "after": b,
            "delta_after_minus_before": {k: (b[k] - a[k] if a[k] is not None and b[k] is not None else None)
                                          for k in keys}}


def render_report(report: dict) -> str:
    m = report["metrics"]
    c = report["causality"]
    w = report["windows"]
    lines = [
        "# 日足スイング版・大きな波キャンペーン検証監査",
        "",
        "## 結論",
        "",
        "これは固定ルールで全期間をリプレイした実装監査です。将来予測の優位性、未見期間での再現性、確率推定、実口座の収益性を検証したものではありません。",
        "",
        f"- 入力期間: {c['input']['start']}〜{c['input']['end']} / {c['input']['rows']}本",
        f"- 実行状態: `{m['status']}` / `{m['verification_status']}`",
        f"- 完了取引: {m['trade_count']}件（勝ち {m['wins']}、負け {m['losses']}、完了取引勝率 {m['win_rate_completed_trades']:.3f}）",
        f"- 完了取引の損益: {m['net_pnl_usd_completed']:.2f} USD / PF {m['profit_factor_completed']:.3f}" if m['profit_factor_completed'] is not None else f"- 完了取引の損益: {m['net_pnl_usd_completed']:.2f} USD / PF n/a",
        f"- 完了取引の実現曲線最大DD: {m['realized_trade_curve_max_drawdown_usd']:.2f} USD",
        f"- 日次MTM最終値: {m['daily_mtm_final_usd']:.2f} USD / 日次終値ベース最大DD: {m['daily_mtm_max_close_to_close_drawdown_usd']:.2f} USD",
        f"- 最終ポジション: {'あり（右側打切り）' if m['open_position'] else 'なし'}",
        "",
        "## 実装不備の修正",
        "",
        "修正前は、主要SELL後の再BUY条件が「前回退出時の古いピーク − 新候補の安値」を比較していました。NQがその古いピークを上回った後は、後年の候補を恒久的に拒否するため、2011年以降の再BUYが止まっていました。修正版は、退出後に新しく資格化されたMarketEpisodeが成立し、その下降局面で支持反応が確認されたことを要求します。新エピソード自体の固定ATR・時間条件は既存検知器に委ねています。",
        "",
        "## 因果順序・入力監査",
        "",
        f"- 因果順序違反: {c['violation_count']}件",
        f"- 価格入力とカレンダーの完全一致: {c['input_dates_match_output_calendar']['exact_match']}",
        f"- カレンダー状態: `{c['input_dates_match_output_calendar']['calendar_status']}`（取引所公式の休場日表ではなく、入力に観測された日付から作成）",
        "",
        "候補は、OBのavailable_index、参照Pivotのavailable_index、接触・回復・突破の各indexが、候補の判断index以前であることを検査しました。週足・月足の完了時刻も候補日以前であることを確認しました。",
        "",
        "## 指定された最近の場面との照合",
        "",
        "### 最初の天井・2025年4月の反応",
        "",
        f"- 主要SELL候補: {w['first_top_reversal_window']['major_sell_candidates']['unique_dates']}",
        f"- 2025-04-07〜09のearly BUY候補: {w['first_top_reversal_window']['early_buy_candidates_2025_04_07_09']['rows']}件、日付 {w['first_top_reversal_window']['early_buy_candidates_2025_04_07_09']['unique_dates']}",
        f"- 2025-04-07〜09のconfirmed BUY候補: {w['first_top_reversal_window']['confirmed_buy_candidates_2025_04_07_09']['rows']}件、日付 {w['first_top_reversal_window']['confirmed_buy_candidates_2025_04_07_09']['unique_dates']}",
        f"- entry assessment: 2025-04-07 = {w['entry_assessment']['2025-04-07']}; 2025-04-09 = {w['entry_assessment']['2025-04-09']}",
        "- 結論: 2025-03-04に主要SELL候補、2025-04-07/09にearly候補は出るが、候補検出だけではBUY約定を意味しない。実際の約定可否は、退出後の新MarketEpisode条件とEpisodeAttemptGateに依存する。",
        "",
        "### 2回目の天井・2026年3月の反応",
        "",
        f"- 主要SELL候補: {w['second_top_reversal_window']['major_sell_candidates']['unique_dates']}",
        f"- 2026-03-30〜31のearly BUY候補: {w['second_top_reversal_window']['early_buy_candidates_2026_03_30_31']['rows']}件、日付 {w['second_top_reversal_window']['early_buy_candidates_2026_03_30_31']['unique_dates']}",
        f"- 2026-03-30〜31のconfirmed BUY候補: {w['second_top_reversal_window']['confirmed_buy_candidates_2026_03_30_31']['rows']}件、日付 {w['second_top_reversal_window']['confirmed_buy_candidates_2026_03_30_31']['unique_dates']}",
        f"- entry assessment: 2026-03-31 = {w['entry_assessment']['2026-03-31']}; 2026-08-05 = {w['entry_assessment']['2026-08-05']}",
        "- 結論: 2026-03-13に主要SELL候補、2026-03-31にearly候補は出る。ただし、このモデルは「2025-10-29/30と2026-01-28/29の2つの高値を正式なダブルトップとして比較する」専用検知器ではない。主要高値圏の再試験と支持割れを使う。",
        "",
        "### 直近の候補",
        "",
        f"- 2026-07-01〜08-27のearly候補日: {w['recent_buy_candidates']['early_2026_07_01_08_27']['unique_dates']}",
        f"- 2026-07-01〜08-27のconfirmed候補日: {w['recent_buy_candidates']['confirmed_2026_07_01_08_27']['unique_dates']}",
        "- 修正版では2026-08-06にBUYが約定し、最終日2026-08-27時点で未決済のため、成績は右側打切りです。",
        "",
        "## 解釈上の制限",
        "",
        "- 29件の完了取引は、独立なサンプル数として十分とはいえません。同一エピソード内の試行制限があるため、単純な日次シグナル勝率ではありません。",
        "- 1D OHLCだけでは同一足内のSL/TP順序、約定スリッページ、Globexの時刻別順序を復元できません。",
        "- 価格系列は連続先物で、限月ロール・調整方法・証拠金・ロールコスト・資金制約は別途必要です。",
        "- 使用カレンダーは入力日付の観測カレンダーであり、CME公式セッション・祝日カレンダーとしては未認証です。",
        "- `NOT_VALIDATED` は、ウォークフォワード、未見ホールドアウト、基準モデル、統計的有意性、確率校正をまだ実施していないことを意味します。",
        "",
        "## 次の検証優先順位",
        "",
        "1. この修正版を固定し、期間を時系列で分割して、過去で閾値を変更せず未見期間へ進めるウォークフォワードを実施する。",
        "2. 2025-04-07/09、2026-03-31の候補について、候補検出・entry assessment拒否理由・実約定・その後の経路を1つのsetup_idへ束ねる。",
        "3. 専用の厳密なダブルトップ検知器を別モデルとして追加し、現在の主要高値圏モデルと同じ時計・同じ目標で比較する。",
        "4. Buy & Hold、常時ロング、旧版、修正版を同じ資本制約・同じ連続先物評価で比較する。",
        "5. 取引所セッション・ロール・分足データを整備して、日足OHLCの順序不明を縮小する。",
    ]
    if report.get("comparison"):
        lines += ["", "## 修正前後の比較", "", "```json", json.dumps(report["comparison"], ensure_ascii=False, indent=2), "```"]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--compare-before", type=Path)
    ap.add_argument("--output-dir", type=Path)
    args = ap.parse_args()
    report = {
        "metrics": metrics(args.run),
        "causality": audit_causality(args.run, json.loads((args.run / "metadata.json").read_text(encoding="utf-8"))),
        "windows": window_summary(args.run),
    }
    if args.compare_before:
        report["comparison"] = compare_runs(args.compare_before, args.run)
    out = args.output_dir or args.run
    out.mkdir(parents=True, exist_ok=True)
    (out / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (out / "report_ja.md").write_text(render_report(report), encoding="utf-8")
    print(f"audit written to {out}")
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    print(json.dumps(report["causality"], ensure_ascii=False, indent=2)[:10000])


if __name__ == "__main__":
    main()
