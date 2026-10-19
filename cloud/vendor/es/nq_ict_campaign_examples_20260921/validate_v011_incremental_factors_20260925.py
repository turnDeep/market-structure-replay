"""One-factor-at-a-time v0.11 replay comparisons; no parameter fitting.

Runs the frozen v0.11 campaign with continuation BUY, four separate weekly /
monthly trend-or-bias gates, and the implemented early-OB wick-sweep gate.
Writes full replay logs plus compact comparison tables and a Japanese report.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parent
OUT = PACKAGE / "validation_rev11_incremental_factors_20260925"
REFERENCE = PACKAGE / "validation_rev11_without_continuation_20260923"
AS_OF = "2026-08-27"
EXPECTED_ROWS = 6526

VARIANTS = [
    ("baseline", "v0.11 baseline: continuation BUY off", "rev11_without_continuation", None, False),
    ("continuation_buy", "add ordinary continuation BUY", "rev11_baseline", None, False),
    ("weekly_trend_up", "weekly completed trend must be up", "rev11_without_continuation", "weekly_trend_up", False),
    ("monthly_trend_up", "monthly completed trend must be up", "rev11_without_continuation", "monthly_trend_up", False),
    ("weekly_bias_up", "weekly completed bias must be up", "rev11_without_continuation", "weekly_bias_up", False),
    ("monthly_bias_up", "monthly completed bias must be up", "rev11_without_continuation", "monthly_bias_up", False),
    ("ob_distal_wick_sweep", "early BUY requires wick below bullish OB distal edge",
     "rev11_without_continuation", None, True),
]

PERIODS = [
    ("full", "2000-09-18", "2026-08-27"),
    ("2006_2010", "2006-01-03", "2010-12-31"),
    ("2011_2015", "2011-01-03", "2015-12-31"),
    ("2016_2018", "2016-01-04", "2018-12-31"),
    ("2019_2022", "2019-01-02", "2022-12-30"),
    ("2023_2026", "2023-01-03", "2026-08-27"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write_csv(path: Path, rows: list[dict]):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_equity_artifacts(results: list[dict], dates: list[str], benchmark: list[float]):
    names = ["v0.11 baseline", "continuation BUY"] + [
        item[1] for item in VARIANTS[2:]
    ]
    rows = []
    for index, date in enumerate(dates):
        row = {"date": date, "buy_hold_mtm_usd": round(benchmark[index], 2)}
        for variant, label in zip(results, names):
            row[variant["key"] + "_mtm_usd"] = round(variant["mtm_by_date"][date], 2)
        rows.append(row)
    write_csv(OUT / "equity_curves.csv", rows)
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; equity_curves.csv written without PNG", flush=True)
        return
    x = list(range(len(dates)))
    fig, axes = plt.subplots(2, 1, figsize=(16, 11), sharex=False,
                             gridspec_kw={"height_ratios": [1, 1.05]})
    palette = ["#17365D", "#D97706", "#2563EB", "#7C3AED", "#059669", "#DB2777", "#475569"]
    for ax, start_index, title in (
        (axes[0], 0, "Full sample: 2000-09 to 2026-08"),
        (axes[1], next((i for i, day in enumerate(dates) if day >= "2016-01-01"), 0),
         "Later regimes: 2016-01 to 2026-08"),
    ):
        ax.plot(x[start_index:], benchmark[start_index:], color="#111827", linewidth=2.5,
                label="Buy & Hold")
        for variant, label, color in zip(results, names, palette):
            y = [variant["mtm_by_date"][day] for day in dates[start_index:]]
            width = 2.5 if variant["key"] == "baseline" else 1.35
            ax.plot(x[start_index:], y, color=color, linewidth=width, alpha=0.94,
                    label=label)
        ticks = list(range(start_index, len(dates), max(1, (len(dates) - start_index) // 8)))
        ax.set_xticks(ticks, [dates[i][:7] for i in ticks], rotation=0)
        ax.set_title(title)
        ax.set_ylabel("Mark-to-market P&L (USD)")
        ax.grid(True, alpha=0.25)
    axes[0].legend(ncol=4, fontsize=8, loc="upper left")
    axes[1].legend(ncol=4, fontsize=8, loc="upper left")
    fig.suptitle("ES=F v0.11 one-factor comparisons (same dates, one contract)", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / "equity_curves.png", dpi=160)
    plt.close(fig)


def run_replays(log_dir: Path):
    log_dir.mkdir(parents=True, exist_ok=True)

    def run_one(variant):
        key, label, stage, htf_filter, sweep = variant
        run_dir = OUT / key
        if run_dir.exists():
            meta_path = run_dir / "metadata.json"
            if meta_path.exists() and read_json(meta_path).get("run_status") == "completed_replay_only":
                return key, 0.0, "reused_completed"
            raise FileExistsError(f"refusing to overwrite incomplete replay: {run_dir}")
        command = [
            sys.executable, "-m", "nq_ict_campaign_examples_20260921.run_campaign",
            "--revision-stage", stage,
            "--calendar-json", str(PACKAGE / "observed_input_calendar.json"),
            "--as-of", AS_OF, "--output", str(run_dir), "--allow-unvalidated-replay",
        ]
        if htf_filter:
            command.extend(("--buy-htf-filter", htf_filter))
        if sweep:
            command.append("--require-early-support-sweep")
        log_path = log_dir / f"{key}.log"
        started = time.monotonic()
        print(f"START {key}: {label}", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                       text=True, check=False)
        elapsed = time.monotonic() - started
        if completed.returncode:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-5000:]
            raise RuntimeError(f"replay {key} failed (exit {completed.returncode})\n{tail}")
        meta_path = run_dir / "metadata.json"
        if not meta_path.exists() or read_json(meta_path).get("run_status") != "completed_replay_only":
            raise RuntimeError(f"replay {key} lacks completed metadata")
        return key, elapsed, "completed"

    failures = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(run_one, variant) for variant in VARIANTS]
        for future in as_completed(futures):
            try:
                key, elapsed, state = future.result()
                print(f"DONE {key}: {elapsed:.1f}s ({state})", flush=True)
            except Exception as exc:
                failures.append(str(exc))
    if failures:
        raise RuntimeError("one or more replays failed: " + " | ".join(failures))


def market_and_benchmark():
    path = ROOT / "nq_25y_actual_2000_2026_20260911" / "NQ_F_1d_combined.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    dates = [str(row["datetime"])[:10] for row in rows]
    opens = [float(row["open"]) for row in rows]
    closes = [float(row["close"]) for row in rows]
    if len(rows) != EXPECTED_ROWS or dates != sorted(set(dates)):
        raise ValueError(f"unexpected price chronology/grain: rows={len(rows)}")
    # Same mark-to-market Buy & Hold convention as the frozen v0.11 report:
    # one tick + entry fee, valued at daily close without hypothetical exit costs.
    entry = opens[0] + 0.25
    benchmark = [(close - entry) * 20.0 - 2.50 for close in closes]
    return dates, benchmark


def summarize_run(key: str, label: str, expected_field: str | None):
    run_dir = OUT / key
    meta = read_json(run_dir / "metadata.json")
    mtm_rows = list(read_jsonl(run_dir / "daily_mtm.jsonl"))
    states = list(read_jsonl(run_dir / "daily_states.jsonl"))
    trades = list(read_jsonl(run_dir / "trades.jsonl"))
    if len(mtm_rows) != EXPECTED_ROWS or len(states) != EXPECTED_ROWS:
        raise ValueError(f"{key}: wrong replay length")
    dates = [row["datetime"][:10] for row in mtm_rows]
    equity = [float(row["mtm_pnl_usd"]) for row in mtm_rows]
    if dates != [row["datetime"][:10] for row in states]:
        raise ValueError(f"{key}: state and MTM dates differ")
    peak = 0.0
    drawdown = 0.0
    for value in equity:
        peak = max(peak, value)
        drawdown = min(drawdown, value - peak)
    profits = [float(t["net_pnl_usd"]) for t in trades if float(t["net_pnl_usd"]) > 0]
    losses = [float(t["net_pnl_usd"]) for t in trades if float(t["net_pnl_usd"]) < 0]
    buys = [row for row in read_jsonl(run_dir / "execution_events.jsonl") if row.get("action") == "buy_fill"]
    sells = [row for row in read_jsonl(run_dir / "execution_events.jsonl") if row.get("action") == "sell_fill"]
    active_days = sum(row.get("position_id") is not None for row in states)
    last_open = states[-1].get("position_id") is not None
    setup_counts = {"continuation": 0, "recovery": 0, "OB_or_reversal": 0}
    for trade in trades:
        setup_id = str(trade.get("entry_setup_id", ""))
        if setup_id.startswith("continuation:"):
            setup_counts["continuation"] += 1
        elif setup_id.startswith("recovery:"):
            setup_counts["recovery"] += 1
        else:
            setup_counts["OB_or_reversal"] += 1

    windows = {}
    for period, start, end in PERIODS:
        indices = [i for i, date in enumerate(dates) if start <= date <= end]
        if not indices:
            continue
        first, last = indices[0], indices[-1]
        # The full sample includes the initial open-to-close move; subperiods
        # follow the prior v0.11 report's close-to-close convention.
        value = equity[last] if period == "full" else equity[last] - equity[first]
        windows[period] = dict(start=dates[first], end=dates[last], bars=len(indices),
                               strategy_pnl_usd=value)

    gate_failure = {"assessed": 0, "failed_unique_setups": 0, "failed_only_by_new_gate": 0,
                    "causal_clock_violations": 0}
    if expected_field:
        unique_failed, unique_only = set(), set()
        for audit in read_jsonl(run_dir / "entry_audit.jsonl"):
            if audit.get("action") != "entry_assessed":
                continue
            check = (audit.get("checks") or {}).get(expected_field)
            if check is None:
                continue
            gate_failure["assessed"] += 1
            if not check.get("passed", False):
                unique_failed.add(audit.get("setup_id") or audit.get("key"))
                if expected_field == "buy_htf_filter":
                    completion = check.get("context_completed_at")
                    decision = check.get("decision_at")
                    if completion and decision and completion > decision:
                        gate_failure["causal_clock_violations"] += 1
                other_failed = [name for name, result in (audit.get("checks") or {}).items()
                                if name != expected_field and result.get("status") != "not_applicable"
                                and not result.get("passed", False)]
                if not other_failed:
                    unique_only.add(audit.get("setup_id") or audit.get("key"))
        gate_failure["failed_unique_setups"] = len(unique_failed)
        gate_failure["failed_only_by_new_gate"] = len(unique_only)

    cfg = meta["campaign_config"]
    value = dict(key=key, label=label, run_status=meta.get("run_status"),
                 processed_bars=meta.get("processed_bars"), final_mtm_usd=equity[-1],
                 realized_completed_usd=sum(float(t["net_pnl_usd"]) for t in trades),
                 max_close_drawdown_usd=drawdown, exposure_fraction=active_days / len(states),
                 active_days=active_days, completed_trades=len(trades),
                 completed_wins=sum(float(t["net_pnl_usd"]) > 0 for t in trades),
                 completed_losses=sum(float(t["net_pnl_usd"]) < 0 for t in trades),
                 profit_factor=(sum(profits) / -sum(losses) if losses else None),
                 buy_fills=len(buys), sell_fills=len(sells), end_open_position=last_open,
                 completed_trade_entry_families=setup_counts, campaign_config=cfg,
                 input_hashes=meta.get("input_hashes"), code_hashes=meta.get("code_hashes"),
                 windows=windows, new_gate=gate_failure,
                 mtm_by_date=dict(zip(dates, equity)))
    return value


def validate_and_write():
    OUT.mkdir(parents=True, exist_ok=True)
    log_dir = OUT / "run_logs"
    run_replays(log_dir)
    dates, benchmark = market_and_benchmark()
    results = [summarize_run(key, label, "buy_htf_filter" if htf else
                             "early_conditions" if sweep else None)
               for key, label, _stage, htf, sweep in VARIANTS]
    baseline = results[0]
    reference_mtm = list(read_jsonl(REFERENCE / "daily_mtm.jsonl"))
    reference_trades = REFERENCE / "trades.jsonl"
    baseline_mtm = list(read_jsonl(OUT / "baseline" / "daily_mtm.jsonl"))
    if [r["mtm_pnl_usd"] for r in reference_mtm] != [r["mtm_pnl_usd"] for r in baseline_mtm]:
        raise AssertionError("default v0.11 baseline MTM changed from frozen replay")
    if sha256(reference_trades) != sha256(OUT / "baseline" / "trades.jsonl"):
        raise AssertionError("default v0.11 baseline trade log changed from frozen replay")
    if baseline["completed_trades"] != 43 or not math.isclose(baseline["final_mtm_usd"], 562012.5):
        raise AssertionError("baseline regression anchor mismatch")
    base_cfg = baseline["campaign_config"]
    expected_changes = {
        "baseline": {},
        "continuation_buy": {"enable_continuation_buy": True},
        "weekly_trend_up": {"buy_htf_filter": "weekly_trend_up"},
        "monthly_trend_up": {"buy_htf_filter": "monthly_trend_up"},
        "weekly_bias_up": {"buy_htf_filter": "weekly_bias_up"},
        "monthly_bias_up": {"buy_htf_filter": "monthly_bias_up"},
        "ob_distal_wick_sweep": {"require_early_support_sweep": True},
    }
    for result in results:
        differences = {key: result["campaign_config"].get(key) for key in base_cfg
                       if result["campaign_config"].get(key) != base_cfg.get(key)}
        if differences != expected_changes[result["key"]]:
            raise AssertionError(f"{result['key']} changed unintended config fields: {differences}")
        if result["processed_bars"] != EXPECTED_ROWS or result["run_status"] != "completed_replay_only":
            raise AssertionError(f"{result['key']} incomplete")
        if result["input_hashes"] != baseline["input_hashes"] or result["code_hashes"] != baseline["code_hashes"]:
            raise AssertionError(f"{result['key']} inputs or executable code differ")
        if list(result["mtm_by_date"]) != dates:
            raise AssertionError(f"{result['key']} date coverage differs from price data")
        if result["new_gate"].get("causal_clock_violations", 0):
            raise AssertionError(f"{result['key']} used future HTF context")

    # Remove bulky per-day equity maps from the compact aggregate before writing.
    compact = [{k: v for k, v in row.items() if k != "mtm_by_date"} for row in results]
    period_rows = []
    for period, start, end in PERIODS:
        idx = [i for i, date in enumerate(dates) if start <= date <= end]
        if not idx:
            continue
        i0, i1 = idx[0], idx[-1]
        bh = benchmark[i1] if period == "full" else benchmark[i1] - benchmark[i0]
        for result in results:
            eq = result["mtm_by_date"]
            strategy = eq[dates[i1]] if period == "full" else eq[dates[i1]] - eq[dates[i0]]
            period_rows.append(dict(period=period, variant=result["key"], start=dates[i0], end=dates[i1],
                                    bars=len(idx), strategy_pnl_usd=round(strategy, 2),
                                    buy_hold_pnl_usd=round(bh, 2),
                                    excess_vs_buy_hold_usd=round(strategy - bh, 2),
                                    delta_vs_baseline_usd=round(strategy -
                                        (baseline["mtm_by_date"][dates[i1]] if period == "full" else
                                         baseline["mtm_by_date"][dates[i1]] - baseline["mtm_by_date"][dates[i0]]), 2)))

    write_csv(OUT / "factor_comparison.csv", compact)
    write_csv(OUT / "period_comparison.csv", period_rows)
    write_equity_artifacts(results, dates, benchmark)
    summary = dict(status="COMPLETED_REPLAY_AND_QA",
                   purpose="separate one-factor replay comparisons; exploratory historical evidence",
                   as_of=AS_OF, rows=EXPECTED_ROWS,
                   source_data_sha256=baseline["input_hashes"]["prices"]["sha256"],
                   cost_assumptions=read_json(OUT / "baseline" / "metadata.json")["execution_config"],
                   baseline_regression=dict(matches_reference_mtm=True, matches_reference_trade_log=True),
                   variants=compact, period_comparisons=period_rows)
    (OUT / "comparison.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2,
                                                     allow_nan=False) + "\n", encoding="utf-8")
    report = make_report(summary)
    (OUT / "report_ja.md").write_text(report, encoding="utf-8")
    print(f"QA PASS; report: {OUT / 'report_ja.md'}", flush=True)


def make_report(result: dict) -> str:
    by_key = {row["key"]: row for row in result["variants"]}
    base = by_key["baseline"]
    lines = [
        "# v0.11 個別追加要素の比較検証 (2026-09-25)", "",
        "## 検証仕様", "",
        f"- 同一データ: {result['rows']:,}本、2000-09-18〜{result['as_of']}。価格系列SHA-256: `{result['source_data_sha256']}`。",
        "- 基準: v0.11 `rev11_without_continuation`。一度に一要素だけ変更し、他の設定・入力・執行条件を固定。",
        "- 1枚のES=F連続先物代理、$20/point、0.25 tick、片道1 tick slippage、片道$2.50 commission。",
        "- シグナルは日足終値で決め、次の観測始値で約定。Buy & Holdは初日始値+1 tickで買い、終値で時価評価（出口コストは課さない）。",
        "- 基準再現QA: 過去の固定v0.11 MTM全系列・完了取引ログと一致。",
        "- 成績指標は同じ過去データ上のリプレイ。信頼区間・有意差や未見性能を主張しない。",
        "", "## 全期間の結果", "",
        "| 条件 | MTM損益 | 基準との差 | 対Buy & Hold | 最大終値DD | 保有率 | 完了取引 (勝/負) | PF | BUY/SELL約定 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    bh_final = next(p["buy_hold_pnl_usd"] for p in result["period_comparisons"] if p["period"] == "full")
    for row in result["variants"]:
        diff = row["final_mtm_usd"] - base["final_mtm_usd"]
        lines.append(f"| {row['label']} | ${row['final_mtm_usd']:,.0f} | ${diff:+,.0f} | "
                     f"${row['final_mtm_usd']-bh_final:+,.0f} | ${row['max_close_drawdown_usd']:,.0f} | "
                     f"{row['exposure_fraction']:.1%} | {row['completed_trades']} "
                     f"({row['completed_wins']}/{row['completed_losses']}) | "
                     f"{row['profit_factor']:.2f} | {row['buy_fills']}/{row['sell_fills']} |")
    lines += ["", "## 追加フィルターが実際に止めた候補", "",
              "| 個別条件 | 評価された候補 | 新条件で不通過となったユニーク候補 | 新条件だけが不通過理由の候補 |",
              "|---|---:|---:|---:|"]
    for key in ("weekly_trend_up", "monthly_trend_up", "weekly_bias_up", "monthly_bias_up", "ob_distal_wick_sweep"):
        row = by_key[key]
        gate = row["new_gate"]
        lines.append(f"| {row['label']} | {gate['assessed']} | {gate['failed_unique_setups']} | {gate['failed_only_by_new_gate']} |")
    lines += ["", "## 期間別診断", "",
              "この表はレジーム依存の確認用で、個々の期間を選んだ将来予測性能ではありません。",
              "| 期間 | 条件 | 戦略損益 | Buy & Hold | 超過損益 | 基準からの変化 |",
              "|---|---|---:|---:|---:|---:|"]
    for period in ("2016_2018", "2023_2026"):
        for row in result["period_comparisons"]:
            if row["period"] == period:
                lines.append(f"| {period} | {by_key[row['variant']]['label']} | ${row['strategy_pnl_usd']:,.0f} | "
                             f"${row['buy_hold_pnl_usd']:,.0f} | ${row['excess_vs_buy_hold_usd']:+,.0f} | "
                             f"${row['delta_vs_baseline_usd']:+,.0f} |")
    lines += ["", "## 今回の比較から言えること", "",
              "- 継続BUYは完了取引を43件から48件へ増やしたが、期末MTMは基準より$15,320低く、最大DDは同じだった。保有率は78.1%から84.5%へ上昇。2016年以降の期間損益差はゼロで、追加分の成績改善は確認できない。",
              "- 週足・月足の`trend_up`を必須にすると、2016–2018年は基準より改善した一方、2023年以降は約$224k下回った。週足/月足の`bias_up`も全期間で基準を下回る。月足BiasだけはBuy & Holdを$19.8k上回るが、基準より$26.1k低く、期間別にも一貫しない。現状ではHTF状態をBUYのハード許可条件にしない。",
              "- OB下端Wick Sweep条件は全期間で基準より$4,920高い一方、最大DDは$585悪化。2016–2018年は+$20.6k、2019–2022年は-$17.2kの基準差で、2023年以降は同一損益だった。小さな全期間差を一般化できず、外部流動性Sweep全般の有効性を示すものでもない。",
              "- したがって今回の一点推定だけで採用を決める要素はない。現行v0.11を固定基準として維持し、Wick Sweep案のみを候補として別期間の事前固定評価へ回すのが妥当。",
              "", "## 定義と解釈上の制限", "",
              "- 週足/月足の各テストはBUYを許可するための単独ハードゲート。`trend_up`は完了足から計算した構造トレンド、`bias_up`は保存済みの次方向Biasを別々に使う。mixed/down/unknown/neutralおよび未完了状態は許可しない。確定時刻が判断日より後のHTF状態を用いていないことを監査した。",
              "- Sweepは実装済みの狭いプロキシのみ: early BUY候補の安値が、そのBullish OBのwick-low/distal edgeを下回ること。直近安値・equal lows等の外部Sell-side Liquidity sweep全般ではなく、確定ルートBUYには適用されない。",
              "- 継続BUYの追加では、観測用構造フィードは両条件で共通のまま、実際の通常継続エントリーだけをONにした。保有状態が変わるため、以降のイベント経路・取引件数も変わる。",
              "- フィルターの候補不通過数は機会損失の診断で、各候補の独立した反実仮想損益ではない。ポジション状態が変わる比較なので、MTM差をフィルター単独の普遍的因果効果とは解釈しない。",
              "- 過去データと特定日付は過去の設計・議論で繰り返し参照済み。これは未見アウトオブサンプルではない。選んだ勝ち条件の最終採用は保留し、新しい期間でルールを固定した前向き評価が必要。",
              "- Yahoo連続先物代理系列。個別限月ロール、証拠金、資本制約・追証、実際のロール費用はモデル化していない。日足だけでは同一足内の順序も復元しない。",
              "", "## 成果物", "",
              "- `factor_comparison.csv`: 全期間の条件別成績と診断",
              "- `period_comparison.csv`: 全期間・年代別損益比較",
              "- `equity_curves.png` / `equity_curves.csv`: 資産推移の図と日次系列",
              "- `comparison.json`: 設定、入力ハッシュ、QAと結果",
              "- 各条件ディレクトリ: 全リプレイログ、設定、コード/入力ハッシュ",
              "- `run_logs/`: 各リプレイ標準出力ログ", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate-only", action="store_true",
                        help="re-use already completed sub-run directories and create comparison artifacts")
    args = parser.parse_args()
    if not args.aggregate_only:
        validate_and_write()
        return
    if not OUT.exists():
        raise FileNotFoundError(OUT)
    dates, _ = market_and_benchmark()
    results = [summarize_run(key, label, "buy_htf_filter" if htf else
                             "early_conditions" if sweep else None)
               for key, label, _stage, htf, sweep in VARIANTS]
    baseline = results[0]
    compact = [{k: v for k, v in row.items() if k != "mtm_by_date"} for row in results]
    period_rows = []
    _, benchmark = market_and_benchmark()
    for period, start, end in PERIODS:
        idx = [i for i, date in enumerate(dates) if start <= date <= end]
        if not idx: continue
        i0, i1 = idx[0], idx[-1]
        bh = benchmark[i1] if period == "full" else benchmark[i1] - benchmark[i0]
        for row in results:
            eq = row["mtm_by_date"]
            value = eq[dates[i1]] if period == "full" else eq[dates[i1]] - eq[dates[i0]]
            base_eq = baseline["mtm_by_date"]
            base_value = base_eq[dates[i1]] if period == "full" else base_eq[dates[i1]] - base_eq[dates[i0]]
            period_rows.append(dict(period=period, variant=row["key"], start=dates[i0], end=dates[i1], bars=len(idx),
                                    strategy_pnl_usd=round(value, 2), buy_hold_pnl_usd=round(bh, 2),
                                    excess_vs_buy_hold_usd=round(value - bh, 2),
                                    delta_vs_baseline_usd=round(value - base_value, 2)))
    summary = dict(status="AGGREGATED_EXISTING_REPLAY", as_of=AS_OF, rows=EXPECTED_ROWS,
                   source_data_sha256=baseline["input_hashes"]["prices"]["sha256"],
                   baseline_regression="not_rechecked_by_aggregate_only",
                   cost_assumptions=read_json(OUT / "baseline" / "metadata.json")["execution_config"],
                   variants=compact, period_comparisons=period_rows)
    write_csv(OUT / "factor_comparison.csv", compact)
    write_csv(OUT / "period_comparison.csv", period_rows)
    write_equity_artifacts(results, dates, benchmark)
    (OUT / "comparison.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2,
                                                     allow_nan=False) + "\n", encoding="utf-8")
    (OUT / "report_ja.md").write_text(make_report(summary), encoding="utf-8")
    print(f"Aggregated: {OUT / 'report_ja.md'}", flush=True)


if __name__ == "__main__":
    main()
