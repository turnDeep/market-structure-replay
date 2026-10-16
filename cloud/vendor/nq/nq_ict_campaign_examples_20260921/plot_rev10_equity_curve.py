"""Plot audited revision 0.10 MTM-PnL curves against the matched Buy & Hold.

This is a visualization of already-produced replay outputs; it does not run a
strategy replay or change any trading rules.
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


HERE = Path(__file__).resolve().parent
DATA_FILE = HERE.parent / "nq_25y_actual_2000_2026_20260911" / "NQ_F_1d_combined.csv"
RELEASE_DIR = HERE / "validation_rev10_comparison_release_20260923"

RUNS = [
    ("基本条件", "validation_rev10_baseline_full_20260923", "#2468A2", "-", 540_992.5),
    ("失敗反応制限のみ", "validation_rev10_episode_only_full_20260923", "#D47A21", "-", 523_567.5),
    ("復帰BUYのみ", "validation_rev10_recovery_only_full_20260923", "#198577", "-", 546_692.5),
    ("併用", "validation_rev10_combined_full_20260923", "#8064A2", "-", 532_232.5),
]
BUY_HOLD_FINAL = 516_127.5
EXPECTED_ROWS = 6_526


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def read_prices(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return rows


def make_chart(output: Path) -> None:
    prices = read_prices(DATA_FILE)
    if len(prices) != EXPECTED_ROWS:
        raise ValueError(f"Price row count changed: expected {EXPECTED_ROWS}, got {len(prices)}")

    dates = [datetime.strptime(row["datetime"][:10], "%Y-%m-%d") for row in prices]
    if dates != sorted(set(dates)):
        raise ValueError("Input dates are not sorted and unique")

    # Match the previously audited B&H convention: buy one contract at the
    # first daily open plus one tick of slippage, then mark to each close.
    first_fill = float(prices[0]["open"]) + 0.25
    buy_hold = [(float(row["close"]) - first_fill) * 20.0 - 2.50 for row in prices]
    if abs(buy_hold[-1] - BUY_HOLD_FINAL) > 0.01:
        raise ValueError(f"Buy & Hold final does not match audited value: {buy_hold[-1]}")

    series: list[tuple[str, list[float], str, str]] = []
    for label, folder, color, linestyle, expected_final in RUNS:
        rows = load_jsonl(HERE / folder / "daily_mtm.jsonl")
        row_dates = [datetime.strptime(row["datetime"][:10], "%Y-%m-%d") for row in rows]
        values = [float(row["mtm_pnl_usd"]) for row in rows]
        if len(rows) != EXPECTED_ROWS or row_dates != dates:
            raise ValueError(f"Daily MTM dates/row count do not match source prices: {folder}")
        if abs(values[-1] - expected_final) > 0.01:
            raise ValueError(f"Final MTM does not match the audited value for {label}: {values[-1]}")
        series.append((label, values, color, linestyle))

    plt.rcParams["font.family"] = "Meiryo"
    fig, ax = plt.subplots(figsize=(17.2, 9.4), constrained_layout=False)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    for label, values, color, linestyle in series:
        final_text = f"${values[-1]:,.0f}"
        ax.plot(dates, values, color=color, linewidth=1.65, linestyle=linestyle,
                label=f"{label}  {final_text}", zorder=3)
    ax.plot(dates, buy_hold, color="#343A40", linewidth=2.0, linestyle=(0, (5, 3)),
            label=f"Buy & Hold  ${buy_hold[-1]:,.0f}", zorder=2)

    ax.axhline(0, color="#8B9299", linewidth=0.9, zorder=1)
    ax.set_xlim(dates[0], dates[-1] + (dates[-1] - dates[0]) * 0.015)
    ax.set_ylim(min(-25_000, min(min(v) for _, v, _, _ in series), min(buy_hold)) * 1.03,
                max(max(max(v) for _, v, _, _ in series), max(buy_hold)) * 1.06)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"${value:,.0f}"))

    tick_dates = [dates[0]]
    tick_dates.extend(datetime(year, 1, 1) for year in range(2002, 2026, 2))
    tick_dates.append(dates[-1])
    ax.set_xticks(tick_dates)
    tick_labels = [tick_dates[0].strftime("%Y-%m")]
    tick_labels.extend(date.strftime("%Y") for date in tick_dates[1:-1])
    tick_labels.append(tick_dates[-1].strftime("%Y-%m-%d"))
    ax.set_xticklabels(tick_labels)

    ax.grid(axis="y", color="#E5E8EB", linewidth=0.8)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#8B9299")
    ax.spines["bottom"].set_color("#8B9299")
    ax.tick_params(axis="both", colors="#343A40", labelsize=10, length=0, pad=8)
    ax.set_ylabel("累積時価評価損益（USD／NQ=F 1枚）", fontsize=12, color="#252A2E", labelpad=12)
    ax.set_xlabel("日付", fontsize=11, color="#252A2E", labelpad=12)
    ax.set_title("NQ=F 改訂0.10：4条件と Buy & Hold の推移", fontsize=18,
                 weight="semibold", color="#202428", loc="left", pad=21)
    fig.text(0.125, 0.913,
             "2000-09-18〜2026-08-27  |  6,526日足  |  累積損益であり口座残高・資産倍率ではありません",
             fontsize=10.5, color="#596168", ha="left")

    legend = ax.legend(loc="upper left", bbox_to_anchor=(0, 1.005), ncol=3,
                       frameon=False, fontsize=10.5, handlelength=2.7,
                       columnspacing=1.6, handletextpad=0.6)
    for line in legend.get_lines():
        line.set_linewidth(2.2)

    fig.text(0.125, 0.035,
             "各戦略は保存済みリプレイのdaily MTM。Buy & Holdは初日始値+1 tickで1枚買い、終値評価（片道手数料 $2.50）。"
             "証拠金・資金拘束・限月ロールコストは含みません。",
             fontsize=9.3, color="#596168", ha="left")
    fig.subplots_adjust(left=0.105, right=0.975, top=0.84, bottom=0.14)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=190, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({
        "output": str(output.resolve()),
        "rows": len(dates),
        "start": dates[0].strftime("%Y-%m-%d"),
        "end": dates[-1].strftime("%Y-%m-%d"),
        "buy_hold_entry_fill": first_fill,
        "buy_hold_final_usd": buy_hold[-1],
        "strategy_final_usd": {label: values[-1] for label, values, _, _ in series},
    }, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=RELEASE_DIR / "equity_curve_rev10_vs_buy_hold.png")
    args = parser.parse_args()
    make_chart(args.output)


if __name__ == "__main__":
    main()
