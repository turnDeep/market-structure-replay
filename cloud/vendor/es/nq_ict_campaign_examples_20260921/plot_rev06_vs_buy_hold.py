"""Plot revision 0.6 ES=F campaign MTM against one-contract Buy & Hold.

This renders fixed-contract cumulative P&L in USD. It deliberately does not
convert the series into an account-return percentage because initial capital,
margin, roll costs, and financing are not specified.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter


POINT_VALUE_USD = 20.0
TICK_SIZE = 0.25
SLIPPAGE_TICKS = 1.0
COMMISSION_PER_SIDE_USD = 2.50


def money(value, _position=None):
    return f"${value:,.0f}"


def drawdown(series: pd.Series) -> pd.Series:
    return series - series.cummax()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prices", type=Path,
                        default=Path("nq_25y_actual_2000_2026_20260911/NQ_F_1d_combined.csv"))
    parser.add_argument("--run", type=Path,
                        default=Path("nq_ict_campaign_examples_20260921/validation_rev06_full_20260922"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    output = args.output or (args.run / "rev06_vs_buy_hold_equity_curve.png")
    prices = pd.read_csv(args.prices, low_memory=False)
    prices["datetime"] = pd.to_datetime(prices["datetime"], errors="raise").dt.normalize()
    prices = prices.sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)
    for column in ("open", "close"):
        prices[column] = pd.to_numeric(prices[column], errors="raise")

    mtm = pd.read_json(args.run / "daily_mtm.jsonl", lines=True)
    mtm["datetime"] = pd.to_datetime(mtm["datetime"], errors="raise").dt.normalize()
    mtm = mtm.sort_values("datetime").reset_index(drop=True)

    if prices["datetime"].iloc[0] != mtm["datetime"].iloc[0] or prices["datetime"].iloc[-1] != mtm["datetime"].iloc[-1]:
        raise ValueError("price and campaign MTM date ranges do not match")
    if len(prices) != len(mtm) or not prices["datetime"].equals(mtm["datetime"]):
        raise ValueError("price and campaign MTM calendars do not match")

    entry_fill = float(prices["open"].iloc[0]) + SLIPPAGE_TICKS * TICK_SIZE
    buy_hold_mtm = ((prices["close"] - entry_fill) * POINT_VALUE_USD
                    - COMMISSION_PER_SIDE_USD)
    strategy_mtm = mtm["mtm_pnl_usd"].astype(float)
    buy_hold_dd = drawdown(buy_hold_mtm)
    strategy_dd = drawdown(strategy_mtm)

    events = load_jsonl(args.run / "execution_events.jsonl")
    buy_events = [event for event in events if event.get("action") == "buy_fill"]
    sell_events = [event for event in events if event.get("action") == "sell_fill"]
    date_to_strategy = dict(zip(mtm["datetime"], strategy_mtm))

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Meiryo", "Yu Gothic", "Arial", "DejaVu Sans"],
        "axes.titlesize": 16,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })

    fig, (ax, dd_ax) = plt.subplots(
        2, 1, figsize=(19, 11), dpi=160, sharex=True,
        gridspec_kw={"height_ratios": [3.1, 1.25], "hspace": 0.08},
    )
    fig.patch.set_facecolor("#fbfcfe")
    for axis in (ax, dd_ax):
        axis.set_facecolor("#fbfcfe")
        axis.grid(True, color="#d9dee7", linewidth=0.7, alpha=0.8)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    dates = mtm["datetime"]
    ax.plot(dates, strategy_mtm, color="#155e9c", linewidth=2.0,
            label="改訂0.6：日次MTM損益")
    ax.plot(dates, buy_hold_mtm, color="#b7791f", linewidth=1.8,
            label="Buy & Hold：日次MTM損益")
    ax.axhline(0, color="#364152", linewidth=0.9)

    for event in buy_events:
        event_date = pd.Timestamp(event["at"]).normalize()
        if event_date in date_to_strategy:
            ax.scatter(event_date, date_to_strategy[event_date], marker="^", s=58,
                       color="#155e9c", edgecolors="#ffffff", linewidths=0.7,
                       zorder=5, label="BUY約定" if event is buy_events[0] else None)
    for event in sell_events:
        event_date = pd.Timestamp(event["at"]).normalize()
        if event_date in date_to_strategy:
            ax.scatter(event_date, date_to_strategy[event_date], marker="v", s=58,
                       color="#b7791f", edgecolors="#ffffff", linewidths=0.7,
                       zorder=5, label="SELL約定" if event is sell_events[0] else None)

    # Annotate only the latest requested cycle so the 26-year line remains readable.
    for event_date, label, offset in (
        (pd.Timestamp("2025-04-10"), "2025-04-10 BUY", (10, 24)),
        (pd.Timestamp("2026-02-06"), "2026-02-06 SELL", (10, -30)),
        (pd.Timestamp("2026-04-01"), "2026-04-01 BUY", (-105, 24)),
    ):
        if event_date in date_to_strategy:
            ax.annotate(label, xy=(event_date, date_to_strategy[event_date]),
                        xytext=offset, textcoords="offset points", fontsize=8.5,
                        color="#344054",
                        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white",
                              "edgecolor": "#b8c2d1", "alpha": 0.92},
                        arrowprops={"arrowstyle": "->", "color": "#667085", "linewidth": 0.8})

    ax.set_title("ES=F 改訂0.6とBuy & Holdの資産推移比較", loc="left",
                 pad=12, fontweight="bold", color="#1f2937")
    ax.set_ylabel("累積時価評価損益（USD / 1枚）")
    ax.yaxis.set_major_formatter(FuncFormatter(money))
    ax.legend(loc="upper left", frameon=False, ncol=4)

    dd_ax.plot(dates, strategy_dd, color="#155e9c", linewidth=1.5,
               label="改訂0.6 DD")
    dd_ax.fill_between(dates, strategy_dd.to_numpy(), 0, color="#155e9c", alpha=0.10)
    dd_ax.plot(dates, buy_hold_dd, color="#b7791f", linewidth=1.4,
               label="Buy & Hold DD")
    dd_ax.fill_between(dates, buy_hold_dd.to_numpy(), 0, color="#b7791f", alpha=0.08)
    dd_ax.axhline(0, color="#364152", linewidth=0.9)
    dd_ax.set_ylabel("DD（USD）")
    dd_ax.yaxis.set_major_formatter(FuncFormatter(money))
    dd_ax.legend(loc="lower left", frameon=False, ncol=2)

    locator = mdates.YearLocator(2)
    dd_ax.xaxis.set_major_locator(locator)
    dd_ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    dd_ax.set_xlabel("観測日")

    strategy_final = float(strategy_mtm.iloc[-1])
    buy_hold_final = float(buy_hold_mtm.iloc[-1])
    fig.text(
        0.01, 0.012,
        f"期間: {dates.iloc[0]:%Y-%m-%d}〜{dates.iloc[-1]:%Y-%m-%d} / "
        f"改訂0.6最終MTM {money(strategy_final)} / Buy & Hold最終MTM {money(buy_hold_final)} / "
        "1枚あたり固定USD損益。初期資本・証拠金・ロール・資金調達費は含まない。",
        fontsize=9, color="#526071",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "source_run": str(args.run.resolve()),
        "prices": str(args.prices.resolve()),
        "output": str(output.resolve()),
        "rows": int(len(mtm)),
        "first_date": str(dates.iloc[0].date()),
        "last_date": str(dates.iloc[-1].date()),
        "strategy_final_mtm_usd": strategy_final,
        "buy_hold_final_mtm_usd": buy_hold_final,
        "difference_strategy_minus_buy_hold_usd": strategy_final - buy_hold_final,
        "strategy_max_drawdown_usd": float(strategy_dd.min()),
        "buy_hold_max_drawdown_usd": float(buy_hold_dd.min()),
        "buy_fill_count": len(buy_events),
        "sell_fill_count": len(sell_events),
        "account_return_not_shown": True,
        "assumptions": {
            "point_value_usd": POINT_VALUE_USD,
            "tick_size": TICK_SIZE,
            "slippage_ticks_per_side": SLIPPAGE_TICKS,
            "commission_per_contract_side_usd": COMMISSION_PER_SIDE_USD,
            "buy_hold_entry": "first observed daily open plus one tick",
            "buy_hold_mark": "daily close; entry commission included",
            "open_strategy_position": "marked at final daily close; no exit commission until closed",
        },
    }
    output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(output)


if __name__ == "__main__":
    main()
