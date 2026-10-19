"""Plot revision 0.7 fixed-contract MTM against Buy & Hold."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
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
    parser.add_argument("--prices", type=Path, default=Path("nq_25y_actual_2000_2026_20260911/NQ_F_1d_combined.csv"))
    parser.add_argument("--run", type=Path, default=Path("nq_ict_campaign_examples_20260921/validation_rev07_full_20260923"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or (args.run / "rev07_vs_buy_hold_equity_curve.png")

    prices = pd.read_csv(args.prices, low_memory=False)
    prices["datetime"] = pd.to_datetime(prices["datetime"], errors="raise").dt.normalize()
    prices = prices.sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)
    prices["open"] = pd.to_numeric(prices["open"], errors="raise")
    prices["close"] = pd.to_numeric(prices["close"], errors="raise")
    mtm = pd.read_json(args.run / "daily_mtm.jsonl", lines=True)
    mtm["datetime"] = pd.to_datetime(mtm["datetime"], errors="raise").dt.normalize()
    mtm = mtm.sort_values("datetime").reset_index(drop=True)
    if len(prices) != len(mtm) or not prices["datetime"].equals(mtm["datetime"]):
        raise ValueError("price and campaign MTM calendars do not match")

    entry_fill = float(prices["open"].iloc[0]) + SLIPPAGE_TICKS * TICK_SIZE
    buy_hold = (prices["close"] - entry_fill) * POINT_VALUE_USD - COMMISSION_PER_SIDE_USD
    strategy = mtm["mtm_pnl_usd"].astype(float)
    buy_hold_dd, strategy_dd = drawdown(buy_hold), drawdown(strategy)
    events = load_jsonl(args.run / "execution_events.jsonl")
    buys = [e for e in events if e.get("action") == "buy_fill"]
    sells = [e for e in events if e.get("action") == "sell_fill"]
    strategy_by_date = dict(zip(mtm["datetime"], strategy))

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Meiryo", "Yu Gothic", "Arial", "DejaVu Sans"]})
    fig, (ax, dd_ax) = plt.subplots(2, 1, figsize=(19, 11), dpi=160, sharex=True,
                                    gridspec_kw={"height_ratios": [3.1, 1.25], "hspace": 0.08})
    fig.patch.set_facecolor("#fbfcfe")
    for axis in (ax, dd_ax):
        axis.set_facecolor("#fbfcfe")
        axis.grid(True, color="#d9dee7", linewidth=0.7, alpha=0.8)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    dates = mtm["datetime"]
    ax.plot(dates, strategy, color="#155e9c", linewidth=2.0, label="改訂0.7：日次MTM損益")
    ax.plot(dates, buy_hold, color="#b7791f", linewidth=1.8, label="Buy & Hold：日次MTM損益")
    ax.axhline(0, color="#364152", linewidth=0.9)
    for event in buys:
        date = pd.Timestamp(event["at"]).normalize()
        if date in strategy_by_date:
            ax.scatter(date, strategy_by_date[date], marker="^", s=48, color="#15803d", edgecolors="white", linewidths=0.6,
                       zorder=5, label="BUY約定" if event is buys[0] else None)
    for event in sells:
        date = pd.Timestamp(event["at"]).normalize()
        if date in strategy_by_date:
            ax.scatter(date, strategy_by_date[date], marker="v", s=48, color="#b91c1c", edgecolors="white", linewidths=0.6,
                       zorder=5, label="SELL約定" if event is sells[0] else None)
    ax.set_title("ES=F 改訂0.7（無制限BUY＋継続BUY）とBuy & Holdの資産推移", loc="left", pad=12,
                 fontweight="bold", color="#1f2937")
    ax.set_ylabel("累積時価評価損益（USD / 1枚）")
    ax.yaxis.set_major_formatter(FuncFormatter(money))
    ax.legend(loc="upper left", frameon=False, ncol=4)

    dd_ax.plot(dates, strategy_dd, color="#155e9c", linewidth=1.5, label="改訂0.7 DD")
    dd_ax.fill_between(dates, strategy_dd.to_numpy(), 0, color="#155e9c", alpha=0.10)
    dd_ax.plot(dates, buy_hold_dd, color="#b7791f", linewidth=1.4, label="Buy & Hold DD")
    dd_ax.fill_between(dates, buy_hold_dd.to_numpy(), 0, color="#b7791f", alpha=0.08)
    dd_ax.axhline(0, color="#364152", linewidth=0.9)
    dd_ax.set_ylabel("DD（USD）")
    dd_ax.yaxis.set_major_formatter(FuncFormatter(money))
    dd_ax.legend(loc="lower left", frameon=False, ncol=2)
    dd_ax.xaxis.set_major_locator(mdates.YearLocator(2))
    dd_ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    dd_ax.set_xlabel("観測日")
    fig.text(0.01, 0.012,
             f"期間: {dates.iloc[0]:%Y-%m-%d}〜{dates.iloc[-1]:%Y-%m-%d} / "
             f"改訂0.7 {strategy.iloc[-1]:,.0f} USD / Buy & Hold {buy_hold.iloc[-1]:,.0f} USD / "
             "固定1枚。初期資本・証拠金・ロール・資金調達費は含まない。",
             fontsize=9, color="#526071")
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    metadata = {"source_run": str(args.run.resolve()), "prices": str(args.prices.resolve()), "output": str(output.resolve()),
                "rows": int(len(mtm)), "first_date": str(dates.iloc[0].date()), "last_date": str(dates.iloc[-1].date()),
                "strategy_final_mtm_usd": float(strategy.iloc[-1]), "buy_hold_final_mtm_usd": float(buy_hold.iloc[-1]),
                "difference_strategy_minus_buy_hold_usd": float(strategy.iloc[-1] - buy_hold.iloc[-1]),
                "strategy_max_drawdown_usd": float(strategy_dd.min()), "buy_hold_max_drawdown_usd": float(buy_hold_dd.min()),
                "buy_fill_count": len(buys), "sell_fill_count": len(sells),
                "assumptions": {"point_value_usd": POINT_VALUE_USD, "tick_size": TICK_SIZE,
                                "slippage_ticks_per_side": SLIPPAGE_TICKS,
                                "commission_per_contract_side_usd": COMMISSION_PER_SIDE_USD,
                                "buy_hold_entry": "first observed daily open plus one tick",
                                "buy_hold_mark": "daily close; entry commission included",
                                "strategy_open_position": "marked at final daily close"}}
    output.with_suffix(".metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
