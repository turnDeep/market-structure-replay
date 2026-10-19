"""Render the campaign's daily MTM asset curve and drawdown.

The primary series is daily mark-to-market P&L for one contract.  The chart
also shows realized-only P&L, completed exit markers, and the right-censored
open position at the end of the input.  It is not an account-return chart.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter


def money(value, _position=None):
    return f"${value:,.0f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    run = args.run
    output = args.output or (run / "equity_curve.png")
    mtm = pd.read_json(run / "daily_mtm.jsonl", lines=True)
    mtm["datetime"] = pd.to_datetime(mtm["datetime"])
    trades = pd.read_json(run / "trades.jsonl", lines=True)
    if not trades.empty:
        trades["entry_at"] = pd.to_datetime(trades["entry_at"])
        trades["exit_at"] = pd.to_datetime(trades["exit_at"])
    final_state = json.loads((run / "final_state.json").read_text(encoding="utf-8"))

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Meiryo", "Yu Gothic", "Arial", "DejaVu Sans"],
        "axes.titlesize": 16,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })

    fig, (ax, dd_ax) = plt.subplots(
        2, 1, figsize=(18, 11), dpi=160, sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.3], "hspace": 0.08},
    )
    fig.patch.set_facecolor("#fbfcfe")
    for axis in (ax, dd_ax):
        axis.set_facecolor("#fbfcfe")
        axis.grid(True, color="#d9dee7", linewidth=0.7, alpha=0.8)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    # MTM includes the floating P&L of an open position. Realized is shown as
    # a dashed comparator so the final right-censored position is explicit.
    ax.plot(mtm["datetime"], mtm["mtm_pnl_usd"], color="#155e9c", linewidth=2.0,
            label="Daily MTM P&L (1 contract)")
    ax.plot(mtm["datetime"], mtm["realized_net_usd"], color="#667085", linewidth=1.25,
            linestyle="--", label="Realized P&L only")
    ax.axhline(0, color="#364152", linewidth=0.9)

    if final_state.get("open_position"):
        entry_at = pd.to_datetime(final_state["open_position"]["entry_at"])
        ax.axvspan(entry_at, mtm["datetime"].iloc[-1], color="#eab308", alpha=0.08,
                   label="Open position / right-censored")
        ax.scatter([entry_at], [float(mtm.loc[mtm["datetime"] == entry_at, "mtm_pnl_usd"].iloc[0])
                                if (mtm["datetime"] == entry_at).any() else 0],
                   color="#b7791f", s=42, zorder=5, marker="D")

    # Exit markers are sparse enough at 29 trades to retain the event history.
    if not trades.empty:
        for _, trade in trades.iterrows():
            exit_row = mtm.loc[mtm["datetime"] == trade["exit_at"]]
            if exit_row.empty:
                continue
            y = float(exit_row["mtm_pnl_usd"].iloc[0])
            color = "#1f7a5a" if trade["net_pnl_usd"] > 0 else "#c05640"
            ax.scatter([trade["exit_at"]], [y], color=color, s=22, zorder=5,
                       edgecolors="#ffffff", linewidths=0.5)

    # Keep the largest single completed trade visible without annotating every
    # event. This prevents the long-run shape from being mistaken for a smooth
    # stationary strategy.
    if not trades.empty:
        largest = trades.loc[trades["net_pnl_usd"].idxmax()]
        largest_row = mtm.loc[mtm["datetime"] == largest["exit_at"]]
        if not largest_row.empty:
            ax.annotate(
                f"Largest completed trade\n{largest['entry_at'].date()} → {largest['exit_at'].date()}\n"
                f"{money(float(largest['net_pnl_usd']))}",
                xy=(largest["exit_at"], float(largest_row["mtm_pnl_usd"].iloc[0])),
                xytext=(-150, 32), textcoords="offset points", fontsize=9,
                bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#b8c2d1"},
                arrowprops={"arrowstyle": "->", "color": "#667085", "linewidth": 0.9},
            )

    ax.set_title("ES=F 日足スイング・キャンペーンの資産推移", loc="left", pad=12,
                 fontweight="bold", color="#1f2937")
    ax.set_ylabel("Cumulative P&L (USD)")
    ax.yaxis.set_major_formatter(FuncFormatter(money))
    ax.legend(loc="upper left", frameon=False, ncol=3)

    dd_ax.plot(mtm["datetime"], mtm["close_to_close_drawdown_usd"], color="#c05640", linewidth=1.6,
               label="Close-to-close drawdown")
    dd_ax.fill_between(mtm["datetime"], mtm["close_to_close_drawdown_usd"], 0,
                       color="#c05640", alpha=0.16)
    dd_ax.axhline(0, color="#364152", linewidth=0.9)
    dd_ax.set_ylabel("Drawdown (USD)")
    dd_ax.yaxis.set_major_formatter(FuncFormatter(money))
    dd_ax.legend(loc="lower left", frameon=False)

    locator = mdates.YearLocator(2)
    formatter = mdates.DateFormatter("%Y")
    dd_ax.xaxis.set_major_locator(locator)
    dd_ax.xaxis.set_major_formatter(formatter)
    dd_ax.set_xlabel("Observation date")
    fig.text(
        0.01, 0.012,
        "Scope: 2000-09-18〜2026-08-27 / 1枚の固定USD損益。MTMは未決済ポジションを時価評価。"
        "最終ポジションは右側打切り。証拠金・ロール・資金制約・口座資産倍率は含まない。",
        fontsize=9, color="#526071",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "source_run": str(run.resolve()),
        "output": str(output.resolve()),
        "series": "daily_mtm.mtm_pnl_usd",
        "comparator": "daily_mtm.realized_net_usd",
        "scope": "one_contract_fixed_usd_pnl",
        "right_censored": bool(final_state.get("right_censored")),
        "open_position": bool(final_state.get("open_position")),
        "rows": int(len(mtm)),
    }
    (output.with_suffix(".metadata.json")).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(output)


if __name__ == "__main__":
    main()
