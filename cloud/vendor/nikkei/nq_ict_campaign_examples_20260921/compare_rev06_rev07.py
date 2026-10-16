"""Recalculate the fixed assumptions and compare campaign revisions 0.6/0.7."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .audit_campaign_revision07 import accounting_audit, buy_hold_audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", type=Path, default=Path("nq_25y_actual_2000_2026_20260911/NQ_F_1d_combined.csv"))
    parser.add_argument("--rev06", type=Path, default=Path("nq_ict_campaign_examples_20260921/validation_rev06_full_20260922"))
    parser.add_argument("--rev07", type=Path, default=Path("nq_ict_campaign_examples_20260921/validation_rev07_full_20260923"))
    parser.add_argument("--output-dir", type=Path, default=Path("nq_ict_campaign_examples_20260921/validation_rev07_full_20260923"))
    args = parser.parse_args()
    rows = {}
    for name, directory in (("rev06", args.rev06), ("rev07", args.rev07)):
        meta = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        accounting = accounting_audit(directory, args.prices, meta)
        benchmark = buy_hold_audit(directory, args.prices)
        rows[name] = {"accounting": accounting, "buy_hold_comparison": benchmark,
                      "config": meta.get("campaign_config", {})}
    a, b = rows["rev06"]["accounting"], rows["rev07"]["accounting"]
    diff = {
        "buy_fill_count": b["execution_buy_fill_count"] - a["execution_buy_fill_count"],
        "closed_trade_count": b["closed_trades"] - a["closed_trades"],
        "realized_net_usd": b["realized_net_usd"] - a["realized_net_usd"],
        "final_mtm_usd": b["final_mtm_usd"] - a["final_mtm_usd"],
        "max_drawdown_usd": b["mtm_max_drawdown_usd"] - a["mtm_max_drawdown_usd"],
        "profit_factor": b["profit_factor"] - a["profit_factor"],
        "win_rate": b["win_rate"] - a["win_rate"],
    }
    result = {"rev06": rows["rev06"], "rev07": rows["rev07"], "rev07_minus_rev06": diff}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "comparison_rev06_rev07.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    r06, r07 = rows["rev06"]["buy_hold_comparison"], rows["rev07"]["buy_hold_comparison"]
    report = ["# 改訂0.6 / 0.7比較", "",
              "同一のNKD=F日足、同一の固定1枚・スリッページ・手数料条件で、履歴への再生結果だけを比較した。未見性能の比較ではない。", "",
              "| 指標 | 改訂0.6 | 改訂0.7 | 差（0.7−0.6） |", "|---|---:|---:|---:|",
              f"| BUY約定 | {a['execution_buy_fill_count']} | {b['execution_buy_fill_count']} | {diff['buy_fill_count']:+d} |",
              f"| 完了取引 | {a['closed_trades']} | {b['closed_trades']} | {diff['closed_trade_count']:+d} |",
              f"| 勝率 | {a['win_rate']:.2%} | {b['win_rate']:.2%} | {diff['win_rate']:+.2%} |",
              f"| 実現損益（USD） | {a['realized_net_usd']:,.0f} | {b['realized_net_usd']:,.0f} | {diff['realized_net_usd']:+,.0f} |",
              f"| PF | {a['profit_factor']:.3f} | {b['profit_factor']:.3f} | {diff['profit_factor']:+.3f} |",
              f"| 最終MTM（USD） | {a['final_mtm_usd']:,.1f} | {b['final_mtm_usd']:,.1f} | {diff['final_mtm_usd']:+,.1f} |",
              f"| 最大DD（USD） | {a['mtm_max_drawdown_usd']:,.1f} | {b['mtm_max_drawdown_usd']:,.1f} | {diff['max_drawdown_usd']:+,.1f} |",
              "", "## Buy & Holdとの差", "",
              f"- 改訂0.6: {r06['strategy_final_minus_buy_hold_usd']:,.1f} USD、最大DD改善 {r06['strategy_max_dd_improvement_usd']:,.1f} USD。",
              f"- 改訂0.7: {r07['strategy_final_minus_buy_hold_usd']:,.1f} USD、最大DD改善 {r07['strategy_max_dd_improvement_usd']:,.1f} USD。", "",
              "## 結論", "",
              "改訂0.7は購入回数制限を外し、継続BUYを加えたことで、全期間の最終MTMを改訂0.6より増やした。一方、取引数と損失も増え、PFは低下し、改訂0.6より最大DDは悪化した。したがって、今回の変更は上昇捕捉を改善したが、リスク調整後の改善を意味しない。",
              "2016年・2023年以降の窓では改訂0.7は改訂0.6より改善したが、各窓のBuy & Holdにはなお遅れた。これは既存LONGを持つ期間の捕捉と、エントリー／退出で空白になった期間の両方を含む。"]
    (args.output_dir / "comparison_rev06_rev07_report_ja.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
