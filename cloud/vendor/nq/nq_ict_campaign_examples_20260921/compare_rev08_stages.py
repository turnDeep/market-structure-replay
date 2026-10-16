"""Compare fixed revision 0.8 stage audits without refitting parameters."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("nq_ict_campaign_examples_20260921"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    stages = {
        "rev07_controls": args.root / "validation_rev08_rev07_controls_full_20260923" / "audit_rev08.json",
        "management": args.root / "validation_rev08_management_full_20260923" / "audit_rev08.json",
        "management_reentry": args.root / "validation_rev08_management_reentry_full_20260923" / "audit_rev08.json",
        "full": args.root / "validation_rev08_full_20260923" / "audit_rev08.json",
    }
    rows = []
    for name, path in stages.items():
        audit = json.loads(path.read_text(encoding="utf-8"))
        accounting = audit["accounting"]
        benchmark = audit["buy_hold_comparison"]
        causal = audit["causal_management_and_continuation"]
        rows.append({
            "stage": name,
            "audit_status": audit["audit_status"],
            "closed_trades": accounting["closed_trades"],
            "right_censored": accounting["right_censored_open_attempts"],
            "wins": accounting["wins"], "losses": accounting["losses"],
            "win_rate": accounting["win_rate"],
            "realized_net_usd": accounting["realized_net_usd"],
            "profit_factor": accounting["profit_factor"],
            "final_mtm_usd": accounting["final_mtm_usd"],
            "max_dd_usd": accounting["mtm_max_drawdown_usd"],
            "vs_buy_hold_usd": benchmark["strategy_final_minus_buy_hold_usd"],
            "dd_improvement_usd": benchmark["strategy_max_dd_improvement_usd"],
            "selected_reversal": causal["selected_entry_family_counts"].get("reversal", 0),
            "selected_continuation": causal["selected_entry_family_counts"].get("continuation", 0),
            "wave_confirmed": causal["wave_confirmed_count"],
            "wave_effective": causal["wave_effective_count"],
            "reentry_events": causal["exit_reentry_event_count"],
        })
    result = {"stages": rows, "buy_hold_final_mtm_usd": rows[0]["final_mtm_usd"] - rows[0]["vs_buy_hold_usd"]}
    output = args.output or (args.root / "validation_rev08_stage_comparison.json")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# 改訂0.8固定ステージ比較", "", "同じ入力・同じ入口/出口検知器で、管理機能を段階的に追加した比較。", "",
             "| stage | 完了取引 | 勝率 | 実現損益 | 最終MTM | 最大DD | Buy&Hold差 | DD改善 | reversal | continuation | |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for row in rows:
        lines.append(f"| {row['stage']} | {row['closed_trades']} | {row['win_rate']:.1%} | {row['realized_net_usd']:,.0f} | {row['final_mtm_usd']:,.0f} | {row['max_dd_usd']:,.0f} | {row['vs_buy_hold_usd']:,.0f} | {row['dd_improvement_usd']:,.0f} | {row['selected_reversal']} | {row['selected_continuation']} |  |")
    lines += ["", "`rev07_controls` は旧管理相当、`management` は波管理、`management_reentry` はExit-reason再BUYまで、`full` は継続BUYのreactionリスク上限まで有効。", ""]
    (args.root / "validation_rev08_stage_comparison_ja.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
