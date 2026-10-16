"""Add component-scoped summary definitions; never change reviewed trading rows."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument("snapshot",type=Path);p.add_argument("--status",choices=["updating","complete"],default="updating");args=p.parse_args()
    snapshot=json.loads(args.snapshot.read_text(encoding="utf-8-sig"))
    def row_hashes():
        return {k:hashlib.sha256(json.dumps(q["rows"],ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()).hexdigest() for k,q in snapshot["queries"].items()}
    before=row_hashes()
    scope=["nq-asof-results"]
    definitions={
        "executions":[
            dict(label="選択日までの実現損益（USD）",componentIds=scope,definition="収録開始から選択日以下のSELL/STOPに保存されたnet_pnl_usdの合計。BUYは加算せず、1枚・20USD/pt・往復手数料5USDを各決済で控除済み。約定価格は片道0.25ptのスリッページ反映済み。"),
            dict(label="決済数・勝敗・勝率",componentIds=scope,definition="選択日以下のSELL/STOPのみを数える。net_pnl_usdが正なら勝ち、負なら負け、0なら同値。勝率=勝ち数÷全決済数×100。同値も分母に含め、未決済・翌始値待ち注文は除外。決済0件の勝率は欠測。"),
            dict(label="最大1取引への依存",componentIds=scope,definition="選択日までに決済済みの最大の正のnet_pnl_usdと、実現損益累計からその利益を1回だけ差し引いた額。将来の決済結果は含めない。"),
        ],
        "market_prices":[dict(label="未決済の含み損益（USD）",componentIds=scope,definition="選択日に保有中の1枚について、(選択日終値−実際のBUY約定価格)×20−新規手数料2.5USD。まだ発生していない決済手数料は控除せず、実現損益に加算しない。")],
        "daily_states":[dict(label="集計の保有判定",componentIds=scope,definition="選択日終値後の保存position_idと、その日までに約定したBUY/SELL/STOPを照合する。未約定の買い注文は保有としない。ノーポジションの含み損益は『保有なし』。")],
    }
    labels={d["label"] for group in definitions.values() for d in group}
    for name,items in definitions.items():
        source=snapshot["queries"][name]["source"]
        source["metricDefinitions"]=[d for d in source.get("metricDefinitions",[]) if d.get("label") not in labels]+items
    snapshot["buildStatus"]=args.status
    snapshot["replay"]["summaryUiRevision"]="2026-10-16-nq-asof-summary"
    assert before==row_hashes()
    args.snapshot.write_text(json.dumps(snapshot,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    print("PASS_METADATA_ONLY: reviewed rows unchanged; summary definitions scoped to nq-asof-results")


if __name__=="__main__":main()
