"""Admit the actual ES alpha replay, preserving NQ and archiving NKD.

No trading-engine calls or Git writes. Export and publish are separate phases;
the latter requires a compiled, reviewed ES view and independent data checks.
"""
import argparse
from copy import deepcopy
import gzip
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from cloud.common import CLOUD,digest,load,now_iso,save
from cloud.export_adapter import export
from cloud.refresh import stage_state
from scripts.import_replays import canonical,package,read_snapshot


def reconcile(work,research):
    import numpy as np
    ref=research/"v012_cross_market_20261019/arms/es/v012_alpha"
    expected=load(ref/"trades.json")
    def lines(name):
        import json
        return [json.loads(x) for x in (work/"result/campaign_replay"/(name+".jsonl")).read_text(encoding="utf-8").splitlines()]
    fields=["entry_at","exit_at","entry_price","exit_price","net_pnl_usd"]
    compact=lambda rows:[{k:r[k] for k in fields} for r in rows if r["exit_at"][:10]<="2026-10-14"]
    assert compact(lines("trades"))==compact(expected),"ES alpha trade mismatch"
    curve=np.load(ref/"curve.npz");count=len(curve["dates"])
    mtm=lines("daily_mtm");states=lines("daily_states")
    assert [r["datetime"][:10] for r in mtm[:count]]==curve["dates"].tolist()
    np.testing.assert_allclose([r["mtm_pnl_usd"] for r in mtm[:count]],curve["mtm"],rtol=0,atol=1e-8)
    assert [r["position_id"] is not None for r in states[:count]]==curve["exposure"].tolist()
    save(work/"result/research_prefix_audit.json",dict(status="PASS",through="2026-10-14",
        closed_trades=len(expected),daily_mtm_and_exposure_rows=count,
        expected_trades_sha256=digest(canonical(compact(expected)))))


def export_es(work,research):
    reconcile(work,research)
    if not (work/"original_manifest.json").exists():
        save(work/"original_manifest.json",load(ROOT/"assets/manifest.json"))
    nq=read_snapshot(gzip.decompress((ROOT/"assets/nq.html.gz").read_bytes()).decode("utf-8"))
    before=deepcopy(nq)
    before.update(id="es-v012-decision-replay-20261019",title="ES=F · v0.12 追体験",surface="dashboard")
    before["replay"]=dict(start="2000-09-18",end="2026-10-14",defaultDate="2026-10-14",
        strategyStatus="NOT_VALIDATED",strategyVersion="v0.12",pointValue=50,commissionSide=2.5,slippagePoints=.25)
    definitions={
        "market_prices":("ES=F 日足・200SMA",["prices.csv"],"1行=有効配信日足。200SMA=当日を含む直近200有効終値の単純平均。"),
        "primary_zones":("v09 OB/FVGの状態",["primary_zone_state_events.jsonl"],"起点日と利用可能日は別。選択日までの保存状態だけを使用。"),
        "tactical_zones":("補助日足FVG/IFVG",["tactical_zone_events.jsonl"],"原本v09とは別定義の補助帯。採用された構造後再テストは紫のR帯で表示。"),
        "executions":("v0.12 約定モデル",["execution_events.jsonl","entry_structural_contexts.json"],"1行=約定モデル。ES 1枚50USD/pt、片道手数料2.5USD、片道スリッページ0.25pt。決済済みだけnet_pnl_usdを公開。"),
        "decisions":("終値判断・採用根拠",["entry_structural_contexts.json","close_plans.jsonl"],"判断時点に未来の翌始値約定・最終決済を含めない。FVG採用の有無は各保存ルートに従う。"),
        "daily_states":("日次保有・未約定状態",["daily_states.jsonl","close_plans.jsonl"],"1行=観測日終値処理後の状態。未約定注文を実保有に含めない。")}
    captured=load(work/"fetch_audit.json")["captured_at_utc"]
    for name,q in before["queries"].items():
        label,tables,definition=definitions[name];q["rows"]=[]
        q["source"]=dict(label=label,tables=tables,executedAt=captured,
            filters=["ES=F / 2000-09-18以降の有効配信日足 / 選択日・表示窓に連動"],
            metricDefinitions=[dict(label=label,definition=definition)],
            caveats=["Yahoo連続先物。限月別ロール約定・証拠金・金利は未再現。日足OHLCのSTOPは代理約定。研究用NOT_VALIDATED。",
                "元取得時の不正日足53行は除外・価格補間なし。過去の配信修正は凍結入力へ遡及適用しない。"],
            evidenceFlow=[dict(title="ES v0.12α照合",detail="同一の凍結入力・半径1設定で6573日分の損益・保有と44件の決済を先行研究に照合。")])
    save(work/"before.json",before)
    fresh=export("es",work,before)
    fresh["replay"]["refresh"].update(dataPolicy="frozen_research_inputs_plus_completed_yahoo_append",
        marketAdded="ES=F",frozenThrough="2026-10-14",appendedBars=len(fresh["queries"]["market_prices"]["rows"])-6573)
    fresh["replay"]["summaryUiRevision"]="2026-10-19-es-asof-summary"
    scope=["es-asof-results"]
    fresh["queries"]["executions"]["source"]["metricDefinitions"] += [
        dict(label="選択日までの実現損益・勝率",componentIds=scope,definition="選択日以下のSELL/STOP net_pnl_usdを合算。勝率=正損益決済数/全決済数。未決済・未約定は除外。50USD/pt、往復手数料5USD控除済み。"),
        dict(label="最大1取引への依存",componentIds=scope,definition="選択日までの最大の正の決済損益と、累計からその1件を引いた値。未来の決済を含めない。")]
    fresh["queries"]["market_prices"]["source"]["metricDefinitions"].append(dict(label="未決済の含み損益（USD）",componentIds=scope,
        definition="保有1枚について(選択日終値−実BUY約定価格)×50−新規手数料2.5USD。未発生の決済手数料は控除しない。実現損益と分離。"))
    save(work/"result/snapshot.json",fresh)
    save(work/"result/app/src/data.json",fresh)
    print("ES_EXPORT_READY",len(fresh["queries"]["market_prices"]["rows"]),fresh["replay"]["end"],flush=True)


def publish(work,html):
    baseline=load(work/"original_manifest.json")
    current=load(ROOT/"assets/manifest.json")
    previous_staged=work/"publication/assets/manifest.json"
    assert current==baseline or (previous_staged.exists() and current==load(previous_staged)),"Another publisher changed the generation"
    assert load(work/"result/research_prefix_audit.json")["status"]=="PASS"
    assert load(work/"result/replay_audit.json")["status"]=="PASS_FROZEN_HISTORY_AND_CONFIGURATION"
    assert load(work/"result/view_audit.json")["status"]=="PASS_DATA_VIEW_AND_LEDGER"
    expected=load(work/"result/snapshot.json")
    assert read_snapshot(html.read_text(encoding="utf-8"))==expected,"Compiled view data mismatch"
    staged=work/"publication";assets=staged/"assets";assets.mkdir(parents=True,exist_ok=True)
    item=package(html,"es",assets)
    item["data_quality"]=dict(provisional_from_date=None,verified_through=item["end"])
    archive=assets/"archive/nkd-v012";archive.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(ROOT/"assets/nikkei.html.gz",archive/"nikkei.html.gz")
    save(archive/"manifest.json",baseline)
    stage_state("es",work,staged)
    manifest=deepcopy(baseline);manifest["markets"]={"nq":baseline["markets"]["nq"],"es":item}
    manifest.update(packaged_at=now_iso(),active_markets_change="NQ + ES; NKD retained in archive/nkd-v012",
        market_addition="es_addition_20261019.json")
    receipt=dict(status="PASS_ES_V012_ADDITION",strategy_status="NOT_VALIDATED",created_at_utc=now_iso(),
        bars=item["bars"],end=item["end"],historical_prefix=item["queries"],
        nq_gzip_sha256_unchanged=baseline["markets"]["nq"]["gzip_sha256"],
        archived_nkd_gzip_sha256=baseline["markets"]["nikkei"]["gzip_sha256"],
        research_prefix=load(work/"result/research_prefix_audit.json"),view=load(work/"result/view_audit.json"),
        replay=load(work/"result/replay_audit.json"),fetch=load(work/"fetch_audit.json"))
    save(assets/"es_addition_20261019.json",receipt)
    save(assets/"manifest.json",manifest)
    for source in sorted(staged.rglob("*")):
        if source.is_file():
            rel=source.relative_to(staged);target=ROOT/("cloud" if rel.parts[0]=="state" else "")/rel
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    assert digest((ROOT/"assets/nq.html.gz").read_bytes())==receipt["nq_gzip_sha256_unchanged"]
    print("PUBLISHED_LOCAL_NQ_ES_NOT_PUSHED",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("mode",choices=["export","publish"])
    p.add_argument("--work",type=Path,required=True);p.add_argument("--research",type=Path);p.add_argument("--html",type=Path)
    a=p.parse_args()
    if a.mode=="export":export_es(a.work.resolve(),a.research.resolve())
    else:publish(a.work.resolve(),a.html.resolve())
