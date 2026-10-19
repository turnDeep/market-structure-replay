"""Portable host for the reviewed route-aware exporter, not trading logic.

Preserve the exact old reviewed presentation rows after reconciling their
factual columns with the newly executed ledger. Export only genuinely new
rows; no old text, order ID, anchor or JSON container is silently rewritten.
"""
from copy import deepcopy
import json
from pathlib import Path

from cloud.common import CLOUD, load, now_iso, save
from cloud.strategy import STRATEGY_VERSION
from cloud.version_labels import normalize_current_labels
from cloud.protective_exit_view import adapt_template, protective_sell


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Reviewed exporter adapter no longer matches: {old[:100]}")
    return source.replace(old,new)


def refresh_metadata(replay, bars, audit, previous_end, prior_rows):
    """Freshness/counts are derived from rows, not an accumulating stale label."""
    result = dict(replay.get("refresh",{}),date=now_iso()[:10],previousEnd=previous_end,
        refreshedThrough=bars[-1]["date"],extendedEnd=bars[-1]["date"],
        latestAppendedBars=len(bars)-prior_rows,capturedAt=audit["captured_at_utc"],
        partialDatesExcluded=audit.get("excluded_in_progress",[]),historicalRowsUnchanged=True,strategyReplayed=True,
        strategyParametersChanged=False,dataPolicy="append_only",automaticScheduleEnabled=True)
    if audit.get("provisional_saved_tail"):
        result.update(dataPolicy="append_only_with_explicit_provisional_tail_finalization",
            provisionalTailReplaced=audit["provisional_saved_tail"],verifiedHistoricalRowsUnchanged=True)
    # Older NKD snapshots had no cumulative-count anchor. Establish one from
    # the actual prior saved end, not an invented original acquisition date.
    result.setdefault("frozenThrough",previous_end)
    result["appendedBars"] = sum(b["date"]>result["frozenThrough"] for b in bars)
    return result


def add_unfilled_sells(ns):
    """Decision evidence comes from close-time logs, never a fictitious fill."""
    known = {d["id"] for d in ns["decisions"]}
    result = []
    bars, day, mark, level, step, visual = [ns[n] for n in ["bars","day","mark","level","step","visual"]]
    for sd in ns["jl"]("campaign_sell_decisions"):
        if sd["order_id"] in known:
            continue
        i = sd["index"]
        t = sd["representative_selection"]["local_trigger"]
        if not t:
            ev,steps,route,reason=protective_sell(ns,sd)
            c=ns["contexts"][sd["position_id"]]
            result.append(dict(id=sd["order_id"],index=i,date=day(i),side="SELL",route=route,
                reason=reason,zone_id=None,initial_stop=c["stop"],visual=json.dumps(ev,ensure_ascii=False,separators=(",",":"))))
            continue
        atr = t["h1"]["atr_at_origin"]
        boundary = t["reference"]["price"]-ns["meta"]["legacy_rules"]["structure_buffer_atr"]*atr
        assert t["first_breaks"]["early"] == i-1 and all(bars[j]["close"]<boundary for j in [i-1,i])
        assert t["peak"]-bars[i]["close"] >= ns["meta"]["legacy_rules"]["sell_departure_atr"]*atr
        marks = [mark("1","H1",t["h1"]["origin_index"],t["h1"]["price"],t["h1"]["available_index"]),
                 mark("N","ネックライン",t["neckline"]["origin_index"],t["neckline"]["price"],t["neckline"]["available_index"]),
                 mark("2","H2",t["peak_index"],t["peak"],i),
                 mark("3a","1本目の終値",i-1,bars[i-1]["close"],i-1),mark("3b","SELL判断",i,bars[i]["close"],i)]
        levels = [level("sell-boundary","SELL判定境界",boundary,t["levels_known_index"]),
                  level("local-support","局所支持",t["reference"]["price"],t["levels_known_index"])]
        steps = [step("1","H1・H2と下落を確認",f"H1 {day(t['h1']['origin_index'])}、H2 {day(t['peak_index'])}。下落幅は固定ATR条件以上。"),
                 step("2","支持下の2終値を確認",f"{day(i-1)} と {day(i)} の終値 < {boundary:,.2f}。"),
                 step("3","翌始値決済を予約","判断時点では翌始値も決済結果も未観測。新規ショートではない。")]
        ev = visual("sell",i,"H1 → H2 → 支持下の2終値 → 翌始値SELL予約",max(0,t["h1"]["origin_index"]-10),marks,levels,steps)
        c = ns["contexts"][sd["position_id"]]
        result.append(dict(id=sd["order_id"],index=i,date=day(i),side="SELL",route="通常SELL",
            reason="\n".join(s["title"]+"："+s["detail"] for s in steps),zone_id=None,initial_stop=c["stop"],
            visual=json.dumps(ev,ensure_ascii=False,separators=(",",":"))))
    return result


def export(market, work, before, *, migration=False):
    dest = work/"result"; app = dest/"app/src"
    app.mkdir(parents=True,exist_ok=True)
    save(app/"data.json",before)
    meta = load(dest/"campaign_replay/metadata.json")
    slip = meta["execution_config"]["tick_size"]*meta["execution_config"]["slippage_ticks"]
    pv = meta["execution_config"]["point_value_usd"]
    commission = meta["execution_config"]["commission_per_contract_side_usd"]
    source = (CLOUD/"export_template.py.txt").read_text(encoding="utf-8")
    for a,b in [
        ("OUT=Path(__file__).resolve().parent",f"OUT=Path({str(dest)!r})"),
        ("BASE=OUT/'NKD_analysis/campaign_replay'","BASE=OUT/'campaign_replay'"),
        ("baseline=load(OUT.parents[1]/'nq_ict_campaign_examples_20260921/validation_rev11_without_continuation_20260923/metadata.json')",f"baseline=load(Path({str(work/'seed/metadata.json')!r}))"),
        ("assert extra_config=={'buy_htf_filter':'none'},extra_config","assert extra_config=={},extra_config"),
        ("assert slip==5 and meta['execution_config']['point_value_usd']==1",f"assert slip=={slip!r} and meta['execution_config']['point_value_usd']=={pv!r}"),
        ("assert meta['execution_config']['commission_per_contract_side_usd']==0",f"assert meta['execution_config']['commission_per_contract_side_usd']=={commission!r}"),
        ("assert abs(pnl-(r['price']-buys[pid]['price']))<1e-8,'independent points ledger'",f"assert abs(pnl-((r['price']-buys[pid]['price'])*{pv!r}-2*{commission!r}))<1e-7,'independent execution ledger'"),
        ("load(OUT/'NKD_F/fetch_manifest.json')['captured_at_utc']",f"load(Path({str(work/'fetch_audit.json')!r}))['captured_at_utc']"),
        ("decision_ids={d['id'] for d in decisions}","decisions.extend(_pending_sells(globals()))\ndecision_ids={d['id'] for d in decisions}")
    ]:
        source = replace_once(source,a,b)
    source = source.replace("2026-10-14以前の有効配信日足",meta["end"][:10]+"以前の有効配信日足")
    for old,new in [("−5pt",f"−{slip:g}pt"),("片道5pt",f"片道{slip:g}pt")]:
        source = source.replace(old,new)
    # Do not execute the NKD-only performance-summary tail for NQ. Its units
    # are intentionally instrument-specific; export/QA is not a new analysis.
    marker = "pd_fills=__import__('pandas').DataFrame"
    if source.count(marker)!=1:
        raise ValueError("Reviewed exporter summary boundary changed")
    source = source.split(marker,1)[0]
    if market in {"nq", "es"}:
        symbol={"nq":"NQ=F","es":"ES=F"}[market]
        source = source.replace("NKD=F",symbol).replace("CMEドル建て / ","").replace("日経先物 "+symbol,symbol)
        source = source.replace(":03d}",":02d}")
    if market == "es":
        from cloud.es_view import retest_labels
        source=retest_labels(source)
    source=adapt_template(source)
    namespace = {"__file__":str(dest/"export_reviewed.py"),"_pending_sells":add_unfilled_sells,"_protective_sell":protective_sell}
    exec(compile(source,str(dest/"export_reviewed.py"),"exec"),namespace)
    fresh = load(app/"data.json")
    if market in {"nq", "es"}:
        for row in fresh["queries"]["executions"]["rows"]:
            row["net_pnl_usd"] = row.pop("net_pnl_points")
            row.pop("net_pnl_usd_reference",None)
        # Keep NQ's original, accurate provenance/contract units rather than
        # inheriting the NKD exporter's point-denominated descriptive text.
        for name,q in fresh["queries"].items():
            q["source"] = deepcopy(before["queries"][name]["source"])
        fresh["title"] = before["title"]
        fresh["replay"].update(pointValue=pv,commissionSide=commission,slippagePoints=slip)
    n = len(before["queries"]["market_prices"]["rows"])
    if migration:
        assert meta["legacy_rules"]["buy_reference_radius"] == 1
        assert fresh["queries"]["market_prices"]["rows"] == before["queries"]["market_prices"]["rows"], "Release must not change prices"
    reconciled = {}
    exact = {"market_prices","primary_zones","tactical_zones","daily_states"}
    for name,query in before["queries"].items():
        current = fresh["queries"][name]["rows"]
        if migration:
            if name == "primary_zones":
                assert current == query["rows"], "Release must not change primary OB input/lifecycle"
            reconciled[name] = len(query["rows"]) if name in {"market_prices", "primary_zones"} else 0
            continue
        prefix = [r for r in current if r["index"]<n]
        if name in exact:
            assert prefix == query["rows"],f"Historical data changed: {name}"
        else:
            by_id = {r["id"]:r for r in prefix}
            assert len(by_id)==len(prefix),f"Duplicate {name} IDs"
            for old in query["rows"]:
                assert old["id"] in by_id,f"Historical {name} ID disappeared"
                new = by_id[old["id"]]
                fields = ["id","index","date","side","kind","price","position_id","order_id","decision_index","decision_date","initial_stop","zone_id","contact_date","net_pnl_usd","net_pnl_points","source_line"]
                for field in fields:
                    if field in old and field in new:
                        assert old[field]==new[field],(name,old["id"],field)
                if "gates" in old and "gates" in new:
                    assert json.loads(old["gates"])==json.loads(new["gates"])
        fresh["queries"][name]["rows"] = deepcopy(query["rows"])+[r for r in current if r["index"]>=n]
        reconciled[name] = len(query["rows"])
    # Existing compiled NQ UI's legacy text does not support a new FVG route.
    # Never silently mislabel that exceptional route: retain the verified
    # generation until the explicitly reviewed renderer adapter is upgraded.
    if market == "nq" and any(json.loads(r["visual"]).get("fvg_used") for r in fresh["queries"]["decisions"]["rows"] if migration or r["index"]>=n):
        raise ValueError("New NQ FVG route requires reviewed UI support; old published generation retained")
    now = now_iso(); fresh["id"] = before["id"]; fresh["buildStatus"] = "complete"; fresh["generatedAt"] = now
    replay = dict(before["replay"],**fresh["replay"])
    replay.update(strategyVersion=STRATEGY_VERSION, revision="rev12_radius_one",
        baseRevision="rev11_without_continuation", buyReferenceRadius=1)
    replay["defaultDate"] = replay["end"]
    replay["refresh"] = refresh_metadata(before["replay"],fresh["queries"]["market_prices"]["rows"],
        load(work/"fetch_audit.json"),before["replay"]["end"],n)
    if migration:
        replay["refresh"].update(strategyParametersChanged=True, historicalRowsUnchanged=False,
            strategyMigration="v0.11 -> v0.12; radius 2 -> 1 only; full history recomputed", dataPolicy="frozen_prices_strategy_release")
        for key in ["provisionalTailReplaced", "verifiedHistoricalRowsUnchanged"]:
            replay["refresh"].pop(key,None)
    fresh["replay"] = replay
    fresh["title"] = fresh["title"].replace("v0.11",STRATEGY_VERSION)
    for name,q in fresh["queries"].items():
        q["source"]["label"] = q["source"]["label"].replace("v0.11",STRATEGY_VERSION)
        q["source"]["executedAt"] = now
        q["source"].setdefault("evidenceFlow",[]).append(dict(title="v0.12リリース" if migration else "クラウド更新",detail=("価格・原本OB入力を固定し、buy_reference_radius 2→1と確定済み半径1ピボット供給だけを変更して全履歴を再計算。β・γは不採用。" if migration else "確定済み日足を追加し、固定v0.12設定で全履歴再計算と既存履歴を照合。過去入力の遡及変更なし。")))
    normalize_current_labels(fresh)
    save(dest/"snapshot.json",fresh)
    save(dest/"export_audit.json",dict(status="PASS_EXPLICIT_STRATEGY_MIGRATION" if migration else "PASS_REPLAY_FACTS_AND_PRESERVED_REVIEWED_HISTORY",preserved_rows=reconciled))
    return fresh
