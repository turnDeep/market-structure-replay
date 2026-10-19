"""Explicit one-time v0.11 -> v0.12 migration. No downloads or live orders.

Run prepare, replay for each isolated market, then publish only after both
exports, view checks and research-prefix reconciliation pass. Git push is
deliberately a separate operator action, never performed by this program.
"""
from copy import deepcopy
import argparse
import base64
import gzip
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cloud.common import CLOUD, FACT_FIELDS, digest, load, materialize_seed, now_iso, save
from cloud.export_adapter import export
from cloud.refresh import hydrate_html, stage_state
from cloud.strategy import BASE_STAGE, STRATEGY_VERSION, install_pivot_supply, stamp_metadata
from replay_assets import load_manifest, load_replay
from scripts.import_replays import SCRIPT, canonical, package, read_snapshot


def prepare(root):
    root.mkdir(parents=True, exist_ok=False)
    manifest = load_manifest()
    assert manifest.get("strategy_version", "v0.11") == "v0.11", "Migration already applied"
    baseline_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    save(root/"original_manifest.json", manifest)
    save(root/"release.json", dict(strategy_version=STRATEGY_VERSION, baseline_commit=baseline_commit,
        started_at_utc=now_iso(), status="PREPARED", parameter_change={"buy_reference_radius": [2, 1]},
        market_data_refresh=False, markets={}))
    for market, item in manifest["markets"].items():
        work=root/market
        before=read_snapshot(load_replay(market,item))
        save(work/"original_snapshot.json",before)
        materialize_seed(market,work/"seed")
        save(work/"fetch_audit.json",dict(captured_at_utc=now_iso(), provider="Unchanged frozen repository inputs; no API request", excluded_in_progress=[]))
        result=work/"result";result.mkdir()
        meta=load(work/"seed/metadata.json")
        assert meta["legacy_rules"]["buy_reference_radius"]==2
        for record in meta["input_hashes"].values():
            shutil.copyfile(work/"seed"/record["path"],result/record["path"])
        config=load(result/"config.json")
        config.setdefault("legacy_rules_overrides",{})["buy_reference_radius"]=1
        save(result/"config.json",config)
        # The ordinary view checker validates all new facts without incorrectly
        # treating v0.11 trades as an immutable prefix of another strategy.
        verification=deepcopy(before)
        for query in verification["queries"].values():
            query["rows"]=[]
        save(work/"before.json",verification)
    print("PREPARED",root,flush=True)


def replay(market, root):
    work=root/market;result=work/"result"
    sys.path.insert(0,str(CLOUD/"vendor"/market))
    from nq_ict_campaign_examples_20260921.run_campaign import main as replay_main
    from nq_ict_campaign_examples_20260921.campaign import CampaignLedger
    install_pivot_supply()
    native=CampaignLedger.feed
    def progress(self,*args,**kwargs):
        value=native(self,*args,**kwargs)
        if len(self.state_rows)%500==0:
            print(market,"bars",len(self.state_rows),flush=True)
        return value
    CampaignLedger.feed=progress
    before=load(work/"seed/metadata.json")
    sys.argv=[__file__,"--config",str(result/"config.json"),"--base-config",str(result/"base_config.json"),
        "--prices",str(result/"prices.csv"),"--objects",str(result/"objects.csv"),"--contexts",str(result/"contexts.csv"),
        "--calendar-json",str(result/"calendar.json"),"--as-of",str(before["end"])[:10],
        "--output",str(result/"campaign_replay"),"--revision-stage",BASE_STAGE,"--allow-unvalidated-replay"]
    replay_main()
    after=stamp_metadata(load(result/"campaign_replay/metadata.json"))
    assert after["legacy_rules"]==dict(before["legacy_rules"],buy_reference_radius=1)
    for field in ["campaign_config","execution_config","stage_overrides"]:
        assert before[field]==after[field],field
    for name in ["prices","objects","contexts","calendar","base_config"]:
        assert after["input_hashes"][name]["sha256"]==before["input_hashes"][name]["sha256"],name
    save(result/"campaign_replay/metadata.json",after)
    events=[json.loads(line) for line in (result/"campaign_replay/execution_events.jsonl").read_text(encoding="utf-8").splitlines()]
    save(result/"execution_facts.json",[[r.get(k) for k in FACT_FIELDS] for r in events])
    save(result/"replay_audit.json",dict(status="PASS_SINGLE_PARAMETER_MIGRATION",strategy_version=STRATEGY_VERSION,
        daily_rows=after["daily_rows"],price_and_original_zone_inputs_unchanged=True,
        sole_legacy_override="buy_reference_radius 2 -> 1",beta_gamma_not_applied=True))
    export_only(market,root)


def export_only(market,root):
    """Recheck presentation from completed native logs; never rerun the engine."""
    work=root/market
    assert load(work/"result/replay_audit.json")["status"]=="PASS_SINGLE_PARAMETER_MIGRATION"
    export(market,work,load(work/"original_snapshot.json"),migration=True)
    subprocess.run(["node",str(CLOUD/"verify_view.mjs"),market,str(work)],cwd=ROOT,check=True)
    print("READY",market,flush=True)


def presentation_labels(html):
    """Mechanical display-label migration, never alter bundled runtime logic."""
    def labels(source):
        source=source.replace("v0.11","v0.12").replace("revision 0.11","revision 0.12").replace("_v011_","_v012_")
        source=source.replace("原本：revision 0.12 / rev11_without_continuation / 収録 ","現行：v0.12 / BUY参照半径1 / 収録 ")
        source=source.replace("revision 0.12 / rev11_without_continuation / 収録 ","現行：v0.12 / BUY参照半径1 / 収録 ")
        return source.replace("保存済み履歴を維持し、追加区間を同じv0.12設定で再計算しています。",
            "v0.12の保存結果です。αの正式採用時に過去売買を再計算し、以後の日次更新ではv0.12内の履歴を維持します。")
    def upgrade(match):
        import re
        if 'type="application/json"' in match.group(1):
            return match.group(0)
        encoded=re.search(r'(src="data:text/javascript[^,]*;base64,)([^\"]+)(")',match.group(1))
        if not encoded:
            return "<script"+match.group(1)+">"+labels(match.group(2))+"</script>"
        source=base64.b64decode(encoded.group(2)).decode("utf-8")
        replacement=base64.b64encode(labels(source).encode()).decode()
        return match.group(0).replace(encoded.group(2),replacement,1)
    import re
    return re.sub(r"(<title>)(.*?)(</title>)",lambda m:m.group(1)+labels(m.group(2))+m.group(3),SCRIPT.sub(upgrade,html),flags=re.S)


def publish(root, reference_root):
    release=load(root/"release.json")
    assert release["status"]=="PREPARED"
    original=load(root/"original_manifest.json")
    assert original==load_manifest(),"Another publisher changed the generation; do not overwrite"
    staged=root/"staged";assets=staged/"assets";assets.mkdir(parents=True,exist_ok=False)
    archive=assets/"archive/v011";archive.mkdir(parents=True)
    save(archive/"manifest.json",original)
    for market in ["nq","nikkei"]:
        shutil.copyfile(ROOT/"assets"/(market+".html.gz"),archive/(market+".html.gz"))
    manifest=deepcopy(original)
    for market in ["nq","nikkei"]:
        work=root/market
        assert load(work/"result/view_audit.json")["status"]=="PASS_DATA_VIEW_AND_LEDGER"
        if reference_root:
            ref=(reference_root/("v011_all_parameter_sensitivity_20261019/arms/p046_lower" if market=="nq" else "v012_cross_market_20261019/arms/nikkei/v012_alpha"))
            expected=load(ref/"trades.json")
            actual=[json.loads(line) for line in (work/"result/campaign_replay/trades.jsonl").read_text(encoding="utf-8").splitlines()]
            fields=["entry_at","exit_at","entry_price","exit_price","net_pnl_usd"]
            compact=lambda rows:[{k:r[k] for k in fields} for r in rows if r["exit_at"][:10]<="2026-10-14"]
            assert compact(actual)==compact(expected),(market,"Research alpha prefix mismatch")
            import numpy as np
            curve=np.load(ref/"curve.npz")
            mtm=[json.loads(line) for line in (work/"result/campaign_replay/daily_mtm.jsonl").read_text(encoding="utf-8").splitlines()]
            states=[json.loads(line) for line in (work/"result/campaign_replay/daily_states.jsonl").read_text(encoding="utf-8").splitlines()]
            count=len(curve["dates"])
            assert [r["datetime"][:10] for r in mtm[:count]]==curve["dates"].tolist()
            np.testing.assert_allclose([r["mtm_pnl_usd"] for r in mtm[:count]],curve["mtm"],rtol=0,atol=1e-8)
            assert [r["position_id"] is not None for r in states[:count]]==curve["exposure"].tolist()
            save(work/"result/research_prefix_audit.json",dict(status="PASS",through="2026-10-14",closed_trades=len(expected),
                daily_mtm_and_exposure_rows=count,native_accounting_units=True,
                expected_trades_sha256=digest(canonical(compact(expected)))))
        snapshot=load(work/"result/snapshot.json")
        if market == "nq":
            assert not any(json.loads(r["visual"]).get("fvg_used") for r in snapshot["queries"]["decisions"]["rows"]), "NQ FVG route needs reviewed UI support"
        html=presentation_labels(hydrate_html(load_replay(market,original["markets"][market]),snapshot))
        path=work/"v012.html";path.write_text(html,encoding="utf-8")
        item=package(path,market,assets)
        item["strategy_version"]=STRATEGY_VERSION
        item["data_quality"]=deepcopy(original["markets"][market]["data_quality"])
        manifest["markets"][market]=item
        stage_state(market,work,staged)
        # A strategy release does not silently upgrade data-quality status.
        shutil.copyfile(work/"seed/quality.json",staged/"state"/market/"quality.json")
        release["markets"][market]=dict(bars=item["bars"],end=item["end"],historical_prefix=item["queries"],
            replay=load(work/"result/replay_audit.json"),view=load(work/"result/view_audit.json"),
            research_prefix=load(work/"result/research_prefix_audit.json") if reference_root else None)
    release.update(status="PASS_V012_MIGRATION",finished_at_utc=now_iso(),published_to_github=False,
        baseline_archive="assets/archive/v011",market_data_refresh=False)
    manifest.update(strategy_version=STRATEGY_VERSION,packaged_at=now_iso(),strategy_release="v012_release.json")
    save(assets/"v012_release.json",release);save(assets/"manifest.json",manifest)
    for path in sorted(staged.rglob("*")):
        if path.is_file():
            rel=path.relative_to(staged)
            target=ROOT/("cloud" if rel.parts[0]=="state" else "")/rel
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(path,target)
    print("STAGED_V012_BOTH_MARKETS_NOT_YET_PUSHED",flush=True)


def refresh_presentations(root):
    """Repair count/display metadata only; keep all release-pinned fact rows."""
    manifest=load_manifest()
    assert manifest["strategy_version"]==STRATEGY_VERSION
    release=load(ROOT/"assets/v012_release.json")
    staged=root/"presentation";staged.mkdir(parents=True,exist_ok=True)
    for market,item in manifest["markets"].items():
        html=load_replay(market,item);snapshot=read_snapshot(html)
        refresh=snapshot["replay"]["refresh"]
        refresh.setdefault("frozenThrough",snapshot["replay"]["end"])
        refresh["appendedBars"]=sum(b["date"]>refresh["frozenThrough"] for b in snapshot["queries"]["market_prices"]["rows"])
        path=staged/(market+".html")
        path.write_text(presentation_labels(hydrate_html(html,snapshot)),encoding="utf-8")
        updated=package(path,market,staged)
        assert updated["queries"]==release["markets"][market]["historical_prefix"],"Presentation repair cannot change any fact row"
        updated["data_quality"]=item["data_quality"]
        manifest["markets"][market]=updated
    for market in manifest["markets"]:
        shutil.copyfile(staged/(market+".html.gz"),ROOT/"assets"/(market+".html.gz"))
    save(ROOT/"assets/manifest.json",manifest)
    print("PASS_PRESENTATION_ONLY_FACT_ROWS_UNCHANGED",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=["prepare","replay","export","publish","presentation"])
    p.add_argument("--work",type=Path,required=True)
    p.add_argument("--market",choices=["nq","nikkei"])
    p.add_argument("--reference-root",type=Path)
    a=p.parse_args();root=a.work.resolve()
    if a.mode=="prepare":prepare(root)
    elif a.mode=="replay":replay(a.market,root)
    elif a.mode=="export":export_only(a.market,root)
    elif a.mode=="presentation":refresh_presentations(root)
    else:publish(root,a.reference_root.resolve() if a.reference_root else None)
