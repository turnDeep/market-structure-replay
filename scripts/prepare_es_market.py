"""Import the already-tested ES alpha inputs; append completed Yahoo bars only.

No production view is replaced here. Run the ordinary isolated replay worker
after preparation, then independently reconcile it with the alpha experiment.
"""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from cloud.common import CLOUD,FACT_FIELDS,digest,load,save
from cloud.refresh import fetch
from cloud.strategy import stamp_metadata


def prepare(work,research):
    work.mkdir(parents=True,exist_ok=False)
    seed=work/"seed";seed.mkdir()
    study=research/"es_v011_comparison_20261016"
    inputs=study/"inputs/es"
    alpha=research/"v012_cross_market_20261019/arms/es/v012_alpha"
    expected=load(alpha/"result.json")
    assert expected["status"]=="COMPLETE" and expected["symbol"]=="ES=F"
    assert expected["extra_supplied_pivot_radii"]==[1]
    # Preserve all vendor bytes except the explicitly reviewed symbol literal.
    for source in sorted((CLOUD/"vendor/nq").rglob("*.py")):
        rel=source.relative_to(CLOUD/"vendor/nq")
        raw=source.read_bytes().replace(b"NQ=F",b"ES=F")
        previous=(study/"engine_es"/rel).read_text(encoding="utf-8")
        assert raw.decode("utf-8").replace("\r\n","\n")==previous.replace("\r\n","\n"),rel
        dest=CLOUD/"vendor/es"/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
    meta=load(study/"replays/es/metadata.json")
    for name,record in meta["input_hashes"].items():
        filename=name+(".csv" if name in ["prices","objects","contexts"] else ".json")
        raw=(inputs/filename).read_bytes()
        assert digest(raw)==record["sha256"],name
        (seed/filename).write_bytes(raw)
        record["path"]=filename
    config=load(seed/"config.json")
    config["legacy_rules_overrides"]["buy_reference_radius"]=1
    save(seed/"config.json",config)
    meta["input_hashes"]["config"]["sha256"]=digest((seed/"config.json").read_bytes())
    meta["legacy_rules"]["buy_reference_radius"]=1
    for key in ["legacy_rules","campaign_config","execution_config"]:
        assert meta[key]==expected["settings"][key],key
    stamp_metadata(meta);save(seed/"metadata.json",meta)
    events=load(alpha/"execution_events.json")
    save(seed/"execution_facts.json",[[r.get(k) for k in FACT_FIELDS] for r in events])
    shutil.copyfile(inputs/"excluded_invalid_ohlc_rows.csv",seed/"excluded_invalid_ohlc_rows.csv")
    save(seed/"quality.json",dict(provisional_from_date=None,verified_through="2026-10-14",
        reason="Frozen audited ES study inputs; actual vendor revisions are not retroactively applied."))
    save(work/"import_audit.json",dict(status="PREPARED_NOT_PUBLISHED",symbol="ES=F",strategy="v0.12",
        research_alpha_closed=expected["closed_trades"],original_rows=expected["rows"],
        original_end=expected["end"],execution_config=meta["execution_config"],NQ_data_or_trades_changed=False))
    has_new=fetch("es",work,meta,datetime.now(timezone.utc))
    if not has_new:
        raise ValueError("No new ES completed bar; use an explicitly frozen full replay, not a fabricated observation")
    print("ES_PREPARED_COMPLETED_ROWS",load(work/"fetch_audit.json")["source_end"],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--work",type=Path,required=True);p.add_argument("--research",type=Path,required=True)
    a=p.parse_args();prepare(a.work.resolve(),a.research.resolve())
