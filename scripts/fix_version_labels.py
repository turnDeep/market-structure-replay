"""One-time presentation repair. No market fetch, strategy run or row changes."""
from copy import deepcopy
from pathlib import Path
import json
import shutil
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from cloud.common import digest, now_iso, save
from cloud.refresh import hydrate_html
from cloud.version_labels import current_labels, normalize_current_labels
from replay_assets import load_manifest,load_replay
from scripts.import_replays import canonical,package,read_snapshot
from scripts.release_v012 import presentation_labels


def repair():
    before=load_manifest();manifest=deepcopy(before)
    archive=ROOT/"assets/archive/v011"
    archive_pins={p.name:digest(p.read_bytes()) for p in archive.iterdir() if p.is_file()}
    audit=dict(status="PASS_VERSION_LABEL_REPAIR",date_utc=now_iso(),strategy_unchanged=True,
        market_data_fetched=False,query_rows_changed=False,markets={})
    with tempfile.TemporaryDirectory(prefix="version-labels-",dir=ROOT/".refresh-work") as tmp:
        stage=Path(tmp)
        for market,item in before["markets"].items():
            html=load_replay(market,item);snap=read_snapshot(html)
            original=deepcopy(snap)
            changes=normalize_current_labels(snap)
            for name,q in snap["queries"].items():
                assert q["rows"]==original["queries"][name]["rows"]
            assert snap["replay"]==original["replay"]
            path=stage/(market+".html")
            path.write_text(presentation_labels(hydrate_html(html,snap)),encoding="utf-8")
            updated=package(path,market,stage)
            assert updated["queries"]==item["queries"],"All price/trade/evidence row hashes must remain unchanged"
            updated["data_quality"]=item["data_quality"]
            manifest["markets"][market]=updated
            audit["markets"][market]=dict(changed_active_labels=changes,current_labels=dict(current_labels(snap)),
                all_query_row_hashes_unchanged=True,compiled_footer_shows_current_model=True)
        assert load_manifest()==before,"Concurrent publisher changed this generation"
        for market in before["markets"]:
            shutil.copyfile(stage/(market+".html.gz"),ROOT/"assets"/(market+".html.gz"))
        manifest["presentation_updated_at"]=now_iso()
        save(ROOT/"assets/manifest.json",manifest)
    assert archive_pins=={p.name:digest(p.read_bytes()) for p in archive.iterdir() if p.is_file()}
    audit["v011_archive_unchanged"]=True
    save(ROOT/"qa/version-label-audit-20261019.json",audit)
    print(json.dumps(audit,ensure_ascii=False,indent=2))


if __name__=="__main__":repair()
