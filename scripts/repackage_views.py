"""Repackage selected compiled UI updates, retaining all reviewed rows and policy."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from replay_assets import load_manifest,load_replay
from scripts.import_replays import package,read_snapshot


def main():
    p=argparse.ArgumentParser();p.add_argument("--nq",type=Path);p.add_argument("--es",type=Path);args=p.parse_args()
    selected=[(market,source) for market,source in [("nq",args.nq),("es",args.es)] if source is not None]
    if not selected:p.error("At least one reviewed --nq or --es HTML is required")
    manifest=load_manifest()
    with tempfile.TemporaryDirectory(prefix="replay-copy-only-") as temporary:
        staged=Path(temporary)
        for market,source in selected:
            old=manifest["markets"][market]
            before=read_snapshot(load_replay(market,old))
            after=read_snapshot(source.read_text(encoding="utf-8"))
            assert before["id"]==after["id"]
            assert {k:v["rows"] for k,v in before["queries"].items()}=={k:v["rows"] for k,v in after["queries"].items()}
            item=package(source,market,staged)
            item["data_quality"]=old["data_quality"]
            manifest["markets"][market]=item
        manifest["packaged_at"]=datetime.now(timezone.utc).isoformat()
        (staged/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        for name in [market+".html.gz" for market,_ in selected]+["manifest.json"]:
            os.replace(staged/name,ROOT/"assets"/name)
    print("PASS_UI_ONLY: selected views updated; all reviewed rows, app IDs, schedule/quality and unselected assets retained")


if __name__=="__main__":main()
