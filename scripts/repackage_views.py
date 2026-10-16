"""Repackage compiled copy-only UI changes, retaining all reviewed rows and policy."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from replay_assets import load_manifest,load_replay
from scripts.import_replays import package,read_snapshot


def main():
    p=argparse.ArgumentParser();p.add_argument("--nq",type=Path,required=True);p.add_argument("--nikkei",type=Path,required=True);args=p.parse_args()
    manifest=load_manifest()
    with tempfile.TemporaryDirectory(prefix="replay-copy-only-") as temporary:
        staged=Path(temporary)
        for market,source in [("nq",args.nq),("nikkei",args.nikkei)]:
            old=manifest["markets"][market]
            before=read_snapshot(load_replay(market,old))
            after=read_snapshot(source.read_text(encoding="utf-8"))
            assert before["id"]==after["id"]
            assert {k:v["rows"] for k,v in before["queries"].items()}=={k:v["rows"] for k,v in after["queries"].items()}
            item=package(source,market,staged)
            item["data_quality"]=old["data_quality"]
            manifest["markets"][market]=item
        (staged/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        for name in ["nq.html.gz","nikkei.html.gz","manifest.json"]:
            os.replace(staged/name,ROOT/"assets"/name)
    print("PASS_COPY_ONLY: all reviewed query rows and app IDs unchanged; schedule/quality retained")


if __name__=="__main__":main()
