"""Explicit adapter smoke check against an existing completed local run."""
import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile

from cloud.common import ROOT, load, materialize_seed, save
from cloud.export_adapter import export
from replay_assets import load_manifest, load_replay
from scripts.import_replays import read_snapshot


def main():
    p=argparse.ArgumentParser();p.add_argument("--source-run",type=Path,required=True);args=p.parse_args()
    for market,item in load_manifest()["markets"].items():
        with tempfile.TemporaryDirectory(prefix="v011-export-check-") as temporary:
            work=Path(temporary)
            materialize_seed(market,work/"seed")
            result=work/"result/campaign_replay";result.mkdir(parents=True)
            for name in ["metadata.json","execution_events.jsonl","entry_structural_contexts.json","entry_audit.jsonl","close_plans.jsonl","tactical_zone_events.jsonl","primary_zone_state_events.jsonl","daily_states.jsonl","campaign_sell_decisions.jsonl"]:
                shutil.copyfile(args.source_run/market/"campaign_replay"/name,result/name)
            shutil.copyfile(args.source_run/market/"fetch_audit.json",work/"fetch_audit.json")
            before=read_snapshot(load_replay(market,item));save(work/"before.json",before)
            export(market,work,before)
            subprocess.run(["node",str(ROOT/"cloud/verify_view.mjs"),market,str(work)],check=True)
            print(market,load(work/"result/export_audit.json"),flush=True)


if __name__=="__main__":main()
