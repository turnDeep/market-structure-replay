"""Fetch, replay, validate and stage a complete generation; never place orders.

GitHub publishes the resulting files in one ordinary commit after pytest.
Verification mode replays the last REAL saved bar from the preceding cutoff
without downloading anything and never updates production files.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
from datetime import datetime, timedelta, timezone
import gzip
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time

from cloud.common import CLOUD, ROOT, MARKETS, completed_cutoff, digest, load, materialize_seed, now_iso, save, validate_rows, verify_package
from cloud.export_adapter import export
from replay_assets import load_manifest, load_replay
from scripts.import_replays import SCRIPT, package, read_snapshot
from cloud.strategy import STRATEGY_VERSION


def fetch(market, work, meta, now):
    import pandas as pd
    import yfinance as yf
    from curl_cffi import requests
    end = str(meta["end"])[:10]
    start = (datetime.fromisoformat(end)-timedelta(days=14)).date().isoformat()
    options = dict(start=start,interval="1d",auto_adjust=False,back_adjust=False,repair=False,actions=False,keepna=True,timeout=30)
    failures = []
    with requests.Session(impersonate="chrome") as session:
        for attempt in range(1,4):
            try:
                frame = yf.Ticker(MARKETS[market],session=session).history(**options)
                if frame.empty:
                    raise ValueError("Empty response is not evidence of a holiday")
                break
            except Exception as exc:
                failures.append(dict(attempt=attempt,error=str(exc)[:240]))
                if attempt == 3:
                    raise
                time.sleep(3*attempt)
    frame.to_csv(work/"provider_raw.csv",encoding="utf-8-sig")
    rows = pd.DataFrame(dict(datetime=[x.date().isoformat() for x in frame.index],
        open=frame.Open.to_numpy(),high=frame.High.to_numpy(),low=frame.Low.to_numpy(),close=frame.Close.to_numpy(),
        adj_close=frame.get("Adj Close",frame.Close).to_numpy(),volume=frame.Volume.to_numpy()))
    source_timezone = str(frame.index.tz)
    cutoff = completed_cutoff(now,source_timezone)
    completed = rows.loc[rows.datetime.le(cutoff)].copy()
    records = completed.to_dict("records")
    validate_rows(records)
    original = list(csv.DictReader((work/"seed/prices.csv").read_text(encoding="utf-8-sig").splitlines()))
    quality = load(work/"seed/quality.json")
    verified_end = quality["verified_through"] if quality.get("provisional_from_date") else end
    if not records or records[-1]["datetime"]<verified_end:
        raise ValueError(f"{market}: source is stale or has no completed rows")
    prior = {r["datetime"][:10]:r for r in original}
    if verified_end not in set(completed.datetime):
        raise ValueError("Provider overlap does not include the last verified date")
    revisions = []
    for row in records:
        if row["datetime"] in prior:
            changed = [c for c in ["open","high","low","close","volume"] if float(row[c])!=float(prior[row["datetime"]][c])]
            if changed:
                revisions.append(dict(date=row["datetime"],fields=changed))
    completed.to_csv(work/"completed.csv",index=False,encoding="utf-8-sig")
    audit = dict(symbol=MARKETS[market],captured_at_utc=now.isoformat(),provider="yfinance / curl-cffi Chrome session",
        source_timezone=source_timezone,completed_cutoff=cutoff,request=options,completed_rows=len(records),
        source_end=records[-1]["datetime"],excluded_in_progress=rows.loc[rows.datetime.gt(cutoff),"datetime"].tolist(),
        overlap_revisions_not_applied=revisions,retries=failures,versions=dict(yfinance=yf.__version__),
        provisional_saved_tail=quality.get("provisional_from_date"),verified_saved_end=verified_end,
        cutoff_policy="Stricter of latest 17:00 America/Chicago boundary and strictly before the provider-local current date; no calendar interpolation")
    save(work/"fetch_audit.json",audit)
    return any(r["datetime"]>verified_end for r in records)


def truncate_seed(work, prior):
    """Discard only a specifically identified provisional/verification tail."""
    seed = work/"seed"; meta = load(seed/"metadata.json")
    remaining = 0
    for name,column in [("prices","datetime"),("contexts","datetime"),("objects","available_at")]:
        path = seed/(name+".csv")
        lines = path.read_bytes().splitlines(keepends=True)
        parsed = list(csv.DictReader(path.read_text(encoding="utf-8-sig").splitlines()))
        assert len(parsed)+1==len(lines),"Multiline CSV needs explicit verified handling"
        kept = [line for line,row in zip(lines[1:],parsed) if row[column][:10]<=prior]
        path.write_bytes(lines[0]+b"".join(kept))
        if name == "prices":
            remaining = len(kept)
    calendar = load(seed/"calendar.json")
    calendar["session_dates"] = [d for d in calendar["session_dates"] if d<=prior]
    calendar["coverage_end"] = prior
    save(seed/"calendar.json",calendar)
    meta["daily_rows"] = remaining; meta["end"] = prior+"T00:00:00"
    for record in meta["input_hashes"].values():
        record["sha256"] = digest((seed/record["path"]).read_bytes())
    save(seed/"metadata.json",meta)
    save(seed/"execution_facts.json",[r for r in load(seed/"execution_facts.json") if r[0]<meta["daily_rows"]])
    return remaining


def prepare_verification(work):
    """Remove one actual saved observation; rerun it without invented prices."""
    seed = work/"seed"
    prices = list(csv.DictReader((seed/"prices.csv").read_text(encoding="utf-8-sig").splitlines()))
    last, prior = prices[-1],prices[-2]["datetime"][:10]
    with (work/"completed.csv").open("w",newline="",encoding="utf-8") as handle:
        writer = csv.DictWriter(handle,fieldnames=list(last)); writer.writeheader(); writer.writerow(last)
    truncate_seed(work,prior)
    save(work/"fetch_audit.json",dict(symbol=last.get("ticker"),captured_at_utc=now_iso(),
        provider="Frozen, genuine last saved daily bar; offline regression, not a live fetch",source_end=last["datetime"][:10],
        overlap_revisions_not_applied=[]))


def hydrate_html(html, snapshot):
    encoded = json.dumps(snapshot,ensure_ascii=False,separators=(",",":"),allow_nan=False).replace("<","\\u003c")
    replaced = False
    def replace(match):
        nonlocal replaced
        if 'type="application/json"' not in match.group(1):
            return match.group(0)
        if replaced:
            return ""
        replaced = True
        return '<script type="application/json" id="data-app-reviewed-snapshot">'+encoded+'</script>'
    result = SCRIPT.sub(replace,html)
    if not replaced or read_snapshot(result)!=snapshot:
        raise ValueError("Compiled-view snapshot hydration failed")
    return result


def stage_state(market, work, staged):
    result = work/"result"; target = staged/"state"/market
    target.mkdir(parents=True,exist_ok=True)
    old = load(work/"seed/metadata.json")
    meta = load(result/"campaign_replay/metadata.json")
    slim = {k:meta[k] for k in old}
    for key in ["strategy_version", "strategy_base_stage", "strategy_change", "extra_supplied_pivot_radii"]:
        if key in meta:
            slim[key] = meta[key]
    for name,record in meta["input_hashes"].items():
        filename = name+(".csv" if name in ["prices","objects","contexts"] else ".json")
        raw = Path(record["path"]).read_bytes()
        (target/(filename+".gz")).write_bytes(gzip.compress(raw,mtime=0))
        slim["input_hashes"][name] = dict(path=filename,sha256=digest(raw))
    save(target/"metadata.json",slim)
    shutil.copyfile(result/"execution_facts.json",target/"execution_facts.json")
    shutil.copyfile(work/"seed/excluded_invalid_ohlc_rows.csv",target/"excluded_invalid_ohlc_rows.csv")
    save(target/"quality.json",dict(provisional_from_date=None,verified_through=str(meta["end"])[:10],
        reason="Admitted only after both CME boundary and provider-local date boundary; later vendor revisions remain possible and are not applied to frozen history."))


def replace_staged_file(source, target):
    """Atomically replace one file even when staging lives on another disk.

    Git publishes the complete generation in one commit after all checks. The
    per-file rename must use a sibling of the destination: GitHub-hosted /tmp
    and the checkout can be separate filesystems (rename otherwise raises EXDEV).
    """
    target = Path(target)
    target.parent.mkdir(parents=True,exist_ok=True)
    sibling = None
    try:
        with tempfile.NamedTemporaryFile(prefix="."+target.name+".",suffix=".tmp",
                                         dir=target.parent,delete=False) as handle:
            sibling = Path(handle.name)
            with Path(source).open("rb") as original:
                shutil.copyfileobj(original,handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(sibling,target)
    finally:
        if sibling is not None:
            sibling.unlink(missing_ok=True)


def run(mode, report_dir):
    verify_package()
    now = datetime.now(timezone.utc)
    original = load_manifest()
    report = dict(started_at_utc=now.isoformat(),mode=mode,status="RUNNING",markets={},
        strategy_parameters_changed=False,strategy_status="NOT_VALIDATED",strategy_version=STRATEGY_VERSION)
    report_dir.mkdir(parents=True,exist_ok=True)
    save(report_dir/"run.json",report)
    with tempfile.TemporaryDirectory(prefix="v011-cloud-") as temp:
        workroot = Path(temp); staged = workroot/"staged"
        staged.mkdir()
        htmls = {}; snapshots = {}; changed = []
        for market,item in original["markets"].items():
            work = workroot/market; work.mkdir()
            meta = materialize_seed(market,work/"seed")
            if meta["daily_rows"]!=item["bars"] or str(meta["end"])[:10]!=item["end"]:
                raise ValueError("Published view and cloud seed are from different generations")
            htmls[market] = load_replay(market,item)
            snapshots[market] = read_snapshot(htmls[market])
            save(work/"before.json",snapshots[market])
            if mode == "verify":
                prepare_verification(work); has_new = True
            else:
                has_new = fetch(market,work,meta,now)
            report["markets"][market] = dict(fetch=load(work/"fetch_audit.json"),new_data=has_new,previous_end=item["end"])
            if has_new:
                quality = load(work/"seed/quality.json")
                if mode == "refresh" and quality.get("provisional_from_date"):
                    old_n = truncate_seed(work,quality["verified_through"])
                    snapshot = deepcopy(snapshots[market])
                    for q in snapshot["queries"].values():
                        q["rows"] = [r for r in q["rows"] if r["index"]<old_n]
                    snapshot["replay"]["end"] = quality["verified_through"]
                    snapshots[market] = snapshot
                    save(work/"before.json",snapshot)
                    report["markets"][market]["bounded_provisional_tail_replaced"] = quality["provisional_from_date"]
                changed.append(market)
            save(report_dir/"run.json",report)
        if not changed:
            report.update(status="NO_NEW_COMPLETED_DATA",finished_at_utc=now_iso(),published=False)
            save(report_dir/"run.json",report)
            print(json.dumps(report,ensure_ascii=False)); return report
        # Separate interpreters prevent the symbol-only adapters from sharing
        # Python's import cache. Serialize to bound private-runner peak memory.
        for market in changed:
            work = workroot/market
            subprocess.run([sys.executable,"-m","cloud.replay_worker",market,str(work)],cwd=ROOT,check=True)
            fresh = export(market,work,snapshots[market])
            subprocess.run(["node",str(CLOUD/"verify_view.mjs"),market,str(work)],cwd=ROOT,check=True)
            if mode == "verify":
                for name,q in fresh["queries"].items():
                    if q["rows"]!=snapshots[market]["queries"][name]["rows"]:
                        raise ValueError("Offline regression did not reproduce the whole published generation")
            for name in ["replay_audit.json","export_audit.json","view_audit.json"]:
                report["markets"][market][name[:-5]] = load(work/"result"/name)
            report["markets"][market]["end"] = fresh["replay"]["end"]
            snapshots[market] = fresh
            save(report_dir/"run.json",report)
        if mode == "verify":
            report.update(status="PASS_OFFLINE_FULL_REPLAY",published=False,finished_at_utc=now_iso())
            save(report_dir/"run.json",report)
            print(json.dumps(report,ensure_ascii=False)); return report
        assets = staged/"assets"; assets.mkdir()
        manifest = deepcopy(original)
        for market in changed:
            work = workroot/market
            path = work/"updated.html"
            path.write_text(hydrate_html(htmls[market],snapshots[market]),encoding="utf-8")
            manifest["markets"][market] = package(path,market,assets)
            stage_state(market,work,staged)
            manifest["markets"][market]["data_quality"] = dict(provisional_from_date=None,verified_through=snapshots[market]["replay"]["end"])
        manifest.update(packaged_at=now_iso(),automatic_refresh=True,
            strategy_version=STRATEGY_VERSION,
            refresh_schedule=dict(provider="GitHub Actions",timezone="Asia/Tokyo",local_time="14:10",cadence="day_after_cme_equity_session",calendar="CME_Equity",calendar_gate=True,failure_policy="retain_previous_verified_generation"))
        report.update(status="PASS_STAGED_GENERATION",published=False,finished_at_utc=now_iso(),
            changed_markets=changed,publication="Staged locally; GitHub commit/push must succeed to publish")
        save(assets/"cloud_refresh_latest.json",report)
        save(assets/"manifest.json",manifest)
        # No production file is touched before BOTH market branches pass.
        # Atomic external publication is the single Git commit after pytest.
        for path in sorted(staged.rglob("*")):
            if path.is_file():
                rel = path.relative_to(staged)
                target = ROOT/("cloud" if rel.parts[0]=="state" else "")/rel
                replace_staged_file(path,target)
        save(report_dir/"run.json",report)
        print(json.dumps(report,ensure_ascii=False)); return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode",choices=["refresh","verify"],default="refresh")
    p.add_argument("--report-dir",type=Path,default=ROOT/".refresh-reports")
    args = p.parse_args()
    try:
        run(args.mode,args.report_dir)
    except Exception as exc:
        report = load(args.report_dir/"run.json") if (args.report_dir/"run.json").exists() else {}
        report.update(status="FAILED_NO_PUBLICATION",error=str(exc)[:500],finished_at_utc=now_iso(),published=False)
        save(args.report_dir/"run.json",report)
        raise


if __name__ == "__main__":
    main()
