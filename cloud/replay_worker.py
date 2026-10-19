"""Run one instrument in an isolated interpreter with frozen strategy sources."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

from cloud.common import CLOUD, FACT_FIELDS, digest, load, save
from cloud.strategy import install_pivot_supply, stamp_metadata, validate_strategy


def append_csv(old, tail, dest):
    import pandas as pd
    original = old.read_bytes()
    if not original.endswith(b"\n"):
        raise ValueError("CSV prefix lacks a newline")
    columns = pd.read_csv(old, nrows=0).columns.tolist()
    dest.write_bytes(original+tail.reindex(columns=columns).to_csv(index=False, header=False).encode("utf-8"))
    assert dest.read_bytes().startswith(original)


def run(market, work):
    import pandas as pd
    seed = work/"seed"; dest = work/"result"
    dest.mkdir(exist_ok=True)
    before = load(seed/"metadata.json")
    validate_strategy(before)
    cutoff = pd.Timestamp(before["end"]).normalize()
    engine = CLOUD/"vendor"/market
    sys.path.insert(0, str(engine))
    import nq_ict_detection_20260911_v03 as detector
    import nq_ict_evaluation_20260912_v08 as v08
    import nq_ict_evaluation_20260912_v09 as v09
    inputs = {k: seed/v["path"] for k, v in before["input_hashes"].items()}
    recent = pd.read_csv(work/"completed.csv")
    tail = recent.loc[recent.datetime.gt(cutoff.date().isoformat())].copy()
    if tail.empty:
        raise ValueError("Worker requires an actual new completed observation")
    append_csv(inputs["prices"], tail, dest/"prices.csv")
    daily, _ = detector.load_daily(dest/"prices.csv")
    daily = v08.add_model_features(daily)
    frames = {"1D": daily, "1W": v08.period_bars(daily,"W-FRI"), "1M": v08.period_bars(daily,"M")}
    frames = {tf: f if tf == "1D" else f.loc[f.certified_calendar].reset_index(drop=True) for tf, f in frames.items()}
    if market == "nikkei":
        v08.SENSITIVITIES = ["balanced"]
    print(f"{market}: detector {len(daily)} bars", flush=True)
    generated, _, pivots, parallel = v08.run_lanes(frames, 2, "process")
    generated = v08.add_object_metadata(generated)
    generated["rule_version"] = v09.SCRIPT_VERSION
    generated = v09.normalize_dates(generated, ["origin_time", "confirmed_at", "available_at"])
    gaps = set(pd.to_datetime(pd.read_csv(seed/"excluded_invalid_ohlc_rows.csv").datetime).dt.normalize())
    generated = v09.mark_gaps(generated, gaps)
    original = pd.read_csv(inputs["objects"], low_memory=False)
    additions = generated.loc[(pd.to_datetime(generated.available_at)>cutoff)&~generated.object_id.isin(original.object_id)].copy()
    for col in ["origin_time", "confirmed_at", "available_at"]:
        additions[col] = pd.to_datetime(additions[col],format="mixed",errors="raise").dt.strftime("%Y-%m-%d")
    append_csv(inputs["objects"], additions, dest/"objects.csv")
    objects = pd.read_csv(dest/"objects.csv",low_memory=False)
    assert objects.object_id.is_unique
    ctx_objects = objects.loc[objects.sensitivity.eq("balanced")&objects.timeframe.isin(["1W","1M"])].copy()
    for col in ["origin_time", "confirmed_at", "available_at"]:
        ctx_objects[col] = pd.to_datetime(ctx_objects[col],format="mixed")
    life = {tf:v08.zone_lifecycle_fast(ctx_objects.loc[ctx_objects.timeframe.eq(tf)],daily) for tf in ["1W","1M"]}
    tail_daily = daily.loc[pd.to_datetime(daily.datetime)>cutoff].reset_index(drop=True)
    original_state = v08.context_state_at
    def full_state(t, _idx, _tail, frame, pivs, tf):
        idx = int(pd.to_datetime(daily.datetime).searchsorted(pd.Timestamp(t)))
        return original_state(t,idx,daily,frame,pivs,tf)
    v08.context_state_at = full_state
    try:
        contexts = v08.build_contexts(tail_daily,frames,pivots,ctx_objects,life)
    finally:
        v08.context_state_at = original_state
    for col in ["datetime", "1w_context_bar_end", "1m_context_bar_end"]:
        contexts[col] = pd.to_datetime(contexts[col],errors="coerce").dt.strftime("%Y-%m-%d")
    append_csv(inputs["contexts"], contexts, dest/"contexts.csv")
    calendar = load(inputs["calendar"])
    calendar["session_dates"] += tail.datetime.tolist()
    assert len(set(calendar["session_dates"])) == len(calendar["session_dates"])
    calendar["coverage_end"] = tail.datetime.iloc[-1]
    calendar["calendar_status"] = "observed_input_dates_not_exchange_certified; yfinance append-only refresh"
    save(dest/"calendar.json",calendar)
    from nq_ict_campaign_examples_20260921.campaign import CampaignLedger
    from nq_ict_campaign_examples_20260921.run_campaign import main as replay_main
    install_pivot_supply()
    feed = CampaignLedger.feed; started = time.monotonic(); count = 0
    def progress(self,*a,**kw):
        nonlocal count
        result = feed(self,*a,**kw); count += 1
        if count%1000 == 0:
            print(f"{market}: replay {count}/{len(daily)} ({time.monotonic()-started:.0f}s)",flush=True)
        return result
    CampaignLedger.feed = progress
    sys.argv = [__file__,"--config",str(inputs["config"]),"--base-config",str(inputs["base_config"]),
        "--prices",str(dest/"prices.csv"),"--objects",str(dest/"objects.csv"),"--contexts",str(dest/"contexts.csv"),
        "--calendar-json",str(dest/"calendar.json"),"--as-of",tail.datetime.iloc[-1],"--output",str(dest/"campaign_replay"),
        "--revision-stage","rev11_without_continuation","--allow-unvalidated-replay"]
    replay_main()
    after = load(dest/"campaign_replay/metadata.json")
    stamp_metadata(after)
    save(dest/"campaign_replay/metadata.json",after)
    for field in ["campaign_config","legacy_rules","execution_config","stage_overrides"]:
        if before[field] != after[field]:
            raise ValueError(f"Strategy settings changed: {field}")
    facts = [[r.get(k) for k in FACT_FIELDS] for r in (json.loads(x) for x in (dest/"campaign_replay/execution_events.jsonl").read_text(encoding="utf-8").splitlines())]
    if [r for r in facts if r[0]<before["daily_rows"]] != load(seed/"execution_facts.json"):
        raise ValueError("Frozen execution history changed")
    for name, record in before["input_hashes"].items():
        if digest(inputs[name].read_bytes()) != record["sha256"]:
            raise ValueError("Original inputs were modified")
    save(dest/"execution_facts.json",facts)
    save(dest/"replay_audit.json",dict(status="PASS_FROZEN_HISTORY_AND_CONFIGURATION",bars=len(daily),
        previous_bars=before["daily_rows"],end=tail.datetime.iloc[-1],new_days=tail.datetime.tolist(),
        appended_objects=len(additions),appended_contexts=len(contexts),parallel=parallel))


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("market",choices=["nq","nikkei"]); p.add_argument("work",type=Path)
    args = p.parse_args(); run(args.market,args.work.resolve())
