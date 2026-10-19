from datetime import datetime, timezone
import gzip
import json
from pathlib import Path

import pytest

from cloud.common import CLOUD, completed_cutoff, materialize_seed, validate_rows, verify_package
from cloud.refresh import hydrate_html, run
from scripts.import_replays import read_snapshot


@pytest.mark.parametrize("clock,expected", [
    ("2026-10-16T00:00:00+00:00", "2026-10-15"),
    ("2026-07-16T21:59:59+00:00", "2026-07-15"),
    ("2026-07-16T22:00:00+00:00", "2026-07-16"),
    ("2026-01-16T22:59:59+00:00", "2026-01-15"),
    ("2026-01-16T23:00:00+00:00", "2026-01-16"),
])
def test_cme_cutoff_dst(clock, expected):
    assert completed_cutoff(datetime.fromisoformat(clock)) == expected


def test_cutoff_rejects_ambiguous_clock():
    with pytest.raises(ValueError):
        completed_cutoff(datetime(2026,10,16))


def test_excludes_yahoo_current_date_after_cme_break():
    now = datetime.fromisoformat("2026-10-16T01:48:00+00:00")
    assert completed_cutoff(now) == "2026-10-15"
    assert completed_cutoff(now,"America/New_York") == "2026-10-14"
    assert completed_cutoff(datetime.fromisoformat("2026-10-16T04:01:00+00:00"),"America/New_York") == "2026-10-15"


def test_code_and_seed_integrity(tmp_path):
    manifest = verify_package()
    assert manifest["strategy_status"] == "NOT_VALIDATED"
    for market in ["nq","nikkei"]:
        meta = materialize_seed(market,tmp_path/market)
        assert meta["campaign_config"]["support_selection"] == "structural_or_leg"
        assert meta["campaign_config"]["enable_recovery_buy"] is True
        assert not meta["campaign_config"]["enable_continuation_buy"]
        assert meta["campaign_config"]["reaction_filter"] == "observe_only"
        assert meta["legacy_rules"]["buy_reference_radius"] == 1
        assert meta["legacy_rules"]["emergency_buffer_atr"] == .10
        assert meta["legacy_rules"]["major_decline_bars"] == 15
        assert meta["strategy_version"] == "v0.12"


def test_nkd_adapter_is_symbol_only():
    nq = CLOUD/"vendor/nq"; nkd = CLOUD/"vendor/nikkei"
    for source in nq.rglob("*.py"):
        assert (nkd/source.relative_to(nq)).read_bytes() == source.read_bytes().replace(b"NQ=F",b"NKD=F")


def test_no_malformed_or_missing_volume():
    good = dict(datetime="2026-10-15",open=10,high=12,low=9,close=11,volume=0)
    validate_rows([good])  # Measured zero is not missing.
    for field,value in [("volume",float("nan")),("volume",-1),("low",11),("high",8)]:
        with pytest.raises(ValueError):
            validate_rows([dict(good,**{field:value})])
    with pytest.raises(ValueError):
        validate_rows([good,good])


def test_hydration_only_replaces_reviewed_snapshot():
    html = '<html><script>unchangedRuntime()</script><script type="application/json" id="data-app-reviewed-snapshot">{}</script><script type="application/json"> </script></html>'
    snapshot = {"text":"</script><script>alert(1)</script>"}
    result = hydrate_html(html,snapshot)
    assert read_snapshot(result)==snapshot
    assert '<script>unchangedRuntime()</script>' in result
    assert 'alert(1)</script>' not in result
    with pytest.raises(ValueError):
        hydrate_html('<html></html>',snapshot)


def test_no_new_bars_does_not_replay_or_modify_assets(tmp_path,monkeypatch):
    import cloud.refresh as refresh
    from replay_assets import ASSETS
    original = {p:p.read_bytes() for p in ASSETS.glob("*") if p.is_file()}
    def no_new(market,work,meta,now):
        refresh.save(work/"fetch_audit.json",dict(captured_at_utc=now.isoformat(),source_end=str(meta["end"])[:10]))
        return False
    monkeypatch.setattr(refresh,"fetch",no_new)
    monkeypatch.setattr(refresh.subprocess,"run",lambda *a,**kw: pytest.fail("No-op must not run replay"))
    report = run("refresh",tmp_path/"reports")
    assert report["status"] == "NO_NEW_COMPLETED_DATA"
    assert not report["published"]
    assert all(p.read_bytes()==b for p,b in original.items())


def test_second_market_fetch_failure_never_publishes_first(tmp_path,monkeypatch):
    import cloud.refresh as refresh
    from replay_assets import ASSETS
    original = {p:p.read_bytes() for p in ASSETS.glob("*") if p.is_file()}
    def failed(market,work,meta,now):
        if market == "nikkei":
            raise ValueError("Provider unavailable")
        refresh.save(work/"fetch_audit.json",dict(captured_at_utc=now.isoformat()))
        return True
    monkeypatch.setattr(refresh,"fetch",failed)
    monkeypatch.setattr(refresh.subprocess,"run",lambda *a,**kw: pytest.fail("Fetch both before any replay"))
    with pytest.raises(ValueError,match="Provider unavailable"):
        run("refresh",tmp_path/"reports")
    assert all(p.read_bytes()==b for p,b in original.items())


def test_schedule_and_permissions_are_scoped():
    text = (CLOUD.parent/".github/workflows/refresh-data.yml").read_text(encoding="utf-8")
    assert "cron: '0 0 * * *'" in text
    assert "workflow_dispatch:" in text
    assert "contents: write" in text
    assert "cancel-in-progress: false" in text
    assert "git push origin HEAD:main" in text
    assert "--force" not in text
    assert "pull_request_target" not in text


def test_pending_sell_has_close_time_evidence_without_a_fictitious_fill():
    from cloud.export_adapter import add_unfilled_sells
    # Small deterministic SOFTWARE fixture, not market observations or a backtest.
    bars = [dict(close=x) for x in [20,22,23,22,19,18]]
    trigger = dict(h1=dict(origin_index=1,price=22,available_index=2,atr_at_origin=1),
        neckline=dict(origin_index=2,price=20,available_index=3),reference=dict(price=20),
        peak_index=2,peak=23,first_breaks=dict(early=4),levels_known_index=3)
    saved = dict(order_id="sell-pending",position_id="long-existing",index=5,
        representative_selection=dict(local_trigger=trigger))
    def visual(family,i,mechanism,focus,marks,levels,steps):
        return dict(family=family,known_index=i,decision_index=i,mechanism=mechanism,marks=marks,levels=levels,steps=steps)
    ns = dict(decisions=[],bars=bars,day=lambda i:f"day-{i}",
        mark=lambda code,label,i,p,k:dict(index=i,price=p,known_index=k),
        level=lambda key,label,p,i:dict(id=key,price=p,from_index=i),
        step=lambda code,title,detail:dict(code=code,title=title,detail=detail),visual=visual,
        jl=lambda name:[saved],meta=dict(legacy_rules=dict(structure_buffer_atr=.05,sell_departure_atr=1)),
        contexts={"long-existing":dict(stop=15)})
    rows = add_unfilled_sells(ns)
    assert len(rows)==1 and rows[0]["side"]=="SELL" and rows[0]["index"]==5
    ev = json.loads(rows[0]["visual"])
    assert "fill" not in ev and "price" not in rows[0]
    assert ev["known_index"]==5 and all(m["known_index"]<=5 for m in ev["marks"])
    ns["decisions"]=rows
    assert add_unfilled_sells(ns)==[]


def test_freshness_counts_are_derived_after_provisional_replacement():
    from cloud.export_adapter import refresh_metadata
    replay = dict(refresh=dict(frozenThrough="2026-08-27",appendedBars=999,extendedEnd="stale"))
    bars = [dict(date=d) for d in ["2026-08-27","2026-08-28","2026-10-15"]]
    result = refresh_metadata(replay,bars,dict(captured_at_utc="now",excluded_in_progress=["2026-10-16"]),"2026-10-14",2)
    assert result["appendedBars"]==2 and result["latestAppendedBars"]==1
    assert result["extendedEnd"]=="2026-10-15"
    assert result["partialDatesExcluded"]==["2026-10-16"]
    assert result["automaticScheduleEnabled"] is True


def test_staged_state_can_be_loaded_without_original_machine_paths(tmp_path):
    from cloud.common import digest,load,save
    from cloud.refresh import stage_state
    work = tmp_path/"work";seed=work/"seed";result=work/"result"
    seed.mkdir(parents=True);(result/"campaign_replay").mkdir(parents=True)
    inputs={}
    for name in ["prices","objects","contexts","calendar","config","base_config"]:
        extension=".csv" if name in ["prices","objects","contexts"] else ".json"
        path=result/(name+extension);path.write_bytes(("column\nrow\n" if extension==".csv" else "{}\n").encode())
        inputs[name]=dict(path=str(path),sha256=digest(path.read_bytes()))
    meta=dict(daily_rows=2,end="2026-10-15",input_hashes=inputs)
    save(seed/"metadata.json",meta);save(result/"campaign_replay/metadata.json",meta)
    save(result/"execution_facts.json",[])
    (seed/"excluded_invalid_ohlc_rows.csv").write_text("datetime\n",encoding="utf-8")
    staged=tmp_path/"staged"
    stage_state("nq",work,staged)
    reopened=materialize_seed("nq",tmp_path/"independent",state_root=staged/"state")
    assert reopened["daily_rows"]==2
    assert all(Path(r["path"]).name==r["path"] for r in reopened["input_hashes"].values())
    assert load(tmp_path/"independent/quality.json")["provisional_from_date"] is None


def test_second_replay_failure_does_not_modify_production_generation(tmp_path,monkeypatch):
    import cloud.refresh as refresh
    from replay_assets import ASSETS
    original={p:p.read_bytes() for p in ASSETS.glob("*") if p.is_file()}
    def fetch(market,work,meta,now):
        refresh.save(work/"fetch_audit.json",dict(captured_at_utc=now.isoformat()))
        return True
    def subprocess_run(args,**kwargs):
        if "cloud.replay_worker" in args:
            market=args[-2];work=Path(args[-1]);(work/"result").mkdir()
            if market=="nikkei":
                raise RuntimeError("Synthetic second-market replay failure")
            for name in ["replay_audit","export_audit","view_audit"]:
                refresh.save(work/"result"/(name+".json"),dict(status="SOFTWARE_TEST_STUB"))
    monkeypatch.setattr(refresh,"fetch",fetch)
    monkeypatch.setattr(refresh.subprocess,"run",subprocess_run)
    monkeypatch.setattr(refresh,"export",lambda market,work,before:before)
    with pytest.raises(RuntimeError,match="second-market replay failure"):
        refresh.run("refresh",tmp_path/"reports")
    assert all(p.read_bytes()==raw for p,raw in original.items())
