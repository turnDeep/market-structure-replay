from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]


def test_nq_summary_all_dates_and_future_boundaries():
    node=shutil.which("node")
    assert node,"Node.js is required to test the same summary function used in the browser"
    result=subprocess.run([node,str(ROOT/"scripts/verify_nq_summary.mjs")],cwd=ROOT,capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stdout+result.stderr
    assert '"status":"PASS_NQ_ASOF_SUMMARY"' in result.stdout


def test_nq_source_has_visible_summary_and_bounded_evidence():
    source=(ROOT/"ui/nq/DashboardContent.jsx").read_text(encoding="utf-8")
    assert 'id="nq-asof-results"' in source
    assert 'title="選択日までのモデル集計"' in source
    assert 'summarizeModel(data.fills,view.latest,view.state' in source
    assert 'sourceRowsByQuery={{executions:view.knownFills,market_prices:[view.latest],daily_states:[view.state]}}' in source
    assert 'model_summary:modelSummary' in source
    assert 'displayRows={[modelSummary]}' in source


def test_ui_repackage_leaves_nikkei_asset_and_policy_untouched(tmp_path,monkeypatch):
    import json
    import scripts.repackage_views as repack
    assets=tmp_path/"assets";assets.mkdir()
    (assets/"nq.html.gz").write_bytes(b"previous nq")
    (assets/"nikkei.html.gz").write_bytes(b"untouched nikkei")
    snapshot=dict(id="fixture",queries=dict(executions=dict(rows=[])))
    html=tmp_path/"nq.html";html.write_text(json.dumps(snapshot),encoding="utf-8")
    item=dict(data_quality=dict(provisional_from_date="2026-10-15"))
    manifest=dict(automatic_refresh=True,refresh_schedule=dict(local_time="09:00"),markets=dict(nq=item,nikkei=dict(asset="nikkei.html.gz")))
    def package(source,market,staged):
        (staged/(market+".html.gz")).write_bytes(b"new nq")
        return dict(asset="nq.html.gz")
    monkeypatch.setattr(repack,"ROOT",tmp_path)
    monkeypatch.setattr(repack,"load_manifest",lambda:manifest)
    monkeypatch.setattr(repack,"load_replay",lambda *args:json.dumps(snapshot))
    monkeypatch.setattr(repack,"read_snapshot",json.loads)
    monkeypatch.setattr(repack,"package",package)
    monkeypatch.setattr(sys,"argv",["repackage_views.py","--nq",str(html)])
    repack.main()
    assert (assets/"nikkei.html.gz").read_bytes()==b"untouched nikkei"
    result=json.loads((assets/"manifest.json").read_text(encoding="utf-8"))
    assert result["automatic_refresh"] is True
    assert result["refresh_schedule"]["local_time"]=="09:00"
    assert result["markets"]["nq"]["data_quality"]==item["data_quality"]
