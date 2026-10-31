import gzip
import json
from pathlib import Path
import shutil
import subprocess

from cloud.common import CLOUD,digest,load,materialize_seed
from replay_assets import ASSETS,MARKETS,load_manifest,load_replay
from scripts.import_replays import canonical,read_snapshot


def test_only_nq_es_active_and_nkd_preserved():
    receipt=load(ASSETS/"es_addition_20261019.json")
    assert MARKETS==("nq","es")
    assert set(load_manifest()["markets"])=={"nq","es"}
    archived=ASSETS/"archive/nkd-v012/nikkei.html.gz"
    assert digest(archived.read_bytes())==receipt["archived_nkd_gzip_sha256"]
    assert archived.read_bytes()==(ASSETS/"nikkei.html.gz").read_bytes()
    # NQ remains an append-only history, even after future scheduled refreshes.
    nq=read_snapshot(load_replay("nq",load_manifest()["markets"]["nq"]))
    assert nq["replay"]["strategyVersion"]=="v0.13"
    text=(CLOUD.parent/".github/workflows/refresh-data.yml").read_text(encoding="utf-8")
    assert "assets/es.html.gz" in text and "assets/nikkei.html.gz" not in text
    assert "cloud/state/es/" in text and "cloud/state/nikkei/" not in text


def test_es_frozen_source_is_symbol_only_and_native_settings(tmp_path):
    nq=CLOUD/"vendor/nq"
    for source in nq.rglob("*.py"):
        assert (CLOUD/"vendor/es"/source.relative_to(nq)).read_bytes()==source.read_bytes().replace(b"NQ=F",b"ES=F")
    meta=materialize_seed("es",tmp_path/"es")
    assert meta["strategy_version"]=="v0.13" and meta["legacy_rules"]["buy_reference_radius"]==1
    assert meta["execution_config"]["point_value_usd"]==50
    assert meta["execution_config"]["commission_per_contract_side_usd"]==2.5
    assert meta["execution_config"]["tick_size"]==.25


def test_es_addition_frozen_prefix_remains_unchanged():
    receipt=load(ASSETS/"es_addition_20261019.json")
    assert receipt["research_prefix"]==dict(status="PASS",through="2026-10-14",closed_trades=44,
        daily_mtm_and_exposure_rows=6573,expected_trades_sha256="99b43cbae15c46317390cc513ee698a0143c145b94a13b2c90db8faeba11be4b")
    # The addition receipt pins the v0.12 generation; since v0.13 the live
    # snapshot is a different strategy, so check the archived v0.12 copy.
    import gzip
    snapshot=read_snapshot(gzip.decompress((ASSETS/"archive/v012/es.html.gz").read_bytes()).decode("utf-8"))
    for name,pin in receipt["historical_prefix"].items():
        rows=[r for r in snapshot["queries"][name]["rows"] if r["index"]<receipt["bars"]]
        assert len(rows)==pin["rows"] and digest(canonical(rows))==pin["rows_sha256"]


def test_es_asof_summary_and_distinct_routes():
    result=subprocess.run([shutil.which("node"),"scripts/verify_es_summary.mjs"],cwd=CLOUD.parent,
        capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stdout+result.stderr
    assert '"status":"PASS_ES_ASOF_SUMMARY"' in result.stdout


def test_es_view_explains_fvg_conditionally_and_uses_own_contract_units():
    source=(CLOUD.parent/"ui/es/DashboardContent.jsx").read_text(encoding="utf-8")
    assert 'id="es-asof-results"' in source and "50USD/pt" in source
    assert "choice.evidence?.retest_zone" in source and "adopted_fvg_retest:view.retestZone" in source
    assert "FVGはこの原本の直接BUY/SELL条件ではありません" not in source
    assert "view.retestSources" in source
