from copy import deepcopy
from pathlib import Path
import shutil
import subprocess

import pytest

from cloud.version_labels import HISTORY_PREFIX, current_labels, normalize_current_labels, validate_current_labels


def test_normalization_changes_labels_not_trades_or_historical_facts():
    sample=dict(title="NQ=F · v0.11 追体験",replay=dict(strategyVersion="v0.13"),queries=dict(
        executions=dict(rows=[dict(price=100,version="v0.11",source_file="v011_source.json")],source=dict(
            label="v0.11 約定履歴",metricDefinitions=[dict(label="v0.11 約定モデル",definition="約定1行")],
            evidenceFlow=[dict(title="クラウド更新",detail="固定v0.11設定で当時再計算。")]))))
    before=deepcopy(sample)
    assert len(normalize_current_labels(sample))==3
    assert sample["queries"]["executions"]["rows"]==before["queries"]["executions"]["rows"]
    event=sample["queries"]["executions"]["source"]["evidenceFlow"][0]
    assert event["detail"]==before["queries"]["executions"]["source"]["evidenceFlow"][0]["detail"]
    assert event["title"].startswith(HISTORY_PREFIX)
    assert normalize_current_labels(sample)==[]
    assert all("v0.11" not in label for _,label in current_labels(sample))


def test_old_strategy_must_not_be_relabelled():
    with pytest.raises(ValueError,match="different strategy"):
        normalize_current_labels(dict(replay=dict(strategyVersion="v0.11")))


@pytest.mark.parametrize("market",["nq","es"])
def test_active_title_and_all_source_measurement_labels(market):
    from replay_assets import load_manifest,load_replay
    from scripts.import_replays import read_snapshot
    snapshot=read_snapshot(load_replay(market,load_manifest()["markets"][market]))
    validate_current_labels(snapshot)
    assert "v0.13" in snapshot["title"]


def test_packager_rejects_stale_source_definition(tmp_path):
    from cloud.refresh import hydrate_html
    from replay_assets import load_manifest,load_replay
    from scripts.import_replays import package,read_snapshot
    html=load_replay("nq",load_manifest()["markets"]["nq"])
    snapshot=read_snapshot(html)
    snapshot["queries"]["executions"]["source"]["metricDefinitions"][0]["label"]="v0.11 約定履歴"
    path=tmp_path/"bad-label.html"
    path.write_text(hydrate_html(html,snapshot),encoding="utf-8")
    with pytest.raises(ValueError,match="metricDefinitions"):
        package(path,"nq",tmp_path/"out")


@pytest.mark.parametrize("market",["nq","es"])
def test_no_stale_version_in_active_compiled_ui_or_page_title(market):
    import base64
    import re
    from cloud.version_labels import OLD_LABEL
    from replay_assets import load_manifest,load_replay
    from scripts.import_replays import SCRIPT
    html=load_replay(market,load_manifest()["markets"][market])
    assert not OLD_LABEL.search(SCRIPT.sub("",html))
    for match in SCRIPT.finditer(html):
        if 'type="application/json"' in match.group(1):
            continue  # Original audit provenance may legitimately name v0.11.
        encoded=re.search(r'base64,([^\"]+)',match.group(1))
        source=base64.b64decode(encoded.group(1)).decode() if encoded else match.group(2)
        assert not OLD_LABEL.search(source)


def test_packaged_title_guard_targets_previous_release():
    # The guard compiled into the packaged bundles must name the current
    # canonical version and the immediately previous release; the v0.13
    # release initially shipped with the previous-version literal still at
    # '11', so v0.12 saved titles were never corrected.
    import base64, re
    from replay_assets import load_manifest, load_replay
    manifest = load_manifest()
    for market in ["nq", "es"]:
        html = load_replay(market, manifest["markets"][market])
        bundles = []
        for match in re.finditer(r'src="data:text/javascript[^,]*;base64,([^"]+)"', html):
            try:
                bundles.append(base64.b64decode(match.group(1)).decode("utf-8"))
            except Exception:
                pass
        guard = [b for b in bundles if "currentReplayTitle" in b]
        assert len(guard) == 1
        assert "prior?.[2] === '12'" in guard[0]
        assert "v0\\.13 追体験" in guard[0]


def test_saved_obsolete_title_guard_uses_public_api_and_preserves_preferences():
    root=Path(__file__).resolve().parents[1]
    result=subprocess.run([shutil.which("node"),str(root/"scripts/verify_saved_title.mjs")],
        cwd=root,capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stdout+result.stderr
    assert '"status":"PASS_SAVED_TITLE_GUARD"' in result.stdout
    for market in ["nq","es"]:
        source=(root/"ui"/market/"DashboardContent.jsx").read_text(encoding="utf-8")
        assert "currentReplayTitle(snapshot,appTitle)" in source
        assert "setAppTitle(title)" in source
    helper=(root/"ui/shared/current-model-title.mjs").read_text(encoding="utf-8")
    assert "localStorage" not in helper and "removeItem" not in helper
