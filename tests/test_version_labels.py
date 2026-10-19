from copy import deepcopy

import pytest

from cloud.version_labels import HISTORY_PREFIX, current_labels, normalize_current_labels, validate_current_labels


def test_normalization_changes_labels_not_trades_or_historical_facts():
    sample=dict(title="NQ=F · v0.11 追体験",replay=dict(strategyVersion="v0.12"),queries=dict(
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
    assert "v0.12" in snapshot["title"]


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
