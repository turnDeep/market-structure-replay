"""Keep the pre-refresh reviewed row prefix pinned across future releases."""
import hashlib
import gzip
import json

import pytest

from replay_assets import ASSETS, load_manifest, load_replay
from scripts.import_replays import canonical, read_snapshot


@pytest.mark.parametrize("market", ["nq", "nikkei"])
def test_pinned_history_before_manual_refresh(market):
    receipt = json.loads((ASSETS / "refresh_20261016.json").read_text(encoding="utf-8"))
    assert receipt["strategy_parameters_changed"] is False
    assert receipt["strategy_validation_status"] == "NOT_VALIDATED"
    audit = receipt["markets"][market]
    # Strategy migration intentionally changes historical trades. The original
    # v0.11 audit must continue to check v0.11, not a different strategy.
    current = read_snapshot(gzip.decompress((ASSETS/"archive/v011"/(market+".html.gz")).read_bytes()).decode("utf-8"))
    for query, pin in audit["historical_prefix"].items():
        rows = [r for r in current["queries"][query]["rows"] if r["index"] < audit["previous_bars"]]
        assert len(rows) == pin["rows"]
        assert hashlib.sha256(canonical(rows)).hexdigest() == pin["rows_sha256"]
    assert audit["model_qa"]["status"] == "PASS_REFRESH_DATA_MODEL_AND_PRIOR_ROWS"
    assert audit["replay_qa"]["status"] == "PASS_FROZEN_HISTORY_AND_CONFIGURATION"


@pytest.mark.parametrize("market", ["nq", "nikkei"])
def test_v012_release_prefix_remains_frozen_during_daily_updates(market):
    release=json.loads((ASSETS/"v012_release.json").read_text(encoding="utf-8"))
    assert release["status"]=="PASS_V012_MIGRATION"
    pin=release["markets"][market]
    if market == "nikkei":
        current=read_snapshot(gzip.decompress((ASSETS/"archive/nkd-v012/nikkei.html.gz").read_bytes()).decode("utf-8"))
    else:
        current=read_snapshot(load_replay(market,load_manifest()["markets"][market]))
    for name,record in pin["historical_prefix"].items():
        rows=[r for r in current["queries"][name]["rows"] if r["index"]<pin["bars"]]
        assert len(rows)==record["rows"]
        assert hashlib.sha256(canonical(rows)).hexdigest()==record["rows_sha256"]
