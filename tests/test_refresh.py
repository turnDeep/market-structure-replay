"""Keep the pre-refresh reviewed row prefix pinned across future releases."""
import hashlib
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
    current = read_snapshot(load_replay(market, load_manifest()["markets"][market]))
    for query, pin in audit["historical_prefix"].items():
        rows = [r for r in current["queries"][query]["rows"] if r["index"] < audit["previous_bars"]]
        assert len(rows) == pin["rows"]
        assert hashlib.sha256(canonical(rows)).hexdigest() == pin["rows_sha256"]
    assert audit["model_qa"]["status"] == "PASS_REFRESH_DATA_MODEL_AND_PRIOR_ROWS"
    assert audit["replay_qa"]["status"] == "PASS_FROZEN_HISTORY_AND_CONFIGURATION"
