from copy import deepcopy
import json
from pathlib import Path
import re
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from replay_assets import AssetError, load_manifest, load_replay
from scripts.import_replays import canonical, check_sensitive_text, digest, read_snapshot


@pytest.mark.parametrize("market", ["nq", "nikkei"])
def test_bundled_snapshot(market):
    manifest = load_manifest()
    item = manifest["markets"][market]
    html = load_replay(market, item)
    snapshot = read_snapshot(html)
    assert snapshot["buildStatus"] == "complete"
    assert snapshot["replay"]["strategyStatus"] == "NOT_VALIDATED"
    assert snapshot["replay"]["end"] == item["end"]
    assert snapshot["replay"]["start"] == item["start"]
    assert isinstance(manifest["automatic_refresh"], bool)
    assert item["reviewed_rows_unchanged"]
    assert len(snapshot["queries"]["market_prices"]["rows"]) == item["bars"]
    check_sensitive_text(html)
    assert not re.search(r'<meta[^>]+name="data-app-local-snapshot"', html)
    assert not re.search(r'<meta[^>]+name="data-app-local-thread', html)
    for name, recorded in item["queries"].items():
        rows = snapshot["queries"][name]["rows"]
        assert len(rows) == recorded["rows"]
        assert digest(canonical(rows)) == recorded["rows_sha256"]
    assert '<script type="application/json" id="data-app-reviewed-snapshot">' in html


def test_reject_unapproved_market_and_path():
    item = load_manifest()["markets"]["nq"]
    with pytest.raises(AssetError):
        load_replay("../../secret", item)
    invalid = deepcopy(item)
    invalid["asset"] = "../../secret"
    with pytest.raises(AssetError):
        load_replay("nq", invalid)


def test_reject_tampered_digest():
    item = deepcopy(load_manifest()["markets"]["nq"])
    item["gzip_sha256"] = "0" * 64
    with pytest.raises(AssetError):
        load_replay("nq", item)


def test_market_price_dates_are_bounded_and_unique():
    for market, item in load_manifest()["markets"].items():
        rows = read_snapshot(load_replay(market, item))["queries"]["market_prices"]["rows"]
        dates = [row["date"] for row in rows]
        assert dates == sorted(set(dates))
        assert dates[0] == item["start"] and dates[-1] == item["end"]


def test_query_input_does_not_choose_file():
    source = (ROOT / "streamlit_app.py").read_text(encoding="utf-8")
    assert 'if requested not in markets:' in source
    assert 'st.iframe(html,' in source
    assert 'load_replay(market, item)' in source


def test_failed_second_import_keeps_previous_assets(tmp_path, monkeypatch):
    import scripts.import_replays as importer
    output = tmp_path / "assets"
    output.mkdir()
    for name in ["nq.html.gz", "nikkei.html.gz", "manifest.json"]:
        (output / name).write_bytes(b"previous-verified-generation")
    def staged_package(path, key, directory):
        if key == "nikkei":
            raise ValueError("Invalid second source")
        (directory / f"{key}.html.gz").write_bytes(b"unpublished-first-market")
        return {}
    monkeypatch.setattr(importer, "package", staged_package)
    monkeypatch.setattr(sys, "argv", ["import_replays.py", "--nq", "first.html", "--nikkei", "bad.html", "--output", str(output)])
    with pytest.raises(ValueError, match="Invalid second source"):
        importer.main()
    for name in ["nq.html.gz", "nikkei.html.gz", "manifest.json"]:
        assert (output / name).read_bytes() == b"previous-verified-generation"
