"""Explicit one-time v0.12 -> v0.13 migration. No downloads or live orders.

v0.13 adds exactly one strategy change: the NFCI persistence regime gate in
cloud/strategy.py (a BUY offer is rejected when NFCI has been >= 0 for at
least four consecutive weeks, read with a 7-day publication lag from the
stored series in cloud/regime/). Prices, zone inputs and every strategy
parameter are frozen; the full history is recomputed under the gate.

Run prepare, replay for each market, then publish only after both exports
and view checks pass. Git push is deliberately a separate operator action,
never performed by this program.
"""
from copy import deepcopy
import argparse
import base64
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cloud.common import CLOUD, FACT_FIELDS, digest, load, materialize_seed, now_iso, save
from cloud.export_adapter import export
from cloud.refresh import hydrate_html, stage_state
from cloud.strategy import BASE_STAGE, REGIME_GATE, STRATEGY_VERSION, install_pivot_supply, install_regime_gate, stamp_metadata
from replay_assets import load_manifest, load_replay
from scripts.import_replays import SCRIPT, package, read_snapshot

MIGRATION_TEXT = ("v0.12 -> v0.13; NFCI persistence regime gate added "
                  "(BUY blocked while NFCI >= 0 for >= 4 consecutive weeks, 7-day publication lag); "
                  "prices and zone inputs frozen; full history recomputed")
MIGRATION_DETAIL = ("価格・原本OB入力を固定し、NFCI持続性レジームゲート"
                    "（NFCI≧0が4週以上継続した局面ではBUYしない、公表ラグ7日）を追加して全履歴を再計算。"
                    "ピボット半径1は不変。")


def prepare(root):
    root.mkdir(parents=True, exist_ok=False)
    manifest = load_manifest()
    assert manifest.get("strategy_version") == "v0.12", "Migration already applied or unexpected base version"
    try:
        baseline_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        baseline_commit = None
    save(root/"original_manifest.json", manifest)
    save(root/"release.json", dict(strategy_version=STRATEGY_VERSION, baseline_commit=baseline_commit,
        started_at_utc=now_iso(), status="PREPARED",
        parameter_change={"regime_gate": REGIME_GATE["id"]}, market_data_refresh=False, markets={}))
    for market, item in manifest["markets"].items():
        work = root/market
        before = read_snapshot(load_replay(market, item))
        save(work/"original_snapshot.json", before)
        materialize_seed(market, work/"seed")
        save(work/"fetch_audit.json", dict(captured_at_utc=now_iso(),
            provider="Unchanged frozen repository inputs; no API request", excluded_in_progress=[]))
        result = work/"result"; result.mkdir()
        meta = load(work/"seed/metadata.json")
        assert meta.get("strategy_version") == "v0.12", "Seed is not the v0.12 generation"
        assert meta["legacy_rules"]["buy_reference_radius"] == 1
        for record in meta["input_hashes"].values():
            shutil.copyfile(work/"seed"/record["path"], result/record["path"])
        # No config override: v0.13 changes no parameter, only the gate.
        verification = deepcopy(before)
        for query in verification["queries"].values():
            query["rows"] = []
        save(work/"before.json", verification)
    print("PREPARED", root, flush=True)


def replay(market, root):
    work = root/market; result = work/"result"
    sys.path.insert(0, str(CLOUD/"vendor"/market))
    from nq_ict_campaign_examples_20260921.run_campaign import main as replay_main
    from nq_ict_campaign_examples_20260921.campaign import CampaignLedger
    install_pivot_supply()
    install_regime_gate()
    native = CampaignLedger.feed
    def progress(self, *args, **kwargs):
        value = native(self, *args, **kwargs)
        if len(self.state_rows) % 500 == 0:
            print(market, "bars", len(self.state_rows), flush=True)
        return value
    CampaignLedger.feed = progress
    before = load(work/"seed/metadata.json")
    sys.argv = [__file__, "--config", str(result/"config.json"), "--base-config", str(result/"base_config.json"),
        "--prices", str(result/"prices.csv"), "--objects", str(result/"objects.csv"), "--contexts", str(result/"contexts.csv"),
        "--calendar-json", str(result/"calendar.json"), "--as-of", str(before["end"])[:10],
        "--output", str(result/"campaign_replay"), "--revision-stage", BASE_STAGE, "--allow-unvalidated-replay"]
    replay_main()
    after = stamp_metadata(load(result/"campaign_replay/metadata.json"))
    assert after["legacy_rules"] == before["legacy_rules"]
    for field in ["campaign_config", "execution_config", "stage_overrides"]:
        assert before[field] == after[field], field
    for name, record in before["input_hashes"].items():
        assert after["input_hashes"][name]["sha256"] == record["sha256"], name
    save(result/"campaign_replay/metadata.json", after)
    events = [json.loads(line) for line in (result/"campaign_replay/execution_events.jsonl").read_text(encoding="utf-8").splitlines()]
    save(result/"execution_facts.json", [[r.get(k) for k in FACT_FIELDS] for r in events])
    save(result/"replay_audit.json", dict(status="PASS_REGIME_GATE_MIGRATION", strategy_version=STRATEGY_VERSION,
        daily_rows=after["daily_rows"], price_and_original_zone_inputs_unchanged=True,
        config_unchanged=True, sole_strategy_change="nfci_persistence_regime_gate"))
    export_only(market, root)


def export_only(market, root):
    """Recheck presentation from completed native logs; never rerun the engine."""
    work = root/market
    assert load(work/"result/replay_audit.json")["status"] == "PASS_REGIME_GATE_MIGRATION"
    export(market, work, load(work/"original_snapshot.json"), migration=MIGRATION_TEXT,
           migration_detail=MIGRATION_DETAIL)
    subprocess.run(["node", str(CLOUD/"verify_view.mjs"), market, str(work)], cwd=ROOT, check=True)
    print("READY", market, flush=True)


def presentation_labels(html):
    """Mechanical display-label migration, never alter bundled runtime logic."""
    def labels(source):
        source = source.replace("v0.12の保存結果です。αの正式採用時に過去売買を再計算し、以後の日次更新ではv0.12内の履歴を維持します。",
            "v0.13の保存結果です。NFCI持続性ゲートの採用時に過去売買を再計算し、以後の日次更新ではv0.13内の履歴を維持します。")
        source = source.replace("v0.12", "v0.13").replace("revision 0.12", "revision 0.13").replace("_v012_", "_v013_")
        # Escaped forms inside bundled regular expressions (e.g. v0\.12).
        source = source.replace("v0\\.12", "v0\\.13").replace("v0\\\\.12", "v0\\\\.13")
        # The saved-title guard compares the saved title's version against the
        # previous release; move that literal forward too, or titles saved
        # under v0.12 are never auto-corrected to the v0.13 canonical title.
        source = source.replace("prior?.[2] === '11'", "prior?.[2] === '12'")
        return source
    def upgrade(match):
        import re
        if 'type="application/json"' in match.group(1):
            return match.group(0)
        encoded = re.search(r'(src="data:text/javascript[^,]*;base64,)([^"]+)(")', match.group(1))
        if not encoded:
            return "<script"+match.group(1)+">"+labels(match.group(2))+"</script>"
        source = base64.b64decode(encoded.group(2)).decode("utf-8")
        replacement = base64.b64encode(labels(source).encode()).decode()
        return match.group(0).replace(encoded.group(2), replacement, 1)
    import re
    return re.sub(r"(<title>)(.*?)(</title>)", lambda m: m.group(1)+labels(m.group(2))+m.group(3), SCRIPT.sub(upgrade, html), flags=re.S)


def publish(root):
    release = load(root/"release.json")
    assert release["status"] == "PREPARED"
    original = load(root/"original_manifest.json")
    assert original == load_manifest(), "Another publisher changed the generation; do not overwrite"
    staged = root/"staged"; assets = staged/"assets"; assets.mkdir(parents=True, exist_ok=False)
    archive = assets/"archive/v012"; archive.mkdir(parents=True)
    save(archive/"manifest.json", original)
    markets = list(original["markets"])
    for market in markets:
        shutil.copyfile(ROOT/"assets"/(market+".html.gz"), archive/(market+".html.gz"))
    manifest = deepcopy(original)
    for market in markets:
        work = root/market
        assert load(work/"result/view_audit.json")["status"] == "PASS_DATA_VIEW_AND_LEDGER"
        snapshot = load(work/"result/snapshot.json")
        html = presentation_labels(hydrate_html(load_replay(market, original["markets"][market]), snapshot))
        path = work/"v013.html"; path.write_text(html, encoding="utf-8")
        item = package(path, market, assets)
        item["strategy_version"] = STRATEGY_VERSION
        item["data_quality"] = deepcopy(original["markets"][market]["data_quality"])
        manifest["markets"][market] = item
        stage_state(market, work, staged)
        # A strategy release does not silently upgrade data-quality status.
        shutil.copyfile(work/"seed/quality.json", staged/"state"/market/"quality.json")
        release["markets"][market] = dict(bars=item["bars"], end=item["end"], historical_prefix=item["queries"],
            replay=load(work/"result/replay_audit.json"), view=load(work/"result/view_audit.json"))
    release.update(status="PASS_V013_MIGRATION", finished_at_utc=now_iso(), published_to_github=False,
        baseline_archive="assets/archive/v012", market_data_refresh=False)
    manifest.update(strategy_version=STRATEGY_VERSION, packaged_at=now_iso(), strategy_release="v013_release.json")
    save(assets/"v013_release.json", release); save(assets/"manifest.json", manifest)
    for path in sorted(staged.rglob("*")):
        if path.is_file():
            rel = path.relative_to(staged)
            target = ROOT/("cloud" if rel.parts[0] == "state" else "")/rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    print("STAGED_V013_BOTH_MARKETS_NOT_YET_PUSHED", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["prepare", "replay", "export", "publish"])
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--market", choices=["nq", "es"])
    a = p.parse_args(); root = a.work.resolve()
    if a.mode == "prepare": prepare(root)
    elif a.mode == "replay": replay(a.market, root)
    elif a.mode == "export": export_only(a.market, root)
    else: publish(root)
