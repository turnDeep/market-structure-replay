"""One-time, explicit import of frozen local research inputs into the private repo.

This is packaging, not a strategy migration. Source bytes are preserved; only
portable metadata paths and the existing NKD symbol-only adapter are derived.
Never run against unreviewed files. No credential or research folder is copied.
"""
from pathlib import Path
import argparse
import ast
import gzip
import hashlib
import json
import shutil

REPO = Path(__file__).resolve().parents[1]


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str)+"\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--source-run", type=Path, required=True)
    args = p.parse_args()
    root = args.workspace.resolve()
    cloud = REPO / "cloud"
    packages = ["nq_ict_campaign_examples_20260921", "nq_ict_macro_reversal_examples_20260916", "nq_ict_priority_entry_examples_20260916"]
    files = set()
    for package in packages:
        files.update(f.relative_to(root) for f in (root/package).glob("*.py"))
    names = ["nq_ict_detection_20260911_v03", "nq_ict_evaluation_20260912_v08", "nq_ict_evaluation_20260912_v09",
             "run_nq_ict_early_buy_comparison_20260916", "run_nq_ict_macro_reversal_20260916",
             "nq_ict_retest_trade_state_machine_20260915", "nq_ict_retest_trade_state_machine_revision_20260915"]
    pending = [Path(n+".py") for n in names]
    while pending:
        rel = pending.pop()
        if rel in files:
            continue
        files.add(rel)
        tree = ast.parse((root/rel).read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            modules = [x.name for x in node.names] if isinstance(node, ast.Import) else [node.module] if isinstance(node, ast.ImportFrom) and node.level == 0 else []
            for module in modules:
                if module:
                    candidate = Path(module.split('.')[0]+".py")
                    if (root/candidate).is_file() and candidate not in files:
                        pending.append(candidate)
    manifest = {"schema": 1, "strategy_status": "NOT_VALIDATED", "files": {}, "markets": {}}
    for key in ["nq", "nikkei"]:
        for rel in sorted(files):
            original = (root/rel).read_bytes()
            raw = original if key == "nq" else original.replace(b"NQ=F", b"NKD=F")
            dest = cloud/"vendor"/key/rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(raw)
            manifest["files"][dest.relative_to(cloud).as_posix()] = dict(sha256=sha(raw), original_sha256=sha(original),
                transformation="none" if key == "nq" else "literal NQ=F to NKD=F only")
        src = args.source_run/key
        meta = json.loads((src/"campaign_replay/metadata.json").read_text(encoding="utf-8-sig"))
        dest = cloud/"state"/key
        dest.mkdir(parents=True, exist_ok=True)
        for kind, record in meta["input_hashes"].items():
            raw = Path(record["path"]).read_bytes()
            assert sha(raw) == record["sha256"], kind
            filename = kind+(".csv" if kind in ["prices", "objects", "contexts"] else ".json")
            packed = gzip.compress(raw, mtime=0)
            (dest/(filename+".gz")).write_bytes(packed)
            record["path"] = filename
        # Full replay outputs are disposable. Only a small immutable facts pin is
        # needed to compare execution against the already published generation.
        facts = []
        for line in (src/"campaign_replay/execution_events.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            facts.append([r.get(k) for k in ["index", "action", "price", "order_id", "position_id", "stop", "net_pnl_usd"]])
        save(dest/"execution_facts.json", facts)
        slim = {k: meta[k] for k in ["daily_rows", "start", "end", "campaign_config", "legacy_rules", "execution_config", "stage_overrides", "input_hashes", "revision_stage"]}
        save(dest/"metadata.json", slim)
        save(dest/"quality.json",dict(provisional_from_date="2026-10-15",verified_through="2026-10-14",
            reason="Initial latest saved Yahoo daily row was observed changing during its provider-local date; retain it visibly as provisional until a later completed-bar refresh replaces this bounded tail."))
        gaps = root/("nq_25y_actual_2000_2026_20260911/excluded_invalid_ohlc_rows.csv" if key == "nq" else "analysis_outputs/nikkei_v011_20261015/NKD_F/excluded_invalid_ohlc_rows.csv")
        shutil.copyfile(gaps, dest/"excluded_invalid_ohlc_rows.csv")
        app = root/("analysis_outputs/v011_browser_replay_20261010/app" if key == "nq" else "analysis_outputs/nikkei_v011_20261015/app")
        for name in ["replay-model.mjs", "decision-focus.mjs"]:
            target = cloud/"views"/key/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(app/"src/content/dashboard"/name, target)
        manifest["markets"][key] = dict(symbol="NQ=F" if key == "nq" else "NKD=F", end=str(meta["end"])[:10], bars=meta["daily_rows"],
            detector_sensitivities=["strict", "balanced", "exploratory"] if key == "nq" else ["balanced"])
    # The reviewed exporter is vendored verbatim. A separate explicit adapter
    # handles portability and per-market units; it never changes engine code.
    target = cloud/"export_template.py.txt"
    shutil.copyfile(root/"analysis_outputs/nikkei_v011_20261015/export_snapshot.py", target)
    manifest["files"][target.relative_to(cloud).as_posix()] = dict(sha256=sha(target.read_bytes()), transformation="none")
    for key in ["nq", "nikkei"]:
        for f in (cloud/"views"/key).glob("*.mjs"):
            manifest["files"][f.relative_to(cloud).as_posix()] = dict(sha256=sha(f.read_bytes()), transformation="none")
    save(cloud/"package_manifest.json", manifest)
    print(json.dumps({"engine_files_per_market": len(files), "markets": manifest["markets"]}))


if __name__ == "__main__":
    main()
