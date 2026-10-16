"""Package existing standalone HTML; do not recalculate or alter trading rows.

Run locally with explicit --nq and --nikkei paths. No source paths are retained.
Output is deterministic apart from the recorded packaging timestamp.
"""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = re.compile(r"<script([^>]*)>(.*?)</script>", re.S)
PRIVATE_PATH = re.compile(r"[A-Za-z]:[\\/]+Users[\\/]+[^\s\"<>]+", re.I)
SECRET = re.compile(r"(?:github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|sk-proj-[A-Za-z0-9_-]{20,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)")


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def read_snapshot(html: str) -> dict:
    chunks = [m.group(2) for m in SCRIPT.finditer(html) if 'type="application/json"' in m.group(1)]
    if not chunks:
        raise ValueError("Standalone reviewed snapshot missing")
    return json.loads("".join(chunks))


def scrub(value: object) -> object:
    if isinstance(value, dict):
        return {key: scrub(child) for key, child in value.items()}
    if isinstance(value, list):
        return [scrub(child) for child in value]
    if isinstance(value, str):
        return PRIVATE_PATH.sub(lambda m: re.split(r"[\\/]+", m.group(0))[-1], value)
    return value


def check_sensitive_text(html: str) -> None:
    strings = [html]
    for match in SCRIPT.finditer(html):
        encoded = re.search(r'src="data:text/javascript[^,]*;base64,([^\"]+)"', match.group(1))
        if encoded:
            strings.append(base64.b64decode(encoded.group(1)).decode("utf-8"))
    for text in strings:
        if PRIVATE_PATH.search(text) or SECRET.search(text):
            raise ValueError("Private machine path or recognizable credential found")


def package(path: Path, key: str, output: Path) -> dict:
    original = path.read_bytes()
    html = original.decode("utf-8")
    before = read_snapshot(html)
    after = scrub(before)
    if before.get("buildStatus") != "complete":
        raise ValueError("The source must be a completed, standalone export")
    if before.get("replay", {}).get("strategyStatus") != "NOT_VALIDATED":
        raise ValueError("Review the unexpected strategy status before importing")
    for query, value in before["queries"].items():
        if value.get("rows") != after["queries"][query].get("rows"):
            raise ValueError(f"Privacy cleanup would change reviewed rows: {query}")
    encoded = json.dumps(after, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    replaced = False

    def replace_snapshot(match: re.Match) -> str:
        nonlocal replaced
        if 'type="application/json"' not in match.group(1):
            return match.group(0)
        if replaced:
            return ""
        replaced = True
        return '<script type="application/json" id="data-app-reviewed-snapshot">' + encoded + '</script>'

    html = SCRIPT.sub(replace_snapshot, html)
    # This is a standalone snapshot. Never follow a previous machine's data URL.
    html = re.sub(r'<meta[^>]+name="data-app-local-snapshot"[^>]*>\s*', "", html)
    html = re.sub(r'<meta[^>]+name="(?:data-app-local-thread|data-app-local-thread-id|data-app-thread-id|data-app-source-thread-id)"[^>]*>\s*', "", html)
    html = re.sub(r'(<meta name="data-app-snapshot-sha256" content=")[^"]+("[^>]*>)', lambda m: m.group(1) + digest(encoded.encode()) + m.group(2), html)
    html = html.replace('<html lang="en">', '<html lang="ja">')
    # Deployment, chat handoff and editing belong to the desktop authoring app,
    # not this read-only Streamlit host. Keep chart/source/export controls.
    html = html.replace("</head>", '<style data-streamlit-replay-host>header[data-data-app-chrome="topbar"] button{display:none!important}</style>\n</head>', 1)
    check_sensitive_text(html)
    verified = read_snapshot(html)
    assert verified == after
    raw = html.encode("utf-8")
    packed = gzip.compress(raw, compresslevel=9, mtime=0)
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{key}.html.gz").write_bytes(packed)
    replay = before["replay"]
    return {
        "symbol": "NQ=F" if key == "nq" else "NKD=F",
        "asset": f"{key}.html.gz",
        "start": replay["start"], "end": replay["end"],
        "bars": len(before["queries"]["market_prices"]["rows"]),
        "snapshot_generated_at": before["generatedAt"],
        "strategy_status": replay["strategyStatus"],
        "html_bytes": len(raw), "html_sha256": digest(raw),
        "gzip_bytes": len(packed), "gzip_sha256": digest(packed),
        "source_html_sha256": digest(original),
        "reviewed_rows_unchanged": True,
        "queries": {name: {"rows": len(query.get("rows", [])), "rows_sha256": digest(canonical(query.get("rows", [])))} for name, query in before["queries"].items()},
        "privacy_cleanup": "Absolute user-directory paths replaced with basenames in provenance only; reviewed query rows unchanged.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nq", type=Path, required=True)
    parser.add_argument("--nikkei", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "assets")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    # A failed second import must not replace the first market's live asset.
    # Validate both in a sibling staging directory, then publish manifest last.
    with tempfile.TemporaryDirectory(prefix="replay-staging-", dir=args.output.parent) as temporary:
        staged = Path(temporary)
        markets = {key: package(path, key, staged) for key, path in [("nq", args.nq), ("nikkei", args.nikkei)]}
        manifest = {"schema_version": 1, "packaged_at": datetime.now(timezone.utc).isoformat(), "automatic_refresh": False, "markets": markets}
        (staged / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        for name in ["nq.html.gz", "nikkei.html.gz", "manifest.json"]:
            os.replace(staged / name, args.output / name)
    print(json.dumps({key: {"bars": item["bars"], "end": item["end"], "gzip_bytes": item["gzip_bytes"]} for key, item in markets.items()}))


if __name__ == "__main__":
    main()
