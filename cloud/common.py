from __future__ import annotations

from datetime import datetime, timedelta, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
CLOUD = ROOT / "cloud"
FACT_FIELDS = ["index", "action", "price", "order_id", "position_id", "stop", "net_pnl_usd"]
MARKETS = {"nq": "NQ=F", "es": "ES=F", "nikkei": "NKD=F"}


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str, allow_nan=False)+"\n", encoding="utf-8")


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def completed_cutoff(now, source_timezone=None):
    if now.tzinfo is None:
        raise ValueError("A timezone-aware clock is required")
    local = now.astimezone(ZoneInfo("America/Chicago"))
    cutoff = local.date() if local.hour >= 17 else local.date()-timedelta(days=1)
    # Yahoo's futures daily label can still be changing after the CME break.
    # Never admit the provider's CURRENT local calendar date, even if its date
    # is already <= the most recent CME boundary. Use the stricter boundary.
    if source_timezone:
        provider_day = now.astimezone(ZoneInfo(source_timezone)).date()
        cutoff = min(cutoff,provider_day-timedelta(days=1))
    return cutoff.isoformat()


def validate_rows(rows):
    dates = [r["datetime"] for r in rows]
    if dates != sorted(set(dates)):
        raise ValueError("Duplicate or unordered provider dates")
    for r in rows:
        o, h, l, c, v = [float(r[k]) for k in ["open", "high", "low", "close", "volume"]]
        if not all(math.isfinite(x) for x in [o, h, l, c, v]) or not (0 < l <= min(o,c) <= max(o,c) <= h and v >= 0):
            raise ValueError(f"Invalid completed OHLCV: {r['datetime']}")


def verify_package():
    manifest = load(CLOUD/"package_manifest.json")
    for rel, pin in manifest["files"].items():
        path = (CLOUD/rel).resolve()
        if not path.is_relative_to(CLOUD.resolve()) or digest(path.read_bytes()) != pin["sha256"]:
            raise ValueError(f"Frozen code/view hash mismatch: {rel}")
    return manifest


def materialize_seed(market, dest, *, state_root=None):
    source = (state_root or CLOUD/"state")/market
    meta = load(source/"metadata.json")
    dest.mkdir(parents=True, exist_ok=True)
    for name, record in meta["input_hashes"].items():
        filename = record["path"]
        if Path(filename).name != filename:
            raise ValueError("Nonportable input path")
        raw = gzip.decompress((source/(filename+".gz")).read_bytes())
        if digest(raw) != record["sha256"]:
            raise ValueError(f"Frozen input mismatch: {market}/{name}")
        (dest/filename).write_bytes(raw)
    save(dest/"metadata.json", meta)
    save(dest/"execution_facts.json", load(source/"execution_facts.json"))
    (dest/"excluded_invalid_ohlc_rows.csv").write_bytes((source/"excluded_invalid_ohlc_rows.csv").read_bytes())
    save(dest/"quality.json",load(source/"quality.json"))
    return meta


def now_iso():
    return datetime.now(timezone.utc).isoformat()
