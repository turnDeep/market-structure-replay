"""Load only the two bundled, hash-checked, trusted replay documents."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
MARKETS = ("nq", "es")
MAX_HTML_BYTES = 40 * 1024 * 1024


class AssetError(ValueError):
    """An expected snapshot is absent or failed its integrity checks."""


def load_manifest() -> dict:
    try:
        manifest = json.loads((ASSETS / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise AssetError("保存済みデータが未配置、または管理ファイルが不正です。") from exc
    if manifest.get("schema_version") != 1 or set(manifest.get("markets", {})) != set(MARKETS):
        raise AssetError("対応していないデータ管理形式です。")
    for market in MARKETS:
        item = manifest["markets"][market]
        if item.get("asset") != f"{market}.html.gz":
            raise AssetError("許可されていないデータファイルです。")
        if item.get("strategy_version", "v0.11") != manifest.get("strategy_version", "v0.11"):
            raise AssetError("売買モデルのバージョンが一致しません。")
    return manifest


def load_replay(market: str, item: dict) -> str:
    if market not in MARKETS or item.get("asset") != f"{market}.html.gz":
        raise AssetError("許可されていない銘柄です。")
    try:
        packed = (ASSETS / item["asset"]).read_bytes()
        if hashlib.sha256(packed).hexdigest() != item["gzip_sha256"]:
            raise AssetError("圧縮データの整合性確認に失敗しました。")
        with gzip.GzipFile(fileobj=io.BytesIO(packed)) as source:
            raw = source.read(MAX_HTML_BYTES + 1)
        if len(raw) > MAX_HTML_BYTES or len(raw) != item["html_bytes"]:
            raise AssetError("HTMLデータのサイズが管理情報と一致しません。")
        if hashlib.sha256(raw).hexdigest() != item["html_sha256"]:
            raise AssetError("HTMLデータの整合性確認に失敗しました。")
        return raw.decode("utf-8")
    except (OSError, KeyError, UnicodeError, EOFError) as exc:
        raise AssetError("保存済みデータを読み込めません。") from exc
