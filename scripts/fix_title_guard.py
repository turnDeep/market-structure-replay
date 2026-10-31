#!/usr/bin/env python3
"""One-time presentation fix for the v0.13 packaged apps.

The packaged bundles contain the saved-title guard copied from the v0.12
export. Its strategy/canonical checks were migrated to v0.13 by the release,
but the previous-version literal stayed '11', so a title saved under v0.12
("NQ=F · v0.12 追体験") is never auto-corrected to the v0.13 canonical title.
Replace exactly that one literal inside the base64 bundle, repackage the
gzip deterministically, and refresh the manifest hashes. Snapshot data is
not touched.
"""
import base64
import gzip
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
OLD = "prior?.[2] === '11'"
NEW = "prior?.[2] === '12'"
SCRIPT = re.compile(r"(<script\b[^>]*>)(.*?)(</script>)", re.S | re.I)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def patch_html(html):
    changed = 0

    def upgrade(match):
        nonlocal changed
        if 'type="application/json"' in match.group(1):
            return match.group(0)
        encoded = re.search(r'(src="data:text/javascript[^,]*;base64,)([^"]+)(")', match.group(1))
        if not encoded:
            body = match.group(2)
            if OLD in body:
                changed += body.count(OLD)
                body = body.replace(OLD, NEW)
            return match.group(1) + body + match.group(3)
        source = base64.b64decode(encoded.group(2)).decode("utf-8")
        if OLD in source:
            changed += source.count(OLD)
            source = source.replace(OLD, NEW)
        replacement = base64.b64encode(source.encode()).decode()
        return match.group(0).replace(encoded.group(2), replacement, 1)

    return SCRIPT.sub(upgrade, html), changed


def main():
    manifest = json.loads((ASSETS / "manifest.json").read_text(encoding="utf-8"))
    for market in ["nq", "es"]:
        item = manifest["markets"][market]
        path = ASSETS / f"{market}.html.gz"
        packed = path.read_bytes()
        assert digest(packed) == item["gzip_sha256"], f"{market}: input gzip hash drift"
        html = gzip.decompress(packed).decode("utf-8")
        assert digest(html.encode()) == item["html_sha256"], f"{market}: input html hash drift"
        snapshot_meta = re.search(r'<meta name="data-app-snapshot-sha256" content="([^"]+)"', html)
        patched, changed = patch_html(html)
        assert changed == 1, f"{market}: expected exactly 1 guard literal, found {changed}"
        decoded = []
        for mm in SCRIPT.finditer(patched):
            enc = re.search(r';base64,([^"]+)', mm.group(1))
            if enc:
                decoded.append(base64.b64decode(enc.group(1)).decode("utf-8"))
        bundle_text = "\n".join(decoded)
        assert OLD not in bundle_text and NEW in bundle_text, f"{market}: guard literal not updated in bundle"
        assert re.search(r'<meta name="data-app-snapshot-sha256" content="([^"]+)"', patched).group(1) == snapshot_meta.group(1), \
            f"{market}: snapshot data must not change"
        raw = patched.encode()
        new_packed = gzip.compress(raw, compresslevel=9, mtime=0)
        path.write_bytes(new_packed)
        item["html_bytes"] = len(raw)
        item["html_sha256"] = digest(raw)
        item["gzip_bytes"] = len(new_packed)
        item["gzip_sha256"] = digest(new_packed)
        print(f"{market}: guard literal patched, gzip {len(packed)} -> {len(new_packed)} bytes")
    (ASSETS / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("PASS_TITLE_GUARD_FIX")
    return 0


if __name__ == "__main__":
    sys.exit(main())
