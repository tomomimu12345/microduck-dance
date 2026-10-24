#!/usr/bin/env bash
# Download the released b11 basketball checkpoint (iteration 6999) that the
# ball dance continues from, and check its SHA-256.
#
# Usage: bash scripts/fetch_basketball.sh
#
# Result: logs/rsl_rl/basketball/b11_6999/checkpoint.pt (logs/ is not tracked by git).
# Source: https://huggingface.co/HannesVonEssen/microduck-basketball (checkpoint.pt).
# If the repo is private for you, export HF_TOKEN first.
# A file that is already present and matches the hash is not downloaded again.
set -euo pipefail

REPO=HannesVonEssen/microduck-basketball
SHA256=57a322eff092cb71e7cba831791232f0cc4fd45ede0441cad6d87813b0033e41
ROOT=$(cd "$(dirname "$0")/.." && pwd)
DEST_DIR=$ROOT/logs/rsl_rl/basketball/b11_6999
DEST=$DEST_DIR/checkpoint.pt
URL=https://huggingface.co/$REPO/resolve/main/checkpoint.pt

check() {
    echo "$SHA256  $1" | sha256sum -c -
}

mkdir -p "$DEST_DIR"
if [[ -f $DEST ]] && check "$DEST"; then
    echo "already present: $DEST"
    exit 0
fi

AUTH=()
if [[ -n ${HF_TOKEN:-} ]]; then
    AUTH=(-H "Authorization: Bearer $HF_TOKEN")
fi
TMP=$DEST.part
curl -fL --retry 3 "${AUTH[@]}" -o "$TMP" "$URL"
if ! check "$TMP"; then
    echo "SHA-256 mismatch for $URL, expected $SHA256" >&2
    rm -f "$TMP"
    exit 1
fi
mv "$TMP" "$DEST"
echo "wrote $DEST"
