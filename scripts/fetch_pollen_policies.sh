#!/usr/bin/env bash
# Download Pollen's released microduck policies that this project uses, at a pinned
# commit, and check their SHA-256.
#
# Usage: bash scripts/fetch_pollen_policies.sh
#
# Result: logs/tricks/pollen-microduck-policies/<file>.onnx (logs/ is not tracked by git).
#   roller.onnx          start of the roller_dance fine-tune (scripts/onnx_to_checkpoint.py)
#   roulade.onnx         front roll trick
#   alpha_sitstand.onnx  sit and stand trick
#   velstand.onnx        standing policy between a roulade and the dance
# Source: https://huggingface.co/pollen-robotics/microduck-policies at commit
# 665ae4eb20bfe13bca1ef4ea6d4bb6b4a096e448, Apache-2.0 (dance/data/CREDITS.md).
# The hashes are the Hugging Face LFS records at that commit.
# A file that is already present and matches its hash is not downloaded again.
set -euo pipefail

REPO=pollen-robotics/microduck-policies
COMMIT=665ae4eb20bfe13bca1ef4ea6d4bb6b4a096e448
ROOT=$(cd "$(dirname "$0")/.." && pwd)
DEST_DIR=$ROOT/logs/tricks/pollen-microduck-policies
declare -A SHA256=(
    [roller.onnx]=cf05651d2708a2f9364212e86b866c97a70ace8131c492500105e8f28bf99afd
    [roulade.onnx]=3d60da08fc13f29c1b57f41977aa898132c0d60042100149d8e775affcbca32b
    [alpha_sitstand.onnx]=a6f102d9fd655782ba85bfffe18175c34621da719cffcdd673be7982a06fc254
    [velstand.onnx]=2fc7f8ee6acd7d0d3f66a99927b5d5a56589aeed02ec714de329b252d5ccf492
)

check() {
    echo "$1  $2" | sha256sum -c --quiet -
}

mkdir -p "$DEST_DIR"
for name in roller.onnx roulade.onnx alpha_sitstand.onnx velstand.onnx; do
    dest=$DEST_DIR/$name
    sha=${SHA256[$name]}
    if [[ -f $dest ]] && check "$sha" "$dest"; then
        echo "already present: $dest"
        continue
    fi
    url=https://huggingface.co/$REPO/resolve/$COMMIT/$name
    tmp=$dest.part
    curl -fsSL --retry 3 -o "$tmp" "$url"
    if ! check "$sha" "$tmp"; then
        echo "SHA-256 mismatch for $url, expected $sha" >&2
        rm -f "$tmp"
        exit 1
    fi
    mv "$tmp" "$dest"
    echo "wrote $dest"
done
