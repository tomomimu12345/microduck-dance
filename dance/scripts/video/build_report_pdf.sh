#!/usr/bin/env bash
# Build the course report PDF from dance/docs/report.md.
#
# Needs pandoc, xelatex with xeCJK and the IPAexGothic font (all checked below).
# Output: out/report/説明資料_PAI最終課題_tom11111111.pdf
# With --png it also renders pages 1 and 3 to out/report/page-1.png and page-3.png
# (needs pdftoppm) so the layout can be checked by eye.
#
# Run from anywhere: bash dance/scripts/video/build_report_pdf.sh [--png]
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SRC="$REPO/dance/docs/report.md"
OUT_DIR="$REPO/out/report"
PDF="$OUT_DIR/説明資料_PAI最終課題_tom11111111.pdf"

for tool in pandoc xelatex; do
    command -v "$tool" >/dev/null || { echo "missing: $tool" >&2; exit 1; }
done
kpsewhich xeCJK.sty >/dev/null || { echo "missing: xeCJK (texlive-lang-cjk)" >&2; exit 1; }
fc-list : family | grep -c IPAexGothic >/dev/null || { echo "missing font: IPAexGothic (fonts-ipaexfont)" >&2; exit 1; }

mkdir -p "$OUT_DIR"
# Figures are referenced relative to dance/docs, so resolve them from there.
pandoc "$SRC" \
    --from markdown \
    --pdf-engine=xelatex \
    --resource-path="$REPO/dance/docs" \
    -o "$PDF"

echo "wrote $PDF"
if command -v pdfinfo >/dev/null; then
    pdfinfo "$PDF" | grep -E '^Pages:'
fi

if [[ "${1:-}" == "--png" ]]; then
    command -v pdftoppm >/dev/null || { echo "missing: pdftoppm (poppler-utils)" >&2; exit 1; }
    for page in 1 3; do
        pdftoppm -png -r 80 -f "$page" -l "$page" -singlefile "$PDF" "$OUT_DIR/page-$page"
        echo "wrote $OUT_DIR/page-$page.png"
    done
fi
