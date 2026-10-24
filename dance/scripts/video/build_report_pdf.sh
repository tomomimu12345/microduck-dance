#!/usr/bin/env bash
# Build the course report PDF from dance/docs/report.md.
#
# Needs pandoc, lualatex with luatexja, and the Windows fonts Arial and BIZ UDGothic
# (dance/docs/report-fonts.tex). Set FONTDIR if the fonts are not in /mnt/c/Windows/Fonts.
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

export FONTDIR="${FONTDIR:-/mnt/c/Windows/Fonts}"
# report.md inputs report-fonts.tex, which lives next to it.
export TEXINPUTS="$REPO/dance/docs:${TEXINPUTS:-}"
for tool in pandoc lualatex; do
    command -v "$tool" >/dev/null || { echo "missing: $tool" >&2; exit 1; }
done
kpsewhich luatexja.sty >/dev/null || { echo "missing: luatexja (texlive-lang-japanese)" >&2; exit 1; }
for f in arial.ttf arialbd.ttf BIZ-UDGothicR.ttc BIZ-UDGothicB.ttc; do
    [[ -f "$FONTDIR/$f" ]] || { echo "missing font: $FONTDIR/$f" >&2; exit 1; }
done

mkdir -p "$OUT_DIR"
# Figures are referenced relative to dance/docs, so resolve them from there.
pandoc "$SRC" \
    --from markdown \
    --pdf-engine=lualatex \
    --resource-path="$REPO/dance/docs" \
    -o "$PDF"

echo "wrote $PDF"
if command -v pdfinfo >/dev/null; then
    pdfinfo "$PDF" | grep -E '^Pages:'
fi

if [[ "${1:-}" == "--png" ]]; then
    command -v pdftoppm >/dev/null || { echo "missing: pdftoppm (poppler-utils)" >&2; exit 1; }
    for page in $(seq 1 "$(pdfinfo "$PDF" | awk '/^Pages:/{print $2}')"); do
        pdftoppm -png -r 80 -f "$page" -l "$page" -singlefile "$PDF" "$OUT_DIR/page-$page"
        echo "wrote $OUT_DIR/page-$page.png"
    done
fi
