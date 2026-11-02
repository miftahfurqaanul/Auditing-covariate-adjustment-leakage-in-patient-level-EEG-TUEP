#!/bin/bash
# Builds the English manuscript in a separate folder (build/out); does not change the project files.
# The main text (main_en.pdf) and the Supplementary Material S1--S3 (supl_S1S3_en.pdf) are two documents that refer to
# each other (xr-hyper): the build order is main_en -> supl_S1S3_en -> main_en so that the numbers of the S2 and S3 tables
# in the main text and the numbers of the main-text tables in the supplement are correct.
# Usage: TECTONIC=tectonic bash build/build_en.sh [--install]   (--install copies the two PDFs to the manuscript folder)
set -e
A="$(cd "$(dirname "$0")" && pwd)"; P="$(dirname "$A")"; B="$A/out"
mkdir -p "$B/figures_en" "$B/naskah"; cd "$B"
cp "$P/main_en.tex" "$P/supl_S1S3_en.tex" "$P/preamble.tex" "$P/referensi.bib" .
cp "$P"/figures_en/*.pdf figures_en/ 2>/dev/null || true
rm -f naskah/*.tex; cp "$P"/naskah/*.tex naskah/
echo "% generated" > nocite_utama.tex; rm -f main_en.aux supl_S1S3_en.aux
T="${TECTONIC:-tectonic} --keep-intermediates"
$T main_en.tex > comp_last.log 2>&1 || true
python3 - <<'PY'
import re
k=[]
for m in re.findall(r'\\citation\{([^}]*)\}',open('main_en.aux').read()):
    for x in m.split(','):
        if x not in k: k.append(x)
open('nocite_utama.tex','w').write('\\nocite{'+','.join(k)+'}\n')
PY
$T supl_S1S3_en.tex > comp_sup.log 2>&1 || true
$T main_en.tex > comp_last.log 2>&1 || true
echo "[main] undefined/errors: $(grep -c -i -E 'undefined|error' comp_last.log); overfull: $(grep -c Overfull comp_last.log); pages $(pdfinfo main_en.pdf | grep Pages | awk '{print $2}')"
echo "[S1-S3] undefined/errors: $(grep -c -i -E 'undefined|error' comp_sup.log); overfull: $(grep -c Overfull comp_sup.log); pages $(pdfinfo supl_S1S3_en.pdf | grep Pages | awk '{print $2}')"
grep -h -i -E "undefined|Overfull|error" comp_last.log comp_sup.log | sort -u | head -12
if [ "$1" = "--install" ]; then cp main_en.pdf supl_S1S3_en.pdf "$P/"; echo "main_en.pdf and supl_S1S3_en.pdf updated"; fi
exit 0
