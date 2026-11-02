#!/usr/bin/env bash
# Rewrites the hard-coded paths of the original workstation (/home/miftah_f/RTX-5090/...) in code/, analysis/
# and the figure scripts so that they point to this clone and to a working directory of your choice.
#
#   bash tools/rewrite_paths.sh /path/to/WORKDIR
#
# WORKDIR must contain, under the same names as in the original workspace:
#   tuep_fresh/data       (TUEP index and derived features; see README, "Data")
#   tuep_fresh/external   (optional: Kural, Emotiv)
#   tuep_fresh/reports    (created by the scripts)
#   EEG-TUEP-v2.0.1/.venv (or edit the interpreter path)
# Run `bash tools/verify_sha256.sh` BEFORE this script: it changes the files, so SHA256SUMS.txt of this repository
# no longer matches afterwards.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$(cd "${1:?usage: bash tools/rewrite_paths.sh /path/to/WORKDIR}" && pwd)"
OLD="/home/miftah_f/RTX-5090"
mapfile -t FILES < <(grep -rIl "$OLD" "$REPO/code" "$REPO/analysis" "$REPO/manuscript/figures" "$REPO/manuscript/figures_en" 2>/dev/null || true)
for f in "${FILES[@]}"; do
  sed -i \
    -e "s#$OLD/tuep_fresh/code#$REPO/code#g" \
    -e "s#$OLD/ijmi-eeg-antigravity/ijmi-eeg/analisis_tambahan#$REPO/analysis#g" \
    -e "s#$OLD/ijmi-eeg-antigravity/ijmi-eeg#$REPO/manuscript#g" \
    -e "s#$OLD#$WORK#g" "$f"
done
echo "rewrote ${#FILES[@]} files; remaining references: $(grep -rIl "$OLD" "$REPO/code" "$REPO/analysis" "$REPO/manuscript/figures" "$REPO/manuscript/figures_en" 2>/dev/null | wc -l)"
