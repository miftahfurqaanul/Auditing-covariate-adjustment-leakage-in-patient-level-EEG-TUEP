#!/usr/bin/env bash
# Verifies every tracked file against SHA256SUMS.txt (run from any directory, before rewrite_paths.sh).
cd "$(dirname "$0")/.." && sha256sum -c --quiet SHA256SUMS.txt && echo "all files match SHA256SUMS.txt"
