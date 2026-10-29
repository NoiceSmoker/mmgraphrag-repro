#!/usr/bin/env bash
# Persist / restore pipeline artefacts on Google Drive between Colab sessions.
#
#   ./scripts/colab/sync.sh push     # VM -> Drive   (run after every finished stage)
#   ./scripts/colab/sync.sh pull     # Drive -> VM   (run at the start of a new session)
#
# Work happens on the VM's local disk (fast, safe for SQLite); Drive only
# stores copies. DRIVE_DIR defaults to /content/drive/MyDrive/mmgraphrag-repro-cache.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_env.sh"
DRIVE_DIR="${DRIVE_DIR:-/content/drive/MyDrive/mmgraphrag-repro-cache}"
DATASET="${DATASET:-MMCoQA}"

pairs=(
  "$LILAC_ROOT/datasets/$DATASET:lilac/datasets/$DATASET"
  "$LILAC_ROOT/artifacts/$DATASET:lilac/artifacts/$DATASET"
  "$LILAC_ROOT/algorithm_results:lilac/algorithm_results"
  "$REPO/work:work"
  "$REPO/results:results"
)

case "${1:-}" in
  push)
    for p in "${pairs[@]}"; do src="${p%%:*}"; dst="$DRIVE_DIR/${p#*:}"
      [[ -e "$src" ]] || continue; mkdir -p "$dst"; rsync -a --info=stats1 "$src/" "$dst/"; done ;;
  pull)
    for p in "${pairs[@]}"; do dst="${p%%:*}"; src="$DRIVE_DIR/${p#*:}"
      [[ -e "$src" ]] || continue; mkdir -p "$dst"; rsync -a --info=stats1 "$src/" "$dst/"; done ;;
  *) echo "usage: $0 push|pull" >&2; exit 1 ;;
esac
