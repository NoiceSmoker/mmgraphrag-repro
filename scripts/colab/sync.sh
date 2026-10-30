#!/usr/bin/env bash
# Persist / restore pipeline artefacts on Google Drive between Colab sessions.
#
#   ./scripts/colab/sync.sh push [lilac|m3kg|all]   # VM -> Drive   (after every finished stage)
#   ./scripts/colab/sync.sh pull [lilac|m3kg|all]   # Drive -> VM   (at the start of a new session)
#
# Work happens on the VM's local disk (fast, safe for SQLite); Drive only
# stores copies. DRIVE_DIR defaults to /content/drive/MyDrive/mmgraphrag-repro-cache.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_env.sh"
DRIVE_DIR="${DRIVE_DIR:-/content/drive/MyDrive/mmgraphrag-repro-cache}"
DATASET="${DATASET:-MMCoQA}"

SCOPE="${2:-all}"   # lilac | m3kg | results | all - each notebook only moves its own artefacts
lilac_pairs=(
  "$LILAC_ROOT/datasets/$DATASET:lilac/datasets/$DATASET"
  "$LILAC_ROOT/artifacts/$DATASET:lilac/artifacts/$DATASET"
  "$LILAC_ROOT/algorithm_results:lilac/algorithm_results"
  "$REPO/results/lilac-repro:results/lilac-repro"
)
m3kg_pairs=(
  "$REPO/work:work"
  "$REPO/results/m3kg-rag-repro:results/m3kg-rag-repro"
)
case "$SCOPE" in
  lilac) pairs=("${lilac_pairs[@]}") ;;
  m3kg)  pairs=("${m3kg_pairs[@]}") ;;
  results) pairs=("$REPO/results:results") ;;
  all)   pairs=("${lilac_pairs[@]}" "${m3kg_pairs[@]}" "$REPO/results:results") ;;
  *) echo "scope must be lilac, m3kg, results or all" >&2; exit 1 ;;
esac

# rsync if available (incremental, with a progress line), otherwise a plain recursive copy
copy_dir() {
  if command -v rsync >/dev/null 2>&1; then rsync -a --info=progress2 "$1/" "$2/"; echo
  else cp -a "$1/." "$2/"; fi
}
t0=$SECONDS
[[ -d "$(dirname "$DRIVE_DIR")" ]] || echo "[sync] WARNING: Google Drive is not mounted ($(dirname "$DRIVE_DIR") missing)"

case "${1:-}" in
  push)
    for p in "${pairs[@]}"; do src="${p%%:*}"; dst="$DRIVE_DIR/${p#*:}"
      [[ -e "$src" ]] || continue; echo "[sync] push ${p#*:} -> Drive"; mkdir -p "$dst"; copy_dir "$src" "$dst"; done ;;
  pull)
    for p in "${pairs[@]}"; do dst="${p%%:*}"; src="$DRIVE_DIR/${p#*:}"
      if [[ ! -e "$src" ]]; then echo "[sync] nothing on Drive for ${p#*:}"; continue; fi
      echo "[sync] pull ${p#*:} <- Drive"; mkdir -p "$dst"; copy_dir "$src" "$dst"; done ;;
  *) echo "usage: $0 push|pull [lilac|m3kg|results|all]" >&2; exit 1 ;;
esac
echo "[sync] $1 $SCOPE done in $((SECONDS - t0)) s"
