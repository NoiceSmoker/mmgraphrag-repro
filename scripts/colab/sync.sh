#!/usr/bin/env bash
# Persist / restore pipeline artefacts on Google Drive between Colab sessions.
#
#   ./scripts/colab/sync.sh push [lilac|m3kg|results|all]   # VM -> Drive   (after every finished stage)
#   ./scripts/colab/sync.sh pull [lilac|m3kg|results|all]   # Drive -> VM   (at the start of a new session)
#
# Work happens on the VM's local disk (fast, safe for SQLite). Each artefact directory is stored on
# Drive as ONE tar file: Drive's FUSE mount is very slow for thousands of small files (VLM outputs,
# crops, audio clips) but fast for one large sequential file. Datasets are not stored: downloading
# them again from Hugging Face is faster than reading them back from Drive.
# DRIVE_DIR defaults to /content/drive/MyDrive/mmgraphrag-repro-store.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_env.sh"
DRIVE_DIR="${DRIVE_DIR:-/content/drive/MyDrive/mmgraphrag-repro-store}"
DATASET="${DATASET:-MMCoQA}"

SCOPE="${2:-all}"   # lilac | m3kg | results | all - each notebook only moves its own artefacts
lilac_pairs=(
  "$LILAC_ROOT/artifacts/$DATASET:lilac-artifacts-$DATASET"
  "$LILAC_ROOT/algorithm_results:lilac-algorithm_results"
  "$REPO/results/lilac-repro:results-lilac-repro"
)
m3kg_pairs=(
  "$REPO/work:m3kg-work"
  "$REPO/results/m3kg-rag-repro:results-m3kg-rag-repro"
)
case "$SCOPE" in
  lilac)   pairs=("${lilac_pairs[@]}") ;;
  m3kg)    pairs=("${m3kg_pairs[@]}") ;;
  results) pairs=("$REPO/results:results-all") ;;
  all)     pairs=("${lilac_pairs[@]}" "${m3kg_pairs[@]}") ;;
  *) echo "scope must be lilac, m3kg, results or all" >&2; exit 1 ;;
esac

t0=$SECONDS
if [[ ! -d "$(dirname "$DRIVE_DIR")" ]]; then
  echo "[sync] WARNING: Google Drive is not mounted ($(dirname "$DRIVE_DIR") missing) - nothing is saved"
  exit 0
fi
mkdir -p "$DRIVE_DIR"

case "${1:-}" in
  push)
    for p in "${pairs[@]}"; do src="${p%%:*}"; tarf="$DRIVE_DIR/${p#*:}.tar"
      [[ -d "$src" ]] || continue
      echo "[sync] push $(du -sh "$src" | cut -f1) ${src#"$REPO"/} -> Drive/${p#*:}.tar"
      tar -C "$(dirname "$src")" -cf "$tarf.partial" "$(basename "$src")"
      mv -f "$tarf.partial" "$tarf"     # replace the old copy only once the new one is complete
    done ;;
  pull)
    for p in "${pairs[@]}"; do dst="${p%%:*}"; tarf="$DRIVE_DIR/${p#*:}.tar"
      if [[ ! -f "$tarf" ]]; then echo "[sync] nothing on Drive for ${p#*:}"; continue; fi
      echo "[sync] pull $(du -sh "$tarf" | cut -f1) Drive/${p#*:}.tar -> ${dst#"$REPO"/}"
      mkdir -p "$(dirname "$dst")"
      tar -C "$(dirname "$dst")" --keep-newer-files -xf "$tarf" 2>&1 | grep -v "is newer or same age" || true
    done ;;
  *) echo "usage: $0 push|pull [lilac|m3kg|results|all]" >&2; exit 1 ;;
esac
echo "[sync] $1 $SCOPE done in $((SECONDS - t0)) s"
