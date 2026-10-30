#!/usr/bin/env bash
# M3KG-RAG (scaled-down, Audio-QA) pipeline. Stages, in order:
#
#   ./scripts/m3kg/pipeline.sh data       # AudioCaps-QA test set + AudioCaps train subset   (CPU, m3kg-audio env)
#   ./scripts/m3kg/pipeline.sh youtube    # optional: YouTube titles/descriptions for the rewriter (CPU)
#   ./scripts/m3kg/pipeline.sh kg         # multi-agent KG construction + Wikipedia look-ups   (GPU, vllm env)
#   ./scripts/m3kg/pipeline.sh index      # CLAP embeddings, leakage guard, tau calibration   (GPU, m3kg-audio env)
#   ./scripts/m3kg/pipeline.sh contexts   # retrieval + TAG grounding for every config         (GPU, m3kg-audio env)
#   ./scripts/m3kg/pipeline.sh filter     # GRASP LLM pruning                                  (GPU, vllm env)
#   ./scripts/m3kg/pipeline.sh generate   # Qwen2.5-Omni answers                               (GPU, vllm env)
#   ./scripts/m3kg/pipeline.sh judge      # AudioBench Model-as-Judge                          (GPU, vllm env)
#   ./scripts/m3kg/pipeline.sh winrate    # optional pairwise preference judge                 (GPU, vllm env)
#   ./scripts/m3kg/pipeline.sh report     # metrics + bootstrap CIs -> W&B                     (CPU)
#
# Env vars: PROFILE (default configs/m3kg/profiles/colab_l4.yaml), SET (extra --set overrides),
#           WORK_DIR (overrides paths.work_dir), EXP_GROUPS (subset of experiment groups for 'contexts';
#           not GROUPS, which bash reserves for the user's group ids).
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_env.sh"
cd "$REPO"

CONFIG="${CONFIG:-configs/m3kg/base.yaml}"
PROFILE="${PROFILE:-configs/m3kg/profiles/colab_l4.yaml}"
EXTRA_SET="${SET:-}"
[[ -n "${WORK_DIR:-}" ]] && EXTRA_SET="paths.work_dir=$WORK_DIR $EXTRA_SET"
AUDIOBENCH_COMMIT="${AUDIOBENCH_COMMIT:-main}"
ARGS=(--config "$CONFIG" --profile "$PROFILE")
# shellcheck disable=SC2206
[[ -n "$EXTRA_SET" ]] && ARGS+=(--set $EXTRA_SET)

stage_data()     { in_env m3kg-audio python -m mmrag.m3kg.data "${ARGS[@]}"; }
stage_youtube()  { in_env m3kg-audio python -m mmrag.m3kg.youtube "${ARGS[@]}"; }
stage_kg()       { in_env vllm python -m mmrag.m3kg.build_kg "${ARGS[@]}"; }
stage_index()    { in_env m3kg-audio python -m mmrag.m3kg.index "${ARGS[@]}"; }
stage_contexts() {
  # shellcheck disable=SC2086
  in_env m3kg-audio python -m mmrag.m3kg.contexts "${ARGS[@]}" ${EXP_GROUPS:+--groups $EXP_GROUPS}
}
stage_filter()   { in_env vllm python -m mmrag.m3kg.vllm_stages filter "${ARGS[@]}"; }
stage_generate() { in_env vllm python -m mmrag.m3kg.vllm_stages generate "${ARGS[@]}"; }
stage_judge() {
  if [[ ! -d third_party/AudioBench ]]; then
    git clone --depth 1 https://github.com/AudioLLMs/AudioBench.git third_party/AudioBench
    [[ "$AUDIOBENCH_COMMIT" != "main" ]] && git -C third_party/AudioBench checkout -q "$AUDIOBENCH_COMMIT"
  fi
  in_env vllm python -m mmrag.m3kg.vllm_stages judge "${ARGS[@]}"
}
stage_winrate()  { in_env vllm python -m mmrag.m3kg.vllm_stages winrate "${ARGS[@]}"; }
stage_report()   { in_env m3kg-audio python -m mmrag.m3kg.report "${ARGS[@]}"; }

for stage in "$@"; do
  echo "=================== M3KG stage: $stage ($(date '+%H:%M:%S')) ==================="
  t_stage=$SECONDS
  "stage_$stage"
  echo "=================== M3KG stage: $stage finished in $(( (SECONDS - t_stage) / 60 )) min $(( (SECONDS - t_stage) % 60 )) s"
done
