# Sourced by the pipeline scripts. Defines `in_env <env> <command...>`, which
# runs a command with that venv first on PATH (the official LILaC shell
# scripts call plain `python3`, so this is how they pick the right env).
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -d /content ]]; then _DEF_ENV_ROOT=/content/envs; else _DEF_ENV_ROOT="$REPO/.envs"; fi
export ENV_ROOT="${ENV_ROOT:-$_DEF_ENV_ROOT}"
export LILAC_ROOT="${LILAC_ROOT:-$REPO/third_party/lilac}"
export MMRAG_RESULTS_DIR="${MMRAG_RESULTS_DIR:-$REPO/results}"
export LILAC_CONDA_HOME="/nonexistent"     # official preamble: skip conda, use PATH python
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export TOKENIZERS_PARALLELISM=false

in_env() {
  local env="$1"; shift
  if [[ ! -x "$ENV_ROOT/$env/bin/python" ]]; then
    echo "env '$env' missing in $ENV_ROOT - run scripts/setup_envs.sh $env" >&2; return 1
  fi
  PATH="$ENV_ROOT/$env/bin:$PATH" VIRTUAL_ENV="$ENV_ROOT/$env" "$@"
}
