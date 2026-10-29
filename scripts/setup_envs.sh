#!/usr/bin/env bash
# Create the isolated Python environments used by the pipelines.
#
# The two reproductions need mutually incompatible stacks (MM-Embed needs
# transformers 4.46; vLLM needs a recent transformers; the TAG grounding
# model needs transformers < 5), so each stage runs in its own uv venv.
#
#   ./scripts/setup_envs.sh lilac-base mmembed vllm      # LILaC
#   ./scripts/setup_envs.sh vllm m3kg-audio              # M3KG-RAG
#   ./scripts/setup_envs.sh all
#
# Env vars:
#   ENV_ROOT         where venvs are created           (default: /content/envs on Colab, ./.envs elsewhere)
#   TORCH_INDEX_URL  PyTorch wheel index for non-vLLM envs (default: CUDA 12.8 wheels;
#                    cu128 is also what RTX 50-series / Blackwell GPUs require)
#   VLLM_SPEC        pip spec for vLLM                 (default: vllm)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -d /content ]]; then DEFAULT_ROOT=/content/envs; else DEFAULT_ROOT="$REPO/.envs"; fi
ENV_ROOT="${ENV_ROOT:-$DEFAULT_ROOT}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
VLLM_SPEC="${VLLM_SPEC:-vllm}"

command -v uv >/dev/null 2>&1 || python3 -m pip install -q uv

mkenv() {  # mkenv <name> <python-version>
  local name="$1" py="$2"
  if [[ ! -x "$ENV_ROOT/$name/bin/python" ]]; then
    uv venv -q --python "$py" "$ENV_ROOT/$name"
  fi
  echo "$ENV_ROOT/$name/bin/python"
}

install_repo() {  # editable install of this repository (no deps)
  uv pip install -q --python "$1" -e "$REPO" --no-deps
}

setup_one() {
  local which="$1" py
  echo "==> setting up env '$which' in $ENV_ROOT"
  case "$which" in
    lilac-base)
      py=$(mkenv lilac-base 3.10)
      uv pip install -q --python "$py" torch --index-url "$TORCH_INDEX_URL"
      uv pip install -q --python "$py" -r "$REPO/requirements/lilac-base.txt"
      ;;
    mmembed)
      py=$(mkenv mmembed 3.10)
      uv pip install -q --python "$py" torch torchvision --index-url "$TORCH_INDEX_URL"
      uv pip install -q --python "$py" -r "$REPO/requirements/mmembed.txt"
      ;;
    vllm)
      py=$(mkenv vllm 3.12)
      uv pip install -q --python "$py" "$VLLM_SPEC"
      uv pip install -q --python "$py" -r "$REPO/requirements/vllm.txt"
      ;;
    m3kg-audio)
      py=$(mkenv m3kg-audio 3.11)
      uv pip install -q --python "$py" torch torchaudio --index-url "$TORCH_INDEX_URL"
      uv pip install -q --python "$py" -r "$REPO/requirements/m3kg-audio.txt"
      ;;
    *) echo "unknown env: $which" >&2; exit 1;;
  esac
  install_repo "$py"
  "$py" -c "import sys, torch; print('  python', sys.version.split()[0], '| torch', torch.__version__, '| cuda', torch.cuda.is_available())"
}

targets=("$@")
[[ ${#targets[@]} -eq 0 || "${targets[0]}" == "all" ]] && targets=(lilac-base mmembed vllm m3kg-audio)
for t in "${targets[@]}"; do setup_one "$t"; done
echo "done. activate an env with: source $ENV_ROOT/<name>/bin/activate"
