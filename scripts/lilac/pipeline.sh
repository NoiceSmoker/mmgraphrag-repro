#!/usr/bin/env bash
# LILaC reproduction pipeline (single GPU). Run stages in order:
#
#   ./scripts/lilac/pipeline.sh fetch        # clone official code at the pinned commit      (CPU)
#   ./scripts/lilac/pipeline.sh data         # download + validate the MMCoQA document corpus (CPU)
#   ./scripts/lilac/pipeline.sh vlm          # Qwen2.5-VL summaries, detections, crop captions (GPU, vllm env)
#   ./scripts/lilac/pipeline.sh graph        # official steps 1,2,3,5-inject,6,7               (CPU/GPU, lilac-base env)
#   ./scripts/lilac/pipeline.sh embed        # MM-Embed corpus embeddings (official step 8)     (GPU >= 20 GB, or T4 + MMEMBED_QUANT)
#   ./scripts/lilac/pipeline.sh decompose    # sub-queries for every decomposer + their embeddings (GPU)
#                                            #   = decompose_llm (vLLM) + decompose_embed (MM-Embed)
#   ./scripts/lilac/pipeline.sh novlm        # extra ablation: image nodes without VLM text     (GPU, optional)
#   ./scripts/lilac/pipeline.sh experiments  # all retrieval experiments + W&B logging           (CPU ok)
#
# Configuration (env vars):
#   DATASET       (default MMCoQA)
#   DECOMPOSERS   space-separated "tag=hf_model_id" pairs; the first one is the main decomposer
#   VLM_MODEL     model for the three image passes (default: the official Qwen/Qwen2.5-VL-7B-Instruct;
#                 on a 16 GB T4 use Qwen/Qwen2.5-VL-7B-Instruct-AWQ)
#   MMEMBED_QUANT 8bit | 4bit: load MM-Embed with bitsandbytes weights so that it fits a 16 GB T4
#                 (default: official fp16, needs >= 20 GB). Must be the same for every MM-Embed stage.
#   MMEMBED_BATCH_SCALE  factor for the official embedding batch sizes (default 1; halved on OOM anyway)
#   GROUPS        experiment groups for the last stage
#   WANDB_PROJECT / WANDB_ENTITY  W&B destination (project default: lilac-repro)
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_env.sh"

DATASET="${DATASET:-MMCoQA}"
DECOMPOSERS="${DECOMPOSERS:-qwen2.5-7b=Qwen/Qwen2.5-7B-Instruct qwen2.5-14b-awq=Qwen/Qwen2.5-14B-Instruct-AWQ qwen2.5-32b-awq=Qwen/Qwen2.5-32B-Instruct-AWQ}"
GROUPS="${GROUPS:-main ablation sens_beam sens_iter decomposer}"
VLM_MODEL="${VLM_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
HF_REPO_MMCOQA="${HF_REPO_MMCOQA:-JoohyungYun/mmcoqa_doc}"
HF_REPO_MMQA="${HF_REPO_MMQA:-JoohyungYun/multimodalqa_doc}"
PINNED_COMMIT=813b6c0cbd7c67ae9a4ec9ca070ac5190a7acfe0

main_tag() { echo "${DECOMPOSERS%% *}" | cut -d= -f1; }
all_tags() { for p in $DECOMPOSERS; do echo -n "${p%%=*} "; done; }

# MM-Embed commands: the _site dir on PYTHONPATH makes every process (incl. the official
# spawn workers) install the quantized loader when MMEMBED_QUANT is set.
in_mmembed() {
  PYTHONPATH="$REPO/src/mmrag/lilac/_site${PYTHONPATH:+:$PYTHONPATH}" in_env mmembed "$@"
}

# corpus and query embeddings must come from the same weights: record and check
mmembed_meta() {
  in_env lilac-base python - "$DATASET" "$1" <<'PY'
import json, sys
from mmrag.lilac import official
from mmrag.lilac.mmembed_quant import describe, write_meta
dataset, action = sys.argv[1], sys.argv[2]
path = official.artifact_dir(dataset) / "mmembed_meta.json"
now = describe()
if action == "write":
    write_meta(path)
elif action == "audit":  # zero vectors = batches whose encoding failed (the official code stores zeros)
    import torch
    meta = json.loads(path.read_text()) if path.exists() else now
    zero = {}
    for pt in sorted(official.embeddings_dir(dataset).glob("*.pt")):
        t = torch.load(pt, map_location="cpu")
        if torch.is_tensor(t) and t.dim() == 2:
            zero[pt.stem] = [int((t.abs().sum(dim=1) == 0).sum()), int(t.shape[0])]
    meta["zero_rows"] = zero
    path.write_text(json.dumps(meta, indent=1))
    bad = {k: v for k, v in zero.items() if v[0]}
    print("zero embeddings per component:", bad if bad else "none")
elif path.exists():
    before = json.loads(path.read_text())
    if before.get("mmembed_weights") != now["mmembed_weights"]:
        sys.exit(f"MM-Embed weights mismatch: corpus embedded with {before['mmembed_weights']}, "
                 f"now MMEMBED_QUANT gives {now['mmembed_weights']}. Use the same setting for all MM-Embed stages.")
print(f"MM-Embed weights: {now['mmembed_weights']}")
PY
}

stage_fetch() {
  if [[ ! -d "$LILAC_ROOT/.git" ]]; then
    git clone https://github.com/joohyung00/lilac.git "$LILAC_ROOT"
  fi
  git -C "$LILAC_ROOT" fetch --depth 1 origin "$PINNED_COMMIT" 2>/dev/null || true
  git -C "$LILAC_ROOT" checkout -q "$PINNED_COMMIT"
  echo "official LILaC at $(git -C "$LILAC_ROOT" rev-parse --short HEAD) in $LILAC_ROOT"
}

stage_data() {
  local repo="$HF_REPO_MMCOQA"; [[ "$DATASET" == "MultimodalQA" ]] && repo="$HF_REPO_MMQA"
  local dst="$LILAC_ROOT/datasets/$DATASET"
  in_env lilac-base python - "$repo" "$dst" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], repo_type="dataset", local_dir=sys.argv[2])
PY
  # Some releases ship archives or parquet + load.py instead of the LILaC layout.
  (cd "$dst" && for a in *.zip; do [[ -e "$a" ]] && unzip -q -n "$a"; done; \
                for a in *.tar *.tar.gz *.tgz; do [[ -e "$a" ]] && tar -xf "$a"; done) || true
  if [[ ! -d "$dst/parsed_documents/dev" && -f "$dst/load.py" ]]; then
    (cd "$LILAC_ROOT" && in_env lilac-base python src/lilac/lcg_constructor/preprocessing/hf_mmcoqa_to_lilac.py \
        --hf_dir "datasets/$DATASET" --out_doc_dir "datasets/$DATASET/parsed_documents/dev" \
        --out_img_dir "datasets/$DATASET/image_components/dev")
  fi
  (cd "$LILAC_ROOT" && in_env lilac-base bash scripts/validate_dataset.sh "$DATASET")
}

stage_vlm() {
  in_env vllm python -m mmrag.lilac.vlm_batch summarize detect caption-crops --dataset "$DATASET" --model "$VLM_MODEL"
}

stage_graph() {
  cd "$LILAC_ROOT"
  in_env lilac-base bash scripts/parse_multimodal_document/step1_dereference.sh -b "$DATASET"
  in_env lilac-base bash scripts/parse_multimodal_document/step2_add_table_segments.sh -b "$DATASET"
  in_env lilac-base python -m mmrag.lilac.official_steps step3 --dataset "$DATASET"
  # detections/crops/captions already exist (stage vlm) -> only register them
  in_env lilac-base python src/lilac/lcg_constructor/step5_add_subimages.py \
      --target_data "$DATASET" --analyzer qwen_bbox --mode add_to_parsed_documents
  in_env lilac-base bash scripts/parse_multimodal_document/step6_add_edges.sh -b "$DATASET"
  in_env lilac-base bash scripts/parse_multimodal_document/step7_serialize_components.sh -b "$DATASET"
}

stage_embed() {
  mmembed_meta check && mmembed_meta write
  local emb_dir
  emb_dir="$(in_env lilac-base python -c "from mmrag.lilac import official; print(official.embeddings_dir('$DATASET'))")"
  for comp in text sentence table table_segment image subimage; do
    if [[ -f "$emb_dir/$comp.pt" && -f "$emb_dir/$comp.json" ]]; then
      echo "embed: $comp already done - skip"; continue
    fi
    (cd "$LILAC_ROOT" && NUM_GPUS=1 in_mmembed python src/lilac/lcg_constructor/step8_embed_mmembed.py \
        -b "$DATASET" -c "$comp" --num_gpus 1)
  done
  mmembed_meta audit
}

stage_decompose_llm() {
  # sub-queries for every decomposer (vLLM env, one model at a time; fits a 16 GB T4 with AWQ models)
  for pair in $DECOMPOSERS; do
    local tag="${pair%%=*}" model="${pair#*=}"
    in_env vllm python -m mmrag.lilac.decompose --dataset "$DATASET" --model "$model" --tag "$tag" --no-activate
  done
}

stage_decompose_embed() {
  # MM-Embed sub-query embeddings per variant (official queryset embedder; same MM-Embed setting as 'embed')
  mmembed_meta check
  for pair in $DECOMPOSERS; do
    local tag="${pair%%=*}"
    in_env lilac-base python -m mmrag.lilac.decompose --dataset "$DATASET" --tag "$tag" --activate-only
    (cd "$LILAC_ROOT" && CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" in_mmembed \
        python src/lilac/retriever/query_decomposer/queryset_embedder_mmembed.py --target_data "$DATASET")
    in_env lilac-base python -m mmrag.lilac.decompose --dataset "$DATASET" --tag "$tag" --store-embeddings
  done
  in_env lilac-base python -m mmrag.lilac.decompose --dataset "$DATASET" --tag "$(main_tag)" --activate-only
}

stage_decompose() { stage_decompose_llm; stage_decompose_embed; }

stage_novlm() {
  mmembed_meta check
  in_env lilac-base python -m mmrag.lilac.serialize_novlm --dataset "$DATASET"
  in_mmembed python -m mmrag.lilac.official_steps embed-extra --dataset "$DATASET" \
      --in-name "image+image" --out-name image_novlm --max-length 256 --batch-size 8
  in_mmembed python -m mmrag.lilac.official_steps embed-extra --dataset "$DATASET" \
      --in-name "subimage+image" --out-name subimage_novlm --max-length 256 --batch-size 8
  mmembed_meta audit
}

stage_experiments() {
  # shellcheck disable=SC2086
  in_env lilac-base python -m mmrag.lilac.run_experiments --dataset "$DATASET" \
      --groups $GROUPS --main-decomposer "$(main_tag)" --decomposers $(all_tags) \
      --project "${WANDB_PROJECT:-lilac-repro}"
}

for stage in "$@"; do
  echo "=================== LILaC stage: $stage ==================="
  "stage_$stage"
done
