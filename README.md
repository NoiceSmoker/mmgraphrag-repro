# mmgraphrag-repro

Code to re-run the experiments of the course report on two papers:

* **LILaC** — *Late Interacting in Layered Component Graph for Open-domain Multimodal Multihop Retrieval* (Yun et al.), reproduced on **MMCoQA** with the official code at a pinned commit.
* **M3KG-RAG** — *Multi-hop Multimodal Knowledge Graph-enhanced Retrieval-Augmented Generation* (Park et al., CVPR 2026), re-implemented at reduced scale on **AudioCaps-QA** (no official code is available).

This README only explains how to run things. Method descriptions, design decisions and results are in the report.

---

## 1. Hardware

| Stage | GPU memory | Free Colab (T4 16 GB) |
|---|---|---|
| LILaC `vlm` (Qwen2.5-VL-7B, vLLM) | ≥ 20 GB | yes, with `VLM_MODEL=Qwen/Qwen2.5-VL-7B-Instruct-AWQ` |
| LILaC `embed`, `decompose_embed`, `novlm` (MM-Embed 8B, fp16) | ≥ 20 GB, ~17 GB system RAM | yes, with `MMEMBED_QUANT=8bit` (~9 GB) |
| LILaC `decompose_llm` (Qwen2.5-7B / 14B-AWQ / 32B-AWQ) | 16 / 12 / 24 GB | yes, 7B-AWQ and 14B-AWQ |
| LILaC `data`, `graph`, `experiments` | none / any (CPU) | yes |
| M3KG `kg`, `filter` (Qwen3-8B) | ≥ 20 GB | yes, `colab_t4` profile (Qwen3-8B-AWQ) |
| M3KG `generate` (Qwen2.5-Omni-7B) | ≥ 22 GB | yes, `colab_t4` profile (Qwen2.5-Omni-3B) |
| M3KG `judge` (Llama-3-8B-Instruct AWQ) | ≥ 8 GB | yes |
| M3KG `data`, `index`, `contexts` (CLAP, TAG) | ≥ 4 GB (or CPU) | yes |

Recommended: **Colab Pro with an L4 (24 GB) or A100 (40 GB) runtime** for GPU stages. On GPUs without bfloat16
(T4 and older) every vLLM stage switches to fp16 automatically; set `VLLM_DTYPE` to override. The notebooks detect a
T4 and select the T4 settings themselves. Free-tier sessions may be interrupted: all stages are resumable.
On a T4, MM-Embed is loaded with bitsandbytes weights (`MMEMBED_QUANT=8bit`, or `4bit`); use the same setting for
all three MM-Embed stages (the pipeline checks this). A local GPU with 8 GB
(e.g. RTX 5050) can run all CPU stages, the CLAP/TAG stages and — with `configs/m3kg/profiles/local_8gb.yaml` —
a smaller version of the M3KG pipeline. RTX 50-series (Blackwell) cards need PyTorch wheels built for CUDA ≥ 12.8,
which is the default index used by `scripts/setup_envs.sh`.

Disk: model weights dominate — about 80 GB for all LILaC models (Qwen2.5-VL-7B, MM-Embed, three decomposers) and
about 45 GB for M3KG-RAG (Qwen3-8B, Qwen2.5-Omni-7B, Llama-3-8B-AWQ, CLAP, TAG); data and embeddings add a few GB.
On Colab, delete the Hugging Face cache of a model once its stage is finished if the disk gets full.

## 2. Setup

```bash
git clone https://github.com/NoiceSmoker/mmgraphrag-repro.git && cd mmgraphrag-repro
bash scripts/setup_envs.sh all          # or a subset, see below
export WANDB_API_KEY=...                 # or `wandb login`; never commit keys
export WANDB_ENTITY=...                  # optional: default = the entity of the API key
export HF_TOKEN=<token>                  # only needed for gated models / higher rate limits
```

`setup_envs.sh` creates isolated [uv](https://github.com/astral-sh/uv) virtual environments under `$ENV_ROOT`
(`/content/envs` on Colab, `./.envs` elsewhere):

| env | used for | key pins |
|---|---|---|
| `lilac-base` | official LILaC graph steps, retrieval experiments | wtpsplit 2.2.1 |
| `mmembed` | MM-Embed encoding | transformers 4.46.3 (as in the official env) |
| `vllm` | all vLLM-served models | `VLLM_SPEC` (default: latest `vllm`) |
| `m3kg-audio` | CLAP, TAG grounding, data preparation | transformers < 5 |

Pipeline scripts select the right environment for every stage automatically.

**Colab:** open `notebooks/lilac_colab.ipynb` or `notebooks/m3kg_colab.ipynb`; they run the same commands as
below cell by cell and use `scripts/colab/sync.sh push|pull lilac|m3kg` to keep artefacts on Google Drive between
sessions. Both notebooks can be started with *Run all*: a failing command stops the notebook, and re-running skips finished work.

## 3. LILaC (MMCoQA)

```bash
bash scripts/lilac/pipeline.sh fetch        # official code -> third_party/lilac (commit 813b6c0)
bash scripts/lilac/pipeline.sh data         # MMCoQA documents from the Hugging Face release + validation
bash scripts/lilac/pipeline.sh vlm          # Qwen2.5-VL summaries, detections, crop captions (vLLM)
bash scripts/lilac/pipeline.sh graph        # official steps 1, 2, 3, 5 (register crops), 6, 7
bash scripts/lilac/pipeline.sh embed        # MM-Embed embeddings of all nodes (official step 8)
bash scripts/lilac/pipeline.sh decompose    # = decompose_llm (sub-queries per decomposer) + decompose_embed (MM-Embed)
bash scripts/lilac/pipeline.sh novlm        # optional: embeddings for the "no VLM text" ablation
bash scripts/lilac/pipeline.sh experiments  # all retrieval runs + metrics -> W&B project `lilac-repro`
```

Environment variables of `pipeline.sh`:

| variable | default |
|---|---|
| `DATASET` | `MMCoQA` (`MultimodalQA` also works; much larger) |
| `DECOMPOSERS` | `qwen2.5-7b=Qwen/Qwen2.5-7B-Instruct qwen2.5-14b-awq=Qwen/Qwen2.5-14B-Instruct-AWQ qwen2.5-32b-awq=Qwen/Qwen2.5-32B-Instruct-AWQ` (first = main) |
| `GROUPS` | `main ablation sens_beam sens_iter decomposer` |
| `VLM_MODEL` | `Qwen/Qwen2.5-VL-7B-Instruct` (official); `Qwen/Qwen2.5-VL-7B-Instruct-AWQ` on a T4 |
| `MMEMBED_QUANT` | unset = official fp16; `8bit` / `4bit` = bitsandbytes weights for 16 GB GPUs |
| `MMEMBED_BATCH_SCALE` | `1` (factor on the official embedding batch sizes; halved automatically on OOM) |
| `WANDB_PROJECT` | `lilac-repro` |

Outputs: official artefacts in `third_party/lilac/artifacts/<DATASET>/`, retrieval runs in
`third_party/lilac/algorithm_results/LILaC/<DATASET>/retrieval/`, metrics in `results/lilac-repro/<group>/`.

The `experiments` stage needs no GPU. To run it on another machine, copy `third_party/lilac/{datasets,artifacts}/<DATASET>`
there (e.g. with `scripts/colab/sync.sh pull`) and run `bash scripts/setup_envs.sh lilac-base` first.
Single variants: `python -m mmrag.lilac.run_experiments --dataset MMCoQA --groups ablation --only full wo_qd`.

## 4. M3KG-RAG (AudioCaps-QA, scaled down)

```bash
bash scripts/m3kg/pipeline.sh data        # AudioCaps-QA test set + AudioCaps train subset (corpus)
bash scripts/m3kg/pipeline.sh youtube     # optional: YouTube metadata for the rewriter agent
bash scripts/m3kg/pipeline.sh kg          # multi-agent KG construction (needs internet: Wikipedia / Wiktionary)
bash scripts/m3kg/pipeline.sh index       # CLAP embeddings, leakage check, distance thresholds
bash scripts/m3kg/pipeline.sh contexts    # retrieval + grounding for every configuration
bash scripts/m3kg/pipeline.sh filter      # LLM pruning
bash scripts/m3kg/pipeline.sh generate    # Qwen2.5-Omni answers
bash scripts/m3kg/pipeline.sh judge       # AudioBench Model-as-Judge (clones AudioBench for the judge prompt)
bash scripts/m3kg/pipeline.sh winrate     # optional pairwise judge
bash scripts/m3kg/pipeline.sh report      # metrics + bootstrap intervals -> W&B project `m3kg-rag-repro`
```

Configuration: `configs/m3kg/base.yaml` + a hardware profile (`PROFILE=configs/m3kg/profiles/{colab_l4,colab_a100,colab_t4,local_8gb}.yaml`).
Any value can be overridden, e.g. `SET='data.corpus_size=1000 data.test_limit=100' bash scripts/m3kg/pipeline.sh data`.
`WORK_DIR` relocates all artefacts (default `work/m3kg`). Set `WIKI_USER_AGENT` to a string containing your
contact address (Wikimedia API policy).

Every LLM / MLLM / judge / Wikipedia call is cached in `work/m3kg/cache.sqlite`; re-running a stage only computes
what is missing, so interrupted Colab sessions can be resumed by re-running the same command.

## 5. W&B links and tables for the report

```bash
python scripts/make_wandb_reports.py --share   # one W&B Report per experiment section (--entity optional)
python scripts/export_results.py --results results --out results/latex  # LaTeX rows / pgfplots data
```

The first command writes `results/wandb_links.json`; the second turns `results/**.json` and those links into the
files `\input` by the report. Make both W&B projects public (*Project → Settings → Visibility*) or use the
share links printed by `--share`.

## 6. Tests

```bash
pip install -r requirements/dev.txt
bash scripts/lilac/pipeline.sh fetch   # some tests run the official LILaC evaluator/retriever
git clone --depth 1 https://github.com/AudioLLMs/AudioBench.git third_party/AudioBench   # optional
PYTHONPATH=src pytest
```

Tests use no models and no network (CPU, < 1 min).

## 7. Repository layout

```
configs/            M3KG-RAG config + hardware profiles
notebooks/          Colab drivers for both pipelines
requirements/       one requirements file per environment
scripts/            setup_envs.sh, pipelines, W&B report builder, LaTeX exporter, Drive sync
src/mmrag/common/   caching, tracking (W&B + local JSON), bootstrap statistics
src/mmrag/lilac/    orchestration of the official LILaC code, batched VLM passes, experiments, metrics
src/mmrag/m3kg/     re-implementation of M3KG-RAG (agents, KG, retrieval, GRASP, generation, judging)
tests/              unit and integration tests
```

## 8. Licenses of third-party components

Code in this repository: MIT. Models and data keep their own licenses — notably MM-Embed is CC-BY-NC-4.0,
Llama-3 follows the Llama 3 license, AudioCaps audio comes from YouTube. Use them for research only.
