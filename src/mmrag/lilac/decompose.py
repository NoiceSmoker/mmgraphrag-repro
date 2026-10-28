"""Batched query decomposition + modality estimation (paper Sec. 4.2.1).

The official scripts send one prompt at a time to vLLM and hard-code
``max_model_len=30000``. This module reuses the *official prompt strings and
output parsers* (imported from the pinned repo) but

  * sends all prompts of a stage in one vLLM batch;
  * uses a short context (prompts are < 1k tokens), which lets a 32B AWQ
    model fit on a single 24 GB GPU;
  * writes exactly the files the official retriever expects
    (``subqueries.json`` and ``subqueries_with_modality.json``);
  * stores every run as a named *decomposition variant* so that several
    decomposer LLMs can be compared without recomputing corpus embeddings;
  * separately measures sequential (one-question-at-a-time) latency on a
    small sample, which is the quantity reported in the paper.

Usage (inside the vLLM environment):
    python -m mmrag.lilac.decompose --dataset MMCoQA --model Qwen/Qwen2.5-7B-Instruct --tag qwen2.5-7b
"""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

from mmrag.common.io import read_json, write_json
from mmrag.lilac import official

SYSTEM_PROMPT = "You are a helpful assistant."  # official Qwen2_5_72B wrapper default


def _official_prompts():
    official.ensure_importable()
    # Importing these modules pulls in vLLM via the official wrapper; we only
    # need their module-level constants and pure parsing helpers.
    import importlib

    qd = importlib.import_module("src.lilac.retriever.query_decomposer.query_decomposer")
    me = importlib.import_module("src.lilac.retriever.query_decomposer.modality_estimator")
    return qd.PROMPT, me.PROMPT, qd.QueryDecomposer.preprocess_and_parse, me.ModalityEstimator._parse_modality


def load_questions(dataset: str) -> dict[str, str]:
    qa = read_json(official.labeled_qa_path(dataset))
    return {q["qid"]: q["question"] for q in qa}


class VLLMChat:
    def __init__(self, model: str, max_model_len: int = 4096, gpu_memory_utilization: float = 0.88,
                 quantization: str | None = None):
        from transformers import AutoTokenizer
        from vllm import LLM

        self.tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        kwargs = dict(model=model, max_model_len=max_model_len,
                      gpu_memory_utilization=gpu_memory_utilization, trust_remote_code=True)
        if quantization:
            kwargs["quantization"] = quantization
        self.llm = LLM(**kwargs)

    def _prompt(self, user: str) -> str:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def generate(self, user_prompts: list[str], max_tokens: int = 256) -> list[str]:
        from vllm import SamplingParams

        # Same decoding parameters as the official wrapper.
        sp = SamplingParams(temperature=0.0, top_p=1.0, repetition_penalty=1.05, max_tokens=max_tokens)
        outs = self.llm.generate([self._prompt(p) for p in user_prompts], sampling_params=sp, use_tqdm=True)
        return [o.outputs[0].text for o in outs]


def run(dataset: str, model: str, tag: str, quantization: str | None, latency_probe: int,
        activate: bool) -> Path:
    dec_prompt, mod_prompt, parse_subqueries, parse_modality = _official_prompts()
    questions = load_questions(dataset)
    qids = list(questions)
    chat = VLLMChat(model, quantization=quantization)

    # -------- stage 1: decomposition (batched) ---------------------------
    t0 = time.time()
    raw = chat.generate([dec_prompt.format(question=questions[q]) for q in qids])
    t_dec = time.time() - t0
    decomposed: dict[str, dict] = {}
    n_fallback = 0
    for qid, txt in zip(qids, raw):
        subs = parse_subqueries(None, txt)
        if not isinstance(subs, list):
            subs = [str(subs)]
        subs = [s if isinstance(s, str) else str(s) for s in subs]
        if len(subs) == 1 and subs[0] == txt:
            n_fallback += 1  # official parser returned the raw text (unparseable JSON)
        decomposed[qid] = {
            "qid": qid,
            "question": questions[qid],
            "subqueries": [{"sqid": f"{qid}___{i + 1}", "subquery": s} for i, s in enumerate(subs)],
            "time": {"decomposition_time": t_dec * 1000 / max(len(qids), 1)},
        }

    # -------- stage 2: modality estimation (batched) ---------------------
    flat = [(qid, i, sq["subquery"]) for qid, blk in decomposed.items() for i, sq in enumerate(blk["subqueries"])]
    t1 = time.time()
    mods = chat.generate([mod_prompt.format(subquery=s) for _, _, s in flat])
    t_mod = time.time() - t1
    for (qid, i, _), m in zip(flat, mods):
        decomposed[qid]["subqueries"][i]["modality"] = parse_modality(None, m)
    for blk in decomposed.values():
        blk["time"]["modality_estimation"] = t_mod * 1000 / max(len(qids), 1)

    # -------- stage 3: sequential latency probe -------------------------
    probe = {}
    if latency_probe > 0:
        sample = qids[:latency_probe]
        lat_dec, lat_mod = [], []
        for q in sample:
            a = time.time()
            out = chat.generate([dec_prompt.format(question=questions[q])])[0]
            lat_dec.append((time.time() - a) * 1000)
            subs = parse_subqueries(None, out) or [questions[q]]
            a = time.time()
            chat.generate([mod_prompt.format(subquery=str(s)) for s in subs])
            lat_mod.append((time.time() - a) * 1000)
        probe = {"probe_n": len(sample),
                 "decomposition_ms_per_query": sum(lat_dec) / len(lat_dec),
                 "modality_ms_per_query": sum(lat_mod) / len(lat_mod)}

    # -------- persist as a variant (+ optionally activate) ---------------
    vdir = variant_dir(dataset, tag)
    vdir.mkdir(parents=True, exist_ok=True)
    plain = {q: {**b, "subqueries": [{k: v for k, v in s.items() if k != "modality"} for s in b["subqueries"]]}
             for q, b in decomposed.items()}
    write_json(plain, vdir / "subqueries.json", indent=1)
    write_json(decomposed, vdir / "subqueries_with_modality.json", indent=1)
    n_sub = [len(b["subqueries"]) for b in decomposed.values()]
    write_json({"model": model, "quantization": quantization, "tag": tag, "n_questions": len(qids),
                "batched_seconds_decomposition": t_dec, "batched_seconds_modality": t_mod,
                "parse_fallback_rate": n_fallback / max(len(qids), 1),
                "avg_subqueries": sum(n_sub) / max(len(n_sub), 1), **probe}, vdir / "meta.json")
    if activate:
        activate_decomposition(dataset, tag)
    print(f"[decompose] stored variant '{tag}' in {vdir}")
    return vdir


# ---------------------------------------------------------------------------
# Variant management (shared with the retrieval runner)
# ---------------------------------------------------------------------------
SUBQ_EMB_FILES = ("subqueries_modality_agnostic.pt", "subqueries_modality_agnostic.json",
                  "subqueries_modality_aware.pt", "subqueries_modality_aware.json")


def variant_dir(dataset: str, tag: str) -> Path:
    return official.artifact_dir(dataset) / "query_decomposition_variants" / tag


def emb_variant_dir(dataset: str, tag: str, embedder: str = official.DEFAULT_EMBEDDER) -> Path:
    return official.embeddings_dir(dataset, embedder) / "subquery_variants" / tag


def activate_decomposition(dataset: str, tag: str) -> None:
    """Copy a stored decomposition into the canonical location read by the
    official query embedder / retriever. Removes stale sub-query embeddings
    so that the official embedder recomputes them."""
    src = variant_dir(dataset, tag)
    dst = official.decomposition_dir(dataset)
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("subqueries.json", "subqueries_with_modality.json"):
        shutil.copy2(src / name, dst / name)
    emb_src = emb_variant_dir(dataset, tag)
    emb_dst = official.embeddings_dir(dataset)
    emb_dst.mkdir(parents=True, exist_ok=True)
    for name in SUBQ_EMB_FILES:
        target = emb_dst / name
        if (emb_src / name).exists():
            shutil.copy2(emb_src / name, target)
        elif target.exists():
            target.unlink()
    (dst / "ACTIVE_VARIANT").write_text(tag)


def store_subquery_embeddings(dataset: str, tag: str) -> None:
    """After the official query embedder has run for the active variant,
    keep a copy of its sub-query embeddings next to the variant."""
    src = official.embeddings_dir(dataset)
    dst = emb_variant_dir(dataset, tag)
    dst.mkdir(parents=True, exist_ok=True)
    for name in SUBQ_EMB_FILES:
        shutil.copy2(src / name, dst / name)


def active_variant(dataset: str) -> str | None:
    p = official.decomposition_dir(dataset) / "ACTIVE_VARIANT"
    return p.read_text().strip() if p.exists() else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="MMCoQA")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--tag", required=True, help="Name of the decomposition variant, e.g. qwen2.5-7b")
    ap.add_argument("--quantization", default=None, help="e.g. awq (vLLM usually auto-detects)")
    ap.add_argument("--latency-probe", type=int, default=50,
                    help="Questions decomposed one at a time to measure per-query latency (0 = skip)")
    ap.add_argument("--no-activate", action="store_true")
    sub = ap.add_mutually_exclusive_group()
    sub.add_argument("--store-embeddings", action="store_true",
                     help="Only copy the canonical sub-query embeddings into the variant folder")
    sub.add_argument("--activate-only", action="store_true", help="Only activate an existing variant")
    args = ap.parse_args()
    if args.store_embeddings:
        store_subquery_embeddings(args.dataset, args.tag)
    elif args.activate_only:
        activate_decomposition(args.dataset, args.tag)
    else:
        run(args.dataset, args.model, args.tag, args.quantization, args.latency_probe, not args.no_activate)


if __name__ == "__main__":
    main()
