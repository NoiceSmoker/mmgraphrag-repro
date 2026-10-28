"""vLLM stages of the M3KG-RAG pipeline (one model per process):

  filter    GRASP step 2: LLM keep/drop of grounded triples      (Qwen3, text)
  generate  graph-augmented answering with the audio MLLM        (Qwen2.5-Omni thinker)
  judge     AudioBench Model-as-Judge score 0-5 (x20)              (Llama-3 judge)
  winrate   optional reference-aware pairwise preference (paper Table 2 / Table 14)

    python -m mmrag.m3kg.vllm_stages filter   --config ... --profile ...
    python -m mmrag.m3kg.vllm_stages generate --config ... --profile ...
    python -m mmrag.m3kg.vllm_stages judge    --config ... --profile ...
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import pandas as pd

from mmrag.common.cache import KVCache
from mmrag.common.io import read_json, read_jsonl, stable_hash, write_json, write_jsonl
from mmrag.m3kg import prompts as P
from mmrag.m3kg.agents import Agents
from mmrag.m3kg.config import WorkDir, load_config
from mmrag.m3kg.llm import CachedLLM, ChatRequest, VLLMBackend

OMNI_SYSTEM = ("You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of "
               "perceiving auditory and visual inputs, as well as generating text and speech.")


def _configs(wd: WorkDir) -> list[str]:
    return sorted(p.name[: -len(".pre.jsonl")] for p in wd.contexts_dir.glob("*.pre.jsonl"))


def _triples(kg: pd.DataFrame, tids: list[int]) -> list[dict]:
    rows = kg.set_index("tid").loc[tids] if tids else kg.iloc[0:0]
    return [{"h": r["head"], "r": r["rel"], "t": r["tail"], "hd": r["head_desc"] or "-", "td": r["tail_desc"] or "-"}
            for _, r in rows.iterrows()]


# ------------------------------------------------------------------ filter --
def stage_filter(cfg: dict) -> None:
    wd = WorkDir(cfg)
    kg = pd.read_parquet(wd.kg_dir / "triplets.parquet")
    test = pd.read_parquet(wd.test_dir / "meta.parquet").set_index("qid")
    g = cfg["grasp"]
    llm = CachedLLM(VLLMBackend(g["filter_llm"], max_model_len=g.get("filter_max_model_len", 8192),
                                gpu_memory_utilization=g.get("gpu_memory_utilization", 0.85)),
                    KVCache(cfg["paths"]["cache_db"]))
    agents = Agents(llm)
    work: list[tuple[str, dict]] = []
    rows_by_cfg = {}
    for name in _configs(wd):
        rows = read_jsonl(wd.contexts_dir / f"{name}.pre.jsonl")
        rows_by_cfg[name] = rows
        work += [(name, r) for r in rows if r["needs_llm_filter"]]
    items = [(test.loc[r["qid"], "question"], [(t["h"], t["r"], t["t"]) for t in _triples(kg, r["after_grounding"])])
             for _, r in work]
    kept = agents.grasp_filter(items)
    for (name, r), k in zip(work, kept):
        r["final"] = [r["after_grounding"][i] for i in k]
    for name, rows in rows_by_cfg.items():
        for r in rows:
            r.setdefault("final", r["after_grounding"])
        write_jsonl(rows, wd.contexts_dir / f"{name}.jsonl")
    print(f"[filter] {len(work)} filter calls ({llm.n_calls} new, {llm.n_cached} cached)")


# ---------------------------------------------------------------- generate --
def build_text(question: str, triples: list[dict]) -> str:
    if not triples:
        return question
    lines = "\n".join(P.TRIPLE_LINE.format(i=i, **t) for i, t in enumerate(triples))
    return P.RAG_INSTRUCTION.format(question=question, triples=lines)


def build_caption_text(question: str, captions: list[str]) -> str:
    if not captions:
        return question
    lines = "\n".join(f"[{i}] {c}" for i, c in enumerate(captions))
    return P.CAPTION_INSTRUCTION.format(question=question, captions=lines)


def stage_generate(cfg: dict) -> None:
    from vllm import LLM, SamplingParams

    from mmrag.m3kg.audio import load_audio

    wd = WorkDir(cfg)
    gcfg = cfg["generation"]
    kg = pd.read_parquet(wd.kg_dir / "triplets.parquet")
    test = pd.read_parquet(wd.test_dir / "meta.parquet").set_index("qid")
    cache = KVCache(cfg["paths"]["cache_db"], table="generation")
    plan = []  # (config, qid, text, cache_key)
    for name in _configs(wd):
        path = wd.contexts_dir / f"{name}.jsonl"
        rows = read_jsonl(path) if path.exists() else read_jsonl(wd.contexts_dir / f"{name}.pre.jsonl")
        for r in rows:
            question = test.loc[r["qid"], "question"]
            if "captions" in r:  # neighbour-caption baseline: no KG, context size = number of captions
                text, n_ctx = build_caption_text(question, r["captions"]), len(r["captions"])
            else:
                tids = r.get("final", r["after_grounding"])
                text, n_ctx = build_text(question, _triples(kg, tids)), len(tids)
            plan.append((name, r["qid"], text, "gen:" + stable_hash([gcfg["model"], r["qid"], text], 24), n_ctx))
    have = cache.get_many([p[3] for p in plan])
    todo = {}
    for p in plan:
        if p[3] not in have and p[3] not in todo:
            todo[p[3]] = p
    print(f"[generate] {len(plan)} (config, question) pairs, {len(todo)} unique prompts to run")
    if todo:
        llm = LLM(model=gcfg["model"], max_model_len=gcfg.get("max_model_len", 8192),
                  max_num_seqs=gcfg.get("max_num_seqs", 16), limit_mm_per_prompt={"audio": 1},
                  gpu_memory_utilization=gcfg.get("gpu_memory_utilization", 0.9), seed=0)
        sp = SamplingParams(temperature=0.0, max_tokens=gcfg.get("max_tokens", 256))
        audio = {q: load_audio(test.loc[q, "path"], 16000, 30.0) for q in {p[1] for p in todo.values()}}
        keys = list(todo)
        for i in range(0, len(keys), 256):
            chunk = keys[i : i + 256]
            reqs = [{"prompt": (f"<|im_start|>system\n{OMNI_SYSTEM}<|im_end|>\n<|im_start|>user\n"
                                f"<|audio_bos|><|AUDIO|><|audio_eos|>{todo[k][2]}<|im_end|>\n<|im_start|>assistant\n"),
                     "multi_modal_data": {"audio": (audio[todo[k][1]], 16000)}} for k in chunk]
            outs = llm.generate(reqs, sampling_params=sp, use_tqdm=True)
            cache.set_many({k: o.outputs[0].text.strip() for k, o in zip(chunk, outs)})
        have = cache.get_many([p[3] for p in plan])
    wd.preds_dir.mkdir(parents=True, exist_ok=True)
    by_cfg: dict[str, list] = {}
    for name, qid, text, key, n_t in plan:
        by_cfg.setdefault(name, []).append({"qid": qid, "prediction": have[key], "n_triples": n_t,
                                            "prompt_chars": len(text)})
    for name, rows in by_cfg.items():
        write_jsonl(rows, wd.preds_dir / f"{name}.jsonl")


# ------------------------------------------------------------------- judge --
def audiobench_template(cfg: dict) -> tuple[str, str]:
    """Load AudioBench's Llama-3 judge prompt from the cloned repo so the
    judging protocol is byte-identical to the benchmark's."""
    path = Path(cfg["judge"]["audiobench_dir"]) / "src" / "dataset_src" / "eval_methods" / "eval_llama3_70b.py"
    src = path.read_text(encoding="utf-8")
    m = re.search(r'PROMPT_TEMPLATE = """\\\n(.*?)"""', src, flags=re.S)
    if not m:
        raise RuntimeError(f"could not find the judge template in {path}")
    return m.group(1), str(path)


def parse_judge(output: str) -> tuple[float, str]:
    """AudioBench parses ``float(output.split()[-1])`` and scores 0 on
    failure; we keep that rule and add a 'Rating: n' regex fallback."""
    try:
        return float(output.split()[-1]), "audiobench"
    except (ValueError, IndexError):
        m = re.search(r"Rating:\s*\(?\s*([0-5](?:\.\d+)?)", output or "")
        if m:
            return float(m.group(1)), "regex"
        return 0.0, "failed"


def stage_judge(cfg: dict) -> None:
    wd = WorkDir(cfg)
    j = cfg["judge"]
    template, src = audiobench_template(cfg)
    test = pd.read_parquet(wd.test_dir / "meta.parquet").set_index("qid")
    llm = CachedLLM(VLLMBackend(j["model"], max_model_len=j.get("max_model_len", 4096),
                                gpu_memory_utilization=j.get("gpu_memory_utilization", 0.88),
                                quantization=j.get("quantization")),
                    KVCache(cfg["paths"]["cache_db"], table="judge"))
    files = sorted(wd.preds_dir.glob("*.jsonl"))
    reqs, index = [], []
    for f in files:
        for r in read_jsonl(f):
            prompt = template.format(question=test.loc[r["qid"], "question"], reference=test.loc[r["qid"], "answer"],
                                     prediction=r["prediction"])
            reqs.append(ChatRequest("", prompt, max_tokens=j.get("max_tokens", 512), temperature=0.0))
            index.append((f.stem, r))
    outs = llm.complete(reqs)
    wd.judge_dir.mkdir(parents=True, exist_ok=True)
    by_cfg: dict[str, list] = {}
    for (name, r), o in zip(index, outs):
        score, mode = parse_judge(o)
        by_cfg.setdefault(name, []).append({**r, "judge_raw": o[-400:], "score_0_5": score, "parse": mode})
    for name, rows in by_cfg.items():
        write_jsonl(rows, wd.judge_dir / f"{name}.jsonl")
    write_json({"judge_model": j["model"], "template_source": src}, wd.judge_dir / "_judge.json")
    print(f"[judge] {len(reqs)} judgements ({llm.n_calls} new)")


# ----------------------------------------------------------------- winrate --
WINRATE_PROMPT = """Compare two answers to the same question about an audio clip, using the reference answer as ground truth.
Judge only by agreement with the reference and the question; do not reward length or speculation.

Question: {question}
Reference answer: {reference}
Answer 1: {a1}
Answer 2: {a2}

For each criterion choose the better answer:
- Comprehensiveness: covers more of the reference's essential points correctly, with fewer errors or omissions.
- Diversity: organises the reference's facts in a more varied way (e.g. different acoustic aspects) without inventing new attributes.
- Empowerment: helps the reader understand the clip more clearly and concisely.
- Overall: most faithful to the reference, with better coverage and clearer presentation (ties: faithfulness, then coverage, then concision).
Return JSON only: {{"Comprehensiveness": 1 or 2, "Diversity": 1 or 2, "Empowerment": 1 or 2, "Overall": 1 or 2}}"""
# NOTE: paraphrase of the paper's Table 14; answer order is randomised per question.

CRITERIA = ("Comprehensiveness", "Diversity", "Empowerment", "Overall")


def parse_winrate(text: str) -> dict[str, int] | None:
    m = re.search(r"\{.*\}", text or "", flags=re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group())
    except json.JSONDecodeError:
        return None
    out = {}
    for c in CRITERIA:
        v = d.get(c)
        if isinstance(v, dict):
            v = v.get("Winner")
        v = str(v)
        if "1" in v:
            out[c] = 1
        elif "2" in v:
            out[c] = 2
    return out if len(out) == len(CRITERIA) else None


def stage_winrate(cfg: dict, pairs: list[str]) -> None:
    wd = WorkDir(cfg)
    j = cfg["judge"]
    test = pd.read_parquet(wd.test_dir / "meta.parquet").set_index("qid")
    llm = CachedLLM(VLLMBackend(j["model"], max_model_len=j.get("max_model_len", 4096),
                                gpu_memory_utilization=j.get("gpu_memory_utilization", 0.88),
                                quantization=j.get("quantization")),
                    KVCache(cfg["paths"]["cache_db"], table="winrate"))
    results = {}
    for pair in pairs:
        ours, base = pair.split(":")
        A = {r["qid"]: r["prediction"] for r in read_jsonl(wd.preds_dir / f"{ours}.jsonl")}
        B = {r["qid"]: r["prediction"] for r in read_jsonl(wd.preds_dir / f"{base}.jsonl")}
        qids = sorted(set(A) & set(B))
        rng = random.Random(0)
        flips = {q: rng.random() < 0.5 for q in qids}
        reqs = []
        for q in qids:
            a1, a2 = (B[q], A[q]) if flips[q] else (A[q], B[q])
            reqs.append(ChatRequest("", WINRATE_PROMPT.format(question=test.loc[q, "question"],
                                                              reference=test.loc[q, "answer"], a1=a1, a2=a2),
                                    max_tokens=96))
        wins = {c: [] for c in CRITERIA}
        n_bad = 0
        for q, o in zip(qids, llm.complete(reqs)):
            d = parse_winrate(o)
            if d is None:
                n_bad += 1
                continue
            for c in CRITERIA:
                ours_slot = 2 if flips[q] else 1
                wins[c].append(1.0 if d[c] == ours_slot else 0.0)
        results[pair] = {c: round(100 * sum(v) / max(len(v), 1), 2) for c, v in wins.items()}
        results[pair]["n"] = len(qids)
        results[pair]["parse_failures"] = n_bad
    write_json(results, wd.judge_dir / "winrate.json")
    print(json.dumps(results, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["filter", "generate", "judge", "winrate"])
    ap.add_argument("--config", default="configs/m3kg/base.yaml")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--pairs", nargs="*", default=["m3kg_rag:no_rag", "m3kg_rag:shared_space", "m3kg_rag:caption_rag"],
                    help="winrate pairs 'ours:baseline'")
    a = ap.parse_args()
    cfg = load_config(a.config, a.profile, a.set)
    {"filter": stage_filter, "generate": stage_generate, "judge": stage_judge}.get(
        a.stage, lambda c: stage_winrate(c, a.pairs))(cfg)


if __name__ == "__main__":
    main()
