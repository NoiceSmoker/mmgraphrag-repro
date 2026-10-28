"""Stage 'contexts' (audio env): retrieval + GRASP grounding for every config.

For each question and configuration we store the candidate triples (after
retrieval / expansion) and their audio-presence scores. Grounding scores are
computed once per (question, triple) for the union of all candidates and
then thresholded per configuration, so threshold sweeps cost nothing.

Output: ``<work>/contexts/<config>.pre.jsonl`` (LLM filtering happens in
the next stage, ``filter``, in the vLLM environment).
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from mmrag.common.cache import KVCache
from mmrag.common.io import read_json, write_json, write_jsonl
from mmrag.m3kg.config import WorkDir, load_config
from mmrag.m3kg.encoders import ClapEncoder, make_grounder
from mmrag.m3kg.experiments import M3Config, all_configs
from mmrag.m3kg.retrieval import KGIndex, shared_space_retrieve


def default_eta(cfg: dict, backend: str) -> float:
    g = cfg["grasp"]
    return g["eta_a"] if backend == "tag" else g["eta_a_clap_window"]


def candidates_for(c: M3Config, q: np.ndarray, idx: KGIndex, taus: dict, hubs: set,
                   tt_emb: np.ndarray, tt_ids: list[int]) -> tuple[list[int], list[tuple[str, float]]]:
    if c.retrieval == "none":
        return [], []
    if c.retrieval == "shared":
        return shared_space_retrieve(q, tt_emb, tt_ids, c.shared_top_m), []
    clips = idx.nearest_clips(q, c.k, taus[c.tau])
    if c.retrieval == "captions":
        return [], clips
    seeds = idx.lift(clips)
    return idx.expand(q, seeds, c.hops, c.max_triplets, hubs), clips


def run(cfg: dict, groups: list[str] | None) -> None:
    wd = WorkDir(cfg)
    ix = wd.index_dir
    kg = pd.read_parquet(wd.kg_dir / "triplets.parquet")
    test = pd.read_parquet(wd.test_dir / "meta.parquet").set_index("qid")
    corpus_ids = read_json(ix / "corpus_ids.json")
    kg = kg[kg.clip_id.isin(set(corpus_ids))].reset_index(drop=True)  # drop leakage-filtered clips
    idx = KGIndex(kg, corpus_ids, np.load(ix / "corpus_emb.npy"))
    q_ids = read_json(ix / "test_ids.json")
    q_emb = np.load(ix / "test_emb.npy")
    tt_emb, tt_ids = np.load(ix / "triple_text_emb.npy"), read_json(ix / "triple_tids.json")
    captions = pd.read_parquet(wd.corpus_dir / "meta.parquet").set_index("clip_id")["caption"].to_dict()
    taus = read_json(ix / "taus.json")
    taus = {k: (float("inf") if v is None or v == "Infinity" else float(v)) for k, v in taus.items()}
    hubs = idx.hub_concepts(cfg["retrieval"].get("hub_max_degree"), cfg["retrieval"].get("hub_top_frac", 0.01))
    configs = all_configs(groups, tuple(cfg["grasp"]["eta_grid"]) if cfg["grasp"].get("eta_grid") else None)

    # 1) candidates for every (config, question)
    t0 = time.time()
    cand: dict[str, dict[str, dict]] = {}
    for c in configs:
        cand[c.name] = {}
        for qi, qid in enumerate(q_ids):
            tids, clips = candidates_for(c, q_emb[qi], idx, taus, hubs, tt_emb, tt_ids)
            cand[c.name][qid] = {"tids": tids, "clips": clips}
    t_retr = (time.time() - t0) / max(len(configs) * len(q_ids), 1) * 1000

    # 2) grounding scores for the union of candidates (cached across runs)
    clap = ClapEncoder(cfg["retrieval"]["clap_model"])
    grounder = make_grounder(cfg, clap)
    gcache = KVCache(cfg["paths"]["cache_db"], table=f"ground_{grounder.name}")
    union: dict[str, set[int]] = {q: set() for q in q_ids}
    for c in configs:
        if c.grounding:
            for q in q_ids:
                union[q].update(cand[c.name][q]["tids"])
    scores: dict[str, dict[int, float]] = {}
    t0, n_scored = time.time(), 0
    for q in q_ids:
        tids = sorted(union[q])
        keys = [f"{q}:{t}" for t in tids]
        have = gcache.get_many(keys)
        todo = [t for t, k in zip(tids, keys) if k not in have]
        if todo:
            vals = grounder.score(test.loc[q, "path"], idx.triple_texts(todo))
            new = {f"{q}:{t}": float(v) for t, v in zip(todo, vals)}
            gcache.set_many(new)
            have.update(new)
            n_scored += len(todo)
        scores[q] = {t: have[f"{q}:{t}"] for t in tids}
    t_ground = (time.time() - t0) * 1000 / max(len(q_ids), 1)

    # 3) apply grounding thresholds and write pre-filter contexts
    wd.contexts_dir.mkdir(parents=True, exist_ok=True)
    for c in configs:
        eta = c.eta if c.eta is not None else default_eta(cfg, grounder.name)
        rows = []
        for q in q_ids:
            tids = cand[c.name][q]["tids"]
            kept = [t for t in tids if scores[q].get(t, -1e9) >= eta] if c.grounding else list(tids)
            extra = ({"captions": [captions[cid] for cid, _ in cand[c.name][q]["clips"] if cid in captions]}
                     if c.retrieval == "captions" else {})
            rows.append({"qid": q, "candidates": tids, "after_grounding": kept, **extra,
                         "scores": {str(t): round(scores[q][t], 4) for t in tids if t in scores[q]},
                         "clips": cand[c.name][q]["clips"], "needs_llm_filter": c.llm_filter and bool(kept)})
        write_jsonl(rows, wd.contexts_dir / f"{c.name}.pre.jsonl")
        write_json({**c.as_config(), "eta_effective": eta, "grounding_backend": grounder.name,
                    "tau_value": taus.get(c.tau)}, wd.contexts_dir / f"{c.name}.config.json")
    write_json({"retrieval_ms_per_query_config": round(t_retr, 3), "grounding_ms_per_query": round(t_ground, 3),
                "n_grounding_pairs_scored": n_scored, "grounding_backend": grounder.name,
                "n_hubs": len(hubs), "hubs_sample": sorted(hubs)[:20]}, wd.contexts_dir / "_timing.json")
    print(f"[contexts] {len(configs)} configs x {len(q_ids)} questions; grounding backend={grounder.name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/m3kg/base.yaml")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--groups", nargs="*", default=None)
    a = ap.parse_args()
    run(load_config(a.config, a.profile, a.set), a.groups)


if __name__ == "__main__":
    main()
