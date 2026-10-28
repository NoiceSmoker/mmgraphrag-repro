"""Multi-agent construction of the multi-hop multimodal KG (paper Sec. 3.1).

Stages (all batched):
  1. rewriter      caption + YouTube title/description -> context-enriched caption (optional)
  2. extractor     caption -> (head, relation, tail) triples
  3. normalizer    entity mention -> canonical, searchable concept (one call per unique mention)
  4. searcher      concept -> candidate descriptions from Wikipedia / Wiktionary,
                   LLM callback for concepts without any hit (+ inspector loop)
  5. selector      choose the candidate matching the sense used in the caption
  6. refiner       adapt the chosen description to the original mention
  7. inspector     score 0-10; < threshold -> re-run the producing agent (sampling),
                   at most ``inspector_max_rounds`` rounds, then discard

Every triple keeps a link to its source clip, which gives the coverage
property of Eq. (2): each fact is reachable from at least one audio item.

Run inside the vLLM environment (needs internet for the KB look-ups):
    python -m mmrag.m3kg.build_kg --config configs/m3kg/base.yaml --profile configs/m3kg/profiles/colab_l4.yaml
"""
from __future__ import annotations

import argparse
import time
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from mmrag.common.cache import KVCache
from mmrag.common.io import write_json
from mmrag.common.tracking import tracked_run
from mmrag.m3kg.agents import Agents
from mmrag.m3kg.config import WorkDir, load_config
from mmrag.m3kg.kb import KnowledgeBase
from mmrag.m3kg.llm import CachedLLM


def _candidates(res: dict, max_candidates: int) -> list[tuple[str, str]]:
    out, seen = [], set()
    for src in ("wikipedia", "wiktionary"):
        for txt in res.get(src, []):
            key = txt.lower()[:200]
            if key not in seen:
                seen.add(key)
                out.append((src, txt))
    return out[:max_candidates]


def inspector_loop(agents: Agents, items: list[dict], produce, threshold: int, max_rounds: int) -> None:
    """Generic self-reflection loop.

    ``items`` are dicts with keys mention, caption, desc. ``produce(items, round)``
    must return new descriptions for the given items (round >= 1 -> sampling).
    Sets ``score``, ``rounds`` and, for items that never pass, ``desc = ""``.
    """
    pending = list(items)
    for rnd in range(max_rounds):
        if not pending:
            break
        if rnd > 0:
            new = produce(pending, rnd)
            for it, d in zip(pending, new):
                it["desc"] = d
        scores = agents.inspect([(it["mention"], it["caption"], it["desc"]) for it in pending])
        nxt = []
        for it, s in zip(pending, scores):
            it["score"], it["rounds"] = s, rnd + 1
            if s < threshold or not it["desc"]:
                nxt.append(it)
        pending = nxt
    for it in pending:  # still failing after the last round -> discard
        it["desc"], it["discarded"] = "", True


def build(cfg: dict, agents: Agents, kb: KnowledgeBase) -> dict:
    wd = WorkDir(cfg)
    k = cfg["kg"]
    thr, rounds = k["inspector_threshold"], k["inspector_max_rounds"]
    meta = pd.read_parquet(wd.corpus_dir / "meta.parquet")
    timings, t = {}, time.time()

    # 1) rewriter (optional, needs YouTube metadata) -----------------------
    captions = dict(zip(meta.clip_id, meta.caption))
    yt_path = wd.corpus_dir / "youtube_meta.parquet"
    n_rewritten = 0
    if k.get("rewriter") and yt_path.exists():
        yt = pd.read_parquet(yt_path).dropna(subset=["title"])
        items = [{"clip_id": r.clip_id, "title": r.title, "description": r.description,
                  "caption": captions[r.clip_id]} for r in yt.itertuples() if r.clip_id in captions]
        for it, new in zip(items, agents.rewrite(items)):
            n_rewritten += int(new.strip() != it["caption"].strip())
            captions[it["clip_id"]] = new
    elif k.get("rewriter"):
        print("[kg] no YouTube metadata found - building from the human captions only (rewriter skipped)")
    timings["rewrite_s"] = time.time() - t

    # 2) extractor ---------------------------------------------------------
    t = time.time()
    clip_ids = list(captions)
    triples = dict(zip(clip_ids, agents.extract([captions[c] for c in clip_ids])))
    timings["extract_s"] = time.time() - t

    # 3) normalizer (unique mentions) ---------------------------------------
    t = time.time()
    mentions = sorted({x for ts in triples.values() for (h, _, tl) in ts for x in (h, tl)})
    concept_of = dict(zip(mentions, agents.normalize(mentions)))
    timings["normalize_s"] = time.time() - t

    # 4) searcher: KB look-up + LLM callback --------------------------------
    t = time.time()
    concepts = sorted(set(concept_of.values()))
    kb_res = kb.lookup_many(concepts)
    cands = {c: _candidates(kb_res.get(c, {}), k["max_candidates"]) for c in concepts}
    first_caption = {}
    for cid, ts in triples.items():
        for h, _, tl in ts:
            for m in (h, tl):
                first_caption.setdefault(concept_of[m], captions[cid])
    missing = [c for c in concepts if not cands[c]]
    cb_items = [{"mention": c, "caption": first_caption[c], "concept": c} for c in missing]
    for it, d in zip(cb_items, agents.callback([(i["concept"], i["caption"]) for i in cb_items])):
        it["desc"] = d
    inspector_loop(agents, cb_items,
                   lambda its, r: agents.callback([(i["concept"], i["caption"]) for i in its],
                                                  seeds=[r] * len(its), temperature=0.7), thr, rounds)
    for it in cb_items:
        cands[it["concept"]] = [("callback", it["desc"])] if it["desc"] else []
    timings["search_s"] = time.time() - t

    # 5-7) selector, refiner, inspector per (clip, mention) -------------------
    t = time.time()
    pairs = []  # one record per distinct (clip, mention)
    seen = set()
    for cid, ts in triples.items():
        for h, _, tl in ts:
            for m in (h, tl):
                if (cid, m) not in seen:
                    seen.add((cid, m))
                    pairs.append({"clip_id": cid, "mention": m, "concept": concept_of[m], "caption": captions[cid]})
    multi = [p for p in pairs if len(cands[p["concept"]]) > 1]
    for p, idx in zip(multi, agents.select([(p["concept"], p["caption"], [x[1] for x in cands[p["concept"]]])
                                             for p in multi])):
        p["sel"] = idx
    for p in pairs:
        cs = cands[p["concept"]]
        if not cs:
            p.update(src="none", base="", desc="")
            continue
        src, base = cs[p.get("sel", 0)]
        p.update(src=src, base=base)
    to_refine = [p for p in pairs if p["base"]]
    needs = [p for p in to_refine if p["mention"].strip().lower() != p["concept"].strip().lower()]
    for p, d in zip(needs, agents.refine([(p["mention"], p["concept"], p["base"]) for p in needs])):
        p["desc"] = d or p["base"]
    for p in to_refine:
        p.setdefault("desc", p["base"])
    inspector_loop(agents, to_refine,
                   lambda its, r: agents.refine([(i["mention"], i["concept"], i["base"]) for i in its],
                                                seeds=[r] * len(its), temperature=0.7), thr, rounds)
    timings["refine_inspect_s"] = time.time() - t

    # ---- assemble the graph ----------------------------------------------
    pinfo = {(p["clip_id"], p["mention"]): p for p in pairs}
    rows = []
    for cid, ts in triples.items():
        for h, r, tl in ts:
            ph, pt = pinfo[(cid, h)], pinfo[(cid, tl)]
            rows.append({"tid": len(rows), "clip_id": cid, "head": h, "rel": r, "tail": tl,
                         "head_concept": concept_of[h], "tail_concept": concept_of[tl],
                         "head_desc": ph.get("desc", ""), "tail_desc": pt.get("desc", ""),
                         "head_src": ph.get("src", "none") if ph.get("desc") else "none",
                         "tail_src": pt.get("src", "none") if pt.get("desc") else "none"})
    kg = pd.DataFrame(rows)
    wd.kg_dir.mkdir(parents=True, exist_ok=True)
    kg.to_parquet(wd.kg_dir / "triplets.parquet", index=False)
    pd.DataFrame([{"clip_id": c, "caption_used": captions[c]} for c in clip_ids]).to_parquet(
        wd.kg_dir / "captions_used.parquet", index=False)

    stats = graph_stats(kg)
    srcs = Counter(p.get("src", "none") if p.get("desc") else "none" for p in pairs)
    first_pass = [p for p in to_refine if p.get("rounds") == 1 and not p.get("discarded")]
    stats.update({
        "n_clips": len(clip_ids), "n_rewritten_captions": n_rewritten,
        "n_clips_without_triples": sum(1 for ts in triples.values() if not ts),
        "n_mentions": len(mentions), "n_concepts": len(concepts),
        "concepts_kb_hit_rate": round(1 - len(missing) / max(len(concepts), 1), 4),
        "callback_discard_rate": round(sum(bool(i.get("discarded")) for i in cb_items) / max(len(cb_items), 1), 4),
        "entity_desc_src": dict(srcs),
        "inspector_first_pass_rate": round(len(first_pass) / max(len(to_refine), 1), 4),
        "inspector_discard_rate": round(sum(bool(p.get("discarded")) for p in to_refine) / max(len(to_refine), 1), 4),
        "llm_calls": agents.llm.n_calls, "llm_cached": agents.llm.n_cached,
        **{f"time/{k_}": round(v, 1) for k_, v in timings.items()},
    })
    write_json(stats, wd.kg_dir / "stats.json")
    return stats


def graph_stats(kg: pd.DataFrame) -> dict:
    """Connectivity of the concept graph (nodes = concepts, edges = triples)."""
    parent: dict[str, str] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    deg = Counter()
    for h, t in zip(kg.head_concept, kg.tail_concept):
        a, b = find(h.lower()), find(t.lower())
        deg[h.lower()] += 1
        deg[t.lower()] += 1
        if a != b:
            parent[a] = b
    comps = Counter(find(x) for x in list(parent))
    n_nodes = len(parent)
    clips_per_concept = defaultdict(set)
    for c, h, t in zip(kg.clip_id, kg.head_concept, kg.tail_concept):
        clips_per_concept[h.lower()].add(c)
        clips_per_concept[t.lower()].add(c)
    shared = sum(1 for v in clips_per_concept.values() if len(v) > 1)
    return {
        "n_triplets": int(len(kg)), "n_nodes": n_nodes, "n_components": len(comps),
        "largest_component_frac": round(max(comps.values()) / n_nodes, 4) if n_nodes else 0.0,
        "mean_degree": round(sum(deg.values()) / n_nodes, 3) if n_nodes else 0.0,
        "concepts_shared_by_clips_frac": round(shared / max(len(clips_per_concept), 1), 4),
        "top_hubs": [f"{c}:{d}" for c, d in deg.most_common(10)],
        "triplets_per_clip": round(len(kg) / max(kg.clip_id.nunique(), 1), 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/m3kg/base.yaml")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_config(a.config, a.profile, a.set)
    from mmrag.m3kg.llm import VLLMBackend

    cache = KVCache(cfg["paths"]["cache_db"])
    k = cfg["kg"]
    llm = CachedLLM(VLLMBackend(k["llm"], max_model_len=k.get("max_model_len", 4096),
                                gpu_memory_utilization=k.get("gpu_memory_utilization", 0.85),
                                quantization=k.get("quantization")), cache)
    kb = KnowledgeBase(cache, threads=k.get("kb_threads", 8))
    with tracked_run(cfg["wandb"]["project"], "kg", "kg-build", cfg, job_type="build",
                     tags=["m3kg", "kg"]) as run:
        stats = build(cfg, Agents(llm), kb)
        run.set_summary({k_: v for k_, v in stats.items() if not isinstance(v, (list, dict))})
        run.set_summary({"entity_desc_src": stats["entity_desc_src"], "top_hubs": ", ".join(stats["top_hubs"])})
    print(stats)


if __name__ == "__main__":
    main()
