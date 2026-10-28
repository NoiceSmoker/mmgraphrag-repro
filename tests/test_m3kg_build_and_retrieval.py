import re

import numpy as np
import pandas as pd

from mmrag.common.cache import KVCache
from mmrag.m3kg import prompts as P
from mmrag.m3kg.agents import Agents
from mmrag.m3kg.build_kg import build, graph_stats
from mmrag.m3kg.kb import KnowledgeBase
from mmrag.m3kg.llm import CachedLLM, FunctionBackend
from mmrag.m3kg.retrieval import KGIndex, calibrate_tau, l2_from_unit, shared_space_retrieve


def fake_llm(system: str, user: str) -> str:
    if system == P.EXTRACTOR_SYSTEM:
        cap = user.split("Caption: ")[1].split("\n")[0]
        if "dog" in cap:
            return "(a small dog, barks at, the mailman)\n(the mailman, rings, a doorbell)"
        if "siren" in cap:
            return "(a raid siren, wails over, the city)"
        return ""
    if system == P.NORMALIZER_SYSTEM:
        m = user.split("Phrase: ")[1].split("\n")[0]
        return {"a small dog": "dog", "the mailman": "mailman", "a doorbell": "doorbell",
                "a raid siren": "air raid siren", "the city": "city"}.get(m, m)
    if system == P.CALLBACK_SYSTEM:
        return "A short definition."
    if system == P.SELECTOR_SYSTEM:
        return "2"
    if system == P.REFINER_SYSTEM:
        m = re.search(r"Original wording: (.*)\n", user).group(1)
        return f"Adapted for {m}."
    if system == P.INSPECTOR_SYSTEM:
        # descriptions about doorbells always fail -> discarded after the max rounds
        return "2" if "doorbell" in user.lower() else "9"
    raise AssertionError(system[:40])


def fake_http(url, params=None):
    q = (params or {}).get("gsrsearch", "")
    if "wikipedia" in url and q in {"dog", "mailman", "doorbell", "city"}:
        return {"query": {"pages": [{"index": 1, "extract": f"{q} sense one."},
                                    {"index": 2, "extract": f"{q} sense two."}]}}
    return {}


def _cfg(tmp_path):
    return {"paths": {"work_dir": str(tmp_path / "w"), "cache_db": str(tmp_path / "c.sqlite")},
            "kg": {"inspector_threshold": 7, "inspector_max_rounds": 3, "max_candidates": 5, "rewriter": False},
            "data": {}}


def test_build_kg_end_to_end_with_fakes(tmp_path):
    cfg = _cfg(tmp_path)
    corpus = tmp_path / "w" / "corpus"
    corpus.mkdir(parents=True)
    pd.DataFrame([{"clip_id": "c1", "caption": "a small dog barks", "youtube_id": "x"},
                  {"clip_id": "c2", "caption": "a raid siren wails", "youtube_id": "y"},
                  {"clip_id": "c3", "caption": "silence", "youtube_id": "z"}]).to_parquet(corpus / "meta.parquet")
    cache = KVCache(cfg["paths"]["cache_db"])
    backend = FunctionBackend(fake_llm)
    stats = build(cfg, Agents(CachedLLM(backend, cache)), KnowledgeBase(cache, threads=1, http_get=fake_http))
    kg = pd.read_parquet(tmp_path / "w" / "kg" / "triplets.parquet")
    assert len(kg) == 3 and stats["n_clips_without_triples"] == 1
    dog = kg[kg["head"] == "a small dog"].iloc[0]
    assert dog.head_concept == "dog" and dog.head_desc == "Adapted for a small dog." and dog.head_src == "wikipedia"
    bell = kg[kg["tail"] == "a doorbell"].iloc[0]
    assert bell.tail_desc == "" and bell.tail_src == "none"          # discarded by the inspector
    siren = kg[kg["head"] == "a raid siren"].iloc[0]
    assert siren.head_src == "callback"                                # no KB hit -> LLM callback
    assert 0 < stats["inspector_discard_rate"] < 1
    # rerun is fully cached
    calls = backend.calls
    build(cfg, Agents(CachedLLM(backend, cache)), KnowledgeBase(cache, threads=1, http_get=fake_http))
    assert backend.calls == calls


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def test_retrieval_lift_expand_and_hubs():
    kg = pd.DataFrame([
        dict(tid=0, clip_id="a", head="dog", rel="barks at", tail="man", head_concept="dog", tail_concept="man"),
        dict(tid=1, clip_id="b", head="dog", rel="chews", tail="bone", head_concept="dog", tail_concept="bone"),
        dict(tid=2, clip_id="c", head="bone", rel="lies on", tail="floor", head_concept="bone", tail_concept="floor"),
        dict(tid=3, clip_id="d", head="man", rel="talks", tail="radio", head_concept="man", tail_concept="radio"),
    ])
    emb = _unit([[1, 0, 0], [0.9, 0.1, 0], [0.5, 0.5, 0], [0, 0, 1]])
    idx = KGIndex(kg, ["a", "b", "c", "d"], emb)
    q = _unit([1, 0, 0])
    near = idx.nearest_clips(q, k=1, tau=10)
    assert [c for c, _ in near] == ["a"]
    seeds = idx.lift(near)
    assert seeds == [0]
    assert idx.expand(q, seeds, hops=0, max_triplets=10, hubs=set()) == [0]
    one = idx.expand(q, seeds, hops=1, max_triplets=10, hubs=set())
    assert one[0] == 0 and set(one) == {0, 1, 3} and one.index(1) < one.index(3)  # closer clip first
    assert set(idx.expand(q, seeds, hops=2, max_triplets=10, hubs=set())) == {0, 1, 2, 3}
    assert set(idx.expand(q, seeds, hops=1, max_triplets=10, hubs={"man"})) == {0, 1}
    assert idx.expand(q, seeds, hops=2, max_triplets=2, hubs=set()) == [0, 1]
    assert idx.nearest_clips(_unit([0, 1, 0]), k=2, tau=0.1) == []   # tau removes off-topic neighbours


def test_tau_calibration_and_shared_space():
    rng = np.random.default_rng(0)
    emb = _unit(rng.normal(size=(50, 8)))
    taus = calibrate_tau(emb, k=5)
    assert taus["q25"] <= taus["q50"] <= taus["q75"] <= taus["q90"] < taus["none"]
    d = l2_from_unit(emb[0], emb)
    assert d[0] < 1e-3
    tt = _unit(rng.normal(size=(6, 8)))
    got = shared_space_retrieve(tt[3], tt, [10, 11, 12, 13, 14, 15], top_m=2)
    assert got[0] == 13 and len(got) == 2


def test_graph_stats_components():
    kg = pd.DataFrame([dict(clip_id="a", head_concept="dog", tail_concept="man"),
                       dict(clip_id="b", head_concept="dog", tail_concept="bone"),
                       dict(clip_id="c", head_concept="siren", tail_concept="city")])
    kg["tid"] = range(3)
    s = graph_stats(kg)
    assert s["n_nodes"] == 5 and s["n_components"] == 2 and s["concepts_shared_by_clips_frac"] == 0.2
