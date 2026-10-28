"""Data flow of the M3KG-RAG stages contexts -> filter -> generate -> judge ->
report, with every model replaced by a fake (no GPU, no downloads)."""
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import REPO
from mmrag.common.io import write_json
from mmrag.m3kg import contexts, report, vllm_stages
from mmrag.m3kg.llm import FunctionBackend


def _unit(m):
    m = np.asarray(m, dtype=np.float32)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


@pytest.fixture
def work(tmp_path):
    import soundfile as sf

    w = tmp_path / "w"
    (w / "test" / "audio").mkdir(parents=True)
    rows = []
    for i in range(4):
        p = w / "test" / "audio" / f"q{i}.wav"
        sf.write(p, np.zeros(16000, dtype="float32"), 16000)
        rows.append({"qid": f"q{i}", "question": f"What is heard in clip {i}?", "answer": "a dog barking", "path": str(p)})
    pd.DataFrame(rows).to_parquet(w / "test" / "meta.parquet")
    (w / "corpus").mkdir()
    pd.DataFrame([{"clip_id": f"c{i}", "caption": cap} for i, cap in enumerate(
        ["a dog barks twice", "a siren wails", "a man speaks", "people laugh", "water runs"])]).to_parquet(
        w / "corpus" / "meta.parquet")
    kg = pd.DataFrame([dict(tid=t, clip_id=f"c{t % 5}", head=h, rel="makes", tail=tl, head_concept=h, tail_concept=tl,
                            head_desc=f"{h} desc", tail_desc="", head_src="wikipedia", tail_src="none")
                       for t, (h, tl) in enumerate([("dog", "bark"), ("dog", "noise"), ("siren", "wail"),
                                                    ("man", "speech"), ("man", "laugh"), ("bark", "echo")])])
    (w / "kg").mkdir()
    kg.to_parquet(w / "kg" / "triplets.parquet")
    ix = w / "index"
    ix.mkdir()
    rng = np.random.default_rng(0)
    c = _unit(rng.normal(size=(5, 8)))
    np.save(ix / "corpus_emb.npy", c)
    write_json([f"c{i}" for i in range(5)], ix / "corpus_ids.json")
    np.save(ix / "test_emb.npy", _unit(c[:4] + 0.05 * rng.normal(size=(4, 8))))
    write_json([f"q{i}" for i in range(4)], ix / "test_ids.json")
    np.save(ix / "triple_text_emb.npy", _unit(rng.normal(size=(6, 8))))
    write_json(list(range(6)), ix / "triple_tids.json")
    write_json({"q25": 0.3, "q50": 0.6, "q75": 1.0, "q90": 1.4, "none": float("inf")}, ix / "taus.json")
    cfg = {"paths": {"work_dir": str(w), "cache_db": str(w / "cache.sqlite")},
           "data": {"corpus_size": 5}, "kg": {"llm": "fake"},
           "retrieval": {"clap_model": "fake", "k": 5, "hub_top_frac": 0.0, "hub_max_degree": 100},
           "grasp": {"eta_a": 0.5, "eta_a_clap_window": 0.2, "eta_grid": [0.1, 0.5, 0.9], "filter_llm": "fake"},
           "generation": {"model": "fake-omni"},
           "judge": {"model": "fake-judge", "audiobench_dir": str(REPO / "third_party" / "AudioBench")},
           "wandb": {"project": "m3kg-flow"}}
    return cfg


class FakeGrounder:
    name = "tag"

    def score(self, path, phrases):
        return np.array([0.9 if ("dog" in p or "bark" in p) else 0.2 for p in phrases])


def fake_vllm_module():
    mod = types.ModuleType("vllm")

    class Out:
        def __init__(self, text):
            self.outputs = [types.SimpleNamespace(text=text)]

    class LLM:
        def __init__(self, **kw):
            pass

        def generate(self, reqs, sampling_params=None, use_tqdm=False):
            return [Out("A dog is barking." if "dog" in r["prompt"] else "Some noise.") for r in reqs]

    mod.LLM, mod.SamplingParams = LLM, lambda **kw: kw
    return mod


def test_stages_flow(work, monkeypatch):
    if not (REPO / "third_party" / "AudioBench").exists():
        pytest.skip("AudioBench not cloned")
    monkeypatch.setattr(contexts, "ClapEncoder", lambda *a, **k: None)
    monkeypatch.setattr(contexts, "make_grounder", lambda cfg, clap: FakeGrounder())
    contexts.run(work, groups=None)

    def fake_filter(system, user):  # keep only the first triple
        return "[0]"

    monkeypatch.setattr(vllm_stages, "VLLMBackend", lambda *a, **k: FunctionBackend(fake_filter))
    vllm_stages.stage_filter(work)

    monkeypatch.setitem(sys.modules, "vllm", fake_vllm_module())
    vllm_stages.stage_generate(work)

    monkeypatch.setattr(vllm_stages, "VLLMBackend", lambda *a, **k: FunctionBackend(
        lambda s, u: "Explanation: fine. Rating: 5" if "dog is barking" in u else "Explanation: no. Rating: 1"))
    vllm_stages.stage_judge(work)

    base = work["paths"]["work_dir"] + "/../base.yaml"
    import yaml

    open(base, "w").write(yaml.safe_dump(work))
    monkeypatch.setattr(sys, "argv", ["report", "--config", base])
    report.main()

    import json
    import mmrag.common.tracking as tr

    res = {p.stem: json.loads(p.read_text())["summary"] for p in (tr.RESULTS_DIR / "m3kg-flow").glob("*/*.json")}
    assert res["no_rag"]["MJ"] == pytest.approx(20.0)          # no context -> "Some noise." -> rating 1
    assert res["m3kg_rag"]["MJ"] > res["no_rag"]["MJ"]           # grounded dog triples reach the generator
    assert res["eta_0.9"]["ctx/final_mean"] <= res["eta_0.1"]["ctx/final_mean"]
    assert res["grasp_tag"]["ctx/llm_removed_pct"] == 0.0
    assert "delta_vs_no_rag/MJ" in res["m3kg_rag"]

    # neighbour-caption baseline: same nearest clips, raw captions, no triples
    from mmrag.common.io import read_jsonl

    wdir = Path(work["paths"]["work_dir"])
    cap_rows = read_jsonl(wdir / "contexts" / "caption_rag.jsonl")
    full_rows = {r["qid"]: r for r in read_jsonl(wdir / "contexts" / "m3kg_rag.jsonl")}
    assert all(r["candidates"] == [] and r["needs_llm_filter"] is False for r in cap_rows)
    assert all([c for c, _ in r["clips"]] == [c for c, _ in full_rows[r["qid"]]["clips"]] for r in cap_rows)
    assert any("a dog barks twice" in r["captions"] for r in cap_rows)
    assert "caption_rag" in res
