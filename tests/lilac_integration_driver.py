"""Driver for test_lilac_integration.py (runs in a fresh interpreter).

Builds a tiny synthetic MMCoQA-like corpus *inside a copy of the official
LILaC repo* (parsed documents, embeddings, decompositions) and runs the
reproduction's RetrieverSession over several variants, exactly as the real
experiment runner does. Prints a JSON summary on stdout.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

root = Path(os.environ["LILAC_ROOT"])
DS = "MMCoQA"
D = 16
rng = np.random.default_rng(0)

docs_dir = root / "datasets" / DS / "parsed_documents" / "dev"
docs_dir.mkdir(parents=True, exist_ok=True)
(root / "datasets" / DS / "image_components" / "dev").mkdir(parents=True, exist_ok=True)

top, low = {"text": [], "image": []}, {"sentence": [], "subimage": []}
topic = {}
N_DOCS = 60  # >= 100 top-level and >= 2048 low-level nodes (official kNN sizes)
for d in range(N_DOCS):
    fn = f"Doc_{d}.json"
    doc = {"title": f"Doc {d}", "hierarchy": {"COMPONENTS": []}, "id_sequence": [], "header": {},
           "text": {}, "table": {}, "image": {}, "sentence": {}, "proposition": {}, "table_segment": {},
           "subimage": {}, "id_to_html": {}}
    for p in range(3):
        cid = f"p_{p}"
        edges = [{"edge": f"Doc_{(d + 1) % N_DOCS}.json"}] if p == 0 else []
        doc["text"][cid] = {"text": f"paragraph {p} of doc {d}", "edges": edges}
        doc["hierarchy"]["COMPONENTS"].append(cid)
        top["text"].append([fn, cid])
        for s in range(12):
            doc["sentence"][f"{cid}_s{s}"] = {"text": f"sentence {s}"}
            low["sentence"].append([fn, f"{cid}_s{s}"])
    doc["image"]["i_0"] = {"filename": None, "caption": {"text": "an image", "edges": []}}
    doc["hierarchy"]["COMPONENTS"].append("i_0")
    top["image"].append([fn, "i_0"])
    doc["subimage"]["i_0_s1"] = {"filename": None, "caption": {"text": "a crop", "edges": []}}
    low["subimage"].append([fn, "i_0_s1"])
    (docs_dir / fn).write_text(json.dumps(doc))

emb_dir = root / "artifacts" / DS / "embeddings" / "MM-Embed"
emb_dir.mkdir(parents=True, exist_ok=True)
vec = {}


def save(name, ids):
    m = rng.normal(size=(len(ids), D)).astype("float32")
    m /= np.linalg.norm(m, axis=1, keepdims=True)
    for i, t in enumerate(ids):
        vec[tuple(t)] = m[i]
    torch.save(torch.from_numpy(m), emb_dir / f"{name}.pt")
    (emb_dir / f"{name}.json").write_text(json.dumps({str(i): t for i, t in enumerate(ids)}))


for k, ids in {**top, **low}.items():
    save(k, ids)

# Questions whose gold evidence is (Doc_d, p_0) + (Doc_{d+1}, i_0) - a two-hop pattern.
qa, subq = [], {}
for q in range(8):
    d = q % 6
    gold = [(f"Doc_{d}", "p_0"), (f"Doc_{(d + 1) % N_DOCS}", "i_0")]
    qa.append({"qid": f"q{q}", "question": f"question {q}", "answers": [{"answer": "a"}],
               "evidences": [{"mmqa_doc_modality": "text", "gold_webpage_title": gold[0][0], "gold_component_id": "p_0"},
                             {"mmqa_doc_modality": "image", "gold_webpage_title": gold[1][0], "gold_component_id": "i_0"}]})
    subq[f"q{q}"] = {"qid": f"q{q}", "question": f"question {q}", "time": {},
                     "subqueries": [{"sqid": f"q{q}___1", "subquery": "text part", "modality": "text"},
                                    {"sqid": f"q{q}___2", "subquery": "image part", "modality": "image"}]}
(root / "datasets" / DS / "QAs_dev_labeled.json").write_text(json.dumps(qa))


def query_vectors(noise):
    qv, sv = [], []
    for q in range(8):
        d = q % 6
        a = vec[(f"Doc_{d}.json", "p_0_s0")]
        b = vec[(f"Doc_{(d + 1) % N_DOCS}.json", "i_0_s1")]
        qv.append(a + 0.3 * b)
        sv += [a + noise * rng.normal(size=D), b + noise * rng.normal(size=D)]
    return np.array(qv, "float32"), np.array(sv, "float32")


sys.path.insert(0, os.environ["MMRAG_SRC"])
from mmrag.common.io import write_json  # noqa: E402
from mmrag.lilac import decompose  # noqa: E402

for tag, noise in (("good", 0.05), ("noisy", 3.0)):
    vdir = decompose.variant_dir(DS, tag)
    vdir.mkdir(parents=True, exist_ok=True)
    write_json(subq, vdir / "subqueries_with_modality.json")
    write_json(subq, vdir / "subqueries.json")
    write_json({"model": f"fake-{tag}", "decomposition_ms_per_query": 1.0}, vdir / "meta.json")
    qv, sv = query_vectors(noise)
    edir = decompose.emb_variant_dir(DS, tag)
    edir.mkdir(parents=True, exist_ok=True)
    sq_ids = {str(i): f"q{i // 2}___{i % 2 + 1}" for i in range(16)}
    for kind in ("aware", "agnostic"):
        torch.save(torch.from_numpy(sv), edir / f"subqueries_modality_{kind}.pt")
        (edir / f"subqueries_modality_{kind}.json").write_text(json.dumps(sq_ids))
    torch.save(torch.from_numpy(qv), emb_dir / "query.pt")
    (emb_dir / "query.json").write_text(json.dumps({str(i): f"q{i}" for i in range(8)}))

decompose.activate_decomposition(DS, "good")

from mmrag.lilac.metrics import evaluate, load_gold  # noqa: E402
from mmrag.lilac.session import RetrieverSession  # noqa: E402
from mmrag.lilac.variants import ablation_group, decomposer_group, sens_iter_group  # noqa: E402

sess = RetrieverSession(DS)
gold = load_gold(root / "datasets" / DS / "QAs_dev_labeled.json")
out = {}
variants = [v for v in ablation_group() if v.embedding_set == "default"] + sens_iter_group() + decomposer_group(["good", "noisy"])
for v in variants:
    from dataclasses import replace

    v = replace(v, decomposer="good" if v.decomposer == "default" else v.decomposer)
    path = sess.run(v, f"{DS}__{v.group}__{v.name}")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    summ, _ = evaluate(rows, gold)
    out[f"{v.group}/{v.name}"] = {"paper_R@3": summ["paper_R@3"], "n": summ["n_questions"],
                                  "top": [[u["nodes"] for u in r["retrieved_units"][:2]] for r in rows[:2]]}
print("RESULT_JSON=" + json.dumps(out))
