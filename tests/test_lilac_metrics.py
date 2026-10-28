"""The 'paper_*' metrics must equal the official LILaC evaluator."""
import json
import random

import pytest

from conftest import needs_lilac
from mmrag.lilac.metrics import evaluate, flatten_units, load_gold, paper_bucketed_rr, paper_recall


def _synthetic(tmp_path, n_q=60, seed=0):
    rng = random.Random(seed)
    docs = [f"Doc_{i}" for i in range(15)]
    comps = [f"{m}_{k}" for m in "pti" for k in range(6)]
    qa, rows = [], []
    for q in range(n_q):
        gold = list({(rng.choice(docs), rng.choice(comps)) for _ in range(rng.randint(1, 3))})
        qa.append({"qid": f"q{q}", "question": f"question {q}", "answers": [{"answer": "x"}],
                   "evidences": [{"mmqa_doc_modality": {"p": "text", "t": "table", "i": "image"}[c[0]],
                                  "gold_webpage_title": d, "gold_component_id": c} for d, c in gold]})
        units = []
        for _ in range(rng.randint(0, 12)):
            a = [f"{rng.choice(docs)}.json", rng.choice(comps) + rng.choice(["", "_s1", "_s3"])]
            if rng.random() < 0.6:
                b = [f"{rng.choice(docs)}.json", rng.choice(comps)]
                units.append({"nodes": [a, b], "edges": [[0, 1]], "score": 1.0, "type": "edge"})
            else:
                units.append({"nodes": [a], "edges": [], "score": 1.0, "type": "node"})
        if rng.random() < 0.3:  # sprinkle in gold hits
            d, c = gold[0]
            units.insert(rng.randint(0, len(units)), {"nodes": [[f"{d}.json", c]], "edges": [], "score": 2.0, "type": "node"})
        if q % 17 != 0:  # some questions have no retrieval output at all
            rows.append({"qid": f"q{q}", "retrieved_units": units, "time": {"retrieval_time(ms)": 1.0}, "middle_results": {}})
    qa_p = tmp_path / "QAs_dev_labeled.json"
    qa_p.write_text(json.dumps(qa))
    ret_p = tmp_path / "ret.jsonl"
    ret_p.write_text("\n".join(json.dumps(r) for r in rows))
    return qa_p, ret_p, rows


def test_flatten_dedup_and_component_ids():
    row = {"retrieved_units": [{"nodes": [["A.json", "p_1_s2"], ["B.json", "t_0"]]},
                               {"nodes": [["A.json", "p_1"], ["C.json", "i_3_s1"]]}]}
    assert flatten_units(row) == [("A", "p_1"), ("B", "t_0"), ("C", "i_3")]


def test_bucketed_rr_and_recall():
    gold = [("A", "p_1")]
    ranked = [("X", "p_0")] * 0 + [("B", "t_0"), ("C", "i_1"), ("D", "p_2"), ("A", "p_1")]
    assert paper_bucketed_rr(gold, ranked) == pytest.approx(0.5)   # rank 4 -> bucket 2
    assert paper_recall(gold, ranked, k=3) == 0.0
    assert paper_recall(gold, ranked, k=9) == 1.0


@needs_lilac
def test_matches_official_evaluator(tmp_path):
    from mmrag.lilac import official

    official.ensure_importable()
    from src.experiment.evaluator.retrieval_accuracy_evaluator import evaluate_retrieval_accuracy
    from src.experiment.utils.constants import AlgorithmName, BenchmarkType, LabelType

    qa_p, ret_p, rows = _synthetic(tmp_path)
    off = evaluate_retrieval_accuracy(AlgorithmName.OMG, BenchmarkType.MULTIMODALQA, LabelType.COMPONENT,
                                      str(qa_p), str(ret_p), k=9)
    ours, per_q = evaluate(rows, load_gold(qa_p))
    assert ours["paper_R@3"] == pytest.approx(100 * off["Component Recall"], abs=1e-3)
    assert ours["paper_MRR@10"] == pytest.approx(100 * off["Component MRR"], abs=1e-3)
    assert ours["n_missing"] == sum(1 for q in range(60) if q % 17 == 0)
    assert {"docs=single/n", "evidence=multi/paper_R@3"} <= set(ours)
