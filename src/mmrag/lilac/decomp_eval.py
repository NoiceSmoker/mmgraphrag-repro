"""Decomposition accuracy (paper Sec. 5.6).

DAcc(q) = Jaccard( predicted modality set, gold modality set ), where the
predicted set is the union of the modality labels of the generated
sub-queries and the gold set comes from the modalities of the gold
evidence components. The dataset-level score is the mean over questions.
"""
from __future__ import annotations

from statistics import mean

from mmrag.common.io import read_json
from mmrag.lilac.metrics import GoldQA


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def decomposition_report(subqueries_with_modality: dict, gold: dict[str, GoldQA]) -> tuple[dict, list[dict]]:
    rows = []
    for qid, g in gold.items():
        blk = subqueries_with_modality.get(qid, {"subqueries": []})
        pred = {sq.get("modality", "text") for sq in blk["subqueries"]}
        gold_set = {m for m in g.modalities if m in {"text", "table", "image"}}
        rows.append({"qid": qid, "n_sub": len(blk["subqueries"]), "pred": "+".join(sorted(pred)),
                     "gold": "+".join(sorted(gold_set)), "jaccard": jaccard(pred, gold_set),
                     "exact": float(pred == gold_set)})
    summary = {
        "DAcc": round(100 * mean(r["jaccard"] for r in rows), 3),
        "modality_set_exact": round(100 * mean(r["exact"] for r in rows), 3),
        "avg_subqueries": round(mean(r["n_sub"] for r in rows), 3),
        "single_subquery_rate": round(100 * mean(r["n_sub"] == 1 for r in rows), 3),
    }
    return summary, rows


def load_variant(path) -> dict:
    return read_json(path)
