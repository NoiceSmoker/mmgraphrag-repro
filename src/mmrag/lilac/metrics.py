"""Retrieval metrics for the webpage benchmarks (MMCoQA, MultimodalQA).

Two families are computed from the same retrieval output:

* ``paper_*``  — exactly the protocol of the official evaluator
  (``OMGMMQABenchmarkRetrievalAccuracyEvaluator`` with ``k: 9`` in
  ``config/experiments/retrieval_accuracy_evaluator.yaml``):
    - ``paper_R@3``  := component recall over the first **9** retrieved
      components (three components are treated as one "screenshot");
    - ``paper_MRR@10`` := reciprocal rank where ranks are bucketed in groups
      of three, ``ceil(rank / 3)``, again within the first 9 components.
  These are the numbers comparable to Table 1 of the paper.

* ``strict_*`` — conventional definitions over individual components
  (R@3 over the first 3 components, MRR@10 over the first 10, Hit@k).

``tests/test_lilac_metrics.py`` checks that ``paper_*`` matches the official
evaluator on synthetic data.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from statistics import mean, median
from typing import Iterable

from mmrag.common.io import read_json

PARSER_CAP = 50  # official retrieval_result_parser.DATA_PARSER_K


@dataclass
class GoldQA:
    qid: str
    question: str
    components: list[tuple[str | None, str | None]]
    modalities: list[str] = field(default_factory=list)

    @property
    def modality_signature(self) -> str:
        return "+".join(sorted(set(self.modalities))) or "unknown"

    @property
    def n_docs(self) -> int:
        return len({d for d, _ in self.components})


def load_gold(path) -> dict[str, GoldQA]:
    """Parse ``QAs_dev_labeled.json`` the same way as the official
    ``ComponentLabeledMMQABenchmarkParser``."""
    gold: dict[str, GoldQA] = {}
    for qa in read_json(path):
        comps, mods = [], []
        for ev in qa["evidences"]:
            comps.append((ev.get("gold_webpage_title"), ev.get("gold_component_id")))
            mods.append(str(ev.get("mmqa_doc_modality", "unknown")))
        gold[qa["qid"]] = GoldQA(qa["qid"], qa["question"], comps, mods)
    return gold


def _component_id(raw: str) -> str:
    """Official ``get_component_id``: 'p_3_s2' -> 'p_3'."""
    return "_".join(raw.split("_")[:2])


def flatten_units(row: dict, cap: int = PARSER_CAP) -> list[tuple[str, str]]:
    """Official ``OMGMMQARetrievalResultParser`` flattening of retrieved units
    (edges or single nodes) into a de-duplicated component ranking."""
    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for unit in row.get("retrieved_units", []):
        for node in unit["nodes"]:
            key = (node[0].replace(".json", ""), _component_id(node[1]))
            if key not in seen:
                seen.add(key)
                out.append(key)
            if len(out) >= cap:
                return out
    return out


def paper_recall(gold: list, ranked: list, k: int = 9) -> float:
    if not gold:
        return 0.0
    return len(set(gold) & set(ranked[:k])) / len(gold)


def paper_bucketed_rr(gold: list, ranked: list, k: int = 9, bucket: int = 3) -> float:
    gold_set = set(gold)
    for rank, comp in enumerate(ranked[:k], 1):
        if comp in gold_set:
            return 1.0 / math.ceil(rank / bucket)
    return 0.0


def strict_recall(gold: list, ranked: list, k: int) -> float:
    return paper_recall(gold, ranked, k)


def strict_rr(gold: list, ranked: list, k: int = 10) -> float:
    gold_set = set(gold)
    for rank, comp in enumerate(ranked[:k], 1):
        if comp in gold_set:
            return 1.0 / rank
    return 0.0


def hit(gold: list, ranked: list, k: int) -> float:
    return float(bool(set(gold) & set(ranked[:k])))


def _pct(x: float) -> float:
    return round(100.0 * x, 3)


def evaluate(rows: Iterable[dict], gold: dict[str, GoldQA], paper_k: int = 9) -> tuple[dict, list[dict]]:
    """Return (summary metrics in %, per-question records)."""
    by_qid = {r["qid"]: r for r in rows}
    per_q: list[dict] = []
    for qid, g in gold.items():
        row = by_qid.get(qid)
        ranked = flatten_units(row) if row is not None else []
        t = (row or {}).get("time", {})
        mid = (row or {}).get("middle_results", {})
        per_q.append(
            {
                "qid": qid,
                "retrieved": row is not None,
                "modality": g.modality_signature,
                "n_gold": len(g.components),
                "n_docs": g.n_docs,
                "paper_R@3": paper_recall(g.components, ranked, paper_k),
                "paper_MRR@10": paper_bucketed_rr(g.components, ranked, paper_k),
                "strict_R@3": strict_recall(g.components, ranked, 3),
                "strict_R@9": strict_recall(g.components, ranked, 9),
                "strict_MRR@10": strict_rr(g.components, ranked, 10),
                "Hit@3": hit(g.components, ranked, 3),
                "Hit@9": hit(g.components, ranked, 9),
                "latency_ms": t.get("retrieval_time(ms)"),
                "subgraph_nodes": mid.get("subgraph_nodes_num"),
                "subgraph_edges": mid.get("subgraph_edges_num"),
            }
        )

    metric_keys = ["paper_R@3", "paper_MRR@10", "strict_R@3", "strict_R@9", "strict_MRR@10", "Hit@3", "Hit@9"]
    summary: dict[str, float] = {k: _pct(mean(r[k] for r in per_q)) for k in metric_keys} if per_q else {}
    summary["n_questions"] = len(per_q)
    summary["n_missing"] = sum(not r["retrieved"] for r in per_q)

    lat = [r["latency_ms"] for r in per_q if r["latency_ms"] is not None]
    if lat:
        lat_sorted = sorted(lat)
        summary["latency_ms_mean"] = round(mean(lat), 3)
        summary["latency_ms_p50"] = round(median(lat), 3)
        summary["latency_ms_p95"] = round(lat_sorted[int(0.95 * (len(lat_sorted) - 1))], 3)
    edges = [r["subgraph_edges"] for r in per_q if r["subgraph_edges"] is not None]
    if edges:
        summary["subgraph_edges_mean"] = round(mean(edges), 2)

    # Breakdowns: which questions does the graph traversal actually help?
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in per_q:
        groups[f"modality={r['modality']}"].append(r)
        groups["evidence=single" if r["n_gold"] <= 1 else "evidence=multi"].append(r)
        groups["docs=single" if r["n_docs"] <= 1 else "docs=multi"].append(r)
    for name, rs in groups.items():
        summary[f"{name}/n"] = len(rs)
        summary[f"{name}/paper_R@3"] = _pct(mean(x["paper_R@3"] for x in rs))
        summary[f"{name}/strict_R@3"] = _pct(mean(x["strict_R@3"] for x in rs))
    return summary, per_q
