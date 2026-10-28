"""Run the LILaC retrieval experiments and log them to W&B.

Prerequisites: corpus embeddings, sub-query embeddings and decomposition
variants already computed (scripts/lilac/10..40_*.sh). This stage needs no
LLM and runs on CPU in minutes, so it can be repeated freely (e.g. on a
laptop) once the artefacts have been copied from Drive.

Example:
    python -m mmrag.lilac.run_experiments --dataset MMCoQA \
        --groups main ablation sens_beam sens_iter decomposer \
        --main-decomposer qwen2.5-7b --decomposers qwen2.5-7b qwen2.5-14b-awq qwen2.5-32b-awq
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from mmrag.common.io import read_json, read_jsonl
from mmrag.common.stats import bootstrap_mean, paired_bootstrap_diff
from mmrag.common.tracking import tracked_run
from mmrag.lilac import official
from mmrag.lilac.decomp_eval import decomposition_report
from mmrag.lilac.decompose import activate_decomposition, variant_dir
from mmrag.lilac.metrics import evaluate, flatten_units, load_gold
from mmrag.lilac.session import RetrieverSession
from mmrag.lilac.variants import GROUPS, Variant, decomposer_group

# Reference variant for paired bootstrap tests, per group.
REFERENCE = {"main": None, "ablation": "wo_lcg_qd", "sens_beam": "b1", "sens_iter": "ni0_b5",
             "decomposer": None}  # decomposer group: filled with the main decomposer at runtime
TIMING_KEYS = ["knn_search(ms)", "subgraph_extraction(ms)", "gpu_calculation(ms)",
               "cpu_late_interaction(ms)", "in_edge_reranking(ms)", "retrieval_time(ms)"]


def _timing_summary(rows: list[dict]) -> dict:
    acc = defaultdict(list)
    for r in rows:
        for k in TIMING_KEYS:
            if k in r.get("time", {}):
                acc[k].append(r["time"][k])
    return {f"time/{k.replace('(ms)', '_ms')}": round(sum(v) / len(v), 3) for k, v in acc.items() if v}


def _embeddings_available(dataset: str, embedding_set: str) -> bool:
    if embedding_set == "default":
        return True
    emb = official.embeddings_dir(dataset)
    return (emb / "image_novlm.pt").exists() and (emb / "subimage_novlm.pt").exists()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="MMCoQA")
    ap.add_argument("--groups", nargs="+", default=["main", "ablation", "sens_beam", "sens_iter"])
    ap.add_argument("--main-decomposer", default="qwen2.5-7b")
    ap.add_argument("--decomposers", nargs="*", default=[])
    ap.add_argument("--project", default="lilac-repro")
    ap.add_argument("--gpu", type=int, default=-1, help="CUDA device for late-interaction matmuls (-1 = CPU)")
    ap.add_argument("--only", nargs="*", default=None, help="run only these variant names")
    args = ap.parse_args()

    gold = load_gold(official.labeled_qa_path(args.dataset))
    activate_decomposition(args.dataset, args.main_decomposer)
    REFERENCE["decomposer"] = f"decomp_{args.main_decomposer}"
    # make sure the reference decomposer runs first inside its group
    args.decomposers = sorted(args.decomposers, key=lambda t: t != args.main_decomposer)

    variants: list[Variant] = []
    for g in args.groups:
        if g == "decomposer":
            variants += decomposer_group(args.decomposers)
        else:
            variants += GROUPS[g]()
    if args.only:
        variants = [v for v in variants if v.name in set(args.only)]
    # decomposer variants last, so the main decomposition stays active for the rest
    variants.sort(key=lambda v: v.group == "decomposer")

    sessions: dict[str, RetrieverSession] = {}
    per_q_cache: dict[tuple[str, str], list[dict]] = {}
    rankings: dict[tuple[str, str], dict[str, list]] = {}

    for v in variants:
        if not _embeddings_available(args.dataset, v.embedding_set):
            print(f"[skip] {v.name}: embedding set '{v.embedding_set}' not computed")
            continue
        if v.embedding_set not in sessions:
            sessions[v.embedding_set] = RetrieverSession(args.dataset, embedding_set=v.embedding_set, gpu=args.gpu)
        sess = sessions[v.embedding_set]
        decomposer = args.main_decomposer if v.decomposer == "default" else v.decomposer
        v_run = Variant(**{**v.as_config(), "decomposer": decomposer, "tags": tuple(v.tags)})
        run_name = f"{args.dataset}__{v.group}__{v.name}"
        jsonl = sess.run(v_run, run_name)
        rows = read_jsonl(jsonl)
        summary, per_q = evaluate(rows, gold)
        summary.update(_timing_summary(rows))
        per_q_cache[(v.group, v.name)] = per_q
        rankings[(v.group, v.name)] = {r["qid"]: flatten_units(r)[:10] for r in rows}

        cfg = {**v_run.as_config(), "dataset": args.dataset, "embedder": official.DEFAULT_EMBEDDER,
               "lilac_commit": official.PINNED_COMMIT, "retrieval_jsonl": str(jsonl)}
        if v.group == "decomposer" or v.group == "main":
            meta_p = variant_dir(args.dataset, decomposer) / "meta.json"
            if meta_p.exists():
                cfg["decomposition_meta"] = read_json(meta_p)
            swm = variant_dir(args.dataset, decomposer) / "subqueries_with_modality.json"
            if swm.exists():
                d_sum, _ = decomposition_report(read_json(swm), gold)
                summary.update({f"decomp/{k}": val for k, val in d_sum.items()})

        ci = bootstrap_mean([r["paper_R@3"] for r in per_q])
        summary.update({"paper_R@3_ci_low": round(100 * ci.low, 3), "paper_R@3_ci_high": round(100 * ci.high, 3)})
        ref = REFERENCE.get(v.group)
        if ref and (v.group, ref) in per_q_cache and ref != v.name:
            base = {r["qid"]: r for r in per_q_cache[(v.group, ref)]}
            mine = {r["qid"]: r for r in per_q}
            qids = sorted(set(base) & set(mine))
            for metric in ("paper_R@3", "strict_R@3"):
                d, p = paired_bootstrap_diff([mine[q][metric] for q in qids], [base[q][metric] for q in qids])
                summary[f"delta_vs_{ref}/{metric}"] = round(100 * d.mean, 3)
                summary[f"delta_vs_{ref}/{metric}_ci_low"] = round(100 * d.low, 3)
                summary[f"delta_vs_{ref}/{metric}_ci_high"] = round(100 * d.high, 3)
                summary[f"delta_vs_{ref}/{metric}_p_le_0"] = round(p, 4)
        if v.name == "official_script_ni2_b5" and ("sens_iter", "ni1_b5") in rankings:
            a, b = rankings[(v.group, v.name)], rankings[("sens_iter", "ni1_b5")]
            same = [a[q] == b.get(q) for q in a]
            summary["identical_top10_to_ni1_b5"] = round(100 * sum(same) / max(len(same), 1), 3)
        if v.name == "ni2_b5" and ("sens_iter", "ni1_b5") in rankings:
            a, b = rankings[(v.group, v.name)], rankings[("sens_iter", "ni1_b5")]
            same = [a[q] == b.get(q) for q in a]
            summary["identical_top10_to_ni1_b5"] = round(100 * sum(same) / max(len(same), 1), 3)

        with tracked_run(args.project, v.group, f"{args.dataset}-{v.name}", cfg,
                         job_type="retrieval", tags=[args.dataset, v.group]) as run:
            run.set_summary(summary)
            if v.group in ("main", "ablation"):
                run.log_table("per_question", per_q)
        print(f"[{v.group}/{v.name}] paper_R@3={summary.get('paper_R@3')}  paper_MRR@10={summary.get('paper_MRR@10')}"
              f"  strict_R@3={summary.get('strict_R@3')}")

    activate_decomposition(args.dataset, args.main_decomposer)


if __name__ == "__main__":
    main()
