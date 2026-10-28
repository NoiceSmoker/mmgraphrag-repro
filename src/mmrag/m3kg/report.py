"""Aggregate judgements into metrics and log one W&B run per configuration.

Metrics per configuration:
  MJ                   AudioBench Model-as-Judge score (mean of 0-5 ratings x 20)
  MJ_ci_low/high       95% bootstrap interval over questions
  delta_vs_<ref>/MJ    paired bootstrap difference to the group's reference config
  ctx/*                context statistics (triples injected, empty-context rate,
                       share of triples removed by grounding / by the LLM filter)
"""
from __future__ import annotations

import argparse
from statistics import mean

from mmrag.common.io import read_json, read_jsonl
from mmrag.common.stats import bootstrap_mean, paired_bootstrap_diff
from mmrag.common.tracking import tracked_run
from mmrag.m3kg.config import WorkDir, load_config

# Reference configuration per group for the paired tests.
REFERENCE = {"main": "no_rag", "ablation": "abl_shared", "grasp": "grasp_none", "hops": "hops0",
             "sens_tau": "tau_q75", "sens_eta": "eta_0.5"}


def _ctx_stats(rows: list[dict]) -> dict:
    if not rows:
        return {}
    if "captions" in rows[0]:  # neighbour-caption baseline: no triples, no pruning
        n = [len(r["captions"]) for r in rows]
        return {"ctx/final_mean": round(mean(n), 3), "ctx/empty_rate": round(100 * mean(x == 0 for x in n), 2),
                "ctx/clips_selected_mean": round(mean(len(r.get("clips", [])) for r in rows), 3)}
    n_c = [len(r["candidates"]) for r in rows]
    n_g = [len(r["after_grounding"]) for r in rows]
    n_f = [len(r.get("final", r["after_grounding"])) for r in rows]
    return {
        "ctx/candidates_mean": round(mean(n_c), 3),
        "ctx/after_grounding_mean": round(mean(n_g), 3),
        "ctx/final_mean": round(mean(n_f), 3),
        "ctx/empty_rate": round(100 * mean(x == 0 for x in n_f), 2),
        "ctx/grounding_removed_pct": round(100 * (1 - sum(n_g) / max(sum(n_c), 1)), 2),
        "ctx/llm_removed_pct": round(100 * (1 - sum(n_f) / max(sum(n_g), 1)), 2),
        "ctx/clips_selected_mean": round(mean(len(r.get("clips", [])) for r in rows), 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/m3kg/base.yaml")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_config(a.config, a.profile, a.set)
    wd = WorkDir(cfg)
    project = cfg["wandb"]["project"]

    judged = {p.stem: {r["qid"]: r for r in read_jsonl(p)} for p in wd.judge_dir.glob("*.jsonl")}
    timing = read_json(wd.contexts_dir / "_timing.json") if (wd.contexts_dir / "_timing.json").exists() else {}
    judge_meta = read_json(wd.judge_dir / "_judge.json") if (wd.judge_dir / "_judge.json").exists() else {}
    kg_stats = read_json(wd.kg_dir / "stats.json") if (wd.kg_dir / "stats.json").exists() else {}
    winrate = read_json(wd.judge_dir / "winrate.json") if (wd.judge_dir / "winrate.json").exists() else {}

    configs = []
    for p in sorted(wd.contexts_dir.glob("*.config.json")):
        c = read_json(p)
        if c["name"] in judged:
            configs.append(c)
    # references first so their per-question scores are available for deltas
    configs.sort(key=lambda c: c["name"] != REFERENCE.get(c["group"]))

    for c in configs:
        name, group = c["name"], c["group"]
        rows = judged[name]
        qids = sorted(rows)
        scores = [20 * rows[q]["score_0_5"] for q in qids]
        ci = bootstrap_mean(scores)
        summary = {"MJ": round(ci.mean, 3), "MJ_ci_low": round(ci.low, 3), "MJ_ci_high": round(ci.high, 3),
                   "n_questions": len(qids),
                   "judge_parse_fail_pct": round(100 * mean(rows[q]["parse"] == "failed" for q in qids), 2),
                   "answer_chars_mean": round(mean(len(rows[q]["prediction"]) for q in qids), 1)}
        ctx_path = wd.contexts_dir / f"{name}.jsonl"
        if ctx_path.exists():
            summary.update(_ctx_stats(read_jsonl(ctx_path)))
        for ref in {REFERENCE.get(group), "no_rag"} - {None, name}:
            if ref in judged:
                common = [q for q in qids if q in judged[ref]]
                d, pval = paired_bootstrap_diff([20 * rows[q]["score_0_5"] for q in common],
                                                [20 * judged[ref][q]["score_0_5"] for q in common])
                summary.update({f"delta_vs_{ref}/MJ": round(d.mean, 3), f"delta_vs_{ref}/MJ_ci_low": round(d.low, 3),
                                f"delta_vs_{ref}/MJ_ci_high": round(d.high, 3), f"delta_vs_{ref}/p_le_0": round(pval, 4)})
        for pair, res in winrate.items():
            if pair.split(":")[0] == name:
                summary.update({f"winrate_vs_{pair.split(':')[1]}/{k}": v for k, v in res.items()})
        run_cfg = {**c, "generator": cfg["generation"]["model"], "judge": judge_meta.get("judge_model"),
                   "kg_llm": cfg["kg"]["llm"], "corpus_size": cfg["data"]["corpus_size"],
                   "clap": cfg["retrieval"]["clap_model"], "timing": timing, "kg_stats": kg_stats}
        with tracked_run(project, group, name, run_cfg, job_type="eval", tags=["m3kg", group]) as run:
            run.set_summary(summary)
            if group == "main":
                run.log_table("per_question", [{"qid": q, "score": 20 * rows[q]["score_0_5"],
                                                "prediction": rows[q]["prediction"][:500],
                                                "n_triples": rows[q].get("n_triples")} for q in qids])
        print(f"[{group}/{name}] MJ={summary['MJ']:.2f} [{summary['MJ_ci_low']:.2f}, {summary['MJ_ci_high']:.2f}]")


if __name__ == "__main__":
    main()
