"""Create one W&B Report per report section and print/save their links.

The course requires a public link next to every experimental section of
the written report (main results, ablation, sensitivity, ...). This script
builds those Reports from the logged runs (filtered by W&B *group*) and
writes the URLs to ``results/wandb_links.json`` (turned into LaTeX macros by
``scripts/export_results.py``).

Usage:
    WANDB_ENTITY=<you> python scripts/make_wandb_reports.py
    python scripts/make_wandb_reports.py --share   # also create view-only share links
                                                   # (needed when the project is not public)
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import wandb_workspaces.reports.v2 as wr

SECTIONS = [
    # macro, project, group, title, summary metrics for bar charts, (x config, y metric) for scatter
    ("wbLilacMain", "lilac-repro", "main", "LILaC - main retrieval results (MMCoQA)",
     ["paper_R@3", "paper_MRR@10", "strict_R@3", "strict_MRR@10"], None),
    ("wbLilacAblation", "lilac-repro", "ablation", "LILaC - ablation study",
     ["paper_R@3", "paper_MRR@10", "strict_R@3", "docs=multi/paper_R@3"], None),
    ("wbLilacBeam", "lilac-repro", "sens_beam", "LILaC - beam width sensitivity",
     ["paper_R@3", "paper_MRR@10", "latency_ms_mean"], ("beam_width", "paper_R@3")),
    ("wbLilacIter", "lilac-repro", "sens_iter", "LILaC - number of traversal iterations",
     ["paper_R@3", "paper_MRR@10", "identical_top10_to_ni1_b5"], ("num_iterations", "paper_R@3")),
    ("wbLilacDecomp", "lilac-repro", "decomposer", "LILaC - query decomposer size",
     ["paper_R@3", "decomp/DAcc", "decomp/avg_subqueries"], None),
    ("wbMkgKG", "m3kg-rag-repro", "kg", "M3KG-RAG - knowledge graph construction",
     ["n_triplets", "n_concepts", "concepts_kb_hit_rate", "inspector_first_pass_rate"], None),
    ("wbMkgMain", "m3kg-rag-repro", "main", "M3KG-RAG - main results (AudioCaps-QA)",
     ["MJ", "ctx/final_mean"], None),
    ("wbMkgAblation", "m3kg-rag-repro", "ablation", "M3KG-RAG - modality-wise retrieval x GRASP",
     ["MJ", "ctx/final_mean"], None),
    ("wbMkgGrasp", "m3kg-rag-repro", "grasp", "M3KG-RAG - GRASP components",
     ["MJ", "ctx/grounding_removed_pct", "ctx/llm_removed_pct"], None),
    ("wbMkgHops", "m3kg-rag-repro", "hops", "M3KG-RAG - multi-hop expansion",
     ["MJ", "ctx/final_mean"], ("hops", "MJ")),
    ("wbMkgTau", "m3kg-rag-repro", "sens_tau", "M3KG-RAG - retrieval threshold tau",
     ["MJ", "ctx/empty_rate"], None),
    ("wbMkgEta", "m3kg-rag-repro", "sens_eta", "M3KG-RAG - grounding threshold eta",
     ["MJ", "ctx/final_mean"], ("eta", "MJ")),
]


def build(entity: str, project_override: dict[str, str], share: bool) -> dict[str, str]:
    links = {}
    for macro, project, group, title, metrics, scatter in SECTIONS:
        project = project_override.get(project, project)
        runset = wr.Runset(entity=entity, project=project, name=group,
                           filters=f"Metric('Group') == '{group}'")
        panels = [wr.BarPlot(title=m, metrics=[wr.SummaryMetric(m)], orientation="h") for m in metrics]
        if scatter:
            panels.append(wr.ScatterPlot(title=f"{scatter[1]} vs {scatter[0]}", x=wr.Config(scatter[0]),
                                         y=wr.SummaryMetric(scatter[1])))
        report = wr.Report(
            project=project, entity=entity, title=title,
            description=f"Runs of W&B group '{group}' (reproduction of the corresponding paper experiment).",
            blocks=[wr.MarkdownBlock(text=f"All runs in group **{group}**; each bar is one configuration. "
                                          "Hover a bar for the run name; the run table lists configs and "
                                          "bootstrap confidence intervals."),
                    wr.PanelGrid(runsets=[runset], panels=panels)],
        )
        report.save()
        url = report.url
        if share:
            try:
                report.enable_share_link()
                url = report.get_share_url() or url
            except Exception as exc:  # personal entities may not support share links
                print(f"[warn] share link for {macro}: {exc}")
        links[macro] = url
        print(f"{macro:16s} {url}")
    return links


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
    ap.add_argument("--lilac-project", default="lilac-repro")
    ap.add_argument("--m3kg-project", default="m3kg-rag-repro")
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    if not a.entity:  # default entity of the logged-in API key (usually the user name)
        import wandb

        a.entity = wandb.Api().default_entity
        if not a.entity:
            raise SystemExit("could not determine the W&B entity: set --entity or WANDB_ENTITY")
        print(f"W&B entity: {a.entity}")
    links = build(a.entity, {"lilac-repro": a.lilac_project, "m3kg-rag-repro": a.m3kg_project}, a.share)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "wandb_links.json").write_text(json.dumps(links, indent=2))
    print(f"wrote {out / 'wandb_links.json'} (scripts/export_results.py turns it into LaTeX macros)")


if __name__ == "__main__":
    main()
