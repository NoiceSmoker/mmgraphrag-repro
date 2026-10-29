"""Turn the local result files (``results/<project>/<group>/<run>.json``,
written by every experiment) into LaTeX table rows and pgfplots data files
for the written report.

    python scripts/export_results.py --results results --out <report>/results
    python scripts/export_results.py --placeholders --out <report>/results   # empty tables

Each output file is self-contained (rows only); the report \\input{}s them.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

DASH = "--"


def load(results: Path, project: str) -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = {}
    base = results / project
    if not base.exists():
        return out
    for f in base.glob("*/*.json"):
        d = json.loads(f.read_text())
        out.setdefault(d["group"], {})[d["name"]] = d
    return out


def fmt(x, nd: int = 1) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return DASH
    if isinstance(x, (int, float)):
        return f"{x:.{nd}f}"
    return str(x)


def get(runs: dict, group: str, name: str, key: str):
    return runs.get(group, {}).get(name, {}).get("summary", {}).get(key)


def ci(runs, group, name, key, nd=1) -> str:
    m, lo, hi = (get(runs, group, name, k) for k in (key, f"{key}_ci_low", f"{key}_ci_high"))
    if m is None:
        return DASH
    if lo is None or hi is None:
        return fmt(m, nd)
    return f"{fmt(m, nd)}\\,{{\\scriptsize[{fmt(lo, nd)},{fmt(hi, nd)}]}}"


def delta(runs, group, name, ref, key, nd=1) -> str:
    p = f"delta_vs_{ref}/{key}"
    m, lo, hi = (get(runs, group, name, k) for k in (p, f"{p}_ci_low", f"{p}_ci_high"))
    if m is None:
        return DASH
    return f"{m:+.{nd}f}\\,{{\\scriptsize[{lo:+.{nd}f},{hi:+.{nd}f}]}}"


def write(out: Path, name: str, text: str) -> None:
    (out / name).write_text(text.rstrip() + "\n", encoding="utf-8")


def export(results: Path, out: Path, ds: str = "MMCoQA") -> None:
    out.mkdir(parents=True, exist_ok=True)
    L = load(results, "lilac-repro")
    M = load(results, "m3kg-rag-repro")
    n = lambda name: f"{ds}-{name}"  # noqa: E731  (LILaC run names carry the dataset)

    # ---- LILaC: main row --------------------------------------------------
    g = "main"
    write(out, "lilac_main_row.tex",
          f"\\lilac{{}}, Qwen2.5-7B-AWQ (ours) & {fmt(get(L, g, n('lilac_full'), 'paper_R@3'))} & "
          f"{fmt(get(L, g, n('lilac_full'), 'paper_MRR@10'))} & {fmt(get(L, g, n('lilac_full'), 'strict_R@3'))} & "
          f"{fmt(get(L, g, n('lilac_full'), 'strict_MRR@10'))} \\\\")

    # ---- LILaC: ablation ---------------------------------------------------
    g = "ablation"
    paper = {"wo_lcg_qd": ("47.94", "43.75"), "wo_qd": ("50.18", "45.59"), "full": ("55.80", "50.77")}
    labels = {"wo_lcg_qd": "w/o LCG \\& QD (kNN)", "wo_qd": "w/o QD (top-down re-rank)", "full": "LILaC (full)",
              "full_agnostic": "full, modality-agnostic sub-queries$^\\dagger$",
              "full_novlm": "full, no VLM text in image nodes$^\\dagger$"}
    rows = []
    for k, lab in labels.items():
        pr = paper.get(k, (DASH, DASH))
        rows.append(f"{lab} & {pr[0]} & {pr[1]} & {fmt(get(L, g, n(k), 'paper_R@3'))} & "
                    f"{fmt(get(L, g, n(k), 'paper_MRR@10'))} & {fmt(get(L, g, n(k), 'strict_R@3'))} & "
                    f"{delta(L, g, n(k), 'wo_lcg_qd', 'paper_R@3') if k != 'wo_lcg_qd' else DASH} \\\\")
    write(out, "lilac_ablation_rows.tex", "\n".join(rows))

    # ---- LILaC: breakdown (full vs kNN) ----------------------------------
    rows = []
    for key, lab in [("docs=single", "evidence in one webpage"), ("docs=multi", "evidence across webpages"),
                     ("evidence=single", "one gold component"), ("evidence=multi", "$\\geq$2 gold components")]:
        cnt = get(L, g, n("full"), f"{key}/n")
        rows.append(f"{lab} & {fmt(cnt, 0)} & {fmt(get(L, g, n('wo_lcg_qd'), f'{key}/paper_R@3'))} & "
                    f"{fmt(get(L, g, n('wo_qd'), f'{key}/paper_R@3'))} & {fmt(get(L, g, n('full'), f'{key}/paper_R@3'))} \\\\")
    write(out, "lilac_breakdown_rows.tex", "\n".join(rows))

    # ---- LILaC: beam width (pgfplots) ----------------------------------------
    lines = ["b R3 MRR strictR3 ms"]
    for b in (1, 2, 3, 4, 5, 10, 20, 30):
        s = L.get("sens_beam", {}).get(n(f"b{b}"), {}).get("summary", {})
        if s:
            lines.append(f"{b} {s.get('paper_R@3', 'nan')} {s.get('paper_MRR@10', 'nan')} "
                         f"{s.get('strict_R@3', 'nan')} {s.get('latency_ms_mean', 'nan')}")
    write(out, "lilac_beam.dat", "\n".join(lines))

    # ---- LILaC: iterations ----------------------------------------------------
    g = "sens_iter"
    labels = [("ni0_b5", "$n_i=0$ (kNN only)", "5"), ("ni1_b5", "$n_i=1$", "5"), ("ni2_b5", "$n_i=2$", "5"),
              ("official_script_ni2_b5", "official script, ``$n_i=2$''", "5"),
              ("ni1_b30", "$n_i=1$", "30"), ("ni2_b30", "$n_i=2$", "30"), ("ni3_b30", "$n_i=3$", "30")]
    rows = [f"{lab} & {b} & {fmt(get(L, g, n(k), 'paper_R@3'))} & {fmt(get(L, g, n(k), 'paper_MRR@10'))} & "
            f"{fmt(get(L, g, n(k), 'identical_top10_to_ni1_b5'))} \\\\" for k, lab, b in labels]
    write(out, "lilac_iter_rows.tex", "\n".join(rows))

    # ---- LILaC: decomposers -----------------------------------------------------
    g = "decomposer"
    rows = []
    for name, d in sorted(L.get(g, {}).items()):
        meta = d.get("config", {}).get("decomposition_meta", {})
        s = d["summary"]
        rows.append(f"{meta.get('model', name).split('/')[-1].replace('_', '-')} & {fmt(s.get('decomp/DAcc'))} & "
                    f"{fmt(s.get('decomp/avg_subqueries'), 2)} & {fmt(s.get('paper_R@3'))} & "
                    f"{fmt(meta.get('decomposition_ms_per_query'), 0)} \\\\")
    write(out, "lilac_decomp_rows.tex", "\n".join(rows) if rows else
          f"Qwen2.5-7B-AWQ & {DASH} & {DASH} & {DASH} & {DASH} \\\\\n"
          f"Qwen2.5-14B-AWQ & {DASH} & {DASH} & {DASH} & {DASH} \\\\")

    # ---- M3KG: KG statistics ----------------------------------------------------
    def _minutes(st):
        t = [v for k, v in st.items() if k.startswith("time/") and k != "time/search_s"]
        return sum(t) / 60 if t else None

    s = M.get("kg", {}).get("kg-build", {}).get("summary", {})
    kg_rows = [("clips in corpus", s.get("n_clips"), 0), ("triples", s.get("n_triplets"), 0),
               ("triples per clip", s.get("triplets_per_clip"), 2), ("unique concepts", s.get("n_concepts"), 0),
               ("concepts found in Wikipedia/Wiktionary (\\%)", _pct(s.get("concepts_kb_hit_rate")), 1),
               ("descriptions accepted at first inspection (\\%)", _pct(s.get("inspector_first_pass_rate")), 1),
               ("descriptions discarded by the inspector (\\%)", _pct(s.get("inspector_discard_rate")), 1),
               ("concepts shared by $\\geq$2 clips (\\%)", _pct(s.get("concepts_shared_by_clips_frac")), 1),
               ("largest connected component (\\% of nodes)", _pct(s.get("largest_component_frac")), 1),
               ("captions rewritten with video metadata", s.get("n_rewritten_captions"), 0),
               ("LLM calls (all agents)", s.get("llm_calls"), 0),
               ("construction time, GPU stages (min)", _minutes(s), 1)]
    write(out, "m3kg_kg_rows.tex", "\n".join(f"{a} & {fmt(v, nd)} \\\\" for a, v, nd in kg_rows))

    # ---- M3KG: main --------------------------------------------------------------
    g = "main"
    rows = [f"No retrieval & 49.00 & {ci(M, g, 'no_rag', 'MJ')} & {DASH} \\\\",
            f"Shared-space text-KG RAG & {DASH} & {ci(M, g, 'shared_space', 'MJ')} & {delta(M, g, 'shared_space', 'no_rag', 'MJ')} \\\\",
            f"Neighbour-caption RAG$^\\dagger$ & {DASH} & {ci(M, g, 'caption_rag', 'MJ')} & {delta(M, g, 'caption_rag', 'no_rag', 'MJ')} \\\\",
            f"\\mkg{{}} (full) & 60.77 & {ci(M, g, 'm3kg_rag', 'MJ')} & {delta(M, g, 'm3kg_rag', 'no_rag', 'MJ')} \\\\"]
    write(out, "m3kg_main_rows.tex", "\n".join(rows))

    # ---- M3KG: ablation + GRASP + hops ---------------------------------------------
    def mrow(group, name, label, ref):
        return (f"{label} & {ci(M, group, name, 'MJ')} & {fmt(get(M, group, name, 'ctx/final_mean'))} & "
                f"{delta(M, group, name, ref, 'MJ') if name != ref else DASH} \\\\")

    write(out, "m3kg_ablation_rows.tex", "\n".join([
        mrow("ablation", "abl_shared", "\\ding{55} & \\ding{55}", "abl_shared"),
        mrow("ablation", "abl_shared_grasp", "\\ding{55} & \\ding{51}", "abl_shared"),
        mrow("ablation", "abl_mw", "\\ding{51} & \\ding{55}", "abl_shared"),
        mrow("ablation", "abl_mw_grasp", "\\ding{51} & \\ding{51}", "abl_shared")]))
    write(out, "m3kg_grasp_rows.tex", "\n".join([
        mrow("grasp", "grasp_none", "no GRASP", "grasp_none"),
        mrow("grasp", "grasp_tag", "TAG grounding only", "grasp_none"),
        mrow("grasp", "grasp_llm", "LLM filter only", "grasp_none"),
        mrow("grasp", "grasp_tag_llm", "TAG + LLM filter (full)", "grasp_none"),
        "\\midrule",
        mrow("hops", "hops0", "full, 0 hops (lifted triples only)", "hops0"),
        mrow("hops", "hops1", "full, 1 hop (default)", "hops0"),
        mrow("hops", "hops2", "full, 2 hops", "hops0")]))

    # ---- M3KG: sensitivity data ---------------------------------------------------
    lines = ["q tau MJ lo hi empty"]
    for qk, qv in (("q25", 25), ("q50", 50), ("q75", 75), ("q90", 90), ("none", 100)):
        d = M.get("sens_tau", {}).get(f"tau_{qk}")
        if d:
            s = d["summary"]
            lines.append(f"{qv} {d['config'].get('tau_value', 'nan')} {s.get('MJ', 'nan')} {s.get('MJ_ci_low', 'nan')} "
                         f"{s.get('MJ_ci_high', 'nan')} {s.get('ctx/empty_rate', 'nan')}")
    write(out, "m3kg_tau.dat", "\n".join(lines))
    lines = ["eta MJ lo hi final"]
    for name, d in sorted(M.get("sens_eta", {}).items(), key=lambda kv: kv[1]["config"].get("eta") or 0):
        s = d["summary"]
        lines.append(f"{d['config'].get('eta')} {s.get('MJ', 'nan')} {s.get('MJ_ci_low', 'nan')} "
                     f"{s.get('MJ_ci_high', 'nan')} {s.get('ctx/final_mean', 'nan')}")
    write(out, "m3kg_eta.dat", "\n".join(lines))

    # ---- W&B links (placeholders unless make_wandb_reports.py was run) ------------
    links = results / "wandb_links.json"
    macros = ["wbLilacMain", "wbLilacAblation", "wbLilacBeam", "wbLilacIter", "wbLilacDecomp", "wbMkgKG",
              "wbMkgMain", "wbMkgAblation", "wbMkgGrasp", "wbMkgHops", "wbMkgTau", "wbMkgEta"]
    have = json.loads(links.read_text()) if links.exists() else {}
    pending = r"\textit{[W\&B link pending]}"
    lines = []
    for m in macros:
        body = (r"\url{" + have[m] + "}") if m in have else pending
        lines.append(r"\newcommand{\%s}{%s}" % (m, body))
    write(out, "wandb_links.tex", "\n".join(lines))
    print(f"exported LaTeX fragments to {out}")


def _pct(x):
    return None if x is None else 100.0 * x


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", default="MMCoQA")
    ap.add_argument("--placeholders", action="store_true", help="ignore results and write empty tables")
    a = ap.parse_args()
    export(Path("/nonexistent") if a.placeholders else Path(a.results), Path(a.out), a.dataset)


if __name__ == "__main__":
    main()
