import json

import export_results
import make_wandb_reports


def test_export_placeholders(tmp_path):
    export_results.export(tmp_path / "none", tmp_path / "out")
    files = {p.name for p in (tmp_path / "out").iterdir()}
    assert {"lilac_main_row.tex", "lilac_ablation_rows.tex", "lilac_beam.dat", "m3kg_main_rows.tex",
            "m3kg_tau.dat", "wandb_links.tex"} <= files
    assert "--" in (tmp_path / "out" / "lilac_main_row.tex").read_text()
    assert "\\newcommand{\\wbLilacMain}" in (tmp_path / "out" / "wandb_links.tex").read_text()


def test_export_with_results(tmp_path):
    res = tmp_path / "results"
    d = res / "lilac-repro" / "main"
    d.mkdir(parents=True)
    (d / "MMCoQA-lilac_full.json").write_text(json.dumps(
        {"group": "main", "name": "MMCoQA-lilac_full", "config": {},
         "summary": {"paper_R@3": 54.321, "paper_MRR@10": 49.0, "strict_R@3": 30.0, "strict_MRR@10": 40.0}}))
    sb = res / "lilac-repro" / "sens_beam"
    sb.mkdir(parents=True)
    (sb / "MMCoQA-b5.json").write_text(json.dumps({"group": "sens_beam", "name": "MMCoQA-b5", "config": {},
                                                   "summary": {"paper_R@3": 50.0}}))
    (res / "wandb_links.json").write_text(json.dumps({"wbLilacMain": "https://wandb.ai/x/y/reports/z"}))
    export_results.export(res, tmp_path / "out")
    assert "54.3" in (tmp_path / "out" / "lilac_main_row.tex").read_text()
    assert "5 50.0" in (tmp_path / "out" / "lilac_beam.dat").read_text()
    assert "\\url{https://wandb.ai/x/y/reports/z}" in (tmp_path / "out" / "wandb_links.tex").read_text()


def test_export_breakdown_and_kg_cost(tmp_path):
    """Coinciding evidence/page splits are merged; a resumed KG run reports total calls, no time."""
    res = tmp_path / "results"
    abl = res / "lilac-repro" / "ablation"
    abl.mkdir(parents=True)
    split = {"evidence=single/n": 9, "evidence=single/paper_R@3": 50.0, "evidence=multi/n": 2,
             "evidence=multi/paper_R@3": 40.0, "docs=single/n": 9, "docs=single/paper_R@3": 50.0,
             "docs=multi/n": 2, "docs=multi/paper_R@3": 40.0, "modality=image/n": 3, "modality=image/paper_R@3": 10.0}
    for v in ("wo_lcg_qd", "wo_qd", "full", "full_novlm"):
        (abl / f"MMCoQA-{v}.json").write_text(json.dumps(
            {"group": "ablation", "name": f"MMCoQA-{v}", "config": {}, "summary": split}))
    kg = res / "m3kg-rag-repro" / "kg"
    kg.mkdir(parents=True)
    (kg / "kg-build.json").write_text(json.dumps(
        {"group": "kg", "name": "kg-build", "config": {},
         "summary": {"n_clips": 100, "n_clips_without_triples": 20, "llm_calls": 10, "llm_cached": 990,
                     "time/extract_s": 6.0, "entity_desc_src": {"none": 3, "wikipedia": 1}}}))
    export_results.export(res, tmp_path / "out")
    rows = (tmp_path / "out" / "lilac_breakdown_rows.tex").read_text()
    assert "across web pages" not in rows and "$\\geq$2 pages & 2 & 40.0 & 40.0 & 40.0 & 40.0" in rows
    assert "image evidence & 3 & 10.0" in rows
    kg_rows = (tmp_path / "out" / "m3kg_kg_rows.tex").read_text()
    assert "LLM calls (all agents) & 1000" in kg_rows and "construction time" not in kg_rows
    assert "clips without any triple (\\%) & 20.0" in kg_rows
    assert "final description (\\%) & 25.0" in kg_rows


def test_wandb_report_objects_build(monkeypatch):
    import wandb_workspaces.reports.v2 as wr

    saved = []
    monkeypatch.setattr(wr.Report, "save", lambda self, *a, **k: saved.append(self.title) or self)
    monkeypatch.setattr(wr.Report, "url", property(lambda self: f"https://wandb.ai/r/{len(saved)}"))
    links = make_wandb_reports.build("me", {}, share=False)
    assert len(links) == len(make_wandb_reports.SECTIONS) == len(saved)
    only = make_wandb_reports.build("me", {}, share=False, only="m3kg")
    assert set(only) == {m for m, p, *_ in make_wandb_reports.SECTIONS if p.startswith("m3kg")}


def test_colab_sync_scopes(tmp_path):
    """Each notebook moves only its own artefacts to Drive."""
    import os
    import subprocess

    from conftest import REPO

    lilac = tmp_path / "lilac"
    (lilac / "artifacts" / "MMCoQA").mkdir(parents=True)
    (lilac / "artifacts" / "MMCoQA" / "a.txt").write_text("a")
    work = REPO / "work"
    made_work = not work.exists()
    (tmp_path / "drive").mkdir()
    env = dict(os.environ, LILAC_ROOT=str(lilac), DRIVE_DIR=str(tmp_path / "drive" / "store"),
               MMRAG_RESULTS_DIR=str(tmp_path / "res"))
    try:
        out = subprocess.run(["bash", str(REPO / "scripts/colab/sync.sh"), "push", "lilac"], env=env,
                             capture_output=True, text=True)
        assert out.returncode == 0, out.stderr
        store = tmp_path / "drive" / "store"
        assert (store / "lilac-artifacts-MMCoQA.tar").is_file() and not list(store.glob("m3kg-*"))
        # pull restores the directory, but never overwrites newer local files
        (lilac / "artifacts" / "MMCoQA" / "a.txt").write_text("newer")
        (lilac / "artifacts" / "MMCoQA" / "b.txt").write_text("b")
        import shutil
        shutil.rmtree(lilac / "artifacts" / "MMCoQA" / "sub", ignore_errors=True)
        out = subprocess.run(["bash", str(REPO / "scripts/colab/sync.sh"), "pull", "lilac"], env=env,
                             capture_output=True, text=True)
        assert out.returncode == 0, out.stderr
        assert (lilac / "artifacts" / "MMCoQA" / "a.txt").read_text() == "newer"
        (lilac / "artifacts" / "MMCoQA" / "a.txt").unlink()
        subprocess.run(["bash", str(REPO / "scripts/colab/sync.sh"), "pull", "lilac"], env=env, check=True,
                       capture_output=True)
        assert (lilac / "artifacts" / "MMCoQA" / "a.txt").read_text() == "a"
        bad = subprocess.run(["bash", str(REPO / "scripts/colab/sync.sh"), "push", "nope"], env=env,
                             capture_output=True, text=True)
        assert bad.returncode != 0
    finally:
        if made_work and work.exists():
            work.rmdir()
