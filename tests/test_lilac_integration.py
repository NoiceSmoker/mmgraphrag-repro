"""End-to-end check of the retrieval session against the *official* LILaC
retriever on a synthetic corpus (no models; runs in ~20 s on CPU)."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from conftest import REPO, needs_lilac


@needs_lilac
def test_session_runs_official_retriever(tmp_path):
    src_root = Path(os.environ.get("LILAC_ROOT", REPO / "third_party" / "lilac")).resolve()
    root = tmp_path / "lilac"
    root.mkdir()
    for sub in ("src", "config"):
        shutil.copytree(src_root / sub, root / sub)
    env = {**os.environ, "LILAC_ROOT": str(root), "MMRAG_SRC": str(REPO / "src"), "WANDB_MODE": "disabled"}
    proc = subprocess.run([sys.executable, str(Path(__file__).with_name("lilac_integration_driver.py"))],
                          env=env, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-4000:]
    line = next(l for l in proc.stdout.splitlines() if l.startswith("RESULT_JSON="))
    res = json.loads(line.split("=", 1)[1])

    # every variant produced output for all 8 questions
    assert all(v["n"] == 8 for v in res.values())
    # graph traversal with good sub-queries finds both hops; plain kNN cannot beat it
    assert res["ablation/full"]["paper_R@3"] >= res["ablation/wo_lcg_qd"]["paper_R@3"]
    assert res["ablation/full"]["paper_R@3"] > 50
    # the official sensitivity-script call (late_interaction) ignores num_iterations
    assert res["sens_iter/official_script_ni2_b5"]["top"] == res["sens_iter/ni1_b5"]["top"]
    # switching the decomposition variant changes the retrieval
    assert res["decomposer/decomp_good"]["paper_R@3"] > res["decomposer/decomp_noisy"]["paper_R@3"]
