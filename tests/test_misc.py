import json
import sys

import numpy as np
import pandas as pd
import pytest

from mmrag.common.stats import bootstrap_mean, paired_bootstrap_diff
from mmrag.common.tracking import tracked_run
from mmrag.lilac.decomp_eval import decomposition_report, jaccard
from mmrag.lilac.metrics import GoldQA
from mmrag.lilac.variants import GROUPS
from mmrag.lilac.vlm_batch import extract_json_objects, rescale_box, smart_resize
from mmrag.m3kg.experiments import all_configs


def test_smart_resize_and_box_rescale():
    assert smart_resize(1000, 2000, max_pixels=12845056) == (1008, 1988)
    h, w = smart_resize(4000, 3000, max_pixels=1280 * 28 * 28)
    assert h * w <= 1280 * 28 * 28 and h % 28 == 0 and w % 28 == 0
    # a box covering the whole resized image maps to the whole original image
    assert rescale_box([0, 0, w, h], (3000, 4000), (h, w)) == (0, 0, 3000, 4000)
    assert rescale_box([10, 20, 5, 1], (100, 100), (50, 50)) == (10, 2, 20, 40)  # swapped corners fixed


def test_extract_json_objects():
    s = 'text [{"class": "dog", "bbox_2d": [1, 2, 3, 4]}, {"class": "x", "bbox_2d": [1,2]} {bad}'
    objs = extract_json_objects(s)
    assert objs[0]["class"] == "dog" and len(objs) == 2


def test_bootstrap():
    ci = bootstrap_mean([1.0] * 10)
    assert ci.mean == ci.low == ci.high == 1.0
    d, p = paired_bootstrap_diff([1, 1, 1, 1], [0, 0, 0, 0])
    assert d.mean == 1 and p == 0.0
    with pytest.raises(ValueError):
        paired_bootstrap_diff([1, 2], [1])


def test_decomposition_accuracy():
    gold = {"q1": GoldQA("q1", "?", [("A", "p_0"), ("B", "i_0")], ["text", "image"]),
            "q2": GoldQA("q2", "?", [("A", "t_0")], ["table"])}
    subq = {"q1": {"subqueries": [{"modality": "text"}]}, "q2": {"subqueries": [{"modality": "table"}, {"modality": "table"}]}}
    s, rows = decomposition_report(subq, gold)
    assert jaccard({"text"}, {"text", "image"}) == 0.5
    assert s["DAcc"] == pytest.approx(75.0) and s["avg_subqueries"] == 1.5 and s["single_subquery_rate"] == 50.0


def test_variant_and_config_names_unique():
    names = [(v.group, v.name) for f in GROUPS.values() for v in f()]
    assert len(names) == len(set(names))
    cfgs = all_configs()
    assert len({c.name for c in cfgs}) == len(cfgs)
    assert {"no_rag", "shared_space", "m3kg_rag", "eta_0.5", "tau_q75", "hops0"} <= {c.name for c in cfgs}


def test_tracking_writes_local_json(tmp_path):
    import mmrag.common.tracking as tr

    with tracked_run("proj", "grp", "run1", {"a": 1}) as run:
        run.set_summary({"x": 1.5})
        run.log_table("t", [{"q": 1}])
    out = json.loads((tr.RESULTS_DIR / "proj" / "grp" / "run1.json").read_text())
    assert out["summary"]["x"] == 1.5 and out["config"]["a"] == 1 and out["tables"]["t"] == [{"q": 1}]


def test_m3kg_report_aggregates(tmp_path, monkeypatch):
    from mmrag.common.io import write_json, write_jsonl
    from mmrag.m3kg import report
    import mmrag.common.tracking as tr

    work = tmp_path / "w"
    for name, group, scores in (("no_rag", "main", [2, 3, 3, 4]), ("m3kg_rag", "main", [3, 4, 4, 5])):
        write_json({"name": name, "group": group}, work / "contexts" / f"{name}.config.json")
        write_jsonl([{"qid": f"q{i}", "candidates": [1, 2], "after_grounding": [1], "final": [1] if name != "no_rag" else [],
                      "clips": []} for i in range(4)], work / "contexts" / f"{name}.jsonl")
        write_jsonl([{"qid": f"q{i}", "prediction": "p", "n_triples": 1, "score_0_5": s, "parse": "audiobench"}
                     for i, s in enumerate(scores)], work / "judgements" / f"{name}.jsonl")
    base = tmp_path / "base.yaml"
    base.write_text(f"""paths: {{work_dir: {work}}}
data: {{corpus_size: 10}}
kg: {{llm: x}}
retrieval: {{clap_model: y}}
generation: {{model: z}}
wandb: {{project: m3kg-test}}
""")
    monkeypatch.setattr(sys, "argv", ["report", "--config", str(base)])
    report.main()
    r = json.loads((tr.RESULTS_DIR / "m3kg-test" / "main" / "m3kg_rag.json").read_text())["summary"]
    assert r["MJ"] == pytest.approx(80.0) and r["delta_vs_no_rag/MJ"] == pytest.approx(20.0)
    assert r["ctx/final_mean"] == 1.0 and r["ctx/grounding_removed_pct"] == 50.0
