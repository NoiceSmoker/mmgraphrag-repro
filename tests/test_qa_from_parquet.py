"""dev.parquet (Hugging Face release) -> QAs_dev_labeled.json in the official schema."""
import json
import subprocess
import sys

import pandas as pd
import pytest

from conftest import REPO
from mmrag.lilac import qa_from_parquet as q
from mmrag.lilac.metrics import load_gold


@pytest.fixture
def ds(tmp_path):
    d = tmp_path / "MMCoQA"
    docs = d / "parsed_documents" / "dev"
    docs.mkdir(parents=True)
    # file stem shortened by load.py, full title inside the JSON
    (docs / "Eiffel Tower.json").write_text(json.dumps({"title": "Eiffel Tower", "text": {
        "p_0": {"text": "a", "label_id": 0}, "p_1": {"text": "b", "label_id": 1}},
        "table": {"t_0": {"table": [], "label_id": 2}}, "image": {"i_0": {"image_name": "x.png", "label_id": 3}}}))
    (docs / "A very long title that load.py cut to fifty charac.json").write_text(json.dumps({
        "title": "A very long title that load.py cut to fifty characters and beyond",
        "text": {"p_0": {"text": "c", "label_id": 0}}, "table": {}, "image": {}}))
    return d


def test_nested_schema_with_title_remap(ds):
    df = pd.DataFrame([
        {"qid": "q1", "question": "Where?", "answers": [{"answer": "Paris"}],
         "evidences": [{"mmqa_doc_modality": "text", "gold_webpage_title": "Eiffel Tower", "gold_component_id": "p_1"},
                       {"mmqa_doc_modality": "image", "gold_webpage_title": "Eiffel Tower", "gold_component_id": "i_0"}]},
        {"qid": "q2", "question": "Long?", "answers": [{"answer": "yes"}],
         "evidences": [{"mmqa_doc_modality": "text",
                        "gold_webpage_title": "A very long title that load.py cut to fifty characters and beyond",
                        "gold_component_id": "p_0"}]},
    ])
    df.to_parquet(ds / "dev.parquet")
    out = subprocess.run([sys.executable, "-m", "mmrag.lilac.qa_from_parquet", "--dataset-dir", str(ds)],
                         capture_output=True, text=True, env={"PYTHONPATH": str(REPO / "src")})
    assert out.returncode == 0, out.stdout + out.stderr
    assert "100.0% found" in out.stdout
    gold = load_gold(ds / "QAs_dev_labeled.json")
    assert gold["q1"].components == [("Eiffel Tower", "p_1"), ("Eiffel Tower", "i_0")]
    assert gold["q2"].components == [("A very long title that load.py cut to fifty charac", "p_0")]


def test_label_id_and_flat_columns(ds):
    comps, t2s, by_label = q.load_documents(ds / "parsed_documents" / "dev")
    recs = [{"id": 7, "query": "Which table?", "answer": "x", "doc_title": ["Eiffel Tower"], "label_id": [2]}]
    out, st = q.convert(recs, comps, t2s, by_label)
    assert out[0]["evidences"] == [{"mmqa_doc_modality": "table", "gold_webpage_title": "Eiffel Tower",
                                    "gold_component_id": "t_0"}]
    assert st["matched"] == 1 and st["via_label_id"] == 1
    assert out[0]["answers"] == [{"answer": "x"}] and out[0]["qid"] == "7"


def test_refuses_to_write_when_labels_do_not_match(ds):
    pd.DataFrame([{"qid": "q1", "question": "?", "answers": [],
                   "evidences": [{"gold_webpage_title": "Nope", "gold_component_id": "p_9"}]}]).to_parquet(ds / "dev.parquet")
    out = subprocess.run([sys.executable, "-m", "mmrag.lilac.qa_from_parquet", "--dataset-dir", str(ds)],
                         capture_output=True, text=True, env={"PYTHONPATH": str(REPO / "src")})
    assert out.returncode != 0 and not (ds / "QAs_dev_labeled.json").exists()


def test_real_hf_schema_pairs_and_underscored_titles(ds):
    """dev.parquet of JoohyungYun/mmcoqa_doc: qid, question, answer (str), evidence [[doc, comp]]."""
    pd.DataFrame([
        {"qid": "C_381_0", "question": "Who?", "answer": "Brian McKnight",
         "evidence": [["Eiffel_Tower", "t_0"], ["Eiffel_Tower", "p_1"]]},
        {"qid": "C_1_0", "question": "Missing?", "answer": "n/a", "evidence": [["Unknown_Page", "p_0"]]},
    ]).to_parquet(ds / "dev.parquet")
    out = subprocess.run([sys.executable, "-m", "mmrag.lilac.qa_from_parquet", "--dataset-dir", str(ds)],
                         capture_output=True, text=True, env={"PYTHONPATH": str(REPO / "src")})
    assert out.returncode == 0, out.stdout + out.stderr
    assert "3 gold evidences, 66.7% found" in out.stdout and "WARNING" in out.stdout
    data = json.loads((ds / "QAs_dev_labeled.json").read_text())
    assert data[0]["answers"] == [{"answer": "Brian McKnight"}]
    assert data[0]["evidences"][0] == {"mmqa_doc_modality": "table", "gold_webpage_title": "Eiffel Tower",
                                       "gold_component_id": "t_0"}
    assert data[1]["evidences"][0]["gold_webpage_title"] == "Unknown_Page"   # kept, like the official file
