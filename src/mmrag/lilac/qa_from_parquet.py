"""Build ``QAs_dev_labeled.json`` from the ``dev.parquet`` of the Hugging Face
MMCoQA-Doc / MultimodalQA-Doc release.

The official LILaC repository expects ``datasets/<DS>/QAs_dev_labeled.json``
(schema ``mmqa_component``: qid, question, answers[{answer}], evidences[{
mmqa_doc_modality, gold_webpage_title, gold_component_id, gold_component_score}])
but does not ship it (``datasets/`` is git-ignored). The Hugging Face release
contains the dev questions as ``dev.parquet`` instead. This script converts it
and checks every gold evidence against the restored documents in
``parsed_documents/dev``:

* document names are mapped to the file stems the retriever reports
  (``load.py`` shortens and sanitises long titles);
* evidences given by ``label_id`` instead of a component key are resolved
  through the documents' ``label_id`` fields.

It always prints the parquet schema and the share of gold evidences found in
the documents, and refuses to write a file if most of them cannot be matched.

    python -m mmrag.lilac.qa_from_parquet --dataset-dir third_party/lilac/datasets/MMCoQA
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

QID_COLS = ("qid", "id", "question_id", "qa_id")
QUESTION_COLS = ("question", "query", "rewritten_question", "q")
ANSWER_COLS = ("answers", "answer", "gold_answers")
EVIDENCE_COLS = ("evidences", "evidence", "gold_evidences", "supporting_facts", "labels")
TITLE_KEYS = ("gold_webpage_title", "webpage_title", "doc_title", "document_title", "title", "document")
COMP_KEYS = ("gold_component_id", "component_id", "component", "cid")
LABEL_KEYS = ("label_id", "gold_label_id")
MODALITY_KEYS = ("mmqa_doc_modality", "modality", "doc_modality")
SCORE_KEYS = ("gold_component_score", "score", "component_score")
MODALITY_OF_PREFIX = {"p": "text", "t": "table", "i": "image"}


def _modality(m, comp: str | None) -> str:
    """Normalise to the official Modality values: text | table | image."""
    m = str(m or "").strip().lower()
    for name, aliases in (("text", ("text", "txt", "p", "passage")), ("table", ("table", "tab", "t")),
                          ("image", ("image", "img", "i", "picture"))):
        if m in aliases or (m and m.startswith(name)):
            return name
    return MODALITY_OF_PREFIX.get((comp or "").split("_")[0], "text")


def _first(d: dict, keys) -> object:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def _maybe_json(v):
    if isinstance(v, str) and v[:1] in "[{":
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            return v
    return v


def load_documents(doc_dir: Path) -> tuple[dict, dict, dict]:
    """stem -> {component keys}, title -> stem, (stem, label_id) -> component key."""
    comps, title_to_stem, by_label = {}, {}, {}
    for p in sorted(doc_dir.glob("*.json")):
        doc = json.loads(p.read_text())
        keys = set()
        for kind in ("text", "table", "image"):
            for cid, comp in (doc.get(kind) or {}).items():
                keys.add(cid)
                if isinstance(comp, dict) and comp.get("label_id") is not None:
                    by_label[(p.stem, str(comp["label_id"]))] = cid
        comps[p.stem] = keys
        if doc.get("title"):
            title_to_stem.setdefault(doc["title"], p.stem)
    return comps, title_to_stem, by_label


def _answers(v) -> list[dict]:
    v = _maybe_json(v)
    if v is None:
        return []
    if isinstance(v, dict):
        v = [v]
    if not isinstance(v, list):
        v = [v]
    out = []
    for a in v:
        a = _maybe_json(a)
        if isinstance(a, dict):
            out.append({**a, "answer": a.get("answer", a.get("text", ""))})
        else:
            out.append({"answer": a})
    return out


def convert(records: list[dict], comps: dict, title_to_stem: dict, by_label: dict) -> tuple[list[dict], dict]:
    cols = set(records[0]) if records else set()
    qid_c = next((c for c in QID_COLS if c in cols), None)
    q_c = next((c for c in QUESTION_COLS if c in cols), None)
    a_c = next((c for c in ANSWER_COLS if c in cols), None)
    e_c = next((c for c in EVIDENCE_COLS if c in cols), None)
    flat = None
    if e_c is None:  # evidences stored as parallel top-level columns (lists or scalars)
        t_c = next((c for c in TITLE_KEYS if c in cols), None)
        c_c = next((c for c in COMP_KEYS + LABEL_KEYS if c in cols), None)
        if t_c and c_c:
            m_c = next((c for c in MODALITY_KEYS if c in cols), None)
            flat = (t_c, c_c, m_c)
    if not (qid_c and q_c and (e_c or flat)):
        raise ValueError(f"unrecognised dev.parquet schema: columns {sorted(cols)}")
    out, stats = [], {"questions": 0, "evidences": 0, "matched": 0, "title_mapped": 0, "via_label_id": 0,
                      "unmatched_examples": []}
    for r in records:
        if e_c:
            evs = _maybe_json(r[e_c]) or []
        else:
            t_c, c_c, m_c = flat
            seq = lambda v: v if isinstance(v, list) else [v]
            ts, cs = seq(_maybe_json(r[t_c])), seq(_maybe_json(r[c_c]))
            ms = seq(_maybe_json(r[m_c])) if m_c else [None] * len(cs)
            evs = [{t_c: t, c_c: c, **({m_c: m} if m_c else {})} for t, c, m in zip(ts, cs, ms)]
        if isinstance(evs, dict):
            evs = [evs]
        new_evs = []
        for ev in evs:
            ev = _maybe_json(ev)
            if not isinstance(ev, dict):
                continue
            title = _first(ev, TITLE_KEYS)
            title = None if title is None else str(title)
            stem = title
            if title is not None and title not in comps and title in title_to_stem:
                stem = title_to_stem[title]
                stats["title_mapped"] += 1
            comp = _first(ev, COMP_KEYS)
            comp = None if comp is None else str(comp)
            if (comp is None or (stem in comps and comp not in comps[stem])) and stem is not None:
                label = _first(ev, LABEL_KEYS)
                if label is not None and (stem, str(label)) in by_label:
                    comp = by_label[(stem, str(label))]
                    stats["via_label_id"] += 1
            modality = _modality(_first(ev, MODALITY_KEYS), comp)
            stats["evidences"] += 1
            ok = stem in comps and comp in comps[stem]
            stats["matched"] += int(ok)
            if not ok and len(stats["unmatched_examples"]) < 5:
                stats["unmatched_examples"].append({"title": title, "component": comp})
            new_ev = {"mmqa_doc_modality": modality, "gold_webpage_title": stem,
                      "gold_component_id": comp}
            score = _first(ev, SCORE_KEYS)
            if score is not None:
                new_ev["gold_component_score"] = score
            new_evs.append(new_ev)
        out.append({"qid": str(r[qid_c]), "question": str(r[q_c]),
                    "answers": _answers(r[a_c]) if a_c else [], "evidences": new_evs})
        stats["questions"] += 1
    return out, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-dir", required=True, help="datasets/<DS> of the official repo")
    ap.add_argument("--parquet", default="dev.parquet")
    ap.add_argument("--out", default="QAs_dev_labeled.json")
    ap.add_argument("--min-match", type=float, default=0.9, help="minimum share of gold evidences found")
    a = ap.parse_args()

    import pandas as pd

    root = Path(a.dataset_dir)
    df = pd.read_parquet(root / a.parquet)
    print(f"[qa] {a.parquet}: {len(df)} rows, columns: {list(df.columns)}")
    print(f"[qa] dtypes: {dict(df.dtypes.astype(str))}")
    records = json.loads(df.to_json(orient="records"))  # numpy arrays / structs -> plain JSON
    if records:
        print("[qa] first row:", json.dumps(records[0], ensure_ascii=False)[:1500])
    comps, title_to_stem, by_label = load_documents(root / "parsed_documents" / "dev")
    print(f"[qa] {len(comps)} documents in parsed_documents/dev")
    out, st = convert(records, comps, title_to_stem, by_label)
    share = st["matched"] / max(st["evidences"], 1)
    print(f"[qa] {st['questions']} questions, {st['evidences']} gold evidences, {share:.1%} found in the documents "
          f"(titles remapped: {st['title_mapped']}, resolved via label_id: {st['via_label_id']})")
    if st["unmatched_examples"]:
        print("[qa] unmatched examples:", st["unmatched_examples"])
    if share < a.min_match:
        sys.exit(f"[qa] only {share:.1%} of the gold evidences match the documents - not writing {a.out}")
    (root / a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"[qa] wrote {root / a.out}")


if __name__ == "__main__":
    main()
