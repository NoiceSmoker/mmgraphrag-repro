"""Official step 3 sentence splitting runs in-process on one GPU (no fork after CUDA init)."""
import json
import types

import pytest

from mmrag.lilac import official, official_steps


def test_official_parallel_split_runs_inline(tmp_path):
    try:
        official.ensure_importable()
        import src.lilac.lcg_constructor.step3_add_sentences as s3
    except Exception as exc:  # official code not fetched
        pytest.skip(f"official LILaC not available: {exc}")
    text = tmp_path / "text.json"
    text.write_text(json.dumps({"a.json": {"p_0": {"text": "One. Two."}, "p_1": {"text": "Three."}},
                                "b.json": {"p_0": {"text": "Four. Five."}}}))
    out = tmp_path / "sentence.json"
    calls = []

    def worker(self, model_id, gpu_id, items, return_dict):   # stands in for the SaT worker
        calls.append((model_id, gpu_id, len(items)))
        part = {}
        for outer, inner, t in items:
            part.setdefault(outer, {})[inner] = [x.strip() + "." for x in t.split(".") if x.strip()]
        return_dict[model_id] = part

    stub = types.SimpleNamespace(_text_component_path=str(text), _sentence_component_path=str(out))
    stub.worker = types.MethodType(worker, stub)
    orig_mp = s3.mp
    try:
        s3.mp = official_steps._InlineMP
        s3.SentenceAdder.parallel_sentence_split(stub, 1, 1)
    finally:
        s3.mp = orig_mp
    assert calls == [(0, 0, 3)]
    assert json.loads(out.read_text()) == {"a.json": {"p_0": ["One.", "Two."], "p_1": ["Three."]},
                                           "b.json": {"p_0": ["Four.", "Five."]}}
