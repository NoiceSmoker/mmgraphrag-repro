"""The quantized MM-Embed loader is installed through sitecustomize in every
process of the mmembed env - including the spawn workers of the official
embedding scripts. Checked here with a fake ``src.models.embedder.mmembed``."""
import json
import os
import subprocess
import sys
import textwrap

import pytest
import torch

from conftest import REPO
from mmrag.lilac import mmembed_quant as mq

FAKE_MODULE = '''
import torch

class MMEmbed:
    def __init__(self, cuda_num=0, show_progress=False):
        self.cuda_num = cuda_num
        self.model_path = "unused"

    def load_model(self):
        raise RuntimeError("official fp16 loader called")

    def encode_corpus(self, data, max_length, batch_size, out_embeddings_path, out_index_path,
                      start_idx=None, end_idx=None):
        self.used_batch_size = batch_size
        out = torch.ones(batch_size, 3)
        out[0] = 0            # one failed row, as the official code writes it
        return out
'''

DRIVER = '''
import json, multiprocessing as mp, sys

def child(q):
    from src.models.embedder.mmembed import MMEmbed
    q.put(getattr(MMEmbed.load_model, "__mmrag_loader__", MMEmbed.load_model.__name__))

if __name__ == "__main__":
    from src.models.embedder.mmembed import MMEmbed
    enc = MMEmbed()
    enc.encode_corpus(data=[], max_length=8, batch_size=32, out_embeddings_path="x", out_index_path="y")
    used_bs = enc.used_batch_size
    try:
        enc.load_model()                      # quantized: fails later (no real model); guard: official loader
    except Exception as exc:
        print("LOAD:", exc)
    mp.set_start_method("spawn", force=True)
    q = mp.Queue()
    p = mp.Process(target=child, args=(q,))
    p.start(); p.join()
    print(json.dumps({"parent": getattr(MMEmbed.load_model, "__mmrag_loader__", MMEmbed.load_model.__name__),
                      "child": q.get(), "batch": used_bs}))
'''


@pytest.fixture
def fake_official(tmp_path):
    pkg = tmp_path / "src" / "models" / "embedder"
    pkg.mkdir(parents=True)
    for d in (tmp_path / "src", tmp_path / "src" / "models", pkg):
        (d / "__init__.py").write_text("")
    (pkg / "mmembed.py").write_text(FAKE_MODULE)
    (tmp_path / "driver.py").write_text(DRIVER)
    return tmp_path


def _run(fake_official, quant, guard=False):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(REPO / "src" / "mmrag" / "lilac" / "_site"), str(REPO / "src"),
                                         str(fake_official)])
    env.pop("MMEMBED_QUANT", None)
    env["HF_HUB_OFFLINE"] = "1"            # the patched loader must not reach the network in tests
    env.pop("MMEMBED_GUARD", None)
    if guard:
        env["MMEMBED_GUARD"] = "1"
    if quant or guard:
        env["MMEMBED_BATCH_SCALE"] = "0.5"
    if quant:
        env["MMEMBED_QUANT"] = quant
    out = subprocess.run([sys.executable, "driver.py"], cwd=fake_official, env=env, capture_output=True,
                         text=True, timeout=120)
    return out


def test_hook_reaches_spawn_workers_and_scales_batches(fake_official):
    out = _run(fake_official, "8bit")
    assert out.returncode == 0, out.stderr
    res = json.loads(out.stdout.strip().splitlines()[-1])
    assert res["parent"] == res["child"] == "load_model_quantized"
    assert res["batch"] == 16           # 32 * MMEMBED_BATCH_SCALE (0.5)
    assert "1 of 16 embeddings are zero vectors" in out.stdout


def test_no_hook_without_env(fake_official):
    out = _run(fake_official, None)
    assert out.returncode == 0, out.stderr
    res = json.loads(out.stdout.strip().splitlines()[-1])
    assert "LOAD: official fp16 loader called" in out.stdout       # official code untouched
    assert res["batch"] == 32 and "zero vectors" not in out.stdout


def test_quant_mode_parsing(monkeypatch):
    for v, want in (("", None), ("fp16", None), ("8bit", "8bit"), ("INT8", "8bit"), ("nf4", "4bit")):
        monkeypatch.setenv("MMEMBED_QUANT", v)
        assert mq.quant_mode() == want
    monkeypatch.setenv("MMEMBED_QUANT", "3bit")
    with pytest.raises(ValueError):
        mq.quant_mode()
    monkeypatch.setenv("MMEMBED_QUANT", "8bit")
    assert mq.describe()["mmembed_weights"] == "8bit"


def test_finalize_devices_rejects_meta_tensors():
    ok = torch.nn.Linear(4, 4)
    assert mq.finalize_devices(ok, "cpu") == []
    with torch.device("meta"):
        bad = torch.nn.Sequential(torch.nn.Linear(4, 4))
    with pytest.raises(RuntimeError, match="meta device"):
        mq.finalize_devices(bad, "cpu")


def test_sitecustomize_is_silent_without_env(tmp_path):
    env = dict(os.environ, PYTHONPATH=str(REPO / "src" / "mmrag" / "lilac" / "_site"))
    env.pop("MMEMBED_QUANT", None)
    code = textwrap.dedent("""
        import sys
        print(any(type(f).__name__ == '_PatchingFinder' for f in sys.meta_path))
    """)
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=60)
    assert out.stdout.strip() == "False" and out.stderr == ""


def test_oom_safe_encode_splits_and_keeps_order():
    calls = []

    def encode(inputs, instruction=None, max_length=8):
        calls.append(len(inputs))
        if len(inputs) > 2:
            raise torch.cuda.OutOfMemoryError("fake OOM")
        assert instruction is None or len(instruction) == len(inputs)
        return {"hidden_states": torch.tensor([[float(x)] for x in inputs])}

    enc = mq.oom_safe_encode(encode)
    out = enc(list(range(7)), instruction=[f"i{k}" for k in range(7)], max_length=8)
    assert out["hidden_states"].squeeze(1).tolist() == list(range(7))
    assert calls[0] == 7 and all(c <= 2 for c in calls[1:] if c not in (7, 3, 4))
    with pytest.raises(torch.cuda.OutOfMemoryError):
        mq.oom_safe_encode(lambda inputs, **k: (_ for _ in ()).throw(torch.cuda.OutOfMemoryError("x")))([1])


def test_guard_keeps_official_loader_but_scales_batches(fake_official):
    out = _run(fake_official, None, guard=True)
    assert out.returncode == 0, out.stderr
    res = json.loads(out.stdout.strip().splitlines()[-1])
    assert res["parent"] == res["child"] == "load_model"        # official loader, wrapped
    assert "LOAD: official fp16 loader called" in out.stdout
    assert res["batch"] == 16 and "1 of 16 embeddings are zero vectors" in out.stdout


def test_oom_retry_happens_outside_the_handler():
    """A single input gets a second attempt after the failed one was released."""
    state = {"n": 0}

    def flaky(inputs, **k):
        state["n"] += 1
        if state["n"] == 1:
            raise torch.cuda.OutOfMemoryError("first try")
        return {"hidden_states": torch.ones(len(inputs), 2)}

    out = mq.oom_safe_encode(flaky)([0])
    assert state["n"] == 2 and out["hidden_states"].shape == (1, 2)
