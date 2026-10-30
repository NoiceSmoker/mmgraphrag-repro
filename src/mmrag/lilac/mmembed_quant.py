"""Guarded and optionally low-memory MM-Embed loading for the official LILaC code.

Guard (``MMEMBED_GUARD=1``, set by ``scripts/lilac/pipeline.sh`` for every
MM-Embed stage): keeps the official fp16 loader but protects the official
encoder against silent failures (see below). Quantization (``MMEMBED_QUANT``)
additionally replaces the loader, for 16 GB GPUs such as the T4 of free Colab:

MM-Embed is about 16.4 GB in fp16, and the official loader
(``src.models.embedder.mmembed.MMEmbed.load_model``) first builds the whole
fp16 model in CPU memory and then moves it to the GPU. Neither fits a free
Colab session (T4 with 15 GB usable, ~12.7 GB of system RAM).

With ``MMEMBED_QUANT=8bit`` (or ``4bit``) this module replaces that one method
so that the checkpoint is streamed shard by shard straight to the GPU with
bitsandbytes weights: LLM.int8 (default) or NF4. The vision tower, the
multimodal projector and the latent-attention pooling stay in fp16. Nothing
else in the official code changes.

The official embedding scripts start their workers with
``multiprocessing`` (spawn), so a monkeypatch in the parent process would not
reach them. The patch is therefore installed by an import hook from
``sitecustomize`` (``src/mmrag/lilac/_site``), which every Python process of
the ``mmembed`` environment runs at start-up when that directory is on
``PYTHONPATH`` (``scripts/lilac/pipeline.sh`` does this).

Other knobs (environment variables):

``MMEMBED_BATCH_SCALE``  multiply the official per-component batch sizes
                         (default 1.0).

The official ``encode_corpus`` catches *every* exception of a batch and
stores zero vectors for it, so an out-of-memory error would silently produce
empty embeddings. The quantized model's ``encode`` is therefore wrapped: on
OOM the batch is split in halves (recursively) and the results concatenated.
``encode_corpus`` additionally reports how many zero rows it produced.
"""
from __future__ import annotations

import importlib.abc
import json
import os
import sys
from pathlib import Path

TARGET_MODULE = "src.models.embedder.mmembed"
KEEP_FP16 = ["vision_tower", "multi_modal_projector", "latent_attention_model"]
_TAG = "[mmembed-quant]"


def quant_mode() -> str | None:
    """Normalised value of MMEMBED_QUANT: '8bit', '4bit' or None (official fp16)."""
    v = os.environ.get("MMEMBED_QUANT", "").strip().lower()
    if v in ("", "0", "none", "fp16", "off", "false"):
        return None
    if v in ("8bit", "int8", "8"):
        return "8bit"
    if v in ("4bit", "nf4", "4"):
        return "4bit"
    raise ValueError(f"MMEMBED_QUANT={v!r}: use 8bit, 4bit or fp16")


def guard_enabled() -> bool:
    """True if the hook should be installed (guard requested or quantization on)."""
    v = os.environ.get("MMEMBED_GUARD", "").strip().lower()
    return v in ("1", "true", "yes", "on") or quant_mode() is not None


def describe() -> dict:
    """What gets recorded next to the embeddings and logged with every run."""
    mode = quant_mode()
    return {"mmembed_weights": mode or "fp16", "guard": guard_enabled(),
            "kept_fp16": KEEP_FP16 if mode else [],
            "batch_scale": float(os.environ.get("MMEMBED_BATCH_SCALE", "1") or 1)}


# ----------------------------------------------------------------- loading --
def _bnb_config(mode: str):
    import torch
    from transformers import BitsAndBytesConfig

    if mode == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=KEEP_FP16)
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.float16, llm_int8_skip_modules=KEEP_FP16)


def finalize_devices(model, device) -> list[str]:
    """Make sure every tensor of a quantized model lives on ``device``.

    With quantization, transformers loads with ``low_cpu_mem_usage``; tensors
    that are not in the checkpoint can end up on the CPU (moved here) or stay
    on the meta device (an error - those weights would be garbage). The
    quantized weights themselves are never touched. Returns the moved names.
    """
    import torch

    dev = torch.device(device)
    moved, meta = [], []
    for mod_name, mod in model.named_modules():
        for kind, store in (("param", mod._parameters), ("buffer", mod._buffers)):
            for name, t in list(store.items()):
                if t is None:
                    continue
                full = f"{mod_name}.{name}" if mod_name else name
                if t.device.type == "meta":
                    meta.append(full)
                    continue
                if t.device == dev or type(t).__module__.startswith("bitsandbytes"):
                    continue
                if kind == "param":
                    store[name] = torch.nn.Parameter(t.data.to(dev), requires_grad=False)
                else:
                    store[name] = t.to(dev)
                moved.append(full)
    if meta:
        raise RuntimeError(f"{_TAG} {len(meta)} tensors were not loaded from the checkpoint "
                           f"(still on the meta device), e.g. {meta[:5]}. Run the MM-Embed stages "
                           "without MMEMBED_QUANT on a GPU with >= 20 GB instead.")
    if moved:
        print(f"{_TAG} moved {len(moved)} tensors to {dev} (e.g. {moved[:3]})", flush=True)
    return moved


def load_model_quantized(self) -> None:
    """Replacement for ``MMEmbed.load_model`` (same attributes afterwards)."""
    import torch
    from transformers import AutoModel

    mode = quant_mode()
    device = f"cuda:{self.cuda_num}"
    print(f"{_TAG} loading MM-Embed with {mode} weights on {device} "
          f"(fp16: {', '.join(KEEP_FP16)})", flush=True)
    self.model = AutoModel.from_pretrained(
        self.model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        quantization_config=_bnb_config(mode),
        device_map={"": device},
    )
    finalize_devices(self.model, device)
    self.model.eval()
    gb = torch.cuda.memory_allocated(device) / 2**30
    print(f"{_TAG} model loaded ({gb:.1f} GiB on GPU)", flush=True)


def oom_safe_encode(encode):
    """Wrap ``model.encode(inputs, ...)``: split the batch on CUDA OOM."""

    def wrapped(inputs, *args, **kwargs):
        import torch

        try:
            return encode(inputs, *args, **kwargs)
        except torch.cuda.OutOfMemoryError:
            if len(inputs) <= 1:
                raise
            torch.cuda.empty_cache()
            mid = len(inputs) // 2
            parts = []
            for sl in (slice(0, mid), slice(mid, None)):
                kw = dict(kwargs)
                instr = kw.get("instruction")
                if isinstance(instr, (list, tuple)) and len(instr) == len(inputs):
                    kw["instruction"] = list(instr)[sl]
                parts.append(wrapped(inputs[sl], *args, **kw)["hidden_states"])
            print(f"{_TAG} out of memory on a batch of {len(inputs)} - encoded it in two halves", flush=True)
            return {"hidden_states": torch.cat(parts, dim=0)}

    wrapped.__wrapped__ = encode
    return wrapped


def _wrap_encode_corpus(orig):
    """Scale the official batch size; report zero vectors (failed batches)."""

    def encode_corpus(self, *args, **kwargs):
        scale = float(os.environ.get("MMEMBED_BATCH_SCALE", "1") or 1)
        if kwargs.get("batch_size"):
            kwargs["batch_size"] = max(1, int(round(kwargs["batch_size"] * scale)))
        out = orig(self, *args, **kwargs)
        try:
            zero = int((out.abs().sum(dim=1) == 0).sum())
            if zero:
                print(f"{_TAG} WARNING: {zero} of {out.shape[0]} embeddings are zero vectors "
                      "(encoding failed for them, see the errors above)", flush=True)
        except Exception:  # unexpected return type: leave it alone
            pass
        return out

    encode_corpus.__wrapped__ = orig
    return encode_corpus


def patch_module(module) -> None:
    cls = getattr(module, "MMEmbed", None)
    if cls is None or getattr(cls, "_mmrag_patched", False):
        return
    loader = load_model_quantized if quant_mode() else cls.load_model   # official fp16 loader otherwise

    def load_model(self):
        loader(self)
        self.model.encode = oom_safe_encode(self.model.encode)   # official encoder zero-fills failed batches

    load_model.__mmrag_loader__ = loader.__name__
    cls.load_model = load_model
    if hasattr(cls, "encode_corpus"):
        cls.encode_corpus = _wrap_encode_corpus(cls.encode_corpus)
    cls._mmrag_patched = True


# ------------------------------------------------------------- import hook --
class _PatchingFinder(importlib.abc.MetaPathFinder):
    """Patches ``src.models.embedder.mmembed`` right after it is executed."""

    def find_spec(self, fullname, path, target=None):
        if fullname != TARGET_MODULE:
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is not None and spec.loader is not None:
                break
        else:
            return None
        orig_exec = spec.loader.exec_module

        def exec_module(module):
            orig_exec(module)
            patch_module(module)

        spec.loader.exec_module = exec_module  # loader instances are per spec
        return spec


def install() -> bool:
    """Install the hook if MMEMBED_GUARD or MMEMBED_QUANT asks for it. Safe to call repeatedly."""
    if not guard_enabled():
        return False
    if TARGET_MODULE in sys.modules:  # imported before the hook: patch in place
        patch_module(sys.modules[TARGET_MODULE])
    if not any(isinstance(f, _PatchingFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _PatchingFinder())
    return True


def write_meta(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(describe(), indent=1))
