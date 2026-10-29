"""GPU helpers shared by all vLLM-based stages."""
from __future__ import annotations

import os


def vllm_dtype() -> str:
    """dtype for vLLM engines.

    ``VLLM_DTYPE`` wins if set. Otherwise GPUs older than Ampere (e.g. the T4 of
    free Colab / Kaggle, compute capability 7.5) get ``half`` because they have
    no bfloat16 support; everything else keeps ``auto`` (the checkpoint dtype).
    """
    env = os.environ.get("VLLM_DTYPE", "").strip()
    if env:
        return env
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.get_device_capability(0) < (8, 0):
            return "half"
    except Exception:  # torch missing or no driver: let vLLM decide
        pass
    return "auto"
