"""Run at start-up by every Python process whose PYTHONPATH contains this
directory (set by scripts/lilac/pipeline.sh for the MM-Embed stages).
It only installs the MM-Embed hook when MMEMBED_GUARD or MMEMBED_QUANT is set; see
mmrag/lilac/mmembed_quant.py."""
import os
import sys

if os.environ.get("MMEMBED_GUARD") or os.environ.get("MMEMBED_QUANT"):
    try:
        from mmrag.lilac.mmembed_quant import install

        install()
    except Exception as exc:  # never break interpreter start-up
        print(f"[mmembed-quant] hook not installed: {exc}", file=sys.stderr)
