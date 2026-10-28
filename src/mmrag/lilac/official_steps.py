"""Single-GPU wrappers around official LILaC steps.

* ``step3``: the official sentence splitter hard-codes ``num_gpus=4``
  (``SentenceAdder.parallel_sentence_split``); on a one-GPU Colab VM the
  workers for cuda:1..3 crash. We patch the default to the number of
  visible GPUs (or 1 process on CPU) and call the official ``main``.
* ``embed-extra``: embed an arbitrary serialisation file with the official
  MM-Embed wrapper (used by the no-VLM-text ablation).

Usage:
    python -m mmrag.lilac.official_steps step3 --dataset MMCoQA
    python -m mmrag.lilac.official_steps embed-extra --dataset MMCoQA \
        --in-name image+image --out-name image_novlm --max-length 256 --batch-size 8
"""
from __future__ import annotations

import argparse
import os
import sys

from mmrag.lilac import official


def _n_gpus() -> int:
    try:
        import torch

        return max(1, torch.cuda.device_count()) if torch.cuda.is_available() else 1
    except Exception:
        return 1


def step3(dataset: str) -> None:
    official.ensure_importable()
    import src.lilac.lcg_constructor.step3_add_sentences as s3

    n = _n_gpus()
    s3.SentenceAdder.parallel_sentence_split.__defaults__ = (n, 1)
    if not _has_cuda():
        _orig_worker = s3.SentenceAdder.worker

        def cpu_worker(self, model_id, gpu_id, items, return_dict):
            # Same as the official worker, but without .half().to("cuda").
            from tqdm import tqdm
            from wtpsplit import SaT

            sat = SaT("sat-3l-sm")
            partial = {}
            for outer, inner, text in tqdm(items, desc=f"SaT cpu worker {model_id}"):
                try:
                    sents = sat.split(text)
                except Exception:
                    sents = [text]
                partial.setdefault(outer, {})[inner] = [s for s in sents if s != ""]
            return_dict[model_id] = partial

        s3.SentenceAdder.worker = cpu_worker
    sys.argv = ["step3_add_sentences.py", "--target_data", dataset, "--mode", "all"]
    s3.main()


def _has_cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def embed_extra(dataset: str, in_name: str, out_name: str, max_length: int, batch_size: int) -> None:
    official.ensure_importable()
    from src.models.embedder.mmembed import MMEmbed
    from src.models.embedder.sharded_runner import encode_one_corpus

    ser = official.artifact_dir(dataset) / "embeddings" / "serializations" / f"{in_name}.json"
    out_dir = official.embeddings_dir(dataset)
    out_dir.mkdir(parents=True, exist_ok=True)
    encode_one_corpus(
        MMEmbed, "mmembed_extra_",
        corpus_in_filepath=str(ser),
        out_embeddings_path=str(out_dir / f"{out_name}.pt"),
        out_index_path=str(out_dir / f"{out_name}.json"),
        num_gpus=int(os.environ.get("NUM_GPUS", "1")),
        max_length=max_length, batch_size=batch_size,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("step3")
    a.add_argument("--dataset", default="MMCoQA")
    b = sub.add_parser("embed-extra")
    b.add_argument("--dataset", default="MMCoQA")
    b.add_argument("--in-name", required=True)
    b.add_argument("--out-name", required=True)
    b.add_argument("--max-length", type=int, default=256)
    b.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()
    if args.cmd == "step3":
        step3(args.dataset)
    else:
        embed_extra(args.dataset, args.in_name, args.out_name, args.max_length, args.batch_size)


if __name__ == "__main__":
    main()
