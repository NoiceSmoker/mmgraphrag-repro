"""Bridge to the official LILaC repository.

The reproduction does not fork LILaC: the official code is cloned at a
pinned commit into ``third_party/lilac`` (``scripts/lilac/pipeline.sh fetch``)
and imported from there. Everything in ``mmrag.lilac`` only (a) orchestrates
the official steps on a single GPU, (b) replaces two slow one-image-at-a-time
VLM loops with batched vLLM calls that write byte-compatible artefacts, and
(c) adds experiment bookkeeping and extra metrics.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PINNED_COMMIT = "813b6c0cbd7c67ae9a4ec9ca070ac5190a7acfe0"
DEFAULT_EMBEDDER = "MM-Embed"


def repo_root() -> Path:
    """Root of *this* repository (mmgraphrag-repro)."""
    return Path(__file__).resolve().parents[3]


def lilac_root() -> Path:
    root = Path(os.environ.get("LILAC_ROOT", repo_root() / "third_party" / "lilac")).resolve()
    if not (root / "src" / "lilac").is_dir():
        raise FileNotFoundError(
            f"Official LILaC code not found at {root}. Run scripts/lilac/pipeline.sh fetch "
            "or set LILAC_ROOT."
        )
    return root


def ensure_importable() -> Path:
    """Put the official repo on sys.path so that ``import src.lilac...`` works."""
    root = lilac_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def dataset_dir(dataset: str) -> Path:
    return lilac_root() / "datasets" / dataset


def artifact_dir(dataset: str) -> Path:
    return lilac_root() / "artifacts" / dataset


def embeddings_dir(dataset: str, embedder: str = DEFAULT_EMBEDDER) -> Path:
    return artifact_dir(dataset) / "embeddings" / embedder


def decomposition_dir(dataset: str) -> Path:
    return artifact_dir(dataset) / "query_decomposition" / "dev"


def labeled_qa_path(dataset: str) -> Path:
    return dataset_dir(dataset) / "QAs_dev_labeled.json"


def retrieval_output_dir(dataset: str, run_name: str) -> Path:
    return lilac_root() / "algorithm_results" / "LILaC" / dataset / "retrieval" / run_name
