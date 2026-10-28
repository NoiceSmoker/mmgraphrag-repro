"""Run many retrieval variants with one loaded official ``Retriever``.

Constructing the official ``Retriever`` loads every corpus embedding, builds
FAISS indexes and the layered component graph -- by far the slowest part of
a retrieval run. All variants of one embedding set share these structures,
so we build the retriever once and only swap the per-run parameters
(mode, beam width, iterations, modality mode, active decomposition) before
calling the official ``run()``.
"""
from __future__ import annotations

import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import yaml

from mmrag.lilac import official
from mmrag.lilac.decompose import activate_decomposition, active_variant
from mmrag.lilac.variants import ITERATIVE, Variant

EMBEDDING_SETS = {
    "default": {"top_image": "image", "low_image": "subimage"},
    "novlm": {"top_image": "image_novlm", "low_image": "subimage_novlm"},
}


@contextmanager
def _clean_argv():
    """The official Retriever parses ``sys.argv`` in its constructor."""
    saved = sys.argv
    sys.argv = [saved[0] if saved else "retriever.py"]
    try:
        yield
    finally:
        sys.argv = saved


class RetrieverSession:
    def __init__(self, dataset: str, embedder: str = official.DEFAULT_EMBEDDER,
                 embedding_set: str = "default", gpu: int = -1):
        root = official.ensure_importable()
        from src.lilac.retriever import retriever as R  # noqa: WPS433 (runtime import of pinned code)

        cfg = yaml.safe_load((root / "config" / "retriever" / "retriever_config.yaml").read_text())
        cfg.update(run_name="_session_init", run_mode=ITERATIVE, target_dataset=dataset,
                   embedding_model=embedder, force_overwrite=True)
        names = EMBEDDING_SETS[embedding_set]
        cfg["top_level_embeddings"]["image"] = names["top_image"]
        cfg["low_level_embeddings"]["image"] = names["low_image"]
        cfg["top_level_embeddings"]["gpu_num"] = -1  # FAISS on CPU (faiss-gpu not required)
        cfg["low_level_embeddings"]["gpu_num"] = gpu  # late-interaction matmul device
        self._tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
        yaml.safe_dump(cfg, self._tmp)
        self._tmp.close()

        self.dataset = dataset
        self.embedding_set = embedding_set
        # the official retriever pickles the graph here without creating the folder
        (official.artifact_dir(dataset) / "components").mkdir(parents=True, exist_ok=True)
        with _clean_argv():
            self.r = R.Retriever(config_path=self._tmp.name)
        self._decomposer = active_variant(dataset) or "default"

    def use_decomposition(self, tag: str) -> None:
        """Activate a stored decomposition variant and reload only the
        question / sub-query embeddings (cheap)."""
        if tag in ("default", self._decomposer):
            return
        activate_decomposition(self.dataset, tag)
        self.r.initaite_questions_manager()  # (sic) official method name
        self._decomposer = tag

    def run(self, v: Variant, run_name: str) -> Path:
        from src.utils.utils import ensure_output_dir

        if v.embedding_set != self.embedding_set:
            raise ValueError(f"variant {v.name} needs embedding set {v.embedding_set}")
        self.use_decomposition(v.decomposer)
        r = self.r
        p = r._run_config["parameters"]
        p.update(beam_width=v.beam_width, num_iterations=v.num_iterations, top_k=v.top_k,
                 target_level=v.target_level, modality_mode=v.modality_mode, hop_mode="1_hop")
        r._run_config["run_mode"] = v.mode
        r._run_config["run_name"] = run_name
        r._run_function_mode = v.mode
        r._modality_mode = v.modality_mode
        r._target_level = v.target_level
        r._run_name = run_name
        out_dir = official.retrieval_output_dir(self.dataset, run_name)
        r._output_dir = str(out_dir)
        ensure_output_dir(str(out_dir), force_overwrite=True)
        with _clean_argv():
            r.run()
        return out_dir / f"{run_name}.jsonl"

    @property
    def decomposer(self) -> str:
        return self._decomposer
