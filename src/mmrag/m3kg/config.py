"""Configuration for the scaled-down M3KG-RAG reproduction (audio QA).

A run is defined by ``configs/m3kg/base.yaml`` merged with a hardware
profile from ``configs/m3kg/profiles/*.yaml`` (model sizes / batch sizes),
then with optional ``--set a.b=c`` overrides on the command line.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def _deep_update(base: dict, upd: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def _parse_scalar(s: str) -> Any:
    try:
        return yaml.safe_load(s)
    except yaml.YAMLError:
        return s


def load_config(base: str | Path, profile: str | Path | None = None,
                overrides: list[str] | None = None) -> dict:
    cfg = yaml.safe_load(Path(base).read_text())
    if profile:
        cfg = _deep_update(cfg, yaml.safe_load(Path(profile).read_text()) or {})
    for item in overrides or []:
        key, _, val = item.partition("=")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _parse_scalar(val)
    work = Path(cfg["paths"]["work_dir"])
    cfg["paths"].setdefault("cache_db", str(work / "cache.sqlite"))
    return cfg


class WorkDir:
    """All artefacts of one reproduction live under ``paths.work_dir``."""

    def __init__(self, cfg: dict):
        self.root = Path(cfg["paths"]["work_dir"])
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def corpus_dir(self) -> Path:
        return self.root / "corpus"

    @property
    def test_dir(self) -> Path:
        return self.root / "test"

    @property
    def kg_dir(self) -> Path:
        return self.root / "kg"

    @property
    def index_dir(self) -> Path:
        return self.root / "index"

    @property
    def contexts_dir(self) -> Path:
        return self.root / "contexts"

    @property
    def preds_dir(self) -> Path:
        return self.root / "predictions"

    @property
    def judge_dir(self) -> Path:
        return self.root / "judgements"
