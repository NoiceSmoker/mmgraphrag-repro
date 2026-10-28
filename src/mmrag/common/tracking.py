"""Experiment tracking.

Every experiment run is logged twice:
  1. to Weights & Biases (project/group/run, config, summary metrics, tables);
  2. to a local JSON file under ``results/<project>/<group>/<run>.json``.

The local copy is what ``scripts/export_results.py`` turns into LaTeX
tables, so the report can be rebuilt even without network access. W&B can
be switched off with ``WANDB_MODE=disabled``; the local copy is always
written.

Environment variables honoured: WANDB_ENTITY, WANDB_PROJECT (overrides the
project passed in code), WANDB_MODE, MMRAG_RESULTS_DIR.
"""
from __future__ import annotations

import os
import platform
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from mmrag.common.io import write_json

RESULTS_DIR = Path(os.environ.get("MMRAG_RESULTS_DIR", "results"))


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def _hardware() -> dict[str, Any]:
    info: dict[str, Any] = {"python": platform.python_version(), "platform": platform.platform()}
    try:
        import torch

        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
    except Exception:
        pass
    return info


class Run:
    """Thin wrapper over a wandb run that mirrors everything to disk."""

    def __init__(self, project: str, group: str, name: str, config: dict[str, Any],
                 job_type: str = "eval", tags: list[str] | None = None):
        self.project = os.environ.get("WANDB_PROJECT", project)
        self.group = group
        self.name = name
        self.config = dict(config)
        self.config.setdefault("_git_commit", _git_commit())
        self.config.setdefault("_hardware", _hardware())
        self.summary: dict[str, Any] = {}
        self.tables: dict[str, list[dict]] = {}
        self._t0 = time.time()
        self._wb = None
        if os.environ.get("WANDB_MODE", "").lower() != "disabled":
            try:
                import wandb

                self._wb = wandb.init(
                    project=self.project,
                    entity=os.environ.get("WANDB_ENTITY") or None,
                    group=group,
                    name=name,
                    job_type=job_type,
                    tags=tags or [],
                    config=self.config,
                    reinit=True,
                )
            except Exception as exc:  # never let tracking kill an experiment
                print(f"[tracking] wandb disabled ({exc}); writing local results only")
                self._wb = None

    @property
    def url(self) -> str | None:
        return getattr(self._wb, "url", None) if self._wb is not None else None

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if self._wb is not None:
            self._wb.log(metrics, step=step)

    def set_summary(self, metrics: dict[str, Any]) -> None:
        self.summary.update(metrics)
        if self._wb is not None:
            for k, v in metrics.items():
                self._wb.summary[k] = v

    def log_table(self, key: str, rows: list[dict], max_rows: int = 5000) -> None:
        rows = rows[:max_rows]
        self.tables[key] = rows
        if self._wb is not None and rows:
            import wandb

            cols = list(rows[0].keys())
            table = wandb.Table(columns=cols, data=[[r.get(c) for c in cols] for r in rows])
            self._wb.log({key: table})

    def finish(self) -> Path:
        self.summary.setdefault("wall_time_s", round(time.time() - self._t0, 2))
        out = RESULTS_DIR / self.project / self.group / f"{self.name}.json"
        write_json(
            {
                "project": self.project,
                "group": self.group,
                "name": self.name,
                "url": self.url,
                "config": self.config,
                "summary": self.summary,
                "tables": {k: v[:200] for k, v in self.tables.items()},
            },
            out,
        )
        if self._wb is not None:
            self._wb.summary["wall_time_s"] = self.summary["wall_time_s"]
            self._wb.finish()
        return out


@contextmanager
def tracked_run(project: str, group: str, name: str, config: dict[str, Any],
                job_type: str = "eval", tags: list[str] | None = None) -> Iterator[Run]:
    run = Run(project, group, name, config, job_type=job_type, tags=tags)
    try:
        yield run
    finally:
        path = run.finish()
        print(f"[tracking] {group}/{name} -> {path}" + (f"  ({run.url})" if run.url else ""))
