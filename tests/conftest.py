import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
os.environ.setdefault("WANDB_MODE", "disabled")


def lilac_available() -> bool:
    root = Path(os.environ.get("LILAC_ROOT", REPO / "third_party" / "lilac"))
    return (root / "src" / "lilac").is_dir()


needs_lilac = pytest.mark.skipif(not lilac_available(), reason="official LILaC code not fetched")


@pytest.fixture(autouse=True)
def _results_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("MMRAG_RESULTS_DIR", str(tmp_path / "results"))
    import mmrag.common.tracking as tr

    monkeypatch.setattr(tr, "RESULTS_DIR", tmp_path / "results")
    yield
