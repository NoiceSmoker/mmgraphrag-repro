"""Experiment grid of the scaled-down M3KG-RAG reproduction (Audio-QA).

Groups (one W&B group / report section each):
  main        no retrieval, naive shared-space RAG, neighbour-caption RAG, M3KG-RAG  (paper Table 1, one MLLM)
  ablation    modality-wise retrieval x GRASP, 2x2                               (paper Table 4)
  grasp       contribution of grounding vs. LLM pruning                          (paper Table 15, audio branch)
  hops        0 / 1 / 2 expansion hops - tests the multi-hop claim directly      (extra)
  sens_tau    retrieval distance threshold                                       (paper Fig. 5 top)
  sens_eta    grounding threshold                                                (paper Fig. 5 bottom)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace


@dataclass(frozen=True)
class M3Config:
    name: str
    group: str
    retrieval: str = "mw"        # none | mw (modality-wise) | shared (text-KG, shared space) | captions
    k: int = 5                   # nearest audio items (paper: k=5)
    tau: str = "q75"             # key of the calibrated thresholds (see retrieval.calibrate_tau)
    hops: int = 1
    max_triplets: int = 60
    shared_top_m: int = 30
    grounding: bool = True       # GRASP step 1 (TAG presence score >= eta)
    eta: float | None = None     # None -> default of the active grounding backend
    llm_filter: bool = True      # GRASP step 2 (LLM keep/drop)
    notes: str = ""

    def as_config(self) -> dict:
        return asdict(self)

    @property
    def uses_kg(self) -> bool:
        return self.retrieval != "none"


NONE = M3Config("no_rag", "main", retrieval="none", grounding=False, llm_filter=False,
                notes="MLLM answers from the audio alone.")
SHARED = M3Config("shared_space", "main", retrieval="shared", grounding=False, llm_filter=False,
                  notes="Triples as text, retrieved with the audio query in CLAP's shared space.")
CAPTIONS = M3Config("caption_rag", "main", retrieval="captions", grounding=False, llm_filter=False,
                    notes="Same nearest clips as M3KG-RAG, but their raw human captions instead of the KG.")
FULL = M3Config("m3kg_rag", "main", notes="Modality-wise retrieval + 1-hop expansion + GRASP.")


def grid(eta_grid: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9)) -> dict[str, list[M3Config]]:
    mw = replace(FULL, name="mw_only", grounding=False, llm_filter=False)
    return {
        "main": [NONE, SHARED, CAPTIONS, FULL],
        "ablation": [
            replace(SHARED, name="abl_shared", group="ablation"),
            replace(SHARED, name="abl_shared_grasp", group="ablation", grounding=True, llm_filter=True),
            replace(mw, name="abl_mw", group="ablation"),
            replace(FULL, name="abl_mw_grasp", group="ablation"),
        ],
        "grasp": [
            replace(mw, name="grasp_none", group="grasp"),
            replace(FULL, name="grasp_tag", group="grasp", llm_filter=False),
            replace(FULL, name="grasp_llm", group="grasp", grounding=False),
            replace(FULL, name="grasp_tag_llm", group="grasp"),
        ],
        "hops": [replace(FULL, name=f"hops{h}", group="hops", hops=h) for h in (0, 1, 2)],
        "sens_tau": [replace(FULL, name=f"tau_{t}", group="sens_tau", tau=t) for t in ("q25", "q50", "q75", "q90", "none")],
        "sens_eta": [replace(FULL, name=f"eta_{e}", group="sens_eta", eta=e) for e in eta_grid],
    }


def all_configs(groups: list[str] | None = None, eta_grid=None) -> list[M3Config]:
    g = grid(eta_grid) if eta_grid else grid()
    out = []
    for name, cfgs in g.items():
        if groups is None or name in groups:
            out.extend(cfgs)
    return out
