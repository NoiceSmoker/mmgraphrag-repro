"""Experiment grid for the LILaC reproduction.

A *variant* is one configuration of the official retriever. Groups map to
the sections of the report (and to W&B groups / report links):

  main         full LILaC, paper defaults (b=30, n_i=1, modality-aware)
  ablation     paper Table 3 rows + two extra ablations
  sens_beam    beam width b in {1,2,3,4,5,10,20,30}         (paper Fig. 5a)
  sens_iter    number of iterations n_i in {0,1,2}          (paper Fig. 5b)
  decomposer   full LILaC with sub-queries from different LLMs (paper Table 4)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace

ITERATIVE = "iterative_late_interaction"
ONE_PASS = "late_interaction"
TOPDOWN = "single_topdown"
KNN = "single_knn"


@dataclass(frozen=True)
class Variant:
    name: str
    group: str
    mode: str = ITERATIVE
    beam_width: int = 30
    num_iterations: int = 1
    top_k: int = 100
    target_level: str = "top"
    modality_mode: str = "aware"           # aware | agnostic
    embedding_set: str = "default"         # default | novlm  (see serialize_novlm.py)
    decomposer: str = "default"            # tag of a stored decomposition variant
    notes: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)

    def as_config(self) -> dict:
        d = asdict(self)
        d["tags"] = list(self.tags)
        return d


FULL = Variant(name="lilac_full", group="main",
               notes="Paper defaults: iterative late interaction, b=30, n_i=1, modality-aware sub-queries.")


def main_group() -> list[Variant]:
    return [FULL]


def ablation_group() -> list[Variant]:
    return [
        Variant(name="wo_lcg_qd", group="ablation", mode=KNN,
                notes="Paper 'w/o LCG & QD': plain kNN over top-level components."),
        Variant(name="wo_qd", group="ablation", mode=TOPDOWN,
                notes="Paper 'w/o QD': top-level kNN (b=30) re-ranked by best sub-component."),
        replace(FULL, name="full", group="ablation"),
        replace(FULL, name="full_agnostic", group="ablation", modality_mode="agnostic",
                notes="Extra: sub-queries embedded with one generic instruction (no modality label)."),
        replace(FULL, name="full_novlm", group="ablation", embedding_set="novlm",
                notes="Extra: image / sub-image nodes embedded without VLM-generated text."),
    ]


def sens_beam_group() -> list[Variant]:
    return [replace(FULL, name=f"b{b}", group="sens_beam", beam_width=b) for b in (1, 2, 3, 4, 5, 10, 20, 30)]


def sens_iter_group() -> list[Variant]:
    out = [Variant(name="ni0_b5", group="sens_iter", mode=KNN, beam_width=5, num_iterations=0,
                   notes="n_i=0 as in the official script: top-level kNN only.")]
    for ni in (1, 2):
        out.append(replace(FULL, name=f"ni{ni}_b5", group="sens_iter", beam_width=5, num_iterations=ni))
    # The official parameter_sensitivity.sh passes --parameter_numiterations to
    # run_mode=late_interaction, which never reads it. This variant reproduces
    # that call so its output can be compared with ni1_b5 / ni2_b5.
    out.append(Variant(name="official_script_ni2_b5", group="sens_iter", mode=ONE_PASS, beam_width=5,
                       num_iterations=2, notes="Official script call (num_iterations ignored by this mode)."))
    for ni in (1, 2, 3):
        out.append(replace(FULL, name=f"ni{ni}_b30", group="sens_iter", beam_width=30, num_iterations=ni))
    return out


def decomposer_group(tags: list[str]) -> list[Variant]:
    return [replace(FULL, name=f"decomp_{t}", group="decomposer", decomposer=t) for t in tags]


GROUPS = {
    "main": main_group,
    "ablation": ablation_group,
    "sens_beam": sens_beam_group,
    "sens_iter": sens_iter_group,
}
