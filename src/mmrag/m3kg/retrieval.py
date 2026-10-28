"""Modality-wise retrieval over the M3KG (paper Sec. 3.2) - pure numpy.

Given the CLAP embedding of the query audio:
  1. nearest audio items of the corpus (same modality as the query),
     top-k by L2 distance, kept only if distance <= tau             (Eq. 3 input S)
  2. lift the selected items to the triples linked to them            (Eq. 3, G_init)
  3. multi-hop expansion: triples of *other* clips that share a concept
     with G_init, ranked by the distance of their own clip to the query;
     hub concepts (very high degree) are not expanded              (implementation choice,
                                                                     the paper only says
                                                                     "expand to the connected
                                                                     multi-hop subgraph")
The ablation baseline "text-only KG in a shared space" (paper Table 4,
first row) embeds every triple as the sentence "h r t" with the CLAP text
encoder and retrieves triples directly with the audio query.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd


def l2_from_unit(q: np.ndarray, X: np.ndarray) -> np.ndarray:
    """L2 distance between unit vectors via the dot product."""
    return np.sqrt(np.clip(2.0 - 2.0 * (X @ q), 0.0, None))


def calibrate_tau(corpus_emb: np.ndarray, k: int = 5, quantiles=(0.25, 0.5, 0.75, 0.9)) -> dict[str, float]:
    """Label-free threshold candidates: quantiles of each corpus clip's
    distance to its k-th nearest *other* corpus clip."""
    n = corpus_emb.shape[0]
    d = np.sqrt(np.clip(2.0 - 2.0 * corpus_emb @ corpus_emb.T, 0.0, None))
    np.fill_diagonal(d, np.inf)
    kth = np.partition(d, kth=min(k, n - 1) - 1, axis=1)[:, min(k, n - 1) - 1]
    out = {f"q{int(q * 100)}": float(np.quantile(kth, q)) for q in quantiles}
    out["none"] = float("inf")
    return out


@dataclass
class KGIndex:
    triplets: pd.DataFrame
    clip_ids: list[str]               # corpus order of the embedding matrix
    clip_emb: np.ndarray              # (N, D) unit vectors

    def __post_init__(self):
        self.row_of_clip = {c: i for i, c in enumerate(self.clip_ids)}
        self.by_clip: dict[str, list[int]] = defaultdict(list)
        self.by_concept: dict[str, list[int]] = defaultdict(list)
        for tid, cid, h, t in zip(self.triplets.tid, self.triplets.clip_id,
                                  self.triplets.head_concept, self.triplets.tail_concept):
            self.by_clip[cid].append(int(tid))
            for c in {h.lower(), t.lower()}:
                self.by_concept[c].append(int(tid))
        self.tid_row = {int(t): i for i, t in enumerate(self.triplets.tid)}
        self._clip_of = dict(zip(self.triplets.tid.astype(int), self.triplets.clip_id))
        self._concepts_of = {int(t): {h.lower(), tl.lower()} for t, h, tl in
                             zip(self.triplets.tid, self.triplets.head_concept, self.triplets.tail_concept)}

    def hub_concepts(self, max_degree: int | None = None, top_frac: float = 0.01) -> set[str]:
        degs = {c: len(v) for c, v in self.by_concept.items()}
        if not degs:
            return set()
        if max_degree is None:
            max_degree = int(np.quantile(list(degs.values()), 1 - top_frac))
        return {c for c, d in degs.items() if d > max(max_degree, 1)}

    def nearest_clips(self, q: np.ndarray, k: int, tau: float) -> list[tuple[str, float]]:
        d = l2_from_unit(q, self.clip_emb)
        idx = np.argsort(d)[:k]
        return [(self.clip_ids[i], float(d[i])) for i in idx if d[i] <= tau]

    def lift(self, clips: list[tuple[str, float]]) -> list[int]:
        out: list[int] = []
        for cid, _ in clips:
            out.extend(self.by_clip.get(cid, []))
        return out

    def expand(self, q: np.ndarray, seed_tids: list[int], hops: int, max_triplets: int,
               hubs: set[str]) -> list[int]:
        """Breadth-first expansion through shared concepts."""
        selected = list(dict.fromkeys(seed_tids))      # lifted triples are always kept
        chosen = set(selected)
        frontier = set().union(*(self._concepts_of[t] for t in selected)) if selected else set()
        visited = set(frontier)
        d_all = l2_from_unit(q, self.clip_emb)
        for _ in range(hops):
            cand = {t for c in frontier - hubs for t in self.by_concept.get(c, []) if t not in chosen}
            if not cand or len(selected) >= max_triplets:
                break
            ranked = sorted(cand, key=lambda t: (d_all[self.row_of_clip[self._clip_of[t]]]
                                                 if self._clip_of[t] in self.row_of_clip else np.inf, t))
            added = ranked[: max_triplets - len(selected)]
            selected.extend(added)
            chosen.update(added)
            frontier = set().union(*(self._concepts_of[t] for t in added)) - visited
            visited |= frontier
        return selected

    def triple_texts(self, tids: list[int]) -> list[str]:
        rows = self.triplets.iloc[[self.tid_row[t] for t in tids]]
        return [f"{h} {r} {t}" for h, r, t in zip(rows["head"], rows["rel"], rows["tail"])]


def shared_space_retrieve(q_audio: np.ndarray, triple_text_emb: np.ndarray, tids: list[int], top_m: int) -> list[int]:
    """Baseline: audio query vs. CLAP text embeddings of 'h r t' sentences."""
    s = triple_text_emb @ q_audio
    idx = np.argsort(-s)[:top_m]
    return [tids[i] for i in idx]
