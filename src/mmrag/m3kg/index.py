"""CLAP index of the corpus (audio env).

Writes to ``<work>/index``:
  corpus_emb.npy / corpus_ids.json   audio embeddings of the knowledge corpus
  test_emb.npy   / test_ids.json     audio embeddings of the evaluation queries
  triple_text_emb.npy / triple_tids.json   'h r t' sentences (shared-space baseline)
  taus.json                          label-free distance thresholds (retrieval.calibrate_tau)
  leakage.json                       corpus clips removed because they (nearly)
                                     duplicate an evaluation clip
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from mmrag.common.io import write_json
from mmrag.m3kg.config import WorkDir, load_config
from mmrag.m3kg.encoders import ClapEncoder
from mmrag.m3kg.retrieval import calibrate_tau


def build_index(cfg: dict) -> None:
    wd = WorkDir(cfg)
    out = wd.index_dir
    out.mkdir(parents=True, exist_ok=True)
    corpus = pd.read_parquet(wd.corpus_dir / "meta.parquet")
    test = pd.read_parquet(wd.test_dir / "meta.parquet")
    kg = pd.read_parquet(wd.kg_dir / "triplets.parquet")
    clap = ClapEncoder(cfg["retrieval"]["clap_model"])

    c_emb = clap.embed_audio_files(corpus.path.tolist())
    q_emb = clap.embed_audio_files(test.path.tolist())

    # leakage guard: drop corpus clips that are near-duplicates of test clips
    sim = q_emb @ c_emb.T
    thr = cfg["data"].get("leakage_cos", 0.98)
    leak_rows = sorted(set(np.where(sim.max(axis=0) >= thr)[0].tolist()))
    keep = np.setdiff1d(np.arange(len(corpus)), leak_rows)
    ids = corpus.clip_id.tolist()
    write_json({"threshold": thr, "removed": [ids[i] for i in leak_rows]}, out / "leakage.json")
    ids_kept = [ids[i] for i in keep]
    np.save(out / "corpus_emb.npy", c_emb[keep].astype(np.float32))
    write_json(ids_kept, out / "corpus_ids.json")
    np.save(out / "test_emb.npy", q_emb.astype(np.float32))
    write_json(test.qid.tolist(), out / "test_ids.json")

    kg_kept = kg[kg.clip_id.isin(set(ids_kept))]
    texts = [f"{h} {r} {t}" for h, r, t in zip(kg_kept["head"], kg_kept["rel"], kg_kept["tail"])]
    np.save(out / "triple_text_emb.npy", clap.embed_text(texts).astype(np.float32))
    write_json([int(t) for t in kg_kept.tid], out / "triple_tids.json")

    taus = calibrate_tau(c_emb[keep], k=cfg["retrieval"].get("k", 5))
    write_json(taus, out / "taus.json")
    print(f"[index] corpus={len(ids_kept)} (removed {len(leak_rows)} near-duplicates), "
          f"test={len(test)}, triples={len(texts)}, taus={taus}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/m3kg/base.yaml")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    build_index(load_config(a.config, a.profile, a.set))


if __name__ == "__main__":
    main()
