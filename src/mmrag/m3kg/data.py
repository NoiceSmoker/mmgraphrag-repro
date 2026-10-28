"""Corpus and evaluation data for the audio-QA reproduction.

* Knowledge corpus  : a random subset of the AudioCaps **train** split
  (audio + human caption + YouTube id), streamed from the Hub so that only
  the sampled clips are downloaded.
* Evaluation set    : AudioCaps-QA from AudioBench (built on AudioCaps test
  clips), the Audio-QA benchmark of the paper.

Audio columns are read with ``decode=False`` and decoded with soundfile, which
avoids the torchcodec dependency of recent ``datasets`` releases.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from mmrag.m3kg.audio import decode_bytes, write_wav
from mmrag.m3kg.config import WorkDir, load_config


def _audio_bytes(cell) -> bytes:
    if isinstance(cell, dict):
        if cell.get("bytes"):
            return cell["bytes"]
        if cell.get("path") and Path(cell["path"]).exists():
            return Path(cell["path"]).read_bytes()
    raise ValueError("audio cell has neither bytes nor a readable path")


def prepare_corpus(cfg: dict) -> pd.DataFrame:
    from datasets import Audio, load_dataset

    d = cfg["data"]
    wd = WorkDir(cfg)
    meta_path = wd.corpus_dir / "meta.parquet"
    if meta_path.exists():
        return pd.read_parquet(meta_path)
    ds = load_dataset(d["corpus_dataset"], split=d["corpus_split"], streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))
    ds = ds.shuffle(seed=d["seed"], buffer_size=d.get("shuffle_buffer", 10_000))
    rows = []
    for ex in ds:
        if len(rows) >= d["corpus_size"]:
            break
        try:
            wav, sr = decode_bytes(_audio_bytes(ex["audio"]))
        except Exception as exc:
            print(f"[corpus] skip {ex.get('audiocap_id')}: {exc}")
            continue
        if wav.size < sr:  # < 1 s of audio
            continue
        cid = f"ac_{ex['audiocap_id']}"
        path = wd.corpus_dir / "audio" / f"{cid}.wav"
        write_wav(path, wav, sr)
        rows.append({"clip_id": cid, "youtube_id": ex.get("youtube_id"), "start_time": ex.get("start_time"),
                     "caption": ex["caption"].strip(), "path": str(path), "duration_s": wav.size / sr})
    df = pd.DataFrame(rows)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(meta_path, index=False)
    print(f"[corpus] {len(df)} clips -> {meta_path}")
    return df


def prepare_test(cfg: dict) -> pd.DataFrame:
    from datasets import Audio, load_dataset

    d = cfg["data"]
    wd = WorkDir(cfg)
    meta_path = wd.test_dir / "meta.parquet"
    if meta_path.exists():
        return pd.read_parquet(meta_path)
    ds = load_dataset(d["test_dataset"], split=d["test_split"])
    audio_col = d.get("test_audio_column", "context")
    ds = ds.cast_column(audio_col, Audio(decode=False))
    limit = d.get("test_limit")
    rows = []
    for i, ex in enumerate(ds):
        if limit and i >= limit:
            break
        wav, sr = decode_bytes(_audio_bytes(ex[audio_col]))
        qid = f"q{i:04d}"
        path = wd.test_dir / "audio" / f"{qid}.wav"
        write_wav(path, wav, sr)
        rows.append({"qid": qid, "question": ex[d.get("test_question_column", "instruction")].strip(),
                     "answer": str(ex[d.get("test_answer_column", "answer")]).strip(), "path": str(path),
                     "duration_s": wav.size / sr})
    df = pd.DataFrame(rows)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(meta_path, index=False)
    print(f"[test] {len(df)} questions -> {meta_path}")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/m3kg/base.yaml")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_config(a.config, a.profile, a.set)
    prepare_test(cfg)
    prepare_corpus(cfg)


if __name__ == "__main__":
    main()
