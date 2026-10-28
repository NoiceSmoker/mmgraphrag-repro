"""Optional crawler for the rewriter agent: YouTube title + description of
each corpus clip (metadata only, no media is downloaded).

Many AudioCaps videos have been removed and YouTube may rate-limit cloud
IPs; clips without metadata simply keep their original caption (which is
also what the rewriter prompt prescribes when the metadata does not match).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from mmrag.common.cache import KVCache
from mmrag.m3kg.config import WorkDir, load_config


def fetch_one(youtube_id: str) -> dict:
    import yt_dlp

    opts = {"quiet": True, "skip_download": True, "no_warnings": True, "noplaylist": True,
            "extract_flat": False, "socket_timeout": 15}
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={youtube_id}", download=False)
        return {"title": info.get("title"), "description": (info.get("description") or "")[:3000]}
    except Exception as exc:
        return {"title": None, "description": None, "error": type(exc).__name__}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/m3kg/base.yaml")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    cfg = load_config(a.config, a.profile, a.set)
    wd = WorkDir(cfg)
    meta = pd.read_parquet(wd.corpus_dir / "meta.parquet")
    cache = KVCache(cfg["paths"]["cache_db"], table="youtube")
    ids = [y for y in meta.youtube_id.dropna().unique()]
    have = cache.get_many(ids)
    todo = [y for y in ids if y not in have]
    with ThreadPoolExecutor(a.threads) as ex:
        for i, (y, res) in enumerate(zip(todo, ex.map(fetch_one, todo))):
            cache.set(y, res)
            have[y] = res
            if (i + 1) % 200 == 0:
                print(f"[youtube] {i + 1}/{len(todo)}")
    rows = [{"clip_id": r.clip_id, **{k: v for k, v in have.get(r.youtube_id, {}).items() if k != "error"}}
            for r in meta.itertuples()]
    df = pd.DataFrame(rows)
    df.to_parquet(wd.corpus_dir / "youtube_meta.parquet", index=False)
    ok = df["title"].notna().mean() if "title" in df else 0.0
    print(f"[youtube] metadata available for {ok:.1%} of clips")


if __name__ == "__main__":
    main()
