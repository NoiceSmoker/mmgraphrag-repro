"""Open knowledge-base look-ups for the *searcher* agent (Wikipedia, Wiktionary).

One Wikipedia request per concept (``generator=search`` + ``prop=extracts``)
returns the intro of the top pages; Wiktionary's REST definition endpoint
adds dictionary senses. Results are cached in SQLite, requests are made in
parallel with retries, and a descriptive User-Agent is sent as required by
the Wikimedia API policy (set ``WIKI_USER_AGENT`` to include your contact).
"""
from __future__ import annotations

import html
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from mmrag.common.cache import KVCache

WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKT_DEF = "https://en.wiktionary.org/api/rest_v1/page/definition/{term}"
DEFAULT_UA = "mmgraphrag-repro/0.1 (academic reproduction of M3KG-RAG; set WIKI_USER_AGENT with a contact address)"
_TAG = re.compile(r"<[^>]+>")


def clean_html(s: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(_TAG.sub("", s or ""))).strip()


def first_sentences(text: str, n: int = 2) -> str:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(parts[:n]).strip()


def parse_wikipedia(js: dict, max_pages: int = 3) -> list[str]:
    pages = (js.get("query") or {}).get("pages") or {}
    if isinstance(pages, dict):
        pages = list(pages.values())
    pages = sorted(pages, key=lambda p: p.get("index", 1e9))
    out = []
    for p in pages[:max_pages]:
        ext = (p.get("extract") or "").strip()
        if not ext or "may refer to" in ext[:200].lower():
            continue  # empty or disambiguation page
        out.append(first_sentences(ext, 2))
    return out


def parse_wiktionary(js: dict, max_defs: int = 3) -> list[str]:
    out = []
    for entry in js.get("en", []) if isinstance(js, dict) else []:
        for d in entry.get("definitions", []):
            txt = clean_html(d.get("definition", ""))
            if txt and len(txt) > 3:
                out.append(txt[0].upper() + txt[1:] if txt else txt)
            if len(out) >= max_defs:
                return out
    return out


class KnowledgeBase:
    def __init__(self, cache: KVCache | None, threads: int = 8, user_agent: str | None = None,
                 http_get: Callable | None = None, max_wiki: int = 3, max_wikt: int = 2):
        self.cache = cache
        self.threads = threads
        self.max_wiki, self.max_wikt = max_wiki, max_wikt
        self.headers = {"User-Agent": user_agent or os.environ.get("WIKI_USER_AGENT", DEFAULT_UA)}
        self._get = http_get or self._requests_get

    def _requests_get(self, url: str, params: dict | None = None) -> dict:
        import requests

        delay = 1.0
        for _ in range(5):
            r = requests.get(url, params=params, headers=self.headers, timeout=20)
            if r.status_code == 404:
                return {}
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(delay)
                delay *= 2
                continue
            r.raise_for_status()
            return r.json()
        return {}

    def _lookup(self, concept: str) -> dict:
        wiki = self._get(WIKI_API, {
            "action": "query", "format": "json", "formatversion": 2, "redirects": 1,
            "generator": "search", "gsrsearch": concept, "gsrlimit": self.max_wiki,
            "prop": "extracts", "exintro": 1, "explaintext": 1, "exlimit": self.max_wiki,
        })
        term = concept.strip().replace(" ", "_")
        wikt = self._get(WIKT_DEF.format(term=term)) if len(concept.split()) <= 3 else {}
        return {"wikipedia": parse_wikipedia(wiki, self.max_wiki), "wiktionary": parse_wiktionary(wikt, self.max_wikt)}

    def lookup_many(self, concepts: list[str]) -> dict[str, dict]:
        uniq = sorted(set(c for c in concepts if c))
        keys = {c: f"kb:{c.lower()}" for c in uniq}
        cached = self.cache.get_many(keys.values()) if self.cache is not None else {}
        out = {c: cached[k] for c, k in keys.items() if k in cached}
        todo = [c for c in uniq if c not in out]

        def work(c: str) -> tuple[str, dict]:
            try:
                return c, self._lookup(c)
            except Exception as exc:  # network errors -> treated as "not found"
                print(f"[kb] lookup failed for {c!r}: {exc}")
                return c, {"wikipedia": [], "wiktionary": [], "error": str(exc)}

        if todo:
            with ThreadPoolExecutor(self.threads) as ex:
                for i, (c, res) in enumerate(ex.map(work, todo)):
                    out[c] = res
                    if self.cache is not None and "error" not in res:
                        self.cache.set(keys[c], res)
                    if (i + 1) % 500 == 0:
                        print(f"[kb] {i + 1}/{len(todo)} concepts looked up")
        return out
