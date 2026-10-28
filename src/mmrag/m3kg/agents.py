"""The seven M3KG construction agents (paper Sec. 3.1, Fig. 2).

Each agent works on a *whole stage* at once (one batched LLM call per stage
and per inspector round) rather than clip by clip; this is what makes the
construction affordable on one GPU. Parsing helpers are pure functions and
are unit-tested.
"""
from __future__ import annotations

import json
import re

from mmrag.m3kg import prompts as P
from mmrag.m3kg.llm import CachedLLM, ChatRequest

MAX_TRIPLES_PER_CLIP = 16


# ---------------------------------------------------------------- parsing ----
def parse_triples(text: str, max_n: int = MAX_TRIPLES_PER_CLIP) -> list[tuple[str, str, str]]:
    """Parse '(h, r, t)' lines. Entities may contain commas: the first field
    is the head, the last the tail and everything between the relation."""
    out, seen = [], set()
    for line in text.splitlines():
        line = line.strip().lstrip("-*0123456789. ").strip()
        m = re.match(r"^\((.*)\)\s*[.,;]?$", line)
        if not m:
            continue
        parts = [p.strip().strip("\"'") for p in m.group(1).split(",")]
        if len(parts) < 3:
            continue
        h, t = parts[0], parts[-1]
        r = ", ".join(parts[1:-1]).strip()
        if not (h and r and t):
            continue
        key = (h.lower(), r.lower(), t.lower())
        if key not in seen:
            seen.add(key)
            out.append((h, r, t))
        if len(out) >= max_n:
            break
    return out


def clean_concept(text: str, fallback: str) -> str:
    line = (text or "").strip().splitlines()[0] if (text or "").strip() else ""
    for _ in range(2):  # e.g. '"Jack Russell Terrier".'
        line = line.strip().rstrip(".").strip().strip("\"'`").strip()
    line = re.sub(r"^(concept|output)\s*:\s*", "", line, flags=re.I)
    return line if 0 < len(line) <= 80 else fallback


def parse_index(text: str, n: int) -> int | None:
    m = re.search(r"\d+", text or "")
    if not m:
        return None
    i = int(m.group())
    return i - 1 if 1 <= i <= n else None


def parse_score(text: str) -> int:
    m = re.search(r"\b(10|[0-9])\b", text or "")
    return int(m.group(1)) if m else 0


def clean_description(text: str) -> str:
    t = re.sub(r"\s+", " ", (text or "").strip()).strip("\"'")
    return t[:600]


def parse_kept_indices(text: str, n: int) -> list[int]:
    """Indices kept by the GRASP LLM filter. On any parsing problem we keep
    everything (the conservative policy of the paper)."""
    m = re.search(r"\[[^\]]*\]", text or "")
    if not m:
        return list(range(n))
    try:
        vals = json.loads(m.group())
    except json.JSONDecodeError:
        return list(range(n))
    keep = sorted({int(v) for v in vals if isinstance(v, (int, float)) or str(v).isdigit()})
    return [i for i in keep if 0 <= i < n]  # the prompt numbers triples from 0


# ---------------------------------------------------------------- agents -----
class Agents:
    def __init__(self, llm: CachedLLM, max_tokens: int = 256):
        self.llm = llm
        self.max_tokens = max_tokens

    def _ask(self, system: str, users: list[str], max_tokens: int | None = None,
             temperature: float = 0.0, seeds: list[int] | None = None) -> list[str]:
        seeds = seeds or [0] * len(users)
        reqs = [ChatRequest(system, u, max_tokens or self.max_tokens, temperature, s) for u, s in zip(users, seeds)]
        return self.llm.complete(reqs)

    def rewrite(self, items: list[dict]) -> list[str]:
        users = [P.REWRITER_USER.format(title=i["title"], description=(i["description"] or "")[:1500],
                                        caption=i["caption"]) for i in items]
        outs = self._ask(P.REWRITER_SYSTEM, users, max_tokens=128)
        return [clean_description(o) or i["caption"] for o, i in zip(outs, items)]

    def extract(self, captions: list[str]) -> list[list[tuple[str, str, str]]]:
        outs = self._ask(P.EXTRACTOR_SYSTEM, [P.EXTRACTOR_USER.format(caption=c) for c in captions], max_tokens=384)
        return [parse_triples(o) for o in outs]

    def normalize(self, mentions: list[str]) -> list[str]:
        outs = self._ask(P.NORMALIZER_SYSTEM, [P.NORMALIZER_USER.format(mention=m) for m in mentions], max_tokens=24)
        return [clean_concept(o, m) for o, m in zip(outs, mentions)]

    def callback(self, pairs: list[tuple[str, str]], seeds: list[int] | None = None, temperature: float = 0.0) -> list[str]:
        users = [P.CALLBACK_USER.format(concept=c, caption=cap) for c, cap in pairs]
        return [clean_description(o) for o in self._ask(P.CALLBACK_SYSTEM, users, 96, temperature, seeds)]

    def select(self, items: list[tuple[str, str, list[str]]]) -> list[int]:
        users = []
        for concept, caption, cands in items:
            listing = "\n".join(f"{k + 1}. {c}" for k, c in enumerate(cands))
            users.append(P.SELECTOR_USER.format(concept=concept, caption=caption, candidates=listing))
        outs = self._ask(P.SELECTOR_SYSTEM, users, max_tokens=8)
        return [parse_index(o, len(c[2])) or 0 for o, c in zip(outs, items)]

    def refine(self, items: list[tuple[str, str, str]], seeds: list[int] | None = None,
               temperature: float = 0.0) -> list[str]:
        users = [P.REFINER_USER.format(mention=m, concept=c, description=d) for m, c, d in items]
        return [clean_description(o) for o in self._ask(P.REFINER_SYSTEM, users, 128, temperature, seeds)]

    def inspect(self, items: list[tuple[str, str, str]]) -> list[int]:
        users = [P.INSPECTOR_USER.format(mention=m, caption=cap, description=d) for m, cap, d in items]
        return [parse_score(o) for o in self._ask(P.INSPECTOR_SYSTEM, users, max_tokens=4)]

    def grasp_filter(self, items: list[tuple[str, list[tuple[str, str, str]]]]) -> list[list[int]]:
        users = []
        for question, triples in items:
            listing = "\n".join(f"[{k}] ({h}, {r}, {t})" for k, (h, r, t) in enumerate(triples))
            users.append(P.GRASP_FILTER_USER.format(question=question, triples=listing))
        outs = self._ask(P.GRASP_FILTER_SYSTEM, users, max_tokens=256)
        return [parse_kept_indices(o, len(t)) for o, (_, t) in zip(outs, items)]
