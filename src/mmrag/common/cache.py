"""SQLite-backed key/value cache.

Every expensive call in this repository (LLM agents, Wikipedia lookups,
MLLM generations, judge calls) goes through this cache, so a Colab session
that is interrupted can simply be re-run: finished items are read back from
disk instead of being recomputed. Keep the database on Google Drive to make
the cache survive VM resets.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterable


class KVCache:
    def __init__(self, path: str | Path, table: str = "kv"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not table.isidentifier():
            raise ValueError(f"invalid table name: {table!r}")
        self.table = table
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False, timeout=60)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            f"CREATE TABLE IF NOT EXISTS {self.table} (k TEXT PRIMARY KEY, v TEXT NOT NULL)"
        )
        self._conn.commit()

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            row = self._conn.execute(
                f"SELECT v FROM {self.table} WHERE k = ?", (key,)
            ).fetchone()
        return default if row is None else json.loads(row[0])

    def get_many(self, keys: Iterable[str]) -> dict[str, Any]:
        keys = list(keys)
        out: dict[str, Any] = {}
        # SQLite has a limit on the number of host parameters per statement.
        for i in range(0, len(keys), 500):
            chunk = keys[i : i + 500]
            marks = ",".join("?" * len(chunk))
            with self._lock:
                rows = self._conn.execute(
                    f"SELECT k, v FROM {self.table} WHERE k IN ({marks})", chunk
                ).fetchall()
            out.update({k: json.loads(v) for k, v in rows})
        return out

    def set(self, key: str, value: Any) -> None:
        self.set_many({key: value})

    def set_many(self, items: dict[str, Any]) -> None:
        if not items:
            return
        payload = [(k, json.dumps(v, ensure_ascii=False)) for k, v in items.items()]
        with self._lock:
            self._conn.executemany(
                f"INSERT OR REPLACE INTO {self.table} (k, v) VALUES (?, ?)", payload
            )
            self._conn.commit()

    def __contains__(self, key: str) -> bool:
        return self.get(key, _MISSING) is not _MISSING

    def __len__(self) -> int:
        with self._lock:
            return self._conn.execute(f"SELECT COUNT(*) FROM {self.table}").fetchone()[0]

    def close(self) -> None:
        with self._lock:
            self._conn.close()


_MISSING = object()
