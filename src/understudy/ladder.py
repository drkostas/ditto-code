"""The model ladder, and the choice of rung for a request.

A rung is any Ollama model whose name starts with the configured prefix (`understudy/` by
default). The ladder is read from Ollama on every choice, so adding or removing a model there is
the whole of changing it; nothing here keeps a second list that could drift.

What a rung needs in memory is measured, not guessed, wherever possible: once a rung has been
loaded, Ollama reports its resident size (weights, context cache and buffers), and that number is
kept per model digest. Until then the need is estimated from the size on disk, deliberately on the
high side, and the estimate says it is one.

The choice, in order:
  1. a rung that is already loaded keeps the request. Reloading costs the whole conversation being
     read again, which takes minutes on a laptop, so a loaded model is never swapped out for a
     bigger one in the middle of work. Ollama unloads it after it has been idle, and the next
     request chooses afresh.
  2. otherwise the largest rung that fits: its need, plus a margin for everything else on the
     machine, within what the system can hand out now (plus whatever Ollama holds, which it
     releases before loading another model) and within the GPU's memory budget.
  3. otherwise nothing, with the numbers that said so. Loading anyway would push the person's own
     programs into swap, which is the one outcome this exists to prevent.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path

from .memory import GB, gb

#: until a rung has been measured: disk size, plus this share of it, plus this many bytes for the
#: context cache and compute buffers. Measured on the first load and replaced then.
ESTIMATE_SHARE = 0.10
ESTIMATE_FIXED = int(2.0 * GB)


@dataclass(frozen=True)
class Rung:
    name: str
    digest: str
    disk: int
    need: int
    measured: bool

    def as_dict(self) -> dict:
        return {"name": self.name, "need_gb": gb(self.need), "disk_gb": gb(self.disk),
                "measured": self.measured}


@dataclass
class Choice:
    rung: Rung | None
    reason: str
    available: int | None
    pool: int | None
    details: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"model": self.rung.name if self.rung else None, "reason": self.reason,
                "available_gb": gb(self.available), "pool_gb": gb(self.pool), **self.details}


class Measurements:
    """Resident sizes seen while a rung was loaded, kept per name and digest in one JSON file."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()

    def _read(self) -> dict:
        try:
            return json.loads(self.path.read_text())
        except (OSError, ValueError):
            return {}

    def get(self, name: str, digest: str) -> int | None:
        row = self._read().get(name)
        return row["size"] if row and row.get("digest") == digest else None

    def put(self, name: str, digest: str, size: int) -> bool:
        with self._lock:
            data = self._read()
            row = data.get(name)
            if row and row.get("digest") == digest and row.get("size", 0) >= size:
                return False            # keep the largest seen: the context grows over a chat
            data[name] = {"digest": digest, "size": size}
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=str(self.path.parent))
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2, sort_keys=True)
            os.replace(tmp, self.path)
            return True


def rungs(tags: list[dict], prefix: str, measured: Measurements) -> list[Rung]:
    """The ladder from Ollama's /api/tags models, largest need first."""
    out = []
    for m in tags:
        name = m.get("name") or m.get("model") or ""
        if not name.startswith(prefix):
            continue
        digest, disk = m.get("digest", ""), int(m.get("size", 0))
        size = measured.get(name, digest)
        need = size if size else int(disk * (1 + ESTIMATE_SHARE)) + ESTIMATE_FIXED
        out.append(Rung(name, digest, disk, need, size is not None))
    return sorted(out, key=lambda r: r.need, reverse=True)


def choose(ladder: list[Rung], available: int | None, loaded: dict[str, int],
           headroom: int, budget: int | None) -> Choice:
    """Which rung takes the next request. `loaded` maps model name to resident bytes (Ollama's
    /api/ps); `budget` is the GPU's working-set limit, None when unknown."""
    names = {r.name: r for r in ladder}
    for name in loaded:
        if name in names:
            return Choice(names[name], "already loaded", available, None)
    if not ladder:
        return Choice(None, "no models on the ladder (none named with the prefix)", available, None)
    if available is None:
        return Choice(None, "free memory could not be read, so no model is loaded blind",
                      None, None)
    pool = available + sum(loaded.values())
    for r in ladder:
        if budget is not None and r.need > budget:
            continue
        if r.need + headroom <= pool:
            return Choice(r, "largest that fits", available, pool,
                          {"need_gb": gb(r.need), "headroom_gb": gb(headroom)})
    smallest = ladder[-1]
    return Choice(None, f"not enough free memory: {gb(pool)} GB can be freed without swapping, "
                        f"the smallest model ({smallest.name}) needs {gb(smallest.need)} GB "
                        f"plus {gb(headroom)} GB of margin. Close some programs or chats and try "
                        "again.", available, pool)
