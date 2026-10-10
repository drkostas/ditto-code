"""The model ladder, and the choice of rung for a request.

A rung is any Ollama model whose name starts with the configured prefix (`understudy/` by
default). The ladder is read from Ollama on every choice, so adding or removing a model there is
the whole of changing it; nothing here keeps a second list that could drift.

One model can be several rungs, one per context window (`understudy/qwen3.5-9b:64k`, `:128k`,
`:256k`). They share the weights on disk, but the context cache is reserved when a model loads, so
a short conversation on a small window leaves gigabytes free that a large window would take.

What a rung needs in memory is measured, not guessed, wherever possible: once a rung has been
loaded, Ollama reports its resident size (weights, context cache and buffers), and that number is
kept per model digest. Until then the need is estimated from the size on disk and the window,
deliberately on the high side, and the estimate says it is one.

The choice, for a request that needs N tokens of window, in order:
  1. a loaded rung whose window holds N keeps the request. Reloading costs the whole conversation
     being read again, which takes minutes on a laptop, so a loaded model is never swapped out for
     a bigger one in the middle of work. Ollama unloads it after it has been idle, and the next
     request chooses afresh.
  2. otherwise, models largest first, each at its smallest window that holds N: the first whose
     need, plus a margin for everything else on the machine, fits in what the system can hand out
     now (plus whatever Ollama holds, which it releases before loading another model) and within
     the GPU's memory budget.
  3. otherwise nothing, with the numbers that said so. Loading anyway would push the person's own
     programs into swap, which is the one outcome this exists to prevent.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path

from .memory import GB, gb

#: until a rung has been measured: disk size plus this share, a fixed amount for buffers, and this
#: many bytes per token of window for the context cache (Qwen3.5/3.6 need 10 to 17 KB a token at
#: q8_0, because only a quarter of their layers keep one). Measured on the first load.
ESTIMATE_SHARE = 0.05
ESTIMATE_FIXED = int(1.0 * GB)
ESTIMATE_PER_TOKEN = 20 * 1024


@dataclass(frozen=True)
class Rung:
    name: str
    digest: str
    disk: int
    need: int
    measured: bool
    context: int | None = None

    @property
    def model(self) -> str:
        return self.name.split(":", 1)[0]

    def as_dict(self) -> dict:
        return {"name": self.name, "context": self.context, "need_gb": gb(self.need),
                "disk_gb": gb(self.disk), "measured": self.measured}


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
                return False            # keep the largest seen
            data[name] = {"digest": digest, "size": size}
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=str(self.path.parent))
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2, sort_keys=True)
            os.replace(tmp, self.path)
            return True


def context_from_tag(name: str) -> int | None:
    """`model:64k` -> 65536, `model:256k` -> 262144; None when the tag does not say."""
    m = re.search(r":(\d+)k$", name)
    return int(m.group(1)) * 1024 if m else None


def num_ctx(parameters: str | None) -> int | None:
    """num_ctx from the `parameters` text of Ollama's /api/show."""
    m = re.search(r"^num_ctx\s+(\d+)", parameters or "", re.M)
    return int(m.group(1)) if m else None


def rungs(tags: list[dict], prefix: str, measured: Measurements,
          contexts: dict[str, int | None] | None = None) -> list[Rung]:
    """The ladder from Ollama's /api/tags models, largest need first. `contexts` maps a model name
    to its num_ctx (from /api/show); the tag (`:64k`) is the fallback."""
    contexts = contexts or {}
    out = []
    for m in tags:
        name = m.get("name") or m.get("model") or ""
        if not name.startswith(prefix):
            continue
        digest, disk = m.get("digest", ""), int(m.get("size", 0))
        ctx = contexts.get(name) or context_from_tag(name)
        size = measured.get(name, digest)
        need = size if size else (int(disk * (1 + ESTIMATE_SHARE)) + ESTIMATE_FIXED
                                  + (ctx or 131072) * ESTIMATE_PER_TOKEN)
        out.append(Rung(name, digest, disk, need, size is not None, ctx))
    return sorted(out, key=lambda r: r.need, reverse=True)


def choose(ladder: list[Rung], available: int | None, loaded: dict[str, int],
           headroom: int, budget: int | None, needed: int = 0) -> Choice:
    """Which rung takes the next request. `loaded` maps model name to resident bytes (Ollama's
    /api/ps); `budget` is the GPU's working-set limit, None when unknown; `needed` is the window
    the request needs, in tokens (its prompt and its answer)."""
    holds = [r for r in ladder if r.context and r.context >= needed]
    for name in loaded:
        for r in holds:
            if r.name == name:
                return Choice(r, "already loaded", available, None, {"needed_tokens": needed})
    if not ladder:
        return Choice(None, "no models on the ladder (none named with the prefix)", available, None)
    if not holds:
        biggest = max((r.context or 0) for r in ladder)
        return Choice(None, f"the conversation needs about {needed} tokens of window and the "
                            f"largest rung has {biggest}", available, None,
                      {"needed_tokens": needed})
    if available is None:
        return Choice(None, "free memory could not be read, so no model is loaded blind",
                      None, None)
    pool = available + sum(loaded.values())
    # models largest first (by their weights), each at its smallest window that holds the request
    best: dict[str, Rung] = {}
    for r in holds:
        if r.model not in best or (r.context or 0) < (best[r.model].context or 0):
            best[r.model] = r
    for r in sorted(best.values(), key=lambda x: x.disk, reverse=True):
        if budget is not None and r.need > budget:
            continue
        if r.need + headroom <= pool:
            return Choice(r, "largest that fits", available, pool,
                          {"need_gb": gb(r.need), "headroom_gb": gb(headroom),
                           "needed_tokens": needed})
    smallest = min(best.values(), key=lambda x: x.need)
    return Choice(None, f"not enough free memory: {gb(pool)} GB can be freed without swapping, "
                        f"the smallest model that holds this conversation ({smallest.name}) needs "
                        f"{gb(smallest.need)} GB plus {gb(headroom)} GB of margin. Close some "
                        "programs or chats and try again.", available, pool,
                  {"needed_tokens": needed})
