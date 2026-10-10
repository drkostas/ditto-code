"""understudy: run Claude Code on a local model, the largest one that fits in free memory.

    understudy serve          the proxy (an Anthropic Messages endpoint in front of Ollama)
    understudy status         free memory, the model ladder, and what the next request would get
    understudy claude [ARGS]  start Claude Code on the local model (`claude --settings <file> ARGS`)
    understudy settings       write the per-chat settings file and print its path
    understudy setup-claude   link the skill and write the settings file
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

from . import claude
from .memory import GB
from .proxy import Config, serve


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="understudy", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=os.environ.get("UNDERSTUDY_URL", "http://127.0.0.1:11435"),
                    help="where the proxy listens (default %(default)s)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--ollama", default=os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434"))
    s.add_argument("--prefix", default="understudy/", help="models with this name prefix are the ladder")
    s.add_argument("--headroom-gb", type=float, default=2.0,
                   help="memory left for everything else when choosing (default %(default)s)")
    s.add_argument("--budget-gb", type=float, default=None,
                   help="the GPU working-set limit Ollama logs at start (default: 78%% of RAM)")
    s.add_argument("--state", type=Path, default=None)
    sub.add_parser("status")
    c = sub.add_parser("claude", add_help=False)
    c.add_argument("args", nargs=argparse.REMAINDER)
    p = sub.add_parser("settings")
    p.add_argument("--context", type=int, default=claude.CONTEXT)
    sub.add_parser("setup-claude")
    a = ap.parse_args(argv)

    if a.cmd == "serve":
        from urllib.parse import urlparse
        u = urlparse(a.url)
        cfg = Config(ollama=a.ollama, host=u.hostname or "127.0.0.1", port=u.port or 11435,
                     prefix=a.prefix, headroom=int(a.headroom_gb * GB),
                     budget=int(a.budget_gb * GB) if a.budget_gb else None)
        if a.state:
            cfg.state = a.state
        serve(cfg)
        return 0

    if a.cmd == "status":
        try:
            with urllib.request.urlopen(a.url + "/status", timeout=20) as r:
                print(json.dumps(json.loads(r.read()), indent=2))
            return 0
        except (OSError, urllib.error.URLError) as e:
            print(f"the proxy at {a.url} did not answer: {e}", file=sys.stderr)
            return 2

    path = claude.default_path()
    if a.cmd == "settings":
        claude.write_settings(path, a.url, a.context)
        print(path)
        return 0
    if a.cmd == "setup-claude":
        print(("settings written: " if claude.write_settings(path, a.url) else "settings unchanged: ")
              + str(path))
        print(claude.link_skill())
        return 0
    if a.cmd == "claude":
        claude.write_settings(path, a.url)
        argv2 = claude.command(path, a.args)
        os.execv(argv2[0], argv2) if os.path.isabs(argv2[0]) else os.execvp(argv2[0], argv2)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
