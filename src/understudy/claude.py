"""Point one Claude Code chat at the proxy, and leave every other chat on Anthropic.

A local chat is an ordinary `claude` started with `--settings <file>`, where the file's `env`
block sends every request (the main loop, background work such as titles, and subagents) to the
proxy under the model name "local". Nothing global changes: the claude.ai login stays saved and
unused for that one chat, and the flag shows in the chat's command line, so a tool that restarts
chats from their command line keeps it local.

Two things a local chat does not have: Remote Control (Claude Code turns it off whenever requests
go anywhere but Anthropic) and Anthropic's own server tools such as web search.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

MODEL = "local"
CONTEXT = 262144          # the largest rung window; Claude Code compacts before it


def settings(url: str = "http://127.0.0.1:11435", context: int = CONTEXT) -> dict:
    env = {
        "ANTHROPIC_BASE_URL": url,
        "ANTHROPIC_AUTH_TOKEN": "understudy",        # any value: the proxy checks none, but one
                                                     # must be set for the base URL to replace the login
        "ANTHROPIC_MODEL": MODEL,
        "ANTHROPIC_DEFAULT_OPUS_MODEL": MODEL,
        "ANTHROPIC_DEFAULT_SONNET_MODEL": MODEL,
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": MODEL,
        "CLAUDE_CODE_SUBAGENT_MODEL": MODEL,
        "CLAUDE_CODE_MAX_CONTEXT_TOKENS": str(context),
        "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "16000",
        "API_TIMEOUT_MS": "3600000",                 # a first turn on a laptop takes minutes
        "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
    }
    return {"model": MODEL, "env": env}


def default_path(home: Path | None = None) -> Path:
    return (home or Path.home()) / ".config" / "understudy" / "claude-settings.json"


def write_settings(path: Path, url: str, context: int = CONTEXT) -> bool:
    body = json.dumps(settings(url, context), indent=2) + "\n"
    if path.exists() and path.read_text() == body:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(body)
    os.replace(tmp, path)
    return True


def command(path: Path, args: list[str]) -> list[str]:
    """argv for a local chat: `claude --settings <file> <args>`. argv[0] stays `claude`."""
    return [shutil.which("claude") or "claude", "--settings", str(path), *args]


def link_skill(home: Path | None = None) -> str:
    home = home or Path.home()
    src = Path(__file__).parent / "skills" / "understudy"
    link = home / ".claude" / "skills" / "understudy"
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        if Path(os.readlink(link)) == src:
            return f"skill already linked: {link}"
        link.unlink()
    elif link.exists():
        return f"skill left alone: {link} exists and is not a link"
    link.symlink_to(src)
    return f"skill linked: {link} -> {src}"
