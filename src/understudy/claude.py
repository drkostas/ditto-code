"""Point one Claude Code chat at the proxy, and leave every other chat on Anthropic.

A local chat is an ordinary `claude` started with `--settings <file>`, where the file's `env`
block sends every request (the main loop, background work such as titles, and subagents) to the
proxy under the model name "local". Nothing global changes: the claude.ai login stays saved and
unused for that one chat, and the flag shows in the chat's command line, so a tool that restarts
chats from their command line keeps it local.

Two things a local chat does not have: Remote Control (Claude Code turns it off whenever requests
go anywhere but Anthropic) and Anthropic's own server tools such as web search.

It also keeps only the MCP servers it is told to keep. Against anything but Anthropic, Claude Code
does not load MCP tools on demand: it sends every connected server's full schemas with every
request. With a few dozen servers that was about 1 MB, some 250,000 tokens, more than a local
model's whole window. A deny rule on a server removes its tools from the request.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

MODEL = "local"
# The window Claude Code is told, so it compacts before the conversation outgrows it. 128k, not the
# 256k the largest rungs have: on a 32 GB machine with other work open, a 9B at 256k (10.6 GB) does
# not fit beside it, and a chat told 256k never compacts and stalls once it passes 128k (measured
# 2026-10-11). Raise it with `understudy settings --context 262144` on a machine with room.
CONTEXT = 131072


def mcp_deny(keep: list[str] | tuple[str, ...] = (), home: Path | None = None) -> list[str]:
    """Deny rules for every MCP server except `keep`: each user server in ~/.claude.json by name,
    and every plugin's, claude.ai connector's and the Chrome extension's servers by pattern."""
    home = home or Path.home()
    try:
        servers = json.loads((home / ".claude.json").read_text()).get("mcpServers", {})
    except (OSError, ValueError):
        servers = {}
    rules = [f"mcp__{name}" for name in sorted(servers) if name not in keep]
    return rules + ["mcp__plugin_*", "mcp__claude_ai_*", "mcp__claude-in-chrome"]


def settings(url: str = "http://127.0.0.1:11435", context: int = CONTEXT,
             deny: list[str] | None = None) -> dict:
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
        # Reading a 100,000-token prompt took over five minutes with no bytes sent, and Claude Code
        # gave up at five: its idle timeout (on for non-Anthropic backends) is turned off and the
        # first-byte and byte watchdogs are set to their 30-minute maximum
        "API_FORCE_IDLE_TIMEOUT": "0",
        "CLAUDE_STREAM_FIRST_BYTE_TIMEOUT_MS": "1800000",
        "CLAUDE_BYTE_STREAM_IDLE_TIMEOUT_MS": "1800000",
        "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
    }
    out = {"model": MODEL, "env": env}
    if deny:
        out["permissions"] = {"deny": deny}
    return out


def default_path(home: Path | None = None) -> Path:
    return (home or Path.home()) / ".config" / "understudy" / "claude-settings.json"


def write_settings(path: Path, url: str, context: int = CONTEXT,
                   keep_mcp: list[str] | tuple[str, ...] = (), home: Path | None = None) -> bool:
    body = json.dumps(settings(url, context, mcp_deny(keep_mcp, home)), indent=2) + "\n"
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
