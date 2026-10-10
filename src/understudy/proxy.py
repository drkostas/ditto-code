"""The proxy: an Anthropic Messages endpoint that hands each request to the right local model.

Claude Code talks to it exactly as it talks to Anthropic. Every request names the model "local"
(any name is accepted); the proxy chooses a rung of the ladder (see ladder.py), writes that name
into the request, and passes it to Ollama's own Anthropic-compatible endpoint, streaming the answer
back unchanged. It listens on the loopback only and holds no credential.

    GET  /health        is Ollama answering
    GET  /status        free memory, the ladder, what is loaded, what the next request would get,
                        and the last choices made
    POST /v1/messages   (and anything else under /v1/) forwarded to Ollama

Two things are changed in a request on the way. The model name, and thinking, which is turned off
unless `thinking` is set: Qwen's templates drop earlier reasoning from the history, so with
thinking on the next turn no longer extends what the model has cached and it reads the whole
conversation again (measured 2026-10-10: 4.1 s against 0.1 s for a 16,000-token prompt, minutes
for a real chat).
"""
from __future__ import annotations

import http.client
import json
import re
import socket
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from . import ladder, memory
from .memory import GB, gb


@dataclass
class Config:
    ollama: str = "http://127.0.0.1:11434"
    host: str = "127.0.0.1"
    port: int = 11435
    prefix: str = "understudy/"
    headroom: int = int(2 * GB)
    budget: int | None = None            # the GPU working-set limit; None = derive from total RAM
    state: Path = field(default_factory=lambda: Path.home() / ".local" / "state" / "understudy")
    timeout: float = 3600.0              # a first turn reads the whole conversation: minutes
    thinking: bool = False               # see the module docstring
    output_cap: int = 16000              # the most of max_tokens counted toward the window
    ping_every: float = 15.0             # seconds between keep-alive events while Ollama reads
    ratio: float = 3.5                   # request bytes per prompt token until one is measured
                                         # (a Claude Code request measured 4.04; lower is safer)

    def gpu_budget(self, total: int | None) -> int | None:
        """Apple Silicon lets the GPU wire about three quarters of memory by default (Ollama logs
        25.0 GiB on a 32 GB machine). Set `budget` to the logged figure to be exact."""
        if self.budget is not None:
            return self.budget
        return int(total * 0.78) if total else None


def _get_json(url: str, timeout: float = 10.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


class Proxy:
    def __init__(self, cfg: Config, available=memory.available, total=memory.total):
        self.cfg = cfg
        self._available, self._total = available, total
        self.measured = ladder.Measurements(cfg.state / "measured.json")
        self.recent: deque = deque(maxlen=50)
        self._choose_lock = threading.Lock()
        self._contexts: dict[tuple[str, str], int | None] = {}
        self._ratios: deque = deque(self._load_ratios(), maxlen=20)

    # ---- how many tokens a request is, learned from what Ollama counted
    def _load_ratios(self) -> list[float]:
        try:
            return [float(x) for x in json.loads((self.cfg.state / "ratios.json").read_text())]
        except (OSError, ValueError, TypeError):
            return []

    def ratio(self) -> float:
        """Request bytes per prompt token: the smallest of the recent measurements (the one that
        estimates the most tokens), or the configured default before any."""
        return min(self._ratios) if self._ratios else self.cfg.ratio

    def observe(self, nbytes: int, tokens: int) -> None:
        if tokens < 1000:
            return                       # short requests say little about long ones
        self._ratios.append(nbytes / tokens)
        try:
            self.cfg.state.mkdir(parents=True, exist_ok=True)
            (self.cfg.state / "ratios.json").write_text(json.dumps(list(self._ratios)))
        except OSError:
            pass

    # ---- what Ollama has
    def tags(self) -> list[dict]:
        return _get_json(self.cfg.ollama + "/api/tags").get("models", [])

    def loaded(self) -> list[dict]:
        return _get_json(self.cfg.ollama + "/api/ps").get("models", [])

    def context_of(self, name: str, digest: str) -> int | None:
        """A rung's num_ctx, from Ollama's /api/show, once per model digest."""
        key = (name, digest)
        if key not in self._contexts:
            req = urllib.request.Request(self.cfg.ollama + "/api/show",
                                         json.dumps({"model": name}).encode(),
                                         {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=10) as r:
                self._contexts[key] = ladder.num_ctx(json.loads(r.read()).get("parameters"))
        return self._contexts[key]

    def ladder(self) -> list[ladder.Rung]:
        tags = self.tags()
        contexts = {m["name"]: self.context_of(m["name"], m.get("digest", ""))
                    for m in tags if m.get("name", "").startswith(self.cfg.prefix)}
        return ladder.rungs(tags, self.cfg.prefix, self.measured, contexts)

    def needed(self, raw: bytes, body: dict) -> int:
        """The window a request needs: its prompt, from the bytes sent and the measured bytes per
        token with 5% to spare, plus room for the answer."""
        return (int(len(raw) / self.ratio() * 1.05)
                + min(int(body.get("max_tokens") or 0), self.cfg.output_cap))

    def choose(self, needed: int = 0) -> ladder.Choice:
        with self._choose_lock:
            resident = {m["name"]: int(m.get("size", 0)) for m in self.loaded()}
            total = self._total()
            c = ladder.choose(self.ladder(), self._available(), resident, self.cfg.headroom,
                              self.cfg.gpu_budget(total), needed)
            c.details["at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            self.recent.appendleft(c.as_dict())
            self._log(c.as_dict())
            return c

    def measure(self) -> None:
        """Record the resident size of every loaded rung, so its need is measured from now on."""
        try:
            for m in self.loaded():
                if m.get("name", "").startswith(self.cfg.prefix):
                    self.measured.put(m["name"], m.get("digest", ""), int(m.get("size", 0)))
        except (OSError, ValueError, urllib.error.URLError):
            pass

    def _log(self, row: dict) -> None:
        p = self.cfg.state / "choices.log"
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            if p.exists() and p.stat().st_size > 1_000_000:
                p.write_text("".join(p.read_text().splitlines(True)[-2000:]))
            with p.open("a") as f:
                f.write(json.dumps(row) + "\n")
        except OSError:
            pass

    def status(self) -> dict:
        total = self._total()
        out = {"listen": f"http://{self.cfg.host}:{self.cfg.port}", "ollama": self.cfg.ollama,
               "total_gb": gb(total), "available_gb": gb(self._available()),
               "gpu_budget_gb": gb(self.cfg.gpu_budget(total)), "headroom_gb": gb(self.cfg.headroom)}
        try:
            out["loaded"] = [{"name": m["name"], "size_gb": gb(int(m.get("size", 0))),
                              "context": m.get("context_length"), "until": m.get("expires_at")}
                             for m in self.loaded()]
            out["ladder"] = [r.as_dict() for r in self.ladder()]
            resident = {m["name"]: m["size_gb"] * GB for m in out["loaded"]}
            nxt = ladder.choose(self.ladder(), self._available(),
                                {k: int(v) for k, v in resident.items()},
                                self.cfg.headroom, self.cfg.gpu_budget(total))
            out["next"] = nxt.as_dict()
        except (OSError, ValueError, urllib.error.URLError) as e:
            out["error"] = f"Ollama did not answer: {e}"
        out["bytes_per_token"] = round(self.ratio(), 2)
        out["recent"] = list(self.recent)[:10]
        return out


INPUT_TOKENS = re.compile(rb'"input_tokens"\s*:\s*(\d+)')
PING = b'event: ping\ndata: {"type": "ping"}\n\n'


def _abort(conn: http.client.HTTPConnection) -> None:
    """Close the connection to Ollama from outside the thread blocked on it; Ollama cancels."""
    try:
        if conn.sock is not None:
            conn.sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    conn.close()


def _error(kind: str, message: str) -> bytes:
    return json.dumps({"type": "error", "error": {"type": kind, "message": message}}).encode()


def handler_for(proxy: Proxy):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"     # the end of a streamed answer is the connection closing
        server_version = "understudy"

        def log_message(self, fmt, *args):
            pass

        def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/health":
                try:
                    v = _get_json(proxy.cfg.ollama + "/api/version", timeout=5)
                    return self._send(200, json.dumps({"ok": True, "ollama": v.get("version")}).encode())
                except (OSError, ValueError, urllib.error.URLError) as e:
                    return self._send(503, json.dumps({"ok": False, "error": str(e)}).encode())
            if path == "/status":
                return self._send(200, json.dumps(proxy.status(), indent=2).encode())
            if path.startswith("/v1/"):
                return self._forward(None)
            return self._send(404, _error("not_found_error", f"no route {path}"))

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(n) if n else b""
            if not urlparse(self.path).path.startswith("/v1/"):
                return self._send(404, _error("not_found_error", f"no route {self.path}"))
            try:
                body = json.loads(raw) if raw else {}
            except ValueError:
                return self._send(400, _error("invalid_request_error", "the body is not JSON"))
            if "model" in body:
                try:
                    c = proxy.choose(proxy.needed(raw, body))
                except (OSError, ValueError, urllib.error.URLError) as e:
                    return self._send(502, _error("api_error", f"understudy: Ollama did not answer: {e}"))
                if c.rung is None:
                    # 529 is "overloaded": Claude Code shows the message and retries with backoff,
                    # which is right for memory that may free up in a minute.
                    code = 529 if c.pool is not None else 500
                    return self._send(code, _error("overloaded_error" if code == 529 else "api_error",
                                                   "understudy: " + c.reason))
                body["model"] = c.rung.name
                if not proxy.cfg.thinking:
                    body["thinking"] = {"type": "disabled"}
            self._forward(json.dumps(body).encode(), stream=bool(body.get("stream")))

        def _forward(self, data: bytes | None, stream: bool = False):
            """Pass the request to Ollama and its answer back.

            A streamed request is answered at once, and a ping event goes to the client every
            `ping_every` seconds until Ollama starts its answer. Reading a long prompt takes minutes
            on a laptop and Ollama sends nothing meanwhile, so without them the client gives up
            (Claude Code timed out at about five minutes). A ping that cannot be delivered means the
            client left, and closing the connection to Ollama cancels the work there, so a retry
            does not wait behind a request nobody is reading."""
            u = urlparse(proxy.cfg.ollama)
            conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=proxy.cfg.timeout)
            headers = {k: v for k, v in self.headers.items()
                       if k.lower() in ("content-type", "accept", "anthropic-version")}
            if data is not None:
                headers["Content-Length"] = str(len(data))
            tokens = 0
            got: dict = {}

            def call():
                try:
                    conn.request(self.command, self.path, body=data, headers=headers)
                    got["r"] = conn.getresponse()
                except OSError as e:
                    got["e"] = e

            worker = threading.Thread(target=call, daemon=True)
            worker.start()
            if stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                while worker.is_alive():
                    worker.join(proxy.cfg.ping_every)
                    if worker.is_alive():
                        try:
                            self.wfile.write(PING)
                            self.wfile.flush()
                        except (BrokenPipeError, ConnectionResetError):
                            _abort(conn)
                            return
            else:
                worker.join()
            if "e" in got or "r" not in got:
                conn.close()
                msg = f"understudy: Ollama did not answer: {got.get('e')}"
                if stream:
                    return self._event_error("api_error", msg)
                return self._send(502, _error("api_error", msg))
            r = got["r"]
            try:
                if stream and r.status != 200:
                    detail = r.read().decode(errors="replace")[:500]
                    return self._event_error("api_error", f"understudy: Ollama answered {r.status}: {detail}")
                if not stream:
                    self.send_response(r.status)
                    for k, v in r.getheaders():
                        if k.lower() in ("content-type", "cache-control"):
                            self.send_header(k, v)
                    self.send_header("Connection", "close")
                    self.end_headers()
                tail = b""
                while True:
                    chunk = r.read1(65536)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
                    # the prompt's token count, as Ollama reports it in the answer's usage
                    for m in INPUT_TOKENS.finditer(tail[-64:] + chunk):
                        tokens = max(tokens, int(m.group(1)))
                    tail = chunk
            except (BrokenPipeError, ConnectionResetError):
                pass                              # the client went away; closing tells Ollama
            finally:
                conn.close()
                if data is not None:
                    if tokens:
                        proxy.observe(len(data), tokens)
                    threading.Thread(target=proxy.measure, daemon=True).start()

        def _event_error(self, kind: str, message: str) -> None:
            """An error after the 200 has gone out: the way the Anthropic API reports one mid-stream."""
            try:
                self.wfile.write(b"event: error\ndata: " + _error(kind, message) + b"\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

    return Handler


def serve(cfg: Config) -> None:
    proxy = Proxy(cfg)
    srv = ThreadingHTTPServer((cfg.host, cfg.port), handler_for(proxy))
    srv.daemon_threads = True
    print(f"understudy listening on http://{cfg.host}:{cfg.port}, models from {cfg.ollama} "
          f"named {cfg.prefix}*", flush=True)
    srv.serve_forever()
