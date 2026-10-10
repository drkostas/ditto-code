import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from understudy import claude
from understudy.memory import GB
from understudy.proxy import Config, Proxy, handler_for


class FakeOllama(BaseHTTPRequestHandler):
    seen: list = []
    models = [{"name": "understudy/small", "digest": "a", "size": int(6 * GB)},
              {"name": "understudy/big", "digest": "b", "size": int(20 * GB)}]
    loaded: list = []

    def log_message(self, *a):
        pass

    def _json(self, obj):
        b = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path == "/api/tags":
            return self._json({"models": self.models})
        if self.path == "/api/ps":
            return self._json({"models": self.loaded})
        if self.path == "/api/version":
            return self._json({"version": "test"})
        self.send_error(404)

    delay = 0.0                        # seconds before answering, like reading a long prompt

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/api/show":
            return self._json({"parameters": "temperature 0.6\nnum_ctx 131072\n"})
        FakeOllama.seen.append((self.path, body, self.headers.get("anthropic-version")))
        import time
        time.sleep(self.delay)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for i in range(3):
            self.wfile.write(f"event: content_block_delta\ndata: {json.dumps({'i': i})}\n\n".encode())
            self.wfile.flush()
        # Ollama reports the prompt's token count in the usage at the end of the answer
        self.wfile.write(b'event: message_delta\ndata: {"usage": {"input_tokens": 2000, "output_tokens": 3}}\n\n')


def start(handler):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


@pytest.fixture
def proxy(tmp_path):
    FakeOllama.seen, FakeOllama.loaded, FakeOllama.delay = [], [], 0.0
    up = start(FakeOllama)
    free = {"bytes": int(12 * GB)}
    cfg = Config(ollama=f"http://127.0.0.1:{up.server_port}", port=0, state=tmp_path,
                 headroom=int(2 * GB), budget=int(25 * GB))
    p = Proxy(cfg, available=lambda: free["bytes"], total=lambda: int(32 * GB))
    srv = start(handler_for(p))
    yield f"http://127.0.0.1:{srv.server_port}", free, p
    srv.shutdown()
    up.shutdown()


def post(url, body):
    req = urllib.request.Request(url + "/v1/messages", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "anthropic-version": "2023-06-01"})
    return urllib.request.urlopen(req, timeout=10)


def test_the_request_goes_to_the_largest_model_that_fits_and_streams_back(proxy):
    url, free, _ = proxy
    r = post(url, {"model": "local", "stream": True, "messages": [{"role": "user", "content": "hi"}]})
    text = r.read().decode()
    assert text.count("content_block_delta") == 3
    path, body, version = FakeOllama.seen[-1]
    # 12 GB free: the 20 GB rung needs ~24 with its estimate, the 6 GB one ~8.6 plus 2 of margin
    assert path == "/v1/messages" and body["model"] == "understudy/small"
    assert version == "2023-06-01" and body["messages"][0]["content"] == "hi"
    # thinking is turned off, so the next turn extends what the model has cached
    assert body["thinking"] == {"type": "disabled"}

    free["bytes"] = int(40 * GB)
    post(url, {"model": "local", "messages": []}).read()
    assert FakeOllama.seen[-1][1]["model"] == "understudy/big"


def test_not_enough_memory_is_overloaded_with_the_reason(proxy):
    url, free, _ = proxy
    free["bytes"] = int(3 * GB)
    with pytest.raises(urllib.error.HTTPError) as e:
        post(url, {"model": "local", "messages": []})
    assert e.value.code == 529
    err = json.loads(e.value.read())["error"]
    assert err["type"] == "overloaded_error" and "not enough free memory" in err["message"]
    assert FakeOllama.seen == []


def test_status_reports_the_ladder_and_the_next_choice(proxy):
    url, _, _ = proxy
    s = json.loads(urllib.request.urlopen(url + "/status", timeout=10).read())
    assert [r["name"] for r in s["ladder"]] == ["understudy/big", "understudy/small"]
    assert s["next"]["model"] == "understudy/small" and s["available_gb"] == 12.0
    assert json.loads(urllib.request.urlopen(url + "/health", timeout=10).read())["ok"] is True


def test_a_loaded_model_is_measured_after_the_request(proxy, tmp_path):
    url, _, p = proxy
    FakeOllama.loaded = [{"name": "understudy/small", "digest": "a", "size": int(7.5 * GB)}]
    p.measure()
    assert p.measured.get("understudy/small", "a") == int(7.5 * GB)


def test_a_local_chat_is_claude_with_a_settings_file(tmp_path):
    path = tmp_path / "s.json"
    assert claude.write_settings(path, "http://127.0.0.1:11435") is True
    assert claude.write_settings(path, "http://127.0.0.1:11435") is False
    s = json.loads(path.read_text())
    env = s["env"]
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:11435" and env["ANTHROPIC_AUTH_TOKEN"]
    # every kind of request (main loop, background, subagents) names the local model
    assert {env[k] for k in ("ANTHROPIC_MODEL", "ANTHROPIC_DEFAULT_OPUS_MODEL",
                             "ANTHROPIC_DEFAULT_SONNET_MODEL", "ANTHROPIC_DEFAULT_HAIKU_MODEL",
                             "CLAUDE_CODE_SUBAGENT_MODEL")} == {"local"}
    argv = claude.command(path, ["-n", "x"])
    assert argv[0].endswith("claude") and argv[1:] == ["--settings", str(path), "-n", "x"]


def test_a_request_longer_than_every_window_is_refused(proxy):
    url, free, _ = proxy
    free["bytes"] = int(64 * GB)
    long = "x" * 450_000                       # about 150,000 tokens, more than the 131,072 window
    with pytest.raises(urllib.error.HTTPError) as e:
        post(url, {"model": "local", "max_tokens": 1000,
                   "messages": [{"role": "user", "content": long}]})
    assert e.value.code == 500 and "window" in json.loads(e.value.read())["error"]["message"]


def test_bytes_per_token_is_learned_from_what_ollama_counted(proxy, tmp_path):
    url, free, p = proxy
    assert p.ratio() == 3.5                       # the default before any measurement
    body = {"model": "local", "stream": True, "messages": [{"role": "user", "content": "y" * 9000}]}
    post(url, body).read()
    sent = len(json.dumps({**body, "model": "understudy/small", "thinking": {"type": "disabled"}}))
    assert abs(p.ratio() - sent / 2000) < 0.01   # what was sent over what Ollama counted
    assert json.loads((tmp_path / "ratios.json").read_text())


def test_a_local_chat_keeps_only_the_mcp_servers_it_is_told_to(tmp_path):
    (tmp_path / ".claude.json").write_text(json.dumps({"mcpServers": {"drlab": {}, "github": {},
                                                                      "playwright-1": {}}}))
    deny = claude.mcp_deny(["drlab"], home=tmp_path)
    assert "mcp__github" in deny and "mcp__playwright-1" in deny and "mcp__drlab" not in deny
    assert {"mcp__plugin_*", "mcp__claude_ai_*", "mcp__claude-in-chrome"} <= set(deny)
    path = tmp_path / "s.json"
    claude.write_settings(path, "http://127.0.0.1:11435", keep_mcp=["drlab"], home=tmp_path)
    assert json.loads(path.read_text())["permissions"]["deny"] == deny


def test_a_slow_answer_is_kept_alive_with_pings(proxy):
    url, free, p = proxy
    p.cfg.ping_every = 0.2
    FakeOllama.delay = 1.0                     # Ollama reads the prompt and sends nothing for a while
    text = post(url, {"model": "local", "stream": True,
                      "messages": [{"role": "user", "content": "hi"}]}).read().decode()
    assert text.count("event: ping") >= 3      # the client heard from us while Ollama was silent
    assert text.index("event: ping") < text.index("content_block_delta")
    assert text.count("content_block_delta") == 3


def test_a_local_chat_waits_as_long_as_reading_takes():
    env = claude.settings()["env"]
    assert env["API_FORCE_IDLE_TIMEOUT"] == "0"
    assert env["CLAUDE_STREAM_FIRST_BYTE_TIMEOUT_MS"] == env["CLAUDE_BYTE_STREAM_IDLE_TIMEOUT_MS"] == "1800000"
