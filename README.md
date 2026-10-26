# understudy

understudy runs Claude Code on a local model. The chat looks and works the same as any other Claude Code chat, with the same tools, instructions, skills and MCP servers. Only the model is different, and it runs on your own machine through [Ollama](https://ollama.com).

It picks the model for you. You keep several sizes of a model in Ollama (the "ladder"), and for each request understudy uses the largest one that fits in the memory your machine can free right now without making other programs use swap.

## How it works

```
claude --settings claude-settings.json
   │   every request names the model "local"
   ▼
understudy proxy (127.0.0.1:11435)
   │   picks a rung of the ladder and writes its name into the request
   ▼
Ollama (127.0.0.1:11434), Anthropic Messages endpoint
```

The ladder is every Ollama model whose name starts with `understudy/`. One model can be several rungs, one for each context window (`understudy/qwen3.5-9b:64k`, `:128k` and `:256k`). They share the same weights on disk, but Ollama reserves the context cache when it loads a model, so a short conversation on a small window leaves gigabytes free.

For each request the proxy estimates how many tokens of window the conversation needs, and then does the following.

1. If a model is already loaded and its window holds the conversation, it uses that one. Changing models in the middle of a chat would mean reading the whole conversation again, which takes minutes on a laptop.
2. Otherwise it takes the models from largest to smallest, each at its smallest window that holds the conversation, and uses the first whose memory need, plus a margin of 2 GB, fits in the memory the system can free now and within the GPU's memory limit.
3. If none fits, it refuses with HTTP 529 and says how much memory was free and how much the smallest model needs. Claude Code shows that message and tries again after a pause.

The proxy also turns thinking off. Qwen's chat templates leave earlier reasoning out of the history, so with thinking on, each new turn no longer continues what the model has cached, and it reads the whole conversation again. On a 16,000-token prompt that was 4.1 seconds against 0.1 seconds with thinking off, and on a real chat it is minutes. `understudy serve --thinking` turns it back on.

A model's memory need is measured. The first time a model is loaded, Ollama reports how much memory it really takes (the weights, the context cache and the working buffers), and understudy keeps that number for that exact model version. Before the first load it uses a cautious estimate from the file size and the context cache a token keeps, which it reads from the model's own attention layout in Ollama (a Qwen3.5-2B keeps about a third of what the 9B and 4B keep), and `understudy status` says which numbers are measured.

## Install

You need Ollama 0.14 or newer (for its Anthropic Messages endpoint) and Claude Code.

```bash
pip install git+https://github.com/drkostas/understudy
understudy setup-claude          # writes the settings file and links the skill into ~/.claude/skills
understudy serve &               # the proxy; run it under launchd or systemd in real use
```

Then create the ladder in Ollama, one model for each window you want. The settings file tells Claude Code that the window is 131,072 tokens, so it compacts the conversation before it needs more. On a machine with room for a larger window, `understudy settings --context 262144` raises it.

```
# qwen3.5-9b.Modelfile
FROM qwen3.5:9b
PARAMETER temperature 0.6
```

```bash
for k in 64 128 256; do
  (cat qwen3.5-9b.Modelfile; echo "PARAMETER num_ctx $((k * 1024))") > /tmp/m
  ollama create "understudy/qwen3.5-9b:${k}k" -f /tmp/m
done
```

Recommended Ollama settings for one person on one machine are `OLLAMA_MAX_LOADED_MODELS=1`, `OLLAMA_NUM_PARALLEL=1`, `OLLAMA_FLASH_ATTENTION=1` and `OLLAMA_KV_CACHE_TYPE=q8_0`. They keep only one model and one context in memory at a time.

## Use

```bash
understudy claude                # a Claude Code chat on the local model
understudy status                # free memory, the ladder, what the next request would get
```

`understudy claude` runs `claude --settings ~/.config/understudy/claude-settings.json` with any arguments you add. Your normal Claude Code login is kept and is not used by that chat. Other chats are not affected.

## Which MCP servers a local chat has

Against anything but Anthropic, Claude Code does not load MCP tools on demand. It sends the full schema of every tool of every connected server with every request, and with a few dozen servers that was about 1 MB, more than a local model's whole window. So the settings file denies every MCP server except the ones you name, which removes their tools from the request.

```bash
understudy --keep-mcp github,notes setup-claude
```

## What a local chat does not have

- Remote Control. Claude Code turns it off when requests go to anything other than Anthropic.
- Anthropic's hosted tools, such as web search.
- A fast first turn. A new chat sends the whole system prompt and every CLAUDE.md file, often 50,000 to 140,000 tokens, and a laptop reads that at a few hundred tokens per second. Later turns reuse the cache while the model stays loaded and take seconds.
- Thinking. It is off so that the cache works (see above).
- The same quality. The harness is the same, but a model that fits on a laptop makes more mistakes than Claude. Check its work with tests and by reading the diff.

## Options

`understudy serve` takes `--ollama` (default `http://127.0.0.1:11434`), `--prefix` (default `understudy/`), `--headroom-gb` (default 2), `--thinking` and `--budget-gb`, the GPU memory limit. On Apple Silicon Ollama prints that limit when it starts (for example `total="25.0 GiB"` on a 32 GB Mac), and the default is 78% of RAM, which matches it. `--url` sets where the proxy listens (default `http://127.0.0.1:11435`).

## License

MIT

An understudy learns the lead's part and goes on stage when the lead cannot.
