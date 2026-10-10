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

The ladder is every Ollama model whose name starts with `understudy/`. For each request the proxy does the following.

1. If one of those models is already loaded, it uses that one. Changing models in the middle of a chat would mean reading the whole conversation again, which takes minutes on a laptop.
2. If none is loaded, it uses the largest one whose memory need, plus a margin of 2 GB, fits in the memory the system can free now, and within the GPU's memory limit.
3. If none fits, it refuses with HTTP 529 and says how much memory was free and how much the smallest model needs. Claude Code shows that message and tries again after a pause.

A model's memory need is measured. The first time a model is loaded, Ollama reports how much memory it really takes (the weights, the context cache and the working buffers), and understudy keeps that number for that exact model version. Before the first load it uses a cautious estimate from the file size, and `understudy status` says which numbers are measured.

## Install

You need Ollama 0.14 or newer (for its Anthropic Messages endpoint) and Claude Code.

```bash
pip install understudy
understudy setup-claude          # writes the settings file and links the skill into ~/.claude/skills
understudy serve &               # the proxy; run it under launchd or systemd in real use
```

Then create the ladder in Ollama. Every model needs `num_ctx 131072`, because the settings file tells Claude Code that the context window is 131,072 tokens and Claude Code compacts the conversation before that.

```
# understudy/qwen3.5-9b.Modelfile
FROM qwen3.5:9b
PARAMETER temperature 0.6
PARAMETER num_ctx 131072
```

```bash
ollama create understudy/qwen3.5-9b -f understudy/qwen3.5-9b.Modelfile
```

Recommended Ollama settings for one person on one machine are `OLLAMA_MAX_LOADED_MODELS=1`, `OLLAMA_NUM_PARALLEL=1`, `OLLAMA_FLASH_ATTENTION=1` and `OLLAMA_KV_CACHE_TYPE=q8_0`. They keep only one model and one context in memory at a time.

## Use

```bash
understudy claude                # a Claude Code chat on the local model
understudy status                # free memory, the ladder, what the next request would get
```

`understudy claude` runs `claude --settings ~/.config/understudy/claude-settings.json` with any arguments you add. Your normal Claude Code login is kept and is not used by that chat. Other chats are not affected.

## What a local chat does not have

- Remote Control. Claude Code turns it off when requests go to anything other than Anthropic.
- Anthropic's hosted tools, such as web search.
- A fast first turn. A new chat sends the whole system prompt and every CLAUDE.md file, and a laptop reads that at a few hundred tokens per second. Later turns are faster while the model stays loaded.
- The same quality. The harness is the same, but a model that fits on a laptop makes more mistakes than Claude. Check its work with tests and by reading the diff.

## Options

`understudy serve` takes `--ollama` (default `http://127.0.0.1:11434`), `--prefix` (default `understudy/`), `--headroom-gb` (default 2) and `--budget-gb`, the GPU memory limit. On Apple Silicon Ollama prints that limit when it starts (for example `total="25.0 GiB"` on a 32 GB Mac), and the default is 78% of RAM, which matches it. `--url` sets where the proxy listens (default `http://127.0.0.1:11435`).

## License

MIT
