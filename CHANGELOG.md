# Changelog

## 0.4.1

- Claude Code is told a 131,072-token window by default, so it compacts before the conversation needs a rung that does not fit beside other work on a 32 GB machine. A chat told 262,144 never compacted and stalled once it passed 128k. `understudy settings --context` raises it

## 0.4.0

- A streamed request is answered at once and kept alive with ping events every 15 seconds until Ollama starts its answer. Reading a long prompt takes minutes and Ollama sends nothing meanwhile, and Claude Code gave up at about five
- A ping that cannot be delivered means the client left, and the request is cancelled in Ollama, so a retry does not queue behind it
- The settings turn off Claude Code's idle timeout and set its first-byte and byte watchdogs to the 30-minute maximum

## 0.3.0

- A local chat keeps only the MCP servers it is told to keep (`--keep-mcp`); every other server, plugin server, claude.ai connector and the Chrome extension is denied in its settings file, which removes their tools from the request. Against a non-Anthropic backend Claude Code sends every connected server's full schemas each time, measured at about 1 MB
- The window a request needs is estimated from bytes per token learned from what Ollama counted (the most cautious of the last 20), with 5% to spare, instead of a fixed three bytes a token

## 0.2.0

- A model can be several rungs, one per context window (`model:64k`, `:128k`, `:256k`). The proxy estimates the window a request needs and picks the largest model that fits in memory at its smallest window that holds the conversation, and a loaded rung gives way when its window is too small
- Thinking is turned off by default, because with it on each turn re-reads the whole conversation instead of reusing the cache (`--thinking` turns it back on)
- The settings declare a 262,144-token window and a 16,000-token answer

## 0.1.0

- The proxy, an Anthropic Messages endpoint in front of Ollama that chooses the largest model named `understudy/*` that fits in the memory free right now, keeps a loaded model for the rest of the chat, measures each model's real memory need, and refuses with HTTP 529 and the numbers when nothing fits
- `understudy claude`, Claude Code started with a settings file that sends every request to the proxy
- `understudy status` and `understudy setup-claude`, and the skill
