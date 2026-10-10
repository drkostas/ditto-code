---
name: understudy
description: Use when work should run on a local model instead of Anthropic's, to save tokens, or when asked to start, check or fix a local chat. understudy runs Claude Code itself on Ollama, choosing the largest local model that fits in the memory free right now, so a local chat has the same tools, instructions and workflow as any other chat.
---

# understudy: Claude Code on a local model

A local chat is an ordinary Claude Code chat whose requests go to a local model. Same harness,
same tools, same CLAUDE.md files, same skills and MCP servers. Only the model differs.

```
claude --settings ~/.config/understudy/claude-settings.json ...    # what a local chat runs
understudy claude [claude args]                                    # the same, by hand
understudy status                                                  # memory, ladder, next choice
```

The settings file points every request (the main loop, background work such as titles, and
subagents) at the understudy proxy on `127.0.0.1:11435` under the model name `local`. The proxy
picks a model and passes the request to Ollama.

## Which model answers

Every Ollama model named `understudy/<x>` is a rung of the ladder, one per context window
(`understudy/qwen3.5-9b:64k`, `:128k`, `:256k`: same weights, different context cache). For each
request the proxy estimates the window it needs (request bytes / 3 plus the answer), then:

1. A loaded rung whose window holds it keeps it. Swapping models mid-chat would mean reading the
   whole conversation again, which takes minutes.
2. Otherwise models largest first, each at its smallest window that holds the conversation: the
   first whose measured memory need, plus 2 GB of margin, fits in what the system can free without
   swapping, and within the GPU budget.
3. Otherwise the request is refused with HTTP 529 and the numbers. Claude Code shows the message
   and retries with backoff, so closing a few chats lets it continue.

`understudy status` prints free memory, every rung's need (and whether it was measured or is
still an estimate), what is loaded, and the last choices with their reasons.

## When to send work to a local chat

Good fits, where saving tokens matters and the result is easy to check:
- routine edits with a clear spec, renames, boilerplate, small scripts, test scaffolding
- summarising logs, files or diffs, drafting commit messages and changelogs
- mechanical refactors that a test suite will verify

Keep on Anthropic: design decisions, anything security-sensitive, anything that needs judgement
about the owner's intent, and long multi-repo work. Always verify a local chat's output (run the
tests, read the diff) before building on it. It is a weaker model, whatever the harness looks like.

## What a local chat does not have

- **Remote Control.** Claude Code turns it off whenever requests go anywhere but Anthropic. Reach
  a local chat through its terminal (or tmux) instead.
- **Anthropic's server tools**, such as hosted web search.
- **Most MCP servers.** Only the servers named with `--keep-mcp` stay; the rest are denied in the
  settings file, because without Anthropic's on-demand tool loading every server's schemas are
  sent with every request (about 1 MB measured). Built-in tools are all there.
- **Speed on the first turn.** A fresh chat sends the whole system prompt and every CLAUDE.md, and
  a laptop reads that at a few hundred tokens a second. Later turns reuse the cache while the model
  stays loaded (Ollama unloads it after 10 idle minutes).
- **Thinking.** The proxy turns it off: Qwen's templates drop earlier reasoning from the history,
  so with thinking on every turn re-reads the whole conversation instead of reusing the cache.
- **A context window as large as Claude's.** The largest rungs run at 262,144 tokens and the
  settings tell Claude Code so, so it compacts before the window fills. A chat in a folder with
  long CLAUDE.md files starts at 100,000 tokens or more before the first message.

## When something is wrong

- `529 ... not enough free memory`: the numbers in the message are the answer. Free memory, or
  wait for Ollama to release a model.
- `502 ... Ollama did not answer`: Ollama is down. `curl -s 127.0.0.1:11434/api/version`.
- A rung's need looks wrong: it is re-measured each time the rung is loaded, keeping the largest
  size seen for that model digest, in `~/.local/state/understudy/measured.json`.
- Adding a rung: create an Ollama model named `understudy/<x>:<n>k` with `num_ctx <n*1024>`.
  Nothing else needs to change, because the ladder is read from Ollama on every request.
