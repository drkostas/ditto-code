# Changelog

## 0.1.0

- The proxy, an Anthropic Messages endpoint in front of Ollama that chooses the largest model named `understudy/*` that fits in the memory free right now, keeps a loaded model for the rest of the chat, measures each model's real memory need, and refuses with HTTP 529 and the numbers when nothing fits
- `understudy claude`, Claude Code started with a settings file that sends every request to the proxy
- `understudy status` and `understudy setup-claude`, and the skill
