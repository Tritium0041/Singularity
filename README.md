# Singularity

Singularity is a minimal TypeScript agent runtime for experimenting with LLM tool use. It provides a small agent loop, pluggable LLM clients, provider selection, tool registration and execution, streaming events, and provider adapters for OpenAI Responses, OpenAI-compatible Chat Completions, and Anthropic Messages.

## Features

- Turn-based agent loop with conversation history.
- Sequential or parallel tool execution.
- Tool argument validation against simple JSON Schema shapes.
- Streaming text, reasoning summary, and tool-call events.
- OpenAI Responses, OpenAI-compatible Chat Completions, and Anthropic Messages support.
- Built-in basic, file, shell, and web/search tools.
- Tool output truncation for large file, command, and URL results.
- Context budgeting uses provider-reported token usage when available, can dynamically summarize stale history in the request view, and still falls back to one-shot no-tool handoff compaction for oversized histories.

## Requirements

- Node.js 18 or newer.
- npm.
- An API key for the provider used by the demo.

## Install

```sh
npm install
```

## Run Checks

```sh
npm test
npm run build
```

## Run The Demo

```sh
LLM_API_KEY=your_key_here npm run demo
```

The demo opens a small terminal chat UI. Each message keeps the same agent history, and the status line shows current provider token totals plus the latest context estimate. When context compaction runs, the demo prints the pre-compaction decision estimate, summary-call usage, and post-compaction context estimate.

You can pass an initial prompt and continue chatting:

```sh
LLM_API_KEY=your_key_here npm run demo -- "What is (123 + 456) * 789?"
```

Use `--once` for the old single-prompt behavior:

```sh
LLM_API_KEY=your_key_here npm run demo -- --once "What is (123 + 456) * 789?"
```

Interactive commands:

- `/usage`: show current session token usage and latest context estimate.
- `/config`: show or change demo settings without restarting with new environment variables.
- `/compact`: manually summarize and compact the current conversation history.
- `/new`: reset conversation history and usage counters.
- `/clear`: clear the terminal view.
- `/exit`: quit the demo.

The TUI config command stores overrides in local `.agent-demo.json`, which is ignored by git:

```txt
/config show
/config keys
/config set provider openai-chat
/config set model qwen3
/config set baseURL http://localhost:11434/v1
/config set apiKey ollama
/config set toolset all
/config set maxTurns 12
/config set tavilyApiKey tvly-dev-...
/config set dynamicCompression true
/config set dynamicAutoSummarize false
/config set compressionModel gpt-4.1-mini
/config unset compressionModel
/config reset
```

Changing config rebuilds the demo agent and resets the active conversation. Use `/config` before a long run when you want the changed model, toolset, context budget, or compression settings to apply from the start.

Optional environment variables:

- `LLM_PROVIDER`: `openai-responses`, `openai-chat`, or `anthropic`. Defaults to `openai-responses`.
- `LLM_MODEL`: model name used by the demo.
- `LLM_API_KEY`: generic API key used before provider-specific fallbacks.
- `LLM_BASE_URL`: generic API base URL used before provider-specific fallbacks.
- `OPENAI_MODEL` / `OPENAI_BASE_URL` / `OPENAI_API_KEY`: fallback values for `openai-responses`.
- `OPENAI_COMPAT_MODEL` / `OPENAI_COMPAT_BASE_URL` / `OPENAI_COMPAT_API_KEY`: fallback values for `openai-chat`.
- `ANTHROPIC_MODEL` / `ANTHROPIC_BASE_URL` / `ANTHROPIC_API_KEY`: fallback values for `anthropic`.
- `AGENT_REASONING_EFFORT`: reasoning effort passed to supported models.
- `AGENT_COMPRESSION_PROVIDER`: optional provider for context compression calls. Defaults to the main provider.
- `AGENT_COMPRESSION_MODEL`: optional model for context compression calls. Defaults to the main model.
- `AGENT_DYNAMIC_COMPRESSION`: set to `1`, `true`, `on`, or `yes` to enable dynamic request-view compression.
- `AGENT_DYNAMIC_AUTO_SUMMARIZE`: optionally enable the older automatic prefix summary fallback. The default dynamic path is offloaded through `compact_context`.
- `AGENT_DYNAMIC_TRIGGER_TOKENS`: optional token estimate threshold for dynamic compression.
- `AGENT_DYNAMIC_KEEP_RECENT_TOKENS`: optional recent-context budget preserved outside the dynamic summary.
- `AGENT_DYNAMIC_MIN_MESSAGES`: optional minimum selected stale messages before dynamic compression runs.
- `AGENT_TOOLSET`: `basic`, `files`, `shell`, `web`, or `all`. Defaults to `basic`.
- `AGENT_MAX_TURNS`: positive integer overriding the demo run's max turn count. The agent default is `8`.
- `TAVILY_API_KEY`: required by the `web_search` tool unless `tavilyApiKey` is set through `/config`.

OpenAI-compatible local or third-party providers can use the Chat Completions adapter:

```sh
LLM_PROVIDER=openai-chat \
LLM_BASE_URL=http://localhost:11434/v1 \
LLM_API_KEY=ollama \
LLM_MODEL=qwen3 \
AGENT_TOOLSET=files \
AGENT_MAX_TURNS=12 \
npm run demo -- "List this directory and read README.md."
```

Long-chain tool stress test:

```sh
LLM_API_KEY=your_key_here \
TAVILY_API_KEY=your_tavily_key_here \
AGENT_TOOLSET=all \
npm run demo -- "Read README.md, search for TypeScript tool-call design notes, fetch one result, and write a short summary to tmp/provider-tool-notes.md."
```

`execute_command` is intentionally powerful and is not a security sandbox. It only constrains its `workdir` to the configured `rootDir`; commands can still invoke the local shell and access the host according to normal OS permissions. Keep `AGENT_TOOLSET` at `basic` unless you want to expose stronger tools.

## Basic Usage

```ts
import { Agent, createCoreTools, createLlmClientFromEnv } from "./src/index.js";

const llm = createLlmClientFromEnv();

const agent = new Agent({
  llm: llm.llm,
  model: llm.model,
  systemPrompt: "You are a concise assistant. Use tools when useful.",
  tools: createCoreTools({ toolset: "basic" })
});

const result = await agent.run("Calculate (123 + 456) * 789.");
console.log(result.output);
```

Agents automatically build a system prompt from Singularity's default coding-agent instructions, the configured `systemPrompt`, and a compact conversation background (`cwd`, date, timezone, shell, and enabled tool names). Override values through `background`, or set `background: false` for the exact system prompt string only.

Dynamic compression is opt-in and does not mutate `agent.history`. When enabled, the request view includes stable message IDs such as `m0001`, a `compact_context` tool, and a compression nudge once the configured token threshold is crossed. The main model only has to call `compact_context`; that tool starts a side compression worker using `compressionLlm`/`compressionModel` when configured, or the main LLM otherwise. The worker inspects the visible context, chooses stale closed ranges, returns JSON summaries, and future requests replace those raw ranges with reusable `bN` summary blocks. The existing threshold-based handoff compaction remains a fallback for oversized histories.

By default the compressed range is replaced by the summary block. Set `preserveUserMessages: true` only when you explicitly want old user messages copied alongside the summary.

Content wrapped in `<protect>...</protect>` inside a compressed range is copied into the final block summary by local code, even if the model omits it from the submitted summary.

```ts
const agent = new Agent({
  llm: llm.llm,
  model: llm.model,
  tools: createCoreTools({ toolset: "basic" }),
  context: {
    dynamicCompression: {
      enabled: true,
      triggerTokens: 50000,
      keepRecentTokens: 20000
    }
  }
});
```

If you want the older automatic prefix summarizer behavior, set `autoSummarize: true`; those calls can use the main LLM or a separate client/model:

```ts
const compression = createLlmClientFromEnv({
  provider: "openai-chat",
  model: "gpt-4.1-mini"
});

const agent = new Agent({
  llm: llm.llm,
  model: llm.model,
  compressionLlm: compression.llm,
  compressionModel: compression.model,
  tools: createCoreTools({ toolset: "basic" }),
  context: {
    dynamicCompression: {
      enabled: true,
      autoSummarize: true,
      triggerTokens: 50000
    }
  }
});
```

## Project Structure

- `src/agent`: the agent loop and event flow.
- `src/context`: system prompt construction, token estimation, request-view truncation, and history compaction.
- `src/llm`: LLM interfaces, provider registry, env factory, and provider clients.
- `src/tools`: tool registry, executor, validation, built-in tools, and core tool factories.
- `examples`: runnable agent demo.
- `tests`: node test coverage for the agent loop and client behavior.

## License

No license has been added yet.
