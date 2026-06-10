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
- Context budgeting uses provider-reported token usage when available, then compacts long histories with a one-shot no-tool handoff summary.

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
- `/compact`: manually summarize and compact the current conversation history.
- `/new`: reset conversation history and usage counters.
- `/clear`: clear the terminal view.
- `/exit`: quit the demo.

Optional environment variables:

- `LLM_PROVIDER`: `openai-responses`, `openai-chat`, or `anthropic`. Defaults to `openai-responses`.
- `LLM_MODEL`: model name used by the demo.
- `LLM_API_KEY`: generic API key used before provider-specific fallbacks.
- `LLM_BASE_URL`: generic API base URL used before provider-specific fallbacks.
- `OPENAI_MODEL` / `OPENAI_BASE_URL` / `OPENAI_API_KEY`: fallback values for `openai-responses`.
- `OPENAI_COMPAT_MODEL` / `OPENAI_COMPAT_BASE_URL` / `OPENAI_COMPAT_API_KEY`: fallback values for `openai-chat`.
- `ANTHROPIC_MODEL` / `ANTHROPIC_BASE_URL` / `ANTHROPIC_API_KEY`: fallback values for `anthropic`.
- `AGENT_REASONING_EFFORT`: reasoning effort passed to supported models.
- `AGENT_TOOLSET`: `basic`, `files`, `shell`, `web`, or `all`. Defaults to `basic`.
- `AGENT_MAX_TURNS`: positive integer overriding the demo run's max turn count. The agent default is `8`.
- `TAVILY_API_KEY`: required by the `web_search` tool.

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

## Project Structure

- `src/agent`: the agent loop and event flow.
- `src/context`: system prompt construction, token estimation, request-view truncation, and history compaction.
- `src/llm`: LLM interfaces, provider registry, env factory, and provider clients.
- `src/tools`: tool registry, executor, validation, built-in tools, and core tool factories.
- `examples`: runnable agent demo.
- `tests`: node test coverage for the agent loop and client behavior.

## License

No license has been added yet.
