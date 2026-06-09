# Singularity

Singularity is a minimal TypeScript agent runtime for experimenting with LLM tool use. It provides a small agent loop, pluggable LLM clients, tool registration and execution, streaming events, and an OpenAI Responses API client.

## Features

- Turn-based agent loop with conversation history.
- Sequential or parallel tool execution.
- Tool argument validation against simple JSON Schema shapes.
- Streaming text, reasoning summary, and tool-call events.
- OpenAI Responses API support with configurable model, base URL, and reasoning options.
- Built-in example tools for calculator and deterministic mock weather.

## Requirements

- Node.js 18 or newer.
- npm.
- `OPENAI_API_KEY` for the OpenAI-powered demo.

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
OPENAI_API_KEY=your_key_here npm run demo
```

You can pass a custom prompt:

```sh
OPENAI_API_KEY=your_key_here npm run demo -- "What is (123 + 456) * 789?"
```

Optional environment variables:

- `OPENAI_MODEL`: model name used by the demo. Defaults to `gpt-4.1-mini`.
- `OPENAI_BASE_URL`: alternate OpenAI-compatible API base URL.
- `AGENT_REASONING_EFFORT`: reasoning effort passed to supported models.

## Basic Usage

```ts
import { Agent, OpenAIResponsesClient, calculatorTool } from "./src/index.js";

const agent = new Agent({
  llm: new OpenAIResponsesClient(),
  model: "gpt-4.1-mini",
  systemPrompt: "You are a concise assistant. Use tools when useful.",
  tools: [calculatorTool]
});

const result = await agent.run("Calculate (123 + 456) * 789.");
console.log(result.output);
```

## Project Structure

- `src/agent`: the agent loop and event flow.
- `src/llm`: LLM interfaces, provider registry, and OpenAI Responses client.
- `src/tools`: tool registry, executor, validation, and built-in tools.
- `examples`: runnable agent demo.
- `tests`: node test coverage for the agent loop and client behavior.

## License

No license has been added yet.
