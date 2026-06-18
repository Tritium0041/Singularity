import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { Agent } from "../src/agent/agent-loop.js";
import { AnthropicMessagesClient } from "../src/llm/anthropic-messages-client.js";
import { createLlmClientFromEnv } from "../src/llm/env-factory.js";
import { OpenAIChatCompletionsClient } from "../src/llm/openai-chat-completions-client.js";
import type { LlmRequest, LlmStreamEvent, StreamingLlmClient } from "../src/llm/types.js";
import { createFileSystemTools, createShellTool, createWebTools } from "../src/tools/core-tools.js";
import type { AgentTool } from "../src/tools/registry.js";
import type { AssistantMessage } from "../src/types.js";

test("createLlmClientFromEnv uses generic env before provider-specific env", async () => {
  let requestUrl = "";
  let requestBody: Record<string, unknown> | undefined;
  let authorization = "";
  const fetchImpl: typeof fetch = async (url, init) => {
    requestUrl = String(url);
    requestBody = JSON.parse(String(init?.body)) as Record<string, unknown>;
    authorization = new Headers(init?.headers).get("authorization") ?? "";
    return Response.json({ choices: [{ message: { content: "ok" } }] });
  };

  const env = {
    LLM_PROVIDER: "openai-chat",
    LLM_API_KEY: "generic-key",
    LLM_BASE_URL: "https://generic.example/v1",
    LLM_MODEL: "generic-model",
    OPENAI_COMPAT_API_KEY: "provider-key",
    OPENAI_COMPAT_BASE_URL: "https://provider.example/v1",
    OPENAI_COMPAT_MODEL: "provider-model"
  } as NodeJS.ProcessEnv;

  const resolved = createLlmClientFromEnv({ env, fetchImpl });
  await resolved.llm.complete({ model: resolved.model, messages: [{ role: "user", content: "hi" }] });

  assert.equal(resolved.provider, "openai-chat");
  assert.equal(resolved.model, "generic-model");
  assert.equal(requestUrl, "https://generic.example/v1/chat/completions");
  assert.equal(requestBody?.model, "generic-model");
  assert.equal(authorization, "Bearer generic-key");
});

test("provider clients report missing API keys before making requests", async () => {
  const client = new AnthropicMessagesClient({ apiKey: "" });
  await assert.rejects(
    () => client.complete({ model: "test-model", messages: [{ role: "user", content: "hi" }] }),
    /Missing Anthropic API key/
  );
});

test("OpenAI Chat client parses streamed tool-call arguments", async () => {
  const sse = [
    sseEvent({
      choices: [
        {
          delta: {
            tool_calls: [{ index: 0, id: "call_1", function: { name: "calculator", arguments: "{\"expression\"" } }]
          }
        }
      ]
    }),
    sseEvent({
      choices: [
        {
          delta: {
            tool_calls: [{ index: 0, function: { arguments: ":\"2 + 2\"}" } }]
          }
        }
      ]
    }),
    "data: [DONE]\n\n"
  ].join("");
  const client = new OpenAIChatCompletionsClient({
    apiKey: "test-key",
    fetchImpl: async () => streamResponse(sse)
  });

  const events = [];
  for await (const event of client.stream({ model: "test-model", messages: [{ role: "user", content: "hi" }] })) {
    events.push(event);
  }

  assert.equal(events[0]?.type, "tool_call_delta");
  assert.equal(events[1]?.type, "tool_call_delta");
  assert.equal(events[2]?.type, "done");
  if (events[2]?.type !== "done") {
    throw new Error("Expected done event");
  }
  assert.deepEqual(events[2].message.toolCalls?.[0], {
    id: "call_1",
    name: "calculator",
    arguments: { expression: "2 + 2" }
  });
});

test("OpenAI Chat client maps non-streaming usage", async () => {
  const client = new OpenAIChatCompletionsClient({
    apiKey: "test-key",
    fetchImpl: async () =>
      Response.json({
        choices: [{ message: { content: "ok" } }],
        usage: {
          prompt_tokens: 21,
          prompt_tokens_details: { cached_tokens: 4 },
          completion_tokens: 8,
          total_tokens: 29
        }
      })
  });

  const message = await client.complete({ model: "test-model", messages: [{ role: "user", content: "hi" }] });

  assert.equal(message.content, "ok");
  assert.deepEqual(message.usage, {
    inputTokens: 21,
    outputTokens: 8,
    totalTokens: 29,
    cacheReadInputTokens: 4,
    cacheCreationInputTokens: undefined
  });
});

test("OpenAI Chat client requests and maps streaming usage chunk", async () => {
  let requestBody: Record<string, unknown> | undefined;
  const sse = [
    sseEvent({ choices: [{ delta: { content: "ok" } }], usage: null }),
    sseEvent({
      choices: [],
      usage: {
        prompt_tokens: 13,
        prompt_tokens_details: { cached_tokens: 2 },
        completion_tokens: 5,
        total_tokens: 18
      }
    }),
    "data: [DONE]\n\n"
  ].join("");
  const client = new OpenAIChatCompletionsClient({
    apiKey: "test-key",
    fetchImpl: async (_url, init) => {
      requestBody = JSON.parse(String(init?.body)) as Record<string, unknown>;
      return streamResponse(sse);
    }
  });

  const events = [];
  for await (const event of client.stream({ model: "test-model", messages: [{ role: "user", content: "hi" }] })) {
    events.push(event);
  }

  assert.deepEqual(requestBody?.stream_options, { include_usage: true });
  assert.equal(events.at(-1)?.type, "done");
  const done = events.at(-1);
  if (done?.type !== "done") {
    throw new Error("Expected done event");
  }
  assert.equal(done.message.content, "ok");
  assert.deepEqual(done.message.usage, {
    inputTokens: 13,
    outputTokens: 5,
    totalTokens: 18,
    cacheReadInputTokens: 2,
    cacheCreationInputTokens: undefined
  });
});

test("Anthropic client sends system prompt, tools, and tool results in Messages format", async () => {
  let requestBody: Record<string, unknown> | undefined;
  const client = new AnthropicMessagesClient({
    apiKey: "test-key",
    fetchImpl: async (_url, init) => {
      requestBody = JSON.parse(String(init?.body)) as Record<string, unknown>;
      return Response.json({ content: [{ type: "text", text: "ok" }] });
    }
  });

  await client.complete({
    model: "claude-test",
    systemPrompt: "system text",
    tools: [
      {
        name: "calculator",
        description: "Calculate",
        parameters: { type: "object", properties: { expression: { type: "string" } }, required: ["expression"] }
      }
    ],
    messages: [
      { role: "user", content: "calculate" },
      { role: "assistant", content: "", toolCalls: [{ id: "toolu_1", name: "calculator", arguments: { expression: "1 + 1" } }] },
      { role: "tool", toolCallId: "toolu_1", toolName: "calculator", content: "2" }
    ]
  });

  assert.equal(requestBody?.system, "system text");
  const messages = requestBody?.messages as Array<{ role: string; content: unknown }>;
  assert.equal(messages[1]?.role, "assistant");
  assert.deepEqual(messages[1]?.content, [{ type: "tool_use", id: "toolu_1", name: "calculator", input: { expression: "1 + 1" } }]);
  assert.deepEqual(messages[2]?.content, [{ type: "tool_result", tool_use_id: "toolu_1", content: "2" }]);
  const tools = requestBody?.tools as Array<{ name: string; input_schema: Record<string, unknown> }>;
  assert.equal(tools[0]?.name, "calculator");
  assert.equal(tools[0]?.input_schema.type, "object");
});

test("Anthropic client maps non-streaming usage including cache tokens", async () => {
  const client = new AnthropicMessagesClient({
    apiKey: "test-key",
    fetchImpl: async () =>
      Response.json({
        content: [{ type: "text", text: "ok" }],
        usage: {
          input_tokens: 10,
          cache_creation_input_tokens: 3,
          cache_read_input_tokens: 7,
          output_tokens: 4
        }
      })
  });

  const message = await client.complete({ model: "claude-test", messages: [{ role: "user", content: "hi" }] });

  assert.equal(message.content, "ok");
  assert.deepEqual(message.usage, {
    inputTokens: 20,
    outputTokens: 4,
    totalTokens: 24,
    cacheReadInputTokens: 7,
    cacheCreationInputTokens: 3
  });
});

test("Anthropic client parses streamed tool input deltas", async () => {
  const sse = [
    sseEvent({ type: "content_block_start", index: 0, content_block: { type: "tool_use", id: "toolu_1", name: "calculator", input: {} } }),
    sseEvent({ type: "content_block_delta", index: 0, delta: { type: "input_json_delta", partial_json: "{\"expression\"" } }),
    sseEvent({ type: "content_block_delta", index: 0, delta: { type: "input_json_delta", partial_json: ":\"3 + 4\"}" } }),
    sseEvent({ type: "message_stop" })
  ].join("");
  const client = new AnthropicMessagesClient({
    apiKey: "test-key",
    fetchImpl: async () => streamResponse(sse)
  });

  const events = [];
  for await (const event of client.stream({ model: "claude-test", messages: [{ role: "user", content: "hi" }] })) {
    events.push(event);
  }

  assert.equal(events[0]?.type, "tool_call_delta");
  assert.equal(events[1]?.type, "tool_call_delta");
  assert.equal(events[2]?.type, "done");
  if (events[2]?.type !== "done") {
    throw new Error("Expected done event");
  }
  assert.deepEqual(events[2].message.toolCalls?.[0], {
    id: "toolu_1",
    name: "calculator",
    arguments: { expression: "3 + 4" }
  });
});

test("Anthropic client maps streaming message_start and message_delta usage", async () => {
  const sse = [
    sseEvent({
      type: "message_start",
      message: {
        id: "msg_1",
        type: "message",
        role: "assistant",
        content: [],
        usage: {
          input_tokens: 10,
          cache_creation_input_tokens: 3,
          cache_read_input_tokens: 7,
          output_tokens: 1
        }
      }
    }),
    sseEvent({ type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "ok" } }),
    sseEvent({
      type: "message_delta",
      delta: { stop_reason: "end_turn", stop_sequence: null },
      usage: {
        output_tokens: 4
      }
    }),
    sseEvent({ type: "message_stop" })
  ].join("");
  const client = new AnthropicMessagesClient({
    apiKey: "test-key",
    fetchImpl: async () => streamResponse(sse)
  });

  const events = [];
  for await (const event of client.stream({ model: "claude-test", messages: [{ role: "user", content: "hi" }] })) {
    events.push(event);
  }

  assert.equal(events.at(-1)?.type, "done");
  const done = events.at(-1);
  if (done?.type !== "done") {
    throw new Error("Expected done event");
  }
  assert.equal(done.message.content, "ok");
  assert.deepEqual(done.message.usage, {
    inputTokens: 20,
    outputTokens: 4,
    totalTokens: 24,
    cacheReadInputTokens: 7,
    cacheCreationInputTokens: 3
  });
});

test("file tools read with truncation and offset, then write and append", async () => {
  const root = await mkdtemp(join(tmpdir(), "agent-files-"));
  try {
    await writeFile(join(root, "logs.txt"), "one\ntwo\nthree\nfour\nfive", "utf8");
    const tools = createFileSystemTools({ rootDir: root, maxLines: 2, maxBytes: 1024 });
    const readTool = findTool(tools, "read_file");
    const writeTool = findTool(tools, "write_file");
    const appendTool = findTool(tools, "append_file");

    const readResult = await readTool.execute({ path: "logs.txt", offset: 2 }, { toolCallId: "call_1" });
    assert.match(readResult.content, /two\nthree/);
    assert.match(readResult.content, /Use offset=4/);
    assert.equal((readResult.details as { truncation: { truncated: boolean } }).truncation.truncated, true);

    await writeTool.execute({ path: "nested/out.txt", content: "" }, { toolCallId: "call_2" });
    await appendTool.execute({ path: "nested/out.txt", content: "hello" }, { toolCallId: "call_3" });
    assert.equal(await readFile(join(root, "nested", "out.txt"), "utf8"), "hello");
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("list_directory returns structured limited entries", async () => {
  const root = await mkdtemp(join(tmpdir(), "agent-list-"));
  try {
    await writeFile(join(root, "a.txt"), "a", "utf8");
    await writeFile(join(root, "b.txt"), "bb", "utf8");
    const listTool = findTool(createFileSystemTools({ rootDir: root }), "list_directory");

    const result = await listTool.execute({ path: ".", limit: 1 }, { toolCallId: "call_1" });
    const details = result.details as {
      entries: Array<{ name: string; type: string; size: number }>;
      entryLimitReached: boolean;
      nextOffset?: number;
    };

    assert.match(result.content, /^Directory: /);
    assert.match(result.content, /Entries: 1 shown \(1-1\) of 2; offset=0 limit=1/);
    assert.match(result.content, /file\t1\ta\.txt/);
    assert.match(result.content, /Re-run list_directory with offset=1 limit=1/);
    assert.equal(details.entries.length, 1);
    assert.equal(details.entries[0]?.name, "a.txt");
    assert.equal(details.entries[0]?.type, "file");
    assert.equal(details.entryLimitReached, true);
    assert.equal(details.nextOffset, 1);

    const second = await listTool.execute({ path: ".", offset: 1, limit: 1 }, { toolCallId: "call_2" });
    const secondDetails = second.details as { entries: Array<{ name: string }>; entryLimitReached: boolean; nextOffset?: number };
    assert.match(second.content, /file\t2\tb\.txt/);
    assert.equal(secondDetails.entries[0]?.name, "b.txt");
    assert.equal(secondDetails.entryLimitReached, false);
    assert.equal(secondDetails.nextOffset, undefined);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("list_directory large output stays compact and continuable", async () => {
  const root = await mkdtemp(join(tmpdir(), "agent-list-large-"));
  try {
    for (let index = 0; index < 120; index += 1) {
      await writeFile(join(root, `file_${String(index).padStart(3, "0")}_${"long_".repeat(8)}.txt`), "x", "utf8");
    }
    const listTool = findTool(createFileSystemTools({ rootDir: root, maxBytes: 2048 }), "list_directory");

    const result = await listTool.execute({ path: ".", limit: 100 }, { toolCallId: "call_1" });

    assert.doesNotMatch(result.content, /^\{/);
    assert.match(result.content, /Directory: /);
    assert.match(result.content, /Entries: 100 shown \(1-100\) of 120; offset=0 limit=100/);
    assert.match(result.content, /offset=100 limit=100/);
    assert.ok(Buffer.byteLength(result.content, "utf8") <= 4096);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("execute_command captures stderr, nonzero exit, and timeout", async () => {
  const root = await mkdtemp(join(tmpdir(), "agent-shell-"));
  try {
    const shellTool = createShellTool({ rootDir: root, defaultTimeoutMs: 1000 });
    const failed = await shellTool.execute(
      { command: "node -e \"console.error('bad'); process.exit(3)\"" },
      { toolCallId: "call_1" }
    );
    assert.equal(failed.isError, true);
    assert.match(failed.content, /Exit code: 3/);
    assert.match(failed.content, /bad/);

    const timedOut = await shellTool.execute(
      { command: "node -e \"setTimeout(() => {}, 500)\"", timeoutMs: 20 },
      { toolCallId: "call_2" }
    );
    assert.equal(timedOut.isError, true);
    assert.equal((timedOut.details as { timedOut: boolean }).timedOut, true);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("execute_command default budget does not truncate medium source output", async () => {
  const root = await mkdtemp(join(tmpdir(), "agent-shell-medium-"));
  try {
    const content = Array.from({ length: 1200 }, (_, index) => `line-${index} ${"source ".repeat(6)}`).join("\n");
    await writeFile(join(root, "medium.txt"), content, "utf8");
    const shellTool = createShellTool({ rootDir: root });

    const result = await shellTool.execute({ command: "cat medium.txt", workdir: "." }, { toolCallId: "call_1" });

    assert.doesNotMatch(result.content, /Truncated command output/);
    assert.match(result.content, /line-0/);
    assert.match(result.content, /line-1199/);
    assert.equal((result.details as { truncation: { truncated: boolean } }).truncation.truncated, false);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("fetch_url extracts HTML text and web_search reports missing Tavily key", async () => {
  const fetchImpl: typeof fetch = async () =>
    new Response("<html><head><title>Hello</title><script>bad()</script></head><body><main>Useful text</main></body></html>", {
      headers: { "content-type": "text/html" }
    });
  const tools = createWebTools({ apiKey: "", fetchImpl, maxBytes: 1024 });
  const searchTool = findTool(tools, "web_search");
  const fetchTool = findTool(tools, "fetch_url");

  const missingKey = await searchTool.execute({ query: "typescript" }, { toolCallId: "call_1" });
  assert.equal(missingKey.isError, true);
  assert.match(missingKey.content, /Missing TAVILY_API_KEY/);

  const fetched = await fetchTool.execute({ url: "https://example.com" }, { toolCallId: "call_2" });
  assert.match(fetched.content, /Title: Hello/);
  assert.match(fetched.content, /Useful text/);
  assert.doesNotMatch(fetched.content, /bad\(\)/);
});

test("web_search returns compact bounded text with full details", async () => {
  const longSnippet = `Lead ${"snippet ".repeat(200)}`;
  const fetchImpl: typeof fetch = async () =>
    Response.json({
      results: [
        { title: "First result", url: "https://example.com/one", content: longSnippet, score: 0.9 },
        { title: "Second result", url: "https://example.com/two", content: "short", score: 0.8 }
      ]
    });
  const searchTool = findTool(createWebTools({ apiKey: "test-key", fetchImpl }), "web_search");

  const result = await searchTool.execute({ query: "typescript", maxResults: 2 }, { toolCallId: "call_1" });
  const details = result.details as { results: Array<{ snippet: string }> };

  assert.match(result.content, /^Search: typescript/);
  assert.match(result.content, /Results: 2/);
  assert.match(result.content, /1\. First result/);
  assert.match(result.content, /URL: https:\/\/example\.com\/one/);
  assert.match(result.content, /\[snippet truncated\]/);
  assert.ok(result.content.length < 2000);
  assert.equal(details.results[0]?.snippet, longSnippet);
});

test("truncation keeps useful content for single long lines", async () => {
  const root = await mkdtemp(join(tmpdir(), "agent-long-line-"));
  try {
    await writeFile(join(root, "long.txt"), "A".repeat(4096), "utf8");
    const readTool = findTool(createFileSystemTools({ rootDir: root, maxBytes: 128 }), "read_file");

    const result = await readTool.execute({ path: "long.txt" }, { toolCallId: "call_1" });

    assert.match(result.content, /^A+/);
    assert.match(result.content, /Use offset=2 to continue/);
    assert.ok((result.details as { truncation: { outputBytes: number } }).truncation.outputBytes > 0);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("fetch_url preserves paragraph text before truncation marker", async () => {
  const html = `<html><head><title>Long page</title></head><body>${Array.from(
    { length: 40 },
    (_, index) => `<p>paragraph-${index} ${"body ".repeat(20)}</p>`
  ).join("")}</body></html>`;
  const fetchImpl: typeof fetch = async () => new Response(html, { headers: { "content-type": "text/html" } });
  const fetchTool = findTool(createWebTools({ apiKey: "test-key", fetchImpl, maxBytes: 512 }), "fetch_url");

  const result = await fetchTool.execute({ url: "https://example.com/long" }, { toolCallId: "call_1" });

  assert.match(result.content, /Title: Long page/);
  assert.match(result.content, /paragraph-0/);
  assert.match(result.content, /paragraph-1/);
  assert.match(result.content, /Truncated fetched content/);
});

test("agent can run a long file-search-fetch-write tool chain with fake streaming LLM", async () => {
  const root = await mkdtemp(join(tmpdir(), "agent-integration-"));
  try {
    await writeFile(join(root, "logs.txt"), "Error: failed to load provider\nStack line", "utf8");
    const fetchImpl: typeof fetch = async (url, init) => {
      if (String(url).includes("tavily")) {
        return Response.json({
          results: [{ title: "Provider fix", url: "https://example.com/fix", content: "Check provider env vars." }]
        });
      }
      return new Response("<html><body>Set LLM_PROVIDER and LLM_API_KEY.</body></html>", {
        headers: { "content-type": "text/html" }
      });
    };
    const tools = [
      ...createFileSystemTools({ rootDir: root }),
      ...createWebTools({ apiKey: "test-key", fetchImpl })
    ];
    const llm = new StreamingSequenceLlm([
      [{ type: "done", message: toolMessage("call_1", "read_file", { path: "logs.txt" }) }],
      [{ type: "done", message: toolMessage("call_2", "web_search", { query: "failed to load provider", maxResults: 1 }) }],
      [{ type: "done", message: toolMessage("call_3", "fetch_url", { url: "https://example.com/fix" }) }],
      [{ type: "done", message: toolMessage("call_4", "write_file", { path: "report.md", content: "Use LLM_PROVIDER and LLM_API_KEY." }) }],
      [{ type: "done", message: { role: "assistant", content: "done" } }]
    ]);

    const agent = new Agent({
      llm,
      model: "fake-model",
      tools,
      maxTurns: 8
    });
    const result = await agent.run("Investigate provider failure");

    assert.equal(result.output, "done");
    assert.equal(await readFile(join(root, "report.md"), "utf8"), "Use LLM_PROVIDER and LLM_API_KEY.");
    assert.equal(result.messages.filter((message) => message.role === "tool").length, 4);
    assert.equal(llm.requests.length, 5);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

class StreamingSequenceLlm implements StreamingLlmClient {
  public readonly requests: LlmRequest[] = [];

  constructor(private readonly responses: LlmStreamEvent[][]) {}

  async complete(): Promise<AssistantMessage> {
    throw new Error("complete should not be called");
  }

  async *stream(request: LlmRequest): AsyncIterable<LlmStreamEvent> {
    this.requests.push(request);
    const response = this.responses.shift();
    if (!response) {
      throw new Error("No fake stream response left");
    }
    for (const event of response) {
      yield event;
    }
  }
}

function toolMessage(id: string, name: string, args: Record<string, unknown>): AssistantMessage {
  return {
    role: "assistant",
    content: "",
    toolCalls: [{ id, name, arguments: args }]
  };
}

function findTool(tools: AgentTool[], name: string): AgentTool {
  const tool = tools.find((candidate) => candidate.name === name);
  if (!tool) {
    throw new Error(`Missing tool: ${name}`);
  }
  return tool;
}

function sseEvent(event: unknown): string {
  return `data: ${JSON.stringify(event)}\n\n`;
}

function streamResponse(sse: string): Response {
  return new Response(
    new ReadableStream({
      start(controller) {
        controller.enqueue(new TextEncoder().encode(sse));
        controller.close();
      }
    })
  );
}
