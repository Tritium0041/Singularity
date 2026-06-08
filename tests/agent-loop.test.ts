import assert from "node:assert/strict";
import test from "node:test";
import { Agent } from "../src/agent/agent-loop.js";
import type { LlmClient, LlmRequest, LlmStreamEvent, StreamingLlmClient } from "../src/llm/types.js";
import { OpenAIResponsesClient } from "../src/llm/openai-responses-client.js";
import type { AgentEvent, AssistantMessage } from "../src/types.js";
import { calculatorTool } from "../src/tools/builtins.js";
import type { AgentTool } from "../src/tools/registry.js";

class SequenceLlm implements LlmClient {
  public readonly requests: LlmRequest[] = [];

  constructor(private readonly responses: AssistantMessage[]) {}

  async complete(request: LlmRequest): Promise<AssistantMessage> {
    this.requests.push(request);
    const response = this.responses.shift();
    if (!response) {
      throw new Error("No fake response left");
    }
    return response;
  }
}

class StreamingSequenceLlm implements StreamingLlmClient {
  public readonly requests: LlmRequest[] = [];
  public completeCalled = false;

  constructor(private readonly responses: LlmStreamEvent[][]) {}

  async complete(): Promise<AssistantMessage> {
    this.completeCalled = true;
    throw new Error("complete should not be called when stream is available");
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

test("agent executes tool calls and feeds results back to the LLM", async () => {
  const llm = new SequenceLlm([
    {
      role: "assistant",
      content: "",
      toolCalls: [{ id: "call_1", name: "calculator", arguments: { expression: "(123 + 456) * 789" } }]
    },
    {
      role: "assistant",
      content: "The result is 456831."
    }
  ]);

  const agent = new Agent({
    llm,
    model: "fake-model",
    tools: [calculatorTool]
  });

  const result = await agent.run("Calculate (123 + 456) * 789");

  assert.equal(result.output, "The result is 456831.");
  assert.equal(result.stoppedBy, "final");
  assert.equal(llm.requests.length, 2);
  assert.equal(llm.requests[1]?.messages.at(-1)?.role, "tool");
  assert.equal(llm.requests[1]?.messages.at(-1)?.content, "456831");
});

test("tool failures are returned as model-visible tool results", async () => {
  const llm = new SequenceLlm([
    {
      role: "assistant",
      content: "",
      toolCalls: [{ id: "call_1", name: "missing_tool", arguments: {} }]
    },
    {
      role: "assistant",
      content: "I could not use that tool."
    }
  ]);

  const agent = new Agent({
    llm,
    model: "fake-model",
    tools: [calculatorTool]
  });

  const result = await agent.run("Use a missing tool");

  assert.equal(result.output, "I could not use that tool.");
  const toolMessage = llm.requests[1]?.messages.at(-1);
  assert.equal(toolMessage?.role, "tool");
  if (toolMessage?.role !== "tool") {
    throw new Error("Expected the last message to be a tool result");
  }
  assert.equal(toolMessage.isError, true);
  assert.match(toolMessage.content, /Tool not found/);
});

test("agent stops at maxTurns when the model keeps calling tools", async () => {
  const llm = new SequenceLlm([
    {
      role: "assistant",
      content: "",
      toolCalls: [{ id: "call_1", name: "calculator", arguments: { expression: "1 + 1" } }]
    }
  ]);

  const agent = new Agent({
    llm,
    model: "fake-model",
    tools: [calculatorTool],
    maxTurns: 1
  });

  const result = await agent.run("Loop once");

  assert.equal(result.stoppedBy, "max_turns");
  assert.equal(result.turns, 1);
});

test("parallel tool execution preserves source order in history", async () => {
  const firstTool: AgentTool = {
    name: "first",
    description: "First tool",
    parameters: { type: "object", properties: {}, additionalProperties: false },
    executionMode: "parallel",
    async execute() {
      await new Promise((resolve) => setTimeout(resolve, 20));
      return { content: "first-result" };
    }
  };
  const secondTool: AgentTool = {
    name: "second",
    description: "Second tool",
    parameters: { type: "object", properties: {}, additionalProperties: false },
    executionMode: "parallel",
    execute() {
      return { content: "second-result" };
    }
  };
  const llm = new SequenceLlm([
    {
      role: "assistant",
      content: "",
      toolCalls: [
        { id: "call_1", name: "first", arguments: {} },
        { id: "call_2", name: "second", arguments: {} }
      ]
    },
    {
      role: "assistant",
      content: "done"
    }
  ]);

  const agent = new Agent({
    llm,
    model: "fake-model",
    tools: [firstTool, secondTool],
    toolExecution: "parallel"
  });

  await agent.run("Run tools");

  const toolMessages = llm.requests[1]?.messages.filter((message) => message.role === "tool");
  assert.equal(toolMessages?.[0]?.content, "first-result");
  assert.equal(toolMessages?.[1]?.content, "second-result");
});

test("invalid tool arguments are returned as tool errors", async () => {
  const llm = new SequenceLlm([
    {
      role: "assistant",
      content: "",
      toolCalls: [{ id: "call_1", name: "calculator", arguments: { expression: 123 } }]
    },
    {
      role: "assistant",
      content: "bad args"
    }
  ]);

  const agent = new Agent({
    llm,
    model: "fake-model",
    tools: [calculatorTool]
  });

  await agent.run("Use bad args");
  const toolMessage = llm.requests[1]?.messages.at(-1);
  assert.equal(toolMessage?.role, "tool");
  if (toolMessage?.role !== "tool") {
    throw new Error("Expected the last message to be a tool result");
  }
  assert.equal(toolMessage.isError, true);
  assert.match(toolMessage.content, /must be a string/);
});

test("agent forwards reasoning config and allows run-level override", async () => {
  const llm = new SequenceLlm([
    {
      role: "assistant",
      content: "first"
    },
    {
      role: "assistant",
      content: "second"
    }
  ]);
  const agent = new Agent({
    llm,
    model: "fake-model",
    reasoning: { effort: "high", summary: "concise" }
  });

  await agent.run("use default reasoning");
  await agent.run("disable reasoning", { reasoning: false });

  assert.deepEqual(llm.requests[0]?.reasoning, { effort: "high", summary: "concise" });
  assert.equal(llm.requests[1]?.reasoning, false);
});

test("agent consumes streaming LLM text deltas", async () => {
  const llm = new StreamingSequenceLlm([
    [
      { type: "text_delta", delta: "hel" },
      { type: "text_delta", delta: "lo" },
      { type: "done", message: { role: "assistant", content: "hello" } }
    ]
  ]);
  const deltas: string[] = [];
  const snapshots: string[] = [];
  const agent = new Agent({
    llm,
    model: "fake-model",
    onEvent(event) {
      if (event.type === "assistant_delta") {
        deltas.push(event.delta);
        snapshots.push(event.content);
      }
    }
  });

  const result = await agent.run("say hi");

  assert.equal(llm.completeCalled, false);
  assert.deepEqual(deltas, ["hel", "lo"]);
  assert.deepEqual(snapshots, ["hel", "hello"]);
  assert.equal(result.output, "hello");
});

test("agent emits streaming thinking deltas", async () => {
  const llm = new StreamingSequenceLlm([
    [
      { type: "thinking_delta", delta: "think" },
      { type: "thinking_delta", delta: "ing" },
      { type: "text_delta", delta: "answer" },
      { type: "done", message: { role: "assistant", content: "answer", reasoning: { summary: "thinking" } } }
    ]
  ]);
  const thinkingSnapshots: string[] = [];
  const agent = new Agent({
    llm,
    model: "fake-model",
    onEvent(event) {
      if (event.type === "thinking_delta") {
        thinkingSnapshots.push(event.content);
      }
    }
  });

  const result = await agent.run("think");

  assert.deepEqual(thinkingSnapshots, ["think", "thinking"]);
  assert.equal(result.output, "answer");
  assert.equal(result.messages.at(-1)?.role, "assistant");
});

test("agent exposes runEvents as an async iterable event stream", async () => {
  const llm = new StreamingSequenceLlm([
    [
      { type: "text_delta", delta: "stre" },
      { type: "text_delta", delta: "am" },
      { type: "done", message: { role: "assistant", content: "stream" } }
    ]
  ]);
  const events: AgentEvent[] = [];
  const agent = new Agent({
    llm,
    model: "fake-model"
  });

  for await (const event of agent.runEvents("say stream")) {
    events.push(event);
  }

  const deltas = events.filter((event) => event.type === "assistant_delta").map((event) => event.delta);
  const agentEnd = events.find((event) => event.type === "agent_end");

  assert.deepEqual(deltas, ["stre", "am"]);
  assert.equal(agentEnd?.type, "agent_end");
  if (agentEnd?.type !== "agent_end") {
    throw new Error("Expected agent_end event");
  }
  assert.equal(agentEnd.result.output, "stream");
});

test("agent streams tool-call arguments and continues the loop", async () => {
  const llm = new StreamingSequenceLlm([
    [
      {
        type: "tool_call_delta",
        toolCallId: "call_1",
        toolName: "calculator",
        delta: "{\"expression\":\"1 + 2\"}",
        argumentsText: "{\"expression\":\"1 + 2\"}"
      },
      {
        type: "done",
        message: {
          role: "assistant",
          content: "",
          toolCalls: [{ id: "call_1", name: "calculator", arguments: { expression: "1 + 2" } }]
        }
      }
    ],
    [{ type: "done", message: { role: "assistant", content: "3" } }]
  ]);
  const toolArgumentSnapshots: string[] = [];
  const agent = new Agent({
    llm,
    model: "fake-model",
    tools: [calculatorTool],
    onEvent(event) {
      if (event.type === "tool_call_delta") {
        toolArgumentSnapshots.push(event.argumentsText);
      }
    }
  });

  const result = await agent.run("calculate");

  assert.deepEqual(toolArgumentSnapshots, ["{\"expression\":\"1 + 2\"}"]);
  assert.equal(result.output, "3");
  assert.equal(llm.requests.length, 2);
  assert.equal(llm.requests[1]?.messages.at(-1)?.role, "tool");
  assert.equal(llm.requests[1]?.messages.at(-1)?.content, "3");
});

test("OpenAI Responses client parses SSE text and tool-call deltas", async () => {
  const sse = [
    sseEvent({ type: "response.output_text.delta", delta: "hi" }),
    sseEvent({
      type: "response.output_item.added",
      item: { type: "function_call", id: "fc_1", call_id: "call_1", name: "calculator", arguments: "" }
    }),
    sseEvent({ type: "response.function_call_arguments.delta", call_id: "call_1", delta: "{\"expression\"" }),
    sseEvent({ type: "response.function_call_arguments.delta", call_id: "call_1", delta: ":\"2 + 2\"}" }),
    sseEvent({ type: "response.completed", response: {} })
  ].join("");

  const client = new OpenAIResponsesClient({
    apiKey: "test-key",
    fetchImpl: async () =>
      new Response(new ReadableStream({
        start(controller) {
          controller.enqueue(new TextEncoder().encode(sse));
          controller.close();
        }
      }))
  });

  const events = [];
  for await (const event of client.stream({ model: "fake-model", messages: [{ role: "user", content: "hi" }] })) {
    events.push(event);
  }

  assert.deepEqual(events[0], { type: "text_delta", delta: "hi" });
  assert.equal(events[1]?.type, "tool_call_delta");
  assert.equal(events[2]?.type, "tool_call_delta");
  assert.equal(events[3]?.type, "done");
  if (events[3]?.type !== "done") {
    throw new Error("Expected final done event");
  }
  assert.equal(events[3].message.content, "hi");
  assert.deepEqual(events[3].message.toolCalls?.[0], {
    id: "call_1",
    name: "calculator",
    arguments: { expression: "2 + 2" }
  });
});

test("OpenAI Responses client sends reasoning options in request body", async () => {
  let requestBody: Record<string, unknown> | undefined;
  const client = new OpenAIResponsesClient({
    apiKey: "test-key",
    defaultReasoning: { effort: "high", summary: "concise" },
    fetchImpl: async (_url, init) => {
      requestBody = JSON.parse(String(init?.body)) as Record<string, unknown>;
      return Response.json({
        output: [{ type: "message", role: "assistant", content: [{ type: "output_text", text: "ok" }] }]
      });
    }
  });

  await client.complete({ model: "fake-model", messages: [{ role: "user", content: "hi" }] });

  assert.deepEqual(requestBody?.reasoning, { effort: "high", summary: "concise" });
  assert.deepEqual(requestBody?.include, ["reasoning.encrypted_content"]);
});

test("OpenAI Responses request reasoning=false disables default reasoning", async () => {
  let requestBody: Record<string, unknown> | undefined;
  const client = new OpenAIResponsesClient({
    apiKey: "test-key",
    defaultReasoning: { effort: "high" },
    fetchImpl: async (_url, init) => {
      requestBody = JSON.parse(String(init?.body)) as Record<string, unknown>;
      return Response.json({
        output: [{ type: "message", role: "assistant", content: [{ type: "output_text", text: "ok" }] }]
      });
    }
  });

  await client.complete({ model: "fake-model", messages: [{ role: "user", content: "hi" }], reasoning: false });

  assert.equal("reasoning" in (requestBody ?? {}), false);
  assert.equal("include" in (requestBody ?? {}), false);
});

test("OpenAI Responses client parses streaming thinking deltas", async () => {
  const sse = [
    sseEvent({ type: "response.reasoning_summary_text.delta", delta: "plan" }),
    sseEvent({ type: "response.output_text.delta", delta: "done" }),
    sseEvent({ type: "response.completed", response: {} })
  ].join("");

  const client = new OpenAIResponsesClient({
    apiKey: "test-key",
    fetchImpl: async () =>
      new Response(new ReadableStream({
        start(controller) {
          controller.enqueue(new TextEncoder().encode(sse));
          controller.close();
        }
      }))
  });

  const events = [];
  for await (const event of client.stream({ model: "fake-model", messages: [{ role: "user", content: "hi" }] })) {
    events.push(event);
  }

  assert.deepEqual(events[0], { type: "thinking_delta", delta: "plan" });
  assert.deepEqual(events[1], { type: "text_delta", delta: "done" });
  assert.equal(events[2]?.type, "done");
  if (events[2]?.type !== "done") {
    throw new Error("Expected final done event");
  }
  assert.equal(events[2].message.reasoning?.summary, "plan");
});

function sseEvent(event: unknown): string {
  return `data: ${JSON.stringify(event)}\n\n`;
}
