import assert from "node:assert/strict";
import test from "node:test";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Agent } from "../src/agent/agent-loop.js";
import type { LlmClient, LlmRequest, LlmStreamEvent, StreamingLlmClient } from "../src/llm/types.js";
import { OpenAIResponsesClient } from "../src/llm/openai-responses-client.js";
import { MarkdownMemoryStore } from "../src/memory/index.js";
import { APPROVE_PLAN_TOOL_NAME, CREATE_PLAN_TOOL_NAME, UPDATE_PLAN_STEP_TOOL_NAME, Planner } from "../src/planning/index.js";
import type { AgentEvent, AssistantMessage } from "../src/types.js";
import { calculatorTool } from "../src/tools/builtins.js";
import { createFileSystemTools, createShellTool } from "../src/tools/core-tools.js";
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

test("agent can resume from configured history without sharing mutable message references", async () => {
  const history: AssistantMessage[] = [
    {
      role: "assistant",
      content: "Earlier answer."
    }
  ];
  const llm = new SequenceLlm([
    {
      role: "assistant",
      content: "continued"
    }
  ]);
  const agent = new Agent({
    llm,
    model: "fake-model",
    history
  });

  history[0]!.content = "mutated outside";
  await agent.run("Continue");

  assert.equal(llm.requests[0]?.messages[0]?.role, "assistant");
  assert.equal(llm.requests[0]?.messages[0]?.content, "Earlier answer.");
  assert.equal(llm.requests[0]?.messages[1]?.role, "user");
  assert.equal(llm.requests[0]?.messages[1]?.content, "Continue");
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

test("planning without a plan exposes only read tools plus create/read plan", async () => {
  const readTool: AgentTool = {
    name: "custom_read",
    description: "Read only",
    access: "read",
    parameters: { type: "object", properties: {}, additionalProperties: false },
    execute() {
      return { content: "read" };
    }
  };
  const unknownTool: AgentTool = {
    name: "custom_unknown",
    description: "Unannotated custom tool",
    parameters: { type: "object", properties: {}, additionalProperties: false },
    execute() {
      return { content: "unknown" };
    }
  };
  const llm = new SequenceLlm([{ role: "assistant", content: "ready" }]);
  const agent = new Agent({
    llm,
    model: "fake-model",
    tools: [readTool, unknownTool, ...createFileSystemTools(), createShellTool()],
    memory: { store: new MarkdownMemoryStore() },
    planning: {}
  });

  await agent.run("plan this");
  const toolNames = new Set(llm.requests[0]?.tools?.map((tool) => tool.name));

  assert.equal(toolNames.has("custom_read"), true);
  assert.equal(toolNames.has("read_file"), true);
  assert.equal(toolNames.has("list_directory"), true);
  assert.equal(toolNames.has("create_plan"), true);
  assert.equal(toolNames.has("read_plan"), true);
  assert.equal(toolNames.has("custom_unknown"), false);
  assert.equal(toolNames.has("write_file"), false);
  assert.equal(toolNames.has("append_file"), false);
  assert.equal(toolNames.has("execute_command"), false);
  assert.equal(toolNames.has("write_note"), false);
  assert.equal(toolNames.has("store_memory"), false);
  assert.equal(toolNames.has("update_plan_step"), false);
  assert.match(llm.requests[0]?.systemPrompt ?? "", /Planning tools are available/);
  assert.doesNotMatch(llm.requests[0]?.systemPrompt ?? "", /<current_plan/);
});

test("forcePlan run option injects an explicit plan-mode directive", async () => {
  const llm = new SequenceLlm([
    {
      role: "assistant",
      content: "",
      toolCalls: [
        {
          id: "call_plan",
          name: CREATE_PLAN_TOOL_NAME,
          arguments: {
            objective: "Forced plan",
            steps: [{ id: "inspect", title: "Inspect first" }]
          }
        }
      ]
    }
  ]);
  const agent = new Agent({
    llm,
    model: "fake-model",
    planning: {}
  });

  const result = await agent.run("do this in plan mode", { forcePlan: true, maxTurns: 1 });

  assert.equal(result.stoppedBy, "plan_review");
  assert.match(llm.requests[0]?.systemPrompt ?? "", /<runtime_plan_mode forced="true">/);
  assert.equal(result.plan?.reviewStatus, "pending");
});

test("planning remains optional when forcePlan is absent", async () => {
  const llm = new SequenceLlm([{ role: "assistant", content: "simple answer" }]);
  const agent = new Agent({
    llm,
    model: "fake-model",
    planning: {}
  });

  const result = await agent.run("simple answer please");

  assert.equal(result.stoppedBy, "final");
  assert.doesNotMatch(llm.requests[0]?.systemPrompt ?? "", /<runtime_plan_mode/);
});

test("planning gate blocks mutation before create_plan and pauses for plan review", async () => {
  const planner = new Planner();
  const llm = new SequenceLlm([
    {
      role: "assistant",
      content: "",
      toolCalls: [{ id: "call_blocked", name: "write_file", arguments: { path: "x.txt", content: "x" } }]
    },
    {
      role: "assistant",
      content: "",
      toolCalls: [
        {
          id: "call_plan",
          name: CREATE_PLAN_TOOL_NAME,
          arguments: {
            objective: "Write file",
            steps: [{ id: "write", title: "Write file" }]
          }
        }
      ]
    },
    {
      role: "assistant",
      content: "should pause before this"
    }
  ]);
  const writeTool: AgentTool = {
    name: "write_file",
    description: "Fake write",
    access: "write",
    parameters: {
      type: "object",
      properties: {
        path: { type: "string" },
        content: { type: "string" }
      },
      required: ["path", "content"],
      additionalProperties: false
    },
    execute() {
      return { content: "wrote" };
    }
  };
  const agent = new Agent({
    llm,
    model: "fake-model",
    tools: [writeTool],
    planning: { planner }
  });

  const result = await agent.run("write a file", { maxTurns: 4 });

  assert.equal(result.stoppedBy, "plan_review");
  assert.match(result.output, /Plan ready for review/);
  const firstToolResult = llm.requests[1]?.messages.at(-1);
  assert.equal(firstToolResult?.role, "tool");
  if (firstToolResult?.role !== "tool") {
    throw new Error("Expected tool result");
  }
  assert.equal(firstToolResult.isError, true);
  assert.match(firstToolResult.content, /Tool not found: write_file/);
  assert.equal(planner.state?.objective, "Write file");
  assert.equal(planner.state?.reviewStatus, "pending");
  assert.equal(llm.requests[2], undefined);
  assert.equal(result.plan?.objective, "Write file");
  assert.equal(result.plan?.reviewStatus, "pending");
  assert.equal(result.plan?.steps[0]?.status, "pending");
});

test("approved plan opens mutation tools and keeps open-step guard", async () => {
  const planner = new Planner();
  planner.create({
    objective: "Write file",
    steps: [{ id: "write", title: "Write file" }]
  });
  planner.approve({ note: "user approved" });
  const llm = new SequenceLlm([
    {
      role: "assistant",
      content: "",
      toolCalls: [{ id: "call_write", name: "write_file", arguments: { path: "x.txt", content: "x" } }]
    },
    {
      role: "assistant",
      content: "done"
    },
    {
      role: "assistant",
      content: "",
      toolCalls: [
        {
          id: "call_complete_plan",
          name: UPDATE_PLAN_STEP_TOOL_NAME,
          arguments: { id: "write", status: "completed", appendEvidence: ["write_file returned wrote"] }
        }
      ]
    },
    {
      role: "assistant",
      content: "done"
    }
  ]);
  const writeTool: AgentTool = {
    name: "write_file",
    description: "Fake write",
    access: "write",
    parameters: {
      type: "object",
      properties: {
        path: { type: "string" },
        content: { type: "string" }
      },
      required: ["path", "content"],
      additionalProperties: false
    },
    execute() {
      return { content: "wrote" };
    }
  };
  const agent = new Agent({
    llm,
    model: "fake-model",
    tools: [writeTool],
    planning: { planner }
  });

  const result = await agent.run("continue after approval", { maxTurns: 4 });

  assert.equal(result.output, "done");
  assert.match(llm.requests[0]?.systemPrompt ?? "", /review_status="approved"/);
  assert.equal(llm.requests[0]?.tools?.some((tool) => tool.name === "write_file"), true);
  assert.equal(llm.requests[1]?.messages.at(-1)?.role, "tool");
  assert.equal(llm.requests[1]?.messages.at(-1)?.content, "wrote");
  assert.equal(llm.requests[2]?.messages.at(-1)?.role, "tool");
  assert.match(llm.requests[2]?.messages.at(-1)?.content ?? "", /Plan mode guard/);
  assert.equal(result.plan?.steps[0]?.status, "completed");
});

test("plan approval can resume the model loop as a visible tool result", async () => {
  const planner = new Planner();
  planner.create({
    objective: "Write file after review",
    steps: [{ id: "write", title: "Write file" }]
  });
  const llm = new SequenceLlm([
    {
      role: "assistant",
      content: "",
      toolCalls: [{ id: "call_write", name: "write_file", arguments: { path: "x.txt", content: "x" } }]
    },
    {
      role: "assistant",
      content: "",
      toolCalls: [
        {
          id: "call_complete_plan",
          name: UPDATE_PLAN_STEP_TOOL_NAME,
          arguments: { id: "write", status: "completed", appendEvidence: ["write_file returned wrote"] }
        }
      ]
    },
    {
      role: "assistant",
      content: "done"
    }
  ]);
  const events: AgentEvent[] = [];
  const writeTool: AgentTool = {
    name: "write_file",
    description: "Fake write",
    access: "write",
    parameters: {
      type: "object",
      properties: {
        path: { type: "string" },
        content: { type: "string" }
      },
      required: ["path", "content"],
      additionalProperties: false
    },
    execute() {
      return { content: "wrote" };
    }
  };
  const agent = new Agent({
    llm,
    model: "fake-model",
    tools: [writeTool],
    planning: { planner },
    onEvent(event) {
      events.push(event);
    }
  });

  const result = await agent.continueWithToolCall(
    { id: "call_approve_plan", name: APPROVE_PLAN_TOOL_NAME, arguments: { note: "Looks good" } },
    { maxTurns: 4 }
  );

  assert.equal(result.output, "done");
  assert.equal(result.stoppedBy, "final");
  assert.equal(planner.state?.reviewStatus, "approved");
  assert.equal(result.plan?.steps[0]?.status, "completed");
  assert.equal(llm.requests.length, 3);
  assert.equal(llm.requests[0]?.messages.some((message) => message.role === "user"), false);
  const approvalAssistant = llm.requests[0]?.messages.at(-2);
  assert.equal(approvalAssistant?.role, "assistant");
  if (approvalAssistant?.role !== "assistant") {
    throw new Error("Expected synthetic approval assistant message");
  }
  assert.equal(approvalAssistant.toolCalls?.[0]?.name, APPROVE_PLAN_TOOL_NAME);
  const approvalResult = llm.requests[0]?.messages.at(-1);
  assert.equal(approvalResult?.role, "tool");
  if (approvalResult?.role !== "tool") {
    throw new Error("Expected approval tool result");
  }
  assert.equal(approvalResult.toolName, APPROVE_PLAN_TOOL_NAME);
  assert.match(approvalResult.content, /Plan approved/);
  assert.equal(llm.requests[0]?.tools?.some((tool) => tool.name === "write_file"), true);
  assert.equal(llm.requests[1]?.messages.at(-1)?.role, "tool");
  assert.equal(llm.requests[1]?.messages.at(-1)?.content, "wrote");
  assert.deepEqual(
    events.filter((event) => event.type === "agent_start").map((event) => event.input),
    ["[tool:approve_plan]"]
  );
  assert.equal(events.filter((event) => event.type === "turn_end").length, 3);
});

test("agent exposes tool-call continuation as an async iterable event stream", async () => {
  const planner = new Planner();
  planner.create({
    objective: "Approve through event stream",
    steps: [{ id: "done", title: "Finish" }]
  });
  planner.updateStep({ id: "done", status: "completed", appendEvidence: ["already complete"] });
  const llm = new SequenceLlm([
    {
      role: "assistant",
      content: "approved and done"
    }
  ]);
  const agent = new Agent({
    llm,
    model: "fake-model",
    planning: { planner }
  });

  const events: AgentEvent[] = [];
  for await (const event of agent.continueWithToolCallEvents({
    id: "call_approve_stream",
    name: APPROVE_PLAN_TOOL_NAME,
    arguments: {}
  })) {
    events.push(event);
  }

  assert.deepEqual(
    events.filter((event) => event.type === "agent_start").map((event) => event.input),
    ["[tool:approve_plan]"]
  );
  assert.equal(events.some((event) => event.type === "tool_end" && event.result.toolName === APPROVE_PLAN_TOOL_NAME), true);
  assert.equal(events.at(-1)?.type, "agent_end");
  assert.equal(planner.state?.reviewStatus, "approved");
  assert.equal(llm.requests[0]?.messages.at(-1)?.role, "tool");
});

test("planning guard prevents a final answer while plan steps remain open", async () => {
  const planner = new Planner();
  planner.create({
    objective: "Complete work",
    steps: [{ id: "work", title: "Do the work" }]
  });
  planner.approve({ note: "approved before execution" });
  const events: AgentEvent[] = [];
  const llm = new SequenceLlm([
    {
      role: "assistant",
      content: "done too early"
    },
    {
      role: "assistant",
      content: "",
      toolCalls: [
        {
          id: "call_complete",
          name: UPDATE_PLAN_STEP_TOOL_NAME,
          arguments: { id: "work", status: "completed", appendEvidence: ["verified work"] }
        }
      ]
    },
    {
      role: "assistant",
      content: "done after plan completion"
    }
  ]);
  const agent = new Agent({
    llm,
    model: "fake-model",
    planning: { planner },
    onEvent(event) {
      events.push(event);
    }
  });

  const result = await agent.run("finish the plan", { maxTurns: 3 });

  assert.equal(result.output, "done after plan completion");
  assert.equal(result.stoppedBy, "final");
  assert.equal(result.turns, 3);
  assert.equal(planner.state?.steps[0]?.status, "completed");
  const firstRequestLastMessage = llm.requests[1]?.messages.at(-1);
  assert.equal(firstRequestLastMessage?.role, "tool");
  if (firstRequestLastMessage?.role !== "tool") {
    throw new Error("Expected plan guard tool result");
  }
  assert.match(firstRequestLastMessage.content, /cannot end while plan steps are pending or in_progress/);
  const guardedAssistant = agent.history.find(
    (message): message is AssistantMessage => message.role === "assistant" && message.content === "done too early"
  );
  assert.equal(guardedAssistant?.toolCalls?.[0]?.name, "read_plan");
  assert.equal(events.filter((event) => event.type === "agent_end").length, 1);
});

test("planning guard allows final answers when all plan steps are closed", async () => {
  const planner = new Planner();
  planner.create({
    objective: "Closed plan",
    steps: [{ id: "done", title: "Done step" }]
  });
  planner.updateStep({ id: "done", status: "completed", appendEvidence: ["already verified"] });
  const llm = new SequenceLlm([
    {
      role: "assistant",
      content: "finished"
    }
  ]);
  const agent = new Agent({
    llm,
    model: "fake-model",
    planning: { planner }
  });

  const result = await agent.run("report");

  assert.equal(result.output, "finished");
  assert.equal(result.stoppedBy, "final");
  assert.equal(result.turns, 1);
  assert.equal(llm.requests.length, 1);
});

test("planning disabled leaves mutation tools visible", async () => {
  const llm = new SequenceLlm([{ role: "assistant", content: "ready" }]);
  const agent = new Agent({
    llm,
    model: "fake-model",
    tools: [createShellTool()],
    planning: false
  });

  await agent.run("hello");

  assert.equal(llm.requests[0]?.tools?.some((tool) => tool.name === "execute_command"), true);
  assert.doesNotMatch(llm.requests[0]?.systemPrompt ?? "", /Planning tools are available/);
});

test("agent adds conversation background to the system prompt when created", async () => {
  const llm = new SequenceLlm([
    {
      role: "assistant",
      content: "ready"
    }
  ]);
  const agent = new Agent({
    llm,
    model: "fake-model",
    systemPrompt: "Base system prompt.",
    tools: [calculatorTool],
    background: {
      cwd: "/workspace/project",
      currentDate: "2026-06-09",
      timezone: "Asia/Shanghai",
      shell: "zsh"
    }
  });

  await agent.run("hello");
  const systemPrompt = llm.requests[0]?.systemPrompt ?? "";

  assert.match(systemPrompt, /^You are Singularity/);
  assert.match(systemPrompt, /Base system prompt\./);
  assert.match(systemPrompt, /<environment_context>/);
  assert.match(systemPrompt, /<cwd>\/workspace\/project<\/cwd>/);
  assert.match(systemPrompt, /<current_date>2026-06-09<\/current_date>/);
  assert.match(systemPrompt, /<timezone>Asia\/Shanghai<\/timezone>/);
  assert.match(systemPrompt, /<shell>zsh<\/shell>/);
  assert.match(systemPrompt, /<available_tools>/);
  assert.match(systemPrompt, /<tool name="calculator" \/>/);
});

test("background false leaves the configured system prompt untouched", async () => {
  const llm = new SequenceLlm([
    {
      role: "assistant",
      content: "ready"
    }
  ]);
  const agent = new Agent({
    llm,
    model: "fake-model",
    systemPrompt: "Only this prompt.",
    background: false
  });

  await agent.run("hello");

  assert.equal(llm.requests[0]?.systemPrompt, "Only this prompt.");
});

test("agent creates a system prompt from background without a base prompt", async () => {
  const llm = new SequenceLlm([
    {
      role: "assistant",
      content: "ready"
    }
  ]);
  const agent = new Agent({
    llm,
    model: "fake-model",
    background: {
      cwd: "/workspace/no-base",
      currentDate: "2026-06-09",
      includeTimezone: false,
      includeShell: false,
      includeTools: false
    }
  });

  await agent.run("hello");
  const systemPrompt = llm.requests[0]?.systemPrompt ?? "";

  assert.match(systemPrompt, /^You are Singularity/);
  assert.match(systemPrompt, /<environment_context>/);
  assert.match(systemPrompt, /<cwd>\/workspace\/no-base<\/cwd>/);
  assert.match(systemPrompt, /<current_date>2026-06-09<\/current_date>/);
  assert.doesNotMatch(systemPrompt, /<available_tools>/);
});

test("agent applies context engine request view without mutating full history", async () => {
  const longOutput = "a".repeat(400);
  const longTool: AgentTool = {
    name: "long_output",
    description: "Return a long output",
    parameters: { type: "object", properties: {}, additionalProperties: false },
    execute() {
      return { content: longOutput };
    }
  };
  const llm = new SequenceLlm([
    {
      role: "assistant",
      content: "",
      toolCalls: [{ id: "call_1", name: "long_output", arguments: {} }]
    },
    {
      role: "assistant",
      content: "done"
    }
  ]);
  const agent = new Agent({
    llm,
    model: "fake-model",
    tools: [longTool],
    context: { maxToolResultTokens: 10 }
  });

  await agent.run("Use long tool");
  const requestToolMessage = llm.requests[1]?.messages.at(-1);
  const historyToolMessage = agent.history.find((message) => message.role === "tool");

  assert.equal(requestToolMessage?.role, "tool");
  assert.equal(historyToolMessage?.role, "tool");
  if (requestToolMessage?.role !== "tool" || historyToolMessage?.role !== "tool") {
    throw new Error("Expected tool messages");
  }
  assert.match(requestToolMessage.content, /ContextEngine truncated tool result/);
  assert.equal(historyToolMessage.content, longOutput);
});

test("agent marks assistant usage from a non-compacted request as reusable", async () => {
  const llm = new SequenceLlm([
    {
      role: "assistant",
      content: "ready",
      usage: { inputTokens: 8, outputTokens: 2, totalTokens: 10 }
    }
  ]);
  const agent = new Agent({
    llm,
    model: "fake-model",
    memory: false,
    context: { contextWindowTokens: 1000, reservedOutputTokens: 0 }
  });

  await agent.run("hello");

  const assistant = agent.history.find((message) => message.role === "assistant");
  assert.equal(assistant?.role, "assistant");
  if (assistant?.role !== "assistant") {
    throw new Error("Expected assistant message");
  }
  assert.equal(assistant.context?.requestCompacted, false);
  assert.deepEqual(assistant.usage, { inputTokens: 8, outputTokens: 2, totalTokens: 10 });
});

test("agent marks assistant usage from a compacted request as non-reusable", async () => {
  const llm = new SequenceLlm([
    {
      role: "assistant",
      content: "handoff summary"
    },
    {
      role: "assistant",
      content: "ready",
      usage: { inputTokens: 8, outputTokens: 2, totalTokens: 10 }
    }
  ]);
  const agent = new Agent({
    llm,
    model: "fake-model",
    context: {
      contextWindowTokens: 8,
      reservedOutputTokens: 0,
      keepRecentTokens: 1
    }
  });

  await agent.run("this request should cross the tiny budget");

  assert.deepEqual(llm.requests[0]?.tools, []);
  assert.match(llm.requests[0]?.messages.at(-1)?.content ?? "", /context checkpoint compaction/i);
  assert.equal(llm.requests[1]?.metadata?.context?.compacted, true);
  const assistant = agent.history.find((message) => message.role === "assistant");
  assert.equal(assistant?.role, "assistant");
  if (assistant?.role !== "assistant") {
    throw new Error("Expected assistant message");
  }
  assert.equal(assistant.context?.requestCompacted, true);
});

test("agent automatically compacts with a full-history summary while preserving the current goal", async () => {
  const llm = new SequenceLlm([
    {
      role: "assistant",
      content: "First answer with important implementation detail."
    },
    {
      role: "assistant",
      content: "Summary: preserve the implementation detail and continue with the second request.",
      usage: { inputTokens: 20, outputTokens: 5, totalTokens: 25 }
    },
    {
      role: "assistant",
      content: "continued from summary"
    }
  ]);
  const agent = new Agent({
    llm,
    model: "fake-model",
    tools: [calculatorTool],
    context: false
  });

  await agent.run("First detailed request", { context: false });
  await agent.run("Second request after history grows", {
    context: {
      contextWindowTokens: 12,
      reservedOutputTokens: 0,
      keepRecentTokens: 1
    }
  });

  const summaryRequest = llm.requests[1];
  assert.ok(summaryRequest);
  assert.deepEqual(summaryRequest.tools, []);
  assert.equal(summaryRequest.messages.some((message) => message.role === "assistant" && message.content.includes("important implementation detail")), true);
  assert.match(summaryRequest.messages.at(-1)?.content ?? "", /context checkpoint compaction/i);

  const compactedRequest = llm.requests[2];
  assert.equal(compactedRequest?.metadata?.context?.compacted, true);
  assert.equal(compactedRequest?.metadata?.context?.compactionSummarySource, "model");
  assert.equal(compactedRequest?.metadata?.context?.compaction?.mode, "automatic");
  assert.equal(compactedRequest?.metadata?.context?.compaction?.summaryCall?.responseUsage?.totalTokens, 25);
  assert.ok((compactedRequest?.metadata?.context?.compaction?.summaryCall?.summaryTokens ?? 0) > 0);
  assert.equal(typeof compactedRequest?.metadata?.context?.compaction?.decision.totalTokens, "number");
  assert.equal(typeof compactedRequest?.metadata?.context?.compaction?.compacted.messageTokens, "number");
  assert.equal(compactedRequest?.tools?.[0]?.name, "calculator");
  assert.match(compactedRequest?.systemPrompt ?? "", /<available_tools>/);
  assert.equal(compactedRequest?.messages[0]?.content, "First detailed request");
  assert.match(compactedRequest?.messages[1]?.content ?? "", /Summary: preserve the implementation detail/);
  assert.equal(compactedRequest?.messages.at(-1)?.content, "Second request after history grows");
  assert.equal(compactedRequest?.messages.some((message) => message.role === "assistant" && message.content.includes("important implementation detail")), false);
  assert.equal(agent.history.some((message) => message.role === "assistant" && message.content.includes("important implementation detail")), true);
  assert.equal(agent.history.at(-1)?.role, "assistant");
  assert.equal(agent.history.at(-1)?.content, "continued from summary");
});

test("agent lets the main model offload compaction to a side worker and projects the block on the next turn", async () => {
  const mainLlm = new SequenceLlm([
    {
      role: "assistant",
      content: "First answer with implementation detail."
    },
    {
      role: "assistant",
      content: "",
      toolCalls: [
        {
          id: "compress_1",
          name: "compact_context",
          arguments: {
            goal: "free context before continuing",
            maxBlocks: 1
          }
        }
      ]
    },
    {
      role: "assistant",
      content: "continued with compressed context"
    }
  ]);
  const compressionLlm = new SequenceLlm([
    {
      role: "assistant",
      content: JSON.stringify({
        blocks: [
          {
            startId: "m0001",
            endId: "m0002",
            topic: "first implementation detail",
            summary: "Summary: preserve the implementation detail."
          }
        ]
      }),
      usage: { inputTokens: 20, outputTokens: 5, totalTokens: 25 }
    }
  ]);
  const agent = new Agent({
    llm: mainLlm,
    model: "main-model",
    compressionLlm,
    compressionModel: "worker-model",
    context: false
  });

  await agent.run("First detailed request", { context: false });
  await agent.run("Second request after history grows", {
    context: {
      contextWindowTokens: 10000,
      reservedOutputTokens: 0,
      compactionThresholdRatio: 1,
      keepRecentTokens: 1,
      dynamicCompression: {
        enabled: true,
        triggerTokens: 1,
        minCompressMessages: 2
      }
    }
  });

  const protocolRequest = mainLlm.requests[1];
  assert.equal(protocolRequest?.tools?.some((tool) => tool.name === "compact_context"), true);
  assert.match(protocolRequest?.systemPrompt ?? "", /dynamic_context_compression/);
  assert.match(protocolRequest?.messages[0]?.content ?? "", /\[context m0001 role=user\]/);
  assert.match(protocolRequest?.messages[1]?.content ?? "", /\[context m0002 role=assistant\]/);

  assert.equal(compressionLlm.requests.length, 1);
  assert.equal(compressionLlm.requests[0]?.model, "worker-model");
  assert.equal(compressionLlm.requests[0]?.systemPrompt, protocolRequest?.systemPrompt);
  assert.equal(compressionLlm.requests[0]?.tools?.length, 0);
  assert.deepEqual(compressionLlm.requests[0]?.messages.slice(0, -1), protocolRequest?.messages);
  assert.match(compressionLlm.requests[0]?.messages[0]?.content ?? "", /\[context m0001 role=user\]/);
  assert.match(compressionLlm.requests[0]?.messages.at(-1)?.content ?? "", /side worker for dynamic context compression/);
  assert.match(compressionLlm.requests[0]?.messages.at(-1)?.content ?? "", /free context before continuing/);

  const compressedMainRequest = mainLlm.requests[2];
  assert.equal(compressedMainRequest?.model, "main-model");
  assert.equal(compressedMainRequest?.metadata?.context?.dynamicCompression?.applied, true);
  assert.equal(compressedMainRequest?.messages.some((message) => message.content.includes("Dynamic context summary: b1")), true);
  assert.equal(compressedMainRequest?.messages.some((message) => message.content.includes("Summary: preserve the implementation detail.")), true);
  assert.equal(compressedMainRequest?.messages.some((message) => message.content.includes("First answer with implementation detail.")), false);
  assert.equal(compressedMainRequest?.messages.some((message) => message.role === "tool" && message.toolName === "compact_context"), true);
  assert.equal(agent.history.some((message) => message.role === "assistant" && message.content === "First answer with implementation detail."), true);
  assert.equal(agent.history.some((message) => message.role === "tool" && message.toolName === "compact_context"), true);
});

test("automatic compaction preserves the live tool-call turn and tool guidance", async () => {
  const longOutput = "tool-result-".repeat(260);
  const longTool: AgentTool = {
    name: "long_output",
    description: "Return a long output",
    parameters: { type: "object", properties: {}, additionalProperties: false },
    execute() {
      return { content: longOutput };
    }
  };
  const llm = new SequenceLlm([
    {
      role: "assistant",
      content: "",
      toolCalls: [{ id: "call_1", name: "long_output", arguments: {} }]
    },
    {
      role: "assistant",
      content: "Summary: keep the live tool result available."
    },
    {
      role: "assistant",
      content: "done"
    }
  ]);
  const agent = new Agent({
    llm,
    model: "fake-model",
    tools: [longTool],
    memory: false,
    context: false
  });

  await agent.run("Use long tool", {
    context: {
      contextWindowTokens: 1000,
      reservedOutputTokens: 0,
      keepRecentTokens: 1,
      maxToolResultTokens: 1000
    }
  });

  const summaryRequest = llm.requests[1];
  assert.ok(summaryRequest);
  assert.deepEqual(summaryRequest.tools, []);
  assert.equal(summaryRequest.messages.some((message) => message.role === "tool" && message.content === longOutput), true);

  const compactedRequest = llm.requests[2];
  assert.equal(compactedRequest?.metadata?.context?.compacted, true);
  assert.equal(compactedRequest?.tools?.[0]?.name, "long_output");
  assert.match(compactedRequest?.systemPrompt ?? "", /<available_tools>/);
  assert.match(compactedRequest?.messages[0]?.content ?? "", /Summary: keep the live tool result/);

  const retainedUser = compactedRequest?.messages.find((message) => message.role === "user" && message.content === "Use long tool");
  const retainedAssistant = compactedRequest?.messages.find((message) => message.role === "assistant" && message.toolCalls?.[0]?.id === "call_1");
  const retainedTool = compactedRequest?.messages.find((message) => message.role === "tool" && message.toolCallId === "call_1");
  assert.ok(retainedUser);
  assert.ok(retainedAssistant);
  assert.ok(retainedTool);
  assert.ok((compactedRequest?.messages.findIndex((message) => message === retainedUser) ?? -1) < (compactedRequest?.messages.findIndex((message) => message === retainedAssistant) ?? -1));
  assert.ok((compactedRequest?.messages.findIndex((message) => message === retainedAssistant) ?? -1) < (compactedRequest?.messages.findIndex((message) => message === retainedTool) ?? -1));
  assert.equal(compactedRequest?.messages.at(-1), retainedTool);

  const historyTool = agent.history.find((message) => message.role === "tool");
  assert.equal(historyTool?.role, "tool");
  assert.equal(historyTool?.content, longOutput);
});

test("agent can manually compact current history", async () => {
  const llm = new SequenceLlm([
    {
      role: "assistant",
      content: "First answer with important implementation detail."
    },
    {
      role: "assistant",
      content: "Second answer kept as recent context."
    },
    {
      role: "assistant",
      content: "Manual summary: preserve the implementation detail.",
      usage: { inputTokens: 18, outputTokens: 4, totalTokens: 22 }
    }
  ]);
  const agent = new Agent({
    llm,
    model: "fake-model",
    tools: [calculatorTool],
    context: false
  });

  await agent.run("First detailed request", { context: false });
  await agent.run("Second detailed request", { context: false });
  const result = await agent.compactHistory({
    context: {
      contextWindowTokens: 1000,
      reservedOutputTokens: 0,
      keepRecentTokens: 1
    }
  });

  assert.equal(result.compacted, true);
  assert.equal(result.context?.compaction?.mode, "manual");
  assert.equal(result.context?.compaction?.summaryCall?.responseUsage?.totalTokens, 22);
  assert.deepEqual(llm.requests[2]?.tools, []);
  assert.match(llm.requests[2]?.messages.at(-1)?.content ?? "", /context checkpoint compaction/i);
  assert.equal(agent.history[0]?.role, "user");
  assert.equal(agent.history[0]?.content, "First detailed request");
  assert.match(agent.history[1]?.content ?? "", /Manual summary/);
  assert.equal(agent.history.some((message) => message.role === "assistant" && message.content.includes("important implementation detail")), false);
  assert.equal(agent.history.at(-1)?.content, "Second answer kept as recent context.");
});

test("agent emits request context metadata on turn_end", async () => {
  const llm = new SequenceLlm([
    {
      role: "assistant",
      content: "ready",
      usage: { inputTokens: 8, outputTokens: 2, totalTokens: 10 }
    }
  ]);
  const events: AgentEvent[] = [];
  const agent = new Agent({
    llm,
    model: "fake-model",
    memory: false,
    context: { contextWindowTokens: 1000, reservedOutputTokens: 0 },
    onEvent(event) {
      events.push(event);
    }
  });

  await agent.run("hello");

  const turnEnd = events.find((event) => event.type === "turn_end");
  assert.equal(turnEnd?.type, "turn_end");
  if (turnEnd?.type !== "turn_end") {
    throw new Error("Expected turn_end event");
  }
  assert.equal(turnEnd.context?.compacted, false);
  assert.equal(turnEnd.context?.tokenEstimateSource, "heuristic");
  assert.equal(typeof turnEnd.context?.estimatedInputTokens, "number");
});

test("agent ignores configured phase summary while the feature is paused", async () => {
  const dir = await mkdtemp(join(tmpdir(), "singularity-phase-summary-"));
  try {
    const mainLlm = new SequenceLlm([
      {
        role: "assistant",
        content: "Implemented the feature."
      }
    ]);
    const summaryLlm = new SequenceLlm([
      {
        role: "assistant",
        content: "This summary should not be requested.",
        usage: { inputTokens: 30, outputTokens: 12, totalTokens: 42 }
      }
    ]);
    const store = new MarkdownMemoryStore({ path: join(dir, "MEMORY.md") });
    const events: AgentEvent[] = [];
    const agent = new Agent({
      llm: mainLlm,
      model: "main-model",
      tools: [calculatorTool],
      memory: {
        store,
        phaseSummary: {
          llm: summaryLlm,
          model: "summary-model",
          tags: ["test"]
        }
      },
      onEvent(event) {
        events.push(event);
      }
    });

    const result = await agent.run("Please implement this");

    assert.equal(result.output, "Implemented the feature.");
    assert.equal(mainLlm.requests.length, 1);
    await agent.waitForBackgroundTasks();
    assert.equal(agent.history.length, 2);
    assert.equal(summaryLlm.requests.length, 0);
    assert.equal(events.some((event) => event.type === "memory_summary_start"), false);
    assert.equal(events.some((event) => event.type === "memory_summary_end"), false);
    assert.equal(events.some((event) => event.type === "memory_summary_error"), false);
    const entries = await store.list({ tag: "phase-summary" });
    assert.equal(entries.length, 0);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test("context false bypasses request-view truncation", async () => {
  const longOutput = "b".repeat(400);
  const longTool: AgentTool = {
    name: "long_output",
    description: "Return a long output",
    parameters: { type: "object", properties: {}, additionalProperties: false },
    execute() {
      return { content: longOutput };
    }
  };
  const llm = new SequenceLlm([
    {
      role: "assistant",
      content: "",
      toolCalls: [{ id: "call_1", name: "long_output", arguments: {} }]
    },
    {
      role: "assistant",
      content: "done"
    }
  ]);
  const agent = new Agent({
    llm,
    model: "fake-model",
    tools: [longTool],
    context: false
  });

  await agent.run("Use long tool", { context: false });
  const requestToolMessage = llm.requests[1]?.messages.at(-1);

  assert.equal(requestToolMessage?.role, "tool");
  if (requestToolMessage?.role !== "tool") {
    throw new Error("Expected tool message");
  }
  assert.equal(requestToolMessage.content, longOutput);
});

test("run-level context options can re-enable agent-level disabled context", async () => {
  const longOutput = "c".repeat(400);
  const longTool: AgentTool = {
    name: "long_output",
    description: "Return a long output",
    parameters: { type: "object", properties: {}, additionalProperties: false },
    execute() {
      return { content: longOutput };
    }
  };
  const llm = new SequenceLlm([
    {
      role: "assistant",
      content: "",
      toolCalls: [{ id: "call_1", name: "long_output", arguments: {} }]
    },
    {
      role: "assistant",
      content: "done"
    }
  ]);
  const agent = new Agent({
    llm,
    model: "fake-model",
    tools: [longTool],
    context: false
  });

  await agent.run("Use long tool", { context: { maxToolResultTokens: 10 } });
  const requestToolMessage = llm.requests[1]?.messages.at(-1);

  assert.equal(requestToolMessage?.role, "tool");
  if (requestToolMessage?.role !== "tool") {
    throw new Error("Expected tool message");
  }
  assert.match(requestToolMessage.content, /ContextEngine truncated tool result/);
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

test("OpenAI Responses client maps non-streaming usage", async () => {
  const client = new OpenAIResponsesClient({
    apiKey: "test-key",
    fetchImpl: async () =>
      Response.json({
        output: [{ type: "message", role: "assistant", content: [{ type: "output_text", text: "ok" }] }],
        usage: {
          input_tokens: 36,
          input_tokens_details: { cached_tokens: 8 },
          output_tokens: 12,
          total_tokens: 48
        }
      })
  });

  const message = await client.complete({ model: "fake-model", messages: [{ role: "user", content: "hi" }] });

  assert.equal(message.content, "ok");
  assert.deepEqual(message.usage, {
    inputTokens: 36,
    outputTokens: 12,
    totalTokens: 48,
    cacheReadInputTokens: 8,
    cacheCreationInputTokens: undefined
  });
});

test("OpenAI Responses client maps response.completed usage while streaming", async () => {
  const sse = [
    sseEvent({ type: "response.output_text.delta", delta: "ok" }),
    sseEvent({
      type: "response.completed",
      response: {
        usage: {
          input_tokens: 11,
          input_tokens_details: { cached_tokens: 3 },
          output_tokens: 5,
          total_tokens: 16
        }
      }
    })
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

  assert.equal(events.at(-1)?.type, "done");
  const done = events.at(-1);
  if (done?.type !== "done") {
    throw new Error("Expected done event");
  }
  assert.deepEqual(done.message.usage, {
    inputTokens: 11,
    outputTokens: 5,
    totalTokens: 16,
    cacheReadInputTokens: 3,
    cacheCreationInputTokens: undefined
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
