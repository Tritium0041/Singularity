import assert from "node:assert/strict";
import test from "node:test";
import {
  BudgetManager,
  ContextEngine,
  DEFAULT_AGENT_INSTRUCTIONS,
  DEFAULT_CONTEXT_ENGINE_OPTIONS,
  estimateMessagesTokens,
  estimateMessageTokens,
  estimateRequestTokens,
  estimateTextTokens,
  PromptBuilder
} from "../src/context/index.js";
import type { LlmRequest } from "../src/llm/types.js";
import type { AgentMessage, ToolResultMessage } from "../src/types.js";

test("token estimator covers text, messages, tool calls, and tool specs", () => {
  const messages: AgentMessage[] = [
    { role: "user", content: "12345678" },
    {
      role: "assistant",
      content: "abcd",
      reasoning: { summary: "reasoning summary" },
      toolCalls: [{ id: "call_1", name: "read_file", arguments: { path: "src/index.ts" } }]
    },
    { role: "tool", toolCallId: "call_1", toolName: "read_file", content: "tool output" }
  ];
  const request: LlmRequest = {
    model: "fake-model",
    systemPrompt: "system prompt",
    messages,
    tools: [
      {
        name: "read_file",
        description: "Read a file",
        parameters: { type: "object", properties: { path: { type: "string" } }, required: ["path"] }
      }
    ]
  };

  assert.equal(estimateTextTokens("1234"), 1);
  assert.equal(estimateTextTokens("12345"), 2);
  assert.ok(estimateMessageTokens(messages[1]!) > estimateTextTokens("abcd"));
  assert.deepEqual(estimateRequestTokens(request), estimateRequestTokens(request));
  assert.ok(estimateRequestTokens(request).totalTokens > estimateTextTokens("system prompt"));
});

test("token estimator calibrates from provider usage plus appended messages", () => {
  const messages: AgentMessage[] = [
    { role: "user", content: "old request" },
    {
      role: "assistant",
      content: "provider counted this output",
      usage: { inputTokens: 1000, outputTokens: 17 }
    },
    { role: "tool", toolCallId: "call_1", toolName: "read_file", content: "fresh tool output" },
    { role: "user", content: "fresh follow-up" }
  ];

  const estimate = estimateRequestTokens({ model: "fake-model", messages });
  const expectedAppended = 17 + estimateMessagesTokens(messages.slice(2));

  assert.equal(estimate.source, "provider_usage");
  assert.equal(estimate.providerInputTokens, 1000);
  assert.equal(estimate.providerOutputTokens, 17);
  assert.equal(estimate.appendedMessageTokens, expectedAppended);
  assert.equal(estimate.totalTokens, 1000 + expectedAppended);
});

test("token estimator ignores provider usage from compacted request views", () => {
  const messages: AgentMessage[] = [
    { role: "user", content: "old request" },
    {
      role: "assistant",
      content: "compacted response",
      usage: { inputTokens: 1000, outputTokens: 17 },
      context: { requestCompacted: true }
    },
    { role: "user", content: "fresh follow-up" }
  ];

  const estimate = estimateRequestTokens({ model: "fake-model", messages });

  assert.equal(estimate.source, "heuristic");
  assert.equal(estimate.providerInputTokens, undefined);
});

test("default context budget uses a 256k window and compacts at 90 percent", () => {
  assert.equal(DEFAULT_CONTEXT_ENGINE_OPTIONS.contextWindowTokens, 256000);
  assert.equal(DEFAULT_CONTEXT_ENGINE_OPTIONS.compactionThresholdRatio, 0.9);

  const budget = new BudgetManager(DEFAULT_CONTEXT_ENGINE_OPTIONS);

  assert.equal(budget.compactionTriggerTokens, 230400);
  assert.equal(budget.shouldCompact(230400), false);
  assert.equal(budget.shouldCompact(230401), true);
});

test("context budget still honors reserved output tokens before threshold", () => {
  const budget = new BudgetManager({
    ...DEFAULT_CONTEXT_ENGINE_OPTIONS,
    contextWindowTokens: 1000,
    compactionThresholdRatio: 0.99,
    reservedOutputTokens: 200
  });

  assert.equal(budget.availableInputTokens, 800);
  assert.equal(budget.compactionTriggerTokens, 800);
});

test("prompt builder renders base prompt with conversation background", () => {
  const prompt = new PromptBuilder().buildConversationSystemPrompt({
    basePrompt: "Base instructions.",
    background: {
      cwd: "/tmp/project&workspace",
      currentDate: "2026-06-09",
      timezone: "Asia/Shanghai",
      shell: "zsh",
      tools: [{ name: "read_file", description: "Read <files>" }],
      includeToolDescriptions: true,
      extra: ["<series>Build An Agent From Scratch</series>"]
    }
  });

  assert.match(prompt ?? "", /^You are Singularity/);
  assert.match(prompt ?? "", /Base instructions\./);
  assert.match(prompt ?? "", /<environment_context>/);
  assert.match(prompt ?? "", /<cwd>\/tmp\/project&amp;workspace<\/cwd>/);
  assert.match(prompt ?? "", /<current_date>2026-06-09<\/current_date>/);
  assert.match(prompt ?? "", /<timezone>Asia\/Shanghai<\/timezone>/);
  assert.match(prompt ?? "", /<shell>zsh<\/shell>/);
  assert.match(prompt ?? "", /<tool name="read_file">Read &lt;files&gt;<\/tool>/);
  assert.match(prompt ?? "", /<series>Build An Agent From Scratch<\/series>/);
});

test("default agent instructions are adapted to current runtime capabilities", () => {
  assert.match(DEFAULT_AGENT_INSTRUCTIONS, /You are Singularity/);
  assert.doesNotMatch(DEFAULT_AGENT_INSTRUCTIONS, /You are Codex/);
  assert.doesNotMatch(DEFAULT_AGENT_INSTRUCTIONS, /apply_patch/);
  assert.doesNotMatch(DEFAULT_AGENT_INSTRUCTIONS, /approval/i);
  assert.doesNotMatch(DEFAULT_AGENT_INSTRUCTIONS, /sandbox/i);
  assert.match(DEFAULT_AGENT_INSTRUCTIONS, /rg/);
  assert.match(DEFAULT_AGENT_INSTRUCTIONS, /dirty git worktree/);
});

test("context engine truncates tool results in request view without mutating history", () => {
  const originalToolMessage: ToolResultMessage = {
    role: "tool",
    toolCallId: "call_1",
    toolName: "read_file",
    content: "a".repeat(400)
  };
  const request: LlmRequest = {
    model: "fake-model",
    messages: [originalToolMessage]
  };

  const prepared = new ContextEngine({ maxToolResultTokens: 10 }).prepare(request);
  const preparedToolMessage = prepared.messages[0] as ToolResultMessage;

  assert.equal(originalToolMessage.content, "a".repeat(400));
  assert.notEqual(preparedToolMessage, originalToolMessage);
  assert.ok(preparedToolMessage.content.length < originalToolMessage.content.length);
  assert.match(preparedToolMessage.content, /ContextEngine truncated tool result/);
  assert.ok(preparedToolMessage.content.startsWith("a"));
});

test("execute_command tool result truncation keeps the tail", () => {
  const request: LlmRequest = {
    model: "fake-model",
    messages: [
      {
        role: "tool",
        toolCallId: "call_1",
        toolName: "execute_command",
        content: `head-${"x".repeat(400)}-tail`
      }
    ]
  };

  const prepared = new ContextEngine({ maxToolResultTokens: 24 }).prepare(request);
  const preparedToolMessage = prepared.messages[0] as ToolResultMessage;

  assert.match(preparedToolMessage.content, /ContextEngine truncated tool result/);
  assert.ok(preparedToolMessage.content.endsWith("-tail"));
});

test("context engine preserves user instructions and drops old model/tool messages when history exceeds budget", () => {
  const request: LlmRequest = {
    model: "fake-model",
    messages: [
      { role: "user", content: "Original goal: inspect the context files and design the engine." },
      {
        role: "assistant",
        content: "",
        toolCalls: [{ id: "call_1", name: "read_file", arguments: { path: "src/agent/agent-loop.ts" } }]
      },
      { role: "tool", toolCallId: "call_1", toolName: "read_file", content: "agent loop source" },
      { role: "assistant", content: "Read the agent loop and found the LLM request construction." },
      { role: "user", content: "Continue with the current implementation." }
    ]
  };

  const prepared = new ContextEngine({
    contextWindowTokens: 20,
    reservedOutputTokens: 5,
    keepRecentTokens: 5
  }).prepare(request);

  assert.equal(prepared.messages[0]?.role, "user");
  assert.equal(prepared.messages[0]?.content, "Original goal: inspect the context files and design the engine.");
  assert.equal(prepared.messages.some((message) => message.role === "assistant"), false);
  assert.equal(prepared.messages.some((message) => message.role === "tool"), false);
  assert.equal(prepared.messages.some((message) => message.content.includes("src/agent/agent-loop.ts")), false);
  assert.equal(prepared.messages.at(-1)?.role, "user");
  assert.equal(prepared.messages.at(-1)?.content, "Continue with the current implementation.");
});

test("context engine compacts when provider-calibrated usage crosses budget", () => {
  const request: LlmRequest = {
    model: "fake-model",
    messages: [
      { role: "user", content: "Original provider-counted request." },
      {
        role: "assistant",
        content: "tool please",
        usage: { inputTokens: 18, outputTokens: 4 },
        toolCalls: [{ id: "call_1", name: "read_file", arguments: { path: "README.md" } }]
      },
      { role: "tool", toolCallId: "call_1", toolName: "read_file", content: "fresh tool output" },
      { role: "user", content: "Continue." }
    ]
  };

  const prepared = new ContextEngine({
    contextWindowTokens: 30,
    reservedOutputTokens: 0,
    keepRecentTokens: 1
  }).prepare(request);

  assert.equal(prepared.metadata?.context?.compacted, true);
  assert.equal(prepared.metadata?.context?.tokenEstimateSource, "heuristic");
  assert.equal(prepared.metadata?.context?.compactionDecisionTokenEstimateSource, "provider_usage");
  assert.ok((prepared.metadata?.context?.compactionDecisionEstimatedInputTokens ?? 0) > 27);
  assert.equal(prepared.messages[0]?.role, "user");
  assert.equal(prepared.messages[0]?.content, "Original provider-counted request.");
  assert.equal(prepared.messages.some((message) => message.role === "assistant"), false);
  assert.equal(prepared.messages.some((message) => message.role === "tool"), false);
  assert.equal(prepared.messages.at(-1)?.content, "Continue.");
});

test("context compaction keeps assistant tool calls paired with tool results", () => {
  const request: LlmRequest = {
    model: "fake-model",
    messages: [
      { role: "user", content: "Old task." },
      { role: "assistant", content: "Old answer." },
      { role: "user", content: "New task that needs a tool." },
      {
        role: "assistant",
        content: "",
        toolCalls: [{ id: "call_2", name: "calculator", arguments: { expression: "2 + 2" } }]
      },
      { role: "tool", toolCallId: "call_2", toolName: "calculator", content: "4" }
    ]
  };

  const prepared = new ContextEngine({
    contextWindowTokens: 16,
    reservedOutputTokens: 4,
    keepRecentTokens: 1
  }).prepare(request);
  const retainedAssistant = prepared.messages.find((message) => message.role === "assistant" && message.toolCalls?.[0]?.id === "call_2");
  const retainedTool = prepared.messages.find((message) => message.role === "tool" && message.toolCallId === "call_2");

  assert.equal(prepared.messages[0]?.role, "user");
  assert.equal(prepared.messages[0]?.content, "Old task.");
  assert.equal(prepared.messages.some((message) => message.role === "assistant" && message.content === "Old answer."), false);
  assert.ok(retainedAssistant);
  assert.ok(retainedTool);
  assert.ok(prepared.messages.findIndex((message) => message === retainedAssistant) < prepared.messages.findIndex((message) => message === retainedTool));
});

test("model handoff compaction summarizes full history and preserves recent complete turns", async () => {
  let summaryRequest: LlmRequest | undefined;
  const request: LlmRequest = {
    model: "fake-model",
    systemPrompt: "Use tools when useful.",
    messages: [
      { role: "user", content: "Old task with important setup." },
      { role: "assistant", content: "Old answer with implementation detail." },
      { role: "user", content: "Current goal: calculate the result with the calculator." },
      {
        role: "assistant",
        content: "",
        toolCalls: [{ id: "call_2", name: "calculator", arguments: { expression: "2 + 2" } }]
      },
      { role: "tool", toolCallId: "call_2", toolName: "calculator", content: "4" }
    ],
    tools: [
      {
        name: "calculator",
        description: "Evaluate arithmetic",
        parameters: { type: "object", properties: { expression: { type: "string" } }, required: ["expression"] }
      }
    ]
  };

  const prepared = await new ContextEngine({
    contextWindowTokens: 16,
    reservedOutputTokens: 0,
    keepRecentTokens: 1
  }).prepareWithHandoff(request, async (nextSummaryRequest) => {
    summaryRequest = nextSummaryRequest;
    return { role: "assistant", content: "Summary: preserve old setup." };
  });

  assert.ok(summaryRequest);
  assert.deepEqual(summaryRequest?.tools, []);
  assert.equal(summaryRequest?.messages.some((message) => message.content.includes("Old task")), true);
  assert.equal(summaryRequest?.messages.some((message) => message.content.includes("Current goal")), true);
  assert.match(summaryRequest?.messages.at(-1)?.content ?? "", /context checkpoint compaction/i);

  assert.equal(prepared.request.systemPrompt, "Use tools when useful.");
  assert.equal(prepared.request.tools?.[0]?.name, "calculator");
  assert.equal(prepared.request.messages[0]?.role, "user");
  assert.equal(prepared.request.messages[0]?.content, "Old task with important setup.");
  assert.equal(prepared.request.messages[1]?.role, "user");
  assert.match(prepared.request.messages[1]?.content ?? "", /Context checkpoint summary/);
  assert.equal(prepared.request.messages.some((message) => message.content.includes("Old answer with implementation detail")), false);

  const retainedUser = prepared.request.messages.find((message) => message.role === "user" && message.content.includes("Current goal"));
  const retainedAssistant = prepared.request.messages.find((message) => message.role === "assistant" && message.toolCalls?.[0]?.id === "call_2");
  const retainedTool = prepared.request.messages.find((message) => message.role === "tool" && message.toolCallId === "call_2");
  assert.ok(retainedUser);
  assert.ok(retainedAssistant);
  assert.ok(retainedTool);
  assert.ok(prepared.request.messages.findIndex((message) => message === retainedUser) < prepared.request.messages.findIndex((message) => message === retainedAssistant));
  assert.ok(prepared.request.messages.findIndex((message) => message === retainedAssistant) < prepared.request.messages.findIndex((message) => message === retainedTool));
  assert.equal(prepared.request.messages.at(-1), retainedTool);
});
