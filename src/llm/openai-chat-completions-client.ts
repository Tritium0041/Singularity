import type { AgentMessage, AssistantMessage, ReasoningOptions, ReasoningReplay, TokenUsage, ToolCall } from "../types.js";
import type { LlmProviderFactory, LlmRequest, LlmStreamEvent, LlmToolSpec, StreamingLlmClient } from "./types.js";
import { parseServerSentEvents } from "./sse.js";
import { cleanTokenUsage, mergeTokenUsage } from "./usage.js";

type OpenAIChatCompletionsClientOptions = {
  apiKey?: string;
  baseURL?: string;
  defaultModel?: string;
  defaultReasoning?: ReasoningOptions | false;
  fetchImpl?: typeof fetch;
};

type ChatMessage =
  | { role: "system"; content: string }
  | { role: "user"; content: string }
  | {
      role: "assistant";
      content: string | null;
      tool_calls?: ChatToolCall[];
      reasoning_content?: string;
      reasoning?: string;
      reasoning_text?: string;
    }
  | { role: "tool"; tool_call_id: string; content: string };

type ChatToolCall = {
  id: string;
  type: "function";
  function: {
    name: string;
    arguments: string;
  };
};

type ChatTool = {
  type: "function";
  function: {
    name: string;
    description: string;
    parameters: Record<string, unknown>;
    strict: false;
  };
};

type ChatChoice = {
  message?: {
    role?: string;
    content?: string | null;
    tool_calls?: ChatToolCall[];
    reasoning_content?: string;
    reasoning?: string;
    reasoning_text?: string;
  };
  delta?: {
    content?: string | null;
    reasoning_content?: string;
    reasoning?: string;
    reasoning_text?: string;
    tool_calls?: Array<{
      index?: number;
      id?: string;
      type?: "function";
      function?: {
        name?: string;
        arguments?: string;
      };
    }>;
  };
  finish_reason?: string | null;
};

type ChatCompletionsBody = {
  choices?: ChatChoice[];
  usage?: ChatUsage | null;
  error?: { message?: string };
};

type ChatUsage = {
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
  prompt_tokens_details?: {
    cached_tokens?: number;
  };
};

type PendingToolCall = {
  id: string;
  name: string;
  argumentsText: string;
};

type OpenAIChatReasoningField = "reasoning_content" | "reasoning" | "reasoning_text";

const OPENAI_CHAT_REASONING_FIELDS: OpenAIChatReasoningField[] = ["reasoning_content", "reasoning", "reasoning_text"];

export class OpenAIChatCompletionsClient implements StreamingLlmClient {
  private readonly apiKey: string | undefined;
  private readonly baseURL: string;
  private readonly defaultModel: string;
  private readonly defaultReasoning: ReasoningOptions | false | undefined;
  private readonly fetchImpl: typeof fetch;

  constructor(options: OpenAIChatCompletionsClientOptions = {}) {
    this.apiKey = options.apiKey ?? process.env.OPENAI_COMPAT_API_KEY ?? process.env.OPENAI_API_KEY;
    this.baseURL = trimTrailingSlash(
      options.baseURL ?? process.env.OPENAI_COMPAT_BASE_URL ?? process.env.OPENAI_BASE_URL ?? "https://api.openai.com/v1"
    );
    this.defaultModel = options.defaultModel ?? process.env.OPENAI_COMPAT_MODEL ?? process.env.OPENAI_MODEL ?? "gpt-4.1-mini";
    this.defaultReasoning = options.defaultReasoning;
    this.fetchImpl = options.fetchImpl ?? fetch;
  }

  async complete(request: LlmRequest): Promise<AssistantMessage> {
    const apiKey = this.requireApiKey();
    const response = await this.createCompletion(request, apiKey, false);
    const body = (await response.json().catch(() => ({}))) as ChatCompletionsBody;
    if (!response.ok) {
      throw new Error(body.error?.message ?? `OpenAI Chat Completions request failed with HTTP ${response.status}`);
    }
    return assistantFromBody(body);
  }

  async *stream(request: LlmRequest): AsyncIterable<LlmStreamEvent> {
    const apiKey = this.requireApiKey();
    const response = await this.createCompletion(request, apiKey, true);
    if (!response.ok) {
      const body = (await response.json().catch(() => ({}))) as ChatCompletionsBody;
      throw new Error(body.error?.message ?? `OpenAI Chat Completions stream failed with HTTP ${response.status}`);
    }
    if (!response.body) {
      throw new Error("OpenAI Chat Completions stream did not include a response body.");
    }

    let content = "";
    let thinkingContent = "";
    let thinkingField: OpenAIChatReasoningField | undefined;
    let usage: TokenUsage | undefined;
    const rawEvents: ChatCompletionsBody[] = [];
    const toolCallsByIndex = new Map<number, PendingToolCall>();
    const toolCallsById = new Map<string, PendingToolCall>();

    for await (const rawEvent of parseServerSentEvents(response.body)) {
      const body = rawEvent as ChatCompletionsBody;
      rawEvents.push(body);
      usage = mergeTokenUsage(usage, usageFromChat(body.usage));
      const choice = body.choices?.[0];
      if (!choice?.delta) {
        continue;
      }

      const textDelta = choice.delta.content;
      if (typeof textDelta === "string" && textDelta.length > 0) {
        content += textDelta;
        yield { type: "text_delta", delta: textDelta };
      }

      const thinkingDelta = firstReasoningField(choice.delta);
      if (thinkingDelta) {
        thinkingField ??= thinkingDelta.field;
        thinkingContent += thinkingDelta.content;
        yield { type: "thinking_delta", delta: thinkingDelta.content };
      }

      for (const deltaToolCall of choice.delta.tool_calls ?? []) {
        const toolCall = resolveToolCall(deltaToolCall, toolCallsByIndex, toolCallsById);
        if (deltaToolCall.function?.name) {
          toolCall.name = deltaToolCall.function.name;
        }
        if (deltaToolCall.id) {
          toolCall.id = deltaToolCall.id;
          toolCallsById.set(deltaToolCall.id, toolCall);
        }

        const argsDelta = deltaToolCall.function?.arguments ?? "";
        if (argsDelta.length > 0) {
          toolCall.argumentsText += argsDelta;
        }
        yield {
          type: "tool_call_delta",
          toolCallId: toolCall.id,
          toolName: toolCall.name,
          delta: argsDelta,
          argumentsText: toolCall.argumentsText
        };
      }
    }

    yield {
      type: "done",
      message: {
        role: "assistant",
        content,
        reasoning: reasoningFromOpenAIChat(thinkingContent, thinkingField),
        toolCalls: pendingToolCallsToToolCalls([...toolCallsByIndex.values()]),
        usage,
        raw: rawEvents
      }
    };
  }

  private requireApiKey(): string {
    if (!this.apiKey) {
      throw new Error("Missing API key. Pass apiKey, set LLM_API_KEY, OPENAI_COMPAT_API_KEY, or OPENAI_API_KEY.");
    }
    return this.apiKey;
  }

  private async createCompletion(request: LlmRequest, apiKey: string, stream: boolean): Promise<Response> {
    const reasoning = request.reasoning !== undefined ? request.reasoning : this.defaultReasoning;
    const body: Record<string, unknown> = {
      model: request.model || this.defaultModel,
      messages: toChatMessages(request.messages, request.systemPrompt),
      tools: request.tools && request.tools.length > 0 ? request.tools.map(toChatTool) : undefined,
      tool_choice: request.tools && request.tools.length > 0 ? "auto" : undefined,
      stream,
      stream_options: stream ? { include_usage: true } : undefined
    };

    if (reasoning && reasoning.effort) {
      body.reasoning_effort = reasoning.effort;
    }

    return this.fetchImpl(`${this.baseURL}/chat/completions`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify(body),
      signal: request.signal
    });
  }
}

export const openAIChatCompletionsProvider: LlmProviderFactory = {
  id: "openai-chat",
  create: (options) => new OpenAIChatCompletionsClient(options)
};

function toChatMessages(messages: AgentMessage[], systemPrompt?: string): ChatMessage[] {
  const result: ChatMessage[] = [];
  let latestReplay: Extract<ReasoningReplay, { provider: "openai-chat" }> | undefined;
  if (systemPrompt) {
    result.push({ role: "system", content: systemPrompt });
  }

  for (const message of messages) {
    if (message.role === "user") {
      result.push({ role: "user", content: message.content });
      continue;
    }

    if (message.role === "assistant") {
      const assistantMessage: ChatMessage = {
        role: "assistant",
        content: message.content || null
      };
      const messageReplay = openAIChatReplay(message);
      const replay = messageReplay ?? (message.toolCalls && message.toolCalls.length > 0 ? latestReplay : undefined);
      if (replay) {
        assistantMessage[replay.field] = replay.content;
      }
      if (message.toolCalls && message.toolCalls.length > 0) {
        assistantMessage.tool_calls = message.toolCalls.map((toolCall) => ({
          id: toolCall.id,
          type: "function",
          function: {
            name: toolCall.name,
            arguments: JSON.stringify(toolCall.arguments ?? {})
          }
        }));
      }
      result.push(assistantMessage);
      if (messageReplay) {
        latestReplay = { ...messageReplay };
      }
      continue;
    }

    result.push({
      role: "tool",
      tool_call_id: message.toolCallId,
      content: message.content
    });
  }

  return result;
}

function toChatTool(tool: LlmToolSpec): ChatTool {
  return {
    type: "function",
    function: {
      name: tool.name,
      description: tool.description,
      parameters: tool.parameters,
      strict: false
    }
  };
}

function assistantFromBody(body: ChatCompletionsBody): AssistantMessage {
  const message = body.choices?.[0]?.message;
  const reasoning = firstReasoningField(message ?? {});
  return {
    role: "assistant",
    content: message?.content ?? "",
    reasoning: reasoning ? reasoningFromOpenAIChat(reasoning.content, reasoning.field) : undefined,
    toolCalls: message?.tool_calls ? chatToolCallsToToolCalls(message.tool_calls) : undefined,
    usage: usageFromChat(body.usage),
    raw: body
  };
}

function reasoningFromOpenAIChat(content: string, field: OpenAIChatReasoningField | undefined): AssistantMessage["reasoning"] {
  if (!content) {
    return undefined;
  }
  return {
    summary: content,
    replay: [
      {
        provider: "openai-chat",
        field: field ?? "reasoning_content",
        content
      }
    ]
  };
}

function openAIChatReplay(message: AssistantMessage): Extract<ReasoningReplay, { provider: "openai-chat" }> | undefined {
  return message.reasoning?.replay?.find(
    (item): item is Extract<ReasoningReplay, { provider: "openai-chat" }> => item.provider === "openai-chat" && item.content.length > 0
  );
}

function usageFromChat(usage: ChatUsage | null | undefined): TokenUsage | undefined {
  if (!usage) {
    return undefined;
  }
  return cleanTokenUsage({
    inputTokens: usage.prompt_tokens,
    outputTokens: usage.completion_tokens,
    totalTokens: usage.total_tokens,
    cacheReadInputTokens: usage.prompt_tokens_details?.cached_tokens
  });
}

function chatToolCallsToToolCalls(toolCalls: ChatToolCall[]): ToolCall[] {
  return toolCalls.map((toolCall) => ({
    id: toolCall.id,
    name: toolCall.function.name,
    arguments: parseJsonObject(toolCall.function.arguments)
  }));
}

function pendingToolCallsToToolCalls(toolCalls: PendingToolCall[]): ToolCall[] | undefined {
  if (toolCalls.length === 0) {
    return undefined;
  }
  return toolCalls.map((toolCall) => ({
    id: toolCall.id,
    name: toolCall.name || "unknown_tool",
    arguments: parseJsonObject(toolCall.argumentsText || "{}")
  }));
}

function resolveToolCall(
  deltaToolCall: { index?: number; id?: string },
  byIndex: Map<number, PendingToolCall>,
  byId: Map<string, PendingToolCall>
): PendingToolCall {
  const index = deltaToolCall.index ?? byIndex.size;
  const fromIndex = byIndex.get(index);
  if (fromIndex) {
    return fromIndex;
  }
  const fromId = deltaToolCall.id ? byId.get(deltaToolCall.id) : undefined;
  if (fromId) {
    byIndex.set(index, fromId);
    return fromId;
  }

  const toolCall = {
    id: deltaToolCall.id ?? `call_${index}`,
    name: "",
    argumentsText: ""
  };
  byIndex.set(index, toolCall);
  if (deltaToolCall.id) {
    byId.set(deltaToolCall.id, toolCall);
  }
  return toolCall;
}

function firstReasoningField(record: Record<string, unknown>): { field: OpenAIChatReasoningField; content: string } | undefined {
  for (const key of OPENAI_CHAT_REASONING_FIELDS) {
    const value = record[key];
    if (typeof value === "string" && value.length > 0) {
      return { field: key, content: value };
    }
  }
  return undefined;
}

function parseJsonObject(value: string): unknown {
  try {
    const parsed = JSON.parse(value);
    return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? parsed : {};
  } catch {
    return {};
  }
}

function trimTrailingSlash(value: string): string {
  return value.replace(/\/+$/, "");
}
