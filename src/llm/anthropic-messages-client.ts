import type {
  AgentMessage,
  AssistantMessage,
  ReasoningOptions,
  ReasoningReplay,
  TokenUsage,
  ToolCall,
  ToolResultMessage
} from "../types.js";
import type { LlmProviderFactory, LlmRequest, LlmStreamEvent, LlmToolSpec, StreamingLlmClient } from "./types.js";
import { parseServerSentEvents } from "./sse.js";
import { cleanTokenUsage, mergeTokenUsage } from "./usage.js";

type AnthropicMessagesClientOptions = {
  apiKey?: string;
  baseURL?: string;
  defaultModel?: string;
  defaultReasoning?: ReasoningOptions | false;
  defaultMaxTokens?: number;
  fetchImpl?: typeof fetch;
};

type AnthropicMessage = {
  role: "user" | "assistant";
  content: string | AnthropicContentBlock[];
};

type AnthropicContentBlock =
  | { type: "text"; text: string }
  | { type: "thinking"; thinking: string; signature?: string }
  | { type: "redacted_thinking"; data: string }
  | { type: "tool_use"; id: string; name: string; input: unknown }
  | { type: "tool_result"; tool_use_id: string; content: string; is_error?: boolean };

type AnthropicTool = {
  name: string;
  description: string;
  input_schema: Record<string, unknown>;
};

type AnthropicBody = {
  content?: Array<{
    type?: string;
    text?: string;
    thinking?: string;
    signature?: string;
    data?: string;
    id?: string;
    name?: string;
    input?: unknown;
  }>;
  usage?: AnthropicUsage;
  error?: { message?: string };
};

type AnthropicUsage = {
  input_tokens?: number | null;
  output_tokens?: number | null;
  cache_read_input_tokens?: number | null;
  cache_creation_input_tokens?: number | null;
};

type AnthropicStreamEvent = {
  type?: string;
  index?: number;
  message?: {
    usage?: AnthropicUsage;
  };
  content_block?: {
    type?: string;
    text?: string;
    thinking?: string;
    signature?: string;
    data?: string;
    id?: string;
    name?: string;
    input?: unknown;
  };
  delta?: {
    type?: string;
    text?: string;
    thinking?: string;
    signature?: string;
    data?: string;
    partial_json?: string;
  };
  usage?: AnthropicUsage;
};

type PendingToolCall = {
  id: string;
  name: string;
  argumentsText: string;
};

type PendingThinkingBlock =
  | { type: "thinking"; thinking: string; signature?: string }
  | { type: "redacted_thinking"; data: string };

export class AnthropicMessagesClient implements StreamingLlmClient {
  private readonly apiKey: string | undefined;
  private readonly baseURL: string;
  private readonly defaultModel: string;
  private readonly defaultReasoning: ReasoningOptions | false | undefined;
  private readonly defaultMaxTokens: number;
  private readonly fetchImpl: typeof fetch;

  constructor(options: AnthropicMessagesClientOptions = {}) {
    this.apiKey = options.apiKey ?? process.env.ANTHROPIC_API_KEY;
    this.baseURL = trimTrailingSlash(options.baseURL ?? process.env.ANTHROPIC_BASE_URL ?? "https://api.anthropic.com/v1");
    this.defaultModel = options.defaultModel ?? process.env.ANTHROPIC_MODEL ?? "claude-3-5-sonnet-latest";
    this.defaultReasoning = options.defaultReasoning;
    this.defaultMaxTokens = options.defaultMaxTokens ?? 20000;
    this.fetchImpl = options.fetchImpl ?? fetch;
  }

  async complete(request: LlmRequest): Promise<AssistantMessage> {
    const apiKey = this.requireApiKey();
    const response = await this.createMessage(request, apiKey, false);
    const body = (await response.json().catch(() => ({}))) as AnthropicBody;
    if (!response.ok) {
      throw new Error(body.error?.message ?? `Anthropic Messages request failed with HTTP ${response.status}`);
    }
    return assistantFromBody(body);
  }

  async *stream(request: LlmRequest): AsyncIterable<LlmStreamEvent> {
    const apiKey = this.requireApiKey();
    const response = await this.createMessage(request, apiKey, true);
    if (!response.ok) {
      const body = (await response.json().catch(() => ({}))) as AnthropicBody;
      throw new Error(body.error?.message ?? `Anthropic Messages stream failed with HTTP ${response.status}`);
    }
    if (!response.body) {
      throw new Error("Anthropic Messages stream did not include a response body.");
    }

    let content = "";
    const thinkingBlocksByIndex = new Map<number, PendingThinkingBlock>();
    let completed = false;
    let usage: TokenUsage | undefined;
    const rawEvents: AnthropicStreamEvent[] = [];
    const toolCallsByIndex = new Map<number, PendingToolCall>();

    for await (const rawEvent of parseServerSentEvents(response.body)) {
      const event = rawEvent as AnthropicStreamEvent;
      rawEvents.push(event);
      usage = mergeTokenUsage(usage, usageFromAnthropic(event.message?.usage), usageFromAnthropic(event.usage));

      if (event.type === "content_block_start" && event.content_block?.type === "tool_use") {
        const index = event.index ?? toolCallsByIndex.size;
        const input = event.content_block.input ?? {};
        const argumentsText = JSON.stringify(input);
        const toolCall = {
          id: event.content_block.id ?? `toolu_${index}`,
          name: event.content_block.name ?? "unknown_tool",
          argumentsText: argumentsText === "{}" ? "" : argumentsText
        };
        toolCallsByIndex.set(index, toolCall);
        if (toolCall.argumentsText) {
          yield {
            type: "tool_call_delta",
            toolCallId: toolCall.id,
            toolName: toolCall.name,
            delta: toolCall.argumentsText,
            argumentsText: toolCall.argumentsText
          };
        }
        continue;
      }

      if (event.type === "content_block_start" && isThinkingBlockType(event.content_block?.type)) {
        const index = event.index ?? thinkingBlocksByIndex.size;
        if (event.content_block.type === "thinking") {
          const block: PendingThinkingBlock = {
            type: "thinking",
            thinking: event.content_block.thinking ?? "",
            ...(event.content_block.signature ? { signature: event.content_block.signature } : {})
          };
          thinkingBlocksByIndex.set(index, block);
          if (block.thinking) {
            yield { type: "thinking_delta", delta: block.thinking };
          }
        } else {
          thinkingBlocksByIndex.set(index, {
            type: "redacted_thinking",
            data: event.content_block.data ?? ""
          });
        }
        continue;
      }

      if (event.type === "content_block_delta" && event.delta?.type === "text_delta" && event.delta.text) {
        content += event.delta.text;
        yield { type: "text_delta", delta: event.delta.text };
        continue;
      }

      if (event.type === "content_block_delta" && event.delta?.type === "thinking_delta" && event.delta.thinking) {
        const block = getOrCreateThinkingBlock(thinkingBlocksByIndex, event.index);
        block.thinking += event.delta.thinking;
        yield { type: "thinking_delta", delta: event.delta.thinking };
        continue;
      }

      if (event.type === "content_block_delta" && event.delta?.type === "signature_delta" && event.delta.signature) {
        const block = getOrCreateThinkingBlock(thinkingBlocksByIndex, event.index);
        block.signature = `${block.signature ?? ""}${event.delta.signature}`;
        continue;
      }

      if (event.type === "content_block_delta" && event.delta?.type === "redacted_thinking_delta" && event.delta.data) {
        const block = getOrCreateRedactedThinkingBlock(thinkingBlocksByIndex, event.index);
        block.data += event.delta.data;
        continue;
      }

      if (event.type === "content_block_delta" && event.delta?.type === "input_json_delta") {
        const index = event.index ?? 0;
        const toolCall = toolCallsByIndex.get(index);
        if (!toolCall) {
          continue;
        }
        const delta = event.delta.partial_json ?? "";
        toolCall.argumentsText += delta;
        yield {
          type: "tool_call_delta",
          toolCallId: toolCall.id,
          toolName: toolCall.name,
          delta,
          argumentsText: toolCall.argumentsText
        };
        continue;
      }

      if (event.type === "message_stop") {
        completed = true;
        yield {
          type: "done",
          message: assistantFromStream(content, [...thinkingBlocksByIndex.values()], [...toolCallsByIndex.values()], usage, rawEvents)
        };
      }
    }

    if (!completed) {
      yield {
        type: "done",
        message: assistantFromStream(content, [...thinkingBlocksByIndex.values()], [...toolCallsByIndex.values()], usage, rawEvents)
      };
    }
  }

  private requireApiKey(): string {
    if (!this.apiKey) {
      throw new Error("Missing Anthropic API key. Pass apiKey, set LLM_API_KEY, or set ANTHROPIC_API_KEY.");
    }
    return this.apiKey;
  }

  private async createMessage(request: LlmRequest, apiKey: string, stream: boolean): Promise<Response> {
    const reasoning = request.reasoning !== undefined ? request.reasoning : this.defaultReasoning;
    const messages = toAnthropicMessages(request.messages);
    const body: Record<string, unknown> = {
      model: request.model || this.defaultModel,
      system: request.systemPrompt,
      messages,
      tools: request.tools && request.tools.length > 0 ? request.tools.map(toAnthropicTool) : undefined,
      max_tokens: this.defaultMaxTokens,
      stream
    };

    if (reasoning && reasoning.effort) {
      body.thinking = {
        type: "enabled",
        budget_tokens: 1024
      };
    }

    return this.fetchImpl(`${this.baseURL}/messages`, {
      method: "POST",
      headers: {
        "x-api-key": apiKey,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json"
      },
      body: JSON.stringify(body),
      signal: request.signal
    });
  }
}

export const anthropicMessagesProvider: LlmProviderFactory = {
  id: "anthropic",
  create: (options) => new AnthropicMessagesClient(options)
};

function toAnthropicMessages(messages: AgentMessage[]): AnthropicMessage[] {
  const result: AnthropicMessage[] = [];
  let latestReplayBlocks: AnthropicContentBlock[] = [];

  for (let index = 0; index < messages.length; index += 1) {
    const message = messages[index];
    if (message.role === "user") {
      if (message.content.trim()) {
        result.push({ role: "user", content: message.content });
      }
      continue;
    }

    if (message.role === "assistant") {
      const blocks: AnthropicContentBlock[] = [];
      const replayBlocks = anthropicReplayBlocks(message);
      const replayPrefix = replayBlocks.length > 0 ? replayBlocks : message.toolCalls && message.toolCalls.length > 0 ? latestReplayBlocks : [];
      blocks.push(...replayPrefix.map((block) => ({ ...block })));
      if (message.content.trim()) {
        blocks.push({ type: "text", text: message.content });
      }
      for (const toolCall of message.toolCalls ?? []) {
        blocks.push({
          type: "tool_use",
          id: toolCall.id,
          name: toolCall.name,
          input: toolCall.arguments ?? {}
        });
      }
      if (blocks.length > 0) {
        result.push({ role: "assistant", content: blocks });
      }
      if (replayBlocks.length > 0) {
        latestReplayBlocks = replayBlocks.map((block) => ({ ...block }));
      }
      continue;
    }

    const toolResults: AnthropicContentBlock[] = [];
    for (; index < messages.length && messages[index]?.role === "tool"; index += 1) {
      toolResults.push(toToolResultBlock(messages[index] as ToolResultMessage));
    }
    index -= 1;
    if (toolResults.length > 0) {
      result.push({ role: "user", content: toolResults });
    }
  }

  return result;
}

function toToolResultBlock(message: ToolResultMessage): AnthropicContentBlock {
  return {
    type: "tool_result",
    tool_use_id: message.toolCallId,
    content: message.content,
    is_error: message.isError
  };
}

function toAnthropicTool(tool: LlmToolSpec): AnthropicTool {
  return {
    name: tool.name,
    description: tool.description,
    input_schema: tool.parameters
  };
}

function assistantFromBody(body: AnthropicBody): AssistantMessage {
  const contentParts: string[] = [];
  const thinkingBlocks: PendingThinkingBlock[] = [];
  const toolCalls: ToolCall[] = [];

  for (const block of body.content ?? []) {
    if (block.type === "text") {
      contentParts.push(block.text ?? "");
    }
    if (block.type === "thinking") {
      const thinkingBlock: PendingThinkingBlock = {
        type: "thinking",
        thinking: block.thinking ?? "",
        ...(block.signature ? { signature: block.signature } : {})
      };
      thinkingBlocks.push(thinkingBlock);
    }
    if (block.type === "redacted_thinking") {
      thinkingBlocks.push({
        type: "redacted_thinking",
        data: block.data ?? ""
      });
    }
    if (block.type === "tool_use") {
      toolCalls.push({
        id: block.id ?? crypto.randomUUID(),
        name: block.name ?? "unknown_tool",
        arguments: block.input ?? {}
      });
    }
  }

  return {
    role: "assistant",
    content: contentParts.join(""),
    reasoning: reasoningFromAnthropicBlocks(thinkingBlocks),
    toolCalls: toolCalls.length > 0 ? toolCalls : undefined,
    usage: usageFromAnthropic(body.usage),
    raw: body
  };
}

function assistantFromStream(
  content: string,
  thinkingBlocks: PendingThinkingBlock[],
  pendingToolCalls: PendingToolCall[],
  usage: TokenUsage | undefined,
  rawEvents: AnthropicStreamEvent[]
): AssistantMessage {
  const toolCalls = pendingToolCalls.map((toolCall) => ({
    id: toolCall.id,
    name: toolCall.name,
    arguments: parseJsonObject(toolCall.argumentsText || "{}")
  }));

  return {
    role: "assistant",
    content,
    reasoning: reasoningFromAnthropicBlocks(thinkingBlocks),
    toolCalls: toolCalls.length > 0 ? toolCalls : undefined,
    usage,
    raw: rawEvents
  };
}

function reasoningFromAnthropicBlocks(thinkingBlocks: PendingThinkingBlock[]): AssistantMessage["reasoning"] {
  const replayBlocks = normalizeAnthropicReplayBlocks(thinkingBlocks);
  const summary = replayBlocks
    .filter((block): block is Extract<PendingThinkingBlock, { type: "thinking" }> => block.type === "thinking" && block.thinking.length > 0)
    .map((block) => block.thinking)
    .join("\n\n");
  if (!summary && replayBlocks.length === 0) {
    return undefined;
  }
  return {
    ...(summary ? { summary } : {}),
    ...(replayBlocks.length > 0 ? { replay: [{ provider: "anthropic", blocks: replayBlocks }] } : {})
  };
}

function normalizeAnthropicReplayBlocks(blocks: PendingThinkingBlock[]): PendingThinkingBlock[] {
  return blocks
    .map((block) => {
      if (block.type === "thinking") {
        return {
          type: "thinking" as const,
          thinking: block.thinking,
          ...(block.signature ? { signature: block.signature } : {})
        };
      }
      return {
        type: "redacted_thinking" as const,
        data: block.data
      };
    })
    .filter((block) => (block.type === "thinking" ? block.thinking.length > 0 : block.data.length > 0));
}

function anthropicReplayBlocks(message: AssistantMessage): AnthropicContentBlock[] {
  const replay = message.reasoning?.replay?.find((item): item is Extract<ReasoningReplay, { provider: "anthropic" }> => item.provider === "anthropic");
  return replay?.blocks.map((block) => ({ ...block })) ?? [];
}

function isThinkingBlockType(type: string | undefined): type is "thinking" | "redacted_thinking" {
  return type === "thinking" || type === "redacted_thinking";
}

function getOrCreateThinkingBlock(blocks: Map<number, PendingThinkingBlock>, index: number | undefined): Extract<PendingThinkingBlock, { type: "thinking" }> {
  const key = index ?? blocks.size;
  const existing = blocks.get(key);
  if (existing?.type === "thinking") {
    return existing;
  }
  const block: PendingThinkingBlock = { type: "thinking", thinking: "" };
  blocks.set(key, block);
  return block;
}

function getOrCreateRedactedThinkingBlock(
  blocks: Map<number, PendingThinkingBlock>,
  index: number | undefined
): Extract<PendingThinkingBlock, { type: "redacted_thinking" }> {
  const key = index ?? blocks.size;
  const existing = blocks.get(key);
  if (existing?.type === "redacted_thinking") {
    return existing;
  }
  const block: PendingThinkingBlock = { type: "redacted_thinking", data: "" };
  blocks.set(key, block);
  return block;
}

function usageFromAnthropic(usage: AnthropicUsage | undefined): TokenUsage | undefined {
  if (!usage) {
    return undefined;
  }
  const baseInputTokens = tokenPart(usage.input_tokens);
  const cacheReadInputTokens = tokenPart(usage.cache_read_input_tokens);
  const cacheCreationInputTokens = tokenPart(usage.cache_creation_input_tokens);
  const inputTokens = sumTokenParts(baseInputTokens, cacheReadInputTokens, cacheCreationInputTokens);
  const outputTokens = tokenPart(usage.output_tokens);
  return cleanTokenUsage({
    inputTokens,
    outputTokens,
    totalTokens: inputTokens !== undefined && outputTokens !== undefined ? inputTokens + outputTokens : undefined,
    cacheReadInputTokens,
    cacheCreationInputTokens
  });
}

function tokenPart(value: number | null | undefined): number | undefined {
  return typeof value === "number" && Number.isFinite(value) && value >= 0 ? Math.ceil(value) : undefined;
}

function sumTokenParts(...values: Array<number | undefined>): number | undefined {
  let total = 0;
  let found = false;
  for (const value of values) {
    if (value === undefined) {
      continue;
    }
    total += value;
    found = true;
  }
  return found ? total : undefined;
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
