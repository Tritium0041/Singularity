import type { AgentMessage, AssistantMessage, ReasoningOptions, ToolCall, ToolResultMessage } from "../types.js";
import type { LlmProviderFactory, LlmRequest, LlmStreamEvent, LlmToolSpec, StreamingLlmClient } from "./types.js";
import { parseServerSentEvents } from "./sse.js";

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
    id?: string;
    name?: string;
    input?: unknown;
  }>;
  error?: { message?: string };
};

type AnthropicStreamEvent = {
  type?: string;
  index?: number;
  content_block?: {
    type?: string;
    text?: string;
    thinking?: string;
    id?: string;
    name?: string;
    input?: unknown;
  };
  delta?: {
    type?: string;
    text?: string;
    thinking?: string;
    partial_json?: string;
  };
};

type PendingToolCall = {
  id: string;
  name: string;
  argumentsText: string;
};

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
    this.defaultMaxTokens = options.defaultMaxTokens ?? 4096;
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
    let thinkingContent = "";
    let completed = false;
    const toolCallsByIndex = new Map<number, PendingToolCall>();

    for await (const rawEvent of parseServerSentEvents(response.body)) {
      const event = rawEvent as AnthropicStreamEvent;

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

      if (event.type === "content_block_delta" && event.delta?.type === "text_delta" && event.delta.text) {
        content += event.delta.text;
        yield { type: "text_delta", delta: event.delta.text };
        continue;
      }

      if (event.type === "content_block_delta" && event.delta?.type === "thinking_delta" && event.delta.thinking) {
        thinkingContent += event.delta.thinking;
        yield { type: "thinking_delta", delta: event.delta.thinking };
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
        yield { type: "done", message: assistantFromStream(content, thinkingContent, [...toolCallsByIndex.values()]) };
      }
    }

    if (!completed) {
      yield { type: "done", message: assistantFromStream(content, thinkingContent, [...toolCallsByIndex.values()]) };
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
    const body: Record<string, unknown> = {
      model: request.model || this.defaultModel,
      system: request.systemPrompt,
      messages: toAnthropicMessages(request.messages),
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
  const thinkingParts: string[] = [];
  const toolCalls: ToolCall[] = [];

  for (const block of body.content ?? []) {
    if (block.type === "text") {
      contentParts.push(block.text ?? "");
    }
    if (block.type === "thinking") {
      thinkingParts.push(block.thinking ?? "");
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
    reasoning: thinkingParts.length > 0 ? { summary: thinkingParts.join("\n\n") } : undefined,
    toolCalls: toolCalls.length > 0 ? toolCalls : undefined,
    raw: body
  };
}

function assistantFromStream(content: string, thinkingContent: string, pendingToolCalls: PendingToolCall[]): AssistantMessage {
  const toolCalls = pendingToolCalls.map((toolCall) => ({
    id: toolCall.id,
    name: toolCall.name,
    arguments: parseJsonObject(toolCall.argumentsText || "{}")
  }));

  return {
    role: "assistant",
    content,
    reasoning: thinkingContent ? { summary: thinkingContent } : undefined,
    toolCalls: toolCalls.length > 0 ? toolCalls : undefined,
    raw: undefined
  };
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
