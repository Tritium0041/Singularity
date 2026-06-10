import type { AgentMessage, AssistantMessage, ReasoningEffort, ReasoningOptions, TokenUsage, ToolCall } from "../types.js";
import type { LlmProviderFactory, LlmRequest, LlmStreamEvent, LlmToolSpec, StreamingLlmClient } from "./types.js";
import { cleanTokenUsage } from "./usage.js";

type OpenAIResponsesClientOptions = {
  apiKey?: string;
  baseURL?: string;
  defaultModel?: string;
  defaultReasoning?: ReasoningOptions | false;
  fetchImpl?: typeof fetch;
};

type ResponsesInputItem =
  | {
      type: "message";
      role: "user" | "assistant";
      content: Array<{ type: "input_text" | "output_text"; text: string }>;
    }
  | {
      type: "function_call";
      call_id: string;
      name: string;
      arguments: string;
    }
  | {
      type: "function_call_output";
      call_id: string;
      output: string;
    };

type ResponsesTool = {
  type: "function";
  name: string;
  description: string;
  parameters: Record<string, unknown>;
  strict: false;
};

type ResponsesOutputItem = {
  id?: string;
  type?: string;
  role?: string;
  name?: string;
  call_id?: string;
  arguments?: string;
  summary?: Array<{ type?: string; text?: string }>;
  content?: Array<{ type?: string; text?: string; refusal?: string }>;
  encrypted_content?: string;
};

type ResponsesReasoning = {
  effort?: ReasoningEffort;
  summary?: "auto" | "concise" | "detailed";
};

type ResolvedReasoning = {
  payload: ResponsesReasoning;
  includeEncryptedContent: boolean;
};

type ResponsesBody = {
  output?: ResponsesOutputItem[];
  usage?: ResponsesUsage;
  error?: { message?: string };
};

type ResponsesUsage = {
  input_tokens?: number;
  output_tokens?: number;
  total_tokens?: number;
  input_tokens_details?: {
    cached_tokens?: number;
  };
};

type ResponseStreamEvent = {
  type?: string;
  delta?: string;
  arguments?: string;
  part?: { type?: string; text?: string };
  item_id?: string;
  call_id?: string;
  item?: ResponsesOutputItem;
  response?: ResponsesBody & { output?: ResponsesOutputItem[] };
};

type PendingToolCall = {
  id: string;
  itemId?: string;
  name: string;
  argumentsText: string;
};

export class OpenAIResponsesClient implements StreamingLlmClient {
  private readonly apiKey: string | undefined;
  private readonly baseURL: string;
  private readonly defaultModel: string;
  private readonly defaultReasoning: ReasoningOptions | false | undefined;
  private readonly fetchImpl: typeof fetch;

  constructor(options: OpenAIResponsesClientOptions = {}) {
    this.apiKey = options.apiKey ?? process.env.OPENAI_API_KEY;
    this.baseURL = trimTrailingSlash(options.baseURL ?? process.env.OPENAI_BASE_URL ?? "https://api.openai.com/v1");
    this.defaultModel = options.defaultModel ?? process.env.OPENAI_MODEL ?? "gpt-4.1-mini";
    this.defaultReasoning = options.defaultReasoning ?? defaultReasoningFromEnv();
    this.fetchImpl = options.fetchImpl ?? fetch;
  }

  async complete(request: LlmRequest): Promise<AssistantMessage> {
    const apiKey = this.apiKey;
    if (!apiKey) {
      throw new Error("Missing OPENAI_API_KEY. Pass apiKey or set the environment variable.");
    }

    const response = await this.createResponse(request, apiKey, false);

    const body = (await response.json().catch(() => ({}))) as ResponsesBody;
    if (!response.ok) {
      throw new Error(body.error?.message ?? `OpenAI Responses request failed with HTTP ${response.status}`);
    }

    return fromResponsesBody(body);
  }

  async *stream(request: LlmRequest): AsyncIterable<LlmStreamEvent> {
    const apiKey = this.apiKey;
    if (!apiKey) {
      throw new Error("Missing OPENAI_API_KEY. Pass apiKey or set the environment variable.");
    }

    const response = await this.createResponse(request, apiKey, true);
    if (!response.ok) {
      const body = (await response.json().catch(() => ({}))) as ResponsesBody;
      throw new Error(body.error?.message ?? `OpenAI Responses stream failed with HTTP ${response.status}`);
    }
    if (!response.body) {
      throw new Error("OpenAI Responses stream did not include a response body.");
    }

    let content = "";
    let thinkingContent = "";
    let completed = false;
    const rawEvents: ResponseStreamEvent[] = [];
    const toolCalls = new Map<string, PendingToolCall>();
    let activeToolCallKey: string | undefined;

    for await (const rawEvent of parseServerSentEvents(response.body)) {
      const event = rawEvent as ResponseStreamEvent;
      rawEvents.push(event);

      if (event.type === "response.output_text.delta" && typeof event.delta === "string") {
        content += event.delta;
        yield { type: "text_delta", delta: event.delta };
        continue;
      }

      if (
        (event.type === "response.reasoning_summary_text.delta" || event.type === "response.reasoning_text.delta") &&
        typeof event.delta === "string"
      ) {
        thinkingContent += event.delta;
        yield { type: "thinking_delta", delta: event.delta };
        continue;
      }

      if (event.type === "response.reasoning_summary_part.done" && thinkingContent) {
        thinkingContent += "\n\n";
        yield { type: "thinking_delta", delta: "\n\n" };
        continue;
      }

      if (event.type === "response.output_item.added" && event.item?.type === "function_call") {
        const toolCall = pendingToolCallFromItem(event.item);
        toolCalls.set(toolCall.id, toolCall);
        activeToolCallKey = toolCall.id;
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

      if (event.type === "response.function_call_arguments.delta" && typeof event.delta === "string") {
        const toolCall = resolvePendingToolCall(toolCalls, activeToolCallKey, event);
        if (!toolCall) {
          continue;
        }
        toolCall.argumentsText += event.delta;
        yield {
          type: "tool_call_delta",
          toolCallId: toolCall.id,
          toolName: toolCall.name,
          delta: event.delta,
          argumentsText: toolCall.argumentsText
        };
        continue;
      }

      if (event.type === "response.function_call_arguments.done") {
        const toolCall = resolvePendingToolCall(toolCalls, activeToolCallKey, event);
        if (toolCall && typeof event.arguments === "string") {
          toolCall.argumentsText = event.arguments;
        }
        continue;
      }

      if (event.type === "response.output_item.done" && event.item?.type === "function_call") {
        const toolCall = pendingToolCallFromItem(event.item);
        toolCalls.set(toolCall.id, toolCall);
        activeToolCallKey = undefined;
        continue;
      }

      if (event.type === "response.completed") {
        completed = true;
        const responseBody = event.response;
        const message =
          responseBody?.output && responseBody.output.length > 0
            ? fromResponsesBody(responseBody)
            : assistantMessageFromStream(content, thinkingContent, [...toolCalls.values()], rawEvents, responseBody?.usage);
        yield { type: "done", message };
      }
    }

    if (!completed) {
      yield {
        type: "done",
        message: assistantMessageFromStream(content, thinkingContent, [...toolCalls.values()], rawEvents)
      };
    }
  }

  private async createResponse(request: LlmRequest, apiKey: string, stream: boolean): Promise<Response> {
    const reasoning = resolveReasoning(request.reasoning, this.defaultReasoning);
    return this.fetchImpl(`${this.baseURL}/responses`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        model: request.model || this.defaultModel,
        instructions: request.systemPrompt,
        input: toResponsesInput(request.messages),
        tools: request.tools?.map(toResponsesTool),
        tool_choice: "auto",
        parallel_tool_calls: true,
        ...(reasoning ? { reasoning: reasoning.payload } : {}),
        ...(reasoning?.includeEncryptedContent ? { include: ["reasoning.encrypted_content"] } : {}),
        stream
      }),
      signal: request.signal
    });
  }
}

export const openAIResponsesProvider: LlmProviderFactory = {
  id: "openai-responses",
  create: (options) => new OpenAIResponsesClient(options)
};

function toResponsesInput(messages: AgentMessage[]): ResponsesInputItem[] {
  const input: ResponsesInputItem[] = [];

  for (const message of messages) {
    if (message.role === "user") {
      input.push({
        type: "message",
        role: "user",
        content: [{ type: "input_text", text: message.content }]
      });
      continue;
    }

    if (message.role === "assistant") {
      if (message.content) {
        input.push({
          type: "message",
          role: "assistant",
          content: [{ type: "output_text", text: message.content }]
        });
      }
      for (const toolCall of message.toolCalls ?? []) {
        input.push({
          type: "function_call",
          call_id: toolCall.id,
          name: toolCall.name,
          arguments: JSON.stringify(toolCall.arguments ?? {})
        });
      }
      continue;
    }

    input.push({
      type: "function_call_output",
      call_id: message.toolCallId,
      output: message.content
    });
  }

  return input;
}

function toResponsesTool(tool: LlmToolSpec): ResponsesTool {
  return {
    type: "function",
    name: tool.name,
    description: tool.description,
    parameters: tool.parameters,
    strict: false
  };
}

function fromResponsesBody(body: ResponsesBody): AssistantMessage {
  const contentParts: string[] = [];
  const reasoningParts: string[] = [];
  const toolCalls: ToolCall[] = [];

  for (const item of body.output ?? []) {
    if (item.type === "reasoning") {
      reasoningParts.push(...reasoningTextFromItem(item));
    }

    if (item.type === "message" && item.role === "assistant") {
      for (const content of item.content ?? []) {
        if (content.type === "output_text" || content.type === "refusal") {
          contentParts.push(content.text ?? content.refusal ?? "");
        }
      }
    }

    if (item.type === "function_call") {
      toolCalls.push({
        id: item.call_id ?? item.id ?? crypto.randomUUID(),
        name: item.name ?? "unknown_tool",
        arguments: parseJsonObject(item.arguments ?? "{}")
      });
    }
  }

  return {
    role: "assistant",
    content: contentParts.join(""),
    reasoning: reasoningParts.length > 0 ? { summary: reasoningParts.join("\n\n") } : undefined,
    toolCalls: toolCalls.length > 0 ? toolCalls : undefined,
    usage: usageFromResponses(body.usage),
    raw: body
  };
}

async function* parseServerSentEvents(body: ReadableStream<Uint8Array>): AsyncIterable<unknown> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }

    buffer += decoder.decode(value, { stream: true });
    yield* drainSseFrames(buffer, (remaining) => {
      buffer = remaining;
    });
  }

  buffer += decoder.decode();
  if (buffer.trim()) {
    yield* parseSseFrame(buffer);
  }
}

function* drainSseFrames(buffer: string, setRemaining: (remaining: string) => void): Iterable<unknown> {
  const normalized = buffer.replace(/\r\n/g, "\n");
  const parts = normalized.split("\n\n");
  setRemaining(parts.pop() ?? "");
  for (const part of parts) {
    yield* parseSseFrame(part);
  }
}

function* parseSseFrame(frame: string): Iterable<unknown> {
  const data = frame
    .split("\n")
    .filter((line) => line.startsWith("data:"))
    .map((line) => line.slice("data:".length).trimStart())
    .join("\n")
    .trim();

  if (!data || data === "[DONE]") {
    return;
  }

  yield JSON.parse(data) as unknown;
}

function pendingToolCallFromItem(item: ResponsesOutputItem): PendingToolCall {
  return {
    id: item.call_id ?? item.id ?? crypto.randomUUID(),
    itemId: item.id,
    name: item.name ?? "unknown_tool",
    argumentsText: item.arguments ?? ""
  };
}

function resolvePendingToolCall(
  toolCalls: Map<string, PendingToolCall>,
  activeToolCallKey: string | undefined,
  event: ResponseStreamEvent
): PendingToolCall | undefined {
  if (event.call_id && toolCalls.has(event.call_id)) {
    return toolCalls.get(event.call_id);
  }
  if (event.item_id) {
    for (const toolCall of toolCalls.values()) {
      if (toolCall.itemId === event.item_id) {
        return toolCall;
      }
    }
  }
  return activeToolCallKey ? toolCalls.get(activeToolCallKey) : undefined;
}

function assistantMessageFromStream(
  content: string,
  reasoningSummary: string,
  pendingToolCalls: PendingToolCall[],
  rawEvents: ResponseStreamEvent[],
  usage?: ResponsesUsage
): AssistantMessage {
  const toolCalls = pendingToolCalls.map((toolCall) => ({
    id: toolCall.id,
    name: toolCall.name,
    arguments: parseJsonObject(toolCall.argumentsText || "{}")
  }));

  return {
    role: "assistant",
    content,
    reasoning: reasoningSummary ? { summary: reasoningSummary.trimEnd() } : undefined,
    toolCalls: toolCalls.length > 0 ? toolCalls : undefined,
    usage: usageFromResponses(usage),
    raw: rawEvents
  };
}

function usageFromResponses(usage: ResponsesUsage | undefined): TokenUsage | undefined {
  if (!usage) {
    return undefined;
  }
  return cleanTokenUsage({
    inputTokens: usage.input_tokens,
    outputTokens: usage.output_tokens,
    totalTokens: usage.total_tokens,
    cacheReadInputTokens: usage.input_tokens_details?.cached_tokens
  });
}

function reasoningTextFromItem(item: ResponsesOutputItem): string[] {
  const summary = (item.summary ?? [])
    .map((part) => part.text ?? "")
    .filter((text) => text.length > 0);
  const content = (item.content ?? [])
    .map((part) => part.text ?? "")
    .filter((text) => text.length > 0);
  return summary.length > 0 ? summary : content;
}

function resolveReasoning(
  requestReasoning: ReasoningOptions | false | undefined,
  defaultReasoning: ReasoningOptions | false | undefined
): ResolvedReasoning | undefined {
  const reasoning = requestReasoning !== undefined ? requestReasoning : defaultReasoning;
  if (reasoning === false || reasoning === undefined) {
    return undefined;
  }

  return {
    payload: {
      effort: reasoning.effort ?? "medium",
      ...(reasoning.summary === "none" ? {} : { summary: reasoning.summary ?? "auto" })
    },
    includeEncryptedContent: reasoning.includeEncryptedContent ?? true
  };
}

function defaultReasoningFromEnv(): ReasoningOptions | false | undefined {
  const mode = process.env.OPENAI_REASONING?.toLowerCase();
  if (mode === "off" || mode === "false" || mode === "0" || mode === "none") {
    return false;
  }

  const effort = parseReasoningEffort(process.env.OPENAI_REASONING_EFFORT);
  const summary = parseReasoningSummary(process.env.OPENAI_REASONING_SUMMARY);
  const includeEncryptedContent = parseOptionalBoolean(process.env.OPENAI_REASONING_INCLUDE_ENCRYPTED_CONTENT);

  if (mode === "on" || mode === "true" || mode === "1" || effort || summary || includeEncryptedContent !== undefined) {
    return {
      effort,
      summary,
      includeEncryptedContent
    };
  }

  return undefined;
}

function parseReasoningEffort(value: string | undefined): ReasoningEffort | undefined {
  if (!value) {
    return undefined;
  }
  const normalized = value.toLowerCase();
  if (normalized === "minimal" || normalized === "low" || normalized === "medium" || normalized === "high" || normalized === "xhigh") {
    return normalized;
  }
  throw new Error(`Invalid OPENAI_REASONING_EFFORT: ${value}`);
}

function parseReasoningSummary(value: string | undefined): ReasoningOptions["summary"] | undefined {
  if (!value) {
    return undefined;
  }
  const normalized = value.toLowerCase();
  if (normalized === "auto" || normalized === "concise" || normalized === "detailed" || normalized === "none") {
    return normalized;
  }
  throw new Error(`Invalid OPENAI_REASONING_SUMMARY: ${value}`);
}

function parseOptionalBoolean(value: string | undefined): boolean | undefined {
  if (!value) {
    return undefined;
  }
  const normalized = value.toLowerCase();
  if (normalized === "true" || normalized === "1" || normalized === "on") {
    return true;
  }
  if (normalized === "false" || normalized === "0" || normalized === "off") {
    return false;
  }
  throw new Error(`Invalid OPENAI_REASONING_INCLUDE_ENCRYPTED_CONTENT: ${value}`);
}

function parseJsonObject(value: string): unknown {
  try {
    return JSON.parse(value);
  } catch {
    return value;
  }
}

function trimTrailingSlash(value: string): string {
  return value.replace(/\/+$/, "");
}
