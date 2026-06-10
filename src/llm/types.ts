import type { AgentMessage, AssistantMessage, JsonSchema, ReasoningOptions, RequestContextMetadata } from "../types.js";

export type LlmToolSpec = {
  name: string;
  description: string;
  parameters: JsonSchema;
};

export type LlmRequest = {
  model: string;
  systemPrompt?: string;
  messages: AgentMessage[];
  tools?: LlmToolSpec[];
  reasoning?: ReasoningOptions | false;
  signal?: AbortSignal;
  metadata?: LlmRequestMetadata;
};

export type LlmRequestMetadata = {
  context?: RequestContextMetadata;
};

export interface LlmClient {
  complete(request: LlmRequest): Promise<AssistantMessage>;
}

export type LlmStreamEvent =
  | { type: "thinking_delta"; delta: string }
  | { type: "text_delta"; delta: string }
  | { type: "tool_call_delta"; toolCallId: string; toolName: string; delta: string; argumentsText: string }
  | { type: "done"; message: AssistantMessage };

export interface StreamingLlmClient extends LlmClient {
  stream(request: LlmRequest): AsyncIterable<LlmStreamEvent>;
}

export function isStreamingLlmClient(client: LlmClient): client is StreamingLlmClient {
  return typeof (client as { stream?: unknown }).stream === "function";
}

export type LlmProviderFactoryOptions = {
  apiKey?: string;
  baseURL?: string;
  defaultModel?: string;
  defaultReasoning?: ReasoningOptions | false;
  fetchImpl?: typeof fetch;
};

export interface LlmProviderFactory {
  id: string;
  create(options?: LlmProviderFactoryOptions): LlmClient;
}
