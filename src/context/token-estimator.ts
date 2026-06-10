import type { AgentMessage, ToolCall } from "../types.js";
import type { LlmRequest, LlmToolSpec } from "../llm/types.js";

export type TokenEstimateSource = "heuristic" | "provider_usage";

export type RequestTokenEstimate = {
  systemPromptTokens: number;
  messageTokens: number;
  toolTokens: number;
  totalTokens: number;
  source: TokenEstimateSource;
  heuristicTotalTokens: number;
  providerInputTokens?: number;
  providerOutputTokens?: number;
  appendedMessageTokens?: number;
};

export type RequestTokenEstimateOptions = {
  useProviderUsage?: boolean;
};

export function estimateTextTokens(text: string | undefined): number {
  if (!text) {
    return 0;
  }
  return Math.ceil(text.length / 4);
}

export function estimateJsonTokens(value: unknown): number {
  return estimateTextTokens(safeStringify(value));
}

export function estimateToolCallTokens(toolCall: ToolCall): number {
  return estimateTextTokens(toolCall.id) + estimateTextTokens(toolCall.name) + estimateJsonTokens(toolCall.arguments);
}

export function estimateMessageTokens(message: AgentMessage): number {
  if (message.role === "user") {
    return estimateTextTokens(message.role) + estimateTextTokens(message.content);
  }

  if (message.role === "assistant") {
    const toolCallTokens = (message.toolCalls ?? []).reduce((sum, toolCall) => sum + estimateToolCallTokens(toolCall), 0);
    return (
      estimateTextTokens(message.role) +
      estimateTextTokens(message.content) +
      estimateTextTokens(message.reasoning?.summary) +
      toolCallTokens
    );
  }

  return (
    estimateTextTokens(message.role) +
    estimateTextTokens(message.toolCallId) +
    estimateTextTokens(message.toolName) +
    estimateTextTokens(message.content)
  );
}

export function estimateMessagesTokens(messages: readonly AgentMessage[]): number {
  return messages.reduce((sum, message) => sum + estimateMessageTokens(message), 0);
}

export function estimateToolSpecTokens(tool: LlmToolSpec): number {
  return estimateTextTokens(tool.name) + estimateTextTokens(tool.description) + estimateJsonTokens(tool.parameters);
}

export function estimateToolSpecsTokens(tools: readonly LlmToolSpec[] | undefined): number {
  return (tools ?? []).reduce((sum, tool) => sum + estimateToolSpecTokens(tool), 0);
}

export function estimateRequestTokens(request: LlmRequest, options: RequestTokenEstimateOptions = {}): RequestTokenEstimate {
  const systemPromptTokens = estimateTextTokens(request.systemPrompt);
  const messageTokens = estimateMessagesTokens(request.messages);
  const toolTokens = estimateToolSpecsTokens(request.tools);
  const heuristicTotalTokens = systemPromptTokens + messageTokens + toolTokens;
  const heuristicEstimate = {
    systemPromptTokens,
    messageTokens,
    toolTokens,
    totalTokens: heuristicTotalTokens,
    source: "heuristic" as const,
    heuristicTotalTokens
  };

  if (options.useProviderUsage === false) {
    return heuristicEstimate;
  }

  const providerEstimate = estimateFromProviderUsage(request.messages);
  if (!providerEstimate) {
    return heuristicEstimate;
  }

  return {
    ...heuristicEstimate,
    totalTokens: providerEstimate.totalTokens,
    source: "provider_usage",
    providerInputTokens: providerEstimate.providerInputTokens,
    providerOutputTokens: providerEstimate.providerOutputTokens,
    appendedMessageTokens: providerEstimate.appendedMessageTokens
  };
}

function estimateFromProviderUsage(messages: readonly AgentMessage[]):
  | { totalTokens: number; providerInputTokens: number; providerOutputTokens?: number; appendedMessageTokens: number }
  | undefined {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    if (message?.role !== "assistant") {
      continue;
    }

    const inputTokens = normalizeTokenCount(message.usage?.inputTokens);
    if (inputTokens === undefined || message.context?.requestCompacted) {
      continue;
    }

    const providerOutputTokens = normalizeTokenCount(message.usage?.outputTokens);
    const assistantTokens = providerOutputTokens ?? estimateMessageTokens(message);
    const appendedMessageTokens = assistantTokens + estimateMessagesTokens(messages.slice(index + 1));
    return {
      totalTokens: inputTokens + appendedMessageTokens,
      providerInputTokens: inputTokens,
      providerOutputTokens,
      appendedMessageTokens
    };
  }

  return undefined;
}

function normalizeTokenCount(value: number | undefined): number | undefined {
  return typeof value === "number" && Number.isFinite(value) && value >= 0 ? Math.ceil(value) : undefined;
}

function safeStringify(value: unknown): string {
  if (value === undefined) {
    return "";
  }
  try {
    return JSON.stringify(value) ?? "";
  } catch {
    return String(value);
  }
}
