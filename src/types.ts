import type { ContextEngineOptions, ContextSummarySource, DynamicCompressionMetadata } from "./context/types.js";
import type { MemoryEntry } from "./memory/types.js";

export type JsonSchema = {
  type?: string;
  properties?: Record<string, JsonSchema>;
  required?: string[];
  additionalProperties?: boolean;
  description?: string;
  enum?: unknown[];
  items?: JsonSchema;
  [key: string]: unknown;
};

export type AgentMessage = UserMessage | AssistantMessage | ToolResultMessage;

export type UserMessage = {
  role: "user";
  content: string;
};

export type AssistantMessage = {
  role: "assistant";
  content: string;
  toolCalls?: ToolCall[];
  reasoning?: ReasoningOutput;
  usage?: TokenUsage;
  context?: AssistantContextMetadata;
  raw?: unknown;
};

export type ToolResultMessage = {
  role: "tool";
  toolCallId: string;
  toolName: string;
  content: string;
  isError?: boolean;
  details?: unknown;
};

export type ToolCall = {
  id: string;
  name: string;
  arguments: unknown;
};

export type ReasoningEffort = "minimal" | "low" | "medium" | "high" | "xhigh";

export type ReasoningSummary = "auto" | "concise" | "detailed" | "none";

export type ReasoningOptions = {
  effort?: ReasoningEffort;
  summary?: ReasoningSummary;
  includeEncryptedContent?: boolean;
};

export type ReasoningOutput = {
  summary?: string;
};

export type TokenUsage = {
  inputTokens?: number;
  outputTokens?: number;
  totalTokens?: number;
  cacheReadInputTokens?: number;
  cacheCreationInputTokens?: number;
};

export type RequestTokenEstimateMetadata = {
  systemPromptTokens: number;
  messageTokens: number;
  toolTokens: number;
  totalTokens: number;
  source: "heuristic" | "provider_usage";
  heuristicTotalTokens: number;
  providerInputTokens?: number;
  providerOutputTokens?: number;
  appendedMessageTokens?: number;
};

export type ContextCompactionMetadata = {
  mode: "automatic" | "manual";
  summarySource: ContextSummarySource;
  messageCountBefore: number;
  messageCountAfter: number;
  decision: RequestTokenEstimateMetadata;
  compacted: RequestTokenEstimateMetadata;
  summaryCall?: {
    messageCount: number;
    model?: string;
    request: RequestTokenEstimateMetadata;
    responseUsage?: TokenUsage;
    summaryTokens: number;
    summaryChars: number;
  };
};

export type RequestContextMetadata = {
  compacted?: boolean;
  estimatedInputTokens?: number;
  tokenEstimateSource?: "heuristic" | "provider_usage";
  compactionDecisionEstimatedInputTokens?: number;
  compactionDecisionTokenEstimateSource?: "heuristic" | "provider_usage";
  compactionSummarySource?: "heuristic" | "model";
  estimate?: RequestTokenEstimateMetadata;
  compaction?: ContextCompactionMetadata;
  dynamicCompression?: DynamicCompressionMetadata;
};

export type AssistantContextMetadata = {
  requestCompacted?: boolean;
};

export type AgentEvent =
  | { type: "agent_start"; input: string }
  | { type: "turn_start"; turn: number }
  | { type: "thinking_delta"; turn: number; delta: string; content: string }
  | { type: "assistant_delta"; turn: number; delta: string; content: string }
  | { type: "tool_call_delta"; turn: number; toolCallId: string; toolName: string; delta: string; argumentsText: string }
  | { type: "message"; message: AgentMessage }
  | { type: "tool_start"; turn: number; toolCall: ToolCall }
  | { type: "tool_end"; turn: number; toolCall: ToolCall; result: ToolResultMessage }
  | {
      type: "turn_end";
      turn: number;
      message: AssistantMessage;
      toolResults: ToolResultMessage[];
      context?: RequestContextMetadata;
    }
  | { type: "memory_summary_start"; model: string; messageCount: number; storePath: string }
  | {
      type: "memory_summary_end";
      model: string;
      storePath: string;
      entry: MemoryEntry;
      action: "created" | "updated";
      summaryTokens: number;
      summaryChars: number;
      usage?: TokenUsage;
    }
  | { type: "memory_summary_error"; model: string; storePath: string; error: string }
  | { type: "agent_end"; result: AgentRunResult };

export type AgentEventSink = (event: AgentEvent) => void | Promise<void>;

export type AgentRunResult = {
  output: string;
  messages: AgentMessage[];
  turns: number;
  stoppedBy: "final" | "max_turns";
};

export type AgentRunOptions = {
  maxTurns?: number;
  reasoning?: ReasoningOptions | false;
  context?: false | Partial<ContextEngineOptions>;
  signal?: AbortSignal;
};

export type AgentCompactOptions = {
  reasoning?: ReasoningOptions | false;
  context?: false | Partial<ContextEngineOptions>;
  signal?: AbortSignal;
};

export type AgentCompactResult = {
  compacted: boolean;
  messages: AgentMessage[];
  context?: RequestContextMetadata;
};
