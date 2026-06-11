import type { LlmRequest } from "../llm/types.js";
import type { ReasoningOptions, RequestTokenEstimateMetadata, TokenUsage } from "../types.js";

export type DynamicCompressionOptions = {
  enabled?: boolean;
  triggerTokens?: number;
  triggerRatio?: number;
  keepRecentTokens?: number;
  minCompressMessages?: number;
  preserveUserMessages?: boolean;
  exposeTool?: boolean;
  autoSummarize?: boolean;
  summaryPrompt?: string;
  state?: DynamicCompressionState;
};

export type ResolvedDynamicCompressionOptions = Required<
  Omit<DynamicCompressionOptions, "state" | "summaryPrompt" | "triggerTokens" | "triggerRatio">
> & {
  triggerTokens?: number;
  triggerRatio?: number;
  summaryPrompt: string;
  state: DynamicCompressionState;
};

export type DynamicCompressionBlock = {
  id: number;
  ref?: string;
  active?: boolean;
  startIndex: number;
  endIndex: number;
  startId?: string;
  endId?: string;
  anchorMessageId?: string;
  messageCount: number;
  coveredMessageIds?: string[];
  coveredToolCallIds?: string[];
  protectedSnippets?: string[];
  consumedBlockIds?: number[];
  parentBlockIds?: number[];
  messageFingerprints: string[];
  summary: string;
  summaryTokens: number;
  createdAt: number;
  model?: string;
  topic?: string;
  source?: "auto" | "tool" | "worker";
  deactivatedByBlockId?: number;
  deactivatedByUser?: boolean;
};

export type DynamicCompressionMessageRef = {
  id: string;
  index: number;
  role: "user" | "assistant" | "tool" | "summary";
  content: string;
  fingerprint: string;
  toolCallIds: string[];
  toolResultId?: string;
  protectedSnippets?: string[];
  kind?: "message" | "block";
  blockId?: number;
  rawStartIndex?: number;
  rawEndIndex?: number;
  rawFingerprints?: string[];
  coveredMessageIds?: string[];
};

export type DynamicCompressionState = {
  nextBlockId: number;
  activeBlock?: DynamicCompressionBlock;
  blocks: DynamicCompressionBlock[];
  messageRefs?: DynamicCompressionMessageRef[];
  workerBaseRequest?: LlmRequest;
  lastTriggerTokens?: number;
};

export type DynamicCompressionMetadata = {
  applied: boolean;
  generated: boolean;
  protocol?: {
    toolName: string;
    messageRefCount: number;
    activeBlockCount: number;
    nudge: boolean;
    triggerTokens: number;
  };
  blockId?: number;
  model?: string;
  messageCountBefore: number;
  messageCountAfter: number;
  coveredMessageCount?: number;
  decision: RequestTokenEstimateMetadata;
  compressed: RequestTokenEstimateMetadata;
  summaryCall?: {
    messageCount: number;
    model: string;
    request: RequestTokenEstimateMetadata;
    responseUsage?: TokenUsage;
    summaryTokens: number;
    summaryChars: number;
  };
};

export type ContextEngineOptions = {
  enabled?: boolean;
  contextWindowTokens?: number;
  compactionThresholdRatio?: number;
  reservedOutputTokens?: number;
  keepRecentTokens?: number;
  maxToolResultTokens?: number;
  summarizeHistory?: boolean;
  compressionModel?: string;
  compressionReasoning?: ReasoningOptions | false;
  dynamicCompression?: false | DynamicCompressionOptions;
};

export type ResolvedContextEngineOptions = Required<
  Omit<ContextEngineOptions, "compressionModel" | "compressionReasoning" | "dynamicCompression">
> & {
  compressionModel?: string;
  compressionReasoning?: ReasoningOptions | false;
  dynamicCompression: false | ResolvedDynamicCompressionOptions;
};

export type PromptFragment = {
  id: string;
  content: string;
  stable?: boolean;
};

export type PromptBackgroundTool = {
  name: string;
  description?: string;
};

export type ContextSummarySource = "heuristic" | "model";

export type SystemPromptBackgroundOptions = {
  cwd?: string;
  currentDate?: string;
  timezone?: string;
  shell?: string;
  tools?: PromptBackgroundTool[];
  includeCwd?: boolean;
  includeCurrentDate?: boolean;
  includeTimezone?: boolean;
  includeShell?: boolean;
  includeTools?: boolean;
  includeToolDescriptions?: boolean;
  extra?: string | string[] | PromptFragment[];
};

export type SystemPromptBuilderOptions = {
  basePrompt?: string;
  defaultInstructions?: false | string;
  background?: false | SystemPromptBackgroundOptions;
  fragments?: readonly PromptFragment[];
};
