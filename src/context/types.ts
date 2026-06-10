export type ContextEngineOptions = {
  enabled?: boolean;
  contextWindowTokens?: number;
  compactionThresholdRatio?: number;
  reservedOutputTokens?: number;
  keepRecentTokens?: number;
  maxToolResultTokens?: number;
  summarizeHistory?: boolean;
};

export type ResolvedContextEngineOptions = Required<ContextEngineOptions>;

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
