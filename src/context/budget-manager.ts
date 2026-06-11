import type {
  ContextEngineOptions,
  DynamicCompressionOptions,
  DynamicCompressionState,
  ResolvedContextEngineOptions,
  ResolvedDynamicCompressionOptions
} from "./types.js";

export const DEFAULT_CONTEXT_ENGINE_OPTIONS: ResolvedContextEngineOptions = {
  enabled: true,
  contextWindowTokens: 256000,
  compactionThresholdRatio: 0.9,
  reservedOutputTokens: 16000,
  keepRecentTokens: 20000,
  maxToolResultTokens: 20000,
  summarizeHistory: true,
  compressionModel: undefined,
  compressionReasoning: undefined,
  dynamicCompression: false
};

export class BudgetManager {
  constructor(private readonly options: ResolvedContextEngineOptions = DEFAULT_CONTEXT_ENGINE_OPTIONS) {}

  shouldCompact(usedTokens: number): boolean {
    return usedTokens > this.compactionTriggerTokens;
  }

  get availableInputTokens(): number {
    return Math.max(0, this.options.contextWindowTokens - this.options.reservedOutputTokens);
  }

  get compactionTriggerTokens(): number {
    const thresholdTokens = Math.floor(this.options.contextWindowTokens * this.options.compactionThresholdRatio);
    return Math.max(0, Math.min(thresholdTokens, this.availableInputTokens));
  }
}

export function resolveContextEngineOptions(options: ContextEngineOptions = {}): ResolvedContextEngineOptions {
  const base = {
    ...DEFAULT_CONTEXT_ENGINE_OPTIONS,
    ...options,
    dynamicCompression: resolveDynamicCompressionOptions(options.dynamicCompression, options.keepRecentTokens ?? DEFAULT_CONTEXT_ENGINE_OPTIONS.keepRecentTokens)
  };

  return base;
}

export function mergeContextEngineOptions(
  agentOptions?: false | ContextEngineOptions,
  runOptions?: false | Partial<ContextEngineOptions>
): false | ResolvedContextEngineOptions {
  if (runOptions === false) {
    return false;
  }
  if (agentOptions === false && runOptions === undefined) {
    return false;
  }

  const merged = resolveContextEngineOptions({
    ...(agentOptions === false ? {} : agentOptions),
    ...(runOptions ?? {})
  });
  return merged.enabled ? merged : false;
}

function resolveDynamicCompressionOptions(
  options: false | DynamicCompressionOptions | undefined,
  defaultKeepRecentTokens: number
): false | ResolvedDynamicCompressionOptions {
  if (options === false || options?.enabled === false) {
    return false;
  }

  if (!options) {
    return false;
  }

  return {
    enabled: true,
    keepRecentTokens: options.keepRecentTokens ?? defaultKeepRecentTokens,
    minCompressMessages: options.minCompressMessages ?? 4,
    preserveUserMessages: options.preserveUserMessages ?? false,
    exposeTool: options.exposeTool ?? true,
    autoSummarize: options.autoSummarize ?? false,
    triggerTokens: options.triggerTokens,
    triggerRatio: options.triggerRatio ?? 0.5,
    summaryPrompt: options.summaryPrompt ?? DYNAMIC_COMPRESSION_SUMMARY_PROMPT,
    state: options.state ?? createDynamicCompressionState()
  };
}

export function createDynamicCompressionState(): DynamicCompressionState {
  return {
    nextBlockId: 1,
    blocks: []
  };
}

export const DYNAMIC_COMPRESSION_SUMMARY_PROMPT = `You are maintaining a dynamic context compression block for a coding agent.

Summarize the selected older conversation so future turns can continue without the raw messages.

Include:
- User goals, constraints, corrections, and acceptance criteria.
- Important decisions, implementation details, file paths, commands, outputs, and unresolved issues.
- Tool results or errors that matter for future work.
- What remains active versus what is already closed.

Be dense and precise. Do not call tools. Return only the summary.`;
