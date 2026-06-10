import type { ContextEngineOptions, ResolvedContextEngineOptions } from "./types.js";

export const DEFAULT_CONTEXT_ENGINE_OPTIONS: ResolvedContextEngineOptions = {
  enabled: true,
  contextWindowTokens: 256000,
  compactionThresholdRatio: 0.9,
  reservedOutputTokens: 16000,
  keepRecentTokens: 20000,
  maxToolResultTokens: 4000,
  maxHandoffUserMessageTokens: 20000,
  summarizeHistory: true
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
  return {
    ...DEFAULT_CONTEXT_ENGINE_OPTIONS,
    ...options
  };
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
