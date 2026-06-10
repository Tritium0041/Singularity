import type { TokenUsage } from "../types.js";

export function tokenCount(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) && value >= 0 ? Math.ceil(value) : undefined;
}

export function cleanTokenUsage(usage: TokenUsage): TokenUsage | undefined {
  const cleaned: TokenUsage = {
    inputTokens: tokenCount(usage.inputTokens),
    outputTokens: tokenCount(usage.outputTokens),
    totalTokens: tokenCount(usage.totalTokens),
    cacheReadInputTokens: tokenCount(usage.cacheReadInputTokens),
    cacheCreationInputTokens: tokenCount(usage.cacheCreationInputTokens)
  };
  return hasTokenUsage(cleaned) ? cleaned : undefined;
}

export function mergeTokenUsage(...usages: Array<TokenUsage | undefined>): TokenUsage | undefined {
  const merged: TokenUsage = {};
  for (const usage of usages) {
    if (!usage) {
      continue;
    }
    merged.inputTokens = tokenCount(usage.inputTokens) ?? merged.inputTokens;
    merged.outputTokens = tokenCount(usage.outputTokens) ?? merged.outputTokens;
    merged.totalTokens = tokenCount(usage.totalTokens) ?? merged.totalTokens;
    merged.cacheReadInputTokens = tokenCount(usage.cacheReadInputTokens) ?? merged.cacheReadInputTokens;
    merged.cacheCreationInputTokens = tokenCount(usage.cacheCreationInputTokens) ?? merged.cacheCreationInputTokens;
  }
  const computedTotal = computeTotalTokens(merged);
  if (computedTotal !== undefined) {
    merged.totalTokens = computedTotal;
  }
  return hasTokenUsage(merged) ? merged : undefined;
}

function hasTokenUsage(usage: TokenUsage): boolean {
  return Object.values(usage).some((value) => value !== undefined);
}

function computeTotalTokens(usage: TokenUsage): number | undefined {
  if (usage.inputTokens === undefined || usage.outputTokens === undefined) {
    return undefined;
  }
  return usage.inputTokens + usage.outputTokens;
}
