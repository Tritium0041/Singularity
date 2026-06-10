import type { AgentMessage, AssistantMessage } from "../types.js";
import type { LlmRequest } from "../llm/types.js";
import { BudgetManager, resolveContextEngineOptions } from "./budget-manager.js";
import { CONTEXT_HANDOFF_SUMMARY_PROMPT, HistoryCompressor } from "./history-compressor.js";
import { PromptBuilder } from "./prompt-builder.js";
import { estimateRequestTokens } from "./token-estimator.js";
import type { ContextEngineOptions, ContextSummarySource, ResolvedContextEngineOptions } from "./types.js";

export type SummaryModel = (request: LlmRequest) => Promise<AssistantMessage>;

export type PreparedContextRequest = {
  request: LlmRequest;
  historyReplacement?: AgentMessage[];
};

export class ContextEngine {
  private readonly options: ResolvedContextEngineOptions;
  private readonly budget: BudgetManager;
  private readonly compressor: HistoryCompressor;
  private readonly promptBuilder = new PromptBuilder();

  constructor(options: ContextEngineOptions = {}) {
    this.options = resolveContextEngineOptions(options);
    this.budget = new BudgetManager(this.options);
    this.compressor = new HistoryCompressor(this.options);
  }

  prepare(request: LlmRequest): LlmRequest {
    return this.prepareSync(request).request;
  }

  async prepareWithHandoff(request: LlmRequest, summarize?: SummaryModel): Promise<PreparedContextRequest> {
    return summarize ? this.prepareInternal(request, summarize) : this.prepareInternal(request);
  }

  private prepareSync(request: LlmRequest): PreparedContextRequest {
    return this.prepareInternal(request);
  }

  private prepareInternal(request: LlmRequest, summarize: SummaryModel): Promise<PreparedContextRequest>;
  private prepareInternal(request: LlmRequest, summarize?: undefined): PreparedContextRequest;
  private prepareInternal(
    request: LlmRequest,
    summarize?: SummaryModel
  ): PreparedContextRequest | Promise<PreparedContextRequest> {
    if (!this.options.enabled) {
      return { request };
    }

    const requestWithTruncatedTools: LlmRequest = {
      ...request,
      systemPrompt: this.promptBuilder.buildSystemPrompt(request.systemPrompt),
      messages: this.compressor.truncateToolResults(request.messages)
    };

    const estimate = estimateRequestTokens(requestWithTruncatedTools);
    if (!this.budget.shouldCompact(estimate.totalTokens)) {
      return { request: withContextMetadata(requestWithTruncatedTools, false, estimate) };
    }

    if (summarize && this.options.summarizeHistory) {
      return this.prepareWithModelSummary(requestWithTruncatedTools, estimate, summarize);
    }

    const compactedRequest: LlmRequest = {
      ...requestWithTruncatedTools,
      messages: this.compressor.compact(requestWithTruncatedTools.messages)
    };
    const compactedEstimate = estimateRequestTokens(compactedRequest, { useProviderUsage: false });
    return {
      request: withContextMetadata(compactedRequest, true, compactedEstimate, estimate, "heuristic")
    };
  }

  private async prepareWithModelSummary(
    request: LlmRequest,
    compactionDecisionEstimate: ReturnType<typeof estimateRequestTokens>,
    summarize: SummaryModel
  ): Promise<PreparedContextRequest> {
    const summaryRequest = buildSummaryRequest(request);
    const summaryMessage = await summarize(summaryRequest);
    const compactedMessages = this.compressor.compactWithHandoffSummary(request.messages, summaryMessage.content);
    const compactedRequest: LlmRequest = {
      ...request,
      messages: compactedMessages
    };
    const compactedEstimate = estimateRequestTokens(compactedRequest, { useProviderUsage: false });
    return {
      request: withContextMetadata(compactedRequest, true, compactedEstimate, compactionDecisionEstimate, "model"),
      historyReplacement: compactedMessages
    };
  }
}

function buildSummaryRequest(request: LlmRequest): LlmRequest {
  return {
    ...request,
    messages: [
      ...request.messages,
      {
        role: "user",
        content: CONTEXT_HANDOFF_SUMMARY_PROMPT
      }
    ],
    tools: []
  };
}

function withContextMetadata(
  request: LlmRequest,
  compacted: boolean,
  estimate: ReturnType<typeof estimateRequestTokens>,
  compactionDecisionEstimate?: ReturnType<typeof estimateRequestTokens>,
  summarySource?: ContextSummarySource
): LlmRequest {
  return {
    ...request,
    metadata: {
      ...request.metadata,
      context: {
        ...request.metadata?.context,
        compacted,
        estimatedInputTokens: estimate.totalTokens,
        tokenEstimateSource: estimate.source,
        compactionDecisionEstimatedInputTokens: compactionDecisionEstimate?.totalTokens,
        compactionDecisionTokenEstimateSource: compactionDecisionEstimate?.source,
        compactionSummarySource: summarySource
      }
    }
  };
}
