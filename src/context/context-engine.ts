import type { AgentMessage, AssistantMessage, ContextCompactionMetadata, RequestTokenEstimateMetadata } from "../types.js";
import type { LlmRequest } from "../llm/types.js";
import { BudgetManager, resolveContextEngineOptions } from "./budget-manager.js";
import { DynamicCompressor } from "./dynamic-compressor.js";
import { CONTEXT_HANDOFF_SUMMARY_PROMPT, HistoryCompressor } from "./history-compressor.js";
import { PromptBuilder } from "./prompt-builder.js";
import { estimateRequestTokens, estimateTextTokens, type RequestTokenEstimate } from "./token-estimator.js";
import type { ContextEngineOptions, ContextSummarySource, ResolvedContextEngineOptions } from "./types.js";

export type ContextSummaryPurpose = "handoff" | "dynamic";
export type SummaryModel = (request: LlmRequest, purpose?: ContextSummaryPurpose) => Promise<AssistantMessage>;

export type PreparedContextRequest = {
  request: LlmRequest;
  historyReplacement?: AgentMessage[];
};

export class ContextEngine {
  private readonly options: ResolvedContextEngineOptions;
  private readonly budget: BudgetManager;
  private readonly compressor: HistoryCompressor;
  private readonly dynamicCompressor: DynamicCompressor;
  private readonly promptBuilder = new PromptBuilder();

  constructor(options: ContextEngineOptions = {}) {
    this.options = resolveContextEngineOptions(options);
    this.budget = new BudgetManager(this.options);
    this.compressor = new HistoryCompressor(this.options);
    this.dynamicCompressor = new DynamicCompressor(this.options);
  }

  prepare(request: LlmRequest): LlmRequest {
    return this.prepareSync(request).request;
  }

  async prepareWithHandoff(request: LlmRequest, summarize?: SummaryModel): Promise<PreparedContextRequest> {
    return summarize ? this.prepareInternal(request, summarize) : this.prepareInternal(request);
  }

  async compactWithHandoff(request: LlmRequest, summarize?: SummaryModel): Promise<PreparedContextRequest> {
    return this.compactInternal(request, summarize, "manual");
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

    const requestWithTruncatedTools = this.buildRequestView(request);
    const baseEstimate = estimateRequestTokens(requestWithTruncatedTools);
    const dynamicResult = this.dynamicCompressor.prepare(requestWithTruncatedTools, baseEstimate, summarize);

    if (dynamicResult instanceof Promise) {
      return dynamicResult.then((result) => this.finishPreparedRequest(result.request, result.applied, summarize, "automatic"));
    }

    return this.finishPreparedRequest(dynamicResult.request, dynamicResult.applied, summarize, "automatic");
  }

  private async compactInternal(
    request: LlmRequest,
    summarize: SummaryModel | undefined,
    mode: ContextCompactionMetadata["mode"]
  ): Promise<PreparedContextRequest> {
    if (!this.options.enabled) {
      return { request };
    }

    const requestWithTruncatedTools = this.buildRequestView(request);
    const estimate = estimateRequestTokens(requestWithTruncatedTools);
    return this.compactRequestView(requestWithTruncatedTools, estimate, summarize, mode);
  }

  private buildRequestView(request: LlmRequest): LlmRequest {
    return {
      ...request,
      systemPrompt: this.promptBuilder.buildSystemPrompt(request.systemPrompt),
      messages: this.compressor.truncateToolResults(request.messages)
    };
  }

  private finishPreparedRequest(
    requestView: LlmRequest,
    dynamicallyCompressed: boolean,
    summarize: SummaryModel | undefined,
    mode: ContextCompactionMetadata["mode"]
  ): PreparedContextRequest | Promise<PreparedContextRequest> {
    const estimate = estimateRequestTokens(requestView, { useProviderUsage: !dynamicallyCompressed });
    if (!this.budget.shouldCompact(estimate.totalTokens)) {
      return { request: withContextMetadata(requestView, dynamicallyCompressed, estimate) };
    }

    return this.compactRequestView(requestView, estimate, summarize, mode);
  }

  private compactRequestView(
    requestWithTruncatedTools: LlmRequest,
    estimate: RequestTokenEstimate,
    summarize: SummaryModel | undefined,
    mode: ContextCompactionMetadata["mode"]
  ): PreparedContextRequest | Promise<PreparedContextRequest> {
    if (summarize && this.options.summarizeHistory) {
      return this.prepareWithModelSummary(requestWithTruncatedTools, estimate, summarize, mode);
    }

    const compactedRequest: LlmRequest = {
      ...requestWithTruncatedTools,
      messages: this.compressor.compact(requestWithTruncatedTools.messages)
    };
    const compactedEstimate = estimateRequestTokens(compactedRequest, { useProviderUsage: false });
    const compaction = buildCompactionMetadata({
      mode,
      summarySource: "heuristic",
      requestBefore: requestWithTruncatedTools,
      requestAfter: compactedRequest,
      decisionEstimate: estimate,
      compactedEstimate
    });
    return {
      request: withContextMetadata(compactedRequest, true, compactedEstimate, estimate, "heuristic", compaction)
    };
  }

  private async prepareWithModelSummary(
    request: LlmRequest,
    compactionDecisionEstimate: ReturnType<typeof estimateRequestTokens>,
    summarize: SummaryModel,
    mode: ContextCompactionMetadata["mode"]
  ): Promise<PreparedContextRequest> {
    const compactionPlan = this.compressor.planCompaction(request.messages);
    const summaryRequest = this.buildSummaryRequest(request);
    const summaryRequestEstimate = estimateRequestTokens(summaryRequest);
    const summaryMessage = await summarize(summaryRequest, "handoff");
    const compactedMessages = this.compressor.compactWithHandoffSummary(request.messages, summaryMessage.content, {
      plan: compactionPlan
    });
    const compactedRequest: LlmRequest = {
      ...request,
      messages: compactedMessages
    };
    const compactedEstimate = estimateRequestTokens(compactedRequest, { useProviderUsage: false });
    const compaction = buildCompactionMetadata({
      mode,
      summarySource: "model",
      requestBefore: request,
      requestAfter: compactedRequest,
      decisionEstimate: compactionDecisionEstimate,
      compactedEstimate,
      summaryCall: {
        messageCount: summaryRequest.messages.length,
        model: summaryRequest.model,
        request: toEstimateMetadata(summaryRequestEstimate),
        responseUsage: summaryMessage.usage,
        summaryTokens: estimateTextTokens(summaryMessage.content),
        summaryChars: summaryMessage.content.length
      }
    });
    return {
      request: withContextMetadata(compactedRequest, true, compactedEstimate, compactionDecisionEstimate, "model", compaction),
      historyReplacement: mode === "manual" ? compactedMessages : undefined
    };
  }

  private buildSummaryRequest(request: LlmRequest): LlmRequest {
    return {
      ...request,
      model: this.options.compressionModel ?? request.model,
      reasoning: this.options.compressionReasoning ?? request.reasoning,
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
}

function withContextMetadata(
  request: LlmRequest,
  compacted: boolean,
  estimate: ReturnType<typeof estimateRequestTokens>,
  compactionDecisionEstimate?: ReturnType<typeof estimateRequestTokens>,
  summarySource?: ContextSummarySource,
  compaction?: ContextCompactionMetadata
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
        compactionSummarySource: summarySource,
        estimate: toEstimateMetadata(estimate),
        compaction
      }
    }
  };
}

function buildCompactionMetadata(options: {
  mode: ContextCompactionMetadata["mode"];
  summarySource: ContextSummarySource;
  requestBefore: LlmRequest;
  requestAfter: LlmRequest;
  decisionEstimate: RequestTokenEstimate;
  compactedEstimate: RequestTokenEstimate;
  summaryCall?: ContextCompactionMetadata["summaryCall"];
}): ContextCompactionMetadata {
  return {
    mode: options.mode,
    summarySource: options.summarySource,
    messageCountBefore: options.requestBefore.messages.length,
    messageCountAfter: options.requestAfter.messages.length,
    decision: toEstimateMetadata(options.decisionEstimate),
    compacted: toEstimateMetadata(options.compactedEstimate),
    summaryCall: options.summaryCall
  };
}

function toEstimateMetadata(estimate: RequestTokenEstimate): RequestTokenEstimateMetadata {
  return {
    systemPromptTokens: estimate.systemPromptTokens,
    messageTokens: estimate.messageTokens,
    toolTokens: estimate.toolTokens,
    totalTokens: estimate.totalTokens,
    source: estimate.source,
    heuristicTotalTokens: estimate.heuristicTotalTokens,
    providerInputTokens: estimate.providerInputTokens,
    providerOutputTokens: estimate.providerOutputTokens,
    appendedMessageTokens: estimate.appendedMessageTokens
  };
}
