import type { AgentMessage } from "../types.js";
import type { LlmRequest } from "../llm/types.js";
import {
  estimateMessagesTokens,
  estimateRequestTokens,
  estimateTextTokens,
  type RequestTokenEstimate
} from "./token-estimator.js";
import type {
  DynamicCompressionBlock,
  DynamicCompressionMetadata,
  DynamicCompressionMessageRef,
  DynamicCompressionState,
  ResolvedContextEngineOptions,
  ResolvedDynamicCompressionOptions
} from "./types.js";
import type { SummaryModel } from "./context-engine.js";

export const DYNAMIC_COMPRESSION_SUMMARY_PREFIX = "Dynamic context summary:";
export const DYNAMIC_COMPRESSION_TOOL_NAME = "compact_context";
export const DYNAMIC_COMPRESSION_MESSAGE_ID_PREFIX = "m";
export const DYNAMIC_COMPRESSION_BLOCK_ID_PREFIX = "b";

export type DynamicCompressionCompactToolArgs = {
  goal?: string;
  targetTokenBudget?: number;
  maxBlocks?: number;
};

export type DynamicCompressionBlockCandidate = {
  startId: string;
  endId: string;
  summary: string;
  topic?: string;
};

export type DynamicCompressionWorkerResult = {
  block: DynamicCompressionBlock;
  content: string;
};

type MessageTurn = {
  startIndex: number;
  endIndex: number;
  messages: AgentMessage[];
};

type DynamicCompressionResult = {
  request: LlmRequest;
  metadata?: DynamicCompressionMetadata;
  applied: boolean;
};

export class DynamicCompressor {
  constructor(private readonly options: ResolvedContextEngineOptions) {}

  prepare(
    request: LlmRequest,
    decisionEstimate: RequestTokenEstimate,
    summarize?: SummaryModel
  ): DynamicCompressionResult | Promise<DynamicCompressionResult> {
    const dynamicOptions = this.options.dynamicCompression;
    if (!dynamicOptions) {
      return { request, applied: false };
    }

    const threshold = getDynamicTriggerTokens(this.options, dynamicOptions);
    dynamicOptions.state.lastTriggerTokens = threshold;
    const shouldNudge = decisionEstimate.totalTokens >= threshold;
    const projected = buildRequestFromActiveBlocks(request, dynamicOptions, decisionEstimate, false);
    const protocolRequest = injectCompressionProtocol(projected.request, dynamicOptions, threshold, shouldNudge);
    dynamicOptions.state.workerBaseRequest = cloneRequestForWorkerBase(protocolRequest);

    if (!dynamicOptions.autoSummarize) {
      return {
        ...projected,
        request: protocolRequest,
        metadata: protocolRequest.metadata?.context?.dynamicCompression,
        applied: projected.applied
      };
    }

    const reusableBlock = getReusableActiveBlock(dynamicOptions, request.messages);
    const shouldGenerate = shouldNudge;

    if (!shouldGenerate) {
      return {
        ...projected,
        request: protocolRequest,
        metadata: protocolRequest.metadata?.context?.dynamicCompression,
        applied: projected.applied
      };
    }

    const selection = selectCompressionPrefix(request.messages, dynamicOptions);
    if (!selection || selection.messageCount < dynamicOptions.minCompressMessages) {
      return {
        ...projected,
        request: protocolRequest,
        metadata: protocolRequest.metadata?.context?.dynamicCompression,
        applied: projected.applied
      };
    }

    if (reusableBlock && reusableBlock.endIndex >= selection.endIndex) {
      return buildRequestFromBlock(request, reusableBlock, dynamicOptions, decisionEstimate, false);
    }

    if (!summarize) {
      return {
        ...projected,
        request: protocolRequest,
        metadata: protocolRequest.metadata?.context?.dynamicCompression,
        applied: projected.applied
      };
    }

    return this.generateBlock(request, decisionEstimate, dynamicOptions, selection, reusableBlock, summarize);
  }

  private async generateBlock(
    request: LlmRequest,
    decisionEstimate: RequestTokenEstimate,
    dynamicOptions: ResolvedDynamicCompressionOptions,
    selection: CompressionSelection,
    previousBlock: DynamicCompressionBlock | undefined,
    summarize: SummaryModel
  ): Promise<DynamicCompressionResult> {
    const summaryRequest = buildDynamicSummaryRequest(request, dynamicOptions, this.options, selection, previousBlock);
    const summaryRequestEstimate = estimateRequestTokens(summaryRequest, { useProviderUsage: false });
    const summaryMessage = await summarize(summaryRequest, "dynamic");
    const block = allocateBlock(dynamicOptions, request.messages, selection, summaryMessage.content, summaryRequest.model);
    const result = buildRequestFromBlock(request, block, dynamicOptions, decisionEstimate, true);

    if (!result.metadata) {
      return result;
    }
    const summaryCall = {
      messageCount: summaryRequest.messages.length,
      model: summaryRequest.model,
      request: toEstimateMetadata(summaryRequestEstimate),
      responseUsage: summaryMessage.usage,
      summaryTokens: estimateTextTokens(summaryMessage.content),
      summaryChars: summaryMessage.content.length
    };
    const metadata = {
      ...result.metadata,
      summaryCall
    };

    return {
      ...result,
      request: {
        ...result.request,
        metadata: {
          ...result.request.metadata,
          context: {
            ...result.request.metadata?.context,
            dynamicCompression: metadata
          }
        }
      },
      metadata
    };
  }
}

type CompressionSelection = {
  endIndex: number;
  messageCount: number;
  messages: AgentMessage[];
};

export function applyDynamicCompressionBlock(
  state: DynamicCompressionState,
  args: DynamicCompressionBlockCandidate,
  source: DynamicCompressionBlock["source"] = "worker"
): { block: DynamicCompressionBlock; content: string } {
  const refs = state.messageRefs ?? [];
  if (refs.length === 0) {
    throw new Error("No dynamic compression message IDs are available for this request.");
  }

  const startRef = refs.find((ref) => ref.id === args.startId);
  const endRef = refs.find((ref) => ref.id === args.endId);
  if (!startRef) {
    throw new Error(`Unknown compression startId: ${args.startId}`);
  }
  if (!endRef) {
    throw new Error(`Unknown compression endId: ${args.endId}`);
  }
  if (startRef.index > endRef.index) {
    throw new Error(`Invalid compression range: ${args.startId} must come before ${args.endId}.`);
  }

  const selectedRefs = refs.filter((ref) => ref.index >= startRef.index && ref.index <= endRef.index);
  if (selectedRefs.length === 0) {
    throw new Error("Compression range did not select any messages.");
  }
  const rawStartIndex = Math.min(...selectedRefs.map(getRefRawStartIndex));
  const rawEndIndex = Math.max(...selectedRefs.map(getRefRawEndIndex));
  const rawFingerprintMap = buildRawFingerprintMap(selectedRefs);
  const messageFingerprints = buildContiguousRawFingerprints(rawFingerprintMap, rawStartIndex, rawEndIndex);
  if (messageFingerprints.length === 0) {
    throw new Error("Compression range did not resolve to any raw messages.");
  }
  const coveredMessageIds = buildCoveredMessageIds(rawStartIndex, rawEndIndex);
  const protectedSnippets = [...new Set(selectedRefs.flatMap((ref) => ref.protectedSnippets ?? []))];
  const summary = appendProtectedSnippets(normalizeSummary(args.summary), protectedSnippets);

  const consumedBlocks = state.blocks.filter(
    (block) =>
      block.active !== false &&
      !block.deactivatedByUser &&
      (selectedRefs.some((ref) => ref.blockId === block.id) || (block.startIndex >= rawStartIndex && block.endIndex <= rawEndIndex))
  );
  const id = state.nextBlockId;
  const block: DynamicCompressionBlock = {
    id,
    ref: toBlockRef(id),
    active: true,
    startIndex: rawStartIndex,
    endIndex: rawEndIndex,
    startId: startRef.id,
    endId: endRef.id,
    anchorMessageId: startRef.id,
    messageCount: messageFingerprints.length,
    coveredMessageIds,
    coveredToolCallIds: [...new Set(selectedRefs.flatMap((ref) => ref.toolCallIds))],
    protectedSnippets,
    consumedBlockIds: consumedBlocks.map((consumedBlock) => consumedBlock.id),
    parentBlockIds: [...new Set(consumedBlocks.flatMap((consumedBlock) => [consumedBlock.id, ...(consumedBlock.parentBlockIds ?? [])]))],
    messageFingerprints,
    summary,
    summaryTokens: estimateTextTokens(summary),
    createdAt: Date.now(),
    topic: args.topic?.trim() || undefined,
    source
  };

  for (const consumedBlock of consumedBlocks) {
    consumedBlock.active = false;
    consumedBlock.deactivatedByBlockId = id;
  }

  state.nextBlockId += 1;
  state.activeBlock = block;
  state.blocks.push(block);
  return {
    block,
    content: `Compressed ${args.startId}-${args.endId} into ${block.ref}. Future requests will replace that range with the supplied summary.`
  };
}

export function applyDynamicCompressionWorkerResult(state: DynamicCompressionState, content: string): DynamicCompressionWorkerResult[] {
  const parsed = parseWorkerJson(content);
  const rawBlocks = Array.isArray(parsed) ? parsed : (parsed as { blocks?: unknown }).blocks;
  if (!Array.isArray(rawBlocks)) {
    throw new Error("Compression worker response must contain a blocks array.");
  }

  const results: DynamicCompressionWorkerResult[] = [];
  for (const rawBlock of rawBlocks) {
    const block = parseWorkerBlock(rawBlock);
    if (!block) {
      continue;
    }
    results.push(applyDynamicCompressionBlock(state, block, "worker"));
  }
  return results;
}

export function buildDynamicCompressionWorkerRequest(options: {
  state: DynamicCompressionState;
  model: string;
  reasoning?: LlmRequest["reasoning"];
  signal?: AbortSignal;
  goal?: string;
  targetTokenBudget?: number;
  maxBlocks?: number;
}): LlmRequest {
  const maxBlocks = normalizePositiveInteger(options.maxBlocks) ?? 3;
  const targetTokenBudget = normalizePositiveInteger(options.targetTokenBudget);
  const activeBlockLines = options.state.blocks
    .filter((block) => block.active !== false && !block.deactivatedByUser)
    .map((block) => `- ${block.ref ?? toBlockRef(block.id)} ${block.startId ?? "?"}-${block.endId ?? "?"}: ${block.topic ?? "untitled"}`);
  const baseRequest = options.state.workerBaseRequest;
  if (!baseRequest) {
    throw new Error("No current dynamic compression request fork is available.");
  }

  return {
    ...baseRequest,
    model: options.model,
    reasoning: options.reasoning,
    signal: options.signal,
    tools: [],
    messages: [
      ...baseRequest.messages.map(cloneAgentMessage),
      buildWorkerInstructionMessage({
        goal: options.goal,
        targetTokenBudget,
        maxBlocks,
        activeBlockLines
      })
    ]
  };
}

function cloneRequestForWorkerBase(request: LlmRequest): LlmRequest {
  return {
    ...request,
    tools: request.tools?.map((tool) => ({ ...tool })),
    messages: request.messages.map(cloneAgentMessage)
  };
}

function cloneAgentMessage(message: AgentMessage): AgentMessage {
  if (message.role === "assistant") {
    const cloned = { ...message };
    if (message.toolCalls) {
      cloned.toolCalls = message.toolCalls.map((toolCall) => ({ ...toolCall }));
    }
    return cloned;
  }
  return { ...message };
}

function getRefRawStartIndex(ref: DynamicCompressionMessageRef): number {
  return ref.rawStartIndex ?? ref.index;
}

function getRefRawEndIndex(ref: DynamicCompressionMessageRef): number {
  return ref.rawEndIndex ?? ref.index;
}

function buildRawFingerprintMap(refs: readonly DynamicCompressionMessageRef[]): Map<number, string> {
  const map = new Map<number, string>();
  for (const ref of refs) {
    const startIndex = getRefRawStartIndex(ref);
    const endIndex = getRefRawEndIndex(ref);
    const fingerprints = ref.rawFingerprints && ref.rawFingerprints.length > 0 ? ref.rawFingerprints : [ref.fingerprint];
    for (let offset = 0; offset < fingerprints.length; offset += 1) {
      const rawIndex = startIndex + offset;
      if (rawIndex > endIndex) {
        break;
      }
      const fingerprint = fingerprints[offset];
      if (fingerprint && !map.has(rawIndex)) {
        map.set(rawIndex, fingerprint);
      }
    }
  }
  return map;
}

function buildContiguousRawFingerprints(rawFingerprintMap: Map<number, string>, startIndex: number, endIndex: number): string[] {
  const fingerprints: string[] = [];
  for (let rawIndex = startIndex; rawIndex <= endIndex; rawIndex += 1) {
    const fingerprint = rawFingerprintMap.get(rawIndex);
    if (!fingerprint) {
      return [];
    }
    fingerprints.push(fingerprint);
  }
  return fingerprints;
}

function buildCoveredMessageIds(startIndex: number, endIndex: number): string[] {
  const ids: string[] = [];
  for (let rawIndex = startIndex; rawIndex <= endIndex; rawIndex += 1) {
    ids.push(toMessageRef(rawIndex + 1));
  }
  return ids;
}

function buildWorkerInstructionMessage(options: {
  goal?: string;
  targetTokenBudget?: number;
  maxBlocks: number;
  activeBlockLines: string[];
}): AgentMessage {
  return {
    role: "user",
    content: [
      "You are the side worker for dynamic context compression.",
      "You are looking at a fork of the exact visible context that was just sent to the main model. Use the [context ...] IDs already attached to those messages. Ranges may use raw message IDs like m0001 or existing summary block IDs like b1.",
      options.goal ? `Compression goal: ${options.goal}` : undefined,
      options.targetTokenBudget !== undefined ? `Target post-compression token budget: ${options.targetTokenBudget}` : undefined,
      options.activeBlockLines.length > 0 ? `Active compression blocks:\n${options.activeBlockLines.join("\n")}` : undefined,
      `Return only JSON with this shape:\n{"blocks":[{"startId":"m0001","endId":"m0004","topic":"short topic","summary":"dense replacement summary"}]}`,
      `Rules:
- Choose only closed or stale continuous ranges from the forked context.
- Do not include the latest active user request, unresolved tool-call turns, or content still needed word-for-word.
- Do not select this worker instruction message; it has no context ID.
- Prefer a small number of high-value ranges. Maximum blocks: ${options.maxBlocks}.
- Preserve goals, decisions, file paths, commands, outputs, errors, and unresolved follow-ups.
- If there is nothing safe to compress, return {"blocks":[]}.`
    ]
      .filter((part): part is string => Boolean(part?.trim()))
      .join("\n\n")
  };
}

function getMessageToolIds(message: AgentMessage): string[] {
  if (message.role === "assistant") {
    return (message.toolCalls ?? []).map((toolCall) => toolCall.id);
  }
  if (message.role === "tool") {
    return [message.toolCallId];
  }
  return [];
}

function selectCompressionPrefix(messages: readonly AgentMessage[], options: ResolvedDynamicCompressionOptions): CompressionSelection | undefined {
  const turns = splitIntoTurns(messages);
  if (turns.length <= 1) {
    return undefined;
  }

  const keptTurns: MessageTurn[] = [];
  let keptTokens = 0;
  for (let index = turns.length - 1; index >= 0; index -= 1) {
    const turn = turns[index];
    if (!turn) {
      continue;
    }
    const turnTokens = estimateMessagesTokens(turn.messages);
    if (keptTurns.length > 0 && keptTokens + turnTokens > options.keepRecentTokens) {
      break;
    }
    keptTurns.unshift(turn);
    keptTokens += turnTokens;
  }

  if (keptTurns.length === turns.length) {
    return undefined;
  }

  const compactedTurns = turns.slice(0, turns.length - keptTurns.length);
  const lastCompactedTurn = compactedTurns.at(-1);
  if (!lastCompactedTurn) {
    return undefined;
  }

  const selectedMessages = messages.slice(0, lastCompactedTurn.endIndex + 1);
  return {
    endIndex: lastCompactedTurn.endIndex,
    messageCount: selectedMessages.length,
    messages: selectedMessages
  };
}

function splitIntoTurns(messages: readonly AgentMessage[]): MessageTurn[] {
  const turns: MessageTurn[] = [];
  let current: AgentMessage[] = [];
  let startIndex = 0;

  for (let index = 0; index < messages.length; index += 1) {
    const message = messages[index];
    if (!message) {
      continue;
    }
    if (message.role === "user" && current.length > 0) {
      turns.push({ startIndex, endIndex: index - 1, messages: current });
      current = [];
      startIndex = index;
    }
    current.push(message);
  }

  if (current.length > 0) {
    turns.push({ startIndex, endIndex: messages.length - 1, messages: current });
  }

  return turns;
}

function getDynamicTriggerTokens(options: ResolvedContextEngineOptions, dynamicOptions: ResolvedDynamicCompressionOptions): number {
  if (dynamicOptions.triggerTokens !== undefined) {
    return dynamicOptions.triggerTokens;
  }

  const ratio = Math.max(0, Math.min(1, dynamicOptions.triggerRatio ?? 0.5));
  return Math.floor(options.contextWindowTokens * ratio);
}

function getReusableActiveBlock(
  options: ResolvedDynamicCompressionOptions,
  messages: readonly AgentMessage[]
): DynamicCompressionBlock | undefined {
  const block = options.state.activeBlock;
  if (!block || block.active === false || block.deactivatedByUser) {
    return undefined;
  }

  if (!isBlockCurrent(block, messages)) {
    options.state.activeBlock = undefined;
    return undefined;
  }

  return block;
}

function getActiveBlocks(options: ResolvedDynamicCompressionOptions, messages: readonly AgentMessage[]): DynamicCompressionBlock[] {
  const validBlocks: DynamicCompressionBlock[] = [];
  for (const block of options.state.blocks) {
    if (block.active === false || block.deactivatedByUser) {
      continue;
    }
    if (!isBlockCurrent(block, messages)) {
      block.active = false;
      if (options.state.activeBlock === block) {
        options.state.activeBlock = undefined;
      }
      continue;
    }
    validBlocks.push(block);
  }

  validBlocks.sort((left, right) => left.startIndex - right.startIndex);
  const nonOverlapping: DynamicCompressionBlock[] = [];
  let lastEnd = -1;
  for (const block of validBlocks) {
    if (block.startIndex <= lastEnd) {
      continue;
    }
    nonOverlapping.push(block);
    lastEnd = block.endIndex;
  }
  return nonOverlapping;
}

function isBlockCurrent(block: DynamicCompressionBlock, messages: readonly AgentMessage[]): boolean {
  if (block.endIndex >= messages.length) {
    return false;
  }

  const currentFingerprints = messages.slice(block.startIndex, block.endIndex + 1).map(fingerprintMessage);
  return arraysEqual(currentFingerprints, block.messageFingerprints);
}

function allocateBlock(
  options: ResolvedDynamicCompressionOptions,
  messages: readonly AgentMessage[],
  selection: CompressionSelection,
  summary: string,
  model: string
): DynamicCompressionBlock {
  const refs = messages.slice(0, selection.endIndex + 1).map((message, index) => buildMessageRef(message, index, index));
  const protectedSnippets = [...new Set(refs.flatMap((ref) => ref.protectedSnippets ?? []))];
  const normalizedSummary = appendProtectedSnippets(normalizeSummary(summary), protectedSnippets);
  const block: DynamicCompressionBlock = {
    id: options.state.nextBlockId,
    ref: toBlockRef(options.state.nextBlockId),
    active: true,
    startIndex: 0,
    endIndex: selection.endIndex,
    startId: refs[0]?.id,
    endId: refs.at(-1)?.id,
    anchorMessageId: refs[0]?.id,
    messageCount: selection.messageCount,
    coveredMessageIds: refs.map((ref) => ref.id),
    coveredToolCallIds: [...new Set(refs.flatMap((ref) => ref.toolCallIds))],
    protectedSnippets,
    consumedBlockIds: [],
    parentBlockIds: [],
    messageFingerprints: messages.slice(0, selection.endIndex + 1).map(fingerprintMessage),
    summary: normalizedSummary,
    summaryTokens: estimateTextTokens(normalizedSummary),
    createdAt: Date.now(),
    model,
    source: "auto"
  };

  options.state.nextBlockId += 1;
  options.state.activeBlock = block;
  options.state.blocks.push(block);
  return block;
}

function buildRequestFromActiveBlocks(
  request: LlmRequest,
  options: ResolvedDynamicCompressionOptions,
  decisionEstimate: RequestTokenEstimate,
  generated: boolean
): DynamicCompressionResult {
  const activeBlocks = getActiveBlocks(options, request.messages);
  const projection = projectMessagesWithBlocks(request.messages, options, activeBlocks);
  options.state.messageRefs = projection.refs;
  const projectedRequest: LlmRequest = {
    ...request,
    messages: projection.messages
  };
  const compressedEstimate = estimateRequestTokens(projectedRequest, { useProviderUsage: false });
  const applied = activeBlocks.length > 0;
  return {
    request: withDynamicProtocolMetadata(
      projectedRequest,
      options,
      applied,
      generated,
      request.messages.length,
      projection.messages.length,
      decisionEstimate,
      compressedEstimate,
      activeBlocks.at(-1)
    ),
    metadata: buildDynamicCompressionMetadata(
      options,
      applied,
      generated,
      request.messages.length,
      projection.messages.length,
      decisionEstimate,
      compressedEstimate,
      activeBlocks.at(-1)
    ),
    applied
  };
}

function buildRequestFromBlock(
  request: LlmRequest,
  block: DynamicCompressionBlock,
  options: ResolvedDynamicCompressionOptions,
  decisionEstimate: RequestTokenEstimate,
  generated: boolean
): DynamicCompressionResult {
  const projection = projectMessagesWithBlocks(request.messages, options, [block]);
  options.state.messageRefs = projection.refs;
  const compressedRequest: LlmRequest = {
    ...request,
    messages: projection.messages
  };
  const compressedEstimate = estimateRequestTokens(compressedRequest, { useProviderUsage: false });

  return {
    request: withDynamicProtocolMetadata(
      compressedRequest,
      options,
      true,
      generated,
      request.messages.length,
      projection.messages.length,
      decisionEstimate,
      compressedEstimate,
      block
    ),
    metadata: buildDynamicCompressionMetadata(
      options,
      true,
      generated,
      request.messages.length,
      projection.messages.length,
      decisionEstimate,
      compressedEstimate,
      block
    ),
    applied: true
  };
}

type MessageProjection = {
  messages: AgentMessage[];
  refs: DynamicCompressionMessageRef[];
};

function projectMessagesWithBlocks(
  messages: readonly AgentMessage[],
  options: ResolvedDynamicCompressionOptions,
  activeBlocks: readonly DynamicCompressionBlock[]
): MessageProjection {
  const projectedMessages: AgentMessage[] = [];
  const refs: DynamicCompressionMessageRef[] = [];
  let rawIndex = 0;

  const pushRawMessage = (message: AgentMessage, sourceRawIndex: number): void => {
    projectedMessages.push(message);
    refs.push(buildMessageRef(message, sourceRawIndex, projectedMessages.length - 1));
  };

  for (const block of activeBlocks) {
    while (rawIndex < block.startIndex) {
      const message = messages[rawIndex];
      if (message) {
        pushRawMessage(message, rawIndex);
      }
      rawIndex += 1;
    }

    pushProtectedUserMessages(messages, block, options, projectedMessages, refs);
    const summaryMessage = buildDynamicSummaryMessage(block);
    projectedMessages.push(summaryMessage);
    refs.push(buildBlockRef(block, summaryMessage, projectedMessages.length - 1));
    rawIndex = block.endIndex + 1;
  }

  while (rawIndex < messages.length) {
    const message = messages[rawIndex];
    if (message) {
      pushRawMessage(message, rawIndex);
    }
    rawIndex += 1;
  }

  return { messages: projectedMessages, refs };
}

function pushProtectedUserMessages(
  messages: readonly AgentMessage[],
  block: DynamicCompressionBlock,
  options: ResolvedDynamicCompressionOptions,
  projectedMessages: AgentMessage[],
  refs: DynamicCompressionMessageRef[]
): void {
  if (!options.preserveUserMessages) {
    return;
  }

  for (let rawIndex = block.startIndex; rawIndex <= block.endIndex; rawIndex += 1) {
    const message = messages[rawIndex];
    if (!message || message.role !== "user" || message.content.startsWith(DYNAMIC_COMPRESSION_SUMMARY_PREFIX)) {
      continue;
    }
    const protectedMessage: AgentMessage = { role: "user", content: message.content };
    projectedMessages.push(protectedMessage);
    refs.push(buildMessageRef(protectedMessage, rawIndex, projectedMessages.length - 1));
  }
}

function buildMessageRef(message: AgentMessage, rawIndex: number, visibleIndex: number): DynamicCompressionMessageRef {
  const id = toMessageRef(rawIndex + 1);
  const fingerprint = fingerprintMessage(message);
  return {
    id,
    index: visibleIndex,
    role: message.role,
    content: message.content,
    fingerprint,
    toolCallIds: getMessageToolIds(message),
    toolResultId: message.role === "tool" ? message.toolCallId : undefined,
    protectedSnippets: extractProtectedSnippets(message.content),
    kind: "message",
    rawStartIndex: rawIndex,
    rawEndIndex: rawIndex,
    rawFingerprints: [fingerprint],
    coveredMessageIds: [id]
  };
}

function buildBlockRef(block: DynamicCompressionBlock, message: AgentMessage, visibleIndex: number): DynamicCompressionMessageRef {
  return {
    id: block.ref ?? toBlockRef(block.id),
    index: visibleIndex,
    role: "summary",
    content: message.content,
    fingerprint: fingerprintMessage(message),
    toolCallIds: block.coveredToolCallIds ?? [],
    protectedSnippets: block.protectedSnippets,
    kind: "block",
    blockId: block.id,
    rawStartIndex: block.startIndex,
    rawEndIndex: block.endIndex,
    rawFingerprints: block.messageFingerprints,
    coveredMessageIds: block.coveredMessageIds ?? buildCoveredMessageIds(block.startIndex, block.endIndex)
  };
}

function withDynamicProtocolMetadata(
  request: LlmRequest,
  options: ResolvedDynamicCompressionOptions,
  applied: boolean,
  generated: boolean,
  messageCountBefore: number,
  messageCountAfter: number,
  decisionEstimate: RequestTokenEstimate,
  compressedEstimate: RequestTokenEstimate,
  block?: DynamicCompressionBlock
): LlmRequest {
  return {
    ...request,
    metadata: {
      ...request.metadata,
      context: {
        ...request.metadata?.context,
        dynamicCompression: buildDynamicCompressionMetadata(
          options,
          applied,
          generated,
          messageCountBefore,
          messageCountAfter,
          decisionEstimate,
          compressedEstimate,
          block
        )
      }
    }
  };
}

function buildDynamicCompressionMetadata(
  options: ResolvedDynamicCompressionOptions,
  applied: boolean,
  generated: boolean,
  messageCountBefore: number,
  messageCountAfter: number,
  decisionEstimate: RequestTokenEstimate,
  compressedEstimate: RequestTokenEstimate,
  block?: DynamicCompressionBlock
): DynamicCompressionMetadata {
  const triggerTokens = options.state.lastTriggerTokens ?? options.triggerTokens ?? 0;
  return {
    applied,
    generated,
    protocol: {
      toolName: DYNAMIC_COMPRESSION_TOOL_NAME,
      messageRefCount: options.state.messageRefs?.length ?? 0,
      activeBlockCount: options.state.blocks.filter((stateBlock) => stateBlock.active !== false && !stateBlock.deactivatedByUser).length,
      nudge: decisionEstimate.totalTokens >= triggerTokens,
      triggerTokens
    },
    blockId: block?.id,
    model: block?.model,
    messageCountBefore,
    messageCountAfter,
    coveredMessageCount: block?.messageCount,
    decision: toEstimateMetadata(decisionEstimate),
    compressed: toEstimateMetadata(compressedEstimate)
  };
}

function injectCompressionProtocol(
  request: LlmRequest,
  options: ResolvedDynamicCompressionOptions,
  triggerTokens: number,
  shouldNudge: boolean
): LlmRequest {
  if (!options.exposeTool) {
    return request;
  }

  const protocolPrompt = buildCompressionProtocolPrompt(triggerTokens, shouldNudge);
  return {
    ...request,
    systemPrompt: [request.systemPrompt, protocolPrompt].filter((part): part is string => Boolean(part?.trim())).join("\n\n"),
    messages: annotateVisibleMessages(request.messages, options.state)
  };
}

function buildCompressionProtocolPrompt(triggerTokens: number, shouldNudge: boolean): string {
  const nudge = shouldNudge
    ? `\n<context_compression_nudge>Estimated context is at or above ${triggerTokens} tokens. Before continuing, call ${DYNAMIC_COMPRESSION_TOOL_NAME} to offload closed-range selection and summarization to the compression worker.</context_compression_nudge>`
    : "";

  return `<dynamic_context_compression>
Messages may include visible context IDs like [context m0001 role=user] and compressed block IDs like [context b1 role=summary].
When context is low, call ${DYNAMIC_COMPRESSION_TOOL_NAME}. The tool starts a side compression worker that chooses completed/stale ranges and writes dense summaries.
Do not choose ranges or write summaries yourself unless a lower-level manual range tool is explicitly available.
After compression, future requests replace worker-selected raw ranges with summary blocks while preserving the original agent history.
</dynamic_context_compression>${nudge}`;
}

function annotateVisibleMessages(messages: readonly AgentMessage[], state: DynamicCompressionState): AgentMessage[] {
  const usedRefs = new Set<string>();
  return messages.map((message) => {
    const ref = resolveVisibleRef(message, state, usedRefs);
    return annotateMessage(message, ref);
  });
}

function resolveVisibleRef(message: AgentMessage, state: DynamicCompressionState, usedRefs: Set<string>): string {
  const blockRef = getBlockRefFromSummary(message);
  if (blockRef) {
    usedRefs.add(blockRef);
    return blockRef;
  }

  const fingerprint = fingerprintMessage(message);
  const ref = state.messageRefs?.find((candidate) => candidate.fingerprint === fingerprint && !usedRefs.has(candidate.id));
  if (ref) {
    usedRefs.add(ref.id);
    return ref.id;
  }

  const fallback = toMessageRef((state.messageRefs?.length ?? usedRefs.size) + usedRefs.size + 1);
  usedRefs.add(fallback);
  return fallback;
}

function annotateMessage(message: AgentMessage, ref: string): AgentMessage {
  const role = ref.startsWith(DYNAMIC_COMPRESSION_BLOCK_ID_PREFIX) ? "summary" : message.role;
  const marker = `[context ${ref} role=${role}]`;
  if (message.content.startsWith(marker)) {
    return message;
  }

  return {
    ...message,
    content: message.content ? `${marker}\n${message.content}` : marker
  };
}

function buildDynamicSummaryRequest(
  request: LlmRequest,
  dynamicOptions: ResolvedDynamicCompressionOptions,
  contextOptions: ResolvedContextEngineOptions,
  selection: CompressionSelection,
  previousBlock: DynamicCompressionBlock | undefined
): LlmRequest {
  const previousCoverageEnd = previousBlock ? previousBlock.endIndex : -1;
  const summaryInputMessages =
    previousBlock && previousCoverageEnd < selection.endIndex
      ? [buildDynamicSummaryMessage(previousBlock), ...request.messages.slice(previousCoverageEnd + 1, selection.endIndex + 1)]
      : selection.messages;

  return {
    ...request,
    model: contextOptions.compressionModel ?? request.model,
    reasoning: contextOptions.compressionReasoning ?? request.reasoning,
    messages: [
      ...summaryInputMessages,
      {
        role: "user",
        content: `${dynamicOptions.summaryPrompt}\n\nCompression range: messages 1-${selection.messageCount}. Return a complete replacement summary for the full range.`
      }
    ],
    tools: []
  };
}

function buildDynamicSummaryMessage(block: DynamicCompressionBlock): AgentMessage {
  return {
    role: "user",
    content: `${DYNAMIC_COMPRESSION_SUMMARY_PREFIX} ${block.ref ?? toBlockRef(block.id)}\n${block.summary}`
  };
}

function normalizeSummary(summary: string): string {
  const trimmed = summary.trim();
  return trimmed || "(no dynamic summary available)";
}

function extractProtectedSnippets(content: string): string[] {
  const snippets: string[] = [];
  const pattern = /<protect>([\s\S]*?)<\/protect>/g;
  for (const match of content.matchAll(pattern)) {
    const snippet = match[1]?.trim();
    if (snippet) {
      snippets.push(snippet);
    }
  }
  return snippets;
}

function appendProtectedSnippets(summary: string, snippets: readonly string[]): string {
  const missingSnippets = snippets.filter((snippet) => !summary.includes(snippet));
  if (missingSnippets.length === 0) {
    return summary;
  }

  const protectedSection = missingSnippets.map((snippet) => `<protect>\n${snippet}\n</protect>`).join("\n");
  return `${summary}\n\nProtected content copied verbatim:\n${protectedSection}`;
}

function parseWorkerJson(content: string): unknown {
  const trimmed = content.trim();
  if (!trimmed) {
    throw new Error("Compression worker returned an empty response.");
  }

  try {
    return JSON.parse(trimmed) as unknown;
  } catch {
    const match = trimmed.match(/\{[\s\S]*\}/);
    if (!match) {
      throw new Error("Compression worker response was not valid JSON.");
    }
    return JSON.parse(match[0]) as unknown;
  }
}

function parseWorkerBlock(value: unknown): DynamicCompressionBlockCandidate | undefined {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return undefined;
  }

  const record = value as Record<string, unknown>;
  if (typeof record.startId !== "string" || typeof record.endId !== "string" || typeof record.summary !== "string") {
    return undefined;
  }

  return {
    startId: record.startId,
    endId: record.endId,
    summary: record.summary,
    topic: typeof record.topic === "string" ? record.topic : undefined
  };
}

function normalizePositiveInteger(value: number | undefined): number | undefined {
  return typeof value === "number" && Number.isInteger(value) && value > 0 ? value : undefined;
}

function toMessageRef(index: number): string {
  return `${DYNAMIC_COMPRESSION_MESSAGE_ID_PREFIX}${String(index).padStart(4, "0")}`;
}

function toBlockRef(id: number): string {
  return `${DYNAMIC_COMPRESSION_BLOCK_ID_PREFIX}${id}`;
}

function getBlockRefFromSummary(message: AgentMessage): string | undefined {
  if (message.role !== "user") {
    return undefined;
  }
  const match = message.content.match(/^Dynamic context summary:\s+(b\d+)/);
  return match?.[1];
}

function fingerprintMessage(message: AgentMessage): string {
  return JSON.stringify(message);
}

function arraysEqual(left: readonly string[], right: readonly string[]): boolean {
  if (left.length !== right.length) {
    return false;
  }

  return left.every((value, index) => value === right[index]);
}

function toEstimateMetadata(estimate: RequestTokenEstimate) {
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
