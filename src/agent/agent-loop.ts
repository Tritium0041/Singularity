import type {
  AgentEvent,
  AgentEventSink,
  AgentCompactOptions,
  AgentCompactResult,
  AgentMessage,
  AgentRunOptions,
  AgentRunResult,
  AssistantMessage,
  ReasoningOptions
} from "../types.js";
import {
  applyDynamicCompressionWorkerResult,
  buildDynamicCompressionWorkerRequest,
  createDynamicCompressionState,
  DYNAMIC_COMPRESSION_TOOL_NAME,
  mergeContextEngineOptions,
  PromptBuilder,
  type ContextEngineOptions,
  type DynamicCompressionCompactToolArgs,
  type DynamicCompressionState,
  type PromptFragment,
  type ResolvedContextEngineOptions,
  type SystemPromptBackgroundOptions,
  ContextEngine
} from "../context/index.js";
import { isStreamingLlmClient, type LlmClient, type LlmRequest } from "../llm/types.js";
import { ToolExecutor, ToolRegistry, type AgentTool, type ToolExecutionMode } from "../tools/registry.js";

type PreparedAgentRequest = {
  request: LlmRequest;
  historyReplacement?: AgentMessage[];
  registry: ToolRegistry;
  executor: ToolExecutor;
};

export type AgentConfig = {
  llm: LlmClient;
  model: string;
  compressionLlm?: LlmClient;
  compressionModel?: string;
  compressionReasoning?: ReasoningOptions | false;
  systemPrompt?: string;
  tools?: AgentTool[] | ToolRegistry;
  toolExecution?: ToolExecutionMode;
  maxTurns?: number;
  reasoning?: ReasoningOptions | false;
  context?: false | ContextEngineOptions;
  background?: false | SystemPromptBackgroundOptions;
  promptFragments?: readonly PromptFragment[];
  onEvent?: AgentEventSink;
};

export class Agent {
  private readonly llm: LlmClient;
  private readonly model: string;
  private readonly compressionLlm: LlmClient | undefined;
  private readonly compressionModel: string | undefined;
  private readonly compressionReasoning: ReasoningOptions | false | undefined;
  private readonly systemPrompt: string | undefined;
  private readonly maxTurns: number;
  private readonly onEvent: AgentEventSink | undefined;
  private readonly toolExecution: ToolExecutionMode;
  private readonly reasoning: ReasoningOptions | false | undefined;
  private readonly context: false | ContextEngineOptions | undefined;
  private readonly tools: ToolRegistry;
  private readonly messages: AgentMessage[] = [];
  private readonly dynamicCompressionState: DynamicCompressionState;

  constructor(config: AgentConfig) {
    this.llm = config.llm;
    this.model = config.model;
    this.compressionLlm = config.compressionLlm;
    this.compressionModel = config.compressionModel;
    this.compressionReasoning = config.compressionReasoning;
    this.maxTurns = config.maxTurns ?? 8;
    this.onEvent = config.onEvent;
    this.toolExecution = config.toolExecution ?? "sequential";
    this.reasoning = config.reasoning;
    this.context = config.context;
    this.tools = config.tools instanceof ToolRegistry ? config.tools : new ToolRegistry(config.tools ?? []);
    this.dynamicCompressionState = getInitialDynamicCompressionState(config.context);
    this.systemPrompt = this.buildSystemPrompt(config);
  }

  get history(): readonly AgentMessage[] {
    return this.messages;
  }

  async run(input: string, options: AgentRunOptions = {}): Promise<AgentRunResult> {
    return this.runInternal(input, options);
  }

  async compactHistory(options: AgentCompactOptions = {}): Promise<AgentCompactResult> {
    if (this.messages.length === 0) {
      return {
        compacted: false,
        messages: []
      };
    }

    const contextOptions = this.resolveContextOptions(options.context);
    if (contextOptions === false) {
      return {
        compacted: false,
        messages: [...this.messages]
      };
    }

    const rawRequest: LlmRequest = {
      model: this.model,
      systemPrompt: this.systemPrompt,
      messages: [...this.messages],
      tools: this.tools.toLlmToolSpecs(),
      reasoning: options.reasoning ?? this.reasoning,
      signal: options.signal
    };
    const prepared = await new ContextEngine(contextOptions).compactWithHandoff(rawRequest, (summaryRequest) => this.completeCompression(summaryRequest));
    const replacement = prepared.historyReplacement ?? prepared.request.messages;
    this.messages.splice(0, this.messages.length, ...replacement);

    return {
      compacted: prepared.request.metadata?.context?.compacted === true,
      messages: [...this.messages],
      context: prepared.request.metadata?.context
    };
  }

  async *runEvents(input: string, options: AgentRunOptions = {}): AsyncIterable<AgentEvent> {
    const events = new AsyncEventQueue<AgentEvent>();
    const runPromise = this.runInternal(input, options, (event) => {
      events.push(event);
    }).then(
      () => events.close(),
      (error: unknown) => events.fail(error)
    );

    try {
      for await (const event of events) {
        yield event;
      }
    } finally {
      await runPromise;
    }
  }

  private async runInternal(
    input: string,
    options: AgentRunOptions = {},
    streamEventSink?: AgentEventSink
  ): Promise<AgentRunResult> {
    const maxTurns = options.maxTurns ?? this.maxTurns;
    const userMessage: AgentMessage = { role: "user", content: input };
    this.messages.push(userMessage);

    await this.emit({ type: "agent_start", input }, streamEventSink);
    await this.emit({ type: "message", message: userMessage }, streamEventSink);

    let lastAssistant: AssistantMessage | undefined;

    for (let turn = 1; turn <= maxTurns; turn += 1) {
      await this.emit({ type: "turn_start", turn }, streamEventSink);

      const rawRequest: LlmRequest = {
        model: this.model,
        systemPrompt: this.systemPrompt,
        messages: [...this.messages],
        tools: this.tools.toLlmToolSpecs(),
        reasoning: options.reasoning ?? this.reasoning,
        signal: options.signal
      };
      const prepared = await this.prepareRequest(rawRequest, options.context);
      const request = prepared.request;
      if (prepared.historyReplacement) {
        this.messages.splice(0, this.messages.length, ...prepared.historyReplacement);
      }
      const assistant = this.withRequestContext(await this.completeAssistant(turn, request, streamEventSink), request);
      lastAssistant = assistant;
      this.messages.push(assistant);
      await this.emit({ type: "message", message: assistant }, streamEventSink);

      const toolCalls = assistant.toolCalls ?? [];
      if (toolCalls.length === 0) {
        const result = this.buildResult(assistant.content, turn, "final");
        await this.emit({ type: "turn_end", turn, message: assistant, toolResults: [], context: request.metadata?.context }, streamEventSink);
        await this.emit({ type: "agent_end", result }, streamEventSink);
        return result;
      }

      const toolResults = await this.executeToolCalls(turn, toolCalls, prepared.registry, prepared.executor, options.signal, streamEventSink);

      this.messages.push(...toolResults);
      for (const message of toolResults) {
        await this.emit({ type: "message", message }, streamEventSink);
      }
      await this.emit({ type: "turn_end", turn, message: assistant, toolResults, context: request.metadata?.context }, streamEventSink);
    }

    const result = this.buildResult(lastAssistant?.content ?? "", maxTurns, "max_turns");
    await this.emit({ type: "agent_end", result }, streamEventSink);
    return result;
  }

  private async completeAssistant(
    turn: number,
    request: LlmRequest,
    streamEventSink?: AgentEventSink
  ): Promise<AssistantMessage> {
    if (!isStreamingLlmClient(this.llm)) {
      return this.llm.complete(request);
    }

    let content = "";
    let thinkingContent = "";
    let finalMessage: AssistantMessage | undefined;
    for await (const event of this.llm.stream(request)) {
      if (event.type === "thinking_delta") {
        thinkingContent += event.delta;
        await this.emit({ type: "thinking_delta", turn, delta: event.delta, content: thinkingContent }, streamEventSink);
        continue;
      }
      if (event.type === "text_delta") {
        content += event.delta;
        await this.emit({ type: "assistant_delta", turn, delta: event.delta, content }, streamEventSink);
        continue;
      }
      if (event.type === "tool_call_delta") {
        await this.emit({
          type: "tool_call_delta",
          turn,
          toolCallId: event.toolCallId,
          toolName: event.toolName,
          delta: event.delta,
          argumentsText: event.argumentsText
        }, streamEventSink);
        continue;
      }
      finalMessage = event.message;
    }

    if (!finalMessage) {
      throw new Error("Streaming LLM finished without a final message.");
    }
    return finalMessage;
  }

  private withRequestContext(message: AssistantMessage, request: LlmRequest): AssistantMessage {
    const compacted = request.metadata?.context?.compacted;
    if (compacted === undefined) {
      return message;
    }
    return {
      ...message,
      context: {
        ...message.context,
        requestCompacted: compacted
      }
    };
  }

  private async prepareRequest(
    request: LlmRequest,
    runContext?: false | Partial<ContextEngineOptions>
  ): Promise<PreparedAgentRequest> {
    const contextOptions = this.resolveContextOptions(runContext);
    const registry = this.buildRuntimeToolRegistry(contextOptions);
    const executor = new ToolExecutor(registry);
    const requestWithRuntimeTools: LlmRequest = {
      ...request,
      tools: registry.toLlmToolSpecs()
    };
    if (contextOptions === false) {
      return { request: requestWithRuntimeTools, registry, executor };
    }
    const prepared = await new ContextEngine(contextOptions).prepareWithHandoff(requestWithRuntimeTools, (summaryRequest) => this.completeCompression(summaryRequest));
    if (contextOptions.dynamicCompression) {
      contextOptions.dynamicCompression.state.workerBaseRequest = prepared.request;
    }
    return {
      ...prepared,
      registry,
      executor
    };
  }

  private completeCompression(request: LlmRequest): Promise<AssistantMessage> {
    return (this.compressionLlm ?? this.llm).complete(request);
  }

  private resolveContextOptions(runContext?: false | Partial<ContextEngineOptions>): false | ResolvedContextEngineOptions {
    const merged = mergeContextEngineOptions(this.context, runContext);
    if (merged === false) {
      return false;
    }

    return {
      ...merged,
      compressionModel: merged.compressionModel ?? this.compressionModel,
      compressionReasoning: merged.compressionReasoning ?? this.compressionReasoning,
      dynamicCompression: merged.dynamicCompression
        ? {
            ...merged.dynamicCompression,
            state: this.dynamicCompressionState
          }
        : false
    };
  }

  private buildRuntimeToolRegistry(contextOptions: false | ResolvedContextEngineOptions): ToolRegistry {
    const tools = this.tools.list();
    if (contextOptions !== false && contextOptions.dynamicCompression && contextOptions.dynamicCompression.exposeTool && !this.tools.get(DYNAMIC_COMPRESSION_TOOL_NAME)) {
      tools.push(
        createDynamicCompressionTool({
          state: this.dynamicCompressionState,
          model: contextOptions.compressionModel ?? this.model,
          reasoning: contextOptions.compressionReasoning ?? this.reasoning,
          complete: (request) => this.completeCompression(request)
        })
      );
    }
    return new ToolRegistry(tools);
  }

  private buildSystemPrompt(config: AgentConfig): string | undefined {
    const background = this.buildBackgroundOptions(config.background);
    return new PromptBuilder().buildConversationSystemPrompt({
      basePrompt: config.systemPrompt,
      defaultInstructions: config.background === false ? false : undefined,
      background,
      fragments: config.promptFragments
    });
  }

  private buildBackgroundOptions(background: AgentConfig["background"]): false | SystemPromptBackgroundOptions {
    if (background === false) {
      return false;
    }

    return {
      cwd: process.cwd(),
      currentDate: new Date().toISOString().slice(0, 10),
      timezone: resolveTimezone(),
      shell: process.env.SHELL,
      tools: this.tools.toLlmToolSpecs().map((tool) => ({ name: tool.name, description: tool.description })),
      ...background
    };
  }

  private async executeToolCalls(
    turn: number,
    toolCalls: NonNullable<AssistantMessage["toolCalls"]>,
    registry: ToolRegistry,
    executor: ToolExecutor,
    signal?: AbortSignal,
    streamEventSink?: AgentEventSink
  ) {
    const mustRunSequentially =
      this.toolExecution === "sequential" ||
      toolCalls.some((toolCall) => registry.get(toolCall.name)?.executionMode === "sequential");

    if (mustRunSequentially) {
      const results = [];
      for (const toolCall of toolCalls) {
        await this.emit({ type: "tool_start", turn, toolCall }, streamEventSink);
        const result = await executor.execute(toolCall, signal);
        results.push(result);
        await this.emit({ type: "tool_end", turn, toolCall, result }, streamEventSink);
      }
      return results;
    }

    for (const toolCall of toolCalls) {
      await this.emit({ type: "tool_start", turn, toolCall }, streamEventSink);
    }

    const results = await Promise.all(
      toolCalls.map(async (toolCall) => {
        const result = await executor.execute(toolCall, signal);
        await this.emit({ type: "tool_end", turn, toolCall, result }, streamEventSink);
        return result;
      })
    );

    return results;
  }

  private buildResult(output: string, turns: number, stoppedBy: AgentRunResult["stoppedBy"]): AgentRunResult {
    return {
      output,
      messages: [...this.messages],
      turns,
      stoppedBy
    };
  }

  private async emit(event: Parameters<NonNullable<AgentEventSink>>[0], streamEventSink?: AgentEventSink): Promise<void> {
    await this.onEvent?.(event);
    await streamEventSink?.(event);
  }
}

class AsyncEventQueue<T> implements AsyncIterable<T> {
  private readonly values: T[] = [];
  private readonly waiters: Array<{
    resolve: (result: IteratorResult<T>) => void;
    reject: (error: unknown) => void;
  }> = [];
  private closed = false;
  private error: unknown;

  push(value: T): void {
    if (this.closed || this.error) {
      return;
    }
    const waiter = this.waiters.shift();
    if (waiter) {
      waiter.resolve({ value, done: false });
      return;
    }
    this.values.push(value);
  }

  close(): void {
    if (this.closed) {
      return;
    }
    this.closed = true;
    for (const waiter of this.waiters.splice(0)) {
      waiter.resolve({ value: undefined, done: true });
    }
  }

  fail(error: unknown): void {
    this.error = error;
    this.closed = true;
    for (const waiter of this.waiters.splice(0)) {
      waiter.reject(error);
    }
  }

  [Symbol.asyncIterator](): AsyncIterator<T> {
    return {
      next: () => this.next()
    };
  }

  private next(): Promise<IteratorResult<T>> {
    if (this.values.length > 0) {
      return Promise.resolve({ value: this.values.shift() as T, done: false });
    }
    if (this.error) {
      return Promise.reject(this.error);
    }
    if (this.closed) {
      return Promise.resolve({ value: undefined, done: true });
    }
    return new Promise<IteratorResult<T>>((resolve, reject) => {
      this.waiters.push({ resolve, reject });
    });
  }
}

function createDynamicCompressionTool(options: {
  state: DynamicCompressionState;
  model: string;
  reasoning?: ReasoningOptions | false;
  complete: (request: LlmRequest) => Promise<AssistantMessage>;
}): AgentTool {
  return {
    name: DYNAMIC_COMPRESSION_TOOL_NAME,
    description: "Offload context compaction to a side worker that selects stale closed message ranges, summarizes them, and installs reusable summary blocks.",
    executionMode: "sequential",
    parameters: {
      type: "object",
      additionalProperties: false,
      properties: {
        goal: {
          type: "string",
          description: "Optional guidance for the compression worker about what context pressure or task phase to optimize for."
        },
        targetTokenBudget: {
          type: "integer",
          description: "Optional target token budget after compression."
        },
        maxBlocks: {
          type: "integer",
          description: "Maximum number of closed ranges the worker should summarize in this pass."
        }
      }
    },
    async execute(args, context) {
      const compactArgs = args as DynamicCompressionCompactToolArgs;
      const request = buildDynamicCompressionWorkerRequest({
        state: options.state,
        model: options.model,
        reasoning: options.reasoning,
        signal: context.signal,
        goal: compactArgs.goal,
        targetTokenBudget: compactArgs.targetTokenBudget,
        maxBlocks: compactArgs.maxBlocks
      });
      const response = await options.complete(request);
      const results = applyDynamicCompressionWorkerResult(options.state, response.content);
      if (results.length === 0) {
        return {
          content: "Compression worker found no closed context ranges safe to summarize.",
          details: {
            responseUsage: response.usage
          }
        };
      }

      return {
        content: `Compression worker installed ${results.length} summary block(s): ${results.map((result) => result.block.ref).join(", ")}.`,
        details: {
          blocks: results.map((result) => ({
            blockId: result.block.id,
            blockRef: result.block.ref,
            startId: result.block.startId,
            endId: result.block.endId,
            topic: result.block.topic,
            coveredMessageIds: result.block.coveredMessageIds,
            coveredToolCallIds: result.block.coveredToolCallIds
          })),
          responseUsage: response.usage
        }
      };
    }
  };
}

function resolveTimezone(): string | undefined {
  if (process.env.TZ) {
    return process.env.TZ;
  }
  try {
    return Intl.DateTimeFormat().resolvedOptions().timeZone;
  } catch {
    return undefined;
  }
}

function getInitialDynamicCompressionState(context?: false | ContextEngineOptions): DynamicCompressionState {
  if (context === false || !context?.dynamicCompression) {
    return createDynamicCompressionState();
  }

  return context.dynamicCompression.state ?? createDynamicCompressionState();
}
