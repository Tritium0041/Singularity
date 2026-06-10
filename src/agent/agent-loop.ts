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
  mergeContextEngineOptions,
  PromptBuilder,
  type ContextEngineOptions,
  type PromptFragment,
  type SystemPromptBackgroundOptions,
  ContextEngine
} from "../context/index.js";
import { isStreamingLlmClient, type LlmClient, type LlmRequest } from "../llm/types.js";
import { ToolExecutor, ToolRegistry, type AgentTool, type ToolExecutionMode } from "../tools/registry.js";

export type AgentConfig = {
  llm: LlmClient;
  model: string;
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
  private readonly systemPrompt: string | undefined;
  private readonly maxTurns: number;
  private readonly onEvent: AgentEventSink | undefined;
  private readonly toolExecution: ToolExecutionMode;
  private readonly reasoning: ReasoningOptions | false | undefined;
  private readonly context: false | ContextEngineOptions | undefined;
  private readonly tools: ToolRegistry;
  private readonly executor: ToolExecutor;
  private readonly messages: AgentMessage[] = [];

  constructor(config: AgentConfig) {
    this.llm = config.llm;
    this.model = config.model;
    this.maxTurns = config.maxTurns ?? 8;
    this.onEvent = config.onEvent;
    this.toolExecution = config.toolExecution ?? "sequential";
    this.reasoning = config.reasoning;
    this.context = config.context;
    this.tools = config.tools instanceof ToolRegistry ? config.tools : new ToolRegistry(config.tools ?? []);
    this.executor = new ToolExecutor(this.tools);
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

    const contextOptions = mergeContextEngineOptions(this.context, options.context);
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
    const prepared = await new ContextEngine(contextOptions).compactWithHandoff(rawRequest, (summaryRequest) =>
      this.llm.complete(summaryRequest)
    );
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

      const toolResults = await this.executeToolCalls(turn, toolCalls, options.signal, streamEventSink);

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
  ): Promise<{ request: LlmRequest; historyReplacement?: AgentMessage[] }> {
    const contextOptions = mergeContextEngineOptions(this.context, runContext);
    if (contextOptions === false) {
      return { request };
    }
    return new ContextEngine(contextOptions).prepareWithHandoff(request, (summaryRequest) => this.llm.complete(summaryRequest));
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
    signal?: AbortSignal,
    streamEventSink?: AgentEventSink
  ) {
    const mustRunSequentially =
      this.toolExecution === "sequential" ||
      toolCalls.some((toolCall) => this.tools.get(toolCall.name)?.executionMode === "sequential");

    if (mustRunSequentially) {
      const results = [];
      for (const toolCall of toolCalls) {
        await this.emit({ type: "tool_start", turn, toolCall }, streamEventSink);
        const result = await this.executor.execute(toolCall, signal);
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
        const result = await this.executor.execute(toolCall, signal);
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
