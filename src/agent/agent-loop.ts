import type {
  AgentEvent,
  AgentEventSink,
  AgentCompactOptions,
  AgentCompactResult,
  AgentMessage,
  AgentRunOptions,
  AgentRunResult,
  AssistantMessage,
  ReasoningOptions,
  ToolCall
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
import {
  buildMemoryInstructions,
  createMemoryStoreTools,
  createWorkspaceTools,
  MarkdownMemoryStore,
  PHASE_SUMMARY_TAG,
  writePhaseSummaryToMemory,
  WorkspaceMemory,
  type PhaseSummaryConfig,
  type WorkspaceState
} from "../memory/index.js";
import {
  buildPlanningInstructions,
  createPlanningTools,
  formatPlanReviewRequest,
  formatPlanSnapshot,
  Planner,
  CREATE_PLAN_TOOL_NAME,
  READ_PLAN_TOOL_NAME,
  hasOpenPlanSteps,
  type PlanState
} from "../planning/index.js";
import { ToolExecutor, ToolRegistry, type AgentTool, type ToolExecutionMode } from "../tools/registry.js";

const PHASE_SUMMARY_RUNTIME_ENABLED = false;

type PreparedAgentRequest = {
  request: LlmRequest;
  historyReplacement?: AgentMessage[];
  registry: ToolRegistry;
  executor: ToolExecutor;
};

type RuntimePlanningMode = "normal" | "force";

type RunLoopOptions = {
  maxTurns: number;
  options: AgentRunOptions;
  planningMode: RuntimePlanningMode;
  streamEventSink?: AgentEventSink;
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
  history?: readonly AgentMessage[];
  memory?:
    | false
    | {
        workspace?: WorkspaceMemory | WorkspaceState;
        store?: MarkdownMemoryStore | { path?: string };
        includeInstructions?: boolean;
        maxMemoryResults?: number;
        phaseSummary?:
          | false
          | {
              llm?: LlmClient;
              model?: string;
              reasoning?: ReasoningOptions | false;
              store?: MarkdownMemoryStore | { path?: string };
              prompt?: string;
              tags?: string[];
            };
      };
  planning?:
    | false
    | {
        planner?: Planner | PlanState;
        includeInstructions?: boolean;
        includeSnapshot?: boolean;
        maxSteps?: number;
        requirePlanBeforeMutation?: boolean;
      };
  onEvent?: AgentEventSink;
};

type ResolvedAgentMemory = {
  workspace: WorkspaceMemory;
  store?: MarkdownMemoryStore;
  includeInstructions: boolean;
  maxMemoryResults: number;
  phaseSummary?: PhaseSummaryConfig;
};

type ResolvedAgentPlanning = {
  planner: Planner;
  includeInstructions: boolean;
  includeSnapshot: boolean;
  maxSteps: number;
  requirePlanBeforeMutation: boolean;
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
  private readonly memory: ResolvedAgentMemory | undefined;
  private readonly planning: ResolvedAgentPlanning | undefined;
  private readonly backgroundTasks = new Set<Promise<void>>();
  private phaseSummaryQueue: Promise<void> = Promise.resolve();

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
    this.memory = this.resolveMemoryConfig(config.memory);
    this.planning = this.resolvePlanningConfig(config.planning);
    this.messages.push(...(config.history ?? []).map(cloneAgentMessage));
    this.systemPrompt = this.buildSystemPrompt(config);
  }

  get history(): readonly AgentMessage[] {
    return this.messages;
  }

  async run(input: string, options: AgentRunOptions = {}): Promise<AgentRunResult> {
    return this.runInternal(input, options);
  }

  async continueWithToolCall(toolCall: ToolCall, options: AgentRunOptions = {}): Promise<AgentRunResult> {
    return this.continueWithToolCallInternal(toolCall, options);
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

  async waitForBackgroundTasks(): Promise<void> {
    while (this.backgroundTasks.size > 0) {
      await Promise.all([...this.backgroundTasks]);
    }
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

  async *continueWithToolCallEvents(toolCall: ToolCall, options: AgentRunOptions = {}): AsyncIterable<AgentEvent> {
    const events = new AsyncEventQueue<AgentEvent>();
    const runPromise = this.continueWithToolCallInternal(toolCall, options, (event) => {
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
    const planningMode: RuntimePlanningMode = options.forcePlan ? "force" : "normal";

    await this.emit({ type: "agent_start", input }, streamEventSink);
    await this.emit({ type: "message", message: userMessage }, streamEventSink);

    return this.runLoop({ maxTurns, options, planningMode, streamEventSink });
  }

  private async continueWithToolCallInternal(
    toolCall: ToolCall,
    options: AgentRunOptions = {},
    streamEventSink?: AgentEventSink
  ): Promise<AgentRunResult> {
    const maxTurns = options.maxTurns ?? this.maxTurns;
    const planningMode: RuntimePlanningMode = options.forcePlan ? "force" : "normal";

    await this.emit({ type: "agent_start", input: `[tool:${toolCall.name}]` }, streamEventSink);

    const contextOptions = this.resolveContextOptions(options.context);
    const registry = this.buildRuntimeToolRegistry(contextOptions);
    const executor = new ToolExecutor(registry);
    const assistant: AssistantMessage = {
      role: "assistant",
      content: "",
      toolCalls: [toolCall]
    };
    this.messages.push(assistant);
    await this.emit({ type: "message", message: assistant }, streamEventSink);
    await this.emit({ type: "tool_start", turn: 0, toolCall }, streamEventSink);
    const toolResult = await executor.execute(toolCall, options.signal);
    this.messages.push(toolResult);
    await this.emit({ type: "tool_end", turn: 0, toolCall, result: toolResult }, streamEventSink);
    await this.emit({ type: "message", message: toolResult }, streamEventSink);

    return this.runLoop({ maxTurns, options, planningMode, streamEventSink });
  }

  private async runLoop({ maxTurns, options, planningMode, streamEventSink }: RunLoopOptions): Promise<AgentRunResult> {
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
      const prepared = await this.prepareRequest(rawRequest, options.context, planningMode);
      const request = prepared.request;
      if (prepared.historyReplacement) {
        this.messages.splice(0, this.messages.length, ...prepared.historyReplacement);
      }
      const assistant = this.withRequestContext(await this.completeAssistant(turn, request, streamEventSink), request);
      const toolCalls = assistant.toolCalls ?? [];
      if (toolCalls.length === 0) {
        const planGuardResult = this.buildPlanGuardResult();
        if (planGuardResult) {
          const guardedAssistant = this.withSyntheticToolCall(assistant, planGuardResult.toolCall);
          lastAssistant = guardedAssistant;
          this.messages.push(guardedAssistant);
          await this.emit({ type: "message", message: guardedAssistant }, streamEventSink);
          await this.emit({ type: "tool_start", turn, toolCall: planGuardResult.toolCall }, streamEventSink);
          this.messages.push(planGuardResult.toolResult);
          await this.emit({ type: "tool_end", turn, toolCall: planGuardResult.toolCall, result: planGuardResult.toolResult }, streamEventSink);
          await this.emit({ type: "message", message: planGuardResult.toolResult }, streamEventSink);
          await this.emit({
            type: "turn_end",
            turn,
            message: guardedAssistant,
            toolResults: [planGuardResult.toolResult],
            context: request.metadata?.context
          }, streamEventSink);
          continue;
        }
        lastAssistant = assistant;
        this.messages.push(assistant);
        await this.emit({ type: "message", message: assistant }, streamEventSink);
        const result = this.buildResult(assistant.content, turn, "final");
        await this.emit({ type: "turn_end", turn, message: assistant, toolResults: [], context: request.metadata?.context }, streamEventSink);
        await this.emit({ type: "agent_end", result }, streamEventSink);
        this.enqueuePhaseSummary(result, options.signal, streamEventSink);
        return result;
      }

      lastAssistant = assistant;
      this.messages.push(assistant);
      await this.emit({ type: "message", message: assistant }, streamEventSink);
      const toolResults = await this.executeToolCalls(turn, toolCalls, prepared.registry, prepared.executor, options.signal, streamEventSink);

      this.messages.push(...toolResults);
      for (const message of toolResults) {
        await this.emit({ type: "message", message }, streamEventSink);
      }
      await this.emit({ type: "turn_end", turn, message: assistant, toolResults, context: request.metadata?.context }, streamEventSink);

      const reviewResult = this.buildPlanReviewResult();
      if (reviewResult) {
        this.messages.push(reviewResult.message);
        await this.emit({ type: "message", message: reviewResult.message }, streamEventSink);
        const result = this.buildResult(reviewResult.message.content, turn, "plan_review");
        await this.emit({ type: "agent_end", result }, streamEventSink);
        return result;
      }
    }

    const result = this.buildResult(lastAssistant?.content ?? "", maxTurns, "max_turns");
    await this.emit({ type: "agent_end", result }, streamEventSink);
    this.enqueuePhaseSummary(result, options.signal, streamEventSink);
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
    runContext?: false | Partial<ContextEngineOptions>,
    planningMode: RuntimePlanningMode = "normal"
  ): Promise<PreparedAgentRequest> {
    const contextOptions = this.resolveContextOptions(runContext);
    const registry = this.buildRuntimeToolRegistry(contextOptions);
    const executor = new ToolExecutor(registry);
    const systemPrompt = this.withRuntimePlanningMode(this.withDynamicPlanningSnapshot(request.systemPrompt), planningMode);
    const requestWithRuntimeTools: LlmRequest = {
      ...request,
      systemPrompt,
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
    if (this.memory) {
      tools.push(...createWorkspaceTools(this.memory.workspace));
      if (this.memory.store) {
        tools.push(...createMemoryStoreTools(this.memory.store, { maxMemoryResults: this.memory.maxMemoryResults }));
      }
    }
    if (this.planning) {
      tools.push(...createPlanningTools(this.planning.planner));
    }
    const visibleTools = this.shouldGatePrePlanTools() ? tools.filter((tool) => isAllowedBeforePlan(tool, this.planning?.planner.state)) : tools;
    return new ToolRegistry(visibleTools);
  }

  private buildSystemPrompt(config: AgentConfig): string | undefined {
    const background = this.buildBackgroundOptions(config.background);
    return new PromptBuilder().buildConversationSystemPrompt({
      basePrompt: config.systemPrompt,
      defaultInstructions: config.background === false ? false : undefined,
      background,
      fragments: this.buildStaticPromptFragments(config)
    });
  }

  private buildStaticPromptFragments(config: AgentConfig): PromptFragment[] {
    const fragments = [...(config.promptFragments ?? [])];
    if (config.background === false) {
      return fragments;
    }

    if (this.memory?.includeInstructions) {
      const memoryInstructions = buildMemoryInstructions({
        hasWorkspace: true,
        hasStore: Boolean(this.memory.store)
      });
      if (memoryInstructions) {
        fragments.push(memoryInstructions);
      }
    }
    if (this.planning?.includeInstructions) {
      fragments.push(buildPlanningInstructions({ requirePlanBeforeMutation: this.planning.requirePlanBeforeMutation }));
    }
    return fragments;
  }

  private resolveMemoryConfig(config: AgentConfig["memory"]): ResolvedAgentMemory | undefined {
    if (config === false) {
      return undefined;
    }

    const memoryConfig = config ?? {};
    const store = resolveMemoryStore(memoryConfig.store);
    return {
      workspace: resolveWorkspace(memoryConfig.workspace),
      store,
      includeInstructions: memoryConfig.includeInstructions ?? true,
      maxMemoryResults: clampMemoryResults(memoryConfig.maxMemoryResults ?? 5),
      phaseSummary: this.resolvePhaseSummaryConfig(memoryConfig.phaseSummary, store)
    };
  }

  private resolvePhaseSummaryConfig(
    config: NonNullable<Exclude<AgentConfig["memory"], false>>["phaseSummary"],
    defaultStore: MarkdownMemoryStore | undefined
  ): PhaseSummaryConfig | undefined {
    if (!PHASE_SUMMARY_RUNTIME_ENABLED) {
      return undefined;
    }
    if (!config) {
      return undefined;
    }

    const store = resolveMemoryStore(config.store) ?? defaultStore ?? new MarkdownMemoryStore();
    return {
      llm: config.llm ?? this.compressionLlm ?? this.llm,
      model: config.model ?? this.compressionModel ?? this.model,
      reasoning: config.reasoning ?? this.compressionReasoning,
      store,
      prompt: config.prompt,
      tags: [PHASE_SUMMARY_TAG, ...(config.tags ?? [])]
    };
  }

  private resolvePlanningConfig(config: AgentConfig["planning"]): ResolvedAgentPlanning | undefined {
    if (config === false || config === undefined) {
      return undefined;
    }

    const planningConfig = config;
    return {
      planner: resolvePlanner(planningConfig.planner),
      includeInstructions: planningConfig.includeInstructions ?? true,
      includeSnapshot: planningConfig.includeSnapshot ?? true,
      maxSteps: clampPlanSnapshotSteps(planningConfig.maxSteps ?? 20),
      requirePlanBeforeMutation: planningConfig.requirePlanBeforeMutation ?? true
    };
  }

  private shouldGatePrePlanTools(): boolean {
    if (!this.planning?.requirePlanBeforeMutation) {
      return false;
    }
    const plan = this.planning.planner.state;
    return !plan || plan.reviewStatus !== "approved";
  }

  private withDynamicPlanningSnapshot(systemPrompt: string | undefined): string | undefined {
    if (!this.planning?.includeSnapshot) {
      return systemPrompt;
    }
    const snapshot = formatPlanSnapshot(this.planning.planner.state, { maxSteps: this.planning.maxSteps });
    if (!snapshot) {
      return systemPrompt;
    }
    if (!systemPrompt || systemPrompt.trim() === "") {
      return snapshot;
    }
    return `${systemPrompt}\n\n${snapshot}`;
  }

  private withRuntimePlanningMode(systemPrompt: string | undefined, planningMode: RuntimePlanningMode): string | undefined {
    if (!this.planning || planningMode !== "force") {
      return systemPrompt;
    }
    const directive = [
      "<runtime_plan_mode forced=\"true\">",
      "The user explicitly requested plan mode for this turn.",
      "You must create or revise the structured plan before execution, then present the plan to the user for review.",
      "Do not perform write, execute, or other mutation work until the plan is approved.",
      "</runtime_plan_mode>"
    ].join("\n");
    if (!systemPrompt || systemPrompt.trim() === "") {
      return directive;
    }
    return `${systemPrompt}\n\n${directive}`;
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
      stoppedBy,
      plan: this.planning?.planner.state
    };
  }

  private buildPlanGuardResult():
    | { toolCall: NonNullable<AssistantMessage["toolCalls"]>[number]; toolResult: AgentMessage & { role: "tool" } }
    | undefined {
    const plan = this.planning?.planner.state;
    if (!plan) {
      return undefined;
    }

    if (plan.reviewStatus !== "approved" || !hasOpenPlanSteps(plan)) {
      return undefined;
    }

    const openSteps = plan.steps.filter((step) => step.status === "pending" || step.status === "in_progress");
    const toolCall = {
      id: `plan_guard_${this.messages.length + 1}_${plan.revision}_${openSteps.length}`,
      name: READ_PLAN_TOOL_NAME,
      arguments: {}
    };
    return {
      toolCall,
      toolResult: {
        role: "tool",
        toolCallId: toolCall.id,
        toolName: toolCall.name,
        content: [
          "Plan mode guard: the conversation cannot end while plan steps are pending or in_progress.",
          "Continue the task, or update the plan by completing, blocking, cancelling, or revising the open steps before the final answer.",
          formatPlanSnapshot(plan, { maxSteps: this.planning?.maxSteps })
        ].filter((line): line is string => Boolean(line)).join("\n\n"),
        details: { plan }
      }
    };
  }

  private buildPlanReviewResult(): { message: AssistantMessage } | undefined {
    const plan = this.planning?.planner.state;
    if (!plan || plan.reviewStatus !== "pending") {
      return undefined;
    }

    const message: AssistantMessage = {
      role: "assistant",
      content: formatPlanReviewRequest(plan, { maxSteps: this.planning?.maxSteps })
    };
    return { message };
  }

  private withSyntheticToolCall(assistant: AssistantMessage, toolCall: NonNullable<AssistantMessage["toolCalls"]>[number]): AssistantMessage {
    return {
      ...assistant,
      toolCalls: [...(assistant.toolCalls ?? []), toolCall]
    };
  }

  private enqueuePhaseSummary(
    result: AgentRunResult,
    signal: AbortSignal | undefined,
    streamEventSink?: AgentEventSink
  ): void {
    if (!this.memory?.phaseSummary) {
      return;
    }

    const task = this.phaseSummaryQueue.then(() => this.writePhaseSummary(result, signal, streamEventSink));
    this.phaseSummaryQueue = task.catch(() => undefined);
    this.trackBackgroundTask(this.phaseSummaryQueue);
  }

  private trackBackgroundTask(task: Promise<void>): void {
    this.backgroundTasks.add(task);
    task.finally(() => {
      this.backgroundTasks.delete(task);
    }).catch(() => undefined);
  }

  private async writePhaseSummary(
    result: AgentRunResult,
    signal: AbortSignal | undefined,
    streamEventSink?: AgentEventSink
  ): Promise<void> {
    const phaseSummary = this.memory?.phaseSummary;
    if (!phaseSummary) {
      return;
    }

    await this.emit(
      {
        type: "memory_summary_start",
        model: phaseSummary.model,
        messageCount: result.messages.length,
        storePath: phaseSummary.store.path
      },
      streamEventSink
    );

    try {
      const summary = await writePhaseSummaryToMemory({
        ...phaseSummary,
        messages: result.messages,
        turns: result.turns,
        stoppedBy: result.stoppedBy,
        signal
      });
      await this.emit(
        {
          type: "memory_summary_end",
          model: phaseSummary.model,
          storePath: phaseSummary.store.path,
          entry: summary.entry,
          action: summary.action,
          summaryTokens: summary.summaryTokens,
          summaryChars: summary.summaryChars,
          usage: summary.message.usage
        },
        streamEventSink
      );
    } catch (error) {
      await this.emit(
        {
          type: "memory_summary_error",
          model: phaseSummary.model,
          storePath: phaseSummary.store.path,
          error: formatError(error)
        },
        streamEventSink
      );
    }
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

function resolveWorkspace(workspace: WorkspaceMemory | WorkspaceState | undefined): WorkspaceMemory {
  if (workspace instanceof WorkspaceMemory) {
    return workspace;
  }
  return new WorkspaceMemory(workspace);
}

function resolveMemoryStore(store: MarkdownMemoryStore | { path?: string } | undefined): MarkdownMemoryStore | undefined {
  if (!store) {
    return undefined;
  }
  if (store instanceof MarkdownMemoryStore) {
    return store;
  }
  return new MarkdownMemoryStore(store);
}

function resolvePlanner(planner: Planner | PlanState | undefined): Planner {
  if (planner instanceof Planner) {
    return planner;
  }
  return new Planner(planner);
}

function clampMemoryResults(value: number): number {
  if (!Number.isFinite(value)) {
    return 5;
  }
  return Math.min(20, Math.max(1, Math.floor(value)));
}

function clampPlanSnapshotSteps(value: number): number {
  if (!Number.isFinite(value)) {
    return 20;
  }
  return Math.min(100, Math.max(1, Math.floor(value)));
}

function isAllowedBeforePlan(tool: AgentTool, plan: PlanState | undefined): boolean {
  if (tool.access === "read") {
    return true;
  }
  if (tool.access !== "planner") {
    return false;
  }
  if (!plan) {
    return tool.name === CREATE_PLAN_TOOL_NAME || tool.name === READ_PLAN_TOOL_NAME;
  }
  return true;
}

function cloneAgentMessage(message: AgentMessage): AgentMessage {
  return JSON.parse(JSON.stringify(message)) as AgentMessage;
}

function formatError(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
