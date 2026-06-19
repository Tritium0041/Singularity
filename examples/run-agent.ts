import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { stdin, stdout } from "node:process";
import { createInterface } from "node:readline/promises";
import {
  Agent,
  AgentSessionStore,
  APPROVE_PLAN_TOOL_NAME,
  createCoreTools,
  formatSkillInvocation,
  createLlmClientFromEnv,
  generateSessionTitle,
  loadSkillsSync,
  MarkdownMemoryStore,
  McpManager,
  Planner,
  WorkspaceMemory,
  type AgentMessage,
  type ContextEngineOptions,
  type AgentEvent,
  type AgentSessionRecord,
  type AgentSessionUsageSnapshot,
  type ContextCompactionMetadata,
  type CoreToolset,
  type EnvLlmClient,
  type McpConfig,
  type ReasoningEffort,
  type RequestContextMetadata,
  type RequestTokenEstimateMetadata,
  type SkillLoadResult,
  type TokenUsage
} from "../src/index.js";

const DEFAULT_PROMPT = "Calculate (123 + 456) * 789, then tell me the result.";
const CONFIG_PATH = ".agent-demo.json";
const SESSION_DIR = ".agent-sessions";
const DEFAULT_SESSION_TITLE = "Untitled session";
const PLAN_DIRECTIVE = "#plan";

type CliOptions = {
  help: boolean;
  once: boolean;
  prompt: string | undefined;
};

type StreamDisplayState = {
  runStreamedText: boolean;
  streamNeedsNewline: boolean;
  turnStreamedThinking: boolean;
  turnStreamedText: boolean;
};

type SessionUsageSnapshot = AgentSessionUsageSnapshot;

type RunUserPromptOptions = {
  forcePlan?: boolean;
  planDirectiveSource?: string;
};

type FinishAgentResultOptions = {
  shouldNameSession?: boolean;
};

type DemoConfig = {
  provider?: string;
  model?: string;
  apiKey?: string;
  baseURL?: string;
  toolset?: CoreToolset;
  maxTurns?: number;
  reasoningEffort?: ReasoningEffort;
  compressionProvider?: string;
  compressionModel?: string;
  compressionApiKey?: string;
  compressionBaseURL?: string;
  summaryProvider?: string;
  summaryModel?: string;
  summaryApiKey?: string;
  summaryBaseURL?: string;
  tavilyApiKey?: string;
  dynamicCompression?: boolean;
  dynamicAutoSummarize?: boolean;
  dynamicTriggerTokens?: number;
  dynamicKeepRecentTokens?: number;
  dynamicMinMessages?: number;
  contextWindowTokens?: number;
  reservedOutputTokens?: number;
  keepRecentTokens?: number;
  maxToolResultTokens?: number;
  compactionThresholdRatio?: number;
  summarizeHistory?: boolean;
  memory?: boolean;
  memoryPath?: string;
  maxMemoryResults?: number;
  phaseSummary?: boolean;
  skillPaths?: string;
  mcpConfigPath?: string;
};

type DemoConfigKey = keyof DemoConfig;

const CONFIG_KEY_HELP: Record<DemoConfigKey, string> = {
  provider: "main provider: openai-responses, openai-chat, anthropic",
  model: "main model name",
  apiKey: "main provider API key, stored locally if saved",
  baseURL: "main provider base URL",
  toolset: "basic, files, shell, web, all",
  maxTurns: "positive integer max turns per prompt",
  reasoningEffort: "minimal, low, medium, high, xhigh",
  compressionProvider: "compression provider override",
  compressionModel: "compression model override",
  compressionApiKey: "compression provider API key",
  compressionBaseURL: "compression provider base URL",
  summaryProvider: "summary provider override for session titles",
  summaryModel: "summary model override for session titles",
  summaryApiKey: "summary provider API key",
  summaryBaseURL: "summary provider base URL",
  tavilyApiKey: "Tavily API key for the web_search tool",
  dynamicCompression: "true/false enable dynamic compression",
  dynamicAutoSummarize: "true/false enable background summary fallback",
  dynamicTriggerTokens: "positive integer trigger threshold",
  dynamicKeepRecentTokens: "positive integer recent context budget",
  dynamicMinMessages: "positive integer minimum stale messages",
  contextWindowTokens: "positive integer context window estimate",
  reservedOutputTokens: "positive integer reserved output budget",
  keepRecentTokens: "positive integer fallback compaction recent budget",
  maxToolResultTokens: "positive integer tool-result truncation budget",
  compactionThresholdRatio: "number between 0 and 1",
  summarizeHistory: "true/false use model handoff summaries",
  memory: "true/false enable long-term Markdown memory store",
  memoryPath: "path for long-term Markdown memory store",
  maxMemoryResults: "positive integer default search_memory result count",
  phaseSummary: "reserved; phase-summary memory is currently paused",
  skillPaths: "comma-separated skill roots; defaults to .singularity/skills",
  mcpConfigPath: "path to a JSON MCP config file"
};

class SessionUsageLedger {
  private providerUsage: TokenUsage = {};
  private providerUsageTurns: number;
  private usageUnavailableCalls: number;
  private assistantTurns: number;
  private compactions: number;
  private latestContext: RequestContextMetadata | undefined;

  constructor(initial: SessionUsageSnapshot = { assistantTurns: 0, compactions: 0, usageUnavailableCalls: 0 }) {
    this.providerUsage = initial.providerUsage ?? {};
    this.providerUsageTurns = initial.providerUsage ? 1 : 0;
    this.usageUnavailableCalls = initial.usageUnavailableCalls;
    this.assistantTurns = initial.assistantTurns;
    this.compactions = initial.compactions;
    this.latestContext = initial.latestContext;
  }

  recordTurn(event: Extract<AgentEvent, { type: "turn_end" }>): SessionUsageSnapshot {
    this.assistantTurns += 1;
    this.latestContext = event.context;
    this.recordCompactionUsage(event.context?.compaction);
    this.recordDynamicCompressionUsage(event.context?.dynamicCompression);

    if (event.message.usage) {
      this.recordProviderUsage(event.message.usage);
    } else {
      this.usageUnavailableCalls += 1;
    }

    return this.snapshot();
  }

  recordManualCompaction(context: RequestContextMetadata | undefined): SessionUsageSnapshot {
    this.latestContext = context;
    this.recordCompactionUsage(context?.compaction);
    return this.snapshot();
  }

  recordMemorySummary(event: Extract<AgentEvent, { type: "memory_summary_end" }>): SessionUsageSnapshot {
    return this.recordSummaryUsage(event.usage);
  }

  recordSessionTitle(usage: TokenUsage | undefined): SessionUsageSnapshot {
    return this.recordSummaryUsage(usage);
  }

  snapshot(): SessionUsageSnapshot {
    return {
      assistantTurns: this.assistantTurns,
      compactions: this.compactions,
      usageUnavailableCalls: this.usageUnavailableCalls,
      latestContext: this.latestContext,
      providerUsage: this.providerUsageTurns > 0 ? this.providerUsage : undefined
    };
  }

  private recordCompactionUsage(compaction: ContextCompactionMetadata | undefined): void {
    if (!compaction) {
      return;
    }

    this.compactions += 1;
    const summaryCall = compaction.summaryCall;
    if (!summaryCall) {
      return;
    }
    if (summaryCall.responseUsage) {
      this.recordProviderUsage(summaryCall.responseUsage);
      return;
    }
    this.usageUnavailableCalls += 1;
  }

  private recordDynamicCompressionUsage(dynamicCompression: RequestContextMetadata["dynamicCompression"]): void {
    const summaryCall = dynamicCompression?.summaryCall;
    if (!summaryCall) {
      return;
    }

    if (summaryCall.responseUsage) {
      this.recordProviderUsage(summaryCall.responseUsage);
      return;
    }

    this.usageUnavailableCalls += 1;
  }

  private recordProviderUsage(usage: TokenUsage): void {
    this.providerUsageTurns += 1;
    this.providerUsage = addTokenUsage(this.providerUsage, usage);
  }

  private recordSummaryUsage(usage: TokenUsage | undefined): SessionUsageSnapshot {
    if (usage) {
      this.recordProviderUsage(usage);
    } else {
      this.usageUnavailableCalls += 1;
    }
    return this.snapshot();
  }
}

const cli = parseCliOptions(process.argv.slice(2));

if (cli.help) {
  printUsage();
  process.exit(0);
}

let demoConfig = loadDemoConfig();
let llm = createMainLlmClient();
let compressionLlm = createCompressionLlmClient(llm);
let summaryLlm = createSummaryLlmClient(llm);
let toolset = resolveToolset();
let maxTurns = resolveMaxTurns();
let displayState = createDisplayState();
let usageLedger = new SessionUsageLedger();
let exchangeCount = 0;
let workspaceMemory = new WorkspaceMemory();
let planner = new Planner();
let loadedSkills = loadDemoSkills();
let mcpManager: McpManager | undefined = await createDemoMcpManager();
let activeSession: AgentSessionRecord | undefined;
let sessions = new AgentSessionStore({ dir: SESSION_DIR });
let sessionPersistenceEnabled = false;
let agent = createDemoAgent();
let sessionWriteQueue: Promise<void> = Promise.resolve();
const demoBackgroundTasks = new Set<Promise<void>>();

if (cli.once || !stdin.isTTY) {
  renderBanner("one-shot");
  const ok = await runUserPrompt(cli.prompt ?? DEFAULT_PROMPT);
  await agent.close();
  if (!ok) {
    process.exitCode = 1;
  }
} else {
  sessionPersistenceEnabled = true;
  await restoreActiveSession();
  renderBanner("tui");
  if (cli.prompt) {
    await runUserPrompt(cli.prompt);
  }
  await runTui();
}

function createDemoAgent(): Agent {
  return new Agent({
    llm: llm.llm,
    model: llm.model,
    compressionLlm: compressionLlm?.llm,
    compressionModel: compressionLlm?.model,
    reasoning: resolveReasoning(),
    systemPrompt: "You are a concise assistant. Use tools when useful, then answer the user.",
    tools: createCoreTools({
      rootDir: process.cwd(),
      toolset,
      web: { apiKey: demoConfig.tavilyApiKey }
    }),
    skills: loadedSkills,
    mcp: mcpManager,
    memory: {
      workspace: workspaceMemory,
      store: createDemoMemoryStore(),
      maxMemoryResults: demoConfig.maxMemoryResults,
      phaseSummary: createDemoPhaseSummaryConfig()
    },
    planning: {
      planner
    },
    history: activeSession?.messages,
    context: buildContextOptions(),
    onEvent(event) {
      if (event.type === "turn_start") {
        displayState.turnStreamedThinking = false;
        displayState.turnStreamedText = false;
        console.log(`\n[turn ${event.turn}]`);
      }
      if (event.type === "assistant_delta") {
        displayState.runStreamedText = true;
        displayState.streamNeedsNewline = true;
        if (!displayState.turnStreamedText && displayState.turnStreamedThinking) {
          console.log("\n[answer]");
        }
        displayState.turnStreamedText = true;
        stdout.write(event.delta);
      }
      if (event.type === "thinking_delta") {
        if (!displayState.turnStreamedThinking) {
          displayState.turnStreamedThinking = true;
          console.log("\n[thinking]");
        }
        displayState.streamNeedsNewline = true;
        stdout.write(event.delta);
      }
      if (event.type === "tool_start") {
        finishStreamingLine();
        console.log(`[tool] ${event.toolCall.name} ${JSON.stringify(event.toolCall.arguments)}`);
      }
      if (event.type === "tool_end") {
        finishStreamingLine();
        console.log(`[observation] ${event.result.content}`);
      }
      if (event.type === "turn_end") {
        finishStreamingLine();
        const snapshot = usageLedger.recordTurn(event);
        console.log(formatTurnTokenUsage(event, snapshot));
        printCompactionDetails(event.context?.compaction);
      }
      if (event.type === "memory_summary_start") {
        finishStreamingLine();
        console.log(`[memory:summary] model=${event.model} messages=${event.messageCount} store=${event.storePath}`);
      }
      if (event.type === "memory_summary_end") {
        finishStreamingLine();
        const snapshot = usageLedger.recordMemorySummary(event);
        console.log(
          [
            `[memory:summary] ${event.action}=${event.entry.id}`,
            `tokens=${formatTokenCount(event.summaryTokens)}`,
            `chars=${formatTokenCount(event.summaryChars)}`,
            formatProviderUsage("provider", event.usage),
            formatProviderUsage("session", snapshot.providerUsage)
          ].join("; ")
        );
      }
      if (event.type === "memory_summary_error") {
        finishStreamingLine();
        console.error(`[memory:summary:error] model=${event.model} store=${event.storePath} ${event.error}`);
      }
    }
  });
}

async function runTui(): Promise<void> {
  const rl = createInterface({ input: stdin, output: stdout });
  rl.on("SIGINT", () => {
    console.log("\n[exit]");
    rl.close();
  });

  try {
    while (true) {
      console.log();
      console.log(formatSessionStatus());
      const input = (await rl.question("you > ")).trim();
      if (!input) {
        continue;
      }

      const handled = await handleCommand(input);
      if (handled === "exit") {
        break;
      }
      if (handled === "handled") {
        continue;
      }

      await runUserPrompt(input);
    }
  } finally {
    await agent.close();
    rl.close();
  }
}

async function runUserPrompt(input: string, options: RunUserPromptOptions = {}): Promise<boolean> {
  if (sessionPersistenceEnabled) {
    await ensureActiveSession();
  }
  const planDirective = extractPlanDirective(input);
  const forcePlan = options.forcePlan === true || planDirective.forcePlan;
  const prompt = planDirective.input;
  const shouldNameSession = shouldGenerateSessionTitle(activeSession);
  exchangeCount += 1;
  displayState = createDisplayState();
  console.log(`\n[user ${exchangeCount}] ${input}`);
  if (forcePlan) {
    console.log(`[plan] forced by ${options.planDirectiveSource ?? (planDirective.forcePlan ? PLAN_DIRECTIVE : "command")}`);
  }

  try {
    const runOptions = {
      ...(maxTurns === undefined ? {} : { maxTurns }),
      ...(forcePlan ? { forcePlan: true } : {})
    };
    const result = await agent.run(prompt, runOptions);
    await finishAgentResult(result, { shouldNameSession });
    return true;
  } catch (error) {
    finishStreamingLine();
    console.error(`[error] ${formatError(error)}`);
    return false;
  }
}

async function continueWithPlanApproval(note?: string): Promise<boolean> {
  if (sessionPersistenceEnabled) {
    await ensureActiveSession();
  }
  if (!planner.state) {
    console.log("[plan] no active plan to approve");
    return true;
  }

  displayState = createDisplayState();
  console.log("[plan] approved; continuing");
  try {
    const result = await agent.continueWithToolCall(
      {
        id: `call_plan_approve_${Date.now().toString(36)}`,
        name: APPROVE_PLAN_TOOL_NAME,
        arguments: note ? { note } : {}
      },
      maxTurns === undefined ? {} : { maxTurns }
    );
    await finishAgentResult(result);
    return true;
  } catch (error) {
    finishStreamingLine();
    console.error(`[error] ${formatError(error)}`);
    return false;
  }
}

async function finishAgentResult(result: Awaited<ReturnType<Agent["run"]>>, options: FinishAgentResultOptions = {}): Promise<void> {
  if (result.output && !displayState.runStreamedText) {
    console.log(`\n${result.output}`);
  } else {
    finishStreamingLine();
  }
  if (result.stoppedBy === "max_turns") {
    console.log(`[stop] reached maxTurns=${result.turns}`);
  }
  if (result.stoppedBy === "plan_review") {
    console.log("[plan] waiting for user review. Use /plan approve to approve and continue, or reply with changes.");
  }
  enqueueActiveSessionTitleAfterFirstRequest(options.shouldNameSession === true, result.messages);
  await saveActiveSession();
}

async function handleCommand(input: string): Promise<"exit" | "handled" | "message"> {
  if (input === "/exit" || input === "/quit" || input === "/q") {
    return "exit";
  }
  if (input === "/help") {
    printTuiHelp();
    return "handled";
  }
  if (input === "/usage") {
    console.log(formatSessionStatus());
    return "handled";
  }
  if (input === "/sessions") {
    await printSessions();
    return "handled";
  }
  if (input === "/session" || input.startsWith("/session ")) {
    await handleSessionCommand(input);
    return "handled";
  }
  if (input === "/config" || input.startsWith("/config ")) {
    await handleConfigCommand(input);
    return "handled";
  }
  if (input === "/memory" || input.startsWith("/memory ")) {
    await handleMemoryCommand(input);
    return "handled";
  }
  if (input === "/plan" || input.startsWith("/plan ")) {
    await handlePlanCommand(input);
    return "handled";
  }
  if (input === "/skill" || input.startsWith("/skill ") || input.startsWith("/skill:")) {
    await handleSkillCommand(input);
    return "handled";
  }
  if (input === "/notes") {
    printWorkspaceNotes();
    return "handled";
  }
  if (input === "/forget-notes") {
    workspaceMemory.clear();
    await saveActiveSession();
    console.log("[notes] workspace cleared");
    return "handled";
  }
  if (input === "/compact") {
    await runManualCompaction();
    return "handled";
  }
  if (input === "/clear") {
    console.clear();
    renderBanner("tui");
    return "handled";
  }
  if (input === "/new") {
    await createAndSwitchSession(DEFAULT_SESSION_TITLE);
    console.log(`[new] created session ${activeSession?.id}`);
    return "handled";
  }
  if (input.startsWith("/")) {
    console.log(`[unknown] ${input}. Type /help for commands.`);
    return "handled";
  }
  return "message";
}

async function restoreActiveSession(): Promise<void> {
  const result = await sessions.loadActiveSession();
  for (const warning of result.warnings) {
    console.warn(`[session:warn] ${warning}`);
  }
  if (!result.session) {
    if (result.warnings.length > 0) {
      await createAndSwitchSession(DEFAULT_SESSION_TITLE);
      console.log(`[session] started fresh session ${activeSession?.id}`);
    }
    return;
  }

  switchToSession(result.session);
  console.log(`[session] restored ${result.session.id} "${result.session.title}" messages=${result.session.messages.length}`);
}

async function ensureActiveSession(title?: string): Promise<AgentSessionRecord> {
  if (activeSession) {
    return activeSession;
  }
  return createAndSwitchSession(title);
}

async function createAndSwitchSession(title?: string): Promise<AgentSessionRecord> {
  const session = await enqueueSessionWrite(() => sessions.createSession({ title }));
  switchToSession(session);
  return session;
}

function switchToSession(session: AgentSessionRecord): void {
  activeSession = session;
  exchangeCount = session.exchangeCount;
  usageLedger = new SessionUsageLedger(session.usage);
  workspaceMemory = new WorkspaceMemory(session.workspace);
  planner = new Planner(session.planning?.plan);
  agent = createDemoAgent();
}

async function saveActiveSession(): Promise<void> {
  if (!sessionPersistenceEnabled || !activeSession) {
    return;
  }

  await enqueueSessionWrite(async () => {
    activeSession = await sessions.saveSession({
      ...activeSession!,
      exchangeCount,
      messages: [...agent.history],
      workspace: workspaceMemory.state,
      planning: {
        plan: planner.state
      },
      usage: usageLedger.snapshot()
    });
  });
}

function shouldGenerateSessionTitle(session: AgentSessionRecord | undefined): boolean {
  return sessionPersistenceEnabled && session?.exchangeCount === 0 && session.title === DEFAULT_SESSION_TITLE;
}

function enqueueActiveSessionTitleAfterFirstRequest(shouldNameSession: boolean, messages: readonly AgentMessage[]): void {
  if (!shouldNameSession || !activeSession) {
    return;
  }

  const sessionId = activeSession.id;
  const titleLlm = summaryLlm;
  const reasoning = resolveReasoning();
  const titleMessages = cloneAgentMessages(messages);
  finishStreamingLine();
  console.log(`[session:title] model=${titleLlm.model} messages=${titleMessages.length}`);
  trackDemoBackgroundTask((async () => {
    try {
      const result = await generateSessionTitle({
        llm: titleLlm.llm,
        model: titleLlm.model,
        messages: titleMessages,
        reasoning
      });
      const snapshot = await applyGeneratedSessionTitle(sessionId, result.title, result.message.usage);
      if (!snapshot) {
        console.log(`[session:title] skipped session=${sessionId}`);
        return;
      }
      console.log(
        [
          `[session:title] "${result.title}"`,
          `tokens=${formatTokenCount(result.titleTokens)}`,
          `chars=${formatTokenCount(result.titleChars)}`,
          formatProviderUsage("provider", result.message.usage),
          formatProviderUsage("session", snapshot.providerUsage)
        ].join("; ")
      );
    } catch (error) {
      console.error(`[session:title:error] model=${titleLlm.model} ${formatError(error)}`);
    }
  })());
}

async function applyGeneratedSessionTitle(
  sessionId: string,
  title: string,
  usage: TokenUsage | undefined
): Promise<SessionUsageSnapshot | undefined> {
  return enqueueSessionWrite(async () => {
    const session = await sessions.loadSession(sessionId);
    const isActive = activeSession?.id === sessionId;
    if (session.title !== DEFAULT_SESSION_TITLE || (isActive && activeSession?.title !== DEFAULT_SESSION_TITLE)) {
      return undefined;
    }

    const diskSnapshot = new SessionUsageLedger(session.usage).recordSessionTitle(usage);
    const activeSnapshot = isActive ? usageLedger.recordSessionTitle(usage) : undefined;
    const saved = await sessions.saveSession(
      {
        ...session,
        title,
        usage: diskSnapshot
      },
      { active: isActive }
    );

    if (isActive && activeSession?.id === sessionId) {
      activeSession = {
        ...activeSession,
        title: saved.title,
        usage: activeSnapshot ?? saved.usage
      };
    }

    return activeSnapshot ?? diskSnapshot;
  });
}

function trackDemoBackgroundTask(task: Promise<void>): void {
  demoBackgroundTasks.add(task);
  task.finally(() => {
    demoBackgroundTasks.delete(task);
  }).catch(() => undefined);
}

function enqueueSessionWrite<T>(operation: () => Promise<T>): Promise<T> {
  const task = sessionWriteQueue.then(operation, operation);
  sessionWriteQueue = task.then(() => undefined, () => undefined);
  return task;
}

function cloneAgentMessages(messages: readonly AgentMessage[]): AgentMessage[] {
  return JSON.parse(JSON.stringify(messages)) as AgentMessage[];
}

async function resetConversationToNewSession(reason: string): Promise<void> {
  await createAndSwitchSession(DEFAULT_SESSION_TITLE);
  console.log(`${reason}; switched to fresh session ${activeSession?.id}`);
}

async function handleSessionCommand(input: string): Promise<void> {
  const [, subcommand = "show", ...args] = input.split(/\s+/);
  const normalized = subcommand.toLowerCase();

  try {
    if (normalized === "show" || normalized === "status") {
      await printActiveSession();
      return;
    }
    if (normalized === "new") {
      const title = args.join(" ").trim() || undefined;
      await createAndSwitchSession(title);
      console.log(`[session] created ${activeSession?.id} "${activeSession?.title}"`);
      return;
    }
    if (normalized === "use" || normalized === "switch") {
      const id = args[0];
      if (!id) {
        console.log("[session] usage: /session use <id>");
        return;
      }
      const session = await sessions.loadSession(id);
      switchToSession(await sessions.saveSession(session));
      console.log(`[session] active ${activeSession?.id} "${activeSession?.title}"`);
      return;
    }
    if (normalized === "rename") {
      const title = args.join(" ").trim();
      if (!title) {
        console.log("[session] usage: /session rename <title>");
        return;
      }
      await ensureActiveSession();
      switchToSession(await sessions.renameSession(activeSession!.id, title));
      console.log(`[session] renamed ${activeSession?.id} "${activeSession?.title}"`);
      return;
    }
    if (normalized === "delete" || normalized === "rm") {
      const id = args[0];
      if (!id) {
        console.log("[session] usage: /session delete <id>");
        return;
      }
      const deletingActive = activeSession?.id === id;
      await sessions.deleteSession(id);
      if (deletingActive) {
        await createAndSwitchSession(DEFAULT_SESSION_TITLE);
        console.log(`[session] deleted ${id}; created ${activeSession?.id}`);
      } else {
        console.log(`[session] deleted ${id}`);
      }
      return;
    }

    console.log("[session] usage: /session show, /session new [title], /session use <id>, /session rename <title>, /session delete <id>");
  } catch (error) {
    console.error(`[session:error] ${formatError(error)}`);
  }
}

async function printSessions(): Promise<void> {
  const allSessions = await sessions.listSessions();
  if (allSessions.length === 0) {
    console.log("[sessions] none");
    return;
  }

  for (const session of allSessions) {
    const marker = session.id === activeSession?.id ? "*" : " ";
    console.log(
      `[sessions] ${marker} ${session.id} "${session.title}" updated=${session.updatedAt} exchanges=${session.exchangeCount} messages=${session.messageCount} notes=${session.workspaceNoteCount}`
    );
  }
}

async function printActiveSession(): Promise<void> {
  await ensureActiveSession();
  const session = activeSession!;
  console.log(`[session] id=${session.id}`);
  console.log(`[session] title=${session.title}`);
  console.log(`[session] file=${sessions.sessionPath(session.id)}`);
  console.log(`[session] created=${session.createdAt} updated=${session.updatedAt}`);
  console.log(`[session] exchanges=${exchangeCount} messages=${agent.history.length} workspace_notes=${workspaceMemory.state.notes.length}`);
}

async function handleConfigCommand(input: string): Promise<void> {
  try {
    await runConfigCommand(input);
  } catch (error) {
    console.error(`[config:error] ${formatError(error)}`);
  }
}

async function runConfigCommand(input: string): Promise<void> {
  const [, subcommand = "show", ...args] = input.split(/\s+/);
  const normalized = subcommand.toLowerCase();

  if (normalized === "show" || normalized === "list") {
    printConfig();
    return;
  }

  if (normalized === "help") {
    printConfigHelp();
    return;
  }

  if (normalized === "keys") {
    printConfigKeys();
    return;
  }

  if (normalized === "set") {
    const key = normalizeConfigKey(args[0]);
    const rawValue = args.slice(1).join(" ").trim();
    if (!key || rawValue.length === 0) {
      console.log("[config] usage: /config set <key> <value>");
      return;
    }
    demoConfig = {
      ...demoConfig,
      [key]: parseConfigValue(key, rawValue)
    };
    saveDemoConfig(demoConfig);
    await reloadRuntimeFromConfig();
    await resetConversationToNewSession("[config] saved and applied; conversation reset");
    console.log(`[config] set ${key}=${formatConfigValue(key, demoConfig[key])}`);
    return;
  }

  if (normalized === "unset") {
    const key = normalizeConfigKey(args[0]);
    if (!key) {
      console.log("[config] usage: /config unset <key>");
      return;
    }
    const next = { ...demoConfig };
    delete next[key];
    demoConfig = next;
    saveDemoConfig(demoConfig);
    await reloadRuntimeFromConfig();
    await resetConversationToNewSession("[config] saved and applied; conversation reset");
    console.log(`[config] unset ${key}`);
    return;
  }

  if (normalized === "save") {
    saveDemoConfig(demoConfig);
    console.log(`[config] saved ${CONFIG_PATH}`);
    return;
  }

  if (normalized === "reload") {
    demoConfig = loadDemoConfig();
    await reloadRuntimeFromConfig();
    await resetConversationToNewSession(`[config] reloaded ${CONFIG_PATH}; conversation reset`);
    return;
  }

  if (normalized === "reset") {
    demoConfig = {};
    saveDemoConfig(demoConfig);
    await reloadRuntimeFromConfig();
    await resetConversationToNewSession(`[config] reset ${CONFIG_PATH}; conversation reset`);
    return;
  }

  console.log(`[config] unknown subcommand: ${subcommand}. Try /config help.`);
}

async function reloadRuntimeFromConfig(): Promise<void> {
  llm = createMainLlmClient();
  compressionLlm = createCompressionLlmClient(llm);
  summaryLlm = createSummaryLlmClient(llm);
  toolset = resolveToolset();
  maxTurns = resolveMaxTurns();
  loadedSkills = loadDemoSkills();
  if (mcpManager) {
    await mcpManager.close();
  }
  mcpManager = await createDemoMcpManager();
  usageLedger = new SessionUsageLedger();
  exchangeCount = 0;
  workspaceMemory = new WorkspaceMemory();
  planner = new Planner();
  activeSession = undefined;
  agent = createDemoAgent();
}

async function handlePlanCommand(input: string): Promise<void> {
  const [, subcommand = "show", ...args] = input.split(/\s+/);
  const normalized = subcommand.toLowerCase();

  if (normalized === "show" || normalized === "status") {
    printPlanStatus();
    return;
  }

  if (normalized === "json") {
    console.log(JSON.stringify(planner.state ?? { plan: undefined }, null, 2));
    return;
  }

  if (normalized === "approve" || normalized === "ok") {
    await continueWithPlanApproval(args.join(" ").trim() || undefined);
    return;
  }

  if (normalized === "run") {
    const prompt = args.join(" ").trim();
    if (!prompt) {
      console.log("[plan] usage: /plan run <request>");
      return;
    }
    await runUserPrompt(prompt, { forcePlan: true, planDirectiveSource: "/plan run" });
    return;
  }

  if (normalized === "clear") {
    planner.clear();
    await saveActiveSession();
    console.log("[plan] cleared");
    return;
  }

  console.log("[plan] usage: /plan, /plan json, /plan approve [note], /plan run <request>, /plan clear");
}

async function handleMemoryCommand(input: string): Promise<void> {
  const [, subcommand = "show"] = input.split(/\s+/);
  const normalized = subcommand.toLowerCase();

  if (normalized === "show" || normalized === "status") {
    printMemoryStatus();
    return;
  }

  if (normalized === "on") {
    demoConfig = { ...demoConfig, memory: true };
    saveDemoConfig(demoConfig);
    reloadRuntimeFromConfig();
    await resetConversationToNewSession("[memory] enabled; conversation reset");
    const store = createRequiredDemoMemoryStore();
    await store.ensureFile();
    console.log(`[memory] store=${store.path}`);
    return;
  }

  if (normalized === "off") {
    demoConfig = { ...demoConfig, memory: false };
    saveDemoConfig(demoConfig);
    reloadRuntimeFromConfig();
    await resetConversationToNewSession("[memory] long-term store disabled; conversation reset");
    return;
  }

  console.log("[memory] usage: /memory, /memory on, /memory off");
}

async function handleSkillCommand(input: string): Promise<void> {
  if (input === "/skill" || input === "/skill list") {
    printSkillStatus();
    return;
  }

  if (input === "/skill reload") {
    loadedSkills = loadDemoSkills();
    agent = createDemoAgent();
    printSkillStatus();
    return;
  }

  if (input.startsWith("/skill:")) {
    const rest = input.slice("/skill:".length).trim();
    const [name = "", ...args] = rest.split(/\s+/);
    await runExplicitSkill(name, args.join(" "));
    return;
  }

  console.log("[skill] usage: /skill, /skill reload, /skill:<name> [args]");
}

function printSkillStatus(): void {
  console.log(`[skill] loaded=${loadedSkills.skills.length} diagnostics=${loadedSkills.diagnostics.length}`);
  for (const skill of loadedSkills.skills) {
    const hidden = skill.disableModelInvocation ? " hidden" : "";
    console.log(`[skill] ${skill.name}${hidden} ${skill.filePath} - ${skill.description}`);
  }
  for (const diagnostic of loadedSkills.diagnostics) {
    console.log(`[skill:${diagnostic.level}] ${diagnostic.path ?? ""} ${diagnostic.message}`.trim());
  }
}

async function runExplicitSkill(name: string, args: string): Promise<void> {
  const skill = loadedSkills.skills.find((candidate) => candidate.name === name);
  if (!skill) {
    console.log(`[skill] not found: ${name}`);
    return;
  }
  const body = await readFile(skill.filePath, "utf8");
  const prompt = [
    "Use this skill for the next task.",
    "",
    formatSkillInvocation(skill, body, args),
    "",
    args ? `User arguments: ${args}` : "Run the skill according to its instructions."
  ].join("\n");
  await runUserPrompt(prompt);
}

async function runManualCompaction(): Promise<void> {
  if (agent.history.length === 0) {
    console.log("[compact] no conversation history yet");
    return;
  }

  finishStreamingLine();
  console.log("[compact] summarizing current history...");
  try {
    const result = await agent.compactHistory();
    const snapshot = usageLedger.recordManualCompaction(result.context);
    if (!result.compacted || !result.context?.compaction) {
      console.log("[compact] skipped");
      return;
    }
    printCompactionDetails(result.context.compaction);
    console.log(`[compact] complete; messages=${result.messages.length}; ${formatProviderUsage("session", snapshot.providerUsage)}`);
    await saveActiveSession();
  } catch (error) {
    console.error(`[compact:error] ${formatError(error)}`);
  }
}

function renderBanner(mode: "tui" | "one-shot"): void {
  console.log("Singularity agent demo");
  console.log(`mode=${mode} provider=${llm.provider} model=${llm.model}`);
  if (compressionLlm) {
    console.log(`compression_provider=${compressionLlm.provider} compression_model=${compressionLlm.model}`);
  }
  console.log(`phase_summary=paused summary_provider=${summaryLlm.provider} summary_model=${summaryLlm.model}`);
  console.log(`toolset=${toolset} maxTurns=${maxTurns ?? "default"} skills=${loadedSkills.skills.length} mcp_tools=${mcpManager?.getToolInfos().length ?? 0}`);
  if (mode === "tui") {
    console.log(`session=${activeSession ? `${activeSession.id} "${activeSession.title}"` : "(new on first message)"}`);
  }
  console.log(
    `workspace_notes=${workspaceMemory.state.notes.length} plan=${formatPlanSummary()} memory_store=${
      demoConfig.memory ? createRequiredDemoMemoryStore().path : "off"
    } phase_memory=off`
  );
  if (mode === "tui") {
    console.log("commands=/help /usage /sessions /session /config /memory /plan /skill /notes /compact /new /clear /exit");
  }
}

function printUsage(): void {
  console.log(`Usage:
  npm run demo
  npm run demo -- "Ask an initial question"
  npm run demo -- --once "Run one prompt and exit"
  npm run demo -- "#plan Plan this request before executing"

Environment:
  LLM_PROVIDER, LLM_MODEL, LLM_API_KEY, LLM_BASE_URL
  AGENT_TOOLSET=basic|files|shell|web|all
  AGENT_MAX_TURNS=<positive integer>
  AGENT_REASONING_EFFORT=minimal|low|medium|high|xhigh
  AGENT_COMPRESSION_PROVIDER, AGENT_COMPRESSION_MODEL
  AGENT_SUMMARY_PROVIDER, AGENT_SUMMARY_MODEL
  AGENT_DYNAMIC_COMPRESSION=1

Interactive commands:
  /usage, /sessions, /session, /config, /memory, /plan, /skill, /notes, /compact, /new, /clear, /exit

Plan mode:
  Add #plan to any request, or use /plan run <request>, to force plan mode.
  Use /plan approve after reviewing the plan to approve it and continue immediately.`);
}

function printTuiHelp(): void {
  console.log(`Commands:
  /usage  show current provider token totals and latest context estimate
  /sessions list saved sessions
  /session show current session; subcommands: new/use/rename/delete
  /config show current config and change runtime settings
  /memory show, enable, or disable the long-term Markdown memory store
  /plan   show current structured plan; subcommands: json/approve/run/clear
  /skill  list skills; use /skill:<name> [args] to invoke one explicitly
  /notes  show current workspace notes
  /forget-notes clear current workspace notes
  /compact manually summarize and compact the current conversation history
  /new    create and switch to a fresh saved session
  /clear  clear the terminal view
  /exit   quit the demo`);
}

function printConfigHelp(): void {
  console.log(`Config commands:
  /config
  /config show
  /config keys
  /config set <key> <value>
  /config unset <key>
  /config reload
  /config reset

Examples:
  /config set provider openai-chat
  /config set model qwen3
  /config set baseURL http://localhost:11434/v1
  /config set apiKey ollama
  /config set toolset all
  /config set maxTurns 12
  /config set tavilyApiKey tvly-dev-...
  /config set dynamicCompression true
  /config set memoryPath .agent-memory/MEMORY.md
  /config set maxMemoryResults 8
  /config set skillPaths .singularity/skills,/tmp/agent-skills
  /config set mcpConfigPath .singularity/mcp.json
  /config set summaryModel gpt-4.1-mini
  /config set compressionModel gpt-4.1-mini`);
}

function printConfigKeys(): void {
  for (const key of Object.keys(CONFIG_KEY_HELP) as DemoConfigKey[]) {
    console.log(`${key.padEnd(26)} ${CONFIG_KEY_HELP[key]}`);
  }
}

function printConfig(): void {
  console.log(`[config] file=${CONFIG_PATH}`);
  console.log(`[config] provider=${llm.provider} model=${llm.model}`);
  if (compressionLlm) {
    console.log(`[config] compressionProvider=${compressionLlm.provider} compressionModel=${compressionLlm.model}`);
  } else {
    console.log("[config] compressionProvider=(main) compressionModel=(main)");
  }
  console.log(`[config] phaseSummary=paused summaryProvider=${summaryLlm.provider} summaryModel=${summaryLlm.model}`);
  console.log(`[config] toolset=${toolset} maxTurns=${maxTurns ?? "default"}`);
  console.log(`[config] skillPaths=${demoConfig.skillPaths ?? ".singularity/skills"} skills=${loadedSkills.skills.length}`);
  console.log(`[config] mcpConfigPath=${demoConfig.mcpConfigPath ?? "(none)"} mcpTools=${mcpManager?.getToolInfos().length ?? 0}`);
  const context = buildContextOptions();
  console.log(`[config] dynamicCompression=${context?.dynamicCompression ? "true" : "false"}`);
  printMemoryStatus();

  const saved = cleanConfig(demoConfig);
  const savedKeys = Object.keys(saved) as DemoConfigKey[];
  if (savedKeys.length === 0) {
    console.log("[config] saved overrides: none");
    return;
  }

  console.log("[config] saved overrides:");
  for (const key of savedKeys) {
    console.log(`  ${key}=${formatConfigValue(key, saved[key])}`);
  }
}

function formatConfigValue(key: DemoConfigKey, value: DemoConfig[DemoConfigKey]): string {
  if (value === undefined) {
    return "(unset)";
  }
  if (key === "apiKey" || key === "compressionApiKey" || key === "summaryApiKey" || key === "tavilyApiKey") {
    return maskSecret(String(value));
  }
  return String(value);
}

function maskSecret(value: string): string {
  if (value.length <= 8) {
    return "***";
  }
  return `${value.slice(0, 4)}...${value.slice(-4)}`;
}

function formatTurnTokenUsage(
  event: Extract<AgentEvent, { type: "turn_end" }>,
  snapshot: SessionUsageSnapshot
): string {
  return [
    `[tokens] exchange ${exchangeCount} turn ${event.turn}`,
    formatRequestContext(event.context),
    formatProviderUsage("provider", event.message.usage),
    formatProviderUsage("session", snapshot.providerUsage)
  ].join("; ");
}

function formatSessionStatus(): string {
  const snapshot = usageLedger.snapshot();
  const parts = [
    `[status] session=${activeSession?.id ?? "new"} "${activeSession?.title ?? "unsaved"}"`,
    `exchanges=${exchangeCount}`,
    `assistant_turns=${snapshot.assistantTurns}`,
    `compactions=${snapshot.compactions}`,
    `messages=${agent.history.length}`,
    `workspace_notes=${workspaceMemory.state.notes.length}`,
    `plan=${formatPlanSummary()}`,
    `memory_store=${demoConfig.memory ? "on" : "off"}`,
    formatProviderUsage("provider_total", snapshot.providerUsage),
    `latest_${formatRequestContext(snapshot.latestContext)}`
  ];

  if (snapshot.usageUnavailableCalls > 0) {
    parts.push(`usage_unavailable_calls=${snapshot.usageUnavailableCalls}`);
  }

  return parts.join(" | ");
}

function printCompactionDetails(compaction: ContextCompactionMetadata | undefined): void {
  if (!compaction) {
    return;
  }

  console.log(
    `[compact] mode=${compaction.mode} summary=${compaction.summarySource} messages=${compaction.messageCountBefore}->${compaction.messageCountAfter}`
  );
  console.log(formatTokenEstimate("[compact:decision]", compaction.decision));
  if (compaction.summaryCall) {
    console.log(
      [
        `[compact:summary-call] messages=${compaction.summaryCall.messageCount}`,
        formatTokenEstimate("request", compaction.summaryCall.request),
        formatProviderUsage("provider", compaction.summaryCall.responseUsage),
        `summary_tokens=${formatTokenCount(compaction.summaryCall.summaryTokens)}`,
        `summary_chars=${formatTokenCount(compaction.summaryCall.summaryChars)}`
      ].join("; ")
    );
  }
  console.log(formatTokenEstimate("[compact:after]", compaction.compacted));
}

function formatTokenEstimate(label: string, estimate: RequestTokenEstimateMetadata): string {
  const parts = [
    `${label} total=${formatTokenCount(estimate.totalTokens)}`,
    `source=${estimate.source}`,
    `system=${formatTokenCount(estimate.systemPromptTokens)}`,
    `messages=${formatTokenCount(estimate.messageTokens)}`,
    `tools=${formatTokenCount(estimate.toolTokens)}`,
    `heuristic=${formatTokenCount(estimate.heuristicTotalTokens)}`,
    formatTokenPart("provider_input", estimate.providerInputTokens),
    formatTokenPart("provider_output", estimate.providerOutputTokens),
    formatTokenPart("appended", estimate.appendedMessageTokens)
  ].filter((part): part is string => Boolean(part));

  return parts.join(" ");
}

function formatRequestContext(context: RequestContextMetadata | undefined): string {
  if (!context || context.estimatedInputTokens === undefined) {
    return "context=unavailable";
  }

  const notes: string[] = [];
  if (context.tokenEstimateSource) {
    notes.push(context.tokenEstimateSource);
  }
  if (context.compacted) {
    notes.push("compacted");
  }
  if (context.compactionSummarySource) {
    notes.push(`summary=${context.compactionSummarySource}`);
  }
  if (context.dynamicCompression?.applied) {
    notes.push(context.dynamicCompression.generated ? "dynamic=new" : "dynamic=reused");
  }
  if (context.compactionDecisionEstimatedInputTokens !== undefined) {
    const source = context.compactionDecisionTokenEstimateSource ? ` ${context.compactionDecisionTokenEstimateSource}` : "";
    notes.push(`decision=${formatTokenCount(context.compactionDecisionEstimatedInputTokens)}${source}`);
  }

  const suffix = notes.length > 0 ? ` (${notes.join(", ")})` : "";
  return `context=${formatTokenCount(context.estimatedInputTokens)}${suffix}`;
}

function createMainLlmClient(): EnvLlmClient {
  return createLlmClientFromEnv({
    provider: demoConfig.provider,
    model: demoConfig.model,
    apiKey: demoConfig.apiKey,
    baseURL: demoConfig.baseURL
  });
}

function createCompressionLlmClient(main: EnvLlmClient): EnvLlmClient | undefined {
  const provider = firstNonEmpty(demoConfig.compressionProvider, process.env.AGENT_COMPRESSION_PROVIDER);
  const model = firstNonEmpty(demoConfig.compressionModel, process.env.AGENT_COMPRESSION_MODEL);
  const usesMainProvider = resolvesToProvider(provider, main.provider);
  const apiKey = firstNonEmpty(demoConfig.compressionApiKey, usesMainProvider ? demoConfig.apiKey : undefined);
  const baseURL = firstNonEmpty(demoConfig.compressionBaseURL, usesMainProvider ? demoConfig.baseURL : undefined);
  if (!provider && !model && !apiKey && !baseURL) {
    return undefined;
  }

  return createLlmClientFromEnv({
    provider: provider ?? main.provider,
    model: model ?? main.model,
    apiKey,
    baseURL
  });
}

function createSummaryLlmClient(main: EnvLlmClient): EnvLlmClient {
  const provider = firstNonEmpty(demoConfig.summaryProvider, process.env.AGENT_SUMMARY_PROVIDER);
  const model = firstNonEmpty(demoConfig.summaryModel, process.env.AGENT_SUMMARY_MODEL);
  const usesMainProvider = resolvesToProvider(provider, main.provider);
  const apiKey = firstNonEmpty(demoConfig.summaryApiKey, process.env.AGENT_SUMMARY_API_KEY, usesMainProvider ? demoConfig.apiKey : undefined);
  const baseURL = firstNonEmpty(demoConfig.summaryBaseURL, process.env.AGENT_SUMMARY_BASE_URL, usesMainProvider ? demoConfig.baseURL : undefined);

  return createLlmClientFromEnv({
    provider: provider ?? main.provider,
    model: model ?? main.model,
    apiKey,
    baseURL
  });
}

function createDemoMemoryStore(): MarkdownMemoryStore | undefined {
  return demoConfig.memory ? createRequiredDemoMemoryStore() : undefined;
}

function createRequiredDemoMemoryStore(): MarkdownMemoryStore {
  return new MarkdownMemoryStore({ path: resolveMemoryPath() });
}

function createDemoPhaseSummaryConfig(): false {
  // Paused while the phase-summary memory shape is being redesigned.
  return false;
}

function resolveMemoryPath(): string {
  return demoConfig.memoryPath ?? ".agent-memory/MEMORY.md";
}

function printMemoryStatus(): void {
  const store = createDemoMemoryStore();
  console.log(`[memory] workspace_notes=${workspaceMemory.state.notes.length}`);
  console.log(`[memory] long_term=${store ? "on" : "off"} store=${store?.path ?? "(disabled)"} maxResults=${demoConfig.maxMemoryResults ?? 5}`);
  console.log("[memory] phase_summary=paused store=(disabled)");
}

function printPlanStatus(): void {
  const state = planner.state;
  if (!state) {
    console.log("[plan] none");
    return;
  }
  console.log(`[plan] objective=${state.objective}`);
  console.log(
    `[plan] revision=${state.revision} review=${state.reviewStatus} current=${state.currentStepId ?? "(none)"} steps=${state.steps.length}`
  );
  if (state.reviewedAt) {
    console.log(`[plan] reviewedAt=${state.reviewedAt}`);
  }
  for (const step of state.steps) {
    const current = step.id === state.currentStepId ? " current" : "";
    console.log(`[plan] ${step.id} status=${step.status}${current} title=${step.title}`);
    if (step.evidence && step.evidence.length > 0) {
      console.log(`[plan] ${step.id} evidence=${step.evidence.join(" | ")}`);
    }
  }
}

function formatPlanSummary(): string {
  const state = planner.state;
  if (!state) {
    return "none";
  }
  const counts = new Map<string, number>();
  for (const step of state.steps) {
    counts.set(step.status, (counts.get(step.status) ?? 0) + 1);
  }
  const status = [...counts.entries()].map(([key, value]) => `${key}:${value}`).join(",");
  return `rev${state.revision} review=${state.reviewStatus}${state.currentStepId ? ` current=${state.currentStepId}` : ""}${
    status ? ` ${status}` : ""
  }`;
}

function extractPlanDirective(input: string): { input: string; forcePlan: boolean } {
  const pattern = /(^|\s)#plan(?=\s|$)/i;
  if (!pattern.test(input)) {
    return { input, forcePlan: false };
  }
  const normalized = input.replace(pattern, "$1").replace(/\s{2,}/g, " ").trim();
  return {
    input: normalized || input.trim(),
    forcePlan: true
  };
}

function printWorkspaceNotes(): void {
  const notes = workspaceMemory.state.notes;
  if (notes.length === 0) {
    console.log("[notes] none");
    return;
  }
  for (const note of notes) {
    console.log(`[notes] ${note.id} kind=${note.kind} updated=${note.updatedAt}`);
    console.log(note.content);
  }
}

function resolvesToProvider(value: string | undefined, provider: EnvLlmClient["provider"]): boolean {
  if (!value) {
    return true;
  }

  const normalized = value.trim().toLowerCase();
  if (provider === "openai-responses") {
    return normalized === "openai" || normalized === "responses" || normalized === "openai-responses";
  }
  if (provider === "openai-chat") {
    return (
      normalized === "chat" ||
      normalized === "openai-chat" ||
      normalized === "openai-compatible" ||
      normalized === "openai-completions" ||
      normalized === "chat-completions"
    );
  }
  return normalized === "anthropic" || normalized === "anthropic-messages" || normalized === "claude";
}

function resolveToolset(): CoreToolset {
  return demoConfig.toolset ?? parseToolset(process.env.AGENT_TOOLSET);
}

function resolveMaxTurns(): number | undefined {
  return demoConfig.maxTurns ?? parseOptionalPositiveInteger(process.env.AGENT_MAX_TURNS, "AGENT_MAX_TURNS");
}

function resolveReasoning() {
  const effort = demoConfig.reasoningEffort ?? parseReasoningEffort(process.env.AGENT_REASONING_EFFORT, "AGENT_REASONING_EFFORT");
  return effort ? { effort, summary: "auto" as const } : undefined;
}

function buildContextOptions(): ContextEngineOptions | undefined {
  const context: ContextEngineOptions = {
    contextWindowTokens: demoConfig.contextWindowTokens,
    reservedOutputTokens: demoConfig.reservedOutputTokens,
    keepRecentTokens: demoConfig.keepRecentTokens,
    maxToolResultTokens: demoConfig.maxToolResultTokens,
    compactionThresholdRatio: demoConfig.compactionThresholdRatio,
    summarizeHistory: demoConfig.summarizeHistory,
    dynamicCompression: parseDynamicCompressionOptions()
  };

  const hasContextValue = Object.values(context).some((value) => value !== undefined && value !== false);
  return hasContextValue ? context : undefined;
}

function parseDynamicCompressionOptions(): ContextEngineOptions["dynamicCompression"] {
  const enabled = demoConfig.dynamicCompression ?? isEnabled(process.env.AGENT_DYNAMIC_COMPRESSION);
  if (!enabled) {
    return undefined;
  }

  return {
    enabled: true,
    autoSummarize: demoConfig.dynamicAutoSummarize ?? isEnabled(process.env.AGENT_DYNAMIC_AUTO_SUMMARIZE),
    triggerTokens: demoConfig.dynamicTriggerTokens ?? parseOptionalPositiveInteger(process.env.AGENT_DYNAMIC_TRIGGER_TOKENS, "AGENT_DYNAMIC_TRIGGER_TOKENS"),
    keepRecentTokens: demoConfig.dynamicKeepRecentTokens ?? parseOptionalPositiveInteger(process.env.AGENT_DYNAMIC_KEEP_RECENT_TOKENS, "AGENT_DYNAMIC_KEEP_RECENT_TOKENS"),
    minCompressMessages: demoConfig.dynamicMinMessages ?? parseOptionalPositiveInteger(process.env.AGENT_DYNAMIC_MIN_MESSAGES, "AGENT_DYNAMIC_MIN_MESSAGES")
  };
}

function loadDemoSkills(): SkillLoadResult {
  return loadSkillsSync({
    roots: parseListConfig(demoConfig.skillPaths) ?? [".singularity/skills"],
    cwd: process.cwd()
  });
}

async function createDemoMcpManager(): Promise<McpManager | undefined> {
  const configPath = demoConfig.mcpConfigPath;
  if (!configPath) {
    return undefined;
  }

  try {
    const raw = await readFile(configPath, "utf8");
    const parsed = JSON.parse(raw) as McpConfig;
    const manager = new McpManager({ config: parsed });
    await manager.start();
    for (const diagnostic of manager.getDiagnostics()) {
      console.warn(`[mcp:${diagnostic.level}] ${diagnostic.serverName} ${diagnostic.message}`);
    }
    return manager;
  } catch (error) {
    console.error(`[mcp:error] failed to load ${configPath}: ${formatError(error)}`);
    return undefined;
  }
}

function parseListConfig(value: string | undefined): string[] | undefined {
  if (!value) {
    return undefined;
  }
  const items = value.split(",").map((item) => item.trim()).filter(Boolean);
  return items.length > 0 ? items : undefined;
}

function resolvePhaseSummaryEnabled(): boolean {
  return false;
}

function loadDemoConfig(): DemoConfig {
  if (!existsSync(CONFIG_PATH)) {
    return {};
  }

  const raw = readFileSync(CONFIG_PATH, "utf8").trim();
  if (!raw) {
    return {};
  }

  const parsed = JSON.parse(raw) as unknown;
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error(`Invalid ${CONFIG_PATH}: expected a JSON object.`);
  }

  const config: DemoConfig = {};
  for (const [rawKey, rawValue] of Object.entries(parsed)) {
    const key = normalizeConfigKey(rawKey);
    if (!key) {
      continue;
    }
    config[key] = parseConfigValue(key, rawValue);
  }
  return config;
}

function saveDemoConfig(config: DemoConfig): void {
  const cleaned = cleanConfig(config);
  writeFileSync(CONFIG_PATH, `${JSON.stringify(cleaned, null, 2)}\n`, "utf8");
}

function cleanConfig(config: DemoConfig): DemoConfig {
  const cleaned: DemoConfig = {};
  for (const key of Object.keys(CONFIG_KEY_HELP) as DemoConfigKey[]) {
    const value = config[key];
    if (value !== undefined) {
      cleaned[key] = value as never;
    }
  }
  return cleaned;
}

function normalizeConfigKey(value: string | undefined): DemoConfigKey | undefined {
  if (!value) {
    return undefined;
  }

  const normalized = value.trim().replace(/[-_](.)/g, (_, char: string) => char.toUpperCase());
  return (Object.keys(CONFIG_KEY_HELP) as DemoConfigKey[]).find((key) => key.toLowerCase() === normalized.toLowerCase());
}

function parseConfigValue(key: DemoConfigKey, value: unknown): never {
  if (isStringKey(key)) {
    if (typeof value !== "string") {
      throw new Error(`Invalid ${key}: expected a string.`);
    }
    return value.trim() as never;
  }

  if (key === "toolset") {
    return parseToolset(String(value)) as never;
  }

  if (key === "reasoningEffort") {
    const effort = parseReasoningEffort(String(value), key);
    if (!effort) {
      throw new Error(`Invalid ${key}: expected minimal, low, medium, high, or xhigh.`);
    }
    return effort as never;
  }

  if (isBooleanKey(key)) {
    return parseBooleanConfig(value, key) as never;
  }

  if (key === "compactionThresholdRatio") {
    return parseRatioConfig(value, key) as never;
  }

  return parsePositiveIntegerConfig(value, key) as never;
}

function isStringKey(key: DemoConfigKey): boolean {
  return (
    key === "provider" ||
    key === "model" ||
    key === "apiKey" ||
    key === "baseURL" ||
    key === "compressionProvider" ||
    key === "compressionModel" ||
    key === "compressionApiKey" ||
    key === "compressionBaseURL" ||
    key === "summaryProvider" ||
    key === "summaryModel" ||
    key === "summaryApiKey" ||
    key === "summaryBaseURL" ||
    key === "tavilyApiKey" ||
    key === "memoryPath" ||
    key === "skillPaths" ||
    key === "mcpConfigPath"
  );
}

function isBooleanKey(key: DemoConfigKey): boolean {
  return key === "dynamicCompression" || key === "dynamicAutoSummarize" || key === "summarizeHistory" || key === "memory" || key === "phaseSummary";
}

function parseBooleanConfig(value: unknown, key: string): boolean {
  if (typeof value === "boolean") {
    return value;
  }
  if (typeof value === "string") {
    const normalized = value.trim().toLowerCase();
    if (normalized === "true" || normalized === "1" || normalized === "on" || normalized === "yes") {
      return true;
    }
    if (normalized === "false" || normalized === "0" || normalized === "off" || normalized === "no") {
      return false;
    }
  }
  throw new Error(`Invalid ${key}: expected true or false.`);
}

function parsePositiveIntegerConfig(value: unknown, key: string): number {
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed < 1) {
    throw new Error(`Invalid ${key}: expected a positive integer.`);
  }
  return parsed;
}

function parseRatioConfig(value: unknown, key: string): number {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed < 0 || parsed > 1) {
    throw new Error(`Invalid ${key}: expected a number between 0 and 1.`);
  }
  return parsed;
}

function isEnabled(value: string | undefined): boolean {
  if (!value) {
    return false;
  }

  const normalized = value.trim().toLowerCase();
  return normalized === "1" || normalized === "true" || normalized === "on" || normalized === "yes";
}

function firstNonEmpty(...values: Array<string | undefined>): string | undefined {
  return values.find((value) => value !== undefined && value.trim().length > 0);
}

function formatProviderUsage(label: string, usage: TokenUsage | undefined): string {
  if (!usage) {
    return `${label}=unavailable`;
  }

  const parts = [
    formatTokenPart("input", usage.inputTokens),
    formatTokenPart("output", usage.outputTokens),
    formatTokenPart("total", usage.totalTokens),
    formatTokenPart("cache_read", usage.cacheReadInputTokens),
    formatTokenPart("cache_create", usage.cacheCreationInputTokens)
  ].filter((part): part is string => Boolean(part));

  return parts.length > 0 ? `${label} ${parts.join(" ")}` : `${label}=unavailable`;
}

function formatTokenPart(label: string, value: number | undefined): string | undefined {
  return value === undefined ? undefined : `${label}=${formatTokenCount(value)}`;
}

function formatTokenCount(value: number): string {
  return Intl.NumberFormat("en-US").format(value);
}

function addTokenUsage(total: TokenUsage, next: TokenUsage): TokenUsage {
  const inputTokens = addTokenCount(total.inputTokens, next.inputTokens);
  const outputTokens = addTokenCount(total.outputTokens, next.outputTokens);
  const nextTotalTokens = next.totalTokens ?? addTokenCount(next.inputTokens, next.outputTokens);

  return {
    inputTokens,
    outputTokens,
    totalTokens: addTokenCount(total.totalTokens, nextTotalTokens),
    cacheReadInputTokens: addTokenCount(total.cacheReadInputTokens, next.cacheReadInputTokens),
    cacheCreationInputTokens: addTokenCount(total.cacheCreationInputTokens, next.cacheCreationInputTokens)
  };
}

function addTokenCount(left: number | undefined, right: number | undefined): number | undefined {
  if (left === undefined) {
    return right;
  }
  if (right === undefined) {
    return left;
  }
  return left + right;
}

function createDisplayState(): StreamDisplayState {
  return {
    runStreamedText: false,
    streamNeedsNewline: false,
    turnStreamedThinking: false,
    turnStreamedText: false
  };
}

function finishStreamingLine(): void {
  if (displayState.streamNeedsNewline) {
    console.log();
    displayState.streamNeedsNewline = false;
  }
}

function formatError(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function parseCliOptions(args: string[]): CliOptions {
  const promptParts: string[] = [];
  let help = false;
  let once = false;

  for (const arg of args) {
    if (arg === "--help" || arg === "-h") {
      help = true;
      continue;
    }
    if (arg === "--once") {
      once = true;
      continue;
    }
    promptParts.push(arg);
  }

  const prompt = promptParts.join(" ").trim();
  return { help, once, prompt: prompt || undefined };
}

function parseToolset(value: string | undefined): CoreToolset {
  if (!value) {
    return "basic";
  }
  if (value === "basic" || value === "files" || value === "shell" || value === "web" || value === "all") {
    return value;
  }
  throw new Error(`Invalid AGENT_TOOLSET: ${value}. Expected basic, files, shell, web, or all.`);
}

function parseReasoningEffort(value: string | undefined, name: string): ReasoningEffort | undefined {
  if (!value) {
    return undefined;
  }
  if (value === "minimal" || value === "low" || value === "medium" || value === "high" || value === "xhigh") {
    return value;
  }
  throw new Error(`Invalid ${name}: ${value}. Expected minimal, low, medium, high, or xhigh.`);
}

function parseOptionalPositiveInteger(value: string | undefined, name: string): number | undefined {
  if (!value) {
    return undefined;
  }
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed < 1) {
    throw new Error(`Invalid ${name}: ${value}. Expected a positive integer.`);
  }
  return parsed;
}
