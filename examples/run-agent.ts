import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { stdin, stdout } from "node:process";
import { createInterface } from "node:readline/promises";
import {
  Agent,
  AgentSessionStore,
  createCoreTools,
  createLlmClientFromEnv,
  MarkdownMemoryStore,
  WorkspaceMemory,
  type ContextEngineOptions,
  type AgentEvent,
  type AgentSessionRecord,
  type AgentSessionUsageSnapshot,
  type ContextCompactionMetadata,
  type CoreToolset,
  type EnvLlmClient,
  type ReasoningEffort,
  type RequestContextMetadata,
  type RequestTokenEstimateMetadata,
  type TokenUsage
} from "../src/index.js";

const DEFAULT_PROMPT = "Calculate (123 + 456) * 789, then tell me the result.";
const CONFIG_PATH = ".agent-demo.json";
const SESSION_DIR = ".agent-sessions";

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
  maxMemoryResults: "positive integer default search_memory result count"
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
}

const cli = parseCliOptions(process.argv.slice(2));

if (cli.help) {
  printUsage();
  process.exit(0);
}

let demoConfig = loadDemoConfig();
let llm = createMainLlmClient();
let compressionLlm = createCompressionLlmClient(llm);
let toolset = resolveToolset();
let maxTurns = resolveMaxTurns();
let displayState = createDisplayState();
let usageLedger = new SessionUsageLedger();
let exchangeCount = 0;
let workspaceMemory = new WorkspaceMemory();
let activeSession: AgentSessionRecord | undefined;
let sessions = new AgentSessionStore({ dir: SESSION_DIR });
let sessionPersistenceEnabled = false;
let agent = createDemoAgent();

if (cli.once || !stdin.isTTY) {
  renderBanner("one-shot");
  const ok = await runUserPrompt(cli.prompt ?? DEFAULT_PROMPT);
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
    memory: {
      workspace: workspaceMemory,
      store: createDemoMemoryStore(),
      maxMemoryResults: demoConfig.maxMemoryResults
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
    rl.close();
  }
}

async function runUserPrompt(input: string): Promise<boolean> {
  if (sessionPersistenceEnabled) {
    await ensureActiveSession();
  }
  exchangeCount += 1;
  displayState = createDisplayState();
  console.log(`\n[user ${exchangeCount}] ${input}`);

  try {
    const result = await agent.run(input, maxTurns === undefined ? {} : { maxTurns });
    if (result.output && !displayState.runStreamedText) {
      console.log(`\n${result.output}`);
    } else {
      finishStreamingLine();
    }
    if (result.stoppedBy === "max_turns") {
      console.log(`[stop] reached maxTurns=${result.turns}`);
    }
    await saveActiveSession();
    return true;
  } catch (error) {
    finishStreamingLine();
    console.error(`[error] ${formatError(error)}`);
    return false;
  }
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
    await createAndSwitchSession("Untitled session");
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
      await createAndSwitchSession("Untitled session");
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
  const session = await sessions.createSession({ title });
  switchToSession(session);
  return session;
}

function switchToSession(session: AgentSessionRecord): void {
  activeSession = session;
  exchangeCount = session.exchangeCount;
  usageLedger = new SessionUsageLedger(session.usage);
  workspaceMemory = new WorkspaceMemory(session.workspace);
  agent = createDemoAgent();
}

async function saveActiveSession(): Promise<void> {
  if (!sessionPersistenceEnabled || !activeSession) {
    return;
  }

  activeSession = await sessions.saveSession({
    ...activeSession,
    exchangeCount,
    messages: [...agent.history],
    workspace: workspaceMemory.state,
    usage: usageLedger.snapshot()
  });
}

async function resetConversationToNewSession(reason: string): Promise<void> {
  await createAndSwitchSession("Untitled session");
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
        await createAndSwitchSession("Untitled session");
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
    reloadRuntimeFromConfig();
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
    reloadRuntimeFromConfig();
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
    reloadRuntimeFromConfig();
    await resetConversationToNewSession(`[config] reloaded ${CONFIG_PATH}; conversation reset`);
    return;
  }

  if (normalized === "reset") {
    demoConfig = {};
    saveDemoConfig(demoConfig);
    reloadRuntimeFromConfig();
    await resetConversationToNewSession(`[config] reset ${CONFIG_PATH}; conversation reset`);
    return;
  }

  console.log(`[config] unknown subcommand: ${subcommand}. Try /config help.`);
}

function reloadRuntimeFromConfig(): void {
  llm = createMainLlmClient();
  compressionLlm = createCompressionLlmClient(llm);
  toolset = resolveToolset();
  maxTurns = resolveMaxTurns();
  usageLedger = new SessionUsageLedger();
  exchangeCount = 0;
  workspaceMemory = new WorkspaceMemory();
  activeSession = undefined;
  agent = createDemoAgent();
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
  console.log(`toolset=${toolset} maxTurns=${maxTurns ?? "default"}`);
  if (mode === "tui") {
    console.log(`session=${activeSession ? `${activeSession.id} "${activeSession.title}"` : "(new on first message)"}`);
  }
  console.log(`workspace_notes=${workspaceMemory.state.notes.length} memory_store=${demoConfig.memory ? createRequiredDemoMemoryStore().path : "off"}`);
  if (mode === "tui") {
    console.log("commands=/help /usage /sessions /session /config /memory /notes /compact /new /clear /exit");
  }
}

function printUsage(): void {
  console.log(`Usage:
  npm run demo
  npm run demo -- "Ask an initial question"
  npm run demo -- --once "Run one prompt and exit"

Environment:
  LLM_PROVIDER, LLM_MODEL, LLM_API_KEY, LLM_BASE_URL
  AGENT_TOOLSET=basic|files|shell|web|all
  AGENT_MAX_TURNS=<positive integer>
  AGENT_REASONING_EFFORT=minimal|low|medium|high|xhigh
  AGENT_COMPRESSION_PROVIDER, AGENT_COMPRESSION_MODEL
  AGENT_DYNAMIC_COMPRESSION=1

Interactive commands:
  /usage, /sessions, /session, /config, /memory, /notes, /compact, /new, /clear, /exit`);
}

function printTuiHelp(): void {
  console.log(`Commands:
  /usage  show current provider token totals and latest context estimate
  /sessions list saved sessions
  /session show current session; subcommands: new/use/rename/delete
  /config show current config and change runtime settings
  /memory show, enable, or disable the long-term Markdown memory store
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
  console.log(`[config] toolset=${toolset} maxTurns=${maxTurns ?? "default"}`);
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
  if (key === "apiKey" || key === "compressionApiKey" || key === "tavilyApiKey") {
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

function createDemoMemoryStore(): MarkdownMemoryStore | undefined {
  return demoConfig.memory ? createRequiredDemoMemoryStore() : undefined;
}

function createRequiredDemoMemoryStore(): MarkdownMemoryStore {
  return new MarkdownMemoryStore({ path: resolveMemoryPath() });
}

function resolveMemoryPath(): string {
  return demoConfig.memoryPath ?? ".agent-memory/MEMORY.md";
}

function printMemoryStatus(): void {
  const store = createDemoMemoryStore();
  console.log(`[memory] workspace_notes=${workspaceMemory.state.notes.length}`);
  console.log(`[memory] long_term=${store ? "on" : "off"} store=${store?.path ?? "(disabled)"} maxResults=${demoConfig.maxMemoryResults ?? 5}`);
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
    key === "tavilyApiKey" ||
    key === "memoryPath"
  );
}

function isBooleanKey(key: DemoConfigKey): boolean {
  return key === "dynamicCompression" || key === "dynamicAutoSummarize" || key === "summarizeHistory" || key === "memory";
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
