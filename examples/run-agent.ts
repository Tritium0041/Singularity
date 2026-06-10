import { stdin, stdout } from "node:process";
import { createInterface } from "node:readline/promises";
import {
  Agent,
  createCoreTools,
  createLlmClientFromEnv,
  type AgentEvent,
  type ContextCompactionMetadata,
  type CoreToolset,
  type RequestContextMetadata,
  type RequestTokenEstimateMetadata,
  type TokenUsage
} from "../src/index.js";

const DEFAULT_PROMPT = "Calculate (123 + 456) * 789, then tell me the result.";

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

type SessionUsageSnapshot = {
  assistantTurns: number;
  compactions: number;
  usageUnavailableCalls: number;
  latestContext?: RequestContextMetadata;
  providerUsage?: TokenUsage;
};

class SessionUsageLedger {
  private providerUsage: TokenUsage = {};
  private providerUsageTurns = 0;
  private usageUnavailableCalls = 0;
  private assistantTurns = 0;
  private compactions = 0;
  private latestContext: RequestContextMetadata | undefined;

  recordTurn(event: Extract<AgentEvent, { type: "turn_end" }>): SessionUsageSnapshot {
    this.assistantTurns += 1;
    this.latestContext = event.context;
    this.recordCompactionUsage(event.context?.compaction);

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

const llm = createLlmClientFromEnv();
const toolset = parseToolset(process.env.AGENT_TOOLSET);
const maxTurns = parseOptionalPositiveInteger(process.env.AGENT_MAX_TURNS, "AGENT_MAX_TURNS");
let displayState = createDisplayState();
let usageLedger = new SessionUsageLedger();
let exchangeCount = 0;
let agent = createDemoAgent();

if (cli.once || !stdin.isTTY) {
  renderBanner("one-shot");
  const ok = await runUserPrompt(cli.prompt ?? DEFAULT_PROMPT);
  if (!ok) {
    process.exitCode = 1;
  }
} else {
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
    reasoning: process.env.AGENT_REASONING_EFFORT
      ? {
          effort: process.env.AGENT_REASONING_EFFORT as "minimal" | "low" | "medium" | "high" | "xhigh",
          summary: "auto"
        }
      : undefined,
    systemPrompt: "You are a concise assistant. Use tools when useful, then answer the user.",
    tools: createCoreTools({ rootDir: process.cwd(), toolset }),
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
    usageLedger = new SessionUsageLedger();
    exchangeCount = 0;
    agent = createDemoAgent();
    console.log("[new] conversation and usage counters reset");
    return "handled";
  }
  if (input.startsWith("/")) {
    console.log(`[unknown] ${input}. Type /help for commands.`);
    return "handled";
  }
  return "message";
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
  } catch (error) {
    console.error(`[compact:error] ${formatError(error)}`);
  }
}

function renderBanner(mode: "tui" | "one-shot"): void {
  console.log("Singularity agent demo");
  console.log(`mode=${mode} provider=${llm.provider} model=${llm.model}`);
  console.log(`toolset=${toolset} maxTurns=${maxTurns ?? "default"}`);
  if (mode === "tui") {
    console.log("commands=/help /usage /compact /new /clear /exit");
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

Interactive commands:
  /usage, /compact, /new, /clear, /exit`);
}

function printTuiHelp(): void {
  console.log(`Commands:
  /usage  show current provider token totals and latest context estimate
  /compact manually summarize and compact the current conversation history
  /new    reset conversation history and usage counters
  /clear  clear the terminal view
  /exit   quit the demo`);
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
    `[status] exchanges=${exchangeCount}`,
    `assistant_turns=${snapshot.assistantTurns}`,
    `compactions=${snapshot.compactions}`,
    `messages=${agent.history.length}`,
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
  if (context.compactionDecisionEstimatedInputTokens !== undefined) {
    const source = context.compactionDecisionTokenEstimateSource ? ` ${context.compactionDecisionTokenEstimateSource}` : "";
    notes.push(`decision=${formatTokenCount(context.compactionDecisionEstimatedInputTokens)}${source}`);
  }

  const suffix = notes.length > 0 ? ` (${notes.join(", ")})` : "";
  return `context=${formatTokenCount(context.estimatedInputTokens)}${suffix}`;
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
