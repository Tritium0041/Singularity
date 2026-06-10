import {
  Agent,
  createCoreTools,
  createLlmClientFromEnv,
  type AgentEvent,
  type CoreToolset,
  type RequestContextMetadata,
  type TokenUsage
} from "../src/index.js";

let streamedText = false;
let streamNeedsNewline = false;
let turnStreamedThinking = false;
let turnStreamedText = false;
const llm = createLlmClientFromEnv();
const toolset = parseToolset(process.env.AGENT_TOOLSET);
const maxTurns = parseOptionalPositiveInteger(process.env.AGENT_MAX_TURNS, "AGENT_MAX_TURNS");

console.log(`[provider] ${llm.provider} ${llm.model}`);
console.log(`[toolset] ${toolset}`);
console.log(`[maxTurns] ${maxTurns ?? "default"}`);

const agent = new Agent({
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
      turnStreamedThinking = false;
      turnStreamedText = false;
      console.log(`\n[turn ${event.turn}]`);
    }
    if (event.type === "assistant_delta") {
      streamedText = true;
      streamNeedsNewline = true;
      if (!turnStreamedText && turnStreamedThinking) {
        console.log("\n[answer]");
      }
      turnStreamedText = true;
      process.stdout.write(event.delta);
    }
    if (event.type === "thinking_delta") {
      if (!turnStreamedThinking) {
        turnStreamedThinking = true;
        console.log("\n[thinking]");
      }
      streamNeedsNewline = true;
      process.stdout.write(event.delta);
    }
    if (event.type === "tool_start") {
      console.log(`[tool] ${event.toolCall.name} ${JSON.stringify(event.toolCall.arguments)}`);
    }
    if (event.type === "tool_end") {
      console.log(`[observation] ${event.result.content}`);
    }
    if (event.type === "turn_end") {
      if (streamNeedsNewline) {
        console.log();
        streamNeedsNewline = false;
      }
      console.log(formatTurnTokenUsage(event));
    }
  }
});

const prompt = process.argv.slice(2).join(" ") || "Calculate (123 + 456) * 789, then tell me the result.";
const result = await agent.run(prompt, maxTurns === undefined ? {} : { maxTurns });
if (result.output && !streamedText) {
  console.log(`\n${result.output}`);
} else if (streamNeedsNewline) {
  console.log();
}

function formatTurnTokenUsage(event: Extract<AgentEvent, { type: "turn_end" }>): string {
  return `[tokens] turn ${event.turn} ${formatRequestContext(event.context)}; ${formatProviderUsage(event.message.usage)}`;
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

function formatProviderUsage(usage: TokenUsage | undefined): string {
  if (!usage) {
    return "provider usage=unavailable";
  }

  const parts = [
    formatTokenPart("input", usage.inputTokens),
    formatTokenPart("output", usage.outputTokens),
    formatTokenPart("total", usage.totalTokens),
    formatTokenPart("cache_read", usage.cacheReadInputTokens),
    formatTokenPart("cache_create", usage.cacheCreationInputTokens)
  ].filter((part): part is string => Boolean(part));

  return parts.length > 0 ? `provider ${parts.join(" ")}` : "provider usage=unavailable";
}

function formatTokenPart(label: string, value: number | undefined): string | undefined {
  return value === undefined ? undefined : `${label}=${formatTokenCount(value)}`;
}

function formatTokenCount(value: number): string {
  return Intl.NumberFormat("en-US").format(value);
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
