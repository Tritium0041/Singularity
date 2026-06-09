import { Agent, createCoreTools, createLlmClientFromEnv, type CoreToolset } from "../src/index.js";

let streamedText = false;
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
      process.stdout.write(event.delta);
    }
    if (event.type === "tool_start") {
      console.log(`[tool] ${event.toolCall.name} ${JSON.stringify(event.toolCall.arguments)}`);
    }
    if (event.type === "tool_end") {
      console.log(`[observation] ${event.result.content}`);
    }
  }
});

const prompt = process.argv.slice(2).join(" ") || "Calculate (123 + 456) * 789, then tell me the result.";
const result = await agent.run(prompt, maxTurns === undefined ? {} : { maxTurns });
if (result.output && !streamedText) {
  console.log(`\n${result.output}`);
} else if (streamedText) {
  console.log();
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
