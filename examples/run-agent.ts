import { Agent, OpenAIResponsesClient, calculatorTool, mockWeatherTool } from "../src/index.js";

let streamedText = false;
let streamedThinking = false;

const agent = new Agent({
  llm: new OpenAIResponsesClient(),
  model: process.env.OPENAI_MODEL ?? "gpt-4.1-mini",
  reasoning: process.env.AGENT_REASONING_EFFORT
    ? {
        effort: process.env.AGENT_REASONING_EFFORT as "minimal" | "low" | "medium" | "high" | "xhigh",
        summary: "auto"
      }
    : undefined,
  systemPrompt: "You are a concise assistant. Use tools when useful, then answer the user.",
  tools: [calculatorTool, mockWeatherTool],
  onEvent(event) {
    if (event.type === "turn_start") {
      console.log(`\n[turn ${event.turn}]`);
    }
    if (event.type === "assistant_delta") {
      streamedText = true;
      process.stdout.write(event.delta);
    }
    if (event.type === "thinking_delta") {
      if (!streamedThinking) {
        streamedThinking = true;
        console.log("\n[thinking]");
      }
      process.stdout.write(event.delta);
    }
    if (event.type === "tool_call_delta") {
      console.log(`\n[tool args] ${event.toolName} ${event.argumentsText}`);
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
const result = await agent.run(prompt);
if (result.output && !streamedText) {
  console.log(`\n${result.output}`);
} else if (streamedText) {
  console.log();
}
