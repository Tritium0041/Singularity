import type { PromptFragment } from "../context/types.js";

export const MEMORY_INSTRUCTIONS_FRAGMENT_ID = "memory-instructions";

export function buildMemoryInstructions(options: { hasWorkspace: boolean; hasStore: boolean }): PromptFragment | undefined {
  if (!options.hasWorkspace && !options.hasStore) {
    return undefined;
  }

  const lines = ["Memory tools are available. Use them proactively when they can materially improve the answer:"];
  if (options.hasStore) {
    lines.push("- Use search_memory before relying on remembered user preferences, project conventions, or prior solutions.");
  }
  if (options.hasWorkspace) {
    lines.push("- Use read_note when continuing a long task and you need the current task workspace.");
    lines.push("- Use write_note for important current-task state that should survive context compaction.");
  }
  if (options.hasStore) {
    lines.push("- Use store_memory only for durable preferences, project conventions, or reusable lessons.");
    lines.push("- Never store secrets or one-off temporary facts.");
  }
  lines.push("- Do not assume memory exists. Treat only tool results as retrieved memory evidence.");

  return {
    id: MEMORY_INSTRUCTIONS_FRAGMENT_ID,
    stable: true,
    content: lines.join("\n")
  };
}
