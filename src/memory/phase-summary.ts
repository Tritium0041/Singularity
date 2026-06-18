import { estimateTextTokens } from "../context/token-estimator.js";
import type { LlmClient, LlmRequest } from "../llm/types.js";
import type { AgentMessage, AssistantMessage, ReasoningOptions } from "../types.js";
import { MarkdownMemoryStore, normalizeTags } from "./memory-store.js";
import type { MemoryEntry, MemoryEntrySource } from "./types.js";

export const PHASE_SUMMARY_TAG = "phase-summary";

export const PHASE_SUMMARY_SYSTEM_PROMPT = `You are a memory checkpoint writer for a local coding agent.

You are looking at the current workspace memory state plus a fork of the agent thread after one user request has finished. Produce the next complete workspace memory state for future continuation.

Include only durable, useful facts:
- What the user asked for in this phase.
- What the agent changed, decided, verified, or could not finish.
- Important file paths, commands, behaviors, and follow-up state.
- User preferences or project conventions that should matter later.

If a previous memory state is provided, revise it in place: preserve still-current facts, update stale facts, remove obsolete detail, and add new durable facts. If no previous state is provided, create the initial state.

Do not call tools. Do not include secrets, credentials, or irrelevant transcript detail. Return only the complete memory entry content.`;

const DEFAULT_PHASE_SUMMARY_PROMPT = `Write the next workspace memory state now.

Use concise Markdown bullets. Prefer concrete file paths, commands, decisions, and remaining tasks over narrative.`;

export type PhaseSummaryConfig = {
  llm: LlmClient;
  model: string;
  store: MarkdownMemoryStore;
  reasoning?: ReasoningOptions | false;
  prompt?: string;
  tags: string[];
  source?: MemoryEntrySource;
};

export type WritePhaseSummaryInput = PhaseSummaryConfig & {
  messages: readonly AgentMessage[];
  turns: number;
  stoppedBy: "final" | "max_turns" | "plan_review";
  signal?: AbortSignal;
};

export type PhaseSummaryResult = {
  entry: MemoryEntry;
  request: LlmRequest;
  message: AssistantMessage;
  action: "created" | "updated";
  previousEntry?: MemoryEntry;
  summaryTokens: number;
  summaryChars: number;
};

export async function writePhaseSummaryToMemory(input: WritePhaseSummaryInput): Promise<PhaseSummaryResult> {
  const previousEntry = await readCurrentPhaseSummary(input.store, input.tags);
  const request = buildPhaseSummaryRequest(input, previousEntry);
  const message = await input.llm.complete(request);
  const summary = normalizeSummary(message.content);
  const result = await input.store.upsertByTag({
    tag: PHASE_SUMMARY_TAG,
    content: summary,
    tags: normalizeTags([PHASE_SUMMARY_TAG, ...input.tags]),
    source: input.source ?? "agent"
  });

  return {
    entry: result.entry,
    request,
    message,
    action: result.created ? "created" : "updated",
    previousEntry,
    summaryTokens: estimateTextTokens(summary),
    summaryChars: summary.length
  };
}

async function readCurrentPhaseSummary(store: MarkdownMemoryStore, tags: readonly string[]): Promise<MemoryEntry | undefined> {
  const tagSet = new Set(normalizeTags([PHASE_SUMMARY_TAG, ...tags]));
  const entries = await store.list({ tag: PHASE_SUMMARY_TAG });
  return entries.find((entry) => entry.tags.every((tag) => tagSet.has(tag)) && [...tagSet].every((tag) => entry.tags.includes(tag))) ?? entries[0];
}

function buildPhaseSummaryRequest(input: WritePhaseSummaryInput, previousEntry: MemoryEntry | undefined): LlmRequest {
  return {
    model: input.model,
    systemPrompt: PHASE_SUMMARY_SYSTEM_PROMPT,
    messages: [
      {
        role: "user",
        content: previousEntry
          ? `<current_workspace_memory id="${previousEntry.id}" updated_at="${previousEntry.updatedAt}">\n${previousEntry.content}\n</current_workspace_memory>`
          : "<current_workspace_memory empty=\"true\" />"
      },
      ...input.messages.map(cloneAgentMessage),
      {
        role: "user",
        content: [
          input.prompt ?? DEFAULT_PHASE_SUMMARY_PROMPT,
          "",
          `<phase_metadata turns="${input.turns}" stopped_by="${input.stoppedBy}" />`
        ].join("\n")
      }
    ],
    tools: [],
    reasoning: input.reasoning,
    signal: input.signal
  };
}

function normalizeSummary(content: string): string {
  const summary = content.trim();
  if (!summary) {
    return "Phase summary unavailable: the summary model returned an empty response.";
  }
  return summary;
}

function cloneAgentMessage(message: AgentMessage): AgentMessage {
  return JSON.parse(JSON.stringify(message)) as AgentMessage;
}
