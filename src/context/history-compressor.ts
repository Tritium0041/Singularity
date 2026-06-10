import type { AgentMessage, AssistantMessage, ToolCall, ToolResultMessage } from "../types.js";
import { estimateMessagesTokens, estimateTextTokens } from "./token-estimator.js";
import type { ResolvedContextEngineOptions } from "./types.js";

export const CONTEXT_HANDOFF_SUMMARY_PROMPT = `You are performing a context checkpoint compaction. Create a handoff summary for the next model that will resume this task.

Include:
- Current progress and key decisions made.
- Important context, constraints, and user preferences.
- What remains to be done, with clear next steps.
- Critical data, examples, file paths, commands, or references needed to continue.

Do not call tools. Return only the summary.`;

export const CONTEXT_HANDOFF_SUMMARY_PREFIX = "Context checkpoint summary for the next model:";

type MessageTurn = {
  messages: AgentMessage[];
};

export class HistoryCompressor {
  constructor(private readonly options: ResolvedContextEngineOptions) {}

  truncateToolResults(messages: readonly AgentMessage[]): AgentMessage[] {
    return messages.map((message) => (message.role === "tool" ? this.truncateToolMessage(message) : message));
  }

  compact(messages: readonly AgentMessage[]): AgentMessage[] {
    const turns = splitIntoTurns(messages);
    if (turns.length <= 1) {
      return [...messages];
    }

    const keptTurns: MessageTurn[] = [];
    let keptTokens = 0;
    for (let index = turns.length - 1; index >= 0; index -= 1) {
      const turn = turns[index];
      if (!turn) {
        continue;
      }
      const turnTokens = estimateMessagesTokens(turn.messages);
      if (keptTurns.length > 0 && keptTokens + turnTokens > this.options.keepRecentTokens) {
        break;
      }
      keptTurns.unshift(turn);
      keptTokens += turnTokens;
    }

    if (keptTurns.length === turns.length) {
      return [...messages];
    }

    const compactedMessages = turns.slice(0, turns.length - keptTurns.length).flatMap((turn) => turn.messages);
    const keptMessages = keptTurns.flatMap((turn) => turn.messages);
    if (!this.options.summarizeHistory || compactedMessages.length === 0) {
      return keptMessages;
    }

    return [buildSummaryMessage(compactedMessages), ...keptMessages];
  }

  compactWithHandoffSummary(messages: readonly AgentMessage[], summary: string): AgentMessage[] {
    const selectedUserMessages = selectRecentUserMessages(messages, this.options.maxHandoffUserMessageTokens);
    const summaryText = normalizeHandoffSummary(summary);
    return [
      ...selectedUserMessages.map((content) => ({ role: "user" as const, content })),
      {
        role: "user",
        content: `${CONTEXT_HANDOFF_SUMMARY_PREFIX}\n${summaryText}`
      }
    ];
  }

  private truncateToolMessage(message: ToolResultMessage): ToolResultMessage {
    const maxChars = Math.max(1, this.options.maxToolResultTokens * 4);
    if (message.content.length <= maxChars) {
      return message;
    }

    const direction = message.toolName === "execute_command" ? "tail" : "head";
    return {
      ...message,
      content: truncateByChars(message.content, maxChars, direction)
    };
  }
}

function splitIntoTurns(messages: readonly AgentMessage[]): MessageTurn[] {
  const turns: MessageTurn[] = [];
  let current: AgentMessage[] = [];

  for (const message of messages) {
    if (message.role === "user" && current.length > 0) {
      turns.push({ messages: current });
      current = [];
    }
    current.push(message);
  }

  if (current.length > 0) {
    turns.push({ messages: current });
  }

  return turns;
}

function buildSummaryMessage(messages: readonly AgentMessage[]): AgentMessage {
  const goals: string[] = [];
  const progress: string[] = [];
  const toolWork: string[] = [];
  const errors: string[] = [];

  for (const message of messages) {
    if (message.role === "user") {
      goals.push(truncateInline(message.content, 220));
      continue;
    }

    if (message.role === "assistant") {
      summarizeAssistantMessage(message, progress, toolWork);
      continue;
    }

    summarizeToolMessage(message, toolWork, errors);
  }

  return {
    role: "user",
    content: [
      "<context_summary>",
      "Goal:",
      ...formatBullets(goals),
      "",
      "Earlier progress:",
      ...formatBullets(progress),
      "",
      "Tool work:",
      ...formatBullets(toolWork),
      "",
      "Important errors:",
      ...formatBullets(errors),
      "</context_summary>"
    ].join("\n")
  };
}

function selectRecentUserMessages(messages: readonly AgentMessage[], maxTokens: number): string[] {
  const selected: string[] = [];
  let remainingTokens = Math.max(0, maxTokens);
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    if (message?.role !== "user" || isHandoffSummaryMessage(message.content)) {
      continue;
    }
    if (remainingTokens === 0) {
      break;
    }

    const messageTokens = estimateTextTokens(message.content);
    if (messageTokens <= remainingTokens) {
      selected.unshift(message.content);
      remainingTokens -= messageTokens;
      continue;
    }

    const truncated = truncateTextToTokenBudget(message.content, remainingTokens);
    if (truncated.trim()) {
      selected.unshift(truncated);
    }
    break;
  }
  return selected;
}

function normalizeHandoffSummary(summary: string): string {
  const trimmed = summary.trim();
  if (!trimmed) {
    return "(no summary available)";
  }
  return isHandoffSummaryMessage(trimmed) ? trimmed.slice(CONTEXT_HANDOFF_SUMMARY_PREFIX.length).trimStart() : trimmed;
}

function isHandoffSummaryMessage(content: string): boolean {
  return content.startsWith(`${CONTEXT_HANDOFF_SUMMARY_PREFIX}\n`);
}

function truncateTextToTokenBudget(text: string, maxTokens: number): string {
  const maxChars = Math.max(0, maxTokens * 4);
  if (maxChars === 0) {
    return "";
  }
  if (text.length <= maxChars) {
    return text;
  }
  const marker = "[ContextEngine truncated user message for handoff.]";
  const bodyChars = Math.max(0, maxChars - marker.length - 1);
  if (bodyChars === 0) {
    return marker.slice(0, maxChars);
  }
  return `${text.slice(-bodyChars)}\n${marker}`;
}

function summarizeAssistantMessage(message: AssistantMessage, progress: string[], toolWork: string[]): void {
  if (message.content.trim()) {
    progress.push(truncateInline(message.content, 220));
  }

  for (const toolCall of message.toolCalls ?? []) {
    toolWork.push(`${toolCall.name}${formatToolArguments(toolCall)}`);
  }
}

function summarizeToolMessage(message: ToolResultMessage, toolWork: string[], errors: string[]): void {
  const status = message.isError ? "error" : "result";
  const summary = `${message.toolName} ${status}: ${truncateInline(message.content, 180)}`;
  toolWork.push(summary);
  if (message.isError) {
    errors.push(summary);
  }
}

function formatToolArguments(toolCall: ToolCall): string {
  if (!isRecord(toolCall.arguments)) {
    const text = truncateInline(JSON.stringify(toolCall.arguments ?? {}), 160);
    return text ? ` ${text}` : "";
  }

  const preferredKeys = ["path", "command", "url", "query", "expression", "workdir", "offset", "limit"];
  const entries: string[] = [];
  for (const key of preferredKeys) {
    if (!(key in toolCall.arguments)) {
      continue;
    }
    const value = toolCall.arguments[key];
    entries.push(`${key}=${formatArgumentValue(value)}`);
  }

  if (entries.length === 0 && "content" in toolCall.arguments && typeof toolCall.arguments.content === "string") {
    entries.push(`content=${toolCall.arguments.content.length} chars`);
  }

  return entries.length > 0 ? ` ${entries.join(" ")}` : "";
}

function formatArgumentValue(value: unknown): string {
  if (typeof value === "string") {
    return truncateInline(value, 120);
  }
  return truncateInline(JSON.stringify(value ?? null), 120);
}

function formatBullets(values: string[]): string[] {
  const filtered = values.map((value) => value.trim()).filter(Boolean).slice(0, 8);
  return filtered.length > 0 ? filtered.map((value) => `- ${value}`) : ["- none"];
}

function truncateByChars(text: string, maxChars: number, direction: "head" | "tail"): string {
  const marker = `[ContextEngine truncated tool result; showing ${direction} of ${text.length} chars.]`;
  const bodyChars = Math.max(1, maxChars - marker.length - 2);
  if (direction === "tail") {
    return `${marker}\n${text.slice(-bodyChars)}`;
  }
  return `${text.slice(0, bodyChars)}\n${marker}`;
}

function truncateInline(text: string | undefined, maxChars: number): string {
  const normalized = (text ?? "").replace(/\s+/g, " ").trim();
  if (normalized.length <= maxChars) {
    return normalized;
  }
  return `${normalized.slice(0, Math.max(0, maxChars - 3))}...`;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
