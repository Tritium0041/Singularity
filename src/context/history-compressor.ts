import type { AgentMessage, ToolResultMessage } from "../types.js";
import { estimateMessagesTokens } from "./token-estimator.js";
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

export type HistoryCompactionPlan = {
  compactedMessages: AgentMessage[];
  keptMessages: AgentMessage[];
};

export class HistoryCompressor {
  constructor(private readonly options: ResolvedContextEngineOptions) {}

  truncateToolResults(messages: readonly AgentMessage[]): AgentMessage[] {
    return messages.map((message) => (message.role === "tool" ? this.truncateToolMessage(message) : message));
  }

  planCompaction(messages: readonly AgentMessage[]): HistoryCompactionPlan {
    const turns = splitIntoTurns(messages);
    if (turns.length <= 1) {
      return { compactedMessages: [], keptMessages: [...messages] };
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
      return { compactedMessages: [], keptMessages: [...messages] };
    }

    const compactedMessages = turns.slice(0, turns.length - keptTurns.length).flatMap((turn) => turn.messages);
    const keptMessages = keptTurns.flatMap((turn) => turn.messages);
    return { compactedMessages, keptMessages };
  }

  compact(messages: readonly AgentMessage[]): AgentMessage[] {
    const plan = this.planCompaction(messages);
    if (plan.compactedMessages.length === 0) {
      return [...messages];
    }

    return [...selectUserInstructionMessages(plan.compactedMessages), ...plan.keptMessages];
  }

  compactWithHandoffSummary(
    messages: readonly AgentMessage[],
    summary: string,
    options: { plan?: HistoryCompactionPlan } = {}
  ): AgentMessage[] {
    const plan = options.plan ?? this.planCompaction(messages);
    return [...selectUserInstructionMessages(plan.compactedMessages), buildHandoffSummaryMessage(summary), ...plan.keptMessages];
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

function buildHandoffSummaryMessage(summary: string): AgentMessage {
  const summaryText = normalizeHandoffSummary(summary);
  return {
    role: "user",
    content: `${CONTEXT_HANDOFF_SUMMARY_PREFIX}\n${summaryText}`
  };
}

function selectUserInstructionMessages(messages: readonly AgentMessage[]): AgentMessage[] {
  return messages
    .filter((message) => message.role === "user" && !isHandoffSummaryMessage(message.content))
    .map((message) => ({ role: "user" as const, content: message.content }));
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

function truncateByChars(text: string, maxChars: number, direction: "head" | "tail"): string {
  const marker = `[ContextEngine truncated tool result; showing ${direction} of ${text.length} chars.]`;
  const bodyChars = Math.max(1, maxChars - marker.length - 2);
  if (direction === "tail") {
    return `${marker}\n${text.slice(-bodyChars)}`;
  }
  return `${text.slice(0, bodyChars)}\n${marker}`;
}
