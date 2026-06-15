import { estimateTextTokens } from "../context/token-estimator.js";
import type { LlmClient, LlmRequest } from "../llm/types.js";
import type { AgentMessage, AssistantMessage, ReasoningOptions } from "../types.js";

export const SESSION_TITLE_SYSTEM_PROMPT = `You name local coding-agent sessions.

You are looking at the finished thread for the first user request in a new saved session. Write a concise session topic that will be shown in a session list.

Rules:
- Use the user's language when it is clear.
- Prefer the actual task or outcome over generic words like chat, session, or conversation.
- Use 3 to 8 words when possible.
- Do not include quotes, Markdown, bullets, trailing punctuation, or explanations.
- Return only the session name.`;

const DEFAULT_SESSION_TITLE = "Untitled session";
const MAX_SESSION_TITLE_CHARS = 80;

export type GenerateSessionTitleInput = {
  llm: LlmClient;
  model: string;
  messages: readonly AgentMessage[];
  reasoning?: ReasoningOptions | false;
  signal?: AbortSignal;
};

export type GenerateSessionTitleResult = {
  title: string;
  request: LlmRequest;
  message: AssistantMessage;
  titleTokens: number;
  titleChars: number;
};

export async function generateSessionTitle(input: GenerateSessionTitleInput): Promise<GenerateSessionTitleResult> {
  const request = buildSessionTitleRequest(input);
  const message = await input.llm.complete(request);
  const title = normalizeSessionTitle(message.content);

  return {
    title,
    request,
    message,
    titleTokens: estimateTextTokens(title),
    titleChars: title.length
  };
}

function buildSessionTitleRequest(input: GenerateSessionTitleInput): LlmRequest {
  return {
    model: input.model,
    systemPrompt: SESSION_TITLE_SYSTEM_PROMPT,
    messages: [
      ...input.messages.map(cloneAgentMessage),
      {
        role: "user",
        content: "Name this saved session now. Return only the session name."
      }
    ],
    tools: [],
    reasoning: input.reasoning,
    signal: input.signal
  };
}

export function normalizeSessionTitle(content: string): string {
  const firstLine = content
    .trim()
    .split(/\r?\n/)
    .map((line) => line.trim())
    .find((line) => line.length > 0);
  if (!firstLine) {
    return DEFAULT_SESSION_TITLE;
  }

  let title = firstLine
    .replace(/^[-*]\s+/, "")
    .replace(/^\d+[.)]\s+/, "")
    .replace(/^title\s*:\s*/i, "")
    .trim();

  title = stripWrappingQuotes(title).replace(/[.!?。！？]+$/u, "").trim();
  if (!title) {
    return DEFAULT_SESSION_TITLE;
  }

  return title.length > MAX_SESSION_TITLE_CHARS ? title.slice(0, MAX_SESSION_TITLE_CHARS).trimEnd() : title;
}

function stripWrappingQuotes(value: string): string {
  const pairs: Array<[string, string]> = [
    ['"', '"'],
    ["'", "'"],
    ["`", "`"],
    ["“", "”"],
    ["‘", "’"],
    ["「", "」"],
    ["《", "》"]
  ];

  for (const [left, right] of pairs) {
    if (value.startsWith(left) && value.endsWith(right) && value.length >= left.length + right.length) {
      return value.slice(left.length, value.length - right.length).trim();
    }
  }

  return value;
}

function cloneAgentMessage(message: AgentMessage): AgentMessage {
  return JSON.parse(JSON.stringify(message)) as AgentMessage;
}
