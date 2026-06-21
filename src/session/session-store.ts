import { randomUUID } from "node:crypto";
import { mkdir, readFile, rename, rm, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import type {
  AgentMessage,
  AssistantContextMetadata,
  ReasoningOutput,
  ReasoningReplay,
  RequestContextMetadata,
  TokenUsage,
  ToolCall
} from "../types.js";
import { WORKSPACE_NOTE_KINDS, type WorkspaceNote, type WorkspaceState } from "../memory/index.js";
import { Planner, type PlanState } from "../planning/index.js";

export type AgentSessionUsageSnapshot = {
  assistantTurns: number;
  compactions: number;
  usageUnavailableCalls: number;
  latestContext?: RequestContextMetadata;
  providerUsage?: TokenUsage;
};

export type AgentSessionRecord = {
  schemaVersion: 1;
  id: string;
  title: string;
  createdAt: string;
  updatedAt: string;
  exchangeCount: number;
  messages: AgentMessage[];
  workspace: WorkspaceState;
  planning?: {
    plan?: PlanState;
  };
  usage: AgentSessionUsageSnapshot;
};

export type AgentSessionMetadata = {
  id: string;
  title: string;
  file: string;
  createdAt: string;
  updatedAt: string;
  exchangeCount: number;
  messageCount: number;
  workspaceNoteCount: number;
};

export type AgentSessionIndex = {
  schemaVersion: 1;
  activeSessionId?: string;
  sessions: AgentSessionMetadata[];
};

export type AgentSessionStoreOptions = {
  dir?: string;
};

export type LoadActiveSessionResult = {
  session?: AgentSessionRecord;
  warnings: string[];
};

export class AgentSessionStore {
  readonly dir: string;
  readonly indexPath: string;

  constructor(options: AgentSessionStoreOptions = {}) {
    this.dir = options.dir ?? ".agent-sessions";
    this.indexPath = join(this.dir, "index.json");
  }

  sessionPath(id: string): string {
    return join(this.dir, sessionFileName(id));
  }

  async loadActiveSession(): Promise<LoadActiveSessionResult> {
    const indexResult = await this.readIndexForRecovery();
    if (!indexResult.index) {
      return { warnings: indexResult.warnings };
    }

    const activeSessionId = indexResult.index.activeSessionId;
    if (!activeSessionId) {
      return { warnings: indexResult.warnings };
    }

    try {
      return {
        session: await this.loadSession(activeSessionId),
        warnings: indexResult.warnings
      };
    } catch (error) {
      return {
        warnings: [...indexResult.warnings, `Could not load active session ${activeSessionId}: ${formatError(error)}`]
      };
    }
  }

  async listSessions(): Promise<AgentSessionMetadata[]> {
    return (await this.readIndex()).sessions;
  }

  async loadSession(id: string): Promise<AgentSessionRecord> {
    const parsed = JSON.parse(await readFile(this.sessionPath(id), "utf8")) as unknown;
    return parseSessionRecord(parsed);
  }

  async createSession(input: { title?: string } = {}): Promise<AgentSessionRecord> {
    const now = new Date().toISOString();
    const session: AgentSessionRecord = {
      schemaVersion: 1,
      id: createSessionId(),
      title: normalizeTitle(input.title),
      createdAt: now,
      updatedAt: now,
      exchangeCount: 0,
      messages: [],
      workspace: { notes: [] },
      planning: {},
      usage: emptyUsageSnapshot()
    };

    return this.saveSession(session, { active: true, touch: false });
  }

  async saveSession(input: AgentSessionRecord, options: { active?: boolean; touch?: boolean } = {}): Promise<AgentSessionRecord> {
    const session = sanitizeSessionRecord(parseSessionRecord({
      ...input,
      title: normalizeTitle(input.title),
      updatedAt: options.touch === false ? input.updatedAt : new Date().toISOString()
    }));

    await mkdir(this.dir, { recursive: true });
    await writeJsonAtomic(this.sessionPath(session.id), session);

    const index = await this.readIndex();
    const metadata = buildMetadata(session);
    const nextSessions = [
      ...index.sessions.filter((candidate) => candidate.id !== session.id),
      metadata
    ].sort((left, right) => right.updatedAt.localeCompare(left.updatedAt));

    await writeJsonAtomic(this.indexPath, {
      schemaVersion: 1,
      activeSessionId: options.active === false ? index.activeSessionId : session.id,
      sessions: nextSessions
    } satisfies AgentSessionIndex);

    return session;
  }

  async renameSession(id: string, title: string): Promise<AgentSessionRecord> {
    const session = await this.loadSession(id);
    return this.saveSession({ ...session, title: normalizeTitle(title) });
  }

  async deleteSession(id: string): Promise<void> {
    const index = await this.readIndex();
    await rm(this.sessionPath(id), { force: true });
    const sessions = index.sessions.filter((session) => session.id !== id);
    await writeJsonAtomic(this.indexPath, {
      schemaVersion: 1,
      activeSessionId: index.activeSessionId === id ? sessions[0]?.id : index.activeSessionId,
      sessions
    } satisfies AgentSessionIndex);
  }

  private async readIndex(): Promise<AgentSessionIndex> {
    const result = await this.readIndexForRecovery();
    return result.index ?? { schemaVersion: 1, sessions: [] };
  }

  private async readIndexForRecovery(): Promise<{ index?: AgentSessionIndex; warnings: string[] }> {
    try {
      const parsed = JSON.parse(await readFile(this.indexPath, "utf8")) as unknown;
      return { index: parseSessionIndex(parsed), warnings: [] };
    } catch (error) {
      if (isNotFoundError(error)) {
        return { warnings: [] };
      }
      return { warnings: [`Could not read session index ${this.indexPath}: ${formatError(error)}`] };
    }
  }
}

function sanitizeSessionRecord(session: AgentSessionRecord): AgentSessionRecord {
  return {
    ...session,
    messages: session.messages.map(sanitizeAgentMessage)
  };
}

function buildMetadata(session: AgentSessionRecord): AgentSessionMetadata {
  return {
    id: session.id,
    title: session.title,
    file: sessionFileName(session.id),
    createdAt: session.createdAt,
    updatedAt: session.updatedAt,
    exchangeCount: session.exchangeCount,
    messageCount: session.messages.length,
    workspaceNoteCount: session.workspace.notes.length
  };
}

function parseSessionIndex(value: unknown): AgentSessionIndex {
  const object = expectObject(value, "session index");
  const schemaVersion = expectNumber(object.schemaVersion, "session index schemaVersion");
  if (schemaVersion !== 1) {
    throw new Error(`Invalid session index schemaVersion: ${schemaVersion}`);
  }
  if (object.activeSessionId !== undefined && typeof object.activeSessionId !== "string") {
    throw new Error("Invalid session index activeSessionId: expected a string.");
  }
  if (!Array.isArray(object.sessions)) {
    throw new Error("Invalid session index sessions: expected an array.");
  }

  return {
    schemaVersion: 1,
    activeSessionId: object.activeSessionId,
    sessions: object.sessions.map(parseSessionMetadata)
  };
}

function parseSessionMetadata(value: unknown): AgentSessionMetadata {
  const object = expectObject(value, "session metadata");
  return {
    id: expectNonEmptyString(object.id, "session metadata id"),
    title: expectNonEmptyString(object.title, "session metadata title"),
    file: expectNonEmptyString(object.file, "session metadata file"),
    createdAt: expectNonEmptyString(object.createdAt, "session metadata createdAt"),
    updatedAt: expectNonEmptyString(object.updatedAt, "session metadata updatedAt"),
    exchangeCount: expectNonNegativeInteger(object.exchangeCount, "session metadata exchangeCount"),
    messageCount: expectNonNegativeInteger(object.messageCount, "session metadata messageCount"),
    workspaceNoteCount: expectNonNegativeInteger(object.workspaceNoteCount, "session metadata workspaceNoteCount")
  };
}

function parseSessionRecord(value: unknown): AgentSessionRecord {
  const object = expectObject(value, "session record");
  const schemaVersion = expectNumber(object.schemaVersion, "session record schemaVersion");
  if (schemaVersion !== 1) {
    throw new Error(`Invalid session record schemaVersion: ${schemaVersion}`);
  }
  if (!Array.isArray(object.messages)) {
    throw new Error("Invalid session record messages: expected an array.");
  }

  return {
    schemaVersion: 1,
    id: expectNonEmptyString(object.id, "session record id"),
    title: expectNonEmptyString(object.title, "session record title"),
    createdAt: expectNonEmptyString(object.createdAt, "session record createdAt"),
    updatedAt: expectNonEmptyString(object.updatedAt, "session record updatedAt"),
    exchangeCount: expectNonNegativeInteger(object.exchangeCount, "session record exchangeCount"),
    messages: object.messages.map(parseAgentMessage),
    workspace: parseWorkspaceState(object.workspace),
    planning: parsePlanningState(object.planning),
    usage: parseUsageSnapshot(object.usage)
  };
}

function parsePlanningState(value: unknown): AgentSessionRecord["planning"] {
  if (value === undefined) {
    return undefined;
  }
  const object = expectObject(value, "planning state");
  if (object.plan === undefined) {
    return {};
  }
  return {
    plan: new Planner(object.plan as PlanState).state
  };
}

function parseWorkspaceState(value: unknown): WorkspaceState {
  const object = expectObject(value, "workspace state");
  if (!Array.isArray(object.notes)) {
    throw new Error("Invalid workspace state notes: expected an array.");
  }
  return { notes: object.notes.map(parseWorkspaceNote) };
}

function parseWorkspaceNote(value: unknown): WorkspaceNote {
  const object = expectObject(value, "workspace note");
  const kind = expectNonEmptyString(object.kind, "workspace note kind");
  if (!(WORKSPACE_NOTE_KINDS as readonly string[]).includes(kind)) {
    throw new Error(`Invalid workspace note kind: ${kind}`);
  }
  return {
    id: expectNonEmptyString(object.id, "workspace note id"),
    kind: kind as WorkspaceNote["kind"],
    content: expectNonEmptyString(object.content, "workspace note content"),
    createdAt: expectNonEmptyString(object.createdAt, "workspace note createdAt"),
    updatedAt: expectNonEmptyString(object.updatedAt, "workspace note updatedAt")
  };
}

function parseUsageSnapshot(value: unknown): AgentSessionUsageSnapshot {
  if (value === undefined) {
    return emptyUsageSnapshot();
  }
  const object = expectObject(value, "session usage");
  return {
    assistantTurns: expectNonNegativeInteger(object.assistantTurns, "session usage assistantTurns"),
    compactions: expectNonNegativeInteger(object.compactions, "session usage compactions"),
    usageUnavailableCalls: expectNonNegativeInteger(object.usageUnavailableCalls, "session usage usageUnavailableCalls"),
    latestContext: object.latestContext === undefined ? undefined : expectObject(object.latestContext, "session usage latestContext") as RequestContextMetadata,
    providerUsage: object.providerUsage === undefined ? undefined : parseTokenUsage(object.providerUsage)
  };
}

function parseAgentMessage(value: unknown): AgentMessage {
  const object = expectObject(value, "agent message");
  const role = expectNonEmptyString(object.role, "agent message role");
  if (role === "user") {
    return {
      role,
      content: expectString(object.content, "user message content")
    };
  }
  if (role === "assistant") {
    return {
      role,
      content: expectString(object.content, "assistant message content"),
      toolCalls: object.toolCalls === undefined ? undefined : parseToolCalls(object.toolCalls),
      reasoning: object.reasoning === undefined ? undefined : parseReasoningOutput(object.reasoning),
      usage: object.usage === undefined ? undefined : parseTokenUsage(object.usage),
      context: object.context === undefined ? undefined : parseAssistantContext(object.context)
    };
  }
  if (role === "tool") {
    return {
      role,
      toolCallId: expectNonEmptyString(object.toolCallId, "tool message toolCallId"),
      toolName: expectNonEmptyString(object.toolName, "tool message toolName"),
      content: expectString(object.content, "tool message content"),
      isError: object.isError === undefined ? undefined : expectBoolean(object.isError, "tool message isError")
    };
  }
  throw new Error(`Invalid agent message role: ${role}`);
}

function sanitizeAgentMessage(message: AgentMessage): AgentMessage {
  if (message.role === "assistant") {
    return {
      role: "assistant",
      content: message.content,
      toolCalls: message.toolCalls,
      reasoning: message.reasoning,
      usage: message.usage,
      context: message.context
    };
  }
  if (message.role === "tool") {
    return {
      role: "tool",
      toolCallId: message.toolCallId,
      toolName: message.toolName,
      content: message.content,
      isError: message.isError
    };
  }
  return {
    role: "user",
    content: message.content
  };
}

function parseToolCalls(value: unknown): ToolCall[] {
  if (!Array.isArray(value)) {
    throw new Error("Invalid assistant toolCalls: expected an array.");
  }
  return value.map((item) => {
    const object = expectObject(item, "tool call");
    return {
      id: expectNonEmptyString(object.id, "tool call id"),
      name: expectNonEmptyString(object.name, "tool call name"),
      arguments: object.arguments
    };
  });
}

function parseReasoningOutput(value: unknown): ReasoningOutput {
  const object = expectObject(value, "assistant reasoning");
  return {
    summary: object.summary === undefined ? undefined : expectString(object.summary, "assistant reasoning summary"),
    replay: object.replay === undefined ? undefined : parseReasoningReplayList(object.replay)
  };
}

function parseReasoningReplayList(value: unknown): ReasoningReplay[] {
  if (!Array.isArray(value)) {
    throw new Error("Invalid assistant reasoning replay: expected an array.");
  }
  return value.map(parseReasoningReplay);
}

function parseReasoningReplay(value: unknown): ReasoningReplay {
  const object = expectObject(value, "assistant reasoning replay");
  const provider = expectNonEmptyString(object.provider, "assistant reasoning replay provider");
  if (provider === "anthropic") {
    return {
      provider,
      blocks: parseAnthropicReplayBlocks(object.blocks)
    };
  }
  if (provider === "openai-chat") {
    return {
      provider,
      field: parseOpenAIChatReasoningField(object.field),
      content: expectString(object.content, "assistant reasoning replay content")
    };
  }
  throw new Error(`Invalid assistant reasoning replay provider: ${provider}`);
}

function parseAnthropicReplayBlocks(value: unknown): Extract<ReasoningReplay, { provider: "anthropic" }>["blocks"] {
  if (!Array.isArray(value)) {
    throw new Error("Invalid assistant reasoning replay blocks: expected an array.");
  }
  return value.map((item) => {
    const object = expectObject(item, "assistant reasoning replay block");
    const type = expectNonEmptyString(object.type, "assistant reasoning replay block type");
    if (type === "thinking") {
      return {
        type,
        thinking: expectString(object.thinking, "assistant reasoning replay thinking"),
        signature: object.signature === undefined ? undefined : expectString(object.signature, "assistant reasoning replay signature")
      };
    }
    if (type === "redacted_thinking") {
      return {
        type,
        data: expectString(object.data, "assistant reasoning replay redacted data")
      };
    }
    throw new Error(`Invalid assistant reasoning replay block type: ${type}`);
  });
}

function parseOpenAIChatReasoningField(value: unknown): Extract<ReasoningReplay, { provider: "openai-chat" }>["field"] {
  const field = expectNonEmptyString(value, "assistant reasoning replay field");
  if (field === "reasoning_content" || field === "reasoning" || field === "reasoning_text") {
    return field;
  }
  throw new Error(`Invalid assistant reasoning replay field: ${field}`);
}

function parseAssistantContext(value: unknown): AssistantContextMetadata {
  const object = expectObject(value, "assistant context");
  return {
    requestCompacted: object.requestCompacted === undefined ? undefined : expectBoolean(object.requestCompacted, "assistant context requestCompacted")
  };
}

function parseTokenUsage(value: unknown): TokenUsage {
  const object = expectObject(value, "token usage");
  return {
    inputTokens: optionalNumber(object.inputTokens, "token usage inputTokens"),
    outputTokens: optionalNumber(object.outputTokens, "token usage outputTokens"),
    totalTokens: optionalNumber(object.totalTokens, "token usage totalTokens"),
    cacheReadInputTokens: optionalNumber(object.cacheReadInputTokens, "token usage cacheReadInputTokens"),
    cacheCreationInputTokens: optionalNumber(object.cacheCreationInputTokens, "token usage cacheCreationInputTokens")
  };
}

function emptyUsageSnapshot(): AgentSessionUsageSnapshot {
  return {
    assistantTurns: 0,
    compactions: 0,
    usageUnavailableCalls: 0
  };
}

async function writeJsonAtomic(path: string, value: unknown): Promise<void> {
  await mkdir(dirname(path), { recursive: true });
  const tempPath = `${path}.${process.pid}.${Date.now()}.${randomUUID()}.tmp`;
  try {
    await writeFile(tempPath, `${JSON.stringify(value, null, 2)}\n`, "utf8");
    await rename(tempPath, path);
  } catch (error) {
    await rm(tempPath, { force: true });
    throw error;
  }
}

function createSessionId(): string {
  return `sess_${new Date().toISOString().replace(/[-:.TZ]/g, "")}_${randomUUID().slice(0, 8)}`;
}

function sessionFileName(id: string): string {
  return `${id}.json`;
}

function normalizeTitle(value: string | undefined): string {
  const normalized = value?.trim();
  return normalized ? normalized : "Untitled session";
}

function expectObject(value: unknown, label: string): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`Invalid ${label}: expected an object.`);
  }
  return value as Record<string, unknown>;
}

function expectString(value: unknown, label: string): string {
  if (typeof value !== "string") {
    throw new Error(`Invalid ${label}: expected a string.`);
  }
  return value;
}

function expectNonEmptyString(value: unknown, label: string): string {
  const string = expectString(value, label).trim();
  if (!string) {
    throw new Error(`Invalid ${label}: expected a non-empty string.`);
  }
  return string;
}

function expectNumber(value: unknown, label: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`Invalid ${label}: expected a number.`);
  }
  return value;
}

function expectNonNegativeInteger(value: unknown, label: string): number {
  const number = expectNumber(value, label);
  if (!Number.isInteger(number) || number < 0) {
    throw new Error(`Invalid ${label}: expected a non-negative integer.`);
  }
  return number;
}

function expectBoolean(value: unknown, label: string): boolean {
  if (typeof value !== "boolean") {
    throw new Error(`Invalid ${label}: expected a boolean.`);
  }
  return value;
}

function optionalNumber(value: unknown, label: string): number | undefined {
  if (value === undefined) {
    return undefined;
  }
  return expectNumber(value, label);
}

function isNotFoundError(error: unknown): boolean {
  return typeof error === "object" && error !== null && "code" in error && (error as { code?: unknown }).code === "ENOENT";
}

function formatError(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
