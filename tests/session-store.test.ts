import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { AgentSessionStore, generateSessionTitle, normalizeSessionTitle, type AgentSessionRecord } from "../src/session/index.js";
import type { LlmClient, LlmRequest } from "../src/llm/types.js";
import type { AssistantMessage } from "../src/types.js";

class SequenceLlm implements LlmClient {
  public readonly requests: LlmRequest[] = [];

  constructor(private readonly responses: AssistantMessage[]) {}

  async complete(request: LlmRequest): Promise<AssistantMessage> {
    this.requests.push(request);
    const response = this.responses.shift();
    if (!response) {
      throw new Error("No fake response left");
    }
    return response;
  }
}

test("session store creates, lists, saves, loads, and renames sessions", async () => {
  const dir = await mkdtemp(join(tmpdir(), "singularity-sessions-"));
  try {
    const store = new AgentSessionStore({ dir });
    const session = await store.createSession({ title: "Planning" });
    const saved = await store.saveSession({
      ...session,
      exchangeCount: 1,
      messages: [
        { role: "user", content: "hello" },
        { role: "assistant", content: "hi", usage: { inputTokens: 4, outputTokens: 2, totalTokens: 6 } }
      ],
      workspace: {
        notes: [
          {
            id: "note_1",
            kind: "decision",
            content: "Keep sessions in JSON.",
            createdAt: "2026-06-15T00:00:00.000Z",
            updatedAt: "2026-06-15T00:00:00.000Z"
          }
        ]
      },
      usage: {
        assistantTurns: 1,
        compactions: 0,
        usageUnavailableCalls: 0,
        providerUsage: { inputTokens: 4, outputTokens: 2, totalTokens: 6 }
      }
    });

    const listed = await store.listSessions();
    assert.equal(listed.length, 1);
    assert.equal(listed[0]?.id, session.id);
    assert.equal(listed[0]?.title, "Planning");
    assert.equal(listed[0]?.exchangeCount, 1);
    assert.equal(listed[0]?.messageCount, 2);
    assert.equal(listed[0]?.workspaceNoteCount, 1);

    const loaded = await store.loadSession(session.id);
    assert.deepEqual(loaded.messages, saved.messages);
    assert.deepEqual(loaded.workspace, saved.workspace);
    assert.equal(loaded.usage.providerUsage?.inputTokens, 4);
    assert.equal(loaded.usage.providerUsage?.outputTokens, 2);
    assert.equal(loaded.usage.providerUsage?.totalTokens, 6);

    const renamed = await store.renameSession(session.id, "Resumed planning");
    assert.equal(renamed.title, "Resumed planning");
    assert.equal((await store.loadActiveSession()).session?.title, "Resumed planning");

    const raw = JSON.parse(await readFile(store.sessionPath(session.id), "utf8")) as AgentSessionRecord;
    assert.equal(raw.schemaVersion, 1);
    assert.equal(raw.id, session.id);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test("session store persists readable turn messages without provider raw delta events", async () => {
  const dir = await mkdtemp(join(tmpdir(), "singularity-sessions-"));
  try {
    const store = new AgentSessionStore({ dir });
    const session = await store.createSession({ title: "Readable" });
    await store.saveSession({
      ...session,
      exchangeCount: 1,
      messages: [
        { role: "user", content: "stream please" },
        {
          role: "assistant",
          content: "final answer",
          usage: { inputTokens: 10, outputTokens: 3, totalTokens: 13 },
          raw: [{ type: "response.output_text.delta", delta: "final " }, { type: "response.output_text.delta", delta: "answer" }]
        },
        {
          role: "tool",
          toolCallId: "call_1",
          toolName: "fetch_url",
          content: "final tool result",
          details: { raw: { html: "<html>noisy</html>" } }
        }
      ]
    });

    const rawText = await readFile(store.sessionPath(session.id), "utf8");
    assert.doesNotMatch(rawText, /response\.output_text\.delta/);
    assert.doesNotMatch(rawText, /"raw"/);
    assert.doesNotMatch(rawText, /"details"/);

    const loaded = await store.loadSession(session.id);
    const assistant = loaded.messages[1];
    assert.equal(assistant?.role, "assistant");
    assert.equal(assistant?.content, "final answer");
    assert.equal("raw" in (assistant ?? {}), false);
    assert.equal("details" in (loaded.messages[2] ?? {}), false);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test("session store deletes sessions and moves active session when needed", async () => {
  const dir = await mkdtemp(join(tmpdir(), "singularity-sessions-"));
  try {
    const store = new AgentSessionStore({ dir });
    const first = await store.createSession({ title: "First" });
    const second = await store.createSession({ title: "Second" });

    assert.equal((await store.loadActiveSession()).session?.id, second.id);

    await store.deleteSession(second.id);
    assert.equal((await store.loadActiveSession()).session?.id, first.id);
    assert.equal((await store.listSessions()).length, 1);

    await store.deleteSession(first.id);
    assert.equal((await store.loadActiveSession()).session, undefined);
    assert.deepEqual(await store.listSessions(), []);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test("session store reports missing active session as a recoverable warning", async () => {
  const dir = await mkdtemp(join(tmpdir(), "singularity-sessions-"));
  try {
    const store = new AgentSessionStore({ dir });
    const session = await store.createSession({ title: "Broken active" });
    await rm(store.sessionPath(session.id), { force: true });

    const result = await store.loadActiveSession();
    assert.equal(result.session, undefined);
    assert.equal(result.warnings.length, 1);
    assert.match(result.warnings[0] ?? "", /Could not load active session/);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test("session store validates malformed messages and workspace notes", async () => {
  const dir = await mkdtemp(join(tmpdir(), "singularity-sessions-"));
  try {
    const store = new AgentSessionStore({ dir });
    const session = await store.createSession({ title: "Invalid" });
    await writeFile(
      store.sessionPath(session.id),
      JSON.stringify({
        ...session,
        messages: [{ role: "not-a-role", content: "bad" }]
      }),
      "utf8"
    );

    await assert.rejects(() => store.loadSession(session.id), /Invalid agent message role/);

    await writeFile(
      store.sessionPath(session.id),
      JSON.stringify({
        ...session,
        workspace: {
          notes: [{ id: "note_1", kind: "bad", content: "bad", createdAt: "now", updatedAt: "now" }]
        }
      }),
      "utf8"
    );

    await assert.rejects(() => store.loadSession(session.id), /Invalid workspace note kind/);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test("session title summary names first request history without tools", async () => {
  const llm = new SequenceLlm([
    {
      role: "assistant",
      content: '"实现 session 自动命名。"',
      usage: { inputTokens: 20, outputTokens: 5, totalTokens: 25 }
    }
  ]);

  const result = await generateSessionTitle({
    llm,
    model: "summary-model",
    messages: [
      { role: "user", content: "对每个session在第一次user request结束时，总结一个主题" },
      { role: "assistant", content: "已实现自动命名。" }
    ],
    reasoning: false
  });

  assert.equal(result.title, "实现 session 自动命名");
  assert.equal(result.titleChars, "实现 session 自动命名".length);
  assert.equal(result.message.usage?.totalTokens, 25);
  assert.equal(llm.requests.length, 1);

  const request = llm.requests[0];
  assert.equal(request?.model, "summary-model");
  assert.deepEqual(request?.tools, []);
  assert.equal(request?.reasoning, false);
  assert.match(request?.systemPrompt ?? "", /You name local coding-agent sessions/);
  assert.equal(request?.messages[0]?.role, "user");
  assert.match(request?.messages.at(-1)?.content ?? "", /Name this saved session/);
});

test("session title normalization keeps a compact single-line name", () => {
  assert.equal(normalizeSessionTitle("Title: Planning Work.\nextra detail"), "Planning Work");
  assert.equal(normalizeSessionTitle(""), "Untitled session");
  assert.equal(normalizeSessionTitle("x".repeat(100)), "x".repeat(80));
});
