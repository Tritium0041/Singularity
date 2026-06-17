import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { Agent } from "../src/agent/agent-loop.js";
import {
  buildMemoryInstructions,
  createMemoryStoreTools,
  createWorkspaceTools,
  MarkdownMemoryStore,
  WorkspaceMemory
} from "../src/memory/index.js";
import { ToolExecutor, ToolRegistry } from "../src/tools/registry.js";
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

test("workspace memory writes, reads, updates, deletes, and validates notes", () => {
  const workspace = new WorkspaceMemory();
  const first = workspace.write({ content: "  implementation detail  " });
  const second = workspace.write({ kind: "decision", content: "Use Markdown memory." });

  assert.equal(first.kind, "note");
  assert.equal(first.content, "implementation detail");
  assert.equal(workspace.read({ id: first.id })[0]?.content, "implementation detail");
  assert.deepEqual(workspace.list().map((note) => note.id), [first.id, second.id]);
  assert.deepEqual(workspace.list({ kind: "decision" }).map((note) => note.id), [second.id]);
  assert.deepEqual(
    workspace.read({ kind: "decision" }).map((note) => note.id),
    [second.id]
  );

  const updated = workspace.update({ id: first.id, kind: "todo", content: "Add tests." });
  assert.equal(updated.kind, "todo");
  assert.equal(updated.content, "Add tests.");
  assert.equal(workspace.delete(second.id), true);
  assert.equal(workspace.read().length, 1);
  assert.throws(() => workspace.write({ content: "   " }), /non-empty/);
  assert.throws(() => workspace.write({ kind: "bad" as never, content: "bad" }), /Invalid workspace note kind/);

  const snapshot = workspace.state;
  snapshot.notes[0]!.content = "mutated outside";
  assert.equal(workspace.read({ id: first.id })[0]?.content, "Add tests.");
});

test("workspace memory lists compact note previews", () => {
  const workspace = new WorkspaceMemory();
  const note = workspace.write({
    kind: "file",
    content: "First line with enough detail.\n\nSecond line should be compacted away from raw whitespace."
  });

  const [listed] = workspace.list({ kind: "file", previewCharacters: 24 });

  assert.equal(listed?.id, note.id);
  assert.equal(listed?.kind, "file");
  assert.equal(listed?.contentLength, Array.from(note.content).length);
  assert.equal(listed?.preview, "First line with enough d...");
  assert.equal(Object.hasOwn(listed ?? {}, "content"), false);
});

test("markdown memory store initializes, appends, lists, searches, and clears entries", async () => {
  const dir = await mkdtemp(join(tmpdir(), "singularity-memory-"));
  try {
    const path = join(dir, "MEMORY.md");
    const store = new MarkdownMemoryStore({ path });

    await store.ensureFile();
    assert.match(await readFile(path, "utf8"), /# Singularity Memory/);

    await store.store({
      content: "User prefers TypeScript for scripts.",
      tags: ["TypeScript", " preference ", "typescript"],
      source: "user"
    });
    await store.store({
      content: "Project convention: keep memory retrieval tool-result only.",
      tags: ["project"],
      source: "agent"
    });

    const text = await readFile(path, "utf8");
    assert.match(text, /## mem_/);
    assert.match(text, /- tags: typescript, preference/);
    assert.match(text, /- source: user/);

    const listed = await store.list({ tag: "typescript" });
    assert.equal(listed.length, 1);
    assert.equal(listed[0]?.tags.includes("typescript"), true);
    assert.equal(typeof listed[0]?.lineStart, "number");
    assert.equal(typeof listed[0]?.lineEnd, "number");

    const contentResults = await store.search("TypeScript scripts");
    assert.equal(contentResults[0]?.entry.content, "User prefers TypeScript for scripts.");
    assert.ok(contentResults[0]?.score ?? 0);
    assert.match(contentResults[0]?.snippet ?? "", /TypeScript/);

    const tagResults = await store.search("project", { tags: ["project"], maxResults: 1 });
    assert.equal(tagResults.length, 1);
    assert.match(tagResults[0]?.entry.content ?? "", /tool-result only/);

    assert.deepEqual(await store.search("missing term"), []);

    await store.clear();
    assert.equal((await store.list()).length, 0);
    assert.match(await readFile(path, "utf8"), /# Singularity Memory/);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test("markdown memory parser ignores non-memory markdown blocks", async () => {
  const dir = await mkdtemp(join(tmpdir(), "singularity-memory-"));
  try {
    const path = join(dir, "MEMORY.md");
    await writeFile(
      path,
      [
        "# Singularity Memory",
        "",
        "Hand-written notes that are not structured entries.",
        "",
        "## mem_manual",
        "- tags: durable",
        "- created_at: 2026-06-13T00:00:00.000Z",
        "- updated_at: 2026-06-13T00:00:00.000Z",
        "",
        "Durable convention."
      ].join("\n"),
      "utf8"
    );

    const store = new MarkdownMemoryStore({ path });
    const entries = await store.list();
    assert.equal(entries.length, 1);
    assert.equal(entries[0]?.id, "mem_manual");
    assert.equal(entries[0]?.content, "Durable convention.");
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test("markdown memory store updates and upserts entries by tag", async () => {
  const dir = await mkdtemp(join(tmpdir(), "singularity-memory-upsert-"));
  try {
    const path = join(dir, "MEMORY.md");
    const store = new MarkdownMemoryStore({ path });

    const created = await store.upsertByTag({
      tag: "phase-summary",
      content: "Initial state.",
      tags: ["workspace"],
      source: "agent"
    });
    assert.equal(created.created, true);
    assert.equal(created.entry.content, "Initial state.");

    const updated = await store.upsertByTag({
      tag: "phase-summary",
      content: "Updated state.",
      tags: ["workspace"],
      source: "agent"
    });
    assert.equal(updated.created, false);
    assert.equal(updated.entry.id, created.entry.id);
    assert.equal(updated.entry.content, "Updated state.");
    assert.equal(updated.entry.tags.includes("phase-summary"), true);
    assert.equal(updated.entry.tags.includes("workspace"), true);

    const entries = await store.list({ tag: "phase-summary" });
    assert.equal(entries.length, 1);
    assert.equal(entries[0]?.content, "Updated state.");

    const text = await readFile(path, "utf8");
    assert.equal((text.match(/^## mem_/gm) ?? []).length, 1);
    assert.doesNotMatch(text, /Initial state/);
    assert.match(text, /Updated state/);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test("memory tools share workspace and store instances", async () => {
  const dir = await mkdtemp(join(tmpdir(), "singularity-memory-tools-"));
  try {
    const workspace = new WorkspaceMemory();
    const workspaceExecutor = new ToolExecutor(new ToolRegistry(createWorkspaceTools(workspace)));
    const writeResult = await workspaceExecutor.execute({
      id: "call_write",
      name: "write_note",
      arguments: { kind: "decision", content: "Keep the API small." }
    });
    assert.equal(writeResult.isError, undefined);
    assert.equal(workspace.read({ kind: "decision" }).length, 1);

    const listResult = await workspaceExecutor.execute({
      id: "call_list",
      name: "list_notes",
      arguments: { kind: "decision" }
    });
    assert.equal(listResult.isError, undefined);
    assert.match(listResult.content, /Keep the API small/);
    assert.doesNotMatch(listResult.content, /"content"/);

    const readResult = await workspaceExecutor.execute({
      id: "call_read",
      name: "read_note",
      arguments: { kind: "decision" }
    });
    assert.match(readResult.content, /Keep the API small/);

    const badResult = await workspaceExecutor.execute({
      id: "call_bad",
      name: "write_note",
      arguments: { kind: "unknown", content: "bad" }
    });
    assert.equal(badResult.isError, true);
    assert.match(badResult.content, /must be one of/);

    const store = new MarkdownMemoryStore({ path: join(dir, "MEMORY.md") });
    const storeExecutor = new ToolExecutor(new ToolRegistry(createMemoryStoreTools(store)));
    await storeExecutor.execute({
      id: "call_store",
      name: "store_memory",
      arguments: { content: "Use TypeScript in examples.", tags: ["preference"], source: "user" }
    });
    const searchResult = await storeExecutor.execute({
      id: "call_search",
      name: "search_memory",
      arguments: { query: "TypeScript", maxResults: 1 }
    });
    assert.match(searchResult.content, /Use TypeScript in examples/);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test("static memory instructions contain only rules, not dynamic memory content", () => {
  const fragment = buildMemoryInstructions({ hasWorkspace: true, hasStore: true });
  assert.match(fragment?.content ?? "", /search_memory/);
  assert.match(fragment?.content ?? "", /write_note/);
  assert.doesNotMatch(fragment?.content ?? "", /User prefers TypeScript/);
});

test("agent registers default workspace memory tools and instructions", async () => {
  const llm = new SequenceLlm([{ role: "assistant", content: "ready" }]);
  const agent = new Agent({ llm, model: "fake-model" });

  await agent.run("hello");

  const request = llm.requests[0];
  assert.equal(request?.tools?.some((tool) => tool.name === "write_note"), true);
  assert.equal(request?.tools?.some((tool) => tool.name === "list_notes"), true);
  assert.equal(request?.tools?.some((tool) => tool.name === "read_note"), true);
  assert.equal(request?.tools?.some((tool) => tool.name === "store_memory"), false);
  assert.match(request?.systemPrompt ?? "", /Memory tools are available/);
  assert.match(request?.systemPrompt ?? "", /list_notes/);
  assert.match(request?.systemPrompt ?? "", /write_note/);
  assert.doesNotMatch(request?.systemPrompt ?? "", /store_memory/);
});

test("agent memory false disables memory tools and instructions", async () => {
  const llm = new SequenceLlm([{ role: "assistant", content: "ready" }]);
  const agent = new Agent({ llm, model: "fake-model", memory: false });

  await agent.run("hello");

  assert.equal(llm.requests[0]?.tools?.some((tool) => tool.name === "write_note"), false);
  assert.doesNotMatch(llm.requests[0]?.systemPrompt ?? "", /Memory tools are available/);
});

test("background false keeps the configured system prompt exact even when memory is enabled", async () => {
  const llm = new SequenceLlm([{ role: "assistant", content: "ready" }]);
  const agent = new Agent({
    llm,
    model: "fake-model",
    systemPrompt: "Only this prompt.",
    background: false
  });

  await agent.run("hello");

  assert.equal(llm.requests[0]?.systemPrompt, "Only this prompt.");
  assert.equal(llm.requests[0]?.tools?.some((tool) => tool.name === "write_note"), true);
});

test("workspace content enters context only through tool results", async () => {
  const workspace = new WorkspaceMemory();
  const llm = new SequenceLlm([
    {
      role: "assistant",
      content: "",
      toolCalls: [{ id: "call_write", name: "write_note", arguments: { kind: "decision", content: "Secret workspace detail." } }]
    },
    { role: "assistant", content: "done" }
  ]);
  const agent = new Agent({
    llm,
    model: "fake-model",
    memory: { workspace }
  });

  await agent.run("remember the detail");

  assert.equal(workspace.read({ kind: "decision" })[0]?.content, "Secret workspace detail.");
  assert.doesNotMatch(llm.requests[0]?.systemPrompt ?? "", /Secret workspace detail/);
  assert.doesNotMatch(llm.requests[1]?.systemPrompt ?? "", /Secret workspace detail/);
  assert.equal(llm.requests[1]?.messages.at(-1)?.role, "tool");
  assert.match(llm.requests[1]?.messages.at(-1)?.content ?? "", /Secret workspace detail/);
});

test("long-term memory is not recalled in a new agent until search_memory is called", async () => {
  const dir = await mkdtemp(join(tmpdir(), "singularity-agent-memory-"));
  try {
    const store = new MarkdownMemoryStore({ path: join(dir, "MEMORY.md") });
    const writerLlm = new SequenceLlm([
      {
        role: "assistant",
        content: "",
        toolCalls: [{ id: "call_store", name: "store_memory", arguments: { content: "User likes concise answers.", tags: ["preference"] } }]
      },
      { role: "assistant", content: "stored" }
    ]);
    const writer = new Agent({ llm: writerLlm, model: "fake-model", memory: { store } });
    await writer.run("remember this");

    const readerLlm = new SequenceLlm([
      {
        role: "assistant",
        content: "",
        toolCalls: [{ id: "call_search", name: "search_memory", arguments: { query: "concise answers" } }]
      },
      { role: "assistant", content: "found it" }
    ]);
    const reader = new Agent({ llm: readerLlm, model: "fake-model", memory: { store } });
    await reader.run("what do you remember?");

    assert.equal(readerLlm.requests[0]?.messages.some((message) => message.content.includes("concise answers")), false);
    assert.equal(readerLlm.requests[0]?.systemPrompt?.includes("concise answers"), false);
    assert.match(readerLlm.requests[1]?.messages.at(-1)?.content ?? "", /User likes concise answers/);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test("workspace source data can be read after history compaction", async () => {
  const workspace = new WorkspaceMemory();
  workspace.write({ kind: "decision", content: "Compaction must not clear workspace." });
  const llm = new SequenceLlm([
    { role: "assistant", content: "old answer that can be compacted" },
    { role: "assistant", content: "Manual summary of old answer." },
    {
      role: "assistant",
      content: "",
      toolCalls: [{ id: "call_read", name: "read_note", arguments: { kind: "decision" } }]
    },
    { role: "assistant", content: "continued" }
  ]);
  const agent = new Agent({
    llm,
    model: "fake-model",
    memory: { workspace }
  });

  await agent.run("old request", { context: false });
  await agent.compactHistory({
    context: {
      contextWindowTokens: 8,
      reservedOutputTokens: 0,
      keepRecentTokens: 1
    }
  });
  await agent.run("continue", { context: false });

  const toolMessage = agent.history.find((message) => message.role === "tool" && message.toolName === "read_note");
  assert.match(toolMessage?.content ?? "", /Compaction must not clear workspace/);
});
