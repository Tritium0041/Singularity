import assert from "node:assert/strict";
import test from "node:test";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { Agent } from "../src/agent/agent-loop.js";
import type { LlmClient, LlmRequest } from "../src/llm/types.js";
import { McpManager } from "../src/mcp/index.js";
import { loadSkillsSync, renderAvailableSkills } from "../src/skills/index.js";
import { createFileSystemTools } from "../src/tools/core-tools.js";
import type { AgentTool } from "../src/tools/registry.js";
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

test("skill loader scans SKILL.md, root markdown files, frontmatter, collisions, and hidden skills", async () => {
  const root = await mkdtemp(join(tmpdir(), "singularity-skills-"));
  try {
    await mkdir(join(root, "review"), { recursive: true });
    await writeFile(
      join(root, "review", "SKILL.md"),
      [
        "---",
        "name: review",
        "description: Review code changes.",
        "---",
        "Read the diff."
      ].join("\n"),
      "utf8"
    );
    await writeFile(
      join(root, "translate.md"),
      [
        "---",
        "description: Translate text.",
        "disable-model-invocation: true",
        "---",
        "Translate carefully."
      ].join("\n"),
      "utf8"
    );
    await mkdir(join(root, "duplicate"), { recursive: true });
    await writeFile(
      join(root, "duplicate", "SKILL.md"),
      [
        "---",
        "name: review",
        "description: Duplicate review skill.",
        "---",
        "duplicate"
      ].join("\n"),
      "utf8"
    );
    await mkdir(join(root, "missing"), { recursive: true });
    await writeFile(join(root, "missing", "SKILL.md"), "---\nname: missing\n---\nmissing", "utf8");

    const result = loadSkillsSync({ roots: [root] });

    assert.deepEqual(result.skills.map((skill) => skill.name), ["review", "translate"]);
    assert.equal(result.skills.find((skill) => skill.name === "translate")?.disableModelInvocation, true);
    assert.ok(result.diagnostics.some((diagnostic) => diagnostic.message.includes("collision")));
    assert.ok(result.diagnostics.some((diagnostic) => diagnostic.message.includes("missing a required description")));
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("skill renderer escapes XML and filters disable-model-invocation skills", () => {
  const rendered = renderAvailableSkills([
    {
      name: "safe",
      description: "Use <care> & quotes \"here\".",
      filePath: "/tmp/SKILL.md",
      baseDir: "/tmp",
      disableModelInvocation: false,
      source: { kind: "directory", root: "/tmp" }
    },
    {
      name: "hidden",
      description: "Hidden",
      filePath: "/tmp/hidden.md",
      baseDir: "/tmp",
      disableModelInvocation: true,
      source: { kind: "file", root: "/tmp" }
    }
  ]);

  assert.match(rendered ?? "", /<available_skills>/);
  assert.match(rendered ?? "", /Use &lt;care&gt; &amp; quotes &quot;here&quot;\./);
  assert.doesNotMatch(rendered ?? "", /hidden/);
});

test("agent exposes skills only when read_file is available", async () => {
  const root = await mkdtemp(join(tmpdir(), "singularity-agent-skills-"));
  try {
    await mkdir(join(root, "review"), { recursive: true });
    await writeFile(join(root, "review", "SKILL.md"), "---\ndescription: Review code.\n---\nBody", "utf8");
    const skills = loadSkillsSync({ roots: [root] });

    const withReadLlm = new SequenceLlm([{ role: "assistant", content: "ok" }]);
    const withRead = new Agent({
      llm: withReadLlm,
      model: "fake-model",
      tools: createFileSystemTools({ rootDir: root }),
      skills
    });
    await withRead.run("hi");
    assert.match(withReadLlm.requests[0]?.systemPrompt ?? "", /<available_skills>/);

    const withoutReadLlm = new SequenceLlm([{ role: "assistant", content: "ok" }]);
    const withoutRead = new Agent({
      llm: withoutReadLlm,
      model: "fake-model",
      skills
    });
    await withoutRead.run("hi");
    assert.doesNotMatch(withoutReadLlm.requests[0]?.systemPrompt ?? "", /<available_skills>/);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("MCP manager starts stdio server, maps tools, filters tools, and calls through", async () => {
  const root = await mkdtemp(join(await ensureWorkspaceTmp(), "mcp-"));
  const serverPath = join(root, "server.mjs");
  try {
    await writeMcpServer(serverPath);
    const manager = new McpManager({
      servers: {
        "docs.server": {
          transport: "stdio",
          command: process.execPath,
          args: [serverPath],
          enabledTools: ["echo"]
        }
      }
    });
    await manager.start();

    const tools = manager.getTools();
    assert.equal(tools.length, 1);
    assert.equal(tools[0]?.name, "mcp_docs_server_echo");
    assert.equal(manager.getToolInfos()[0]?.rawName, "echo");

    const result = await tools[0]!.execute({ text: "hello" }, { toolCallId: "call_1" });
    assert.equal(result.content, "echo: hello");
    assert.equal(result.isError, false);
    await manager.close();
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("MCP manager isolates startup and call failures", async () => {
  const root = await mkdtemp(join(await ensureWorkspaceTmp(), "mcp-failure-"));
  const serverPath = join(root, "server.mjs");
  try {
    await writeMcpServer(serverPath);
    const manager = new McpManager({
      servers: {
        missing: {
          transport: "stdio",
          command: join(root, "does-not-exist")
        },
        ok: {
          transport: "stdio",
          command: process.execPath,
          args: [serverPath]
        }
      }
    });
    await manager.start();

    assert.ok(manager.getDiagnostics().some((diagnostic) => diagnostic.serverName === "missing"));
    const failTool = manager.getTools().find((tool) => tool.name === "mcp_ok_fail");
    assert.ok(failTool);
    const result = await failTool.execute({}, { toolCallId: "call_1" });
    assert.equal(result.isError, true);
    assert.match(result.content, /forced failure/);
    await manager.close();
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("agent merges MCP tools and planning gate keeps execute tools before approval", async () => {
  const mcpTool: AgentTool = {
    name: "mcp_docs_search",
    description: "Search docs",
    access: "read",
    parameters: { type: "object", properties: {}, additionalProperties: false },
    execute: () => ({ content: "docs" })
  };
  const writeTool: AgentTool = {
    name: "mcp_docs_write",
    description: "Write docs",
    access: "write",
    parameters: { type: "object", properties: {}, additionalProperties: false },
    execute: () => ({ content: "wrote" })
  };
  class StaticMcpManager extends McpManager {
    constructor() {
      super({ servers: {} });
    }

    override getTools(): AgentTool[] {
      return [mcpTool, writeTool];
    }
  }
  const llm = new SequenceLlm([{ role: "assistant", content: "ok" }]);
  const agent = new Agent({
    llm,
    model: "fake-model",
    mcp: new StaticMcpManager(),
    planning: {}
  });

  await agent.run("hi");

  const toolNames = llm.requests[0]?.tools?.map((tool) => tool.name) ?? [];
  assert.ok(toolNames.includes("mcp_docs_search"));
  assert.ok(!toolNames.includes("mcp_docs_write"));
});

test("agent can start MCP config lazily before the first request", async () => {
  const root = await mkdtemp(join(await ensureWorkspaceTmp(), "mcp-agent-config-"));
  const serverPath = join(root, "server.mjs");
  let agent: Agent | undefined;
  try {
    await writeMcpServer(serverPath);
    const llm = new SequenceLlm([{ role: "assistant", content: "ok" }]);
    agent = new Agent({
      llm,
      model: "fake-model",
      mcp: {
        servers: {
          docs: {
            transport: "stdio",
            command: process.execPath,
            args: [serverPath]
          }
        }
      }
    });

    await agent.run("hi");

    const toolNames = llm.requests[0]?.tools?.map((tool) => tool.name) ?? [];
    assert.ok(toolNames.includes("mcp_docs_echo"));
    assert.ok(toolNames.includes("mcp_docs_fail"));
  } finally {
    await agent?.close();
    await rm(root, { recursive: true, force: true });
  }
});

async function writeMcpServer(path: string): Promise<void> {
  await mkdir(dirname(path), { recursive: true });
  await writeFile(
    path,
    `
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { CallToolRequestSchema, ListToolsRequestSchema } from "@modelcontextprotocol/sdk/types.js";

const server = new Server({ name: "test-server", version: "1.0.0" }, { capabilities: { tools: {} } });

server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: [
    {
      name: "echo",
      description: "Echo text",
      inputSchema: {
        type: "object",
        properties: { text: { type: "string" } },
        required: ["text"],
        additionalProperties: false
      }
    },
    {
      name: "fail",
      description: "Fail",
      inputSchema: { type: "object", properties: {}, additionalProperties: false }
    }
  ]
}));

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  if (request.params.name === "fail") {
    return { content: [{ type: "text", text: "forced failure" }], isError: true };
  }
  return { content: [{ type: "text", text: "echo: " + request.params.arguments.text }], isError: false };
});

await server.connect(new StdioServerTransport());
`,
    "utf8"
  );
}

async function ensureWorkspaceTmp(): Promise<string> {
  const path = join(process.cwd(), ".tmp-tests");
  await mkdir(path, { recursive: true });
  return path;
}
