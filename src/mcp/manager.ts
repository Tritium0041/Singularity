import { createHash } from "node:crypto";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { SSEClientTransport } from "@modelcontextprotocol/sdk/client/sse.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";
import { WebSocketClientTransport } from "@modelcontextprotocol/sdk/client/websocket.js";
import type { Transport } from "@modelcontextprotocol/sdk/shared/transport.js";
import { WebSocket } from "ws";
import type { JsonSchema } from "../types.js";
import type { AgentTool, ToolAccess, ToolResult } from "../tools/registry.js";
import type { McpConfig, McpDiagnostic, McpManagerOptions, McpServerConfig, McpToolInfo } from "./types.js";

type RunningServer = {
  name: string;
  config: McpServerConfig;
  client: Client;
  transport: Transport;
};

type ListedMcpTool = {
  name: string;
  description?: string;
  inputSchema: JsonSchema;
  annotations?: {
    readOnlyHint?: boolean;
    destructiveHint?: boolean;
  };
};

type ToolRoute = {
  server: RunningServer;
  info: McpToolInfo;
  parameters: JsonSchema;
  access: ToolAccess;
};

export class McpManager {
  private readonly config: McpConfig;
  private readonly clientName: string;
  private readonly clientVersion: string;
  private readonly routes = new Map<string, ToolRoute>();
  private readonly runningServers = new Map<string, RunningServer>();
  private readonly diagnostics: McpDiagnostic[] = [];
  private started = false;

  constructor(options: McpManagerOptions | McpConfig) {
    if ("servers" in options) {
      this.config = options;
      this.clientName = "singularity";
      this.clientVersion = "0.1.0";
      return;
    }
    this.config = options.config;
    this.clientName = options.clientName ?? "singularity";
    this.clientVersion = options.clientVersion ?? "0.1.0";
  }

  getDiagnostics(): McpDiagnostic[] {
    return [...this.diagnostics];
  }

  getToolInfos(): McpToolInfo[] {
    return [...this.routes.values()].map((route) => route.info);
  }

  async start(): Promise<void> {
    if (this.started) {
      return;
    }
    this.started = true;

    const usedNames = new Set<string>();
    for (const [serverName, serverConfig] of Object.entries(this.config.servers)) {
      if (serverConfig.enabled === false) {
        continue;
      }
      try {
        const server = await this.startServer(serverName, serverConfig);
        this.runningServers.set(serverName, server);
        const listed = await server.client.listTools();
        for (const tool of listed.tools as ListedMcpTool[]) {
          if (!isToolEnabled(tool.name, serverConfig)) {
            continue;
          }
          const modelName = uniqueModelToolName(serverName, tool.name, usedNames);
          usedNames.add(modelName);
          const info: McpToolInfo = {
            serverName,
            rawName: tool.name,
            name: modelName,
            description: tool.description ?? `MCP tool ${tool.name} from server ${serverName}.`
          };
          this.routes.set(modelName, { server, info, parameters: normalizeInputSchema(tool.inputSchema), access: inferToolAccess(tool) });
        }
      } catch (error) {
        this.diagnostics.push({
          level: "error",
          serverName,
          message: `Failed to start MCP server: ${formatError(error)}`
        });
      }
    }
  }

  getTools(): AgentTool[] {
    return [...this.routes.values()].map((route) => this.toAgentTool(route));
  }

  async close(): Promise<void> {
    const servers = [...this.runningServers.values()];
    this.runningServers.clear();
    this.routes.clear();
    this.started = false;
    await Promise.allSettled(servers.map((server) => server.transport.close()));
  }

  private async startServer(serverName: string, config: McpServerConfig): Promise<RunningServer> {
    const transport = createTransport(config);
    const client = new Client({
      name: this.clientName,
      version: this.clientVersion
    });
    await client.connect(transport);
    return {
      name: serverName,
      config,
      client,
      transport
    };
  }

  private toAgentTool(route: ToolRoute): AgentTool {
    return {
      name: route.info.name,
      description: `[MCP:${route.info.serverName}] ${route.info.description}`,
      parameters: route.parameters,
      access: route.access,
      executionMode: "sequential",
      execute: async (args) => {
        try {
          const result = await route.server.client.callTool({
            name: route.info.rawName,
            arguments: asRecord(args)
          });
          return formatMcpToolResult(result, route.info);
        } catch (error) {
          return {
            content: `MCP tool ${route.info.serverName}/${route.info.rawName} failed: ${formatError(error)}`,
            isError: true,
            details: {
              serverName: route.info.serverName,
              toolName: route.info.rawName
            }
          };
        }
      }
    };
  }
}

function createTransport(config: McpServerConfig): Transport {
  if (config.transport === "stdio") {
    return new StdioClientTransport({
      command: config.command,
      args: config.args,
      env: config.env,
      cwd: config.cwd,
      stderr: "pipe"
    });
  }

  if (config.transport === "streamable-http" || config.transport === "streamable_http" || config.transport === "streamableHttp" || config.transport === "http") {
    return new StreamableHTTPClientTransport(parseUrl(config.url), {
      requestInit: headersToRequestInit(config.headers),
      sessionId: config.sessionId
    });
  }

  if (config.transport === "sse") {
    return new SSEClientTransport(parseUrl(config.url), {
      requestInit: headersToRequestInit(config.headers),
      eventSourceInit: config.headers ? { fetch: fetchWithHeaders(config.headers) } : undefined
    });
  }

  ensureWebSocketGlobal();
  return new WebSocketClientTransport(parseUrl(config.url));
}

function ensureWebSocketGlobal(): void {
  if (typeof globalThis.WebSocket === "function") {
    return;
  }
  (globalThis as unknown as { WebSocket: unknown }).WebSocket = WebSocket;
}

function parseUrl(value: string): URL {
  try {
    return new URL(value);
  } catch (error) {
    throw new Error(`Invalid MCP server URL ${value}: ${formatError(error)}`);
  }
}

function headersToRequestInit(headers: Record<string, string> | undefined): RequestInit | undefined {
  return headers ? { headers } : undefined;
}

function fetchWithHeaders(headers: Record<string, string>): typeof fetch {
  return (input, init) => {
    const mergedHeaders = new Headers(init?.headers);
    for (const [key, value] of Object.entries(headers)) {
      mergedHeaders.set(key, value);
    }
    return fetch(input, {
      ...init,
      headers: mergedHeaders
    });
  };
}

function isToolEnabled(toolName: string, config: McpServerConfig): boolean {
  if (config.enabledTools && !config.enabledTools.includes(toolName)) {
    return false;
  }
  if (config.disabledTools?.includes(toolName)) {
    return false;
  }
  return true;
}

function inferToolAccess(tool: ListedMcpTool): ToolAccess {
  if (tool.annotations?.readOnlyHint === true) {
    return "read";
  }
  if (tool.annotations?.destructiveHint === true) {
    return "write";
  }
  return "execute";
}

function uniqueModelToolName(serverName: string, toolName: string, usedNames: Set<string>): string {
  const base = sanitizeToolName(`mcp_${serverName}_${toolName}`);
  if (!usedNames.has(base)) {
    return base;
  }
  const hash = createHash("sha1").update(`${serverName}:${toolName}`).digest("hex").slice(0, 8);
  const hashed = `${base.slice(0, Math.max(1, 55 - hash.length))}_${hash}`;
  if (!usedNames.has(hashed)) {
    return hashed;
  }
  let index = 2;
  while (usedNames.has(`${hashed}_${index}`)) {
    index += 1;
  }
  return `${hashed}_${index}`;
}

function sanitizeToolName(value: string): string {
  const sanitized = value.toLowerCase().replace(/[^a-z0-9_-]+/g, "_").replace(/^_+|_+$/g, "");
  if (!sanitized) {
    return "mcp_tool";
  }
  if (/^[a-z_]/.test(sanitized)) {
    return sanitized.slice(0, 64);
  }
  return `mcp_${sanitized}`.slice(0, 64);
}

function asRecord(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return {};
  }
  return value as Record<string, unknown>;
}

function normalizeInputSchema(schema: unknown): JsonSchema {
  if (!schema || typeof schema !== "object" || Array.isArray(schema)) {
    return { type: "object", additionalProperties: true };
  }
  return schema as JsonSchema;
}

function formatMcpToolResult(result: unknown, info: McpToolInfo): ToolResult {
  if (!result || typeof result !== "object") {
    return {
      content: String(result ?? ""),
      details: { serverName: info.serverName, toolName: info.rawName, raw: result }
    };
  }

  const record = result as Record<string, unknown>;
  if (!Array.isArray(record.content)) {
    return {
      content: JSON.stringify(record.toolResult ?? record),
      isError: readBoolean(record.isError),
      details: { serverName: info.serverName, toolName: info.rawName, raw: result }
    };
  }

  const text: string[] = [];
  const nonText: unknown[] = [];
  for (const block of record.content) {
    if (isTextContent(block)) {
      text.push(block.text);
    } else {
      nonText.push(block);
    }
  }

  return {
    content: text.join("\n\n") || (nonText.length > 0 ? `[MCP tool returned ${nonText.length} non-text content block(s).]` : ""),
    isError: readBoolean(record.isError),
    details: {
      serverName: info.serverName,
      toolName: info.rawName,
      structuredContent: record.structuredContent,
      nonTextContent: nonText,
      meta: record._meta
    }
  };
}

function isTextContent(value: unknown): value is { type: "text"; text: string } {
  return Boolean(value && typeof value === "object" && (value as { type?: unknown }).type === "text" && typeof (value as { text?: unknown }).text === "string");
}

function readBoolean(value: unknown): boolean | undefined {
  return typeof value === "boolean" ? value : undefined;
}

function formatError(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
