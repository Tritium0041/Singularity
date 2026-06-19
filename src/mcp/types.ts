import type { ToolAccess } from "../tools/registry.js";

export type McpToolFilter = {
  enabledTools?: string[];
  disabledTools?: string[];
};

export type McpServerConfig = McpToolFilter & {
  transport: "stdio";
  command: string;
  args?: string[];
  env?: Record<string, string>;
  cwd?: string;
  enabled?: boolean;
  access?: ToolAccess;
};

export type McpConfig = {
  servers: Record<string, McpServerConfig>;
};

export type McpToolInfo = {
  serverName: string;
  rawName: string;
  name: string;
  description: string;
  access: ToolAccess;
};

export type McpDiagnostic = {
  level: "warning" | "error";
  serverName: string;
  message: string;
  toolName?: string;
};

export type McpManagerOptions = {
  config: McpConfig;
  clientName?: string;
  clientVersion?: string;
};
