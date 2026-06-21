export type McpToolFilter = {
  enabledTools?: string[];
  disabledTools?: string[];
};

export type McpBaseServerConfig = McpToolFilter & {
  enabled?: boolean;
};

export type McpStdioServerConfig = McpBaseServerConfig & {
  transport: "stdio";
  command: string;
  args?: string[];
  env?: Record<string, string>;
  cwd?: string;
};

export type McpHttpServerConfig = McpBaseServerConfig & {
  transport: "streamable-http" | "streamable_http" | "streamableHttp" | "http";
  url: string;
  headers?: Record<string, string>;
  sessionId?: string;
};

export type McpSseServerConfig = McpBaseServerConfig & {
  transport: "sse";
  url: string;
  headers?: Record<string, string>;
};

export type McpWebSocketServerConfig = McpBaseServerConfig & {
  transport: "websocket" | "websocket-client" | "ws";
  url: string;
};

export type McpServerConfig = McpStdioServerConfig | McpHttpServerConfig | McpSseServerConfig | McpWebSocketServerConfig;

export type McpConfig = {
  servers: Record<string, McpServerConfig>;
};

export type McpToolInfo = {
  serverName: string;
  rawName: string;
  name: string;
  description: string;
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
