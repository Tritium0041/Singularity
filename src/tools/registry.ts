import type { JsonSchema, ToolCall, ToolResultMessage } from "../types.js";
import type { LlmToolSpec } from "../llm/types.js";

export type ToolResult = {
  content: string;
  isError?: boolean;
  details?: unknown;
};

export type ToolExecutionContext = {
  toolCallId: string;
  signal?: AbortSignal;
};

export type ToolExecutionMode = "sequential" | "parallel";

export type ToolAccess = "read" | "write" | "execute" | "planner";

export type AgentTool = {
  name: string;
  description: string;
  parameters: JsonSchema;
  access?: ToolAccess;
  executionMode?: ToolExecutionMode;
  execute(args: unknown, context: ToolExecutionContext): Promise<ToolResult> | ToolResult;
};

export class ToolRegistry {
  private readonly tools = new Map<string, AgentTool>();

  constructor(tools: AgentTool[] = []) {
    for (const tool of tools) {
      this.register(tool);
    }
  }

  register(tool: AgentTool): void {
    if (this.tools.has(tool.name)) {
      throw new Error(`Tool already registered: ${tool.name}`);
    }
    this.tools.set(tool.name, tool);
  }

  get(name: string): AgentTool | undefined {
    return this.tools.get(name);
  }

  list(): AgentTool[] {
    return [...this.tools.values()];
  }

  toLlmToolSpecs(): LlmToolSpec[] {
    return this.list().map((tool) => ({
      name: tool.name,
      description: tool.description,
      parameters: tool.parameters
    }));
  }
}

export class ToolExecutor {
  constructor(private readonly registry: ToolRegistry) {}

  async execute(toolCall: ToolCall, signal?: AbortSignal): Promise<ToolResultMessage> {
    const tool = this.registry.get(toolCall.name);
    if (!tool) {
      return toToolResultMessage(toolCall, {
        content: `Tool not found: ${toolCall.name}`,
        isError: true
      });
    }

    try {
      validateArguments(tool.parameters, toolCall.arguments);
      const result = await tool.execute(toolCall.arguments, {
        toolCallId: toolCall.id,
        signal
      });
      return toToolResultMessage(toolCall, result);
    } catch (error) {
      return toToolResultMessage(toolCall, {
        content: error instanceof Error ? error.message : String(error),
        isError: true
      });
    }
  }
}

function toToolResultMessage(toolCall: ToolCall, result: ToolResult): ToolResultMessage {
  return {
    role: "tool",
    toolCallId: toolCall.id,
    toolName: toolCall.name,
    content: result.content,
    isError: result.isError,
    details: result.details
  };
}

function validateArguments(schema: JsonSchema, args: unknown): void {
  if (schema.type !== "object") {
    return;
  }
  if (!args || typeof args !== "object" || Array.isArray(args)) {
    throw new Error("Tool arguments must be an object.");
  }

  const record = args as Record<string, unknown>;
  for (const key of schema.required ?? []) {
    if (!(key in record)) {
      throw new Error(`Missing required argument: ${key}`);
    }
  }

  const properties = schema.properties ?? {};
  if (schema.additionalProperties === false) {
    for (const key of Object.keys(record)) {
      if (!(key in properties)) {
        throw new Error(`Unknown argument: ${key}`);
      }
    }
  }

  for (const [key, propertySchema] of Object.entries(properties)) {
    if (!(key in record)) {
      continue;
    }
    validatePrimitiveType(key, propertySchema.type, record[key]);
  }
}

function validatePrimitiveType(key: string, type: unknown, value: unknown): void {
  if (type === undefined) {
    return;
  }
  if (type === "string" && typeof value !== "string") {
    throw new Error(`Argument ${key} must be a string.`);
  }
  if ((type === "number" || type === "integer") && typeof value !== "number") {
    throw new Error(`Argument ${key} must be a number.`);
  }
  if (type === "boolean" && typeof value !== "boolean") {
    throw new Error(`Argument ${key} must be a boolean.`);
  }
}
