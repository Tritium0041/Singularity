import type { AgentTool } from "../tools/registry.js";
import { MEMORY_ENTRY_SOURCES, WORKSPACE_NOTE_KINDS, type MemoryEntrySource, type WorkspaceNoteKind } from "./types.js";
import type { MarkdownMemoryStore } from "./memory-store.js";
import { normalizeTags } from "./memory-store.js";
import type { WorkspaceMemory } from "./workspace.js";

export const WRITE_NOTE_TOOL_NAME = "write_note";
export const LIST_NOTES_TOOL_NAME = "list_notes";
export const READ_NOTE_TOOL_NAME = "read_note";
export const UPDATE_WORKSPACE_TOOL_NAME = "update_workspace";
export const STORE_MEMORY_TOOL_NAME = "store_memory";
export const SEARCH_MEMORY_TOOL_NAME = "search_memory";

export function createWorkspaceTools(workspace: WorkspaceMemory): AgentTool[] {
  return [
    {
      name: WRITE_NOTE_TOOL_NAME,
      description:
        "Write a note to the current task workspace. Use this for important current-task state that should survive context compaction.",
      access: "write",
      executionMode: "sequential",
      parameters: {
        type: "object",
        additionalProperties: false,
        properties: {
          kind: {
            type: "string",
            enum: [...WORKSPACE_NOTE_KINDS],
            description: "Workspace note kind. Defaults to note."
          },
          content: {
            type: "string",
            description: "Non-empty note content."
          }
        },
        required: ["content"]
      },
      execute(args) {
        const note = workspace.write({
          kind: readOptionalWorkspaceKind(args, "kind"),
          content: readStringArg(args, "content")
        });
        return jsonToolResult(note);
      }
    },
    {
      name: LIST_NOTES_TOOL_NAME,
      description:
        "List compact summaries of notes in the current task workspace. Use this to discover note ids and distinguish notes before reading full content.",
      access: "read",
      parameters: {
        type: "object",
        additionalProperties: false,
        properties: {
          kind: {
            type: "string",
            enum: [...WORKSPACE_NOTE_KINDS],
            description: "Optional note kind filter."
          }
        }
      },
      execute(args) {
        const notes = workspace.list({
          kind: readOptionalWorkspaceKind(args, "kind")
        });
        return jsonToolResult({ notes });
      }
    },
    {
      name: READ_NOTE_TOOL_NAME,
      description: "Read notes from the current task workspace by optional id or kind filter.",
      access: "read",
      parameters: {
        type: "object",
        additionalProperties: false,
        properties: {
          id: {
            type: "string",
            description: "Optional note id."
          },
          kind: {
            type: "string",
            enum: [...WORKSPACE_NOTE_KINDS],
            description: "Optional note kind filter."
          }
        }
      },
      execute(args) {
        const notes = workspace.read({
          id: readOptionalStringArg(args, "id"),
          kind: readOptionalWorkspaceKind(args, "kind")
        });
        return jsonToolResult({ notes });
      }
    },
    {
      name: UPDATE_WORKSPACE_TOOL_NAME,
      description: "Update or delete a note in the current task workspace.",
      access: "write",
      executionMode: "sequential",
      parameters: {
        type: "object",
        additionalProperties: false,
        properties: {
          id: {
            type: "string",
            description: "Workspace note id."
          },
          kind: {
            type: "string",
            enum: [...WORKSPACE_NOTE_KINDS],
            description: "Optional replacement note kind."
          },
          content: {
            type: "string",
            description: "Optional replacement note content."
          },
          delete: {
            type: "boolean",
            description: "Delete the note instead of updating it."
          }
        },
        required: ["id"]
      },
      execute(args) {
        const id = readStringArg(args, "id");
        if (readOptionalBooleanArg(args, "delete") === true) {
          const deleted = workspace.delete(id);
          return jsonToolResult({ deleted, id });
        }
        const note = workspace.update({
          id,
          kind: readOptionalWorkspaceKind(args, "kind"),
          content: readOptionalStringArg(args, "content")
        });
        return jsonToolResult(note);
      }
    }
  ];
}

export function createMemoryStoreTools(store: MarkdownMemoryStore, options: { maxMemoryResults?: number } = {}): AgentTool[] {
  const defaultMaxResults = clampResultLimit(options.maxMemoryResults ?? 5);
  return [
    {
      name: STORE_MEMORY_TOOL_NAME,
      description:
        "Store durable long-term memory as a local Markdown entry. Only save user preferences, project conventions, or reusable lessons. Never store secrets, credentials, or one-off temporary facts.",
      access: "write",
      executionMode: "sequential",
      parameters: {
        type: "object",
        additionalProperties: false,
        properties: {
          content: {
            type: "string",
            description: "Non-empty durable memory content."
          },
          tags: {
            type: "array",
            items: { type: "string" },
            description: "Optional tags for later search."
          },
          source: {
            type: "string",
            enum: [...MEMORY_ENTRY_SOURCES],
            description: "Optional memory source."
          }
        },
        required: ["content"]
      },
      async execute(args) {
        const entry = await store.store({
          content: readStringArg(args, "content"),
          tags: readOptionalStringArrayArg(args, "tags"),
          source: readOptionalMemorySource(args, "source")
        });
        return jsonToolResult(entry);
      }
    },
    {
      name: SEARCH_MEMORY_TOOL_NAME,
      description:
        "Search durable long-term memory for remembered user preferences, project conventions, and prior reusable solutions. Use this before relying on memory.",
      access: "read",
      parameters: {
        type: "object",
        additionalProperties: false,
        properties: {
          query: {
            type: "string",
            description: "Non-empty search query."
          },
          tags: {
            type: "array",
            items: { type: "string" },
            description: "Optional tag filter. Entries matching any tag are eligible."
          },
          maxResults: {
            type: "number",
            description: `Maximum results from 1 to 20. Defaults to ${defaultMaxResults}.`
          }
        },
        required: ["query"]
      },
      async execute(args) {
        const results = await store.search(readStringArg(args, "query"), {
          tags: readOptionalStringArrayArg(args, "tags"),
          maxResults: readOptionalNumberArg(args, "maxResults") ?? defaultMaxResults
        });
        const payload = {
          results: results.map((result) => ({
            id: result.entry.id,
            content: result.entry.content,
            tags: result.entry.tags,
            source: result.entry.source,
            createdAt: result.entry.createdAt,
            updatedAt: result.entry.updatedAt,
            lineStart: result.entry.lineStart,
            lineEnd: result.entry.lineEnd,
            score: result.score,
            matchedTerms: result.matchedTerms,
            snippet: result.snippet
          }))
        };
        return jsonToolResult(payload);
      }
    }
  ];
}

function jsonToolResult(details: unknown) {
  return {
    content: JSON.stringify(details, null, 2),
    details
  };
}

function readRecord(args: unknown): Record<string, unknown> {
  if (!args || typeof args !== "object" || Array.isArray(args)) {
    throw new Error("Tool arguments must be an object.");
  }
  return args as Record<string, unknown>;
}

function readStringArg(args: unknown, key: string): string {
  const value = readRecord(args)[key];
  if (typeof value !== "string" || value.trim() === "") {
    throw new Error(`Argument ${key} must be a non-empty string.`);
  }
  return value.trim();
}

function readOptionalStringArg(args: unknown, key: string): string | undefined {
  const value = readRecord(args)[key];
  if (value === undefined) {
    return undefined;
  }
  if (typeof value !== "string") {
    throw new Error(`Argument ${key} must be a string.`);
  }
  return value;
}

function readOptionalNumberArg(args: unknown, key: string): number | undefined {
  const value = readRecord(args)[key];
  if (value === undefined) {
    return undefined;
  }
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`Argument ${key} must be a finite number.`);
  }
  return value;
}

function readOptionalBooleanArg(args: unknown, key: string): boolean | undefined {
  const value = readRecord(args)[key];
  if (value === undefined) {
    return undefined;
  }
  if (typeof value !== "boolean") {
    throw new Error(`Argument ${key} must be a boolean.`);
  }
  return value;
}

function readOptionalStringArrayArg(args: unknown, key: string): string[] | undefined {
  const value = readRecord(args)[key];
  if (value === undefined) {
    return undefined;
  }
  if (!Array.isArray(value) || value.some((item) => typeof item !== "string")) {
    throw new Error(`Argument ${key} must be an array of strings.`);
  }
  return normalizeTags(value);
}

function readOptionalWorkspaceKind(args: unknown, key: string): WorkspaceNoteKind | undefined {
  const value = readOptionalStringArg(args, key);
  if (value === undefined) {
    return undefined;
  }
  if (!(WORKSPACE_NOTE_KINDS as readonly string[]).includes(value)) {
    throw new Error(`Argument ${key} must be one of: ${WORKSPACE_NOTE_KINDS.join(", ")}.`);
  }
  return value as WorkspaceNoteKind;
}

function readOptionalMemorySource(args: unknown, key: string): MemoryEntrySource | undefined {
  const value = readOptionalStringArg(args, key);
  if (value === undefined) {
    return undefined;
  }
  if (!(MEMORY_ENTRY_SOURCES as readonly string[]).includes(value)) {
    throw new Error(`Argument ${key} must be one of: ${MEMORY_ENTRY_SOURCES.join(", ")}.`);
  }
  return value as MemoryEntrySource;
}

function clampResultLimit(value: number): number {
  return Math.min(20, Math.max(1, Math.floor(value)));
}
