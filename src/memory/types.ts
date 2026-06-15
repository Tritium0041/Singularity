export const WORKSPACE_NOTE_KINDS = ["note", "decision", "file", "error", "todo"] as const;

export type WorkspaceNoteKind = (typeof WORKSPACE_NOTE_KINDS)[number];

export type WorkspaceNote = {
  id: string;
  kind: WorkspaceNoteKind;
  content: string;
  createdAt: string;
  updatedAt: string;
};

export type WorkspaceState = {
  notes: WorkspaceNote[];
};

export const MEMORY_ENTRY_SOURCES = ["user", "agent", "tool"] as const;

export type MemoryEntrySource = (typeof MEMORY_ENTRY_SOURCES)[number];

export type MemoryEntry = {
  id: string;
  content: string;
  tags: string[];
  source?: MemoryEntrySource;
  createdAt: string;
  updatedAt: string;
  lineStart?: number;
  lineEnd?: number;
};

export type MemorySearchResult = {
  entry: MemoryEntry;
  score: number;
  matchedTerms: string[];
  snippet: string;
};
