import {
  WORKSPACE_NOTE_KINDS,
  type WorkspaceNote,
  type WorkspaceNoteKind,
  type WorkspaceNoteListItem,
  type WorkspaceState
} from "./types.js";

export const DEFAULT_WORKSPACE_NOTE_PREVIEW_CHARS = 80;

export class WorkspaceMemory {
  private readonly notes: WorkspaceNote[];

  constructor(initial?: WorkspaceState) {
    this.notes = (initial?.notes ?? []).map(cloneNote);
    for (const note of this.notes) {
      validateKind(note.kind);
      if (note.content.trim() === "") {
        throw new Error("Workspace note content must be non-empty.");
      }
    }
  }

  get state(): WorkspaceState {
    return {
      notes: this.notes.map(cloneNote)
    };
  }

  write(input: { kind?: WorkspaceNoteKind; content: string }): WorkspaceNote {
    const kind = input.kind ?? "note";
    validateKind(kind);
    const content = normalizeContent(input.content);
    const now = new Date().toISOString();
    const note: WorkspaceNote = {
      id: createWorkspaceNoteId(),
      kind,
      content,
      createdAt: now,
      updatedAt: now
    };
    this.notes.push(note);
    return cloneNote(note);
  }

  read(filter: { id?: string; kind?: WorkspaceNoteKind } = {}): WorkspaceNote[] {
    if (filter.kind !== undefined) {
      validateKind(filter.kind);
    }
    return this.notes
      .filter((note) => (filter.id === undefined || note.id === filter.id) && (filter.kind === undefined || note.kind === filter.kind))
      .map(cloneNote);
  }

  list(filter: { kind?: WorkspaceNoteKind; previewCharacters?: number } = {}): WorkspaceNoteListItem[] {
    if (filter.kind !== undefined) {
      validateKind(filter.kind);
    }
    const previewCharacters = normalizePreviewCharacters(filter.previewCharacters ?? DEFAULT_WORKSPACE_NOTE_PREVIEW_CHARS);
    return this.notes
      .filter((note) => filter.kind === undefined || note.kind === filter.kind)
      .map((note) => ({
        id: note.id,
        kind: note.kind,
        preview: previewContent(note.content, previewCharacters),
        contentLength: Array.from(note.content).length,
        createdAt: note.createdAt,
        updatedAt: note.updatedAt
      }));
  }

  update(input: { id: string; kind?: WorkspaceNoteKind; content?: string }): WorkspaceNote {
    const note = this.notes.find((candidate) => candidate.id === input.id);
    if (!note) {
      throw new Error(`Workspace note not found: ${input.id}`);
    }
    if (input.kind !== undefined) {
      validateKind(input.kind);
      note.kind = input.kind;
    }
    if (input.content !== undefined) {
      note.content = normalizeContent(input.content);
    }
    note.updatedAt = new Date().toISOString();
    return cloneNote(note);
  }

  delete(id: string): boolean {
    const index = this.notes.findIndex((note) => note.id === id);
    if (index === -1) {
      return false;
    }
    this.notes.splice(index, 1);
    return true;
  }

  clear(): void {
    this.notes.splice(0, this.notes.length);
  }
}

function cloneNote(note: WorkspaceNote): WorkspaceNote {
  return { ...note };
}

function normalizeContent(content: string): string {
  const normalized = content.trim();
  if (normalized === "") {
    throw new Error("Workspace note content must be non-empty.");
  }
  return normalized;
}

function normalizePreviewCharacters(value: number): number {
  if (!Number.isFinite(value)) {
    throw new Error("Workspace note preview length must be finite.");
  }
  return Math.min(200, Math.max(10, Math.floor(value)));
}

function previewContent(content: string, maxCharacters: number): string {
  const compact = content.replace(/\s+/g, " ").trim();
  const characters = Array.from(compact);
  if (characters.length <= maxCharacters) {
    return compact;
  }
  return `${characters.slice(0, maxCharacters).join("")}...`;
}

function validateKind(kind: string): asserts kind is WorkspaceNoteKind {
  if (!(WORKSPACE_NOTE_KINDS as readonly string[]).includes(kind)) {
    throw new Error(`Invalid workspace note kind: ${kind}`);
  }
}

function createWorkspaceNoteId(): string {
  return `note_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;
}
