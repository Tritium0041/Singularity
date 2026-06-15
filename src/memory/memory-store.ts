import { appendFile, mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { MEMORY_ENTRY_SOURCES, type MemoryEntry, type MemoryEntrySource, type MemorySearchResult } from "./types.js";

const DEFAULT_MEMORY_PATH = ".agent-memory/MEMORY.md";
const MEMORY_HEADER = "# Singularity Memory\n\nLong-term memory for this local agent. Entries are append-only Markdown blocks.\n";

type ParsedMemoryBlock = {
  entry: MemoryEntry;
  raw: string;
  body: string;
};

export class MarkdownMemoryStore {
  private readonly filePath: string;

  constructor(options: { path?: string } = {}) {
    this.filePath = resolve(options.path ?? DEFAULT_MEMORY_PATH);
  }

  get path(): string {
    return this.filePath;
  }

  async ensureFile(): Promise<void> {
    await mkdir(dirname(this.filePath), { recursive: true });
    try {
      await readFile(this.filePath, "utf8");
    } catch (error) {
      if (isNotFoundError(error)) {
        await writeFile(this.filePath, MEMORY_HEADER, "utf8");
        return;
      }
      throw error;
    }
  }

  async store(input: { content: string; tags?: string[]; source?: MemoryEntrySource }): Promise<MemoryEntry> {
    const content = normalizeContent(input.content);
    const tags = normalizeTags(input.tags);
    if (input.source !== undefined) {
      validateSource(input.source);
    }
    await this.ensureFile();

    const now = new Date().toISOString();
    const entry: MemoryEntry = {
      id: createMemoryEntryId(new Date(now)),
      content,
      tags,
      source: input.source,
      createdAt: now,
      updatedAt: now
    };
    const existing = await readFile(this.filePath, "utf8");
    const separator = existing.endsWith("\n\n") ? "" : existing.endsWith("\n") ? "\n" : "\n\n";
    await appendFile(this.filePath, `${separator}${formatEntry(entry)}\n`, "utf8");
    return { ...entry, tags: [...entry.tags] };
  }

  async update(input: { id: string; content: string; tags?: string[]; source?: MemoryEntrySource }): Promise<MemoryEntry> {
    const content = normalizeContent(input.content);
    const tags = normalizeTags(input.tags);
    if (input.source !== undefined) {
      validateSource(input.source);
    }
    await this.ensureFile();

    const text = await readFile(this.filePath, "utf8");
    const blocks = parseMemoryBlocks(text);
    const target = blocks.find((block) => block.entry.id === input.id);
    if (!target) {
      throw new Error(`Memory entry not found: ${input.id}`);
    }

    const entry: MemoryEntry = {
      ...target.entry,
      content,
      tags,
      source: input.source,
      updatedAt: new Date().toISOString()
    };
    const updated = replaceLines(text, target.entry.lineStart, target.entry.lineEnd, formatEntry(entry));
    await writeFile(this.filePath, updated, "utf8");
    return cloneEntry(entry);
  }

  async upsertByTag(input: { tag: string; content: string; tags?: string[]; source?: MemoryEntrySource }): Promise<{ entry: MemoryEntry; created: boolean }> {
    const tag = normalizeTag(input.tag);
    if (!tag) {
      throw new Error("Memory upsert tag must be non-empty.");
    }
    const existing = (await this.list({ tag }))[0];
    const tags = normalizeTags([tag, ...(input.tags ?? [])]);
    if (!existing) {
      return {
        entry: await this.store({
          content: input.content,
          tags,
          source: input.source
        }),
        created: true
      };
    }

    return {
      entry: await this.update({
        id: existing.id,
        content: input.content,
        tags,
        source: input.source ?? existing.source
      }),
      created: false
    };
  }

  async list(options: { limit?: number; tag?: string } = {}): Promise<MemoryEntry[]> {
    const entries = (await this.readBlocks()).map((block) => block.entry);
    const tag = options.tag === undefined ? undefined : normalizeTag(options.tag);
    const filtered = tag ? entries.filter((entry) => entry.tags.includes(tag)) : entries;
    const limit = options.limit === undefined ? filtered.length : Math.max(0, Math.floor(options.limit));
    return filtered.slice(0, limit).map(cloneEntry);
  }

  async search(query: string, options: { maxResults?: number; tags?: string[] } = {}): Promise<MemorySearchResult[]> {
    const normalizedQuery = query.trim().toLowerCase();
    if (normalizedQuery === "") {
      throw new Error("Memory search query must be non-empty.");
    }

    const maxResults = clampResultLimit(options.maxResults ?? 5);
    const filterTags = normalizeTags(options.tags);
    const terms = buildSearchTerms(normalizedQuery);
    const results: MemorySearchResult[] = [];

    for (const block of await this.readBlocks()) {
      if (filterTags.length > 0 && !filterTags.some((tag) => block.entry.tags.includes(tag))) {
        continue;
      }
      const scored = scoreBlock(block, normalizedQuery, terms, filterTags);
      if (scored.score <= 0) {
        continue;
      }
      results.push({
        entry: cloneEntry(block.entry),
        score: scored.score,
        matchedTerms: scored.matchedTerms,
        snippet: buildSnippet(block.raw, block.body, scored.matchedTerms)
      });
    }

    return results
      .sort((left, right) => right.score - left.score || right.entry.updatedAt.localeCompare(left.entry.updatedAt))
      .slice(0, maxResults);
  }

  async clear(): Promise<void> {
    await mkdir(dirname(this.filePath), { recursive: true });
    await writeFile(this.filePath, MEMORY_HEADER, "utf8");
  }

  private async readBlocks(): Promise<ParsedMemoryBlock[]> {
    await this.ensureFile();
    const text = await readFile(this.filePath, "utf8");
    return parseMemoryBlocks(text);
  }
}

function formatEntry(entry: MemoryEntry): string {
  const lines = [
    `## ${entry.id}`,
    `- tags: ${entry.tags.join(", ")}`,
    ...(entry.source ? [`- source: ${entry.source}`] : []),
    `- created_at: ${entry.createdAt}`,
    `- updated_at: ${entry.updatedAt}`,
    "",
    entry.content
  ];
  return lines.join("\n");
}

function replaceLines(text: string, lineStart: number | undefined, lineEnd: number | undefined, replacement: string): string {
  if (!lineStart || !lineEnd || lineStart < 1 || lineEnd < lineStart) {
    throw new Error("Memory entry is missing source line information.");
  }

  const hadTrailingNewline = text.endsWith("\n");
  const lines = text.split("\n");
  if (hadTrailingNewline) {
    lines.pop();
  }

  lines.splice(lineStart - 1, lineEnd - lineStart, ...replacement.split("\n"));
  return `${lines.join("\n")}\n`;
}

function parseMemoryBlocks(text: string): ParsedMemoryBlock[] {
  const lines = text.split("\n");
  const startIndexes: number[] = [];
  for (let index = 0; index < lines.length; index += 1) {
    if (/^## mem_/.test(lines[index] ?? "")) {
      startIndexes.push(index);
    }
  }

  const blocks: ParsedMemoryBlock[] = [];
  for (let blockIndex = 0; blockIndex < startIndexes.length; blockIndex += 1) {
    const start = startIndexes[blockIndex]!;
    const endExclusive = startIndexes[blockIndex + 1] ?? lines.length;
    const blockLines = lines.slice(start, endExclusive);
    const parsed = parseMemoryBlock(blockLines, start + 1, endExclusive);
    if (parsed) {
      blocks.push(parsed);
    }
  }
  return blocks;
}

function parseMemoryBlock(lines: string[], lineStart: number, lineEnd: number): ParsedMemoryBlock | undefined {
  const heading = lines[0]?.match(/^##\s+(mem_[^\s]+)/);
  if (!heading) {
    return undefined;
  }
  const metadataEnd = lines.findIndex((line, index) => index > 0 && line.trim() === "");
  const metadataLines = metadataEnd === -1 ? lines.slice(1) : lines.slice(1, metadataEnd);
  const bodyLines = metadataEnd === -1 ? [] : lines.slice(metadataEnd + 1);
  const metadata = parseMetadata(metadataLines);
  const content = trimTrailingBlankLines(bodyLines).join("\n").trim();
  if (content === "") {
    return undefined;
  }

  return {
    entry: {
      id: heading[1]!,
      content,
      tags: normalizeTags(metadata.tags ? metadata.tags.split(",") : []),
      source: parseOptionalSource(metadata.source),
      createdAt: metadata.created_at ?? "",
      updatedAt: metadata.updated_at ?? metadata.created_at ?? "",
      lineStart,
      lineEnd
    },
    raw: trimTrailingBlankLines(lines).join("\n"),
    body: content
  };
}

function parseMetadata(lines: string[]): Record<string, string> {
  const metadata: Record<string, string> = {};
  for (const line of lines) {
    const match = line.match(/^-\s*([^:]+):\s*(.*)$/);
    if (!match) {
      continue;
    }
    metadata[match[1]!.trim()] = match[2]!.trim();
  }
  return metadata;
}

function scoreBlock(
  block: ParsedMemoryBlock,
  normalizedQuery: string,
  terms: string[],
  filterTags: string[]
): { score: number; matchedTerms: string[] } {
  const raw = block.raw.toLowerCase();
  const tagText = block.entry.tags.map((tag) => tag.toLowerCase());
  const matched = new Set<string>();
  let score = 0;

  if (raw.includes(normalizedQuery)) {
    score += 6;
    matched.add(normalizedQuery);
  }

  for (const term of terms) {
    for (const tag of tagText) {
      if (tag === term) {
        score += 4;
        matched.add(term);
      } else if (tag.includes(term)) {
        score += 2;
        matched.add(term);
      }
    }
    if (raw.includes(term)) {
      score += 1;
      matched.add(term);
    }
  }

  if (filterTags.length > 0 && filterTags.some((tag) => tagText.includes(tag))) {
    score += 2;
    for (const tag of filterTags) {
      matched.add(tag);
    }
  }

  return {
    score,
    matchedTerms: [...matched]
  };
}

function buildSearchTerms(normalizedQuery: string): string[] {
  const splitTerms = normalizedQuery
    .split(/[\s,.;:!?()[\]{}"'`/\\|]+/g)
    .map((term) => term.trim())
    .filter((term) => term.length > 0);
  return [...new Set([normalizedQuery, ...splitTerms])];
}

function buildSnippet(raw: string, body: string, matchedTerms: string[]): string {
  const normalizedBody = body.toLowerCase();
  for (const term of matchedTerms) {
    const index = normalizedBody.indexOf(term.toLowerCase());
    if (index !== -1) {
      return compactWhitespace(body.slice(Math.max(0, index - 80), Math.min(body.length, index + term.length + 120)));
    }
  }

  const normalizedRaw = raw.toLowerCase();
  for (const term of matchedTerms) {
    const index = normalizedRaw.indexOf(term.toLowerCase());
    if (index !== -1) {
      return compactWhitespace(raw.slice(Math.max(0, index - 80), Math.min(raw.length, index + term.length + 120)));
    }
  }
  return compactWhitespace(body.slice(0, 200));
}

function compactWhitespace(value: string): string {
  return value.replace(/\s+/g, " ").trim();
}

function normalizeContent(content: string): string {
  const normalized = content.trim();
  if (normalized === "") {
    throw new Error("Memory content must be non-empty.");
  }
  return normalized;
}

export function normalizeTags(tags: readonly string[] | undefined): string[] {
  if (!tags) {
    return [];
  }
  return [...new Set(tags.map(normalizeTag).filter((tag) => tag.length > 0))];
}

function normalizeTag(tag: string): string {
  return tag.trim().toLowerCase();
}

function parseOptionalSource(source: string | undefined): MemoryEntrySource | undefined {
  if (!source) {
    return undefined;
  }
  return (MEMORY_ENTRY_SOURCES as readonly string[]).includes(source) ? (source as MemoryEntrySource) : undefined;
}

function validateSource(source: string): asserts source is MemoryEntrySource {
  if (!(MEMORY_ENTRY_SOURCES as readonly string[]).includes(source)) {
    throw new Error(`Invalid memory source: ${source}`);
  }
}

function cloneEntry(entry: MemoryEntry): MemoryEntry {
  return {
    ...entry,
    tags: [...entry.tags]
  };
}

function trimTrailingBlankLines(lines: string[]): string[] {
  const copy = [...lines];
  while (copy.length > 0 && copy.at(-1)?.trim() === "") {
    copy.pop();
  }
  return copy;
}

function clampResultLimit(value: number): number {
  if (!Number.isFinite(value)) {
    return 5;
  }
  return Math.min(20, Math.max(1, Math.floor(value)));
}

function createMemoryEntryId(date: Date): string {
  const stamp = date.toISOString().replace(/[-:]/g, "").replace(/\.\d{3}Z$/, "").replace("T", "_");
  return `mem_${stamp}_${Math.random().toString(36).slice(2, 6)}`;
}

function isNotFoundError(error: unknown): boolean {
  return Boolean(error && typeof error === "object" && "code" in error && (error as { code?: unknown }).code === "ENOENT");
}
