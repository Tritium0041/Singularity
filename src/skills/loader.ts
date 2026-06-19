import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { basename, dirname, extname, isAbsolute, join, resolve } from "node:path";
import { parse } from "yaml";
import type { Skill, SkillDiagnostic, SkillLoadOptions, SkillLoadResult } from "./types.js";

type SkillFrontmatter = {
  name?: unknown;
  description?: unknown;
  "disable-model-invocation"?: unknown;
  disableModelInvocation?: unknown;
};

type ParsedMarkdown = {
  frontmatter: SkillFrontmatter;
  body: string;
};

export const DEFAULT_SKILL_ROOTS = [".singularity/skills"];

export async function loadSkills(options: SkillLoadOptions = {}): Promise<SkillLoadResult> {
  return loadSkillsSync(options);
}

export function loadSkillsSync(options: SkillLoadOptions = {}): SkillLoadResult {
  const cwd = options.cwd ?? process.cwd();
  const roots = options.roots ?? DEFAULT_SKILL_ROOTS;
  const diagnostics: SkillDiagnostic[] = [];
  const discovered: Skill[] = [];

  for (const root of roots) {
    const rootPath = resolvePath(cwd, root);
    if (!existsSync(rootPath)) {
      continue;
    }
    scanRoot(rootPath, rootPath, discovered, diagnostics);
  }

  const byName = new Map<string, Skill>();
  for (const skill of discovered) {
    const existing = byName.get(skill.name);
    if (existing) {
      diagnostics.push({
        level: "warning",
        message: `Skill name collision: ${skill.name}. Keeping ${existing.filePath}; ignoring ${skill.filePath}.`,
        path: skill.filePath,
        name: skill.name
      });
      continue;
    }
    byName.set(skill.name, skill);
  }

  return {
    skills: [...byName.values()].sort((a, b) => a.name.localeCompare(b.name)),
    diagnostics
  };
}

function scanRoot(rootPath: string, currentPath: string, skills: Skill[], diagnostics: SkillDiagnostic[]): void {
  const metadata = safeStat(currentPath);
  if (!metadata) {
    return;
  }

  if (metadata.isFile()) {
    if (isMarkdownFile(currentPath)) {
      loadSkillFile(rootPath, currentPath, "file", skills, diagnostics);
    }
    return;
  }

  if (!metadata.isDirectory()) {
    return;
  }

  const skillFile = join(currentPath, "SKILL.md");
  if (existsSync(skillFile)) {
    loadSkillFile(rootPath, skillFile, "directory", skills, diagnostics);
    return;
  }

  const entries = readdirSync(currentPath, { withFileTypes: true });
  for (const entry of entries.sort((a, b) => a.name.localeCompare(b.name))) {
    if (entry.name.startsWith(".")) {
      continue;
    }
    const childPath = join(currentPath, entry.name);
    if (entry.isDirectory()) {
      scanRoot(rootPath, childPath, skills, diagnostics);
      continue;
    }
    if (entry.isFile() && currentPath === rootPath && isMarkdownFile(childPath)) {
      loadSkillFile(rootPath, childPath, "file", skills, diagnostics);
    }
  }
}

function loadSkillFile(
  rootPath: string,
  filePath: string,
  kind: "directory" | "file",
  skills: Skill[],
  diagnostics: SkillDiagnostic[]
): void {
  let raw: string;
  try {
    raw = readFileSync(filePath, "utf8");
  } catch (error) {
    diagnostics.push({ level: "error", message: `Failed to read skill: ${formatError(error)}`, path: filePath });
    return;
  }

  let parsed: ParsedMarkdown;
  try {
    parsed = parseMarkdown(raw);
  } catch (error) {
    diagnostics.push({ level: "error", message: `Failed to parse skill frontmatter: ${formatError(error)}`, path: filePath });
    return;
  }

  const baseDir = kind === "directory" ? dirname(filePath) : dirname(filePath);
  const fallbackName = kind === "directory" ? basename(baseDir) : basename(filePath, extname(filePath));
  const name = normalizeSkillName(readOptionalString(parsed.frontmatter.name) ?? fallbackName);
  if (!name) {
    diagnostics.push({ level: "warning", message: "Skill has an invalid empty name.", path: filePath });
    return;
  }

  const description = readOptionalString(parsed.frontmatter.description);
  if (!description) {
    diagnostics.push({
      level: "warning",
      message: `Skill ${name} is missing a required description and was not loaded.`,
      path: filePath,
      name
    });
    return;
  }

  const disableModelInvocation = readOptionalBoolean(parsed.frontmatter["disable-model-invocation"]) ?? readOptionalBoolean(parsed.frontmatter.disableModelInvocation) ?? false;
  skills.push({
    name,
    description,
    filePath,
    baseDir,
    disableModelInvocation,
    source: {
      root: rootPath,
      kind
    }
  });
}

function parseMarkdown(raw: string): ParsedMarkdown {
  if (!raw.startsWith("---\n") && !raw.startsWith("---\r\n")) {
    return { frontmatter: {}, body: raw };
  }

  const newline = raw.startsWith("---\r\n") ? "\r\n" : "\n";
  const endMarker = `${newline}---${newline}`;
  const endIndex = raw.indexOf(endMarker, 4);
  if (endIndex === -1) {
    return { frontmatter: {}, body: raw };
  }

  const frontmatterText = raw.slice(4, endIndex);
  const body = raw.slice(endIndex + endMarker.length);
  const frontmatter = parse(frontmatterText) as unknown;
  if (!frontmatter || typeof frontmatter !== "object" || Array.isArray(frontmatter)) {
    return { frontmatter: {}, body };
  }
  return { frontmatter: frontmatter as SkillFrontmatter, body };
}

function resolvePath(cwd: string, path: string): string {
  return isAbsolute(path) ? path : resolve(cwd, path);
}

function safeStat(path: string) {
  try {
    return statSync(path);
  } catch {
    return undefined;
  }
}

function isMarkdownFile(path: string): boolean {
  return extname(path).toLowerCase() === ".md";
}

function readOptionalString(value: unknown): string | undefined {
  if (typeof value !== "string") {
    return undefined;
  }
  const trimmed = value.trim();
  return trimmed || undefined;
}

function readOptionalBoolean(value: unknown): boolean | undefined {
  if (typeof value === "boolean") {
    return value;
  }
  if (typeof value !== "string") {
    return undefined;
  }
  const normalized = value.trim().toLowerCase();
  if (normalized === "true" || normalized === "yes" || normalized === "1" || normalized === "on") {
    return true;
  }
  if (normalized === "false" || normalized === "no" || normalized === "0" || normalized === "off") {
    return false;
  }
  return undefined;
}

function normalizeSkillName(value: string): string {
  return value.trim().toLowerCase().replace(/[^a-z0-9-]+/g, "-").replace(/^-+|-+$/g, "");
}

function formatError(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
