import { isAbsolute, relative, resolve } from "node:path";

export function readStringArg(args: unknown, key: string, options: { allowEmpty?: boolean } = {}): string {
  const value = readRecord(args)[key];
  if (typeof value !== "string" || (!options.allowEmpty && value.trim() === "")) {
    throw new Error(`Argument ${key} must be a non-empty string.`);
  }
  return value;
}

export function readOptionalStringArg(args: unknown, key: string): string | undefined {
  const value = readRecord(args)[key];
  if (value === undefined) {
    return undefined;
  }
  if (typeof value !== "string") {
    throw new Error(`Argument ${key} must be a string.`);
  }
  return value;
}

export function readOptionalNumberArg(args: unknown, key: string): number | undefined {
  const value = readRecord(args)[key];
  if (value === undefined) {
    return undefined;
  }
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`Argument ${key} must be a finite number.`);
  }
  return value;
}

export function resolveWithinRoot(rootDir: string, path: string): string {
  const root = resolve(rootDir);
  const target = resolve(root, path || ".");
  const rel = relative(root, target);
  if (rel === "" || (!rel.startsWith("..") && !isAbsolute(rel))) {
    return target;
  }
  throw new Error(`Path escapes rootDir: ${path}`);
}

function readRecord(args: unknown): Record<string, unknown> {
  if (!args || typeof args !== "object" || Array.isArray(args)) {
    throw new Error("Tool arguments must be an object.");
  }
  return args as Record<string, unknown>;
}
