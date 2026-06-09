import { spawn } from "node:child_process";
import { appendFile, mkdir, readFile, readdir, stat, writeFile } from "node:fs/promises";
import { dirname } from "node:path";
import { load } from "cheerio";
import { calculatorTool, mockWeatherTool } from "./builtins.js";
import { readOptionalNumberArg, readOptionalStringArg, readStringArg, resolveWithinRoot } from "./helpers.js";
import type { AgentTool, ToolResult } from "./registry.js";
import { DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, formatBytes, truncateHead, truncateTail, type TruncationOptions } from "./truncation.js";

export type FileSystemToolsOptions = TruncationOptions & {
  rootDir?: string;
};

export type ShellToolOptions = TruncationOptions & {
  rootDir?: string;
  defaultTimeoutMs?: number;
};

export type WebToolsOptions = TruncationOptions & {
  apiKey?: string;
  fetchImpl?: typeof fetch;
  tavilyEndpoint?: string;
};

export type CoreToolset = "basic" | "files" | "shell" | "web" | "all";

export type CoreToolsOptions = {
  rootDir?: string;
  toolset?: CoreToolset;
  fileSystem?: Omit<FileSystemToolsOptions, "rootDir">;
  shell?: Omit<ShellToolOptions, "rootDir">;
  web?: WebToolsOptions;
};

type CommandExecution = {
  stdout: string;
  stderr: string;
  exitCode: number | null;
  timedOut: boolean;
  aborted: boolean;
  durationMs: number;
};

export function createFileSystemTools(options: FileSystemToolsOptions = {}): AgentTool[] {
  const rootDir = options.rootDir ?? process.cwd();
  const truncationOptions = toTruncationOptions(options);

  return [
    {
      name: "read_file",
      description:
        "Read a UTF-8 text file under rootDir. Supports 1-indexed offset and line limit. Large output is truncated from the head.",
      parameters: {
        type: "object",
        properties: {
          path: { type: "string", description: "Path to read, relative to rootDir unless absolute and still inside rootDir." },
          offset: { type: "number", description: "1-indexed line number to start from." },
          limit: { type: "number", description: "Maximum number of lines to read before truncation." }
        },
        required: ["path"],
        additionalProperties: false
      },
      async execute(args) {
        const rawPath = readStringArg(args, "path");
        const offset = Math.max(1, Math.floor(readOptionalNumberArg(args, "offset") ?? 1));
        const limitValue = readOptionalNumberArg(args, "limit");
        const limit = limitValue === undefined ? undefined : Math.max(1, Math.floor(limitValue));
        const path = resolveWithinRoot(rootDir, rawPath);
        const text = await readFile(path, "utf8");
        const lines = text.split("\n");
        const startIndex = offset - 1;
        if (startIndex >= lines.length) {
          throw new Error(`Offset ${offset} is beyond end of file (${lines.length} lines).`);
        }

        const selectedLines = lines.slice(startIndex, limit === undefined ? undefined : startIndex + limit);
        const truncated = truncateHead(selectedLines.join("\n"), truncationOptions);
        const endLine = startIndex + Math.max(truncated.details.outputLines, 1);
        let content = truncated.content;
        if (truncated.details.truncated) {
          content += `\n\n[Truncated: showing lines ${offset}-${endLine} of ${lines.length}; ${formatBytes(
            truncated.details.outputBytes
          )}/${formatBytes(truncated.details.totalBytes)} selected bytes. Use offset=${endLine + 1} to continue.]`;
        } else if (limit !== undefined && startIndex + limit < lines.length) {
          content += `\n\n[${lines.length - (startIndex + limit)} more lines in file. Use offset=${startIndex + limit + 1} to continue.]`;
        }

        return {
          content,
          details: {
            path,
            offset,
            limit,
            truncation: truncated.details
          }
        };
      }
    },
    {
      name: "write_file",
      description: "Write UTF-8 content to a file under rootDir, creating parent directories and overwriting existing content.",
      parameters: {
        type: "object",
        properties: {
          path: { type: "string", description: "Path to write, relative to rootDir unless absolute and still inside rootDir." },
          content: { type: "string", description: "Full content to write." }
        },
        required: ["path", "content"],
        additionalProperties: false
      },
      async execute(args) {
        const rawPath = readStringArg(args, "path");
        const content = readStringArg(args, "content", { allowEmpty: true });
        const path = resolveWithinRoot(rootDir, rawPath);
        await mkdir(dirname(path), { recursive: true });
        await writeFile(path, content, "utf8");
        const bytesWritten = Buffer.byteLength(content, "utf8");
        return {
          content: `Wrote ${bytesWritten} bytes to ${rawPath}.`,
          details: { path, bytesWritten }
        };
      }
    },
    {
      name: "append_file",
      description: "Append UTF-8 content to a file under rootDir, creating parent directories and the file if needed.",
      parameters: {
        type: "object",
        properties: {
          path: { type: "string", description: "Path to append to, relative to rootDir unless absolute and still inside rootDir." },
          content: { type: "string", description: "Content to append." }
        },
        required: ["path", "content"],
        additionalProperties: false
      },
      async execute(args) {
        const rawPath = readStringArg(args, "path");
        const content = readStringArg(args, "content", { allowEmpty: true });
        const path = resolveWithinRoot(rootDir, rawPath);
        await mkdir(dirname(path), { recursive: true });
        await appendFile(path, content, "utf8");
        const bytesWritten = Buffer.byteLength(content, "utf8");
        return {
          content: `Appended ${bytesWritten} bytes to ${rawPath}.`,
          details: { path, bytesWritten }
        };
      }
    },
    {
      name: "list_directory",
      description: "List a directory under rootDir as structured JSON with name, type, and size. Output is sorted and entry-limited.",
      parameters: {
        type: "object",
        properties: {
          path: { type: "string", description: "Directory to list. Defaults to rootDir." },
          limit: { type: "number", description: "Maximum number of entries to return. Defaults to 500." }
        },
        required: [],
        additionalProperties: false
      },
      async execute(args) {
        const rawPath = readOptionalStringArg(args, "path") ?? ".";
        const limit = Math.max(1, Math.floor(readOptionalNumberArg(args, "limit") ?? 500));
        const path = resolveWithinRoot(rootDir, rawPath);
        const metadata = await stat(path);
        if (!metadata.isDirectory()) {
          throw new Error(`Not a directory: ${rawPath}`);
        }

        const dirents = (await readdir(path, { withFileTypes: true })).sort((a, b) => a.name.localeCompare(b.name));
        const entries = [];
        for (const dirent of dirents.slice(0, limit)) {
          const childPath = resolveWithinRoot(path, dirent.name);
          const childStat = await stat(childPath);
          entries.push({
            name: dirent.name,
            type: dirent.isDirectory() ? "directory" : dirent.isFile() ? "file" : "other",
            size: childStat.size
          });
        }
        const entryLimitReached = dirents.length > entries.length;
        const payload = {
          path,
          entries,
          entryLimitReached,
          totalEntries: dirents.length
        };
        const truncated = truncateHead(JSON.stringify(payload, null, 2), truncationOptions);
        return {
          content:
            truncated.content +
            (truncated.details.truncated ? `\n\n[Truncated directory output at ${formatBytes(truncated.details.maxBytes)}.]` : ""),
          details: {
            ...payload,
            truncation: truncated.details
          }
        };
      }
    }
  ];
}

export function createShellTool(options: ShellToolOptions = {}): AgentTool {
  const rootDir = options.rootDir ?? process.cwd();
  const defaultTimeoutMs = options.defaultTimeoutMs ?? 30_000;
  const truncationOptions = toTruncationOptions(options);

  return {
    name: "execute_command",
    description:
      "Execute a shell command in a workdir under rootDir. Captures stdout, stderr, exit code, timeout, and duration. This is not a security sandbox.",
    executionMode: "sequential",
    parameters: {
      type: "object",
      properties: {
        command: { type: "string", description: "Shell command to execute." },
        workdir: { type: "string", description: "Working directory under rootDir. Defaults to rootDir." },
        timeoutMs: { type: "number", description: "Maximum runtime in milliseconds. Defaults to 30000." }
      },
      required: ["command"],
      additionalProperties: false
    },
    async execute(args, context) {
      const command = readStringArg(args, "command");
      const workdir = resolveWithinRoot(rootDir, readOptionalStringArg(args, "workdir") ?? ".");
      const timeoutMs = Math.max(1, Math.floor(readOptionalNumberArg(args, "timeoutMs") ?? defaultTimeoutMs));
      const result = await runCommand(command, workdir, timeoutMs, context.signal);
      const content = formatCommandResult(result, truncationOptions);
      return {
        content: content.content,
        isError: result.aborted || result.timedOut || result.exitCode !== 0,
        details: {
          command,
          workdir,
          ...result,
          truncation: content.details
        }
      };
    }
  };
}

export function createWebTools(options: WebToolsOptions = {}): AgentTool[] {
  const fetchImpl = options.fetchImpl ?? fetch;
  const truncationOptions = toTruncationOptions(options);

  return [
    {
      name: "web_search",
      description: "Search the web through Tavily and return structured title, URL, and snippet results.",
      parameters: {
        type: "object",
        properties: {
          query: { type: "string", description: "Search query." },
          maxResults: { type: "number", description: "Maximum number of results, from 1 to 10. Defaults to 5." }
        },
        required: ["query"],
        additionalProperties: false
      },
      async execute(args) {
        const query = readStringArg(args, "query");
        const maxResults = Math.min(10, Math.max(1, Math.floor(readOptionalNumberArg(args, "maxResults") ?? 5)));
        const apiKey = options.apiKey ?? process.env.TAVILY_API_KEY;
        if (!apiKey) {
          return {
            content: "Missing TAVILY_API_KEY. Set it or pass apiKey to createWebTools().",
            isError: true
          };
        }

        const response = await fetchImpl(options.tavilyEndpoint ?? "https://api.tavily.com/search", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            api_key: apiKey,
            query,
            max_results: maxResults,
            search_depth: "basic"
          })
        });
        const text = await response.text();
        if (!response.ok) {
          return {
            content: `Tavily search failed with HTTP ${response.status}: ${text}`,
            isError: true,
            details: { status: response.status, body: text }
          };
        }

        const body = parseJson(text) as {
          results?: Array<{ title?: string; url?: string; content?: string; score?: number }>;
        };
        const results = (body.results ?? []).slice(0, maxResults).map((result) => ({
          title: result.title ?? "",
          url: result.url ?? "",
          snippet: result.content ?? "",
          score: result.score
        }));
        return {
          content: JSON.stringify({ query, results }, null, 2),
          details: { query, results, raw: body }
        };
      }
    },
    {
      name: "fetch_url",
      description: "Fetch an HTTP(S) URL and return readable text. HTML pages are converted to body text and truncated from the head.",
      parameters: {
        type: "object",
        properties: {
          url: { type: "string", description: "HTTP or HTTPS URL to fetch." }
        },
        required: ["url"],
        additionalProperties: false
      },
      async execute(args) {
        const url = readStringArg(args, "url");
        const parsedUrl = new URL(url);
        if (parsedUrl.protocol !== "http:" && parsedUrl.protocol !== "https:") {
          throw new Error("fetch_url only supports http and https URLs.");
        }

        const response = await fetchImpl(parsedUrl);
        const body = await response.text();
        if (!response.ok) {
          return {
            content: `Fetch failed with HTTP ${response.status}: ${body.slice(0, 1000)}`,
            isError: true,
            details: { url, status: response.status }
          };
        }

        const contentType = response.headers.get("content-type") ?? "";
        const readableText = contentType.includes("html") ? htmlToText(body) : body;
        const truncated = truncateHead(readableText, truncationOptions);
        return {
          content:
            truncated.content +
            (truncated.details.truncated ? `\n\n[Truncated fetched content at ${formatBytes(truncated.details.maxBytes)}.]` : ""),
          details: {
            url,
            status: response.status,
            contentType,
            truncation: truncated.details
          }
        };
      }
    }
  ];
}

export function createCoreTools(options: CoreToolsOptions = {}): AgentTool[] {
  const rootDir = options.rootDir ?? process.cwd();
  const toolset = options.toolset ?? "basic";
  const tools: AgentTool[] = [calculatorTool, mockWeatherTool];

  if (toolset === "files" || toolset === "all") {
    tools.push(...createFileSystemTools({ rootDir, ...options.fileSystem }));
  }
  if (toolset === "shell" || toolset === "all") {
    tools.push(createShellTool({ rootDir, ...options.shell }));
  }
  if (toolset === "web" || toolset === "all") {
    tools.push(...createWebTools(options.web));
  }

  return tools;
}

function toTruncationOptions(options: TruncationOptions): Required<TruncationOptions> {
  return {
    maxLines: options.maxLines ?? DEFAULT_MAX_LINES,
    maxBytes: options.maxBytes ?? DEFAULT_MAX_BYTES
  };
}

function runCommand(command: string, cwd: string, timeoutMs: number, signal?: AbortSignal): Promise<CommandExecution> {
  const startedAt = Date.now();
  return new Promise((resolve, reject) => {
    const child = spawn(command, {
      cwd,
      shell: true,
      env: process.env,
      stdio: ["ignore", "pipe", "pipe"]
    });
    let stdout = "";
    let stderr = "";
    let timedOut = false;
    let aborted = false;
    let settled = false;

    const timeout = setTimeout(() => {
      timedOut = true;
      child.kill("SIGTERM");
    }, timeoutMs);

    const onAbort = () => {
      aborted = true;
      child.kill("SIGTERM");
    };

    child.stdout.on("data", (chunk: Buffer) => {
      stdout += chunk.toString("utf8");
    });
    child.stderr.on("data", (chunk: Buffer) => {
      stderr += chunk.toString("utf8");
    });
    child.on("error", (error) => {
      clearTimeout(timeout);
      signal?.removeEventListener("abort", onAbort);
      if (!settled) {
        settled = true;
        reject(error);
      }
    });
    child.on("close", (exitCode) => {
      clearTimeout(timeout);
      signal?.removeEventListener("abort", onAbort);
      if (!settled) {
        settled = true;
        resolve({
          stdout,
          stderr,
          exitCode,
          timedOut,
          aborted,
          durationMs: Date.now() - startedAt
        });
      }
    });

    if (signal) {
      if (signal.aborted) {
        onAbort();
      } else {
        signal.addEventListener("abort", onAbort, { once: true });
      }
    }
  });
}

function formatCommandResult(result: CommandExecution, options: TruncationOptions): ToolResult {
  const raw = [
    `Exit code: ${result.exitCode ?? "null"}`,
    `Timed out: ${result.timedOut}`,
    `Aborted: ${result.aborted}`,
    `Wall time: ${(result.durationMs / 1000).toFixed(1)} seconds`,
    "",
    "[stdout]",
    result.stdout || "(empty)",
    "",
    "[stderr]",
    result.stderr || "(empty)"
  ].join("\n");
  const truncated = truncateTail(raw, options);
  return {
    content:
      truncated.content +
      (truncated.details.truncated ? `\n\n[Truncated command output; showing tail within ${formatBytes(truncated.details.maxBytes)}.]` : ""),
    details: truncated.details
  };
}

function htmlToText(html: string): string {
  const $ = load(html);
  $("script, style, noscript").remove();
  const title = $("title").first().text().trim();
  const bodyText = $("body").text();
  const lines = bodyText
    .replace(/\r/g, "")
    .split("\n")
    .map((line) => line.replace(/\s+/g, " ").trim())
    .filter(Boolean);
  return [title ? `Title: ${title}` : "", ...lines].filter(Boolean).join("\n");
}

function parseJson(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return {};
  }
}
