export type TruncationDirection = "head" | "tail";

export type TruncationDetails = {
  truncated: boolean;
  direction: TruncationDirection;
  truncatedBy: "lines" | "bytes" | null;
  totalLines: number;
  totalBytes: number;
  outputLines: number;
  outputBytes: number;
  maxLines: number;
  maxBytes: number;
};

export type TruncationOptions = {
  maxLines?: number;
  maxBytes?: number;
};

export type TruncatedText = {
  content: string;
  details: TruncationDetails;
};

export const DEFAULT_MAX_LINES = 10_000;
export const DEFAULT_MAX_BYTES = 80 * 1024;

export function truncateHead(text: string, options: TruncationOptions = {}): TruncatedText {
  return truncate(text, "head", options);
}

export function truncateTail(text: string, options: TruncationOptions = {}): TruncatedText {
  return truncate(text, "tail", options);
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) {
    return `${bytes}B`;
  }
  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)}KB`;
  }
  return `${(bytes / (1024 * 1024)).toFixed(1)}MB`;
}

function truncate(text: string, direction: TruncationDirection, options: TruncationOptions): TruncatedText {
  const maxLines = options.maxLines ?? DEFAULT_MAX_LINES;
  const maxBytes = options.maxBytes ?? DEFAULT_MAX_BYTES;
  const lines = splitLines(text);
  const totalBytes = Buffer.byteLength(text, "utf8");
  const totalLines = lines.length;

  if (totalLines <= maxLines && totalBytes <= maxBytes) {
    return {
      content: text,
      details: {
        truncated: false,
        direction,
        truncatedBy: null,
        totalLines,
        totalBytes,
        outputLines: totalLines,
        outputBytes: totalBytes,
        maxLines,
        maxBytes
      }
    };
  }

  const selected = direction === "head" ? takeFromHead(lines, maxLines, maxBytes) : takeFromTail(lines, maxLines, maxBytes);
  const content = selected.lines.join("\n");
  const outputBytes = Buffer.byteLength(content, "utf8");

  return {
    content,
    details: {
      truncated: true,
      direction,
      truncatedBy: selected.truncatedBy,
      totalLines,
      totalBytes,
      outputLines: selected.lines.length,
      outputBytes,
      maxLines,
      maxBytes
    }
  };
}

function takeFromHead(lines: string[], maxLines: number, maxBytes: number): { lines: string[]; truncatedBy: "lines" | "bytes" } {
  const output: string[] = [];
  let bytes = 0;
  let truncatedBy: "lines" | "bytes" = "lines";

  for (const line of lines) {
    if (output.length >= maxLines) {
      truncatedBy = "lines";
      break;
    }
    const lineBytes = Buffer.byteLength(line, "utf8") + (output.length > 0 ? 1 : 0);
    if (bytes + lineBytes > maxBytes) {
      truncatedBy = "bytes";
      if (output.length === 0 && maxLines > 0) {
        const clipped = sliceUtf8ByBytes(line, maxBytes, "head");
        if (clipped) {
          output.push(clipped);
        }
      }
      break;
    }
    output.push(line);
    bytes += lineBytes;
  }

  return { lines: output, truncatedBy };
}

function takeFromTail(lines: string[], maxLines: number, maxBytes: number): { lines: string[]; truncatedBy: "lines" | "bytes" } {
  const output: string[] = [];
  let bytes = 0;
  let truncatedBy: "lines" | "bytes" = "lines";

  for (let index = lines.length - 1; index >= 0; index -= 1) {
    if (output.length >= maxLines) {
      truncatedBy = "lines";
      break;
    }
    const line = lines[index] ?? "";
    const lineBytes = Buffer.byteLength(line, "utf8") + (output.length > 0 ? 1 : 0);
    if (bytes + lineBytes > maxBytes) {
      truncatedBy = "bytes";
      if (output.length === 0 && maxLines > 0) {
        const clipped = sliceUtf8ByBytes(line, maxBytes, "tail");
        if (clipped) {
          output.unshift(clipped);
        }
      }
      break;
    }
    output.unshift(line);
    bytes += lineBytes;
  }

  return { lines: output, truncatedBy };
}

function sliceUtf8ByBytes(text: string, maxBytes: number, direction: TruncationDirection): string {
  if (maxBytes <= 0) {
    return "";
  }

  const characters = Array.from(text);
  if (direction === "head") {
    let bytes = 0;
    const output: string[] = [];
    for (const character of characters) {
      const characterBytes = Buffer.byteLength(character, "utf8");
      if (bytes + characterBytes > maxBytes) {
        break;
      }
      output.push(character);
      bytes += characterBytes;
    }
    return output.join("");
  }

  let bytes = 0;
  const output: string[] = [];
  for (let index = characters.length - 1; index >= 0; index -= 1) {
    const character = characters[index] ?? "";
    const characterBytes = Buffer.byteLength(character, "utf8");
    if (bytes + characterBytes > maxBytes) {
      break;
    }
    output.unshift(character);
    bytes += characterBytes;
  }
  return output.join("");
}

function splitLines(text: string): string[] {
  if (!text) {
    return [];
  }
  const lines = text.split("\n");
  if (text.endsWith("\n")) {
    lines.pop();
  }
  return lines;
}
