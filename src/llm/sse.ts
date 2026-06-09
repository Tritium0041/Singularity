export async function* parseServerSentEvents(body: ReadableStream<Uint8Array>): AsyncIterable<unknown> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }

    buffer += decoder.decode(value, { stream: true });
    yield* drainSseFrames(buffer, (remaining) => {
      buffer = remaining;
    });
  }

  buffer += decoder.decode();
  if (buffer.trim()) {
    yield* parseSseFrame(buffer);
  }
}

function* drainSseFrames(buffer: string, setRemaining: (remaining: string) => void): Iterable<unknown> {
  const normalized = buffer.replace(/\r\n/g, "\n");
  const parts = normalized.split("\n\n");
  setRemaining(parts.pop() ?? "");
  for (const part of parts) {
    yield* parseSseFrame(part);
  }
}

function* parseSseFrame(frame: string): Iterable<unknown> {
  const data = frame
    .split(/\r?\n/)
    .filter((line) => line.startsWith("data:"))
    .map((line) => line.slice("data:".length).trimStart())
    .join("\n")
    .trim();

  if (!data || data === "[DONE]") {
    return;
  }

  yield JSON.parse(data) as unknown;
}
