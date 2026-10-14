import assert from "node:assert/strict";
import test from "node:test";
import { connectLocalSession } from "../src/lib/local-session.ts";

test("reconnect only establishes a session and reads service identity", async () => {
  const requests = [];
  const original = globalThis.fetch;
  globalThis.fetch = async (url, options) => {
    requests.push([url, options.method || "GET"]);
    return new Response("{}");
  };
  try {
    await connectLocalSession(new AbortController().signal);
    assert.deepEqual(requests, [
      ["/session", "POST"],
      ["/api/service", "GET"],
    ]);
  } finally {
    globalThis.fetch = original;
  }
});

test("an unresponsive local service returns a retryable timeout", async () => {
  const original = globalThis.fetch;
  const keepAlive = setInterval(() => {}, 100);
  globalThis.fetch = async (_url, { signal }) =>
    new Promise((_resolve, reject) => {
      signal.addEventListener("abort", () => reject(signal.reason), {
        once: true,
      });
    });
  try {
    await assert.rejects(
      connectLocalSession(new AbortController().signal, 20),
      /连接本机服务超时/,
    );
  } finally {
    clearInterval(keepAlive);
    globalThis.fetch = original;
  }
});

test("a successful cookie response does not hide an unavailable backend", async () => {
  const original = globalThis.fetch;
  globalThis.fetch = async (url) =>
    new Response("{}", { status: url === "/session" ? 200 : 503 });
  try {
    await assert.rejects(
      connectLocalSession(new AbortController().signal),
      /本机服务暂未就绪/,
    );
  } finally {
    globalThis.fetch = original;
  }
});
