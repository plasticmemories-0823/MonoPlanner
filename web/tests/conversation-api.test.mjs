import assert from "node:assert/strict";
import test from "node:test";
import {
  ConversationMonitor,
  requestJson,
  readStorageFailure,
  StorageRequestError,
} from "../src/lib/conversation-api.ts";

const delay = (ms = 100) => new Promise((resolve) => setTimeout(resolve, ms));
const snapshot = (cursor, text = "已收到") => ({
  thread_id: "thread-a",
  event_cursor: cursor,
  messages: [{ id: "message-1", type: "ai", content: text, source: "chat" }],
  status: "idle",
  chat_status: "idle",
  work: null,
  repo: "",
  workspace_bound: false,
  pending_messages: [],
  activities: [],
  usage: {},
  read_only: false,
});
class Connection {
  closed = false;
  listeners = new Map();
  onerror = null;
  constructor(url) {
    this.url = url;
  }
  addEventListener(type, listener) {
    this.listeners.set(type, listener);
  }
  close() {
    this.closed = true;
  }
  emit(seq, conversation_id = "thread-a") {
    this.listeners.get("update")?.({
      data: JSON.stringify({ seq, kind: "message", conversation_id, payload: {} }),
    });
  }
}
function fixture(fetcher) {
  const applied = [],
    errors = [],
    health = [],
    connections = [];
  const monitor = new ConversationMonitor(
    "/api",
    "thread-a",
    {
      snapshot: (value) => applied.push(value),
      error: (value) => errors.push(value),
      storageError: (value) => health.push(value),
    },
    fetcher,
    (url) => {
      const connection = new Connection(url);
      connections.push(connection);
      return connection;
    },
  );
  return { monitor, applied, errors, health, connections };
}

const storageFault = {
  code: "storage_unavailable",
  dispatch_suspended: true,
  operation: "work_state",
  message: "internal diagnostic must not become public text",
};

test("saved delivery proposals reach the view without creating a command", async () => {
  const proposal = snapshot(8);
  proposal.messages = [{ id: "work-proposal:" + "a".repeat(64), type: "ai",
    source: "work_proposal", content: "Confirmed delivery proposal", proposal_status: "" }];
  const value = fixture(async () => Response.json(proposal));
  await value.monitor.start();
  assert.equal(value.errors.filter(Boolean).length, 0);
  assert.equal(value.applied[0].messages[0].source, "work_proposal");
  assert.equal(value.applied[0].work, null);
  value.monitor.dispose();
});

test("a storage rejection retains its health fact without retrying a command", async () => {
  let calls = 0;
  await assert.rejects(
    requestJson("/commands", { method: "POST" }, async () => {
      calls++;
      return Response.json({ storage_error: storageFault }, { status: 503 });
    }),
    (error) => {
      assert.ok(error instanceof StorageRequestError);
      assert.equal(error.storageError.dispatch_suspended, true);
      assert.match(error.message, /记录写入失败/);
      assert.ok(!error.message.includes("internal diagnostic"));
      return true;
    },
  );
  assert.equal(calls, 1);
  assert.equal(readStorageFailure({ code: "different", dispatch_suspended: true }), null);
});

test("storage health events do not require a writable event cursor or replace public messages", async () => {
  const value = fixture(async () => Response.json(snapshot(12)));
  await value.monitor.start();
  value.connections[0].listeners.get("health")({
    data: JSON.stringify({ storage_error: storageFault }),
  });
  assert.equal(value.health.at(-1).code, "storage_unavailable");
  assert.equal(value.applied.length, 1);
  value.monitor.dispose();
  value.connections[0].listeners.get("health")({ data: JSON.stringify({ storage_error: null }) });
  assert.equal(value.health.length, 1, "a disposed stream cannot change current service health");
});

test("failed snapshot reads surface storage failure and retain the last valid conversation", async () => {
  let calls = 0;
  const value = fixture(async () =>
    ++calls === 1
      ? Response.json(snapshot(9))
      : Response.json({ storage_error: storageFault }, { status: 503 }),
  );
  await value.monitor.start();
  await value.monitor.refresh();
  assert.equal(value.applied.length, 1);
  assert.equal(value.health.at(-1).dispatch_suspended, true);
  assert.equal(calls, 2);
  value.monitor.dispose();
});

test("one event subscription invalidates one authoritative snapshot without merging work text", async () => {
  let calls = 0,
    cursor = 0;
  const value = fixture(async () => {
    calls++;
    return Response.json(snapshot(cursor));
  });
  await value.monitor.start();
  assert.equal(value.connections.length, 1);
  assert.match(value.connections[0].url, /events\?after=0$/);
  cursor = 2;
  value.connections[0].emit(1);
  value.connections[0].emit(2);
  value.connections[0].emit(2);
  value.connections[0].emit(99, "other-thread");
  await delay();
  assert.equal(calls, 2);
  assert.equal(value.applied.length, 2);
  assert.deepEqual(value.applied[1].messages, snapshot(2).messages);
  value.monitor.dispose();
  assert.equal(value.connections[0].closed, true);
});

test("concurrent invalidations never create concurrent snapshot requests", async () => {
  let reads = 0,
    active = 0,
    peak = 0;
  const value = fixture(async () => {
    active++;
    peak = Math.max(peak, active);
    reads++;
    await delay(12);
    active--;
    return Response.json(snapshot(reads));
  });
  await Promise.all([value.monitor.refresh(), value.monitor.refresh(), value.monitor.refresh()]);
  assert.equal(peak, 1);
  assert.equal(reads, 2);
  value.monitor.dispose();
});

test("switching away aborts reads and ignores late messages instead of archiving into a new conversation", async () => {
  let resolve, signal;
  const value = fixture(async (_url, init) => {
    signal = init.signal;
    return new Promise((done) => {
      resolve = done;
    });
  });
  const pending = value.monitor.start();
  value.monitor.dispose();
  assert.equal(signal.aborted, true);
  resolve(Response.json(snapshot(8, "old thread")));
  await pending;
  assert.equal(value.applied.length, 0);
  assert.equal(value.connections.length, 0);
});

test("older snapshot cannot overwrite a newer snapshot; read failures preserve the last messages", async () => {
  const replies = [snapshot(5, "newer"), snapshot(4, "older"), new Error("connection lost")];
  const value = fixture(async () => {
    const next = replies.shift();
    if (next instanceof Error) throw next;
    return Response.json(next);
  });
  await value.monitor.start();
  await value.monitor.refresh();
  await value.monitor.refresh();
  assert.equal(value.applied.length, 1);
  assert.equal(value.applied[0].messages[0].content, "newer");
  assert.match(value.errors.at(-1), /connection lost/);
  value.monitor.dispose();
});

test("broken event connection stops and reconnect only reads snapshot and event cursor", async () => {
  const requests = [];
  const value = fixture(async (url, init) => {
    requests.push({ url, method: init.method || "GET" });
    return Response.json(snapshot(12));
  });
  await value.monitor.start();
  value.connections[0].onerror(new Event("error"));
  assert.equal(value.connections[0].closed, true);
  await delay();
  assert.equal(requests.length, 1);
  await value.monitor.start();
  assert.equal(value.connections.length, 2);
  assert.match(value.connections[1].url, /after=12$/);
  assert.ok(
    requests.every((request) => request.method === "GET" && request.url.endsWith("/snapshot")),
  );
  value.monitor.dispose();
});

test("malformed or wrong-owner snapshot is not published", async () => {
  const value = fixture(async () => Response.json({ ...snapshot(1), thread_id: "another" }));
  await value.monitor.start();
  assert.equal(value.applied.length, 0);
  assert.equal(value.connections.length, 0);
  assert.match(value.errors.at(-1), /格式不完整/);
  value.monitor.dispose();
});

test("legacy read-only snapshot never opens a live work subscription", async () => {
  const value = fixture(async () => Response.json({ ...snapshot(0), read_only: true }));
  await value.monitor.start();
  assert.equal(value.applied.length, 1);
  assert.equal(value.connections.length, 0);
  value.monitor.dispose();
});

test("unknown command response never automatically retries or creates a replacement command", async () => {
  let calls = 0;
  await assert.rejects(
    requestJson(
      "/api/conversations/thread-a/commands",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id: "command-1", kind: "message", text: "看看进度" }),
      },
      async (_url, init) => {
        calls++;
        assert.equal(JSON.parse(init.body).id, "command-1");
        throw new Error("connection lost after send");
      },
    ),
    /connection lost/,
  );
  assert.equal(calls, 1);
});
