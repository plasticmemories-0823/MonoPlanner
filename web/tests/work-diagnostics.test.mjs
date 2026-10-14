import assert from "node:assert/strict";
import test from "node:test";
import {
  appendDiagnosticPage,
  WorkActivityMonitor,
  visibleWorkActivity,
  groupActivity,
} from "../src/lib/work-diagnostics.ts";

test("diagnostic pagination appends a fragment once and replaces it with the saved response", () => {
  const first = { id: "call-1", status: "running", detail: "第一段", version: 1 };
  const part = { ...first, detail: "第二段", version: 2, append: true };
  const combined = appendDiagnosticPage([first], [part, part]);
  assert.equal(combined[0].detail, "第一段第二段");
  assert.equal(combined.length, 1);
  const complete = { ...first, status: "completed", detail: "完整原文", version: 3, append: false };
  assert.equal(appendDiagnosticPage(combined, [complete])[0].detail, "完整原文");
});

const delay = (ms = 35) => new Promise((resolve) => setTimeout(resolve, ms));
const page = (items = [], cursors = {}, more = false) => ({ items, cursors, more });

test("active work reads incremental records, drains final pages and stops without commands", async () => {
  const reads = [],
    views = [];
  let monitor;
  monitor = new WorkActivityMonitor(
    async (after) => {
      reads.push(after);
      const version = reads.length;
      if (version === 2) monitor.update("task:completed:2", false);
      return page(
        [
          {
            id: "task-thinking-1",
            detail: String(version),
            version,
            status: version < 3 ? "running" : "completed",
          },
        ],
        { task: version },
        version === 2,
      );
    },
    (items) => views.push(items),
    5,
  );
  try {
    monitor.update("task:running:1", true);
    await delay(80);
    assert.equal(reads.length, 3);
    assert.deepEqual(reads, [{}, { task: 1 }, { task: 2 }]);
    assert.equal(views.at(-1)[0].detail, "3");
    assert.equal(views.at(-1)[0].status, "completed");
  } finally {
    monitor.dispose();
  }
});

test("concurrent state changes never overlap reads and terminal transition still gets a final read", async () => {
  let release,
    active = 0,
    peak = 0,
    reads = 0;
  const monitor = new WorkActivityMonitor(
    async () => {
      reads++;
      peak = Math.max(peak, ++active);
      if (reads === 1)
        await new Promise((resolve) => {
          release = resolve;
        });
      active--;
      return page();
    },
    () => {},
    5,
  );
  try {
    monitor.update("running:1", true);
    monitor.update("running:2", true);
    monitor.update("paused:3", false);
    release();
    await delay(60);
    assert.equal(reads, 2);
    assert.equal(peak, 1);
  } finally {
    monitor.dispose();
  }
});

test("display failures retain rows and cursor and only explicit reconnect retries", async () => {
  const reads = [],
    views = [];
  const monitor = new WorkActivityMonitor(
    async (after) => {
      reads.push(after);
      if (reads.length === 2) throw new Error("offline");
      return page(
        [{ id: "saved", detail: "saved text", version: reads.length, status: "running" }],
        { task: 7 },
      );
    },
    (items, error) => views.push({ items, error }),
    5,
  );
  try {
    monitor.update("running:1", true);
    await delay();
    assert.equal(reads.length, 2);
    assert.equal(views.at(-1).error, "offline");
    assert.equal(views.at(-1).items[0].detail, "saved text");
    monitor.update("paused:2", false);
    await delay();
    assert.equal(reads.length, 2);
    monitor.reconnect();
    await delay();
    assert.equal(reads.length, 3);
    assert.deepEqual(reads[2], { task: 7 });
    assert.equal(views.at(-1).error, "");
  } finally {
    monitor.dispose();
  }
});

test("switching conversation aborts reads and ignores a late result", async () => {
  let release, signal;
  const views = [];
  const monitor = new WorkActivityMonitor(
    async (_, incomingSignal) => {
      signal = incomingSignal;
      await new Promise((resolve) => {
        release = resolve;
      });
      return page([{ id: "old-thread", detail: "not current" }]);
    },
    (items) => views.push(items),
  );
  monitor.update("running", true);
  monitor.dispose();
  release();
  await delay();
  assert.equal(signal.aborted, true);
  assert.deepEqual(views, []);
});

test("preparation has visible status before reasoning arrives and terminal state closes stale spinners", () => {
  const work = { task_id: "task", input_id: "user-1", status: "preparing" };
  const initial = visibleWorkActivity([], work);
  assert.equal(initial[0].label, "正在准备项目环境");
  assert.equal(initial[0].after_id, "user-1");
  const rows = [
    { id: "task-thinking-1", task_id: "task", kind: "thinking", status: "running", detail: "片段" },
  ];
  assert.equal(visibleWorkActivity(rows, { ...work, status: "paused" })[0].status, "interrupted");
  assert.equal(
    visibleWorkActivity(rows, { ...work, status: "execution_unknown" })[0].status,
    "unknown",
  );
  assert.equal(rows[0].status, "running", "display projection does not rewrite saved originals");
  assert.equal(
    visibleWorkActivity([], { ...work, status: "preparation_failed" })[0].status,
    "failed",
  );
});

test("work stays with its original request while simultaneous chat keeps its own message group", () => {
  const messages = [
    { id: "user-1", type: "human", content: "开始工作" },
    { id: "reply-1", type: "ai", content: "已交接" },
    { id: "user-2", type: "human", content: "普通询问" },
  ];
  const work = { id: "work-1", after_id: "reply-1", source: "work", detail: "command" };
  const chat = { id: "chat-1", after_id: "user-2", source: "chat", detail: "reasoning" };
  const groups = groupActivity(messages, [work, chat]);
  assert.deepEqual(groups.get("user-1"), [work]);
  assert.deepEqual(groups.get("user-2"), [chat]);
  assert.equal(messages.length, 3, "display grouping does not add messages");
});

test("late live diagnostics cannot replace a committed final result", () => {
  const saved = { id: "call-1", status: "completed", detail: "原件", seq: 19 };
  const rows = appendDiagnosticPage(
    [saved],
    [
      { id: "call-1", status: "running", detail: "旧片段", version: 100, append: true },
      { ...saved, detail: "旧原件", seq: 18 },
    ],
  );
  assert.equal(rows[0].detail, "原件");
});

test("empty pages and rejected duplicate fragments preserve the existing array", () => {
  const saved = [{ id: "call", status: "completed", detail: "原件", version: 3 }];
  assert.equal(appendDiagnosticPage(saved, []), saved);
  assert.equal(appendDiagnosticPage(saved, [{ ...saved[0], version: 2 }]), saved);
});

test("empty increments advance the cursor without notifying unchanged views", async () => {
  const reads = [],
    views = [];
  const monitor = new WorkActivityMonitor(
    async (after) => {
      reads.push(after);
      const sequence = reads.length;
      return {
        ...page(sequence === 1 ? [{ id: "saved", status: "completed", detail: "原件" }] : [], {
          task: sequence,
        }),
        display_warning: sequence === 3 || sequence === 4 ? "显示警告" : "",
      };
    },
    (items, warning) => views.push({ items, warning }),
  );
  try {
    for (let sequence = 1; sequence <= 5; sequence++) {
      monitor.update(`paused:${sequence}`, false);
      await delay(5);
    }
    assert.deepEqual(reads, [{}, { task: 1 }, { task: 2 }, { task: 3 }, { task: 4 }]);
    assert.deepEqual(
      views.map((view) => view.warning),
      ["", "显示警告", ""],
    );
    assert.ok(views.every((view) => view.items === views[0].items));
  } finally {
    monitor.dispose();
  }
});

test("starting a new task cannot revive unfinished display rows from the previous task", () => {
  const original = [
    {
      id: "old-thinking-1",
      task_id: "old",
      kind: "thinking",
      label: "正在思考",
      status: "running",
      detail: "旧片段",
    },
    {
      id: "new-operation-1",
      task_id: "new",
      kind: "operation",
      label: "正在读取",
      status: "running",
      detail: "new.py",
    },
  ];
  const rows = visibleWorkActivity(original, { task_id: "new", status: "executing" });
  assert.equal(rows[0].status, "unknown");
  assert.equal(rows[0].label, "旧任务的执行状态待核对");
  assert.equal(rows[0].detail, "旧片段");
  assert.equal(rows[1].status, "running");
  assert.equal(original[0].status, "running", "originals remain untouched by display projection");
});
