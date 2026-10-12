import assert from "node:assert/strict";
import test from "node:test";
import {
  mergePublicMessages,
  messagesForThread,
} from "../src/lib/public-messages.ts";
import { activityAnchors } from "../src/lib/activity-groups.ts";
import { combineUsage } from "../src/lib/activity-usage.ts";

const message = (id, content, type = "ai") => ({ id, content, type });

test("switching a running stream never archives its messages into another thread", () => {
  const running = [message("new-user", "第二条正在等待", "human")];
  const oldThread = [message("old-user", "第一条", "human")];
  const beforeReset = messagesForThread(
    running,
    "api\0old",
    "api\0new",
    "old",
    "new",
  );
  assert.deepEqual(beforeReset, []);
  const historyStillStale = messagesForThread(
    running,
    "api\0old",
    "api\0old",
    "old",
    "new",
  );
  assert.deepEqual(historyStillStale, []);
  assert.strictEqual(
    mergePublicMessages(oldThread, historyStillStale),
    oldThread,
  );
  assert.strictEqual(
    messagesForThread(oldThread, "api\0old", "api\0old", "old", "old"),
    oldThread,
  );
});

test("new-thread optimistic messages and interrupted commentary stay visible under their owner", () => {
  const first = [message("first", "开始讨论", "human")];
  assert.strictEqual(
    messagesForThread(first, "api\0new", "api\0new", "new"),
    first,
  );
  const published = [...first, message("progress", "我已经找到入口")];
  const stopped = messagesForThread([], "api\0new", "api\0new", "new", "new");
  assert.strictEqual(mergePublicMessages(published, stopped), published);
  assert.deepEqual(messagesForThread(published, "api\0", "api\0new", null), []);
});

test("usage combines cache weights but uses only the latest context", () => {
  const row = (input, cached, used) => ({
    input_tokens: input,
    cached_input_tokens: cached,
    cache_known_input_tokens: input,
    context: { used_tokens: used, window_tokens: 1000000, estimated: false },
  });
  const result = combineUsage([row(100, 80, 120), row(900, 180, 940)]);
  assert.equal(result.cache_rate, 0.26);
  assert.equal(result.context.used_tokens, 940);
  assert.equal(combineUsage([]).cache_rate, null);
  assert.equal(combineUsage([row(100, 80, 120), {}]).context, null);
});

test("older recovery and empty stream do not swallow published commentary", () => {
  const published = [
    message("user", "直接检查", "human"),
    message("progress", "我找到了入口，接下来检查调用方。"),
  ];
  assert.strictEqual(mergePublicMessages(published, []), published);
  assert.strictEqual(mergePublicMessages(published, [published[0]]), published);
  assert.strictEqual(
    mergePublicMessages(published, [message("progress", "")]),
    published,
  );
  assert.strictEqual(
    mergePublicMessages(published, [message("progress", "我找到了入口")]),
    published,
  );
});

test("new replies append once and growing text keeps its original position", () => {
  const first = [message("progress", "正在检查")];
  const updated = mergePublicMessages(first, [
    message("progress", "正在检查调用方"),
    message("final", "检查完成"),
  ]);
  assert.deepEqual(
    updated.map((item) => item.id),
    ["progress", "final"],
  );
  assert.equal(updated[0].content, "正在检查调用方");
  assert.strictEqual(
    mergePublicMessages(updated, structuredClone(updated)),
    updated,
  );
  assert.equal(first[0].content, "正在检查");
});

test("recovery then live fragments converge without another archive update", () => {
  const latest = {
    ...message("p1", "已取得材料"),
    response_metadata: { snapshot: "live" },
  };
  const archive = [latest];
  const fragments = [
    { ...latest, response_metadata: { snapshot: "recovery" } },
    structuredClone(latest),
  ];
  for (let i = 0; i < 100; i++)
    assert.strictEqual(mergePublicMessages(archive, fragments), archive);
});

test("optimistic and queued user inputs survive partial views without duplicates", () => {
  const users = [
    message("u1", "开始检查", "human"),
    message("u2", "补充要求", "human"),
  ];
  const restored = mergePublicMessages(users, [message("reply", "正在处理")]);
  assert.deepEqual(
    mergePublicMessages(restored, [users[0]]).map((item) => item.id),
    ["u1", "u2", "reply"],
  );
  assert.strictEqual(mergePublicMessages(restored, [users[1]]), restored);
});

test("all private steps share their human turn while public replies remain separate", () => {
  const messages = [
    message("u1", "检查", "human"),
    message("p1", "正在检查"),
    message("p2", "继续核对"),
    message("u2", "补充", "human"),
  ];
  assert.deepEqual(
    [...activityAnchors(messages)],
    [
      ["u1", "u1"],
      ["p1", "u1"],
      ["p2", "u1"],
      ["u2", "u2"],
    ],
  );
});
