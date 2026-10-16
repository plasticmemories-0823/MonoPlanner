import assert from "node:assert/strict";
import test from "node:test";
import { ThreadHistoryStore } from "../src/lib/thread-history.ts";
import { combineUsage } from "../src/lib/activity-usage.ts";

const thread = (id, fields = {}) => ({
  thread_id: id,
  created_at: `2026-10-11T12:00:0${id}Z`,
  updated_at: "2026-10-11T12:00:00Z",
  status: "idle",
  metadata: { graph_id: "monoplaner" },
  values: {},
  visibility: "active",
  ...fields,
});
const title = (revision, value = "整理项目目录") => ({
  title: value,
  status: "completed",
  revision,
});

test("created threads appear immediately and survive an older empty search", () => {
  const store = new ThreadHistoryStore();
  const search = store.startRefresh();
  store.upsert(thread("1"), "帮我整理一下项目目录");
  assert.equal(store.rows[0].thread_id, "1");
  assert.equal(store.rows[0].values.messages[0].content, "帮我整理一下项目目录");
  store.applyRefresh(search, []);
  assert.equal(store.rows.length, 1);
});

test("multiple creations merge by id and metadata-only search preserves preview", () => {
  const store = new ThreadHistoryStore();
  store.upsert(thread("1"), "第一条消息");
  store.upsert(thread("2"), "第二条消息");
  store.applyRefresh(store.startRefresh(), [thread("1"), thread("2")]);
  assert.deepEqual(
    store.rows.map((row) => row.thread_id),
    ["2", "1"],
  );
  assert.equal(store.rows[0].values.messages[0].content, "第二条消息");
  store.upsert(thread("2"));
  assert.equal(store.rows.length, 2);
  assert.equal(store.rows[0].values.messages[0].content, "第二条消息");
});

test("out-of-order searches cannot overwrite the latest list or generated title", () => {
  const store = new ThreadHistoryStore();
  const first = store.startRefresh();
  const second = store.startRefresh();
  store.applyRefresh(second, [thread("2", { title: title(2) })]);
  store.applyRefresh(first, [thread("1")]);
  assert.equal(store.rows[0].thread_id, "2");
  store.applyTitle("2", title(3, "项目结构整理"));
  store.applyRefresh(store.startRefresh(), [thread("2", { title: title(1) })]);
  assert.equal(store.rows[0].title.title, "项目结构整理");
});

test("archive and trash mutations win over an in-flight snapshot", () => {
  const store = new ThreadHistoryStore();
  store.upsert(thread("1"));
  const old = store.startRefresh();
  store.setVisibility("1", "archived");
  store.applyRefresh(old, [thread("1")]);
  assert.equal(store.rows[0].visibility, "archived");
  store.setVisibility("1", "deleted");
  store.applyTitle("1", title(1));
  store.applyRefresh(store.startRefresh(), [thread("1")]);
  assert.equal(store.rows[0].visibility, "deleted");
  assert.equal(store.rows[0].title.title, "整理项目目录");
});

test("server acknowledgement releases local visibility override for later changes", () => {
  const store = new ThreadHistoryStore();
  store.upsert(thread("1"));
  store.setVisibility("1", "archived");
  store.applyRefresh(store.startRefresh(), [thread("1", { visibility: "archived" })]);
  store.applyRefresh(store.startRefresh(), [thread("1")]);
  assert.equal(store.rows[0].visibility, "active");
});

test("purged thread cannot return from stale search, activity or an upsert", () => {
  const store = new ThreadHistoryStore();
  store.upsert(thread("1"));
  store.purge("1");
  store.applyRefresh(store.startRefresh(), [thread("1")]);
  store.applyTitle("1", title(10));
  store.upsert(thread("1"));
  assert.equal(store.rows.length, 0);
});

test("repeated activity titles do not produce continuous React updates", () => {
  const store = new ThreadHistoryStore();
  store.upsert(thread("1"));
  store.applyTitle("1", title(2));
  const rows = store.rows;
  assert.strictEqual(store.applyTitle("1", title(2)), rows);
  assert.strictEqual(store.applyTitle("1", title(1)), rows);
});

test("title token costs accumulate without resetting the main request context", () => {
  const context = {
    used_tokens: 1200,
    window_tokens: 1000000,
    estimated: false,
  };
  const result = combineUsage([
    {
      input_tokens: 1000,
      cached_input_tokens: 900,
      cache_known_input_tokens: 1000,
      calls: 1,
      context,
    },
    {
      purpose: "title",
      input_tokens: 100,
      cached_input_tokens: 0,
      cache_known_input_tokens: 100,
      calls: 1,
      context: null,
    },
  ]);
  assert.equal(result.calls, 2);
  assert.equal(result.input_tokens, 1100);
  assert.equal(result.cache_rate, 900 / 1100);
  assert.deepEqual(result.context, context);
});
