import assert from "node:assert/strict";
import test from "node:test";
import { mergePublicMessages } from "../src/lib/public-messages.ts";

const message = (id, content, type = "ai") => ({ id, content, type });

test("older recovery and empty stream do not swallow published commentary", () => {
  const published = [message("user", "直接检查", "human"), message("progress", "我找到了入口，接下来检查调用方。")];
  assert.strictEqual(mergePublicMessages(published, []), published);
  assert.strictEqual(mergePublicMessages(published, [published[0]]), published);
  assert.strictEqual(mergePublicMessages(published, [message("progress", "")]), published);
  assert.strictEqual(mergePublicMessages(published, [message("progress", "我找到了入口")]), published);
});

test("new replies append once and growing text keeps its original position", () => {
  const first = [message("progress", "正在检查")];
  const updated = mergePublicMessages(first, [message("progress", "正在检查调用方"), message("final", "检查完成")]);
  assert.deepEqual(updated.map((item) => item.id), ["progress", "final"]);
  assert.equal(updated[0].content, "正在检查调用方");
  assert.strictEqual(mergePublicMessages(updated, structuredClone(updated)), updated);
  assert.equal(first[0].content, "正在检查");
});
