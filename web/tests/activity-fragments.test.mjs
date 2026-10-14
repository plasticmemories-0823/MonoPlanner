import assert from "node:assert/strict";
import test from "node:test";
import { mergeActivity } from "../src/lib/activity-fragments.ts";

test("journal fragments append once and the final response replaces partial display", () => {
  let row = { id: "r1", status: "running", detail: "", version: 1 };
  row = mergeActivity(row, {
    ...row,
    detail: "第一段",
    append: true,
    version: 2,
  });
  assert.equal(row.detail, "第一段");
  assert.strictEqual(
    mergeActivity(row, { ...row, detail: "第一段", append: true }),
    row,
  );
  row = mergeActivity(row, {
    ...row,
    detail: "第二段",
    append: true,
    version: 3,
  });
  assert.equal(row.detail, "第一段第二段");
  row = mergeActivity(row, {
    ...row,
    status: "completed",
    detail: "完整原文",
    append: false,
    version: 4,
  });
  assert.equal(row.detail, "完整原文");
  assert.strictEqual(
    mergeActivity(row, {
      ...row,
      status: "running",
      detail: "旧片段",
      version: 3,
    }),
    row,
  );
});

test("recovery and repeated polls do not regress a completed display", () => {
  const row = { id: "r1", status: "completed", detail: "原文", version: 4 };
  for (let i = 0; i < 100; i++) {
    assert.strictEqual(mergeActivity(row, structuredClone(row)), row);
    assert.strictEqual(
      mergeActivity(row, { ...row, status: "running", version: 5 }),
      row,
    );
  }
  const full = mergeActivity(undefined, { ...row, append: false });
  assert.equal(full.detail, "原文");
});
