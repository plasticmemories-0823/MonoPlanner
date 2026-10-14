import assert from "node:assert/strict";
import test from "node:test";
import { appendDiagnosticPage } from "../src/lib/work-diagnostics.ts";

test("manual diagnostic pagination appends a fragment once and replaces it with the saved response", () => {
  const first = { id: "call-1", status: "running", detail: "第一段", version: 1 };
  const part = { ...first, detail: "第二段", version: 2, append: true };
  const combined = appendDiagnosticPage([first], [part, part]);
  assert.equal(combined[0].detail, "第一段第二段");
  assert.equal(combined.length, 1);
  const complete = { ...first, status: "completed", detail: "完整原文", version: 3, append: false };
  assert.equal(appendDiagnosticPage(combined, [complete])[0].detail, "完整原文");
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
