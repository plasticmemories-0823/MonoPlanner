import assert from "node:assert/strict";
import test from "node:test";
import { canSupplementWork } from "../src/lib/conversation-types.ts";

test("supplements are available only while the work can consume them", () => {
  for (const status of ["queued", "running", "needs_plan", "executing", "paused", "awaiting_user"])
    assert.equal(canSupplementWork(status), true, status);
});

test("finished, uncertain and explicitly resumable states cannot queue supplements", () => {
  for (const status of [
    undefined,
    "idle",
    "verified_success",
    "task_blocked",
    "task_cancelled",
    "chat_complete",
    "existing_complete",
    "stalled",
    "budget_exhausted",
    "model_request_unknown",
    "execution_unknown",
    "state_conflict",
    "pausing",
    "output_limit_reached",
    "unrecognized_status",
  ])
    assert.equal(canSupplementWork(status), false, String(status));
});
