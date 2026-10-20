import assert from "node:assert/strict";
import test from "node:test";
import { canSupplementWork, commandNotice, isWorking } from "../src/lib/conversation-types.ts";

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
    "preparation_failed",
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

test("failed preparation is terminal for activity display and an unrecorded pause is never reported saved", () => {
  assert.equal(isWorking("preparation_failed"), false);
  const notice = commandNotice("pause", {
    id: "pause-a",
    received: false,
    recorded: false,
    control_requested: true,
    termination_confirmed: false,
    applied: false,
    status: "control_unconfirmed",
    event_cursor: 0,
  });
  assert.equal(notice.warning, true);
  assert.match(notice.message, /结果待核对/);
  assert.doesNotMatch(notice.message, /已保存|已停止/);
});

await test("acceptance budget pause exposes Continue but never supplement or running state", async () => {
  const { acceptanceLimitLabel, canResumeWork, resumeCommand } = await import(
    "../src/lib/conversation-types.ts"
  );
  const limit = {
    active: true,
    completed_rounds: 3,
    round_limit: 3,
    round_increment: 3,
    pause_id: "pause-3",
    last_verification_status: "failed",
  };
  assert.equal(canResumeWork("acceptance_limit_reached"), true);
  assert.equal(canSupplementWork("acceptance_limit_reached"), false);
  assert.equal(isWorking("acceptance_limit_reached"), false);
  assert.match(acceptanceLimitLabel(limit), /3 \/ 3 轮验收/);
  assert.equal(acceptanceLimitLabel({ ...limit, active: false }), null);
  assert.deepEqual(resumeCommand({ task_id: "task", acceptance_limit: limit }, "resume-id"), {
    kind: "resume",
    id: "resume-id",
    acceptance_pause_id: "pause-3",
  });
  assert.deepEqual(resumeCommand({ task_id: "task", status: "paused" }, "ordinary"), {
    kind: "resume",
    id: "ordinary",
  });
  for (const status of ["stalled", "execution_unknown", "state_conflict", "verified_success"])
    assert.equal(canResumeWork(status), false, status);
});
