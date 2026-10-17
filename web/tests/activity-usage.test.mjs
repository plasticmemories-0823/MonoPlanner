import assert from "node:assert/strict";
import test from "node:test";
import { combineUsage, conversationUsage } from "../src/lib/activity-usage.ts";

test("cache rate is input-weighted and unknown records are not treated as misses", () => {
  const result = combineUsage([
    {
      input_tokens: 100,
      cached_input_tokens: 80,
      cache_known_input_tokens: 100,
      total_tokens: 120,
    },
    {
      input_tokens: 900,
      cached_input_tokens: 180,
      cache_known_input_tokens: 900,
      total_tokens: 940,
    },
    { unknown_calls: 1 },
  ]);
  assert.equal(result.cache_rate, 0.26);
  assert.equal(result.unknown_calls, 1);
  assert.equal(result.total_tokens, 1060);
  assert.equal(combineUsage([]).cache_rate, null);
});

test("progress and title costs accumulate without replacing the active main context", () => {
  const groups = {
    chat: { total_tokens: 120, calls: 1, context: { used_tokens: 300, window_tokens: 1000 } },
    work: { total_tokens: 1000, calls: 3, context: { used_tokens: 800, window_tokens: 1000 } },
    progress: { total_tokens: 30, calls: 1, context: null },
    title: { total_tokens: 20, calls: 1, context: null },
  };
  assert.equal(conversationUsage(groups, true).context.used_tokens, 800);
  assert.equal(conversationUsage(groups, false).context.used_tokens, 300);
  assert.equal(conversationUsage(groups, true).total_tokens, 1170);
  assert.equal(conversationUsage(groups, true).calls, 6);
  assert.equal(conversationUsage({ chat: groups.chat }, true).context, null);
});

test("web service fees and unknown reservations do not become model requests", () => {
  const result = conversationUsage({
    chat: { calls: 2, total_tokens: 50, service_calls: 1, service_cost_micros: 5000 },
    work: { calls: 3, total_tokens: 100, service_calls: 2, service_cost_micros: 5000,
      service_reserved_cost_micros: 5000, service_unknown_calls: 1 },
  }, false);
  assert.equal(result.calls, 5);
  assert.equal(result.total_tokens, 150);
  assert.equal(result.service_calls, 3);
  assert.equal(result.service_cost_micros, 10000);
  assert.equal(result.service_reserved_cost_micros, 5000);
  assert.equal(result.service_unknown_calls, 1);
  assert.equal(result.unknown_calls, 0);
});
