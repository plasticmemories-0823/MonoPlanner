export type ActivityUsage = {
  total_tokens: number;
  input_tokens: number;
  cached_input_tokens: number;
  cache_known_input_tokens: number;
  cache_unknown_calls: number;
  uncached_input_tokens: number;
  calls: number;
  unknown_calls: number;
  model_seconds: number;
  service_calls?: number;
  service_cost_micros?: number;
  service_reserved_cost_micros?: number;
  service_unknown_calls?: number;
  purpose?: string;
  context?: {
    used_tokens: number;
    window_tokens: number;
    estimated: boolean;
  } | null;
};

// 缓存比例按输入 Token 加权；上下文占用取自
// 最新一次任务请求，不累加整段对话的所有调用。
export function combineUsage(rows: ActivityUsage[]) {
  const totals: ActivityUsage = {
    total_tokens: 0,
    input_tokens: 0,
    cached_input_tokens: 0,
    cache_known_input_tokens: 0,
    cache_unknown_calls: 0,
    uncached_input_tokens: 0,
    calls: 0,
    unknown_calls: 0,
    model_seconds: 0,
    service_calls: 0,
    service_cost_micros: 0,
    service_reserved_cost_micros: 0,
    service_unknown_calls: 0,
    context: null,
  };
  for (const value of rows) {
    for (const key of [
      "total_tokens",
      "input_tokens",
      "cached_input_tokens",
      "cache_known_input_tokens",
      "cache_unknown_calls",
      "uncached_input_tokens",
      "calls",
      "unknown_calls",
      "model_seconds",
      "service_calls",
      "service_cost_micros",
      "service_reserved_cost_micros",
      "service_unknown_calls",
    ] as const) {
      totals[key] = (totals[key] || 0) + (value[key] || 0);
    }
    if (value.purpose !== "title") totals.context = value.context || null;
  }
  return {
    ...totals,
    cache_rate: totals.cache_known_input_tokens
      ? totals.cached_input_tokens / totals.cache_known_input_tokens
      : null,
  };
}

/** 所有分支计入总用量，只有当前主分支决定上下文占用。 */
export function conversationUsage(
  groups: Partial<Record<"chat" | "work" | "progress" | "title", ActivityUsage>>,
  workActive: boolean,
) {
  const context_source = workActive ? "work" : "chat";
  return {
    ...combineUsage(Object.entries(groups).map(([purpose, value]) => ({ ...value, purpose }))),
    context: groups[context_source]?.context || null,
    context_source,
  };
}
