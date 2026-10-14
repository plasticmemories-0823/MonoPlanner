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
    ] as const) {
      totals[key] += value[key] || 0;
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
