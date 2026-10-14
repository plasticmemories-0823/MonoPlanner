type Row = {
  id: string;
  status: string;
  detail: string;
  append?: boolean;
  version?: number;
};

/** 通过日志游标和片段版本避免重复轮询造成重复更新。 */
export function mergeActivity<T extends Row>(
  previous: T | undefined,
  next: T,
): T {
  if (previous?.status === "completed" && next.status !== "completed")
    return previous;
  if (
    previous?.version !== undefined &&
    next.version !== undefined &&
    next.version <= previous.version
  )
    return previous;
  const row = {
    ...previous,
    ...next,
    detail: next.append ? (previous?.detail || "") + next.detail : next.detail,
  };
  return JSON.stringify(previous) === JSON.stringify(row) ? previous! : row;
}
