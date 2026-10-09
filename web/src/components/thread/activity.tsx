"use client";

import { useEffect, useRef, useState } from "react";
import {
  Brain,
  Check,
  CircleDot,
  CircleHelp,
  LoaderCircle,
} from "lucide-react";

type Activity = {
  id: string;
  after_id: string;
  kind: "thinking" | "operation";
  label: string;
  detail: string;
  status: string;
  stage?: "trial" | "acceptance";
  summary?: string;
  started_at?: number;
  seconds?: number;
  exit_code?: number;
  child_usage?: { status: string; total_tokens?: number; source?: string };
};

type Usage = {
  total_tokens: number;
  uncached_input_tokens: number;
  calls: number;
  unknown_calls: number;
  model_seconds: number;
};

// Display records never become chat messages or model input.
export function useActivity(
  apiUrl: string,
  threadId: string | null,
  running: boolean,
) {
  const [items, setItems] = useState<Activity[]>([]);
  const [error, setError] = useState("");
  const [usage, setUsage] = useState<Usage | null>(null);
  const archive = useRef({
    thread: threadId,
    api: apiUrl,
    cursors: {},
    rows: new Map<string, Activity>(),
    usage: {} as Record<string, Usage>,
  });
  useEffect(() => {
    if (archive.current.thread !== threadId || archive.current.api !== apiUrl) {
      archive.current = {
        thread: threadId,
        api: apiUrl,
        cursors: {},
        rows: new Map(),
        usage: {},
      };
      setItems([]);
      setError("");
      setUsage(null);
    }
    if (!threadId) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;
    const controller = new AbortController();
    const poll = async () => {
      try {
        const query = new URLSearchParams({
          after: JSON.stringify(archive.current.cursors),
        });
        const response = await fetch(
          `${apiUrl}/threads/${threadId}/activity?${query}`,
          { signal: controller.signal },
        );
        const result = await response.json();
        if (!response.ok)
          throw new Error(result.error || "无法读取本地活动记录");
        if (cancelled) return;
        archive.current.cursors = result.cursors;
        for (const row of result.items as Activity[])
          archive.current.rows.set(row.id, row);
        Object.assign(archive.current.usage, result.usage || {});
        const totals = Object.values(archive.current.usage).reduce(
          (sum, value) => ({
            total_tokens: sum.total_tokens + value.total_tokens,
            uncached_input_tokens:
              sum.uncached_input_tokens + value.uncached_input_tokens,
            calls: sum.calls + value.calls,
            unknown_calls: sum.unknown_calls + value.unknown_calls,
            model_seconds: sum.model_seconds + value.model_seconds,
          }),
          {
            total_tokens: 0,
            uncached_input_tokens: 0,
            calls: 0,
            unknown_calls: 0,
            model_seconds: 0,
          },
        );
        setUsage(totals);
        setItems([...archive.current.rows.values()]);
        setError("");
        if (result.more || running)
          timer = setTimeout(poll, result.more ? 20 : 800);
      } catch (e) {
        if (!cancelled)
          setError(e instanceof Error ? e.message : "无法读取本地活动记录");
      }
    };
    poll();
    return () => {
      cancelled = true;
      controller.abort();
      clearTimeout(timer);
    };
  }, [apiUrl, threadId, running]);
  return { items, error, usage };
}

export function ActivityRows({ items }: { items: Activity[] }) {
  const [now, setNow] = useState(Date.now() / 1000);
  const hasRunning = items.some((row) => row.status === "running");
  useEffect(() => {
    if (!hasRunning) return;
    const timer = setInterval(() => setNow(Date.now() / 1000), 1000);
    return () => clearInterval(timer);
  }, [hasRunning]);
  return (
    <>
      {items.map((row) => {
        const busy = row.status === "running";
        const needsAttention = ["failed", "error"].includes(row.status);
        const Icon = busy
          ? LoaderCircle
          : row.status === "unknown"
            ? CircleHelp
            : needsAttention
              ? CircleDot
              : row.kind === "thinking"
                ? Brain
                : Check;
        const title =
          row.kind === "thinking"
            ? row.label
            : row.summary ||
              `${busy ? "" : row.status === "unknown" ? "结果待核对 · " : needsAttention ? "需要处理 · " : "已完成 · "}${row.label.replace(/^正在/, "")}`;
        return (
          <details
            key={row.id}
            className="text-muted-foreground mr-auto w-full text-sm"
            data-activity={row.kind}
          >
            <summary className="hover:bg-muted/60 flex cursor-pointer list-none items-start gap-2 rounded-lg px-1 py-1">
              <Icon
                className={`mt-0.5 size-4 shrink-0 ${busy ? "animate-spin" : ""}`}
              />
              <span className="min-w-0">
                <span>
                  {title}
                  {row.kind === "thinking" && !busy && !row.detail
                    ? "（未返回思考内容）"
                    : ""}
                </span>
                {(row.seconds !== undefined || row.started_at) && (
                  <span className="ml-2 text-xs">
                    {Math.max(
                      0,
                      row.seconds ?? now - (row.started_at || now),
                    ).toFixed(1)}{" "}
                    秒
                  </span>
                )}
                {row.kind === "operation" && row.detail && !row.stage && (
                  <code className="ml-2 text-xs break-all">
                    {row.detail.split("\n")[0]}
                  </code>
                )}
              </span>
              {row.detail && (
                <span className="ml-auto shrink-0 text-xs">
                  {row.stage ? "查看诊断" : "展开"}
                </span>
              )}
            </summary>
            {row.detail && (
              <pre className="mt-2 ml-6 max-w-full border-l pl-3 font-mono text-xs leading-6 break-words whitespace-pre-wrap">
                {row.detail}
              </pre>
            )}
            {row.child_usage && (
              <p className="ml-6 text-xs">
                退出码：{row.exit_code ?? "未知"} · 被运行任务 Token：
                {row.child_usage.status === "reported_by_task"
                  ? `${row.child_usage.total_tokens?.toLocaleString()}（任务自行报告）`
                  : "未知"}
              </p>
            )}
            {busy && row.kind === "thinking" && (
              <p className="ml-6 text-xs">响应保存后显示思考全文。</p>
            )}
          </details>
        );
      })}
    </>
  );
}
