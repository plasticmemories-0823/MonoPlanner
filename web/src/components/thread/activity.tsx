"use client";

import { memo, useEffect, useRef, useState } from "react";
import { combineUsage, type ActivityUsage } from "@/lib/activity-usage";
import {
  Brain,
  Check,
  CircleDot,
  CircleHelp,
  LoaderCircle,
  ChevronDown,
} from "lucide-react";

export type Activity = {
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

type Usage = ReturnType<typeof combineUsage>;

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
    usage: {} as Record<string, ActivityUsage>,
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
        let changed = false;
        for (const row of [
          ...result.items,
          ...(result.live || []),
        ] as Activity[]) {
          const previous = archive.current.rows.get(row.id);
          if (previous?.status === "completed" && row.status === "running")
            continue;
          if (JSON.stringify(previous) !== JSON.stringify(row)) {
            archive.current.rows.set(row.id, row);
            changed = true;
          }
        }
        Object.assign(archive.current.usage, result.usage || {});
        const totals = combineUsage(Object.values(archive.current.usage));
        setUsage((previous) =>
          JSON.stringify(previous) === JSON.stringify(totals)
            ? previous
            : totals,
        );
        if (changed) setItems([...archive.current.rows.values()]);
        setError("");
        if (result.more || running)
          timer = setTimeout(poll, result.more ? 20 : 400);
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

const ActivityRow = memo(function ActivityRow({ row }: { row: Activity }) {
  const [open, setOpen] = useState(false);
  const [now, setNow] = useState(Date.now() / 1000);
  const hasRunning = row.status === "running";
  useEffect(() => {
    if (!hasRunning) return;
    const timer = setInterval(() => setNow(Date.now() / 1000), 1000);
    return () => clearInterval(timer);
  }, [hasRunning]);
  return (() => {
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
        onToggle={(event) => setOpen(event.currentTarget.open)}
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
        {open && row.detail && (
          <pre className="mt-2 ml-6 max-h-80 max-w-full overflow-auto border-l pl-3 font-mono text-xs leading-6 break-words whitespace-pre-wrap">
            {row.detail}
          </pre>
        )}
        {open && row.child_usage && (
          <p className="ml-6 text-xs">
            退出码：{row.exit_code ?? "未知"} · 被运行任务 Token：
            {row.child_usage.status === "reported_by_task"
              ? `${row.child_usage.total_tokens?.toLocaleString()}（任务自行报告）`
              : "未知"}
          </p>
        )}
        {open && busy && row.kind === "thinking" && (
          <p className="ml-6 text-xs">思考内容正在实时接收。</p>
        )}
      </details>
    );
  })();
});

export function ActivityRows({ items }: { items: Activity[] }) {
  const [open, setOpen] = useState(false);
  if (!items.length) return null;
  const running = items.findLast((row) => row.status === "running");
  const needsAttention = items.some((row) =>
    ["failed", "error", "unknown"].includes(row.status),
  );
  const interrupted = items.some((row) => row.status === "interrupted");
  const label = running
    ? running.label
    : needsAttention
      ? "步骤已结束，有结果需要核对"
      : interrupted
        ? "当前回复已中断"
        : "思考与工作步骤已完成";
  return (
    <details
      data-testid="activity-group"
      className="text-muted-foreground border-border/50 w-full rounded-lg border text-sm"
      onToggle={(event) => {
        if (event.target === event.currentTarget)
          setOpen(event.currentTarget.open);
      }}
    >
      <summary className="flex cursor-pointer list-none items-center gap-2 px-3 py-2">
        {running ? (
          <LoaderCircle className="size-4 animate-spin" />
        ) : (
          <Check className="size-4" />
        )}
        <span>{label}</span>
        <span className="ml-auto text-xs">{items.length} 项</span>
        <ChevronDown
          className={`size-4 transition-transform ${open ? "rotate-180" : ""}`}
        />
      </summary>
      {open && (
        <div className="space-y-2 border-t px-3 py-3">
          {items.map((row) => (
            <ActivityRow
              key={row.id}
              row={row}
            />
          ))}
        </div>
      )}
    </details>
  );
}
