"use client";

import { memo, useEffect, useMemo, useRef, useState } from "react";
import { conversationUsage } from "@/lib/activity-usage";
import { useStreamContext } from "@/providers/Stream";
import type { Activity } from "@/lib/conversation-types";
import { requestJson } from "@/lib/conversation-api";
import {
  WorkActivityMonitor,
  visibleWorkActivity,
  type DiagnosticPage,
} from "@/lib/work-diagnostics";
import { Brain, Check, CircleDot, CircleHelp, LoaderCircle, ChevronDown } from "lucide-react";

export type { Activity } from "@/lib/conversation-types";

// 聊天取公开快照，工作只读本地原件的增量游标；两者只在显示层组合。
export function useActivity() {
  const stream = useStreamContext();
  const { apiUrl } = stream;
  const threadId = stream.values.thread_id;
  const work = stream.values.work;
  const enabled = Boolean(threadId && work && !stream.values.read_only && !stream.storageError);
  const monitor = useRef<WorkActivityMonitor | null>(null);
  const [workView, setWorkView] = useState({ threadId: "", items: [] as Activity[], error: "" });
  useEffect(() => {
    if (!enabled) return;
    const reader = new WorkActivityMonitor(
      (after, signal) => {
        const query = new URLSearchParams({ after: JSON.stringify(after) });
        return requestJson<DiagnosticPage>(
          `${apiUrl}/conversations/${threadId}/diagnostics?${query}`,
          { signal },
        );
      },
      (items, error) => setWorkView({ threadId, items, error }),
    );
    monitor.current = reader;
    return () => {
      reader.dispose();
      if (monitor.current === reader) monitor.current = null;
    };
  }, [apiUrl, threadId, enabled]);
  useEffect(() => {
    if (enabled && work)
      monitor.current?.update(
        `${work.task_id}:${work.status}:${work.revision || 0}`,
        stream.workRunning,
      );
  }, [enabled, work, stream.workRunning]);
  const usage = useMemo(
    () => conversationUsage(stream.values.usage || {}, stream.workRunning),
    [stream.values.usage, stream.workRunning],
  );
  const workRows = workView.threadId === threadId ? workView.items : [];
  const items: Activity[] = [
    ...(stream.values.activities || []).map((row) => ({ ...row, source: "chat" as const })),
    ...visibleWorkActivity(workRows, work).map((row) => ({ ...row, source: "work" as const })),
  ];
  return {
    items: items.map((item) =>
      stream.storageError && item.status === "running"
        ? { ...item, status: "unknown", label: "记录写入失败，状态待核对" }
        : item,
    ),
    error: [stream.error?.message, workView.threadId === threadId && workView.error]
      .filter(Boolean)
      .join("；"),
    usage,
    groups: stream.values.usage || {},
    reconnect: () => {
      stream.reconnect();
      monitor.current?.reconnect();
    },
  };
}

const ActivityRow = memo(function ActivityRow({ row }: { row: Activity }) {
  const [open, setOpen] = useState(false);
  const [now, setNow] = useState(() => Date.now() / 1000);
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
          <Icon className={`mt-0.5 size-4 shrink-0 ${busy ? "animate-spin" : ""}`} />
          <span className="min-w-0">
            <span>
              {title}
              {row.kind === "thinking" && !busy && !row.detail ? "（未返回思考内容）" : ""}
            </span>
            {(row.seconds !== undefined || row.started_at) && (
              <span className="ml-2 text-xs">
                {Math.max(0, row.seconds ?? now - (row.started_at || now)).toFixed(1)} 秒
              </span>
            )}
            {row.kind === "operation" && row.detail && !row.stage && (
              <code className="ml-2 text-xs break-all">{row.detail.split("\n")[0]}</code>
            )}
          </span>
          {row.detail && (
            <span className="ml-auto shrink-0 text-xs">{row.stage ? "查看诊断" : "展开"}</span>
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

export function ActivityRows({
  items,
  title = "思考与操作",
}: {
  items: Activity[];
  title?: string;
}) {
  const [open, setOpen] = useState(false);
  if (!items.length) return null;
  const running = items.findLast((row) => row.status === "running");
  const needsAttention = items.some((row) => ["failed", "error", "unknown"].includes(row.status));
  const interrupted = items.some((row) => row.status === "interrupted");
  const currentState = items.findLast((row) => row.id.endsWith("-state") && row.source === "work");
  const label = running
    ? running.label
    : currentState
      ? currentState.label
      : needsAttention
        ? "步骤已结束，有结果需要核对"
        : interrupted
          ? "当前回复已中断"
          : "思考与工作步骤已完成";
  return (
    <details
      data-testid="activity-group"
      data-source={items[0]?.source}
      className="text-muted-foreground border-border/50 w-full rounded-lg border text-sm"
      onToggle={(event) => {
        if (event.target === event.currentTarget) setOpen(event.currentTarget.open);
      }}
    >
      <summary className="flex cursor-pointer list-none items-center gap-2 px-3 py-2">
        {running ? (
          <LoaderCircle className="size-4 animate-spin" />
        ) : needsAttention ? (
          <CircleHelp className="size-4" />
        ) : (
          <Check className="size-4" />
        )}
        <span>
          {title} · {label}
        </span>
        <span className="ml-auto text-xs">{items.length} 项</span>
        <ChevronDown className={`size-4 transition-transform ${open ? "rotate-180" : ""}`} />
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
