import { useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import { useStreamContext } from "@/providers/Stream";
import { commandNotice } from "@/lib/conversation-types";

export function statusLabel(status?: string) {
  return (
    (
      {
        idle: "就绪",
        running: "正在处理",
        queued: "已排队",
        starting: "正在检查开工条件",
        preparing: "正在准备项目环境",
        pausing: "正在停止当前操作",
        stopped: "已停止",
        chat_complete: "回复完成",
        verified_success: "实际验收通过",
        existing_complete: "已有任务已结束",
        preparation_failed: "开工准备失败",
        control_unconfirmed: "已请求终止，结果待核对",
        paused: "已暂停",
        awaiting_user: "等待补充或继续",
        task_blocked: "任务受阻",
        runtime_error: "任务已停止，运行记录已保留",
        output_limit_reached: "回复达到输出上限，等待手动继续",
        model_request_unknown: "模型请求结果未确认，已保留记录",
        web_request_unknown: "联网请求结果未确认，已保留记录",
        execution_unknown: "操作结果未确认，已保留记录",
        compaction_commit_unknown: "上下文整理结果待核对，已保留记录",
        state_conflict: "运行条件发生变化，需要核对后继续",
        task_cancelled: "任务已取消",
        budget_exhausted: "已达到本次运行边界，记录已保留",
        stalled: "尚未取得新的进展，记录已保留",
        needs_plan: "正在制定计划",
        executing: "正在执行",
      } as Record<string, string>
    )[status || ""] || (status ? "运行状态已记录" : "就绪")
  );
}

export function useRunControls(
  stream: ReturnType<typeof useStreamContext>,
  threadId: string | null,
) {
  const [pauseRequested, setPauseRequested] = useState(false);
  const busy = useRef(false);
  const pending = useRef<{ kind: "pause" | "resume"; id: string } | null>(null);
  useEffect(() => {
    setPauseRequested(false);
    pending.current = null;
  }, [threadId]);
  useEffect(() => {
    if (!stream.isLoading) setPauseRequested(false);
  }, [stream.isLoading]);

  async function control(kind: "pause" | "resume") {
    if (!threadId || busy.current) return;
    busy.current = true;
    if (kind === "pause") setPauseRequested(true);
    if (pending.current?.kind !== kind) pending.current = { kind, id: crypto.randomUUID() };
    try {
      const receipt = await stream.command(pending.current, threadId);
      const notice = commandNotice(kind, receipt);
      if (notice.warning) toast.warning(notice.message);
      else {
        pending.current = null;
        toast.success(notice.message);
      }
    } catch (error) {
      // 结果不明时保留同一命令编号，显式重试不会重复执行。
      toast.error(error instanceof Error ? error.message : "请求结果未确认，请重新连接后核对。");
    } finally {
      busy.current = false;
      setPauseRequested(false);
    }
  }
  return {
    pause: () => control("pause"),
    resume: () => control("resume"),
    pauseRequested:
      pauseRequested || (!stream.storageError && stream.values.work?.status === "pausing"),
  };
}
