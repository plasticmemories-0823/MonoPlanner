import { useEffect, useRef, useState } from "react";
import { Message } from "@langchain/langgraph-sdk";
import { v4 as uuid } from "uuid";
import { toast } from "sonner";
import { useStreamContext } from "@/providers/Stream";

const options = {
  streamMode: ["values", "custom"] as ("values" | "custom")[],
  streamResumable: true,
  config: { recursion_limit: 10000 },
};

export function statusLabel(status?: string) {
  return (
    (
      {
        chat_complete: "回复完成",
        verified_success: "实际验收通过",
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
  const [queued, setQueued] = useState<Message[]>([]);
  const [pauseRequested, setPauseRequested] = useState(false);
  const attempted = useRef("");
  const resumeRequest = useRef<string | null>(null);
  const resuming = useRef(false);
  const pendingMessages = queued.filter(
    (m) => !(stream.values.messages || []).some((v) => v.id === m.id),
  );

  useEffect(() => {
    if (!stream.isLoading) setPauseRequested(false);
    if (stream.isLoading) resumeRequest.current = null;
  }, [stream.isLoading]);

  useEffect(() => {
    setQueued([]);
    attempted.current = "";
    resumeRequest.current = null;
    resuming.current = false;
    if (!threadId) return;
    let cancelled = false;
    fetch(`${stream.apiUrl}/threads/${threadId}/control`)
      .then((r) => r.json())
      .then((r) => {
        if (cancelled) return;
        setQueued((previous) => [
          ...previous,
          ...(r.pending_messages || []).filter(
            (m: Message) => !previous.some((v) => v.id === m.id),
          ),
        ]);
      })
      .catch(console.error);
    return () => {
      cancelled = true;
    };
  }, [threadId, stream.apiUrl]);

  useEffect(() => {
    if (stream.isLoading || stream.error || !pendingMessages.length) return;
    if (
      ["runtime_error", "state_conflict", "paused"].includes(
        stream.values.status || "",
      )
    )
      return;
    const first = pendingMessages[0];
    if (attempted.current === first.id) return;
    attempted.current = first.id!;
    // 消息可能在图的最后一个节点结束后到达；仅在该次运行关闭后
    // 处理已保存的消息 ID，后端会对已经应用的输入去重。
    stream.submit({ messages: [first], repo: stream.values.repo }, options);
  }, [
    stream.isLoading,
    stream.error,
    pendingMessages,
    stream,
    stream.values.status,
  ]);

  async function control(kind: string, text?: string, id = uuid()) {
    try {
      const r = await fetch(`${stream.apiUrl}/threads/${threadId}/control`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id, kind, text }),
      });
      if (!r.ok) throw new Error("请求未保存，请保留内容重试");
      return true;
    } catch (e) {
      toast.error(String(e));
      return false;
    }
  }

  async function queueMessage(message: Message) {
    if (!(await control("message", message.content as string, message.id)))
      return false;
    stream.retainMessage(message);
    setQueued((old) => [...old, message]);
    toast.success("已收到，正在中断当前回复并调整后续工作");
    return true;
  }
  async function pause() {
    if (await control("pause")) {
      setPauseRequested(true);
      toast.success("正在暂停；当前模型回复将被中断");
    }
  }
  async function resume() {
    if (resuming.current) return;
    resuming.current = true;
    resumeRequest.current ||= uuid();
    try {
      if (await control("resume", undefined, resumeRequest.current))
        await stream.submit({ messages: [] }, options);
    } finally {
      resuming.current = false;
    }
  }
  return { pendingMessages, queueMessage, pause, resume, pauseRequested };
}
