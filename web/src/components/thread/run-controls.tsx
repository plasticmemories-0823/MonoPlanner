import { useEffect, useRef, useState } from "react";
import { Message } from "@langchain/langgraph-sdk";
import { v4 as uuid } from "uuid";
import { toast } from "sonner";
import { useStreamContext } from "@/providers/Stream";

const options = { streamMode: ["values", "custom"] as ("values" | "custom")[],
  streamResumable: true, config: { recursion_limit: 10000 } };

export function statusLabel(status?: string) {
  return ({ chat_complete: "回复完成", verified_success: "实际验收通过", paused: "已暂停",
    awaiting_user: "等待补充或继续", task_blocked: "任务受阻", runtime_error: "结果不明，已停止",
    needs_plan: "正在制定计划", executing: "正在执行" } as Record<string, string>)[status || ""] || status || "就绪";
}

export function useRunControls(stream: ReturnType<typeof useStreamContext>, threadId: string | null) {
  const [queued, setQueued] = useState<Message[]>([]);
  const attempted = useRef("");
  const pendingMessages = queued.filter((m) => !stream.messages.some((v) => v.id === m.id));

  useEffect(() => {
    setQueued([]);
    attempted.current = "";
    if (!threadId) return;
    fetch(`${stream.apiUrl}/threads/${threadId}/control`).then((r) => r.json()).then((r) =>
      setQueued((previous) => [...previous, ...(r.pending_messages || [])
        .filter((m: Message) => !previous.some((v) => v.id === m.id))])).catch(console.error);
  }, [threadId, stream.apiUrl]);

  useEffect(() => {
    if (stream.isLoading || stream.error || !pendingMessages.length) return;
    if (["runtime_error", "state_conflict"].includes(stream.values.status || "")) return;
    const first = pendingMessages[0];
    if (attempted.current === first.id) return;
    attempted.current = first.id!;
    // A message can arrive after the final graph node. Drain its saved ID only
    // after that run closes; the backend deduplicates already applied input.
    stream.submit({ messages: [first], repo: stream.values.repo }, options);
  }, [stream.isLoading, stream.error, pendingMessages, stream, stream.values.status]);

  async function control(kind: string, text?: string, id = uuid()) {
    try {
      const r = await fetch(`${stream.apiUrl}/threads/${threadId}/control`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id, kind, text }),
      });
      if (!r.ok) throw new Error("请求未保存，请保留内容重试");
      return true;
    } catch (e) { toast.error(String(e)); return false; }
  }

  async function queueMessage(message: Message) {
    if (!await control("message", message.content as string, message.id)) return false;
    setQueued((old) => [...old, message]);
    toast.success("已收到，将在当前请求或操作结束后接入");
    return true;
  }
  async function pause() {
    if (await control("pause")) toast.success("暂停已收到，将在安全位置停止");
  }
  async function resume() {
    if (await control("resume")) stream.submit({ messages: [] }, options);
  }
  return { pendingMessages, queueMessage, pause, resume };
}
