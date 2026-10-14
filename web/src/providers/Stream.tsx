import { localApiUrl } from "@/lib/local-service";
import React, {
  createContext,
  useContext,
  ReactNode,
  useState,
  useEffect,
  useRef,
} from "react";
import { useStream } from "@langchain/langgraph-sdk/react";
import { type Message } from "@langchain/langgraph-sdk";
import { useQueryState } from "nuqs";
import { mergePublicMessages, messagesForThread } from "@/lib/public-messages";
import { getThreadSearchMetadata, useThreads } from "./Thread";
import { createClient } from "./client";
import { toast } from "sonner";
import type { RuntimeSettings } from "@/components/thread/model-settings";

// 流状态与上下文约定
export type ProgressEvent = { type: "progress"; phase: string; text: string };
export type StateType = {
  messages: Message[];
  repo?: string;
  status?: string;
  runtime_settings?: RuntimeSettings;
  turn_status?: string;
  workspace_bound?: boolean;
  read_roots?: string[];
};

const useTypedStream = useStream<
  StateType,
  {
    UpdateType: {
      messages?: Message[] | Message | string;
      repo?: string;
      read_roots?: string[];
    };
    CustomEventType: ProgressEvent;
  }
>;

type StreamContextType = ReturnType<typeof useTypedStream> & {
  apiUrl: string;
  progress?: ProgressEvent;
  retainMessage: (message: Message, forThreadId?: string) => void;
  ensureThread: (firstMessage?: string) => Promise<string>;
  preparingThread: boolean;
};
const StreamContext = createContext<StreamContextType | undefined>(undefined);

// 本地服务状态检查
async function checkGraphStatus(apiUrl: string): Promise<boolean> {
  try {
    return (await fetch(`${apiUrl}/service`)).ok;
  } catch {
    return false;
  }
}

// 会话创建、流订阅与公开消息恢复
const StreamSession = ({
  children,
  apiUrl,
  assistantId,
}: {
  children: ReactNode;
  apiUrl: string;
  assistantId: string;
}) => {
  const [threadId, setThreadId] = useQueryState("threadId");
  const { getThreads, upsertThread, waitForTitle } = useThreads();
  const [progress, setProgress] = useState<ProgressEvent>();
  const [recovery, setRecovered] = useState<{
    key: string;
    value: StateType;
  }>();
  const key = `${apiUrl}\0${threadId || ""}`;
  const recovered = recovery?.key === key ? recovery.value : undefined;
  const activeKey = useRef(key);
  activeKey.current = key;
  const selectedThread = useRef(threadId);
  selectedThread.current = threadId;
  const setup = useRef<Promise<string> | null>(null);
  const threadCommit = useRef<{
    id: string;
    resolve: () => void;
    reject: (error: Error) => void;
  } | null>(null);
  const [preparingThread, setPreparingThread] = useState(false);
  useEffect(() => {
    const waiting = threadCommit.current;
    if (!waiting) return;
    if (threadId === waiting.id) {
      threadCommit.current = null;
      waiting.resolve();
    } else if (threadId) {
      threadCommit.current = null;
      waiting.reject(new Error("对话已切换，请在当前对话重新操作"));
    }
  }, [threadId]);
  useEffect(
    () => () => {
      threadCommit.current?.reject(new Error("对话已关闭，设置未保存"));
      threadCommit.current = null;
    },
    [],
  );
  const ensureThread = async (firstMessage?: string) => {
    if (setup.current) return setup.current;
    if (selectedThread.current) {
      if (firstMessage) waitForTitle(selectedThread.current, firstMessage);
      return selectedThread.current;
    }
    const expectedKey = activeKey.current;
    const pending = (async () => {
      const client = createClient(apiUrl);
      const created = await client.threads.create({
        metadata: getThreadSearchMetadata(assistantId),
      });
      upsertThread(created, firstMessage);
      if (firstMessage) waitForTitle(created.thread_id, firstMessage);
      if (activeKey.current !== expectedKey)
        throw new Error("对话已切换，请在当前对话重新操作");
      // URL 更新可能先于 React 提交完成；桌面控件必须
      // 先完成当前会话的绑定，再保存其权限。
      const committed = new Promise<void>((resolve, reject) => {
        threadCommit.current = { id: created.thread_id, resolve, reject };
      });
      try {
        await setThreadId(created.thread_id);
        await committed;
      } finally {
        if (threadCommit.current?.id === created.thread_id)
          threadCommit.current = null;
      }
      void getThreads();
      return created.thread_id;
    })();
    setup.current = pending;
    setPreparingThread(true);
    try {
      return await pending;
    } finally {
      if (setup.current === pending) {
        setup.current = null;
        setPreparingThread(false);
      }
    }
  };
  const recoveryVersion = useRef(0);
  const [archive, setArchive] = useState<{ key: string; messages: Message[] }>({
    key: "",
    messages: [],
  });
  const retainMessage = (message: Message, forThreadId?: string) => {
    const targetKey = forThreadId ? `${apiUrl}\0${forThreadId}` : key;
    setArchive((old) => ({
      key: targetKey,
      messages: mergePublicMessages(old.key === targetKey ? old.messages : [], [
        message,
      ]),
    }));
  };
  const streamValue = useTypedStream({
    apiUrl,
    assistantId,
    threadId: threadId ?? null,
    reconnectOnMount: true,
    // 合并密集 SDK 流引起的界面更新；在 SDK 存储中保留所有事件，
    // 并保持请求和执行时序不变。
    throttle: 50,
    fetchStateHistory: { limit: 1 },
    onFinish: async (_state, metadata) => {
      void getThreads();
      const finishedThread = metadata?.thread_id || threadId;
      if (!finishedThread) return;
      const expectedKey = `${apiUrl}\0${finishedThread}`;
      const version = ++recoveryVersion.current;
      try {
        const r = await fetch(`${apiUrl}/threads/${finishedThread}/control`, {
          cache: "no-store",
        });
        if (r.ok) {
          const value = await r.json();
          if (
            activeKey.current === expectedKey &&
            recoveryVersion.current === version
          )
            setRecovered({ key: expectedKey, value });
        }
      } catch (error) {
        console.error("Public message recovery failed", error);
      }
    },
    onCustomEvent: (event) => {
      if (activeKey.current !== key) return;
      if ("type" in event && event.type === "progress") {
        setProgress(event);
        return;
      }
    },
    onThreadId: (id) => {
      if (!threadId)
        setArchive((old) =>
          old.key === key ? { ...old, key: `${apiUrl}\0${id}` } : old,
        );
      setThreadId(id);
      void getThreads();
    },
  });
  const [settledStreamKey, setSettledStreamKey] = useState(key);
  useEffect(() => {
    // 在 SDK 重置流并开始加载历史记录后执行。
    // 在此之前，公开消息仍属于此前选中的会话。
    setSettledStreamKey(key);
  }, [key]);
  const latestStream = useRef(streamValue);
  latestStream.current = streamValue;
  // ensureThread 等待 URL 更新和 React 提交；调用方仍持有
  // 此前的渲染结果，因此需明确通过最新的 SDK 闭包提交。
  const submit: typeof streamValue.submit = (values, options) => {
    if (options?.threadId && options.threadId !== selectedThread.current)
      return Promise.reject(new Error("对话已切换，消息尚未发送"));
    return latestStream.current.submit(values, options);
  };

  useEffect(() => setProgress(undefined), [threadId]);
  useEffect(() => {
    if (streamValue.isLoading) recoveryVersion.current += 1;
  }, [streamValue.isLoading]);
  useEffect(() => {
    setRecovered(undefined);
    if (!threadId) return;
    let cancelled = false;
    const version = ++recoveryVersion.current;
    fetch(`${apiUrl}/threads/${threadId}/control`)
      .then((r) => (r.ok ? r.json() : null))
      .then((r) => {
        if (
          r &&
          Array.isArray(r.messages) &&
          !cancelled &&
          recoveryVersion.current === version
        )
          setRecovered({ key, value: r });
      })
      .catch(console.error);
    return () => {
      cancelled = true;
    };
  }, [apiUrl, threadId]);
  const publicFallback = !streamValue.isLoading && recovered?.messages?.length;
  // SDK 在副作用中清理此前的会话；重置完成前，不要将旧会话
  // 最后一次渲染的消息归档到新选中的会话。
  const loadedThread = streamValue.history[0]?.checkpoint.thread_id;
  const liveMessages = messagesForThread(
    streamValue.messages,
    key,
    settledStreamKey,
    threadId,
    loadedThread,
  );
  const messages = mergePublicMessages(
    archive.key === key ? archive.messages : [],
    mergePublicMessages(recovered?.messages || [], liveMessages),
  );
  useEffect(() => {
    setArchive((old) => {
      const merged = mergePublicMessages(
        old.key === key ? old.messages : [],
        messages,
      );
      return old.key === key && merged === old.messages
        ? old
        : { key, messages: merged };
    });
  }, [key, messages]);

  useEffect(() => {
    checkGraphStatus(apiUrl).then((ok) => {
      if (ok) void getThreads();
      if (!ok) {
        toast.error("无法连接 MonoPlanner 服务", {
          description: () => (
            <p>
              请确认本地服务已启动：<code>{apiUrl}</code>。
            </p>
          ),
          duration: 10000,
          richColors: true,
          closeButton: true,
        });
      }
    });
  }, [apiUrl, getThreads]);

  return (
    <StreamContext.Provider
      value={{
        ...streamValue,
        submit,
        ...(publicFallback
          ? {
              values: { ...streamValue.values, ...recovered },
            }
          : {}),
        messages,
        apiUrl,
        progress,
        retainMessage,
        ensureThread,
        preparingThread,
      }}
    >
      {children}
    </StreamContext.Provider>
  );
};

export const StreamProvider: React.FC<{ children: ReactNode }> = ({
  children,
}) => (
  <StreamSession
    apiUrl={localApiUrl()}
    assistantId="monoplaner"
  >
    {children}
  </StreamSession>
);

// 提供读取上下文的自定义 Hook。
export const useStreamContext = (): StreamContextType => {
  const context = useContext(StreamContext);
  if (context === undefined) {
    throw new Error("useStreamContext must be used within a StreamProvider");
  }
  return context;
};

export default StreamContext;
