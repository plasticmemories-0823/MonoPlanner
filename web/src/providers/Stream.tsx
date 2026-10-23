"use client";

import React, {
  createContext,
  useContext,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { useQueryState } from "nuqs";
import { localApiUrl } from "@/lib/local-service";
import { connectLocalSession } from "@/lib/local-session";
import {
  ConversationMonitor,
  requestJson,
  readStorageFailure,
  StorageRequestError,
} from "@/lib/conversation-api";
import { useInitialServiceHealth } from "./LocalSession";
import {
  emptySnapshot,
  isWorking,
  type Conversation,
  type ConversationSnapshot,
  type ConversationCommand,
  type CommandReceipt,
  type StorageFailure,
} from "@/lib/conversation-types";
import { useThreads } from "./Thread";

type StreamContextType = {
  apiUrl: string;
  values: ConversationSnapshot;
  messages: ConversationSnapshot["messages"];
  isLoading: boolean;
  workRunning: boolean;
  error: Error | undefined;
  storageError: StorageFailure | null;
  preparingThread: boolean;
  ensureThread: (firstMessage?: string, repo?: string) => Promise<string>;
  command: (value: ConversationCommand, threadId?: string) => Promise<CommandReceipt>;
  selectWorkspace: (repo: string) => Promise<void>;
  reconnect: () => void;
};

const StreamContext = createContext<StreamContextType | undefined>(undefined);

export function StreamProvider({ children }: { children: ReactNode }) {
  const apiUrl = localApiUrl();
  const [threadId, setThreadId] = useQueryState("threadId");
  const { getThreads, upsertThread, updateTitle } = useThreads();
  const [view, setView] = useState<{ thread: string; snapshot: ConversationSnapshot }>();
  const [connectionError, setConnectionError] = useState<{ thread: string; message: string }>();
  const [preparingThread, setPreparingThread] = useState(false);
  const initialHealth = useInitialServiceHealth();
  const [storageError, setStorageError] = useState<StorageFailure | null>(initialHealth);
  const selected = useRef(threadId);
  const setup = useRef<Promise<string> | null>(null);
  const creation = useRef<{ id: string; repo: string; owner: string | null } | null>(null);
  const selectionCommit = useRef<{
    id: string;
    resolve: () => void;
    reject: (error: Error) => void;
  } | null>(null);
  const monitor = useRef<ConversationMonitor | null>(null);
  const values = view?.thread === threadId ? view.snapshot : emptySnapshot();
  const latest = useRef(values);
  const reconnecting = useRef<AbortController | null>(null);

  useLayoutEffect(() => {
    selected.current = threadId;
    latest.current = values;
  }, [threadId, values]);

  useEffect(() => {
    const waiting = selectionCommit.current;
    if (!waiting) return;
    if (threadId === waiting.id) {
      selectionCommit.current = null;
      waiting.resolve();
    } else if (threadId) {
      selectionCommit.current = null;
      waiting.reject(new Error("对话已切换，操作尚未提交。"));
    }
  }, [threadId]);
  useEffect(
    () => () => {
      selectionCommit.current?.reject(new Error("窗口已关闭，操作尚未提交。"));
      selectionCommit.current = null;
      reconnecting.current?.abort();
    },
    [],
  );

  useEffect(() => {
    if (!threadId) return;
    let previousStatus = "";
    const next = new ConversationMonitor(apiUrl, threadId, {
      snapshot(snapshot) {
        setView({ thread: threadId, snapshot });
        if (snapshot.title) updateTitle(threadId, snapshot.title);
        if (snapshot.status !== previousStatus) {
          previousStatus = snapshot.status;
          void getThreads();
        }
      },
      error(message) {
        setConnectionError({ thread: threadId, message });
      },
      storageError: setStorageError,
    });
    monitor.current = next;
    void next.start();
    return () => {
      next.dispose();
      if (monitor.current === next) monitor.current = null;
    };
  }, [apiUrl, threadId, updateTitle, getThreads]);

  async function createConversation(repo = "", firstMessage?: string) {
    if (storageError) throw new StorageRequestError(storageError);
    const expected = selected.current;
    if (creation.current?.repo !== repo || creation.current.owner !== expected)
      creation.current = { id: crypto.randomUUID(), repo, owner: expected };
    const draft = creation.current;
    let created: Conversation;
    try {
      created = await requestJson<Conversation>(`${apiUrl}/conversations`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id: draft.id, repo }),
      });
    } catch (error) {
      if (error instanceof StorageRequestError) setStorageError(error.storageError);
      throw error;
    }
    if (creation.current?.id === draft.id) creation.current = null;
    upsertThread(created, firstMessage);
    if (selected.current !== expected)
      throw new Error("对话已切换，新对话已保存在侧栏，消息尚未发送。");
    // 授权窗口依赖已提交的会话属性，不能仅等待 URL 写入完成。
    const committed = new Promise<void>((resolve, reject) => {
      selectionCommit.current = { id: created.thread_id, resolve, reject };
    });
    try {
      await setThreadId(created.thread_id);
      await committed;
    } finally {
      if (selectionCommit.current?.id === created.thread_id) selectionCommit.current = null;
    }
    return created.thread_id;
  }

  async function ensureThread(firstMessage?: string, repo = "") {
    if (setup.current) return setup.current;
    if (selected.current) return selected.current;
    const pending = createConversation(repo, firstMessage);
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
  }

  async function command(value: ConversationCommand, target = selected.current || "") {
    if (!target) throw new Error("请先创建对话。");
    if (target === latest.current.thread_id && latest.current.read_only)
      throw new Error("旧版会话仅供查看，请新建对话继续。");
    if (storageError && value.kind !== "pause" && value.kind !== "resume")
      throw new StorageRequestError(storageError);
    let receipt: CommandReceipt;
    try {
      receipt = await requestJson<CommandReceipt>(`${apiUrl}/conversations/${target}/commands`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(value),
      });
    } catch (error) {
      if (error instanceof StorageRequestError) setStorageError(error.storageError);
      throw error;
    }
    if (receipt.storage_error) setStorageError(readStorageFailure(receipt.storage_error));
    if (selected.current === target) await monitor.current?.refresh();
    void getThreads();
    return receipt;
  }

  async function selectWorkspace(repo: string) {
    if (storageError) throw new StorageRequestError(storageError);
    const current = latest.current;
    let target = selected.current;
    if (target && current.repo !== repo && current.work) {
      if (isWorking(current.work.status))
        await command({ id: crypto.randomUUID(), kind: "pause" }, target);
      target = await createConversation(repo);
    } else if (!target) target = await ensureThread(undefined, repo);
    await command({ id: crypto.randomUUID(), kind: "workspace", repo }, target);
    localStorage.setItem("monoplaner.repo", repo);
  }

  async function reconnect() {
    if (reconnecting.current) return;
    const controller = new AbortController();
    reconnecting.current = controller;
    const target = selected.current || "";
    try {
      const service = await connectLocalSession(controller.signal);
      if (Object.hasOwn(service, "storage_error"))
        setStorageError(readStorageFailure(service.storage_error));
      if (selected.current === target) await monitor.current?.start();
      await getThreads();
    } catch (error) {
      if (!controller.signal.aborted)
        setConnectionError({
          thread: target,
          message: error instanceof Error ? error.message : "无法重新连接本机服务。",
        });
    } finally {
      reconnecting.current = null;
    }
  }

  const message = connectionError?.thread === threadId ? connectionError.message : "";
  const workRunning = !storageError && isWorking(values.work?.status);
  return (
    <StreamContext.Provider
      value={{
        apiUrl,
        values,
        messages: values.messages,
        isLoading: !storageError && (isWorking(values.chat_status) || workRunning),
        workRunning,
        storageError,
        error: message ? new Error(message) : undefined,
        preparingThread,
        ensureThread,
        command,
        selectWorkspace,
        reconnect: () => {
          void reconnect();
        },
      }}
    >
      {children}
    </StreamContext.Provider>
  );
}

export function useStreamContext(): StreamContextType {
  const context = useContext(StreamContext);
  if (!context) throw new Error("useStreamContext must be used within a StreamProvider");
  return context;
}

export default StreamContext;
