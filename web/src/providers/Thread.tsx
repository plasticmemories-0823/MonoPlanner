"use client";

import {
  createContext,
  useContext,
  useCallback,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { localApiUrl } from "@/lib/local-service";
import { requestJson } from "@/lib/conversation-api";
import type { Conversation } from "@/lib/conversation-types";
import {
  ThreadHistoryStore,
  type ConversationTitle,
  type ManagedThread,
  type Visibility,
} from "@/lib/thread-history";

export type { ManagedThread, Visibility } from "@/lib/thread-history";

type ThreadContextType = {
  getThreads: () => Promise<ManagedThread[]>;
  threads: ManagedThread[];
  threadsLoading: boolean;
  threadsError: string;
  upsertThread: (thread: Conversation, firstMessage?: string) => void;
  updateTitle: (thread: string, title: ConversationTitle) => void;
  setVisibility: (thread: string, visibility: Visibility) => Promise<void>;
  deletePermanently: (thread: string, retainEnvironment?: boolean) => Promise<void>;
};
const ThreadContext = createContext<ThreadContextType | undefined>(undefined);

export function ThreadProvider({ children }: { children: ReactNode }) {
  const apiUrl = localApiUrl();
  const [threads, setThreads] = useState<ManagedThread[]>([]);
  const [threadsLoading, setLoading] = useState(false);
  const [threadsError, setError] = useState("");
  const store = useRef(new ThreadHistoryStore());
  const active = useRef(true);
  const controller = useRef<AbortController | null>(null);
  const pending = useRef<Promise<ManagedThread[]> | null>(null);
  const generation = useRef(0);

  const upsertThread = useCallback((thread: Conversation, text?: string) => {
    setThreads(store.current.upsert(thread, text));
  }, []);
  const updateTitle = useCallback((id: string, title: ConversationTitle) => {
    setThreads(store.current.applyTitle(id, title));
  }, []);

  const getThreads = useCallback((): Promise<ManagedThread[]> => {
    if (pending.current && !controller.current?.signal.aborted) return pending.current;
    const current = ++generation.current;
    const abort = new AbortController();
    controller.current = abort;
    const ticket = store.current.startRefresh();
    setLoading(true);
    const request = (async () => {
      try {
        const result = await requestJson<{ conversations: Conversation[] }>(
          `${apiUrl}/conversations`,
          { signal: abort.signal },
        );
        if (!Array.isArray(result.conversations)) throw new Error("无法读取对话记录。");
        if (!active.current || abort.signal.aborted || current !== generation.current)
          return store.current.rows;
        const rows = result.conversations.map((row) => ({
          ...row,
          visibility: row.visibility || "active",
        }));
        const merged = store.current.applyRefresh(ticket, rows);
        setThreads(merged);
        setError("");
        return merged;
      } catch (error) {
        if (active.current && !abort.signal.aborted && current === generation.current)
          setError(error instanceof Error ? error.message : "无法读取对话记录。");
        return store.current.rows;
      } finally {
        if (current === generation.current) {
          if (active.current && !abort.signal.aborted) setLoading(false);
          pending.current = null;
        }
      }
    })();
    pending.current = request;
    return request;
  }, [apiUrl]);

  useEffect(() => {
    active.current = true;
    void getThreads();
    const refresh = () => {
      void getThreads();
    };
    window.addEventListener("focus", refresh);
    window.addEventListener("online", refresh);
    return () => {
      active.current = false;
      controller.current?.abort();
      window.removeEventListener("focus", refresh);
      window.removeEventListener("online", refresh);
    };
  }, [getThreads]);

  async function setVisibility(thread: string, visibility: Visibility) {
    await requestJson(`${apiUrl}/conversations/${thread}/visibility`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ visibility }),
    });
    setThreads(store.current.setVisibility(thread, visibility));
  }
  async function deletePermanently(thread: string, retainEnvironment = false) {
    await requestJson(`${apiUrl}/conversations/${thread}/purge`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ confirm_thread_id: thread, retain_environment: retainEnvironment }),
    });
    setThreads(store.current.purge(thread));
  }

  return (
    <ThreadContext.Provider
      value={{
        getThreads,
        threads,
        threadsLoading,
        threadsError,
        upsertThread,
        updateTitle,
        setVisibility,
        deletePermanently,
      }}
    >
      {children}
    </ThreadContext.Provider>
  );
}

export function useThreads() {
  const context = useContext(ThreadContext);
  if (!context) throw new Error("useThreads must be used within a ThreadProvider");
  return context;
}
