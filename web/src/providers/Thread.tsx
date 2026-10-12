import { validate } from "uuid";
import { getApiKey } from "@/lib/api-key";
import { resolveApiUrl } from "@/lib/resolve-api-url";
import { Thread } from "@langchain/langgraph-sdk";
import { useQueryState } from "nuqs";
import {
  createContext,
  useContext,
  ReactNode,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
import { createClient } from "./client";
import {
  ThreadHistoryStore,
  type ConversationTitle,
  type ManagedThread,
  type Visibility,
} from "@/lib/thread-history";

export type { ManagedThread, Visibility } from "@/lib/thread-history";

interface ThreadContextType {
  getThreads: () => Promise<ManagedThread[]>;
  threads: ManagedThread[];
  threadsLoading: boolean;
  threadsError: string;
  upsertThread: (thread: Thread, firstMessage?: string) => void;
  updateTitle: (thread: string, title: ConversationTitle) => void;
  waitForTitle: (thread: string, firstMessage: string) => void;
  setVisibility: (thread: string, visibility: Visibility) => Promise<void>;
  deletePermanently: (thread: string) => Promise<void>;
}

const ThreadContext = createContext<ThreadContextType | undefined>(undefined);

export function getThreadSearchMetadata(
  assistantId: string,
): { graph_id: string } | { assistant_id: string } {
  if (validate(assistantId)) {
    return { assistant_id: assistantId };
  } else {
    return { graph_id: assistantId };
  }
}

export function ThreadProvider({ children }: { children: ReactNode }) {
  const envApiUrl = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:2024";
  const envAssistantId: string | undefined =
    process.env.NEXT_PUBLIC_ASSISTANT_ID || "monoplaner";
  const envAuthScheme: string | undefined = process.env.NEXT_PUBLIC_AUTH_SCHEME;

  const [apiUrl] = useQueryState("apiUrl", {
    defaultValue: envApiUrl || "",
  });
  const [assistantId] = useQueryState("assistantId");
  const [authScheme] = useQueryState("authScheme", {
    defaultValue: envAuthScheme || "",
  });
  const [threads, setThreads] = useState<ManagedThread[]>([]);
  const [threadsLoading, setThreadsLoading] = useState(false);
  const [threadsError, setThreadsError] = useState("");
  const [titleError, setTitleError] = useState("");
  const [titlePoll, setTitlePoll] = useState(0);
  const scope = `${apiUrl}\0${assistantId || envAssistantId}\0${authScheme}`;
  const activeScope = useRef(scope);
  activeScope.current = scope;
  const store = useRef(new ThreadHistoryStore());
  const request = useRef(0);
  useEffect(() => {
    store.current = new ThreadHistoryStore();
    setThreads([]);
    setThreadsError("");
    setTitleError("");
  }, [scope]);

  const upsertThread = useCallback((thread: Thread, firstMessage?: string) => {
    setThreads(store.current.upsert(thread, firstMessage));
  }, []);
  const updateTitle = useCallback(
    (thread: string, title: ConversationTitle) => {
      setThreads(store.current.applyTitle(thread, title));
    },
    [],
  );
  const waitForTitle = useCallback((thread: string, firstMessage: string) => {
    setThreads(store.current.waitForTitle(thread, firstMessage));
    if (store.current.titleRefreshState().pending)
      setTitlePoll((value) => value + 1);
  }, []);

  const getThreads = useCallback(async (): Promise<ManagedThread[]> => {
    const finalApiUrl = resolveApiUrl(apiUrl, envApiUrl);
    const resolvedAssistantId = assistantId || envAssistantId;
    if (!finalApiUrl || !resolvedAssistantId) return [];
    const expectedStore = store.current;
    expectedStore.retryTitles();
    const ticket = expectedStore.startRefresh();
    const version = ++request.current;
    setThreadsLoading(true);
    const client = createClient(
      finalApiUrl,
      getApiKey(finalApiUrl) ?? undefined,
      authScheme || undefined,
    );

    try {
      const threads = await client.threads.search({
        metadata: {
          ...getThreadSearchMetadata(resolvedAssistantId),
        },
        limit: 100,
      });

      const response = await fetch(`${finalApiUrl}/conversations`, {
        cache: "no-store",
      });
      if (!response.ok) throw new Error("无法读取对话管理信息");
      const {
        visibility = {},
        titles = {},
        pending_deletions = [],
      } = await response.json();
      const savedIds = new Set(threads.map((thread) => thread.thread_id));
      // A partial cleanup may have removed the SDK thread already. Keep a
      // minimal trash entry so the user can explicitly retry the cleanup.
      const pending: Thread[] = pending_deletions
        .filter((item: { thread_id: string }) => !savedIds.has(item.thread_id))
        .map((item: { thread_id: string; title: string }) => ({
          thread_id: item.thread_id,
          created_at: "",
          updated_at: "",
          metadata: {},
          status: "idle",
          values: { messages: [{ content: item.title }] },
        }));
      const incoming: ManagedThread[] = [...threads, ...pending]
        .filter((thread) => visibility[thread.thread_id] !== "purged")
        .map((thread) => ({
          ...thread,
          title: titles[thread.thread_id],
          visibility:
            visibility[thread.thread_id] === "purging"
              ? "deleted"
              : visibility[thread.thread_id] || "active",
        }));
      if (activeScope.current !== scope || store.current !== expectedStore)
        return [];
      const merged = expectedStore.applyRefresh(ticket, incoming, visibility);
      setThreads(merged);
      if (expectedStore.titleRefreshState().pending)
        setTitlePoll((value) => value + 1);
      if (request.current === version) setThreadsError("");
      return merged;
    } catch (error) {
      if (activeScope.current === scope && request.current === version)
        setThreadsError(
          error instanceof Error ? error.message : "无法读取对话记录",
        );
      // Refresh failures must not clear the last usable list.
      return expectedStore.rows;
    } finally {
      if (activeScope.current === scope && request.current === version)
        setThreadsLoading(false);
    }
  }, [apiUrl, envApiUrl, assistantId, authScheme, envAssistantId, scope]);

  useEffect(() => {
    const expectedStore = store.current;
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    const poll = async () => {
      const state = expectedStore.titleRefreshState();
      if (state.expired) setTitleError("标题尚未同步，可重试查询");
      if (!state.pending) return;
      try {
        // One lightweight watcher survives conversation selection changes.
        // It only reads display metadata, not SDK message histories.
        const response = await fetch(
          `${resolveApiUrl(apiUrl, envApiUrl)}/conversations`,
          {
            cache: "no-store",
            signal: controller.signal,
          },
        );
        if (!response.ok) throw new Error("无法同步对话标题");
        const result = await response.json();
        if (controller.signal.aborted || store.current !== expectedStore)
          return;
        setThreads(expectedStore.applyTitles(result.titles || {}));
        const next = expectedStore.titleRefreshState();
        setTitleError(next.expired ? "标题尚未同步，可重试查询" : "");
        if (next.pending) timer = setTimeout(poll, 1000);
      } catch (error) {
        if (!controller.signal.aborted)
          setTitleError(
            error instanceof Error ? error.message : "无法同步对话标题",
          );
      }
    };
    void poll();
    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, [scope, apiUrl, envApiUrl, titlePoll]);

  useEffect(() => {
    void getThreads();
    const refresh = () => {
      void getThreads();
    };
    window.addEventListener("focus", refresh);
    window.addEventListener("online", refresh);
    return () => {
      window.removeEventListener("focus", refresh);
      window.removeEventListener("online", refresh);
    };
  }, [getThreads]);

  const setVisibility = async (thread: string, visibility: Visibility) => {
    const response = await fetch(
      `${resolveApiUrl(apiUrl, envApiUrl)}/threads/${thread}/visibility`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ visibility }),
      },
    );
    const value = await response.json();
    if (!response.ok) throw new Error(value.error || "对话管理操作未完成");
    setThreads(store.current.setVisibility(thread, visibility));
  };

  const deletePermanently = async (thread: string) => {
    const response = await fetch(
      `${resolveApiUrl(apiUrl, envApiUrl)}/threads/${thread}/purge`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ confirm_thread_id: thread }),
      },
    );
    const value = await response.json();
    if (!response.ok) throw new Error(value.error || "彻底删除未完成");
    setThreads(store.current.purge(thread));
  };

  const value = {
    getThreads,
    threads,
    threadsLoading,
    threadsError: threadsError || titleError,
    upsertThread,
    updateTitle,
    waitForTitle,
    setVisibility,
    deletePermanently,
  };

  return (
    <ThreadContext.Provider value={value}>{children}</ThreadContext.Provider>
  );
}

export function useThreads() {
  const context = useContext(ThreadContext);
  if (context === undefined) {
    throw new Error("useThreads must be used within a ThreadProvider");
  }
  return context;
}
