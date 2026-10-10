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
  useState,
  Dispatch,
  SetStateAction,
} from "react";
import { createClient } from "./client";

export type Visibility = "active" | "archived" | "deleted";
export type ManagedThread = Thread & { visibility: Visibility };

interface ThreadContextType {
  getThreads: () => Promise<ManagedThread[]>;
  threads: ManagedThread[];
  setThreads: Dispatch<SetStateAction<ManagedThread[]>>;
  threadsLoading: boolean;
  setThreadsLoading: Dispatch<SetStateAction<boolean>>;
  setVisibility: (thread: string, visibility: Visibility) => Promise<void>;
}

const ThreadContext = createContext<ThreadContextType | undefined>(undefined);

function getThreadSearchMetadata(
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

  const getThreads = useCallback(async (): Promise<ManagedThread[]> => {
    const finalApiUrl = resolveApiUrl(apiUrl, envApiUrl);
    const resolvedAssistantId = assistantId || envAssistantId;
    if (!finalApiUrl || !resolvedAssistantId) return [];
    const client = createClient(
      finalApiUrl,
      getApiKey(finalApiUrl) ?? undefined,
      authScheme || undefined,
    );

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
    const { visibility } = await response.json();
    return threads.map((thread) => ({
      ...thread,
      visibility: visibility[thread.thread_id] || "active",
    }));
  }, [apiUrl, envApiUrl, assistantId, authScheme, envAssistantId]);

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
    setThreads((old) =>
      old.map((item) =>
        item.thread_id === thread ? { ...item, visibility } : item,
      ),
    );
  };

  const value = {
    getThreads,
    threads,
    setThreads,
    threadsLoading,
    setThreadsLoading,
    setVisibility,
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
