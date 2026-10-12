import React, {
  createContext,
  useContext,
  ReactNode,
  useState,
  useEffect,
  useMemo,
  useRef,
} from "react";
import { useStream } from "@langchain/langgraph-sdk/react";
import { type Message } from "@langchain/langgraph-sdk";
import { useQueryState } from "nuqs";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { BrandMark } from "@/components/brand";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { ArrowRight } from "lucide-react";
import { PasswordInput } from "@/components/ui/password-input";
import { getApiKey, setApiKey as storeApiKey } from "@/lib/api-key";
import { resolveApiUrl } from "@/lib/resolve-api-url";
import { mergePublicMessages, messagesForThread } from "@/lib/public-messages";
import { getThreadSearchMetadata, useThreads } from "./Thread";
import { createClient } from "./client";
import { toast } from "sonner";
import type { RuntimeSettings } from "@/components/thread/model-settings";

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

async function checkGraphStatus(
  apiUrl: string,
  apiKey: string | null,
  authScheme?: string,
): Promise<boolean> {
  try {
    const headers = new Headers();
    if (apiKey) headers.set("X-Api-Key", apiKey);
    if (authScheme) headers.set("X-Auth-Scheme", authScheme);

    const res = await fetch(`${apiUrl}/info`, {
      headers,
    });

    return res.ok;
  } catch (e) {
    console.error(e);
    return false;
  }
}

const StreamSession = ({
  children,
  apiKey,
  apiUrl,
  assistantId,
  authScheme,
}: {
  children: ReactNode;
  apiKey: string | null;
  apiUrl: string;
  assistantId: string;
  authScheme?: string;
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
      const client = createClient(apiUrl, apiKey ?? undefined, authScheme);
      const created = await client.threads.create({
        metadata: getThreadSearchMetadata(assistantId),
      });
      upsertThread(created, firstMessage);
      if (firstMessage) waitForTitle(created.thread_id, firstMessage);
      if (activeKey.current !== expectedKey)
        throw new Error("对话已切换，请在当前对话重新操作");
      // URL updates can resolve before React commits. Desktop controls must
      // finish binding to this conversation before saving its permissions.
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
    apiKey: apiKey ?? undefined,
    assistantId,
    ...(authScheme && {
      defaultHeaders: {
        "X-Auth-Scheme": authScheme,
      },
    }),
    threadId: threadId ?? null,
    reconnectOnMount: true,
    // Coalesce presentation updates from dense SDK streams; keep every event
    // in the SDK store and leave request/execution timing unchanged.
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
    // Runs after the SDK has reset its stream and started history loading.
    // Until then its public messages still belong to the previous selection.
    setSettledStreamKey(key);
  }, [key]);
  const latestStream = useRef(streamValue);
  latestStream.current = streamValue;
  // ensureThread waits for the URL and React commit. The caller still holds
  // its earlier render, so submit through the latest SDK closure explicitly.
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
  // The SDK clears a previous thread in an effect. Do not archive its last
  // render under the newly selected thread while that reset is pending.
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
    checkGraphStatus(apiUrl, apiKey, authScheme).then((ok) => {
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
  }, [apiKey, apiUrl, authScheme, getThreads]);

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

// Default values for the form
const DEFAULT_API_URL = "http://localhost:2024";
const DEFAULT_ASSISTANT_ID = "agent";
const AGENT_BUILDER_AUTH_SCHEME = "langsmith-api-key";

export const StreamProvider: React.FC<{ children: ReactNode }> = ({
  children,
}) => {
  // Get environment variables
  const envApiUrl = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:2024";
  const envAssistantId: string | undefined =
    process.env.NEXT_PUBLIC_ASSISTANT_ID || "monoplaner";
  const envAuthScheme: string | undefined = process.env.NEXT_PUBLIC_AUTH_SCHEME;

  // Use URL params with env var fallbacks
  const [apiUrl, setApiUrl] = useQueryState("apiUrl", {
    defaultValue: envApiUrl || "",
  });
  const [assistantId, setAssistantId] = useQueryState("assistantId", {
    defaultValue: envAssistantId || "",
  });
  const [authScheme, setAuthScheme] = useQueryState("authScheme", {
    defaultValue: envAuthScheme || "",
  });
  const [isAgentBuilder, setIsAgentBuilder] = useState(
    () =>
      (authScheme || envAuthScheme || "").toLowerCase() ===
      AGENT_BUILDER_AUTH_SCHEME,
  );

  const finalApiUrl = resolveApiUrl(apiUrl, envApiUrl);
  const finalAssistantId = assistantId || envAssistantId;
  const finalAuthScheme = authScheme || envAuthScheme || "";

  const apiKey = useMemo(() => getApiKey(finalApiUrl) || "", [finalApiUrl]);

  // Show the form if we: don't have an API URL, or don't have an assistant ID
  if (!finalApiUrl || !finalAssistantId) {
    return (
      <div className="flex min-h-screen w-full items-center justify-center p-4">
        <div className="animate-in fade-in-0 zoom-in-95 bg-background flex max-w-3xl flex-col rounded-lg border shadow-lg">
          <div className="mt-14 flex flex-col gap-2 border-b p-6">
            <div className="flex flex-col items-start gap-2">
              <BrandMark />
              <h1 className="text-xl font-semibold tracking-tight">
                MonoPlanner
              </h1>
            </div>
            <p className="text-muted-foreground">
              连接 MonoPlanner 本地服务后，即可开始对话与工作。
            </p>
          </div>
          <form
            onSubmit={(e) => {
              e.preventDefault();

              const form = e.target as HTMLFormElement;
              const formData = new FormData(form);
              const apiUrl = formData.get("apiUrl") as string;
              const assistantId = formData.get("assistantId") as string;
              const apiKey = formData.get("apiKey") as string;

              setApiUrl(apiUrl);
              storeApiKey(apiUrl, apiKey);
              setAssistantId(assistantId);
              setAuthScheme(isAgentBuilder ? AGENT_BUILDER_AUTH_SCHEME : "");

              form.reset();
            }}
            className="bg-muted/50 flex flex-col gap-6 p-6"
          >
            <div className="flex flex-col gap-2">
              <Label htmlFor="apiUrl">
                服务地址<span className="text-rose-500">*</span>
              </Label>
              <p className="text-muted-foreground text-sm">
                MonoPlanner 后端的连接地址。
              </p>
              <Input
                id="apiUrl"
                name="apiUrl"
                className="bg-background"
                defaultValue={apiUrl || DEFAULT_API_URL}
                required
              />
            </div>

            <div className="flex flex-col gap-2">
              <Label htmlFor="assistantId">
                Assistant / Graph ID<span className="text-rose-500">*</span>
              </Label>
              <p className="text-muted-foreground text-sm">
                This is the ID of the graph (can be the graph name), or
                assistant to fetch threads from, and invoke when actions are
                taken.
              </p>
              <Input
                id="assistantId"
                name="assistantId"
                className="bg-background"
                defaultValue={assistantId || DEFAULT_ASSISTANT_ID}
                required
              />
            </div>

            <div className="flex flex-col gap-2">
              <Label htmlFor="apiKey">服务访问密钥（可选）</Label>
              <p className="text-muted-foreground text-sm">
                This is <strong>NOT</strong> required if using a local LangGraph
                server. This value is stored in your browser's local storage and
                is only used to authenticate requests sent to your LangGraph
                server.
              </p>
              <PasswordInput
                id="apiKey"
                name="apiKey"
                defaultValue={apiKey ?? ""}
                className="bg-background"
                placeholder="lsv2_pt_..."
              />
            </div>

            <div className="flex flex-col gap-3">
              <div className="flex items-center justify-between gap-4">
                <div className="flex flex-col gap-1">
                  <Label htmlFor="agentBuilderEnabled">自定义服务认证</Label>
                  <p className="text-muted-foreground text-sm">
                    Enable this for Agent Builder deployments.
                  </p>
                </div>
                <Switch
                  id="agentBuilderEnabled"
                  checked={isAgentBuilder}
                  onCheckedChange={setIsAgentBuilder}
                />
              </div>
            </div>

            <div className="mt-2 flex justify-end">
              <Button
                type="submit"
                size="lg"
              >
                Continue
                <ArrowRight className="size-5" />
              </Button>
            </div>
          </form>
        </div>
      </div>
    );
  }

  return (
    <StreamSession
      apiKey={apiKey}
      apiUrl={finalApiUrl}
      assistantId={finalAssistantId}
      authScheme={finalAuthScheme || undefined}
    >
      {children}
    </StreamSession>
  );
};

// Create a custom hook to use the context
export const useStreamContext = (): StreamContextType => {
  const context = useContext(StreamContext);
  if (context === undefined) {
    throw new Error("useStreamContext must be used within a StreamProvider");
  }
  return context;
};

export default StreamContext;
