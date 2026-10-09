import { v4 as uuidv4 } from "uuid";
import { Fragment, ReactNode, useEffect, useRef } from "react";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";
import { useStreamContext } from "@/providers/Stream";
import { useState, FormEvent } from "react";
import { Button } from "../ui/button";
import { Checkpoint, Message } from "@langchain/langgraph-sdk";
import { AssistantMessage, AssistantMessageLoading } from "./messages/ai";
import { HumanMessage } from "./messages/human";
import {
  DO_NOT_RENDER_ID_PREFIX,
  ensureToolCallsHaveResponses,
} from "@/lib/ensure-tool-responses";
import { BrandMark } from "../brand";
import { TooltipIconButton } from "./tooltip-icon-button";
import {
  ArrowDown,
  LoaderCircle,
  PanelRightOpen,
  PanelRightClose,
  SquarePen,
  XIcon,
  ArrowUp,
} from "lucide-react";
import { useQueryState, parseAsBoolean } from "nuqs";
import { StickToBottom, useStickToBottomContext } from "use-stick-to-bottom";
import ThreadHistory from "./history";
import { toast } from "sonner";
import { useMediaQuery } from "@/hooks/useMediaQuery";
import { Workspace, useWorkspace } from "./workspace";
import { useRunControls, statusLabel } from "./run-controls";
import { ActivityRows, useActivity } from "./activity";
import { ReadSources } from "./read-sources";
import { ModelSettings } from "./model-settings";
import {
  useArtifactOpen,
  ArtifactContent,
  ArtifactTitle,
  useArtifactContext,
} from "./artifact";

function StickyToBottomContent(props: {
  content: ReactNode;
  footer?: ReactNode;
  className?: string;
  contentClassName?: string;
}) {
  const context = useStickToBottomContext();
  return (
    <div
      ref={context.scrollRef}
      style={{ width: "100%", height: "100%" }}
      className={props.className}
    >
      <div
        ref={context.contentRef}
        className={props.contentClassName}
      >
        {props.content}
      </div>

      {props.footer}
    </div>
  );
}

function ScrollToBottom(props: { className?: string }) {
  const { isAtBottom, scrollToBottom } = useStickToBottomContext();

  if (isAtBottom) return null;
  return (
    <Button
      variant="outline"
      className={props.className}
      onClick={() => scrollToBottom()}
    >
      <ArrowDown className="h-4 w-4" />
      <span>回到最新消息</span>
    </Button>
  );
}

export function Thread() {
  const [artifactContext, setArtifactContext] = useArtifactContext();
  const [artifactOpen, closeArtifact] = useArtifactOpen();

  const [threadId, _setThreadId] = useQueryState("threadId");
  const [chatHistoryOpen, setChatHistoryOpen] = useQueryState(
    "chatHistoryOpen",
    parseAsBoolean.withDefault(false),
  );
  const [hideToolCalls, setHideToolCalls] = useQueryState(
    "hideToolCalls",
    parseAsBoolean.withDefault(false),
  );
  const [input, setInput] = useState("");
  const [firstTokenReceived, setFirstTokenReceived] = useState(false);
  const isLargeScreen = useMediaQuery("(min-width: 1024px)");

  const stream = useStreamContext();
  const { repo, setRepo } = useWorkspace(stream.apiUrl, stream.values.repo);
  const { pendingMessages, queueMessage, pause, resume } = useRunControls(
    stream,
    threadId,
  );
  const messages = [...stream.messages, ...pendingMessages];
  const isLoading = stream.isLoading;
  const activity = useActivity(stream.apiUrl, threadId, isLoading);
  const [readRoots, setReadRoots] = useState<string[]>([]);
  const savedReadRoots = JSON.stringify(stream.values.read_roots || []);
  useEffect(() => { setReadRoots(JSON.parse(savedReadRoots)); }, [threadId, savedReadRoots]);
  const messageIds = new Set(messages.map((message) => message.id));

  const lastError = useRef<string | undefined>(undefined);

  const setThreadId = (id: string | null) => {
    _setThreadId(id);

    // close artifact and reset artifact context
    closeArtifact();
    setArtifactContext({});
  };

  useEffect(() => {
    if (!stream.error) {
      lastError.current = undefined;
      return;
    }
    try {
      const message = (stream.error as any).message;
      if (!message || lastError.current === message) {
        // Message has already been logged. do not modify ref, return early.
        return;
      }

      // Message is defined, and it has not been logged yet. Save it, and send the error
      lastError.current = message;
      toast.error("An error occurred. Please try again.", {
        description: (
          <p>
            <strong>Error:</strong> <code>{message}</code>
          </p>
        ),
        richColors: true,
        closeButton: true,
      });
    } catch {
      // no-op
    }
  }, [stream.error]);

  // Stop the waiting indicator when the first assistant reply arrives.
  const prevMessageLength = useRef(0);
  useEffect(() => {
    if (
      messages.length !== prevMessageLength.current &&
      messages?.length &&
      messages[messages.length - 1].type === "ai"
    ) {
      setFirstTokenReceived(true);
    }

    prevMessageLength.current = messages.length;
  }, [messages]);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (input.trim().length === 0) return;
    setFirstTokenReceived(false);
    // Chat needs no workspace. The backend validates it when work is requested.
    if (repo.trim()) localStorage.setItem("monoplaner.repo", repo.trim());

    const newHumanMessage: Message = {
      id: uuidv4(),
      type: "human",
      content: input,
    };

    if (isLoading && threadId) {
      if (await queueMessage(newHumanMessage)) setInput("");
      return;
    }

    const toolMessages = ensureToolCallsHaveResponses(stream.messages);

    const context =
      Object.keys(artifactContext).length > 0 ? artifactContext : undefined;

    stream.submit(
      {
        messages: [...toolMessages, newHumanMessage],
        context,
        repo: repo.trim(),
        read_roots: readRoots,
      },
      {
        streamMode: ["values", "custom"],
        streamSubgraphs: true,
        streamResumable: true,
        config: { recursion_limit: 10000 },
        optimisticValues: (prev) => ({
          ...prev,
          context,
          messages: [
            ...(prev.messages ?? []),
            ...toolMessages,
            newHumanMessage,
          ],
        }),
      },
    );

    setInput("");
  };

  const handleRegenerate = (
    parentCheckpoint: Checkpoint | null | undefined,
  ) => {
    // Do this so the loading state is correct
    prevMessageLength.current = prevMessageLength.current - 1;
    setFirstTokenReceived(false);
    toast.info("请追加新要求；已执行操作不能通过回退聊天记录重跑。");
  };

  const chatStarted = !!threadId || !!messages.length;
  const hasNoAIOrToolMessages = !messages.find(
    (m) => m.type === "ai" || m.type === "tool",
  );

  return (
    <div className="flex h-screen w-full overflow-hidden">
      <div className="relative hidden lg:flex">
        <motion.div
          className="absolute z-20 h-full overflow-hidden border-r bg-white"
          style={{ width: 300 }}
          animate={
            isLargeScreen
              ? { x: chatHistoryOpen ? 0 : -300 }
              : { x: chatHistoryOpen ? 0 : -300 }
          }
          initial={{ x: -300 }}
          transition={
            isLargeScreen
              ? { type: "spring", stiffness: 300, damping: 30 }
              : { duration: 0 }
          }
        >
          <div
            className="relative h-full"
            style={{ width: 300 }}
          >
            <ThreadHistory />
          </div>
        </motion.div>
      </div>

      <div
        className={cn(
          "grid w-full grid-cols-[1fr_0fr] transition-all duration-500",
          artifactOpen && "grid-cols-[3fr_2fr]",
        )}
      >
        <motion.div
          className={cn(
            "relative flex min-w-0 flex-1 flex-col overflow-hidden",
            !chatStarted && "grid-rows-[1fr]",
          )}
          layout={isLargeScreen}
          animate={{
            marginLeft: chatHistoryOpen ? (isLargeScreen ? 300 : 0) : 0,
            width: chatHistoryOpen
              ? isLargeScreen
                ? "calc(100% - 300px)"
                : "100%"
              : "100%",
          }}
          transition={
            isLargeScreen
              ? { type: "spring", stiffness: 300, damping: 30 }
              : { duration: 0 }
          }
        >
          <header className="relative z-10 flex h-16 shrink-0 items-center justify-between px-3 sm:px-6">
            <div className="flex items-center gap-2">
              <TooltipIconButton
                tooltip="对话记录"
                variant="ghost"
                onClick={() => setChatHistoryOpen((p) => !p)}
              >
                {chatHistoryOpen ? (
                  <PanelRightOpen className="size-5" />
                ) : (
                  <PanelRightClose className="size-5" />
                )}
              </TooltipIconButton>
              <button
                type="button"
                onClick={() => setThreadId(null)}
                className="flex items-center gap-2 rounded-lg px-1"
                aria-label="MonoPlanner 首页"
              >
                <BrandMark className="size-8" />
                <span className="text-base font-semibold tracking-tight">
                  MonoPlanner
                </span>
              </button>
            </div>
            <TooltipIconButton
              tooltip="新建对话"
              variant="ghost"
              onClick={() => setThreadId(null)}
            >
              <SquarePen className="size-5" />
            </TooltipIconButton>
          </header>

          <StickToBottom className="relative flex-1 overflow-hidden">
            <StickyToBottomContent
              className={cn(
                "absolute inset-0 overflow-y-auto px-4 [&::-webkit-scrollbar]:w-1.5 [&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-thumb]:bg-gray-300 [&::-webkit-scrollbar-track]:bg-transparent",
                !chatStarted && "flex flex-col items-stretch justify-center",
                chatStarted && "grid grid-rows-[1fr_auto]",
              )}
              contentClassName={cn(
                "mx-auto flex w-full max-w-3xl flex-col gap-5",
                chatStarted ? "pt-8 pb-12" : "hidden",
              )}
              content={
                <>
                  {messages
                    .filter((m) => !m.id?.startsWith(DO_NOT_RENDER_ID_PREFIX))
                    .map((message, index) => (
                      <Fragment key={message.id || `${message.type}-${index}`}>
                        {message.type === "human" ? (
                          <HumanMessage
                            key={message.id || `${message.type}-${index}`}
                            message={message}
                            isLoading={isLoading}
                          />
                        ) : (
                          <AssistantMessage
                            key={message.id || `${message.type}-${index}`}
                            message={message}
                            isLoading={isLoading}
                            handleRegenerate={handleRegenerate}
                          />
                        )}
                        <ActivityRows
                          items={activity.items.filter(
                            (row) => row.after_id === message.id,
                          )}
                        />
                      </Fragment>
                    ))}
                  <ActivityRows
                    items={activity.items.filter(
                      (row) => !messageIds.has(row.after_id),
                    )}
                  />
                  {activity.error && (
                    <p className="text-muted-foreground text-sm">
                      活动记录暂未显示：{activity.error}
                    </p>
                  )}
                  {/* Special rendering case where there are no AI/tool messages, but there is an interrupt.
                    We need to render it outside of the messages list, since there are no messages to render */}
                  {hasNoAIOrToolMessages && !!stream.interrupt && (
                    <AssistantMessage
                      key="interrupt-msg"
                      message={undefined}
                      isLoading={isLoading}
                      handleRegenerate={handleRegenerate}
                    />
                  )}
                  {isLoading && !firstTokenReceived && (
                    <AssistantMessageLoading />
                  )}
                </>
              }
              footer={
                <div
                  className={cn(
                    "bg-background sticky bottom-0 flex flex-col items-center pb-4 sm:pb-6",
                    chatStarted ? "pt-3" : "gap-8",
                  )}
                >
                  {!chatStarted && (
                    <div className="mb-2 flex flex-col items-center text-center">
                      <BrandMark className="mb-4 size-20" />
                      <h1 className="text-3xl font-semibold tracking-tight sm:text-4xl">
                        从一个想法开始
                      </h1>
                      <p className="text-muted-foreground mt-3 text-sm">
                        聊一聊，或选择工作目录后开始构建。
                      </p>
                    </div>
                  )}

                  <ScrollToBottom className="animate-in fade-in-0 zoom-in-95 absolute bottom-full left-1/2 mb-4 -translate-x-1/2" />

                  <div
                    className={cn(
                      "bg-background relative z-10 mx-auto w-full max-w-3xl rounded-3xl shadow-[0_4px_28px_-8px_rgba(0,0,0,0.12)] transition-shadow focus-within:shadow-[0_6px_32px_-8px_rgba(0,0,0,0.16)]",
                      "border border-solid",
                    )}
                  >
                    <form
                      onSubmit={handleSubmit}
                      className="mx-auto grid max-w-3xl grid-rows-[1fr_auto] gap-2"
                    >
                      <textarea
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyDown={(e) => {
                          if (
                            e.key === "Enter" &&
                            !e.shiftKey &&
                            !e.metaKey &&
                            !e.nativeEvent.isComposing
                          ) {
                            e.preventDefault();
                            const el = e.target as HTMLElement | undefined;
                            const form = el?.closest("form");
                            form?.requestSubmit();
                          }
                        }}
                        aria-label="消息输入框"
                        placeholder="发消息，或描述你想完成的工作…"
                        className="placeholder:text-muted-foreground/70 field-sizing-content max-h-80 min-h-28 w-full resize-none border-none bg-transparent px-5 pt-5 pb-3 text-base shadow-none ring-0 outline-none focus:ring-0 focus:outline-none"
                      />

                      <div
                        className="flex items-end justify-between gap-2 px-3 pb-3"
                        data-testid="composer-toolbar"
                      >
                        <div className="flex min-w-0 flex-wrap items-center gap-1">
                          <Workspace
                            apiUrl={stream.apiUrl}
                            repo={repo}
                            onChange={setRepo}
                            disabled={
                              isLoading ||
                              Boolean(
                                stream.values.workspace_bound &&
                                ![
                                  "chat_complete",
                                  "verified_success",
                                  "task_blocked",
                                ].includes(stream.values.status || ""),
                              )
                            }
                          />
                          <ModelSettings
                            apiUrl={stream.apiUrl}
                            current={stream.values.runtime_settings}
                          />
                          <ReadSources apiUrl={stream.apiUrl} roots={readRoots} onChange={setReadRoots} disabled={isLoading} />
                        </div>
                        <div className="flex shrink-0 items-center gap-2">
                          {isLoading && (
                            <Button
                              type="button"
                              variant="outline"
                              size="sm"
                              className="rounded-full"
                              onClick={pause}
                              aria-label="安全暂停"
                            >
                              <LoaderCircle className="size-4 animate-spin" />
                              <span className="hidden sm:inline">暂停</span>
                            </Button>
                          )}
                          <Button
                            type="submit"
                            size="icon"
                            className="size-9 rounded-full shadow-none"
                            disabled={!input.trim()}
                            aria-label={isLoading ? "发送补充" : "发送"}
                            title={isLoading ? "发送补充" : "发送"}
                          >
                            <ArrowUp className="size-5" />
                          </Button>
                        </div>
                      </div>
                    </form>
                    {activity.usage && <p className="text-muted-foreground px-4 pb-2 text-xs" data-testid="usage-summary">
                      助手已记录 Token：{activity.usage.total_tokens.toLocaleString()} · 未缓存输入：{activity.usage.uncached_input_tokens.toLocaleString()}
                      {" · "}请求：{activity.usage.calls} · 模型用时：{activity.usage.model_seconds.toFixed(1)} 秒
                      {activity.usage.unknown_calls > 0 && ` · ${activity.usage.unknown_calls} 次用量未知`}
                    </p>}
                    {(isLoading || stream.values.status) && (
                      <div
                        role="status"
                        className="border-border/60 text-muted-foreground flex items-center justify-between gap-3 border-t px-4 py-2 text-xs"
                      >
                        <span className="flex min-w-0 items-center gap-2">
                          <span
                            className={cn(
                              "size-1.5 shrink-0 rounded-full",
                              isLoading
                                ? "animate-pulse bg-emerald-500"
                                : "bg-muted-foreground/50",
                            )}
                          />
                          {isLoading
                            ? stream.progress?.text || "正在处理请求"
                            : statusLabel(stream.values.status)}
                        </span>
                        {!isLoading &&
                          ["paused", "awaiting_user"].includes(
                            stream.values.status || "",
                          ) && (
                            <Button
                              type="button"
                              variant="ghost"
                              size="sm"
                              className="h-6 px-2 text-xs"
                              onClick={resume}
                            >
                              继续任务
                            </Button>
                          )}
                      </div>
                    )}
                  </div>
                </div>
              }
            />
          </StickToBottom>
        </motion.div>
        <div className="relative flex flex-col border-l">
          <div className="absolute inset-0 flex min-w-[30vw] flex-col">
            <div className="grid grid-cols-[1fr_auto] border-b p-4">
              <ArtifactTitle className="truncate overflow-hidden" />
              <button
                onClick={closeArtifact}
                className="cursor-pointer"
              >
                <XIcon className="size-5" />
              </button>
            </div>
            <ArtifactContent className="relative flex-grow" />
          </div>
        </div>
      </div>
    </div>
  );
}
