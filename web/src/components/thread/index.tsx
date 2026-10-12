import { v4 as uuidv4 } from "uuid";
import { Fragment, ReactNode, useEffect, useRef } from "react";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";
import { useStreamContext } from "@/providers/Stream";
import { useState, FormEvent } from "react";
import { Button } from "../ui/button";
import { Message } from "@langchain/langgraph-sdk";
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
  ArrowUp,
} from "lucide-react";
import { useQueryState, parseAsBoolean } from "nuqs";
import { StickToBottom, useStickToBottomContext } from "use-stick-to-bottom";
import ThreadHistory from "./history";
import { ResizableHistory } from "./history/resizable";
import { mergePublicMessages } from "@/lib/public-messages";
import { activityAnchors } from "@/lib/activity-groups";
import { toast } from "sonner";
import { useMediaQuery } from "@/hooks/useMediaQuery";
import { Workspace, useWorkspace } from "./workspace";
import { useRunControls, statusLabel } from "./run-controls";
import { ActivityRows, useActivity } from "./activity";
import { ReadSources } from "./read-sources";
import { ModelSettings } from "./model-settings";
import { DesktopCards } from "./desktop-controls";
import { ExecutionPermissions } from "./execution-permissions";

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
  const [threadId, setThreadId] = useQueryState("threadId");
  const [chatHistoryOpen, setChatHistoryOpen] = useQueryState(
    "chatHistoryOpen",
    parseAsBoolean.withDefault(false),
  );
  const [input, setInput] = useState("");
  const [firstTokenReceived, setFirstTokenReceived] = useState(false);
  const isLargeScreen = useMediaQuery("(min-width: 1024px)");

  const stream = useStreamContext();
  const { repo, setRepo } = useWorkspace(stream.apiUrl, stream.values.repo);
  const { pendingMessages, queueMessage, pause, resume, pauseRequested } =
    useRunControls(stream, threadId);
  const messages = mergePublicMessages(stream.messages, pendingMessages);
  const isLoading = stream.isLoading;
  const activity = useActivity(stream.apiUrl, threadId, isLoading);
  const messageIds = new Set(messages.map((message) => message.id));
  const anchors = activityAnchors(messages);
  const groupedActivity = new Map<string, typeof activity.items>();
  for (const row of activity.items) {
    const anchor = anchors.get(row.after_id) || row.after_id;
    const group = groupedActivity.get(anchor) || [];
    group.push(row);
    groupedActivity.set(anchor, group);
  }

  const lastError = useRef<string | undefined>(undefined);

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
      toast.error("当前请求未完成，消息和运行记录已保留", {
        description: (
          <p>
            <strong>诊断：</strong> <code>{message}</code>
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
    if (input.trim().length === 0 || stream.preparingThread) return;
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

    let targetThread: string;
    try {
      targetThread = await stream.ensureThread(input);
    } catch (error) {
      toast.error(
        error instanceof Error ? error.message : "无法创建对话，消息尚未发送",
      );
      return;
    }
    stream.retainMessage(newHumanMessage, targetThread);
    void stream
      .submit(
        {
          messages: [...toolMessages, newHumanMessage],
          repo: repo.trim(),
        },
        {
          threadId: targetThread,
          streamMode: ["values", "custom"],
          streamSubgraphs: true,
          streamResumable: true,
          config: { recursion_limit: 10000 },
          optimisticValues: (prev) => ({
            ...prev,
            messages: [
              ...(prev.messages ?? []),
              ...toolMessages,
              newHumanMessage,
            ],
          }),
        },
      )
      .catch((error) => {
        toast.error(error instanceof Error ? error.message : "消息未能发送");
      });

    setInput("");
  };

  const chatStarted = !!threadId || !!messages.length;
  return (
    <div className="flex h-screen w-full overflow-hidden">
      <ResizableHistory open={chatHistoryOpen}>
        <ThreadHistory />
      </ResizableHistory>

      <div className="flex min-w-0 flex-1">
        <motion.div
          className={cn(
            "relative flex min-w-0 flex-1 flex-col overflow-hidden",
            !chatStarted && "grid-rows-[1fr]",
          )}
          layout={isLargeScreen}
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
                          />
                        )}
                        <ActivityRows
                          items={groupedActivity.get(message.id || "") || []}
                        />
                      </Fragment>
                    ))}
                  <ActivityRows
                    items={[...groupedActivity]
                      .filter(([anchor]) => !messageIds.has(anchor))
                      .flatMap(([, items]) => items)}
                  />
                  <DesktopCards
                    apiUrl={stream.apiUrl}
                    threadId={threadId}
                    running={isLoading}
                  />
                  {activity.error && (
                    <p className="text-muted-foreground text-sm">
                      {activity.error}
                      <button
                        type="button"
                        className="ml-2 underline"
                        onClick={activity.reconnect}
                      >
                        重新连接
                      </button>
                    </p>
                  )}
                  {isLoading &&
                    !firstTokenReceived &&
                    !activity.items.some((row) => row.status === "running") && (
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
                        rows={1}
                        className="placeholder:text-muted-foreground/70 field-sizing-content max-h-48 min-h-12 w-full resize-none border-none bg-transparent px-4 pt-3 pb-1 text-base shadow-none ring-0 outline-none focus:ring-0 focus:outline-none"
                      />

                      <div
                        className="flex items-end justify-between gap-2 px-3 pb-3"
                        data-testid="composer-toolbar"
                      >
                        <div className="flex min-w-0 flex-1 items-center gap-0.5 overflow-x-auto whitespace-nowrap">
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
                            trailingControls={
                              <ExecutionPermissions
                                apiUrl={stream.apiUrl}
                                threadId={threadId}
                                running={isLoading}
                                ensureThread={stream.ensureThread}
                              />
                            }
                            menuItems={
                              <>
                                <ReadSources
                                  apiUrl={stream.apiUrl}
                                  threadId={threadId}
                                  ensureThread={stream.ensureThread}
                                  disabled={isLoading}
                                />
                              </>
                            }
                          />
                        </div>
                        <div className="flex shrink-0 items-center gap-2">
                          {isLoading && (
                            <Button
                              type="button"
                              variant="outline"
                              size="sm"
                              className="rounded-full"
                              onClick={pause}
                              disabled={pauseRequested}
                              aria-label={pauseRequested ? "正在暂停" : "暂停"}
                            >
                              <LoaderCircle className="size-4 animate-spin" />
                              <span className="hidden sm:inline">
                                {pauseRequested ? "正在暂停" : "暂停"}
                              </span>
                            </Button>
                          )}
                          <Button
                            type="submit"
                            size="icon"
                            className="size-9 rounded-full shadow-none"
                            disabled={!input.trim() || stream.preparingThread}
                            aria-label={isLoading ? "发送补充" : "发送"}
                            title={isLoading ? "发送补充" : "发送"}
                          >
                            <ArrowUp className="size-5" />
                          </Button>
                        </div>
                      </div>
                    </form>
                    {
                      <p
                        className="text-muted-foreground px-4 pb-2 text-xs"
                        data-testid="usage-summary"
                      >
                        <span title="按已返回缓存用量的输入 Token 加权；含联网辅助调用。">
                          缓存命中：
                          {activity.usage?.cache_rate != null
                            ? `${(activity.usage.cache_rate * 100).toFixed(1)}%${activity.usage.cache_unknown_calls ? "（部分记录）" : ""}`
                            : "暂无数据"}
                        </span>
                        {" · "}
                        <span title="最近一次主模型请求的输入与已完成输出，包含提示词、工具和必要思考续接；等待回复时为估算。不是累计 Token。">
                          上下文：
                          {activity.usage?.context &&
                          activity.usage.context.window_tokens > 0
                            ? `${activity.usage.context.estimated ? "≈" : ""}${((activity.usage.context.used_tokens / activity.usage.context.window_tokens) * 100).toFixed(1)}%`
                            : "暂无数据"}
                        </span>
                        {activity.usage && (
                          <>
                            {" · "}已记录 Token：
                            {activity.usage.total_tokens.toLocaleString()}
                            {" · "}请求：{activity.usage.calls}
                            {activity.usage.unknown_calls > 0 &&
                              ` · ${activity.usage.unknown_calls} 次用量未知`}
                          </>
                        )}
                      </p>
                    }
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
      </div>
    </div>
  );
}
